#include "Passes.h"
#include "CleanupPasses.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

using namespace sys;

namespace {

constexpr int kDefaultBudget = 64;
constexpr int kDefaultMaxOps = 2000;

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

int envInt(const char *name, int fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::atoi(env);
}

bool isRecursiveFunc(FuncOp *func) {
  if (!func || !func->has<CallerAttr>())
    return false;
  const auto &name = NAME(func);
  const auto &callers = CALLER(func);
  return std::find(callers.begin(), callers.end(), name) != callers.end();
}

bool isSpecializedName(const std::string &name) {
  return name.rfind("__carg_", 0) == 0;
}

std::string constSuffix(int value) {
  if (value < 0)
    return "m" + std::to_string(-value);
  return std::to_string(value);
}

std::string cloneName(const std::string &name, int index, int value) {
  return "__carg_" + std::to_string(index) + "_" + constSuffix(value) + "_" + name;
}

bool foldTinyIntConstants(ModuleOp *module) {
  bool changed = false;
  Builder builder;
  auto tryFold = [&](Op *op) {
    if (op->getOperandCount() != 2 || !isa<IntOp>(op->DEF(0)) || !isa<IntOp>(op->DEF(1)))
      return false;
    int a = V(op->DEF(0));
    int b = V(op->DEF(1));
    int value = 0;
    if (isa<AddIOp>(op))
      value = a + b;
    else if (isa<SubIOp>(op))
      value = a - b;
    else if (isa<MulIOp>(op))
      value = a * b;
    else
      return false;
    builder.replace<IntOp>(op, { new IntAttr(value) });
    changed = true;
    return true;
  };

  for (auto func : module->findAll<FuncOp>()) {
    for (auto bb : func->getRegion()->getBlocks()) {
      auto ops = bb->getOps();
      for (auto op : ops)
        tryFold(op);
    }
  }
  return changed;
}

int opCount(FuncOp *func) {
  int count = 0;
  for (auto bb : func->getRegion()->getBlocks())
    for ([[maybe_unused]] auto op : bb->getOps())
      count++;
  return count;
}

std::optional<int> immutableScalarGlobalValue(
    const std::string &name,
    const std::map<std::string, GlobalOp*> &gMap,
    const std::set<std::string> &mutableGlobals) {
  if (mutableGlobals.count(name))
    return std::nullopt;
  auto it = gMap.find(name);
  if (it == gMap.end())
    return std::nullopt;
  auto global = it->second;
  if (!global->has<ConstAttr>())
    return std::nullopt;
  if (!global->has<DimensionAttr>() || DIM(global).size() != 1 || DIM(global)[0] != 1)
    return std::nullopt;
  auto iarr = global->find<IntArrayAttr>();
  if (!iarr || iarr->size != 1)
    return std::nullopt;
  if (!iarr->vi)
    return 0;
  return iarr->vi[0];
}

bool addressMentionsGlobal(const std::string &name, Op *addr) {
  if (!addr)
    return false;
  if (isa<GetGlobalOp>(addr))
    return NAME(addr) == name;
  auto alias = addr->find<AliasAttr>();
  if (!alias || alias->unknown)
    return false;
  for (auto &[base, _] : alias->location) {
    if (base && isa<GlobalOp>(base) && NAME(base) == name)
      return true;
  }
  return false;
}

bool mayStoreToGlobal(const std::string &name, Op *addr,
                      const std::set<std::string> &escapedGlobals) {
  if (addressMentionsGlobal(name, addr))
    return true;
  auto alias = addr ? addr->find<AliasAttr>() : nullptr;
  return escapedGlobals.count(name) && (!alias || alias->unknown);
}

std::optional<std::string> zeroOffsetGlobalName(Op *addr) {
  if (!addr)
    return std::nullopt;
  if (isa<GetGlobalOp>(addr))
    return NAME(addr);
  auto alias = addr->find<AliasAttr>();
  if (!alias || alias->unknown || alias->location.size() != 1)
    return std::nullopt;
  auto &[base, offsets] = *alias->location.begin();
  if (!base || !isa<GlobalOp>(base) || offsets.size() != 1 || offsets[0] != 0)
    return std::nullopt;
  return NAME(base);
}

std::optional<int> tryGetConstantValue(
    Op *op,
    const std::map<std::string, GlobalOp*> &gMap,
    const std::set<std::string> &mutableGlobals,
    std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return std::nullopt;
  seen.insert(op);

  if (isa<IntOp>(op))
    return V(op);

  if (isa<LoadOp>(op)) {
    auto name = zeroOffsetGlobalName(op->DEF(0));
    if (!name)
      return std::nullopt;
    return immutableScalarGlobalValue(*name, gMap, mutableGlobals);
  }

  if (op->getOperandCount() != 2)
    return std::nullopt;
  auto lhs = tryGetConstantValue(op->DEF(0), gMap, mutableGlobals, seen);
  auto rhs = tryGetConstantValue(op->DEF(1), gMap, mutableGlobals, seen);
  if (!lhs || !rhs)
    return std::nullopt;

  if (isa<AddIOp>(op))
    return *lhs + *rhs;
  if (isa<SubIOp>(op))
    return *lhs - *rhs;
  if (isa<MulIOp>(op))
    return *lhs * *rhs;
  if (isa<DivIOp>(op) && *rhs != 0)
    return *lhs / *rhs;
  if (isa<ModIOp>(op) && *rhs != 0)
    return *lhs % *rhs;
  return std::nullopt;
}

void copyRegion(Region *dst, Region *src) {
  std::unordered_map<BasicBlock*, BasicBlock*> blockMap;
  std::unordered_map<Op*, Op*> opMap;
  Builder builder;

  for (auto bb : src->getBlocks())
    blockMap[bb] = dst->appendBlock();

  for (auto [oldBlock, newBlock] : blockMap) {
    builder.setToBlockStart(newBlock);
    for (auto op : oldBlock->getOps()) {
      auto copied = builder.copy(op);
      opMap[op] = copied;
    }
  }

  for (auto [oldOp, newOp] : opMap) {
    (void) oldOp;
    auto operands = newOp->getOperands();
    newOp->removeAllOperands();
    for (auto operand : operands) {
      auto def = operand.defining;
      newOp->pushOperand(opMap.count(def) ? opMap[def]->getResult() : operand);
    }
  }

  for (auto [oldBlock, newBlock] : blockMap) {
    (void) oldBlock;
    auto term = newBlock->getLastOp();
    if (!term)
      continue;
    if (auto target = term->find<TargetAttr>(); target && blockMap.count(target->bb))
      target->bb = blockMap[target->bb];
    if (auto els = term->find<ElseAttr>(); els && blockMap.count(els->bb))
      els->bb = blockMap[els->bb];
  }

  for (auto [oldOp, newOp] : opMap) {
    (void) oldOp;
    if (!isa<PhiOp>(newOp))
      continue;
    for (auto attr : newOp->getAttrs())
      if (auto from = dyn_cast<FromAttr>(attr); from && blockMap.count(from->bb))
        from->bb = blockMap[from->bb];
  }
}

void replaceSpecializedArg(FuncOp *func, int index, int value) {
  Builder builder;
  auto args = func->findAll<GetArgOp>();
  for (auto arg : args) {
    if (V(arg) != index)
      continue;
    builder.setBeforeOp(arg);
    auto constant = builder.create<IntOp>({ new IntAttr(value) });
    arg->replaceAllUsesWith(constant);
    arg->erase();
  }
}

} // namespace

std::map<std::string, int> ConstArgSpecialize::stats() {
  return {
    { "cloned", cloned },
    { "calls-retargeted", retargeted },
    { "skipped-budget", skippedBudget },
  };
}

void ConstArgSpecialize::run() {
  if (!envEnabled("SISY_ENABLE_CONST_ARG_SPECIALIZE", true))
    return;

  CallGraph(module).run();

  const int budget = envInt("SISY_CONST_ARG_SPECIALIZE_BUDGET", kDefaultBudget);
  const int maxOps = envInt("SISY_CONST_ARG_SPECIALIZE_MAX_OPS", kDefaultMaxOps);
  std::set<std::string> produced;
  auto gMap = getGlobalMap();
  std::set<std::string> escapedGlobals;
  for (auto call : module->findAll<CallOp>()) {
    for (auto operand : call->getOperands()) {
      auto def = operand.defining;
      if (!def || def->getResultType() != Value::i64)
        continue;
      for (auto &[name, _] : gMap)
        if (addressMentionsGlobal(name, def))
          escapedGlobals.insert(name);
    }
  }
  std::set<std::string> mutableGlobals;
  for (auto store : module->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    auto addr = store->DEF(1);
    if (!addr)
      continue;
    for (auto &[name, _] : gMap)
      if (mayStoreToGlobal(name, addr, escapedGlobals))
        mutableGlobals.insert(name);
  }
  mutableGlobals.insert(escapedGlobals.begin(), escapedGlobals.end());

  for (int round = 0; round < budget; round++) {
    while (foldTinyIntConstants(module));
    auto fmap = getFunctionMap();
    auto calls = module->findAll<CallOp>();
    bool changed = false;

    for (auto call : calls) {
      auto &callee = NAME(call);
      if (isExtern(callee) || isSpecializedName(callee) || !fmap.count(callee))
        continue;

      auto func = fmap[callee];
      if (!func->has<ArgCountAttr>() || opCount(func) > maxOps)
        continue;
      if (call->getOperandCount() != func->get<ArgCountAttr>()->count)
        continue;
      auto argTypes = func->find<ArgTypesAttr>();
      if (!argTypes || (int) argTypes->types.size() != call->getOperandCount())
        continue;
      if (isRecursiveFunc(func) &&
          !envEnabled("SISY_ENABLE_RECURSIVE_CONST_ARG_SPECIALIZE", true))
        continue;

      int argIndex = -1;
      int argValue = 0;
      for (int i = 0; i < call->getOperandCount(); i++) {
        if (argTypes->types[i] != Value::i32)
          continue;
        auto def = call->DEF(i);
        if (!def)
          continue;
        std::set<Op*> seen;
        auto maybeValue = tryGetConstantValue(def, gMap, mutableGlobals, seen);
        if (!maybeValue)
          continue;
        int value = *maybeValue;
        if (value < -1 || value > 16)
          continue;
        argIndex = i;
        argValue = value;
        break;
      }
      if (argIndex < 0)
        continue;

      auto specialized = cloneName(callee, argIndex, argValue);
      if (!fmap.count(specialized) && !produced.count(specialized)) {
        if (cloned >= budget) {
          skippedBudget++;
          continue;
        }

        Builder builder;
        builder.setToRegionStart(module->getRegion());
        std::vector<Attr*> attrs = {
          new NameAttr(specialized),
          func->get<ArgCountAttr>()->clone()
        };
        if (auto argTypes = func->find<ArgTypesAttr>())
          attrs.push_back(argTypes->clone());
        auto clone = builder.create<FuncOp>(attrs);
        if (func->has<ImpureAttr>())
          clone->add<ImpureAttr>();
        clone->appendRegion();
        copyRegion(clone->getRegion(), func->getRegion());
        replaceSpecializedArg(clone, argIndex, argValue);

        produced.insert(specialized);
        cloned++;
      }

      callee = specialized;
      retargeted++;
      changed = true;
    }

    if (!changed)
      break;

    while (foldTinyIntConstants(module));
    RegularFold(module).run();
    SimplifyCFG(module).run();
    GVN(module).run();
    DCE(module).run();
  }

  CallGraph(module).run();
}
