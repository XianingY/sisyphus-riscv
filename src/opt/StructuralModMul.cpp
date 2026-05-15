#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>

using namespace sys;

namespace {

struct ModMulCandidate {
  FuncOp *func = nullptr;
  int mod = 0;
};

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

std::string helperName(const std::string &name) {
  return "__sisy_struct_modmul_" + name;
}

bool hasFunctionNamed(ModuleOp *module, const std::string &name) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == name)
      return true;
  return false;
}

bool isInt(Op *op, int value) {
  return op && isa<IntOp>(op) && V(op) == value;
}

bool same(Op *a, Op *b) {
  return a && b && a == b;
}

bool hasI32Args(FuncOp *func, int argc) {
  if (!func->has<ArgCountAttr>() || func->get<ArgCountAttr>()->count != argc)
    return false;
  if (auto types = func->find<ArgTypesAttr>()) {
    if ((int) types->types.size() != argc)
      return false;
    for (auto ty : types->types)
      if (ty != Value::i32)
        return false;
  }
  return true;
}

bool hasOnlySelfCalls(FuncOp *func) {
  const auto &self = NAME(func);
  int calls = 0;
  for (auto call : func->findAll<CallOp>()) {
    calls++;
    if (NAME(call) != self)
      return false;
  }
  return calls > 0;
}

bool isModuloBy(Op *op, int mod, Op *&value) {
  auto modOp = dyn_cast<ModIOp>(op);
  if (!modOp || modOp->getOperandCount() != 2 || !isInt(modOp->DEF(1), mod))
    return false;
  value = modOp->DEF(0);
  return true;
}

bool isSignedHalfOf(Op *op, Op *value) {
  if (!op || !value)
    return false;
  if (auto div = dyn_cast<DivIOp>(op))
    return same(div->DEF(0), value) && isInt(div->DEF(1), 2);

  auto shift = dyn_cast<RShiftOp>(op);
  if (!shift || !isInt(shift->DEF(1), 1))
    return false;
  if (same(shift->DEF(0), value))
    return true;

  // Canonical signed division by two:
  // (x + ((x >> 31) & 1)) >> 1
  auto add = dyn_cast<AddIOp>(shift->DEF(0));
  if (!add)
    return false;
  Op *bias = nullptr;
  if (same(add->DEF(0), value))
    bias = add->DEF(1);
  else if (same(add->DEF(1), value))
    bias = add->DEF(0);
  else
    return false;

  auto andOp = dyn_cast<AndIOp>(bias);
  if (!andOp || !isInt(andOp->DEF(1), 1))
    return false;
  auto sign = dyn_cast<RShiftOp>(andOp->DEF(0));
  return sign && same(sign->DEF(0), value) && isInt(sign->DEF(1), 31);
}

bool isSelfHalfCall(Op *op, FuncOp *func, Op *arg0, Op *arg1) {
  auto call = dyn_cast<CallOp>(op);
  return call && NAME(call) == NAME(func) && call->getOperandCount() == 2 &&
         same(call->DEF(0), arg0) && isSignedHalfOf(call->DEF(1), arg1);
}

bool isDoubleMod(Op *op, Op *selfCall, int mod) {
  Op *value = nullptr;
  if (!isModuloBy(op, mod, value))
    return false;
  if (auto mul = dyn_cast<MulIOp>(value))
    return ((same(mul->DEF(0), selfCall) && isInt(mul->DEF(1), 2)) ||
            (same(mul->DEF(1), selfCall) && isInt(mul->DEF(0), 2)));
  if (auto add = dyn_cast<AddIOp>(value))
    return same(add->DEF(0), selfCall) && same(add->DEF(1), selfCall);
  return false;
}

bool isAddModWithArg(Op *op, Op *doubleMod, Op *arg0, int mod) {
  Op *value = nullptr;
  if (!isModuloBy(op, mod, value))
    return false;
  auto add = dyn_cast<AddIOp>(value);
  return add && ((same(add->DEF(0), doubleMod) && same(add->DEF(1), arg0)) ||
                 (same(add->DEF(1), doubleMod) && same(add->DEF(0), arg0)));
}

bool isArgMod(Op *op, Op *arg0, int mod) {
  Op *value = nullptr;
  return isModuloBy(op, mod, value) && same(value, arg0);
}

bool returnPhiHasValues(FuncOp *func, Op *zero, Op *argMod, Op *oddMod, Op *doubleMod) {
  auto returns = func->findAll<ReturnOp>();
  if (returns.size() != 1 || returns[0]->getOperandCount() != 1)
    return false;

  auto phi = dyn_cast<PhiOp>(returns[0]->DEF(0));
  if (!phi)
    return false;

  bool hasZero = false;
  bool hasArgMod = false;
  bool hasOddMod = false;
  bool hasDoubleMod = false;
  for (int i = 0; i < phi->getOperandCount(); i++) {
    Op *in = phi->DEF(i);
    if (same(in, zero))
      hasZero = true;
    else if (same(in, argMod))
      hasArgMod = true;
    else if (same(in, oddMod))
      hasOddMod = true;
    else if (same(in, doubleMod))
      hasDoubleMod = true;
    else
      return false;
  }
  return hasZero && hasArgMod && hasOddMod && hasDoubleMod;
}

std::set<int> modulusCandidates(FuncOp *func) {
  std::set<int> values;
  for (auto op : func->findAll<IntOp>()) {
    int value = V(op);
    if (value > 1)
      values.insert(value);
  }
  return values;
}

std::vector<Op*> allOps(FuncOp *func) {
  std::vector<Op*> ops;
  for (auto region : func->getRegions())
    for (auto block : region->getBlocks())
      for (auto op : block->getOps())
        ops.push_back(op);
  return ops;
}

std::optional<ModMulCandidate> classify(FuncOp *func) {
  if (!hasI32Args(func, 2) || !hasOnlySelfCalls(func))
    return std::nullopt;

  Op *arg0 = nullptr;
  Op *arg1 = nullptr;
  Op *zero = nullptr;
  for (auto op : func->findAll<GetArgOp>()) {
    if (V(op) == 0)
      arg0 = op;
    else if (V(op) == 1)
      arg1 = op;
  }
  for (auto op : func->findAll<IntOp>())
    if (V(op) == 0)
      zero = op;
  if (!arg0 || !arg1 || !zero)
    return std::nullopt;

  Op *selfCall = nullptr;
  for (auto call : func->findAll<CallOp>()) {
    if (!isSelfHalfCall(call, func, arg0, arg1))
      return std::nullopt;
    if (selfCall)
      return std::nullopt;
    selfCall = call;
  }
  if (!selfCall)
    return std::nullopt;

  for (int mod : modulusCandidates(func)) {
    Op *argMod = nullptr;
    Op *doubleMod = nullptr;
    Op *oddMod = nullptr;
    for (auto op : allOps(func)) {
      if (!argMod && isArgMod(op, arg0, mod))
        argMod = op;
      if (!doubleMod && isDoubleMod(op, selfCall, mod))
        doubleMod = op;
    }
    if (!argMod || !doubleMod)
      continue;
    for (auto op : allOps(func)) {
      if (isAddModWithArg(op, doubleMod, arg0, mod)) {
        oddMod = op;
        break;
      }
    }
    if (!oddMod)
      continue;
    if (returnPhiHasValues(func, zero, argMod, oddMod, doubleMod))
      return ModMulCandidate{ func, mod };
  }
  return std::nullopt;
}

bool provenNonNegative(Op *op) {
  if (!op)
    return false;
  if (isa<IntOp>(op))
    return V(op) >= 0;
  if (!op->has<RangeAttr>())
    return false;
  auto [low, high] = RANGE(op);
  (void) high;
  return low >= 0;
}

void ensureHelper(ModuleOp *module, const std::string &originalName, int mod) {
  auto name = helperName(originalName);
  if (hasFunctionNamed(module, name))
    return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(name),
    new ArgCountAttr(2),
    new ArgTypesAttr({ Value::i32, Value::i32 })
  });
  func->add<ImpureAttr>();
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto slow = region->appendBlock();
  auto fast = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto lhs = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  auto rhs = builder.create<GetArgOp>(Value::i32, { new IntAttr(1) });
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto lhsNeg = builder.create<LtOp>(std::vector<Value>{ lhs, zero });
  auto rhsNeg = builder.create<LtOp>(std::vector<Value>{ rhs, zero });
  auto anyNeg = builder.create<OrIOp>(std::vector<Value>{ lhsNeg, rhsNeg });
  builder.create<BranchOp>(std::vector<Value>{ anyNeg },
                           { new TargetAttr(slow), new ElseAttr(fast) });

  builder.setToBlockEnd(slow);
  auto fallback = builder.create<CallOp>(Value::i32, std::vector<Value>{ lhs, rhs },
                                         { new NameAttr(originalName) });
  builder.create<ReturnOp>(std::vector<Value>{ fallback });

  builder.setToBlockEnd(fast);
  auto product = builder.create<MulLOp>(std::vector<Value>{ lhs, rhs });
  auto modOp = builder.create<IntOp>({ new IntAttr(mod) });
  auto reduced = builder.create<ModLOp>(std::vector<Value>{ product, modOp });
  builder.create<ReturnOp>(std::vector<Value>{ reduced });
}

} // namespace

std::map<std::string, int> StructuralModMul::stats() {
  return {
    { "classified", classified },
    { "guarded-calls", guarded },
    { "replaced-calls", replaced },
  };
}

void StructuralModMul::run() {
  if (!envEnabled("SISY_ENABLE_STRUCTURAL_MODMUL", true))
    return;

  CallGraph(module).run();

  std::map<std::string, ModMulCandidate> candidates;
  for (auto func : collectFuncs()) {
    if (isExtern(NAME(func)))
      continue;
    auto candidate = classify(func);
    if (!candidate)
      continue;
    candidates[NAME(func)] = *candidate;
    classified++;
  }
  if (candidates.empty())
    return;

  Builder builder;
  runRewriter([&](CallOp *call) {
    auto it = candidates.find(NAME(call));
    if (it == candidates.end() || call->getResultType() != Value::i32 ||
        call->getOperandCount() != 2)
      return false;

    auto parent = call->getParentOp<FuncOp>();
    if (parent == it->second.func)
      return false;

    builder.setBeforeOp(call);
    if (!provenNonNegative(call->DEF(0)) || !provenNonNegative(call->DEF(1))) {
      ensureHelper(module, NAME(call), it->second.mod);
      NAME(call) = helperName(NAME(call));
      guarded++;
      replaced++;
      return false;
    }

    auto product = builder.create<MulLOp>(std::vector<Value>{ call->DEF(0), call->DEF(1) });
    auto mod = builder.create<IntOp>({ new IntAttr(it->second.mod) });
    auto reduced = builder.create<ModLOp>(std::vector<Value>{ product, mod });
    call->replaceAllUsesWith(reduced);
    call->erase();
    replaced++;
    return true;
  });
}
