#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

constexpr int kDim0 = 64;
constexpr int kDim1 = 2048;
constexpr int kTotal = kDim0 * kDim1;

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

std::string symbolFor(const std::string &prefix, const std::string &name) {
  std::string out = prefix;
  for (char ch : name) {
    if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
        (ch >= '0' && ch <= '9') || ch == '_')
      out.push_back(ch);
    else
      out.push_back('_');
  }
  return out;
}

bool exprReferencesGlobal(Op *op, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (isa<GetGlobalOp>(op))
    return true;
  for (auto operand : op->getOperands())
    if (exprReferencesGlobal(operand.defining, seen))
      return true;
  return false;
}

bool storesToGlobal(Op *func) {
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    std::set<Op*> seen;
    if (exprReferencesGlobal(store->DEF(1), seen))
      return true;
  }
  return false;
}

bool hasOnlySelfCalls(FuncOp *func) {
  const auto &self = NAME(func);
  for (auto call : func->findAll<CallOp>()) {
    if (NAME(call) != self)
      return false;
  }
  return true;
}

std::vector<Op*> orderedArgs(FuncOp *func, int argCount) {
  std::vector<Op*> args(argCount, nullptr);
  for (auto getarg : func->findAll<GetArgOp>()) {
    if (getarg->getResultType() != Value::i32)
      return {};
    int idx = V(getarg);
    if (idx < 0 || idx >= argCount)
      return {};
    args[idx] = getarg;
  }
  for (auto arg : args)
    if (!arg)
      return {};
  return args;
}

bool eligible(FuncOp *func) {
  if (!func->has<ArgCountAttr>() || !func->has<ArgTypesAttr>())
    return false;
  if (func->get<ArgCountAttr>()->count != 2)
    return false;
  for (auto ty : func->get<ArgTypesAttr>()->types)
    if (ty != Value::i32)
      return false;

  int opCount = 0;
  int selfCalls = 0;
  const auto &self = NAME(func);
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      opCount++;
      if (auto call = dyn_cast<CallOp>(op)) {
        if (NAME(call) != self)
          return false;
        selfCalls++;
      }
      if (isa<CloneOp>(op) || isa<JoinOp>(op) || isa<WakeOp>(op))
        return false;
    }
  }
  if (selfCalls == 0 || opCount > 220)
    return false;
  if (!hasOnlySelfCalls(func) || storesToGlobal(func))
    return false;

  bool hasI32Return = false;
  for (auto ret : func->findAll<ReturnOp>()) {
    if (ret->getOperandCount() == 0)
      return false;
    if (ret->DEF()->getResultType() != Value::i32)
      return false;
    hasI32Return = true;
  }
  return hasI32Return;
}

GlobalOp *createIntGlobal(ModuleOp *module, const std::string &name, int elements,
                          int initValue = 0) {
  Builder builder;
  builder.setToRegionStart(module->getRegion());
  int *values = nullptr;
  if (initValue != 0) {
    values = new int[elements];
    for (int i = 0; i < elements; i++)
      values[i] = initValue;
  }
  return builder.create<GlobalOp>({
    new SizeAttr((size_t) elements * 4),
    new IntArrayAttr(values, elements),
    new NameAttr(name),
    new DimensionAttr({ elements }),
  });
}

Value cacheOffset(Builder &builder, Value arg0, Value arg1) {
  auto stride = builder.create<IntOp>({ new IntAttr(kDim1) });
  auto row = builder.create<MulIOp>({ arg0, stride });
  auto idx = builder.create<AddIOp>({ row, arg1 });
  auto four = builder.create<IntOp>({ new IntAttr(4) });
  return builder.create<MulIOp>({ idx, four });
}

Value indexedAddress(Builder &builder, const std::string &globalName, Value offset) {
  auto base = builder.create<GetGlobalOp>({ new NameAttr(globalName) });
  return builder.create<AddLOp>({ base, offset });
}

Value inRange2D(Builder &builder, Value arg0, Value arg1) {
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto dim0 = builder.create<IntOp>({ new IntAttr(kDim0) });
  auto dim1 = builder.create<IntOp>({ new IntAttr(kDim1) });
  auto a0lo = builder.create<LeOp>({ zero, arg0 });
  auto a0hi = builder.create<LtOp>({ arg0, dim0 });
  auto a1lo = builder.create<LeOp>({ zero, arg1 });
  auto a1hi = builder.create<LtOp>({ arg1, dim1 });
  auto c0 = builder.create<AndIOp>(std::vector<Value>{ a0lo, a0hi });
  auto c1 = builder.create<AndIOp>(std::vector<Value>{ c0, a1lo });
  return builder.create<AndIOp>(std::vector<Value>{ c1, a1hi });
}

void bumpEpochBefore(CallOp *call, const std::string &epochName, int &count) {
  Builder builder;
  builder.setBeforeOp(call);
  auto ep = builder.create<GetGlobalOp>({ new NameAttr(epochName) });
  auto old = builder.create<LoadOp>(Value::i32, { ep }, { new SizeAttr(4) });
  auto one = builder.create<IntOp>({ new IntAttr(1) });
  auto next = builder.create<AddIOp>({ old, one });
  builder.create<StoreOp>({ next, ep }, { new SizeAttr(4) });
  count++;
}

void addEntryCheck(FuncOp *func, const std::string &valueName,
                   const std::string &seenName, const std::string &epochName,
                   int &entryChecks) {
  auto args = orderedArgs(func, 2);
  assert(args.size() == 2);

  auto region = func->getRegion();
  auto entry = region->getFirstBlock();
  auto term = entry->getLastOp();
  if (!isa<GotoOp>(term))
    return;
  auto original = TARGET(term);

  auto checkBlock = region->insertAfter(entry);
  auto hitBlock = region->insertAfter(checkBlock);

  Builder builder;
  builder.setBeforeOp(term);
  Value inRange = inRange2D(builder, args[0], args[1]);
  Value offset = cacheOffset(builder, args[0], args[1]);
  builder.replace<BranchOp>(term, { inRange },
                            { new TargetAttr(checkBlock), new ElseAttr(original) });

  builder.setToBlockEnd(checkBlock);
  auto seenAddr = indexedAddress(builder, seenName, offset);
  auto seen = builder.create<LoadOp>(Value::i32, { seenAddr }, { new SizeAttr(4) });
  auto ep = builder.create<GetGlobalOp>({ new NameAttr(epochName) });
  auto epoch = builder.create<LoadOp>(Value::i32, { ep }, { new SizeAttr(4) });
  auto hit = builder.create<EqOp>(std::vector<Value>{ seen, epoch });
  builder.create<BranchOp>(std::vector<Value>{ hit },
                           { new TargetAttr(hitBlock), new ElseAttr(original) });

  builder.setToBlockEnd(hitBlock);
  auto valueAddr = indexedAddress(builder, valueName, offset);
  auto cached = builder.create<LoadOp>(Value::i32, { valueAddr }, { new SizeAttr(4) });
  builder.create<ReturnOp>({ cached });
  entryChecks++;
}

void addReturnStores(FuncOp *func, const std::string &valueName,
                     const std::string &seenName, const std::string &epochName,
                     int &storesAdded) {
  auto args = orderedArgs(func, 2);
  if (args.size() != 2)
    return;
  std::vector<ReturnOp*> returns;
  for (auto ret : func->findAll<ReturnOp>())
    returns.push_back(cast<ReturnOp>(ret));

  auto region = func->getRegion();
  Builder builder;
  for (auto ret : returns) {
    if (ret->getOperandCount() == 0)
      continue;

    auto retBlock = ret->getParent();
    auto storeBlock = region->insertAfter(retBlock);
    auto finalBlock = region->insertAfter(storeBlock);
    Value retValue = ret->getOperand();

    retBlock->splitOpsAfter(finalBlock, ret);

    builder.setToBlockEnd(retBlock);
    Value inRange = inRange2D(builder, args[0], args[1]);
    builder.create<BranchOp>({ inRange }, { new TargetAttr(storeBlock), new ElseAttr(finalBlock) });

    builder.setToBlockEnd(storeBlock);
    Value offset = cacheOffset(builder, args[0], args[1]);
    auto valueAddr = indexedAddress(builder, valueName, offset);
    auto seenAddr = indexedAddress(builder, seenName, offset);
    auto ep = builder.create<GetGlobalOp>({ new NameAttr(epochName) });
    auto epoch = builder.create<LoadOp>(Value::i32, { ep }, { new SizeAttr(4) });
    builder.create<StoreOp>({ retValue, valueAddr }, { new SizeAttr(4) });
    builder.create<StoreOp>({ epoch, seenAddr }, { new SizeAttr(4) });
    builder.create<GotoOp>({ new TargetAttr(finalBlock) });
    storesAdded++;
  }
}

} // namespace

std::map<std::string, int> RuntimeMemoize::stats() {
  return {
    { "memoized-functions", memoized },
    { "entry-checks", entryChecks },
    { "call-epoch-bumps", callEpochBumps },
  };
}

void RuntimeMemoize::run() {
  if (!envEnabled("SISY_ENABLE_RUNTIME_MEMOIZE", true))
    return;

  CallGraph(module).run();
  auto funcs = collectFuncs();
  std::set<FuncOp*> candidates;
  for (auto func : funcs) {
    if (eligible(func))
      candidates.insert(func);
  }
  if (candidates.empty())
    return;

  for (auto func : candidates) {
    const auto &fname = NAME(func);
    std::string valueName = symbolFor("__sisy_memo_value_", fname);
    std::string seenName = symbolFor("__sisy_memo_seen_", fname);
    std::string epochName = symbolFor("__sisy_memo_epoch_", fname);
    createIntGlobal(module, valueName, kTotal);
    createIntGlobal(module, seenName, kTotal);
    createIntGlobal(module, epochName, 1, 1);

    for (auto call : module->findAll<CallOp>()) {
      if (NAME(call) != fname)
        continue;
      auto parent = call->getParentOp<FuncOp>();
      if (parent == func)
        continue;
      bumpEpochBefore(cast<CallOp>(call), epochName, callEpochBumps);
    }

    addEntryCheck(func, valueName, seenName, epochName, entryChecks);
    int storesAdded = 0;
    addReturnStores(func, valueName, seenName, epochName, storesAdded);
    if (storesAdded > 0)
      memoized++;
  }
}
