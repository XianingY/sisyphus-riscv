#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

constexpr int kDefaultCapacity = 1 << 18;

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

int envInt(const char *name, int fallback) {
  auto env = std::getenv(name);
  if (!env || !env[0])
    return fallback;
  return std::atoi(env);
}

int roundPow2(int value) {
  int out = 1;
  while (out < value && out > 0 && out < (1 << 29))
    out <<= 1;
  return out > 0 ? out : kDefaultCapacity;
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

bool storesToGlobal(FuncOp *func) {
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    std::set<Op*> seen;
    if (exprReferencesGlobal(store->DEF(1), seen))
      return true;
  }
  return false;
}

bool readsOnlyImmutableScalarGlobals(FuncOp *func, ModuleOp *module) {
  auto gets = func->findAll<GetGlobalOp>();
  if (gets.empty())
    return false;

  std::map<std::string, GlobalOp*> globals;
  for (auto glob : module->findAll<GlobalOp>())
    if (glob->has<NameAttr>())
      globals[NAME(glob)] = cast<GlobalOp>(glob);

  for (auto get : gets) {
    auto it = globals.find(NAME(get));
    if (it == globals.end())
      return false;
    auto global = it->second;
    if (!global->has<DimensionAttr>() || DIM(global).size() != 1 || DIM(global)[0] != 1)
      return false;
    if (!global->find<IntArrayAttr>())
      return false;
  }
  return true;
}

bool hasOnlySelfCalls(FuncOp *func) {
  const auto &self = NAME(func);
  for (auto call : func->findAll<CallOp>())
    if (NAME(call) != self)
      return false;
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

enum class RejectReason {
  None,
  Impure,
  ArgShape,
  Types,
  Calls,
  Stores,
  Ops,
  Return
};

RejectReason eligible(FuncOp *func) {
  if (func->has<ImpureAttr>() || !func->has<ArgCountAttr>() || !func->has<ArgTypesAttr>())
    return RejectReason::Impure;
  if (func->get<ArgCountAttr>()->count != 2)
    return RejectReason::ArgShape;
  for (auto ty : func->get<ArgTypesAttr>()->types)
    if (ty != Value::i32)
      return RejectReason::Types;

  int opCount = 0;
  int selfCalls = 0;
  const auto &self = NAME(func);
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      opCount++;
      if (auto call = dyn_cast<CallOp>(op)) {
        if (NAME(call) != self)
          return RejectReason::Calls;
        selfCalls++;
      }
      if (isa<CloneOp>(op) || isa<JoinOp>(op) || isa<WakeOp>(op))
        return RejectReason::Calls;
    }
  }
  if (selfCalls == 0 || opCount > 260)
    return RejectReason::Ops;
  if (!hasOnlySelfCalls(func) || storesToGlobal(func))
    return storesToGlobal(func) ? RejectReason::Stores : RejectReason::Calls;

  bool hasI32Return = false;
  for (auto ret : func->findAll<ReturnOp>()) {
    if (ret->getOperandCount() == 0)
      return RejectReason::Return;
    if (ret->DEF()->getResultType() != Value::i32)
      return RejectReason::Return;
    hasI32Return = true;
  }
  return hasI32Return ? RejectReason::None : RejectReason::Return;
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

Value indexedAddress(Builder &builder, const std::string &globalName, Value index) {
  auto base = builder.create<GetGlobalOp>({ new NameAttr(globalName) });
  auto four = builder.create<IntOp>({ new IntAttr(4) });
  auto offset = builder.create<MulIOp>({ index, four });
  return builder.create<AddLOp>({ base, offset });
}

Value cacheIndex(Builder &builder, Value arg0, Value arg1, int mask) {
  auto c0 = builder.create<IntOp>({ new IntAttr(1103515245) });
  auto c1 = builder.create<IntOp>({ new IntAttr(-1640531535) });
  auto maskOp = builder.create<IntOp>({ new IntAttr(mask) });
  auto h0 = builder.create<MulIOp>({ arg0, c0 });
  auto h1 = builder.create<MulIOp>({ arg1, c1 });
  auto mix = builder.create<XorIOp>({ h0, h1 });
  return builder.create<AndIOp>({ mix, maskOp });
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

void addEntryCheck(FuncOp *func, const std::string &key0Name, const std::string &key1Name,
                   const std::string &valueName, const std::string &seenName,
                   const std::string &epochName, int mask, int &entryChecks) {
  auto args = orderedArgs(func, 2);
  assert(args.size() == 2);

  auto region = func->getRegion();
  auto entry = region->getFirstBlock();
  auto term = entry->getLastOp();
  BasicBlock *original = nullptr;
  bool replaceTerm = false;
  if (term && isa<GotoOp>(term)) {
    original = TARGET(term);
    replaceTerm = true;
  } else {
    original = entry->nextBlock();
  }
  if (!original)
    return;

  auto checkSeen = region->insertAfter(entry);
  auto checkKeys = region->insertAfter(checkSeen);
  auto hitBlock = region->insertAfter(checkKeys);

  Builder builder;
  if (replaceTerm)
    builder.setBeforeOp(term);
  else
    builder.setToBlockEnd(entry);
  Value index = cacheIndex(builder, args[0], args[1], mask);
  if (replaceTerm)
    builder.replace<GotoOp>(term, { new TargetAttr(checkSeen) });
  else
    builder.create<GotoOp>({ new TargetAttr(checkSeen) });

  builder.setToBlockEnd(checkSeen);
  auto seenAddr = indexedAddress(builder, seenName, index);
  auto seen = builder.create<LoadOp>(Value::i32, { seenAddr }, { new SizeAttr(4) });
  auto ep = builder.create<GetGlobalOp>({ new NameAttr(epochName) });
  auto epoch = builder.create<LoadOp>(Value::i32, { ep }, { new SizeAttr(4) });
  auto seenHit = builder.create<EqOp>(std::vector<Value>{ seen, epoch });
  builder.create<BranchOp>(std::vector<Value>{ seenHit },
                           { new TargetAttr(checkKeys), new ElseAttr(original) });

  builder.setToBlockEnd(checkKeys);
  auto key0Addr = indexedAddress(builder, key0Name, index);
  auto key1Addr = indexedAddress(builder, key1Name, index);
  auto key0 = builder.create<LoadOp>(Value::i32, { key0Addr }, { new SizeAttr(4) });
  auto key1 = builder.create<LoadOp>(Value::i32, { key1Addr }, { new SizeAttr(4) });
  auto eq0 = builder.create<EqOp>(std::vector<Value>{ key0, args[0] });
  auto eq1 = builder.create<EqOp>(std::vector<Value>{ key1, args[1] });
  auto both = builder.create<AndIOp>(std::vector<Value>{ eq0, eq1 });
  builder.create<BranchOp>(std::vector<Value>{ both },
                           { new TargetAttr(hitBlock), new ElseAttr(original) });

  builder.setToBlockEnd(hitBlock);
  auto valueAddr = indexedAddress(builder, valueName, index);
  auto cached = builder.create<LoadOp>(Value::i32, { valueAddr }, { new SizeAttr(4) });
  builder.create<ReturnOp>({ cached });
  entryChecks++;
}

void addReturnStores(FuncOp *func, const std::string &key0Name, const std::string &key1Name,
                     const std::string &valueName, const std::string &seenName,
                     const std::string &epochName, int mask, int &storesAdded) {
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
    builder.create<GotoOp>({ new TargetAttr(storeBlock) });

    builder.setToBlockEnd(storeBlock);
    Value index = cacheIndex(builder, args[0], args[1], mask);
    auto key0Addr = indexedAddress(builder, key0Name, index);
    auto key1Addr = indexedAddress(builder, key1Name, index);
    auto valueAddr = indexedAddress(builder, valueName, index);
    auto seenAddr = indexedAddress(builder, seenName, index);
    auto ep = builder.create<GetGlobalOp>({ new NameAttr(epochName) });
    auto epoch = builder.create<LoadOp>(Value::i32, { ep }, { new SizeAttr(4) });
    builder.create<StoreOp>({ args[0], key0Addr }, { new SizeAttr(4) });
    builder.create<StoreOp>({ args[1], key1Addr }, { new SizeAttr(4) });
    builder.create<StoreOp>({ retValue, valueAddr }, { new SizeAttr(4) });
    builder.create<StoreOp>({ epoch, seenAddr }, { new SizeAttr(4) });
    builder.create<GotoOp>({ new TargetAttr(finalBlock) });
    storesAdded++;
  }
}

} // namespace

std::map<std::string, int> RuntimeMemoize::stats() {
  return {
    { "candidates", candidates },
    { "memoized-functions", memoized },
    { "entry-checks", entryChecks },
    { "call-epoch-bumps", callEpochBumps },
    { "rejected-impure", rejectedImpure },
    { "rejected-arg-shape", rejectedArgShape },
    { "rejected-types", rejectedTypes },
    { "rejected-calls", rejectedCalls },
    { "rejected-stores", rejectedStores },
    { "rejected-ops", rejectedOps },
    { "rejected-return", rejectedReturn },
  };
}

void RuntimeMemoize::run() {
  if (!envEnabled("SISY_ENABLE_RUNTIME_MEMOIZE", true))
    return;

  int capacity = roundPow2(envInt("SISY_AUTO_MEMOIZE_CAPACITY", kDefaultCapacity));
  int mask = capacity - 1;

  CallGraph(module).run();
  auto funcs = collectFuncs();
  std::set<FuncOp*> candidates;
  for (auto func : funcs) {
    if (readsOnlyImmutableScalarGlobals(func, module)) {
      rejectedCalls++;
      continue;
    }
    switch (eligible(func)) {
    case RejectReason::None:
      candidates.insert(func);
      this->candidates++;
      break;
    case RejectReason::Impure:
      rejectedImpure++;
      break;
    case RejectReason::ArgShape:
      rejectedArgShape++;
      break;
    case RejectReason::Types:
      rejectedTypes++;
      break;
    case RejectReason::Calls:
      rejectedCalls++;
      break;
    case RejectReason::Stores:
      rejectedStores++;
      break;
    case RejectReason::Ops:
      rejectedOps++;
      break;
    case RejectReason::Return:
      rejectedReturn++;
      break;
    }
  }
  if (candidates.empty())
    return;

  for (auto func : candidates) {
    const auto &fname = NAME(func);
    std::string key0Name = symbolFor("__sisy_memo_k0_", fname);
    std::string key1Name = symbolFor("__sisy_memo_k1_", fname);
    std::string valueName = symbolFor("__sisy_memo_v_", fname);
    std::string seenName = symbolFor("__sisy_memo_seen_", fname);
    std::string epochName = symbolFor("__sisy_memo_epoch_", fname);
    createIntGlobal(module, key0Name, capacity);
    createIntGlobal(module, key1Name, capacity);
    createIntGlobal(module, valueName, capacity);
    createIntGlobal(module, seenName, capacity);
    createIntGlobal(module, epochName, 1, 1);

    for (auto call : module->findAll<CallOp>()) {
      if (NAME(call) != fname)
        continue;
      auto parent = call->getParentOp<FuncOp>();
      if (parent == func)
        continue;
      bumpEpochBefore(cast<CallOp>(call), epochName, callEpochBumps);
    }

    addEntryCheck(func, key0Name, key1Name, valueName, seenName, epochName, mask, entryChecks);
    int storesAdded = 0;
    addReturnStores(func, key0Name, key1Name, valueName, seenName, epochName,
                    mask, storesAdded);
    if (storesAdded > 0)
      memoized++;
  }
}
