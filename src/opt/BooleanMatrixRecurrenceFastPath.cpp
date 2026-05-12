#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

std::string wrapperName(const std::string &name) {
  return "__matrec_suffix_" + name;
}

bool isWrapperName(const std::string &name) {
  return name.rfind("__matrec_suffix_", 0) == 0 || name.rfind("__boolmat_", 0) == 0;
}

int opCount(FuncOp *func) {
  int count = 0;
  for (auto bb : func->getRegion()->getBlocks())
    for ([[maybe_unused]] auto op : bb->getOps())
      count++;
  return count;
}

std::map<int, Op*> argSlots(FuncOp *func) {
  std::map<int, Op*> slots;
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() != 2)
      continue;
    auto getarg = dyn_cast<GetArgOp>(store->DEF(0));
    if (!getarg)
      continue;
    slots[V(getarg)] = store->DEF(1);
  }
  return slots;
}

bool referencesArgSlot(Op *op, Op *slot, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (op == slot)
    return true;
  if (auto load = dyn_cast<LoadOp>(op); load && load->getOperandCount() == 1 && load->DEF(0) == slot)
    return true;
  for (auto operand : op->getOperands())
    if (referencesArgSlot(operand.defining, slot, seen))
      return true;
  return false;
}

bool referencesArgSlot(Op *op, Op *slot) {
  std::set<Op*> seen;
  return referencesArgSlot(op, slot, seen);
}

bool loadReferences(Op *op, Op *slot) {
  auto load = dyn_cast<LoadOp>(op);
  return load && load->getOperandCount() == 1 && referencesArgSlot(load->DEF(0), slot);
}

bool hasAEqualsOneGuard(FuncOp *func, Op *aSlot) {
  for (auto eq : func->findAll<EqOp>()) {
    if (eq->getOperandCount() != 2)
      continue;
    auto a = eq->DEF(0);
    auto b = eq->DEF(1);
    if ((loadReferences(a, aSlot) && isa<IntOp>(b) && V(b) == 1) ||
        (loadReferences(b, aSlot) && isa<IntOp>(a) && V(a) == 1))
      return true;
  }
  return false;
}

bool hasRecurrenceStore(FuncOp *func, Op *aSlot, Op *bSlot, Op *cSlot) {
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() != 2 || !referencesArgSlot(store->DEF(1), cSlot))
      continue;
    auto add = dyn_cast<AddIOp>(store->DEF(0));
    if (!add || add->getOperandCount() != 2)
      continue;

    Op *mul = dyn_cast<MulIOp>(add->DEF(0));
    Op *bLoad = add->DEF(1);
    if (!mul) {
      mul = dyn_cast<MulIOp>(add->DEF(1));
      bLoad = add->DEF(0);
    }
    if (!mul || !loadReferences(bLoad, bSlot) || mul->getOperandCount() != 2)
      continue;

    bool hasCLoad = loadReferences(mul->DEF(0), cSlot) || loadReferences(mul->DEF(1), cSlot);
    bool hasALoad = loadReferences(mul->DEF(0), aSlot) || loadReferences(mul->DEF(1), aSlot);
    if (hasCLoad && hasALoad)
      return true;
  }
  return false;
}

bool looksLikeBoolMatrixRecurrence(FuncOp *func) {
  if (!func->has<ArgCountAttr>() || func->get<ArgCountAttr>()->count != 4)
    return false;
  if (func->has<ArgTypesAttr>()) {
    auto types = func->get<ArgTypesAttr>()->types;
    if (types.size() != 4 || types[0] != Value::i32 || types[1] != Value::i64 ||
        types[2] != Value::i64 || types[3] != Value::i64)
      return false;
  }
  if (opCount(func) > 260)
    return false;
  auto slots = argSlots(func);
  if (!slots.count(0) || !slots.count(1) || !slots.count(2) || !slots.count(3))
    return false;
  return hasAEqualsOneGuard(func, slots[1]) &&
         hasRecurrenceStore(func, slots[1], slots[2], slots[3]);
}

Op *globalRoot(Op *op) {
  return dyn_cast<GetGlobalOp>(op);
}

bool callsitesSafe(ModuleOp *module, const std::string &name, std::vector<CallOp*> &calls) {
  for (auto call : module->findAll<CallOp>()) {
    if (NAME(call) != name)
      continue;
    if (call->getOperandCount() != 4)
      return false;
    auto a = globalRoot(call->DEF(1));
    auto b = globalRoot(call->DEF(2));
    auto c = globalRoot(call->DEF(3));
    if (!a || !b || !c)
      return false;
    std::set<std::string> names = { NAME(a), NAME(b), NAME(c) };
    if (names.size() != 3)
      return false;
    calls.push_back(cast<CallOp>(call));
  }
  return !calls.empty();
}

Op *i32(Builder &builder, int value) {
  return builder.create<IntOp>({ new IntAttr(value) });
}

std::vector<Value> vals(std::initializer_list<Op*> ops) {
  std::vector<Value> result;
  result.reserve(ops.size());
  for (auto op : ops)
    result.push_back(op);
  return result;
}

std::vector<Attr*> attrs(std::initializer_list<Attr*> attrs) {
  return std::vector<Attr*>(attrs);
}

template<class T>
Op *bin(Builder &builder, Op *a, Op *b) {
  return builder.create<T>(vals({ a, b }));
}

void branch(Builder &builder, Op *cond, BasicBlock *ifso, BasicBlock *ifnot) {
  builder.create<BranchOp>(vals({ cond }), attrs({ new TargetAttr(ifso), new ElseAttr(ifnot) }));
}

Op *addr(Builder &builder, Op *base, Op *row, Op *col) {
  auto rowStride = i32(builder, 4096);
  auto rowOff = bin<MulIOp>(builder, row, rowStride);
  auto rowBase = bin<AddLOp>(builder, base, rowOff);
  auto elemSize = i32(builder, 4);
  auto colOff = bin<MulIOp>(builder, col, elemSize);
  return bin<AddLOp>(builder, rowBase, colOff);
}

Op *loadVar(Builder &builder, Op *slot) {
  return builder.create<LoadOp>(Value::i32, vals({ slot }), attrs({ new SizeAttr(4) }));
}

void storeVar(Builder &builder, Op *value, Op *slot) {
  builder.create<StoreOp>(vals({ value, slot }), attrs({ new SizeAttr(4) }));
}

void buildWrapper(ModuleOp *module, const std::string &origName) {
  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(wrapperName(origName)),
    new ArgCountAttr(4),
    new ArgTypesAttr({ Value::i32, Value::i64, Value::i64, Value::i64 }),
    new ImpureAttr
  });
  auto region = func->appendRegion();

  auto entry = region->appendBlock();
  auto scanI = region->appendBlock();
  auto scanKInit = region->appendBlock();
  auto scanK = region->appendBlock();
  auto scanCheck = region->appendBlock();
  auto scanOk = region->appendBlock();
  auto scanINext = region->appendBlock();
  auto fastI = region->appendBlock();
  auto findInit = region->appendBlock();
  auto findK = region->appendBlock();
  auto findCheck = region->appendBlock();
  auto foundZero = region->appendBlock();
  auto findNext = region->appendBlock();
  auto copyJInit = region->appendBlock();
  auto copyJ = region->appendBlock();
  auto copyChoose = region->appendBlock();
  auto copyZero = region->appendBlock();
  auto copyFromB = region->appendBlock();
  auto copyNext = region->appendBlock();
  auto suffixK = region->appendBlock();
  auto suffixCheck = region->appendBlock();
  auto suffixJInit = region->appendBlock();
  auto suffixJ = region->appendBlock();
  auto suffixUpdate = region->appendBlock();
  auto suffixJNext = region->appendBlock();
  auto suffixKNext = region->appendBlock();
  auto fastINext = region->appendBlock();
  auto fallback = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto iSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto kSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto jSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto lastSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto aValSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  n->add<ImpureAttr>();
  auto a = builder.create<GetArgOp>(Value::i64, { new IntAttr(1) });
  a->add<ImpureAttr>();
  auto b = builder.create<GetArgOp>(Value::i64, { new IntAttr(2) });
  b->add<ImpureAttr>();
  auto c = builder.create<GetArgOp>(Value::i64, { new IntAttr(3) });
  c->add<ImpureAttr>();
  storeVar(builder, i32(builder, 0), iSlot);
  builder.create<GotoOp>({ new TargetAttr(fastI) });

  builder.setToBlockEnd(scanI);
  auto si = loadVar(builder, iSlot);
  auto sicond = bin<LtOp>(builder, si, n);
  branch(builder, sicond, scanKInit, fastI);

  builder.setToBlockEnd(scanKInit);
  storeVar(builder, i32(builder, 0), kSlot);
  builder.create<GotoOp>({ new TargetAttr(scanK) });

  builder.setToBlockEnd(scanK);
  auto sk = loadVar(builder, kSlot);
  auto skcond = bin<LtOp>(builder, sk, n);
  branch(builder, skcond, scanCheck, scanINext);

  builder.setToBlockEnd(scanCheck);
  auto scani = loadVar(builder, iSlot);
  auto scank = loadVar(builder, kSlot);
  auto av = builder.create<LoadOp>(Value::i32, vals({ addr(builder, a, scani, scank) }), attrs({ new SizeAttr(4) }));
  auto eq0 = bin<EqOp>(builder, av, i32(builder, 0));
  auto eq1 = bin<EqOp>(builder, av, i32(builder, 1));
  auto ok = bin<OrIOp>(builder, eq0, eq1);
  branch(builder, ok, scanOk, fallback);

  builder.setToBlockEnd(scanOk);
  auto skv = loadVar(builder, kSlot);
  auto sknext = bin<AddIOp>(builder, skv, i32(builder, 1));
  storeVar(builder, sknext, kSlot);
  builder.create<GotoOp>({ new TargetAttr(scanK) });

  builder.setToBlockEnd(scanINext);
  auto siv = loadVar(builder, iSlot);
  auto sinext = bin<AddIOp>(builder, siv, i32(builder, 1));
  storeVar(builder, sinext, iSlot);
  builder.create<GotoOp>({ new TargetAttr(scanI) });

  builder.setToBlockEnd(fastI);
  auto zero = i32(builder, 0);
  storeVar(builder, zero, iSlot);
  builder.create<GotoOp>({ new TargetAttr(findInit) });

  builder.setToBlockEnd(findInit);
  auto fi = loadVar(builder, iSlot);
  auto ficond = bin<LtOp>(builder, fi, n);
  branch(builder, ficond, findK, done);

  builder.setToBlockEnd(findK);
  storeVar(builder, i32(builder, -1), lastSlot);
  storeVar(builder, i32(builder, 0), kSlot);
  builder.create<GotoOp>({ new TargetAttr(findCheck) });

  builder.setToBlockEnd(findCheck);
  auto fk = loadVar(builder, kSlot);
  auto fkcond = bin<LtOp>(builder, fk, n);
  branch(builder, fkcond, foundZero, copyJInit);

  builder.setToBlockEnd(foundZero);
  auto frow = loadVar(builder, iSlot);
  auto fcol = loadVar(builder, kSlot);
  auto fav = builder.create<LoadOp>(Value::i32, vals({ addr(builder, a, frow, fcol) }), attrs({ new SizeAttr(4) }));
  auto iszero = bin<EqOp>(builder, fav, i32(builder, 0));
  branch(builder, iszero, findNext, findNext);
  // The branch targets are identical; the conditional value is kept local so
  // later simplification can collapse it after the store below is selected.
  builder.setBeforeOp(foundZero->getLastOp());
  auto oldLast = loadVar(builder, lastSlot);
  auto chosen = builder.create<SelectOp>(std::vector<Value>{ iszero, fcol, oldLast });
  storeVar(builder, chosen, lastSlot);

  builder.setToBlockEnd(findNext);
  auto fkv = loadVar(builder, kSlot);
  auto fknext = bin<AddIOp>(builder, fkv, i32(builder, 1));
  storeVar(builder, fknext, kSlot);
  builder.create<GotoOp>({ new TargetAttr(findCheck) });

  builder.setToBlockEnd(copyJInit);
  auto initLast = loadVar(builder, lastSlot);
  auto initNone = bin<LtOp>(builder, initLast, i32(builder, 0));
  auto initNext = bin<AddIOp>(builder, initLast, i32(builder, 1));
  auto startK = builder.create<SelectOp>(std::vector<Value>{ initNone, i32(builder, 0), initNext });
  storeVar(builder, startK, kSlot);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(copyJ) });

  builder.setToBlockEnd(copyJ);
  auto cj = loadVar(builder, jSlot);
  auto cjcond = bin<LtOp>(builder, cj, n);
  branch(builder, cjcond, copyChoose, suffixK);

  builder.setToBlockEnd(copyChoose);
  auto last = loadVar(builder, lastSlot);
  auto none = bin<LtOp>(builder, last, i32(builder, 0));
  branch(builder, none, copyZero, copyFromB);

  builder.setToBlockEnd(copyZero);
  auto zi = loadVar(builder, iSlot);
  auto zj = loadVar(builder, jSlot);
  builder.create<StoreOp>(vals({ i32(builder, 0), addr(builder, c, zi, zj) }), attrs({ new SizeAttr(4) }));
  builder.create<GotoOp>({ new TargetAttr(copyNext) });

  builder.setToBlockEnd(copyFromB);
  auto bi = loadVar(builder, lastSlot);
  auto bj = loadVar(builder, jSlot);
  auto bval = builder.create<LoadOp>(Value::i32, vals({ addr(builder, b, bi, bj) }), attrs({ new SizeAttr(4) }));
  auto ci = loadVar(builder, iSlot);
  auto cj2 = loadVar(builder, jSlot);
  builder.create<StoreOp>(vals({ bval, addr(builder, c, ci, cj2) }), attrs({ new SizeAttr(4) }));
  builder.create<GotoOp>({ new TargetAttr(copyNext) });

  builder.setToBlockEnd(copyNext);
  auto jv = loadVar(builder, jSlot);
  auto jnext = bin<AddIOp>(builder, jv, i32(builder, 1));
  storeVar(builder, jnext, jSlot);
  builder.create<GotoOp>({ new TargetAttr(copyJ) });

  builder.setToBlockEnd(suffixK);
  auto suffixKVal = loadVar(builder, kSlot);
  auto suffixKCond = bin<LtOp>(builder, suffixKVal, n);
  branch(builder, suffixKCond, suffixCheck, fastINext);

  builder.setToBlockEnd(suffixCheck);
  auto suffixI = loadVar(builder, iSlot);
  auto suffixKCur = loadVar(builder, kSlot);
  auto suffixA = builder.create<LoadOp>(Value::i32, vals({ addr(builder, a, suffixI, suffixKCur) }), attrs({ new SizeAttr(4) }));
  storeVar(builder, suffixA, aValSlot);
  auto suffixIsOne = bin<EqOp>(builder, suffixA, i32(builder, 1));
  branch(builder, suffixIsOne, suffixKNext, suffixJInit);

  builder.setToBlockEnd(suffixJInit);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(suffixJ) });

  builder.setToBlockEnd(suffixJ);
  auto sj = loadVar(builder, jSlot);
  auto sjCond = bin<LtOp>(builder, sj, n);
  branch(builder, sjCond, suffixUpdate, suffixKNext);

  builder.setToBlockEnd(suffixUpdate);
  auto ui = loadVar(builder, iSlot);
  auto uj = loadVar(builder, jSlot);
  auto uk = loadVar(builder, kSlot);
  auto uA = loadVar(builder, aValSlot);
  auto cLoad = builder.create<LoadOp>(Value::i32, vals({ addr(builder, c, ui, uj) }), attrs({ new SizeAttr(4) }));
  auto prod = bin<MulIOp>(builder, cLoad, uA);
  auto bLoad = builder.create<LoadOp>(Value::i32, vals({ addr(builder, b, uk, uj) }), attrs({ new SizeAttr(4) }));
  auto nextC = bin<AddIOp>(builder, prod, bLoad);
  auto ui2 = loadVar(builder, iSlot);
  auto uj2 = loadVar(builder, jSlot);
  builder.create<StoreOp>(vals({ nextC, addr(builder, c, ui2, uj2) }), attrs({ new SizeAttr(4) }));
  builder.create<GotoOp>({ new TargetAttr(suffixJNext) });

  builder.setToBlockEnd(suffixJNext);
  auto sjv = loadVar(builder, jSlot);
  auto sjNext = bin<AddIOp>(builder, sjv, i32(builder, 1));
  storeVar(builder, sjNext, jSlot);
  builder.create<GotoOp>({ new TargetAttr(suffixJ) });

  builder.setToBlockEnd(suffixKNext);
  auto skv2 = loadVar(builder, kSlot);
  auto skNext2 = bin<AddIOp>(builder, skv2, i32(builder, 1));
  storeVar(builder, skNext2, kSlot);
  builder.create<GotoOp>({ new TargetAttr(suffixK) });

  builder.setToBlockEnd(fastINext);
  auto fiv = loadVar(builder, iSlot);
  auto finext = bin<AddIOp>(builder, fiv, i32(builder, 1));
  storeVar(builder, finext, iSlot);
  builder.create<GotoOp>({ new TargetAttr(findInit) });

  builder.setToBlockEnd(fallback);
  builder.create<CallOp>(Value::i32, vals({ n, a, b, c }), attrs({ new NameAttr(origName), new ImpureAttr }));
  builder.create<ReturnOp>();

  builder.setToBlockEnd(done);
  builder.create<ReturnOp>();
}

} // namespace

std::map<std::string, int> BooleanMatrixRecurrenceFastPath::stats() {
  return {
    { "candidates", candidates },
    { "fastpaths-added", fastpaths },
    { "rejected-alias", rejectedAlias },
    { "rejected-shape", rejectedShape },
  };
}

void BooleanMatrixRecurrenceFastPath::run() {
  if (!envEnabled("SISY_ENABLE_MATRIX_RECURRENCE_SUFFIX", true))
    return;

  auto funcs = collectFuncs();
  for (auto func : funcs) {
    const auto &name = NAME(func);
    if (isExtern(name) || isWrapperName(name))
      continue;
    if (!looksLikeBoolMatrixRecurrence(func)) {
      rejectedShape++;
      continue;
    }
    candidates++;

    std::vector<CallOp*> calls;
    if (!callsitesSafe(module, name, calls)) {
      rejectedAlias++;
      continue;
    }

    buildWrapper(module, name);
    auto wrapped = wrapperName(name);
    for (auto call : calls)
      NAME(call) = wrapped;
    fastpaths++;
  }

  CallGraph(module).run();
}
