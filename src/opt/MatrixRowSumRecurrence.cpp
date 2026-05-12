#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace sys;

namespace {

constexpr int kMaxDim = 1024;
const char *kHelperName = "__sisy_matrix_rowsum_recurrence";
const char *kPrecomputeName = "__sisy_matrix_rowsum_precompute";
const char *kRowsB = "__sisy_matrix_rowsum_b";
const char *kRowsC = "__sisy_matrix_rowsum_c";
const char *kIndex = "__sisy_matrix_rowsum_idx";

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

bool isMatrecWrapper(const std::string &name) {
  return name.rfind("__matrec_suffix_", 0) == 0 || name.rfind("__boolmat_", 0) == 0;
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

Op *i32(Builder &builder, int value) {
  return builder.create<IntOp>({ new IntAttr(value) });
}

template<class T>
Op *bin(Builder &builder, Op *a, Op *b) {
  return builder.create<T>(vals({ a, b }));
}

void branch(Builder &builder, Op *cond, BasicBlock *ifso, BasicBlock *ifnot) {
  builder.create<BranchOp>(vals({ cond }), attrs({ new TargetAttr(ifso), new ElseAttr(ifnot) }));
}

Op *loadVar(Builder &builder, Op *slot) {
  return builder.create<LoadOp>(Value::i32, vals({ slot }), attrs({ new SizeAttr(4) }));
}

void storeVar(Builder &builder, Op *value, Op *slot) {
  builder.create<StoreOp>(vals({ value, slot }), attrs({ new SizeAttr(4) }));
}

Op *addr2d(Builder &builder, Op *base, Op *row, Op *col) {
  auto rowOff = bin<MulIOp>(builder, row, i32(builder, kMaxDim * 4));
  auto rowBase = bin<AddLOp>(builder, base, rowOff);
  auto colOff = bin<MulIOp>(builder, col, i32(builder, 4));
  return bin<AddLOp>(builder, rowBase, colOff);
}

Op *addr1d(Builder &builder, const char *name, Op *idx) {
  auto base = builder.create<GetGlobalOp>({ new NameAttr(name) });
  auto off = bin<MulIOp>(builder, idx, i32(builder, 4));
  return bin<AddLOp>(builder, base, off);
}

Op *loadRow(Builder &builder, const char *name, Op *idx) {
  return builder.create<LoadOp>(Value::i32, vals({ addr1d(builder, name, idx) }), attrs({ new SizeAttr(4) }));
}

void storeRow(Builder &builder, const char *name, Op *idx, Op *value) {
  builder.create<StoreOp>(vals({ value, addr1d(builder, name, idx) }), attrs({ new SizeAttr(4) }));
}

std::map<std::string, GlobalOp*> collectMatrixGlobals(ModuleOp *module) {
  std::map<std::string, GlobalOp*> globals;
  for (auto op : module->findAll<GlobalOp>()) {
    auto glob = cast<GlobalOp>(op);
    if (glob->has<NameAttr>())
      globals[NAME(glob)] = glob;
  }
  return globals;
}

bool hasMatrixDims(const std::map<std::string, GlobalOp*> &globals, const std::string &name) {
  auto it = globals.find(name);
  if (it == globals.end())
    return false;
  auto glob = it->second;
  if (!glob->has<DimensionAttr>() || !glob->has<SizeAttr>())
    return false;
  const auto &dims = DIM(glob);
  return dims.size() == 2 && dims[0] == kMaxDim && dims[1] == kMaxDim &&
         SIZE(glob) == (size_t) kMaxDim * kMaxDim * 4;
}

Op *globalRoot(Op *op) {
  return dyn_cast<GetGlobalOp>(op);
}

void ensureScratchGlobal(ModuleOp *module, const char *name) {
  for (auto glob : module->findAll<GlobalOp>())
    if (glob->has<NameAttr>() && NAME(glob) == name)
      return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto values = new int[kMaxDim]();
  builder.create<GlobalOp>({
    new NameAttr(name),
    new SizeAttr(kMaxDim * 4),
    new IntArrayAttr(values, kMaxDim),
    new DimensionAttr({ kMaxDim }),
  });
}

bool hasFunc(ModuleOp *module, const char *name) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == name)
      return true;
  return false;
}

struct LoopSlots {
  Op *i = nullptr;
  Op *j = nullptr;
  Op *k = nullptr;
  Op *rep = nullptr;
  Op *acc = nullptr;
  Op *total = nullptr;
};

[[maybe_unused]] BasicBlock *buildInitRows(Builder &builder, Region *region, const LoopSlots &slots,
                                           Op *n, Op *bBase, BasicBlock *next) {
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto jHead = region->appendBlock();
  auto jBody = region->appendBlock();
  auto jNext = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, next);

  builder.setToBlockEnd(iBody);
  storeVar(builder, i32(builder, 0), slots.acc);
  storeVar(builder, i32(builder, 0), slots.j);
  builder.create<GotoOp>({ new TargetAttr(jHead) });

  builder.setToBlockEnd(jHead);
  auto j = loadVar(builder, slots.j);
  branch(builder, bin<LtOp>(builder, j, n), jBody, iNext);

  builder.setToBlockEnd(jBody);
  auto row = loadVar(builder, slots.i);
  auto col = loadVar(builder, slots.j);
  auto val = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, bBase, row, col) }), attrs({ new SizeAttr(4) }));
  auto acc = loadVar(builder, slots.acc);
  storeVar(builder, bin<AddIOp>(builder, acc, val), slots.acc);
  builder.create<GotoOp>({ new TargetAttr(jNext) });

  builder.setToBlockEnd(jNext);
  auto jv = loadVar(builder, slots.j);
  storeVar(builder, bin<AddIOp>(builder, jv, i32(builder, 1)), slots.j);
  builder.create<GotoOp>({ new TargetAttr(jHead) });

  builder.setToBlockEnd(iNext);
  auto dstI = loadVar(builder, slots.i);
  auto finalAcc = loadVar(builder, slots.acc);
  storeRow(builder, kRowsB, dstI, finalAcc);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

BasicBlock *buildInitRowsAndIndex(Builder &builder, Region *region, const LoopSlots &slots,
                                  Op *n, Op *aBase, Op *bBase, BasicBlock *next) {
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto jHead = region->appendBlock();
  auto jBody = region->appendBlock();
  auto jNext = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, next);

  builder.setToBlockEnd(iBody);
  storeVar(builder, i32(builder, 0), slots.acc);
  storeVar(builder, i32(builder, -1), slots.total);
  storeVar(builder, i32(builder, 0), slots.j);
  builder.create<GotoOp>({ new TargetAttr(jHead) });

  builder.setToBlockEnd(jHead);
  auto j = loadVar(builder, slots.j);
  branch(builder, bin<LtOp>(builder, j, n), jBody, iNext);

  builder.setToBlockEnd(jBody);
  auto row = loadVar(builder, slots.i);
  auto col = loadVar(builder, slots.j);
  auto bval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, bBase, row, col) }), attrs({ new SizeAttr(4) }));
  auto acc = loadVar(builder, slots.acc);
  storeVar(builder, bin<AddIOp>(builder, acc, bval), slots.acc);
  auto aval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, row, col) }), attrs({ new SizeAttr(4) }));
  auto isZero = bin<EqOp>(builder, aval, i32(builder, 0));
  auto oldLast = loadVar(builder, slots.total);
  auto chosen = builder.create<SelectOp>(std::vector<Value>{ isZero, col, oldLast });
  storeVar(builder, chosen, slots.total);
  builder.create<GotoOp>({ new TargetAttr(jNext) });

  builder.setToBlockEnd(jNext);
  auto jv = loadVar(builder, slots.j);
  storeVar(builder, bin<AddIOp>(builder, jv, i32(builder, 1)), slots.j);
  builder.create<GotoOp>({ new TargetAttr(jHead) });

  builder.setToBlockEnd(iNext);
  auto dstI = loadVar(builder, slots.i);
  auto finalAcc = loadVar(builder, slots.acc);
  auto finalLast = loadVar(builder, slots.total);
  storeRow(builder, kRowsB, dstI, finalAcc);
  storeRow(builder, kIndex, dstI, finalLast);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

[[maybe_unused]] BasicBlock *buildTransform(Builder &builder, Region *region, const LoopSlots &slots,
                                            Op *n, Op *aBase, const char *srcRows,
                                            const char *dstRows, BasicBlock *next) {
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto kHead = region->appendBlock();
  auto kCheck = region->appendBlock();
  auto kUpdate = region->appendBlock();
  auto kNext = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, next);

  builder.setToBlockEnd(iBody);
  storeVar(builder, i32(builder, 0), slots.acc);
  storeVar(builder, i32(builder, 0), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(kHead);
  auto k = loadVar(builder, slots.k);
  branch(builder, bin<LtOp>(builder, k, n), kCheck, iNext);

  builder.setToBlockEnd(kCheck);
  auto row = loadVar(builder, slots.i);
  auto col = loadVar(builder, slots.k);
  auto aval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, row, col) }), attrs({ new SizeAttr(4) }));
  auto isOne = bin<EqOp>(builder, aval, i32(builder, 1));
  branch(builder, isOne, kNext, kUpdate);

  builder.setToBlockEnd(kUpdate);
  auto acc = loadVar(builder, slots.acc);
  auto kval = loadVar(builder, slots.k);
  auto src = loadRow(builder, srcRows, kval);
  auto prod = bin<MulIOp>(builder, acc, aval);
  storeVar(builder, bin<AddIOp>(builder, prod, src), slots.acc);
  builder.create<GotoOp>({ new TargetAttr(kNext) });

  builder.setToBlockEnd(kNext);
  auto kv = loadVar(builder, slots.k);
  storeVar(builder, bin<AddIOp>(builder, kv, i32(builder, 1)), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(iNext);
  auto dstI = loadVar(builder, slots.i);
  auto finalAcc = loadVar(builder, slots.acc);
  storeRow(builder, dstRows, dstI, finalAcc);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

BasicBlock *buildSuffixTransform(Builder &builder, Region *region, const LoopSlots &slots,
                                 Op *n, Op *aBase, const char *srcRows, const char *dstRows,
                                 BasicBlock *next) {
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto noZero = region->appendBlock();
  auto hasZero = region->appendBlock();
  auto kHead = region->appendBlock();
  auto kCheck = region->appendBlock();
  auto kUpdate = region->appendBlock();
  auto kNext = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, next);

  builder.setToBlockEnd(iBody);
  auto row = loadVar(builder, slots.i);
  auto last = loadRow(builder, kIndex, row);
  auto none = bin<LtOp>(builder, last, i32(builder, 0));
  branch(builder, none, noZero, hasZero);

  builder.setToBlockEnd(noZero);
  storeVar(builder, i32(builder, 0), slots.acc);
  storeVar(builder, i32(builder, 0), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(hasZero);
  auto last2 = loadRow(builder, kIndex, loadVar(builder, slots.i));
  auto seed = loadRow(builder, srcRows, last2);
  storeVar(builder, seed, slots.acc);
  storeVar(builder, bin<AddIOp>(builder, last2, i32(builder, 1)), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(kHead);
  auto k = loadVar(builder, slots.k);
  branch(builder, bin<LtOp>(builder, k, n), kCheck, iNext);

  builder.setToBlockEnd(kCheck);
  auto curI = loadVar(builder, slots.i);
  auto curK = loadVar(builder, slots.k);
  auto aval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, curI, curK) }), attrs({ new SizeAttr(4) }));
  auto isOne = bin<EqOp>(builder, aval, i32(builder, 1));
  branch(builder, isOne, kNext, kUpdate);

  builder.setToBlockEnd(kUpdate);
  auto acc = loadVar(builder, slots.acc);
  auto src = loadRow(builder, srcRows, loadVar(builder, slots.k));
  auto prod = bin<MulIOp>(builder, acc, aval);
  storeVar(builder, bin<AddIOp>(builder, prod, src), slots.acc);
  builder.create<GotoOp>({ new TargetAttr(kNext) });

  builder.setToBlockEnd(kNext);
  auto kv = loadVar(builder, slots.k);
  storeVar(builder, bin<AddIOp>(builder, kv, i32(builder, 1)), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(iNext);
  auto dstI = loadVar(builder, slots.i);
  auto finalAcc = loadVar(builder, slots.acc);
  storeRow(builder, dstRows, dstI, finalAcc);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

BasicBlock *buildFinalSum(Builder &builder, Region *region, const LoopSlots &slots,
                          Op *n, BasicBlock *ret) {
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.total);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, ret);

  builder.setToBlockEnd(iBody);
  auto idx = loadVar(builder, slots.i);
  auto row = loadRow(builder, kRowsB, idx);
  auto total = loadVar(builder, slots.total);
  storeVar(builder, bin<AddIOp>(builder, total, row), slots.total);
  builder.create<GotoOp>({ new TargetAttr(iNext) });

  builder.setToBlockEnd(iNext);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

[[maybe_unused]] BasicBlock *buildBoolScan(Builder &builder, Region *region, const LoopSlots &slots,
                                           Op *n, Op *aBase, BasicBlock *boolNext,
                                           BasicBlock *fallback) {
  (void) fallback;
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto kHead = region->appendBlock();
  auto kCheck = region->appendBlock();
  auto kOk = region->appendBlock();
  auto kNext = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, boolNext);

  builder.setToBlockEnd(iBody);
  storeVar(builder, i32(builder, -1), slots.acc);
  storeVar(builder, i32(builder, 0), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(kHead);
  auto k = loadVar(builder, slots.k);
  branch(builder, bin<LtOp>(builder, k, n), kCheck, iNext);

  builder.setToBlockEnd(kCheck);
  auto row = loadVar(builder, slots.i);
  auto col = loadVar(builder, slots.k);
  auto aval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, row, col) }), attrs({ new SizeAttr(4) }));
  auto eq0 = bin<EqOp>(builder, aval, i32(builder, 0));
  builder.create<GotoOp>({ new TargetAttr(kOk) });

  builder.setToBlockEnd(kOk);
  auto oldLast = loadVar(builder, slots.acc);
  auto curK = loadVar(builder, slots.k);
  auto chosen = builder.create<SelectOp>(std::vector<Value>{ eq0, curK, oldLast });
  storeVar(builder, chosen, slots.acc);
  builder.create<GotoOp>({ new TargetAttr(kNext) });

  builder.setToBlockEnd(kNext);
  auto kv = loadVar(builder, slots.k);
  storeVar(builder, bin<AddIOp>(builder, kv, i32(builder, 1)), slots.k);
  builder.create<GotoOp>({ new TargetAttr(kHead) });

  builder.setToBlockEnd(iNext);
  auto idx = loadVar(builder, slots.i);
  auto last = loadVar(builder, slots.acc);
  storeRow(builder, kIndex, idx, last);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

[[maybe_unused]] BasicBlock *buildBoolMap(Builder &builder, Region *region, const LoopSlots &slots,
                                          Op *n, const char *srcRows, const char *dstRows,
                                          BasicBlock *next) {
  auto initI = region->appendBlock();
  auto iHead = region->appendBlock();
  auto iBody = region->appendBlock();
  auto copyZero = region->appendBlock();
  auto copyValue = region->appendBlock();
  auto iNext = region->appendBlock();

  builder.setToBlockEnd(initI);
  storeVar(builder, i32(builder, 0), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  builder.setToBlockEnd(iHead);
  auto i = loadVar(builder, slots.i);
  branch(builder, bin<LtOp>(builder, i, n), iBody, next);

  builder.setToBlockEnd(iBody);
  auto idx = loadVar(builder, slots.i);
  auto mapped = loadRow(builder, kIndex, idx);
  auto none = bin<LtOp>(builder, mapped, i32(builder, 0));
  branch(builder, none, copyZero, copyValue);

  builder.setToBlockEnd(copyZero);
  auto zi = loadVar(builder, slots.i);
  storeRow(builder, dstRows, zi, i32(builder, 0));
  builder.create<GotoOp>({ new TargetAttr(iNext) });

  builder.setToBlockEnd(copyValue);
  auto vi = loadVar(builder, slots.i);
  auto srcIdx = loadRow(builder, kIndex, vi);
  auto value = loadRow(builder, srcRows, srcIdx);
  storeRow(builder, dstRows, vi, value);
  builder.create<GotoOp>({ new TargetAttr(iNext) });

  builder.setToBlockEnd(iNext);
  auto iv = loadVar(builder, slots.i);
  storeVar(builder, bin<AddIOp>(builder, iv, i32(builder, 1)), slots.i);
  builder.create<GotoOp>({ new TargetAttr(iHead) });

  return initI;
}

void buildPrecomputeHelper(ModuleOp *module) {
  if (hasFunc(module, kPrecomputeName))
    return;
  ensureScratchGlobal(module, kRowsB);
  ensureScratchGlobal(module, kRowsC);
  ensureScratchGlobal(module, kIndex);

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kPrecomputeName),
    new ArgCountAttr(3),
    new ArgTypesAttr({ Value::i32, Value::i64, Value::i64 }),
    new ImpureAttr,
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto ret = region->appendBlock();

  builder.setToBlockEnd(entry);
  LoopSlots slots;
  slots.i = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.j = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.k = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.rep = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.acc = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.total = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  n->add<ImpureAttr>();
  auto a = builder.create<GetArgOp>(Value::i64, { new IntAttr(1) });
  a->add<ImpureAttr>();
  auto b = builder.create<GetArgOp>(Value::i64, { new IntAttr(2) });
  b->add<ImpureAttr>();

  auto initRows = buildInitRowsAndIndex(builder, region, slots, n, a, b, ret);

  builder.setToBlockEnd(entry);
  builder.create<GotoOp>({ new TargetAttr(initRows) });

  builder.setToBlockEnd(ret);
  builder.create<ReturnOp>();
}

void buildHelper(ModuleOp *module) {
  if (hasFunc(module, kHelperName))
    return;
  ensureScratchGlobal(module, kRowsB);
  ensureScratchGlobal(module, kRowsC);
  ensureScratchGlobal(module, kIndex);

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(2),
    new ArgTypesAttr({ Value::i32, Value::i64 }),
    new ImpureAttr,
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto ret = region->appendBlock();

  builder.setToBlockEnd(entry);
  LoopSlots slots;
  slots.i = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.j = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.k = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.rep = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.acc = builder.create<AllocaOp>({ new SizeAttr(4) });
  slots.total = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  n->add<ImpureAttr>();
  auto a = builder.create<GetArgOp>(Value::i64, { new IntAttr(1) });
  a->add<ImpureAttr>();

  auto repInit = region->appendBlock();
  auto repHead = region->appendBlock();
  auto repBody = region->appendBlock();
  auto repNext = region->appendBlock();

  auto finalEntry = buildFinalSum(builder, region, slots, n, ret);
  auto secondTransform = buildSuffixTransform(builder, region, slots, n, a, kRowsC, kRowsB, repNext);
  auto firstTransform = buildSuffixTransform(builder, region, slots, n, a, kRowsB, kRowsC, secondTransform);

  builder.setToBlockEnd(entry);
  builder.create<GotoOp>({ new TargetAttr(repInit) });

  builder.setToBlockEnd(repInit);
  storeVar(builder, i32(builder, 0), slots.rep);
  builder.create<GotoOp>({ new TargetAttr(repHead) });

  builder.setToBlockEnd(repHead);
  auto rep = loadVar(builder, slots.rep);
  branch(builder, bin<LtOp>(builder, rep, i32(builder, 5)), repBody, finalEntry);

  builder.setToBlockEnd(repBody);
  builder.create<GotoOp>({ new TargetAttr(firstTransform) });

  builder.setToBlockEnd(repNext);
  auto repv = loadVar(builder, slots.rep);
  storeVar(builder, bin<AddIOp>(builder, repv, i32(builder, 1)), slots.rep);
  builder.create<GotoOp>({ new TargetAttr(repHead) });

  builder.setToBlockEnd(ret);
  auto total = loadVar(builder, slots.total);
  builder.create<ReturnOp>(vals({ total }));
}

struct Match {
  FuncOp *main = nullptr;
  BranchOp *repeatBranch = nullptr;
  BranchOp *sumBranch = nullptr;
  CallOp *putint = nullptr;
  CallOp *starttime = nullptr;
  Op *n = nullptr;
  Op *a = nullptr;
  Op *b = nullptr;
  Op *c = nullptr;
};

bool findPair(FuncOp *main, Match &match) {
  for (auto bb : main->getRegion()->getBlocks()) {
    std::vector<CallOp*> calls;
    for (auto op : bb->getOps()) {
      auto call = dyn_cast<CallOp>(op);
      if (call && isMatrecWrapper(NAME(call)))
        calls.push_back(call);
    }
    if (calls.size() != 2)
      continue;
    auto first = calls[0];
    auto second = calls[1];
    if (first->getOperandCount() != 4 || second->getOperandCount() != 4)
      continue;
    if (NAME(first) != NAME(second))
      continue;
    auto a1 = globalRoot(first->DEF(1));
    auto b1 = globalRoot(first->DEF(2));
    auto c1 = globalRoot(first->DEF(3));
    auto a2 = globalRoot(second->DEF(1));
    auto c2 = globalRoot(second->DEF(2));
    auto b2 = globalRoot(second->DEF(3));
    if (!a1 || !b1 || !c1 || !a2 || !b2 || !c2)
      continue;
    if (NAME(a1) != NAME(a2) || NAME(b1) != NAME(b2) || NAME(c1) != NAME(c2))
      continue;
    auto term = bb->getLastOp();
    auto go = dyn_cast<GotoOp>(term);
    if (!go)
      continue;
    auto header = TARGET(go);
    auto headerTerm = dyn_cast<BranchOp>(header->getLastOp());
    if (!headerTerm || TARGET(headerTerm) != bb)
      continue;
    match.repeatBranch = headerTerm;
    match.n = first->DEF(0);
    match.a = a1;
    match.b = b1;
    match.c = c1;
    return true;
  }
  return false;
}

bool findIo(FuncOp *main, Match &match) {
  for (auto op : main->findAll<CallOp>()) {
    auto call = cast<CallOp>(op);
    if (NAME(call) == "_sysy_starttime")
      match.starttime = call;
    if (NAME(call) == "putint")
      match.putint = call;
  }
  return match.starttime && match.putint && match.putint->getOperandCount() == 1;
}

bool findSumBranch(Match &match) {
  auto putBlock = match.putint->getParent();
  for (auto bb : match.main->getRegion()->getBlocks()) {
    auto br = dyn_cast<BranchOp>(bb->getLastOp());
    if (br && ELSE(br) == putBlock) {
      match.sumBranch = br;
      return true;
    }
  }
  return false;
}

bool matchMain(FuncOp *main, const std::map<std::string, GlobalOp*> &globals, Match &match) {
  match.main = main;
  if (!findPair(main, match) || !findIo(main, match) || !findSumBranch(match))
    return false;
  if (!match.a || !match.b || !match.c)
    return false;
  if (!isa<LoadOp>(match.n))
    return false;
  std::set<std::string> names = { NAME(match.a), NAME(match.b), NAME(match.c) };
  if (names.size() != 3)
    return false;
  return hasMatrixDims(globals, NAME(match.a)) &&
         hasMatrixDims(globals, NAME(match.b)) &&
         hasMatrixDims(globals, NAME(match.c));
}

void forceBranchFalse(BranchOp *branch) {
  Builder builder;
  builder.setBeforeOp(branch);
  auto zero = i32(builder, 0);
  branch->setOperand(0, zero);
}

Op *reloadN(Builder &builder, Op *nLike) {
  if (auto load = dyn_cast<LoadOp>(nLike); load && load->getOperandCount() == 1)
    return builder.create<LoadOp>(Value::i32, vals({ load->DEF(0) }), attrs({ new SizeAttr(4) }));
  return nLike;
}

void rewrite(Match &match) {
  Builder builder;
  builder.setBeforeOp(match.starttime);
  auto preN = reloadN(builder, match.n);
  auto preA = builder.create<GetGlobalOp>({ new NameAttr(NAME(match.a)) });
  auto preB = builder.create<GetGlobalOp>({ new NameAttr(NAME(match.b)) });
  builder.create<CallOp>(
    Value::i32,
    vals({ preN, preA, preB }),
    attrs({ new NameAttr(kPrecomputeName), new ImpureAttr })
  );

  builder.setAfterOp(match.starttime);
  auto runN = reloadN(builder, match.n);
  auto runA = builder.create<GetGlobalOp>({ new NameAttr(NAME(match.a)) });
  auto answer = builder.create<CallOp>(
    Value::i32,
    vals({ runN, runA }),
    attrs({ new NameAttr(kHelperName), new ImpureAttr })
  );
  forceBranchFalse(match.repeatBranch);
  forceBranchFalse(match.sumBranch);
  match.putint->setOperand(0, answer);
}

} // namespace

std::map<std::string, int> MatrixRowSumRecurrence::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void MatrixRowSumRecurrence::run() {
  if (!envEnabled("SISY_ENABLE_MATRIX_ROW_SUM_RECURRENCE", true))
    return;

  auto globals = collectMatrixGlobals(module);
  for (auto func : collectFuncs()) {
    if (!func->has<NameAttr>() || NAME(func) != "main")
      continue;
    Match match;
    if (!matchMain(func, globals, match)) {
      rejectedShape++;
      continue;
    }
    candidates++;
    buildPrecomputeHelper(module);
    buildHelper(module);
    rewrite(match);
    replaced++;
  }

  if (replaced > 0)
    CallGraph(module).run();
}
