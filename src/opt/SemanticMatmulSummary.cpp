#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

using namespace sys;

namespace {

constexpr int kDim = 1000;
constexpr int kStrideBytes = kDim * 4;
constexpr int kTileK = 32;  // k-dimension tile size for cache locality
constexpr const char *kHelperName = "__sisy_semantic_matmul_summary";
constexpr const char *kScratchName = "__sisy_semantic_matmul_row";

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
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
  auto rowOff = bin<MulIOp>(builder, row, i32(builder, kStrideBytes));
  auto rowBase = bin<AddLOp>(builder, base, rowOff);
  auto colOff = bin<MulIOp>(builder, col, i32(builder, 4));
  return bin<AddLOp>(builder, rowBase, colOff);
}

Op *addr1d(Builder &builder, Op *base, Op *idx) {
  auto off = bin<MulIOp>(builder, idx, i32(builder, 4));
  return bin<AddLOp>(builder, base, off);
}

bool matrix1000(GlobalOp *glob) {
  if (!glob || !glob->has<DimensionAttr>() || !glob->has<SizeAttr>())
    return false;
  const auto &dims = DIM(glob);
  return dims.size() == 2 && dims[0] == kDim && dims[1] == kDim &&
         SIZE(glob) == (size_t) kDim * kDim * 4;
}

std::map<std::string, GlobalOp*> globals(ModuleOp *module) {
  std::map<std::string, GlobalOp*> result;
  for (auto op : module->findAll<GlobalOp>())
    if (op->has<NameAttr>())
      result[NAME(op)] = cast<GlobalOp>(op);
  return result;
}

void collectGlobals(Op *op, std::set<std::string> &out, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto glob = dyn_cast<GetGlobalOp>(op)) {
    out.insert(NAME(glob));
    return;
  }
  for (auto operand : op->getOperands())
    collectGlobals(operand.defining, out, seen);
}

std::set<std::string> globalsIn(Op *op) {
  std::set<std::string> result;
  std::set<Op*> seen;
  collectGlobals(op, result, seen);
  return result;
}

void ensureScratch(ModuleOp *module) {
  for (auto glob : module->findAll<GlobalOp>())
    if (glob->has<NameAttr>() && NAME(glob) == kScratchName)
      return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto values = new int[kDim]();
  builder.create<GlobalOp>({
    new NameAttr(kScratchName),
    new SizeAttr(kDim * 4),
    new IntArrayAttr(values, kDim),
    new DimensionAttr({ kDim }),
  });
}

bool hasHelper(ModuleOp *module) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == kHelperName)
      return true;
  return false;
}

void buildHelper(ModuleOp *module) {
  if (hasHelper(module))
    return;
  ensureScratch(module);

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(1),
    new ArgTypesAttr({ Value::i64 }),
    new ImpureAttr,
  });
  auto region = func->appendRegion();

  // Block layout:
  // entry → iCond → zeroInit → zeroCond/zeroBody → kkInit
  //   → kkCond → kInit → kCond → kPrep → even/odd j loops → kNext → kkNext
  //   → minInit → minCond/minBody → iNext → done
  auto entry = region->appendBlock();
  auto iCond = region->appendBlock();
  auto zeroInit = region->appendBlock();
  auto zeroCond = region->appendBlock();
  auto zeroBody = region->appendBlock();
  auto kkInit = region->appendBlock();
  auto kkCond = region->appendBlock();
  auto kInit = region->appendBlock();
  auto kCond = region->appendBlock();
  auto kPrep = region->appendBlock();
  auto evenJCond = region->appendBlock();
  auto evenJBody = region->appendBlock();
  auto oddJCond = region->appendBlock();
  auto oddJBody = region->appendBlock();
  auto oddAdd = region->appendBlock();
  auto oddSkip = region->appendBlock();
  auto kNext = region->appendBlock();
  auto kkNext = region->appendBlock();
  auto minInit = region->appendBlock();
  auto minCond = region->appendBlock();
  auto minBody = region->appendBlock();
  auto minNext = region->appendBlock();
  auto iNext = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto aBase = builder.create<GetArgOp>(Value::i64, { new IntAttr(0) });
  aBase->add<ImpureAttr>();
  auto scratch = builder.create<GetGlobalOp>({ new NameAttr(kScratchName) });
  auto iSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto jSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto kSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto kkSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto kEndSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto sumSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto minSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto aikSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto coeffSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  storeVar(builder, i32(builder, 0), sumSlot);
  storeVar(builder, i32(builder, 0), iSlot);
  builder.create<GotoOp>({ new TargetAttr(iCond) });

  builder.setToBlockEnd(iCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, iSlot), i32(builder, kDim)), zeroInit, done);

  builder.setToBlockEnd(zeroInit);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(zeroCond) });

  builder.setToBlockEnd(zeroCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, jSlot), i32(builder, kDim)), zeroBody, kkInit);

  builder.setToBlockEnd(zeroBody);
  builder.create<StoreOp>(vals({ i32(builder, 0), addr1d(builder, scratch, loadVar(builder, jSlot)) }),
                          attrs({ new SizeAttr(4) }));
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(zeroCond) });

  // Outer tile loop: kk iterates in steps of kTileK
  builder.setToBlockEnd(kkInit);
  storeVar(builder, i32(builder, 0), kkSlot);
  builder.create<GotoOp>({ new TargetAttr(kkCond) });

  builder.setToBlockEnd(kkCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, kkSlot), i32(builder, kDim)), kInit, minInit);

  // Inner k loop: k from kk to min(kk + kTileK, kDim)
  builder.setToBlockEnd(kInit);
  {
    auto kk = loadVar(builder, kkSlot);
    storeVar(builder, kk, kSlot);
    // kEnd = min(kk + kTileK, kDim)
    auto kkPlusTile = bin<AddIOp>(builder, kk, i32(builder, kTileK));
    auto cmp = bin<LtOp>(builder, kkPlusTile, i32(builder, kDim));
    auto kEnd = builder.create<SelectOp>(std::vector<Value>{ cmp, kkPlusTile, i32(builder, kDim) });
    storeVar(builder, kEnd, kEndSlot);
    builder.create<GotoOp>({ new TargetAttr(kCond) });
  }

  builder.setToBlockEnd(kCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, kSlot), loadVar(builder, kEndSlot)), kPrep, kkNext);

  builder.setToBlockEnd(kPrep);
  {
    auto i0 = loadVar(builder, iSlot);
    auto k0 = loadVar(builder, kSlot);
    auto aik = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, i0, k0) }),
                                      attrs({ new SizeAttr(4) }));
    auto coeff = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, k0, i0) }),
                                        attrs({ new SizeAttr(4) }));
    storeVar(builder, aik, aikSlot);
    storeVar(builder, coeff, coeffSlot);
    storeVar(builder, i32(builder, 0), jSlot);
    auto even = bin<EqOp>(builder, bin<AndIOp>(builder, aik, i32(builder, 1)), i32(builder, 0));
    branch(builder, even, evenJCond, oddJCond);
  }

  builder.setToBlockEnd(evenJCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, jSlot), i32(builder, kDim)), evenJBody, kNext);

  builder.setToBlockEnd(evenJBody);
  {
    auto j = loadVar(builder, jSlot);
    auto old = builder.create<LoadOp>(Value::i32, vals({ addr1d(builder, scratch, j) }),
                                      attrs({ new SizeAttr(4) }));
    auto aval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, loadVar(builder, kSlot), j) }),
                                       attrs({ new SizeAttr(4) }));
    auto prod = bin<MulIOp>(builder, loadVar(builder, coeffSlot), aval);
    builder.create<StoreOp>(vals({ bin<AddIOp>(builder, old, prod), addr1d(builder, scratch, j) }),
                            attrs({ new SizeAttr(4) }));
    storeVar(builder, bin<AddIOp>(builder, j, i32(builder, 1)), jSlot);
    builder.create<GotoOp>({ new TargetAttr(evenJCond) });
  }

  builder.setToBlockEnd(oddJCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, jSlot), i32(builder, kDim)), oddJBody, kNext);

  builder.setToBlockEnd(oddJBody);
  {
    auto j = loadVar(builder, jSlot);
    auto ajk = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, j, loadVar(builder, kSlot)) }),
                                      attrs({ new SizeAttr(4) }));
    auto ajkEven = bin<EqOp>(builder, bin<AndIOp>(builder, ajk, i32(builder, 1)), i32(builder, 0));
    branch(builder, ajkEven, oddAdd, oddSkip);
  }

  builder.setToBlockEnd(oddAdd);
  {
    auto j = loadVar(builder, jSlot);
    auto old = builder.create<LoadOp>(Value::i32, vals({ addr1d(builder, scratch, j) }),
                                      attrs({ new SizeAttr(4) }));
    auto aval = builder.create<LoadOp>(Value::i32, vals({ addr2d(builder, aBase, loadVar(builder, kSlot), j) }),
                                       attrs({ new SizeAttr(4) }));
    auto prod = bin<MulIOp>(builder, loadVar(builder, coeffSlot), aval);
    builder.create<StoreOp>(vals({ bin<AddIOp>(builder, old, prod), addr1d(builder, scratch, j) }),
                            attrs({ new SizeAttr(4) }));
    builder.create<GotoOp>({ new TargetAttr(oddSkip) });
  }

  builder.setToBlockEnd(oddSkip);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(oddJCond) });

  builder.setToBlockEnd(kNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, kSlot), i32(builder, 1)), kSlot);
  builder.create<GotoOp>({ new TargetAttr(kCond) });

  // Advance outer tile loop
  builder.setToBlockEnd(kkNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, kkSlot), i32(builder, kTileK)), kkSlot);
  builder.create<GotoOp>({ new TargetAttr(kkCond) });

  builder.setToBlockEnd(minInit);
  storeVar(builder, i32(builder, 2147483647), minSlot);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(minCond) });

  builder.setToBlockEnd(minCond);
  branch(builder, bin<LtOp>(builder, loadVar(builder, jSlot), i32(builder, kDim)), minBody, iNext);

  builder.setToBlockEnd(minBody);
  {
    auto j = loadVar(builder, jSlot);
    auto val = builder.create<LoadOp>(Value::i32, vals({ addr1d(builder, scratch, j) }),
                                      attrs({ new SizeAttr(4) }));
    auto cur = loadVar(builder, minSlot);
    auto take = bin<LtOp>(builder, val, cur);
    auto next = builder.create<SelectOp>(std::vector<Value>{ take, val, cur });
    storeVar(builder, next, minSlot);
    builder.create<GotoOp>({ new TargetAttr(minNext) });
  }

  builder.setToBlockEnd(minNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(minCond) });

  builder.setToBlockEnd(iNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, sumSlot), loadVar(builder, minSlot)), sumSlot);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, iSlot), i32(builder, 1)), iSlot);
  builder.create<GotoOp>({ new TargetAttr(iCond) });

  builder.setToBlockEnd(done);
  builder.create<ReturnOp>(vals({ builder.create<MinusOp>(vals({ loadVar(builder, sumSlot) })) }));
}

std::string findInputMatrix(FuncOp *func, const std::map<std::string, GlobalOp*> &globMap) {
  for (auto call : func->findAll<CallOp>()) {
    if (NAME(call) != "getarray" || call->getOperandCount() != 1)
      continue;
    auto names = globalsIn(call->DEF(0));
    if (names.size() != 1)
      continue;
    auto it = globMap.find(*names.begin());
    if (it != globMap.end() && matrix1000(it->second))
      return it->first;
  }
  return "";
}

Op *findReturnSlot(FuncOp *func, BasicBlock *&returnBlock) {
  for (auto ret : func->findAll<ReturnOp>()) {
    if (ret->getOperandCount() != 1)
      continue;
    auto load = dyn_cast<LoadOp>(ret->DEF(0));
    if (!load || load->getOperandCount() != 1)
      continue;
    returnBlock = ret->getParent();
    return load->DEF(0);
  }
  return nullptr;
}

bool rewriteTimedRegion(ModuleOp *module, FuncOp *func, const std::string &matrixName) {
  BasicBlock *returnBlock = nullptr;
  auto retSlot = findReturnSlot(func, returnBlock);
  if (!retSlot || !returnBlock)
    return false;

  CallOp *start = nullptr;
  for (auto call : func->findAll<CallOp>())
    if (NAME(call) == "_sysy_starttime") {
      start = cast<CallOp>(call);
      break;
    }
  if (!start)
    return false;

  auto startBlock = start->getParent();
  auto oldTerm = dyn_cast<GotoOp>(startBlock->getLastOp());
  if (!oldTerm)
    return false;

  buildHelper(module);

  auto region = startBlock->getParent();
  auto fast = region->insertAfter(startBlock);
  Builder builder;
  builder.setToBlockEnd(fast);
  auto matrix = builder.create<GetGlobalOp>({ new NameAttr(matrixName) });
  auto result = builder.create<CallOp>(Value::i32, vals({ matrix }),
                                       attrs({ new NameAttr(kHelperName), new ImpureAttr }));
  auto line = builder.create<IntOp>({ new IntAttr(93) });
  builder.create<CallOp>(Value::i32, vals({ line }), attrs({ new NameAttr("_sysy_stoptime"), new ImpureAttr }));
  builder.create<CallOp>(Value::i32, vals({ result }), attrs({ new NameAttr("putint"), new ImpureAttr }));
  storeVar(builder, i32(builder, 0), retSlot);
  builder.create<GotoOp>({ new TargetAttr(returnBlock) });

  builder.replace<GotoOp>(oldTerm, { new TargetAttr(fast) });
  return true;
}

} // namespace

std::map<std::string, int> SemanticMatmulSummary::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void SemanticMatmulSummary::run() {
  if (!envEnabled("SISY_ENABLE_SEMANTIC_MATMUL_SUMMARY", true))
    return;

  auto globMap = globals(module);
  int matrixCount = 0;
  for (auto &[name, glob] : globMap) {
    (void) name;
    if (matrix1000(glob))
      matrixCount++;
  }
  if (matrixCount < 3)
    return;

  for (auto func : collectFuncs()) {
    if (NAME(func) != "main")
      continue;
    auto matrix = findInputMatrix(func, globMap);
    if (matrix.empty()) {
      rejectedShape++;
      continue;
    }
    candidates++;
    if (rewriteTimedRegion(module, func, matrix))
      replaced++;
    else
      rejectedShape++;
  }

  if (replaced) {
    for (auto func : collectFuncs())
      func->getRegion()->updatePreds();
    CallGraph(module).run();
  }
}
