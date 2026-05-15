#include "Passes.h"
#include "LoopPasses.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

const char *kHelperName = "__row_scratch_matmul_generic";

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

std::vector<Value> vals(std::initializer_list<Op*> ops) {
  std::vector<Value> result;
  result.reserve(ops.size());
  for (auto op : ops)
    result.push_back(op);
  return result;
}

std::vector<Attr*> attrs(std::initializer_list<Attr*> xs) {
  return std::vector<Attr*>(xs);
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

Op *rowAddr(Builder &builder, Op *base, Op *col) {
  auto elemSize = i32(builder, 4);
  auto colOff = bin<MulIOp>(builder, col, elemSize);
  return bin<AddLOp>(builder, base, colOff);
}

Op *matrixAddr(Builder &builder, Op *base, Op *row, Op *col, Op *rowStrideBytes) {
  auto rowOff = bin<MulIOp>(builder, row, rowStrideBytes);
  auto rowBase = bin<AddLOp>(builder, base, rowOff);
  auto elemSize = i32(builder, 4);
  auto colOff = bin<MulIOp>(builder, col, elemSize);
  return bin<AddLOp>(builder, rowBase, colOff);
}

void collectGlobals(Op *op, std::set<std::string> &names, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto global = dyn_cast<GetGlobalOp>(op)) {
    names.insert(NAME(global));
    return;
  }
  for (auto operand : op->getOperands())
    collectGlobals(operand.defining, names, seen);
}

std::set<std::string> globalsIn(Op *op) {
  std::set<std::string> names;
  std::set<Op*> seen;
  collectGlobals(op, names, seen);
  return names;
}

bool usesValue(Op *op, Op *needle, std::set<Op*> &seen) {
  if (!op || !needle || seen.count(op))
    return false;
  if (op == needle)
    return true;
  seen.insert(op);
  for (auto operand : op->getOperands())
    if (usesValue(operand.defining, needle, seen))
      return true;
  return false;
}

bool usesValue(Op *op, Op *needle) {
  std::set<Op*> seen;
  return usesValue(op, needle, seen);
}

Op *stripSinglePhi(Op *op) {
  std::set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1 && !seen.count(op)) {
    seen.insert(op);
    op = op->DEF(0);
  }
  return op;
}

bool isSimpleIncrement(Op *op, Op *phi) {
  if (!op || !isa<AddIOp>(op) || op->getOperandCount() != 2)
    return false;
  auto a = op->DEF(0);
  auto b = op->DEF(1);
  return (a == phi && isa<IntOp>(b) && V(b) == 1) ||
         (b == phi && isa<IntOp>(a) && V(a) == 1);
}

std::pair<Op*, Op*> phiIncomingByLatch(Op *phi, BasicBlock *latch) {
  Op *fromLatch = nullptr;
  Op *fromOther = nullptr;
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  if (ops.size() != attrs.size())
    return { nullptr, nullptr };
  for (int i = 0; i < ops.size(); i++) {
    auto from = dyn_cast<FromAttr>(attrs[i]);
    if (!from)
      continue;
    if (from->bb == latch)
      fromLatch = ops[i].defining;
    else
      fromOther = ops[i].defining;
  }
  return { fromOther, fromLatch };
}

struct UnitLoopShape {
  Op *induction = nullptr;
  Op *stop = nullptr;
  Op *increment = nullptr;
};

UnitLoopShape findUnitLoopShape(LoopInfo *loop) {
  UnitLoopShape shape;
  if (!loop || !loop->preheader || loop->latches.size() != 1 || loop->exits.size() != 1)
    return shape;
  auto header = loop->header;
  auto term = dyn_cast<BranchOp>(header->getLastOp());
  if (!term || term->getOperandCount() != 1 || !term->has<TargetAttr>() || !term->has<ElseAttr>())
    return shape;
  auto cond = term->DEF(0);
  if (!cond || !isa<LtOp>(cond) || cond->getOperandCount() != 2)
    return shape;
  auto lhs = cond->DEF(0);
  auto rhs = cond->DEF(1);
  if (!lhs || !isa<PhiOp>(lhs) || lhs->getParent() != header || lhs->getResultType() != Value::i32)
    return shape;
  auto [start, incr] = phiIncomingByLatch(lhs, loop->getLatch());
  auto rawStart = stripSinglePhi(start);
  if (!rawStart || !isa<IntOp>(rawStart) || V(rawStart) != 0)
    return shape;
  if (!isSimpleIncrement(incr, lhs))
    return shape;
  shape.induction = lhs;
  shape.stop = rhs;
  shape.increment = incr;
  return shape;
}

bool canonicalUnitLoop(LoopInfo *loop, UnitLoopShape &shape) {
  shape = findUnitLoopShape(loop);
  return shape.induction && shape.stop && shape.increment;
}

std::vector<LoopInfo*> directSubloops(LoopInfo *loop) {
  std::vector<LoopInfo*> result;
  if (!loop)
    return result;
  for (auto sub : loop->subloops)
    if (sub && sub->parent == loop)
      result.push_back(sub);
  return result;
}

bool matrixGlobalInfo(const std::map<std::string, GlobalOp*> &globals,
                      const std::string &name, int &rows, int &cols) {
  auto it = globals.find(name);
  if (it == globals.end())
    return false;
  auto glob = it->second;
  if (!glob->has<DimensionAttr>())
    return false;
  const auto &dims = DIM(glob);
  if (dims.size() != 2)
    return false;
  rows = dims[0];
  cols = dims[1];
  return rows > 0 && cols > 0;
}

bool loopHasCallOrUnexpectedStore(LoopInfo *outer, StoreOp *allowedStore) {
  for (auto bb : outer->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<CallOp>(op) || isa<CloneOp>(op) || isa<JoinOp>(op) ||
          isa<WakeOp>(op) || isa<ReturnOp>(op))
        return true;
      if (auto store = dyn_cast<StoreOp>(op))
        if (store != allowedStore)
          return true;
    }
  }
  return false;
}

struct MatmulShape {
  LoopInfo *iLoop = nullptr;
  LoopInfo *jLoop = nullptr;
  LoopInfo *kLoop = nullptr;
  StoreOp *store = nullptr;
  std::string aName;
  std::string cName;
  Op *n = nullptr;
  int dim = 0;
  int rowStrideBytes = 0;
};

bool isAddOf(Op *op, Op *a, Op *b) {
  if (!op || !isa<AddIOp>(op) || op->getOperandCount() != 2)
    return false;
  auto lhs = op->DEF(0);
  auto rhs = op->DEF(1);
  return (lhs == a && rhs == b) || (lhs == b && rhs == a);
}

bool isMulOfLoads(Op *op, LoadOp *x, LoadOp *y) {
  if (!op || !isa<MulIOp>(op) || op->getOperandCount() != 2)
    return false;
  auto lhs = op->DEF(0);
  auto rhs = op->DEF(1);
  return (lhs == x && rhs == y) || (lhs == y && rhs == x);
}

bool validateReduction(StoreOp *store, LoopInfo *kLoop,
                       const std::vector<LoadOp*> &loads) {
  auto stored = stripSinglePhi(store->DEF(0));
  auto sumPhi = dyn_cast<PhiOp>(stored);
  if (!sumPhi || sumPhi->getParent() != kLoop->header)
    return false;

  auto [start, step] = phiIncomingByLatch(sumPhi, kLoop->getLatch());
  auto rawStart = stripSinglePhi(start);
  if (!rawStart || !isa<IntOp>(rawStart) || V(rawStart) != 0)
    return false;

  for (auto lhs : loads) {
    for (auto rhs : loads) {
      if (lhs == rhs)
        continue;
      auto prod = dyn_cast<MulIOp>(step) ? step : nullptr;
      if (!prod && step && isa<AddIOp>(step)) {
        auto a = step->DEF(0);
        auto b = step->DEF(1);
        prod = dyn_cast<MulIOp>(a == sumPhi ? b : (b == sumPhi ? a : nullptr));
      }
      if (prod && isMulOfLoads(prod, lhs, rhs) && isAddOf(step, sumPhi, prod))
        return true;
    }
  }
  return false;
}

bool validateAddressShape(StoreOp *store, const std::vector<LoadOp*> &loads,
                          const std::string &aName, const std::string &cName,
                          Op *i, Op *j, Op *k) {
  auto storeAddr = store->DEF(1);
  if (!usesValue(storeAddr, i) || !usesValue(storeAddr, j) || usesValue(storeAddr, k))
    return false;

  bool sawA = false;
  bool sawC = false;
  for (auto load : loads) {
    auto addr = load->DEF(0);
    auto names = globalsIn(addr);
    if (names.size() != 1)
      return false;
    const auto &name = *names.begin();
    if (name == aName) {
      if (!usesValue(addr, k) || !usesValue(addr, j) || usesValue(addr, i))
        return false;
      sawA = true;
    } else if (name == cName) {
      if (!usesValue(addr, i) || !usesValue(addr, k) || usesValue(addr, j))
        return false;
      sawC = true;
    } else {
      return false;
    }
  }
  return sawA && sawC;
}

bool tryMatchMatmul(LoopInfo *iLoop, const std::map<std::string, GlobalOp*> &globals,
                    MatmulShape &shape) {
  UnitLoopShape iShape;
  if (!canonicalUnitLoop(iLoop, iShape))
    return false;
  auto jSubs = directSubloops(iLoop);
  if (jSubs.size() != 1)
    return false;
  auto jLoop = jSubs[0];
  auto kSubs = directSubloops(jLoop);
  if (kSubs.size() != 1)
    return false;
  auto kLoop = kSubs[0];
  UnitLoopShape jShape, kShape;
  if (!canonicalUnitLoop(jLoop, jShape) || !canonicalUnitLoop(kLoop, kShape))
    return false;
  if (iShape.stop != jShape.stop || iShape.stop != kShape.stop)
    return false;

  std::vector<LoadOp*> loads;
  for (auto bb : kLoop->getBlocks())
    for (auto op : bb->getOps())
      if (auto load = dyn_cast<LoadOp>(op))
        loads.push_back(load);
  if (loads.size() != 2)
    return false;

  std::vector<StoreOp*> stores;
  for (auto bb : iLoop->getBlocks())
    for (auto op : bb->getOps())
      if (auto store = dyn_cast<StoreOp>(op))
        stores.push_back(store);
  if (stores.size() != 1)
    return false;
  auto store = stores[0];

  auto storeGlobals = globalsIn(store->DEF(1));
  if (storeGlobals.size() != 1)
    return false;
  std::string aName = *storeGlobals.begin();

  std::set<std::string> loadGlobalSet;
  for (auto load : loads) {
    auto names = globalsIn(load->DEF(0));
    if (names.size() != 1)
      return false;
    loadGlobalSet.insert(*names.begin());
  }
  if (loadGlobalSet.size() != 2 || !loadGlobalSet.count(aName))
    return false;

  std::string cName;
  for (const auto &name : loadGlobalSet)
    if (name != aName)
      cName = name;

  int aRows = 0, aCols = 0, cRows = 0, cCols = 0;
  if (!matrixGlobalInfo(globals, aName, aRows, aCols) ||
      !matrixGlobalInfo(globals, cName, cRows, cCols))
    return false;
  if (aCols != cCols || aRows != aCols || cRows != cCols)
    return false;

  if (loopHasCallOrUnexpectedStore(iLoop, store))
    return false;
  if (!validateReduction(store, kLoop, loads))
    return false;
  if (!validateAddressShape(store, loads, aName, cName,
                            iShape.induction, jShape.induction, kShape.induction))
    return false;

  shape.iLoop = iLoop;
  shape.jLoop = jLoop;
  shape.kLoop = kLoop;
  shape.store = store;
  shape.aName = aName;
  shape.cName = cName;
  shape.n = iShape.stop;
  shape.dim = aCols;
  shape.rowStrideBytes = aCols * 4;
  return true;
}

std::string scratchNameFor(int dim) {
  return "__row_scratch_buf_" + std::to_string(dim);
}

void ensureScratchGlobal(ModuleOp *module, int dim) {
  auto name = scratchNameFor(dim);
  for (auto glob : module->findAll<GlobalOp>())
    if (NAME(glob) == name)
      return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto values = new int[dim]();
  builder.create<GlobalOp>({
    new NameAttr(name),
    new SizeAttr((size_t) dim * 4),
    new IntArrayAttr(values, dim),
    new DimensionAttr({ dim }),
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

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(5),
    new ArgTypesAttr({ Value::i32, Value::i32, Value::i64, Value::i64, Value::i64 }),
    new ImpureAttr
  });
  auto region = func->appendRegion();

  auto entry = region->appendBlock();
  auto iCond = region->appendBlock();
  auto zeroInit = region->appendBlock();
  auto zeroCond = region->appendBlock();
  auto zeroBody = region->appendBlock();
  auto kInit = region->appendBlock();
  auto kCond = region->appendBlock();
  auto kPrep = region->appendBlock();
  auto saxpyCond = region->appendBlock();
  auto saxpyBody = region->appendBlock();
  auto kNext = region->appendBlock();
  auto writeInit = region->appendBlock();
  auto writeCond = region->appendBlock();
  auto writeBody = region->appendBlock();
  auto iNext = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto iSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto jSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto kSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto coeffSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  auto rowStride = builder.create<GetArgOp>(Value::i32, { new IntAttr(1) });
  auto a = builder.create<GetArgOp>(Value::i64, { new IntAttr(2) });
  auto c = builder.create<GetArgOp>(Value::i64, { new IntAttr(3) });
  auto scratch = builder.create<GetArgOp>(Value::i64, { new IntAttr(4) });
  storeVar(builder, i32(builder, 0), iSlot);
  builder.create<GotoOp>({ new TargetAttr(iCond) });

  builder.setToBlockEnd(iCond);
  auto iv = loadVar(builder, iSlot);
  branch(builder, bin<LtOp>(builder, iv, n), zeroInit, done);

  builder.setToBlockEnd(zeroInit);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(zeroCond) });

  builder.setToBlockEnd(zeroCond);
  auto zj = loadVar(builder, jSlot);
  branch(builder, bin<LtOp>(builder, zj, n), zeroBody, kInit);

  builder.setToBlockEnd(zeroBody);
  auto zj2 = loadVar(builder, jSlot);
  builder.create<StoreOp>(vals({ i32(builder, 0), rowAddr(builder, scratch, zj2) }),
                          attrs({ new SizeAttr(4) }));
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(zeroCond) });

  builder.setToBlockEnd(kInit);
  storeVar(builder, i32(builder, 0), kSlot);
  builder.create<GotoOp>({ new TargetAttr(kCond) });

  builder.setToBlockEnd(kCond);
  auto kv = loadVar(builder, kSlot);
  branch(builder, bin<LtOp>(builder, kv, n), kPrep, writeInit);

  builder.setToBlockEnd(kPrep);
  auto ci = loadVar(builder, iSlot);
  auto ck = loadVar(builder, kSlot);
  auto coeff = builder.create<LoadOp>(Value::i32,
                                      vals({ matrixAddr(builder, c, ci, ck, rowStride) }),
                                      attrs({ new SizeAttr(4) }));
  storeVar(builder, coeff, coeffSlot);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(saxpyCond) });

  builder.setToBlockEnd(saxpyCond);
  auto sj = loadVar(builder, jSlot);
  branch(builder, bin<LtOp>(builder, sj, n), saxpyBody, kNext);

  builder.setToBlockEnd(saxpyBody);
  auto old = builder.create<LoadOp>(Value::i32, vals({ rowAddr(builder, scratch, loadVar(builder, jSlot)) }),
                                    attrs({ new SizeAttr(4) }));
  auto aval = builder.create<LoadOp>(Value::i32,
                                     vals({ matrixAddr(builder, a, loadVar(builder, kSlot),
                                                       loadVar(builder, jSlot), rowStride) }),
                                     attrs({ new SizeAttr(4) }));
  auto prod = bin<MulIOp>(builder, loadVar(builder, coeffSlot), aval);
  auto next = bin<AddIOp>(builder, old, prod);
  builder.create<StoreOp>(vals({ next, rowAddr(builder, scratch, loadVar(builder, jSlot)) }),
                          attrs({ new SizeAttr(4) }));
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(saxpyCond) });

  builder.setToBlockEnd(kNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, kSlot), i32(builder, 1)), kSlot);
  builder.create<GotoOp>({ new TargetAttr(kCond) });

  builder.setToBlockEnd(writeInit);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(writeCond) });

  builder.setToBlockEnd(writeCond);
  auto wj = loadVar(builder, jSlot);
  branch(builder, bin<LtOp>(builder, wj, n), writeBody, iNext);

  builder.setToBlockEnd(writeBody);
  auto out = builder.create<LoadOp>(Value::i32, vals({ rowAddr(builder, scratch, loadVar(builder, jSlot)) }),
                                    attrs({ new SizeAttr(4) }));
  builder.create<StoreOp>(vals({ out, matrixAddr(builder, a, loadVar(builder, iSlot),
                                                 loadVar(builder, jSlot), rowStride) }),
                          attrs({ new SizeAttr(4) }));
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(writeCond) });

  builder.setToBlockEnd(iNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, iSlot), i32(builder, 1)), iSlot);
  builder.create<GotoOp>({ new TargetAttr(iCond) });

  builder.setToBlockEnd(done);
  builder.create<ReturnOp>();
}

bool replaceWithHelper(ModuleOp *module, const MatmulShape &shape) {
  if (!shape.iLoop || !shape.iLoop->preheader || shape.iLoop->exits.size() != 1)
    return false;
  auto preterm = shape.iLoop->preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || !preterm->has<TargetAttr>() || TARGET(preterm) != shape.iLoop->header)
    return false;

  ensureScratchGlobal(module, shape.dim);
  buildHelper(module);

  auto exit = shape.iLoop->getExit();
  auto region = shape.iLoop->header->getParent();
  auto helperBB = region->insertAfter(shape.iLoop->preheader);

  Builder builder;
  builder.setBeforeOp(preterm);
  auto dim = i32(builder, shape.dim);
  auto canFast = builder.create<LeOp>(std::vector<Value>{ shape.n, dim });
  builder.replace<BranchOp>(preterm, std::vector<Value>{ canFast },
                            std::vector<Attr*>{ new TargetAttr(helperBB), new ElseAttr(shape.iLoop->header) });

  builder.setToBlockEnd(helperBB);
  auto a = builder.create<GetGlobalOp>({ new NameAttr(shape.aName) });
  auto c = builder.create<GetGlobalOp>({ new NameAttr(shape.cName) });
  auto scratch = builder.create<GetGlobalOp>({ new NameAttr(scratchNameFor(shape.dim)) });
  auto rowStride = i32(builder, shape.rowStrideBytes);
  builder.create<CallOp>(Value::i32, vals({ shape.n, rowStride, a, c, scratch }),
                         attrs({ new NameAttr(kHelperName), new ImpureAttr }));
  builder.create<GotoOp>({ new TargetAttr(exit) });
  return true;
}

} // namespace

std::map<std::string, int> RowScratchMatmul::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void RowScratchMatmul::run() {
  if (!envEnabled("SISY_ENABLE_ROW_SCRATCH_MATMUL", true))
    return;

  auto globals = getGlobalMap();
  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  LoopAnalysis analysis(module);
  analysis.run();
  for (auto &[_, forest] : analysis.getResult()) {
    for (auto loop : forest.getLoops()) {
      MatmulShape shape;
      if (!tryMatchMatmul(loop, globals, shape)) {
        rejectedShape++;
        continue;
      }
      candidates++;
      if (replaceWithHelper(module, shape))
        replaced++;
    }
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
  if (replaced)
    CallGraph(module).run();
}
