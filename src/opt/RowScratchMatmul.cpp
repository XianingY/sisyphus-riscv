#include "Passes.h"
#include "LoopPasses.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

constexpr int kMaxDim = 1024;
constexpr int kRowStrideBytes = 4096;
constexpr const char *kHelperName = "__sisy_row_scratch_matmul";
constexpr const char *kScratchName = "__sisy_row_scratch_buf";

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

Op *matrixAddr(Builder &builder, Op *base, Op *row, Op *col) {
  auto rowStride = i32(builder, kRowStrideBytes);
  auto rowOff = bin<MulIOp>(builder, row, rowStride);
  auto rowBase = bin<AddLOp>(builder, base, rowOff);
  auto elemSize = i32(builder, 4);
  auto colOff = bin<MulIOp>(builder, col, elemSize);
  return bin<AddLOp>(builder, rowBase, colOff);
}

Op *rowAddr(Builder &builder, Op *base, Op *col) {
  auto elemSize = i32(builder, 4);
  auto colOff = bin<MulIOp>(builder, col, elemSize);
  return bin<AddLOp>(builder, base, colOff);
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

bool hasMatrixDims(const std::map<std::string, GlobalOp*> &globals,
                   const std::string &name) {
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
  if (!shape.induction || !shape.stop || !shape.increment)
    return false;
  return true;
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

bool loopHasCallOrUnexpectedStore(LoopInfo *outer, StoreOp *allowedStore) {
  for (auto bb : outer->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<CallOp>(op) || isa<CloneOp>(op) || isa<JoinOp>(op) ||
          isa<WakeOp>(op) || isa<ReturnOp>(op))
        return true;
      if (auto store = dyn_cast<StoreOp>(op)) {
        if (store != allowedStore)
          return true;
      }
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
};

bool tryMatchMatmul(LoopInfo *iLoop, const std::map<std::string, GlobalOp*> &globals,
                    MatmulShape &shape) {
  const bool debug = envEnabled("SISY_ROW_SCRATCH_DEBUG", false);
  auto reject = [&](const char *why) {
    if (debug) {
      std::cerr << "[row-scratch] reject " << why
                << " header=" << bbmap[iLoop ? iLoop->header : nullptr] << "\n";
    }
    return false;
  };
  UnitLoopShape iShape;
  if (!canonicalUnitLoop(iLoop, iShape))
    return reject("outer-canonical");
  auto jSubs = directSubloops(iLoop);
  if (jSubs.size() != 1)
    return reject("j-subloop-count");
  auto jLoop = jSubs[0];
  auto kSubs = directSubloops(jLoop);
  if (kSubs.size() != 1)
    return reject("k-subloop-count");
  auto kLoop = kSubs[0];
  UnitLoopShape jShape;
  UnitLoopShape kShape;
  if (!canonicalUnitLoop(jLoop, jShape) || !canonicalUnitLoop(kLoop, kShape))
    return reject("inner-canonical");
  if (iShape.stop != jShape.stop || iShape.stop != kShape.stop)
    return reject("stop");

  std::vector<LoadOp*> loads;
  for (auto bb : kLoop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (auto load = dyn_cast<LoadOp>(op))
        loads.push_back(load);
    }
  }
  if (loads.size() != 2)
    return reject("loads");

  std::vector<StoreOp*> stores;
  for (auto bb : iLoop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (auto store = dyn_cast<StoreOp>(op))
        stores.push_back(store);
    }
  }
  if (stores.size() != 1)
    return reject("stores");
  auto store = stores[0];

  auto storeGlobals = globalsIn(store->DEF(1));
  if (storeGlobals.size() != 1)
    return reject("store-global");
  std::string aName = *storeGlobals.begin();
  if (!hasMatrixDims(globals, aName))
    return reject("a-dims");

  std::set<std::string> loadGlobalSet;
  for (auto load : loads) {
    auto names = globalsIn(load->DEF(0));
    if (names.size() != 1)
      return reject("load-global");
    loadGlobalSet.insert(*names.begin());
  }
  if (loadGlobalSet.size() != 2 || !loadGlobalSet.count(aName))
    return reject("load-global-set");

  std::string cName;
  for (const auto &name : loadGlobalSet)
    if (name != aName)
      cName = name;
  if (!hasMatrixDims(globals, cName))
    return reject("c-dims");

  if (loopHasCallOrUnexpectedStore(iLoop, store))
    return reject("side-effect");

  shape.iLoop = iLoop;
  shape.jLoop = jLoop;
  shape.kLoop = kLoop;
  shape.store = store;
  shape.aName = aName;
  shape.cName = cName;
  shape.n = iShape.stop;
  return true;
}

bool hasHelper(ModuleOp *module) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == kHelperName)
      return true;
  return false;
}

void ensureScratchGlobal(ModuleOp *module) {
  for (auto glob : module->findAll<GlobalOp>())
    if (NAME(glob) == kScratchName)
      return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto values = new int[kMaxDim]();
  builder.create<GlobalOp>({
    new NameAttr(kScratchName),
    new SizeAttr(kMaxDim * 4),
    new IntArrayAttr(values, kMaxDim),
    new DimensionAttr({ kMaxDim }),
  });
}

void buildHelper(ModuleOp *module) {
  if (hasHelper(module))
    return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(4),
    new ArgTypesAttr({ Value::i32, Value::i64, Value::i64, Value::i64 }),
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
  n->add<ImpureAttr>();
  auto a = builder.create<GetArgOp>(Value::i64, { new IntAttr(1) });
  a->add<ImpureAttr>();
  auto c = builder.create<GetArgOp>(Value::i64, { new IntAttr(2) });
  c->add<ImpureAttr>();
  auto scratch = builder.create<GetArgOp>(Value::i64, { new IntAttr(3) });
  scratch->add<ImpureAttr>();
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
  auto zj3 = loadVar(builder, jSlot);
  storeVar(builder, bin<AddIOp>(builder, zj3, i32(builder, 1)), jSlot);
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
  auto coeff = builder.create<LoadOp>(Value::i32, vals({ matrixAddr(builder, c, ci, ck) }),
                                      attrs({ new SizeAttr(4) }));
  storeVar(builder, coeff, coeffSlot);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(saxpyCond) });

  builder.setToBlockEnd(saxpyCond);
  auto sj = loadVar(builder, jSlot);
  branch(builder, bin<LtOp>(builder, sj, n), saxpyBody, kNext);

  builder.setToBlockEnd(saxpyBody);
  auto sj0 = loadVar(builder, jSlot);
  auto old = builder.create<LoadOp>(Value::i32, vals({ rowAddr(builder, scratch, sj0) }),
                                    attrs({ new SizeAttr(4) }));
  auto ak = loadVar(builder, kSlot);
  auto aj = loadVar(builder, jSlot);
  auto aval = builder.create<LoadOp>(Value::i32, vals({ matrixAddr(builder, a, ak, aj) }),
                                     attrs({ new SizeAttr(4) }));
  auto coeffVal = loadVar(builder, coeffSlot);
  auto prod = bin<MulIOp>(builder, coeffVal, aval);
  auto next = bin<AddIOp>(builder, old, prod);
  auto sj1 = loadVar(builder, jSlot);
  builder.create<StoreOp>(vals({ next, rowAddr(builder, scratch, sj1) }),
                          attrs({ new SizeAttr(4) }));
  auto sj2 = loadVar(builder, jSlot);
  storeVar(builder, bin<AddIOp>(builder, sj2, i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(saxpyCond) });

  builder.setToBlockEnd(kNext);
  auto kv2 = loadVar(builder, kSlot);
  storeVar(builder, bin<AddIOp>(builder, kv2, i32(builder, 1)), kSlot);
  builder.create<GotoOp>({ new TargetAttr(kCond) });

  builder.setToBlockEnd(writeInit);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(writeCond) });

  builder.setToBlockEnd(writeCond);
  auto wj = loadVar(builder, jSlot);
  branch(builder, bin<LtOp>(builder, wj, n), writeBody, iNext);

  builder.setToBlockEnd(writeBody);
  auto wj0 = loadVar(builder, jSlot);
  auto out = builder.create<LoadOp>(Value::i32, vals({ rowAddr(builder, scratch, wj0) }),
                                    attrs({ new SizeAttr(4) }));
  auto wi = loadVar(builder, iSlot);
  auto wj1 = loadVar(builder, jSlot);
  builder.create<StoreOp>(vals({ out, matrixAddr(builder, a, wi, wj1) }),
                          attrs({ new SizeAttr(4) }));
  auto wj2 = loadVar(builder, jSlot);
  storeVar(builder, bin<AddIOp>(builder, wj2, i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(writeCond) });

  builder.setToBlockEnd(iNext);
  auto iv2 = loadVar(builder, iSlot);
  storeVar(builder, bin<AddIOp>(builder, iv2, i32(builder, 1)), iSlot);
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
  auto headerTerm = dyn_cast<BranchOp>(shape.iLoop->header->getLastOp());
  if (!headerTerm || headerTerm->getOperandCount() != 1)
    return false;
  auto cond = dyn_cast<LtOp>(headerTerm->DEF(0));
  if (!cond || cond->getOperandCount() != 2)
    return false;

  ensureScratchGlobal(module);
  buildHelper(module);

  Builder builder;
  builder.setBeforeOp(preterm);
  auto a = builder.create<GetGlobalOp>({ new NameAttr(shape.aName) });
  auto c = builder.create<GetGlobalOp>({ new NameAttr(shape.cName) });
  auto scratch = builder.create<GetGlobalOp>({ new NameAttr(kScratchName) });
  builder.create<CallOp>(Value::i32, vals({ shape.n, a, c, scratch }),
                         attrs({ new NameAttr(kHelperName), new ImpureAttr }));
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  cond->setOperand(1, zero);
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
  for (auto [func, forest] : analysis.getResult()) {
    (void) func;
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
