#include "Passes.h"
#include "LoopPasses.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

using namespace sys;

namespace {

constexpr int kAffineUnroll = 4;

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

Op *ptrOffset(Builder &builder, Op *base, int bytes) {
  if (bytes == 0)
    return base;
  return bin<AddLOp>(builder, base, i32(builder, bytes));
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
  const auto &attrList = phi->getAttrs();
  if (ops.size() != attrList.size())
    return { nullptr, nullptr };
  for (int i = 0; i < ops.size(); i++) {
    auto from = dyn_cast<FromAttr>(attrList[i]);
    if (!from)
      continue;
    if (from->bb == latch)
      fromLatch = ops[i].defining;
    else
      fromOther = ops[i].defining;
  }
  return { fromOther, fromLatch };
}

bool collectGlobals(Op *op, std::set<std::string> &names, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return true;
  seen.insert(op);
  if (auto global = dyn_cast<GetGlobalOp>(op)) {
    names.insert(NAME(global));
    return true;
  }
  for (auto operand : op->getOperands())
    if (!collectGlobals(operand.defining, names, seen))
      return false;
  return true;
}

std::set<std::string> globalsIn(Op *op) {
  std::set<std::string> names;
  std::set<Op*> seen;
  collectGlobals(op, names, seen);
  return names;
}

bool disjointGlobals(Op *a, Op *b) {
  auto ga = globalsIn(a);
  auto gb = globalsIn(b);
  if (ga.empty() || gb.empty())
    return false;
  for (auto &name : ga)
    if (gb.count(name))
      return false;
  return true;
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

struct UnitLoopShape {
  Op *induction = nullptr;
  Op *stop = nullptr;
  Op *increment = nullptr;
};

UnitLoopShape findUnitLoopShape(LoopInfo *loop) {
  UnitLoopShape shape;
  if (!loop || !loop->preheader || loop->latches.size() != 1 || loop->exits.size() != 1)
    return shape;
  auto term = dyn_cast<BranchOp>(loop->header->getLastOp());
  if (!term || term->getOperandCount() != 1 || !term->has<TargetAttr>() || !term->has<ElseAttr>())
    return shape;
  auto cond = term->DEF(0);
  if (!cond || !isa<LtOp>(cond) || cond->getOperandCount() != 2)
    return shape;
  auto lhs = cond->DEF(0);
  auto rhs = cond->DEF(1);
  if (!lhs || !isa<PhiOp>(lhs) || lhs->getParent() != loop->header || lhs->getResultType() != Value::i32)
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

bool loopHasForbiddenSideEffect(LoopInfo *loop, StoreOp *allowedStore) {
  for (auto bb : loop->getBlocks()) {
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

std::vector<StoreOp*> collectStores(LoopInfo *loop) {
  std::vector<StoreOp*> stores;
  for (auto bb : loop->getBlocks())
    for (auto op : bb->getOps())
      if (auto store = dyn_cast<StoreOp>(op))
        stores.push_back(store);
  return stores;
}

bool sameOp(Op *a, Op *b) {
  return a == b;
}

struct StoreRecurrence {
  StoreOp *store = nullptr;
  LoadOp *oldDstLoad = nullptr;
  LoadOp *srcLoad = nullptr;
  LoadOp *factorLoad = nullptr;
  Op *dstAddr = nullptr;
  Op *srcAddr = nullptr;
  Op *factorAddr = nullptr;
};

bool matchStoreRecurrence(StoreOp *store, Op *innerIV, StoreRecurrence &rec) {
  if (!store || store->getOperandCount() != 2)
    return false;
  auto dstAddr = store->DEF(1);
  auto stored = stripSinglePhi(store->DEF(0));
  auto add = dyn_cast<AddIOp>(stored);
  if (!add || add->getOperandCount() != 2)
    return false;

  auto trySide = [&](Op *maybeMul, Op *maybeSrc) -> bool {
    auto mul = dyn_cast<MulIOp>(stripSinglePhi(maybeMul));
    auto srcLoad = dyn_cast<LoadOp>(stripSinglePhi(maybeSrc));
    if (!mul || !srcLoad || mul->getOperandCount() != 2 || srcLoad->getOperandCount() != 1)
      return false;
    LoadOp *oldDst = nullptr;
    LoadOp *factor = nullptr;
    for (int i = 0; i < 2; i++) {
      auto load = dyn_cast<LoadOp>(stripSinglePhi(mul->DEF(i)));
      if (!load || load->getOperandCount() != 1)
        continue;
      if (sameOp(load->DEF(0), dstAddr))
        oldDst = load;
      else
        factor = load;
    }
    if (!oldDst || !factor || factor->getOperandCount() != 1)
      return false;
    if (usesValue(factor->DEF(0), innerIV))
      return false;
    if (!usesValue(dstAddr, innerIV) || !usesValue(srcLoad->DEF(0), innerIV))
      return false;
    if (!disjointGlobals(factor->DEF(0), dstAddr))
      return false;
    rec.store = store;
    rec.oldDstLoad = oldDst;
    rec.srcLoad = srcLoad;
    rec.factorLoad = factor;
    rec.dstAddr = dstAddr;
    rec.srcAddr = srcLoad->DEF(0);
    rec.factorAddr = factor->DEF(0);
    return true;
  };

  if (trySide(add->DEF(0), add->DEF(1)))
    return true;
  if (trySide(add->DEF(1), add->DEF(0)))
    return true;
  return false;
}

bool cloneableAddressOp(Op *op) {
  return isa<IntOp>(op) || isa<GetGlobalOp>(op) ||
         isa<AddIOp>(op) || isa<AddLOp>(op) ||
         isa<SubIOp>(op) || isa<SubLOp>(op) ||
         isa<MulIOp>(op) || isa<MulLOp>(op);
}

bool canCloneExpr(Op *op, LoopInfo *oldLoop, const std::set<Op*> &substituted,
                  std::set<Op*> &seen) {
  if (!op)
    return false;
  if (substituted.count(op))
    return true;
  if (!oldLoop->contains(op->getParent()))
    return true;
  if (seen.count(op))
    return true;
  seen.insert(op);
  if (!cloneableAddressOp(op))
    return false;
  for (auto operand : op->getOperands())
    if (!canCloneExpr(operand.defining, oldLoop, substituted, seen))
      return false;
  return true;
}

Op *cloneExpr(Op *op, LoopInfo *oldLoop, Builder &builder, std::map<Op*, Op*> &subst) {
  if (!op)
    return nullptr;
  if (subst.count(op))
    return subst[op];
  if (!oldLoop->contains(op->getParent()))
    return op;
  if (!cloneableAddressOp(op))
    return nullptr;

  std::vector<Value> operands;
  for (auto operand : op->getOperands()) {
    auto mapped = cloneExpr(operand.defining, oldLoop, builder, subst);
    if (!mapped)
      return nullptr;
    operands.push_back(mapped);
  }

  Op *clone = nullptr;
  if (isa<IntOp>(op))
    clone = builder.create<IntOp>({ new IntAttr(V(op)) });
  else if (auto global = dyn_cast<GetGlobalOp>(op))
    clone = builder.create<GetGlobalOp>({ new NameAttr(NAME(global)) });
  else if (isa<AddIOp>(op))
    clone = builder.create<AddIOp>(operands);
  else if (isa<AddLOp>(op))
    clone = builder.create<AddLOp>(operands);
  else if (isa<SubIOp>(op))
    clone = builder.create<SubIOp>(operands);
  else if (isa<SubLOp>(op))
    clone = builder.create<SubLOp>(operands);
  else if (isa<MulIOp>(op))
    clone = builder.create<MulIOp>(operands);
  else if (isa<MulLOp>(op))
    clone = builder.create<MulLOp>(operands);
  if (clone)
    subst[op] = clone;
  return clone;
}

void emitRowStore(Builder &builder, Op *dstPtr, Op *srcPtr, Op *factor, int offset) {
  auto dstLane = ptrOffset(builder, dstPtr, offset);
  auto srcLane = ptrOffset(builder, srcPtr, offset);
  auto srcVal = builder.create<LoadOp>(Value::i32, vals({ srcLane }), attrs({ new SizeAttr(4) }));
  Op *out = srcVal;
  if (factor) {
    auto dstVal = builder.create<LoadOp>(Value::i32, vals({ dstLane }), attrs({ new SizeAttr(4) }));
    out = bin<AddIOp>(builder, bin<MulIOp>(builder, dstVal, factor), srcVal);
  }
  builder.create<StoreOp>(vals({ out, dstLane }), attrs({ new SizeAttr(4) }));
}

bool buildRowAffinePath(LoopInfo *loop, const UnitLoopShape &shape, const StoreRecurrence &rec) {
  auto preterm = loop->preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || !preterm->has<TargetAttr>() || TARGET(preterm) != loop->header)
    return false;

  std::set<Op*> substituted = { shape.induction };
  std::set<Op*> seen;
  if (!canCloneExpr(rec.srcAddr, loop, substituted, seen))
    return false;
  seen.clear();
  if (!canCloneExpr(rec.dstAddr, loop, substituted, seen))
    return false;
  seen.clear();
  if (!canCloneExpr(rec.factorAddr, loop, substituted, seen))
    return false;

  auto region = loop->header->getParent();
  auto dispatch = region->insertAfter(loop->preheader);
  auto copyCond = region->insertAfter(dispatch);
  auto copyBody = region->insertAfter(copyCond);
  auto copyTailCond = region->insertAfter(copyBody);
  auto copyTailBody = region->insertAfter(copyTailCond);
  auto affineCond = region->insertAfter(copyTailBody);
  auto affineBody = region->insertAfter(affineCond);
  auto affineTailCond = region->insertAfter(affineBody);
  auto affineTailBody = region->insertAfter(affineTailCond);
  auto exit = loop->getExit();

  Builder builder;
  builder.setBeforeOp(preterm);
  auto zero = i32(builder, 0);
  auto four = i32(builder, kAffineUnroll);
  auto rem = bin<ModIOp>(builder, shape.stop, four);
  auto mainStop = bin<SubIOp>(builder, shape.stop, rem);
  std::map<Op*, Op*> subst;
  subst[shape.induction] = zero;
  auto dstBase = cloneExpr(rec.dstAddr, loop, builder, subst);
  auto srcBase = cloneExpr(rec.srcAddr, loop, builder, subst);
  auto factorAddr = cloneExpr(rec.factorAddr, loop, builder, subst);
  if (!dstBase || !srcBase || !factorAddr)
    return false;
  auto factor = builder.create<LoadOp>(Value::i32, vals({ factorAddr }), attrs({ new SizeAttr(4) }));
  auto isZero = bin<EqOp>(builder, factor, zero);
  builder.replace<BranchOp>(preterm, vals({ i32(builder, 1) }),
                            attrs({ new TargetAttr(dispatch), new ElseAttr(loop->header) }));

  builder.setToBlockEnd(dispatch);
  builder.create<BranchOp>(vals({ isZero }), attrs({ new TargetAttr(copyCond), new ElseAttr(affineCond) }));

  builder.setToBlockEnd(copyCond);
  auto copyJ = builder.create<PhiOp>(vals({ zero }), attrs({ new FromAttr(dispatch) }));
  auto copyDst = builder.create<PhiOp>(vals({ dstBase }), attrs({ new FromAttr(dispatch) }));
  auto copySrc = builder.create<PhiOp>(vals({ srcBase }), attrs({ new FromAttr(dispatch) }));
  builder.create<BranchOp>(vals({ bin<LtOp>(builder, copyJ, mainStop) }),
                           attrs({ new TargetAttr(copyBody), new ElseAttr(copyTailCond) }));

  builder.setToBlockEnd(copyBody);
  for (int lane = 0; lane < kAffineUnroll; lane++)
    emitRowStore(builder, copyDst, copySrc, nullptr, lane * 4);
  auto copyJNext = bin<AddIOp>(builder, copyJ, four);
  auto copyDstNext = ptrOffset(builder, copyDst, kAffineUnroll * 4);
  auto copySrcNext = ptrOffset(builder, copySrc, kAffineUnroll * 4);
  builder.create<GotoOp>({ new TargetAttr(copyCond) });
  copyJ->pushOperand(copyJNext);
  copyJ->add<FromAttr>(copyBody);
  copyDst->pushOperand(copyDstNext);
  copyDst->add<FromAttr>(copyBody);
  copySrc->pushOperand(copySrcNext);
  copySrc->add<FromAttr>(copyBody);

  builder.setToBlockEnd(copyTailCond);
  auto copyTailJ = builder.create<PhiOp>(vals({ copyJ }), attrs({ new FromAttr(copyCond) }));
  auto copyTailDst = builder.create<PhiOp>(vals({ copyDst }), attrs({ new FromAttr(copyCond) }));
  auto copyTailSrc = builder.create<PhiOp>(vals({ copySrc }), attrs({ new FromAttr(copyCond) }));
  builder.create<BranchOp>(vals({ bin<LtOp>(builder, copyTailJ, shape.stop) }),
                           attrs({ new TargetAttr(copyTailBody), new ElseAttr(exit) }));

  builder.setToBlockEnd(copyTailBody);
  emitRowStore(builder, copyTailDst, copyTailSrc, nullptr, 0);
  auto copyTailJNext = bin<AddIOp>(builder, copyTailJ, i32(builder, 1));
  auto copyTailDstNext = ptrOffset(builder, copyTailDst, 4);
  auto copyTailSrcNext = ptrOffset(builder, copyTailSrc, 4);
  builder.create<GotoOp>({ new TargetAttr(copyTailCond) });
  copyTailJ->pushOperand(copyTailJNext);
  copyTailJ->add<FromAttr>(copyTailBody);
  copyTailDst->pushOperand(copyTailDstNext);
  copyTailDst->add<FromAttr>(copyTailBody);
  copyTailSrc->pushOperand(copyTailSrcNext);
  copyTailSrc->add<FromAttr>(copyTailBody);

  builder.setToBlockEnd(affineCond);
  auto affineJ = builder.create<PhiOp>(vals({ zero }), attrs({ new FromAttr(dispatch) }));
  auto affineDst = builder.create<PhiOp>(vals({ dstBase }), attrs({ new FromAttr(dispatch) }));
  auto affineSrc = builder.create<PhiOp>(vals({ srcBase }), attrs({ new FromAttr(dispatch) }));
  builder.create<BranchOp>(vals({ bin<LtOp>(builder, affineJ, mainStop) }),
                           attrs({ new TargetAttr(affineBody), new ElseAttr(affineTailCond) }));

  builder.setToBlockEnd(affineBody);
  for (int lane = 0; lane < kAffineUnroll; lane++)
    emitRowStore(builder, affineDst, affineSrc, factor, lane * 4);
  auto affineJNext = bin<AddIOp>(builder, affineJ, four);
  auto affineDstNext = ptrOffset(builder, affineDst, kAffineUnroll * 4);
  auto affineSrcNext = ptrOffset(builder, affineSrc, kAffineUnroll * 4);
  builder.create<GotoOp>({ new TargetAttr(affineCond) });
  affineJ->pushOperand(affineJNext);
  affineJ->add<FromAttr>(affineBody);
  affineDst->pushOperand(affineDstNext);
  affineDst->add<FromAttr>(affineBody);
  affineSrc->pushOperand(affineSrcNext);
  affineSrc->add<FromAttr>(affineBody);

  builder.setToBlockEnd(affineTailCond);
  auto affineTailJ = builder.create<PhiOp>(vals({ affineJ }), attrs({ new FromAttr(affineCond) }));
  auto affineTailDst = builder.create<PhiOp>(vals({ affineDst }), attrs({ new FromAttr(affineCond) }));
  auto affineTailSrc = builder.create<PhiOp>(vals({ affineSrc }), attrs({ new FromAttr(affineCond) }));
  builder.create<BranchOp>(vals({ bin<LtOp>(builder, affineTailJ, shape.stop) }),
                           attrs({ new TargetAttr(affineTailBody), new ElseAttr(exit) }));

  builder.setToBlockEnd(affineTailBody);
  emitRowStore(builder, affineTailDst, affineTailSrc, factor, 0);
  auto affineTailJNext = bin<AddIOp>(builder, affineTailJ, i32(builder, 1));
  auto affineTailDstNext = ptrOffset(builder, affineTailDst, 4);
  auto affineTailSrcNext = ptrOffset(builder, affineTailSrc, 4);
  builder.create<GotoOp>({ new TargetAttr(affineTailCond) });
  affineTailJ->pushOperand(affineTailJNext);
  affineTailJ->add<FromAttr>(affineTailBody);
  affineTailDst->pushOperand(affineTailDstNext);
  affineTailDst->add<FromAttr>(affineTailBody);
  affineTailSrc->pushOperand(affineTailSrcNext);
  affineTailSrc->add<FromAttr>(affineTailBody);
  return true;
}

bool buildZeroCopyPath(LoopInfo *loop, const UnitLoopShape &shape, const StoreRecurrence &rec) {
  auto preterm = loop->preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || !preterm->has<TargetAttr>() || TARGET(preterm) != loop->header)
    return false;
  std::set<Op*> substituted = { shape.induction };
  std::set<Op*> seen;
  if (!canCloneExpr(rec.srcAddr, loop, substituted, seen))
    return false;
  seen.clear();
  if (!canCloneExpr(rec.dstAddr, loop, substituted, seen))
    return false;

  auto region = loop->header->getParent();
  auto copyHeader = region->insertAfter(loop->preheader);
  auto copyBody = region->insertAfter(copyHeader);
  auto copyDone = region->insertAfter(copyBody);
  auto exit = loop->getExit();

  Builder builder;
  builder.setBeforeOp(preterm);
  auto factor = builder.create<LoadOp>(Value::i32, vals({ rec.factorAddr }), attrs({ new SizeAttr(4) }));
  auto zero = i32(builder, 0);
  auto isZero = bin<EqOp>(builder, factor, zero);
  builder.replace<BranchOp>(preterm, vals({ isZero }), attrs({ new TargetAttr(copyHeader), new ElseAttr(loop->header) }));

  builder.setToBlockEnd(copyHeader);
  auto j = builder.create<PhiOp>(vals({ i32(builder, 0) }), attrs({ new FromAttr(loop->preheader) }));
  auto cond = bin<LtOp>(builder, j, shape.stop);
  builder.create<BranchOp>(vals({ cond }), attrs({ new TargetAttr(copyBody), new ElseAttr(copyDone) }));

  builder.setToBlockEnd(copyBody);
  std::map<Op*, Op*> subst;
  subst[shape.induction] = j;
  auto srcAddr = cloneExpr(rec.srcAddr, loop, builder, subst);
  auto dstAddr = cloneExpr(rec.dstAddr, loop, builder, subst);
  if (!srcAddr || !dstAddr)
    return false;
  auto srcVal = builder.create<LoadOp>(Value::i32, vals({ srcAddr }), attrs({ new SizeAttr(4) }));
  builder.create<StoreOp>(vals({ srcVal, dstAddr }), attrs({ new SizeAttr(4) }));
  auto jNext = bin<AddIOp>(builder, j, i32(builder, 1));
  builder.create<GotoOp>({ new TargetAttr(copyHeader) });

  j->pushOperand(jNext);
  j->add<FromAttr>(copyBody);

  builder.setToBlockEnd(copyDone);
  builder.create<GotoOp>({ new TargetAttr(exit) });
  return true;
}

bool tryReplace(LoopInfo *loop, bool &usedAffine) {
  usedAffine = false;
  auto shape = findUnitLoopShape(loop);
  if (!shape.induction || !shape.stop)
    return false;
  auto stores = collectStores(loop);
  if (stores.size() != 1)
    return false;
  if (loopHasForbiddenSideEffect(loop, stores[0]))
    return false;
  StoreRecurrence rec;
  if (!matchStoreRecurrence(stores[0], shape.induction, rec))
    return false;
  if (buildRowAffinePath(loop, shape, rec)) {
    usedAffine = true;
    return true;
  }
  return buildZeroCopyPath(loop, shape, rec);
}

} // namespace

std::map<std::string, int> ZeroFactorStoreLoop::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "affine-replaced", affineReplaced },
    { "rejected-shape", rejectedShape },
  };
}

void ZeroFactorStoreLoop::run() {
  if (!envEnabled("SISY_ENABLE_ZERO_FACTOR_STORE_LOOP", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  LoopAnalysis analysis(module);
  analysis.run();
  for (auto &[_, forest] : analysis.getResult()) {
    for (auto loop : forest.getLoops()) {
      auto stores = collectStores(loop);
      if (stores.size() == 1)
        candidates++;
      bool usedAffine = false;
      if (tryReplace(loop, usedAffine)) {
        replaced++;
        if (usedAffine)
          affineReplaced++;
      } else if (stores.size() == 1)
        rejectedShape++;
    }
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
