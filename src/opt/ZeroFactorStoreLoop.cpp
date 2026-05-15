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

bool hasLargeGlobalArray(Op *op, const std::map<std::string, GlobalOp*> &globals) {
  auto names = globalsIn(op);
  for (auto &name : names) {
    auto it = globals.find(name);
    if (it == globals.end())
      continue;
    auto global = it->second;
    if (!global->has<DimensionAttr>())
      continue;
    long long elems = 1;
    for (int dim : DIM(global))
      elems *= dim;
    if (elems >= 262144)
      return true;
  }
  return false;
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

bool matchStoreRecurrence(StoreOp *store, Op *innerIV,
                          const std::map<std::string, GlobalOp*> &globals,
                          StoreRecurrence &rec) {
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
    if (!hasLargeGlobalArray(factor->DEF(0), globals) ||
        !hasLargeGlobalArray(srcLoad->DEF(0), globals) ||
        !hasLargeGlobalArray(dstAddr, globals))
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

bool tryReplace(LoopInfo *loop, const std::map<std::string, GlobalOp*> &globals) {
  auto shape = findUnitLoopShape(loop);
  if (!shape.induction || !shape.stop)
    return false;
  auto stores = collectStores(loop);
  if (stores.size() != 1)
    return false;
  if (loopHasForbiddenSideEffect(loop, stores[0]))
    return false;
  StoreRecurrence rec;
  if (!matchStoreRecurrence(stores[0], shape.induction, globals, rec))
    return false;
  return buildZeroCopyPath(loop, shape, rec);
}

} // namespace

std::map<std::string, int> ZeroFactorStoreLoop::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void ZeroFactorStoreLoop::run() {
  if (!envEnabled("SISY_ENABLE_ZERO_FACTOR_STORE_LOOP", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  auto globals = getGlobalMap();
  LoopAnalysis analysis(module);
  analysis.run();
  for (auto &[_, forest] : analysis.getResult()) {
    for (auto loop : forest.getLoops()) {
      auto stores = collectStores(loop);
      if (stores.size() == 1)
        candidates++;
      if (tryReplace(loop, globals))
        replaced++;
      else if (stores.size() == 1)
        rejectedShape++;
    }
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
