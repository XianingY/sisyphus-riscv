#include "LoopPasses.h"

using namespace sys;

std::map<std::string, int> BoundsCheck::stats() {
  return {
    { "branches-eliminated", branchesEliminated },
    { "bounds-proved", boundsProved },
    { "bounds-rejected", boundsRejected },
  };
}

namespace {

Op *peel(Op *op) {
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1)
    op = op->DEF();
  return op;
}

struct Affine {
  Op *base = nullptr;
  int offset = 0;
  bool valid = false;
  bool constant = false;
};

struct IvInfo {
  Op *iv = nullptr;
  Op *start = nullptr;
  Op *stop = nullptr;
  Op *step = nullptr;
};

Affine affine(Op *op) {
  op = peel(op);
  if (!op)
    return {};
  if (auto *c = dyn_cast<IntOp>(op))
    return { nullptr, V(c), true, true };
  if (auto *add = dyn_cast<AddIOp>(op)) {
    auto lhs = affine(add->DEF(0));
    auto rhs = affine(add->DEF(1));
    if (lhs.valid && rhs.constant)
      return { lhs.base, lhs.offset + rhs.offset, true, lhs.constant };
    if (rhs.valid && lhs.constant)
      return { rhs.base, rhs.offset + lhs.offset, true, rhs.constant };
    return {};
  }
  if (auto *sub = dyn_cast<SubIOp>(op)) {
    auto lhs = affine(sub->DEF(0));
    auto rhs = affine(sub->DEF(1));
    if (lhs.valid && rhs.constant)
      return { lhs.base, lhs.offset - rhs.offset, true, lhs.constant };
    return {};
  }
  return { op, 0, true, false };
}

bool extractIndexOffset(Op *expr, Op *iv, int &offset) {
  auto a = affine(expr);
  if (!a.valid || a.constant || peel(a.base) != peel(iv))
    return false;
  offset = a.offset;
  return true;
}

bool isAddStep(Op *op, Op *iv, Op *&step) {
  op = peel(op);
  auto *add = dyn_cast<AddIOp>(op);
  if (!add)
    return false;
  if (peel(add->DEF(0)) == peel(iv)) {
    step = add->DEF(1);
    return true;
  }
  if (peel(add->DEF(1)) == peel(iv)) {
    step = add->DEF(0);
    return true;
  }
  return false;
}

Op *findStopFromTerm(Op *term, Op *iv) {
  auto *branch = dyn_cast<BranchOp>(term);
  if (!branch || !branch->getOperandCount())
    return nullptr;
  auto *cond = peel(branch->DEF());
  auto *lt = dyn_cast<LtOp>(cond);
  if (!lt)
    return nullptr;
  if (peel(lt->DEF(0)) == peel(iv))
    return lt->DEF(1);
  if (auto *add = dyn_cast<AddIOp>(peel(lt->DEF(0)))) {
    if (peel(add->DEF(0)) == peel(iv) || peel(add->DEF(1)) == peel(iv))
      return lt->DEF(1);
  }
  return nullptr;
}

std::vector<IvInfo> collectIvs(LoopInfo *loop) {
  std::vector<IvInfo> result;
  if (!loop->preheader || loop->latches.size() != 1)
    return result;
  auto *latch = loop->getLatch();

  for (auto *phi : loop->header->getPhis()) {
    if (phi->getOperandCount() != 2)
      continue;
    auto from0 = FROM(phi->getAttrs()[0]);
    auto from1 = FROM(phi->getAttrs()[1]);
    auto *def0 = phi->DEF(0);
    auto *def1 = phi->DEF(1);
    if (from0 == latch && from1 == loop->preheader) {
      std::swap(from0, from1);
      std::swap(def0, def1);
    }
    if (from0 != loop->preheader || from1 != latch)
      continue;

    Op *step = nullptr;
    if (!isAddStep(def1, phi, step) || !isa<IntOp>(step) || V(step) <= 0)
      continue;

    Op *stop = findStopFromTerm(loop->header->getLastOp(), phi);
    if (!stop)
      stop = findStopFromTerm(latch->getLastOp(), phi);
    if (!stop)
      continue;

    result.push_back({ phi, def0, stop, step });
  }
  return result;
}

bool proveLower(Op *idx, const IvInfo &iv) {
  int offset = 0;
  if (!extractIndexOffset(idx, iv.iv, offset))
    return false;

  auto start = affine(iv.start);
  if (!start.valid || !start.constant)
    return false;
  return start.offset + offset >= 0;
}

bool proveUpper(Op *idx, Op *bound, bool inclusive, const IvInfo &iv) {
  int idxOffset = 0;
  if (!extractIndexOffset(idx, iv.iv, idxOffset))
    return false;

  auto stop = affine(iv.stop);
  auto upper = affine(bound);
  if (!stop.valid || !upper.valid)
    return false;

  if (stop.constant && upper.constant) {
    int maxIdx = stop.offset - 1 + idxOffset;
    return inclusive ? maxIdx <= upper.offset : maxIdx < upper.offset;
  }

  if (stop.constant || upper.constant || peel(stop.base) != peel(upper.base))
    return false;

  // The loop is canonicalized as iv < stop for positive steps, so the last
  // possible iv is <= stop - 1.  Compare offsets relative to the same base.
  int maxIdxOffset = stop.offset - 1 + idxOffset;
  return inclusive ? maxIdxOffset <= upper.offset : maxIdxOffset < upper.offset;
}

bool provePredicateTrue(Op *cond, const IvInfo &iv) {
  cond = peel(cond);
  if (!cond)
    return false;
  if (auto *c = dyn_cast<IntOp>(cond))
    return V(c) != 0;
  if (auto *andOp = dyn_cast<AndIOp>(cond))
    return provePredicateTrue(andOp->DEF(0), iv) &&
           provePredicateTrue(andOp->DEF(1), iv);
  if (auto *lt = dyn_cast<LtOp>(cond))
    return proveUpper(lt->DEF(0), lt->DEF(1), /*inclusive=*/ false, iv);
  if (auto *le = dyn_cast<LeOp>(cond)) {
    auto lhs = peel(le->DEF(0));
    auto rhs = peel(le->DEF(1));
    if (auto *c = dyn_cast<IntOp>(lhs); c && V(c) == 0)
      return proveLower(rhs, iv);
    return proveUpper(lhs, rhs, /*inclusive=*/ true, iv);
  }
  return false;
}

void removePhiIncoming(BasicBlock *bb, BasicBlock *from) {
  if (!bb || !from)
    return;
  for (auto *phi : bb->getPhis()) {
    for (int i = (int) phi->getAttrs().size() - 1; i >= 0; i--) {
      if (FROM(phi->getAttrs()[i]) == from) {
        phi->removeOperand(i);
        phi->removeAttribute(i);
      }
    }
  }
}

} // namespace

void BoundsCheck::runImpl(LoopInfo *loop) {
  for (auto *subloop : loop->subloops)
    runImpl(subloop);

  if (!loop->subloops.empty())
    return;
  auto ivs = collectIvs(loop);
  if (ivs.empty())
    return;

  Builder builder;

  for (auto *bb : loop->getBlocks()) {
    auto *term = bb->getLastOp();
    auto *branch = dyn_cast<BranchOp>(term);
    if (!branch || !branch->has<TargetAttr>() || !branch->has<ElseAttr>())
      continue;
    if (!loop->contains(TARGET(branch)) || !loop->contains(ELSE(branch)))
      continue;

    bool proven = false;
    for (const auto &iv : ivs) {
      if (provePredicateTrue(branch->DEF(), iv)) {
        proven = true;
        break;
      }
    }
    if (!proven) {
      boundsRejected++;
      continue;
    }

    auto *trueTarget = TARGET(branch);
    auto *falseTarget = ELSE(branch);
    if (trueTarget != falseTarget)
      removePhiIncoming(falseTarget, bb);
    builder.replace<GotoOp>(branch, { new TargetAttr(trueTarget) });
    boundsProved++;
    branchesEliminated++;
  }
}

void BoundsCheck::run() {
  LoopAnalysis analysis(module);
  analysis.run();
  auto forests = analysis.getResult();

  for (auto *func : collectFuncs()) {
    auto &forest = forests[func];
    for (auto *loop : forest.getLoops()) {
      if (!loop->getParent())
        runImpl(loop);
    }
    func->getRegion()->updatePreds();
  }
}
