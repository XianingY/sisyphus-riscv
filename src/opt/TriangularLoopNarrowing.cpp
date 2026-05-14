#include "LoopPasses.h"

using namespace sys;

std::map<std::string, int> TriangularLoopNarrowing::stats() {
  return {
    { "narrowed-lower", narrowedLower },
    { "rejected", rejected },
  };
}

namespace {

Op *peel(Op *op) {
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1)
    op = op->DEF();
  return op;
}

bool matchAffineBasePlusConst(Op *op, Op *base, int &offset) {
  op = peel(op);
  base = peel(base);
  if (!op || !base)
    return false;
  if (op == base) {
    offset = 0;
    return true;
  }
  if (auto *add = dyn_cast<AddIOp>(op)) {
    if (peel(add->DEF(0)) == base && isa<IntOp>(add->DEF(1))) {
      offset = V(add->DEF(1));
      return true;
    }
    if (peel(add->DEF(1)) == base && isa<IntOp>(add->DEF(0))) {
      offset = V(add->DEF(0));
      return true;
    }
  }
  if (auto *sub = dyn_cast<SubIOp>(op)) {
    if (peel(sub->DEF(0)) == base && isa<IntOp>(sub->DEF(1))) {
      offset = -V(sub->DEF(1));
      return true;
    }
  }
  return false;
}

struct HeaderIv {
  Op *phi = nullptr;
  Value start;
  int preIdx = -1;
  int step = 0;
  bool valid = false;
};

HeaderIv findHeaderIv(LoopInfo *loop) {
  HeaderIv result;
  if (!loop || !loop->preheader)
    return result;

  for (auto *phi : loop->header->getPhis()) {
    int preIdx = -1;
    Value start;
    int step = 0;
    bool good = true;

    for (int i = 0; i < phi->getOperandCount(); i++) {
      auto *from = FROM(phi->getAttrs()[i]);
      if (from == loop->preheader) {
        if (preIdx >= 0) {
          good = false;
          break;
        }
        preIdx = i;
        start = phi->getOperand(i);
        continue;
      }
      if (!loop->latches.count(from)) {
        good = false;
        break;
      }
      auto *def = phi->DEF(i);
      auto *add = dyn_cast<AddIOp>(peel(def));
      if (!add) {
        good = false;
        break;
      }
      Op *other = nullptr;
      if (peel(add->DEF(0)) == peel(phi))
        other = add->DEF(1);
      else if (peel(add->DEF(1)) == peel(phi))
        other = add->DEF(0);
      else {
        good = false;
        break;
      }
      if (!isa<IntOp>(other) || V(other) <= 0) {
        good = false;
        break;
      }
      if (!step)
        step = V(other);
      else if (step != V(other)) {
        good = false;
        break;
      }
    }

    if (good && preIdx >= 0 && step > 0)
      return { phi, start, preIdx, step, true };
  }

  return result;
}

bool singleGotoTo(BasicBlock *bb, BasicBlock *target) {
  if (!bb || bb->getOpCount() == 0)
    return false;
  auto *term = bb->getLastOp();
  return isa<GotoOp>(term) && TARGET(term) == target;
}

bool isSkipOnlyBackedge(BasicBlock *bb, BasicBlock *header, Op *iv) {
  if (!singleGotoTo(bb, header))
    return false;
  for (auto *op : bb->getOps()) {
    if (op == bb->getLastOp())
      break;
    if (isa<IntOp>(op))
      continue;
    if (auto *add = dyn_cast<AddIOp>(op)) {
      if ((peel(add->DEF(0)) == peel(iv) && isa<IntOp>(add->DEF(1))) ||
          (peel(add->DEF(1)) == peel(iv) && isa<IntOp>(add->DEF(0))))
        continue;
    }
    return false;
  }
  return true;
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

void TriangularLoopNarrowing::runImpl(LoopInfo *loop) {
  for (auto *sub : loop->subloops)
    runImpl(sub);

  if (!loop->parent || !loop->preheader)
    return;
  auto inner = findHeaderIv(loop);
  auto outer = findHeaderIv(loop->parent);
  if (!inner.valid || !outer.valid)
    return;
  auto *outerIv = outer.phi;

  auto *phi = inner.phi;
  auto *header = loop->header;
  BasicBlock *condBlock = nullptr;
  BasicBlock *skipBlock = nullptr;
  BasicBlock *bodyBlock = nullptr;
  int upperBias = 0;
  bool found = false;

  for (auto *bb : loop->getBlocks()) {
    auto *term = bb->getLastOp();
    auto *br = dyn_cast<BranchOp>(term);
    if (!br || !br->has<TargetAttr>() || !br->has<ElseAttr>())
      continue;
    auto *cond = peel(br->DEF());
    if (!cond)
      continue;

    int localBias = 0;
    bool matched = false;
    if (auto *lt = dyn_cast<LtOp>(cond)) {
      matched = matchAffineBasePlusConst(lt->DEF(0), outerIv, localBias) &&
                peel(lt->DEF(1)) == peel(phi);
      if (matched)
        upperBias = localBias + 1;
    } else if (auto *le = dyn_cast<LeOp>(cond)) {
      matched = matchAffineBasePlusConst(le->DEF(0), outerIv, localBias) &&
                peel(le->DEF(1)) == peel(phi);
      if (matched)
        upperBias = localBias;
    }
    if (!matched)
      continue;

    auto *trueTarget = TARGET(br);
    auto *falseTarget = ELSE(br);
    if (!loop->contains(trueTarget) || !loop->contains(falseTarget))
      continue;
    if (!isSkipOnlyBackedge(trueTarget, header, phi))
      continue;
    condBlock = bb;
    skipBlock = trueTarget;
    bodyBlock = falseTarget;
    found = true;
    break;
  }

  if (!found) {
    rejected++;
    return;
  }

  Builder builder;
  builder.setBeforeOp(loop->preheader->getLastOp());
  Value limit = outerIv;
  if (upperBias != 0) {
    auto delta = builder.create<IntOp>({ new IntAttr(upperBias) });
    limit = builder.create<AddIOp>(std::vector<Value>{ outerIv, delta });
  }
  auto *headerCond = dyn_cast<LtOp>(peel(header->getLastOp()->DEF()));
  if (!headerCond || peel(headerCond->DEF(0)) != peel(phi)) {
    rejected++;
    return;
  }
  auto oldStop = headerCond->getOperand(1);
  auto cmp = builder.create<LtOp>(std::vector<Value>{ limit, oldStop });
  auto narrowedStop = builder.create<SelectOp>(std::vector<Value>{ cmp, limit, oldStop });
  headerCond->setOperand(1, narrowedStop);

  builder.replace<GotoOp>(condBlock->getLastOp(), { new TargetAttr(bodyBlock) });
  removePhiIncoming(header, skipBlock);
  narrowedLower++;
}

void TriangularLoopNarrowing::run() {
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
