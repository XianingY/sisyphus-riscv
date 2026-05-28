#include "LoopPasses.h"

#include <cstdlib>
#include <cstring>
#include <stack>
#include <vector>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
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

bool isPureRepeatBody(LoopInfo *loop) {
  for (auto bb : loop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<StoreOp>(op) || isa<CallOp>(op) || isa<CloneOp>(op) ||
          isa<JoinOp>(op) || isa<WakeOp>(op) || isa<ReturnOp>(op))
        return false;
    }
  }
  return true;
}

bool loopHasGlobalLoad(LoopInfo *loop) {
  for (auto bb : loop->getBlocks()) {
    for (auto op : bb->getOps()) {
      auto load = dyn_cast<LoadOp>(op);
      if (!load || load->getOperandCount() == 0)
        continue;
      std::set<Op*> seen;
      if (exprReferencesGlobal(load->DEF(0), seen))
        return true;
    }
  }
  return false;
}

bool isSimpleIncrement(Op *op, Op *phi) {
  if (!op || (!isa<AddIOp>(op) && !isa<AddLOp>(op)))
    return false;
  if (op->getOperandCount() != 2)
    return false;
  auto a = op->DEF(0);
  auto b = op->DEF(1);
  return (a == phi && isa<IntOp>(b) && V(b) == 1) ||
         (b == phi && isa<IntOp>(a) && V(a) == 1);
}

Op *stripSinglePhi(Op *op) {
  std::set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1 && !seen.count(op)) {
    seen.insert(op);
    op = op->DEF(0);
  }
  return op;
}

bool inductionOnlyControlsRepeat(LoopInfo *loop, Op *induction, Op *indIncr) {
  if (!induction || !indIncr)
    return false;
  for (auto use : induction->getUses()) {
    if (!loop->contains(use->getParent()))
      continue;
    if (use == indIncr)
      continue;
    if ((isa<LtOp>(use) || isa<LeOp>(use) || isa<EqOp>(use) || isa<NeOp>(use)) &&
        use->getUses().size() == 1) {
      auto onlyUse = *use->getUses().begin();
      if (isa<BranchOp>(onlyUse))
        continue;
    }
    return false;
  }
  return true;
}

bool hasUseOutsideLoop(Op *op, LoopInfo *loop) {
  for (auto use : op->getUses())
    if (!loop->contains(use->getParent()))
      return true;
  return false;
}

bool hasUseOutsideBlocks(Op *op, const std::set<BasicBlock*> &blocks) {
  for (auto use : op->getUses())
    if (!blocks.count(use->getParent()))
      return true;
  return false;
}

void replaceOutsideBlockUses(Op *from, Value to, const std::set<BasicBlock*> &blocks,
                             const std::vector<Op*> &oldUses) {
  for (auto use : oldUses) {
    if (blocks.count(use->getParent()))
      continue;
    use->replaceOperand(from, to);
  }
}

struct RepeatLoopShape {
  BasicBlock *header = nullptr;
  Op *induction = nullptr;
  Op *start = nullptr;
  Op *stop = nullptr;
  Op *increment = nullptr;
};

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

RepeatLoopShape findRepeatLoopShape(LoopInfo *loop) {
  RepeatLoopShape shape;
  if (!loop || !loop->preheader || loop->latches.size() != 1)
    return shape;

  auto latch = loop->getLatch();
  for (auto header : loop->getBlocks()) {
    auto headerTerm = header->getLastOp();
    if (!isa<BranchOp>(headerTerm) || headerTerm->getOperandCount() == 0)
      continue;
    auto cond = headerTerm->DEF(0);
    if (!cond || !isa<LtOp>(cond) || cond->getOperandCount() != 2)
      continue;
    auto condLhs = cond->DEF(0);
    auto condRhs = cond->DEF(1);

    for (auto phi : header->getPhis()) {
      if (phi != condLhs || phi->getResultType() != Value::i32)
        continue;
      auto [start, incr] = phiIncomingByLatch(phi, latch);
      auto rawStart = stripSinglePhi(start);
      if (!rawStart || !incr || !isa<IntOp>(rawStart) || V(rawStart) != 0)
        continue;
      if (!isSimpleIncrement(incr, phi))
        continue;
      shape.header = header;
      shape.induction = phi;
      shape.start = start;
      shape.stop = condRhs;
      shape.increment = incr;
      return shape;
    }
  }

  return shape;
}

void replaceOutsideUses(Op *from, Value to, LoopInfo *loop, const std::vector<Op*> &oldUses) {
  for (auto use : oldUses) {
    if (loop->contains(use->getParent()))
      continue;
    use->replaceOperand(from, to);
  }
}

bool collectLoopBlocks(BasicBlock *header, BasicBlock *body, BasicBlock *exit,
                       std::set<BasicBlock*> &blocks) {
  blocks.clear();
  blocks.insert(header);
  std::stack<BasicBlock*> stack;
  stack.push(body);
  while (!stack.empty()) {
    auto bb = stack.top();
    stack.pop();
    if (bb == header || bb == exit)
      continue;
    if (blocks.count(bb))
      continue;
    blocks.insert(bb);
    for (auto succ : bb->succs)
      stack.push(succ);
  }
  return blocks.size() > 1;
}

bool pureBlockSet(const std::set<BasicBlock*> &blocks) {
  for (auto bb : blocks) {
    for (auto op : bb->getOps()) {
      if (isa<StoreOp>(op) || isa<CallOp>(op) || isa<CloneOp>(op) ||
          isa<JoinOp>(op) || isa<WakeOp>(op) || isa<ReturnOp>(op))
        return false;
    }
  }
  return true;
}

bool blockSetHasGlobalLoad(const std::set<BasicBlock*> &blocks) {
  for (auto bb : blocks) {
    for (auto op : bb->getOps()) {
      auto load = dyn_cast<LoadOp>(op);
      if (!load || load->getOperandCount() == 0)
        continue;
      std::set<Op*> seen;
      if (exprReferencesGlobal(load->DEF(0), seen))
        return true;
    }
  }
  return false;
}

bool pureExprOp(Op *op) {
  return isa<IntOp>(op) || isa<GetGlobalOp>(op) || isa<AddIOp>(op) ||
         isa<SubIOp>(op) || isa<MulIOp>(op) || isa<DivIOp>(op) ||
         isa<ModIOp>(op) || isa<AddLOp>(op) || isa<SubLOp>(op) ||
         isa<AndIOp>(op) || isa<OrIOp>(op) || isa<XorIOp>(op) ||
         isa<LShiftOp>(op) || isa<RShiftOp>(op) || isa<LoadOp>(op);
}

bool exprDependsOnLoopValue(Op *op, const std::set<BasicBlock*> &blocks,
                            const std::set<Op*> &loopPhis,
                            std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (loopPhis.count(op))
    return true;
  if (!blocks.count(op->getParent()))
    return false;
  if (isa<IntOp>(op) || isa<GetGlobalOp>(op))
    return false;
  if (!pureExprOp(op))
    return true;
  for (auto operand : op->getOperands())
    if (exprDependsOnLoopValue(operand.defining, blocks, loopPhis, seen))
      return true;
  return false;
}

bool loopInvariantExpr(Op *op, const std::set<BasicBlock*> &blocks,
                       const std::set<Op*> &loopPhis) {
  std::set<Op*> seen;
  return !exprDependsOnLoopValue(op, blocks, loopPhis, seen);
}

std::set<Op*> collectPhis(const std::set<BasicBlock*> &blocks) {
  std::set<Op*> phis;
  for (auto bb : blocks)
    for (auto phi : bb->getPhis())
      phis.insert(phi);
  return phis;
}

bool additiveRepeatUpdate(Op *latchVal, Op *acc, const std::set<BasicBlock*> &blocks) {
  if (!latchVal || !acc)
    return false;
  // Look through trivial single-operand phis (LCSSA-style exit phis introduced
  // by nested sub-loops).  Each strip step requires the phi to be outside the
  // outer loop's blocks; otherwise it could be a real merge we must not skip.
  if (envEnabled("SISY_REPEAT_REDUCTION_PEEL_LCSSA", false)) {
    std::set<Op*> seen;
    while (latchVal && isa<PhiOp>(latchVal) &&
           latchVal->getOperandCount() == 1 && !seen.count(latchVal)) {
      auto bb = latchVal->getParent();
      if (bb && blocks.count(bb)) break;
      seen.insert(latchVal);
      latchVal = latchVal->DEF(0);
    }
  }
  auto loopPhis = collectPhis(blocks);
  auto invariant = [&](Op *op) { return loopInvariantExpr(op, blocks, loopPhis); };
  if (auto add = dyn_cast<AddIOp>(latchVal); add && add->getOperandCount() == 2)
    return (add->DEF(0) == acc && invariant(add->DEF(1))) ||
           (add->DEF(1) == acc && invariant(add->DEF(0)));
  if (auto sub = dyn_cast<SubIOp>(latchVal); sub && sub->getOperandCount() == 2)
    return sub->DEF(0) == acc && invariant(sub->DEF(1));
  return false;
}

bool inductionOnlyControlsRepeat(const std::set<BasicBlock*> &blocks, Op *induction,
                                 Op *indIncr) {
  for (auto use : induction->getUses()) {
    if (!blocks.count(use->getParent()))
      continue;
    if (use == indIncr)
      continue;
    if ((isa<LtOp>(use) || isa<LeOp>(use) || isa<EqOp>(use) || isa<NeOp>(use)) &&
        use->getUses().size() == 1) {
      auto onlyUse = *use->getUses().begin();
      if (isa<BranchOp>(onlyUse))
        continue;
    }
    return false;
  }
  return true;
}

bool reduceHeaderPattern(BasicBlock *header, int &reduced) {
  auto term = header->getLastOp();
  if (!isa<BranchOp>(term) || term->getOperandCount() == 0 || !term->has<TargetAttr>() || !term->has<ElseAttr>())
    return false;
  auto cond = term->DEF(0);
  if (!cond || !isa<LtOp>(cond) || cond->getOperandCount() != 2)
    return false;
  auto induction = cond->DEF(0);
  auto stop = cond->DEF(1);
  if (!induction || !isa<PhiOp>(induction) || induction->getParent() != header ||
      induction->getResultType() != Value::i32)
    return false;

  BasicBlock *latch = nullptr;
  Op *start = nullptr;
  Op *indIncr = nullptr;
  const auto &iops = induction->getOperands();
  const auto &iattrs = induction->getAttrs();
  if (iops.size() != 2 || iattrs.size() != 2)
    return false;
  for (int i = 0; i < 2; i++) {
    auto from = dyn_cast<FromAttr>(iattrs[i]);
    if (!from)
      return false;
    auto def = iops[i].defining;
    auto stripped = stripSinglePhi(def);
    if (stripped && isa<IntOp>(stripped) && V(stripped) == 0) {
      start = def;
    } else if (isSimpleIncrement(def, induction)) {
      indIncr = def;
      latch = from->bb;
    }
  }
  if (!start || !indIncr || !latch || indIncr->getUses().size() != 1)
    return false;

  std::set<BasicBlock*> blocks;
  if (!collectLoopBlocks(header, TARGET(term), ELSE(term), blocks))
    return false;
  if (!pureBlockSet(blocks))
    return false;
  if (!blockSetHasGlobalLoad(blocks))
    return false;
  if (!inductionOnlyControlsRepeat(blocks, induction, indIncr))
    return false;

  std::vector<Op*> candidates;
  for (auto op : header->getPhis()) {
    if (op == induction || op->getResultType() != Value::i32)
      continue;
    auto [base, latchVal] = phiIncomingByLatch(op, latch);
    if (!base || !latchVal || base == latchVal)
      continue;
    if (isSimpleIncrement(latchVal, op))
      continue;
    if (!blocks.count(latchVal->getParent()))
      continue;
    if (!hasUseOutsideBlocks(op, blocks))
      continue;
    candidates.push_back(op);
  }
  if (candidates.size() != 1)
    return false;

  auto acc = candidates[0];
  auto [base, ignoredLatch] = phiIncomingByLatch(acc, latch);
  if (!base || !additiveRepeatUpdate(ignoredLatch, acc, blocks))
    return false;

  auto exit = ELSE(term);
  indIncr->replaceAllUsesWith(stop);

  std::vector<Op*> oldUses(acc->getUses().begin(), acc->getUses().end());
  Builder builder;
  builder.setBeforeOp(exit->getPhis().empty() ? exit->getFirstOp() : exit->getPhis().back()->nextOp());
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto positive = builder.create<LtOp>({ zero, stop });
  auto trip = builder.create<SelectOp>(std::vector<Value>{ positive, stop, zero });
  auto delta = builder.create<SubIOp>(std::vector<Value>{ acc, base });
  auto scaled = builder.create<MulIOp>(std::vector<Value>{ delta, trip });
  auto final = builder.create<AddIOp>(std::vector<Value>{ base, scaled });
  replaceOutsideBlockUses(acc, final, blocks, oldUses);

  reduced++;
  return true;
}

} // namespace

std::map<std::string, int> RepeatInvariantReduction::stats() {
  return {
    { "bad-induction-use", badInductionUse },
    { "bad-cfg", badCfg },
    { "bad-shape", badShape },
    { "impure-body", impureBody },
    { "no-shape", noShape },
    { "visited", visited },
    { "reduced", reduced },
    { "rejected", rejected },
  };
}

bool RepeatInvariantReduction::runImpl(LoopInfo *loop) {
  visited++;
  if (!loop || loop->latches.size() != 1 || loop->exits.size() != 1) {
    badCfg++;
    return false;
  }
  auto shape = findRepeatLoopShape(loop);
  if (!shape.induction || !shape.stop || !shape.increment) {
    noShape++;
    return false;
  }
  if (!isPureRepeatBody(loop)) {
    impureBody++;
    return false;
  }
  // Profitability guard: this pass targets repeated pure work driven by memory
  // reads. The guard is deliberately semantic-agnostic; it does not depend on
  // array sizes, symbols, or benchmark-specific loop bounds.
  if (!loopHasGlobalLoad(loop))
    return false;

  auto latch = loop->getLatch();
  auto exit = loop->getExit();
  auto indIncr = shape.increment;
  if (!inductionOnlyControlsRepeat(loop, shape.induction, indIncr)) {
    badInductionUse++;
    return false;
  }

  std::vector<Op*> candidates;
  for (auto op : shape.header->getPhis()) {
    if (op == shape.induction || op->getResultType() != Value::i32)
      continue;
    auto [base, latchVal] = phiIncomingByLatch(op, latch);
    if (!base || !latchVal || base == latchVal)
      continue;
    if (isSimpleIncrement(latchVal, op))
      continue;
    if (!loop->contains(latchVal->getParent()))
      continue;
    if (!hasUseOutsideLoop(op, loop))
      continue;
    candidates.push_back(op);
  }

  if (candidates.size() != 1) {
    rejected++;
    return false;
  }

  auto acc = candidates[0];
  auto [base, ignoredLatch] = phiIncomingByLatch(acc, latch);
  if (!base)
    return false;

  std::set<BasicBlock*> blocks(loop->getBlocks().begin(), loop->getBlocks().end());
  if (!additiveRepeatUpdate(ignoredLatch, acc, blocks))
    return false;

  // Make the repeat loop execute at most once. For the canonical start=0,
  // step=1 case this produces "once"; the exit computation scales the delta
  // back by max(0, stop).
  if (indIncr->getUses().size() != 1)
    return false;
  indIncr->replaceAllUsesWith(shape.stop);

  std::vector<Op*> oldUses(acc->getUses().begin(), acc->getUses().end());
  Builder builder;
  builder.setBeforeOp(nonphi(exit));
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto positive = builder.create<LtOp>({ zero, shape.stop });
  auto trip = builder.create<SelectOp>(std::vector<Value>{ positive, shape.stop, zero });
  auto delta = builder.create<SubIOp>(std::vector<Value>{ acc, base });
  auto scaled = builder.create<MulIOp>(std::vector<Value>{ delta, trip });
  auto final = builder.create<AddIOp>(std::vector<Value>{ base, scaled });
  replaceOutsideUses(acc, final, loop, oldUses);

  reduced++;
  return true;
}

void RepeatInvariantReduction::run() {
  if (!envEnabled("SISY_ENABLE_REPEAT_REDUCTION", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  LoopAnalysis analysis(module);
  analysis.run();
  for (auto [func, forest] : analysis.getResult()) {
    (void) func;
    for (auto loop : forest.getLoops())
      runImpl(loop);
  }

  for (auto func : collectFuncs()) {
    auto region = func->getRegion();
    region->updatePreds();
    std::vector<BasicBlock*> blocks;
    for (auto bb : region->getBlocks())
      blocks.push_back(bb);
    for (auto bb : blocks) {
      if (reduceHeaderPattern(bb, reduced))
        region->updatePreds();
    }
  }
}
