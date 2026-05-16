#include "LoopPasses.h"
#include "Passes.h"
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

int envIntClamped(const char *name, int fallback, int low, int high) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  int value = std::atoi(env);
  if (value < low || value > high)
    return fallback;
  return value;
}

Op *stripSinglePhi(Op *op) {
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1)
    op = op->DEF();
  return op;
}

int loopOpCount(LoopInfo *loop) {
  int loopsize = 0;
  for (auto bb : loop->getBlocks())
    loopsize += bb->getOpCount();
  return loopsize;
}

Op *headerStop(LoopInfo *loop, bool &inclusive) {
  inclusive = false;
  if (!loop || !loop->header || !loop->getInduction())
    return nullptr;
  auto term = dyn_cast<BranchOp>(loop->header->getLastOp());
  if (!term || term->getOperandCount() != 1)
    return nullptr;
  auto cond = stripSinglePhi(term->DEF());
  if (auto lt = dyn_cast<LtOp>(cond)) {
    if (stripSinglePhi(lt->DEF(0)) == stripSinglePhi(loop->getInduction()))
      return stripSinglePhi(lt->DEF(1));
  }
  if (auto le = dyn_cast<LeOp>(cond)) {
    if (stripSinglePhi(le->DEF(0)) == stripSinglePhi(loop->getInduction())) {
      inclusive = true;
      return stripSinglePhi(le->DEF(1));
    }
  }
  return nullptr;
}

bool smallConstantTrip(LoopInfo *loop, int loopsize, int &trip) {
  auto lower = stripSinglePhi(loop->getStart());
  auto upper = stripSinglePhi(loop->getStop());
  bool inclusive = false;
  if (!upper)
    upper = headerStop(loop, inclusive);
  auto step = stripSinglePhi(loop->getStepOp());
  if (!lower || !upper || !step || !isa<IntOp>(lower) ||
      !isa<IntOp>(upper) || !isa<IntOp>(step))
    return false;
  int stepValue = V(step);
  if (stepValue <= 0)
    return false;
  int span = V(upper) - V(lower) + (inclusive ? 1 : 0);
  if (span <= 0 || span % stepValue != 0)
    return false;
  trip = span / stepValue;
  if (trip <= 1 || trip > 16)
    return false;
  return loopsize > 0 && loopsize * (trip - 1) <= 200;
}

bool tryRotateSmallConstantLoop(LoopInfo *info) {
  if (!info || !info->preheader || info->latches.size() != 1 ||
      info->exits.size() != 1 || !info->getInduction())
    return false;

  int trip = 0;
  int loopsize = loopOpCount(info);
  if (!smallConstantTrip(info, loopsize, trip))
    return false;

  for (auto bb : info->getBlocks())
    for (auto op : bb->getOps())
      if (isa<CallOp>(op))
        return false;

  auto exit = info->getExit();
  auto induction = info->getInduction();
  auto header = info->header;
  auto term = dyn_cast<BranchOp>(header->getLastOp());
  if (!term || term->getOperandCount() != 1 || !term->has<TargetAttr>() ||
      !term->has<ElseAttr>() || ELSE(term) != exit)
    return false;

  auto cond = stripSinglePhi(term->DEF());
  Op *upper = nullptr;
  if (auto lt = dyn_cast<LtOp>(cond)) {
    if (stripSinglePhi(lt->DEF(0)) != stripSinglePhi(induction))
      return false;
    upper = stripSinglePhi(lt->DEF(1));
  } else {
    return false;
  }
  if (!upper || !isa<IntOp>(upper))
    return false;

  auto latch = info->getLatch();
  auto latchterm = latch->getLastOp();
  if (!isa<GotoOp>(latchterm) || !latchterm->has<TargetAttr>() ||
      TARGET(latchterm) != header)
    return false;

  auto preheader = info->preheader;
  auto preterm = preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || !preterm->has<TargetAttr>() ||
      TARGET(preterm) != header)
    return false;

  auto headerPhis = header->getPhis();
  std::map<Op*, Op*> valueMap, initMap;
  for (auto phi : headerPhis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    if (attrs.size() != 2 || ops.size() != 2)
      return false;

    auto bb1 = FROM(attrs[0]);
    auto bb2 = FROM(attrs[1]);
    if (bb1 == latch && bb2 == preheader) {
      valueMap[phi] = ops[0].defining;
      initMap[phi] = ops[1].defining;
    } else if (bb2 == latch && bb1 == preheader) {
      valueMap[phi] = ops[1].defining;
      initMap[phi] = ops[0].defining;
    } else {
      return false;
    }
  }
  if (!valueMap.count(induction))
    return false;

  for (auto phi : exit->getPhis()) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    for (size_t i = 0; i < ops.size(); i++) {
      if (FROM(attrs[i]) != header)
        continue;
      auto def = ops[i].defining;
      if (!valueMap.count(def) || !initMap.count(def))
        return false;
    }
  }

  Builder builder;
  builder.setBeforeOp(preterm);
  Value cmp = builder.create<LtOp>(std::vector<Value> { info->getStart(), upper });
  builder.replace<BranchOp>(preterm, { cmp }, { new TargetAttr(header), new ElseAttr(exit) });

  auto target = TARGET(term);
  builder.replace<GotoOp>(term, { new TargetAttr(target) });

  builder.setBeforeOp(latchterm);
  cmp = builder.create<LtOp>(std::vector<Value> { valueMap[induction], upper });
  builder.replace<BranchOp>(latchterm, { cmp }, { new TargetAttr(header), new ElseAttr(exit) });

  for (auto phi : exit->getPhis()) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    for (size_t i = 0; i < ops.size(); i++) {
      if (FROM(attrs[i]) != header)
        continue;
      auto def = ops[i].defining;
      FROM(attrs[i]) = latch;
      if (valueMap.count(def))
        phi->setOperand(i, valueMap[def]);
      phi->pushOperand(initMap.count(def) ? initMap[def] : def);
      phi->add<FromAttr>(preheader);
      break;
    }
  }

  info->stop = upper;
  header->getParent()->updatePreds();
  (void)trip;
  return true;
}

}

std::map<std::string, int> ConstLoopUnroll::stats() {
  return {
    { "unrolled", unrolled },
    { "factor-unrolled", factorUnrolled },
    { "rotated-for-unroll", rotatedForUnroll }
  };
}

BasicBlock *ConstLoopUnroll::copyLoop(LoopInfo *loop, BasicBlock *bb, int unroll) {
  std::map<Op*, Op*> cloneMap, revcloneMap, prevLatch;
  std::map<BasicBlock*, BasicBlock*> rewireMap;
  Builder builder;
  BasicBlock *latch = loop->getLatch();
  BasicBlock *lastLatch = loop->getLatch();
  BasicBlock *header = loop->header;
  BasicBlock *exit = loop->getExit();
  BasicBlock *latchRewire = nullptr;
  auto region = lastLatch->getParent();

  // We're rewiring it to header, so we've got a copy.
  --unroll;

  for (int i = 0; i < unroll; i++) {
    cloneMap.clear();
    revcloneMap.clear();
    rewireMap.clear();
    
    // First shallow copy everything.
    std::vector<Op*> created;

    for (auto block : loop->getBlocks()) {
      bb = region->insertAfter(bb);
      builder.setToBlockStart(bb);
      for (auto op : block->getOps()) {
        auto copied = builder.copy(op);
        cloneMap[op] = copied;
        created.push_back(copied);
        if (isa<PhiOp>(op))
          revcloneMap[copied] = op;
      }
      rewireMap[block] = bb;
    }

    // Rewire operands.
    for (auto op : created) {
      auto operands = op->getOperands();
      op->removeAllOperands();
      for (auto operand : operands) {
        auto def = operand.defining;
        op->pushOperand(cloneMap.count(def) ? cloneMap[def] : def);
      }
    }

    // Rewire blocks.
    for (auto [k, v] : rewireMap) {
      auto term = v->getLastOp();
      if (auto attr = term->find<TargetAttr>(); attr && rewireMap.count(attr->bb))
        attr->bb = rewireMap[attr->bb];
      if (auto attr = term->find<ElseAttr>(); attr && rewireMap.count(attr->bb))
        attr->bb = rewireMap[attr->bb];
    }

    // The current (unappended) latch of the loop should connect to the new region instead.
    // We shouldn't change the original latch for side-loop; otherwise all future copies will break.
    auto term = lastLatch->getLastOp();
    auto rewired = rewireMap[header];
    if (lastLatch != latch)
      builder.replace<GotoOp>(term, { new TargetAttr(rewired) });
    else
      latchRewire = rewired;

    // The new latch now branches to either the rewire header or exit.
    // Redirect it to the real header.

    auto curLatch = rewireMap[latch];
    term = curLatch->getLastOp();
    if (TARGET(term) == rewired)
      TARGET(term) = header;
    if (ELSE(term) == rewired)
      ELSE(term) = header;

    // Update the current latch.
    lastLatch = curLatch;
    
    // Replace phis at header.
    // All phis come from either the preheader or the latch.
    // Now the "preheader" is the previous latch. 
    // The value wouldn't come from the latch because it's no longer a predecessor.
    auto phis = rewireMap[header]->getPhis();

    for (auto copiedphi : phis) {
      auto origphi = revcloneMap[copiedphi];
      // We should use the updated version of the variable.
      // This means the operand from latch in the original phi (phiMap[origphi]).
      auto latchvalue = phiMap[origphi];

      // For the block succeeding the original loop body, `prevLatch` is empty.
      // Just use the latch value.
      Op *value;
      if (!prevLatch.count(latchvalue))
        value = latchvalue;
      else 
        // Otherwise, use `prevLatch` (which is actually the cloneMap of the previous iteration)
        // to find the inherited value.
        value = prevLatch[latchvalue];

      cloneMap[origphi] = value;
      copiedphi->replaceAllUsesWith(value);
      copiedphi->erase();
    }
    

    // All remaining phis should have their blocks renamed.
    // Note that `revcloneMap` contains all phis.
    std::set<Op*> erased(phis.begin(), phis.end());
    for (auto [k, _] : revcloneMap) {
      if (erased.count(k))
        continue;

      for (auto attr : k->getAttrs())
        FROM(attr) = rewireMap[FROM(attr)];
    }

    prevLatch = cloneMap;
  }

  // Rewire the old (first) latch now. It should go to `latchRewire`.
  auto term = latch->getLastOp();
  builder.replace<GotoOp>(term, { new TargetAttr(latchRewire) });

  // Also, the last latch can only get to exit. There's no chance of returning to header.
  auto final = lastLatch->getLastOp();
  builder.replace<GotoOp>(final, { new TargetAttr(exit) });

  // The exit can only receive operand from the latest latch.
  for (auto [k, v] : exitlatch) {
    Op *def = cloneMap.count(v) ? cloneMap[v] : v;
    
    int index = k->replaceOperand(v, def);
    k->setAttribute(index, new FromAttr(lastLatch));
  }

  // Phis at the header should also now point to the new latch.
  auto phis = header->getPhis();
  for (auto phi : phis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();

    // The side loop will only be executed once.
    // We remove the operand from latch.
    for (int i = 0; i < ops.size(); i++) {
      if (FROM(attrs[i]) == latch) {
        phi->removeOperand(i);
        phi->removeAttribute(i);
        break;
      }
    }
  }

  return lastLatch;
}

bool ConstLoopUnroll::runImpl(LoopInfo *loop) {
  if (!loop->getInduction())
    return false;

  if (loop->exits.size() != 1)
    return false;
  if (loop->latches.size() != 1)
    return false;

  auto header = loop->header;
  auto preheader = loop->preheader;
  auto latch = loop->getLatch();
  // The loop is not rotated. Don't unroll it.
  if (!isa<BranchOp>(latch->getLastOp())) {
    if (tryRotateSmallConstantLoop(loop)) {
      rotatedForUnroll++;
      preheader = loop->preheader;
    }
    if (!isa<BranchOp>(latch->getLastOp()))
      return false;
  }

  auto exit = loop->getExit();
  // We don't want an internal `break` to interfere.
  // It should come from either preheader or latch.
  if (exit->preds.size() > 2)
    return false;

  int loopsize = loopOpCount(loop);

  // Not every loop can be unrolled, even not all constant-bounded loops.
  // See 65_color.sy, where we are attempting to unroll a nested loop with a total of 18^5*7 = 13226976 iterations.
  // Relaxed limits for better performance on tight loops.
  if (loopsize > 500)
    return false;

  auto phis = header->getPhis();
  // Phis will give immense register pressure. Don't unroll when there are too many.
  if (phis.size() >= 5)
    return false;

  // Ensure that every phi at header is either from preheader or from the latch.
  // Also, finds the value of each phi from latch.
  phiMap.clear();
  for (auto phi : phis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    if (ops.size() != 2)
      return false;
    auto bb1 = cast<FromAttr>(attrs[0])->bb;
    auto bb2 = cast<FromAttr>(attrs[1])->bb;
    if (!(bb1 == latch && bb2 == preheader
       || bb2 == latch && bb1 == preheader))
      return false;

    if (bb1 == latch)
      phiMap[phi] = ops[0].defining;
    if (bb2 == latch)
      phiMap[phi] = ops[1].defining;
  }

  int unroll = -1;

  auto lower = loop->getStart();
  auto upper = loop->getStop();
  if (!upper)
    return false;

  // Check the real op.
  while (isa<PhiOp>(lower) && lower->getOperandCount() == 1)
    lower = lower->DEF();
  while (isa<PhiOp>(upper) && upper->getOperandCount() == 1)
    upper = upper->DEF();

  auto step = loop->getStepOp();
  if (!isa<IntOp>(step))
    return false;

  auto runtimeUnrollEnabled = []() {
    auto env = std::getenv("SISY_ENABLE_RUNTIME_UNROLL");
    return env && std::strcmp(env, "1") == 0;
  };

  auto estimateUpperBound = [](Op* expr) -> int64_t {
    if (!expr)
      return -1;
    // If it's a constant, return directly
    if (isa<IntOp>(expr))
      return V(expr);
    // If it's a GetArgOp (function argument), use conservative bound
    if (isa<GetArgOp>(expr)) {
      // Conservative upper bound for common size arguments.
      return 8192;
    }
    // If it's a division by constant, try to estimate the operand
    if (auto div = dyn_cast<DivIOp>(expr)) {
      auto rhs = div->DEF(1);
      if (isa<IntOp>(rhs)) {
        int divisor = V(rhs);
        if (divisor > 0) {
          auto lhs = div->DEF(0);
          // Recursively estimate the numerator
          if (isa<GetArgOp>(lhs)) {
            // Conservative estimate for size-like arguments.
            return 4096 / divisor;
          }
          // Check if lhs is a load from the function argument alloca
          if (auto load = dyn_cast<LoadOp>(lhs)) {
            auto src = load->DEF(0);
            // If the source is an alloca for a function argument, use conservative bound
            if (isa<AllocaOp>(src)) {
              return 4096 / divisor;
            }
          }
        }
      }
    }
    // For other expressions, we can't estimate
    return -1;
  };

  // Fully unroll constant-bounded loops if it's small enough.
  // Increased limit for better performance on tight inner loops.
  if (lower && upper && isa<IntOp>(lower) && isa<IntOp>(upper)) {
    int low = V(lower);
    int high = V(upper);
    int times = (high - low) / V(step);
    if (times <= 2000 / loopsize)
      unroll = times;
  } else if (runtimeUnrollEnabled()) {
    // Partial unrolling for runtime-bounded loops.
    // Try to estimate the upper bound
    int64_t estUpper = estimateUpperBound(upper);
    if (estUpper > 0 && loopsize > 0) {
      // Partial unrolling based on estimated trip count
      // Choose unroll factor based on loop size and estimated trip count
      int64_t estTimes = estUpper / V(step);

      // Calculate potential benefit: unrolling reduces loop overhead
      // and can expose ILP. Be more aggressive for small loops.
      if (loopsize <= 20) {
        // Very small tight loop - can unroll more
        if (estTimes <= 64)
          unroll = 8;
        else if (estTimes <= 256)
          unroll = 4;
        else
          unroll = 2;  // Large trip count but tight loop
      } else if (loopsize <= 50) {
        // Small loop
        if (estTimes <= 32)
          unroll = 4;
        else if (estTimes <= 256)
          unroll = 2;
        else
          unroll = 2;  // Conservative: even large loops benefit from 2x unroll
      }
      // Larger loops: don't partially unroll
    }
  }
  if (unroll == -1) {
    if (!envEnabled("SISY_ENABLE_FACTOR_UNROLL", true))
      return false;

    int factor = envIntClamped("SISY_FACTOR_UNROLL_FACTOR", 4, 2, 8);
    if (tryFactorUnroll(loop, factor)) {
      factorUnrolled++;
      return true;
    }
    return false;
  }
  if (unroll <= 1)
    return false;

  // Guardrail: keep duplicated loop body within a bounded growth budget.
  // Relaxed for better performance on small tight loops.
  int estimatedGrowth = loopsize * (unroll - 1);
  if (estimatedGrowth > 800)
    return false;

  // Record the phi values at the beginning of `exit` that are taken from the latch.
  // Note that "taken from latch" doesn't necessarily mean it's in the loop.
  // It can be from something that passes through all the loop, for example.
  auto exitphis = exit->getPhis();
  exitlatch.clear();
  for (auto phi : exitphis)
    exitlatch[phi] = Op::getPhiFrom(phi, latch);

  // We construct a side-loop with checks in it. The code is roughly the same.
  // Note that preheader won't change if it's used, because it's only used when `true`.
  copyLoop(loop, loop->getLatch(), unroll);

  unrolled++;
  return true;
}

// Factor unroll for non-constant trip-count loops.  This uses guarded copies:
// every copied latch keeps the original exit test, and only the final copied
// latch returns to the original header.  That preserves exact tail semantics
// without constructing a separate remainder loop.
bool ConstLoopUnroll::tryFactorUnroll(LoopInfo *loop, int factor) {
  if (factor <= 1)
    return false;

  auto induction = loop->getInduction();
  if (!induction)
    return false;
  if (!loop->getStart() || !loop->getStop())
    return false;

  auto step = loop->getStepOp();
  if (!step || !isa<IntOp>(step))
    return false;
  if (V(step) <= 0)
    return false;

  if (loop->exits.size() != 1)
    return false;
  if (loop->latches.size() != 1)
    return false;

  auto latch = loop->getLatch();
  if (!isa<BranchOp>(latch->getLastOp()))
    return false;

  auto header = loop->header;
  auto preheader = loop->preheader;
  if (!preheader)
    return false;

  auto exit = loop->getExit();
  if (exit->preds.size() > 2)
    return false;

  int loopsize = 0;
  for (auto bb : loop->getBlocks()) {
    loopsize += bb->getOpCount();
    for (auto op : bb->getOps()) {
      if (isa<CallOp>(op) && op->has<ImpureAttr>())
        return false;
    }
  }
  if (loopsize == 0 || loopsize >= 32)
    return false;
  if (loopsize * factor > 128)
    return false;

  auto phis = header->getPhis();
  if (phis.size() >= 5)
    return false;

  phiMap.clear();
  for (auto phi : phis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    if (ops.size() != 2)
      return false;
    auto bb1 = cast<FromAttr>(attrs[0])->bb;
    auto bb2 = cast<FromAttr>(attrs[1])->bb;
    bool expected = (bb1 == latch && bb2 == preheader)
                 || (bb2 == latch && bb1 == preheader);
    if (!expected)
      return false;

    if (bb1 == latch)
      phiMap[phi] = ops[0].defining;
    if (bb2 == latch)
      phiMap[phi] = ops[1].defining;
  }

  exitlatch.clear();
  for (auto phi : exit->getPhis())
    exitlatch[phi] = Op::getPhiFrom(phi, latch);

  std::map<Op*, Op*> prevLatch;
  BasicBlock *lastLatch = latch;
  BasicBlock *insertAfter = latch;
  auto region = latch->getParent();
  Builder builder;

  for (int i = 1; i < factor; i++) {
    std::map<Op*, Op*> cloneMap, revcloneMap;
    std::map<BasicBlock*, BasicBlock*> rewireMap;
    std::vector<Op*> created;

    for (auto block : loop->getBlocks()) {
      insertAfter = region->insertAfter(insertAfter);
      builder.setToBlockStart(insertAfter);
      for (auto op : block->getOps()) {
        auto copied = builder.copy(op);
        cloneMap[op] = copied;
        created.push_back(copied);
        if (isa<PhiOp>(op))
          revcloneMap[copied] = op;
      }
      rewireMap[block] = insertAfter;
    }

    for (auto op : created) {
      auto operands = op->getOperands();
      op->removeAllOperands();
      for (auto operand : operands) {
        auto def = operand.defining;
        op->pushOperand(cloneMap.count(def) ? cloneMap[def] : def);
      }
    }

    for (auto [oldBlock, newBlock] : rewireMap) {
      (void)oldBlock;
      auto term = newBlock->getLastOp();
      if (auto attr = term->find<TargetAttr>(); attr && rewireMap.count(attr->bb))
        attr->bb = rewireMap[attr->bb];
      if (auto attr = term->find<ElseAttr>(); attr && rewireMap.count(attr->bb))
        attr->bb = rewireMap[attr->bb];
    }

    auto rewiredHeader = rewireMap[header];
    auto previousTerm = lastLatch->getLastOp();
    bool rewiredBackedge = false;
    if (previousTerm->has<TargetAttr>() && TARGET(previousTerm) == header) {
      TARGET(previousTerm) = rewiredHeader;
      rewiredBackedge = true;
    }
    if (previousTerm->has<ElseAttr>() && ELSE(previousTerm) == header) {
      ELSE(previousTerm) = rewiredHeader;
      rewiredBackedge = true;
    }
    if (!rewiredBackedge)
      return false;

    auto clonedLatch = rewireMap[latch];
    auto clonedLatchTerm = clonedLatch->getLastOp();
    if (clonedLatchTerm->has<TargetAttr>() && TARGET(clonedLatchTerm) == rewiredHeader)
      TARGET(clonedLatchTerm) = header;
    if (clonedLatchTerm->has<ElseAttr>() && ELSE(clonedLatchTerm) == rewiredHeader)
      ELSE(clonedLatchTerm) = header;

    auto copiedHeaderPhis = rewireMap[header]->getPhis();
    for (auto copiedPhi : copiedHeaderPhis) {
      auto origPhi = revcloneMap[copiedPhi];
      auto latchValue = phiMap[origPhi];
      auto value = prevLatch.count(latchValue) ? prevLatch[latchValue] : latchValue;
      cloneMap[origPhi] = value;
      copiedPhi->replaceAllUsesWith(value);
      copiedPhi->erase();
    }

    std::set<Op*> erased(copiedHeaderPhis.begin(), copiedHeaderPhis.end());
    for (auto [copiedPhi, _] : revcloneMap) {
      if (erased.count(copiedPhi))
        continue;

      for (auto attr : copiedPhi->getAttrs())
        FROM(attr) = rewireMap[FROM(attr)];
    }

    for (auto [phi, latchValue] : exitlatch) {
      auto def = cloneMap.count(latchValue) ? cloneMap[latchValue] : latchValue;
      phi->pushOperand(def);
      phi->add<FromAttr>(clonedLatch);
    }

    prevLatch = cloneMap;
    lastLatch = clonedLatch;
  }

  for (auto phi : phis) {
    auto latchValue = phiMap[phi];
    auto backedgeValue = prevLatch.count(latchValue) ? prevLatch[latchValue] : latchValue;
    const auto &attrs = phi->getAttrs();
    for (int i = 0; i < (int) attrs.size(); i++) {
      if (FROM(attrs[i]) != latch)
        continue;
      phi->setOperand(i, backedgeValue);
      phi->setAttribute(i, new FromAttr(lastLatch));
      break;
    }
  }

  (void)induction;
  (void)step;
  return true;
}

static void postorder(LoopInfo *loop, std::vector<LoopInfo*> &order) {
  for (auto subloop : loop->subloops)
    postorder(subloop, order);
  order.push_back(loop);
}

void ConstLoopUnroll::run() {
  LoopAnalysis analysis(module);

  auto funcs = collectFuncs();
  for (auto func : funcs) {
    auto region = func->getRegion();
    auto forest = analysis.runImpl(region);

    bool changed;
    do {
      changed = false;
      // Don't unroll too much.
      if (region->getBlocks().size() > 1000)
        break;

      // We want to unroll small loops first.
      std::vector<LoopInfo*> order;
      for (auto loop : forest.getLoops()) {
        if (!loop->getParent())
          postorder(loop, order);
      }

      for (auto loop : order) {
        // We only want to unroll innermost loops.
        // Also, we can't unroll nested loops correctly.
        if (loop->subloops.size() > 0)
          continue;

        if (runImpl(loop)) {
          // We need to run a fold to completely flatten the loop.
          // (We've only copied the body without dealing with branches and phis.)
          forest = analysis.runImpl(region);
          changed = true;
          break;
        }
      }
    } while (changed);
  }
}
