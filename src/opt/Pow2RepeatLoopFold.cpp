#include "LoopPasses.h"

#include <cstdlib>
#include <cstring>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

bool isConst(Op *op, int value) {
  return op && isa<IntOp>(op) && V(op) == value;
}

bool isDecOne(Op *op, Op *phi) {
  if (!op || op->getOperandCount() != 2)
    return false;
  if (isa<SubIOp>(op))
    return op->DEF(0) == phi && isConst(op->DEF(1), 1);
  if (isa<AddIOp>(op))
    return (op->DEF(0) == phi && isConst(op->DEF(1), -1)) ||
           (op->DEF(1) == phi && isConst(op->DEF(0), -1));
  return false;
}

bool isDouble(Op *op, Op *phi) {
  if (!op || op->getOperandCount() != 2)
    return false;
  if (isa<MulIOp>(op))
    return (op->DEF(0) == phi && isConst(op->DEF(1), 2)) ||
           (op->DEF(1) == phi && isConst(op->DEF(0), 2));
  if (isa<AddIOp>(op))
    return op->DEF(0) == phi && op->DEF(1) == phi;
  if (isa<LShiftOp>(op))
    return op->DEF(0) == phi && isConst(op->DEF(1), 1);
  return false;
}

std::pair<Op*, Op*> incomingByLatch(Op *phi, BasicBlock *latch) {
  Op *base = nullptr;
  Op *back = nullptr;
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  if (ops.size() != attrs.size())
    return { nullptr, nullptr };
  for (int i = 0; i < ops.size(); i++) {
    auto from = dyn_cast<FromAttr>(attrs[i]);
    if (!from)
      continue;
    if (from->bb == latch)
      back = ops[i].defining;
    else
      base = ops[i].defining;
  }
  return { base, back };
}

bool outsideLoopUse(Op *op, LoopInfo *loop) {
  for (auto use : op->getUses())
    if (!loop->contains(use->getParent()))
      return true;
  return false;
}

void replaceOutsideUses(Op *from, Op *to, LoopInfo *loop) {
  std::vector<Op*> uses(from->getUses().begin(), from->getUses().end());
  for (auto use : uses)
    if (!loop->contains(use->getParent()))
      use->replaceOperand(from, to->getResult());
}

bool condExitsOnZero(Op *cond, Op *counterPhi, BasicBlock *trueTarget,
                     BasicBlock *falseTarget, BasicBlock *latch,
                     BasicBlock *exit) {
  if (cond == counterPhi)
    return trueTarget == latch && falseTarget == exit;

  if (!cond || cond->getOperandCount() != 2)
    return false;

  Op *other = nullptr;
  if (cond->DEF(0) == counterPhi)
    other = cond->DEF(1);
  else if (cond->DEF(1) == counterPhi)
    other = cond->DEF(0);
  else
    return false;
  if (!isConst(other, 0))
    return false;

  if (isa<NeOp>(cond))
    return trueTarget == latch && falseTarget == exit;
  if (isa<EqOp>(cond))
    return trueTarget == exit && falseTarget == latch;
  return false;
}

bool purePow2LoopBody(LoopInfo *loop, Op *cond, Op *branch, Op *counterBack,
                      Op *valueBack) {
  for (auto bb : loop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<PhiOp>(op) || op == cond || op == branch || op == counterBack ||
          op == valueBack || isa<IntOp>(op) || isa<GotoOp>(op))
        continue;
      return false;
    }
  }
  return true;
}

} // namespace

std::map<std::string, int> Pow2RepeatLoopFold::stats() {
  return {
    { "bad-branch", badBranch },
    { "bad-cfg", badCfg },
    { "bad-counter", badCounter },
    { "bad-value", badValue },
    { "visited", visited },
    { "folded", folded },
    { "impure-body", impureBody },
    { "rejected", rejected },
  };
}

bool Pow2RepeatLoopFold::runImpl(LoopInfo *loop) {
  visited++;
  if (!loop || !loop->preheader || loop->latches.size() != 1 || loop->exits.size() != 1 ||
      loop->getBlocks().size() != 2) {
    badCfg++;
    rejected++;
    return false;
  }

  auto header = loop->header;
  auto latch = loop->getLatch();
  auto exit = loop->getExit();
  auto branch = dyn_cast<BranchOp>(header->getLastOp());
  if (!branch || !branch->has<TargetAttr>() || !branch->has<ElseAttr>() ||
      branch->getOperandCount() != 1) {
    badBranch++;
    rejected++;
    return false;
  }

  auto cond = branch->DEF(0);
  Op *counterPhi = nullptr;
  Op *counterBase = nullptr;
  Op *counterBack = nullptr;
  Op *valuePhi = nullptr;
  Op *valueBase = nullptr;
  Op *valueBack = nullptr;

  for (auto phi : header->getPhis()) {
    auto [base, back] = incomingByLatch(phi, latch);
    if (!base || !back || phi->getResultType() != Value::i32)
      continue;
    if (!counterPhi && isDecOne(back, phi) &&
        condExitsOnZero(cond, phi, TARGET(branch), ELSE(branch), latch, exit)) {
      counterPhi = phi;
      counterBase = base;
      counterBack = back;
      continue;
    }
  }

  if (!counterPhi || !counterBase || !counterBack) {
    badCounter++;
    rejected++;
    return false;
  }

  for (auto phi : header->getPhis()) {
    if (phi == counterPhi || phi->getResultType() != Value::i32)
      continue;
    auto [base, back] = incomingByLatch(phi, latch);
    if (!base || !back)
      continue;
    if (isDouble(back, phi) && outsideLoopUse(phi, loop)) {
      if (valuePhi) {
        rejected++;
        return false;
      }
      valuePhi = phi;
      valueBase = base;
      valueBack = back;
    }
  }

  if (!valuePhi || !valueBase || !valueBack) {
    badValue++;
    rejected++;
    return false;
  }

  if (!purePow2LoopBody(loop, cond, branch, counterBack, valueBack)) {
    impureBody++;
    rejected++;
    return false;
  }

  Builder builder;
  builder.setBeforeOp(loop->preheader->getLastOp());
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto one = builder.create<IntOp>({ new IntAttr(1) });
  auto thirtyTwo = builder.create<IntOp>({ new IntAttr(32) });
  auto nonNegative = builder.create<LeOp>({ zero, counterBase });
  auto small = builder.create<LtOp>({ counterBase, thirtyTwo });
  auto validShift = builder.create<AndIOp>({ nonNegative, small });
  auto shiftAmount = builder.create<SelectOp>(std::vector<Value>{ validShift, counterBase, zero });
  auto shifted = builder.create<LShiftOp>({ one, shiftAmount });
  auto factor = builder.create<SelectOp>(std::vector<Value>{ validShift, shifted, zero });
  auto result = builder.create<MulIOp>({ valueBase, factor });

  replaceOutsideUses(valuePhi, result, loop);
  counterBack->replaceAllUsesWith(zero);

  folded++;
  return true;
}

void Pow2RepeatLoopFold::run() {
  if (!envEnabled("SISY_ENABLE_POW2_REPEAT_LOOP_FOLD", true))
    return;

  for (int round = 0; round < 16; round++) {
    for (auto func : collectFuncs())
      func->getRegion()->updatePreds();

    bool changed = false;
    LoopAnalysis analysis(module);
    analysis.run();
    for (auto [func, forest] : analysis.getResult()) {
      (void) func;
      for (auto loop : forest.getLoops()) {
        if (runImpl(loop)) {
          changed = true;
          break;
        }
      }
      if (changed)
        break;
    }
    if (!changed)
      break;
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
