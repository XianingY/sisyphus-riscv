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

bool isAddOne(Op *op, Op *phi) {
  if (!op || !isa<AddIOp>(op) || op->getOperandCount() != 2)
    return false;
  return (op->DEF(0) == phi && isConst(op->DEF(1), 1)) ||
         (op->DEF(1) == phi && isConst(op->DEF(0), 1));
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
      use->replaceOperand(from, to);
}

} // namespace

std::map<std::string, int> DivPow2LoopFold::stats() {
  return {
    { "visited", visited },
    { "folded", folded },
    { "rejected", rejected },
  };
}

bool DivPow2LoopFold::runImpl(LoopInfo *loop) {
  visited++;
  if (!loop || !loop->preheader || loop->latches.size() != 1 || loop->exits.size() != 1 ||
      loop->getBlocks().size() != 2) {
    rejected++;
    return false;
  }

  auto header = loop->header;
  auto latch = loop->getLatch();
  auto exit = loop->getExit();
  auto branch = dyn_cast<BranchOp>(header->getLastOp());
  if (!branch || !branch->has<TargetAttr>() || !branch->has<ElseAttr>() ||
      TARGET(branch) != latch || ELSE(branch) != exit || branch->getOperandCount() != 1) {
    rejected++;
    return false;
  }
  auto cond = branch->DEF(0);
  if (!cond || !isa<LtOp>(cond) || cond->getOperandCount() != 2) {
    rejected++;
    return false;
  }
  auto stop = cond->DEF(1);
  bool constTrip = isa<IntOp>(stop);
  int trip = constTrip ? V(stop) : -1;
  if (constTrip && (trip < 0 || trip > 16)) {
    rejected++;
    return false;
  }

  auto induction = cond->DEF(0);
  if (!induction || !isa<PhiOp>(induction) || induction->getParent() != header) {
    rejected++;
    return false;
  }

  Op *valuePhi = nullptr;
  Op *valueBase = nullptr;
  Op *valueBack = nullptr;
  Op *indBack = nullptr;
  for (auto phi : header->getPhis()) {
    auto [base, back] = incomingByLatch(phi, latch);
    if (!base || !back)
      continue;
    if (phi == induction) {
      if (!isConst(base, 0) || !isAddOne(back, phi)) {
        rejected++;
        return false;
      }
      indBack = back;
      continue;
    }
    if (phi->getResultType() == Value::i32 && isa<DivIOp>(back) &&
        back->getOperandCount() == 2 && back->DEF(0) == phi && isConst(back->DEF(1), 16) &&
        outsideLoopUse(phi, loop)) {
      valuePhi = phi;
      valueBase = base;
      valueBack = back;
    }
  }
  if (!indBack || !valuePhi || !valueBase || !valueBack) {
    rejected++;
    return false;
  }

  for (auto bb : loop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<PhiOp>(op) || op == cond || op == branch || op == valueBack ||
          op == indBack || isa<IntOp>(op) || isa<GotoOp>(op))
        continue;
      rejected++;
      return false;
    }
  }

  auto preterm = loop->preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || TARGET(preterm) != header) {
    rejected++;
    return false;
  }

  Builder builder;
  builder.setBeforeOp(preterm);
  Op *result = valueBase;
  if (constTrip) {
    for (int i = 0; i < trip; i++) {
      auto sixteen = builder.create<IntOp>({ new IntAttr(16) });
      result = builder.create<DivIOp>({ result, sixteen });
    }
  } else {
    for (int i = 0; i < 8; i++) {
      auto sixteen = builder.create<IntOp>({ new IntAttr(16) });
      auto divided = builder.create<DivIOp>({ result, sixteen });
      auto guard = builder.create<LtOp>({ builder.create<IntOp>({ new IntAttr(i) }), stop });
      result = builder.create<SelectOp>(std::vector<Value>{ guard, divided, result });
    }
  }

  replaceOutsideUses(valuePhi, result, loop);
  for (auto phi : exit->getPhis()) {
    bool fromLoop = false;
    for (auto attr : phi->getAttrs())
      if (auto from = dyn_cast<FromAttr>(attr); from && loop->contains(from->bb))
        fromLoop = true;
    if (!fromLoop)
      continue;
    if (phi->getResultType() == Value::i32) {
      phi->replaceAllUsesWith(result);
      phi->erase();
    }
  }

  builder.replace<GotoOp>(preterm, { new TargetAttr(exit) });
  folded++;
  return true;
}

void DivPow2LoopFold::run() {
  if (!envEnabled("SISY_ENABLE_DIV_POW2_LOOP_FOLD", true))
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

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
