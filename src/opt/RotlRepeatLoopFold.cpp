#include "LoopPasses.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>

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

bool isRotl1(Op *op, Op *phi, Op *&mul, Op *&mod) {
  if (!op || !isa<AddIOp>(op) || op->getOperandCount() != 2)
    return false;
  auto a = op->DEF(0);
  auto b = op->DEF(1);
  auto matchMul = [&](Op *x) {
    return x && isa<MulIOp>(x) && x->getOperandCount() == 2 &&
           ((x->DEF(0) == phi && isConst(x->DEF(1), 2)) ||
            (x->DEF(1) == phi && isConst(x->DEF(0), 2)));
  };
  auto matchMod = [&](Op *x) {
    return x && isa<ModIOp>(x) && x->getOperandCount() == 2 &&
           x->DEF(0) == phi && isConst(x->DEF(1), 2);
  };
  if (matchMul(a) && matchMod(b)) {
    mul = a;
    mod = b;
    return true;
  }
  if (matchMul(b) && matchMod(a)) {
    mul = b;
    mod = a;
    return true;
  }
  return false;
}

void collectAllocaRefs(Op *op, std::set<Op*> &refs, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (isa<AllocaOp>(op)) {
    refs.insert(op);
    return;
  }
  for (auto operand : op->getOperands())
    collectAllocaRefs(operand.defining, refs, seen);
}

std::set<Op*> collectAllocaRefs(Op *op) {
  std::set<Op*> refs;
  std::set<Op*> seen;
  collectAllocaRefs(op, refs, seen);
  return refs;
}

bool referencesAlloca(Op *op, Op *alloca) {
  auto refs = collectAllocaRefs(op);
  return refs.count(alloca);
}

std::optional<std::pair<int, int>> minMaxConstStoredToLoadedAlloca(FuncOp *func, Op *stop) {
  auto load = dyn_cast<LoadOp>(stop);
  if (!load || load->getOperandCount() != 1)
    return std::nullopt;

  auto refs = collectAllocaRefs(load->DEF(0));
  if (refs.size() != 1)
    return std::nullopt;
  auto base = *refs.begin();

  bool sawStore = false;
  int minValue = 0;
  int maxValue = 0;
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() != 2 || !referencesAlloca(store->DEF(1), base))
      continue;
    auto value = store->DEF(0);
    if (!isa<IntOp>(value))
      return std::nullopt;
    if (!sawStore) {
      minValue = V(value);
      maxValue = V(value);
      sawStore = true;
    } else {
      minValue = std::min(minValue, V(value));
      maxValue = std::max(maxValue, V(value));
    }
  }
  if (!sawStore)
    return std::nullopt;
  return std::make_pair(minValue, maxValue);
}

std::optional<std::pair<int, int>> minMaxStopValue(FuncOp *func, Op *stop) {
  if (isa<IntOp>(stop))
    return std::make_pair(V(stop), V(stop));
  return minMaxConstStoredToLoadedAlloca(func, stop);
}

Op *createRotlStep(Builder &builder, Op *value) {
  auto twoA = builder.create<IntOp>({ new IntAttr(2) });
  auto mul = builder.create<MulIOp>({ value, twoA });
  auto twoB = builder.create<IntOp>({ new IntAttr(2) });
  auto mod = builder.create<ModIOp>({ value, twoB });
  return builder.create<AddIOp>({ mul, mod });
}

} // namespace

std::map<std::string, int> RotlRepeatLoopFold::stats() {
  return {
    { "visited", visited },
    { "folded", folded },
    { "rejected", rejected },
  };
}

bool RotlRepeatLoopFold::runImpl(FuncOp *func, LoopInfo *loop) {
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
  auto induction = cond->DEF(0);
  auto stop = cond->DEF(1);
  if (!induction || !isa<PhiOp>(induction) || induction->getParent() != header ||
      induction->getResultType() != Value::i32) {
    rejected++;
    return false;
  }

  Op *valuePhi = nullptr;
  Op *valueBase = nullptr;
  Op *valueBack = nullptr;
  Op *indBack = nullptr;
  Op *rotMul = nullptr;
  Op *rotMod = nullptr;
  for (auto phi : header->getPhis()) {
    auto [base, back] = incomingByLatch(phi, latch);
    if (!base || !back)
      continue;
    if (phi == induction) {
      if (!isConst(base, 1) || !isAddOne(back, phi)) {
        rejected++;
        return false;
      }
      indBack = back;
      continue;
    }
    Op *mul = nullptr;
    Op *mod = nullptr;
    if (phi->getResultType() == Value::i32 && isRotl1(back, phi, mul, mod) &&
        outsideLoopUse(phi, loop)) {
      valuePhi = phi;
      valueBase = base;
      valueBack = back;
      rotMul = mul;
      rotMod = mod;
    }
  }
  if (!indBack || !valuePhi || !valueBase || !valueBack || !rotMul || !rotMod) {
    rejected++;
    return false;
  }

  auto stopRange = minMaxStopValue(func, stop);
  if (!stopRange || stopRange->first < 1 || stopRange->second > 8) {
    rejected++;
    return false;
  }

  for (auto bb : loop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<PhiOp>(op) || op == cond || op == branch || op == valueBack ||
          op == indBack || op == rotMul || op == rotMod || isa<IntOp>(op) ||
          isa<GotoOp>(op))
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
  for (int i = 1; i < stopRange->first; i++)
    result = createRotlStep(builder, result);

  for (int i = stopRange->first; i < stopRange->second; i++) {
    auto rotated = createRotlStep(builder, result);
    auto idx = builder.create<IntOp>({ new IntAttr(i) });
    auto guard = builder.create<LtOp>({ idx, stop });
    result = builder.create<SelectOp>(std::vector<Value>{ guard, rotated, result });
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

void RotlRepeatLoopFold::run() {
  if (!envEnabled("SISY_ENABLE_ROTL_REPEAT_FOLD", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  LoopAnalysis analysis(module);
  analysis.run();
  for (auto [func, forest] : analysis.getResult()) {
    (void) func;
    for (auto loop : forest.getLoops())
      runImpl(func, loop);
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
