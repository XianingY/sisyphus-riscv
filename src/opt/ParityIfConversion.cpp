#include "Passes.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

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

bool isAbsConst(Op *op, int value) {
  return op && isa<IntOp>(op) && (V(op) == value || V(op) == -value);
}

Op *intConst(Builder &builder, int value) {
  return builder.create<IntOp>({ new IntAttr(value) });
}

Op *parityTruthValue(Builder &builder, Op *cond) {
  if (!cond)
    return cond;

  auto buildBit = [&](Op *value) {
    return builder.create<AndIOp>(std::vector<Value>{ value, intConst(builder, 1) });
  };

  if (auto mod = dyn_cast<ModIOp>(cond);
      mod && mod->getOperandCount() == 2 && isAbsConst(mod->DEF(1), 2))
    return buildBit(mod->DEF(0));

  auto normalizeCompare = [&](auto *cmp, bool isEq) -> Op* {
    if (!cmp || cmp->getOperandCount() != 2)
      return nullptr;
    auto lhs = cmp->DEF(0);
    auto rhs = cmp->DEF(1);
    if (auto mod = dyn_cast<ModIOp>(lhs);
        mod && mod->getOperandCount() == 2 && isAbsConst(mod->DEF(1), 2) &&
        isConst(rhs, 0)) {
      auto bit = buildBit(mod->DEF(0));
      if (isEq)
        return builder.create<EqOp>(std::vector<Value>{ bit, intConst(builder, 0) });
      return builder.create<NeOp>(std::vector<Value>{ bit, intConst(builder, 0) });
    }
    if (auto mod = dyn_cast<ModIOp>(rhs);
        mod && mod->getOperandCount() == 2 && isAbsConst(mod->DEF(1), 2) &&
        isConst(lhs, 0)) {
      auto bit = buildBit(mod->DEF(0));
      if (isEq)
        return builder.create<EqOp>(std::vector<Value>{ intConst(builder, 0), bit });
      return builder.create<NeOp>(std::vector<Value>{ intConst(builder, 0), bit });
    }
    return nullptr;
  };

  if (auto eq = dyn_cast<EqOp>(cond))
    if (auto folded = normalizeCompare(eq, true))
      return folded;
  if (auto ne = dyn_cast<NeOp>(cond))
    if (auto folded = normalizeCompare(ne, false))
      return folded;

  return cond;
}

bool containsParityMod(Op *op, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (isa<ModIOp>(op) && op->getOperandCount() == 2 && isAbsConst(op->DEF(1), 2))
    return true;
  for (auto operand : op->getOperands())
    if (containsParityMod(operand.defining, seen))
      return true;
  return false;
}

bool isParityCondition(Op *cond) {
  std::set<Op*> seen;
  return containsParityMod(cond, seen);
}

bool pureHoistable(Op *op) {
  return isa<IntOp>(op) || isa<GetGlobalOp>(op) || isa<AddIOp>(op) ||
         isa<SubIOp>(op) || isa<MulIOp>(op) || isa<AddLOp>(op) ||
         isa<SubLOp>(op) || isa<AndIOp>(op) || isa<OrIOp>(op) ||
         isa<XorIOp>(op) || isa<LoadOp>(op);
}

bool shortPureBlock(BasicBlock *bb) {
  if (!bb || !isa<GotoOp>(bb->getLastOp()))
    return false;
  int count = 0;
  for (auto op : bb->getOps()) {
    if (op == bb->getLastOp())
      continue;
    if (!pureHoistable(op))
      return false;
    if (++count > 12)
      return false;
  }
  return count > 0;
}

bool emptyGotoBlock(BasicBlock *bb) {
  return bb && bb->getOpCount() == 1 && isa<GotoOp>(bb->getLastOp());
}

Op *incomingFrom(Op *phi, BasicBlock *bb) {
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  if (ops.size() != attrs.size())
    return nullptr;
  for (int i = 0; i < ops.size(); i++) {
    auto from = dyn_cast<FromAttr>(attrs[i]);
    if (from && from->bb == bb)
      return ops[i].defining;
  }
  return nullptr;
}

bool onlyLocalUsesOrPhi(Op *op, BasicBlock *bb, Op *phi) {
  for (auto use : op->getUses())
    if (use != phi && use->getParent() != bb)
      return false;
  return true;
}

bool convertOne(BranchOp *branch) {
  if (!branch || branch->getOperandCount() != 1 || !branch->has<TargetAttr>() ||
      !branch->has<ElseAttr>() || !isParityCondition(branch->DEF(0)))
    return false;

  auto trueBlock = TARGET(branch);
  auto falseBlock = ELSE(branch);
  BasicBlock *calcBlock = nullptr;
  BasicBlock *emptyBlock = nullptr;
  BasicBlock *join = nullptr;
  BasicBlock *directFrom = nullptr;
  bool eraseEmpty = true;
  if (shortPureBlock(trueBlock) && emptyGotoBlock(falseBlock)) {
    calcBlock = trueBlock;
    emptyBlock = falseBlock;
  } else if (shortPureBlock(falseBlock) && emptyGotoBlock(trueBlock)) {
    calcBlock = falseBlock;
    emptyBlock = trueBlock;
  } else if (shortPureBlock(trueBlock)) {
    auto calcTerm = dyn_cast<GotoOp>(trueBlock->getLastOp());
    if (!calcTerm || TARGET(calcTerm) != falseBlock)
      return false;
    calcBlock = trueBlock;
    join = falseBlock;
    directFrom = branch->getParent();
    eraseEmpty = false;
  } else if (shortPureBlock(falseBlock)) {
    auto calcTerm = dyn_cast<GotoOp>(falseBlock->getLastOp());
    if (!calcTerm || TARGET(calcTerm) != trueBlock)
      return false;
    calcBlock = falseBlock;
    join = trueBlock;
    directFrom = branch->getParent();
    eraseEmpty = false;
  } else {
    return false;
  }

  if (eraseEmpty) {
    if (calcBlock->preds.size() != 1 || emptyBlock->preds.size() != 1 ||
        *calcBlock->preds.begin() != branch->getParent() ||
        *emptyBlock->preds.begin() != branch->getParent())
      return false;

    auto calcTerm = dyn_cast<GotoOp>(calcBlock->getLastOp());
    auto emptyTerm = dyn_cast<GotoOp>(emptyBlock->getLastOp());
    if (!calcTerm || !emptyTerm || TARGET(calcTerm) != TARGET(emptyTerm))
      return false;
    join = TARGET(calcTerm);
  } else {
    if (calcBlock->preds.size() != 1 || *calcBlock->preds.begin() != branch->getParent())
      return false;
  }

  std::vector<Op*> phis;
  for (auto phi : join->getPhis()) {
    if (phi->getOperandCount() != 2 || phi->getResultType() != Value::i32)
      continue;
    auto fromCalc = incomingFrom(phi, calcBlock);
    auto fromEmpty = incomingFrom(phi, eraseEmpty ? emptyBlock : directFrom);
    if (!fromCalc || !fromEmpty)
      continue;
    if (fromCalc->getParent() == calcBlock && onlyLocalUsesOrPhi(fromCalc, calcBlock, phi))
      phis.push_back(phi);
  }
  if (phis.size() != 1)
    return false;

  auto phi = phis[0];
  Builder builder;
  auto calcValue = incomingFrom(phi, calcBlock);
  auto trueValue = (trueBlock == join && !eraseEmpty) ? incomingFrom(phi, directFrom) : incomingFrom(phi, trueBlock);
  auto falseValue = (falseBlock == join && !eraseEmpty) ? incomingFrom(phi, directFrom) : incomingFrom(phi, falseBlock);
  if (!calcValue || !trueValue || !falseValue)
    return false;

  std::vector<Op*> moving;
  for (auto op : calcBlock->getOps()) {
    if (op == calcBlock->getLastOp())
      continue;
    moving.push_back(op);
  }
  for (auto op : moving)
    op->moveBefore(branch);

  builder.setBeforeOp(branch);
  auto cond = parityTruthValue(builder, branch->DEF(0));
  auto select = builder.create<SelectOp>(std::vector<Value>{ cond, trueValue, falseValue });
  phi->replaceAllUsesWith(select);
  phi->erase();
  builder.replace<GotoOp>(branch, { new TargetAttr(join) });
  calcBlock->forceErase();
  if (eraseEmpty)
    emptyBlock->forceErase();
  return true;
}

} // namespace

std::map<std::string, int> ParityIfConversion::stats() {
  return {
    { "converted", converted },
    { "rejected", rejected },
  };
}

void ParityIfConversion::run() {
  if (!envEnabled("SISY_ENABLE_PARITY_IF_CONVERSION", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  auto branches = module->findAll<BranchOp>();
  for (auto op : branches) {
    if (convertOne(cast<BranchOp>(op)))
      converted++;
    else
      rejected++;
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
