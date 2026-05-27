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

Op *parityBit(Op *op) {
  if (!op)
    return nullptr;
  if (auto mod = dyn_cast<ModIOp>(op);
      mod && mod->getOperandCount() == 2 && isAbsConst(mod->DEF(1), 2))
    return mod->DEF(0);
  if (auto andi = dyn_cast<AndIOp>(op);
      andi && andi->getOperandCount() == 2 && isConst(andi->DEF(1), 1))
    return op;
  if (auto andi = dyn_cast<AndIOp>(op);
      andi && andi->getOperandCount() == 2 && isConst(andi->DEF(0), 1))
    return op;
  return nullptr;
}

Op *parityTruthValue(Builder &builder, Op *cond) {
  if (!cond)
    return cond;

  auto buildBit = [&](Op *value) {
    return builder.create<AndIOp>(std::vector<Value>{ value, intConst(builder, 1) });
  };

  if (auto bit = parityBit(cond))
    return bit == cond ? cond : buildBit(bit);

  auto normalizeCompare = [&](auto *cmp, bool isEq) -> Op* {
    if (!cmp || cmp->getOperandCount() != 2)
      return nullptr;
    auto lhs = cmp->DEF(0);
    auto rhs = cmp->DEF(1);
    auto makeCompare = [&](Op *bitSource, Op *constant) -> Op* {
      if (!isa<IntOp>(constant) || (V(constant) != 0 && V(constant) != 1))
        return nullptr;
      auto bit = bitSource == lhs || bitSource == rhs ? bitSource : buildBit(bitSource);
      if (V(constant) == 1)
        return isEq ? bit : builder.create<EqOp>(std::vector<Value>{ bit, intConst(builder, 0) });
      return isEq ? builder.create<EqOp>(std::vector<Value>{ bit, intConst(builder, 0) }) : bit;
    };
    if (auto bit = parityBit(lhs))
      if (auto normalized = makeCompare(bit == lhs ? lhs : bit, rhs))
        return normalized;
    if (auto bit = parityBit(rhs))
      if (auto normalized = makeCompare(bit == rhs ? rhs : bit, lhs))
        return normalized;
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
  if (parityBit(op))
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

bool pureConditionHoistable(Op *op) {
  return isa<IntOp>(op) || isa<AddIOp>(op) || isa<SubIOp>(op) ||
         isa<MulIOp>(op) || isa<DivIOp>(op) || isa<ModIOp>(op) ||
         isa<AndIOp>(op) || isa<OrIOp>(op) || isa<XorIOp>(op) ||
         isa<EqOp>(op) || isa<NeOp>(op) || isa<LtOp>(op) ||
         isa<LeOp>(op);
}

bool collectConditionDeps(Op *op, BasicBlock *bb, std::vector<Op*> &out,
                          std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return true;
  seen.insert(op);
  if (op->getParent() != bb)
    return true;
  if (!pureConditionHoistable(op))
    return false;
  for (auto operand : op->getOperands())
    if (!collectConditionDeps(operand.defining, bb, out, seen))
      return false;
  out.push_back(op);
  return true;
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

bool convertNestedShortCircuit(BranchOp *branch) {
  if (!branch || branch->getOperandCount() != 1 || !branch->has<TargetAttr>() ||
      !branch->has<ElseAttr>() || !isParityCondition(branch->DEF(0)))
    return false;

  auto outer = branch->getParent();
  auto trueBlock = TARGET(branch);
  auto falseBlock = ELSE(branch);

  BasicBlock *calcBlock = nullptr;
  BasicBlock *testBlock = nullptr;
  BasicBlock *emptyBlock = nullptr;
  bool isOrShape = false;
  Op *innerCond = nullptr;

  auto matchInner = [&](BasicBlock *test, BasicBlock *calc, BasicBlock *empty,
                        bool buildOr) -> bool {
    if (!test || !calc || !empty || !shortPureBlock(calc) || !emptyGotoBlock(empty))
      return false;
    if (test->preds.size() != 1 || *test->preds.begin() != outer)
      return false;

    auto inner = dyn_cast<BranchOp>(test->getLastOp());
    if (!inner || inner->getOperandCount() != 1 || !inner->has<TargetAttr>() ||
        !inner->has<ElseAttr>())
      return false;

    if (TARGET(inner) != calc || ELSE(inner) != empty)
      return false;
    if (!isParityCondition(inner->DEF(0)))
      return false;
    if (!calc->preds.count(test))
      return false;
    if (buildOr) {
      if (empty->preds.size() != 1 || *empty->preds.begin() != test)
        return false;
      if (calc->preds.size() != 2 || !calc->preds.count(outer))
        return false;
    } else {
      if (calc->preds.size() != 1)
        return false;
      if (empty->preds.size() != 2 || !empty->preds.count(outer) ||
          !empty->preds.count(test))
        return false;
    }

    innerCond = inner->DEF(0);
    return true;
  };

  // if (c1) calc; else if (c2) calc; else passthrough;
  if (trueBlock && falseBlock && trueBlock->preds.count(outer)) {
    if (auto inner = dyn_cast<BranchOp>(falseBlock->getLastOp())) {
      if (matchInner(falseBlock, trueBlock, ELSE(inner), true)) {
        calcBlock = trueBlock;
        testBlock = falseBlock;
        emptyBlock = ELSE(inner);
        isOrShape = true;
      }
    }
  }

  // if (c1) { if (c2) calc; else passthrough; } else passthrough;
  if (!calcBlock && trueBlock && falseBlock && emptyGotoBlock(falseBlock)) {
    if (auto inner = dyn_cast<BranchOp>(trueBlock->getLastOp())) {
      if (matchInner(trueBlock, TARGET(inner), falseBlock, false)) {
        testBlock = trueBlock;
        calcBlock = TARGET(inner);
        emptyBlock = falseBlock;
      }
    }
  }

  if (!calcBlock || !testBlock || !emptyBlock || !innerCond)
    return false;

  auto calcTerm = dyn_cast<GotoOp>(calcBlock->getLastOp());
  auto emptyTerm = dyn_cast<GotoOp>(emptyBlock->getLastOp());
  if (!calcTerm || !emptyTerm || TARGET(calcTerm) != TARGET(emptyTerm))
    return false;
  auto join = TARGET(calcTerm);
  if (!join || join->getPhis().size() != 1)
    return false;

  auto phi = join->getPhis()[0];
  if (phi->getOperandCount() != 2 || phi->getResultType() != Value::i32)
    return false;
  auto calcValue = incomingFrom(phi, calcBlock);
  auto passValue = incomingFrom(phi, emptyBlock);
  if (!calcValue || !passValue)
    return false;
  if (calcValue->getParent() != calcBlock ||
      !onlyLocalUsesOrPhi(calcValue, calcBlock, phi))
    return false;

  std::vector<Op*> moving;
  for (auto op : calcBlock->getOps()) {
    if (op == calcBlock->getLastOp())
      continue;
    if (!pureHoistable(op))
      return false;
    // Avoid speculating memory reads across a short-circuit branch. The generic
    // single-branch converter already handles simple cases; this nested form is
    // meant for arithmetic parity kernels.
    if (isa<LoadOp>(op))
      return false;
    moving.push_back(op);
  }
  if (moving.empty() || moving.size() > 12)
    return false;

  std::vector<Op*> condMoving;
  std::set<Op*> condSeen;
  if (!collectConditionDeps(innerCond, testBlock, condMoving, condSeen))
    return false;
  if (moving.size() + condMoving.size() > 20)
    return false;

  for (auto op : moving)
    op->moveBefore(branch);
  for (auto op : condMoving)
    op->moveBefore(branch);

  Builder builder;
  builder.setBeforeOp(branch);
  auto outerCond = parityTruthValue(builder, branch->DEF(0));
  innerCond = parityTruthValue(builder, innerCond);
  Op *combinedCond = nullptr;
  if (isOrShape)
    combinedCond = builder.create<OrIOp>(std::vector<Value>{ outerCond, innerCond });
  else
    combinedCond = builder.create<AndIOp>(std::vector<Value>{ outerCond, innerCond });
  auto select = builder.create<SelectOp>(
      std::vector<Value>{ combinedCond, calcValue, passValue });
  phi->replaceAllUsesWith(select);
  phi->erase();
  builder.replace<GotoOp>(branch, { new TargetAttr(join) });

  calcBlock->forceErase();
  testBlock->forceErase();
  emptyBlock->forceErase();
  return true;
}

bool sameAddressValue(Op *a, Op *b) {
  if (a == b)
    return true;
  if (!a || !b || a->opid != b->opid ||
      a->getOperandCount() != b->getOperandCount())
    return false;
  if (isa<IntOp>(a) || isa<FloatOp>(a))
    return false;
  for (int i = 0; i < a->getOperandCount(); i++)
    if (a->DEF(i) != b->DEF(i))
      return false;
  return true;
}

bool matchLoadModifyStore(StoreOp *store, Op *&oldValue, Op *&newValue) {
  if (!store || store->getOperandCount() < 2)
    return false;
  auto stored = store->DEF(0);
  auto addr = store->DEF(1);
  auto add = dyn_cast<AddIOp>(stored);
  if (!add || add->getOperandCount() != 2)
    return false;
  for (int i = 0; i < 2; i++) {
    auto load = dyn_cast<LoadOp>(add->DEF(i));
    if (!load || load->getOperandCount() != 1)
      continue;
    if (!sameAddressValue(load->DEF(0), addr))
      continue;
    oldValue = load;
    newValue = stored;
    return true;
  }
  return false;
}

bool convertParityStoreUpdate(BranchOp *branch) {
  if (!branch || branch->getOperandCount() != 1 || !branch->has<TargetAttr>() ||
      !branch->has<ElseAttr>() || !isParityCondition(branch->DEF(0)))
    return false;

  auto trueBlock = TARGET(branch);
  auto falseBlock = ELSE(branch);
  BasicBlock *calcBlock = nullptr;
  BasicBlock *emptyBlock = nullptr;
  if (shortPureBlock(trueBlock) && emptyGotoBlock(falseBlock)) {
    // shortPureBlock intentionally rejects stores; handle the store shape
    // below, so don't take this arm.
    return false;
  }

  auto blockHasSingleStoreUpdate = [&](BasicBlock *bb, StoreOp *&store,
                                       Op *&oldValue, Op *&newValue) -> bool {
    if (!bb || !isa<GotoOp>(bb->getLastOp()))
      return false;
    int stores = 0;
    for (auto op : bb->getOps()) {
      if (op == bb->getLastOp())
        continue;
      if (auto st = dyn_cast<StoreOp>(op)) {
        stores++;
        store = st;
        continue;
      }
      if (!pureHoistable(op))
        return false;
    }
    if (stores != 1)
      return false;
    return matchLoadModifyStore(store, oldValue, newValue);
  };

  StoreOp *store = nullptr;
  Op *oldValue = nullptr;
  Op *newValue = nullptr;
  bool calcOnTrue = false;
  if (blockHasSingleStoreUpdate(trueBlock, store, oldValue, newValue) &&
      emptyGotoBlock(falseBlock)) {
    calcBlock = trueBlock;
    emptyBlock = falseBlock;
    calcOnTrue = true;
  } else if (blockHasSingleStoreUpdate(falseBlock, store, oldValue, newValue) &&
             emptyGotoBlock(trueBlock)) {
    calcBlock = falseBlock;
    emptyBlock = trueBlock;
    calcOnTrue = false;
  } else {
    return false;
  }

  if (calcBlock->preds.size() != 1 || emptyBlock->preds.size() != 1 ||
      *calcBlock->preds.begin() != branch->getParent() ||
      *emptyBlock->preds.begin() != branch->getParent())
    return false;

  auto calcTerm = dyn_cast<GotoOp>(calcBlock->getLastOp());
  auto emptyTerm = dyn_cast<GotoOp>(emptyBlock->getLastOp());
  if (!calcTerm || !emptyTerm || TARGET(calcTerm) != TARGET(emptyTerm))
    return false;
  auto join = TARGET(calcTerm);
  if (!join || !join->getPhis().empty())
    return false;

  std::vector<Op*> moving;
  for (auto op : calcBlock->getOps()) {
    if (op == calcBlock->getLastOp() || op == store)
      continue;
    if (!pureHoistable(op))
      return false;
    moving.push_back(op);
  }
  if (moving.empty() || moving.size() > 16)
    return false;

  for (auto op : moving)
    op->moveBefore(branch);

  Builder builder;
  builder.setBeforeOp(branch);
  auto cond = parityTruthValue(builder, branch->DEF(0));
  auto selected = calcOnTrue
      ? builder.create<SelectOp>(std::vector<Value>{ cond, newValue, oldValue })
      : builder.create<SelectOp>(std::vector<Value>{ cond, oldValue, newValue });
  size_t storeSize = store->has<SizeAttr>() ? SIZE(store) : 4;
  builder.create<StoreOp>(std::vector<Value>{ selected, store->DEF(1) },
                          std::vector<Attr*>{ new SizeAttr(storeSize) });
  builder.replace<GotoOp>(branch, { new TargetAttr(join) });

  calcBlock->forceErase();
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
    if (convertParityStoreUpdate(cast<BranchOp>(op)) ||
        convertNestedShortCircuit(cast<BranchOp>(op)) ||
        convertOne(cast<BranchOp>(op)))
      converted++;
    else
      rejected++;
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
