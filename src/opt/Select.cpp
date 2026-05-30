#include "Passes.h"

#include <set>
#include <vector>

using namespace sys;

namespace {

#define ALLOWED(Ty) || isa<Ty>(op)
bool hoistable(Op *op) {
  return isa<AddIOp>(op)
    ALLOWED(IntOp)
    ALLOWED(AddLOp)
    ALLOWED(SubIOp)
    ALLOWED(OrIOp)
    ALLOWED(AndIOp)
    ALLOWED(XorIOp)
    ALLOWED(EqOp)
    ALLOWED(LeOp)
    ALLOWED(LtOp)
    ALLOWED(NeOp)
  ;
}

bool identical(Op *a, Op *b) {
  if (a->opid != b->opid || a->getOperandCount() != b->getOperandCount())
    return false;

  // Avoid infinite loops.
  if (isa<PhiOp>(a))
    return false;

  if (isa<IntOp>(a))
    return V(a) == V(b);
  
  for (int i = 0; i < a->getOperandCount(); i++) {
    if (!identical(a->DEF(i), b->DEF(i)))
      return false;
  }
  return true;
}

bool onlyPhiInMerge(Op *phi) {
  if (!phi || !isa<PhiOp>(phi))
    return false;
  auto phis = phi->getParent()->getPhis();
  return phis.size() == 1 && phis[0] == phi;
}

bool emptyGotoBlock(BasicBlock *bb) {
  return bb && bb->getOpCount() == 1 && isa<GotoOp>(bb->getLastOp());
}

bool pureSpeculatable(Op *op) {
  return isa<IntOp>(op) || isa<AddIOp>(op) || isa<SubIOp>(op) ||
         isa<MulIOp>(op) || isa<AddLOp>(op) || isa<SubLOp>(op) ||
         isa<AndIOp>(op) || isa<OrIOp>(op) || isa<XorIOp>(op) ||
         isa<LShiftOp>(op) || isa<RShiftOp>(op) || isa<EqOp>(op) ||
         isa<NeOp>(op) || isa<LtOp>(op) || isa<LeOp>(op) ||
         isa<SetNotZeroOp>(op) || isa<NotOp>(op);
}

bool shortPureBlock(BasicBlock *bb) {
  if (!bb || !isa<GotoOp>(bb->getLastOp()))
    return false;
  int count = 0;
  for (auto op : bb->getOps()) {
    if (op == bb->getLastOp())
      continue;
    if (!pureSpeculatable(op))
      return false;
    if (++count > 10)
      return false;
  }
  return count > 0;
}

Op *incomingFrom(Op *phi, BasicBlock *bb) {
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  if (ops.size() != attrs.size())
    return nullptr;
  for (size_t i = 0; i < ops.size(); i++) {
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

int convertShortPureBranch(BranchOp *branch) {
  if (!branch || branch->getOperandCount() != 1 || !branch->has<TargetAttr>() ||
      !branch->has<ElseAttr>())
    return 0;

  auto trueBlock = TARGET(branch);
  auto falseBlock = ELSE(branch);
  BasicBlock *calcBlock = nullptr;
  BasicBlock *emptyBlock = nullptr;
  if (shortPureBlock(trueBlock) && emptyGotoBlock(falseBlock)) {
    calcBlock = trueBlock;
    emptyBlock = falseBlock;
  } else if (shortPureBlock(falseBlock) && emptyGotoBlock(trueBlock)) {
    calcBlock = falseBlock;
    emptyBlock = trueBlock;
  } else {
    return 0;
  }

  if (calcBlock->preds.size() != 1 || emptyBlock->preds.size() != 1 ||
      *calcBlock->preds.begin() != branch->getParent() ||
      *emptyBlock->preds.begin() != branch->getParent())
    return 0;

  auto calcTerm = dyn_cast<GotoOp>(calcBlock->getLastOp());
  auto emptyTerm = dyn_cast<GotoOp>(emptyBlock->getLastOp());
  if (!calcTerm || !emptyTerm || TARGET(calcTerm) != TARGET(emptyTerm))
    return 0;
  auto join = TARGET(calcTerm);
  if (!join || join->getPhis().empty())
    return 0;

  std::vector<Op*> phis;
  for (auto phi : join->getPhis()) {
    if (phi->getOperandCount() != 2 || phi->getResultType() != Value::i32)
      continue;
    auto fromCalc = incomingFrom(phi, calcBlock);
    auto fromEmpty = incomingFrom(phi, emptyBlock);
    if (!fromCalc || !fromEmpty)
      continue;
    if (fromCalc->getParent() == calcBlock && onlyLocalUsesOrPhi(fromCalc, calcBlock, phi))
      phis.push_back(phi);
  }
  if (phis.empty() || phis.size() != join->getPhis().size())
    return 0;
  for (auto phi : phis) {
    if (!incomingFrom(phi, trueBlock) || !incomingFrom(phi, falseBlock))
      return 0;
  }

  std::vector<Op*> moving;
  for (auto op : calcBlock->getOps()) {
    if (op == calcBlock->getLastOp())
      continue;
    if (!pureSpeculatable(op))
      return 0;
    moving.push_back(op);
  }

  for (auto op : moving)
    op->moveBefore(branch);

  Builder builder;
  builder.setBeforeOp(branch);
  int converted = 0;
  for (auto phi : phis) {
    auto trueValue = incomingFrom(phi, trueBlock);
    auto falseValue = incomingFrom(phi, falseBlock);
    auto select = builder.create<SelectOp>(
        std::vector<Value>{ branch->DEF(0), trueValue, falseValue });
    phi->replaceAllUsesWith(select);
    phi->erase();
    converted++;
  }
  builder.replace<GotoOp>(branch, { new TargetAttr(join) });
  calcBlock->forceErase();
  emptyBlock->forceErase();
  return converted;
}

}

std::map<std::string, int> Select::stats() {
  return {
    { "raised-selects", raised }
  };
}

void Select::run() {
  Builder builder;
  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  auto branches = module->findAll<BranchOp>();
  for (auto op : branches) {
    raised += convertShortPureBranch(cast<BranchOp>(op));
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  auto phis = module->findAll<PhiOp>();

  // Hoist identical ops out of an if.
  for (auto phi : phis) {
    if (phi->getOperandCount() != 2)
      continue;

    const auto &attrs = phi->getAttrs();
    auto bb1 = FROM(attrs[0]), bb2 = FROM(attrs[1]);

    // Check if their unique predecessors are the same.
    if (bb1->preds.size() != 1 || bb2->preds.size() != 1)
      continue;
    auto pred1 = *bb1->preds.begin(), pred2 = *bb2->preds.begin();
    if (pred1 != pred2)
      continue;

    auto pred = pred1;
    auto term = pred->getLastOp();
    
    // Now check if both bb1 and bb2 have identical initial op sequence.
    auto r1 = bb1->getFirstOp(), r2 = bb2->getFirstOp();
    while (hoistable(r1) && identical(r1, r2)) {
      auto x1 = r1->nextOp(), x2 = r2->nextOp();
      
      r1->moveBefore(term);
      r2->replaceAllUsesWith(r1);
      r2->erase();

      r1 = x1, r2 = x2;
    }
  }

  for (auto phi : phis) {
    // If the phi has two empty blocks as predecessors, then it is a `select`.
    if (phi->getOperandCount() != 2)
      continue;
    if (!onlyPhiInMerge(phi))
      continue;

    const auto &attrs = phi->getAttrs();
    auto bb1 = FROM(attrs[0]), bb2 = FROM(attrs[1]);
    if (bb1->getOpCount() != 1 || bb2->getOpCount() != 1)
      continue;

    // TODO: check for a single block that have a single non-terminator.

    // Check if their unique predecessors are the same.
    if (bb1->preds.size() != 1 || bb2->preds.size() != 1)
      continue;
    auto pred1 = *bb1->preds.begin(), pred2 = *bb2->preds.begin();
    if (pred1 != pred2)
      continue;

    auto pred = pred1;
    // We know the last op of `pred` is a branch op,
    // which is the condition of the select.

    auto term = pred->getLastOp();
    auto cond = term->getOperand(0);

    // Replace the phi with a select.
    if (phi->DEF(0)->getResultType() != Value::i32)
      continue;

    auto _true = TARGET(term), _false = ELSE(term);
    auto select = builder.replace<SelectOp>(phi, { cond, Op::getPhiFrom(phi, _true), Op::getPhiFrom(phi, _false) });
    
    // Move it below all the phis in the same block.
    auto parent = select->getParent();
    auto insert = nonphi(parent);
    select->moveBefore(insert);

    // Replace the branch of `pred` to connect to `parent` instead.
    builder.replace<GotoOp>(term, { new TargetAttr(parent) });

    // The empty blocks `bb1` and `bb2` are dead. Erase them.
    bb1->forceErase();
    bb2->forceErase();
  }

  // Moreover, raise the following to select:
  //   br %1 <bb1>, <bb2>
  // bb1:
  //   %2 = <relocatable>
  //   goto <bb3>
  // bb2:
  //   goto <bb3>
  // bb3:
  //   phi = [%2, bb1], [%0, bb2]
  //
  // It should become
  //   select %1 %2 %0
  phis = module->findAll<PhiOp>();

  for (auto phi : phis) {
    if (phi->getOperandCount() != 2)
      continue;
    if (!onlyPhiInMerge(phi))
      continue;

    const auto &attrs = phi->getAttrs();
    auto bb1 = FROM(attrs[0]), bb2 = FROM(attrs[1]);
    if (bb1->getOpCount() == 1 && bb2->getOpCount() != 1)
      std::swap(bb1, bb2);
    if (bb2->getOpCount() != 1 || bb1->getOpCount() != 2)
      continue;

    // The `%2` above is the first op in `bb1`.
    auto v2 = bb1->getFirstOp();
    // It has to be relocatable, and has to be "worth it",
    // because we're pulling it out of an if.
    // We use a whitelist.
    if (!hoistable(v2))
      continue;

    // Check if their unique predecessors are the same.
    if (bb1->preds.size() != 1 || bb2->preds.size() != 1)
      continue;
    auto pred1 = *bb1->preds.begin(), pred2 = *bb2->preds.begin();
    if (pred1 != pred2)
      continue;

    auto pred = pred1;
    
    // The rest is quite similar.
    auto term = pred->getLastOp();
    auto cond = term->getOperand(0);

    // Replace the phi with a select.
    if (phi->DEF(0)->getResultType() != Value::i32)
      continue;

    v2->moveBefore(pred->getLastOp());

    auto _true = TARGET(term), _false = ELSE(term);
    auto select = builder.replace<SelectOp>(phi, { cond, Op::getPhiFrom(phi, _true), Op::getPhiFrom(phi, _false) });
    
    // Move it below all the phis in the same block.
    auto parent = select->getParent();
    auto insert = nonphi(parent);
    select->moveBefore(insert);

    // Replace the branch of `pred` to connect to `parent` instead.
    builder.replace<GotoOp>(term, { new TargetAttr(parent) });

    // The empty blocks `bb1` and `bb2` are dead. Erase them.
    bb1->forceErase();
    bb2->forceErase();
  }
}
