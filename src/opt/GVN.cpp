#include "Passes.h"
#include "../rv/RvOps.h"
#include "../arm/ArmOps.h"

using namespace sys;

std::map<std::string, int> GVN::stats() {
  return {
    { "eliminated-ops", elim },
    { "pre-sunk", preSunk }
  };
}

bool GVN::Expr::operator<(const Expr &other) const {
  if (id != other.id)
    return id < other.id;
  if (operands.size() != other.operands.size())
    return operands.size() < other.operands.size();
  for (size_t i = 0; i < operands.size(); i++) {
    if (operands[i] != other.operands[i])
      return operands[i] < other.operands[i];
  }
  if (vi != other.vi)
    return vi < other.vi;
  if (vf != other.vf)
    return vf < other.vf;
  return name < other.name;
}

#define ALLOW(Ty) || isa<Ty>(op)
bool allowed(Op *op) {
  return
    (isa<CallOp>(op) && !op->has<ImpureAttr>())
    ALLOW(AddIOp)
    ALLOW(SubIOp)
    ALLOW(MulIOp)
    ALLOW(DivIOp)
    ALLOW(ModIOp)
    ALLOW(AddFOp)
    ALLOW(SubFOp)
    ALLOW(MulFOp)
    ALLOW(DivFOp)
    ALLOW(ModFOp)
    ALLOW(AddLOp)
    ALLOW(SubLOp)
    ALLOW(MulLOp)
    ALLOW(DivLOp)
    ALLOW(ModLOp)
    ALLOW(AndIOp)
    ALLOW(OrIOp)
    ALLOW(XorIOp)
    ALLOW(AddVOp)
    ALLOW(SubVOp)
    ALLOW(MulVOp)
    ALLOW(AddFVOp)
    ALLOW(SubFVOp)
    ALLOW(MulFVOp)
    ALLOW(EqOp)
    ALLOW(NeOp)
    ALLOW(LtOp)
    ALLOW(LeOp)
    ALLOW(EqFOp)
    ALLOW(NeFOp)
    ALLOW(LtFOp)
    ALLOW(LeFOp)
    ALLOW(IntOp)
    ALLOW(FloatOp)
    ALLOW(MinusOp)
    ALLOW(MinusFOp)
    ALLOW(F2IOp)
    ALLOW(I2FOp)
    ALLOW(NotOp)
    ALLOW(LShiftOp)
    ALLOW(RShiftOp)
    ALLOW(LShiftLOp)
    ALLOW(RShiftLOp)
    ALLOW(MulshOp)
    ALLOW(MuluhOp)
    ALLOW(SetNotZeroOp)
    ALLOW(SelectOp)
    ALLOW(GetGlobalOp)
    ALLOW(BroadcastOp)
    ALLOW(BroadcastFOp)

  // RISC-V GVN
    ALLOW(rv::AddOp)
    ALLOW(rv::AddwOp)
    ALLOW(rv::AndOp)
    ALLOW(rv::OrOp)
    ALLOW(rv::XorOp)
    ALLOW(rv::LiOp)

  // ARM GVN
    ALLOW(arm::MovIOp)
  ;
}

bool phiPreAllowed(Op *op) {
  return isa<AddIOp>(op) || isa<SubIOp>(op) || isa<MulIOp>(op) ||
         isa<DivIOp>(op) || isa<ModIOp>(op) || isa<AddLOp>(op) ||
         isa<SubLOp>(op) || isa<MulLOp>(op) || isa<DivLOp>(op) ||
         isa<ModLOp>(op) || isa<AndIOp>(op) || isa<OrIOp>(op) ||
         isa<XorIOp>(op) || isa<EqOp>(op) || isa<NeOp>(op) ||
         isa<LtOp>(op) || isa<LeOp>(op) || isa<MinusOp>(op) ||
         isa<NotOp>(op) || isa<LShiftOp>(op) || isa<RShiftOp>(op) ||
         isa<LShiftLOp>(op) || isa<RShiftLOp>(op) ||
         isa<SetNotZeroOp>(op) || isa<SelectOp>(op);
}

bool samePhiPreTemplate(Op *a, Op *b) {
  if (!a || !b || a->opid != b->opid ||
      a->getOperandCount() != b->getOperandCount())
    return false;
  auto ai = a->find<IntAttr>(), bi = b->find<IntAttr>();
  if (!!ai != !!bi || (ai && ai->value != bi->value))
    return false;
  auto af = a->find<FloatAttr>(), bf = b->find<FloatAttr>();
  if (!!af != !!bf || (af && af->value != bf->value))
    return false;
  auto an = a->find<NameAttr>(), bn = b->find<NameAttr>();
  if (!!an != !!bn || (an && an->name != bn->name))
    return false;
  return true;
}

void GVN::dvnt(BasicBlock *bb, Domtree &domtree) {
  SemanticScope scope(*this);

  auto phis = bb->getPhis();
  for (auto phi : phis) {
    // Empty phi is not allowed.
    assert(phi->getOperandCount() > 0);

    Value common = phi->getOperand(0);
    if (symbols.count(common.defining)) {
      // This phi is meaningless if all of its operands share the same value.
      bool meaningless = true;
      for (auto v : phi->getOperands()) {
        if (!(symbols.count(v.defining) && symbols[v.defining] == symbols[common.defining])) {
          meaningless = false;
          break;
        }
      }

      if (meaningless) {
        elim++;
        phi->replaceAllUsesWith(common.defining);
        phi->erase();
        continue;
      }
    }

    // This phi is a new value. Add it to symbols.
    symbols[phi] = num++;
  }

  auto ops = bb->getOps();
  for (auto op : ops) {
    // Processed beforehand.
    if (isa<PhiOp>(op))
      continue;

    if (!allowed(op)) {
      symbols[op] = num++;
      continue;
    }
    
    Expr key { .id = op->opid };
    bool missingDef = false;
    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      if (!symbols.count(def)) {
        missingDef = true;
        break;
      }
      key.operands.push_back(symbols[def]);
    }
    if (missingDef) {
      symbols[op] = num++;
      continue;
    }

    // Canonicalize for commutative Ops.
    if (isa<AddIOp>(op) || isa<AddFOp>(op) || isa<AddLOp>(op) ||
        isa<MulIOp>(op) || isa<MulFOp>(op) || isa<MulLOp>(op) ||
        isa<AndIOp>(op) || isa<OrIOp>(op) || isa<XorIOp>(op) ||
        isa<EqOp>(op) || isa<NeOp>(op) || isa<EqFOp>(op) ||
        isa<NeFOp>(op) || isa<rv::AddOp>(op) || isa<rv::AddwOp>(op) ||
        isa<rv::AndOp>(op) || isa<rv::OrOp>(op) || isa<rv::XorOp>(op))
      std::sort(key.operands.begin(), key.operands.end());

    if (auto attr = op->find<IntAttr>())
      key.vi = attr->value;
    if (auto attr = op->find<FloatAttr>())
      key.vf = attr->value;
    if (auto attr = op->find<NameAttr>())
      key.name = attr->name;

    if (exprNum.count(key)) {
      assert(numOp.count(exprNum[key]));
      elim++;
      op->replaceAllUsesWith(numOp[exprNum[key]]);
      op->erase();
    } else {
      symbols[op] = exprNum[key] = num;
      numOp[num] = op;
      num++;
    }
  }

  for (auto succ : domtree[bb])
    dvnt(succ, domtree);
}

bool GVN::sinkPhiExpressions(Region *region) {
  bool changed = false;

  for (auto bb : region->getBlocks()) {
    auto phis = bb->getPhis();
    if (phis.empty())
      continue;

    Op *insertBefore = nullptr;
    for (auto op : bb->getOps()) {
      if (!isa<PhiOp>(op)) {
        insertBefore = op;
        break;
      }
    }
    if (!insertBefore)
      continue;

    for (auto phi : phis) {
      if (phi->getParent() != bb || phi->getOperandCount() < 2)
        continue;
      if (phi->getUses().empty())
        continue;

      std::vector<Op*> incomingExprs;
      bool ok = true;
      for (auto operand : phi->getOperands()) {
        auto expr = operand.defining;
        if (!phiPreAllowed(expr)) {
          ok = false;
          break;
        }
        if (!incomingExprs.empty() && !samePhiPreTemplate(incomingExprs.front(), expr)) {
          ok = false;
          break;
        }
        incomingExprs.push_back(expr);
      }
      if (!ok || incomingExprs.empty())
        continue;

      auto arity = incomingExprs.front()->getOperandCount();
      std::vector<Value> newOperands;
      std::vector<Op*> createdPhis;
      for (int operandIndex = 0; operandIndex < arity && ok; operandIndex++) {
        std::vector<Value> values;
        std::vector<Attr*> attrs;
        Op *common = nullptr;
        bool allSame = true;

        for (int i = 0; i < phi->getOperandCount(); i++) {
          auto from = dyn_cast<FromAttr>(phi->getAttrs()[i]);
          if (!from) {
            ok = false;
            break;
          }
          auto value = incomingExprs[i]->getOperand(operandIndex);
          if (value.defining == phi) {
            ok = false;
            break;
          }
          values.push_back(value);
          attrs.push_back(new FromAttr(from->bb));
          if (!common)
            common = value.defining;
          else if (common != value.defining)
            allSame = false;
        }
        if (!ok) {
          for (auto attr : attrs)
            delete attr;
          break;
        }

        if (allSame) {
          for (auto attr : attrs)
            delete attr;
          newOperands.push_back(common);
          continue;
        }

        Op *existing = nullptr;
        for (auto candidate : bb->getPhis()) {
          if (candidate == phi || candidate->getOperandCount() != (int)values.size())
            continue;
          bool same = true;
          for (int i = 0; i < candidate->getOperandCount(); i++) {
            auto from = dyn_cast<FromAttr>(candidate->getAttrs()[i]);
            if (!from || FROM(attrs[i]) != from->bb ||
                candidate->getOperand(i).defining != values[i].defining) {
              same = false;
              break;
            }
          }
          if (same) {
            existing = candidate;
            break;
          }
        }

        if (existing) {
          for (auto attr : attrs)
            delete attr;
          newOperands.push_back(existing);
          continue;
        }

        Builder phiBuilder;
        phiBuilder.setToBlockStart(bb);
        auto argPhi = phiBuilder.create<PhiOp>(values, attrs);
        createdPhis.push_back(argPhi);
        newOperands.push_back(argPhi);
      }
      if (!ok || (int)newOperands.size() != arity) {
        for (auto createdPhi : createdPhis)
          createdPhi->erase();
        continue;
      }

      Builder builder;
      builder.setBeforeOp(insertBefore);
      auto sunk = builder.copy(incomingExprs.front());
      sunk->removeAllOperands();
      for (auto operand : newOperands)
        sunk->pushOperand(operand);

      phi->replaceAllUsesWith(sunk);
      phi->erase();
      preSunk++;
      changed = true;
    }
  }

  if (changed)
    region->updatePreds();
  return changed;
}

// See https://www.cs.tufts.edu/~nr/cs257/archive/keith-cooper/value-numbering.pdf,
// "Value Numbering", Briggs, 1997
// Refer to figure 4.
void GVN::runImpl(Region *region) {
  Domtree domtree;
  for (auto &[bb, children] : getDomTree(region))
    domtree[bb] = children;
  dvnt(region->getFirstBlock(), domtree);
  sinkPhiExpressions(region);
}

void GVN::run() {
  auto funcs = collectFuncs();
  for (auto func : funcs)
    runImpl(func->getRegion());

  // Tidy up remaining phis after gvn.
  runRewriter([&](PhiOp *op) {
    // Discard trivial phis.
    if (op->getOperands().size() == 1) {
      auto def = op->getOperand().defining;
      op->replaceAllUsesWith(def);
      op->erase();
      return true;
    }
    return false;
  });
}

PreservedAnalyses GVN::run(PassContext &ctx) {
  activeContext = &ctx;
  run();
  activeContext = nullptr;
  return PreservedAnalyses::cfg();
}
