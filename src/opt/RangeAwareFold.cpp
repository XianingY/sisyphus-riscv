#include "CleanupPasses.h"
#include "Analysis.h"
#include "../utils/Matcher.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <set>
#include <unordered_set>
#include <vector>

using namespace sys;

std::map<std::string, int> RangeAwareFold::stats() {
  return {
    { "folded-ops", folded },
    { "path-replacements", pathReplacements },
    { "threaded-edges", threadedEdges },
  };
}

// Defined in Specialize.cpp.
void removeRange(Region *region);

namespace {

std::optional<std::pair<int, int>> intRange(Op *op);

Op *phiIncoming(Op *phi, BasicBlock *pred) {
  if (!phi || !pred || !isa<PhiOp>(phi))
    return nullptr;
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  size_t count = std::min(ops.size(), attrs.size());
  for (size_t i = 0; i < count; i++) {
    if (!isa<FromAttr>(attrs[i]))
      continue;
    if (FROM(attrs[i]) == pred)
      return ops[i].defining;
  }
  return nullptr;
}

void removePhiIncoming(Op *phi, BasicBlock *pred) {
  auto ops = phi->getOperands();
  std::vector<Attr*> attrs;
  for (auto attr : phi->getAttrs())
    attrs.push_back(attr->clone());

  phi->removeAllOperands();
  phi->removeAllAttributes();

  size_t count = std::min(ops.size(), attrs.size());
  for (size_t i = 0; i < count; i++) {
    if (!isa<FromAttr>(attrs[i]))
      continue;
    auto from = FROM(attrs[i]);
    if (from == pred)
      continue;
    phi->pushOperand(ops[i]);
    phi->add<FromAttr>(from);
  }

  for (auto attr : attrs)
    delete attr;
}

bool isThreadLocalPure(Op *op) {
  return isa<IntOp>(op) || isa<AddIOp>(op) || isa<SubIOp>(op) ||
         isa<MulIOp>(op) || isa<DivIOp>(op) || isa<ModIOp>(op) ||
         isa<AndIOp>(op) || isa<OrIOp>(op) || isa<XorIOp>(op) ||
         isa<EqOp>(op) || isa<NeOp>(op) || isa<LtOp>(op) ||
         isa<LeOp>(op) || isa<NotOp>(op) || isa<SetNotZeroOp>(op);
}

bool decisionBlockCanBeSkipped(BasicBlock *bb) {
  if (!bb->getPhis().empty())
    return false;

  auto term = bb->getLastOp();
  for (auto op : bb->getOps()) {
    if (isa<PhiOp>(op) || op == term)
      continue;
    if (!isThreadLocalPure(op))
      return false;
    for (auto use : op->getUses()) {
      if (use->getParent() != bb)
        return false;
    }
  }
  return true;
}

Op *edgeValue(Op *op, BasicBlock *pred, BasicBlock *through) {
  std::unordered_set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getParent() == through && !seen.count(op)) {
    seen.insert(op);
    auto incoming = phiIncoming(op, pred);
    if (!incoming)
      return nullptr;
    op = incoming;
  }
  return op;
}

Op *refinedValueOnPred(Op *op, BasicBlock *pred, BasicBlock *through) {
  op = edgeValue(op, pred, through);
  if (!op || !pred || !op->has<EqClassAttr>())
    return op;

  auto originalRange = intRange(op);
  int originalWidth = originalRange ? originalRange->second - originalRange->first : INT_MAX;
  Op *best = op;
  for (auto candidate : pred->getOps()) {
    if (candidate == op || candidate->getResultType() == Value::f32 ||
        !candidate->has<EqClassAttr>() || EQCLASS(candidate) != EQCLASS(op))
      continue;
    auto range = intRange(candidate);
    if (!range)
      continue;
    int width = range->second - range->first;
    if (width <= originalWidth) {
      best = candidate;
      originalWidth = width;
    }
  }
  return best;
}

std::optional<std::pair<int, int>> intRange(Op *op) {
  if (!op || op->getResultType() == Value::f32)
    return std::nullopt;
  if (isa<IntOp>(op))
    return std::pair<int, int>{ V(op), V(op) };
  if (!op->has<RangeAttr>())
    return std::nullopt;
  auto range = RANGE(op);
  if (range.first > range.second)
    return std::nullopt;
  return range;
}

std::optional<int> truthFromRange(Op *op) {
  auto range = intRange(op);
  if (!range)
    return std::nullopt;
  auto [low, high] = *range;
  if (low == 0 && high == 0)
    return 0;
  if (high < 0 || low > 0)
    return 1;
  return std::nullopt;
}

std::optional<int> evaluateComparisonOnEdge(Op *cond, BasicBlock *pred, BasicBlock *through) {
  if (!cond || cond->getOperandCount() != 2)
    return std::nullopt;

  auto lhs = refinedValueOnPred(cond->DEF(0), pred, through);
  auto rhs = refinedValueOnPred(cond->DEF(1), pred, through);
  if (!lhs || !rhs)
    return std::nullopt;

  if (lhs == rhs) {
    if (isa<EqOp>(cond) || isa<LeOp>(cond))
      return 1;
    if (isa<NeOp>(cond) || isa<LtOp>(cond))
      return 0;
  }

  auto lr = intRange(lhs);
  auto rr = intRange(rhs);
  if (!lr || !rr)
    return std::nullopt;

  auto [ll, lh] = *lr;
  auto [rl, rh] = *rr;
  if (isa<LtOp>(cond)) {
    if (lh < rl)
      return 1;
    if (ll >= rh)
      return 0;
  } else if (isa<LeOp>(cond)) {
    if (lh <= rl)
      return 1;
    if (ll > rh)
      return 0;
  } else if (isa<EqOp>(cond)) {
    if (ll == lh && rl == rh && ll == rl)
      return 1;
    if (lh < rl || rh < ll)
      return 0;
  } else if (isa<NeOp>(cond)) {
    if (ll == lh && rl == rh && ll == rl)
      return 0;
    if (lh < rl || rh < ll)
      return 1;
  }
  return std::nullopt;
}

std::optional<int> branchTruthOnEdge(Op *cond, BasicBlock *pred, BasicBlock *through) {
  auto resolved = edgeValue(cond, pred, through);
  if (!resolved)
    return std::nullopt;
  if (auto truth = truthFromRange(resolved))
    return truth;
  if (resolved->getParent() != through)
    return std::nullopt;
  if (isa<EqOp>(resolved) || isa<NeOp>(resolved) ||
      isa<LtOp>(resolved) || isa<LeOp>(resolved))
    return evaluateComparisonOnEdge(resolved, pred, through);
  if (isa<NotOp>(resolved) && resolved->getOperandCount() == 1) {
    auto inner = branchTruthOnEdge(resolved->DEF(), pred, through);
    if (inner)
      return *inner ? 0 : 1;
  }
  return std::nullopt;
}

bool availableBeforeEdge(Op *op, BasicBlock *pred, BasicBlock *through) {
  op = edgeValue(op, pred, through);
  if (!op)
    return false;
  return op->getParent() != through;
}

}

void RangeAwareFold::run() {
  Builder builder;
  auto sameEqClass = [](Op *a, Op *b) -> bool {
    return a && b && a->has<EqClassAttr>() && b->has<EqClassAttr>() && EQCLASS(a) == EQCLASS(b);
  };
  auto sameValueOrEqClass = [&](Op *a, Op *b) -> bool {
    return a && b && (a == b || sameEqClass(a, b));
  };

  auto singletonRange = [](Op *op) -> std::optional<int> {
    auto range = intRange(op);
    if (!range)
      return std::nullopt;
    auto [low, high] = *range;
    if (low != high)
      return std::nullopt;
    return low;
  };

  auto setAfterPhis = [&](BasicBlock *bb) {
    auto phis = bb->getPhis();
    if (phis.empty())
      builder.setToBlockStart(bb);
    else
      builder.setAfterOp(phis.back());
  };

  auto replacePhiUsesWith = [&](PhiOp *phi, Op *replacement) -> bool {
    if (!phi || !replacement || phi == replacement)
      return false;
    if (!replacement->getParent()->dominates(phi->getParent()))
      return false;
    phi->replaceAllUsesWith(replacement);
    phi->erase();
    pathReplacements++;
    return true;
  };

  auto branchEqualityReplacement = [&](PhiOp *phi) -> Op* {
    if (!phi || phi->getOperandCount() != 1)
      return nullptr;

    FromAttr *from = nullptr;
    for (auto attr : phi->getAttrs()) {
      if (!isa<FromAttr>(attr))
        continue;
      if (from)
        return nullptr;
      from = cast<FromAttr>(attr);
    }
    if (!from)
      return nullptr;

    auto pred = from->bb;
    if (!pred || !pred->getLastOp() || !isa<BranchOp>(pred->getLastOp()))
      return nullptr;

    auto branch = cast<BranchOp>(pred->getLastOp());
    bool onTrueEdge = TARGET(branch) == phi->getParent();
    bool onFalseEdge = ELSE(branch) == phi->getParent();
    if (!onTrueEdge && !onFalseEdge)
      return nullptr;

    auto cond = branch->DEF();
    bool equalityHolds = (onTrueEdge && isa<EqOp>(cond)) || (onFalseEdge && isa<NeOp>(cond));
    if (!equalityHolds)
      return nullptr;

    auto lhs = cond->DEF(0);
    auto rhs = cond->DEF(1);
    auto original = phi->DEF();

    // Canonicalize non-constant x == y paths by substituting the LHS split
    // value with RHS.  Replacing both split operands would just swap them and
    // hide x-x/y-y folds from later rewriters.
    if (original == lhs)
      return rhs;

    // Constants are safe and profitable whichever side they appear on.
    if (original == rhs && isa<IntOp>(lhs))
      return lhs;

    return nullptr;
  };

  auto funcs = collectFuncs();
  for (auto func : funcs)
    func->getRegion()->updateDoms();

  auto phis = module->findAll<PhiOp>();
  for (auto op : phis) {
    auto phi = cast<PhiOp>(op);
    if (auto replacement = branchEqualityReplacement(phi)) {
      replacePhiUsesWith(phi, replacement);
      continue;
    }

    // SCCP-like constant propagation for edge phis narrowed to a singleton by
    // path-sensitive range splitting.  Restrict this to single-predecessor
    // blocks so the materialized constant dominates all non-phi uses without
    // pretending to be available on unrelated incoming edges.
    auto cst = singletonRange(phi);
    if (!cst || phi->getOperandCount() != 1 || phi->getParent()->preds.size() != 1)
      continue;

    bool usedByPhiInSameBlock = false;
    for (auto use : phi->getUses()) {
      if (isa<PhiOp>(use) && use->getParent() == phi->getParent()) {
        usedByPhiInSameBlock = true;
        break;
      }
    }
    if (usedByPhiInSameBlock)
      continue;

    setAfterPhis(phi->getParent());
    auto value = builder.create<IntOp>({ new IntAttr(*cst) });
    phi->replaceAllUsesWith(value);
    phi->erase();
    pathReplacements++;
  }

  bool enableEqFold = true;
  if (const char *env = std::getenv("SISY_ENABLE_EQCLASS_FOLD"))
    enableEqFold = env[0] && std::strcmp(env, "0") != 0;

  auto foldKnownBool = [&](Op *op) {
    if (!op->has<RangeAttr>())
      return false;
    auto [low, high] = RANGE(op);
    if (low != high || (low != 0 && low != 1))
      return false;
    folded++;
    builder.replace<IntOp>(op, { new IntAttr(low) });
    return true;
  };

  auto knownTruthyValue = [](Op *op) -> std::optional<int> {
    auto range = intRange(op);
    if (!range)
      return std::nullopt;
    auto [low, high] = *range;
    if (low == 0 && high == 0)
      return 0;
    if (high < 0 || low > 0)
      return 1;
    return std::nullopt;
  };

  runRewriter([&](EqOp *op) { return foldKnownBool(op); });
  runRewriter([&](NeOp *op) { return foldKnownBool(op); });
  runRewriter([&](LtOp *op) { return foldKnownBool(op); });
  runRewriter([&](LeOp *op) { return foldKnownBool(op); });

  runRewriter([&](NotOp *op) {
    auto truth = knownTruthyValue(op->DEF());
    if (!truth)
      return false;
    folded++;
    builder.replace<IntOp>(op, { new IntAttr(*truth ? 0 : 1) });
    return true;
  });

  runRewriter([&](SetNotZeroOp *op) {
    auto truth = knownTruthyValue(op->DEF());
    if (!truth)
      return false;
    folded++;
    builder.replace<IntOp>(op, { new IntAttr(*truth) });
    return true;
  });

  runRewriter([&](BranchOp *op) {
    auto truth = knownTruthyValue(op->DEF());
    if (!truth)
      return false;
    folded++;
    builder.setBeforeOp(op);
    auto value = builder.create<IntOp>({ new IntAttr(*truth) });
    op->replaceOperand(op->DEF(), value);
    return false;
  });

  auto threadOneEdge = [&]() -> bool {
    for (auto func : funcs) {
      auto region = func->getRegion();
      region->updatePreds();
      auto blocks = region->getBlocks();
      for (auto bb : blocks) {
        auto term = bb->getLastOp();
        if (!term || !isa<BranchOp>(term) || bb->preds.size() < 2 ||
            !decisionBlockCanBeSkipped(bb))
          continue;

        std::vector<BasicBlock*> preds(bb->preds.begin(), bb->preds.end());
        for (auto pred : preds) {
          if (!pred || pred == bb)
            continue;
          auto truth = branchTruthOnEdge(term->DEF(), pred, bb);
          if (!truth)
            continue;

          auto target = *truth ? TARGET(term) : ELSE(term);
          if (!target || target == bb || target->preds.count(pred))
            continue;

          bool ok = true;
          std::vector<std::pair<Op*, Op*>> targetIncoming;
          for (auto phi : target->getPhis()) {
            auto incoming = phiIncoming(phi, bb);
            if (!incoming || !availableBeforeEdge(incoming, pred, bb)) {
              ok = false;
              break;
            }
            targetIncoming.push_back({ phi, edgeValue(incoming, pred, bb) });
          }
          if (!ok)
            continue;

          auto predTerm = pred->getLastOp();
          bool rewired = false;
          if (isa<GotoOp>(predTerm) && TARGET(predTerm) == bb) {
            TARGET(predTerm) = target;
            rewired = true;
          } else if (isa<BranchOp>(predTerm)) {
            if (TARGET(predTerm) == bb) {
              TARGET(predTerm) = target;
              rewired = true;
            }
            if (ELSE(predTerm) == bb) {
              ELSE(predTerm) = target;
              rewired = true;
            }
          }
          if (!rewired)
            continue;

          for (auto [phi, incoming] : targetIncoming) {
            phi->pushOperand(incoming);
            phi->add<FromAttr>(pred);
          }
          for (auto phi : bb->getPhis())
            removePhiIncoming(phi, pred);

          region->updatePreds();
          threadedEdges++;
          return true;
        }
      }
    }
    return false;
  };

  while (threadOneEdge()) {}

  // Fold left/right shifts early.
  runRewriter([&](DivIOp *op) {
    auto x = op->DEF(0);
    auto y = op->DEF(1);
    if (!isa<IntOp>(y) || V(y) < 0 || !x->has<RangeAttr>())
      return false;

    auto [low, high] = RANGE(x);
    if (low > high)
      return false;
    if (low < 0)
      return false;

    if (__builtin_popcount(V(y)) != 1)
      return false;

    // This can be replaced to an ordinary right-shift.
    folded++;
    builder.setBeforeOp(op);
    auto vi = builder.create<IntOp>({ new IntAttr(__builtin_ctz(V(y))) });
    builder.replace<RShiftOp>(op, { x, vi });
    return false;
  });

  runRewriter([&](ModIOp *op) {
    auto x = op->DEF(0);
    auto y = op->DEF(1);
    if (!isa<IntOp>(y) || !x->has<RangeAttr>())
      return false;

    if (V(y) < 0)
      V(y) = -V(y);

    auto [low, high] = RANGE(x);
    if (low > high)
      return false;
    if (low < 0)
      return false;

    if (__builtin_popcount(V(y)) != 1)
      return false;

    // Replace with bit-and.
    folded++;
    builder.setBeforeOp(op);
    auto vi = builder.create<IntOp>({ new IntAttr(V(y) - 1) });
    builder.replace<AndIOp>(op, { x, vi });
    return false;
  });

  Rule eq_or("(or (eq x 1) (not x))");
  runRewriter([&](OrIOp *op) {
    if (eq_or.match(op)) {
      auto x = eq_or.extract("x");
      if (!x->has<RangeAttr>())
        return false;
      auto [low, high] = RANGE(x);
      if (low == 0) {
        folded++;
        builder.setBeforeOp(op);
        auto two = builder.create<IntOp>({ new IntAttr(2) });
        builder.replace<LtOp>(op, { x, two });
        return false;
      }
    }
    return false;
  });

  if (enableEqFold) {
    runRewriter([&](EqOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameValueOrEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(1) });
      return true;
    });

    runRewriter([&](NeOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameValueOrEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(0) });
      return true;
    });

    runRewriter([&](LtOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameValueOrEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(0) });
      return true;
    });

    runRewriter([&](LeOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameValueOrEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(1) });
      return true;
    });

    runRewriter([&](SubIOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameValueOrEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(0) });
      return true;
    });
  }
  
  for (auto func : funcs) {
    auto region = func->getRegion();
    removeRange(region);
    removeEqClass(region);
  }
}
