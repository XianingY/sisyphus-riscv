#include "CleanupPasses.h"
#include "Analysis.h"
#include "../utils/Matcher.h"

#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

using namespace sys;

std::map<std::string, int> RangeAwareFold::stats() {
  return {
    { "folded-ops", folded },
    { "path-replacements", pathReplacements },
  };
}

// Defined in Specialize.cpp.
void removeRange(Region *region);

void RangeAwareFold::run() {
  Builder builder;
  auto sameEqClass = [](Op *a, Op *b) -> bool {
    return a && b && a->has<EqClassAttr>() && b->has<EqClassAttr>() && EQCLASS(a) == EQCLASS(b);
  };
  auto sameValueOrEqClass = [&](Op *a, Op *b) -> bool {
    return a && b && (a == b || sameEqClass(a, b));
  };

  auto singletonRange = [](Op *op) -> std::optional<int> {
    if (!op || op->getResultType() == Value::f32 || !op->has<RangeAttr>())
      return std::nullopt;
    auto [low, high] = RANGE(op);
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
    if (!op || op->getResultType() == Value::f32 || !op->has<RangeAttr>())
      return std::nullopt;
    auto [low, high] = RANGE(op);
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

  // Fold left/right shifts early.
  runRewriter([&](DivIOp *op) {
    auto x = op->DEF(0);
    auto y = op->DEF(1);
    if (!isa<IntOp>(y) || V(y) < 0 || !x->has<RangeAttr>())
      return false;

    auto [low, high] = RANGE(x);
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
