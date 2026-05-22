#include "CleanupPasses.h"
#include "Analysis.h"
#include "../utils/Matcher.h"

#include <cstdlib>
#include <cstring>
#include <optional>

using namespace sys;

std::map<std::string, int> RangeAwareFold::stats() {
  return {
    { "folded-ops", folded }
  };
}

// Defined in Specialize.cpp.
void removeRange(Region *region);

void RangeAwareFold::run() {
  Builder builder;
  auto sameEqClass = [](Op *a, Op *b) -> bool {
    return a && b && a->has<EqClassAttr>() && b->has<EqClassAttr>() && EQCLASS(a) == EQCLASS(b);
  };
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
      if (!sameEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(1) });
      return true;
    });

    runRewriter([&](NeOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(0) });
      return true;
    });

    runRewriter([&](LtOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(0) });
      return true;
    });

    runRewriter([&](LeOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(1) });
      return true;
    });

    runRewriter([&](SubIOp *op) {
      auto l = op->DEF(0);
      auto r = op->DEF(1);
      if (!sameEqClass(l, r))
        return false;
      folded++;
      builder.replace<IntOp>(op, { new IntAttr(0) });
      return true;
    });
  }
  
  auto funcs = collectFuncs();

  for (auto func : funcs) {
    auto region = func->getRegion();
    removeRange(region);
    removeEqClass(region);
  }
}
