#include "PatternRewriter.h"

#include <algorithm>

using namespace sys;

namespace {

bool isInt(Op *op, int value) {
  return isa<IntOp>(op) && op->has<IntAttr>() && V(op) == value;
}

bool sameValue(Op *lhs, Op *rhs) {
  return lhs == rhs;
}

std::vector<Op*> collectOps(Region *region) {
  std::vector<Op*> result;
  if (!region)
    return result;
  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      result.push_back(op);
      for (auto nested : op->getRegions()) {
        auto child = collectOps(nested);
        result.insert(result.end(), child.begin(), child.end());
      }
    }
  }
  return result;
}

class AddIdentityPattern final : public Pattern {
public:
  std::string name() const override { return "arith.addi.identity"; }
  PatternBenefit benefit() const override { return PatternBenefit(10); }
  bool matchAndRewrite(Op *op, PatternRewriter &rewriter) const override {
    if (!isa<AddIOp>(op) || op->getOperandCount() != 2)
      return false;
    Op *lhs = op->DEF(0);
    Op *rhs = op->DEF(1);
    if (isInt(rhs, 0))
      return rewriter.replaceOp(op, lhs);
    if (isInt(lhs, 0))
      return rewriter.replaceOp(op, rhs);
    if (isa<IntOp>(lhs) && isa<IntOp>(rhs) && lhs->has<IntAttr>() && rhs->has<IntAttr>())
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(V(lhs) + V(rhs)) });
    return false;
  }
};

class SubIdentityPattern final : public Pattern {
public:
  std::string name() const override { return "arith.subi.identity"; }
  PatternBenefit benefit() const override { return PatternBenefit(9); }
  bool matchAndRewrite(Op *op, PatternRewriter &rewriter) const override {
    if (!isa<SubIOp>(op) || op->getOperandCount() != 2)
      return false;
    Op *lhs = op->DEF(0);
    Op *rhs = op->DEF(1);
    if (isInt(rhs, 0))
      return rewriter.replaceOp(op, lhs);
    if (sameValue(lhs, rhs))
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(0) });
    if (isa<IntOp>(lhs) && isa<IntOp>(rhs) && lhs->has<IntAttr>() && rhs->has<IntAttr>())
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(V(lhs) - V(rhs)) });
    return false;
  }
};

class MulIdentityPattern final : public Pattern {
public:
  std::string name() const override { return "arith.muli.identity"; }
  PatternBenefit benefit() const override { return PatternBenefit(10); }
  bool matchAndRewrite(Op *op, PatternRewriter &rewriter) const override {
    if (!isa<MulIOp>(op) || op->getOperandCount() != 2)
      return false;
    Op *lhs = op->DEF(0);
    Op *rhs = op->DEF(1);
    if (isInt(rhs, 1))
      return rewriter.replaceOp(op, lhs);
    if (isInt(lhs, 1))
      return rewriter.replaceOp(op, rhs);
    if (isInt(rhs, 0) || isInt(lhs, 0))
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(0) });
    if (isa<IntOp>(lhs) && isa<IntOp>(rhs) && lhs->has<IntAttr>() && rhs->has<IntAttr>())
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(V(lhs) * V(rhs)) });
    return false;
  }
};

class AndIdentityPattern final : public Pattern {
public:
  std::string name() const override { return "arith.andi.identity"; }
  PatternBenefit benefit() const override { return PatternBenefit(10); }
  bool matchAndRewrite(Op *op, PatternRewriter &rewriter) const override {
    if (!isa<AndIOp>(op) || op->getOperandCount() != 2)
      return false;
    Op *lhs = op->DEF(0);
    Op *rhs = op->DEF(1);
    if (isInt(rhs, 0) || isInt(lhs, 0))
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(0) });
    if (isInt(rhs, -1))
      return rewriter.replaceOp(op, lhs);
    if (isInt(lhs, -1))
      return rewriter.replaceOp(op, rhs);
    if (sameValue(lhs, rhs))
      return rewriter.replaceOp(op, lhs);
    if (isa<IntOp>(lhs) && isa<IntOp>(rhs) && lhs->has<IntAttr>() && rhs->has<IntAttr>())
      return rewriter.replaceOpWithNew<IntOp>(op, { new IntAttr(V(lhs) & V(rhs)) });
    return false;
  }
};

class SelectFoldPattern final : public Pattern {
public:
  std::string name() const override { return "arith.select.fold"; }
  PatternBenefit benefit() const override { return PatternBenefit(8); }
  bool matchAndRewrite(Op *op, PatternRewriter &rewriter) const override {
    if (!isa<SelectOp>(op) || op->getOperandCount() != 3)
      return false;
    Op *cond = op->DEF(0);
    Op *trueValue = op->DEF(1);
    Op *falseValue = op->DEF(2);
    if (sameValue(trueValue, falseValue))
      return rewriter.replaceOp(op, trueValue);
    if (isInt(cond, 0))
      return rewriter.replaceOp(op, falseValue);
    if (isa<IntOp>(cond) && cond->has<IntAttr>())
      return rewriter.replaceOp(op, trueValue);
    return false;
  }
};

} // namespace

bool PatternRewriter::replaceOp(Op *op, Op *replacement) {
  if (!op || !replacement || op == replacement)
    return false;
  op->replaceAllUsesWith(replacement);
  op->erase();
  changed = true;
  return true;
}

bool PatternRewriter::eraseOp(Op *op) {
  if (!op || !op->getUses().empty())
    return false;
  op->erase();
  changed = true;
  return true;
}

RewriteStats RewriteDriver::runGreedy(
    Region *region,
    const std::vector<std::unique_ptr<Pattern>> &patterns,
    int maxIterations) {
  RewriteStats stats;
  if (!region || patterns.empty())
    return stats;

  std::vector<const Pattern*> ordered;
  ordered.reserve(patterns.size());
  for (const auto &pattern : patterns)
    ordered.push_back(pattern.get());
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const Pattern *lhs, const Pattern *rhs) {
                     if (lhs->benefit().value != rhs->benefit().value)
                       return lhs->benefit().value > rhs->benefit().value;
                     return lhs->name() < rhs->name();
                   });

  PatternRewriter rewriter;
  bool changed = false;
  do {
    changed = false;
    stats.iterations++;
    if (stats.iterations > maxIterations) {
      stats.convergenceBailouts++;
      break;
    }
    auto ops = collectOps(region);
    for (auto op : ops) {
      if (!op || !op->getParent())
        continue;
      bool rewritten = false;
      for (const auto *pattern : ordered) {
        stats.attempts++;
        rewriter.resetChanged();
        if (pattern->matchAndRewrite(op, rewriter)) {
          if (rewriter.hasChanged()) {
            stats.rewrites++;
            changed = true;
            rewritten = true;
          }
          break;
        }
      }
      if (!rewritten)
        stats.rejected++;
    }
    Op::release();
  } while (changed);

  return stats;
}

void sys::populateCoreCanonicalizationPatterns(
    std::vector<std::unique_ptr<Pattern>> &patterns) {
  patterns.emplace_back(std::make_unique<AddIdentityPattern>());
  patterns.emplace_back(std::make_unique<SubIdentityPattern>());
  patterns.emplace_back(std::make_unique<MulIdentityPattern>());
  patterns.emplace_back(std::make_unique<AndIdentityPattern>());
  patterns.emplace_back(std::make_unique<SelectFoldPattern>());
}
