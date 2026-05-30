#ifndef SISY_PATTERN_REWRITER_H
#define SISY_PATTERN_REWRITER_H

#include "Pass.h"
#include "../codegen/Attrs.h"
#include "../codegen/CodeGen.h"

#include <memory>
#include <string>
#include <vector>

namespace sys {

struct PatternBenefit {
  int value = 1;

  explicit PatternBenefit(int value = 1): value(value) {}
  bool operator<(const PatternBenefit &other) const {
    return value < other.value;
  }
};

class PatternRewriter {
  Builder builder;
  bool changed = false;

public:
  bool replaceOp(Op *op, Op *replacement);
  template<class T>
  T *replaceOpWithNew(Op *op, const std::vector<Attr*> &attrs) {
    builder.setBeforeOp(op);
    auto created = builder.create<T>(attrs);
    op->replaceAllUsesWith(created);
    op->erase();
    changed = true;
    return created;
  }

  bool eraseOp(Op *op);
  bool hasChanged() const { return changed; }
  void resetChanged() { changed = false; }
};

class Pattern {
public:
  virtual ~Pattern() = default;
  virtual std::string name() const = 0;
  virtual PatternBenefit benefit() const { return PatternBenefit(1); }
  virtual bool matchAndRewrite(Op *op, PatternRewriter &rewriter) const = 0;
};

struct RewriteStats {
  int attempts = 0;
  int rewrites = 0;
  int iterations = 0;
  int rejected = 0;
  int convergenceBailouts = 0;
};

class RewriteDriver {
public:
  static RewriteStats runGreedy(
      Region *region,
      const std::vector<std::unique_ptr<Pattern>> &patterns,
      int maxIterations = 64);
};

void populateCoreCanonicalizationPatterns(
    std::vector<std::unique_ptr<Pattern>> &patterns);

} // namespace sys

#endif
