#ifndef HIR_AFFINE_H
#define HIR_AFFINE_H

#include "HIROps.h"

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sys::hir::affine {

struct Expr {
  bool valid = false;
  int64_t constant = 0;
  std::map<std::string, int64_t> coeffs;
};

struct Access {
  const Op *op = nullptr;
  std::string base;
  std::vector<Expr> indices;
  bool isStore = false;
};

struct CanonicalLoop {
  std::string iv;
  const Op *bound = nullptr;
  const Op *body = nullptr;
  const Op *step = nullptr;
};

Expr analyzeExpr(const Op *op);
bool matchCanonicalLoop(const Op *op, CanonicalLoop &loop);
std::vector<Access> collectArrayAccesses(const Op *op);

bool exprUsesAny(const Expr &expr, const std::unordered_set<std::string> &symbols);
bool opWritesAnyScalarUsedBy(const Op *op, const Op *expr);
bool hasAffineArrayAccessUsing(const Op *op, const std::string &symbol,
                               int minRank = 1);

bool fusionMemorySafe(const Op *loopA, const Op *loopB);

}  // namespace sys::hir::affine

#endif
