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

struct PresburgerFusionResult {
  bool safe = false;
  int queries = 0;
  int noReorderedDependence = 0;
  int mayReorderedDependence = 0;
  int unknown = 0;
};

struct PresburgerInterchangeResult {
  bool safe = false;
  int queries = 0;
  int noViolatingDependence = 0;
  int mayViolatingDependence = 0;
  int unknown = 0;
  int projectedDims = 0;
  int projectionUnknown = 0;
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
PresburgerFusionResult fusionMemorySafePresburger(const Op *loopA, const Op *loopB);
PresburgerFusionResult fusionMemorySafePresburger(const Op *loopA, const Op *loopB,
                                                  const Op *initA, const Op *initB);
PresburgerInterchangeResult interchangeMemorySafePresburger(const Op *outerLoop,
                                                            const Op *innerLoop,
                                                            const Op *outerInit,
                                                            const Op *innerInit);
PresburgerInterchangeResult permutationMemorySafePresburger(
    const std::vector<const Op*> &loopOps,
    const std::vector<const Op*> &initOps,
    const std::vector<int> &permutation);

}  // namespace sys::hir::affine

#endif
