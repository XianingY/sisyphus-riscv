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

struct Coeff {
  int64_t constant = 0;
  std::map<std::string, int64_t> symbols;

  bool isZero() const { return constant == 0 && symbols.empty(); }
};

bool operator==(const Coeff &lhs, const Coeff &rhs);
bool operator!=(const Coeff &lhs, const Coeff &rhs);

struct Expr {
  bool valid = false;
  int64_t constant = 0;
  std::map<std::string, Coeff> coeffs;
};

struct Access {
  const Op *op = nullptr;
  std::string base;
  std::vector<Expr> indices;
  bool isStore = false;
};

enum class DependenceStatus {
  Safe,
  Unsafe,
  Unknown,
};

struct DependenceResult {
  DependenceStatus status = DependenceStatus::Unknown;
  int queries = 0;
  int noDeps = 0;
  int mayDeps = 0;
  int unknown = 0;
  int projectedDims = 0;
  int projectionUnknown = 0;

  bool safe() const { return status == DependenceStatus::Safe; }
};

struct MemoryAccess {
  const Op *op = nullptr;
  std::string base;
  std::vector<Expr> indices;
  bool isWrite = false;
  bool isAffine = false;
  bool isContiguous = false;
  bool isStrided = false;
  int64_t strideBytes = 0;
};

struct ReductionKernelPlan {
  bool legal = false;
  bool conditional = false;
  bool inPlace = false;
  bool needsScratch = true;
  int mr = 1;
  int nr = 0;
  int kc = 0;
  int nc = 0;
  int scratchElems = 0;
  DependenceStatus dependence = DependenceStatus::Unknown;
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

struct LoopDomain {
  std::string iv;
  Expr lower;
  Expr upper;
  int64_t step = 1;
  bool valid = false;
};

struct SideEffectSummary {
  bool hasCall = false;
  bool hasBreakOrContinue = false;
  bool hasReturn = false;
  std::unordered_set<std::string> scalarReads;
  std::unordered_set<std::string> scalarWrites;
  std::unordered_set<std::string> arrayReads;
  std::unordered_set<std::string> arrayWrites;
};

struct AffineNest {
  std::vector<CanonicalLoop> loops;
  std::vector<LoopDomain> domains;
  std::vector<Access> accesses;
  std::vector<MemoryAccess> memory;
  std::vector<const Op*> guards;
  SideEffectSummary effects;
  bool imperfect = false;
  bool hasSymbolicAccesses = false;
};

Expr analyzeExpr(const Op *op);
Expr analyzeExpr(const Op *op, const std::unordered_set<std::string> &loopIVs);
bool matchCanonicalLoop(const Op *op, CanonicalLoop &loop);
bool collectAffineNest(const Op *op, AffineNest &nest, int maxDepth = 5,
                       bool allowGuards = true);
std::vector<Access> collectArrayAccesses(const Op *op);
std::vector<Access> collectArrayAccesses(const Op *op,
                                         const std::unordered_set<std::string> &loopIVs);
bool hasSymbolicCoefficients(const Expr &expr);
bool coeffIsConstant(const Expr &expr, const std::string &symbol, int64_t value);

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
DependenceResult toDependenceResult(const PresburgerFusionResult &result);
DependenceResult toDependenceResult(const PresburgerInterchangeResult &result);
DependenceResult fusionDependence(const Op *loopA, const Op *loopB,
                                  const Op *initA, const Op *initB);
DependenceResult interchangeDependence(const Op *outerLoop,
                                       const Op *innerLoop,
                                       const Op *outerInit,
                                       const Op *innerInit);
DependenceResult permutationDependence(
    const std::vector<const Op*> &loopOps,
    const std::vector<const Op*> &initOps,
    const std::vector<int> &permutation);

}  // namespace sys::hir::affine

#endif
