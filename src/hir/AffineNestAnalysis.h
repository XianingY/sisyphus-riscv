#ifndef HIR_AFFINE_NEST_ANALYSIS_H
#define HIR_AFFINE_NEST_ANALYSIS_H

#include "HIROps.h"

namespace sys::hir::affine {

struct AffineNestAnalysisSummary {
  int opsVisited = 0;
  int nests = 0;
  int perfect2D = 0;
  int perfect3D = 0;
  int maxDepth = 0;
  int maxAccessRank = 0;
  int affineAccesses = 0;
  int nonAffineAccesses = 0;
  int contiguousAccesses = 0;
  int stridedAccesses = 0;
  int nestsWithCalls = 0;
  int nestsWithStores = 0;
  int nestsWithReductions = 0;
  int nestsWithSymbolicAccesses = 0;
  int pressureEstimateMax = 0;
};

AffineNestAnalysisSummary analyzeAffineNests(const Module &module);

}  // namespace sys::hir::affine

#endif
