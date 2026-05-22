#ifndef HIR_POLYHEDRAL_H
#define HIR_POLYHEDRAL_H

#include "HIROps.h"
#include <string>
#include <unordered_set>

namespace sys::hir {

struct PolyhedralStats {
  int reductionJammed = 0;
  int reductionInterchanged = 0;
  int conditionalReductionInterchanged = 0;
  int repeatReduced = 0;
  int repeatRejected = 0;
  int repeatRejectShape = 0;
  int repeatRejectInit = 0;
  int repeatRejectBound = 0;
  int repeatRejectLegal = 0;
  int repeatRejectClone = 0;
  int rejected = 0;
  // Loop tiling counters
  int tilingApplied = 0;
  int tilingRejected = 0;
  int tilingRejectShape = 0;
  int tilingRejectControl = 0;
  int tilingRejectBoundWrite = 0;
  int tilingRejectAffineAccess = 0;
  int tilingRejectNoInner = 0;
  int tilingRejectIdempotent = 0;
  // Loop fusion counters
  int fusionApplied = 0;
  int fusionRejected = 0;
  int fusionRejectShape = 0;
  int fusionRejectInit = 0;
  int fusionRejectBounds = 0;
  int fusionRejectControl = 0;
  int fusionRejectScalar = 0;
  int fusionRejectMemory = 0;
  int forwardedArrayStoreLoads = 0;
  // Stats-only affine nest scanner. These counters do not enable rewrites.
  int affineNestCandidates = 0;
  int affineNestRejectedShape = 0;
  int affineNestRejectedControl = 0;
  int affineNestRejectedAccess = 0;
  int affineNestPerfect2D = 0;
  int affineNestPerfect3D = 0;
  int matmulLikeCandidates = 0;
};

class PolyhedralOptimizer {
public:
  PolyhedralStats run(Module &module);

private:
  int uniqueId = 0;
  int hirTileSize = 32;
  std::unordered_set<std::string> globalArrays;

  bool optimizeBlock(Op *block, PolyhedralStats &stats);
  void scanAffineNest(Op *op, PolyhedralStats &stats);
  bool tryReductionInterchange(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryReductionJam(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryRepeatInvariantReduction(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop tiling: strip-mine 2-level or 3-level perfect nests in HIR.
  bool tryLoopTiling(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop fusion: merge two adjacent loops with identical bounds over same arrays.
  bool tryLoopFusion(Op *block, size_t idx, PolyhedralStats &stats);
  bool forwardArrayStoreLoads(Op *block, PolyhedralStats &stats);
};

}  // namespace sys::hir

#endif
