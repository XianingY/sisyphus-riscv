#ifndef HIR_POLYHEDRAL_H
#define HIR_POLYHEDRAL_H

#include "HIROps.h"
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace sys::hir {

struct PolyhedralStats {
  int reductionJammed = 0;
  int reductionPrivatized = 0;
  int reductionRowJamConditional = 0;
  int reductionMicroTiled = 0;
  int reductionMicroTileInPlace = 0;
  int reductionMicroTileConditional = 0;
  int reductionMicroTile3D = 0;
  int reductionMicroTileRejectDependence = 0;
  int reductionMicroTileRejectPressure = 0;
  int microTileMrSum = 0;
  int microTileNrSum = 0;
  int microTileKcSum = 0;
  int microTileNcSum = 0;
  int reductionInterchanged = 0;
  int conditionalReductionInterchanged = 0;
  int repeatReduced = 0;
  int repeatRejected = 0;
  int repeatRejectShape = 0;
  int repeatRejectInit = 0;
  int repeatRejectBound = 0;
  int repeatRejectLegal = 0;
  int repeatRejectClone = 0;
  int overwriteRepeatReduced = 0;
  int overwriteRepeatRejected = 0;
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
  // Loop interchange counters
  int interchangeApplied = 0;
  int interchangeRejected = 0;
  int interchangeRejectShape = 0;
  int interchangeRejectInit = 0;
  int interchangeRejectBounds = 0;
  int interchangeRejectControl = 0;
  int interchangeRejectAccess = 0;
  int interchangeRejectMemory = 0;
  // 3D j/k loop interchange counters.
  int interchange3DApplied = 0;
  int interchange3DRejected = 0;
  int interchange3DRejectShape = 0;
  int interchange3DRejectInit = 0;
  int interchange3DRejectBounds = 0;
  int interchange3DRejectControl = 0;
  int interchange3DRejectAccess = 0;
  int interchange3DRejectMemory = 0;
  // Unroll-and-jam counters
  int unrollJammed = 0;
  int unrollJamRejected = 0;
  int unrollJamRejectShape = 0;
  int unrollJamRejectInit = 0;
  int unrollJamRejectBounds = 0;
  int unrollJamRejectControl = 0;
  int unrollJamRejectAccess = 0;
  int unrollJamRejectMemory = 0;
  // Order-preserving innermost partial unroll counters.
  int partialUnrolled = 0;
  int partialUnrollRejected = 0;
  int partialUnrollRejectShape = 0;
  int partialUnrollRejectControl = 0;
  int partialUnrollRejectAccess = 0;
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
  int presburgerFusionQueries = 0;
  int presburgerFusionNoDeps = 0;
  int presburgerFusionMayDeps = 0;
  int presburgerFusionUnknown = 0;
  int presburgerInterchangeQueries = 0;
  int presburgerInterchangeNoDeps = 0;
  int presburgerInterchangeMayDeps = 0;
  int presburgerInterchangeUnknown = 0;
  // Stats-only affine nest scanner. These counters do not enable rewrites.
  int affineNestCandidates = 0;
  int affineNestRejectedShape = 0;
  int affineNestRejectedControl = 0;
  int affineNestRejectedAccess = 0;
  int affineNestPerfect2D = 0;
  int affineNestPerfect3D = 0;
  int matmulLikeCandidates = 0;
  // Stencil/conv-like boundary dispatch. This keeps the original boundary
  // guarded body as fallback and exposes an unchecked interior body.
  int stencilInteriorDispatched = 0;
  int stencilInteriorRejected = 0;
  int stencilInteriorRejectShape = 0;
  int stencilInteriorRejectBounds = 0;
  // Hoist an invariant conjunct from a guarded inner loop when the loop has
  // no observable work outside that guard.
  int invariantGuardHoisted = 0;
  int invariantGuardRejected = 0;
  // Monotone guard tightening:
  //   while (j < B) { if (i < j) { j++; continue; } body; j++; }
  // becomes j < min(B, i + 1).  This is a general triangular-loop cleanup.
  int monotoneGuardTightened = 0;
  int monotoneGuardRejected = 0;
  int monotoneGuardRejectShape = 0;
  int monotoneGuardRejectUse = 0;
  // Loop distribution / fission counters. Splits a single canonical loop
  // body into two adjacent loops when the two halves access disjoint
  // memory (no array/scalar overlap), exposing each half independently
  // to vectorize / regalloc.
  int loopDistributionApplied = 0;
  int loopDistributionRejected = 0;
  int loopDistributionRejectShape = 0;     // not a canonical perfect while
  int loopDistributionRejectControl = 0;   // calls / nested while in body
  int loopDistributionRejectNoSplit = 0;   // no viable split point found
};

class PolyhedralOptimizer {
public:
  PolyhedralStats run(Module &module);

private:
  int uniqueId = 0;
  int hirTileSize = 32;
  int hirJamFactor = 4;
  std::unordered_set<std::string> globalArrays;
  std::unordered_map<std::string, std::vector<int>> globalArrayDims;
  std::unordered_set<const Op*> monotoneTightenedLoops;
  std::unordered_set<const Op*> partialUnrollRemainders;

  bool optimizeBlock(Op *block, PolyhedralStats &stats);
  void scanAffineNest(Op *op, PolyhedralStats &stats);
  bool tryReductionInterchange(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryReductionMicroTile(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryReductionRowPrivatize(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryReductionJam(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryLoopInterchange3D(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryLoopInterchange(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryLoopUnrollJam(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryInnermostPartialUnroll(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryRepeatInvariantReduction(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryDeadOverwriteRepeat(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop tiling: strip-mine 2-level or 3-level perfect nests in HIR.
  bool tryLoopTiling(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop fusion: merge two adjacent loops with identical bounds over same arrays.
  bool tryLoopFusion(Op *block, size_t idx, PolyhedralStats &stats);
  bool forwardArrayStoreLoads(Op *block, PolyhedralStats &stats);
  bool tryStencilInteriorDispatch(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryInvariantGuardHoist(Op *block, size_t idx, PolyhedralStats &stats);
  bool tryMonotoneGuardBoundTightening(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop distribution: split one canonical perfect while into two adjacent
  // whiles over the same iteration space, when the two halves of the body
  // access disjoint arrays and scalars.
  bool tryLoopDistribute(Op *block, size_t idx, PolyhedralStats &stats);
  std::unordered_map<std::string, Op*> functions;
};

}  // namespace sys::hir

#endif
