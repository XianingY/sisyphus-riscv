#ifndef HIR_POLYHEDRAL_H
#define HIR_POLYHEDRAL_H

#include "HIROps.h"
#include <string>
#include <unordered_set>

namespace sys::hir {

struct PolyhedralStats {
  int reductionJammed = 0;
  int reductionInterchanged = 0;
  int repeatReduced = 0;
  int repeatRejected = 0;
  int rejected = 0;
  // Loop tiling counters
  int tilingApplied = 0;
  int tilingRejected = 0;
  // Loop fusion counters
  int fusionApplied = 0;
  int fusionRejected = 0;
};

class PolyhedralOptimizer {
public:
  PolyhedralStats run(Module &module);

private:
  int uniqueId = 0;
  int hirTileSize = 32;
  std::unordered_set<std::string> globalArrays;

  bool optimizeBlock(Op *block, PolyhedralStats &stats);
  bool tryReductionInterchange(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryReductionJam(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryRepeatInvariantReduction(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop tiling: strip-mine 2-level or 3-level perfect nests in HIR.
  bool tryLoopTiling(Op *block, size_t idx, PolyhedralStats &stats);

  // Loop fusion: merge two adjacent loops with identical bounds over same arrays.
  bool tryLoopFusion(Op *block, size_t idx, PolyhedralStats &stats);
};

}  // namespace sys::hir

#endif
