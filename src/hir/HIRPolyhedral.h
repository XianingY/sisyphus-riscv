#ifndef HIR_POLYHEDRAL_H
#define HIR_POLYHEDRAL_H

#include "HIROps.h"

namespace sys::hir {

struct PolyhedralStats {
  int reductionJammed = 0;
  int rejected = 0;
};

class PolyhedralOptimizer {
public:
  PolyhedralStats run(Module &module);

private:
  int uniqueId = 0;

  bool optimizeBlock(Op *block, PolyhedralStats &stats);
  bool tryReductionJam(Op *block, size_t initIndex, PolyhedralStats &stats);
};

}  // namespace sys::hir

#endif
