#ifndef HIR_POLYHEDRAL_H
#define HIR_POLYHEDRAL_H

#include "HIROps.h"
#include <string>
#include <unordered_set>

namespace sys::hir {

struct PolyhedralStats {
  int reductionJammed = 0;
  int reductionInterchanged = 0;
  int rejected = 0;
};

class PolyhedralOptimizer {
public:
  PolyhedralStats run(Module &module);

private:
  int uniqueId = 0;
  std::unordered_set<std::string> globalArrays;

  bool optimizeBlock(Op *block, PolyhedralStats &stats);
  bool tryReductionInterchange(Op *block, size_t initIndex, PolyhedralStats &stats);
  bool tryReductionJam(Op *block, size_t initIndex, PolyhedralStats &stats);
};

}  // namespace sys::hir

#endif
