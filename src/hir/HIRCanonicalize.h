#ifndef HIR_CANONICALIZE_H
#define HIR_CANONICALIZE_H

#include "HIROps.h"

namespace sys::hir {

struct CanonStats {
  int constFolded = 0;
  int deadBranchesEliminated = 0;
  int simpleAffineCallsInlined = 0;
};

class Canonicalizer {
public:
  CanonStats run(Module &module);

private:
  bool foldConstExpr(Op *op, CanonStats &stats);
  bool simplifyStructuredControl(Op *op, CanonStats &stats);
  bool inlineSimpleAffineCalls(Module &module, CanonStats &stats);
};

}  // namespace sys::hir

#endif
