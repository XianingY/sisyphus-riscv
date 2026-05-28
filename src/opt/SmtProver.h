#ifndef SMT_PROVER_H
#define SMT_PROVER_H

#include "../codegen/Ops.h"
#include "../utils/smt/SMT.h"

namespace sys::smt_prover {

// Try to prove that two i32 SSA values produce equal results for every
// possible assignment of their free inputs.
//
// Supported ops: IntOp, GetArgOp, AddIOp, SubIOp, MulIOp, AndIOp, OrIOp,
// XorIOp, LShiftOp, RShiftOp, MinusOp, NotOp.  Encountering any other op
// returns Result::Unknown so callers can stay conservative.
//
// This is intended as a verification helper for narrow, local peephole
// folds (superopt style).  Default consumers should treat Unknown as
// "do not fold".  The pass is opt-in via SISY_ENABLE_SMT_PROVER=1 in the
// callers that use it.
enum class Result { Equal, NotEqual, Unknown };

Result tryProveEqualI32(Op *a, Op *b);

// Same supported op subset.  Returns true iff the solver proves that
// `cond` evaluates to a non-zero value under every assignment of free
// inputs.  Used by `SmtBranchPrune` to fold always-taken branches.  Any
// op outside the supported subset yields `false` (treated as "unknown").
bool tryProveNonZeroI32(Op *cond);

// Returns true iff the solver proves `cond` evaluates to zero under
// every assignment of free inputs.  Folds always-not-taken branches.
bool tryProveZeroI32(Op *cond);

}

#endif
