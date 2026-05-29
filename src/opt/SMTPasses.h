#ifndef SMT_PASSES_H
#define SMT_PASSES_H

#include "Pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"
#include "../utils/smt/SMT.h"

namespace sys {

// Use SMT solver to guess a formula for proven readonly constant arrays.
class SynthConstArray : public Pass {
  smt::BvExprContext ctx;

  std::vector<smt::BvExpr*> candidates;
  Builder builder;
  int arraysConsidered = 0;
  int arraysSynthesized = 0;
  int loadsReplaced = 0;
  int rejectMutable = 0;
  int rejectNonAffineIndex = 0;
  int rejectNoFormula = 0;

  Op *reconstruct(smt::BvExpr *expr, Op *subscript, int c0, int c1);
public:
  SynthConstArray(ModuleOp *module);

  std::string name() override { return "synth-const-array"; };
  std::map<std::string, int> stats() override {
    return {
      { "arrays-considered", arraysConsidered },
      { "arrays-synthesized", arraysSynthesized },
      { "loads-replaced", loadsReplaced },
      { "reject-mutable", rejectMutable },
      { "reject-non-affine-index", rejectNonAffineIndex },
      { "reject-no-formula", rejectNoFormula },
    };
  }
  void run() override;
};

}

#endif
