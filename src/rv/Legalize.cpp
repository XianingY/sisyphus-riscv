#include "RvPasses.h"
#include "../backend/shared/MIRLegalizer.h"

#include <cstdlib>
#include <iostream>

using namespace sys;
using namespace sys::rv;
using namespace sys::backend::shared;

namespace {

bool fitsSImm12(int v) {
  return v >= -2048 && v <= 2047;
}

bool legalRvMemSize(Op *op) {
  if (!op->has<SizeAttr>())
    return false;
  int size = SIZE(op);
  return size == 4 || size == 8;
}

class RVLegalizerInfo final : public TargetLegalizerInfo {
public:
  const char *name() const override { return "rv"; }

  LegalizeAction classify(Op *op, std::string &reason) const override {
    if (isa<sys::rv::LoadOp>(op) || isa<sys::rv::StoreOp>(op)) {
      if (!legalRvMemSize(op)) {
        reason = "unsupported scalar memory size";
        return LegalizeAction::Illegal;
      }
      if (op->has<IntAttr>() && !fitsSImm12(V(op)))
        return LegalizeAction::Custom;
      return LegalizeAction::Legal;
    }
    if (isa<FldOp>(op) || isa<FsdOp>(op)) {
      if (!op->has<IntAttr>() || !fitsSImm12(V(op)))
        return LegalizeAction::Custom;
      return LegalizeAction::Legal;
    }
    if (isa<Vle32Op>(op) || isa<Vlse32Op>(op) || isa<Vse32Op>(op) ||
        isa<Vsse32Op>(op) || isa<VaddvvOp>(op) || isa<VsubvvOp>(op) ||
        isa<VmulvvOp>(op) || isa<VmvvxOp>(op) || isa<VfaddvvOp>(op) ||
        isa<VfsubvvOp>(op) || isa<VfmulvvOp>(op) || isa<VfmvvfOp>(op) ||
        isa<VredsumOp>(op) || isa<VfredsumOp>(op) ||
        isa<VsetvliOp>(op) || isa<VsetvliResultOp>(op))
      return LegalizeAction::Legal;
    if (op->getResultType() == Value::i128 ||
        op->getResultType() == Value::f128 ||
        op->getResultType() == Value::vscale_i32 ||
        op->getResultType() == Value::vscale_f32) {
      reason = "generic vector op reached RV MIR";
      return LegalizeAction::Illegal;
    }
    return LegalizeAction::Legal;
  }
};

} // namespace

std::map<std::string, int> Legalize::stats() {
  return {
    { "legal", legal },
    { "promoted", promoted },
    { "split", split },
    { "expanded", expanded },
    { "custom", custom },
    { "illegal", illegal },
    { "verifier-errors", verifierErrors },
  };
}

void Legalize::run() {
  RVLegalizerInfo info;
  auto s = verifyAndCount(module, info, /*emitDiagnostics=*/ true);
  legal += s.legal;
  promoted += s.promoted;
  split += s.split;
  expanded += s.expanded;
  custom += s.custom;
  illegal += s.illegal;
  verifierErrors += s.verifierErrors;
  if (strictMIRLegalizer() && s.verifierErrors) {
    std::cerr << "rv-legalize: strict MIR verification failed\n";
    std::exit(1);
  }
}
