#include "ArmPasses.h"
#include "../backend/shared/MIRLegalizer.h"

#include <cstdlib>
#include <iostream>

using namespace sys;
using namespace sys::arm;
using namespace sys::backend::shared;

namespace {

bool fitsAddImm12(int imm) {
  return imm >= -4095 && imm <= 4095;
}

class ArmLegalizerInfo final : public TargetLegalizerInfo {
public:
  const char *name() const override { return "arm"; }

  LegalizeAction classify(Op *op, std::string &reason) const override {
    if ((isa<AddXIOp>(op) || isa<AddWIOp>(op)) && op->has<IntAttr>() &&
        !fitsAddImm12(V(op)))
      return LegalizeAction::Expand;
    if (isa<MovIOp>(op) && op->has<IntAttr>()) {
      int v = V(op);
      if (v >= 65536 || v < -65536)
        return LegalizeAction::Expand;
    }
    if (op->getResultType() == Value::vscale_i32 ||
        op->getResultType() == Value::vscale_f32) {
      reason = "scalable vector op reached ARM MIR";
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
  ArmLegalizerInfo info;
  auto s = verifyAndCount(module, info, /*emitDiagnostics=*/ true);
  legal += s.legal;
  promoted += s.promoted;
  split += s.split;
  expanded += s.expanded;
  custom += s.custom;
  illegal += s.illegal;
  verifierErrors += s.verifierErrors;
  if (strictMIRLegalizer() && s.verifierErrors) {
    std::cerr << "arm-legalize: strict MIR verification failed\n";
    std::exit(1);
  }
}
