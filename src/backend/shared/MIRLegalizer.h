#ifndef SISY_BACKEND_SHARED_MIR_LEGALIZER_H
#define SISY_BACKEND_SHARED_MIR_LEGALIZER_H

#include "../../codegen/Ops.h"

#include <map>
#include <string>

namespace sys::backend::shared {

enum class LegalizeAction {
  Legal,
  Promote,
  Split,
  Expand,
  Custom,
  Illegal,
};

struct LegalizeStats {
  int legal = 0;
  int promoted = 0;
  int split = 0;
  int expanded = 0;
  int custom = 0;
  int illegal = 0;
  int verifierErrors = 0;

  std::map<std::string, int> toMap() const;
};

class TargetLegalizerInfo {
public:
  virtual ~TargetLegalizerInfo() = default;
  virtual const char *name() const = 0;
  virtual LegalizeAction classify(Op *op, std::string &reason) const = 0;
};

bool strictMIRLegalizer();
LegalizeStats verifyAndCount(ModuleOp *module,
                             const TargetLegalizerInfo &target,
                             bool emitDiagnostics);

} // namespace sys::backend::shared

#endif
