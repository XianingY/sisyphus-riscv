#ifndef SISY_TARGET_COST_MODEL_H
#define SISY_TARGET_COST_MODEL_H

#include "../../codegen/Ops.h"

namespace sys {

enum class TargetKind {
  Generic,
  RiscV,
  Arm,
};

struct OperationCost {
  int latency = 1;
  int throughput = 1;
  int registerPressure = 1;
  int spillRisk = 0;
  bool legalAddressing = true;
};

class TargetCostModel {
  TargetKind kind = TargetKind::Generic;

public:
  explicit TargetCostModel(TargetKind kind = TargetKind::Generic): kind(kind) {}

  static TargetCostModel forCurrentTarget();
  TargetKind target() const { return kind; }
  OperationCost cost(Op *op) const;
  int loadUseDelay() const;
  int multiplyUseDelay() const;
  int gprBudget() const;
};

} // namespace sys

#endif
