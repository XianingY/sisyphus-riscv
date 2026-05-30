#include "TargetCostModel.h"

#include <cstdlib>
#include <cstring>

using namespace sys;

TargetCostModel TargetCostModel::forCurrentTarget() {
  if (const char *rv = std::getenv("SISY_TARGET_RISCV"))
    if (rv[0] && std::strcmp(rv, "0") != 0)
      return TargetCostModel(TargetKind::RiscV);
  if (const char *arm = std::getenv("SISY_TARGET_ARM"))
    if (arm[0] && std::strcmp(arm, "0") != 0)
      return TargetCostModel(TargetKind::Arm);
  return TargetCostModel(TargetKind::Generic);
}

OperationCost TargetCostModel::cost(Op *op) const {
  OperationCost result;
  if (!op)
    return result;
  if (isa<LoadOp>(op)) {
    result.latency = loadUseDelay() + 1;
    result.registerPressure = 1;
    result.spillRisk = 1;
  } else if (isa<StoreOp>(op)) {
    result.latency = 1;
    result.registerPressure = 0;
  } else if (isa<MulIOp>(op) || isa<MulLOp>(op)) {
    result.latency = multiplyUseDelay() + 1;
  } else if (isa<CallOp>(op)) {
    result.latency = 8;
    result.registerPressure = 8;
    result.spillRisk = 4;
  }
  return result;
}

int TargetCostModel::loadUseDelay() const {
  switch (kind) {
  case TargetKind::RiscV: return 2;
  case TargetKind::Arm: return 2;
  case TargetKind::Generic: return 2;
  }
  return 2;
}

int TargetCostModel::multiplyUseDelay() const {
  switch (kind) {
  case TargetKind::RiscV: return 3;
  case TargetKind::Arm: return 3;
  case TargetKind::Generic: return 3;
  }
  return 3;
}

int TargetCostModel::gprBudget() const {
  switch (kind) {
  case TargetKind::RiscV: return 26;
  case TargetKind::Arm: return 12;
  case TargetKind::Generic: return 16;
  }
  return 16;
}
