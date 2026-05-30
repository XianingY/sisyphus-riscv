#include "BlockArgumentBridge.h"

#include "../codegen/Attrs.h"

#include <iostream>
#include <set>

namespace sys::ir {

namespace {

bool phiMatchesPredecessors(Op *phi, BasicBlock *bb) {
  if (!phi || !isa<PhiOp>(phi))
    return false;
  if (phi->getOperandCount() != (int) bb->preds.size())
    return false;
  std::set<BasicBlock*> incoming;
  const auto &attrs = phi->getAttrs();
  if (attrs.size() != phi->getOperands().size())
    return false;
  for (auto *attr : attrs) {
    if (!isa<FromAttr>(attr))
      return false;
    auto *from = FROM(attr);
    if (!bb->preds.count(from))
      return false;
    incoming.insert(from);
  }
  return incoming.size() == bb->preds.size();
}

void refreshPreds(ModuleOp *module) {
  for (auto *func : module->findAll<FuncOp>())
    if (func->getRegionCount())
      func->getRegion()->updatePreds();
}

} // namespace

BlockArgumentBridgeStats PhiToBlockArgumentBridge::materialize(ModuleOp *module) {
  BlockArgumentBridgeStats stats;
  if (!module)
    return stats;
  refreshPreds(module);
  for (auto *func : module->findAll<FuncOp>()) {
    if (!func->getRegionCount())
      continue;
    for (auto *bb : func->getRegion()->getBlocks()) {
      int existing = (int) bb->getArguments().size();
      for (auto *phi : bb->getPhis()) {
        stats.phiCandidates++;
        if (phi->getAttrs().size() != phi->getOperands().size()) {
          stats.rejectedMalformedPhi++;
          continue;
        }
        if (!phiMatchesPredecessors(phi, bb)) {
          stats.rejectedPredMismatch++;
          continue;
        }
        auto &arg = bb->addArgument(phi->getResultType(),
                                    "phi" + std::to_string(existing++));
        (void) Value(&arg);
        stats.argumentsCreated++;
      }
    }
  }
  return stats;
}

BlockArgumentBridgeStats PhiToBlockArgumentBridge::lower(ModuleOp *module) {
  BlockArgumentBridgeStats stats;
  if (!module)
    return stats;
  for (auto *func : module->findAll<FuncOp>()) {
    if (!func->getRegionCount())
      continue;
    for (auto *bb : func->getRegion()->getBlocks()) {
      stats.loweredArguments += (int) bb->getArguments().size();
      bb->clearArguments();
    }
  }
  return stats;
}

bool PhiToBlockArgumentBridge::roundTrip(ModuleOp *module, std::ostream &os) {
  auto materialized = materialize(module);
  auto lowered = lower(module);
  os << "[block-arg-bridge] phi-candidates=" << materialized.phiCandidates
     << " arguments-created=" << materialized.argumentsCreated
     << " reject-pred-mismatch=" << materialized.rejectedPredMismatch
     << " reject-malformed-phi=" << materialized.rejectedMalformedPhi
     << " lowered-arguments=" << lowered.loweredArguments << "\n";
  return materialized.argumentsCreated == lowered.loweredArguments;
}

} // namespace sys::ir
