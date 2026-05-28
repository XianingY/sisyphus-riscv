#include "MIRLegalizer.h"

#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace sys;
using namespace sys::backend::shared;

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

void bump(LegalizeStats &stats, LegalizeAction action) {
  switch (action) {
  case LegalizeAction::Legal: stats.legal++; break;
  case LegalizeAction::Promote: stats.promoted++; break;
  case LegalizeAction::Split: stats.split++; break;
  case LegalizeAction::Expand: stats.expanded++; break;
  case LegalizeAction::Custom: stats.custom++; break;
  case LegalizeAction::Illegal: stats.illegal++; break;
  }
}

void collectOps(Region *region, std::vector<Op*> &ops) {
  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      ops.push_back(op);
      for (auto child : op->getRegions())
        collectOps(child, ops);
    }
  }
}

} // namespace

std::map<std::string, int> LegalizeStats::toMap() const {
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

bool sys::backend::shared::strictMIRLegalizer() {
  return envEnabled("SISY_MIR_LEGALIZER_STRICT", false);
}

LegalizeStats sys::backend::shared::verifyAndCount(
    ModuleOp *module,
    const TargetLegalizerInfo &target,
    bool emitDiagnostics) {
  LegalizeStats stats;
  std::vector<Op*> ops;
  collectOps(module->getRegion(), ops);
  for (auto op : ops) {
    if (isa<ModuleOp>(op) || isa<FuncOp>(op) || isa<GlobalOp>(op))
      continue;
    std::string reason;
    auto action = target.classify(op, reason);
    bump(stats, action);
    if (action == LegalizeAction::Illegal) {
      stats.verifierErrors++;
      if (emitDiagnostics) {
        std::cerr << "[" << target.name() << "-legalize] illegal op "
                  << op->getName();
        if (!reason.empty())
          std::cerr << ": " << reason;
        std::cerr << "\n";
      }
    }
  }
  return stats;
}
