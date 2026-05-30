#include "DialectConversion.h"

#include "../codegen/Ops.h"

#include <functional>
#include <iostream>

namespace sys::ir {

bool ConversionTarget::isLegal(const OpDescriptor &op) const {
  return legalDialects.count(op.dialect) ||
         legalOps.count(std::string(op.dialect) + "." + op.name);
}

void ConversionTarget::dump(std::ostream &os) const {
  for (const auto &dialect : legalDialects)
    os << "[dialect-conversion] legal-dialect " << dialect << "\n";
  for (const auto &op : legalOps)
    os << "[dialect-conversion] legal-op " << op << "\n";
}

ConversionLegalitySummary
DialectConversionDriver::analyzeDescriptors(const ConversionTarget &target) {
  ConversionLegalitySummary summary;
  for (const auto &op : OpDescriptorTable::all()) {
    if (target.isLegal(op)) {
      summary.legal++;
      continue;
    }
    summary.illegal++;
    summary.illegalOps.push_back(std::string(op.dialect) + "." + op.name);
  }
  return summary;
}

ConversionTarget DialectConversionDriver::standardScalarTarget() {
  ConversionTarget target;
  target.addLegalDialect("arith");
  target.addLegalDialect("scf");
  target.addLegalDialect("memref");
  return target;
}

void DialectConversionDriver::dumpStandardScalarLegality(std::ostream &os) {
  auto target = standardScalarTarget();
  target.dump(os);
  auto summary = analyzeDescriptors(target);
  os << "[dialect-conversion] legal-count=" << summary.legal
     << " illegal-count=" << summary.illegal << "\n";
  for (const auto &op : summary.illegalOps)
    os << "[dialect-conversion] illegal-op " << op << "\n";
}

ConversionStats DialectConversionDriver::runLegacyDryRun(ModuleOp *module,
                                                         std::ostream &os) {
  ConversionStats stats;
  auto target = standardScalarTarget();
  if (!module) {
    os << "[dialect-conversion] target=legacy visited=0 legal=0 converted=0 failed=0 rollbacks=0\n";
    return stats;
  }

  std::function<void(Op*)> walk = [&](Op *op) {
    if (!op)
      return;
    stats.visited++;
    auto *operation = Operation::fromLegacy(op);
    auto *descriptor = operation->getDescriptor();
    if (descriptor && target.isLegal(*descriptor)) {
      stats.legal++;
    } else if (operation->getDialect() == "legacy") {
      // Already in legacy form. The first bridge stage treats this as an
      // identity conversion; real replacement patterns can be added later.
      stats.converted++;
    } else {
      stats.failed++;
    }
    for (auto *region : op->getRegions())
      for (auto *bb : region->getBlocks())
        for (auto *child : bb->getOps())
          walk(child);
  };
  walk(module);
  os << "[dialect-conversion] target=legacy"
     << " visited=" << stats.visited
     << " legal=" << stats.legal
     << " converted=" << stats.converted
     << " failed=" << stats.failed
     << " rollbacks=" << stats.rollbacks << "\n";
  return stats;
}

ConversionStats DialectConversionDriver::runRollbackSelfTest(std::ostream &os) {
  ConversionStats stats;
  stats.visited = 1;
  ConversionPatternRewriter rewriter;
  rewriter.signalFailure();
  if (rewriter.failed()) {
    stats.failed = 1;
    stats.rollbacks = 1;
  }
  os << "[dialect-conversion] target=rollback-test"
     << " visited=" << stats.visited
     << " legal=0 converted=0 failed=" << stats.failed
     << " rollbacks=" << stats.rollbacks << "\n";
  return stats;
}

} // namespace sys::ir
