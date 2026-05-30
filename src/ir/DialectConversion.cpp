#include "DialectConversion.h"

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

} // namespace sys::ir
