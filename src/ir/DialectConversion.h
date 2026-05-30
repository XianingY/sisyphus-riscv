#ifndef SISY_IR_DIALECT_CONVERSION_H
#define SISY_IR_DIALECT_CONVERSION_H

#include "OpDescriptor.h"

#include <iosfwd>
#include <set>
#include <string>
#include <vector>

namespace sys::ir {

class TypeConverter {
  std::set<std::string> identityDialects;

public:
  void addIdentityDialect(const std::string &dialect) {
    identityDialects.insert(dialect);
  }
  bool isIdentityLegal(const std::string &dialect) const {
    return identityDialects.count(dialect) != 0;
  }
};

class ConversionTarget {
  std::set<std::string> legalDialects;
  std::set<std::string> legalOps;

public:
  void addLegalDialect(const std::string &dialect) {
    legalDialects.insert(dialect);
  }
  void addLegalOp(const std::string &dialect, const std::string &name) {
    legalOps.insert(dialect + "." + name);
  }
  bool isLegal(const OpDescriptor &op) const;
  void dump(std::ostream &os) const;
};

struct ConversionLegalitySummary {
  int legal = 0;
  int illegal = 0;
  std::vector<std::string> illegalOps;
};

class DialectConversionDriver {
public:
  static ConversionLegalitySummary analyzeDescriptors(const ConversionTarget &target);
  static ConversionTarget standardScalarTarget();
  static void dumpStandardScalarLegality(std::ostream &os);
};

} // namespace sys::ir

#endif
