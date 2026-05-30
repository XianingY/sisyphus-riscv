#ifndef SISY_IR_DIALECT_CONVERSION_H
#define SISY_IR_DIALECT_CONVERSION_H

#include "OpDescriptor.h"
#include "Operation.h"

#include <functional>
#include <iosfwd>
#include <set>
#include <string>
#include <utility>
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

struct ConversionStats {
  int visited = 0;
  int legal = 0;
  int converted = 0;
  int failed = 0;
  int rollbacks = 0;
};

class ConversionPatternRewriter {
  bool transactionFailed = false;
public:
  void replaceOp(Operation *, Operation *) {}
  void eraseOp(Operation *) {}
  void signalFailure() { transactionFailed = true; }
  bool failed() const { return transactionFailed; }
};

class ConversionPattern {
  std::string rootName;
public:
  explicit ConversionPattern(std::string rootName): rootName(std::move(rootName)) {}
  virtual ~ConversionPattern() = default;
  const std::string &getRootName() const { return rootName; }
  virtual bool matchAndRewrite(Operation *op,
                               ConversionPatternRewriter &rewriter) const = 0;
};

class DialectConversionDriver {
public:
  static ConversionLegalitySummary analyzeDescriptors(const ConversionTarget &target);
  static ConversionTarget standardScalarTarget();
  static void dumpStandardScalarLegality(std::ostream &os);
  static ConversionStats runLegacyDryRun(ModuleOp *module, std::ostream &os);
  static ConversionStats runRollbackSelfTest(std::ostream &os);
};

} // namespace sys::ir

#endif
