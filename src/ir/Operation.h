#ifndef SISY_IR_OPERATION_H
#define SISY_IR_OPERATION_H

#include "IRContext.h"
#include "OpDescriptor.h"
#include "../codegen/OpBase.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace sys {
class ModuleOp;
}

namespace sys::ir {

Type legacyTypeToIRType(Value::Type type);

class Operation {
  Op *legacy = nullptr;
  const OpDescriptor *descriptor = nullptr;
  std::string dialect = "legacy";
  std::string name = "unknown";
  std::vector<Value> operands;
  std::vector<Attr*> attrs;
  std::vector<Region*> regions;
  std::vector<Type> resultTypes;
  Attribute location;

  explicit Operation(Op *legacy);

public:
  static Operation *fromLegacy(Op *op);

  Op *getLegacyOp() const { return legacy; }
  const OpDescriptor *getDescriptor() const { return descriptor; }
  const std::string &getDialect() const { return dialect; }
  const std::string &getName() const { return name; }
  std::string getQualifiedName() const { return dialect + "." + name; }

  int getOperandCount() const { return (int) operands.size(); }
  int getResultCount() const { return (int) resultTypes.size(); }
  int getAttrCount() const { return (int) attrs.size(); }
  int getRegionCount() const { return (int) regions.size(); }

  Value getOperand(int i) const { return operands[i]; }
  Type getResultType(int i = 0) const { return resultTypes[i]; }
  Attribute getLocation() const { return location; }

  bool hasTrait(const std::string &trait) const;
  bool implementsInterface(const std::string &interfaceName) const;
  bool isPure() const;
  bool hasMemoryEffects() const;
  bool isTerminator() const;
  bool isRegionBranch() const;

  void syncFromLegacy();
  bool verifyBridge(std::string *error = nullptr) const;
  void print(std::ostream &os) const;

  static void dumpModule(ModuleOp *module, std::ostream &os);
  static bool verifyModuleBridge(ModuleOp *module, std::ostream &os);
};

} // namespace sys::ir

#endif
