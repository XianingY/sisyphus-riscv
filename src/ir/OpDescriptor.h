#ifndef SISY_IR_OP_DESCRIPTOR_H
#define SISY_IR_OP_DESCRIPTOR_H

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace sys::ir {

struct OpDescriptor {
  const char *dialect;
  const char *name;
  int operandCount;
  int resultCount;
  int attrCount;
  const char *verifyHook;
  const char *foldHook;
  const char *const *traits;
  std::size_t traitCount;
  const char *const *operandNames;
  std::size_t operandNameCount;
  const char *const *resultNames;
  std::size_t resultNameCount;
  const char *const *attrNames;
  std::size_t attrNameCount;
  const char *cppClass;
  const char *typeFormat;
  const char *assemblyFormat;
  const char *const *interfaces;
  std::size_t interfaceCount;
};

class OpDescriptorTable {
public:
  static const std::vector<OpDescriptor> &all();
  static const OpDescriptor *find(const std::string &dialect,
                                  const std::string &name);
  static bool verify(std::string *error = nullptr);
  static void dump(std::ostream &os);
};

} // namespace sys::ir

#endif
