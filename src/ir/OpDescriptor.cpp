#include "OpDescriptor.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>

#include "GeneratedOpDescriptors.inc"

namespace sys::ir {

const std::vector<OpDescriptor> &OpDescriptorTable::all() {
  static const std::vector<OpDescriptor> table(
      std::begin(kGeneratedOpDescriptors), std::end(kGeneratedOpDescriptors));
  return table;
}

const OpDescriptor *OpDescriptorTable::find(const std::string &dialect,
                                            const std::string &name) {
  for (const auto &op : all())
    if (op.dialect == dialect && op.name == name)
      return &op;
  return nullptr;
}

bool OpDescriptorTable::verify(std::string *error) {
  static const std::set<std::string> allowedTraits = {
    "Pure",
    "MemoryEffect",
    "BranchLike",
    "LoopLike",
    "SameOperandsAndResultType",
    "Terminator",
    "HasRegion",
    "NoSideEffect",
    "Commutative",
  };

  std::set<std::string> seen;
  std::string previous;
  for (const auto &op : all()) {
    if (!op.dialect || !op.dialect[0] || !op.name || !op.name[0]) {
      if (error)
        *error = "op descriptor has empty dialect/name";
      return false;
    }
    if (!op.verifyHook || !op.verifyHook[0] || !op.foldHook || !op.foldHook[0]) {
      if (error)
        *error = std::string(op.dialect) + "." + op.name +
                 " has empty verify/fold hook";
      return false;
    }
    if (op.operandCount < -1 || op.resultCount < 0 || op.attrCount < 0) {
      if (error)
        *error = std::string(op.dialect) + "." + op.name +
                 " has invalid arity";
      return false;
    }
    std::string key = std::string(op.dialect) + "." + op.name;
    if (!previous.empty() && key < previous) {
      if (error)
        *error = "op descriptor table is not sorted: " + previous + " before " + key;
      return false;
    }
    previous = key;
    if (!seen.insert(key).second) {
      if (error)
        *error = "duplicate op descriptor: " + key;
      return false;
    }
    for (std::size_t i = 0; i < op.traitCount; ++i) {
      if (!op.traits || !op.traits[i] || !allowedTraits.count(op.traits[i])) {
        if (error)
          *error = "unknown trait on " + key;
        return false;
      }
    }
  }
  return true;
}

void OpDescriptorTable::dump(std::ostream &os) {
  for (const auto &op : all()) {
    os << op.dialect << "." << op.name
       << " operands=" << op.operandCount
       << " results=" << op.resultCount
       << " attrs=" << op.attrCount
       << " verify=" << op.verifyHook
       << " fold=" << op.foldHook
       << " traits=";
    for (std::size_t i = 0; i < op.traitCount; ++i) {
      if (i)
        os << ",";
      os << op.traits[i];
    }
    os << "\n";
  }
}

} // namespace sys::ir
