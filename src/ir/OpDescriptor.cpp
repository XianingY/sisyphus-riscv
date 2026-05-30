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
    "IsolatedFromAbove",
    "Symbol",
    "FunctionLike",
    "AffineLike",
    "VectorLike",
    "MachineOp",
    "RegisterOp",
  };
  static const std::set<std::string> allowedInterfaces = {
    "PureOpInterface",
    "MemoryEffectOpInterface",
    "RegionBranchOpInterface",
    "TerminatorOpInterface",
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
    if (op.operandCount >= 0 && op.operandNameCount != (std::size_t) op.operandCount) {
      if (error)
        *error = std::string(op.dialect) + "." + op.name +
                 " has mismatched operand names";
      return false;
    }
    if (op.resultNameCount != (std::size_t) op.resultCount ||
        op.attrNameCount != (std::size_t) op.attrCount) {
      if (error)
        *error = std::string(op.dialect) + "." + op.name +
                 " has mismatched result/attr names";
      return false;
    }
    if (!op.cppClass || !op.cppClass[0] || !op.typeFormat ||
        !op.typeFormat[0] || !op.assemblyFormat ||
        !op.assemblyFormat[0]) {
      if (error)
        *error = std::string(op.dialect) + "." + op.name +
                 " has empty generated class/type/assembly metadata";
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
    for (std::size_t i = 0; i < op.interfaceCount; ++i) {
      if (!op.interfaces || !op.interfaces[i] ||
          !allowedInterfaces.count(op.interfaces[i])) {
        if (error)
          *error = "unknown interface on " + key;
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
       << " cppClass=" << op.cppClass
       << " typeFormat=" << op.typeFormat
       << " assemblyFormat=" << op.assemblyFormat
       << " traits=";
    for (std::size_t i = 0; i < op.traitCount; ++i) {
      if (i)
        os << ",";
      os << op.traits[i];
    }
    os << " operands=[";
    for (std::size_t i = 0; i < op.operandNameCount; ++i) {
      if (i)
        os << ",";
      os << op.operandNames[i];
    }
    os << "] results=[";
    for (std::size_t i = 0; i < op.resultNameCount; ++i) {
      if (i)
        os << ",";
      os << op.resultNames[i];
    }
    os << "] attrs=[";
    for (std::size_t i = 0; i < op.attrNameCount; ++i) {
      if (i)
        os << ",";
      os << op.attrNames[i];
    }
    os << "]";
    os << " interfaces=[";
    for (std::size_t i = 0; i < op.interfaceCount; ++i) {
      if (i)
        os << ",";
      os << op.interfaces[i];
    }
    os << "]";
    os << "\n";
  }
}

} // namespace sys::ir
