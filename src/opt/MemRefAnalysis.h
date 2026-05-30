#ifndef SISY_MEMREF_ANALYSIS_H
#define SISY_MEMREF_ANALYSIS_H

#include "DataLayout.h"
#include "../codegen/Attrs.h"
#include "../codegen/Ops.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace sys {

enum class MemRefBaseKind {
  Unknown,
  Global,
  Local,
  Param,
};

struct MemRefType {
  std::vector<int64_t> shape;
  Value::Type elemType = Value::i32;
  std::vector<int64_t> strides;
  std::string layout = "identity";
  MemRefBaseKind baseKind = MemRefBaseKind::Unknown;
  bool readonly = false;
  std::size_t elemSize = 4;
  std::size_t storageSize = 0;
};

enum class AliasRelation {
  NoAlias,
  MustAlias,
  PartialAlias,
  Unknown,
};

struct MemRefAliasStats {
  int bases = 0;
  int queries = 0;
  int noAlias = 0;
  int mustAlias = 0;
  int partialAlias = 0;
  int unknown = 0;
};

class MemRefAliasAnalysis {
  ModuleOp *module;
  const DataLayout &layout;
  std::map<Op*, MemRefType> bases;
  mutable MemRefAliasStats stats;

  void recordBase(Op *op);

public:
  MemRefAliasAnalysis(ModuleOp *module, const DataLayout &layout):
      module(module), layout(layout) {}

  void build();
  const MemRefType *lookupBase(Op *op) const;
  AliasRelation alias(Op *lhsAddr, std::size_t lhsSize,
                      Op *rhsAddr, std::size_t rhsSize) const;
  const MemRefAliasStats &getStats() const { return stats; }
};

const char *aliasRelationName(AliasRelation relation);

} // namespace sys

#endif
