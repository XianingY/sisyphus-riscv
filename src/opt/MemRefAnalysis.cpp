#include "MemRefAnalysis.h"

#include <algorithm>
#include <limits>

using namespace sys;

namespace {

std::vector<int64_t> dimsOf(Op *op) {
  std::vector<int64_t> dims;
  if (op && op->has<DimensionAttr>()) {
    for (int dim : DIM(op))
      dims.push_back(dim);
  }
  return dims;
}

std::vector<int64_t> rowMajorStrides(const std::vector<int64_t> &shape,
                                     std::size_t elemSize) {
  std::vector<int64_t> strides(shape.size(), (int64_t) elemSize);
  int64_t stride = (int64_t) elemSize;
  for (int i = (int) shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    if (shape[i] > 0)
      stride *= shape[i];
  }
  return strides;
}

bool intervalNoOverlap(int64_t lhsOff, std::size_t lhsSize,
                       int64_t rhsOff, std::size_t rhsSize) {
  int64_t lhsEnd = lhsOff + (int64_t) lhsSize;
  int64_t rhsEnd = rhsOff + (int64_t) rhsSize;
  return lhsEnd <= rhsOff || rhsEnd <= lhsOff;
}

int64_t firstKnownOffset(const std::vector<int> &offsets, bool &known) {
  known = false;
  if (offsets.empty())
    return 0;
  int first = offsets.front();
  if (first < 0)
    return 0;
  for (int offset : offsets) {
    if (offset < 0 || offset != first)
      return 0;
  }
  known = true;
  return first;
}

} // namespace

void MemRefAliasAnalysis::recordBase(Op *op) {
  if (!op || bases.count(op))
    return;

  MemRefType type;
  type.shape = dimsOf(op);
  type.elemType = op->has<FPAttr>() ? Value::f32 : Value::i32;
  type.elemSize = layout.sizeOf(type.elemType);
  type.strides = rowMajorStrides(type.shape, type.elemSize);
  type.storageSize = type.elemSize;
  for (auto dim : type.shape) {
    if (dim > 0 && type.storageSize < std::numeric_limits<std::size_t>::max() / (std::size_t) dim)
      type.storageSize *= (std::size_t) dim;
  }
  if (isa<GlobalOp>(op)) {
    type.baseKind = MemRefBaseKind::Global;
    type.readonly = op->has<ConstAttr>();
  } else if (isa<AllocaOp>(op)) {
    type.baseKind = MemRefBaseKind::Local;
  } else if (isa<GetArgOp>(op)) {
    type.baseKind = MemRefBaseKind::Param;
  }
  bases[op] = type;
}

void MemRefAliasAnalysis::build() {
  bases.clear();
  stats = {};
  for (auto global : module->findAll<GlobalOp>())
    recordBase(global);
  for (auto alloca : module->findAll<AllocaOp>())
    recordBase(alloca);
  for (auto arg : module->findAll<GetArgOp>())
    recordBase(arg);
  stats.bases = (int) bases.size();
}

const MemRefType *MemRefAliasAnalysis::lookupBase(Op *op) const {
  auto it = bases.find(op);
  if (it == bases.end())
    return nullptr;
  return &it->second;
}

AliasRelation MemRefAliasAnalysis::alias(Op *lhsAddr, std::size_t lhsSize,
                                         Op *rhsAddr, std::size_t rhsSize) const {
  stats.queries++;
  auto record = [&](AliasRelation relation) {
    switch (relation) {
    case AliasRelation::NoAlias: stats.noAlias++; break;
    case AliasRelation::MustAlias: stats.mustAlias++; break;
    case AliasRelation::PartialAlias: stats.partialAlias++; break;
    case AliasRelation::Unknown: stats.unknown++; break;
    }
    return relation;
  };

  auto *lhsAlias = lhsAddr ? lhsAddr->find<AliasAttr>() : nullptr;
  auto *rhsAlias = rhsAddr ? rhsAddr->find<AliasAttr>() : nullptr;
  if (!lhsAlias || !rhsAlias || lhsAlias->unknown || rhsAlias->unknown)
    return record(AliasRelation::Unknown);

  bool sawSameBase = false;
  bool allDisjoint = true;
  bool exactSingle = lhsAlias->location.size() == 1 && rhsAlias->location.size() == 1;
  for (const auto &[lhsBase, lhsOffsets] : lhsAlias->location) {
    for (const auto &[rhsBase, rhsOffsets] : rhsAlias->location) {
      if (lhsBase != rhsBase)
        continue;
      sawSameBase = true;
      bool lhsKnown = false;
      bool rhsKnown = false;
      int64_t lhsOff = firstKnownOffset(lhsOffsets, lhsKnown);
      int64_t rhsOff = firstKnownOffset(rhsOffsets, rhsKnown);
      if (!lhsKnown || !rhsKnown)
        return record(AliasRelation::Unknown);
      if (!intervalNoOverlap(lhsOff, lhsSize, rhsOff, rhsSize))
        allDisjoint = false;
      if (exactSingle && lhsOff == rhsOff && lhsSize == rhsSize)
        return record(AliasRelation::MustAlias);
    }
  }
  if (!sawSameBase)
    return record(AliasRelation::NoAlias);
  if (allDisjoint)
    return record(AliasRelation::NoAlias);
  return record(AliasRelation::PartialAlias);
}

const char *sys::aliasRelationName(AliasRelation relation) {
  switch (relation) {
  case AliasRelation::NoAlias: return "no-alias";
  case AliasRelation::MustAlias: return "must-alias";
  case AliasRelation::PartialAlias: return "partial-alias";
  case AliasRelation::Unknown: return "unknown";
  }
  return "unknown";
}
