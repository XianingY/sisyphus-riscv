#ifndef SISY_SELF_MLIR_INTERNAL_H
#define SISY_SELF_MLIR_INTERNAL_H

#include "SelfMLIR.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sys::mlir {

int64_t parseIntegerAttr(Attribute attr);
std::string symbolAttr(Attribute attr, const std::string &fallback = "");
bool isFloatType(Type type);
bool hasScalarHome(Type type);
bool isMemrefType(Type type);
bool hasValueHome(Type type);
std::string valueKey(Value value);
bool envEnabled(const char *name, bool defaultValue);
bool positivePowerOfTwoShift(int64_t value, int &shift);
bool constantIntegerValue(Value value, int64_t &out);

struct MemrefInfo {
  std::vector<int64_t> shape;
  int elemBytes = 4;
  bool valid = false;
};

MemrefInfo parseMemrefInfo(Type type);
int64_t memrefAllocationBytes(Type type);
bool isScalarWordMemref(Type type);
bool isI32Like(Type type);
std::string sanitizeLabel(std::string label);
int operationIndexInBlock(Block &block, Operation *needle);

} // namespace sys::mlir

#endif
