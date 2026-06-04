#include "SelfMLIRInternal.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace sys::mlir {

int64_t parseIntegerAttr(Attribute attr) {
  if (!attr)
    return 0;
  std::string text = attr.str();
  size_t pos = 0;
  while (pos < text.size() && text[pos] == '"')
    pos++;
  size_t end = pos;
  if (end < text.size() && (text[end] == '-' || text[end] == '+'))
    end++;
  while (end < text.size() && std::isdigit((unsigned char) text[end]))
    end++;
  if (end == pos)
    return 0;
  return std::stoll(text.substr(pos, end - pos));
}

std::string symbolAttr(Attribute attr, const std::string &fallback) {
  if (!attr)
    return fallback;
  std::string text = attr.str();
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
    return text.substr(1, text.size() - 2);
  return text.empty() ? fallback : text;
}


bool isFloatType(Type type) {
  return type.kind() == TypeKind::Float || type.str() == "f32";
}

bool hasScalarHome(Type type) {
  return type.kind() == TypeKind::Integer || type.kind() == TypeKind::Index ||
         type.kind() == TypeKind::Float || type.str() == "i32" || type.str() == "f32";
}

bool isMemrefType(Type type) {
  return type.kind() == TypeKind::MemRef || type.str().rfind("memref<", 0) == 0;
}

bool hasValueHome(Type type) {
  return hasScalarHome(type) || isMemrefType(type);
}

std::string valueKey(Value value) {
  return value.identityKey();
}

bool envEnabled(const char *name, bool defaultValue) {
  if (const char *value = std::getenv(name))
    return std::string(value) != "0";
  return defaultValue;
}

bool positivePowerOfTwoShift(int64_t value, int &shift) {
  if (value <= 0 || (value & (value - 1)) != 0)
    return false;
  shift = 0;
  while ((int64_t(1) << shift) != value && shift < 31)
    shift++;
  return shift < 31;
}

bool constantIntegerValue(Value value, int64_t &out) {
  if (isFloatType(value.type()))
    return false;
  auto *op = value.getDefiningOp();
  if (!op)
    return false;
  if (op->name() != "arith.constant" && op->name() != "rv_machine.li" &&
      op->name() != "arm_machine.mov")
    return false;
  if (!op->attr("value"))
    return false;
  out = parseIntegerAttr(op->attr("value"));
  return true;
}

MemrefInfo parseMemrefInfo(Type type) {
  MemrefInfo info;
  std::string text = type.str();
  auto begin = text.find("memref<");
  if (begin == std::string::npos)
    return info;
  begin += 7;
  auto end = text.find('>', begin);
  if (end == std::string::npos)
    end = text.size();
  std::string shape = text.substr(begin, end - begin);
  auto elemPos = shape.find("xi32");
  if (elemPos == std::string::npos) {
    elemPos = shape.find("xf32");
  }
  if (elemPos == std::string::npos)
    return info;
  shape = shape.substr(0, elemPos);
  std::size_t pos = 0;
  while (pos < shape.size()) {
    std::size_t next = shape.find('x', pos);
    std::string part = shape.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
    if (!part.empty() && part[0] != '?') {
      try {
        info.shape.push_back(std::stoll(part));
      } catch (...) {
        info.shape.push_back(-1);
      }
    }
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }
  info.valid = true;
  return info;
}

int64_t memrefAllocationBytes(Type type) {
  MemrefInfo info = parseMemrefInfo(type);
  if (!info.valid)
    return 4;
  int64_t elems = 1;
  for (int64_t dim : info.shape) {
    if (dim > 0)
      elems *= dim;
  }
  return std::max<int64_t>(4, elems * info.elemBytes);
}

bool isScalarWordMemref(Type type) {
  MemrefInfo info = parseMemrefInfo(type);
  return info.valid && info.shape.size() == 1 && info.shape[0] == 1 &&
         (type.str().find("xi32") != std::string::npos ||
          type.str().find("xf32") != std::string::npos);
}

bool isI32Like(Type type) {
  return type.kind() == TypeKind::Integer || type.kind() == TypeKind::Index ||
         type.str() == "i32" || type.str() == "index";
}

std::string sanitizeLabel(std::string label) {
  if (label.empty())
    label = "anon";
  for (char &c : label) {
    if (!std::isalnum((unsigned char) c) && c != '_' && c != '.')
      c = '_';
  }
  if (std::isdigit((unsigned char) label[0]))
    label = "_" + label;
  return label;
}

int operationIndexInBlock(Block &block, Operation *needle) {
  for (size_t i = 0; i < block.ops().size(); i++)
    if (block.ops()[i].get() == needle)
      return (int) i;
  return -1;
}


} // namespace sys::mlir
