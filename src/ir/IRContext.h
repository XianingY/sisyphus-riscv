#ifndef SISY_IR_CONTEXT_H
#define SISY_IR_CONTEXT_H

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sys::ir {

struct TypeStorage;
struct AttributeStorage;

class Type {
  const TypeStorage *storage = nullptr;
public:
  explicit Type(const TypeStorage *storage = nullptr): storage(storage) {}
  bool operator==(Type other) const { return storage == other.storage; }
  bool operator!=(Type other) const { return storage != other.storage; }
  explicit operator bool() const { return storage != nullptr; }
  std::string str() const;
  const TypeStorage *impl() const { return storage; }
};

class Attribute {
  const AttributeStorage *storage = nullptr;
public:
  explicit Attribute(const AttributeStorage *storage = nullptr): storage(storage) {}
  bool operator==(Attribute other) const { return storage == other.storage; }
  bool operator!=(Attribute other) const { return storage != other.storage; }
  explicit operator bool() const { return storage != nullptr; }
  std::string str() const;
  const AttributeStorage *impl() const { return storage; }
};

struct Location {
  std::string file;
  int line = 0;
  int column = 0;
};

class IRContext {
  std::unordered_map<std::string, std::unique_ptr<TypeStorage>> types;
  std::unordered_map<std::string, std::unique_ptr<AttributeStorage>> attrs;

  Type internType(std::string key, std::string text);
  Attribute internAttr(std::string key, std::string text);

public:
  static IRContext &global();

  Type getIntegerType(int width);
  Type getFloatType(int width);
  Type getIndexType();
  Type getMemRefType(const std::vector<int64_t> &shape, Type elemType,
                     const std::vector<int64_t> &strides = {},
                     const std::string &layout = "identity",
                     bool readonly = false);
  Type getVectorType(Type elemType, int64_t lanes, bool scalable);

  Attribute getIntegerAttr(int64_t value, Type type);
  Attribute getStringAttr(const std::string &value);
  Attribute getLocationAttr(const std::string &file, int line, int column);

  std::size_t typeCount() const { return types.size(); }
  std::size_t attrCount() const { return attrs.size(); }
  void dump(std::ostream &os);
};

} // namespace sys::ir

#endif
