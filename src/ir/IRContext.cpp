#include "IRContext.h"

#include <sstream>

namespace sys::ir {

struct TypeStorage {
  std::string key;
  std::string text;
};

struct AttributeStorage {
  std::string key;
  std::string text;
};

std::string Type::str() const {
  return storage ? storage->text : "<null-type>";
}

std::string Attribute::str() const {
  return storage ? storage->text : "<null-attr>";
}

IRContext &IRContext::global() {
  static IRContext context;
  return context;
}

Type IRContext::internType(std::string key, std::string text) {
  auto it = types.find(key);
  if (it != types.end())
    return Type(it->second.get());
  auto storage = std::make_unique<TypeStorage>();
  storage->key = key;
  storage->text = text;
  auto *ptr = storage.get();
  types.emplace(std::move(key), std::move(storage));
  return Type(ptr);
}

Attribute IRContext::internAttr(std::string key, std::string text) {
  auto it = attrs.find(key);
  if (it != attrs.end())
    return Attribute(it->second.get());
  auto storage = std::make_unique<AttributeStorage>();
  storage->key = key;
  storage->text = text;
  auto *ptr = storage.get();
  attrs.emplace(std::move(key), std::move(storage));
  return Attribute(ptr);
}

Type IRContext::getIntegerType(int width) {
  return internType("i" + std::to_string(width), "i" + std::to_string(width));
}

Type IRContext::getFloatType(int width) {
  return internType("f" + std::to_string(width), "f" + std::to_string(width));
}

Type IRContext::getIndexType() {
  return internType("index", "index");
}

Type IRContext::getMemRefType(const std::vector<int64_t> &shape, Type elemType,
                              const std::vector<int64_t> &strides,
                              const std::string &layout, bool readonly) {
  std::ostringstream key;
  key << "memref<";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      key << "x";
    key << (shape[i] < 0 ? "?" : std::to_string(shape[i]));
  }
  key << "x" << elemType.str();
  if (!strides.empty()) {
    key << ",strides=[";
    for (size_t i = 0; i < strides.size(); ++i) {
      if (i)
        key << ",";
      key << strides[i];
    }
    key << "]";
  }
  key << ",layout=" << layout;
  if (readonly)
    key << ",readonly";
  key << ">";
  return internType(key.str(), key.str());
}

Type IRContext::getVectorType(Type elemType, int64_t lanes, bool scalable) {
  std::ostringstream text;
  text << "vector<" << (scalable ? "vscale x " : "")
       << lanes << "x" << elemType.str() << ">";
  return internType(text.str(), text.str());
}

Attribute IRContext::getIntegerAttr(int64_t value, Type type) {
  std::string key = "int:" + type.str() + ":" + std::to_string(value);
  return internAttr(key, std::to_string(value) + " : " + type.str());
}

Attribute IRContext::getStringAttr(const std::string &value) {
  return internAttr("str:" + value, "\"" + value + "\"");
}

Attribute IRContext::getLocationAttr(const std::string &file, int line, int column) {
  std::ostringstream text;
  text << "loc(\"" << file << "\":" << line << ":" << column << ")";
  return internAttr("loc:" + file + ":" + std::to_string(line) + ":" +
                        std::to_string(column),
                    text.str());
}

void IRContext::dump(std::ostream &os) {
  auto i32 = getIntegerType(32);
  auto i32Again = getIntegerType(32);
  auto f32 = getFloatType(32);
  auto mem = getMemRefType({4, -1}, i32, {16, 4}, "row-major", true);
  auto vec = getVectorType(f32, 4, true);
  auto loc = getLocationAttr("unknown", 0, 0);
  auto locAgain = getLocationAttr("unknown", 0, 0);

  os << "[ir-context] type " << i32.str()
     << " uniqued=" << (i32 == i32Again ? 1 : 0) << "\n";
  os << "[ir-context] type " << mem.str() << "\n";
  os << "[ir-context] type " << vec.str() << "\n";
  os << "[ir-context] attr " << loc.str()
     << " uniqued=" << (loc == locAgain ? 1 : 0) << "\n";
  os << "[ir-context] counts types=" << typeCount()
     << " attrs=" << attrCount() << "\n";
}

} // namespace sys::ir
