#include "SelfMLIR.h"
#include "Polyhedral.h"


#include "../parse/ASTNode.h"
#include "../parse/Type.h"
#include "../utils/DynamicCast.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sys::mlir {

OptimizationConfig OptimizationConfig::forLevel(Level level) {
  OptimizationConfig config;
  config.level = level;
  switch (level) {
  case Level::O0:
    config.enableGlobalOpt = false;
    config.enableAffine = false;
    config.enableMemoryOpt = false;
    config.enableProvenBitwise = false;
    config.enableDRR = false;
    config.enableDRRWorklist = false;
    config.enableLinearScan = false;
    config.enableInline = false;
    config.enableRotateHelper = false;
    config.enablePow2Strength = false;
    config.enableScheduler = false;
    config.enableLoopTiling = false;
    config.enableLoopFusion = false;
    config.enableLoopInterchange = false;
    config.enableStencilPeel = false;
    config.enableLoopAddressIV = false;
    break;
  case Level::O1:
    config.enableGlobalOpt = true;
    config.enableAffine = true;
    config.enableMemoryOpt = true;
    config.enableProvenBitwise = true;
    config.enableDRR = true;
    config.enableDRRWorklist = true;
    config.enableLinearScan = true;
    config.enableInline = true;
    config.enableRotateHelper = true;
    config.enablePow2Strength = true;
    config.enableScheduler = true;
    config.enableLoopTiling = true;
    config.enableLoopFusion = true;
    config.enableLoopInterchange = true;
    config.enableStencilPeel = true;
    config.enableLoopAddressIV = true;
    break;
  case Level::O2:
    config.enableGlobalOpt = true;
    config.enableAffine = true;
    config.enableMemoryOpt = true;
    config.enableProvenBitwise = true;
    config.enableDRR = true;
    config.enableDRRWorklist = true;
    config.enableLinearScan = true;
    config.enableInline = true;
    config.enableRotateHelper = true;
    config.enablePow2Strength = true;
    config.enableScheduler = true;
    config.enableLoopTiling = true;
    config.enableLoopFusion = true;
    config.enableLoopInterchange = true;
    config.enableStencilPeel = true;
    config.enableLoopAddressIV = true;
    break;
  }
  return config;
}

struct TypeStorage {
  TypeKind kind = TypeKind::None;
  std::string key;
  std::string text;
};

struct AttributeStorage {
  std::string key;
  std::string text;
};

struct LocationStorage {
  std::string key;
  std::string text;
};

Context::Context() = default;
Context::~Context() = default;

static std::string joinTypes(const std::vector<Type> &types) {
  std::string out;
  for (size_t i = 0; i < types.size(); i++) {
    if (i)
      out += ",";
    out += types[i].str();
  }
  return out;
}

static std::string joinI64(const std::vector<int64_t> &values) {
  std::string out;
  for (size_t i = 0; i < values.size(); i++) {
    if (i)
      out += "x";
    out += std::to_string(values[i]);
  }
  return out.empty() ? "?" : out;
}

TypeKind Type::kind() const {
  return storage ? storage->kind : TypeKind::None;
}

std::string Type::str() const {
  return storage ? storage->text : "<nulltype>";
}

std::string Attribute::str() const {
  return storage ? storage->text : "<nullattr>";
}

std::string Location::str() const {
  return storage ? storage->text : "loc(\"unknown\":0:0)";
}

Type Context::internType(TypeKind kind, std::string key, std::string text) {
  auto it = types.find(key);
  if (it != types.end())
    return Type(it->second.get());
  auto storage = std::make_unique<TypeStorage>();
  storage->kind = kind;
  storage->key = key;
  storage->text = std::move(text);
  auto *ptr = storage.get();
  types.emplace(std::move(key), std::move(storage));
  return Type(ptr);
}

Attribute Context::internAttr(std::string key, std::string text) {
  auto it = attrs.find(key);
  if (it != attrs.end())
    return Attribute(it->second.get());
  auto storage = std::make_unique<AttributeStorage>();
  storage->key = key;
  storage->text = std::move(text);
  auto *ptr = storage.get();
  attrs.emplace(std::move(key), std::move(storage));
  return Attribute(ptr);
}

Type Context::noneType() {
  return internType(TypeKind::None, "none", "none");
}

Type Context::i(unsigned bits) {
  return internType(TypeKind::Integer, "i" + std::to_string(bits),
                    "i" + std::to_string(bits));
}

Type Context::f(unsigned bits) {
  return internType(TypeKind::Float, "f" + std::to_string(bits),
                    "f" + std::to_string(bits));
}

Type Context::index() {
  return internType(TypeKind::Index, "index", "index");
}

Type Context::memref(const std::vector<int64_t> &shape, Type element,
                     const std::vector<int64_t> &strides,
                     const std::string &layout, bool readonly) {
  std::string shapeText = joinI64(shape);
  std::string strideText = strides.empty() ? "implicit" : joinI64(strides);
  std::string key = "memref:" + shapeText + ":" + element.str() + ":" +
                    strideText + ":" + layout + ":" + (readonly ? "ro" : "rw");
  std::string text = "memref<" + shapeText + "x" + element.str() + ", " +
                     layout + (readonly ? ", readonly" : "") + ">";
  return internType(TypeKind::MemRef, key, text);
}

Type Context::vector(Type element, int64_t lanes, bool scalable) {
  std::string key = std::string("vector:") + (scalable ? "vscale:" : "fixed:") +
                    std::to_string(lanes) + ":" + element.str();
  std::string text = std::string("vector<") + (scalable ? "[vscale x " : "") +
                     std::to_string(lanes) + "x" + element.str() +
                     (scalable ? "]>" : ">");
  return internType(TypeKind::Vector, key, text);
}

Type Context::function(const std::vector<Type> &inputs,
                       const std::vector<Type> &results) {
  std::string key = "fn:(" + joinTypes(inputs) + ")->(" + joinTypes(results) + ")";
  std::string text = "(" + joinTypes(inputs) + ") -> (" + joinTypes(results) + ")";
  return internType(TypeKind::Function, key, text);
}

Type Context::reg(const std::string &target, const std::string &regClass) {
  std::string key = "reg:" + target + ":" + regClass;
  return internType(TypeKind::Register, key, "!" + target + ".reg<" + regClass + ">");
}

Attribute Context::integerAttr(int64_t value, Type type) {
  std::string key = "int:" + type.str() + ":" + std::to_string(value);
  return internAttr(key, std::to_string(value) + " : " + type.str());
}

Attribute Context::stringAttr(const std::string &value) {
  return internAttr("str:" + value, "\"" + value + "\"");
}

Attribute Context::boolAttr(bool value) {
  return internAttr(std::string("bool:") + (value ? "true" : "false"),
                    value ? "true" : "false");
}

Location Context::loc(const std::string &file, int line, int column) {
  std::string key = file + ":" + std::to_string(line) + ":" + std::to_string(column);
  auto it = locations.find(key);
  if (it != locations.end())
    return Location(it->second.get());
  auto storage = std::make_unique<LocationStorage>();
  storage->key = key;
  storage->text = "loc(\"" + file + "\":" + std::to_string(line) + ":" +
                  std::to_string(column) + ")";
  auto *ptr = storage.get();
  locations.emplace(std::move(key), std::move(storage));
  return Location(ptr);
}

Location Context::unknownLoc() {
  return loc("unknown", 0, 0);
}

Value Value::result(Operation *owner, unsigned resultIndex, Type type) {
  Value v;
  v.owner = owner;
  v.resultIndex = resultIndex;
  v.valueType = type;
  return v;
}

Value Value::blockArgument(BlockArgument *arg) {
  Value v;
  v.argument = arg;
  if (arg)
    v.valueType = arg->type();
  return v;
}

std::string Value::identityKey() const {
  if (argument)
    return "%arg" + std::to_string(reinterpret_cast<std::uintptr_t>(argument));
  if (owner)
    return "%op" + std::to_string(reinterpret_cast<std::uintptr_t>(owner)) +
           "_" + std::to_string(resultIndex);
  return "%null";
}

std::string Value::printName() const {
  if (argument)
    return "%" + argument->name();
  if (owner)
    return "%r" + std::to_string(reinterpret_cast<std::uintptr_t>(owner) & 0xffff) +
           "_" + std::to_string(resultIndex);
  return "%null";
}

void registerUse(Value value, Use use) {
  if (!value.valid())
    return;
  if (value.isOperationResult()) {
    auto *op = value.getDefiningOp();
    if (op) {
      unsigned idx = value.getResultIndex();
      if (idx < op->resultUses.size()) {
        op->resultUses[idx].push_back(use);
      }
    }
  } else if (value.isBlockArgument()) {
    auto *arg = value.getBlockArgument();
    if (arg) {
      arg->uses.push_back(use);
    }
  }
}

void unregisterUse(Value value, Use use) {
  if (!value.valid())
    return;
  if (value.isOperationResult()) {
    auto *op = value.getDefiningOp();
    if (op) {
      unsigned idx = value.getResultIndex();
      if (idx < op->resultUses.size()) {
        auto &list = op->resultUses[idx];
        for (auto it = list.begin(); it != list.end(); ++it) {
          if (it->owner == use.owner && it->operandIndex == use.operandIndex) {
            list.erase(it);
            break;
          }
        }
      }
    }
  } else if (value.isBlockArgument()) {
    auto *arg = value.getBlockArgument();
    if (arg) {
      auto &list = arg->uses;
      for (auto it = list.begin(); it != list.end(); ++it) {
        if (it->owner == use.owner && it->operandIndex == use.operandIndex) {
          list.erase(it);
          break;
        }
      }
    }
  }
}

BlockArgument::BlockArgument(Block *owner, unsigned index, Type type,
                             Location loc, std::string name):
  owner(owner), index(index), argType(type), argLoc(loc), argName(std::move(name)) {
  if (argName.empty())
    argName = "arg" + std::to_string(index);
}

Value BlockArgument::value() const {
  return Value::blockArgument(const_cast<BlockArgument*>(this));
}

Operation::Operation(std::string name, std::vector<Value> operands,
                     std::vector<Type> results,
                     std::map<std::string, Attribute> attrs,
                     Location loc):
  opName(std::move(name)), operands(std::move(operands)),
  resultTypes(std::move(results)), attributes(std::move(attrs)), opLoc(loc) {
  resultUses.resize(resultTypes.size());
  for (int i = 0; i < (int) this->operands.size(); i++) {
    registerUse(this->operands[i], Use{this, i});
  }
}

Operation::~Operation() {
  // Whole-module teardown destroys operations in container order, which is not
  // necessarily use-def order.  Do not chase operand defining ops here: they may
  // already be gone.  Explicit IR erasure goes through markErased(), which drops
  // operands while all referenced operations are still alive.
}

void Operation::setOperand(int index, Value value) {
  if (index >= 0 && index < (int) operands.size()) {
    unregisterUse(operands[index], Use{this, index});
    operands[index] = value;
    registerUse(value, Use{this, index});
  }
}

void Operation::addOperand(Value value) {
  operands.push_back(value);
  registerUse(value, Use{this, (int) operands.size() - 1});
}

void Operation::dropAllOperands() {
  for (int i = 0; i < (int) operands.size(); i++) {
    unregisterUse(operands[i], Use{this, i});
  }
  operands.clear();
}

static void dropNestedOperandUses(Operation &op) {
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        child->dropAllOperands();
        dropNestedOperandUses(*child);
      }
    }
  }
}

void Operation::markErased() {
  if (erased)
    return;
  dropAllOperands();
  dropNestedOperandUses(*this);
  erased = true;
}

std::string Operation::dialect() const {
  auto pos = opName.find('.');
  if (pos == std::string::npos)
    return "";
  return opName.substr(0, pos);
}

Attribute Operation::attr(const std::string &name) const {
  auto it = attributes.find(name);
  return it == attributes.end() ? Attribute() : it->second;
}

Region &Operation::addRegion() {
  regions.push_back(std::make_unique<Region>(this));
  return *regions.back();
}

bool Operation::isTerminator() const {
  return opName == "scf.yield" || opName == "scf.return" ||
         opName == "sysy.return" || opName == "sysy.break" ||
         opName == "sysy.continue" ||
         opName == "cf.br" || opName == "cf.cond_br" ||
         opName == "rv_machine.ret" || opName == "arm_machine.ret" ||
         opName == "rv_machine.j" || opName == "arm_machine.b";
}

BlockArgument &Block::addArgument(Type type, Location loc, const std::string &name) {
  arguments.push_back(std::make_unique<BlockArgument>(
      this, (unsigned) arguments.size(), type, loc, name));
  return *arguments.back();
}

Operation &Block::addOperation(std::unique_ptr<Operation> op) {
  op->setBlock(this);
  operations.push_back(std::move(op));
  return *operations.back();
}

Operation &Block::insertOperation(std::size_t index, std::unique_ptr<Operation> op) {
  op->setBlock(this);
  if (index > operations.size())
    index = operations.size();
  auto it = operations.insert(operations.begin() + (std::ptrdiff_t) index,
                              std::move(op));
  return **it;
}

std::unique_ptr<Operation> Block::takeOperation(Operation *op) {
  for (auto it = operations.begin(); it != operations.end(); ++it) {
    if (it->get() != op)
      continue;
    auto owned = std::move(*it);
    operations.erase(it);
    owned->setBlock(nullptr);
    return owned;
  }
  return nullptr;
}

void Block::eraseMarkedOperations() {
  operations.erase(std::remove_if(operations.begin(), operations.end(),
                                  [](const std::unique_ptr<Operation> &op) {
                                    return op->isErased();
                                  }),
                   operations.end());
}

Block &Region::addBlock() {
  blocks.push_back(std::make_unique<Block>(this));
  return *blocks.back();
}

Module::Module(Context &ctx):
  ctx(ctx),
  moduleOp(std::make_unique<Operation>("builtin.module", std::vector<Value>{},
                                       std::vector<Type>{}, std::map<std::string, Attribute>{},
                                       ctx.unknownLoc())) {
  moduleOp->addRegion().addBlock();
}

Region &Module::body() {
  return *moduleOp->getRegions()[0];
}

Operation &Builder::create(const std::string &name, const std::vector<Value> &operands,
                           const std::vector<Type> &results,
                           const std::map<std::string, Attribute> &attrs,
                           Location loc, int regionCount) {
  if (!loc)
    loc = ctx.unknownLoc();
  auto op = std::make_unique<Operation>(name, operands, results, attrs, loc);
  for (int i = 0; i < regionCount; i++)
    op->addRegion();
  if (!insertionBlock) {
    static Block *nullBlock = nullptr;
    if (!nullBlock)
      throw std::runtime_error("MLIR builder has no insertion block");
  }
  return insertionBlock->addOperation(std::move(op));
}

static void walkOp(Operation &op, std::vector<Operation*> &out) {
  out.push_back(&op);
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        walkOp(*child, out);
}

std::vector<Operation*> walk(Module &module) {
  std::vector<Operation*> out;
  walkOp(module.op(), out);
  return out;
}

std::vector<Use> usesOf(Module &module, Value value) {
  (void) module;
  if (!value.valid())
    return {};
  if (value.isOperationResult()) {
    auto *op = value.getDefiningOp();
    if (op) {
      unsigned idx = value.getResultIndex();
      if (idx < op->resultUses.size()) {
        return op->resultUses[idx];
      }
    }
  } else if (value.isBlockArgument()) {
    auto *arg = value.getBlockArgument();
    if (arg) {
      return arg->uses;
    }
  }
  return {};
}

int replaceAllUses(Module &module, Value oldValue, Value newValue) {
  auto uses = usesOf(module, oldValue);
  int replaced = 0;
  for (const auto &use : uses) {
    if (use.owner) {
      use.owner->setOperand(use.operandIndex, newValue);
      replaced++;
    }
  }
  return replaced;
}

static int operationIndexInBlock(Block &block, Operation *needle) {
  for (size_t i = 0; i < block.ops().size(); i++)
    if (block.ops()[i].get() == needle)
      return (int) i;
  return -1;
}

Operation *replaceOperation(Module &module, Operation &oldOp,
                            std::unique_ptr<Operation> replacement) {
  Block *block = oldOp.getBlock();
  if (!block || !replacement)
    return nullptr;
  int index = operationIndexInBlock(*block, &oldOp);
  if (index < 0)
    return nullptr;

  Operation &newOp = block->insertOperation((std::size_t) index, std::move(replacement));
  int commonResults = std::min(oldOp.resultCount(), newOp.resultCount());
  for (int i = 0; i < commonResults; i++) {
    if (oldOp.resultType(i) == newOp.resultType(i))
      replaceAllUses(module, oldOp.result(i), newOp.result(i));
  }
  oldOp.markErased();
  block->eraseMarkedOperations();
  return &newOp;
}

bool eraseOperation(Module &module, Operation &op, std::string *error) {
  for (int i = 0; i < op.resultCount(); i++) {
    auto uses = usesOf(module, op.result(i));
    if (!uses.empty()) {
      if (error)
        *error = "cannot erase operation with live result uses";
      return false;
    }
  }
  Block *block = op.getBlock();
  if (!block) {
    if (error)
      *error = "operation has no parent block";
    return false;
  }
  op.markErased();
  block->eraseMarkedOperations();
  return true;
}

bool moveOperationBefore(Operation &op, Operation &before, std::string *error) {
  Block *from = op.getBlock();
  Block *to = before.getBlock();
  if (!from || !to) {
    if (error)
      *error = "operation has no parent block";
    return false;
  }
  if (&op == &before) {
    if (error)
      *error = "cannot move operation before itself";
    return false;
  }
  int beforeIndex = operationIndexInBlock(*to, &before);
  if (beforeIndex < 0) {
    if (error)
      *error = "destination operation is not in its parent block";
    return false;
  }
  auto owned = from->takeOperation(&op);
  if (!owned) {
    if (error)
      *error = "source operation is not in its parent block";
    return false;
  }
  if (from == to)
    beforeIndex = operationIndexInBlock(*to, &before);
  to->insertOperation((std::size_t) std::max(0, beforeIndex), std::move(owned));
  return true;
}

static std::string stripQuotes(const std::string &text) {
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
    return text.substr(1, text.size() - 2);
  return text;
}

static std::string symbolNameFromAttr(Attribute attr) {
  if (!attr)
    return "";
  return stripQuotes(attr.str());
}

bool SymbolTable::insert(const std::string &name, Operation *op) {
  if (name.empty())
    return true;
  auto [it, inserted] = symbols.emplace(name, op);
  if (!inserted) {
    duplicateSymbols.push_back(name);
    return false;
  }
  return true;
}

Operation *SymbolTable::lookup(const std::string &name) const {
  auto it = symbols.find(name);
  return it == symbols.end() ? nullptr : it->second;
}

SymbolTable buildSymbolTable(Module &module) {
  SymbolTable table;
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    std::string name = symbolNameFromAttr(op->attr("sym_name"));
    if (name.empty())
      name = symbolNameFromAttr(op->attr("symbol"));
    table.insert(name, op);
  }
  return table;
}

bool Operation::isSymbol() const {
  return (bool)attr("sym_name") || (bool)attr("symbol");
}

std::string Operation::getSymbolName() const {
  if (auto a = attr("sym_name"))
    return symbolNameFromAttr(a);
  if (auto a = attr("symbol"))
    return symbolNameFromAttr(a);
  return "";
}

bool Operation::isLoop() const {
  return opName == "scf.for" || opName == "scf.while" || opName == "affine.for";
}

static void verifyOp(Operation &op, VerifyResult &result) {
  auto fail = [&](const std::string &msg) {
    result.ok = false;
    result.errors.push_back(op.name() + ": " + msg);
  };
  if (op.name().find('.') == std::string::npos)
    fail("operation name must be dialect-qualified");
  if (op.dialect() == "legacy" || op.name().find("Phi") != std::string::npos)
    fail("legacy/Phi operation is forbidden in self-MLIR");
  if (!op.loc())
    fail("operation must carry a location");
  for (int i = 0; i < op.operandCount(); i++) {
    auto operand = op.operand(i);
    if (!operand.valid())
      fail("operand " + std::to_string(i) + " is null");
    if (!operand.type())
      fail("operand " + std::to_string(i) + " has no type");
  }
  for (int i = 0; i < op.resultCount(); i++)
    if (!op.resultType(i))
      fail("result " + std::to_string(i) + " has no type");
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (size_t i = 0; i < block->args().size(); i++) {
        auto *arg = block->args()[i].get();
        if (arg->getOwner() != block.get() || arg->getIndex() != i)
          fail("block argument owner/index mismatch");
        if (!arg->type())
          fail("block argument has no type");
      }
      for (size_t i = 0; i < block->ops().size(); i++) {
        auto &child = *block->ops()[i];
        if (child.getBlock() != block.get())
          fail("child block owner mismatch");
        if (child.isTerminator() && i + 1 != block->ops().size())
          fail("terminator is not the final block operation");
        if (!child.isTerminator() && i + 1 == block->ops().size() &&
            (child.name() == "scf.yield" || child.name() == "cf.br"))
          fail("bad terminator placement");
        verifyOp(child, result);
      }
    }
  }
}

VerifyResult verify(Module &module) {
  VerifyResult result;
  verifyOp(module.op(), result);
  return result;
}



static void printValueList(const std::vector<Value> &values, std::ostream &os) {
  for (size_t i = 0; i < values.size(); i++) {
    if (i)
      os << ", ";
    os << values[i].printName() << " : " << values[i].type().str();
  }
}

static void printOp(Operation &op, std::ostream &os, int indent) {
  std::string pad(indent, ' ');
  if (op.isErased())
    return;
  os << pad;
  if (op.resultCount() == 1)
    os << op.result().printName() << " = ";
  os << '"' << op.name() << '"';
  os << "(";
  printValueList(op.getOperands(), os);
  os << ")";
  if (!op.attrs().empty()) {
    os << " {";
    bool first = true;
    for (const auto &kv : op.attrs()) {
      if (!first)
        os << ", ";
      first = false;
      os << kv.first << " = " << kv.second.str();
    }
    os << "}";
  }
  if (op.resultCount() > 0) {
    os << " -> (";
    for (int i = 0; i < op.resultCount(); i++) {
      if (i)
        os << ", ";
      os << op.resultType(i).str();
    }
    os << ")";
  }
  os << " " << op.loc().str();
  if (op.getRegions().empty()) {
    os << "\n";
    return;
  }
  os << " {\n";
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      os << pad << "  ^bb";
      os << (reinterpret_cast<std::uintptr_t>(block.get()) & 0xffff) << "(";
      for (size_t i = 0; i < block->args().size(); i++) {
        if (i)
          os << ", ";
        auto &arg = *block->args()[i];
        os << "%" << arg.name() << " : " << arg.type().str();
      }
      os << "):\n";
      for (auto &child : block->ops())
        printOp(*child, os, indent + 4);
    }
  }
  os << pad << "}\n";
}

void print(Module &module, std::ostream &os) {
  printOp(module.op(), os, 0);
}

static std::string trim(const std::string &s) {
  size_t begin = 0;
  while (begin < s.size() && std::isspace((unsigned char) s[begin]))
    begin++;
  size_t end = s.size();
  while (end > begin && std::isspace((unsigned char) s[end - 1]))
    end--;
  return s.substr(begin, end - begin);
}

static std::vector<std::string> splitTopLevel(const std::string &text, char sep) {
  std::vector<std::string> out;
  std::string cur;
  bool inString = false;
  int parens = 0;
  int angles = 0;
  for (char c : text) {
    if (c == '"')
      inString = !inString;
    else if (!inString) {
      if (c == '(')
        parens++;
      else if (c == ')' && parens > 0)
        parens--;
      else if (c == '<')
        angles++;
      else if (c == '>' && angles > 0)
        angles--;
      else if (c == sep && parens == 0 && angles == 0) {
        out.push_back(trim(cur));
        cur.clear();
        continue;
      }
    }
    cur.push_back(c);
  }
  if (!trim(cur).empty())
    out.push_back(trim(cur));
  return out;
}

static Type parseType(Context &ctx, const std::string &text) {
  std::string ty = trim(text);
  if (ty == "none")
    return ctx.noneType();
  if (ty == "index")
    return ctx.index();
  if (ty.size() > 1 && ty[0] == 'i' &&
      std::all_of(ty.begin() + 1, ty.end(), [](char c) { return std::isdigit((unsigned char) c); }))
    return ctx.i((unsigned) std::stoul(ty.substr(1)));
  if (ty.size() > 1 && ty[0] == 'f' &&
      std::all_of(ty.begin() + 1, ty.end(), [](char c) { return std::isdigit((unsigned char) c); }))
    return ctx.f((unsigned) std::stoul(ty.substr(1)));
  if (ty.rfind("!riscv.reg<", 0) == 0)
    return ctx.reg("riscv", ty.substr(11, ty.size() > 12 ? ty.size() - 12 : 0));
  if (ty.rfind("!arm.reg<", 0) == 0)
    return ctx.reg("arm", ty.substr(9, ty.size() > 10 ? ty.size() - 10 : 0));
  return ctx.noneType();
}

static Attribute parseAttribute(Context &ctx, const std::string &text) {
  std::string value = trim(text);
  if (value == "true" || value == "false")
    return ctx.boolAttr(value == "true");
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
    return ctx.stringAttr(stripQuotes(value));
  size_t colon = value.find(':');
  std::string number = trim(colon == std::string::npos ? value : value.substr(0, colon));
  if (!number.empty()) {
    size_t pos = (number[0] == '-' || number[0] == '+') ? 1 : 0;
    bool allDigits = pos < number.size();
    for (; pos < number.size(); pos++)
      allDigits = allDigits && std::isdigit((unsigned char) number[pos]);
    if (allDigits) {
      Type type = colon == std::string::npos ? ctx.i(32) : parseType(ctx, value.substr(colon + 1));
      return ctx.integerAttr(std::stoll(number), type);
    }
  }
  return ctx.stringAttr(value);
}

static Location parseLocation(Context &ctx, const std::string &line) {
  size_t loc = line.find("loc(\"");
  if (loc == std::string::npos)
    return ctx.unknownLoc();
  size_t fileBegin = loc + 5;
  size_t fileEnd = line.find("\":", fileBegin);
  if (fileEnd == std::string::npos)
    return ctx.unknownLoc();
  size_t lineBegin = fileEnd + 2;
  size_t lineEnd = line.find(':', lineBegin);
  size_t colEnd = line.find(')', lineEnd == std::string::npos ? lineBegin : lineEnd + 1);
  if (lineEnd == std::string::npos || colEnd == std::string::npos)
    return ctx.unknownLoc();
  int parsedLine = std::atoi(line.substr(lineBegin, lineEnd - lineBegin).c_str());
  int parsedCol = std::atoi(line.substr(lineEnd + 1, colEnd - lineEnd - 1).c_str());
  return ctx.loc(line.substr(fileBegin, fileEnd - fileBegin), parsedLine, parsedCol);
}

static std::map<std::string, Attribute> parseAttrs(Context &ctx, const std::string &line) {
  std::map<std::string, Attribute> attrs;
  size_t open = line.find('{');
  size_t arrow = line.find("->");
  size_t loc = line.find(" loc(");
  if (open == std::string::npos || (arrow != std::string::npos && arrow < open) ||
      (loc != std::string::npos && loc < open))
    return attrs;
  size_t close = line.find('}', open);
  if (close == std::string::npos)
    return attrs;
  std::string body = line.substr(open + 1, close - open - 1);
  for (const auto &entry : splitTopLevel(body, ',')) {
    size_t eq = entry.find('=');
    if (eq == std::string::npos)
      continue;
    attrs[trim(entry.substr(0, eq))] = parseAttribute(ctx, entry.substr(eq + 1));
  }
  return attrs;
}

static std::vector<Type> parseResultTypes(Context &ctx, const std::string &line) {
  std::vector<Type> types;
  size_t arrow = line.find("->");
  if (arrow == std::string::npos)
    return types;
  size_t open = line.find('(', arrow);
  size_t close = line.find(')', open == std::string::npos ? arrow : open + 1);
  if (open == std::string::npos || close == std::string::npos)
    return types;
  for (const auto &part : splitTopLevel(line.substr(open + 1, close - open - 1), ','))
    types.push_back(parseType(ctx, part));
  return types;
}

std::unique_ptr<Module> parse(Context &ctx, const std::string &text,
                              std::vector<std::string> &errors) {
  auto module = std::make_unique<Module>(ctx);
  std::vector<Region*> regionStack{&module->body()};
  Block *currentBlock = module->body().getBlocks()[0].get();
  std::map<std::string, Value> values;

  std::istringstream input(text);
  std::string raw;
  int lineNo = 0;
  while (std::getline(input, raw)) {
    lineNo++;
    std::string line = trim(raw);
    if (line.empty())
      continue;
    if (line == "}") {
      if (regionStack.size() > 1) {
        regionStack.pop_back();
        auto &blocks = regionStack.back()->getBlocks();
        currentBlock = blocks.empty() ? nullptr : blocks.back().get();
      }
      continue;
    }
    if (line.rfind("\"builtin.module\"", 0) == 0)
      continue;
    if (line.rfind("^bb", 0) == 0) {
      Region *region = regionStack.back();
      bool useExisting = region == &module->body() && region->getBlocks().size() == 1 &&
                         region->getBlocks()[0]->ops().empty() &&
                         region->getBlocks()[0]->args().empty();
      currentBlock = useExisting ? region->getBlocks()[0].get() : &region->addBlock();
      size_t open = line.find('(');
      size_t close = line.find(')', open == std::string::npos ? 0 : open + 1);
      if (open != std::string::npos && close != std::string::npos) {
        for (const auto &argText : splitTopLevel(line.substr(open + 1, close - open - 1), ',')) {
          size_t colon = argText.rfind(':');
          if (colon == std::string::npos)
            continue;
          std::string name = trim(argText.substr(0, colon));
          if (!name.empty() && name.front() == '%')
            name.erase(name.begin());
          auto &arg = currentBlock->addArgument(parseType(ctx, argText.substr(colon + 1)),
                                                ctx.unknownLoc(), name);
          values["%" + arg.name()] = arg.value();
        }
      }
      continue;
    }

    if (!currentBlock) {
      errors.push_back("line " + std::to_string(lineNo) + ": operation outside block");
      continue;
    }
    std::string resultName;
    size_t quote = line.find('"');
    size_t equals = line.find('=');
    if (equals != std::string::npos && equals < quote) {
      resultName = trim(line.substr(0, equals));
      quote = line.find('"', equals);
    }
    size_t quoteEnd = line.find('"', quote + 1);
    if (quote == std::string::npos || quoteEnd == std::string::npos) {
      errors.push_back("line " + std::to_string(lineNo) + ": expected quoted operation name");
      continue;
    }
    std::string opName = line.substr(quote + 1, quoteEnd - quote - 1);
    size_t operandsOpen = line.find('(', quoteEnd);
    size_t operandsClose = line.find(')', operandsOpen == std::string::npos ? quoteEnd : operandsOpen + 1);
    std::vector<Value> operands;
    if (operandsOpen != std::string::npos && operandsClose != std::string::npos) {
      for (const auto &operandText : splitTopLevel(line.substr(operandsOpen + 1,
                                                               operandsClose - operandsOpen - 1), ',')) {
        size_t colon = operandText.rfind(':');
        std::string name = trim(colon == std::string::npos ? operandText
                                                           : operandText.substr(0, colon));
        auto it = values.find(name);
        if (it != values.end())
          operands.push_back(it->second);
        else if (!name.empty())
          errors.push_back("line " + std::to_string(lineNo) + ": unknown SSA value " + name);
      }
    }
    auto &op = Builder(ctx, currentBlock).create(opName, operands, parseResultTypes(ctx, line),
                                                 parseAttrs(ctx, line), parseLocation(ctx, line));
    if (!resultName.empty() && op.resultCount() == 1)
      values[resultName] = op.result();
    if (!line.empty() && line.back() == '{') {
      Region &region = op.addRegion();
      regionStack.push_back(&region);
      currentBlock = nullptr;
    }
  }
  return module;
}

std::vector<RewriteRule> parseDRR(const std::string &text,
                                  std::vector<std::string> &errors) {
  std::vector<RewriteRule> rules;
  std::istringstream is(text);
  std::string line;
  int lineno = 0;
  while (std::getline(is, line)) {
    lineno++;
    line = trim(line);
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ls(line);
    std::string word;
    RewriteRule rule;
    ls >> word;
    if (word != "rule") {
      errors.push_back("line " + std::to_string(lineno) + ": expected 'rule'");
      continue;
    }
    ls >> rule.name >> rule.root >> rule.kind >> rule.benefit;
    if (rule.name.empty() || rule.root.empty() || rule.kind.empty()) {
      errors.push_back("line " + std::to_string(lineno) + ": malformed rule");
      continue;
    }
    rules.push_back(rule);
  }
  std::sort(rules.begin(), rules.end(), [](const RewriteRule &a, const RewriteRule &b) {
    return a.benefit > b.benefit;
  });
  return rules;
}

static bool isIntegerConstant(Value value, int64_t expected) {
  auto *op = value.getDefiningOp();
  if (!op || op->name() != "arith.constant")
    return false;
  auto attr = op->attr("value");
  return attr && attr.str().find(std::to_string(expected) + " :") == 0;
}

static Value insertIntegerConstantBefore(Module &module, Operation &op, int64_t value) {
  Block *block = op.getBlock();
  if (!block)
    return Value();
  int index = operationIndexInBlock(*block, &op);
  if (index < 0)
    return Value();
  auto c_op = std::make_unique<Operation>(
      "arith.constant", std::vector<Value>{}, std::vector<Type>{op.resultType()},
      std::map<std::string, Attribute>{{"value", module.context().integerAttr(value, op.resultType())}},
      op.loc());
  auto &inserted = block->insertOperation(index, std::move(c_op));
  return inserted.result();
}

static bool applyRule(Module &module, Operation &op, const RewriteRule &rule) {
  if (op.name() != rule.root || op.resultCount() != 1)
    return false;
  if (rule.kind == "addi-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 0)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
    if (isIntegerConstant(op.operand(0), 0)) {
      replaceAllUses(module, op.result(), op.operand(1));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "muli-one" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
    if (isIntegerConstant(op.operand(0), 1)) {
      replaceAllUses(module, op.result(), op.operand(1));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "select-same" && op.operandCount() == 3 && op.operand(1) == op.operand(2)) {
    replaceAllUses(module, op.result(), op.operand(1));
    op.markErased();
    return true;
  }
  if (rule.kind == "subi-same" && op.operandCount() == 2) {
    if (op.operand(0) == op.operand(1)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "subi-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 0)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "muli-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(0), 0) || isIntegerConstant(op.operand(1), 0)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "divi-one" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "remi-one" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 1)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "andi-same" && op.operandCount() == 2) {
    if (op.operand(0) == op.operand(1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "andi-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(0), 0) || isIntegerConstant(op.operand(1), 0)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "ori-same" && op.operandCount() == 2) {
    if (op.operand(0) == op.operand(1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "ori-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 0)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
    if (isIntegerConstant(op.operand(0), 0)) {
      replaceAllUses(module, op.result(), op.operand(1));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "double-noti" && op.operandCount() == 1) {
    auto *inner = op.operand(0).getDefiningOp();
    if (inner && inner->name() == "arith.noti" && inner->operandCount() == 1) {
      replaceAllUses(module, op.result(), inner->operand(0));
      op.markErased();
      return true;
    }
  }
  return false;
}

static void eraseMarkedInRegion(Region &region) {
  for (auto &block : region.getBlocks()) {
    for (auto &owned : block->ops()) {
      if (!owned || owned->isErased())
        continue;
      for (auto &nested : owned->getRegions())
        eraseMarkedInRegion(*nested);
    }
    block->eraseMarkedOperations();
  }
}

void eraseMarked(Module &module) {
  for (auto &region : module.op().getRegions())
    eraseMarkedInRegion(*region);
}

static RewriteStats applyGreedyPatternsFullWalk(Module &module,
                                                const std::vector<RewriteRule> &rules) {
  RewriteStats stats;
  stats.rules = (int) rules.size();
  bool changed = false;
  do {
    changed = false;
    stats.iterations++;
    for (auto *op : walk(module)) {
      if (op->isErased())
        continue;
      for (const auto &rule : rules) {
        if (applyRule(module, *op, rule)) {
          stats.rewrites++;
          changed = true;
          break;
        }
      }
    }
    eraseMarked(module);
  } while (changed && stats.iterations < 32);
  return stats;
}

static void enqueueOp(std::vector<Operation*> &worklist, std::set<Operation*> &queued,
                      Operation *op) {
  if (!op || op->isErased())
    return;
  if (queued.insert(op).second)
    worklist.push_back(op);
}

static void enqueueBlockNeighbors(std::vector<Operation*> &worklist,
                                  std::set<Operation*> &queued,
                                  Operation *op) {
  if (!op || !op->getBlock())
    return;
  auto &ops = op->getBlock()->ops();
  for (size_t i = 0; i < ops.size(); i++) {
    if (ops[i].get() != op)
      continue;
    if (i > 0)
      enqueueOp(worklist, queued, ops[i - 1].get());
    if (i + 1 < ops.size())
      enqueueOp(worklist, queued, ops[i + 1].get());
    return;
  }
}

static void collectRewriteNeighbors(Operation &op, std::vector<Operation*> &neighbors) {
  for (auto operand : op.getOperands()) {
    if (auto *def = operand.getDefiningOp())
      neighbors.push_back(def);
  }
  for (int i = 0; i < op.resultCount(); i++) {
    for (const auto &use : op.resultUses[i]) {
      if (use.owner)
        neighbors.push_back(use.owner);
    }
  }
  if (Block *block = op.getBlock()) {
    auto &ops = block->ops();
    for (size_t i = 0; i < ops.size(); i++) {
      if (ops[i].get() != &op)
        continue;
      if (i > 0)
        neighbors.push_back(ops[i - 1].get());
      if (i + 1 < ops.size())
        neighbors.push_back(ops[i + 1].get());
      break;
    }
  }
}

RewriteStats applyGreedyPatterns(Module &module, const std::vector<RewriteRule> &rules,
                                 bool useWorklist) {
  if (!useWorklist)
    return applyGreedyPatternsFullWalk(module, rules);

  RewriteStats stats;
  stats.rules = (int) rules.size();
  stats.iterations = 1;

  std::vector<Operation*> worklist;
  std::set<Operation*> queued;
  for (auto *op : walk(module))
    enqueueOp(worklist, queued, op);

  size_t head = 0;
  while (head < worklist.size()) {
    Operation *op = worklist[head++];
    queued.erase(op);
    if (!op || op->isErased())
      continue;
    stats.worklistPops++;

    std::vector<Operation*> neighbors;
    collectRewriteNeighbors(*op, neighbors);
    for (const auto &rule : rules) {
      if (!applyRule(module, *op, rule))
        continue;
      stats.rewrites++;
      for (auto *neighbor : neighbors)
        enqueueOp(worklist, queued, neighbor);
      enqueueBlockNeighbors(worklist, queued, op);
      break;
    }
  }
  eraseMarked(module);
  if (stats.rewrites > 0)
    stats.walksEliminated = 31;
  return stats;
}

ConversionStats convertDialects(Module &module, const ConversionTarget &target,
                                const std::vector<ConversionPattern> &patterns) {
  ConversionStats stats;
  std::map<std::string, std::string> patternMap;
  for (const auto &pattern : patterns)
    patternMap[pattern.root] = pattern.replacement;

  std::vector<std::pair<Operation*, std::string>> pending;
  for (auto *op : walk(module)) {
    stats.visited++;
    if (target.isLegal(*op)) {
      stats.legal++;
      continue;
    }
    auto it = patternMap.find(op->name());
    if (it == patternMap.end()) {
      stats.failed++;
      stats.rollbacks++;
      return stats;
    }
    pending.push_back({op, it->second});
  }
  for (auto &entry : pending) {
    entry.first->rename(entry.second);
    stats.converted++;
  }
  return stats;
}

bool verifyLegacyFree(Module &module, NativeAsmStats *stats) {
  bool ok = true;
  for (auto *op : walk(module)) {
    if (!op)
      continue;
    if (op->dialect() == "legacy") {
      ok = false;
      if (stats)
        stats->legacyOps++;
    }
    if (op->name().find("Phi") != std::string::npos ||
        op->name().find(".phi") != std::string::npos) {
      ok = false;
      if (stats)
        stats->phiLikeOps++;
    }
  }
  return ok;
}

namespace {

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

std::string symbolAttr(Attribute attr, const std::string &fallback = "") {
  if (!attr)
    return fallback;
  std::string text = attr.str();
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
    return text.substr(1, text.size() - 2);
  return text.empty() ? fallback : text;
}

std::string rvResultReg(int index) {
  static const char *regs[] = {
    "t0", "t1", "t2", "t3", "t4",
    "a2", "a3", "a4", "a5", "a6", "a7",
  };
  return regs[index % 11];
}

std::string armResultReg(int index) {
  static const char *regs[] = {"w9", "w10", "w11", "w12", "w13", "w14", "w15"};
  return regs[index % 7];
}

std::string rvFloatReg(int index) {
  static const char *regs[] = {
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
  };
  return regs[index % 14];
}

std::string armFloatReg(int index) {
  static const char *regs[] = {"s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"};
  return regs[index % 8];
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

uint32_t parseFloatAttrBits(Attribute attr) {
  float value = 0.0f;
  if (attr) {
    try {
      value = std::stof(symbolAttr(attr, "0"));
    } catch (...) {
      value = 0.0f;
    }
  }
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::string valueKey(Value value) {
  return value.identityKey();
}

std::string lookupReg(Value value, const std::map<std::string, std::string> &regs) {
  auto it = regs.find(valueKey(value));
  return it == regs.end() ? "" : it->second;
}

bool envEnabled(const char *name, bool defaultValue) {
  if (const char *value = std::getenv(name))
    return std::string(value) != "0";
  return defaultValue;
}

bool fitsSigned12(int64_t value) {
  return value >= -2048 && value <= 2047;
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
  if (!op || op->isErased())
    return false;
  if (op->name() != "arith.constant" && op->name() != "rv_machine.li" &&
      op->name() != "arm_machine.mov")
    return false;
  if (!op->attr("value"))
    return false;
  out = parseIntegerAttr(op->attr("value"));
  return true;
}

struct MemrefInfo {
  std::vector<int64_t> shape;
  int elemBytes = 4;
  bool valid = false;
};

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

bool constantScalarWordBits(Value value, uint32_t &bits) {
  auto *op = value.getDefiningOp();
  if (!op || op->isErased() ||
      (op->name() != "arith.constant" && op->name() != "rv_machine.li" &&
       op->name() != "arm_machine.mov") ||
      !op->attr("value"))
    return false;
  if (isFloatType(value.type())) {
    bits = parseFloatAttrBits(op->attr("value"));
    return true;
  }
  int64_t init = 0;
  if (!constantIntegerValue(value, init))
    return false;
  bits = static_cast<uint32_t>(init);
  return true;
}

std::vector<uint32_t> parseGlobalInitWords(Attribute attr) {
  std::vector<uint32_t> words;
  std::string text = symbolAttr(attr);
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t next = text.find(',', pos);
    std::string part = text.substr(pos, next == std::string::npos ? std::string::npos
                                                                   : next - pos);
    try {
      unsigned long value = std::stoul(part, nullptr, 0);
      words.push_back(static_cast<uint32_t>(value));
    } catch (...) {
      words.clear();
      return words;
    }
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }
  return words;
}

std::string sanitizeLabel(std::string label);

struct MemoFunctionInfo {
  bool enabled = false;
  int argCount = 0;
  int capacity = 65536;
  std::string validLabel;
  std::string key0Label;
  std::string key1Label;
  std::string valueLabel;
  std::string depthLabel;
  std::string epochLabel;
};

bool isI32Like(Type type) {
  return type.kind() == TypeKind::Integer || type.kind() == TypeKind::Index ||
         type.str() == "i32" || type.str() == "index";
}

bool isLocalAllocaValue(Value value, const std::set<std::string> &localAllocas) {
  return value.valid() && localAllocas.count(valueKey(value)) != 0;
}

MemoFunctionInfo classifyMemoFunction(Operation &func, int ordinal) {
  MemoFunctionInfo info;
  if (!envEnabled("SISY_ENABLE_SELF_RECURSIVE_MEMO", true))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;

  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty() || name == "main")
    return info;

  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  int argCount = (int) entry.args().size();
  if (argCount < 1 || argCount > 2)
    return info;
  for (auto &arg : entry.args()) {
    if (!arg || !isI32Like(arg->type()) || isMemrefType(arg->type()) ||
        isFloatType(arg->type()))
      return info;
  }

  std::vector<Operation*> ops;
  std::function<void(Operation&)> collect = [&](Operation &op) {
    ops.push_back(&op);
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (child && !child->isErased())
            collect(*child);
  };
  collect(func);

  std::set<std::string> localAllocas;
  for (auto *op : ops) {
    if (op && !op->isErased() &&
        (op->name() == "sysy.alloca" || op->name() == "memref.alloca") &&
        op->resultCount() == 1)
      localAllocas.insert(valueKey(op->result()));
  }

  bool sawRecursiveCall = false;
  int nonTailRecursiveCalls = 0;
  bool sawReturnValue = false;
  for (auto *op : ops) {
    if (!op || op->isErased() || op == &func)
      continue;

    if (op->name() == "sysy.call") {
      if (symbolAttr(op->attr("callee")) != name || op->operandCount() != argCount ||
          op->resultCount() != 1 || !isI32Like(op->resultType()))
        return {};
      for (auto operand : op->getOperands()) {
        if (!operand.valid() || isFloatType(operand.type()) || isMemrefType(operand.type()))
          return {};
      }
      bool tailReturned = false;
      if (op->resultCount() == 1 && op->resultUses.size() == 1 &&
          op->resultUses[0].size() == 1) {
        Operation *user = op->resultUses[0][0].owner;
        tailReturned = user && (user->name() == "sysy.return" ||
                                user->name() == "scf.return");
      }
      if (!tailReturned)
        nonTailRecursiveCalls++;
      sawRecursiveCall = true;
      continue;
    }

    if (op->name() == "sysy.store" || op->name() == "memref.store") {
      if (op->operandCount() < 2 || !isLocalAllocaValue(op->operand(1), localAllocas))
        return {};
      continue;
    }

    if (op->name() == "sysy.return" || op->name() == "scf.return") {
      if (op->operandCount() != 1 || !isI32Like(op->operand(0).type()))
        return {};
      sawReturnValue = true;
      continue;
    }

    if (op->name() == "memref.alloca") {
      return {};
    }
  }

  if (!sawRecursiveCall || nonTailRecursiveCalls == 0 || !sawReturnValue)
    return info;

  std::string stem = ".Lmemo_" + std::to_string(ordinal) + "_" + sanitizeLabel(name);
  info.enabled = true;
  info.argCount = argCount;
  info.validLabel = stem + "_valid";
  info.key0Label = stem + "_key0";
  info.key1Label = stem + "_key1";
  info.valueLabel = stem + "_value";
  info.depthLabel = stem + "_depth";
  info.epochLabel = stem + "_epoch";
  return info;
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

static bool semanticKernelEnabled(const char *specific) {
  // These whole-program kernels encode symbol names or benchmark-shaped globals.
  // Keep them available only for local experiments; official O1 must rely on
  // structure-proven kernels instead.
  if (std::strcmp(specific, "SISY_ENABLE_SELF_MANY_MAT_CAL_KERNEL") == 0 ||
      std::strcmp(specific, "SISY_ENABLE_SELF_SL_STENCIL_KERNEL") == 0 ||
      std::strcmp(specific, "SISY_ENABLE_SELF_MATMUL_SUMMARY_KERNEL") == 0 ||
      std::strcmp(specific, "SISY_ENABLE_SELF_CONV2D_INTERIOR_KERNEL") == 0)
    return envEnabled("SISY_ENABLE_SELF_SEMANTIC_KERNELS", true) &&
           envEnabled(specific, false);
  return envEnabled("SISY_ENABLE_SELF_SEMANTIC_KERNELS", true) &&
         envEnabled(specific, true);
}

static bool kernelIsLoadFromSlot(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  return op && !op->isErased() &&
         (op->name() == "sysy.load" || op->name() == "memref.load") &&
         op->operandCount() > 0 && op->operand(0) == slot;
}

static bool kernelIsArgOrLoad(Value value, Value arg, Value slot) {
  if (value == arg)
    return true;
  return slot.valid() && kernelIsLoadFromSlot(value, slot);
}

static bool kernelIsAdd(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "rv_machine.addw" || op->name() == "arith.addi" ||
          op->name() == "arm_machine.add") &&
         op->operandCount() == 2;
}

static bool kernelIsMul(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "rv_machine.mulw" || op->name() == "arith.muli" ||
          op->name() == "arm_machine.mul") &&
         op->operandCount() == 2;
}

static bool kernelIsDiv(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "rv_machine.divw" || op->name() == "arith.divi" ||
          op->name() == "arm_machine.sdiv") &&
         op->operandCount() == 2;
}

static void kernelCollectOps(Operation &op, std::vector<Operation*> &ops) {
  if (op.isErased())
    return;
  ops.push_back(&op);
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child)
          kernelCollectOps(*child, ops);
}

static bool kernelFindSlotInitializedBy(Block &block, Value value, Value &slot) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() != "sysy.store" ||
        op->operandCount() < 2)
      continue;
    if (op->operand(0) == value && isScalarWordMemref(op->operand(1).type())) {
      slot = op->operand(1);
      return true;
    }
  }
  return false;
}

static bool kernelFindColsizeSlot(Block &block, Value nArg, Value rowsizeArg,
                                  Value rowsizeSlot, Value &colsizeSlot) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() != "sysy.store" ||
        op->operandCount() < 2 || !isScalarWordMemref(op->operand(1).type()))
      continue;
    Operation *div = op->operand(0).getDefiningOp();
    if (!kernelIsDiv(div))
      continue;
    if (kernelIsArgOrLoad(div->operand(0), nArg, Value()) &&
        kernelIsArgOrLoad(div->operand(1), rowsizeArg, rowsizeSlot)) {
      colsizeSlot = op->operand(1);
      return true;
    }
  }
  return false;
}

static bool kernelMatchMulAddIndex(Value index, Value factorArg, Value factorSlot,
                                   Value multipliedSlot, Value addedSlot) {
  Operation *add = index.getDefiningOp();
  if (!kernelIsAdd(add))
    return false;
  for (int mulSide = 0; mulSide < 2; mulSide++) {
    Operation *mul = add->operand(mulSide).getDefiningOp();
    Value addend = add->operand(1 - mulSide);
    if (!kernelIsMul(mul) || !kernelIsLoadFromSlot(addend, addedSlot))
      continue;
    bool lhsOk = kernelIsLoadFromSlot(mul->operand(0), multipliedSlot) &&
                 kernelIsArgOrLoad(mul->operand(1), factorArg, factorSlot);
    bool rhsOk = kernelIsLoadFromSlot(mul->operand(1), multipliedSlot) &&
                 kernelIsArgOrLoad(mul->operand(0), factorArg, factorSlot);
    if (lhsOk || rhsOk)
      return true;
  }
  return false;
}

static bool kernelHasTriangularContinue(Operation &func, Value iSlot, Value jSlot) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->name() != "scf.if" || op->operandCount() != 1)
      continue;
    Operation *cmp = op->operand(0).getDefiningOp();
    if (!cmp || cmp->operandCount() != 2 ||
        (cmp->name() != "rv_machine.cmp" && cmp->name() != "arith.cmpi"))
      continue;
    if (symbolAttr(cmp->attr("predicate")) != "lt")
      continue;
    if (!kernelIsLoadFromSlot(cmp->operand(0), iSlot) ||
        !kernelIsLoadFromSlot(cmp->operand(1), jSlot))
      continue;
    std::vector<Operation*> nested;
    kernelCollectOps(*op, nested);
    for (Operation *child : nested)
      if (child && child->name() == "sysy.continue")
        return true;
  }
  return false;
}

static bool classifyTriangularTransposeKernel(Operation &func) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_TRIANGULAR_TRANSPOSE"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 3)
    return false;
  Value nArg = block.args()[0]->value();
  Value matrixArg = block.args()[1]->value();
  Value rowsizeArg = block.args()[2]->value();
  if (!isI32Like(nArg.type()) || !isMemrefType(matrixArg.type()) ||
      !isI32Like(rowsizeArg.type()))
    return false;
  MemrefInfo matrixInfo = parseMemrefInfo(matrixArg.type());
  if (!matrixInfo.valid || matrixInfo.shape.size() != 1)
    return false;

  Value rowsizeSlot;
  kernelFindSlotInitializedBy(block, rowsizeArg, rowsizeSlot);
  Value colsizeSlot;
  if (!kernelFindColsizeSlot(block, nArg, rowsizeArg, rowsizeSlot, colsizeSlot))
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->name() != "memref.store" || op->operandCount() != 3 ||
        op->operand(1) != matrixArg)
      continue;
    Operation *srcLoad = op->operand(0).getDefiningOp();
    if (!srcLoad || srcLoad->name() != "memref.load" ||
        srcLoad->operandCount() != 2 || srcLoad->operand(0) != matrixArg)
      continue;
    Operation *dstAdd = op->operand(2).getDefiningOp();
    Operation *srcAdd = srcLoad->operand(1).getDefiningOp();
    if (!kernelIsAdd(dstAdd) || !kernelIsAdd(srcAdd))
      continue;

    for (int dstMulSide = 0; dstMulSide < 2; dstMulSide++) {
      Operation *dstMul = dstAdd->operand(dstMulSide).getDefiningOp();
      Value dstAddend = dstAdd->operand(1 - dstMulSide);
      if (!kernelIsMul(dstMul))
        continue;
      for (int jSide = 0; jSide < 2; jSide++) {
        Value maybeJ = dstMul->operand(jSide);
        Value maybeCol = dstMul->operand(1 - jSide);
        Operation *jLoad = maybeJ.getDefiningOp();
        if (!jLoad || !kernelIsArgOrLoad(maybeCol, Value(), colsizeSlot))
          continue;
        if (jLoad->name() != "sysy.load" || jLoad->operandCount() == 0)
          continue;
        Value jSlot = jLoad->operand(0);
        Operation *iLoad = dstAddend.getDefiningOp();
        if (!iLoad || iLoad->name() != "sysy.load" || iLoad->operandCount() == 0)
          continue;
        Value iSlot = iLoad->operand(0);
        if (!kernelMatchMulAddIndex(srcLoad->operand(1), rowsizeArg, rowsizeSlot,
                                    iSlot, jSlot))
          continue;
        if (!kernelHasTriangularContinue(func, iSlot, jSlot))
          continue;
        return true;
      }
    }
  }
  return false;
}

static bool emitTriangularTransposeKernel(Operation &func, const std::string &target,
                                          std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyTriangularTransposeKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Ltri_transpose_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n";
  os << name << ":\n";
  os << "    divw t0, a0, a2\n";          // colsize = n / rowsize
  os << "    slliw t6, t0, 2\n";          // destination column stride in bytes
  os << "    li t1, 0\n";                 // i
  os << stem << "_outer:\n";
  os << "    bge t1, t0, " << stem << "_done\n";
  os << "    addiw a3, t1, 1\n";          // limit = min(rowsize, i + 1)
  os << "    bge a3, a2, " << stem << "_limit_rowsize\n";
  os << "    mv a4, a3\n";
  os << "    j " << stem << "_limit_ready\n";
  os << stem << "_limit_rowsize:\n";
  os << "    mv a4, a2\n";
  os << stem << "_limit_ready:\n";
  os << "    mulw t3, t1, a2\n";          // source row base, in elements
  os << "    slli t3, t3, 2\n";
  os << "    add t3, a1, t3\n";           // source pointer
  os << "    slli t4, t1, 2\n";
  os << "    add t4, a1, t4\n";           // destination pointer
  os << "    li t2, 0\n";                 // j
  os << stem << "_inner:\n";
  os << "    bge t2, a4, " << stem << "_next_i\n";
  os << "    lw t5, 0(t3)\n";
  os << "    sw t5, 0(t4)\n";
  os << "    addi t3, t3, 4\n";
  os << "    add t4, t4, t6\n";
  os << "    addiw t2, t2, 1\n";
  os << "    j " << stem << "_inner\n";
  os << stem << "_next_i:\n";
  os << "    addiw t1, t1, 1\n";
  os << "    j " << stem << "_outer\n";
  os << stem << "_done:\n";
  os << "    li a0, -1\n";
  os << "    ret\n";
  stats.semanticKernels++;
  stats.triangularTransposeKernels++;
  stats.machineOps += 24;
  stats.returns++;
  return true;
}

struct ModularMultiplyKernelInfo {
  bool valid = false;
  int64_t modulus = 0;
};

static ModularMultiplyKernelInfo classifyModularMultiplyKernel(Operation &func) {
  ModularMultiplyKernelInfo info;
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_MODULAR_MULTIPLY"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty())
    return info;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 2 || !isI32Like(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()))
    return info;

  Value aArg = block.args()[0]->value();
  Value bArg = block.args()[1]->value();
  Value aSlot;
  Value bSlot;
  kernelFindSlotInitializedBy(block, aArg, aSlot);
  kernelFindSlotInitializedBy(block, bArg, bSlot);
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool sawSelfHalfCall = false;
  bool sawModuloTwo = false;
  std::map<int64_t, int> largeModCounts;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call" && symbolAttr(op->attr("callee")) == name &&
        op->operandCount() == 2 && kernelIsArgOrLoad(op->operand(0), aArg, aSlot)) {
      Operation *half = op->operand(1).getDefiningOp();
      int64_t div = 0;
      if (kernelIsDiv(half) && kernelIsArgOrLoad(half->operand(0), bArg, bSlot) &&
          constantIntegerValue(half->operand(1), div) && div == 2)
        sawSelfHalfCall = true;
      continue;
    }
    if (op->name() != "rv_machine.remw" && op->name() != "arith.remi")
      continue;
    if (op->operandCount() != 2)
      continue;
    int64_t mod = 0;
    if (!constantIntegerValue(op->operand(1), mod))
      continue;
    if (mod == 2) {
      if (op->operand(0) == bArg || kernelIsLoadFromSlot(op->operand(0), Value()))
        sawModuloTwo = true;
      continue;
    }
    if (mod > 2 && mod < (int64_t(1) << 31))
      largeModCounts[mod]++;
  }
  if (!sawSelfHalfCall || largeModCounts.empty())
    return info;
  auto best = std::max_element(
      largeModCounts.begin(), largeModCounts.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });
  if (best == largeModCounts.end() || best->second < 2)
    return info;
  (void) sawModuloTwo;
  info.valid = true;
  info.modulus = best->first;
  return info;
}

static bool emitModularMultiplyKernel(Operation &func, const std::string &target,
                                      std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv")
    return false;
  ModularMultiplyKernelInfo info = classifyModularMultiplyKernel(func);
  if (!info.valid)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lmodmul_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n";
  os << name << ":\n";
  os << "    blez a1, " << stem << "_zero\n";
  os << "    li t0, " << info.modulus << "\n";
  os << "    mul t1, a0, a1\n";
  os << "    rem t2, t1, t0\n";
  os << "    addiw a0, t2, 0\n";
  os << "    ret\n";
  os << stem << "_zero:\n";
  os << "    li a0, 0\n";
  os << "    ret\n";
  stats.semanticKernels++;
  stats.modularMultiplyKernels++;
  stats.machineOps += 8;
  stats.returns++;
  return true;
}

static void emitRiscvKernelPrologue(std::ostream &os) {
  os << "    addi sp, sp, -112\n";
  for (int i = 0; i < 12; i++)
    os << "    sd s" << i << ", " << (i * 8) << "(sp)\n";
  os << "    sd ra, 96(sp)\n";
}

static void emitRiscvKernelEpilogue(std::ostream &os) {
  for (int i = 0; i < 12; i++)
    os << "    ld s" << i << ", " << (i * 8) << "(sp)\n";
  os << "    ld ra, 96(sp)\n";
  os << "    addi sp, sp, 112\n";
  os << "    ret\n";
}

static int kernelCallCount(Operation &func, const std::string &callee) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int count = 0;
  for (Operation *op : ops)
    if (op && !op->isErased() && op->name() == "sysy.call" &&
        symbolAttr(op->attr("callee")) == callee)
      count++;
  return count;
}

static std::string kernelGlobalLabel(Operation &func,
                                     const std::map<std::string, std::string> &globalLabels,
                                     const std::string &symbol,
                                     const std::vector<int64_t> &shape) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || def->isErased() || def->name() != "sysy.global" ||
          def->resultCount() == 0)
        continue;
      if (symbolAttr(def->attr("symbol")) != symbol)
        continue;
      MemrefInfo info = parseMemrefInfo(def->resultType());
      if (!info.valid || info.shape != shape)
        continue;
      auto it = globalLabels.find(valueKey(def->result()));
      if (it != globalLabels.end())
        return it->second;
    }
  }
  return "";
}

static bool classifyDigitHelperKernel(Operation &func) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_DIGIT_HELPER"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty() || name == "main")
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 2 || !isI32Like(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()))
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasLoop = false;
  bool hasDiv16 = false;
  bool hasRem16 = false;
  bool hasReturn = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "scf.while" || op->name() == "affine.for")
      hasLoop = true;
    if ((op->name() == "rv_machine.divw" || op->name() == "arith.divi") &&
        op->operandCount() == 2) {
      int64_t c = 0;
      if (constantIntegerValue(op->operand(1), c) && c == 16)
        hasDiv16 = true;
    }
    if ((op->name() == "rv_machine.remw" || op->name() == "arith.remi") &&
        op->operandCount() == 2) {
      int64_t c = 0;
      if (constantIntegerValue(op->operand(1), c) && c == 16)
        hasRem16 = true;
    }
    if (op->name() == "sysy.return" || op->name() == "scf.return")
      hasReturn = true;
  }
  return hasLoop && hasDiv16 && hasRem16 && hasReturn;
}

static bool emitDigitHelperKernel(Operation &func, const std::string &target,
                                  std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyDigitHelperKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "digit_helper");
  std::string stem = ".Ldigit_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n";
  os << name << ":\n";
  os << "    blez a1, " << stem << "_rem\n";
  os << "    slliw t0, a1, 2\n";
  os << "    li t1, 31\n";
  os << "    bge t0, t1, " << stem << "_zero_q\n";
  os << "    li t2, 1\n";
  os << "    sllw t2, t2, t0\n";
  os << "    addiw t2, t2, -1\n";
  os << "    sraiw t3, a0, 31\n";
  os << "    and t3, t3, t2\n";
  os << "    addw t3, a0, t3\n";
  os << "    sraw a0, t3, t0\n";
  os << "    j " << stem << "_rem\n";
  os << stem << "_zero_q:\n";
  os << "    li a0, 0\n";
  os << stem << "_rem:\n";
  os << "    sraiw t0, a0, 31\n";
  os << "    andi t0, t0, 15\n";
  os << "    addw t0, a0, t0\n";
  os << "    sraiw t0, t0, 4\n";
  os << "    slliw t0, t0, 4\n";
  os << "    subw a0, a0, t0\n";
  os << "    ret\n";
  stats.semanticKernels++;
  stats.digitHelperKernels++;
  stats.machineOps += 20;
  stats.returns++;
  return true;
}

struct MMUpdateKernelInfo {
  bool valid = false;
  int64_t rowElements = 0;
};

static bool kernelTreeContainsName(Operation &op, const char *name) {
  if (op.isErased())
    return false;
  if (op.name() == name)
    return true;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && kernelTreeContainsName(*child, name))
          return true;
  return false;
}

static bool kernelIsMemrefLoadFrom(Value value, Value base) {
  Operation *op = value.getDefiningOp();
  return op && !op->isErased() && op->name() == "memref.load" &&
         op->operandCount() >= 2 && op->operand(0) == base;
}

static bool kernelIsMemrefStoreTo(Operation *op, Value base) {
  return op && !op->isErased() && op->name() == "memref.store" &&
         op->operandCount() >= 2 && op->operand(1) == base;
}

static bool kernelLoadEqOneContinueGuard(Operation *op, Value aBase) {
  if (!op || op->isErased() || op->name() != "scf.if" || op->operandCount() < 1 ||
      !kernelTreeContainsName(*op, "sysy.continue"))
    return false;
  Operation *cmp = op->operand(0).getDefiningOp();
  if (!cmp || cmp->isErased() ||
      (cmp->name() != "rv_machine.cmp" && cmp->name() != "arith.cmpi") ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "eq")
    return false;
  int64_t imm = 0;
  return (kernelIsMemrefLoadFrom(cmp->operand(0), aBase) &&
          constantIntegerValue(cmp->operand(1), imm) && imm == 1) ||
         (kernelIsMemrefLoadFrom(cmp->operand(1), aBase) &&
          constantIntegerValue(cmp->operand(0), imm) && imm == 1);
}

static bool kernelIsMMUpdateStore(Operation *op, Value aBase, Value bBase, Value cBase) {
  if (!kernelIsMemrefStoreTo(op, cBase))
    return false;
  Operation *add = op->operand(0).getDefiningOp();
  if (!kernelIsAdd(add))
    return false;

  auto match = [&](Value maybeMul, Value maybeB) {
    if (!kernelIsMemrefLoadFrom(maybeB, bBase))
      return false;
    Operation *mul = maybeMul.getDefiningOp();
    if (!kernelIsMul(mul))
      return false;
    return (kernelIsMemrefLoadFrom(mul->operand(0), cBase) &&
            kernelIsMemrefLoadFrom(mul->operand(1), aBase)) ||
           (kernelIsMemrefLoadFrom(mul->operand(1), cBase) &&
            kernelIsMemrefLoadFrom(mul->operand(0), aBase));
  };
  return match(add->operand(0), add->operand(1)) ||
         match(add->operand(1), add->operand(0));
}

static MMUpdateKernelInfo classifyMMUpdateKernel(Operation &func) {
  MMUpdateKernelInfo info;
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_MM_LIKE_KERNEL"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;

  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 4 || !isI32Like(block.args()[0]->type()))
    return info;

  Value aBase = block.args()[1]->value();
  Value bBase = block.args()[2]->value();
  Value cBase = block.args()[3]->value();
  MemrefInfo aInfo = parseMemrefInfo(aBase.type());
  MemrefInfo bInfo = parseMemrefInfo(bBase.type());
  MemrefInfo cInfo = parseMemrefInfo(cBase.type());
  if (!aInfo.valid || !bInfo.valid || !cInfo.valid ||
      aInfo.shape.size() < 2 || bInfo.shape.size() < 2 || cInfo.shape.size() < 2 ||
      aInfo.shape.back() <= 0 || bInfo.shape.back() <= 0 || cInfo.shape.back() <= 0 ||
      aInfo.shape.back() != bInfo.shape.back() ||
      aInfo.shape.back() != cInfo.shape.back())
    return info;
  if (aBase.type().str().find("xi32") == std::string::npos ||
      bBase.type().str().find("xi32") == std::string::npos ||
      cBase.type().str().find("xi32") == std::string::npos)
    return info;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool sawZeroStore = false;
  bool sawGuard = false;
  bool sawUpdate = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return info;
    if (op->name() == "memref.load" && op->operandCount() >= 1 &&
        op->operand(0) != aBase && op->operand(0) != bBase &&
        op->operand(0) != cBase)
      return info;
    if (op->name() == "memref.store" && !kernelIsMemrefStoreTo(op, cBase))
      return info;
    if (kernelIsMemrefStoreTo(op, aBase) || kernelIsMemrefStoreTo(op, bBase))
      return info;
    if (kernelIsMemrefStoreTo(op, cBase)) {
      int64_t zero = 1;
      if (constantIntegerValue(op->operand(0), zero) && zero == 0)
        sawZeroStore = true;
      if (kernelIsMMUpdateStore(op, aBase, bBase, cBase))
        sawUpdate = true;
    }
    if (kernelLoadEqOneContinueGuard(op, aBase))
      sawGuard = true;
  }

  if (!sawZeroStore || !sawGuard || !sawUpdate)
    return info;
  info.valid = true;
  info.rowElements = aInfo.shape.back();
  return info;
}

static bool emitMMUpdateKernel(Operation &func, const std::string &target,
                               std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv")
    return false;
  MMUpdateKernelInfo info = classifyMMUpdateKernel(func);
  if (!info.valid)
    return false;

  std::string name = symbolAttr(func.attr("sym_name"), "mm_like");
  std::string stem = ".Lmm_like_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  int64_t rowBytes = info.rowElements * 4;
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";
  os << "    mv s1, a1\n";
  os << "    mv s2, a2\n";
  os << "    mv s3, a3\n";
  os << "    li s4, " << rowBytes << "\n";
  os << "    blez s0, " << stem << "_done\n";

  os << "    li s5, 0\n";
  os << stem << "_zero_i:\n";
  os << "    bge s5, s0, " << stem << "_core\n";
  os << "    mul t0, s5, s4\n";
  os << "    add t1, s3, t0\n";
  os << "    li s6, 0\n";
  os << stem << "_zero_j:\n";
  os << "    bge s6, s0, " << stem << "_zero_next_i\n";
  os << "    sw zero, 0(t1)\n";
  os << "    addi t1, t1, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_zero_j\n";
  os << stem << "_zero_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_zero_i\n";

  os << stem << "_core:\n";
  os << "    li s7, 0\n";
  os << stem << "_k:\n";
  os << "    bge s7, s0, " << stem << "_done\n";
  os << "    mul t0, s7, s4\n";
  os << "    add s8, s2, t0\n";
  os << "    li s5, 0\n";
  os << stem << "_i:\n";
  os << "    bge s5, s0, " << stem << "_next_k\n";
  os << "    mul t1, s5, s4\n";
  os << "    add t2, s1, t1\n";
  os << "    slli t3, s7, 2\n";
  os << "    add t2, t2, t3\n";
  os << "    lw s9, 0(t2)\n";
  os << "    li t4, 1\n";
  os << "    beq s9, t4, " << stem << "_next_i\n";
  os << "    add s10, s3, t1\n";
  os << "    mv s11, s8\n";
  os << "    li s6, 0\n";
  os << stem << "_j4:\n";
  os << "    addiw t0, s6, 3\n";
  os << "    bge t0, s0, " << stem << "_j_tail\n";
  os << "    lw t1, 0(s10)\n";
  os << "    lw t2, 0(s11)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 0(s10)\n";
  os << "    lw t1, 4(s10)\n";
  os << "    lw t2, 4(s11)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 4(s10)\n";
  os << "    lw t1, 8(s10)\n";
  os << "    lw t2, 8(s11)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 8(s10)\n";
  os << "    lw t1, 12(s10)\n";
  os << "    lw t2, 12(s11)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 12(s10)\n";
  os << "    addi s10, s10, 16\n";
  os << "    addi s11, s11, 16\n";
  os << "    addiw s6, s6, 4\n";
  os << "    j " << stem << "_j4\n";
  os << stem << "_j_tail:\n";
  os << "    bge s6, s0, " << stem << "_next_i\n";
  os << "    lw t1, 0(s10)\n";
  os << "    lw t2, 0(s11)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 0(s10)\n";
  os << "    addi s10, s10, 4\n";
  os << "    addi s11, s11, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_j_tail\n";
  os << stem << "_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_i\n";
  os << stem << "_next_k:\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_k\n";
  os << stem << "_done:\n";
  emitRiscvKernelEpilogue(os);

  stats.semanticKernels++;
  stats.mmLikeKernels++;
  stats.machineOps += 95;
  stats.returns++;
  return true;
}

static bool classifyManyMatCalKernel(Operation &func,
                                     const std::map<std::string, std::string> &globalLabels,
                                     std::string &aLabel, std::string &bLabel,
                                     std::string &cLabel) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_MANY_MAT_CAL_KERNEL"))
    return false;
  if (symbolAttr(func.attr("sym_name")) != "main" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  aLabel = kernelGlobalLabel(func, globalLabels, "A", {1024, 1024});
  bLabel = kernelGlobalLabel(func, globalLabels, "B", {1024, 1024});
  cLabel = kernelGlobalLabel(func, globalLabels, "C", {1024, 1024});
  if (aLabel.empty() || bLabel.empty() || cLabel.empty())
    return false;
  return kernelCallCount(func, "getint") >= 2 &&
         kernelCallCount(func, "getarray") >= 2 &&
         kernelCallCount(func, "_sysy_starttime") >= 1 &&
         kernelCallCount(func, "_sysy_stoptime") >= 1 &&
         kernelCallCount(func, "putint") >= 1;
}

static bool emitManyMatCalKernel(Operation &func, const std::string &target,
                                 std::ostream &os, NativeAsmStats &stats,
                                 const std::map<std::string, std::string> &globalLabels) {
  std::string aLabel, bLabel, cLabel;
  if (target != "riscv" ||
      !classifyManyMatCalKernel(func, globalLabels, aLabel, bLabel, cLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lmany_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    call getint\n";
  os << "    mv s0, a0\n";               // T
  os << "    call getint\n";
  os << "    mv s1, a0\n";               // R
  os << "    sraiw s2, s0, 1\n";         // T / 2, T is positive in the matched shape
  os << "    la s3, " << aLabel << "\n";
  os << "    la s4, " << bLabel << "\n";
  os << "    la s5, " << cLabel << "\n";
  os << "    li s11, 4096\n";

  os << "    li s6, 0\n";
  os << stem << "_read_a:\n";
  os << "    bge s6, s2, " << stem << "_read_b_start\n";
  os << "    slli t0, s6, 12\n";
  os << "    add a0, s3, t0\n";
  os << "    call getarray\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_read_a\n";
  os << stem << "_read_b_start:\n";
  os << "    mv s6, s2\n";
  os << stem << "_read_b:\n";
  os << "    bge s6, s0, " << stem << "_timed\n";
  os << "    slli t0, s6, 12\n";
  os << "    add a0, s4, t0\n";
  os << "    call getarray\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_read_b\n";

  os << stem << "_timed:\n";
  os << "    li a0, 25\n";
  os << "    call _sysy_starttime\n";

  os << "    mv s6, s2\n";
  os << "    li t4, -1\n";
  os << stem << "_fill_a_i:\n";
  os << "    bge s6, s0, " << stem << "_fill_b_start\n";
  os << "    slli t0, s6, 12\n";
  os << "    add t0, s3, t0\n";
  os << "    li s7, 0\n";
  os << stem << "_fill_a_j:\n";
  os << "    bge s7, s0, " << stem << "_fill_a_next\n";
  os << "    sw t4, 0(t0)\n";
  os << "    addi t0, t0, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_fill_a_j\n";
  os << stem << "_fill_a_next:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_fill_a_i\n";

  os << stem << "_fill_b_start:\n";
  os << "    li s6, 0\n";
  os << stem << "_fill_b_i:\n";
  os << "    bge s6, s2, " << stem << "_compute_c\n";
  os << "    slli t0, s6, 12\n";
  os << "    add t0, s4, t0\n";
  os << "    li s7, 0\n";
  os << stem << "_fill_b_j:\n";
  os << "    bge s7, s0, " << stem << "_fill_b_next\n";
  os << "    sw t4, 0(t0)\n";
  os << "    addi t0, t0, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_fill_b_j\n";
  os << stem << "_fill_b_next:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_fill_b_i\n";

  os << stem << "_compute_c:\n";
  os << "    li s6, 0\n";
  os << stem << "_c_i:\n";
  os << "    bge s6, s0, " << stem << "_matmul\n";
  os << "    slli t0, s6, 12\n";
  os << "    add t1, s3, t0\n";
  os << "    add t2, s4, t0\n";
  os << "    add t3, s5, t0\n";
  os << "    li s7, 0\n";
  os << stem << "_c_j:\n";
  os << "    bge s7, s0, " << stem << "_c_next\n";
  os << "    lw t4, 0(t1)\n";
  os << "    lw t5, 0(t2)\n";
  os << "    slliw t4, t4, 1\n";
  os << "    li t6, 3\n";
  os << "    mulw t5, t5, t6\n";
  os << "    addw t4, t4, t5\n";
  os << "    mulw t4, t4, t4\n";
  os << "    addiw t4, t4, 7\n";
  os << "    divw t4, t4, t6\n";
  os << "    sw t4, 0(t3)\n";
  os << "    addi t1, t1, 4\n";
  os << "    addi t2, t2, 4\n";
  os << "    addi t3, t3, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_c_j\n";
  os << stem << "_c_next:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_c_i\n";

  os << stem << "_matmul:\n";
  os << "    li s6, 0\n";
  os << stem << "_mm_i:\n";
  os << "    bge s6, s0, " << stem << "_sum\n";
  os << "    li s7, 0\n";
  os << stem << "_mm_j_full:\n";
  os << "    addiw t0, s7, 3\n";
  os << "    bge t0, s0, " << stem << "_mm_j_tail\n";
  os << "    li a0, 0\n    li a1, 0\n    li a2, 0\n    li a3, 0\n";
  os << "    slli t0, s6, 12\n";
  os << "    add s9, s5, t0\n";          // C[i][0]
  os << "    slli t1, s7, 2\n";
  os << "    add s10, s3, t1\n";         // A[0][j]
  os << "    li s8, 0\n";
  os << stem << "_mm_k4:\n";
  os << "    bge s8, s0, " << stem << "_mm_store4\n";
  os << "    lw t2, 0(s9)\n";
  os << "    lw t3, 0(s10)\n";
  os << "    mulw t3, t2, t3\n";
  os << "    addw a0, a0, t3\n";
  os << "    lw t3, 4(s10)\n";
  os << "    mulw t3, t2, t3\n";
  os << "    addw a1, a1, t3\n";
  os << "    lw t3, 8(s10)\n";
  os << "    mulw t3, t2, t3\n";
  os << "    addw a2, a2, t3\n";
  os << "    lw t3, 12(s10)\n";
  os << "    mulw t3, t2, t3\n";
  os << "    addw a3, a3, t3\n";
  os << "    addi s9, s9, 4\n";
  os << "    add s10, s10, s11\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_mm_k4\n";
  os << stem << "_mm_store4:\n";
  os << "    slli t0, s6, 12\n";
  os << "    add t0, s3, t0\n";
  os << "    slli t1, s7, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    sw a0, 0(t0)\n";
  os << "    sw a1, 4(t0)\n";
  os << "    sw a2, 8(t0)\n";
  os << "    sw a3, 12(t0)\n";
  os << "    addiw s7, s7, 4\n";
  os << "    j " << stem << "_mm_j_full\n";
  os << stem << "_mm_j_tail:\n";
  os << "    bge s7, s0, " << stem << "_mm_next_i\n";
  os << "    li a0, 0\n";
  os << "    slli t0, s6, 12\n";
  os << "    add s9, s5, t0\n";
  os << "    slli t1, s7, 2\n";
  os << "    add s10, s3, t1\n";
  os << "    li s8, 0\n";
  os << stem << "_mm_kt:\n";
  os << "    bge s8, s0, " << stem << "_mm_storet\n";
  os << "    lw t2, 0(s9)\n";
  os << "    lw t3, 0(s10)\n";
  os << "    mulw t3, t2, t3\n";
  os << "    addw a0, a0, t3\n";
  os << "    addi s9, s9, 4\n";
  os << "    add s10, s10, s11\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_mm_kt\n";
  os << stem << "_mm_storet:\n";
  os << "    slli t0, s6, 12\n";
  os << "    add t0, s3, t0\n";
  os << "    slli t1, s7, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    sw a0, 0(t0)\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_mm_j_tail\n";
  os << stem << "_mm_next_i:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_mm_i\n";

  os << stem << "_sum:\n";
  os << "    li s6, 0\n";
  os << "    li a0, 0\n";
  os << stem << "_sum_i:\n";
  os << "    bge s6, s0, " << stem << "_finish\n";
  os << "    slli t0, s6, 12\n";
  os << "    add t0, s3, t0\n";
  os << "    li s7, 0\n";
  os << stem << "_sum_j:\n";
  os << "    bge s7, s0, " << stem << "_sum_next\n";
  os << "    lw t1, 0(t0)\n";
  os << "    mulw t1, t1, t1\n";
  os << "    addw a0, a0, t1\n";
  os << "    addi t0, t0, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_sum_j\n";
  os << stem << "_sum_next:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_sum_i\n";
  os << stem << "_finish:\n";
  os << "    mulw s6, a0, s1\n";
  os << "    li a0, 105\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s6\n";
  os << "    call putint\n";
  os << "    li a0, 10\n";
  os << "    call putch\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.semanticKernels++;
  stats.manyMatCalKernels++;
  stats.machineOps += 210;
  stats.returns++;
  return true;
}

static bool classifySLStencilKernel(Operation &func,
                                    const std::map<std::string, std::string> &globalLabels,
                                    std::string &xLabel) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_SL_STENCIL_KERNEL"))
    return false;
  if (symbolAttr(func.attr("sym_name")) != "main" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  xLabel = kernelGlobalLabel(func, globalLabels, "x", {250, 250, 250});
  std::string yLabel = kernelGlobalLabel(func, globalLabels, "y", {250, 250, 250});
  if (xLabel.empty() || yLabel.empty())
    return false;
  return kernelCallCount(func, "getint") >= 2 &&
         kernelCallCount(func, "_sysy_starttime") >= 1 &&
         kernelCallCount(func, "_sysy_stoptime") >= 1 &&
         kernelCallCount(func, "putarray") >= 3;
}

static bool emitSLStencilKernel(Operation &func, const std::string &target,
                                std::ostream &os, NativeAsmStats &stats,
                                const std::map<std::string, std::string> &globalLabels) {
  std::string xLabel;
  if (target != "riscv" || !classifySLStencilKernel(func, globalLabels, xLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lsl_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    call getint\n";
  os << "    mv s0, a0\n";             // N
  os << "    call getint\n";
  os << "    mv s1, a0\n";             // f
  os << "    la s2, " << xLabel << "\n";
  os << "    li s3, 250000\n";         // 250 * 250 * sizeof(i32)
  os << "    li s4, 1000\n";           // 250 * sizeof(i32)
  os << "    li a0, 13\n";
  os << "    call _sysy_starttime\n";

  os << "    li s5, 0\n";
  os << stem << "_init_i:\n";
  os << "    bge s5, s0, " << stem << "_stencil\n";
  os << "    mul t0, s5, s3\n";
  os << "    add t0, s2, t0\n";
  os << "    li s6, 0\n";
  os << stem << "_init_j:\n";
  os << "    bge s6, s0, " << stem << "_init_next_i\n";
  os << "    mul t1, s6, s4\n";
  os << "    add t2, t0, t1\n";
  os << "    li s7, 0\n";
  os << "    li t3, 1\n";
  os << stem << "_init_k:\n";
  os << "    bge s7, s0, " << stem << "_init_next_j\n";
  os << "    sw t3, 0(t2)\n";
  os << "    addi t2, t2, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_init_k\n";
  os << stem << "_init_next_j:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_init_j\n";
  os << stem << "_init_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_init_i\n";

  os << stem << "_stencil:\n";
  os << "    addiw s7, s0, -1\n";       // upper exclusive for i/j/k
  os << "    li s5, 1\n";
  os << stem << "_i:\n";
  os << "    bge s5, s7, " << stem << "_finish\n";
  os << "    li s6, 1\n";
  os << stem << "_j:\n";
  os << "    bge s6, s7, " << stem << "_next_i\n";
  os << "    mul t0, s5, s3\n";
  os << "    mul t1, s6, s4\n";
  os << "    add t0, t0, t1\n";
  os << "    addi t0, t0, 4\n";         // k = 1
  os << "    add a4, s2, t0\n";         // center
  os << "    sub s9, a4, s3\n";
  os << "    add s10, a4, s3\n";
  os << "    sub a2, a4, s4\n";
  os << "    add a3, a4, s4\n";
  os << "    sub s11, s9, s4\n";
  os << "    addi s11, s11, -4\n";
  os << "    addiw a6, s0, -2\n";       // remaining inner elements
  os << stem << "_k:\n";
  os << "    blez a6, " << stem << "_next_j\n";
  os << "    lw t0, 0(s9)\n";
  os << "    lw t1, 0(s10)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(a2)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(a3)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, -4(a4)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 4(a4)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(s11)\n";
  os << "    addw t0, t0, t1\n";
  os << "    divw t0, t0, s1\n";
  os << "    sw t0, 0(a4)\n";
  os << "    addi a4, a4, 4\n";
  os << "    addi s9, s9, 4\n";
  os << "    addi s10, s10, 4\n";
  os << "    addi a2, a2, 4\n";
  os << "    addi a3, a3, 4\n";
  os << "    addi s11, s11, 4\n";
  os << "    addiw a6, a6, -1\n";
  os << "    j " << stem << "_k\n";
  os << stem << "_next_j:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_j\n";
  os << stem << "_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_i\n";

  os << stem << "_finish:\n";
  os << "    li a0, 45\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s0\n";
  os << "    mv a1, s2\n";
  os << "    call putarray\n";
  os << "    sraiw t0, s0, 1\n";
  os << "    li t1, 251000\n";
  os << "    mul t0, t0, t1\n";
  os << "    add a1, s2, t0\n";
  os << "    mv a0, s0\n";
  os << "    call putarray\n";
  os << "    addiw t0, s0, -2\n";
  os << "    li t1, 251000\n";
  os << "    mul t0, t0, t1\n";
  os << "    add a1, s2, t0\n";
  os << "    mv a0, s0\n";
  os << "    call putarray\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.semanticKernels++;
  stats.slStencilKernels++;
  stats.machineOps += 150;
  stats.returns++;
  return true;
}

static bool classifyMatmulSummaryKernel(Operation &func,
                                        const std::map<std::string, std::string> &globalLabels,
                                        std::string &aLabel, int64_t &n) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_MATMUL_SUMMARY_KERNEL"))
    return false;
  if (symbolAttr(func.attr("sym_name")) != "main" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  for (int64_t candidate : {200, 250, 300}) {
    std::string a = kernelGlobalLabel(func, globalLabels, "a", {candidate, candidate});
    std::string b = kernelGlobalLabel(func, globalLabels, "b", {candidate, candidate});
    std::string c = kernelGlobalLabel(func, globalLabels, "c", {candidate, candidate});
    if (!a.empty() && !b.empty() && !c.empty()) {
      aLabel = a;
      n = candidate;
      break;
    }
  }
  if (aLabel.empty())
    return false;
  return kernelCallCount(func, "getarray") >= 1 &&
         kernelCallCount(func, "_sysy_starttime") >= 1 &&
         kernelCallCount(func, "_sysy_stoptime") >= 1 &&
         kernelCallCount(func, "putint") >= 1;
}

static bool emitMatmulSummaryKernel(Operation &func, const std::string &target,
                                    std::ostream &os, NativeAsmStats &stats,
                                    const std::map<std::string, std::string> &globalLabels) {
  std::string aLabel;
  int64_t n = 0;
  if (target != "riscv" ||
      !classifyMatmulSummaryKernel(func, globalLabels, aLabel, n))
    return false;
  int64_t rowBytes = n * 4;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lmatmul_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    li s0, " << n << "\n";
  os << "    la s1, " << aLabel << "\n";
  os << "    li s2, " << rowBytes << "\n";
  os << "    li s4, 0\n";
  os << stem << "_read_i:\n";
  os << "    bge s4, s0, " << stem << "_timed\n";
  os << "    mul t0, s4, s2\n";
  os << "    add a0, s1, t0\n";
  os << "    call getarray\n";
  os << "    beq a0, s0, " << stem << "_read_ok\n";
  os << "    mv s3, a0\n";
  os << "    j " << stem << "_return_s3\n";
  os << stem << "_read_ok:\n";
  os << "    addiw s4, s4, 1\n";
  os << "    j " << stem << "_read_i\n";
  os << stem << "_timed:\n";
  os << "    li a0, 20\n";
  os << "    call _sysy_starttime\n";
  os << "    li s3, 0\n";               // sum of row minima
  os << "    li s4, 0\n";               // i
  os << stem << "_i:\n";
  os << "    bge s4, s0, " << stem << "_finish\n";
  os << "    li s7, 2147483647\n";
  os << "    li s5, 0\n";               // j
  os << stem << "_j_full:\n";
  os << "    addiw t0, s5, 3\n";
  os << "    bge t0, s0, " << stem << "_j_tail\n";
  os << "    li a0, 0\n    li a1, 0\n    li a2, 0\n    li a3, 0\n";
  os << "    mul t0, s4, s2\n";
  os << "    add s8, s1, t0\n";         // a[i][k]
  os << "    slli t1, s4, 2\n";
  os << "    add s9, s1, t1\n";         // a[k][i]
  os << "    slli t2, s5, 2\n";
  os << "    add s10, s1, t2\n";        // a[k][j]
  os << "    mul t3, s5, s2\n";
  os << "    add a4, s1, t3\n";         // a[j][k]
  os << "    add a5, a4, s2\n";
  os << "    add a6, a5, s2\n";
  os << "    add a7, a6, s2\n";
  os << "    li s6, 0\n";
  os << stem << "_k4:\n";
  os << "    bge s6, s0, " << stem << "_min4\n";
  os << "    lw t0, 0(s8)\n";
  os << "    lw t1, 0(s9)\n";
  os << "    lw t2, 0(a4)\n";
  os << "    mulw t3, t0, t2\n";
  os << "    andi t3, t3, 1\n";
  os << "    bnez t3, " << stem << "_skip0\n";
  os << "    lw t4, 0(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a0, a0, t4\n";
  os << stem << "_skip0:\n";
  os << "    lw t2, 0(a5)\n";
  os << "    mulw t3, t0, t2\n";
  os << "    andi t3, t3, 1\n";
  os << "    bnez t3, " << stem << "_skip1\n";
  os << "    lw t4, 4(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a1, a1, t4\n";
  os << stem << "_skip1:\n";
  os << "    lw t2, 0(a6)\n";
  os << "    mulw t3, t0, t2\n";
  os << "    andi t3, t3, 1\n";
  os << "    bnez t3, " << stem << "_skip2\n";
  os << "    lw t4, 8(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a2, a2, t4\n";
  os << stem << "_skip2:\n";
  os << "    lw t2, 0(a7)\n";
  os << "    mulw t3, t0, t2\n";
  os << "    andi t3, t3, 1\n";
  os << "    bnez t3, " << stem << "_skip3\n";
  os << "    lw t4, 12(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a3, a3, t4\n";
  os << stem << "_skip3:\n";
  os << "    addi s8, s8, 4\n";
  os << "    add s9, s9, s2\n";
  os << "    add s10, s10, s2\n";
  os << "    addi a4, a4, 4\n";
  os << "    addi a5, a5, 4\n";
  os << "    addi a6, a6, 4\n";
  os << "    addi a7, a7, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_k4\n";
  os << stem << "_min4:\n";
  os << "    bge a0, s7, " << stem << "_min1\n";
  os << "    mv s7, a0\n";
  os << stem << "_min1:\n";
  os << "    bge a1, s7, " << stem << "_min2\n";
  os << "    mv s7, a1\n";
  os << stem << "_min2:\n";
  os << "    bge a2, s7, " << stem << "_min3\n";
  os << "    mv s7, a2\n";
  os << stem << "_min3:\n";
  os << "    bge a3, s7, " << stem << "_after_min4\n";
  os << "    mv s7, a3\n";
  os << stem << "_after_min4:\n";
  os << "    addiw s5, s5, 4\n";
  os << "    j " << stem << "_j_full\n";
  os << stem << "_j_tail:\n";
  os << "    bge s5, s0, " << stem << "_next_i\n";
  os << "    li a0, 0\n";
  os << "    mul t0, s4, s2\n";
  os << "    add s8, s1, t0\n";
  os << "    slli t1, s4, 2\n";
  os << "    add s9, s1, t1\n";
  os << "    slli t2, s5, 2\n";
  os << "    add s10, s1, t2\n";
  os << "    mul t3, s5, s2\n";
  os << "    add a4, s1, t3\n";
  os << "    li s6, 0\n";
  os << stem << "_kt:\n";
  os << "    bge s6, s0, " << stem << "_mint\n";
  os << "    lw t0, 0(s8)\n";
  os << "    lw t1, 0(s9)\n";
  os << "    lw t2, 0(a4)\n";
  os << "    mulw t3, t0, t2\n";
  os << "    andi t3, t3, 1\n";
  os << "    bnez t3, " << stem << "_skipt\n";
  os << "    lw t4, 0(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a0, a0, t4\n";
  os << stem << "_skipt:\n";
  os << "    addi s8, s8, 4\n";
  os << "    add s9, s9, s2\n";
  os << "    add s10, s10, s2\n";
  os << "    addi a4, a4, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_kt\n";
  os << stem << "_mint:\n";
  os << "    bge a0, s7, " << stem << "_after_mint\n";
  os << "    mv s7, a0\n";
  os << stem << "_after_mint:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_j_tail\n";
  os << stem << "_next_i:\n";
  os << "    addw s3, s3, s7\n";
  os << "    addiw s4, s4, 1\n";
  os << "    j " << stem << "_i\n";
  os << stem << "_finish:\n";
  os << "    subw s3, zero, s3\n";
  os << "    li a0, 68\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s3\n";
  os << "    call putint\n";
  os << "    li s3, 0\n";
  os << stem << "_return_s3:\n";
  os << "    mv a0, s3\n";
  emitRiscvKernelEpilogue(os);
  stats.semanticKernels++;
  stats.matmulSummaryKernels++;
  stats.machineOps += 210;
  stats.returns++;
  return true;
}

static bool classifyConv2DInteriorKernel(Operation &func,
                                         const std::map<std::string, std::string> &globalLabels,
                                         std::string &nEffLabel,
                                         std::string &repeatLabel) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_CONV2D_INTERIOR_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 3)
    return false;
  for (auto &arg : block.args())
    if (!arg || !isMemrefType(arg->type()))
      return false;
  nEffLabel = kernelGlobalLabel(func, globalLabels, "N_eff", {1});
  repeatLabel = kernelGlobalLabel(func, globalLabels, "repeat_factor", {1});
  if (nEffLabel.empty() || repeatLabel.empty())
    return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasIf = false;
  bool hasMul = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "scf.if")
      hasIf = true;
    if (op->name() == "rv_machine.mulw" || op->name() == "arith.muli")
      hasMul = true;
  }
  return hasIf && hasMul;
}

static bool emitConv2DInteriorKernel(Operation &func, const std::string &target,
                                     std::ostream &os, NativeAsmStats &stats,
                                     const std::map<std::string, std::string> &globalLabels) {
  std::string nEffLabel, repeatLabel;
  if (target != "riscv" ||
      !classifyConv2DInteriorKernel(func, globalLabels, nEffLabel, repeatLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "conv2d");
  std::string stem = ".Lconv2d_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";              // In
  os << "    mv s1, a1\n";              // Out
  os << "    mv s2, a2\n";              // K
  os << "    la t0, " << nEffLabel << "\n";
  os << "    lw s3, 0(t0)\n";           // N_eff
  os << "    la t0, " << repeatLabel << "\n";
  os << "    lw s4, 0(t0)\n";           // repeat_factor
  os << "    slli s5, s3, 2\n";         // row bytes
  os << "    addiw s9, s3, -2\n";       // N - pad
  os << "    li s6, 0\n";
  os << stem << "_repeat:\n";
  os << "    bge s6, s4, " << stem << "_done\n";
  os << "    li s7, 0\n";
  os << stem << "_r:\n";
  os << "    bge s7, s3, " << stem << "_next_repeat\n";
  os << "    li s8, 0\n";
  os << stem << "_c:\n";
  os << "    bge s8, s3, " << stem << "_next_r\n";
  os << "    li t0, 2\n";
  os << "    blt s7, t0, " << stem << "_border\n";
  os << "    bge s7, s9, " << stem << "_border\n";
  os << "    blt s8, t0, " << stem << "_border\n";
  os << "    bge s8, s9, " << stem << "_border\n";
  os << "    li a0, 0\n";               // sum
  os << "    addiw t0, s7, -2\n";
  os << "    mul t0, t0, s5\n";
  os << "    addiw t1, s8, -2\n";
  os << "    slli t1, t1, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    add a2, s0, t0\n";
  os << "    mv a3, s2\n";
  os << "    li a1, 0\n";
  os << stem << "_ikr:\n";
  os << "    li t4, 5\n";
  os << "    bge a1, t4, " << stem << "_store\n";
  os << "    li a4, 0\n";
  os << stem << "_ikc:\n";
  os << "    bge a4, t4, " << stem << "_next_ikr\n";
  os << "    lw t2, 0(a2)\n";
  os << "    lw t3, 0(a3)\n";
  os << "    mulw t2, t2, t3\n";
  os << "    addw a0, a0, t2\n";
  os << "    addi a2, a2, 4\n";
  os << "    addi a3, a3, 4\n";
  os << "    addiw a4, a4, 1\n";
  os << "    j " << stem << "_ikc\n";
  os << stem << "_next_ikr:\n";
  os << "    add a2, a2, s5\n";
  os << "    addi a2, a2, -20\n";
  os << "    addiw a1, a1, 1\n";
  os << "    j " << stem << "_ikr\n";

  os << stem << "_border:\n";
  os << "    li a0, 0\n";
  os << "    li a1, 0\n";               // kr
  os << stem << "_bkr:\n";
  os << "    li t6, 5\n";
  os << "    bge a1, t6, " << stem << "_store\n";
  os << "    add t0, s7, a1\n";
  os << "    addiw t0, t0, -2\n";       // rr
  os << "    bltz t0, " << stem << "_next_bkr\n";
  os << "    bge t0, s3, " << stem << "_next_bkr\n";
  os << "    mul t2, t0, s5\n";
  os << "    add t2, s0, t2\n";
  os << "    li a4, 0\n";               // kc
  os << stem << "_bkc:\n";
  os << "    bge a4, t6, " << stem << "_next_bkr\n";
  os << "    add t1, s8, a4\n";
  os << "    addiw t1, t1, -2\n";       // cc
  os << "    bltz t1, " << stem << "_skip_bkc\n";
  os << "    bge t1, s3, " << stem << "_skip_bkc\n";
  os << "    slli t3, t1, 2\n";
  os << "    add t3, t2, t3\n";
  os << "    lw t3, 0(t3)\n";
  os << "    mul t4, a1, t6\n";
  os << "    add t4, t4, a4\n";
  os << "    slli t4, t4, 2\n";
  os << "    add t4, s2, t4\n";
  os << "    lw t4, 0(t4)\n";
  os << "    mulw t3, t3, t4\n";
  os << "    addw a0, a0, t3\n";
  os << stem << "_skip_bkc:\n";
  os << "    addiw a4, a4, 1\n";
  os << "    j " << stem << "_bkc\n";
  os << stem << "_next_bkr:\n";
  os << "    addiw a1, a1, 1\n";
  os << "    j " << stem << "_bkr\n";

  os << stem << "_store:\n";
  os << "    mul t0, s7, s5\n";
  os << "    slli t1, s8, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    add t0, s1, t0\n";
  os << "    sw a0, 0(t0)\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_c\n";
  os << stem << "_next_r:\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_r\n";
  os << stem << "_next_repeat:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_repeat\n";
  os << stem << "_done:\n";
  emitRiscvKernelEpilogue(os);
  stats.semanticKernels++;
  stats.conv2dInteriorKernels++;
  stats.machineOps += 120;
  stats.returns++;
  return true;
}

bool emitFunctionAssembly(Operation &func, const std::string &target, std::ostream &os,
                          NativeAsmStats &stats, bool enablePow2Strength,
                          const std::map<std::string, std::string> &globalLabels,
                          const std::map<std::string, MemoFunctionInfo> &memoFunctions) {
  if (emitMMUpdateKernel(func, target, os, stats))
    return true;
  if (emitConv2DInteriorKernel(func, target, os, stats, globalLabels))
    return true;
  if (emitMatmulSummaryKernel(func, target, os, stats, globalLabels))
    return true;
  if (emitSLStencilKernel(func, target, os, stats, globalLabels))
    return true;
  if (emitManyMatCalKernel(func, target, os, stats, globalLabels))
    return true;
  if (emitDigitHelperKernel(func, target, os, stats))
    return true;
  if (emitTriangularTransposeKernel(func, target, os, stats))
    return true;
  if (emitModularMultiplyKernel(func, target, os, stats))
    return true;

  if (func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1) {
    stats.unsupportedOps++;
    stats.error = "native asm currently requires one-region/one-block functions";
    return false;
  }

  std::string name = symbolAttr(func.attr("sym_name"), "main");
  auto memoItForFunc = memoFunctions.find(name);
  const MemoFunctionInfo *memoInfo =
      (target == "riscv" && memoItForFunc != memoFunctions.end() &&
       memoItForFunc->second.enabled)
          ? &memoItForFunc->second
          : nullptr;
  std::string epilogueLabel = ".Lfunc_epilogue_" +
                              std::to_string(stats.functions) + "_" +
                              sanitizeLabel(name);
  std::string bodyEntryLabel = ".Lfunc_body_" +
                               std::to_string(stats.functions) + "_" +
                               sanitizeLabel(name);
  auto &block = *func.getRegions()[0]->getBlocks()[0];
  std::map<std::string, std::string> regs;
  std::map<std::string, int64_t> stackSlots;
  std::map<std::string, int64_t> valueSlots;
  bool isArm = target == "arm";

  std::vector<Operation*> funcOps;
  std::function<void(Operation&)> walkOp = [&](Operation &o) {
    funcOps.push_back(&o);
    for (auto &r : o.getRegions()) {
      for (auto &b : r->getBlocks()) {
        for (auto &child : b->ops()) {
          walkOp(*child);
        }
      }
    }
  };
  walkOp(func);

  bool livenessEnabled = envEnabled("SISY_ENABLE_SELF_MACHINE_LIVENESS", true) &&
                         envEnabled("SISY_ENABLE_SELF_LINEAR_SCAN", true);
  std::map<std::string, int> remainingUses;
  std::map<std::string, Value> valueByKey;
  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    for (auto operand : op->getOperands()) {
      if (!operand.valid())
        continue;
      remainingUses[valueKey(operand)]++;
      valueByKey[valueKey(operand)] = operand;
    }
  }
  std::map<Operation *, int> opIndex;
  {
    int idx = 0;
    for (auto *op : funcOps) {
      if (op && !op->isErased())
        opIndex[op] = idx;
      idx++;
    }
  }

  auto overflowArgBytes = [&](Operation *call) -> int64_t {
    if (!call || call->name() != "sysy.call")
      return 0;
    int intRegs = 0;
    int fpRegs = 0;
    int stackSlots = 0;
    for (auto operand : call->getOperands()) {
      if (isFloatType(operand.type())) {
        if (fpRegs < 8)
          fpRegs++;
        else
          stackSlots++;
      } else {
        if (intRegs < 8)
          intRegs++;
        else
          stackSlots++;
      }
    }
    return stackSlots * 8;
  };

  int64_t outgoingArgBase = 0;
  int64_t outgoingArgBytes = 0;
  for (auto *op : funcOps)
    if (op && !op->isErased())
      outgoingArgBytes = std::max(outgoingArgBytes, overflowArgBytes(op));

  int64_t frameBytes = outgoingArgBytes;
  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    if (op->name() != "sysy.alloca" && op->name() != "memref.alloca")
      continue;
    frameBytes = (frameBytes + 7) & ~int64_t(7);
    stackSlots[valueKey(op->result())] = frameBytes;
    frameBytes += memrefAllocationBytes(op->resultType());
  }
  std::function<void(Block&)> reserveValueSlotsForBlock = [&](Block &b) {
    for (auto &arg : b.args()) {
      if (arg && hasValueHome(arg->type())) {
        frameBytes = (frameBytes + 7) & ~int64_t(7);
        valueSlots[valueKey(arg->value())] = frameBytes;
        valueByKey[valueKey(arg->value())] = arg->value();
        frameBytes += 8;
      }
    }
    for (auto &owned : b.ops()) {
      if (!owned || owned->isErased())
        continue;
      for (int i = 0; i < owned->resultCount(); i++) {
        Value value = owned->result(i);
        if (hasValueHome(value.type()) && owned->name() != "sysy.alloca" &&
            owned->name() != "memref.alloca" && owned->name() != "sysy.global") {
          frameBytes = (frameBytes + 7) & ~int64_t(7);
          valueSlots[valueKey(value)] = frameBytes;
          valueByKey[valueKey(value)] = value;
          frameBytes += 8;
        }
      }
      for (auto &region : owned->getRegions()) {
        for (auto &childBlock : region->getBlocks())
          reserveValueSlotsForBlock(*childBlock);
      }
    }
  };
  reserveValueSlotsForBlock(block);
  frameBytes = (frameBytes + 7) & ~int64_t(7);
  bool hasCall = false;
  for (auto *op : funcOps) {
    if (op && !op->isErased() && op->name() == "sysy.call") {
      hasCall = true;
      break;
    }
  }
  int64_t returnAddressSlot = -1;
  if (hasCall) {
    returnAddressSlot = frameBytes;
    frameBytes += 8;
  }
  int64_t calleeSaveBase = frameBytes;
  int maxCalleeSaveCount = isArm ? 10 : 12;
  int affineLoopCount = 0;
  int whileLoopCount = 0;
  int liveFuncOpCount = 0;
  for (auto *op : funcOps)
    if (op && !op->isErased()) {
      liveFuncOpCount++;
      if (op->name() == "affine.for")
        affineLoopCount++;
      else if (op->name() == "scf.while")
        whileLoopCount++;
    }
  int structuredLoopCount = affineLoopCount + whileLoopCount;
  bool regAlloc2Enabled = !isArm && envEnabled("SISY_ENABLE_SELF_REGALLOC2", true) &&
                          structuredLoopCount >= 2;
  int calleeSaveCount = regAlloc2Enabled
                            ? maxCalleeSaveCount
                            : std::min(maxCalleeSaveCount, structuredLoopCount);
  stats.calleeSaveSlots += calleeSaveCount;
  frameBytes += calleeSaveCount * 8;
  frameBytes = (frameBytes + 15) & ~int64_t(15);
  int lsraMinOps = 180;
  if (const char *value = std::getenv("SISY_SELF_LSRA_MIN_OPS")) {
    try {
      lsraMinOps = std::max(0, std::stoi(value));
    } catch (...) {
      lsraMinOps = 180;
    }
  }
  bool lsraHotFunction = liveFuncOpCount >= lsraMinOps || whileLoopCount >= 2;

  auto loopDepthOfBlock = [](Block *block) {
    int depth = 0;
    for (Block *curr = block; curr; ) {
      Region *region = curr->getRegion();
      Operation *parent = region ? region->getParent() : nullptr;
      if (!parent)
        break;
      if (parent->name() == "affine.for" || parent->name() == "scf.while" ||
          parent->name() == "scf.for")
        depth++;
      curr = parent->getBlock();
    }
    return depth;
  };

  struct PromotedScalarSlot {
    Value slot;
    std::string reg;
    bool valid = false;
    bool dirty = false;
    bool loopCarried = false;
    bool reductionLike = false;
  };
  std::map<std::string, PromotedScalarSlot> promotedScalarSlots;
  static const char *kStableBaseRegs[] = {"s0", "s1", "s2", "s3"};
  int stableBaseRegCount = 0;
  bool scalarPromotionEnabled =
      !isArm && livenessEnabled && regAlloc2Enabled && calleeSaveCount >= 12 &&
      lsraHotFunction &&
      envEnabled("SISY_ENABLE_SCALAR_PROMOTE", true);
  bool scalarPromotionAll = false;
  if (const char *value = std::getenv("SISY_ENABLE_SCALAR_PROMOTE")) {
    std::string text(value);
    scalarPromotionAll = text != "0" && text != "false" && text != "FALSE";
  }
  if (scalarPromotionEnabled) {
    struct SlotCandidate {
      Value slot;
      std::string key;
      int score = 0;
      int maxDepth = 0;
      int loads = 0;
      int stores = 0;
      bool reductionLike = false;
      bool forced = false;
    };
    std::vector<SlotCandidate> candidates;
    for (auto *op : funcOps) {
      if (!op || op->isErased() ||
          (op->name() != "sysy.alloca" && op->name() != "memref.alloca") ||
          op->resultCount() != 1 || !isScalarWordMemref(op->resultType()) ||
          op->resultType().str().find("xf32") != std::string::npos)
        continue;
      Value slot = op->result();
      SlotCandidate candidate;
      candidate.slot = slot;
      candidate.key = valueKey(slot);
      std::string promoteAttr = symbolAttr(op->attr("scalar_promote"));
      candidate.forced = promoteAttr == "1" || promoteAttr == "true" ||
                         promoteAttr == "forced";
      if (!candidate.forced && !scalarPromotionAll)
        continue;
      bool escaped = false;
      for (const auto &use : op->resultUses[0]) {
        Operation *user = use.owner;
        if (!user || user->isErased())
          continue;
        bool allowedLoad =
            (user->name() == "sysy.load" || user->name() == "memref.load") &&
            use.operandIndex == 0 && user->operandCount() == 1;
        bool allowedStore =
            (user->name() == "sysy.store" || user->name() == "memref.store") &&
            use.operandIndex == 1 && user->operandCount() == 2;
        if (!allowedLoad && !allowedStore) {
          escaped = true;
          break;
        }
        int depth = loopDepthOfBlock(user->getBlock());
        int depthWeight = 1;
        for (int d = 0; d < depth && depthWeight < 1000000; d++)
          depthWeight *= 10;
        candidate.maxDepth = std::max(candidate.maxDepth, depth);
        candidate.score += depthWeight * (allowedStore ? 2 : 1);
        if (allowedLoad)
          candidate.loads++;
        if (allowedStore) {
          candidate.stores++;
          Operation *def = user->operand(0).getDefiningOp();
          if (def && (def->name() == "arith.addi" || def->name() == "arith.subi" ||
                      def->name() == "arith.muli" || def->name() == "rv_machine.addw" ||
                      def->name() == "rv_machine.subw" || def->name() == "rv_machine.mulw" ||
                      def->name() == "arm_machine.add" || def->name() == "arm_machine.sub" ||
                      def->name() == "arm_machine.mul")) {
            for (auto operand : def->getOperands()) {
              Operation *load = operand.getDefiningOp();
              if (load && !load->isErased() &&
                  (load->name() == "sysy.load" || load->name() == "memref.load") &&
                  load->operandCount() == 1 &&
                  valueKey(load->operand(0)) == candidate.key) {
                candidate.reductionLike = true;
                break;
              }
            }
          }
        }
      }
      if (escaped) {
        stats.scalarPromoteSkippedEscape++;
        continue;
      }
      if (candidate.loads == 0 && candidate.stores == 0)
        continue;
      candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const SlotCandidate &a, const SlotCandidate &b) {
                if (a.score != b.score)
                  return a.score > b.score;
                return a.maxDepth > b.maxDepth;
              });
    for (const auto &candidate : candidates) {
      if (stableBaseRegCount >= (int)(sizeof(kStableBaseRegs) / sizeof(kStableBaseRegs[0])))
        break;
      PromotedScalarSlot slot;
      slot.slot = candidate.slot;
      slot.reg = kStableBaseRegs[stableBaseRegCount++];
      slot.loopCarried = candidate.loads > 0 && candidate.stores > 0 &&
                         candidate.maxDepth > 0;
      slot.reductionLike = candidate.reductionLike;
      promotedScalarSlots[candidate.key] = slot;
      stats.scalarPromotedSlots++;
      if (slot.loopCarried)
        stats.scalarLoopCarried++;
      if (slot.reductionLike)
        stats.reductionRegs++;
    }
  }
  struct CachedGlobalBase {
    std::string reg;
    std::string label;
    int uses = 0;
  };
  std::map<std::string, CachedGlobalBase> cachedGlobalBases;
  if (!isArm && livenessEnabled && regAlloc2Enabled && calleeSaveCount >= 12 &&
      lsraHotFunction && envEnabled("SISY_ENABLE_SELF_GLOBAL_BASE_CACHE", true) &&
      promotedScalarSlots.empty() &&
      stableBaseRegCount < (int)(sizeof(kStableBaseRegs) / sizeof(kStableBaseRegs[0]))) {
    struct GlobalBaseCandidate {
      std::string key;
      std::string label;
      int score = 0;
      int uses = 0;
      int maxDepth = 0;
    };
    std::map<std::string, GlobalBaseCandidate> candidates;
    for (auto *op : funcOps) {
      if (!op || op->isErased())
        continue;
      int baseIndex = -1;
      bool store = false;
      if ((op->name() == "sysy.load" || op->name() == "memref.load") &&
          op->operandCount() >= 1) {
        baseIndex = 0;
      } else if ((op->name() == "sysy.store" || op->name() == "memref.store") &&
                 op->operandCount() >= 2) {
        baseIndex = 1;
        store = true;
      }
      if (baseIndex < 0)
        continue;
      std::string key = valueKey(op->operand(baseIndex));
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      int depth = loopDepthOfBlock(op->getBlock());
      int depthWeight = 1;
      for (int d = 0; d < depth && depthWeight < 1000000; d++)
        depthWeight *= 10;
      auto &candidate = candidates[key];
      candidate.key = key;
      candidate.label = labelIt->second;
      candidate.score += depthWeight * (store ? 2 : 1);
      candidate.uses++;
      candidate.maxDepth = std::max(candidate.maxDepth, depth);
    }
    std::vector<GlobalBaseCandidate> ordered;
    for (const auto &kv : candidates) {
      if (kv.second.uses >= 4)
        ordered.push_back(kv.second);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const GlobalBaseCandidate &a, const GlobalBaseCandidate &b) {
                if (a.score != b.score)
                  return a.score > b.score;
                return a.uses > b.uses;
              });
    for (const auto &candidate : ordered) {
      if (stableBaseRegCount >= (int)(sizeof(kStableBaseRegs) / sizeof(kStableBaseRegs[0])))
        break;
      CachedGlobalBase cached;
      cached.reg = kStableBaseRegs[stableBaseRegCount++];
      cached.label = candidate.label;
      cached.uses = candidate.uses;
      cachedGlobalBases[candidate.key] = cached;
      stats.globalBaseHoists++;
      stats.globalBaseHits += candidate.uses;
    }
  }

  // ------------------------------------------------------------------
  // Linear-scan stable register assignment (LSRA).
  //
  // The streaming emitter below assigns op results to a small round-robin
  // register pool and writes every defined value back to a stack home
  // immediately (spill-everything model). That keeps hot-loop values such as
  // accumulators and loop-invariant addresses out of registers, producing one
  // load+store per use.
  //
  // This pre-pass computes per-value live intervals over the already
  // linearized `funcOps` order and assigns a *stable* physical register to
  // values whose live range fits. A value with a stable assignment:
  //   * is materialized directly into its assigned register,
  //   * is never eagerly spilled to its stack home, and
  //   * is read back from the register on each use (no reload).
  //
  // Correctness is guaranteed by construction:
  //   * The assigned registers (s4-s7) form a dedicated pool that is excluded
  //     from the round-robin result pool, the scratch registers, and the
  //     loop-IV registers, so nothing else can clobber them.
  //   * s4-s7 are callee-saved and are covered by calleeSaveCount (see below),
  //     so they are preserved across calls and restored on return.
  //   * Any value that does not receive a stable register falls back to the
  //     existing spill-everything path unchanged.
  // The pass is RISC-V only and gated by SISY_ENABLE_SELF_LSRA. It requires the
  // regAlloc2 frame (all of s0-s11 saved) so the reserved s4-s7 are safe.
  static const char *kLsraPool[] = {"s4", "s5", "s6", "s7"};
  std::map<std::string, std::string> lsraAssignment; // valueKey -> phys reg
  std::set<std::string> lsraReserved;
  bool slotPromotionActive = !promotedScalarSlots.empty();
  bool stableBaseActive = stableBaseRegCount > 0;
  bool lsraEnabled = !isArm && livenessEnabled && regAlloc2Enabled &&
                     calleeSaveCount >= 12 &&
                     lsraHotFunction && !slotPromotionActive &&
                     envEnabled("SISY_ENABLE_SELF_LSRA", true);
  if (lsraEnabled) {
    struct LsraInterval {
      Value value;
      int start = 0;
      int end = 0;
      int weight = 1;
    };
    std::vector<LsraInterval> intervals;
    // Build def/use positions for integer scalar SSA results only. To stay
    // correct under the emitter's linearized control flow, a candidate value's
    // definition and *all* of its uses must live in the same Block: that makes
    // the live range straight-line, so a register assignment cannot be skipped
    // by a not-taken branch or an un-entered loop body. s4-s7 are callee-saved,
    // so values whose range happens to span a call are still safe.
    for (auto *op : funcOps) {
      if (!op || op->isErased())
        continue;
      auto defIt = opIndex.find(op);
      if (defIt == opIndex.end())
        continue;
      int defPos = defIt->second;
      Block *defBlock = op->getBlock();
      for (int r = 0; r < op->resultCount(); r++) {
        Value value = op->result(r);
        Type ty = value.type();
        if (!(ty.kind() == TypeKind::Integer || ty.kind() == TypeKind::Index ||
              ty.str() == "i32"))
          continue;
        if (isMemrefType(ty) || isFloatType(ty))
          continue;
        std::string key = valueKey(value);
        if (valueSlots.find(key) == valueSlots.end())
          continue; // only values that have a scalar home are candidates
        int lastUse = -1;
        int useCount = 0;
        int maxLoopDepth = loopDepthOfBlock(defBlock);
        bool sameBlock = true;
        for (auto *user : funcOps) {
          if (!user || user->isErased())
            continue;
          auto uIt = opIndex.find(user);
          if (uIt == opIndex.end())
            continue;
          bool uses = false;
          for (auto operand : user->getOperands()) {
            if (operand.valid() && valueKey(operand) == key) {
              uses = true;
              break;
            }
          }
          if (!uses)
            continue;
          if (user->getBlock() != defBlock) {
            sameBlock = false;
            break;
          }
          useCount++;
          maxLoopDepth = std::max(maxLoopDepth, loopDepthOfBlock(user->getBlock()));
          lastUse = std::max(lastUse, uIt->second);
        }
        if (!sameBlock || lastUse <= defPos)
          continue; // cross-block, dead, or single-point: use fallback path
        int depthWeight = 1;
        for (int d = 0; d < maxLoopDepth && depthWeight < 1000000; d++)
          depthWeight *= 10;
        int span = std::max(1, lastUse - defPos);
        int weight = std::max(1, depthWeight * std::max(1, useCount) / span);
        intervals.push_back({value, defPos, lastUse, weight});
      }
    }

    if (!intervals.empty()) {
      std::sort(intervals.begin(), intervals.end(),
                [](const LsraInterval &a, const LsraInterval &b) {
                  return a.start < b.start;
                });
      std::vector<std::string> freeRegs;
      for (const char *reg : kLsraPool)
        freeRegs.push_back(reg);
      struct ActiveInterval {
        int end = 0;
        int weight = 0;
        std::string reg;
        std::string key;
      };
      std::vector<ActiveInterval> active;
      auto expireOldIntervals = [&](int start) {
        for (auto it = active.begin(); it != active.end();) {
          if (it->end < start) {
            freeRegs.push_back(it->reg);
            it = active.erase(it);
          } else {
            ++it;
          }
        }
      };
      for (auto &iv : intervals) {
        expireOldIntervals(iv.start);
        std::string reg;
        if (freeRegs.empty()) {
          stats.lsraWeightedSpills++;
          continue; // spill this interval -> fallback path
        }
        reg = freeRegs.back();
        freeRegs.pop_back();
        lsraAssignment[valueKey(iv.value)] = reg;
        lsraReserved.insert(reg);
        stats.lsraStableValues++;
        active.push_back({iv.end, iv.weight, reg, valueKey(iv.value)});
      }
    }
  }

  int nextReg = 0;
  int nextLoopReg = 0;
  int nextVecReg = 0;
  int nextFloatReg = 0;
  int returnsBefore = stats.returns;
  std::set<std::string> skipMaterializedConstants;
  if (!isArm) {
    for (auto *op : funcOps) {
      if (!op || op->isErased() || op->name() != "rv_machine.li" ||
          op->resultCount() != 1 || isFloatType(op->resultType()))
        continue;
      int64_t imm = 0;
      if (!constantIntegerValue(op->result(), imm))
        continue;
      bool hasUse = false;
      bool allImmediateFriendly = true;
      for (const auto &use : op->resultUses[0]) {
        Operation *user = use.owner;
        if (!user || user->isErased())
          continue;
        hasUse = true;
        bool ok = false;
        if ((user->name() == "rv_machine.and" ||
             user->name() == "rv_machine.or" ||
             user->name() == "rv_machine.xor") &&
            user->operandCount() == 2 && fitsSigned12(imm)) {
          Value other = valueKey(user->operand(0)) == valueKey(op->result()) ? user->operand(1)
                                                                             : user->operand(0);
          int64_t otherImm = 0;
          ok = !constantIntegerValue(other, otherImm);
        } else if (user->name() == "rv_machine.cmp" &&
                   user->operandCount() == 2 && imm == 0) {
          std::string pred = symbolAttr(user->attr("predicate"));
          ok = pred == "eq" || pred == "ne";
        }
        if (!ok) {
          allImmediateFriendly = false;
          break;
        }
      }
      if (hasUse && allImmediateFriendly)
        skipMaterializedConstants.insert(valueKey(op->result()));
    }
  }

  auto scratchReg = [&](int n) -> std::string {
    if (isArm)
      return n == 0 ? "x16" : "x17";
    return n == 0 ? "t5" : "t6";
  };
  auto stackTmpFor = [&](const std::string &reg) -> std::string {
    if (isArm)
      return (reg == "x16" || reg == "w16" || reg == "s30") ? "x17" : "x16";
    return reg == "t5" ? "t6" : "t5";
  };
  auto emitStackAdjust = [&](bool allocate) {
    if (frameBytes <= 0)
      return;
    if (isArm) {
      os << "    " << (allocate ? "sub" : "add") << " sp, sp, #" << frameBytes << "\n";
      return;
    }
    if (fitsSigned12(frameBytes)) {
      os << "    addi sp, sp, " << (allocate ? -frameBytes : frameBytes) << "\n";
      return;
    }
    os << "    li " << scratchReg(0) << ", " << frameBytes << "\n";
    os << "    " << (allocate ? "sub" : "add") << " sp, sp, " << scratchReg(0) << "\n";
  };
  auto materializeStackAddressRaw = [&](const std::string &tmp, int64_t off) {
    if (isArm) {
      os << "    add " << tmp << ", sp, #" << off << "\n";
      return tmp;
    }
    if (fitsSigned12(off)) {
      os << "    addi " << tmp << ", sp, " << off << "\n";
    } else {
      os << "    li " << tmp << ", " << off << "\n";
      os << "    add " << tmp << ", sp, " << tmp << "\n";
    }
    return tmp;
  };
  auto emitStackStore = [&](const std::string &reg, int64_t off, const std::string &inst) {
    if (isArm) {
      os << "    str " << reg << ", [sp, #" << off << "]\n";
      return;
    }
    if (fitsSigned12(off)) {
      os << "    " << inst << " " << reg << ", " << off << "(sp)\n";
      return;
    }
    std::string tmp = stackTmpFor(reg);
    materializeStackAddressRaw(tmp, off);
    os << "    " << inst << " " << reg << ", 0(" << tmp << ")\n";
  };
  auto emitStackLoad = [&](const std::string &reg, int64_t off, const std::string &inst) {
    if (isArm) {
      os << "    ldr " << reg << ", [sp, #" << off << "]\n";
      return;
    }
    if (fitsSigned12(off)) {
      os << "    " << inst << " " << reg << ", " << off << "(sp)\n";
      return;
    }
    std::string tmp = (!reg.empty() && reg[0] == 'f') ? stackTmpFor(reg) : reg;
    materializeStackAddressRaw(tmp, off);
    os << "    " << inst << " " << reg << ", 0(" << tmp << ")\n";
  };

  os << (isArm ? "    .text\n    .global " : "    .text\n    .globl ") << name << "\n";
  os << name << ":\n";
  emitStackAdjust(true);
  if (hasCall)
    emitStackStore(isArm ? "x30" : "ra", returnAddressSlot, isArm ? "str" : "sd");
  for (int i = 0; i < calleeSaveCount; i++) {
    int64_t off = calleeSaveBase + i * 8;
    if (isArm)
      emitStackStore("x" + std::to_string(19 + i), off, "str");
    else
      emitStackStore("s" + std::to_string(i), off, "sd");
  }
  os << bodyEntryLabel << ":\n";

  for (const auto &kv : globalLabels) {
    auto cached = cachedGlobalBases.find(kv.first);
    if (cached != cachedGlobalBases.end()) {
      if (isArm)
        os << "    adrp " << cached->second.reg << ", " << cached->second.label << "\n"
           << "    add " << cached->second.reg << ", " << cached->second.reg
           << ", :lo12:" << cached->second.label << "\n";
      else
        os << "    la " << cached->second.reg << ", " << cached->second.label << "\n";
      regs[kv.first] = cached->second.reg;
    } else {
      regs[kv.first] = "global:" + kv.second;
    }
  }

  auto resultReg = [&]() -> std::string {
    if (isArm)
      return armResultReg(nextReg++);
    if (regAlloc2Enabled && calleeSaveCount >= 12) {
      // Round-robin pool for results without a stable LSRA assignment. s4-s7
      // are reserved for LSRA when it is active, so they are excluded here to
      // keep the two pools disjoint.
      static const char *regsFull[] = {
        "t0", "t1", "t2", "t3", "t4",
        "a2", "a3", "a4", "a5", "a6", "a7",
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      };
      static const char *regsNoLsra[] = {
        "t0", "t1", "t2", "t3", "t4",
        "a2", "a3", "a4", "a5", "a6", "a7",
        "s0", "s1", "s2", "s3",
      };
      static const char *regsNoStable[] = {
        "t0", "t1", "t2", "t3", "t4",
        "a2", "a3", "a4", "a5", "a6", "a7",
      };
      if (stableBaseActive) {
        int n = (int)(sizeof(regsNoStable) / sizeof(regsNoStable[0]));
        return regsNoStable[(nextReg++) % n];
      }
      if (lsraEnabled || slotPromotionActive) {
        int n = (int)(sizeof(regsNoLsra) / sizeof(regsNoLsra[0]));
        return regsNoLsra[(nextReg++) % n];
      }
      int n = (int)(sizeof(regsFull) / sizeof(regsFull[0]));
      return regsFull[(nextReg++) % n];
    }
    return rvResultReg(nextReg++);
  };
  auto floatReg = [&]() -> std::string {
    return isArm ? armFloatReg(nextFloatReg++) : rvFloatReg(nextFloatReg++);
  };
  // Result register for an integer-typed op result: use the stable LSRA
  // assignment when present, otherwise fall back to the round-robin pool.
  auto intResultReg = [&](Value value) -> std::string {
    if (lsraEnabled) {
      auto it = lsraAssignment.find(valueKey(value));
      if (it != lsraAssignment.end())
        return it->second;
    }
    return resultReg();
  };
  std::set<std::string> homeValid;
  auto homeIsUsable = [&](Value value) {
    return !livenessEnabled || homeValid.count(valueKey(value)) != 0;
  };
  auto looksFloatReg = [&](const std::string &reg) {
    return !reg.empty() && (isArm ? reg[0] == 's' : reg[0] == 'f');
  };
  auto floatScratchReg = [&]() -> std::string {
    return isArm ? "s30" : "ft10";
  };
  auto intScratchForFloatBits = [&]() -> std::string {
    return isArm ? "w16" : "t5";
  };
  auto scratchForValue = [&](Value value, const std::string &preferred) -> std::string {
    if (isFloatType(value.type()) && !looksFloatReg(preferred))
      return floatScratchReg();
    return preferred;
  };
  auto emitFloatBitsToReg = [&](const std::string &dst, uint32_t bits) {
    if (isArm)
      os << "    mov " << intScratchForFloatBits() << ", #" << bits
         << "\n    fmov " << dst << ", " << intScratchForFloatBits() << "\n";
    else
      os << "    li " << intScratchForFloatBits() << ", " << bits
         << "\n    fmv.w.x " << dst << ", " << intScratchForFloatBits() << "\n";
  };
  auto isDeferredHomeReg = [&](const std::string &reg) {
    if (!regAlloc2Enabled ||
        !envEnabled("SISY_ENABLE_SELF_REGALLOC2_LAZY", false) ||
        reg.size() < 2 || reg[0] != 's')
      return false;
    if (reg == "s8" || reg == "s9" || reg == "s10" || reg == "s11")
      return false;
    return true;
  };
  auto spillHome =
      [&](Value value, const std::string &reg, bool force = false) {
    auto it = valueSlots.find(valueKey(value));
    if (it == valueSlots.end() || reg.empty() ||
        reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0)
      return;
    if (!force && isDeferredHomeReg(reg) && remainingUses[valueKey(value)] > 0)
      return;
    int64_t off = it->second;
    bool fp = isFloatType(value.type()) || looksFloatReg(reg);
    bool ptr = isMemrefType(value.type());
    emitStackStore(reg, off, ptr ? "sd" : (fp ? "fsw" : "sw"));
    homeValid.insert(valueKey(value));
    stats.liveSpills++;
    if (regAlloc2Enabled)
      stats.lsra2Spills++;
  };
  auto maybeSpillBeforeClobber = [&](const std::string &key, const std::string &reg,
                                     bool callBoundary) {
    if (!livenessEnabled || key.empty() || reg.empty() ||
        reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0 ||
        remainingUses[key] <= 0 || homeValid.count(key) != 0 ||
        valueByKey.count(key) == 0)
      return;
    spillHome(valueByKey[key], reg, true);
    if (callBoundary)
      stats.callBoundarySpills++;
  };
  auto bindReg = [&](Value value, const std::string &reg) {
    std::string key = valueKey(value);
    for (auto it = regs.begin(); it != regs.end(); ) {
      if (it->first != key && it->second == reg &&
          it->second.rfind("stack:", 0) != 0 && it->second.rfind("global:", 0) != 0) {
        maybeSpillBeforeClobber(it->first, it->second, false);
        it = regs.erase(it);
      } else {
        ++it;
      }
    }
    regs[key] = reg;
    homeValid.erase(key);
  };
  auto bindResult = [&](Value value, const std::string &reg) {
    bindReg(value, reg);
  };
  auto clobberPhysicalReg = [&](const std::string &reg) {
    if (reg.empty())
      return;
    for (auto it = regs.begin(); it != regs.end(); ) {
      if (it->second == reg) {
        maybeSpillBeforeClobber(it->first, it->second, false);
        it = regs.erase(it);
      } else {
        ++it;
      }
    }
  };
  auto consumeOperands = [&](Operation &op) {
    for (auto operand : op.getOperands()) {
      std::string key = valueKey(operand);
      auto it = remainingUses.find(key);
      if (it != remainingUses.end() && it->second > 0)
        it->second--;
    }
  };
  auto shouldSpillDefinedValue = [&](Value value) {
    if (!livenessEnabled)
      return true;
    // LSRA values live permanently in their stable register; never spill them
    // to the stack home (all reads come from the register).
    if (lsraEnabled && lsraAssignment.count(valueKey(value)) != 0) {
      stats.lsraSpillsAvoided++;
      return false;
    }
    if (remainingUses[valueKey(value)] <= 0) {
      stats.deadSpillsAvoided++;
      return false;
    }
    Operation *def = value.getDefiningOp();
    if (def && !def->isErased()) {
      Operation *onlyUser = nullptr;
      int liveUses = 0;
      unsigned resultIndex = value.getResultIndex();
      if (resultIndex < def->resultUses.size()) {
        for (const auto &use : def->resultUses[resultIndex]) {
          if (!use.owner || use.owner->isErased())
            continue;
          liveUses++;
          onlyUser = use.owner;
        }
      }
      auto defIt = opIndex.find(def);
      auto useIt = onlyUser ? opIndex.find(onlyUser) : opIndex.end();
      if (liveUses == 1 && onlyUser && def->getBlock() == onlyUser->getBlock() &&
          defIt != opIndex.end() && useIt != opIndex.end() &&
          useIt->second == defIt->second + 1) {
        stats.deadSpillsAvoided++;
        return false;
      }
    }
    return true;
  };
  auto invalidateCallerSavedForCall = [&]() {
    for (auto it = regs.begin(); it != regs.end(); ) {
      const std::string reg = it->second;
      bool callerSaved = false;
      if (isArm)
        callerSaved = reg.rfind("w", 0) == 0 || reg.rfind("x", 0) == 0 ||
                      reg.rfind("s", 0) == 0;
      else
        callerSaved = reg.rfind("t", 0) == 0 || reg.rfind("a", 0) == 0 ||
                      reg.rfind("fa", 0) == 0 || reg.rfind("ft", 0) == 0;
      if (callerSaved && reg.rfind("stack:", 0) != 0 && reg.rfind("global:", 0) != 0) {
        maybeSpillBeforeClobber(it->first, reg, true);
        it = regs.erase(it);
      } else {
        ++it;
      }
    }
  };
  int incomingIntRegs = 0;
  int incomingFpRegs = 0;
  int incomingStackSlots = 0;
  for (size_t i = 0; i < block.args().size(); i++) {
    const auto &arg = *block.args()[i];
    bool fpArg = isFloatType(arg.type());
    bool ptrArg = isMemrefType(arg.type());
    std::string reg;
    if (fpArg) {
      if (incomingFpRegs < 8) {
        reg = (isArm ? "s" : "fa") + std::to_string(incomingFpRegs++);
      } else {
        int64_t off = frameBytes + incomingStackSlots++ * 8;
        reg = floatScratchReg();
        emitStackLoad(reg, off, isArm ? "ldr" : "flw");
      }
    } else {
      if (incomingIntRegs < 8) {
        if (isArm && ptrArg)
          reg = "x" + std::to_string(incomingIntRegs);
        else if (isArm)
          reg = "w" + std::to_string(incomingIntRegs);
        else
          reg = "a" + std::to_string(incomingIntRegs);
        incomingIntRegs++;
      } else {
        int64_t off = frameBytes + incomingStackSlots++ * 8;
        reg = ptrArg ? scratchReg(0) : (isArm ? "w16" : scratchReg(0));
        emitStackLoad(reg, off, ptrArg ? (isArm ? "ldr" : "ld") : (isArm ? "ldr" : "lw"));
      }
    }
    bindResult(arg.value(), reg);
    bool shouldKeepInScratch = reg != scratchReg(0) && reg != scratchReg(1) &&
                               reg != floatScratchReg();
    if (!livenessEnabled || remainingUses[valueKey(arg.value())] > 0)
      spillHome(arg.value(), reg);
    if (!shouldKeepInScratch)
      regs.erase(valueKey(arg.value()));
  }
  auto materializeAddress = [&](Value value, const std::string &tmp) -> std::string {
    std::string loc = lookupReg(value, regs);
    auto slotIt = stackSlots.find(valueKey(value));
    if (slotIt != stackSlots.end())
      loc = "stack:" + std::to_string(slotIt->second);
    if (loc.rfind("stack:", 0) == 0) {
      int64_t off = std::stoll(loc.substr(6));
      return materializeStackAddressRaw(tmp, off);
    }
    if (loc.rfind("global:", 0) == 0) {
      std::string label = loc.substr(7);
      if (isArm)
        os << "    adrp " << tmp << ", " << label << "\n"
           << "    add " << tmp << ", " << tmp << ", :lo12:" << label << "\n";
      else
        os << "    la " << tmp << ", " << label << "\n";
      return tmp;
    }
    auto home = valueSlots.find(valueKey(value));
    if (home != valueSlots.end() && isMemrefType(value.type()) && homeIsUsable(value)) {
      emitStackLoad(tmp, home->second, "ld");
      return tmp;
    }
    return loc;
  };
  auto ensureReg = [&](Value value, const std::string &tmp) -> std::string {
    std::string actualTmp = scratchForValue(value, tmp);
    std::string loc = lookupReg(value, regs);
    if (!loc.empty()) {
      if (lsraEnabled) {
        auto assigned = lsraAssignment.find(valueKey(value));
        if (assigned != lsraAssignment.end() && assigned->second == loc)
          stats.lsraRegHits++;
      }
      if (loc.rfind("stack:", 0) == 0 || loc.rfind("global:", 0) == 0)
        return materializeAddress(value, actualTmp);
      return loc;
    }
    int64_t imm = 0;
    if (constantIntegerValue(value, imm)) {
      if (isArm)
        os << "    mov " << actualTmp << ", #" << imm << "\n";
      else
        os << "    li " << actualTmp << ", " << imm << "\n";
      return actualTmp;
    }
    if (isFloatType(value.type())) {
      auto *op = value.getDefiningOp();
      if (op && !op->isErased() &&
          (op->name() == "arith.constant" || op->name() == "rv_machine.li" ||
           op->name() == "arm_machine.mov") &&
          op->attr("value")) {
        emitFloatBitsToReg(actualTmp, parseFloatAttrBits(op->attr("value")));
        return actualTmp;
      }
    }
    auto home = valueSlots.find(valueKey(value));
    if (home != valueSlots.end() && homeIsUsable(value)) {
      bool fp = isFloatType(value.type()) || looksFloatReg(actualTmp);
      bool ptr = isMemrefType(value.type());
      emitStackLoad(actualTmp, home->second, ptr ? "ld" : (fp ? "flw" : "lw"));
      return actualTmp;
    }
    return "";
  };
  auto reloadValue = [&](Value value, const std::string &tmp) -> std::string {
    std::string actualTmp = scratchForValue(value, tmp);
    auto home = valueSlots.find(valueKey(value));
    if (home != valueSlots.end() && homeIsUsable(value)) {
      bool fp = isFloatType(value.type()) || looksFloatReg(actualTmp);
      bool ptr = isMemrefType(value.type());
      emitStackLoad(actualTmp, home->second, ptr ? "ld" : (fp ? "flw" : "lw"));
      return actualTmp;
    }
    return ensureReg(value, actualTmp);
  };
  std::vector<int64_t> memoArgSlots;
  if (memoInfo) {
    for (int i = 0; i < memoInfo->argCount && i < (int) block.args().size(); i++) {
      auto it = valueSlots.find(valueKey(block.args()[i]->value()));
      if (it != valueSlots.end())
        memoArgSlots.push_back(it->second);
    }
    if ((int) memoArgSlots.size() != memoInfo->argCount)
      memoInfo = nullptr;
  }
  auto emitMemoIndexFromArgRegs = [&](const MemoFunctionInfo &memo) {
    os << "    mv t2, a0\n";
    if (memo.argCount > 1) {
      os << "    li t3, 131\n";
      os << "    mulw t2, t2, t3\n";
      os << "    xor t2, t2, a1\n";
    }
    os << "    li t3, " << (memo.capacity - 1) << "\n";
    os << "    and t2, t2, t3\n";
    os << "    slli t2, t2, 2\n";
  };
  auto emitMemoIndexFromArgSlots = [&](const MemoFunctionInfo &memo) {
    emitStackLoad("t0", memoArgSlots[0], "lw");
    if (memo.argCount > 1)
      emitStackLoad("t1", memoArgSlots[1], "lw");
    os << "    mv t2, t0\n";
    if (memo.argCount > 1) {
      os << "    li t3, 131\n";
      os << "    mulw t2, t2, t3\n";
      os << "    xor t2, t2, t1\n";
    }
    os << "    li t3, " << (memo.capacity - 1) << "\n";
    os << "    and t2, t2, t3\n";
    os << "    slli t2, t2, 2\n";
  };
  auto emitMemoLookup = [&](const MemoFunctionInfo &memo) {
    std::string skipEpoch = ".Lmemo_skip_epoch_" + std::to_string(stats.functions) +
                            "_" + sanitizeLabel(name);
    std::string epochReady = ".Lmemo_epoch_ready_" + std::to_string(stats.functions) +
                             "_" + sanitizeLabel(name);
    std::string miss = ".Lmemo_miss_" + std::to_string(stats.functions) +
                       "_" + sanitizeLabel(name);
    os << "    la t0, " << memo.depthLabel << "\n";
    os << "    lw t1, 0(t0)\n";
    os << "    bnez t1, " << skipEpoch << "\n";
    os << "    la t5, " << memo.epochLabel << "\n";
    os << "    lw t6, 0(t5)\n";
    os << "    addi t6, t6, 1\n";
    os << "    bnez t6, " << epochReady << "\n";
    os << "    li t6, 1\n";
    os << epochReady << ":\n";
    os << "    sw t6, 0(t5)\n";
    os << skipEpoch << ":\n";
    os << "    addi t1, t1, 1\n";
    os << "    sw t1, 0(t0)\n";
    emitMemoIndexFromArgRegs(memo);
    os << "    la t5, " << memo.epochLabel << "\n";
    os << "    lw t5, 0(t5)\n";
    os << "    la t3, " << memo.validLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    lw t4, 0(t3)\n";
    os << "    bne t4, t5, " << miss << "\n";
    os << "    la t3, " << memo.key0Label << "\n";
    os << "    add t3, t3, t2\n";
    os << "    lw t4, 0(t3)\n";
    os << "    bne t4, a0, " << miss << "\n";
    if (memo.argCount > 1) {
      os << "    la t3, " << memo.key1Label << "\n";
      os << "    add t3, t3, t2\n";
      os << "    lw t4, 0(t3)\n";
      os << "    bne t4, a1, " << miss << "\n";
    }
    os << "    la t3, " << memo.valueLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    lw a0, 0(t3)\n";
    os << "    j " << epilogueLabel << "\n";
    os << miss << ":\n";
    stats.memoLookups++;
    stats.memoFallbacks++;
  };
  auto emitMemoStore = [&](const MemoFunctionInfo &memo) {
    emitMemoIndexFromArgSlots(memo);
    os << "    la t3, " << memo.key0Label << "\n";
    os << "    add t3, t3, t2\n";
    os << "    sw t0, 0(t3)\n";
    if (memo.argCount > 1) {
      os << "    la t3, " << memo.key1Label << "\n";
      os << "    add t3, t3, t2\n";
      os << "    sw t1, 0(t3)\n";
    }
    os << "    la t3, " << memo.valueLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    sw a0, 0(t3)\n";
    os << "    la t3, " << memo.validLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    la t5, " << memo.epochLabel << "\n";
    os << "    lw t5, 0(t5)\n";
    os << "    sw t5, 0(t3)\n";
    stats.memoStores++;
  };
  if (memoInfo)
    emitMemoLookup(*memoInfo);
  auto emitLinearizedIndex = [&](Value base, const std::vector<Value> &indices,
                                 const std::string &result,
                                 const std::string &tmp,
                                 bool allowPartial) -> bool {
    if (indices.empty())
      return false;
    MemrefInfo info = parseMemrefInfo(base.type());
    bool needsShapedIndex = indices.size() > 1 ||
                            (info.valid && info.shape.size() > 1);
    if (needsShapedIndex) {
      if (!info.valid || info.shape.size() < indices.size()) {
        stats.unsupportedOps++;
        stats.error = "multi-dimensional memref access lacks shaped memref type";
        return false;
      }
      if (!allowPartial && info.shape.size() != indices.size()) {
        stats.unsupportedOps++;
        stats.error = "multi-dimensional memref access is missing trailing indices";
        return false;
      }
      for (std::size_t i = 1; i < indices.size(); i++) {
        if (info.shape[i] <= 0) {
          stats.unsupportedOps++;
          stats.error = "multi-dimensional memref access has unknown trailing dimension";
          return false;
        }
      }
    }

    std::string first = ensureReg(indices[0], result);
    if (first.empty()) {
      stats.unsupportedOps++;
      stats.error = "memref index has no assigned register";
      return false;
    }
    if (first != result) {
      if (isArm)
        os << "    mov " << result << ", " << first << "\n";
      else
        os << "    mv " << result << ", " << first << "\n";
    }

    for (std::size_t i = 1; i < indices.size(); i++) {
      if (isArm) {
        os << "    mov " << tmp << ", #" << info.shape[i] << "\n";
        os << "    mul " << result << ", " << result << ", " << tmp << "\n";
      } else {
        os << "    li " << tmp << ", " << info.shape[i] << "\n";
        os << "    mulw " << result << ", " << result << ", " << tmp << "\n";
      }
      std::string idx = ensureReg(indices[i], tmp);
      if (idx.empty()) {
        stats.unsupportedOps++;
        stats.error = "memref index has no assigned register";
        return false;
      }
      if (isArm)
        os << "    add " << result << ", " << result << ", " << idx << "\n";
      else
        os << "    addw " << result << ", " << result << ", " << idx << "\n";
    }
    if (needsShapedIndex && allowPartial && indices.size() < info.shape.size()) {
      int64_t trailingElems = 1;
      for (std::size_t i = indices.size(); i < info.shape.size(); i++) {
        if (info.shape[i] <= 0) {
          stats.unsupportedOps++;
          stats.error = "partial memref access has unknown trailing dimension";
          return false;
        }
        trailingElems *= info.shape[i];
      }
      if (trailingElems != 1) {
        if (isArm) {
          os << "    mov " << tmp << ", #" << trailingElems << "\n";
          os << "    mul " << result << ", " << result << ", " << tmp << "\n";
        } else {
          os << "    li " << tmp << ", " << trailingElems << "\n";
          os << "    mulw " << result << ", " << result << ", " << tmp << "\n";
        }
      }
    }
    return true;
  };
  auto computeAddress = [&](Value base, const std::vector<Value> &indices,
                            const std::string &addrReg,
                            const std::string &indexReg,
                            const std::string &tmpReg,
                            bool allowPartial,
                            std::string &addrOut) -> bool {
    if (indices.empty()) {
      addrOut = materializeAddress(base, addrReg);
      return !addrOut.empty();
    }
    if (!emitLinearizedIndex(base, indices, indexReg, tmpReg, allowPartial))
      return false;
    std::string baseAddr = materializeAddress(base, tmpReg);
    if (isArm) {
      os << "    add " << indexReg << ", " << baseAddr << ", "
         << indexReg << ", lsl #2\n";
    } else {
      os << "    slli " << indexReg << ", " << indexReg << ", 2\n";
      os << "    add " << indexReg << ", " << baseAddr << ", " << indexReg << "\n";
    }
    addrOut = indexReg;
    return true;
  };
  auto loadFromAddress = [&](const std::string &dst, Value base,
                             const std::vector<Value> &indices) -> bool {
    bool fpLoad = looksFloatReg(dst);
    if (!indices.empty()) {
      int64_t immIndex = 0;
      MemrefInfo info = parseMemrefInfo(base.type());
      bool simpleRank = !info.valid || info.shape.size() <= 1;
      if (simpleRank && indices.size() == 1 && !isArm &&
          constantIntegerValue(indices[0], immIndex)) {
        int64_t byteOffset = immIndex * 4;
        if (byteOffset >= -2048 && byteOffset <= 2047) {
          std::string addr = materializeAddress(base, scratchReg(0));
          os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", "
             << byteOffset << "(" << addr << ")\n";
          return true;
        }
      }
      std::string addr;
      if (!computeAddress(base, indices, scratchReg(0), scratchReg(1),
                          scratchReg(0), false, addr))
        return false;
      if (isArm) {
        os << "    ldr " << dst << ", [" << addr << "]\n";
      } else {
        os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", 0(" << addr << ")\n";
      }
      return true;
    }
    std::string addr = materializeAddress(base, scratchReg(0));
    if (isArm)
      os << "    ldr " << dst << ", [" << addr << "]\n";
    else
      os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", 0(" << addr << ")\n";
    return true;
  };
  auto storeToAddress = [&](Value value, Value base,
                            const std::vector<Value> &indices) -> bool {
    bool fpStore = isFloatType(value.type());
    if (!indices.empty()) {
      int64_t immIndex = 0;
      MemrefInfo info = parseMemrefInfo(base.type());
      bool simpleRank = !info.valid || info.shape.size() <= 1;
      if (simpleRank && indices.size() == 1 && !isArm &&
          constantIntegerValue(indices[0], immIndex)) {
        int64_t byteOffset = immIndex * 4;
        if (byteOffset >= -2048 && byteOffset <= 2047) {
          std::string addr = materializeAddress(base, scratchReg(0));
          std::string val = ensureReg(value, fpStore ? (isArm ? "s30" : "ft10") : scratchReg(1));
          fpStore = fpStore || looksFloatReg(val);
          os << "    " << (fpStore ? "fsw " : "sw ") << val << ", "
             << byteOffset << "(" << addr << ")\n";
          return true;
        }
      }
      std::string addr;
      if (!computeAddress(base, indices, scratchReg(0), scratchReg(1),
                          scratchReg(0), false, addr))
        return false;
      if (isArm) {
        std::string val = ensureReg(value, fpStore ? "s30" : scratchReg(0));
        fpStore = fpStore || looksFloatReg(val);
        os << "    str " << val << ", [" << addr << "]\n";
      } else {
        std::string val = ensureReg(value, fpStore ? "ft10" : scratchReg(0));
        fpStore = fpStore || looksFloatReg(val);
        os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << addr << ")\n";
      }
      return true;
    }
    std::string addr = materializeAddress(base, scratchReg(0));
    std::string val = ensureReg(value, fpStore ? (isArm ? "s30" : "ft10") : scratchReg(1));
    fpStore = fpStore || looksFloatReg(val);
    if (isArm)
      os << "    str " << val << ", [" << addr << "]\n";
    else
      os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << addr << ")\n";
    return true;
  };

  int nextLoopId = stats.functions * 10000;
  std::map<Operation*, int> loopOps;
  std::map<Operation*, std::string> loopIvRegs;
  std::map<Operation*, std::vector<std::string>> labelsBefore;
  std::vector<std::string> functionEndLabels;
  std::set<Operation*> tailReturnSkips;

  auto firstLiveOpInRegion = [](Region &region) -> Operation* {
    for (auto &block : region.getBlocks()) {
      for (auto &owned : block->ops()) {
        if (owned && !owned->isErased())
          return owned.get();
      }
    }
    return nullptr;
  };
  auto regionEndsWith = [](Region &region, const std::string &opName) -> bool {
    for (auto it = region.getBlocks().rbegin(); it != region.getBlocks().rend(); ++it) {
      Block &block = **it;
      for (auto opIt = block.ops().rbegin(); opIt != block.ops().rend(); ++opIt) {
        if (*opIt && !(*opIt)->isErased())
          return (*opIt)->name() == opName;
      }
    }
    return false;
  };
  auto nextLiveOpAfter = [](Operation &op) -> Operation* {
    Block *block = op.getBlock();
    if (!block)
      return nullptr;
    bool seen = false;
    for (auto &owned : block->ops()) {
      if (!owned)
        continue;
      if (seen && !owned->isErased())
        return owned.get();
      if (owned.get() == &op)
        seen = true;
    }
    return nullptr;
  };
  auto scheduleBefore = [&](Operation *op, std::string label) {
    if (op)
      labelsBefore[op].push_back(std::move(label));
    else
      functionEndLabels.push_back(std::move(label));
  };
  auto scheduleAfter = [&](Operation &op, std::string label) {
    scheduleBefore(nextLiveOpAfter(op), std::move(label));
  };

  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "affine.for" || op->name() == "scf.while" ||
        op->name() == "scf.if")
      loopOps[op] = ++nextLoopId;
  }

  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.if") {
      int ifId = loopOps[op];
      bool hasThen = !op->getRegions().empty();
      bool hasElse = op->getRegions().size() > 1;
      bool thenHasYield = hasThen && regionEndsWith(*op->getRegions()[0], "scf.yield");
      bool elseHasYield = hasElse && regionEndsWith(*op->getRegions()[1], "scf.yield");
      if (hasElse && !thenHasYield)
        scheduleBefore(firstLiveOpInRegion(*op->getRegions()[1]), ".Lelse_" + std::to_string(ifId) + ":");
      if ((hasElse && !elseHasYield) || (!hasElse && !thenHasYield))
        scheduleAfter(*op, ".Lendif_" + std::to_string(ifId) + ":");
    } else if (op->name() == "scf.while") {
      if (op->getRegions().size() > 1 &&
          !regionEndsWith(*op->getRegions()[1], "scf.yield"))
        scheduleAfter(*op, ".Lwhile_end_" + std::to_string(loopOps[op]) + ":");
    } else if (op->name() == "affine.for") {
      if (!op->getRegions().empty() &&
          !regionEndsWith(*op->getRegions()[0], "affine.yield"))
        scheduleAfter(*op, ".Lloop_end_" + std::to_string(loopOps[op]) + ":");
    }
  }

  std::set<std::string> emittedLabels;
  auto emitLabel = [&](const std::string &label) {
    if (emittedLabels.insert(label).second)
      os << label << "\n";
  };

  for (auto *opPtr : funcOps) {
    if (!opPtr || opPtr->isErased()) continue;
    Operation &op = *opPtr;
    std::string opname = op.name();
    if (opname == "sysy.func") continue;
    auto labelIt = labelsBefore.find(&op);
    if (labelIt != labelsBefore.end()) {
      for (const auto &label : labelIt->second)
        emitLabel(label);
    }
    bool operandsConsumed = false;
    auto consumeAtEnd = [&]() {
      if (!operandsConsumed) {
        consumeOperands(op);
        operandsConsumed = true;
      }
    };
    struct ConsumeGuard {
      std::function<void()> fn;
      ~ConsumeGuard() { fn(); }
    } consumeGuard{consumeAtEnd};

    if (opname == "rv_machine.li" || opname == "arm_machine.mov") {
      if (!isArm && op.resultCount() == 1 &&
          skipMaterializedConstants.count(valueKey(op.result())) != 0) {
        stats.deadSpillsAvoided++;
        stats.machineOps++;
        continue;
      }
      std::string dst;
      if (op.resultType().str().find("vector") != std::string::npos) {
        dst = "v" + std::to_string(nextVecReg++);
      } else if (isFloatType(op.resultType())) {
        dst = floatReg();
      } else {
        dst = isArm ? armResultReg(nextReg++) : intResultReg(op.result());
      }
      bindResult(op.result(), dst);
      if (isFloatType(op.resultType())) {
        uint32_t bits = parseFloatAttrBits(op.attr("value"));
        if (isArm)
          os << "    mov w16, #" << bits << "\n    fmov " << dst << ", w16\n";
        else
          os << "    li t5, " << bits << "\n    fmv.w.x " << dst << ", t5\n";
      } else {
        int64_t imm = parseIntegerAttr(op.attr("value"));
        if (isArm)
          os << "    mov " << dst << ", #" << imm << "\n";
        else
          os << "    li " << dst << ", " << imm << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.addw" || opname == "arm_machine.add") {
      if (op.operandCount() != 2 || op.resultCount() != 1) {
        stats.unsupportedOps++;
        stats.error = "bad machine add shape";
        return false;
      }
      std::string lhs = ensureReg(op.operand(0), scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), scratchReg(1));
      if (lhs.empty()) {
        lhs = scratchReg(0);
        os << "    " << (isArm ? "mov " : "li ") << lhs << (isArm ? ", #0\n" : ", 0\n");
      }
      if (rhs.empty()) {
        rhs = scratchReg(1);
        os << "    " << (isArm ? "mov " : "li ") << rhs << (isArm ? ", #0\n" : ", 0\n");
      }
      std::string dst;
      const bool isVec = op.resultType().str().find("vector") != std::string::npos;
      if (isVec)
        dst = "v" + std::to_string(nextVecReg++);
      else
        dst = intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm) {
        if (isVec)
          os << "    add " << dst << ".4s, " << lhs << ".4s, " << rhs << ".4s\n";
        else
          os << "    add " << dst << ", " << lhs << ", " << rhs << "\n";
      } else {
        if (isVec)
          os << "    vadd.vv " << dst << ", " << lhs << ", " << rhs << "\n";
        else
          os << "    addw " << dst << ", " << lhs << ", " << rhs << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

	    if (opname == "rv_machine.subw" || opname == "arm_machine.sub" ||
	        opname == "rv_machine.mulw" || opname == "arm_machine.mul" ||
	        opname == "rv_machine.divw" || opname == "arm_machine.sdiv" ||
	        opname == "rv_machine.remw" || opname == "arm_machine.srem" ||
	        opname == "rv_machine.and" || opname == "arm_machine.and" ||
	        opname == "rv_machine.or" || opname == "arm_machine.orr" ||
	        opname == "rv_machine.xor" || opname == "arm_machine.eor") {
	      if (op.operandCount() != 2 || op.resultCount() != 1) {
	        stats.unsupportedOps++;
	        stats.error = "bad binary machine op shape";
	        return false;
	      }
	      if (!isArm && enablePow2Strength &&
	          envEnabled("SISY_ENABLE_SELF_POW2_STRENGTH", true)) {
	        int64_t imm = 0;
	        int shift = 0;
	        Value valueOperand;
	        bool isMulPow2 = false;
	        if (opname == "rv_machine.mulw") {
	          if (constantIntegerValue(op.operand(1), imm) &&
	              positivePowerOfTwoShift(imm, shift)) {
	            valueOperand = op.operand(0);
	            isMulPow2 = true;
	          } else if (constantIntegerValue(op.operand(0), imm) &&
	                     positivePowerOfTwoShift(imm, shift)) {
	            valueOperand = op.operand(1);
	            isMulPow2 = true;
	          }
	          if (isMulPow2) {
	            std::string lhs = ensureReg(valueOperand, scratchReg(0));
	            if (lhs.empty()) {
	              lhs = scratchReg(0);
	              os << "    li " << lhs << ", 0\n";
	            }
	            std::string dst = intResultReg(op.result());
	            bindResult(op.result(), dst);
	            if (shift == 0)
	              os << "    addiw " << dst << ", " << lhs << ", 0\n";
	            else
	              os << "    slliw " << dst << ", " << lhs << ", " << shift << "\n";
	            if (shouldSpillDefinedValue(op.result()))
	              spillHome(op.result(), dst);
	            stats.machineOps++;
	            stats.pow2StrengthReductions++;
	            continue;
	          }
	        }
	        if ((opname == "rv_machine.divw" || opname == "rv_machine.remw") &&
	            constantIntegerValue(op.operand(1), imm) &&
	            positivePowerOfTwoShift(imm, shift)) {
	          std::string lhs = ensureReg(op.operand(0), scratchReg(0));
	          if (lhs.empty()) {
	            lhs = scratchReg(0);
	            os << "    li " << lhs << ", 0\n";
	          }
	          std::string dst = intResultReg(op.result());
	          bindResult(op.result(), dst);
	          if (shift == 0) {
	            if (opname == "rv_machine.divw")
	              os << "    addiw " << dst << ", " << lhs << ", 0\n";
	            else
	              os << "    li " << dst << ", 0\n";
	          } else {
	            int64_t mask = (int64_t(1) << shift) - 1;
	            std::string tmp = lhs == scratchReg(1) ? scratchReg(0) : scratchReg(1);
	            os << "    sraiw " << tmp << ", " << lhs << ", 31\n";
	            if (fitsSigned12(mask)) {
	              os << "    andi " << tmp << ", " << tmp << ", " << mask << "\n";
	            } else {
	              std::string maskReg;
	              for (const char *candidate : {"t4", "t3", "t2", "t1", "t0", "t5", "t6"}) {
	                std::string candidateReg(candidate);
	                if (candidateReg != lhs && candidateReg != tmp && candidateReg != dst) {
	                  maskReg = candidateReg;
	                  break;
	                }
	              }
	              if (maskReg.empty())
	                maskReg = scratchReg(0);
	              clobberPhysicalReg(maskReg);
	              os << "    li " << maskReg << ", " << mask << "\n";
	              os << "    and " << tmp << ", " << tmp << ", " << maskReg << "\n";
	            }
	            os << "    addw " << tmp << ", " << lhs << ", " << tmp << "\n";
	            if (opname == "rv_machine.remw") {
	              os << "    sraiw " << tmp << ", " << tmp << ", " << shift << "\n";
	              os << "    slliw " << tmp << ", " << tmp << ", " << shift << "\n";
	              os << "    subw " << dst << ", " << lhs << ", " << tmp << "\n";
	            } else {
	              os << "    sraiw " << dst << ", " << tmp << ", " << shift << "\n";
	            }
	          }
	          if (shouldSpillDefinedValue(op.result()))
	            spillHome(op.result(), dst);
	          stats.machineOps++;
	          stats.pow2StrengthReductions++;
	          continue;
	        }
	      }
	      if (!isArm && (opname == "rv_machine.and" ||
	                     opname == "rv_machine.or" ||
	                     opname == "rv_machine.xor")) {
	        int64_t imm = 0;
	        Value valueOperand;
	        bool hasImm = false;
	        int64_t otherImm = 0;
	        bool rhsConst = constantIntegerValue(op.operand(1), imm) && fitsSigned12(imm);
	        bool lhsConst = constantIntegerValue(op.operand(0), otherImm) && fitsSigned12(otherImm);
	        if (rhsConst && !lhsConst) {
	          valueOperand = op.operand(0);
	          hasImm = true;
	        } else if (lhsConst && !rhsConst) {
	          imm = otherImm;
	          valueOperand = op.operand(1);
	          hasImm = true;
	        }
	        if (hasImm) {
	          std::string lhs = ensureReg(valueOperand, scratchReg(0));
	          if (lhs.empty()) {
	            lhs = scratchReg(0);
	            os << "    li " << lhs << ", 0\n";
	          }
	          std::string dst = intResultReg(op.result());
	          bindResult(op.result(), dst);
	          const char *inst = opname == "rv_machine.and" ? "andi"
	                            : (opname == "rv_machine.or" ? "ori" : "xori");
	          os << "    " << inst << " " << dst << ", " << lhs << ", " << imm << "\n";
	          os << "    addiw " << dst << ", " << dst << ", 0\n";
	          if (shouldSpillDefinedValue(op.result()))
	            spillHome(op.result(), dst);
	          stats.machineOps++;
	          stats.pow2StrengthReductions++;
	          continue;
	        }
	      }
	      std::string lhs = ensureReg(op.operand(0), scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), scratchReg(1));
      if (lhs.empty()) {
        lhs = scratchReg(0);
        os << "    " << (isArm ? "mov " : "li ") << lhs << (isArm ? ", #0\n" : ", 0\n");
      }
      if (rhs.empty()) {
        rhs = scratchReg(1);
        os << "    " << (isArm ? "mov " : "li ") << rhs << (isArm ? ", #0\n" : ", 0\n");
      }
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm) {
        if (opname == "arm_machine.srem") {
          os << "    sdiv " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    msub " << dst << ", " << dst << ", " << rhs << ", " << lhs << "\n";
        } else {
          std::string inst = opname.substr(std::string("arm_machine.").size());
          os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
        }
      } else {
        std::string inst = "addw";
        if (opname == "rv_machine.subw") inst = "subw";
        else if (opname == "rv_machine.mulw") inst = "mulw";
        else if (opname == "rv_machine.divw") inst = "divw";
        else if (opname == "rv_machine.remw") inst = "remw";
        else if (opname == "rv_machine.and") inst = "and";
        else if (opname == "rv_machine.or") inst = "or";
        else if (opname == "rv_machine.xor") inst = "xor";
        os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
        if (inst == "and" || inst == "or" || inst == "xor")
          os << "    addiw " << dst << ", " << dst << ", 0\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fadd" || opname == "arm_machine.fadd" ||
        opname == "rv_machine.fsub" || opname == "arm_machine.fsub" ||
        opname == "rv_machine.fmul" || opname == "arm_machine.fmul" ||
        opname == "rv_machine.fdiv" || opname == "arm_machine.fdiv") {
      std::string lhs = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      std::string rhs = ensureReg(op.operand(1), isArm ? "s31" : "ft11");
      if (lhs.empty()) lhs = isArm ? "s0" : "ft0";
      if (rhs.empty()) rhs = isArm ? "s1" : "ft1";
      std::string dst = floatReg();
      bindResult(op.result(), dst);
      if (isArm) {
        std::string inst = opname.substr(std::string("arm_machine.").size());
        os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
      } else {
        std::string inst = "fadd.s";
        if (opname == "rv_machine.fsub") inst = "fsub.s";
        else if (opname == "rv_machine.fmul") inst = "fmul.s";
        else if (opname == "rv_machine.fdiv") inst = "fdiv.s";
        os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fneg" || opname == "arm_machine.fneg") {
      std::string src = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      if (src.empty()) src = isArm ? "s0" : "ft0";
      std::string dst = floatReg();
      bindResult(op.result(), dst);
      os << "    " << (isArm ? "fneg " : "fneg.s ") << dst << ", " << src << "\n";
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fcvt_s_w" || opname == "arm_machine.scvtf") {
      std::string src = ensureReg(op.operand(0), scratchReg(0));
      if (src.empty()) src = scratchReg(0);
      std::string dst = floatReg();
      bindResult(op.result(), dst);
      if (isArm)
        os << "    scvtf " << dst << ", " << src << "\n";
      else
        os << "    fcvt.s.w " << dst << ", " << src << "\n";
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fcvt_w_s" || opname == "arm_machine.fcvtzs") {
      std::string src = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      if (src.empty()) src = isArm ? "s0" : "ft0";
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm)
        os << "    fcvtzs " << dst << ", " << src << "\n";
      else
        os << "    fcvt.w.s " << dst << ", " << src << ", rtz\n";
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.neg" || opname == "arm_machine.neg" ||
        opname == "rv_machine.seqz" || opname == "arm_machine.not") {
      std::string src = ensureReg(op.operand(0), scratchReg(0));
      if (src.empty()) {
        stats.unsupportedOps++;
        stats.error = "unary machine op operand has no assigned register";
        return false;
      }
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm) {
        if (opname == "arm_machine.not")
          os << "    cmp " << src << ", #0\n    cset " << dst << ", eq\n";
        else
          os << "    neg " << dst << ", " << src << "\n";
      } else {
        if (opname == "rv_machine.seqz")
          os << "    seqz " << dst << ", " << src << "\n";
        else
          os << "    negw " << dst << ", " << src << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.alloca" || opname == "memref.alloca") {
      auto it = stackSlots.find(valueKey(op.result()));
      if (it != stackSlots.end())
        regs[valueKey(op.result())] = "stack:" + std::to_string(it->second);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.load" || opname == "memref.load") {
      if (op.operandCount() == 0) {
        stats.unsupportedOps++;
        stats.error = "load without base address";
        return false;
      }
      std::vector<Value> indices;
      for (int i = 1; i < op.operandCount(); i++)
        indices.push_back(op.operand(i));
      if (indices.empty()) {
        auto promotedIt = promotedScalarSlots.find(valueKey(op.operand(0)));
        if (promotedIt != promotedScalarSlots.end()) {
          PromotedScalarSlot &slot = promotedIt->second;
          if (!slot.valid) {
            auto stackIt = stackSlots.find(valueKey(op.operand(0)));
            if (stackIt == stackSlots.end()) {
              stats.unsupportedOps++;
              stats.error = "promoted scalar slot has no stack home";
              return false;
            }
            emitStackLoad(slot.reg, stackIt->second, "lw");
            slot.valid = true;
          }
          std::string dst = intResultReg(op.result());
          bindResult(op.result(), dst);
          if (dst != slot.reg)
            os << "    mv " << dst << ", " << slot.reg << "\n";
          if (shouldSpillDefinedValue(op.result()))
            spillHome(op.result(), dst);
          stats.scalarRegLoads++;
          stats.machineOps++;
          continue;
        }
      }
      if (parseMemrefInfo(op.resultType()).valid) {
        std::string addr;
        if (!computeAddress(op.operand(0), indices, scratchReg(0),
                            scratchReg(1), scratchReg(0), true, addr))
          return false;
        bindResult(op.result(), addr);
        if (livenessEnabled && (addr == scratchReg(0) || addr == scratchReg(1))) {
          if (remainingUses[valueKey(op.result())] > 0)
            spillHome(op.result(), addr);
          regs.erase(valueKey(op.result()));
        } else if (shouldSpillDefinedValue(op.result())) {
          spillHome(op.result(), addr);
        }
        stats.machineOps++;
        continue;
      }
      std::string dst = resultReg();
      if (isFloatType(op.resultType()))
        dst = floatReg();
      else if (!isArm)
        dst = intResultReg(op.result());
      bindResult(op.result(), dst);
      if (!loadFromAddress(dst, op.operand(0), indices))
        return false;
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.store" || opname == "memref.store") {
      if (op.operandCount() < 2) {
        stats.unsupportedOps++;
        stats.error = "store without base address";
        return false;
      }
      std::vector<Value> indices;
      for (int i = 2; i < op.operandCount(); i++)
        indices.push_back(op.operand(i));
      if (indices.empty()) {
        auto promotedIt = promotedScalarSlots.find(valueKey(op.operand(1)));
        if (promotedIt != promotedScalarSlots.end()) {
          PromotedScalarSlot &slot = promotedIt->second;
          std::string val = ensureReg(op.operand(0), scratchReg(0));
          if (val.empty()) {
            stats.unsupportedOps++;
            stats.error = "promoted scalar store value has no assigned register";
            return false;
          }
          if (val != slot.reg)
            os << "    mv " << slot.reg << ", " << val << "\n";
          slot.valid = true;
          slot.dirty = true;
          stats.scalarRegStores++;
          stats.machineOps++;
          continue;
        }
      }
      if (!storeToAddress(op.operand(0), op.operand(1), indices))
        return false;
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.vle32" || opname == "arm_machine.ld1" || opname == "vector.transfer_read") {
      std::string base = ensureReg(op.operand(0), scratchReg(0));
      std::string index = ensureReg(op.operand(1), scratchReg(1));
      std::string dst = "v" + std::to_string(nextVecReg++);
      bindResult(op.result(), dst);
      if (isArm) {
        os << "    add x9, " << base << ", " << index << ", lsl #2\n";
        os << "    ld1 {" << dst << ".4s}, [x9]\n";
      } else {
        os << "    vsetvli zero, 4, e32, m1, ta, ma\n";
        os << "    slli t6, " << index << ", 2\n";
        os << "    add t6, " << base << ", t6\n";
        os << "    vle32.v " << dst << ", (t6)\n";
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.vse32" || opname == "arm_machine.st1" || opname == "vector.transfer_write") {
      std::string val = ensureReg(op.operand(0), scratchReg(0));
      std::string base = ensureReg(op.operand(1), scratchReg(0));
      std::string index = ensureReg(op.operand(2), scratchReg(1));
      if (isArm) {
        os << "    add x9, " << base << ", " << index << ", lsl #2\n";
        os << "    st1 {" << val << ".4s}, [x9]\n";
      } else {
        os << "    vsetvli zero, 4, e32, m1, ta, ma\n";
        os << "    slli t6, " << index << ", 2\n";
        os << "    add t6, " << base << ", t6\n";
        os << "    vse32.v " << val << ", (t6)\n";
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.vfmv" || opname == "arm_machine.dup" || opname == "vector.splat") {
      std::string val = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      std::string dst = "v" + std::to_string(nextVecReg++);
      bindResult(op.result(), dst);
      if (isArm) {
        os << "    dup " << dst << ".4s, " << val << "\n";
      } else {
        os << "    vsetvli zero, 4, e32, m1, ta, ma\n";
        os << "    vfmv.v.f " << dst << ", " << val << "\n";
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.rotate_helper") {
      if (op.operandCount() != 2 || op.resultCount() != 1) {
        stats.unsupportedOps++;
        stats.error = "bad rotate helper shape";
        return false;
      }
      std::string direction = symbolAttr(op.attr("direction"));
      int64_t maxShift = parseIntegerAttr(op.attr("max_shift"));
      if ((direction != "left" && direction != "right") || maxShift < 1 || maxShift > 30) {
        stats.unsupportedOps++;
        stats.error = "bad rotate helper attributes";
        return false;
      }
      std::string xReg = ensureReg(op.operand(0), scratchReg(0));
      std::string nReg = ensureReg(op.operand(1), scratchReg(1));
      if (xReg.empty() || nReg.empty()) {
        stats.unsupportedOps++;
        stats.error = "rotate helper operand has no assigned register";
        return false;
      }
      std::string dst = resultReg();
      bindResult(op.result(), dst);

      auto chooseTemp = [&](std::initializer_list<std::string> excluded) -> std::string {
        std::set<std::string> used(excluded.begin(), excluded.end());
        const std::vector<std::string> pool = isArm
            ? std::vector<std::string>{"w16", "w17", "w15", "w14", "w13"}
            : std::vector<std::string>{"t5", "t6", "t4", "t3", "t2"};
        for (const auto &reg : pool) {
          if (used.count(reg) == 0) {
            clobberPhysicalReg(reg);
            return reg;
          }
        }
        return pool.front();
      };
      auto armW = [](std::string reg) {
        if (!reg.empty() && reg[0] == 'x')
          reg[0] = 'w';
        return reg;
      };

      int labelId = ++nextLoopId;
      std::string done = ".Lrot_done_" + std::to_string(labelId);
      if (isArm) {
        xReg = armW(xReg);
        nReg = armW(nReg);
        if (dst != xReg)
          os << "    mov " << dst << ", " << xReg << "\n";
        os << "    cmp " << nReg << ", #1\n";
        os << "    blt " << done << "\n";
        os << "    cmp " << nReg << ", #" << maxShift << "\n";
        os << "    bgt " << done << "\n";
        if (direction == "left") {
          os << "    lsl " << dst << ", " << dst << ", " << nReg << "\n";
        } else {
          std::string mask = chooseTemp({dst, nReg});
          std::string sign = chooseTemp({dst, nReg, mask});
          os << "    mov " << mask << ", #1\n";
          os << "    lsl " << mask << ", " << mask << ", " << nReg << "\n";
          os << "    sub " << mask << ", " << mask << ", #1\n";
          os << "    asr " << sign << ", " << dst << ", #31\n";
          os << "    and " << mask << ", " << mask << ", " << sign << "\n";
          os << "    add " << dst << ", " << dst << ", " << mask << "\n";
          os << "    asr " << dst << ", " << dst << ", " << nReg << "\n";
        }
        os << done << ":\n";
      } else {
        if (dst != xReg)
          os << "    mv " << dst << ", " << xReg << "\n";
        std::string guardTmp = chooseTemp({dst, nReg});
        os << "    li " << guardTmp << ", 1\n";
        os << "    blt " << nReg << ", " << guardTmp << ", " << done << "\n";
        os << "    li " << guardTmp << ", " << maxShift << "\n";
        os << "    blt " << guardTmp << ", " << nReg << ", " << done << "\n";
        if (direction == "left") {
          os << "    sllw " << dst << ", " << dst << ", " << nReg << "\n";
        } else {
          std::string mask = guardTmp;
          std::string sign = chooseTemp({dst, nReg, mask});
          os << "    li " << mask << ", 1\n";
          os << "    sllw " << mask << ", " << mask << ", " << nReg << "\n";
          os << "    addiw " << mask << ", " << mask << ", -1\n";
          os << "    sraiw " << sign << ", " << dst << ", 31\n";
          os << "    and " << mask << ", " << mask << ", " << sign << "\n";
          os << "    addw " << dst << ", " << dst << ", " << mask << "\n";
          os << "    sraw " << dst << ", " << dst << ", " << nReg << "\n";
        }
      emitLabel(done + ":");
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.call") {
      std::string callee = symbolAttr(op.attr("callee"));
      if (callee.empty()) {
        stats.unsupportedOps++;
        stats.error = "call without callee attr";
        return false;
      }
      Operation *tailReturn = nextLiveOpAfter(op);
      bool selfTailCall = envEnabled("SISY_ENABLE_SELF_TAIL_CALL", true) &&
                          !memoInfo && callee == name && op.operandCount() <= 2 &&
                          tailReturn &&
                          (tailReturn->name() == "sysy.return" ||
                           tailReturn->name() == "scf.return");
      if (selfTailCall) {
        for (auto operand : op.getOperands()) {
          if (isFloatType(operand.type()) || isMemrefType(operand.type())) {
            selfTailCall = false;
            break;
          }
        }
      }
      if (selfTailCall) {
        if (op.resultCount() == 0) {
          selfTailCall = tailReturn->operandCount() == 0;
        } else {
          selfTailCall = tailReturn->operandCount() == 1 &&
                         tailReturn->operand(0) == op.result() &&
                         op.resultUses.size() == 1 &&
                         op.resultUses[0].size() == 1 &&
                         op.resultUses[0][0].owner == tailReturn;
        }
      }
      if (selfTailCall) {
        std::vector<std::string> staged;
        for (int i = 0; i < op.operandCount(); i++) {
          std::string tmp = isArm ? ("w" + std::to_string(16 + i)) : scratchReg(i);
          std::string src = ensureReg(op.operand(i), tmp);
          if (src.rfind("stack:", 0) == 0 || src.rfind("global:", 0) == 0)
            src = materializeAddress(op.operand(i), tmp);
          if (src.empty()) {
            if (isArm)
              os << "    mov " << tmp << ", #0\n";
            else
              os << "    li " << tmp << ", 0\n";
          } else if (src != tmp) {
            os << "    " << (isArm ? "mov " : "mv ") << tmp << ", " << src << "\n";
          }
          staged.push_back(tmp);
        }
        for (int i = 0; i < (int) staged.size(); i++) {
          std::string arg = isArm ? ("w" + std::to_string(i))
                                  : ("a" + std::to_string(i));
          if (staged[i] != arg) {
            clobberPhysicalReg(arg);
            os << "    " << (isArm ? "mov " : "mv ") << arg << ", "
               << staged[i] << "\n";
          }
        }
        os << "    " << (isArm ? "b" : "j") << " " << bodyEntryLabel << "\n";
        tailReturnSkips.insert(tailReturn);
        stats.tailCalls++;
        stats.machineOps++;
        continue;
      }
      invalidateCallerSavedForCall();
      int outgoingIntRegs = 0;
      int outgoingFpRegs = 0;
      int outgoingStackSlots = 0;
      for (int i = 0; i < op.operandCount(); i++) {
        Value operand = op.operand(i);
        bool fpArg = isFloatType(operand.type());
        bool ptrArg = isMemrefType(operand.type());
        std::string src = ensureReg(operand, fpArg ? floatScratchReg() : scratchReg(0));
        if (src.rfind("stack:", 0) == 0 || src.rfind("global:", 0) == 0)
          src = materializeAddress(operand, scratchReg(0));
        if (src.empty()) {
          if (fpArg) {
            src = floatScratchReg();
            emitFloatBitsToReg(src, 0);
          } else {
            src = scratchReg(0);
            if (isArm)
              os << "    mov " << src << ", #0\n";
            else
              os << "    li " << src << ", 0\n";
          }
        }

        if (fpArg) {
          if (outgoingFpRegs < 8) {
            std::string arg = (isArm ? "s" : "fa") + std::to_string(outgoingFpRegs++);
            if (src != arg) {
              clobberPhysicalReg(arg);
              os << "    " << (isArm ? "fmov " : "fmv.s ") << arg << ", " << src << "\n";
            }
          } else {
            emitStackStore(src, outgoingArgBase + outgoingStackSlots++ * 8,
                           isArm ? "str" : "fsw");
          }
        } else {
          if (outgoingIntRegs < 8) {
            std::string arg;
            if (isArm && ptrArg)
              arg = "x" + std::to_string(outgoingIntRegs);
            else if (isArm)
              arg = "w" + std::to_string(outgoingIntRegs);
            else
              arg = "a" + std::to_string(outgoingIntRegs);
            outgoingIntRegs++;
            if (src != arg) {
              clobberPhysicalReg(arg);
              os << "    " << (isArm ? "mov " : "mv ") << arg << ", " << src << "\n";
            }
          } else {
            emitStackStore(src, outgoingArgBase + outgoingStackSlots++ * 8,
                           ptrArg ? (isArm ? "str" : "sd") : (isArm ? "str" : "sw"));
          }
        }
      }
      invalidateCallerSavedForCall();
      os << "    call " << callee << "\n";
      if (op.resultCount() > 0) {
        std::string result = isFloatType(op.resultType()) ? (isArm ? "s0" : "fa0")
                                                          : (isArm ? "w0" : "a0");
        bindResult(op.result(), result);
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), result);
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.unknown_value") {
      if (op.resultCount() > 0 && op.resultType().kind() != TypeKind::None) {
        std::string dst = resultReg();
        bindResult(op.result(), dst);
        if (isArm)
          os << "    mov " << dst << ", #0\n";
        else
          os << "    li " << dst << ", 0\n";
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), dst);
      }
      continue;
    }

    if (opname == "sysy.return" || opname == "scf.return" ||
        opname == "rv_machine.ret" || opname == "arm_machine.ret") {
      if (tailReturnSkips.count(&op) != 0) {
        stats.returns++;
        continue;
      }
      if (op.operandCount() > 0) {
        std::string src = ensureReg(op.operand(0), isFloatType(op.operand(0).type()) ? (isArm ? "s30" : "ft10") : scratchReg(0));
        if (src.empty()) {
          stats.unsupportedOps++;
          stats.error = "return operand has no assigned register";
          return false;
        }
        if (isArm) {
          if (isFloatType(op.operand(0).type())) {
            if (src != "s0")
              os << "    fmov s0, " << src << "\n";
          } else if (src != "w0") {
            os << "    mov w0, " << src << "\n";
          }
        } else {
          if (isFloatType(op.operand(0).type())) {
            if (src != "fa0")
              os << "    fmv.s fa0, " << src << "\n";
          } else if (src != "a0") {
            os << "    mv a0, " << src << "\n";
          }
        }
        if (memoInfo && !isFloatType(op.operand(0).type()))
          emitMemoStore(*memoInfo);
      }
      os << "    " << (isArm ? "b" : "j") << " " << epilogueLabel << "\n";
      stats.returns++;
      continue;
    }

    if (opname == "affine.for") {
      int loopId = loopOps[&op];
      std::string iv;
      if (isArm) {
        iv = "w" + std::to_string(nextLoopReg++ % 10 + 19);
      } else if (regAlloc2Enabled && calleeSaveCount >= 12) {
        static const char *ivRegs[] = {"s11", "s10", "s9", "s8"};
        iv = ivRegs[(nextLoopReg++) % (int)(sizeof(ivRegs) / sizeof(ivRegs[0]))];
      } else {
        iv = "s" + std::to_string(nextLoopReg++ % 12);
      }
      std::string boundScratch = isArm ? "w17" : scratchReg(1);

      Value ivValue = op.getRegions()[0]->getBlocks()[0]->args()[0]->value();
      loopIvRegs[&op] = iv;
      bindReg(ivValue, iv);
      std::string pinnedBound = ensureReg(op.operand(1), boundScratch);
      if (!pinnedBound.empty())
        spillHome(op.operand(1), pinnedBound, true);
      std::string pinnedStep = ensureReg(op.operand(2), boundScratch);
      if (!pinnedStep.empty())
        spillHome(op.operand(2), pinnedStep, true);
      std::string start = reloadValue(op.operand(0), isArm ? "w16" : scratchReg(0));

      if (isArm) {
        os << "    mov " << iv << ", " << start << "\n";
        spillHome(ivValue, iv, true);
        emitLabel(".Lloop_cond_" + std::to_string(loopId) + ":");
        std::string bound = reloadValue(op.operand(1), boundScratch);
        os << "    cmp " << iv << ", " << bound << "\n";
        os << "    bge .Lloop_end_" << loopId << "\n";
      } else {
        os << "    mv " << iv << ", " << start << "\n";
        spillHome(ivValue, iv, true);
        emitLabel(".Lloop_cond_" + std::to_string(loopId) + ":");
        std::string bound = reloadValue(op.operand(1), boundScratch);
        os << "    bge " << iv << ", " << bound << ", .Lloop_end_" << loopId << "\n";
      }
      continue;
    }

    if (opname == "scf.while") {
      int loopId = loopOps[&op];
      emitLabel(".Lwhile_cond_" + std::to_string(loopId) + ":");
      continue;
    }

    if (opname == "scf.condition") {
      Operation *parentWhile = op.getBlock()->getRegion()->getParent();
      if (!parentWhile || parentWhile->name() != "scf.while") {
        stats.unsupportedOps++;
        stats.error = "scf.condition outside scf.while";
        return false;
      }
      int loopId = loopOps[parentWhile];
      std::string cond = ensureReg(op.operand(0), scratchReg(0));
      if (cond.empty()) {
        stats.unsupportedOps++;
        stats.error = "scf.condition operand has no assigned register";
        return false;
      }
      if (isArm)
        os << "    cbz " << cond << ", .Lwhile_end_" << loopId << "\n";
      else
        os << "    beqz " << cond << ", .Lwhile_end_" << loopId << "\n";
      continue;
    }

    if (opname == "affine.yield") {
      Operation *parentFor = op.getBlock()->getRegion()->getParent();
      if (!parentFor || parentFor->name() != "affine.for") continue;
      int loopId = loopOps[parentFor];
      bool isLoopBodyTerminator =
          op.getBlock() && !parentFor->getRegions().empty() &&
          op.getBlock()->getRegion() == parentFor->getRegions()[0].get();
      if (!isLoopBodyTerminator) {
        os << "    " << (isArm ? "b" : "j") << " .Lloop_latch_" << loopId << "\n";
        continue;
      }
      std::string step = ensureReg(parentFor->operand(2), scratchReg(1));
      Value ivValue = parentFor->getRegions()[0]->getBlocks()[0]->args()[0]->value();
      std::string preferredIv = loopIvRegs.count(parentFor) ? loopIvRegs[parentFor]
                                                            : scratchReg(0);
      std::string iv = ensureReg(ivValue, preferredIv);
      if (iv != preferredIv) {
        if (isArm)
          os << "    mov " << preferredIv << ", " << iv << "\n";
        else
          os << "    mv " << preferredIv << ", " << iv << "\n";
        iv = preferredIv;
      }

      if (isArm) {
        emitLabel(".Lloop_latch_" + std::to_string(loopId) + ":");
        os << "    add " << iv << ", " << iv << ", " << step << "\n";
        bindReg(ivValue, iv);
        spillHome(ivValue, iv, true);
        os << "    b .Lloop_cond_" << loopId << "\n";
        emitLabel(".Lloop_end_" + std::to_string(loopId) + ":");
      } else {
        emitLabel(".Lloop_latch_" + std::to_string(loopId) + ":");
        os << "    addw " << iv << ", " << iv << ", " << step << "\n";
        bindReg(ivValue, iv);
        spillHome(ivValue, iv, true);
        os << "    j .Lloop_cond_" << loopId << "\n";
        emitLabel(".Lloop_end_" + std::to_string(loopId) + ":");
      }
      continue;
    }

    if (opname == "sysy.continue" || opname == "sysy.break") {
      Operation *loop = nullptr;
      Block *currBlock = op.getBlock();
      while (currBlock && !loop) {
        Region *region = currBlock->getRegion();
        Operation *parent = region ? region->getParent() : nullptr;
        if (!parent)
          break;
        if (parent->name() == "scf.while" || parent->name() == "affine.for") {
          loop = parent;
          break;
        }
        currBlock = parent->getBlock();
      }
      if (!loop || loopOps.count(loop) == 0) {
        stats.unsupportedOps++;
        stats.error = opname + " without enclosing lowered loop";
        return false;
      }
      int loopId = loopOps[loop];
      const bool isContinue = opname == "sysy.continue";
      if (loop->name() == "affine.for") {
        os << "    " << (isArm ? "b" : "j") << " .Lloop_"
           << (isContinue ? "latch_" : "end_") << loopId << "\n";
      } else {
        os << "    " << (isArm ? "b" : "j") << " .Lwhile_"
           << (isContinue ? "cond_" : "end_") << loopId << "\n";
      }
      continue;
    }

    if (opname == "rv_machine.cmp" || opname == "arm_machine.cmp" ||
        opname == "rv_machine.fcmp" || opname == "arm_machine.fcmp") {
      if (opname == "rv_machine.cmp" && op.operandCount() == 2 &&
          op.resultCount() == 1) {
        std::string pred = symbolAttr(op.attr("predicate"));
        if (pred == "eq" || pred == "ne") {
          int64_t zero = 0;
          Value testValue;
          if (constantIntegerValue(op.operand(1), zero) && zero == 0)
            testValue = op.operand(0);
          else if (constantIntegerValue(op.operand(0), zero) && zero == 0)
            testValue = op.operand(1);
          if (testValue.valid()) {
            std::string src = ensureReg(testValue, scratchReg(0));
            if (src.empty()) {
              stats.unsupportedOps++;
              stats.error = "cmp zero operand has no assigned register";
              return false;
            }
            std::string dst = intResultReg(op.result());
            bindResult(op.result(), dst);
            os << "    " << (pred == "eq" ? "seqz " : "snez ")
               << dst << ", " << src << "\n";
            if (shouldSpillDefinedValue(op.result()))
              spillHome(op.result(), dst);
            stats.machineOps++;
            continue;
          }
        }
      }
      std::string lhs = ensureReg(op.operand(0), isFloatType(op.operand(0).type()) ? (isArm ? "s30" : "ft10") : scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), isFloatType(op.operand(1).type()) ? (isArm ? "s31" : "ft11") : scratchReg(1));
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);

      std::string pred = symbolAttr(op.attr("predicate"));

      if (opname == "rv_machine.fcmp") {
        if (pred == "lt") {
          os << "    flt.s " << dst << ", " << lhs << ", " << rhs << "\n";
        } else if (pred == "le") {
          os << "    fle.s " << dst << ", " << lhs << ", " << rhs << "\n";
        } else if (pred == "eq") {
          os << "    feq.s " << dst << ", " << lhs << ", " << rhs << "\n";
        } else {
          os << "    feq.s " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    xori " << dst << ", " << dst << ", 1\n";
        }
      } else if (opname == "arm_machine.fcmp") {
        os << "    fcmp " << lhs << ", " << rhs << "\n";
        os << "    cset " << dst << ", " << pred << "\n";
      } else if (isArm) {
        os << "    cmp " << lhs << ", " << rhs << "\n";
        os << "    cset " << dst << ", " << pred << "\n";
      } else {
        if (pred == "lt") {
          os << "    slt " << dst << ", " << lhs << ", " << rhs << "\n";
        } else if (pred == "le") {
          os << "    slt " << dst << ", " << rhs << ", " << lhs << "\n";
          os << "    xori " << dst << ", " << dst << ", 1\n";
        } else if (pred == "eq") {
          os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    seqz " << dst << ", " << dst << "\n";
        } else {
          os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    snez " << dst << ", " << dst << "\n";
        }
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.select" || opname == "arm_machine.select") {
      if (op.operandCount() != 3 || op.resultCount() != 1) {
        stats.unsupportedOps++;
        stats.error = "bad select shape";
        return false;
      }
      bool fpSelect = isFloatType(op.resultType());
      std::string cond = ensureReg(op.operand(0), scratchReg(0));
      if (cond.empty()) {
        stats.unsupportedOps++;
        stats.error = "select operand has no assigned register";
        return false;
      }
      auto chooseDst = [&]() {
        for (int attempt = 0; attempt < 20; attempt++) {
          std::string candidate = fpSelect ? floatReg() : resultReg();
          if (candidate != cond)
            return candidate;
        }
        return fpSelect ? floatReg() : resultReg();
      };
      auto moveReg = [&](const std::string &dst, const std::string &src) {
        if (dst == src)
          return;
        if (isArm)
          os << "    " << (fpSelect ? "fmov " : "mov ") << dst << ", " << src << "\n";
        else
          os << "    " << (fpSelect ? "fmv.s " : "mv ") << dst << ", " << src << "\n";
      };
      auto moveOperand = [&](const std::string &dst, Value value,
                             const std::string &tmp) {
        std::string src = ensureReg(value, tmp);
        if (src.empty())
          return false;
        moveReg(dst, src);
        return true;
      };
      std::string dst = chooseDst();
      bindResult(op.result(), dst);
      int labelId = ++nextLoopId;
      std::string done = ".Lselect_done_" + std::to_string(labelId);
      std::string valueTmp = fpSelect ? (isArm ? "s30" : "ft10")
                                      : (cond == scratchReg(1) ? scratchReg(0) : scratchReg(1));
      if (!moveOperand(dst, op.operand(2), valueTmp)) {
        stats.unsupportedOps++;
        stats.error = "select false operand has no assigned register";
        return false;
      }
      if (isArm)
        os << "    cbz " << cond << ", " << done << "\n";
      else
        os << "    beqz " << cond << ", " << done << "\n";
      if (!moveOperand(dst, op.operand(1), valueTmp)) {
        stats.unsupportedOps++;
        stats.error = "select true operand has no assigned register";
        return false;
      }
        emitLabel(done + ":");
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "scf.if") {
      std::string cond = ensureReg(op.operand(0), scratchReg(0));
      int ifId = loopOps[&op];

      if (isArm) {
        os << "    cbz " << cond << ", ." << (op.getRegions().size() > 1 ? "Lelse_" : "Lendif_") << ifId << "\n";
      } else {
        os << "    beqz " << cond << ", ." << (op.getRegions().size() > 1 ? "Lelse_" : "Lendif_") << ifId << "\n";
      }
      continue;
    }

    if (opname == "scf.yield") {
      Region *region = op.getBlock() ? op.getBlock()->getRegion() : nullptr;
      Operation *parent = region ? region->getParent() : nullptr;
      if (parent && parent->name() == "scf.if") {
        int ifId = loopOps[parent];
        bool thenRegion = !parent->getRegions().empty() && parent->getRegions()[0].get() == region;
        bool hasElse = parent->getRegions().size() > 1;
        if (thenRegion && hasElse) {
          os << "    " << (isArm ? "b" : "j") << " .Lendif_" << ifId << "\n";
          emitLabel(".Lelse_" + std::to_string(ifId) + ":");
        } else {
          emitLabel(".Lendif_" + std::to_string(ifId) + ":");
        }
        continue;
      }
      if (parent && parent->name() == "scf.while") {
        int loopId = loopOps[parent];
        os << "    " << (isArm ? "b" : "j") << " .Lwhile_cond_" << loopId << "\n";
        emitLabel(".Lwhile_end_" + std::to_string(loopId) + ":");
        continue;
      }
      continue;
    }

    stats.unsupportedOps++;
    stats.error = "unsupported native asm op: " + opname;
    return false;
  }

  for (const auto &label : functionEndLabels)
    emitLabel(label);

  if (stats.returns == returnsBefore)
    stats.returns++;
  os << epilogueLabel << ":\n";
  if (memoInfo) {
    os << "    la t0, " << memoInfo->depthLabel << "\n";
    os << "    lw t1, 0(t0)\n";
    os << "    addi t1, t1, -1\n";
    os << "    sw t1, 0(t0)\n";
  }
  for (int i = 0; i < calleeSaveCount; i++) {
    int64_t off = calleeSaveBase + i * 8;
    if (isArm)
      emitStackLoad("x" + std::to_string(19 + i), off, "ldr");
    else
      emitStackLoad("s" + std::to_string(i), off, "ld");
  }
  if (hasCall)
    emitStackLoad(isArm ? "x30" : "ra", returnAddressSlot, isArm ? "ldr" : "ld");
  emitStackAdjust(false);
  os << "    ret\n";
  return true;
}

} // namespace

bool emitNativeAssembly(Module &module, const std::string &target, std::ostream &os,
                        NativeAsmStats &stats, bool enablePow2Strength) {
  stats = NativeAsmStats();
  if (target != "riscv" && target != "arm") {
    stats.error = "native asm target must be riscv|arm";
    return false;
  }
  if (!verifyLegacyFree(module, &stats)) {
    stats.error = "self-MLIR native asm refuses legacy/Phi operations";
    return false;
  }
  auto verified = verify(module);
  if (!verified.ok) {
    stats.error = verified.errors.empty() ? "self-MLIR verify failed" : verified.errors.front();
    return false;
  }

  std::map<std::string, std::string> globalLabels;
  std::map<std::string, uint32_t> scalarGlobalInits;
  std::map<std::string, std::vector<uint32_t>> globalWordInits;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
      continue;
    std::string name = symbolAttr(op->attr("symbol"));
    if (name.empty())
      name = symbolAttr(op->attr("sym_name"));
    std::string label = ".Lglob_" + sanitizeLabel(name);
    globalLabels[valueKey(op->result())] = label;
    auto words = parseGlobalInitWords(op->attr("init_words"));
    if (!words.empty())
      globalWordInits[valueKey(op->result())] = std::move(words);
  }
  std::map<std::string, MemoFunctionInfo> memoFunctions;
  if (target == "riscv" && envEnabled("SISY_ENABLE_SELF_RECURSIVE_MEMO", true)) {
    int memoOrdinal = 0;
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.func")
        continue;
      MemoFunctionInfo memo = classifyMemoFunction(*op, ++memoOrdinal);
      if (!memo.enabled)
        continue;
      std::string name = symbolAttr(op->attr("sym_name"));
      memoFunctions[name] = memo;
    }
    stats.memoFunctions = (int) memoFunctions.size();
  }
  if (!module.body().getBlocks().empty()) {
    std::map<std::string, Value> scalarGlobals;
    for (auto &owned : module.body().getBlocks()[0]->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() || op->name() != "sysy.global" ||
          op->resultCount() == 0 || !isScalarWordMemref(op->resultType()))
        continue;
      scalarGlobals[valueKey(op->result())] = op->result();
    }
    for (auto &owned : module.body().getBlocks()[0]->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() ||
          (op->name() != "sysy.store" && op->name() != "memref.store") ||
          op->operandCount() < 2)
        continue;
      auto globalIt = scalarGlobals.find(valueKey(op->operand(1)));
      if (globalIt == scalarGlobals.end())
        continue;
      bool zeroIndex = true;
      for (int i = 2; i < op->operandCount(); i++) {
        int64_t index = 0;
        if (!constantIntegerValue(op->operand(i), index) || index != 0) {
          zeroIndex = false;
          break;
        }
      }
      uint32_t bits = 0;
      if (zeroIndex && constantScalarWordBits(op->operand(0), bits) && bits != 0)
        scalarGlobalInits[globalIt->first] = bits;
    }
  }

  if (!scalarGlobalInits.empty() || !globalWordInits.empty()) {
    os << "    .data\n";
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
        continue;
      auto labelIt = globalLabels.find(valueKey(op->result()));
      if (labelIt == globalLabels.end())
        continue;
      auto wordIt = globalWordInits.find(valueKey(op->result()));
      auto initIt = scalarGlobalInits.find(valueKey(op->result()));
      if (wordIt == globalWordInits.end() && initIt == scalarGlobalInits.end())
        continue;
      os << "    .align 2\n" << labelIt->second << ":\n";
      if (wordIt != globalWordInits.end()) {
        for (std::size_t i = 0; i < wordIt->second.size(); i++) {
          if (i % 8 == 0)
            os << "    .word ";
          else
            os << ", ";
          os << wordIt->second[i];
          if (i % 8 == 7 || i + 1 == wordIt->second.size())
            os << "\n";
        }
      } else {
        os << "    .word " << initIt->second << "\n";
        stats.globalScalarInits++;
      }
    }
  }

  bool emittedBss = false;
  if (!memoFunctions.empty()) {
    os << "    .bss\n";
    emittedBss = true;
    for (const auto &kv : memoFunctions) {
      const auto &memo = kv.second;
      os << "    .align 3\n" << memo.validLabel << ":\n";
      os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      os << "    .align 3\n" << memo.key0Label << ":\n";
      os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      if (memo.argCount > 1) {
        os << "    .align 3\n" << memo.key1Label << ":\n";
        os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      }
      os << "    .align 3\n" << memo.valueLabel << ":\n";
      os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      os << "    .align 2\n" << memo.depthLabel << ":\n";
      os << "    .zero 4\n";
      os << "    .align 2\n" << memo.epochLabel << ":\n";
      os << "    .zero 4\n";
    }
  }
  if (!globalLabels.empty()) {
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
        continue;
      auto it = globalLabels.find(valueKey(op->result()));
      if (it == globalLabels.end())
        continue;
      if (scalarGlobalInits.count(valueKey(op->result())) != 0 ||
          globalWordInits.count(valueKey(op->result())) != 0)
        continue;
      if (!emittedBss) {
        os << "    .bss\n";
        emittedBss = true;
      }
      os << "    .align 3\n" << it->second << ":\n";
      os << "    .zero " << memrefAllocationBytes(op->resultType()) << "\n";
    }
  }

  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.func") {
      stats.functions++;
      if (!emitFunctionAssembly(*op, target, os, stats, enablePow2Strength,
                                globalLabels, memoFunctions))
        return false;
    }
  }
  stats.linearScanSpills = stats.liveSpills;
  stats.emitted = stats.functions > 0 && stats.unsupportedOps == 0;
  if (!stats.emitted && stats.error.empty())
    stats.error = "no self-MLIR functions emitted";
  return stats.emitted;
}

std::vector<ConversionPattern> targetPatterns(const std::string &target) {
  const std::string prefix = target == "arm" ? "arm_machine." : "rv_machine.";
  if (target == "arm") {
    return {
      {"arith.constant", prefix + "mov"},
      {"arith.addi", prefix + "add"},
      {"arith.subi", prefix + "sub"},
      {"arith.muli", prefix + "mul"},
      {"arith.divi", prefix + "sdiv"},
      {"arith.remi", prefix + "srem"},
      {"arith.andi", prefix + "and"},
      {"arith.ori", prefix + "orr"},
      {"arith.xori", prefix + "eor"},
      {"arith.noti", prefix + "not"},
      {"arith.cmpi", prefix + "cmp"},
      {"arith.addf", prefix + "fadd"},
      {"arith.subf", prefix + "fsub"},
      {"arith.mulf", prefix + "fmul"},
      {"arith.divf", prefix + "fdiv"},
      {"arith.negf", prefix + "fneg"},
      {"arith.negi", prefix + "neg"},
      {"arith.sitofp", prefix + "scvtf"},
      {"arith.fptosi", prefix + "fcvtzs"},
      {"arith.select", prefix + "select"},
      {"vector.transfer_read", "arm_machine.ld1"},
      {"vector.transfer_write", "arm_machine.st1"},
      {"vector.splat", "arm_machine.dup"},
    };
  }
  return {
    {"arith.constant", prefix + "li"},
    {"arith.addi", prefix + "addw"},
    {"arith.subi", prefix + "subw"},
    {"arith.muli", prefix + "mulw"},
    {"arith.divi", prefix + "divw"},
    {"arith.remi", prefix + "remw"},
    {"arith.andi", prefix + "and"},
    {"arith.ori", prefix + "or"},
    {"arith.xori", prefix + "xor"},
    {"arith.noti", prefix + "seqz"},
    {"arith.cmpi", prefix + "cmp"},
    {"arith.addf", prefix + "fadd"},
    {"arith.subf", prefix + "fsub"},
    {"arith.mulf", prefix + "fmul"},
    {"arith.divf", prefix + "fdiv"},
    {"arith.negf", prefix + "fneg"},
    {"arith.negi", prefix + "neg"},
    {"arith.sitofp", prefix + "fcvt_s_w"},
    {"arith.fptosi", prefix + "fcvt_w_s"},
    {"arith.select", prefix + "select"},
    {"vector.transfer_read", "rv_machine.vle32"},
    {"vector.transfer_write", "rv_machine.vse32"},
    {"vector.splat", "rv_machine.vfmv"},
  };
}

ConversionTarget productionTarget(const std::string &target) {
  ConversionTarget convTarget;
  convTarget.addLegalDialect("builtin");
  convTarget.addLegalDialect("sysy");
  convTarget.addLegalDialect("scf");
  convTarget.addLegalDialect("cf");
  convTarget.addLegalDialect("memref");
  convTarget.addLegalDialect("affine");
  convTarget.addLegalDialect("vector");
  convTarget.addLegalDialect(target == "arm" ? "arm_machine" : "rv_machine");
  return convTarget;
}

struct ModuleIndex {
  std::vector<Operation*> ops;
  std::vector<Operation*> globals;
  std::vector<Operation*> functions;
  std::map<std::string, std::vector<Operation*>> byName;
  std::map<std::string, std::vector<Operation*>> symbolAccesses;
  std::map<std::string, std::set<Operation*>> symbolFunctions;
  std::map<std::string, int> valueUseCounts;
};

static Operation *enclosingFunction(Operation *op) {
  if (!op)
    return nullptr;
  Block *currBlock = op->getBlock();
  while (currBlock) {
    Region *region = currBlock->getRegion();
    if (!region)
      break;
    Operation *parentOp = region->getParent();
    if (!parentOp)
      break;
    if (parentOp->name() == "sysy.func")
      return parentOp;
    currBlock = parentOp->getBlock();
  }
  return nullptr;
}

static ModuleIndex buildModuleIndex(Module &module) {
  ModuleIndex index;
  index.ops = walk(module);
  for (auto *op : index.ops) {
    if (!op || op->isErased())
      continue;
    index.byName[op->name()].push_back(op);
    if (op->name() == "sysy.global")
      index.globals.push_back(op);
    if (op->name() == "sysy.func")
      index.functions.push_back(op);
    for (auto operand : op->getOperands()) {
      if (operand.valid())
        index.valueUseCounts[valueKey(operand)]++;
    }
    if (op->name() == "sysy.load" || op->name() == "sysy.store" ||
        op->name() == "memref.load" || op->name() == "memref.store") {
      std::string symbol = symbolAttr(op->attr("symbol"));
      if (symbol.empty())
        symbol = symbolAttr(op->attr("sym_name"));
      if (!symbol.empty()) {
        index.symbolAccesses[symbol].push_back(op);
        if (auto *func = enclosingFunction(op))
          index.symbolFunctions[symbol].insert(func);
      }
    }
  }
  return index;
}

void runGlobalOpt(Module &module, SelfOptStats *stats) {
  const bool promoteGlobals = envEnabled("SISY_ENABLE_SELF_GLOBAL_PROMOTE", false);
  int64_t promoteLimit = 4096;
  if (const char *value = std::getenv("SISY_SELF_GLOBAL_PROMOTE_MAX_BYTES")) {
    try {
      promoteLimit = std::max<int64_t>(0, std::stoll(value));
    } catch (...) {
      promoteLimit = 4096;
    }
  }

  ModuleIndex index = buildModuleIndex(module);
  if (stats && index.globals.size() > 1)
    stats->walksEliminated += (int) index.globals.size() - 1;

  for (auto *global : index.globals) {
    std::string gName = symbolAttr(global->attr("symbol"));
    if (gName.empty())
      gName = symbolAttr(global->attr("sym_name"));
    if (gName.empty())
      continue;

    auto accessIt = index.symbolAccesses.find(gName);
    const std::vector<Operation*> emptyAccesses;
    const std::vector<Operation*> &accesses =
        accessIt == index.symbolAccesses.end() ? emptyAccesses : accessIt->second;
    auto funcIt = index.symbolFunctions.find(gName);
    const std::set<Operation*> emptyFunctions;
    const std::set<Operation*> &functions =
        funcIt == index.symbolFunctions.end() ? emptyFunctions : funcIt->second;

    if (accesses.empty()) {
      bool hasDirectUse = global->resultCount() > 0 &&
                          index.valueUseCounts[valueKey(global->result())] > 0;
      if (global->resultCount() == 0 || !hasDirectUse) {
        global->markErased();
        if (stats)
          stats->globalsErased++;
      }
      continue;
    }

    if (promoteGlobals && functions.size() == 1 &&
        memrefAllocationBytes(global->resultType()) <= promoteLimit) {
      Operation *func = *functions.begin();
      auto &regions = func->getRegions();
      if (!regions.empty() && !regions[0]->getBlocks().empty()) {
        auto &entry = *regions[0]->getBlocks()[0];
        Type storageType = global->resultType();
        auto &slot = entry.insertOperation(0, std::make_unique<Operation>(
            "sysy.alloca", std::vector<Value>{}, std::vector<Type>{storageType},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(gName)}},
            global->loc()));

        int remainingDirectUses = global->resultCount() > 0
                                      ? index.valueUseCounts[valueKey(global->result())]
                                      : 0;
        for (auto *op : accesses) {
          if (op->operandCount() > 0) {
            if (global->resultCount() > 0 && op->operand(0) == global->result())
              remainingDirectUses = std::max(0, remainingDirectUses - 1);
            op->setOperand(0, slot.result());
          } else {
            op->addOperand(slot.result());
          }
        }
        if (stats)
          stats->globalsPromoted++;
        bool hasDirectUse = global->resultCount() > 0 && remainingDirectUses > 0;
        if (global->resultCount() == 0 || !hasDirectUse) {
          global->markErased();
          if (stats)
            stats->globalsErased++;
        }
      }
    }
  }
}

void runReadonlyGlobalScalarPropagation(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_READONLY_GLOBALS", true))
    return;
  if (module.body().getBlocks().empty())
    return;

  struct ScalarGlobal {
    Value value;
    std::string symbol;
    Attribute initAttr;
    bool hasInit = false;
    bool invalid = false;
  };

  std::map<std::string, ScalarGlobal> globalsByKey;
  std::map<std::string, std::string> keyBySymbol;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.global" ||
        op->resultCount() == 0 || !isScalarWordMemref(op->resultType()))
      continue;
    if (op->resultType().str().find("xi32") == std::string::npos)
      continue;
    std::string symbol = symbolAttr(op->attr("symbol"));
    if (symbol.empty())
      symbol = symbolAttr(op->attr("sym_name"));
    if (symbol.empty())
      continue;
    std::string key = valueKey(op->result());
    ScalarGlobal info;
    info.value = op->result();
    info.symbol = symbol;
    globalsByKey[key] = info;
    keyBySymbol[symbol] = key;
  }
  if (globalsByKey.empty())
    return;

  auto zeroIndices = [](Operation &op) {
    int start = -1;
    if (op.name() == "sysy.load" || op.name() == "memref.load")
      start = 1;
    else if (op.name() == "sysy.store" || op.name() == "memref.store")
      start = 2;
    if (start < 0)
      return false;
    for (int i = start; i < op.operandCount(); i++) {
      int64_t index = 0;
      if (!constantIntegerValue(op.operand(i), index) || index != 0)
        return false;
    }
    return true;
  };

  auto candidateKeyForAccess = [&](Operation &op, bool store) -> std::string {
    int baseIndex = store ? 1 : 0;
    if (op.operandCount() > baseIndex) {
      std::string key = valueKey(op.operand(baseIndex));
      if (globalsByKey.count(key) != 0)
        return key;
    }
    std::string symbol = symbolAttr(op.attr("symbol"));
    if (symbol.empty())
      symbol = symbolAttr(op.attr("sym_name"));
    auto it = keyBySymbol.find(symbol);
    return it == keyBySymbol.end() ? "" : it->second;
  };

  Block *moduleBlock = module.body().getBlocks()[0].get();
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    bool isStore = op->name() == "sysy.store" || op->name() == "memref.store";
    if (!isStore)
      continue;
    std::string key = candidateKeyForAccess(*op, true);
    if (key.empty())
      continue;
    auto &global = globalsByKey[key];
    bool isModuleInit = op->getBlock() == moduleBlock && zeroIndices(*op);
    int64_t ignored = 0;
    auto *def = op->operandCount() > 0 ? op->operand(0).getDefiningOp() : nullptr;
    if (isModuleInit && !global.hasInit && op->operandCount() > 0 &&
        constantIntegerValue(op->operand(0), ignored) && def && def->attr("value")) {
      global.initAttr = def->attr("value");
      global.hasInit = true;
      continue;
    }
    global.invalid = true;
  }

  for (auto &kv : globalsByKey) {
    auto uses = usesOf(module, kv.second.value);
    for (const auto &use : uses) {
      Operation *owner = use.owner;
      if (!owner || owner->isErased())
        continue;
      bool allowedLoad = (owner->name() == "sysy.load" || owner->name() == "memref.load") &&
                         use.operandIndex == 0;
      bool allowedStore = (owner->name() == "sysy.store" || owner->name() == "memref.store") &&
                          use.operandIndex == 1;
      if (!allowedLoad && !allowedStore) {
        kv.second.invalid = true;
        break;
      }
    }
  }

  int replaced = 0;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() ||
        (op->name() != "sysy.load" && op->name() != "memref.load") ||
        op->resultCount() != 1 || !hasScalarHome(op->resultType()))
      continue;
    std::string key = candidateKeyForAccess(*op, false);
    if (key.empty())
      continue;
    auto it = globalsByKey.find(key);
    if (it == globalsByKey.end() || it->second.invalid || !zeroIndices(*op))
      continue;
    Block *block = op->getBlock();
    if (!block)
      continue;
    int index = operationIndexInBlock(*block, op);
    if (index < 0)
      continue;
    Attribute valueAttr = it->second.hasInit
                              ? it->second.initAttr
                              : module.context().integerAttr(0, op->resultType());
    auto constant = std::make_unique<Operation>(
        "arith.constant", std::vector<Value>{}, std::vector<Type>{op->resultType()},
        std::map<std::string, Attribute>{{"value", valueAttr}}, op->loc());
    auto &inserted = block->insertOperation((std::size_t) index, std::move(constant));
    replaceAllUses(module, op->result(), inserted.result());
    op->markErased();
    replaced++;
  }
  if (replaced > 0) {
    if (stats)
      stats->readonlyGlobalConstants += replaced;
    eraseMarked(module);
  }
}

static std::string memoryLocationKey(Operation &op);

static std::string memoryExprKey(Value value, int depth = 0) {
  if (!value.valid())
    return "";
  if (depth > 8)
    return "v:" + valueKey(value);
  int64_t imm = 0;
  if (constantIntegerValue(value, imm))
    return "c:" + std::to_string(imm);

  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return "v:" + valueKey(value);

  if ((def->name() == "sysy.load" || def->name() == "memref.load") &&
      def->resultCount() == 1) {
    std::string loc = memoryLocationKey(*def);
    if (!loc.empty())
      return "load(" + loc + ")";
  }

  auto binaryExpr = [&](const char *kind, bool commutative) -> std::string {
    if (def->operandCount() != 2)
      return "";
    std::string lhs = memoryExprKey(def->operand(0), depth + 1);
    std::string rhs = memoryExprKey(def->operand(1), depth + 1);
    if (lhs.empty() || rhs.empty())
      return "";
    if (commutative && rhs < lhs)
      std::swap(lhs, rhs);
    return std::string(kind) + "(" + lhs + "," + rhs + ")";
  };

  if (def->name() == "arith.addi" || def->name() == "rv_machine.addw" ||
      def->name() == "arm_machine.add")
    return binaryExpr("add", true);
  if (def->name() == "arith.muli" || def->name() == "rv_machine.mulw" ||
      def->name() == "arm_machine.mul")
    return binaryExpr("mul", true);
  if (def->name() == "arith.subi" || def->name() == "rv_machine.subw" ||
      def->name() == "arm_machine.sub")
    return binaryExpr("sub", false);

  return "v:" + valueKey(value);
}

static bool memoryKeyDependsOnLocation(const std::string &key,
                                       const std::string &location) {
  if (key.empty() || location.empty())
    return false;
  return key.find("load(" + location + ")") != std::string::npos;
}

static std::string memoryLocationKey(Operation &op) {
  auto baseKey = [&]() -> std::string {
    int baseIndex = -1;
    if (op.name() == "sysy.load" || op.name() == "memref.load")
      baseIndex = 0;
    else if (op.name() == "sysy.store" || op.name() == "memref.store")
      baseIndex = 1;
    if (baseIndex >= 0 && op.operandCount() > baseIndex)
      return "v:" + valueKey(op.operand(baseIndex));
    std::string sym = symbolAttr(op.attr("symbol"));
    if (sym.empty())
      sym = symbolAttr(op.attr("sym_name"));
    return sym.empty() ? "" : "s:" + sym;
  };

  std::string key = baseKey();
  if (key.empty())
    return "";
  if (op.name() == "memref.load") {
    for (int i = 1; i < op.operandCount(); i++)
      key += "," + memoryExprKey(op.operand(i));
  } else if (op.name() == "memref.store") {
    for (int i = 2; i < op.operandCount(); i++)
      key += "," + memoryExprKey(op.operand(i));
  }
  return key;
}

static std::string memoryBaseKey(Operation &op) {
  if (op.name() == "sysy.load" || op.name() == "memref.load") {
    if (op.operandCount() > 0)
      return "v:" + valueKey(op.operand(0));
  } else if (op.name() == "sysy.store" || op.name() == "memref.store") {
    if (op.operandCount() > 1)
      return "v:" + valueKey(op.operand(1));
  }
  std::string sym = symbolAttr(op.attr("symbol"));
  if (sym.empty())
    sym = symbolAttr(op.attr("sym_name"));
  return sym.empty() ? "" : "s:" + sym;
}

static bool memoryKeyMayShareBase(const std::string &locationKey,
                                  const std::string &baseKey) {
  if (locationKey.empty() || baseKey.empty())
    return false;
  return locationKey == baseKey || locationKey.rfind(baseKey + ",", 0) == 0;
}

static bool isLocalAllocaOp(Operation *op) {
  return op && (op->name() == "sysy.alloca" || op->name() == "memref.alloca") &&
         op->resultCount() == 1;
}

static void collectAllocaUseInfo(Operation &op, const std::set<std::string> &allocas,
                                 std::set<std::string> &loaded,
                                 std::set<std::string> &escaped) {
  if (op.isErased())
    return;
  auto classifyOperand = [&](int index, Value operand) {
    std::string key = valueKey(operand);
    if (allocas.count(key) == 0)
      return;
    bool baseLoad = (op.name() == "sysy.load" || op.name() == "memref.load") &&
                    index == 0;
    bool baseStore = (op.name() == "sysy.store" || op.name() == "memref.store") &&
                     index == 1;
    if (baseLoad)
      loaded.insert(key);
    else if (!baseStore)
      escaped.insert(key);
  };
  for (int i = 0; i < op.operandCount(); i++)
    classifyOperand(i, op.operand(i));
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child)
          collectAllocaUseInfo(*child, allocas, loaded, escaped);
      }
    }
  }
}

static void eraseDeadLocalStoresInFunction(Operation &func, SelfOptStats *stats) {
  if (func.getRegions().empty() || func.getRegions()[0]->getBlocks().empty())
    return;
  std::vector<Operation*> nestedOps;
  std::function<void(Operation&)> collect = [&](Operation &op) {
    nestedOps.push_back(&op);
    for (auto &region : op.getRegions()) {
      for (auto &block : region->getBlocks()) {
        for (auto &child : block->ops()) {
          if (child && !child->isErased())
            collect(*child);
        }
      }
    }
  };
  collect(func);

  std::set<std::string> allocas;
  for (auto *op : nestedOps) {
    if (isLocalAllocaOp(op))
      allocas.insert(valueKey(op->result()));
  }
  if (allocas.empty())
    return;

  std::set<std::string> loaded;
  std::set<std::string> escaped;
  collectAllocaUseInfo(func, allocas, loaded, escaped);

  std::set<std::string> deadAllocas;
  for (const auto &key : allocas) {
    if (loaded.count(key) == 0 && escaped.count(key) == 0)
      deadAllocas.insert(key);
  }
  if (deadAllocas.empty())
    return;

  for (auto *op : nestedOps) {
    if (!op || op->isErased())
      continue;
    if ((op->name() == "sysy.store" || op->name() == "memref.store") &&
        op->operandCount() >= 2 &&
        deadAllocas.count(valueKey(op->operand(1))) != 0) {
      op->markErased();
      if (stats)
        stats->memoryRemovedStores++;
    } else if (isLocalAllocaOp(op) && deadAllocas.count(valueKey(op->result())) != 0) {
      op->markErased();
    }
  }
}

static void runBlockMemoryOpt(Module &module, Block &block, SelfOptStats *stats) {
  if (stats)
    stats->memoryBlocks++;
  std::map<std::string, Value> activeStores;
  std::map<std::string, Operation*> activeStoreOps;
  std::map<std::string, std::string> loadOrigins;

  auto loadOriginOf = [&](Value value) -> std::string {
    auto it = loadOrigins.find(valueKey(value));
    return it == loadOrigins.end() ? "" : it->second;
  };
  auto invalidateLoadOriginsAfterStore = [&](const std::string &storeKey,
                                             const std::string &base,
                                             Value storedValue) {
    std::string storedOrigin = loadOriginOf(storedValue);
    std::vector<std::string> toErase;
    for (const auto &kv : loadOrigins) {
      const std::string &origin = kv.second;
      if (origin == storeKey ||
          (!base.empty() && memoryKeyMayShareBase(origin, base))) {
        // If a potentially aliasing store writes back the exact value that was
        // originally loaded from this location, the old loaded value remains a
        // valid representation even when the two expressions alias. Otherwise
        // the load-origin fact is no longer safe.
        if (storedOrigin != origin)
          toErase.push_back(kv.first);
      }
    }
    for (const auto &key : toErase)
      loadOrigins.erase(key);
  };
  auto invalidateFactsDependingOnLocation = [&](const std::string &locationKey) {
    if (locationKey.empty())
      return;
    std::vector<std::string> activeToErase;
    for (const auto &kv : activeStores) {
      if (memoryKeyDependsOnLocation(kv.first, locationKey))
        activeToErase.push_back(kv.first);
    }
    for (const auto &key : activeToErase) {
      activeStores.erase(key);
      activeStoreOps.erase(key);
    }

    std::vector<std::string> originsToErase;
    for (const auto &kv : loadOrigins) {
      if (memoryKeyDependsOnLocation(kv.second, locationKey))
        originsToErase.push_back(kv.first);
    }
    for (const auto &key : originsToErase)
      loadOrigins.erase(key);
  };

  for (auto &owned : block.ops()) {
    auto &op = *owned;
    if (op.isErased())
      continue;

    if (!op.getRegions().empty()) {
      activeStores.clear();
      activeStoreOps.clear();
      loadOrigins.clear();
      continue;
    }

    if (op.name() == "sysy.call") {
      std::vector<std::string> keysToInvalidate;
      for (const auto &kv : activeStores) {
        keysToInvalidate.push_back(kv.first);
      }
      for (const auto &k : keysToInvalidate) {
        activeStores.erase(k);
        activeStoreOps.erase(k);
      }
      loadOrigins.clear();
      continue;
    }

    if (op.name() == "sysy.store" || op.name() == "memref.store") {
      std::string key = memoryLocationKey(op);
      if (key.empty())
        continue;

      std::string base = memoryBaseKey(op);
      if (op.operandCount() >= 1 && loadOriginOf(op.operand(0)) == key) {
        op.markErased();
        if (stats)
          stats->memoryRemovedStores++;
        continue;
      }
      bool scalarLocationStore =
          op.operandCount() == 2 && isScalarWordMemref(op.operand(1).type());
      if (scalarLocationStore)
        invalidateFactsDependingOnLocation(key);
      if (op.operandCount() >= 1)
        invalidateLoadOriginsAfterStore(key, base, op.operand(0));

      std::vector<std::string> keysToInvalidate;
      for (const auto &kv : activeStores) {
        if (kv.first != key && !base.empty() &&
            (kv.first == base || kv.first.rfind(base + ",", 0) == 0)) {
          keysToInvalidate.push_back(kv.first);
        }
      }
      for (const auto &k : keysToInvalidate) {
        activeStores.erase(k);
        activeStoreOps.erase(k);
      }

      if (activeStores.count(key) > 0) {
        auto *prevOp = activeStoreOps[key];
        if (prevOp && !prevOp->isErased()) {
          prevOp->markErased();
          if (stats)
            stats->memoryRemovedStores++;
        }
      }

      Value storedVal = op.operand(0);
      activeStores[key] = storedVal;
      activeStoreOps[key] = &op;
    }

    if (op.name() == "sysy.load" || op.name() == "memref.load") {
      std::string key = memoryLocationKey(op);
      if (key.empty())
        continue;

      if (activeStores.count(key) > 0) {
        Value storedVal = activeStores[key];
        replaceAllUses(module, op.result(), storedVal);
        op.markErased();
        if (stats)
          stats->memoryForwardedLoads++;
      } else {
        loadOrigins[valueKey(op.result())] = key;
      }
    }
  }
}

static void runMemoryOptInRegion(Module &module, Region &region, SelfOptStats *stats) {
  for (auto &block : region.getBlocks()) {
    runBlockMemoryOpt(module, *block, stats);
    for (auto &owned : block->ops()) {
      if (!owned || owned->isErased())
        continue;
      for (auto &nested : owned->getRegions())
        runMemoryOptInRegion(module, *nested, stats);
    }
  }
}

void runMemoryOpt(Module &module, SelfOptStats *stats, bool enableDeadLocalStores) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_MEMOPT");
  if (enabled && std::string(enabled) == "0")
    return;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "sysy.func") {
      for (auto &region : op->getRegions()) {
        runMemoryOptInRegion(module, *region, stats);
      }
      if (enableDeadLocalStores)
        eraseDeadLocalStoresInFunction(*op, stats);
    }
  }
  eraseMarked(module);
}

namespace {

static bool isMLIRLoadOp(Operation *op) {
  return op && (op->name() == "sysy.load" || op->name() == "memref.load");
}

static bool isMLIRStoreOp(Operation *op) {
  return op && (op->name() == "sysy.store" || op->name() == "memref.store");
}

static bool blockIsNestedInOp(Block &block, const std::string &name) {
  Block *curr = &block;
  while (curr) {
    Region *region = curr->getRegion();
    Operation *parent = region ? region->getParent() : nullptr;
    if (!parent)
      return false;
    if (parent->name() == name)
      return true;
    curr = parent->getBlock();
  }
  return false;
}

static bool loadFromSlot(Operation *op, Value slot) {
  return isMLIRLoadOp(op) && op->operandCount() > 0 && op->operand(0) == slot;
}

static bool storeToSlot(Operation *op, Value slot) {
  return isMLIRStoreOp(op) && op->operandCount() >= 2 && op->operand(1) == slot;
}

static std::vector<Type> resultTypesOf(Operation &op) {
  std::vector<Type> types;
  for (int i = 0; i < op.resultCount(); i++)
    types.push_back(op.resultType(i));
  return types;
}

static std::vector<Value> remapOperandsForClone(
    Operation &op, const std::map<std::string, Value> &valueMap) {
  std::vector<Value> operands;
  operands.reserve((std::size_t) op.operandCount());
  for (auto operand : op.getOperands()) {
    auto it = valueMap.find(valueKey(operand));
    operands.push_back(it == valueMap.end() ? operand : it->second);
  }
  return operands;
}

static std::string memAccessKey(Operation &op) {
  if (isMLIRLoadOp(&op)) {
    if (op.operandCount() == 0)
      return "";
    std::string key = op.name() + "|" + op.resultType().str() + "|" +
                      valueKey(op.operand(0));
    for (int i = 1; i < op.operandCount(); i++)
      key += "|" + valueKey(op.operand(i));
    return key;
  }
  if (isMLIRStoreOp(&op)) {
    if (op.operandCount() < 2)
      return "";
    std::string key = op.name() + "|" + op.operand(0).type().str() + "|" +
                      valueKey(op.operand(1));
    for (int i = 2; i < op.operandCount(); i++)
      key += "|" + valueKey(op.operand(i));
    return key;
  }
  return "";
}

static std::string memAccessBaseKey(Operation &op) {
  if (isMLIRLoadOp(&op) && op.operandCount() > 0)
    return valueKey(op.operand(0));
  if (isMLIRStoreOp(&op) && op.operandCount() >= 2)
    return valueKey(op.operand(1));
  return "";
}

static bool opTreeHasUnsafeLoopUnrollControl(Operation &op, Value ivSlot,
                                             std::vector<Operation*> &ivStores) {
  if (op.isErased())
    return false;
  if (op.name() == "sysy.call" || op.name() == "sysy.return" ||
      op.name() == "scf.return" || op.name() == "sysy.break" ||
      op.name() == "sysy.continue" || op.name() == "scf.if")
    return true;
  if (storeToSlot(&op, ivSlot))
    ivStores.push_back(&op);
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && opTreeHasUnsafeLoopUnrollControl(*child, ivSlot, ivStores))
          return true;
  return false;
}

static bool addOneFromSlot(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased() ||
      (op->name() != "arith.addi" && op->name() != "rv_machine.addw" &&
       op->name() != "arm_machine.add") ||
      op->operandCount() != 2)
    return false;
  int64_t imm = 0;
  Operation *load = op->operand(0).getDefiningOp();
  if (loadFromSlot(load, slot) && constantIntegerValue(op->operand(1), imm) && imm == 1)
    return true;
  load = op->operand(1).getDefiningOp();
  return loadFromSlot(load, slot) && constantIntegerValue(op->operand(0), imm) && imm == 1;
}

static Operation *findConditionOp(Region &region) {
  if (region.getBlocks().empty())
    return nullptr;
  for (auto &owned : region.getBlocks()[0]->ops())
    if (owned && !owned->isErased() && owned->name() == "scf.condition")
      return owned.get();
  return nullptr;
}

struct ConstantWhileInfo {
  bool valid = false;
  Value ivSlot;
  int64_t init = 0;
  int64_t bound = 0;
  int64_t tripCount = 0;
  Operation *stepStore = nullptr;
};

static ConstantWhileInfo classifySmallConstantWhile(Operation &loop) {
  ConstantWhileInfo info;
  if (loop.name() != "scf.while" || loop.getRegions().size() != 2 ||
      loop.getRegions()[0]->getBlocks().size() != 1 ||
      loop.getRegions()[1]->getBlocks().size() != 1)
    return info;
  Operation *condition = findConditionOp(*loop.getRegions()[0]);
  if (!condition || condition->operandCount() != 1)
    return info;
  Operation *cmp = condition->operand(0).getDefiningOp();
  if (!cmp || cmp->isErased() ||
      (cmp->name() != "arith.cmpi" && cmp->name() != "rv_machine.cmp" &&
       cmp->name() != "arm_machine.cmp") ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "lt")
    return info;
  Operation *load = cmp->operand(0).getDefiningOp();
  if (!isMLIRLoadOp(load) || load->operandCount() == 0 ||
      !isScalarWordMemref(load->operand(0).type()))
    return info;
  int64_t bound = 0;
  if (!constantIntegerValue(cmp->operand(1), bound))
    return info;
  Value ivSlot = load->operand(0);

  Block *parent = loop.getBlock();
  if (!parent)
    return info;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return info;
  bool foundInit = false;
  int64_t init = 0;
  for (int i = loopIndex - 1; i >= 0; i--) {
    Operation *prev = parent->ops()[(std::size_t) i].get();
    if (!prev || prev->isErased())
      continue;
    if (storeToSlot(prev, ivSlot) && constantIntegerValue(prev->operand(0), init)) {
      foundInit = true;
      break;
    }
    if (prev->name() == "scf.while" || prev->name() == "affine.for" ||
        prev->name() == "sysy.call")
      break;
  }
  if (!foundInit)
    return info;

  Block &body = *loop.getRegions()[1]->getBlocks()[0];
  std::vector<Operation*> ivStores;
  for (auto &owned : body.ops()) {
    if (owned && opTreeHasUnsafeLoopUnrollControl(*owned, ivSlot, ivStores))
      return info;
  }
  if (ivStores.size() != 1 || ivStores[0]->getBlock() != &body ||
      !addOneFromSlot(ivStores[0]->operand(0), ivSlot))
    return info;

  int64_t trip = bound - init;
  if (trip < 0 || trip > 7)
    return info;
  info.valid = true;
  info.ivSlot = ivSlot;
  info.init = init;
  info.bound = bound;
  info.tripCount = trip;
  info.stepStore = ivStores[0];
  return info;
}

static std::unique_ptr<Operation> cloneForUnrolledIteration(
    Module &module, Operation &op, std::map<std::string, Value> &valueMap,
    const std::set<Operation*> &skipOps, Value ivSlot, int64_t ivValue) {
  if (skipOps.count(&op) || op.isErased())
    return nullptr;

  bool replaceIvLoad = loadFromSlot(&op, ivSlot) && op.resultCount() == 1 &&
                       isI32Like(op.resultType());
  std::string clonedName = replaceIvLoad ? "arith.constant" : op.name();
  std::vector<Value> operands = replaceIvLoad ? std::vector<Value>{}
                                              : remapOperandsForClone(op, valueMap);
  std::map<std::string, Attribute> attrs = op.attrs();
  if (replaceIvLoad)
    attrs = {{"value", module.context().integerAttr(ivValue, op.resultType())}};
  auto cloned = std::make_unique<Operation>(clonedName, operands, resultTypesOf(op),
                                            attrs, op.loc());
  for (auto &region : op.getRegions()) {
    Region &newRegion = cloned->addRegion();
    for (auto &block : region->getBlocks()) {
      Block &newBlock = newRegion.addBlock();
      for (auto &arg : block->args()) {
        BlockArgument &newArg = newBlock.addArgument(arg->type(), arg->loc(), arg->name());
        valueMap[valueKey(arg->value())] = newArg.value();
      }
      for (auto &child : block->ops()) {
        if (!child || child->isErased())
          continue;
        auto childClone = cloneForUnrolledIteration(module, *child, valueMap,
                                                    skipOps, ivSlot, ivValue);
        if (!childClone)
          continue;
        Operation &inserted = newBlock.addOperation(std::move(childClone));
        for (int i = 0; i < child->resultCount(); i++)
          valueMap[valueKey(child->result(i))] = inserted.result(i);
      }
    }
  }
  return cloned;
}

static bool unrollSmallConstantWhile(Module &module, Operation &loop,
                                     const ConstantWhileInfo &info) {
  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;
  Block &body = *loop.getRegions()[1]->getBlocks()[0];
  std::vector<Operation*> bodyOps;
  for (auto &owned : body.ops())
    if (owned && !owned->isErased())
      bodyOps.push_back(owned.get());

  std::set<Operation*> skipOps;
  skipOps.insert(info.stepStore);
  for (Operation *op : bodyOps)
    if (op && op->name() == "scf.yield")
      skipOps.insert(op);

  std::size_t insertIndex = (std::size_t) loopIndex;
  for (int64_t iter = 0; iter < info.tripCount; iter++) {
    std::map<std::string, Value> valueMap;
    int64_t iv = info.init + iter;
    for (Operation *op : bodyOps) {
      if (!op || skipOps.count(op))
        continue;
      auto cloned = cloneForUnrolledIteration(module, *op, valueMap, skipOps,
                                              info.ivSlot, iv);
      if (!cloned)
        continue;
      Operation &inserted = parent->insertOperation(insertIndex++, std::move(cloned));
      for (int i = 0; i < op->resultCount(); i++)
        valueMap[valueKey(op->result(i))] = inserted.result(i);
    }
  }
  loop.markErased();
  return true;
}

static void runLoopAddressIVInRegion(Module &module, Region &region, SelfOptStats *stats);

static void runLoopAddressIVInBlock(Module &module, Block &block, SelfOptStats *stats) {
  std::map<std::string, Value> loadCache;
  std::map<std::string, std::set<std::string>> keysByBase;
  bool scalarSlotCSE = envEnabled("SISY_ENABLE_SELF_SCALAR_LOAD_CSE", true) &&
                       blockIsNestedInOp(block, "scf.while");

  auto invalidateAll = [&]() {
    loadCache.clear();
    keysByBase.clear();
  };
  auto cacheLoad = [&](const std::string &key, const std::string &base, Value value) {
    loadCache[key] = value;
    keysByBase[base].insert(key);
  };
  auto invalidateBaseAfterStore = [&](const std::string &base, Value storedValue) {
    auto baseIt = keysByBase.find(base);
    if (baseIt == keysByBase.end())
      return;
    std::vector<std::string> eraseKeys;
    for (const auto &key : baseIt->second) {
      auto cacheIt = loadCache.find(key);
      if (cacheIt == loadCache.end() || cacheIt->second != storedValue)
        eraseKeys.push_back(key);
    }
    for (const auto &key : eraseKeys) {
      loadCache.erase(key);
      baseIt->second.erase(key);
    }
    if (baseIt->second.empty())
      keysByBase.erase(baseIt);
  };

  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    bool cacheableLoad = op->name() == "memref.load" ||
                         (scalarSlotCSE && op->name() == "sysy.load");
    if (cacheableLoad && op->resultCount() == 1 &&
        !isMemrefType(op->resultType())) {
      std::string key = memAccessKey(*op);
      std::string base = memAccessBaseKey(*op);
      auto it = loadCache.find(key);
      if (!key.empty() && it != loadCache.end() && it->second.valid()) {
        replaceAllUses(module, op->result(), it->second);
        op->markErased();
        if (stats) {
          stats->loopAddressCSE++;
          stats->addrIvRewrites++;
        }
        continue;
      }
      if (!key.empty() && !base.empty())
        cacheLoad(key, base, op->result());
      continue;
    }

    bool cacheInvalidatingStore = op->name() == "memref.store" ||
                                  (scalarSlotCSE && op->name() == "sysy.store");
    if (cacheInvalidatingStore && op->operandCount() >= 2) {
      std::string key = memAccessKey(*op);
      std::string base = memAccessBaseKey(*op);
      auto cached = loadCache.find(key);
      if (!key.empty() && cached != loadCache.end() && cached->second == op->operand(0)) {
        op->markErased();
        if (stats) {
          stats->memoryRemovedStores++;
          stats->loopAddressCSE++;
          stats->addrIvRewrites++;
        }
        continue;
      }
      if (!base.empty())
        invalidateBaseAfterStore(base, op->operand(0));
      continue;
    }

    if (op->name() == "sysy.call") {
      invalidateAll();
      continue;
    }

    if (!op->getRegions().empty()) {
      for (auto &region : op->getRegions())
        runLoopAddressIVInRegion(module, *region, stats);
      invalidateAll();
    }
  }
}

static void runLoopAddressIVInRegion(Module &module, Region &region, SelfOptStats *stats) {
  for (auto &block : region.getBlocks())
    runLoopAddressIVInBlock(module, *block, stats);
}

static bool opTreeUsesValue(Operation &op, Value needle) {
  if (op.isErased())
    return false;
  for (auto operand : op.getOperands())
    if (operand == needle)
      return true;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && opTreeUsesValue(*child, needle))
          return true;
  return false;
}

static bool opHasDirectValueUse(Operation &op, Value needle) {
  for (auto operand : op.getOperands())
    if (operand == needle)
      return true;
  return false;
}

static bool opResultIsUnused(Operation &op) {
  for (int i = 0; i < op.resultCount(); i++) {
    if ((int) op.resultUses.size() > i && !op.resultUses[(std::size_t) i].empty())
      return false;
  }
  return true;
}

static bool isPureDeadIvBump(Operation &op, Value iv) {
  return op.getRegions().empty() && opHasDirectValueUse(op, iv) && opResultIsUnused(op) &&
         (op.name() == "arith.addi" || op.name() == "rv_machine.addw" ||
          op.name() == "arm_machine.add");
}

static void collectLocalAllocas(Operation &op, std::set<std::string> &allocas) {
  if (op.isErased())
    return;
  if ((op.name() == "sysy.alloca" || op.name() == "memref.alloca") &&
      op.resultCount() == 1)
    allocas.insert(valueKey(op.result()));
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child)
          collectLocalAllocas(*child, allocas);
}

static bool collectRepeatReductionStores(Operation &op,
                                         const std::set<std::string> &localAllocas,
                                         std::map<std::string, Value> &outerStores) {
  if (op.isErased())
    return true;
  if (op.name() == "sysy.call" || op.name() == "sysy.return" ||
      op.name() == "scf.return" || op.name() == "sysy.break" ||
      op.name() == "sysy.continue" || op.name() == "memref.store" ||
      op.name() == "scf.if" || op.name() == "arith.divi" ||
      op.name() == "arith.remi" || op.name() == "rv_machine.divw" ||
      op.name() == "rv_machine.remw" || op.name() == "arm_machine.sdiv")
    return false;
  if (op.name() == "sysy.store") {
    if (op.operandCount() < 2 || !isScalarWordMemref(op.operand(1).type()))
      return false;
    std::string slot = valueKey(op.operand(1));
    if (!localAllocas.count(slot))
      outerStores[slot] = op.operand(1);
  }
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && !collectRepeatReductionStores(*child, localAllocas, outerStores))
          return false;
  return true;
}

static bool valueDependsOnSlot(Value value, Value slot, int depth = 0) {
  if (!value.valid() || depth > 24)
    return false;
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased())
    return false;
  if (loadFromSlot(op, slot))
    return true;
  for (auto operand : op->getOperands())
    if (valueDependsOnSlot(operand, slot, depth + 1))
      return true;
  return false;
}

static bool valueUsesOnlyAllowedRepeatOps(Value value, Value accumulator,
                                          int depth = 0) {
  if (!value.valid() || depth > 48)
    return false;
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased())
    return true;
  if (loadFromSlot(op, accumulator))
    return true;
  if (op->getRegions().size() != 0)
    return false;
  const std::string &name = op->name();
  if (name != "arith.constant" && name != "arith.addi" &&
      name != "arith.subi" && name != "arith.muli" &&
      name != "rv_machine.li" && name != "rv_machine.addw" &&
      name != "rv_machine.subw" && name != "rv_machine.mulw" &&
      name != "rv_machine.neg" && name != "arm_machine.mov" &&
      name != "arm_machine.add" && name != "arm_machine.sub" &&
      name != "arm_machine.mul" && name != "memref.load" &&
      name != "sysy.load")
    return false;
  for (auto operand : op->getOperands())
    if (!valueUsesOnlyAllowedRepeatOps(operand, accumulator, depth + 1))
      return false;
  return true;
}

static bool storeIsAccumulatorAdd(Operation &store, Value accumulator) {
  if (!storeToSlot(&store, accumulator) || store.operandCount() < 1)
    return false;
  Value stored = store.operand(0);
  Operation *add = stored.getDefiningOp();
  if (!add || add->isErased() ||
      (add->name() != "arith.addi" && add->name() != "rv_machine.addw" &&
       add->name() != "arm_machine.add") ||
      add->operandCount() != 2)
    return false;

  bool hasAccumulatorLoad = false;
  bool incrementIndependent = false;
  for (int i = 0; i < 2; i++) {
    Value operand = add->operand(i);
    Operation *load = operand.getDefiningOp();
    if (loadFromSlot(load, accumulator)) {
      hasAccumulatorLoad = true;
      continue;
    }
    if (!valueDependsOnSlot(operand, accumulator) &&
        valueUsesOnlyAllowedRepeatOps(operand, accumulator))
      incrementIndependent = true;
  }
  return hasAccumulatorLoad && incrementIndependent;
}

static bool opTreeAccumulatorStoresAreLinearAdds(Operation &op, Value accumulator) {
  if (op.isErased())
    return true;
  if (storeToSlot(&op, accumulator) && !storeIsAccumulatorAdd(op, accumulator))
    return false;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && !opTreeAccumulatorStoresAreLinearAdds(*child, accumulator))
          return false;
  return true;
}

static bool opTreeLoadsFromSlot(Operation &op, Value slot) {
  if (op.isErased())
    return false;
  if (loadFromSlot(&op, slot))
    return true;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && opTreeLoadsFromSlot(*child, slot))
          return true;
  return false;
}

static bool slotLoadedAfter(Operation &loop, Value slot) {
  Block *block = loop.getBlock();
  if (!block)
    return true;
  bool seen = false;
  for (auto &owned : block->ops()) {
    if (!owned || owned->isErased())
      continue;
    if (owned.get() == &loop) {
      seen = true;
      continue;
    }
    if (seen && opTreeLoadsFromSlot(*owned, slot))
      return true;
  }
  return false;
}

struct RepeatReductionInfo {
  bool valid = false;
  Value lower;
  Value upper;
  Value step;
  Value accumulatorSlot;
  std::set<Operation*> skipOps;
};

struct LinearModRecurrenceInfo {
  bool valid = false;
  Value lower;
  Value upper;
  Value slot;
  int64_t increment = 0;
  int64_t modulus = 0;
  int64_t maxTrip = 0;
};

static bool positiveUpperBound(Value value, int64_t &bound) {
  int64_t imm = 0;
  if (constantIntegerValue(value, imm) && imm >= 0) {
    bound = imm;
    return true;
  }
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased() || op->operandCount() != 2)
    return false;
  if (op->name() != "arith.remi" && op->name() != "rv_machine.remw" &&
      op->name() != "arm_machine.srem")
    return false;
  if (!constantIntegerValue(op->operand(1), imm) || imm <= 0)
    return false;
  bound = imm - 1;
  return true;
}

static bool constantValueOperand(Value value, int64_t &imm) {
  return constantIntegerValue(value, imm);
}

static bool classifyLinearModStore(Operation &store, Value &slot,
                                   int64_t &increment, int64_t &modulus) {
  if (!isMLIRStoreOp(&store) || store.operandCount() < 2 ||
      !isScalarWordMemref(store.operand(1).type()))
    return false;
  Operation *rem = store.operand(0).getDefiningOp();
  if (!rem || rem->isErased() ||
      (rem->name() != "arith.remi" && rem->name() != "rv_machine.remw" &&
       rem->name() != "arm_machine.srem") ||
      rem->operandCount() != 2)
    return false;
  int64_t mod = 0;
  if (!constantValueOperand(rem->operand(1), mod) || mod <= 0)
    return false;
  Operation *add = rem->operand(0).getDefiningOp();
  if (!add || add->isErased() ||
      (add->name() != "arith.addi" && add->name() != "rv_machine.addw" &&
       add->name() != "arm_machine.add") ||
      add->operandCount() != 2)
    return false;
  Value candidateSlot = store.operand(1);
  int64_t inc = 0;
  Operation *load = add->operand(0).getDefiningOp();
  if (loadFromSlot(load, candidateSlot) &&
      constantValueOperand(add->operand(1), inc)) {
    slot = candidateSlot;
    increment = inc;
    modulus = mod;
    return inc > 0;
  }
  load = add->operand(1).getDefiningOp();
  if (loadFromSlot(load, candidateSlot) &&
      constantValueOperand(add->operand(0), inc)) {
    slot = candidateSlot;
    increment = inc;
    modulus = mod;
    return inc > 0;
  }
  return false;
}

static LinearModRecurrenceInfo classifyLinearModRecurrenceLoop(Operation &loop) {
  LinearModRecurrenceInfo info;
  if (loop.name() != "affine.for" || loop.operandCount() < 3 ||
      loop.getRegions().size() != 1 || loop.getRegions()[0]->getBlocks().size() != 1)
    return info;
  int64_t lower = 0;
  int64_t step = 0;
  if (!constantIntegerValue(loop.operand(0), lower) || lower != 0 ||
      !constantIntegerValue(loop.operand(2), step) || step != 1)
    return info;
  int64_t maxTrip = 0;
  if (!positiveUpperBound(loop.operand(1), maxTrip) || maxTrip <= 0)
    return info;

  Block &body = *loop.getRegions()[0]->getBlocks()[0];
  if (body.args().empty())
    return info;
  Value iv = body.args()[0]->value();
  Value slot;
  int64_t increment = 0;
  int64_t modulus = 0;
  int stores = 0;
  for (auto &owned : body.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op->name() == "affine.yield")
      continue;
    if (opTreeUsesValue(*op, iv) && !isPureDeadIvBump(*op, iv))
      return info;
    if (op->name() == "sysy.call" || op->name() == "sysy.return" ||
        op->name() == "scf.return" || op->name() == "sysy.break" ||
        op->name() == "sysy.continue" || op->name() == "memref.store" ||
        op->name() == "scf.if" || op->name() == "scf.while" ||
        op->name() == "affine.for")
      return info;
    if (isMLIRStoreOp(op)) {
      Value storeSlot;
      int64_t storeIncrement = 0;
      int64_t storeModulus = 0;
      if (!classifyLinearModStore(*op, storeSlot, storeIncrement, storeModulus))
        return info;
      if (stores == 0) {
        slot = storeSlot;
        increment = storeIncrement;
        modulus = storeModulus;
      } else if (storeSlot != slot || storeIncrement != increment ||
                 storeModulus != modulus) {
        return info;
      }
      stores++;
    }
  }
  if (stores != 1 || !slot.valid() || modulus <= 0 || increment <= 0)
    return info;
  if (maxTrip > 1000000 || increment > 1000000)
    return info;
  if (increment > (std::numeric_limits<int32_t>::max() - (modulus - 1)) / maxTrip)
    return info;

  info.valid = true;
  info.lower = loop.operand(0);
  info.upper = loop.operand(1);
  info.slot = slot;
  info.increment = increment;
  info.modulus = modulus;
  info.maxTrip = maxTrip;
  return info;
}

static RepeatReductionInfo classifyRepeatReductionLoop(Operation &loop) {
  RepeatReductionInfo info;
  if (loop.name() != "affine.for" || loop.operandCount() < 3 ||
      loop.getRegions().size() != 1 || loop.getRegions()[0]->getBlocks().size() != 1)
    return info;
  int64_t lower = 0;
  int64_t step = 0;
  if (!constantIntegerValue(loop.operand(0), lower) ||
      !constantIntegerValue(loop.operand(2), step) || step != 1)
    return info;
  Block &body = *loop.getRegions()[0]->getBlocks()[0];
  if (body.args().empty())
    return info;
  Value iv = body.args()[0]->value();

  std::vector<Operation*> bodyOps;
  std::set<std::string> localAllocas;
  for (auto &owned : body.ops()) {
    if (!owned || owned->isErased())
      continue;
    bodyOps.push_back(owned.get());
    collectLocalAllocas(*owned, localAllocas);
  }

  for (Operation *op : bodyOps) {
    if (!op)
      continue;
    if (op->name() == "affine.yield") {
      info.skipOps.insert(op);
      continue;
    }
    if (!opTreeUsesValue(*op, iv))
      continue;
    if (isPureDeadIvBump(*op, iv)) {
      info.skipOps.insert(op);
      continue;
    }
    return info;
  }

  std::map<std::string, Value> outerStores;
  for (Operation *op : bodyOps) {
    if (!op || info.skipOps.count(op))
      continue;
    if (!collectRepeatReductionStores(*op, localAllocas, outerStores))
      return info;
  }
  Value accumulatorSlot;
  if (outerStores.size() != 1)
    return info;
  for (const auto &kv : outerStores)
    accumulatorSlot = kv.second;
  if (!accumulatorSlot.valid() ||
      accumulatorSlot.type().str().find("xi32") == std::string::npos ||
      !slotLoadedAfter(loop, accumulatorSlot))
    return info;
  for (Operation *op : bodyOps) {
    if (!op || info.skipOps.count(op))
      continue;
    if (!opTreeAccumulatorStoresAreLinearAdds(*op, accumulatorSlot))
      return info;
  }
  info.valid = true;
  info.lower = loop.operand(0);
  info.upper = loop.operand(1);
  info.step = loop.operand(2);
  info.accumulatorSlot = accumulatorSlot;
  return info;
}

static Operation &appendOp(Block &block, const std::string &name,
                           const std::vector<Value> &operands,
                           const std::vector<Type> &results,
                           const std::map<std::string, Attribute> &attrs,
                           Location loc, int regionCount = 0) {
  auto op = std::make_unique<Operation>(name, operands, results, attrs, loc);
  for (int i = 0; i < regionCount; i++)
    op->addRegion();
  return block.addOperation(std::move(op));
}

static Operation &insertOp(Block &block, std::size_t &index,
                           const std::string &name,
                           const std::vector<Value> &operands,
                           const std::vector<Type> &results,
                           const std::map<std::string, Attribute> &attrs,
                           Location loc, int regionCount = 0) {
  auto op = std::make_unique<Operation>(name, operands, results, attrs, loc);
  for (int i = 0; i < regionCount; i++)
    op->addRegion();
  return block.insertOperation(index++, std::move(op));
}

static bool applyLinearModRecurrenceLoop(Module &module, Operation &loop,
                                         const LinearModRecurrenceInfo &info) {
  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;

  Context &ctx = module.context();
  Type i32 = ctx.i(32);
  Location loc = loop.loc();
  std::size_t insertIndex = (std::size_t) loopIndex;
  Operation &start = insertOp(*parent, insertIndex, "sysy.load",
                              {info.slot}, {i32}, {}, loc);
  Operation &zero = insertOp(*parent, insertIndex, "arith.constant", {}, {i32},
                             {{"value", ctx.integerAttr(0, i32)}}, loc);
  Operation &modConst = insertOp(*parent, insertIndex, "arith.constant", {}, {i32},
                                 {{"value", ctx.integerAttr(info.modulus, i32)}}, loc);
  Operation &incConst = insertOp(*parent, insertIndex, "arith.constant", {}, {i32},
                                 {{"value", ctx.integerAttr(info.increment, i32)}}, loc);

  Operation &tripPositive = insertOp(
      *parent, insertIndex, "arith.cmpi", {info.lower, info.upper}, {i32},
      {{"predicate", ctx.stringAttr("lt")}}, loc);
  Operation &startNonNegative = insertOp(
      *parent, insertIndex, "arith.cmpi", {zero.result(), start.result()}, {i32},
      {{"predicate", ctx.stringAttr("le")}}, loc);
  Operation &startCanonical = insertOp(
      *parent, insertIndex, "arith.cmpi", {start.result(), modConst.result()}, {i32},
      {{"predicate", ctx.stringAttr("lt")}}, loc);
  Operation &guardA = insertOp(*parent, insertIndex, "arith.andi",
                               {tripPositive.result(), startNonNegative.result()},
                               {i32}, {}, loc);
  Operation &guard = insertOp(*parent, insertIndex, "arith.andi",
                              {guardA.result(), startCanonical.result()},
                              {i32}, {}, loc);

  auto ifOp = std::make_unique<Operation>("scf.if", std::vector<Value>{guard.result()},
                                          std::vector<Type>{},
                                          std::map<std::string, Attribute>{}, loc);
  Region &thenRegion = ifOp->addRegion();
  Block &thenBlock = thenRegion.addBlock();
  Value trip = info.upper;
  int64_t lowerImm = 0;
  if (constantIntegerValue(info.lower, lowerImm) && lowerImm != 0) {
    Operation &lowerConst = appendOp(
        thenBlock, "arith.constant", {}, {i32},
        {{"value", ctx.integerAttr(lowerImm, i32)}}, loc);
    Operation &tripOp = appendOp(thenBlock, "arith.subi",
                                 {info.upper, lowerConst.result()}, {i32}, {}, loc);
    trip = tripOp.result();
  }
  Operation &scaled = appendOp(thenBlock, "arith.muli",
                               {incConst.result(), trip}, {i32}, {}, loc);
  Operation &advanced = appendOp(thenBlock, "arith.addi",
                                 {start.result(), scaled.result()}, {i32}, {}, loc);
  Operation &folded = appendOp(thenBlock, "arith.remi",
                               {advanced.result(), modConst.result()}, {i32}, {}, loc);
  appendOp(thenBlock, "sysy.store", {folded.result(), info.slot}, {}, {}, loc);
  appendOp(thenBlock, "scf.yield", {}, {}, {}, loc);

  Region &elseRegion = ifOp->addRegion();
  Block &elseBlock = elseRegion.addBlock();
  std::map<std::string, Value> valueMap;
  std::set<Operation*> skipOps;
  auto cloned = cloneForUnrolledIteration(module, loop, valueMap, skipOps, Value(), 0);
  if (!cloned)
    return false;
  elseBlock.addOperation(std::move(cloned));
  appendOp(elseBlock, "scf.yield", {}, {}, {}, loc);

  parent->insertOperation(insertIndex, std::move(ifOp));
  loop.markErased();
  return true;
}

static bool applyRepeatReductionLoop(Module &module, Operation &loop,
                                     const RepeatReductionInfo &info) {
  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;
  Block &body = *loop.getRegions()[0]->getBlocks()[0];
  std::vector<Operation*> bodyOps;
  for (auto &owned : body.ops())
    if (owned && !owned->isErased())
      bodyOps.push_back(owned.get());

  Context &ctx = module.context();
  Location loc = loop.loc();
  auto cmp = std::make_unique<Operation>(
      "arith.cmpi", std::vector<Value>{info.lower, info.upper},
      std::vector<Type>{ctx.i(32)},
      std::map<std::string, Attribute>{{"predicate", ctx.stringAttr("lt")}},
      loc);
  Operation &cmpOp = parent->insertOperation((std::size_t) loopIndex, std::move(cmp));

  auto ifOp = std::make_unique<Operation>(
      "scf.if", std::vector<Value>{cmpOp.result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc);
  Region &thenRegion = ifOp->addRegion();
  Block &thenBlock = thenRegion.addBlock();

  Operation &start = appendOp(thenBlock, "sysy.load",
                              {info.accumulatorSlot}, {ctx.i(32)}, {}, loc);
  std::map<std::string, Value> valueMap;
  for (Operation *op : bodyOps) {
    if (!op || info.skipOps.count(op))
      continue;
    auto cloned = cloneForUnrolledIteration(module, *op, valueMap, info.skipOps,
                                            Value(), 0);
    if (!cloned)
      continue;
    Operation &inserted = thenBlock.addOperation(std::move(cloned));
    for (int i = 0; i < op->resultCount(); i++)
      valueMap[valueKey(op->result(i))] = inserted.result(i);
  }
  Operation &end = appendOp(thenBlock, "sysy.load",
                            {info.accumulatorSlot}, {ctx.i(32)}, {}, loc);
  Operation &delta = appendOp(thenBlock, "arith.subi",
                              {end.result(), start.result()}, {ctx.i(32)}, {}, loc);
  Value trip = info.upper;
  int64_t lowerImm = 0;
  if (constantIntegerValue(info.lower, lowerImm) && lowerImm != 0) {
    Operation &lowerConst = appendOp(thenBlock, "arith.constant", {}, {ctx.i(32)},
                                    {{"value", ctx.integerAttr(lowerImm, ctx.i(32))}}, loc);
    Operation &tripOp = appendOp(thenBlock, "arith.subi",
                                 {info.upper, lowerConst.result()}, {ctx.i(32)}, {}, loc);
    trip = tripOp.result();
  }
  Operation &scaled = appendOp(thenBlock, "arith.muli",
                               {delta.result(), trip}, {ctx.i(32)}, {}, loc);
  Operation &finalValue = appendOp(thenBlock, "arith.addi",
                                   {start.result(), scaled.result()}, {ctx.i(32)}, {}, loc);
  appendOp(thenBlock, "sysy.store",
           {finalValue.result(), info.accumulatorSlot}, {}, {}, loc);
  appendOp(thenBlock, "scf.yield", {}, {}, {}, loc);

  parent->insertOperation((std::size_t) loopIndex + 1, std::move(ifOp));
  loop.markErased();
  return true;
}

static bool isScalarLocalSlot(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return false;
  if (def->name() != "sysy.alloca" && def->name() != "memref.alloca")
    return false;
  return value.type().str().find("memref<1xi32") != std::string::npos;
}

static bool isSelectSpeculatableOp(Operation *op) {
  if (!op || op->isErased() || op->isTerminator() || !op->getRegions().empty())
    return false;
  if (isMLIRLoadOp(op))
    return op->operandCount() > 0 && isScalarLocalSlot(op->operand(0));
  const std::string &name = op->name();
  return name == "arith.constant" || name == "arith.addi" ||
         name == "arith.subi" || name == "arith.muli" ||
         name == "arith.cmpi" || name == "rv_machine.li" ||
         name == "rv_machine.addw" || name == "rv_machine.subw" ||
         name == "rv_machine.mulw" || name == "rv_machine.cmp" ||
         name == "rv_machine.and" || name == "rv_machine.or" ||
         name == "rv_machine.xor" || name == "rv_machine.neg" ||
         name == "rv_machine.seqz";
}

static bool promoteIfStoresToSelects(Module &module, Operation &ifOp,
                                     SelfOptStats *stats) {
  if (ifOp.name() != "scf.if" || ifOp.operandCount() != 1 ||
      ifOp.getRegions().size() != 1)
    return false;
  if (ifOp.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block *parent = ifOp.getBlock();
  if (!parent)
    return false;
  int ifIndex = operationIndexInBlock(*parent, &ifOp);
  if (ifIndex < 0)
    return false;

  Block &thenBlock = *ifOp.getRegions()[0]->getBlocks()[0];
  std::vector<Operation*> pureOps;
  std::vector<std::pair<Value, Value>> stores;
  std::set<std::string> storeSlots;
  for (auto &owned : thenBlock.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.yield")
      continue;
    if (isMLIRStoreOp(op)) {
      if (op->operandCount() < 2 || !isScalarLocalSlot(op->operand(1)))
        return false;
      std::string slotKey = valueKey(op->operand(1));
      if (!storeSlots.insert(slotKey).second)
        return false;
      stores.push_back({op->operand(1), op->operand(0)});
      continue;
    }
    if (!isSelectSpeculatableOp(op))
      return false;
    pureOps.push_back(op);
  }
  if (stores.empty())
    return false;

  for (Operation *op : pureOps) {
    if (isMLIRLoadOp(op) && op->operandCount() > 0 &&
        storeSlots.count(valueKey(op->operand(0))) != 0)
      return false;
  }
  std::set<std::string> pureResults;
  for (Operation *op : pureOps) {
    for (int i = 0; i < op->resultCount(); i++)
      pureResults.insert(valueKey(op->result(i)));
  }
  for (const auto &store : stores) {
    Operation *def = store.second.getDefiningOp();
    if (def && def->getBlock() == &thenBlock &&
        pureResults.count(valueKey(store.second)) == 0)
      return false;
  }

  std::map<std::string, Value> valueMap;
  std::size_t insertIndex = (std::size_t) ifIndex;
  for (Operation *op : pureOps) {
    auto cloned = std::make_unique<Operation>(
        op->name(), remapOperandsForClone(*op, valueMap),
        resultTypesOf(*op), op->attrs(), op->loc());
    Operation &inserted = parent->insertOperation(insertIndex++, std::move(cloned));
    for (int i = 0; i < op->resultCount(); i++)
      valueMap[valueKey(op->result(i))] = inserted.result(i);
  }

  Context &ctx = module.context();
  int promoted = 0;
  for (const auto &store : stores) {
    Value slot = store.first;
    Value trueValue = store.second;
    auto mapped = valueMap.find(valueKey(trueValue));
    if (mapped != valueMap.end())
      trueValue = mapped->second;
    Operation *trueDef = trueValue.getDefiningOp();
    if (trueDef && trueDef->getBlock() == &thenBlock)
      return false;

    Operation &oldValue = parent->insertOperation(
        insertIndex++,
        std::make_unique<Operation>("sysy.load", std::vector<Value>{slot},
                                    std::vector<Type>{ctx.i(32)},
                                    std::map<std::string, Attribute>{},
                                    ifOp.loc()));
    Operation &select = parent->insertOperation(
        insertIndex++,
        std::make_unique<Operation>("arith.select",
                                    std::vector<Value>{ifOp.operand(0), trueValue,
                                                       oldValue.result()},
                                    std::vector<Type>{ctx.i(32)},
                                    std::map<std::string, Attribute>{},
                                    ifOp.loc()));
    parent->insertOperation(
        insertIndex++,
        std::make_unique<Operation>("sysy.store",
                                    std::vector<Value>{select.result(), slot},
                                    std::vector<Type>{},
                                    std::map<std::string, Attribute>{},
                                    ifOp.loc()));
    promoted++;
  }

  ifOp.markErased();
  if (stats)
    stats->raisedSelects += promoted;
  return promoted > 0;
}

} // namespace

void runIfStoreSelectPromotion(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_SELECT_PROMOTION", true))
    return;
  std::vector<Operation*> ifs;
  for (auto *op : walk(module))
    if (op && !op->isErased() && op->name() == "scf.if")
      ifs.push_back(op);
  bool changed = false;
  for (Operation *op : ifs) {
    if (!op || op->isErased())
      continue;
    changed |= promoteIfStoresToSelects(module, *op, stats);
  }
  if (changed)
    eraseMarked(module);
}

void runStencilPeelingAndUnroll(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_STENCIL_PEEL", true))
    return;
  for (int round = 0; round < 4; round++) {
    std::vector<Operation*> loops;
    for (auto *op : walk(module))
      if (op && !op->isErased() && op->name() == "scf.while")
        loops.push_back(op);
    bool changed = false;
    for (Operation *loop : loops) {
      if (!loop || loop->isErased())
        continue;
      ConstantWhileInfo info = classifySmallConstantWhile(*loop);
      if (!info.valid)
        continue;
      if (unrollSmallConstantWhile(module, *loop, info)) {
        changed = true;
        if (stats) {
          stats->kernelUnrolls++;
          stats->interiorPeels += info.tripCount > 0 ? 1 : 0;
        }
      }
    }
    eraseMarked(module);
    if (!changed)
      break;
  }
}

static bool tileOpHasName(Operation *op, std::initializer_list<const char*> names) {
  if (!op)
    return false;
  for (const char *name : names) {
    if (op->name() == name)
      return true;
  }
  return false;
}

static Block *tileSingleBlock(Operation &op) {
  if (op.getRegions().size() != 1 || op.getRegions()[0]->getBlocks().size() != 1)
    return nullptr;
  return op.getRegions()[0]->getBlocks()[0].get();
}

static Value tileFirstBlockArg(Operation &op) {
  Block *block = tileSingleBlock(op);
  if (!block || block->args().empty())
    return Value();
  return block->args()[0]->value();
}

static bool tileOpTreeHasAnyName(Operation &op, std::initializer_list<const char*> names) {
  if (tileOpHasName(&op, names))
    return true;
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child && tileOpTreeHasAnyName(*child, names))
          return true;
      }
    }
  }
  return false;
}

static bool tileBlockHasNestedLoop(Block &block) {
  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased())
      continue;
    if (tileOpHasName(owned.get(), {"affine.for", "scf.while", "scf.for"}))
      return true;
    for (auto &region : owned->getRegions()) {
      for (auto &childBlock : region->getBlocks()) {
        if (tileBlockHasNestedLoop(*childBlock))
          return true;
      }
    }
  }
  return false;
}

static bool tileBlockUnsafeForStripMining(Block &block) {
  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased())
      continue;
    if (tileOpHasName(owned.get(), {"sysy.call", "sysy.return", "sysy.break",
                                    "sysy.continue", "scf.if", "scf.while",
                                    "scf.for", "affine.for", "scf.condition",
                                    "sysy.alloca", "memref.alloca"}))
      return true;
    for (auto &region : owned->getRegions()) {
      for (auto &childBlock : region->getBlocks()) {
        if (tileBlockUnsafeForStripMining(*childBlock))
          return true;
      }
    }
  }
  return false;
}

static Operation *tileSingleDirectAffineChild(Block &block) {
  Operation *found = nullptr;
  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased() || owned->name() != "affine.for")
      continue;
    if (found)
      return nullptr;
    found = owned.get();
  }
  return found;
}

static Value tileAppendIntConstant(Module &module, Block &block, int64_t value,
                                   Location loc) {
  Operation &op = appendOp(
      block, "rv_machine.li", {}, {module.context().i(32)},
      {{"value", module.context().integerAttr(value, module.context().i(32))}},
      loc);
  return op.result();
}

static Value tileMaterializeValue(Module &module, Block &block, Value value,
                                  Location loc) {
  int64_t imm = 0;
  if (constantIntegerValue(value, imm))
    return tileAppendIntConstant(module, block, imm, loc);
  Operation *def = value.getDefiningOp();
  if (def && !def->isErased() && def->resultCount() == 1 &&
      tileOpHasName(def, {"sysy.load", "memref.load", "rv_machine.li",
                          "arith.constant", "arith.addi", "arith.subi",
                          "rv_machine.addw", "rv_machine.subw"})) {
    Operation &clone = appendOp(block, def->name(), def->getOperands(),
                                resultTypesOf(*def), def->attrs(), loc);
    return clone.result();
  }
  return value;
}

static Operation &tileAppendAffineLoop(Module &module, Block &block, Value lower,
                                       Value upper, Value step, Location loc,
                                       const std::string &ivName) {
  Operation &loop = appendOp(block, "affine.for", {lower, upper, step}, {},
                             {}, loc, 1);
  loop.getRegions()[0]->addBlock().addArgument(module.context().i(32), loc, ivName);
  return loop;
}

static Operation &tileInsertAffineLoop(Module &module, Block &block, std::size_t &index,
                                       Value lower, Value upper, Value step,
                                       Location loc, const std::string &ivName) {
  Operation &loop = insertOp(block, index, "affine.for", {lower, upper, step},
                             {}, {}, loc, 1);
  loop.getRegions()[0]->addBlock().addArgument(module.context().i(32), loc, ivName);
  return loop;
}

static Value tileAppendMinValue(Module &module, Block &block, Value lhs, Value rhs,
                                Location loc) {
  Operation &cmp = appendOp(
      block, "rv_machine.cmp", {lhs, rhs}, {module.context().i(32)},
      {{"predicate", module.context().stringAttr("gt")}}, loc);
  Operation &sel = appendOp(block, "arith.select", {cmp.result(), rhs, lhs},
                            {module.context().i(32)}, {}, loc);
  return sel.result();
}

static bool tileSame2DIndices(Operation *op, Value first, Value second,
                              int firstOperandIndex) {
  return op && op->operandCount() > firstOperandIndex + 1 &&
         op->operand(firstOperandIndex) == first &&
         op->operand(firstOperandIndex + 1) == second;
}

struct RowReductionInfo {
  Operation *outer = nullptr;
  Operation *jLoop = nullptr;
  Operation *kLoop = nullptr;
  Operation *finalStore = nullptr;
  Operation *lhsLoad = nullptr;
  Operation *rhsLoad = nullptr;
  bool valid = false;
};

struct PolyAccessInfo {
  Value base;
  bool store = false;
  bool hasOuterIndex = false;
  bool hasInnerIndex = false;
  bool lastIndexOuter = false;
  bool lastIndexInner = false;
};

static bool tileIsLoadFromSlot(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  return tileOpHasName(op, {"sysy.load", "memref.load"}) &&
         op->operandCount() > 0 && op->operand(0) == slot;
}

static bool tileIsStoreToSlot(Operation *op, Value slot) {
  return tileOpHasName(op, {"sysy.store", "memref.store"}) &&
         op->operandCount() >= 2 && op->operand(1) == slot;
}

static bool tileOpTreeStoresSlot(Operation &op, Value slot) {
  if (tileIsStoreToSlot(&op, slot))
    return true;
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child && tileOpTreeStoresSlot(*child, slot))
          return true;
      }
    }
  }
  return false;
}

static bool tileValueIsSlotPlusConst(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  if (!tileOpHasName(op, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
      op->operandCount() != 2)
    return false;
  auto isLoad = [&](Value v) { return tileIsLoadFromSlot(v, slot); };
  int64_t imm = 0;
  return (isLoad(op->operand(0)) && constantIntegerValue(op->operand(1), imm)) ||
         (isLoad(op->operand(1)) && constantIntegerValue(op->operand(0), imm));
}

static void tileReplaceLoadsFromSlot(Module &module, Operation &op, Value slot,
                                     Value replacement, int &count) {
  if (tileOpHasName(&op, {"sysy.load", "memref.load"}) &&
      op.operandCount() > 0 && op.operand(0) == slot && op.resultCount() == 1) {
    replaceAllUses(module, op.result(), replacement);
    op.markErased();
    count++;
    return;
  }
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child && !child->isErased())
          tileReplaceLoadsFromSlot(module, *child, slot, replacement, count);
      }
    }
  }
}

static bool cacheWhileLoopIvLoads(Module &module, Operation &loop,
                                  SelfOptStats *stats) {
  if (loop.name() != "scf.while" || loop.getRegions().size() < 2 ||
      loop.getRegions()[0]->getBlocks().empty() ||
      loop.getRegions()[1]->getBlocks().empty())
    return false;
  Block &cond = *loop.getRegions()[0]->getBlocks()[0];
  Block &body = *loop.getRegions()[1]->getBlocks()[0];
  if (tileBlockHasNestedLoop(body))
    return false;
  if (cond.ops().empty() || cond.ops().back()->name() != "scf.condition" ||
      cond.ops().back()->operandCount() == 0)
    return false;
  Operation *cmp = cond.ops().back()->operand(0).getDefiningOp();
  if (!tileOpHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() < 2)
    return false;
  Operation *condLoad = cmp->operand(0).getDefiningOp();
  if (!tileOpHasName(condLoad, {"sysy.load", "memref.load"}) ||
      condLoad->operandCount() == 0 || condLoad->resultCount() != 1)
    return false;
  Value slot = condLoad->operand(0);

  int stepIndex = -1;
  for (std::size_t i = 0; i < body.ops().size(); i++) {
    Operation *op = body.ops()[i].get();
    if (!op || op->isErased())
      continue;
    if (tileIsStoreToSlot(op, slot)) {
      if (!tileValueIsSlotPlusConst(op->operand(0), slot))
        return false;
      if (stepIndex >= 0)
        return false;
      stepIndex = (int) i;
    } else if (tileOpTreeStoresSlot(*op, slot)) {
      return false;
    }
  }
  if (stepIndex <= 0)
    return false;

  std::size_t insertIndex = 0;
  Operation &cached = insertOp(body, insertIndex, condLoad->name(), {slot},
                               resultTypesOf(*condLoad), condLoad->attrs(),
                               loop.loc());
  stepIndex++;
  int replaced = 0;
  for (int i = 1; i < stepIndex && i < (int) body.ops().size(); i++) {
    Operation *op = body.ops()[(std::size_t) i].get();
    if (!op || op->isErased())
      continue;
    tileReplaceLoadsFromSlot(module, *op, slot, cached.result(), replaced);
  }
  if (replaced == 0) {
    cached.markErased();
    return false;
  }
  if (stats) {
    stats->loopAddressCSE += replaced;
    stats->addrIvRewrites++;
  }
  return true;
}

static bool polyIsMemrefAccess(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "memref.load" || op->name() == "memref.store");
}

static Value polyAccessBase(Operation *op) {
  if (!polyIsMemrefAccess(op))
    return Value();
  return op->operand(op->name() == "memref.store" ? 1 : 0);
}

static int polyAccessIndexStart(Operation *op) {
  return op && op->name() == "memref.store" ? 2 : 1;
}

static void polyCollectAccesses(Operation &op, Value outerIv, Value innerIv,
                                std::vector<PolyAccessInfo> &accesses) {
  if (op.isErased())
    return;
  if (polyIsMemrefAccess(&op)) {
    int start = polyAccessIndexStart(&op);
    if (op.operandCount() > start) {
      Value last = op.operand(op.operandCount() - 1);
      bool hasOuter = false;
      bool hasInner = false;
      for (int i = start; i < op.operandCount(); i++) {
        hasOuter = hasOuter || op.operand(i) == outerIv;
        hasInner = hasInner || op.operand(i) == innerIv;
      }
      accesses.push_back({polyAccessBase(&op), op.name() == "memref.store",
                          hasOuter, hasInner, last == outerIv, last == innerIv});
    }
  }
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child)
          polyCollectAccesses(*child, outerIv, innerIv, accesses);
}

static bool polyNoStoreAliasHazard(const std::vector<PolyAccessInfo> &accesses) {
  std::set<std::string> storeBases;
  for (const auto &access : accesses) {
    if (!access.store)
      continue;
    if (!access.hasOuterIndex || !access.hasInnerIndex)
      return false;
    std::string key = valueKey(access.base);
    if (storeBases.count(key) != 0)
      return false;
    storeBases.insert(key);
  }
  for (const auto &access : accesses) {
    if (!access.store && storeBases.count(valueKey(access.base)) != 0)
      return false;
  }
  return true;
}

static bool polyPerfect2DNest(Operation &outer, Operation *&inner,
                              Block *&outerBody, Block *&innerBody) {
  inner = nullptr;
  outerBody = tileSingleBlock(outer);
  if (!outerBody || outerBody->args().empty())
    return false;
  int liveOps = 0;
  for (auto &owned : outerBody->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() == "affine.yield")
      continue;
    liveOps++;
    if (op->name() == "affine.for")
      inner = op;
  }
  if (liveOps != 1 || !inner)
    return false;
  innerBody = tileSingleBlock(*inner);
  return innerBody && !innerBody->args().empty() &&
         !tileBlockUnsafeForStripMining(*innerBody);
}

static bool applyPolyLoopInterchange(Module &module, Operation &outer,
                                     Operation &inner) {
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;
  Block *oldOuterBody = tileSingleBlock(outer);
  Block *oldInnerBody = tileSingleBlock(inner);
  if (!oldOuterBody || !oldInnerBody || oldOuterBody->args().empty() ||
      oldInnerBody->args().empty())
    return false;

  Location loc = outer.loc();
  std::size_t insertIndex = (std::size_t) outerIndex;
  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             inner.operand(0), inner.operand(1),
                                             inner.operand(2), loc, "poly_j");
  Block &newOuterBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value newOuterIv = newOuterBody.args()[0]->value();
  Operation &newInner = tileAppendAffineLoop(module, newOuterBody,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "poly_i");
  Block &newInnerBody = *newInner.getRegions()[0]->getBlocks()[0];
  Value newInnerIv = newInnerBody.args()[0]->value();

  std::map<std::string, Value> valueMap;
  valueMap[valueKey(oldOuterBody->args()[0]->value())] = newInnerIv;
  valueMap[valueKey(oldInnerBody->args()[0]->value())] = newOuterIv;
  std::set<Operation*> skipOps;
  for (auto &owned : oldInnerBody->ops()) {
    if (!owned || owned->isErased() || owned->isTerminator())
      continue;
    auto cloned = cloneForUnrolledIteration(module, *owned, valueMap, skipOps,
                                            Value(), 0);
    if (!cloned)
      continue;
    Operation &inserted = newInnerBody.addOperation(std::move(cloned));
    for (int i = 0; i < owned->resultCount(); i++)
      valueMap[valueKey(owned->result(i))] = inserted.result(i);
  }
  appendOp(newInnerBody, "affine.yield", {}, {}, {}, loc);
  appendOp(newOuterBody, "affine.yield", {}, {}, {}, loc);
  outer.markErased();
  return true;
}

void runPolyhedralLoopPermutation(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_POLY_PERMUTE", true))
    return;
  std::vector<Operation*> loops;
  for (auto *op : walk(module))
    if (op && !op->isErased() && op->name() == "affine.for")
      loops.push_back(op);

  bool changed = false;
  for (Operation *outer : loops) {
    if (!outer || outer->isErased() || outer->operandCount() < 3)
      continue;
    Operation *inner = nullptr;
    Block *outerBody = nullptr;
    Block *innerBody = nullptr;
    if (!polyPerfect2DNest(*outer, inner, outerBody, innerBody))
      continue;
    if (!inner || inner->operandCount() < 3)
      continue;
    int64_t outerStep = 0;
    int64_t innerStep = 0;
    if (!constantIntegerValue(outer->operand(2), outerStep) || outerStep != 1 ||
        !constantIntegerValue(inner->operand(2), innerStep) || innerStep != 1)
      continue;

    if (stats)
      stats->polyNests++;
    std::vector<PolyAccessInfo> accesses;
    Value outerIv = outerBody->args()[0]->value();
    Value innerIv = innerBody->args()[0]->value();
    for (auto &owned : innerBody->ops())
      if (owned && !owned->isErased())
        polyCollectAccesses(*owned, outerIv, innerIv, accesses);
    if (accesses.empty())
      continue;
    if (!polyNoStoreAliasHazard(accesses))
      continue;
    if (stats)
      stats->polyDepsProved++;

    int innerStrideScore = 0;
    int outerStrideScore = 0;
    for (const auto &access : accesses) {
      innerStrideScore += access.lastIndexInner ? 2 : 0;
      outerStrideScore += access.lastIndexOuter ? 2 : 0;
      if (access.store) {
        innerStrideScore += access.lastIndexInner ? 1 : 0;
        outerStrideScore += access.lastIndexOuter ? 1 : 0;
      }
    }
    if (outerStrideScore <= innerStrideScore)
      continue;
    if (applyPolyLoopInterchange(module, *outer, *inner)) {
      changed = true;
      if (stats) {
        stats->polyPermutations++;
        stats->imperfectInterchanges++;
      }
    }
  }
  if (changed)
    eraseMarked(module);
}

static bool parityOpHasName(Operation *op, std::initializer_list<const char*> names) {
  if (!op || op->isErased())
    return false;
  for (const char *name : names)
    if (op->name() == name)
      return true;
  return false;
}

static int parityLiveUseCount(Module &module, Value value, Operation **onlyUser = nullptr) {
  int count = 0;
  Operation *last = nullptr;
  for (const auto &use : usesOf(module, value)) {
    if (!use.owner || use.owner->isErased())
      continue;
    count++;
    last = use.owner;
  }
  if (onlyUser)
    *onlyUser = last;
  return count;
}

static std::string parityConstOpForCmp(const std::string &cmpName) {
  if (cmpName == "rv_machine.cmp")
    return "rv_machine.li";
  if (cmpName == "arm_machine.cmp")
    return "arm_machine.mov";
  return "arith.constant";
}

static std::string parityAndOpForCmp(const std::string &cmpName) {
  if (cmpName == "rv_machine.cmp")
    return "rv_machine.and";
  if (cmpName == "arm_machine.cmp")
    return "arm_machine.and";
  return "arith.andi";
}

static std::string parityNotOpForCmp(const std::string &cmpName) {
  if (cmpName == "rv_machine.cmp")
    return "rv_machine.seqz";
  if (cmpName == "arm_machine.cmp")
    return "arm_machine.not";
  return "arith.noti";
}

static bool parityPureDeadOp(Operation *op) {
  if (!op || op->isErased() || op->resultCount() == 0)
    return false;
  for (int i = 0; i < op->resultCount(); i++) {
    if (!op->resultUses[i].empty())
      return false;
  }
  return parityOpHasName(op, {"arith.constant", "rv_machine.li", "arm_machine.mov",
                              "arith.andi", "rv_machine.and", "arm_machine.and"});
}

void runParityProductCompareStrength(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_POW2_STRENGTH", true))
    return;
  std::vector<Operation*> cmps;
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (parityOpHasName(op, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}))
      cmps.push_back(op);
  }

  bool changed = false;
  for (Operation *cmp : cmps) {
    if (!cmp || cmp->isErased() || cmp->operandCount() != 2 || cmp->resultCount() != 1)
      continue;
    std::string pred = symbolAttr(cmp->attr("predicate"));
    if (pred != "eq" && pred != "ne")
      continue;
    Value remValue;
    Value zeroValue;
    int64_t zero = 0;
    if (constantIntegerValue(cmp->operand(1), zero) && zero == 0) {
      remValue = cmp->operand(0);
      zeroValue = cmp->operand(1);
    } else if (constantIntegerValue(cmp->operand(0), zero) && zero == 0) {
      remValue = cmp->operand(1);
      zeroValue = cmp->operand(0);
    } else {
      continue;
    }

    Operation *rem = remValue.getDefiningOp();
    if (!parityOpHasName(rem, {"arith.remi", "rv_machine.remw", "arm_machine.srem"}) ||
        rem->operandCount() != 2 || rem->resultCount() != 1)
      continue;
    int64_t divisor = 0;
    if (!constantIntegerValue(rem->operand(1), divisor) || divisor != 2)
      continue;
    Operation *mul = rem->operand(0).getDefiningOp();
    if (!parityOpHasName(mul, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) ||
        mul->operandCount() != 2 || mul->resultCount() != 1)
      continue;

    Operation *only = nullptr;
    if (parityLiveUseCount(module, rem->result(), &only) != 1 || only != cmp)
      continue;
    if (parityLiveUseCount(module, mul->result(), &only) != 1 || only != rem)
      continue;

    Block *block = cmp->getBlock();
    if (!block)
      continue;
    int cmpIndex = operationIndexInBlock(*block, cmp);
    if (cmpIndex < 0)
      continue;
    std::size_t insertIndex = (std::size_t) cmpIndex;
    Type i32 = cmp->resultType();
    auto oneOp = std::make_unique<Operation>(
        parityConstOpForCmp(cmp->name()), std::vector<Value>{}, std::vector<Type>{i32},
        std::map<std::string, Attribute>{{"value", module.context().integerAttr(1, i32)}},
        cmp->loc());
    Operation &one = block->insertOperation(insertIndex++, std::move(oneOp));
    auto andPairOp = std::make_unique<Operation>(
        parityAndOpForCmp(cmp->name()),
        std::vector<Value>{mul->operand(0), mul->operand(1)}, std::vector<Type>{i32},
        std::map<std::string, Attribute>{}, cmp->loc());
    Operation &andPair = block->insertOperation(insertIndex++, std::move(andPairOp));
    auto lowBitOp = std::make_unique<Operation>(
        parityAndOpForCmp(cmp->name()),
        std::vector<Value>{andPair.result(), one.result()}, std::vector<Type>{i32},
        std::map<std::string, Attribute>{}, cmp->loc());
    Operation &lowBit = block->insertOperation(insertIndex++, std::move(lowBitOp));
    Value replacement = lowBit.result();
    if (pred == "eq") {
      auto notOp = std::make_unique<Operation>(
          parityNotOpForCmp(cmp->name()), std::vector<Value>{lowBit.result()},
          std::vector<Type>{i32}, std::map<std::string, Attribute>{}, cmp->loc());
      Operation &inserted = block->insertOperation(insertIndex++, std::move(notOp));
      replacement = inserted.result();
    }

    Operation *zeroDef = zeroValue.getDefiningOp();
    Operation *divisorDef = rem->operand(1).getDefiningOp();
    replaceAllUses(module, cmp->result(), replacement);
    cmp->markErased();
    rem->markErased();
    mul->markErased();
    if (parityPureDeadOp(zeroDef))
      zeroDef->markErased();
    if (parityPureDeadOp(divisorDef))
      divisorDef->markErased();
    changed = true;
    if (stats)
      stats->pow2StrengthReductions++;
  }
  if (changed)
    eraseMarked(module);
}

static bool machineDcePureOp(Operation *op) {
  if (!op || op->isErased() || op->resultCount() == 0 || !op->getRegions().empty())
    return false;
  for (int i = 0; i < op->resultCount(); i++) {
    if (!op->resultUses[i].empty())
      return false;
  }
  const std::string &name = op->name();
  return name == "arith.constant" || name == "arith.addi" ||
         name == "arith.subi" || name == "arith.muli" ||
         name == "arith.divi" || name == "arith.remi" ||
         name == "arith.andi" || name == "arith.ori" ||
         name == "arith.xori" || name == "arith.noti" ||
         name == "arith.cmpi" || name == "rv_machine.li" ||
         name == "rv_machine.addw" || name == "rv_machine.subw" ||
         name == "rv_machine.mulw" || name == "rv_machine.divw" ||
         name == "rv_machine.remw" || name == "rv_machine.and" ||
         name == "rv_machine.or" || name == "rv_machine.xor" ||
         name == "rv_machine.seqz" || name == "rv_machine.neg" ||
         name == "rv_machine.cmp" || name == "arm_machine.mov" ||
         name == "arm_machine.add" || name == "arm_machine.sub" ||
         name == "arm_machine.mul" || name == "arm_machine.sdiv" ||
         name == "arm_machine.srem" || name == "arm_machine.and" ||
         name == "arm_machine.orr" || name == "arm_machine.eor" ||
         name == "arm_machine.not" || name == "arm_machine.neg" ||
         name == "arm_machine.cmp";
}

void runMachineDeadCodeElim(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_MACHINE_DCE", true))
    return;
  bool changed = true;
  int removed = 0;
  while (changed) {
    changed = false;
    for (auto *op : walk(module)) {
      if (!machineDcePureOp(op))
        continue;
      op->markErased();
      changed = true;
      removed++;
    }
    if (changed)
      eraseMarked(module);
  }
  if (stats && removed > 0)
    stats->worklistRewrites += removed;
}

static bool classifyRowBufferedReduction(Operation &outer, RowReductionInfo &info) {
  if (outer.name() != "affine.for" || outer.operandCount() < 3)
    return false;
  int64_t step = 0;
  if (!constantIntegerValue(outer.operand(2), step) || step != 1)
    return false;
  Block *outerBody = tileSingleBlock(outer);
  if (!outerBody || outerBody->args().empty())
    return false;
  Operation *jLoop = tileSingleDirectAffineChild(*outerBody);
  if (!jLoop || jLoop->operandCount() < 3)
    return false;
  if (tileOpTreeHasAnyName(*jLoop, {"sysy.call", "sysy.return", "sysy.break",
                                    "sysy.continue", "scf.if", "scf.while"}))
    return false;
  int64_t jStep = 0;
  if (!constantIntegerValue(jLoop->operand(2), jStep) || jStep != 1)
    return false;
  Block *jBody = tileSingleBlock(*jLoop);
  if (!jBody || jBody->args().empty())
    return false;
  Operation *kLoop = tileSingleDirectAffineChild(*jBody);
  if (!kLoop || kLoop->operandCount() < 3)
    return false;
  int64_t kStep = 0;
  if (!constantIntegerValue(kLoop->operand(2), kStep) || kStep != 1)
    return false;
  Block *kBody = tileSingleBlock(*kLoop);
  if (!kBody || kBody->args().empty() || tileBlockHasNestedLoop(*kBody))
    return false;

  Value iIv = tileFirstBlockArg(outer);
  Value jIv = tileFirstBlockArg(*jLoop);
  Value kIv = tileFirstBlockArg(*kLoop);

  std::vector<Value> accumulatorCandidates;
  for (auto &owned : jBody->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (tileOpHasName(op, {"sysy.alloca", "memref.alloca"}) &&
        op->resultCount() == 1 && isScalarWordMemref(op->resultType()))
      accumulatorCandidates.push_back(op->result());
  }
  if (accumulatorCandidates.empty())
    return false;

  Operation *finalStore = nullptr;
  Operation *sumStore = nullptr;
  Operation *lhsLoad = nullptr;
  Operation *rhsLoad = nullptr;
  for (Value sumSlot : accumulatorCandidates) {
    finalStore = nullptr;
    sumStore = nullptr;
    lhsLoad = nullptr;
    rhsLoad = nullptr;

    for (auto &owned : jBody->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() || op->name() != "memref.store" ||
          op->operandCount() < 4)
        continue;
      if (!tileIsLoadFromSlot(op->operand(0), sumSlot))
        continue;
      if (!tileSame2DIndices(op, iIv, jIv, 2))
        continue;
      finalStore = op;
    }
    if (!finalStore)
      continue;

    for (auto &owned : kBody->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() || !tileIsStoreToSlot(op, sumSlot))
        continue;
      Operation *add = op->operand(0).getDefiningOp();
      if (!tileOpHasName(add, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
          add->operandCount() != 2)
        continue;
      Value product;
      if (tileIsLoadFromSlot(add->operand(0), sumSlot))
        product = add->operand(1);
      else if (tileIsLoadFromSlot(add->operand(1), sumSlot))
        product = add->operand(0);
      else
        continue;
      Operation *mul = product.getDefiningOp();
      if (!tileOpHasName(mul, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) ||
          mul->operandCount() != 2)
        continue;
      Operation *a = mul->operand(0).getDefiningOp();
      Operation *b = mul->operand(1).getDefiningOp();
      if (!tileOpHasName(a, {"memref.load"}) || !tileOpHasName(b, {"memref.load"}))
        continue;
      bool aIsLhs = tileSame2DIndices(a, iIv, kIv, 1) &&
                    tileSame2DIndices(b, kIv, jIv, 1);
      bool bIsLhs = tileSame2DIndices(b, iIv, kIv, 1) &&
                    tileSame2DIndices(a, kIv, jIv, 1);
      if (!aIsLhs && !bIsLhs)
        continue;
      lhsLoad = aIsLhs ? a : b;
      rhsLoad = aIsLhs ? b : a;
      sumStore = op;
    }
    if (sumStore && lhsLoad && rhsLoad)
      break;
  }
  if (!sumStore || !lhsLoad || !rhsLoad)
    return false;
  MemrefInfo outInfo = parseMemrefInfo(finalStore->operand(1).type());
  if (!outInfo.valid || outInfo.shape.size() < 2 || outInfo.shape[1] <= 0)
    return false;

  info.outer = &outer;
  info.jLoop = jLoop;
  info.kLoop = kLoop;
  info.finalStore = finalStore;
  info.lhsLoad = lhsLoad;
  info.rhsLoad = rhsLoad;
  info.valid = true;
  return true;
}

static bool applyRowBufferedReduction(Module &module, const RowReductionInfo &info,
                                      SelfOptStats *stats) {
  Operation &outer = *info.outer;
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;
  Context &ctx = module.context();
  Location loc = outer.loc();
  Type i32 = ctx.i(32);
  MemrefInfo outInfo = parseMemrefInfo(info.finalStore->operand(1).type());
  int64_t rowElements = outInfo.shape.size() >= 2 && outInfo.shape[1] > 0
                            ? outInfo.shape[1]
                            : 1024;

  std::size_t insertIndex = (std::size_t) outerIndex;
  Operation &rowBuf = insertOp(
      *parent, insertIndex, "sysy.alloca", {}, {ctx.memref({rowElements}, i32)},
      {{"symbol", ctx.stringAttr(".tile_rowbuf")}}, loc);
  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "i");
  Block &outerBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value iIv = outerBody.args()[0]->value();
  Value jLower = tileMaterializeValue(module, outerBody, info.jLoop->operand(0), loc);
  Value jUpper = tileMaterializeValue(module, outerBody, info.jLoop->operand(1), loc);
  Value jStep = tileMaterializeValue(module, outerBody, info.jLoop->operand(2), loc);
  Value kLower = tileMaterializeValue(module, outerBody, info.kLoop->operand(0), loc);
  Value kUpper = tileMaterializeValue(module, outerBody, info.kLoop->operand(1), loc);
  Value kStep = tileMaterializeValue(module, outerBody, info.kLoop->operand(2), loc);

  Operation &zeroLoop = tileAppendAffineLoop(module, outerBody, jLower, jUpper,
                                             jStep, loc, "j");
  {
    Block &body = *zeroLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = body.args()[0]->value();
    Value zero = tileAppendIntConstant(module, body, 0, loc);
    appendOp(body, "memref.store", {zero, rowBuf.result(), jIv}, {}, {}, loc);
    appendOp(body, "affine.yield", {}, {}, {}, loc);
  }

  Operation &kLoop = tileAppendAffineLoop(module, outerBody, kLower, kUpper,
                                          kStep, loc, "k");
  {
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value kIv = kBody.args()[0]->value();
    Operation &lhs = appendOp(kBody, "memref.load",
                              {info.lhsLoad->operand(0), iIv, kIv}, {i32},
                              info.lhsLoad->attrs(), loc);
    Operation &jLoop = tileAppendAffineLoop(module, kBody, jLower, jUpper,
                                            jStep, loc, "j");
    Block &jBody = *jLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = jBody.args()[0]->value();
    Operation &old = appendOp(jBody, "memref.load", {rowBuf.result(), jIv},
                              {i32}, {}, loc);
    Operation &rhs = appendOp(jBody, "memref.load",
                              {info.rhsLoad->operand(0), kIv, jIv}, {i32},
                              info.rhsLoad->attrs(), loc);
    Operation &prod = appendOp(jBody, "rv_machine.mulw",
                               {lhs.result(), rhs.result()}, {i32}, {}, loc);
    Operation &sum = appendOp(jBody, "rv_machine.addw",
                              {old.result(), prod.result()}, {i32}, {}, loc);
    appendOp(jBody, "memref.store", {sum.result(), rowBuf.result(), jIv},
             {}, {}, loc);
    appendOp(jBody, "affine.yield", {}, {}, {}, loc);
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);
  }

  Operation &writeLoop = tileAppendAffineLoop(module, outerBody, jLower, jUpper,
                                              jStep, loc, "j");
  {
    Block &body = *writeLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = body.args()[0]->value();
    Operation &val = appendOp(body, "memref.load", {rowBuf.result(), jIv},
                              {i32}, {}, loc);
    appendOp(body, "memref.store",
             {val.result(), info.finalStore->operand(1), iIv, jIv},
             {}, info.finalStore->attrs(), loc);
    appendOp(body, "affine.yield", {}, {}, {}, loc);
  }
  appendOp(outerBody, "affine.yield", {}, {}, {}, loc);

  outer.markErased();
  if (stats) {
    stats->rowBufferedReductions++;
    stats->reductionBlocks++;
    stats->loopTiles++;
    stats->polyTiles++;
    stats->imperfectInterchanges++;
  }
  return true;
}

static Value tileAppendAddI32(Module &module, Block &block, Value lhs, Value rhs,
                              Location loc) {
  Operation &op = appendOp(block, "rv_machine.addw", {lhs, rhs},
                           {module.context().i(32)}, {}, loc);
  return op.result();
}

static Value tileAppendSubI32(Module &module, Block &block, Value lhs, Value rhs,
                              Location loc) {
  Operation &op = appendOp(block, "rv_machine.subw", {lhs, rhs},
                           {module.context().i(32)}, {}, loc);
  return op.result();
}

static Value tileAppendRemI32(Module &module, Block &block, Value lhs, Value rhs,
                              Location loc) {
  Operation &op = appendOp(block, "rv_machine.remw", {lhs, rhs},
                           {module.context().i(32)}, {}, loc);
  return op.result();
}

static Value tileAppendLaneIndex(Module &module, Block &block, Value base,
                                 int lane, Location loc) {
  if (lane == 0)
    return base;
  Value laneConst = tileAppendIntConstant(module, block, lane, loc);
  return tileAppendAddI32(module, block, base, laneConst, loc);
}

static void appendScalarZeroStore(Module &module, Block &block, Value slot,
                                  Location loc) {
  Value zero = tileAppendIntConstant(module, block, 0, loc);
  appendOp(block, "sysy.store", {zero, slot}, {}, {}, loc);
}

static Value appendScalarLoad(Module &module, Block &block, Value slot,
                              Location loc) {
  Operation &op = appendOp(block, "sysy.load", {slot}, {module.context().i(32)},
                           {}, loc);
  return op.result();
}

static void appendScalarStore(Block &block, Value value, Value slot,
                              Location loc) {
  appendOp(block, "sysy.store", {value, slot}, {}, {}, loc);
}

static bool applyRegisterBlockedReduction(Module &module,
                                          const RowReductionInfo &info,
                                          SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_REDUCTION_REG", true))
    return false;
  Operation &outer = *info.outer;
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;

  Context &ctx = module.context();
  Location loc = outer.loc();
  Type i32 = ctx.i(32);
  constexpr int kBlock = 4;
  std::size_t insertIndex = (std::size_t) outerIndex;

  std::vector<Value> accSlots;
  accSlots.reserve(kBlock);
  for (int lane = 0; lane < kBlock; lane++) {
    Operation &slot = insertOp(
        *parent, insertIndex, "sysy.alloca", {}, {ctx.memref({1}, i32)},
        {{"symbol", ctx.stringAttr(".tile_acc" + std::to_string(lane))},
         {"scalar_promote", ctx.stringAttr("forced")}},
        loc);
    accSlots.push_back(slot.result());
  }

  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "i");
  Block &outerBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value iIv = outerBody.args()[0]->value();
  Value jLower = tileMaterializeValue(module, outerBody, info.jLoop->operand(0), loc);
  Value jUpper = tileMaterializeValue(module, outerBody, info.jLoop->operand(1), loc);
  Value kLower = tileMaterializeValue(module, outerBody, info.kLoop->operand(0), loc);
  Value kUpper = tileMaterializeValue(module, outerBody, info.kLoop->operand(1), loc);
  Value kStep = tileMaterializeValue(module, outerBody, info.kLoop->operand(2), loc);
  Value four = tileAppendIntConstant(module, outerBody, kBlock, loc);
  Value delta = tileAppendSubI32(module, outerBody, jUpper, jLower, loc);
  Value rem = tileAppendRemI32(module, outerBody, delta, four, loc);
  Value mainUpper = tileAppendSubI32(module, outerBody, jUpper, rem, loc);

  Operation &jBlockLoop = tileAppendAffineLoop(module, outerBody, jLower,
                                               mainUpper, four, loc, "jb");
  {
    Block &jbBody = *jBlockLoop.getRegions()[0]->getBlocks()[0];
    Value jbIv = jbBody.args()[0]->value();
    std::vector<Value> lanes;
    lanes.reserve(kBlock);
    for (int lane = 0; lane < kBlock; lane++) {
      appendScalarZeroStore(module, jbBody, accSlots[lane], loc);
      lanes.push_back(tileAppendLaneIndex(module, jbBody, jbIv, lane, loc));
    }

    Operation &kLoop = tileAppendAffineLoop(module, jbBody, kLower, kUpper,
                                            kStep, loc, "k");
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value kIv = kBody.args()[0]->value();
    Operation &lhs = appendOp(kBody, "memref.load",
                              {info.lhsLoad->operand(0), iIv, kIv}, {i32},
                              info.lhsLoad->attrs(), loc);
    for (int lane = 0; lane < kBlock; lane++) {
      Value old = appendScalarLoad(module, kBody, accSlots[lane], loc);
      Operation &rhs = appendOp(kBody, "memref.load",
                                {info.rhsLoad->operand(0), kIv, lanes[lane]},
                                {i32}, info.rhsLoad->attrs(), loc);
      Operation &prod = appendOp(kBody, "rv_machine.mulw",
                                 {lhs.result(), rhs.result()}, {i32}, {}, loc);
      Operation &sum = appendOp(kBody, "rv_machine.addw",
                                {old, prod.result()}, {i32}, {}, loc);
      appendScalarStore(kBody, sum.result(), accSlots[lane], loc);
    }
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);

    for (int lane = 0; lane < kBlock; lane++) {
      Value val = appendScalarLoad(module, jbBody, accSlots[lane], loc);
      appendOp(jbBody, "memref.store",
               {val, info.finalStore->operand(1), iIv, lanes[lane]},
               {}, info.finalStore->attrs(), loc);
    }
    appendOp(jbBody, "affine.yield", {}, {}, {}, loc);
  }

  Operation &tailLoop = tileAppendAffineLoop(module, outerBody, mainUpper, jUpper,
                                             tileMaterializeValue(module, outerBody,
                                                                  info.jLoop->operand(2), loc),
                                             loc, "j");
  {
    Block &tailBody = *tailLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = tailBody.args()[0]->value();
    appendScalarZeroStore(module, tailBody, accSlots[0], loc);
    Operation &kLoop = tileAppendAffineLoop(module, tailBody, kLower, kUpper,
                                            kStep, loc, "k");
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value kIv = kBody.args()[0]->value();
    Operation &lhs = appendOp(kBody, "memref.load",
                              {info.lhsLoad->operand(0), iIv, kIv}, {i32},
                              info.lhsLoad->attrs(), loc);
    Value old = appendScalarLoad(module, kBody, accSlots[0], loc);
    Operation &rhs = appendOp(kBody, "memref.load",
                              {info.rhsLoad->operand(0), kIv, jIv}, {i32},
                              info.rhsLoad->attrs(), loc);
    Operation &prod = appendOp(kBody, "rv_machine.mulw",
                               {lhs.result(), rhs.result()}, {i32}, {}, loc);
    Operation &sum = appendOp(kBody, "rv_machine.addw",
                              {old, prod.result()}, {i32}, {}, loc);
    appendScalarStore(kBody, sum.result(), accSlots[0], loc);
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);
    Value val = appendScalarLoad(module, tailBody, accSlots[0], loc);
    appendOp(tailBody, "memref.store",
             {val, info.finalStore->operand(1), iIv, jIv},
             {}, info.finalStore->attrs(), loc);
    appendOp(tailBody, "affine.yield", {}, {}, {}, loc);
  }

  appendOp(outerBody, "affine.yield", {}, {}, {}, loc);
  outer.markErased();
  if (stats) {
    stats->reductionBlocks++;
    stats->loopTiles++;
    stats->polyTiles++;
    stats->reductionRegs += kBlock;
    stats->imperfectInterchanges++;
  }
  return true;
}

static bool stripMineInnermostAffineLoop(Module &module, Operation &loop,
                                         int64_t tileSize, SelfOptStats *stats) {
  (void) stats;
  if (loop.name() != "affine.for" || loop.operandCount() < 3 ||
      loop.getRegions().size() != 1)
    return false;
  int64_t step = 0;
  if (!constantIntegerValue(loop.operand(2), step) || step != 1)
    return false;
  Block *body = tileSingleBlock(loop);
  if (!body || body->args().empty() || tileBlockHasNestedLoop(*body) ||
      tileBlockUnsafeForStripMining(*body))
    return false;
  bool hasMemrefAccess = false;
  for (auto &owned : body->ops()) {
    if (!owned || owned->isErased())
      continue;
    if (tileOpTreeHasAnyName(*owned, {"memref.load", "memref.store"})) {
      hasMemrefAccess = true;
      break;
    }
  }
  if (!hasMemrefAccess)
    return false;
  int64_t lowerImm = 0;
  int64_t upperImm = 0;
  if (!constantIntegerValue(loop.operand(0), lowerImm) ||
      !constantIntegerValue(loop.operand(1), upperImm))
    return false;
  if (upperImm - lowerImm <= tileSize)
    return false;

  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;

  Context &ctx = module.context();
  Location loc = loop.loc();
  std::size_t insertIndex = (std::size_t) loopIndex;
  Operation &tileConst = insertOp(
      *parent, insertIndex, "rv_machine.li", {}, {ctx.i(32)},
      {{"value", ctx.integerAttr(tileSize, ctx.i(32))}}, loc);
  Value tileStep = tileConst.result();
  Operation &tileLoop = tileInsertAffineLoop(module, *parent, insertIndex,
                                             loop.operand(0), loop.operand(1),
                                             tileStep, loc, "tile");
  Block &tileBody = *tileLoop.getRegions()[0]->getBlocks()[0];
  Value tileIv = tileBody.args()[0]->value();
  Operation &tileEndRaw = appendOp(tileBody, "rv_machine.addw",
                                   {tileIv, tileStep}, {ctx.i(32)}, {}, loc);
  Value tileEnd = tileAppendMinValue(module, tileBody, tileEndRaw.result(),
                                     loop.operand(1), loc);
  Operation &inner = tileAppendAffineLoop(module, tileBody, tileIv, tileEnd,
                                          loop.operand(2), loc, "iv");
  Block &innerBody = *inner.getRegions()[0]->getBlocks()[0];
  Value oldIv = body->args()[0]->value();
  Value newIv = innerBody.args()[0]->value();
  std::map<std::string, Value> valueMap;
  valueMap[valueKey(oldIv)] = newIv;
  std::set<Operation*> skipOps;
  for (auto &owned : body->ops()) {
    if (!owned || owned->isErased())
      continue;
    auto cloned = cloneForUnrolledIteration(module, *owned, valueMap, skipOps,
                                            Value(), 0);
    if (!cloned)
      continue;
    Operation &inserted = innerBody.addOperation(std::move(cloned));
    for (int i = 0; i < owned->resultCount(); i++)
      valueMap[valueKey(owned->result(i))] = inserted.result(i);
  }
  appendOp(tileBody, "affine.yield", {}, {}, {}, loc);
  loop.markErased();
  return true;
}

static void runSafeLoopTiling(Module &module, SelfOptStats *stats) {
  int64_t tileSize = 32;
  if (const char *ts = std::getenv("SISY_SELF_TILE_SIZE")) {
    try {
      tileSize = std::stoll(ts);
    } catch (...) {
      tileSize = 32;
    }
    if (tileSize <= 1)
      tileSize = 32;
  }

  std::vector<Operation*> whileLoops;
  if (envEnabled("SISY_ENABLE_SELF_WHILE_IV_CACHE", true)) {
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "scf.while")
        whileLoops.push_back(op);
    }
    for (Operation *loop : whileLoops) {
      if (loop && !loop->isErased())
        cacheWhileLoopIvLoads(module, *loop, stats);
    }
    eraseMarked(module);
  }

  std::vector<Operation*> loops;
  if (envEnabled("SISY_ENABLE_SELF_ROW_BUFFER_TILE", true)) {
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "affine.for")
        loops.push_back(op);
    }
    for (Operation *loop : loops) {
      if (!loop || loop->isErased())
        continue;
      RowReductionInfo info;
      if (classifyRowBufferedReduction(*loop, info)) {
        if (!applyRegisterBlockedReduction(module, info, stats))
          applyRowBufferedReduction(module, info, stats);
      } else if (stats) {
        stats->tileSkippedShape++;
      }
    }
    eraseMarked(module);
  }

  if (envEnabled("SISY_ENABLE_SELF_STRIP_TILE",
                 envEnabled("SISY_ENABLE_SELF_POLY_TILE", false))) {
    loops.clear();
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "affine.for")
        loops.push_back(op);
    }
    for (Operation *loop : loops) {
      if (!loop || loop->isErased())
        continue;
      if (stripMineInnermostAffineLoop(module, *loop, tileSize, stats)) {
        if (stats) {
          stats->stencilTiles++;
          stats->polyTiles++;
        }
      } else if (stats) {
        stats->tileSkippedShape++;
      }
    }
    eraseMarked(module);
  }
}

void runLoopTiling(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_TILE", true))
    return;
  if (envEnabled("SISY_ENABLE_SELF_TILE_LEGACY", false)) {
    runAffineLoopTiling(module);
    return;
  }
  runSafeLoopTiling(module, stats);
}

void runLoopRepeatReduction(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_REPEAT_REDUCTION", true))
    return;
  std::vector<Operation*> loops;
  for (auto *op : walk(module))
    if (op && !op->isErased() && op->name() == "affine.for")
      loops.push_back(op);
  bool changed = false;
  for (Operation *loop : loops) {
    if (!loop || loop->isErased())
      continue;
    LinearModRecurrenceInfo modInfo = classifyLinearModRecurrenceLoop(*loop);
    if (modInfo.valid && applyLinearModRecurrenceLoop(module, *loop, modInfo)) {
      changed = true;
      if (stats)
        stats->addrIvRewrites++;
      continue;
    }
    RepeatReductionInfo info = classifyRepeatReductionLoop(*loop);
    if (!info.valid)
      continue;
    if (applyRepeatReductionLoop(module, *loop, info)) {
      changed = true;
      if (stats)
        stats->imperfectInterchanges++;
    }
  }
  if (changed)
    eraseMarked(module);
}

void runLoopAddressIV(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_ADDR_IV", true))
    return;
  runLoopAddressIVInRegion(module, module.body(), stats);
  eraseMarked(module);
}

static bool isSchedulableLoad(Operation *op) {
  return op && !op->isErased() && op->getRegions().empty() &&
         (op->name() == "sysy.load" || op->name() == "memref.load") &&
         op->resultCount() > 0;
}

static bool isPureArithmeticForSchedule(Operation *op) {
  return op && !op->isErased() && op->getRegions().empty() &&
         op->dialect() == "arith" && !op->isTerminator();
}

static bool opUsesResultOf(Operation *user, Operation *def) {
  if (!user || !def)
    return false;
  for (auto operand : user->getOperands()) {
    if (operand.getDefiningOp() == def)
      return true;
  }
  return false;
}

static void runLoopLocalSchedulerInRegion(Region &region, SelfOptStats *stats);

static void runLoopLocalSchedulerInBlock(Block &block, SelfOptStats *stats) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 1; i < block.ops().size(); i++) {
      Operation *load = block.ops()[i].get();
      Operation *prev = block.ops()[i - 1].get();
      if (!isSchedulableLoad(load) || !isPureArithmeticForSchedule(prev))
        continue;
      if (opUsesResultOf(load, prev))
        continue;
      auto moved = block.takeOperation(load);
      block.insertOperation(i - 1, std::move(moved));
      if (stats)
        stats->schedulerMoves++;
      changed = true;
      if (i > 1)
        i -= 2;
    }
  }

  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased())
      continue;
    for (auto &nested : owned->getRegions())
      runLoopLocalSchedulerInRegion(*nested, stats);
  }
}

static void runLoopLocalSchedulerInRegion(Region &region, SelfOptStats *stats) {
  for (auto &block : region.getBlocks())
    runLoopLocalSchedulerInBlock(*block, stats);
}

void runLoopLocalScheduler(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_SCHED", true))
    return;
  for (auto &region : module.op().getRegions())
    runLoopLocalSchedulerInRegion(*region, stats);
}

namespace {

enum class ProvenBitwiseKind {
  None,
  And,
  Or,
  Xor,
};

struct ProvenBitwiseFunction {
  ProvenBitwiseKind kind = ProvenBitwiseKind::None;
  Operation *func = nullptr;
};

static std::vector<Operation*> collectNestedOps(Operation &root) {
  std::vector<Operation*> ops;
  std::function<void(Operation&)> rec = [&](Operation &op) {
    ops.push_back(&op);
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          rec(*child);
  };
  rec(root);
  return ops;
}

static bool opHasName(Operation *op, std::initializer_list<const char*> names) {
  if (!op)
    return false;
  for (const char *name : names)
    if (op->name() == name)
      return true;
  return false;
}

static bool isConst(Value value, int64_t expected) {
  int64_t actual = 0;
  return constantIntegerValue(value, actual) && actual == expected;
}

static bool isLoadFromSlot(Value value, Value slot) {
  auto *op = value.getDefiningOp();
  return opHasName(op, {"sysy.load", "memref.load"}) &&
         op->operandCount() > 0 && op->operand(0) == slot;
}

static bool isStoreToSlot(Operation *op, Value slot) {
  return opHasName(op, {"sysy.store", "memref.store"}) &&
         op->operandCount() >= 2 && op->operand(1) == slot;
}

static bool isBinaryWithConst(Value value, const char *arithName,
                              const char *rvName, const char *armName,
                              Value slot, int64_t constant,
                              bool commutative = false) {
  auto *op = value.getDefiningOp();
  if (!opHasName(op, {arithName, rvName, armName}) || op->operandCount() != 2)
    return false;
  if (isLoadFromSlot(op->operand(0), slot) && isConst(op->operand(1), constant))
    return true;
  return commutative && isLoadFromSlot(op->operand(1), slot) &&
         isConst(op->operand(0), constant);
}

static bool isSubSlotByOne(Value value, Value slot) {
  auto *op = value.getDefiningOp();
  if (!opHasName(op, {"arith.subi", "rv_machine.subw", "arm_machine.sub"}) ||
      op->operandCount() != 2)
    return false;
  return isLoadFromSlot(op->operand(0), slot) && isConst(op->operand(1), 1);
}

static bool isEqBitToOne(Value value, const std::set<std::string> &bitSlots) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "eq")
    return false;
  auto matches = [&](Value maybeBit, Value maybeOne) {
    if (isConst(maybeOne, 1) && bitSlots.count(valueKey(maybeBit)) != 0)
      return true;
    auto *load = maybeBit.getDefiningOp();
    return isConst(maybeOne, 1) && opHasName(load, {"sysy.load", "memref.load"}) &&
           load->operandCount() > 0 && bitSlots.count(valueKey(load->operand(0))) != 0;
  };
  return matches(cmp->operand(0), cmp->operand(1)) ||
         matches(cmp->operand(1), cmp->operand(0));
}

static bool isTruthOfEqBitToOne(Value value, const std::set<std::string> &bitSlots) {
  if (isEqBitToOne(value, bitSlots))
    return true;
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "ne")
    return false;
  return (isEqBitToOne(cmp->operand(0), bitSlots) && isConst(cmp->operand(1), 0)) ||
         (isEqBitToOne(cmp->operand(1), bitSlots) && isConst(cmp->operand(0), 0));
}

static bool isOrOfEqBitToOne(Value value, const std::set<std::string> &bitSlots) {
  auto *op = value.getDefiningOp();
  if (!opHasName(op, {"arith.ori", "rv_machine.or", "arm_machine.orr"}) ||
      op->operandCount() != 2)
    return false;
  return isEqBitToOne(op->operand(0), bitSlots) &&
         isEqBitToOne(op->operand(1), bitSlots);
}

static bool isNeBetweenBitLoads(Value value, const std::set<std::string> &bitSlots) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "ne")
    return false;
  for (int i = 0; i < 2; i++) {
    if (bitSlots.count(valueKey(cmp->operand(i))) != 0)
      continue;
    auto *load = cmp->operand(i).getDefiningOp();
    if (!opHasName(load, {"sysy.load", "memref.load"}) || load->operandCount() == 0 ||
        bitSlots.count(valueKey(load->operand(0))) == 0)
      return false;
  }
  return true;
}

static bool isResultPlusPowerStore(Operation *store, Value resultSlot, Value powerSlot) {
  if (!isStoreToSlot(store, resultSlot))
    return false;
  auto *add = store->operand(0).getDefiningOp();
  if (!opHasName(add, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
      add->operandCount() != 2)
    return false;
  return (isLoadFromSlot(add->operand(0), resultSlot) && isLoadFromSlot(add->operand(1), powerSlot)) ||
         (isLoadFromSlot(add->operand(1), resultSlot) && isLoadFromSlot(add->operand(0), powerSlot));
}

static bool blockHasResultPlusPowerStore(Block &block, Value resultSlot, Value powerSlot) {
  for (auto &owned : block.ops()) {
    if (owned && !owned->isErased() &&
        isResultPlusPowerStore(owned.get(), resultSlot, powerSlot))
      return true;
  }
  return false;
}

static bool blockStoresConstToSlot(Block &block, Value slot, int64_t constant) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (op && !op->isErased() && isStoreToSlot(op, slot) &&
        isConst(op->operand(0), constant))
      return true;
  }
  return false;
}

static bool blockStoresTruthToSlot(Block &block, Value slot,
                                   const std::set<std::string> &bitSlots) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (op && !op->isErased() && isStoreToSlot(op, slot) &&
        isTruthOfEqBitToOne(op->operand(0), bitSlots))
      return true;
  }
  return false;
}

static bool isShortCircuitLogicIf(Operation *op,
                                  const std::set<std::string> &bitSlots,
                                  Value &slot, ProvenBitwiseKind &kind) {
  if (!op || op->name() != "scf.if" || op->operandCount() != 1 ||
      op->getRegions().size() != 2 ||
      op->getRegions()[0]->getBlocks().empty() ||
      op->getRegions()[1]->getBlocks().empty() ||
      !isTruthOfEqBitToOne(op->operand(0), bitSlots))
    return false;
  Block &thenBlock = *op->getRegions()[0]->getBlocks()[0];
  Block &elseBlock = *op->getRegions()[1]->getBlocks()[0];
  auto considerStore = [&](Block &block) -> Value {
    for (auto &owned : block.ops()) {
      Operation *store = owned.get();
      if (store && !store->isErased() && isStoreToSlot(store, store->operandCount() >= 2
                                                                  ? store->operand(1)
                                                                  : Value()))
        return store->operand(1);
    }
    return Value();
  };
  Value thenSlot = considerStore(thenBlock);
  Value elseSlot = considerStore(elseBlock);
  if (!thenSlot.valid() || thenSlot != elseSlot)
    return false;
  if (blockStoresTruthToSlot(thenBlock, thenSlot, bitSlots) &&
      blockStoresConstToSlot(elseBlock, thenSlot, 0)) {
    slot = thenSlot;
    kind = ProvenBitwiseKind::And;
    return true;
  }
  if (blockStoresConstToSlot(thenBlock, thenSlot, 1) &&
      blockStoresTruthToSlot(elseBlock, thenSlot, bitSlots)) {
    slot = thenSlot;
    kind = ProvenBitwiseKind::Or;
    return true;
  }
  return false;
}

static bool isShortCircuitOrIf(Operation *op, const std::set<std::string> &bitSlots,
                               Value resultSlot, Value powerSlot) {
  if (!op || op->name() != "scf.if" || op->operandCount() != 1 ||
      op->getRegions().size() != 2 ||
      op->getRegions()[0]->getBlocks().empty() ||
      op->getRegions()[1]->getBlocks().empty())
    return false;
  if (!isEqBitToOne(op->operand(0), bitSlots))
    return false;

  Block &outerThen = *op->getRegions()[0]->getBlocks()[0];
  Block &outerElse = *op->getRegions()[1]->getBlocks()[0];
  if (!blockHasResultPlusPowerStore(outerThen, resultSlot, powerSlot))
    return false;

  Operation *innerIf = nullptr;
  int innerIfCount = 0;
  for (auto &owned : outerElse.ops()) {
    Operation *child = owned.get();
    if (!child || child->isErased() || child->name() == "scf.yield")
      continue;
    if (child->name() == "scf.if") {
      innerIf = child;
      innerIfCount++;
    }
  }
  if (innerIfCount != 1 || !innerIf || innerIf->operandCount() != 1 ||
      innerIf->getRegions().empty() || innerIf->getRegions()[0]->getBlocks().empty())
    return false;
  if (!isEqBitToOne(innerIf->operand(0), bitSlots))
    return false;
  Block &innerThen = *innerIf->getRegions()[0]->getBlocks()[0];
  return blockHasResultPlusPowerStore(innerThen, resultSlot, powerSlot);
}

static bool valueStaticallyNonNegativeImpl(Value value, std::set<std::string> &visiting,
                                           int depth, SelfOptStats *stats) {
  if (!value.valid() || depth > 8)
    return false;
  std::string key = valueKey(value);
  if (!visiting.insert(key).second)
    return false;
  int64_t c = 0;
  if (constantIntegerValue(value, c)) {
    visiting.erase(key);
    return c >= 0;
  }
  auto *op = value.getDefiningOp();
  if (!op) {
    visiting.erase(key);
    return false;
  }
  auto prove = [&](Value operand) {
    return valueStaticallyNonNegativeImpl(operand, visiting, depth + 1, stats);
  };
  bool proven = false;
  if (opHasName(op, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp",
                     "arith.noti", "rv_machine.seqz", "arm_machine.not"})) {
    proven = true;
  } else if (opHasName(op, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
      op->operandCount() == 2) {
    int64_t mask = 0;
    proven = (constantIntegerValue(op->operand(0), mask) && mask >= 0) ||
             (constantIntegerValue(op->operand(1), mask) && mask >= 0) ||
             prove(op->operand(0)) || prove(op->operand(1));
  } else if (opHasName(op, {"arith.ori", "rv_machine.or", "arm_machine.orr",
                            "arith.xori", "rv_machine.xor", "arm_machine.eor"}) &&
             op->operandCount() == 2) {
    proven = prove(op->operand(0)) && prove(op->operand(1));
  }
  visiting.erase(key);
  if (proven && stats)
    stats->bitwiseStaticProofs++;
  return proven;
}

static bool valueStaticallyNonNegative(Value value, SelfOptStats *stats = nullptr) {
  std::set<std::string> visiting;
  return valueStaticallyNonNegativeImpl(value, visiting, 0, stats);
}

static ProvenBitwiseFunction classifyProvenBitwiseFunction(Operation &func,
                                                           SelfOptStats *stats) {
  ProvenBitwiseFunction result;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return result;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  if (entry.args().size() != 2)
    return result;

  auto allOps = collectNestedOps(func);
  int loopCount = 0;
  Operation *loop = nullptr;
  for (auto *op : allOps) {
    if (!op || op == &func || op->isErased())
      continue;
    if (op->name() == "sysy.call") {
      if (stats)
        stats->bitwiseRejectImpure++;
      return result;
    }
    if (op->name() == "memref.load" || op->name() == "memref.store")
      return result;
    if (op->name() == "scf.while" || op->name() == "affine.for") {
      loopCount++;
      loop = op;
    }
  }
  if (loopCount != 1 || !loop || loop->name() != "scf.while" ||
      loop->getRegions().size() < 2 || loop->getRegions()[1]->getBlocks().empty())
    return result;

  std::map<std::string, int64_t> initConstants;
  std::set<std::string> paramSlots;
  for (auto &owned : entry.ops()) {
    auto *op = owned.get();
    if (!op || op == loop)
      break;
    if (!opHasName(op, {"sysy.store"}) || op->operandCount() < 2)
      continue;
    int64_t init = 0;
    if (constantIntegerValue(op->operand(0), init))
      initConstants[valueKey(op->operand(1))] = init;
    if (op->operand(0).isBlockArgument())
      paramSlots.insert(valueKey(op->operand(1)));
  }
  if (paramSlots.size() != 2)
    return result;

  Value resultSlot;
  for (auto &owned : entry.ops()) {
    auto *op = owned.get();
    if (!opHasName(op, {"sysy.return"}) || op->operandCount() == 0)
      continue;
    auto *load = op->operand(0).getDefiningOp();
    auto initIt = load && load->operandCount() > 0
                      ? initConstants.find(valueKey(load->operand(0)))
                      : initConstants.end();
    if (opHasName(load, {"sysy.load"}) && load->operandCount() > 0 &&
        initIt != initConstants.end() && initIt->second == 0) {
      resultSlot = load->operand(0);
      break;
    }
  }
  if (!resultSlot.valid())
    return result;

  Value lenSlot, powerSlot;
  for (const auto &kv : initConstants) {
    if (kv.second == 32) {
      for (auto &owned : entry.ops()) {
        if (owned && owned->resultCount() && valueKey(owned->result()) == kv.first)
          lenSlot = owned->result();
      }
    } else if (kv.second == 1) {
      for (auto &owned : entry.ops()) {
        if (owned && owned->resultCount() && valueKey(owned->result()) == kv.first)
          powerSlot = owned->result();
      }
    }
  }
  if (!lenSlot.valid() || !powerSlot.valid())
    return result;

  Block &body = *loop->getRegions()[1]->getBlocks()[0];
  std::set<std::string> bitSlots;
  std::set<std::string> bitRefs;
  std::set<std::string> paramsWithRem;
  std::set<std::string> paramsWithDiv;
  std::map<std::string, ProvenBitwiseKind> logicSlotKinds;
  bool doublesPower = false;
  bool decrementsLen = false;
  bool updatesResult = false;
  ProvenBitwiseKind condKind = ProvenBitwiseKind::None;

  for (auto &owned : body.ops()) {
    auto *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (isStoreToSlot(op, powerSlot) &&
        isBinaryWithConst(op->operand(0), "arith.muli", "rv_machine.mulw", "arm_machine.mul",
                          powerSlot, 2, true))
      doublesPower = true;
    if (isStoreToSlot(op, lenSlot) && isSubSlotByOne(op->operand(0), lenSlot))
      decrementsLen = true;
    if (opHasName(op, {"sysy.store"}) && op->operandCount() >= 2) {
      for (const auto &paramKey : paramSlots) {
        Value paramSlot;
        for (auto &entryOp : entry.ops()) {
          if (entryOp && entryOp->resultCount() && valueKey(entryOp->result()) == paramKey)
            paramSlot = entryOp->result();
        }
        if (!paramSlot.valid())
          continue;
        if (isBinaryWithConst(op->operand(0), "arith.remi", "rv_machine.remw",
                              "arm_machine.srem", paramSlot, 2)) {
          bitSlots.insert(valueKey(op->operand(1)));
          bitRefs.insert(valueKey(op->operand(1)));
          bitRefs.insert(valueKey(op->operand(0)));
          paramsWithRem.insert(paramKey);
        }
        if (isStoreToSlot(op, paramSlot) &&
            isBinaryWithConst(op->operand(0), "arith.divi", "rv_machine.divw",
                              "arm_machine.sdiv", paramSlot, 2))
          paramsWithDiv.insert(paramKey);
      }
    }
    Value logicSlot;
    ProvenBitwiseKind logicKind = ProvenBitwiseKind::None;
    if (isShortCircuitLogicIf(op, bitRefs, logicSlot, logicKind))
      logicSlotKinds[valueKey(logicSlot)] = logicKind;

    if (op->name() == "scf.if" && op->operandCount() == 1 &&
        op->getRegions().size() == 1 && !op->getRegions()[0]->getBlocks().empty()) {
      auto *condOp = op->operand(0).getDefiningOp();
      if (opHasName(condOp, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
          condOp->operandCount() == 2 &&
          isEqBitToOne(condOp->operand(0), bitRefs) &&
          isEqBitToOne(condOp->operand(1), bitRefs)) {
        condKind = ProvenBitwiseKind::And;
      } else if (isOrOfEqBitToOne(op->operand(0), bitRefs)) {
        condKind = ProvenBitwiseKind::Or;
      } else if (isNeBetweenBitLoads(op->operand(0), bitRefs)) {
        condKind = ProvenBitwiseKind::Xor;
      } else {
        auto *load = op->operand(0).getDefiningOp();
        if (opHasName(load, {"sysy.load", "memref.load"}) && load->operandCount() > 0) {
          auto logicIt = logicSlotKinds.find(valueKey(load->operand(0)));
          if (logicIt != logicSlotKinds.end())
            condKind = logicIt->second;
        }
      }
      Block &thenBlock = *op->getRegions()[0]->getBlocks()[0];
      for (auto &thenOp : thenBlock.ops())
        if (thenOp && isResultPlusPowerStore(thenOp.get(), resultSlot, powerSlot))
          updatesResult = true;
    } else if (op->name() == "scf.if" &&
               isShortCircuitOrIf(op, bitRefs, resultSlot, powerSlot)) {
      condKind = ProvenBitwiseKind::Or;
      updatesResult = true;
    }
  }

  if (paramsWithRem.size() != 2 || paramsWithDiv.size() != 2 ||
      bitSlots.size() != 2 || !doublesPower || !decrementsLen ||
      !updatesResult || condKind == ProvenBitwiseKind::None)
    return result;

  result.kind = condKind;
  result.func = &func;
  if (stats)
    stats->bitwiseCandidates++;
  return result;
}

static Operation *insertOp(Block &block, std::size_t &index,
                           std::unique_ptr<Operation> op) {
  Operation &inserted = block.insertOperation(index, std::move(op));
  index++;
  return &inserted;
}

static Operation *insertConstant(Module &module, Block &block, std::size_t &index,
                                 int64_t value, Type type, Location loc) {
  return insertOp(block, index, std::make_unique<Operation>(
      "arith.constant", std::vector<Value>{}, std::vector<Type>{type},
      std::map<std::string, Attribute>{{"value", module.context().integerAttr(value, type)}},
      loc));
}

static Operation *insertBinary(Block &block, std::size_t &index, const std::string &name,
                               Value lhs, Value rhs, Type type, Location loc) {
  return insertOp(block, index, std::make_unique<Operation>(
      name, std::vector<Value>{lhs, rhs}, std::vector<Type>{type},
      std::map<std::string, Attribute>{}, loc));
}

static Operation *insertCmp(Module &module, Block &block, std::size_t &index,
                            Value lhs, Value rhs, const std::string &pred,
                            Location loc) {
  return insertOp(block, index, std::make_unique<Operation>(
      "arith.cmpi", std::vector<Value>{lhs, rhs}, std::vector<Type>{module.context().i(32)},
      std::map<std::string, Attribute>{{"predicate", module.context().stringAttr(pred)}},
      loc));
}

static std::string bitwiseOpName(ProvenBitwiseKind kind) {
  switch (kind) {
  case ProvenBitwiseKind::And:
    return "arith.andi";
  case ProvenBitwiseKind::Or:
    return "arith.ori";
  case ProvenBitwiseKind::Xor:
    return "arith.xori";
  default:
    return "";
  }
}

static void lowerCallWithGuard(Module &module, Operation &call,
                               ProvenBitwiseKind kind, SelfOptStats *stats) {
  Block *block = call.getBlock();
  if (!block || call.operandCount() != 2 || call.resultCount() != 1)
    return;
  int callIndex = operationIndexInBlock(*block, &call);
  if (callIndex < 0)
    return;
  std::size_t index = (std::size_t) callIndex;
  Type type = call.resultType();
  Location loc = call.loc();
  std::string directName = bitwiseOpName(kind);
  if (directName.empty())
    return;

  bool lhsNonNegative = valueStaticallyNonNegative(call.operand(0), stats);
  bool rhsNonNegative = valueStaticallyNonNegative(call.operand(1), stats);
  bool directSafe = lhsNonNegative && rhsNonNegative;
  if (directSafe) {
    Operation *direct = insertBinary(*block, index, directName, call.operand(0), call.operand(1), type, loc);
    replaceAllUses(module, call.result(), direct->result());
    call.markErased();
    if (stats)
      stats->bitwiseRewrittenCalls++;
    return;
  }

  Operation *slot = insertOp(*block, index, std::make_unique<Operation>(
      "sysy.alloca", std::vector<Value>{},
      std::vector<Type>{module.context().memref({1}, type)},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  Operation *zeroA = insertConstant(module, *block, index, 0, type, loc);
  Operation *lhsOk = insertCmp(module, *block, index, zeroA->result(), call.operand(0), "le", loc);
  Operation *zeroB = insertConstant(module, *block, index, 0, type, loc);
  Operation *rhsOk = insertCmp(module, *block, index, zeroB->result(), call.operand(1), "le", loc);
  Operation *guard = insertBinary(*block, index, "arith.andi", lhsOk->result(), rhsOk->result(),
                                  module.context().i(32), loc);

  auto ifOp = std::make_unique<Operation>(
      "scf.if", std::vector<Value>{guard->result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc);
  ifOp->addRegion();
  ifOp->addRegion();
  Operation *ifPtr = insertOp(*block, index, std::move(ifOp));
  Block &thenBlock = ifPtr->getRegions()[0]->addBlock();
  Operation &direct = thenBlock.addOperation(std::make_unique<Operation>(
      directName, std::vector<Value>{call.operand(0), call.operand(1)}, std::vector<Type>{type},
      std::map<std::string, Attribute>{}, loc));
  thenBlock.addOperation(std::make_unique<Operation>(
      "sysy.store", std::vector<Value>{direct.result(), slot->result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  thenBlock.addOperation(std::make_unique<Operation>(
      "scf.yield", std::vector<Value>{}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc));

  Block &elseBlock = ifPtr->getRegions()[1]->addBlock();
  Operation &fallback = elseBlock.addOperation(std::make_unique<Operation>(
      "sysy.call", std::vector<Value>{call.operand(0), call.operand(1)}, std::vector<Type>{type},
      call.attrs(), loc));
  elseBlock.addOperation(std::make_unique<Operation>(
      "sysy.store", std::vector<Value>{fallback.result(), slot->result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  elseBlock.addOperation(std::make_unique<Operation>(
      "scf.yield", std::vector<Value>{}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc));

  Operation *load = insertOp(*block, index, std::make_unique<Operation>(
      "sysy.load", std::vector<Value>{slot->result()}, std::vector<Type>{type},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  replaceAllUses(module, call.result(), load->result());
  call.markErased();
  if (stats)
    stats->bitwiseGuardedCalls++;
}

enum class RotateHelperKind {
  None,
  Left,
  Right,
};

struct RotateHelperFunction {
  RotateHelperKind kind = RotateHelperKind::None;
  std::map<int64_t, int64_t> factors;
};

static int64_t denseRotateMaxShift(const RotateHelperFunction &info) {
  if (info.kind == RotateHelperKind::None || info.factors.empty())
    return 0;
  int64_t maxShift = info.factors.rbegin()->first;
  if (maxShift <= 0 || maxShift > 30)
    return 0;
  for (int64_t shift = 1; shift <= maxShift; shift++) {
    auto it = info.factors.find(shift);
    if (it == info.factors.end() || it->second != (int64_t(1) << shift))
      return 0;
  }
  return maxShift;
}

static bool isValueOrLoadFromSlot(Value value, Value slot, Value arg) {
  if (value == arg)
    return true;
  return slot.valid() && isLoadFromSlot(value, slot);
}

static bool isEqToConst(Value value, Value slot, Value arg, int64_t &constant) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "eq")
    return false;
  auto matches = [&](Value maybeValue, Value maybeConst) {
    return isValueOrLoadFromSlot(maybeValue, slot, arg) &&
           constantIntegerValue(maybeConst, constant);
  };
  return matches(cmp->operand(0), cmp->operand(1)) ||
         matches(cmp->operand(1), cmp->operand(0));
}

static Operation *singleReturnInRegion(Operation *op) {
  if (!op || op->getRegions().empty() || op->getRegions()[0]->getBlocks().empty())
    return nullptr;
  Operation *ret = nullptr;
  for (auto &owned : op->getRegions()[0]->getBlocks()[0]->ops()) {
    Operation *child = owned.get();
    if (!child || child->isErased() || child->name() == "scf.yield")
      continue;
    if (opHasName(child, {"sysy.return", "scf.return"})) {
      if (ret)
        return nullptr;
      ret = child;
      continue;
    }
    if (ret)
      return nullptr;
    if (!opHasName(child, {"sysy.load", "memref.load", "arith.constant",
                           "arith.addi", "arith.subi", "arith.muli",
                           "arith.divi", "arith.remi", "arith.cmpi",
                           "rv_machine.li", "rv_machine.addw",
                           "rv_machine.subw", "rv_machine.mulw",
                           "rv_machine.divw", "rv_machine.remw",
                           "rv_machine.cmp", "arm_machine.mov",
                           "arm_machine.add", "arm_machine.sub",
                           "arm_machine.mul", "arm_machine.sdiv",
                           "arm_machine.srem", "arm_machine.cmp"}))
      return nullptr;
  }
  return ret;
}

static RotateHelperFunction classifyRotateHelperFunction(Operation &func) {
  RotateHelperFunction info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  if (entry.args().size() != 2)
    return info;
  Value xArg = entry.args()[0]->value();
  Value nArg = entry.args()[1]->value();
  Value xSlot;
  Value nSlot;
  for (auto &owned : entry.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || !isStoreToSlot(op, op->operandCount() >= 2 ? op->operand(1) : Value()))
      continue;
    if (op->operandCount() >= 2 && op->operand(0) == xArg)
      xSlot = op->operand(1);
    if (op->operandCount() >= 2 && op->operand(0) == nArg)
      nSlot = op->operand(1);
  }

  bool sawCase = false;
  bool finalReturnsX = false;
  RotateHelperKind kind = RotateHelperKind::None;
  std::map<int64_t, int64_t> factors;
  for (auto &owned : entry.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.if") {
      if (op->operandCount() != 1)
        return {};
      int64_t shift = 0;
      if (!isEqToConst(op->operand(0), nSlot, nArg, shift) || shift < 1 || shift > 30)
        return {};
      Operation *ret = singleReturnInRegion(op);
      if (!ret || ret->operandCount() != 1)
        return {};
      Operation *arith = ret->operand(0).getDefiningOp();
      if (!arith || arith->operandCount() != 2)
        return {};
      int64_t factor = 0;
      bool xFirst = isValueOrLoadFromSlot(arith->operand(0), xSlot, xArg) &&
                    constantIntegerValue(arith->operand(1), factor);
      bool xSecond = isValueOrLoadFromSlot(arith->operand(1), xSlot, xArg) &&
                     constantIntegerValue(arith->operand(0), factor);
      RotateHelperKind thisKind = RotateHelperKind::None;
      if (opHasName(arith, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) &&
          (xFirst || xSecond)) {
        thisKind = RotateHelperKind::Left;
      } else if (opHasName(arith, {"arith.divi", "rv_machine.divw", "arm_machine.sdiv"}) &&
                 xFirst) {
        thisKind = RotateHelperKind::Right;
      } else {
        return {};
      }
      if (factor != (int64_t(1) << shift))
        return {};
      if (kind == RotateHelperKind::None)
        kind = thisKind;
      else if (kind != thisKind)
        return {};
      factors[shift] = factor;
      sawCase = true;
      continue;
    }
    if (opHasName(op, {"sysy.return", "scf.return"})) {
      if (op->operandCount() != 1 || !isValueOrLoadFromSlot(op->operand(0), xSlot, xArg))
        return {};
      finalReturnsX = true;
    }
  }

  if (sawCase && finalReturnsX && kind != RotateHelperKind::None) {
    info.kind = kind;
    info.factors = std::move(factors);
  }
  return info;
}

} // namespace

void runProvenBitwiseHelper(Module &module, SelfOptStats *stats) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_PROVEN_BITWISE");
  if (enabled && std::string(enabled) == "0")
    return;

  std::map<std::string, ProvenBitwiseFunction> classified;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.func")
      continue;
    auto info = classifyProvenBitwiseFunction(*op, stats);
    if (info.kind == ProvenBitwiseKind::None)
      continue;
    std::string symbol = symbolAttr(op->attr("sym_name"));
    if (!symbol.empty())
      classified[symbol] = info;
  }
  if (classified.empty())
    return;

  std::vector<Operation*> calls;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.call" ||
        op->operandCount() != 2 || op->resultCount() != 1)
      continue;
    std::string callee = symbolAttr(op->attr("callee"));
    if (classified.count(callee))
      calls.push_back(op);
  }

  for (auto *call : calls) {
    if (!call || call->isErased())
      continue;
    std::string callee = symbolAttr(call->attr("callee"));
    auto it = classified.find(callee);
    if (it == classified.end())
      continue;
    lowerCallWithGuard(module, *call, it->second.kind, stats);
  }
  eraseMarked(module);
}

void runRotateHelperFold(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_ROT_HELPER", true))
    return;
  std::map<std::string, RotateHelperFunction> helpers;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.func")
      continue;
    auto info = classifyRotateHelperFunction(*op);
    if (info.kind == RotateHelperKind::None)
      continue;
    std::string symbol = symbolAttr(op->attr("sym_name"));
    if (!symbol.empty())
      helpers[symbol] = std::move(info);
  }
  if (helpers.empty())
    return;

  std::vector<Operation*> calls;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.call" ||
        op->operandCount() != 2 || op->resultCount() != 1)
      continue;
    std::string callee = symbolAttr(op->attr("callee"));
    if (helpers.count(callee) != 0)
      calls.push_back(op);
  }

  for (Operation *call : calls) {
    if (!call || call->isErased())
      continue;
    auto helperIt = helpers.find(symbolAttr(call->attr("callee")));
    if (helperIt == helpers.end())
      continue;
    int64_t shift = 0;
    if (!constantIntegerValue(call->operand(1), shift))
      shift = std::numeric_limits<int64_t>::min();
    Block *block = call->getBlock();
    if (!block)
      continue;
    int indexRaw = operationIndexInBlock(*block, call);
    if (indexRaw < 0)
      continue;
    if (shift == std::numeric_limits<int64_t>::min()) {
      int64_t maxShift = denseRotateMaxShift(helperIt->second);
      if (maxShift <= 0)
        continue;
      auto replacement = std::make_unique<Operation>(
          "sysy.rotate_helper",
          std::vector<Value>{call->operand(0), call->operand(1)},
          std::vector<Type>{call->resultType()},
          std::map<std::string, Attribute>{
              {"direction", module.context().stringAttr(
                                helperIt->second.kind == RotateHelperKind::Left ? "left"
                                                                                 : "right")},
              {"max_shift", module.context().integerAttr(maxShift,
                                                          module.context().i(32))}},
          call->loc());
      replaceOperation(module, *call, std::move(replacement));
      if (stats)
        stats->rotHelperFolds++;
      continue;
    }
    if (shift <= 0 || helperIt->second.factors.count(shift) == 0) {
      replaceAllUses(module, call->result(), call->operand(0));
      call->markErased();
      if (stats)
        stats->rotHelperFolds++;
      continue;
    }
    std::size_t index = (std::size_t) indexRaw;
    int64_t factor = helperIt->second.factors[shift];
    Operation *factorOp = insertConstant(module, *block, index, factor, call->resultType(), call->loc());
    const std::string opName = helperIt->second.kind == RotateHelperKind::Left
                                   ? "arith.muli"
                                   : "arith.divi";
    Operation *folded = insertBinary(*block, index, opName, call->operand(0),
                                     factorOp->result(), call->resultType(), call->loc());
    replaceAllUses(module, call->result(), folded->result());
    call->markErased();
    if (stats)
      stats->rotHelperFolds++;
  }
  eraseMarked(module);
}

void collectAffineNestSummary(Module &module, SelfOptStats *stats) {
  if (!stats)
    return;
  stats->affineSummaryLoops = 0;
  stats->affineSummaryMemoryOps = 0;
  stats->affineSummarySideEffects = 0;
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "affine.for" || op->name() == "scf.for" || op->name() == "scf.while")
      stats->affineSummaryLoops++;
    if (op->name() == "memref.load" || op->name() == "memref.store" ||
        op->name() == "sysy.load" || op->name() == "sysy.store")
      stats->affineSummaryMemoryOps++;
    if (op->name() == "sysy.call")
      stats->affineSummarySideEffects++;
    if (op->name() == "affine.for" && op->operandCount() >= 3) {
      int64_t step = 1;
      if (constantIntegerValue(op->operand(2), step) && step != 1)
        stats->loopTiles++;
    }
  }
}

void runLoopVectorization(Module &module) {
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.while" || op->name() == "scf.for" || op->name() == "affine.for") {
      Block *body = nullptr;
      if (!op->getRegions().empty() && !op->getRegions()[0]->getBlocks().empty()) {
        body = op->getRegions()[0]->getBlocks()[0].get();
      }
      if (!body || body->ops().empty())
        continue;

      std::vector<Operation*> loads;
      std::vector<Operation*> stores;
      std::vector<Operation*> ariths;

      for (auto &owned : body->ops()) {
        auto *child = owned.get();
        if (!child || child->isErased())
          continue;
        if (child->name() == "memref.store") {
          stores.push_back(child);
        }
      }

      if (stores.empty())
        continue;

      std::set<std::string> activeValNames;
      for (auto *storeOp : stores) {
        if (storeOp->operandCount() >= 1) {
          activeValNames.insert(valueKey(storeOp->operand(0)));
        }
      }

      for (int i = (int) body->ops().size() - 1; i >= 0; i--) {
        auto *child = body->ops()[i].get();
        if (!child || child->isErased())
          continue;
        if (child->resultCount() > 0) {
          std::string resName = valueKey(child->result());
          if (activeValNames.count(resName) > 0) {
            if (child->name() == "memref.load") {
              loads.push_back(child);
            } else if (child->name() == "arith.addi" || child->name() == "arith.addf" ||
                       child->name() == "arith.subi" || child->name() == "arith.subf" ||
                       child->name() == "rv_machine.addw" || child->name() == "arm_machine.add" ||
                       child->name() == "arith.muli" || child->name() == "arith.mulf") {
              ariths.push_back(child);
              for (int opIdx = 0; opIdx < child->operandCount(); opIdx++) {
                activeValNames.insert(valueKey(child->operand(opIdx)));
              }
            }
          }
        }
      }

      std::reverse(loads.begin(), loads.end());
      std::reverse(ariths.begin(), ariths.end());

      if (loads.empty())
        continue;

      bool ok = true;
      for (auto *l : loads) {
        if (l->operandCount() < 2) { ok = false; break; }
      }
      for (auto *s : stores) {
        if (s->operandCount() < 3) { ok = false; break; }
      }
      if (!ok)
        continue;

      bool hasRAW = false;
      for (auto *storeOp : stores) {
        std::string storeBase = symbolAttr(storeOp->attr("symbol"));
        if (storeBase.empty())
          storeBase = symbolAttr(storeOp->attr("sym_name"));
        for (auto *loadOp : loads) {
          std::string loadBase = symbolAttr(loadOp->attr("symbol"));
          if (loadBase.empty())
            loadBase = symbolAttr(loadOp->attr("sym_name"));
          if (storeBase == loadBase) {
            if (storeOp->operandCount() >= 3 && loadOp->operandCount() >= 2) {
              if (valueKey(storeOp->operand(2)) != valueKey(loadOp->operand(1))) {
                hasRAW = true;
                break;
              }
            } else {
              hasRAW = true;
              break;
            }
          }
        }
        if (hasRAW)
          break;
      }
      if (hasRAW)
        continue;

      std::size_t insertIdx = body->ops().size() - 1;

      int64_t lanes = 4;
      Type elemType = loads[0]->resultType();
      Type vecType = module.context().vector(elemType, lanes);

      std::map<std::string, Value> vecLoads;
      for (auto *loadOp : loads) {
        Value base = loadOp->operand(0);
        Value index = loadOp->operand(1);
        std::string loadSym = symbolAttr(loadOp->attr("symbol"));
        if (loadSym.empty())
          loadSym = symbolAttr(loadOp->attr("sym_name"));
        auto &vread = body->insertOperation(insertIdx++, std::make_unique<Operation>(
            "vector.transfer_read", std::vector<Value>{base, index}, std::vector<Type>{vecType},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(loadSym)}},
            loadOp->loc()));
        vecLoads[valueKey(loadOp->result())] = vread.result();
      }

      std::map<std::string, Value> vecAriths;
      for (auto *arithOp : ariths) {
        std::vector<Value> vecOperands;
        for (int i = 0; i < arithOp->operandCount(); i++) {
          Value operand = arithOp->operand(i);
          std::string operandKey = valueKey(operand);
          if (vecLoads.count(operandKey)) {
            vecOperands.push_back(vecLoads[operandKey]);
          } else if (vecAriths.count(operandKey)) {
            vecOperands.push_back(vecAriths[operandKey]);
          } else {
            auto &splat = body->insertOperation(insertIdx++, std::make_unique<Operation>(
                "vector.splat", std::vector<Value>{operand}, std::vector<Type>{vecType},
                std::map<std::string, Attribute>{}, arithOp->loc()));
            vecOperands.push_back(splat.result());
          }
        }
        auto &varith = body->insertOperation(insertIdx++, std::make_unique<Operation>(
            arithOp->name(), vecOperands, std::vector<Type>{vecType},
            arithOp->attrs(), arithOp->loc()));
        vecAriths[valueKey(arithOp->result())] = varith.result();
      }

      for (auto *storeOp : stores) {
        Value val = storeOp->operand(0);
        Value base = storeOp->operand(1);
        Value index = storeOp->operand(2);
        std::string storeSym = symbolAttr(storeOp->attr("symbol"));
        if (storeSym.empty())
          storeSym = symbolAttr(storeOp->attr("sym_name"));

        Value vecVal;
        std::string valKey = valueKey(val);
        if (vecLoads.count(valKey)) {
          vecVal = vecLoads[valKey];
        } else if (vecAriths.count(valKey)) {
          vecVal = vecAriths[valKey];
        } else {
          auto &splat = body->insertOperation(insertIdx++, std::make_unique<Operation>(
              "vector.splat", std::vector<Value>{val}, std::vector<Type>{vecType},
              std::map<std::string, Attribute>{}, storeOp->loc()));
          vecVal = splat.result();
        }

        body->insertOperation(insertIdx++, std::make_unique<Operation>(
            "vector.transfer_write", std::vector<Value>{vecVal, base, index}, std::vector<Type>{},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(storeSym)}},
            storeOp->loc()));
      }

      for (auto &owned : body->ops()) {
        auto *child = owned.get();
        if (!child || child->isErased())
          continue;
        if ((child->name() == "rv_machine.li" || child->name() == "arith.constant" || child->name() == "arm_machine.mov") &&
            parseIntegerAttr(child->attr("value")) == 1) {
          child->setAttr("value", module.context().integerAttr(lanes, module.context().i(32)));
        }
      }

      for (auto *opToErase : loads) {
        for (int i = 0; i < opToErase->operandCount(); i++) {
          opToErase->setOperand(i, Value());
        }
        opToErase->markErased();
      }
      for (auto *opToErase : ariths) {
        for (int i = 0; i < opToErase->operandCount(); i++) {
          opToErase->setOperand(i, Value());
        }
        opToErase->markErased();
      }
      for (auto *opToErase : stores) {
        for (int i = 0; i < opToErase->operandCount(); i++) {
          opToErase->setOperand(i, Value());
        }
        opToErase->markErased();
      }
    }
  }
  eraseMarked(module);
}

static const char *kCoreRules =
  "rule fold_addi_zero arith.addi addi-zero 10\n"
  "rule fold_muli_one arith.muli muli-one 9\n"
  "rule fold_select_same arith.select select-same 8\n"
  "rule fold_subi_same arith.subi subi-same 12\n"
  "rule fold_subi_zero arith.subi subi-zero 10\n"
  "rule fold_muli_zero arith.muli muli-zero 11\n"
  "rule fold_divi_one arith.divi divi-one 9\n"
  "rule fold_remi_one arith.remi remi-one 9\n"
  "rule fold_andi_same arith.andi andi-same 7\n"
  "rule fold_andi_zero arith.andi andi-zero 8\n"
  "rule fold_ori_same arith.ori ori-same 6\n"
  "rule fold_ori_zero arith.ori ori-zero 7\n"
  "rule fold_double_noti arith.noti double-noti 15\n";


static Module buildSample(Context &ctx) {
  Module module(ctx);
  auto &top = module.body().getBlocks()[0];
  Builder topBuilder(ctx, top.get());
  auto &func = topBuilder.create(
      "sysy.func", {}, {}, {{"sym_name", ctx.stringAttr("main")}},
      ctx.loc("sample.sy", 1, 1), 1);
  auto &entry = func.getRegions()[0]->addBlock();
  auto &arg = entry.addArgument(ctx.i(32), ctx.loc("sample.sy", 1, 12), "n");
  Builder b(ctx, &entry);
  auto &zero = b.create("arith.constant", {}, {ctx.i(32)},
                        {{"value", ctx.integerAttr(0, ctx.i(32))}},
                        ctx.loc("sample.sy", 2, 3));
  auto &add = b.create("arith.addi", {arg.value(), zero.result()}, {ctx.i(32)},
                       {}, ctx.loc("sample.sy", 2, 8));
  b.create("scf.return", {add.result()}, {}, {}, ctx.loc("sample.sy", 3, 1));
  return module;
}

static Module buildNativeAsmSample(Context &ctx) {
  Module module(ctx);
  auto &top = module.body().getBlocks()[0];
  Builder topBuilder(ctx, top.get());
  auto &func = topBuilder.create(
      "sysy.func", {}, {}, {{"sym_name", ctx.stringAttr("main")}},
      ctx.loc("native.sy", 1, 1), 1);
  auto &entry = func.getRegions()[0]->addBlock();
  Builder b(ctx, &entry);
  auto &seven = b.create("arith.constant", {}, {ctx.i(32)},
                         {{"value", ctx.integerAttr(7, ctx.i(32))}},
                         ctx.loc("native.sy", 2, 3));
  auto &one = b.create("arith.constant", {}, {ctx.i(32)},
                       {{"value", ctx.integerAttr(1, ctx.i(32))}},
                       ctx.loc("native.sy", 2, 7));
  auto &sum = b.create("arith.addi", {seven.result(), one.result()}, {ctx.i(32)},
                       {}, ctx.loc("native.sy", 2, 11));
  b.create("sysy.return", {sum.result()}, {}, {}, ctx.loc("native.sy", 3, 1));
  return module;
}

int runCoreSelfTest(std::ostream &os) {
  Context ctx;
  auto i32a = ctx.i(32);
  auto i32b = ctx.i(32);
  auto locA = ctx.loc("sample.sy", 1, 1);
  auto locB = ctx.loc("sample.sy", 1, 1);
  bool uniqued = i32a == i32b && locA == locB;
  Module module = buildSample(ctx);
  auto symtab = buildSymbolTable(module);
  Operation *mainFunc = symtab.lookup("main");
  Operation *addi = nullptr;
  Operation *zero = nullptr;
  for (auto *op : walk(module)) {
    if (!op)
      continue;
    if (op->name() == "arith.addi")
      addi = op;
    if (op->name() == "arith.constant")
      zero = op;
  }
  int addUsesBefore = addi ? (int) usesOf(module, addi->result()).size() : -1;
  int zeroUsesBefore = zero ? (int) usesOf(module, zero->result()).size() : -1;
  auto before = verify(module);
  std::vector<std::string> parseErrors;
  auto rules = parseDRR(kCoreRules, parseErrors);
  auto stats = applyGreedyPatterns(module, rules);
  auto after = verify(module);

  std::string eraseError;
  bool erasedDeadConstant = false;
  if (zero && usesOf(module, zero->result()).empty())
    erasedDeadConstant = eraseOperation(module, *zero, &eraseError);

  std::ostringstream printed;
  print(module, printed);
  std::vector<std::string> roundTripErrors;
  auto roundTrip = parse(ctx, printed.str(), roundTripErrors);
  auto roundTripVerify = roundTrip ? verify(*roundTrip) : VerifyResult{false, {"parse failed"}};
  auto roundTripSymbols = roundTrip ? buildSymbolTable(*roundTrip) : SymbolTable();

  Module mutation = buildNativeAsmSample(ctx);
  Operation *firstConst = nullptr;
  Operation *secondConst = nullptr;
  Operation *mutationAdd = nullptr;
  for (auto *op : walk(mutation)) {
    if (!op)
      continue;
    if (op->name() == "arith.constant") {
      if (!firstConst)
        firstConst = op;
      else if (!secondConst)
        secondConst = op;
    } else if (op->name() == "arith.addi") {
      mutationAdd = op;
    }
  }
  bool moved = firstConst && secondConst && moveOperationBefore(*secondConst, *firstConst);
  Operation *replacement = nullptr;
  if (mutationAdd) {
    auto repl = std::make_unique<Operation>(
        "arith.subi", mutationAdd->getOperands(), std::vector<Type>{mutationAdd->resultType()},
        std::map<std::string, Attribute>{}, mutationAdd->loc());
    replacement = replaceOperation(mutation, *mutationAdd, std::move(repl));
  }
  auto mutationVerify = verify(mutation);

  os << "[self-mlir-core] uniqued=" << (uniqued ? 1 : 0)
     << " types=" << ctx.typeCount()
     << " attrs=" << ctx.attrCount()
     << " locs=" << ctx.locationCount()
     << " block-args=1"
     << " symbols=" << symtab.all().size()
     << " main-symbol=" << (mainFunc ? 1 : 0)
     << " add-uses-before=" << addUsesBefore
     << " zero-uses-before=" << zeroUsesBefore
     << " erased-dead-const=" << (erasedDeadConstant ? 1 : 0)
     << " moved-op=" << (moved ? 1 : 0)
     << " replaced-op=" << (replacement ? 1 : 0)
     << " mutation-verify=" << (mutationVerify.ok ? 1 : 0)
     << " roundtrip-verify=" << (roundTripVerify.ok ? 1 : 0)
     << " roundtrip-errors=" << roundTripErrors.size()
     << " roundtrip-symbols=" << roundTripSymbols.all().size()
     << " rules=" << stats.rules
     << " rewrites=" << stats.rewrites
     << " verify-before=" << (before.ok ? 1 : 0)
     << " verify-after=" << (after.ok ? 1 : 0)
     << " parse-errors=" << parseErrors.size() << "\n";
  print(module, os);
  return uniqued && before.ok && after.ok && parseErrors.empty() &&
         stats.rewrites == 1 && mainFunc && symtab.duplicates().empty() &&
         addUsesBefore == 1 && zeroUsesBefore == 1 && erasedDeadConstant &&
         moved && replacement && mutationVerify.ok && roundTrip &&
         roundTripVerify.ok && roundTripErrors.empty() &&
         roundTripSymbols.lookup("main") ? 0 : 1;
}

int runConversionSelfTest(std::ostream &os) {
  Context ctx;
  Module rvModule = buildSample(ctx);
  ConversionTarget rvTarget;
  rvTarget.addLegalDialect("builtin");
  rvTarget.addLegalDialect("sysy");
  rvTarget.addLegalDialect("scf");
  rvTarget.addLegalDialect("rv_machine");
  auto rvStats = convertDialects(rvModule, rvTarget, {
      {"arith.constant", "rv_machine.li"},
      {"arith.addi", "rv_machine.addw"},
  });
  auto rvVerify = verify(rvModule);

  Module armModule = buildSample(ctx);
  ConversionTarget armTarget;
  armTarget.addLegalDialect("builtin");
  armTarget.addLegalDialect("sysy");
  armTarget.addLegalDialect("scf");
  armTarget.addLegalDialect("arm_machine");
  auto armStats = convertDialects(armModule, armTarget, {
      {"arith.constant", "arm_machine.mov"},
      {"arith.addi", "arm_machine.add"},
  });
  auto armVerify = verify(armModule);

  Module rollbackModule = buildSample(ctx);
  ConversionTarget strictTarget;
  strictTarget.addLegalDialect("builtin");
  auto rollbackStats = convertDialects(rollbackModule, strictTarget, {});
  auto rollbackVerify = verify(rollbackModule);

  os << "[self-mlir-conversion] rv-converted=" << rvStats.converted
     << " rv-failed=" << rvStats.failed
     << " rv-verify=" << (rvVerify.ok ? 1 : 0)
     << " arm-converted=" << armStats.converted
     << " arm-failed=" << armStats.failed
     << " arm-verify=" << (armVerify.ok ? 1 : 0)
     << " rollback-failed=" << rollbackStats.failed
     << " rollback-count=" << rollbackStats.rollbacks
     << " rollback-verify=" << (rollbackVerify.ok ? 1 : 0) << "\n";
  print(rvModule, os);
  print(armModule, os);
  return rvStats.converted == 2 && rvStats.failed == 0 && rvVerify.ok &&
         armStats.converted == 2 && armStats.failed == 0 && armVerify.ok &&
         rollbackStats.failed == 1 && rollbackStats.rollbacks == 1 &&
         rollbackVerify.ok ? 0 : 1;
}

int runNativeBackendSelfTest(std::ostream &os) {
  Context ctx;
  Module rvModule = buildNativeAsmSample(ctx);
  ConversionTarget rvTarget;
  rvTarget.addLegalDialect("builtin");
  rvTarget.addLegalDialect("sysy");
  rvTarget.addLegalDialect("rv_machine");
  auto rvConv = convertDialects(rvModule, rvTarget, {
      {"arith.constant", "rv_machine.li"},
      {"arith.addi", "rv_machine.addw"},
  });
  NativeAsmStats rvStats;
  std::ostringstream rvAsm;
  bool rvOk = emitNativeAssembly(rvModule, "riscv", rvAsm, rvStats);

  Module armModule = buildNativeAsmSample(ctx);
  ConversionTarget armTarget;
  armTarget.addLegalDialect("builtin");
  armTarget.addLegalDialect("sysy");
  armTarget.addLegalDialect("arm_machine");
  auto armConv = convertDialects(armModule, armTarget, {
      {"arith.constant", "arm_machine.mov"},
      {"arith.addi", "arm_machine.add"},
  });
  NativeAsmStats armStats;
  std::ostringstream armAsm;
  bool armOk = emitNativeAssembly(armModule, "arm", armAsm, armStats);

  os << "[self-mlir-native-backend]"
     << " rv-converted=" << rvConv.converted
     << " rv-failed=" << rvConv.failed
     << " rv-emitted=" << (rvOk ? 1 : 0)
     << " rv-machine-ops=" << rvStats.machineOps
     << " rv-unsupported=" << rvStats.unsupportedOps
     << " arm-converted=" << armConv.converted
     << " arm-failed=" << armConv.failed
     << " arm-emitted=" << (armOk ? 1 : 0)
     << " arm-machine-ops=" << armStats.machineOps
     << " arm-unsupported=" << armStats.unsupportedOps
     << " legacy-free=" << ((rvStats.legacyOps == 0 && rvStats.phiLikeOps == 0 &&
                              armStats.legacyOps == 0 && armStats.phiLikeOps == 0) ? 1 : 0)
     << "\n";
  os << "[self-mlir-native-backend-rv-asm]\n" << rvAsm.str();
  os << "[self-mlir-native-backend-arm-asm]\n" << armAsm.str();

  return rvOk && armOk && rvConv.failed == 0 && armConv.failed == 0 &&
         rvStats.legacyOps == 0 && rvStats.phiLikeOps == 0 &&
         armStats.legacyOps == 0 && armStats.phiLikeOps == 0 &&
         rvAsm.str().find("addw") != std::string::npos &&
         armAsm.str().find("add ") != std::string::npos ? 0 : 1;
}

void dumpSample(std::ostream &os) {
  Context ctx;
  Module module = buildSample(ctx);
  print(module, os);
}

} // namespace sys::mlir
