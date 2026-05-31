#include "SelfMLIR.h"


#include "../parse/ASTNode.h"
#include "../parse/Type.h"
#include "../utils/DynamicCast.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sys::mlir {

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

RewriteStats applyGreedyPatterns(Module &module, const std::vector<RewriteRule> &rules) {
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
  static const char *regs[] = {"t0", "t1", "t2", "t3", "t4"};
  return regs[index % 5];
}

std::string armResultReg(int index) {
  static const char *regs[] = {"w9", "w10", "w11", "w12", "w13", "w14", "w15"};
  return regs[index % 7];
}

std::string rvFloatReg(int index) {
  static const char *regs[] = {"ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7"};
  return regs[index % 8];
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
  return value.printName();
}

std::string lookupReg(Value value, const std::map<std::string, std::string> &regs) {
  auto it = regs.find(valueKey(value));
  return it == regs.end() ? "" : it->second;
}

bool constantIntegerValue(Value value, int64_t &out) {
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

int64_t memrefAllocationBytes(Type type) {
  std::string text = type.str();
  auto begin = text.find("memref<");
  if (begin == std::string::npos)
    return 4;
  begin += 7;
  auto end = text.find('>', begin);
  if (end == std::string::npos)
    end = text.size();
  std::string shape = text.substr(begin, end - begin);
  auto elemPos = shape.find("xi32");
  int elemBytes = 4;
  if (elemPos == std::string::npos) {
    elemPos = shape.find("xf32");
    elemBytes = 4;
  }
  if (elemPos == std::string::npos)
    return 8;
  shape = shape.substr(0, elemPos);
  int64_t elems = 1;
  std::size_t pos = 0;
  while (pos < shape.size()) {
    std::size_t next = shape.find('x', pos);
    std::string part = shape.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
    if (!part.empty() && part[0] != '?') {
      int64_t dim = std::stoll(part);
      if (dim > 0)
        elems *= dim;
    }
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }
  return std::max<int64_t>(4, elems * elemBytes);
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

bool emitFunctionAssembly(Operation &func, const std::string &target, std::ostream &os,
                          NativeAsmStats &stats,
                          const std::map<std::string, std::string> &globalLabels) {
  if (func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1) {
    stats.unsupportedOps++;
    stats.error = "native asm currently requires one-region/one-block functions";
    return false;
  }

  std::string name = symbolAttr(func.attr("sym_name"), "main");
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

  bool livenessEnabled = true;
  if (const char *value = std::getenv("SISY_ENABLE_SELF_MACHINE_LIVENESS"))
    livenessEnabled = std::string(value) != "0";
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

  int64_t frameBytes = 0;
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
      if (arg && hasScalarHome(arg->type())) {
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
        if (hasScalarHome(value.type())) {
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
  int64_t calleeSaveBase = frameBytes;
  int calleeSaveCount = isArm ? 10 : 12;
  frameBytes += calleeSaveCount * 8;
  frameBytes = (frameBytes + 15) & ~int64_t(15);

  os << (isArm ? "    .text\n    .global " : "    .text\n    .globl ") << name << "\n";
  os << name << ":\n";
  if (frameBytes > 0) {
    if (isArm)
      os << "    sub sp, sp, #" << frameBytes << "\n";
    else
      os << "    addi sp, sp, -" << frameBytes << "\n";
  }
  for (int i = 0; i < calleeSaveCount; i++) {
    int64_t off = calleeSaveBase + i * 8;
    if (isArm)
      os << "    str x" << (19 + i) << ", [sp, #" << off << "]\n";
    else
      os << "    sd s" << i << ", " << off << "(sp)\n";
  }

  for (const auto &kv : globalLabels)
    regs[kv.first] = "global:" + kv.second;

  int nextReg = 0;
  int nextVecReg = 0;
  int nextFloatReg = 0;
  int returnsBefore = stats.returns;

  auto scratchReg = [&](int n) -> std::string {
    if (isArm)
      return n == 0 ? "x16" : "x17";
    return n == 0 ? "t5" : "t6";
  };
  auto resultReg = [&]() -> std::string {
    return isArm ? armResultReg(nextReg++) : rvResultReg(nextReg++);
  };
  auto floatReg = [&]() -> std::string {
    return isArm ? armFloatReg(nextFloatReg++) : rvFloatReg(nextFloatReg++);
  };
  std::set<std::string> homeValid;
  auto looksFloatReg = [&](const std::string &reg) {
    return !reg.empty() && (isArm ? reg[0] == 's' : reg[0] == 'f');
  };
  std::function<void(Value, const std::string&)> spillHome =
      [&](Value value, const std::string &reg) {
    auto it = valueSlots.find(valueKey(value));
    if (it == valueSlots.end() || reg.empty() ||
        reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0)
      return;
    int64_t off = it->second;
    bool fp = isFloatType(value.type()) || looksFloatReg(reg);
    if (isArm)
      os << "    str " << reg << ", [sp, #" << off << "]\n";
    else
      os << "    " << (fp ? "fsw " : "sw ") << reg << ", " << off << "(sp)\n";
    homeValid.insert(valueKey(value));
    stats.liveSpills++;
  };
  auto maybeSpillBeforeClobber = [&](const std::string &key, const std::string &reg,
                                     bool callBoundary) {
    if (!livenessEnabled || key.empty() || reg.empty() ||
        reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0 ||
        remainingUses[key] <= 0 || homeValid.count(key) != 0 ||
        valueByKey.count(key) == 0)
      return;
    spillHome(valueByKey[key], reg);
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
    if (remainingUses[valueKey(value)] <= 0) {
      stats.deadSpillsAvoided++;
      return false;
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
  for (size_t i = 0; i < block.args().size(); i++) {
    const auto &arg = *block.args()[i];
    std::string reg;
    if (isFloatType(arg.type()))
      reg = (isArm ? "s" : "fa") + std::to_string((int) i);
    else if (isArm)
      reg = "w" + std::to_string((int) i);
    else
      reg = "a" + std::to_string((int) i);
    bindResult(arg.value(), reg);
    if (!livenessEnabled || remainingUses[valueKey(arg.value())] > 0)
      spillHome(arg.value(), reg);
  }
  auto materializeAddress = [&](Value value, const std::string &tmp) -> std::string {
    std::string loc = lookupReg(value, regs);
    auto slotIt = stackSlots.find(valueKey(value));
    if (slotIt != stackSlots.end())
      loc = "stack:" + std::to_string(slotIt->second);
    if (loc.rfind("stack:", 0) == 0) {
      int64_t off = std::stoll(loc.substr(6));
      if (isArm)
        os << "    add " << tmp << ", sp, #" << off << "\n";
      else
        os << "    addi " << tmp << ", sp, " << off << "\n";
      return tmp;
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
    return loc;
  };
  auto ensureReg = [&](Value value, const std::string &tmp) -> std::string {
    std::string loc = lookupReg(value, regs);
    if (!loc.empty()) {
      if (loc.rfind("stack:", 0) == 0 || loc.rfind("global:", 0) == 0)
        return materializeAddress(value, tmp);
      return loc;
    }
    int64_t imm = 0;
    if (constantIntegerValue(value, imm)) {
      if (isArm)
        os << "    mov " << tmp << ", #" << imm << "\n";
      else
        os << "    li " << tmp << ", " << imm << "\n";
      return tmp;
    }
    auto home = valueSlots.find(valueKey(value));
    if (home != valueSlots.end()) {
      bool fp = isFloatType(value.type()) || looksFloatReg(tmp);
      if (isArm)
        os << "    ldr " << tmp << ", [sp, #" << home->second << "]\n";
      else
        os << "    " << (fp ? "flw " : "lw ") << tmp << ", " << home->second << "(sp)\n";
      return tmp;
    }
    return "";
  };
  auto loadFromAddress = [&](const std::string &dst, Value base, Value index) {
    std::string addr = materializeAddress(base, scratchReg(0));
    bool fpLoad = looksFloatReg(dst);
    if (index.valid()) {
      int64_t immIndex = 0;
      if (!isArm && constantIntegerValue(index, immIndex)) {
        int64_t byteOffset = immIndex * 4;
        if (byteOffset >= -2048 && byteOffset <= 2047) {
          os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", "
             << byteOffset << "(" << addr << ")\n";
          return;
        }
      }
      std::string idx = ensureReg(index, scratchReg(1));
      if (idx.empty()) {
        idx = scratchReg(1);
        if (isArm)
          os << "    mov " << idx << ", #0\n";
        else
          os << "    li " << idx << ", 0\n";
      }
      if (isArm) {
        os << "    ldr " << dst << ", [" << addr << ", " << idx << ", lsl #2]\n";
      } else {
        os << "    slli " << scratchReg(1) << ", " << idx << ", 2\n";
        os << "    add " << scratchReg(1) << ", " << addr << ", " << scratchReg(1) << "\n";
        os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", 0(" << scratchReg(1) << ")\n";
      }
      return;
    }
    if (isArm)
      os << "    ldr " << dst << ", [" << addr << "]\n";
    else
      os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", 0(" << addr << ")\n";
  };
  auto storeToAddress = [&](Value value, Value base, Value index) {
    std::string addr = materializeAddress(base, scratchReg(0));
    bool fpStore = isFloatType(value.type());
    if (index.valid()) {
      int64_t immIndex = 0;
      if (!isArm && constantIntegerValue(index, immIndex)) {
        int64_t byteOffset = immIndex * 4;
        if (byteOffset >= -2048 && byteOffset <= 2047) {
          std::string val = ensureReg(value, fpStore ? (isArm ? "s30" : "ft10") : scratchReg(1));
          fpStore = fpStore || looksFloatReg(val);
          os << "    " << (fpStore ? "fsw " : "sw ") << val << ", "
             << byteOffset << "(" << addr << ")\n";
          return;
        }
      }
      std::string idx = ensureReg(index, scratchReg(1));
      if (idx.empty()) {
        idx = scratchReg(1);
        if (isArm)
          os << "    mov " << idx << ", #0\n";
        else
          os << "    li " << idx << ", 0\n";
      }
      if (isArm) {
        std::string val = ensureReg(value, fpStore ? "s30" : "x17");
        fpStore = fpStore || looksFloatReg(val);
        os << "    str " << val << ", [" << addr << ", " << idx << ", lsl #2]\n";
      } else {
        os << "    slli " << scratchReg(1) << ", " << idx << ", 2\n";
        os << "    add " << scratchReg(1) << ", " << addr << ", " << scratchReg(1) << "\n";
        std::string val = ensureReg(value, fpStore ? "ft10" : scratchReg(0));
        fpStore = fpStore || looksFloatReg(val);
        os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << scratchReg(1) << ")\n";
      }
      return;
    }
    std::string val = ensureReg(value, fpStore ? (isArm ? "s30" : "ft10") : scratchReg(1));
    fpStore = fpStore || looksFloatReg(val);
    if (isArm)
      os << "    str " << val << ", [" << addr << "]\n";
    else
      os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << addr << ")\n";
  };

  int nextLoopId = stats.functions * 10000;
  std::map<Operation*, int> loopOps;

  for (auto *opPtr : funcOps) {
    if (!opPtr || opPtr->isErased()) continue;
    Operation &op = *opPtr;
    std::string opname = op.name();
    if (opname == "sysy.func") continue;
    consumeOperands(op);

    if (opname == "rv_machine.li" || opname == "arm_machine.mov") {
      std::string dst;
      if (op.resultType().str().find("vector") != std::string::npos) {
        dst = "v" + std::to_string(nextVecReg++);
      } else if (isFloatType(op.resultType())) {
        dst = floatReg();
      } else {
        dst = isArm ? armResultReg(nextReg++) : rvResultReg(nextReg++);
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
        dst = resultReg();
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
      std::string dst = resultReg();
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
      std::string dst = resultReg();
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
      std::string dst = resultReg();
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
      std::string dst = resultReg();
      if (isFloatType(op.resultType()))
        dst = floatReg();
      bindResult(op.result(), dst);
      loadFromAddress(dst, op.operand(0), op.operandCount() > 1 ? op.operand(1) : Value());
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
      storeToAddress(op.operand(0), op.operand(1), op.operandCount() > 2 ? op.operand(2) : Value());
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

    if (opname == "sysy.call") {
      std::string callee = symbolAttr(op.attr("callee"));
      if (callee.empty()) {
        stats.unsupportedOps++;
        stats.error = "call without callee attr";
        return false;
      }
      for (int i = 0; i < op.operandCount(); i++) {
        std::string src = ensureReg(op.operand(i), scratchReg(0));
        if (src.rfind("stack:", 0) == 0 || src.rfind("global:", 0) == 0)
          src = materializeAddress(op.operand(i), scratchReg(0));
        if (src.empty()) {
          src = scratchReg(0);
          if (isArm)
            os << "    mov " << src << ", #0\n";
          else
            os << "    li " << src << ", 0\n";
        }
        bool fpArg = op.operand(i).type().kind() == TypeKind::Float || looksFloatReg(src);
        if (isArm) {
          std::string arg = fpArg ? ("s" + std::to_string(i)) : ("w" + std::to_string(i));
          if (!fpArg && src.size() > 0 && src[0] == 'x')
            arg = "x" + std::to_string(i);
          if (src != arg)
            os << "    " << (fpArg ? "fmov " : "mov ") << arg << ", " << src << "\n";
        } else {
          std::string arg = fpArg ? ("fa" + std::to_string(i)) : ("a" + std::to_string(i));
          if (src != arg)
            os << "    " << (fpArg ? "fmv.s " : "mv ") << arg << ", " << src << "\n";
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
      }
      for (int i = 0; i < calleeSaveCount; i++) {
        int64_t off = calleeSaveBase + i * 8;
        if (isArm)
          os << "    ldr x" << (19 + i) << ", [sp, #" << off << "]\n";
        else
          os << "    ld s" << i << ", " << off << "(sp)\n";
      }
      if (frameBytes > 0) {
        if (isArm)
          os << "    add sp, sp, #" << frameBytes << "\n";
        else
          os << "    addi sp, sp, " << frameBytes << "\n";
      }
      os << "    ret\n";
      stats.returns++;
      continue;
    }

    if (opname == "affine.for") {
      int loopId = ++nextLoopId;
      loopOps[&op] = loopId;
      std::string start = ensureReg(op.operand(0), scratchReg(0));
      std::string boundTmp = ensureReg(op.operand(1), scratchReg(1));
      std::string iv = isArm ? ("w" + std::to_string(nextReg++ % 5 + 19))
                             : ("s" + std::to_string(nextReg++ % 12));
      std::string bound = isArm ? ("w" + std::to_string(nextReg++ % 5 + 24))
                                : ("s" + std::to_string(nextReg++ % 12));

      Value ivValue = op.getRegions()[0]->getBlocks()[0]->args()[0]->value();
      bindReg(ivValue, iv);

      if (isArm) {
        os << "    mov " << iv << ", " << start << "\n";
        os << "    mov " << bound << ", " << boundTmp << "\n";
        spillHome(ivValue, iv);
        os << ".Lloop_cond_" << loopId << ":\n";
        os << "    cmp " << iv << ", " << bound << "\n";
        os << "    bge .Lloop_end_" << loopId << "\n";
      } else {
        os << "    mv " << iv << ", " << start << "\n";
        os << "    mv " << bound << ", " << boundTmp << "\n";
        spillHome(ivValue, iv);
        os << ".Lloop_cond_" << loopId << ":\n";
        os << "    bge " << iv << ", " << bound << ", .Lloop_end_" << loopId << "\n";
      }
      continue;
    }

    if (opname == "scf.while") {
      int loopId = ++nextLoopId;
      loopOps[&op] = loopId;
      os << ".Lwhile_cond_" << loopId << ":\n";
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
      std::string step = ensureReg(parentFor->operand(2), scratchReg(1));
      Value ivValue = parentFor->getRegions()[0]->getBlocks()[0]->args()[0]->value();
      std::string iv = ensureReg(ivValue, scratchReg(0));

      if (isArm) {
        os << "    add " << iv << ", " << iv << ", " << step << "\n";
        bindReg(ivValue, iv);
        spillHome(ivValue, iv);
        os << "    b .Lloop_cond_" << loopId << "\n";
        os << ".Lloop_end_" << loopId << ":\n";
      } else {
        os << "    addw " << iv << ", " << iv << ", " << step << "\n";
        bindReg(ivValue, iv);
        spillHome(ivValue, iv);
        os << "    j .Lloop_cond_" << loopId << "\n";
        os << ".Lloop_end_" << loopId << ":\n";
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
           << (isContinue ? "cond_" : "end_") << loopId << "\n";
      } else {
        os << "    " << (isArm ? "b" : "j") << " .Lwhile_"
           << (isContinue ? "cond_" : "end_") << loopId << "\n";
      }
      continue;
    }

    if (opname == "rv_machine.cmp" || opname == "arm_machine.cmp" ||
        opname == "rv_machine.fcmp" || opname == "arm_machine.fcmp") {
      std::string lhs = ensureReg(op.operand(0), isFloatType(op.operand(0).type()) ? (isArm ? "s30" : "ft10") : scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), isFloatType(op.operand(1).type()) ? (isArm ? "s31" : "ft11") : scratchReg(1));
      std::string dst = resultReg();
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

    if (opname == "scf.if") {
      std::string cond = ensureReg(op.operand(0), scratchReg(0));
      static int nextIfId = 0;
      int ifId = ++nextIfId;
      loopOps[&op] = ifId;

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
          os << ".Lelse_" << ifId << ":\n";
        } else {
          os << ".Lendif_" << ifId << ":\n";
        }
        continue;
      }
      if (parent && parent->name() == "scf.while") {
        int loopId = loopOps[parent];
        os << "    " << (isArm ? "b" : "j") << " .Lwhile_cond_" << loopId << "\n";
        os << ".Lwhile_end_" << loopId << ":\n";
        continue;
      }
      continue;
    }

    stats.unsupportedOps++;
    stats.error = "unsupported native asm op: " + opname;
    return false;
  }

  if (stats.returns == returnsBefore) {
    for (int i = 0; i < calleeSaveCount; i++) {
      int64_t off = calleeSaveBase + i * 8;
      if (isArm)
        os << "    ldr x" << (19 + i) << ", [sp, #" << off << "]\n";
      else
        os << "    ld s" << i << ", " << off << "(sp)\n";
    }
    if (frameBytes > 0) {
      if (isArm)
        os << "    add sp, sp, #" << frameBytes << "\n";
      else
        os << "    addi sp, sp, " << frameBytes << "\n";
    }
    os << "    ret\n";
    stats.returns++;
  }
  return true;
}

} // namespace

bool emitNativeAssembly(Module &module, const std::string &target, std::ostream &os,
                        NativeAsmStats &stats) {
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
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
      continue;
    std::string name = symbolAttr(op->attr("symbol"));
    if (name.empty())
      name = symbolAttr(op->attr("sym_name"));
    std::string label = ".Lglob_" + sanitizeLabel(name);
    globalLabels[valueKey(op->result())] = label;
  }

  if (!globalLabels.empty()) {
    os << "    .bss\n";
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
        continue;
      auto it = globalLabels.find(valueKey(op->result()));
      if (it == globalLabels.end())
        continue;
      os << "    .align 3\n" << it->second << ":\n";
      os << "    .zero " << memrefAllocationBytes(op->resultType()) << "\n";
    }
  }

  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.func") {
      stats.functions++;
      if (!emitFunctionAssembly(*op, target, os, stats, globalLabels))
        return false;
    }
  }
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

void runGlobalOpt(Module &module, SelfOptStats *stats) {
  std::vector<Operation*> globals;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "sysy.global") {
      globals.push_back(op);
    }
  }

  for (auto *global : globals) {
    std::string gName = symbolAttr(global->attr("symbol"));
    if (gName.empty())
      gName = symbolAttr(global->attr("sym_name"));
    if (gName.empty())
      continue;

    std::vector<Operation*> accesses;
    std::set<Operation*> functions;
    for (auto *op : walk(module)) {
      if (!op || op->isErased())
        continue;
      if (op->name() == "sysy.load" || op->name() == "sysy.store" ||
          op->name() == "memref.load" || op->name() == "memref.store") {
        if (symbolAttr(op->attr("symbol")) == gName) {
          accesses.push_back(op);
          Block *currBlock = op->getBlock();
          while (currBlock) {
            Region *r = currBlock->getRegion();
            if (!r)
              break;
            Operation *parentOp = r->getParent();
            if (!parentOp)
              break;
            if (parentOp->name() == "sysy.func") {
              functions.insert(parentOp);
              break;
            }
            currBlock = parentOp->getBlock();
          }
        }
      }
    }

    if (accesses.empty()) {
      if (global->resultCount() == 0 || usesOf(module, global->result()).empty()) {
        global->markErased();
        if (stats)
          stats->globalsErased++;
      }
      continue;
    }

    if (functions.size() == 1) {
      Operation *func = *functions.begin();
      auto &regions = func->getRegions();
      if (!regions.empty() && !regions[0]->getBlocks().empty()) {
        auto &entry = *regions[0]->getBlocks()[0];
        Type storageType = global->resultType();
        auto &slot = entry.insertOperation(0, std::make_unique<Operation>(
            "sysy.alloca", std::vector<Value>{}, std::vector<Type>{storageType},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(gName)}},
            global->loc()));

        for (auto *op : accesses) {
          if (op->operandCount() > 0) {
            op->setOperand(0, slot.result());
          } else {
            op->addOperand(slot.result());
          }
        }
        if (stats)
          stats->globalsPromoted++;
        if (global->resultCount() == 0 || usesOf(module, global->result()).empty()) {
          global->markErased();
          if (stats)
            stats->globalsErased++;
        }
      }
    }
  }
}

static std::string memoryLocationKey(Operation &op) {
  std::string sym = symbolAttr(op.attr("symbol"));
  if (sym.empty())
    sym = symbolAttr(op.attr("sym_name"));
  if (sym.empty())
    return "";
  std::string key = sym;
  if (op.name() == "memref.load") {
    for (int i = 1; i < op.operandCount(); i++) {
      key += "," + op.operand(i).printName();
    }
  } else if (op.name() == "memref.store") {
    for (int i = 2; i < op.operandCount(); i++) {
      key += "," + op.operand(i).printName();
    }
  }
  return key;
}

static void runBlockMemoryOpt(Module &module, Block &block, SelfOptStats *stats) {
  if (stats)
    stats->memoryBlocks++;
  std::map<std::string, Value> activeStores;
  std::map<std::string, Operation*> activeStoreOps;

  for (auto &owned : block.ops()) {
    auto &op = *owned;
    if (op.isErased())
      continue;

    if (!op.getRegions().empty()) {
      activeStores.clear();
      activeStoreOps.clear();
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
      continue;
    }

    if (op.name() == "sysy.store" || op.name() == "memref.store") {
      std::string key = memoryLocationKey(op);
      if (key.empty())
        continue;

      std::string baseSym = symbolAttr(op.attr("symbol"));
      if (baseSym.empty())
        baseSym = symbolAttr(op.attr("sym_name"));
      std::vector<std::string> keysToInvalidate;
      for (const auto &kv : activeStores) {
        if (kv.first != key && !baseSym.empty() && kv.first.find(baseSym) == 0) {
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

void runMemoryOpt(Module &module, SelfOptStats *stats) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_MEMOPT");
  if (enabled && std::string(enabled) == "0")
    return;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "sysy.func") {
      for (auto &region : op->getRegions()) {
        runMemoryOptInRegion(module, *region, stats);
      }
    }
  }
  eraseMarked(module);
}

namespace {

enum class ProvenBitwiseKind {
  None,
  And,
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
    auto *load = maybeBit.getDefiningOp();
    return isConst(maybeOne, 1) && opHasName(load, {"sysy.load", "memref.load"}) &&
           load->operandCount() > 0 && bitSlots.count(valueKey(load->operand(0))) != 0;
  };
  return matches(cmp->operand(0), cmp->operand(1)) ||
         matches(cmp->operand(1), cmp->operand(0));
}

static bool isNeBetweenBitLoads(Value value, const std::set<std::string> &bitSlots) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "ne")
    return false;
  for (int i = 0; i < 2; i++) {
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

static bool valueStaticallyNonNegative(Value value) {
  int64_t c = 0;
  if (constantIntegerValue(value, c))
    return c >= 0;
  auto *op = value.getDefiningOp();
  if (!op)
    return false;
  if (opHasName(op, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp",
                     "arith.noti", "rv_machine.seqz", "arm_machine.not"}))
    return true;
  if (opHasName(op, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
      op->operandCount() == 2) {
    int64_t mask = 0;
    return (constantIntegerValue(op->operand(0), mask) && mask >= 0) ||
           (constantIntegerValue(op->operand(1), mask) && mask >= 0);
  }
  return false;
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
  std::set<std::string> paramsWithRem;
  std::set<std::string> paramsWithDiv;
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
          paramsWithRem.insert(paramKey);
        }
        if (isStoreToSlot(op, paramSlot) &&
            isBinaryWithConst(op->operand(0), "arith.divi", "rv_machine.divw",
                              "arm_machine.sdiv", paramSlot, 2))
          paramsWithDiv.insert(paramKey);
      }
    }
    if (op->name() == "scf.if" && op->operandCount() == 1 &&
        op->getRegions().size() == 1 && !op->getRegions()[0]->getBlocks().empty()) {
      auto *condOp = op->operand(0).getDefiningOp();
      if (opHasName(condOp, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
          condOp->operandCount() == 2 &&
          isEqBitToOne(condOp->operand(0), bitSlots) &&
          isEqBitToOne(condOp->operand(1), bitSlots)) {
        condKind = ProvenBitwiseKind::And;
      } else if (isNeBetweenBitLoads(op->operand(0), bitSlots)) {
        condKind = ProvenBitwiseKind::Xor;
      }
      Block &thenBlock = *op->getRegions()[0]->getBlocks()[0];
      for (auto &thenOp : thenBlock.ops())
        if (thenOp && isResultPlusPowerStore(thenOp.get(), resultSlot, powerSlot))
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

  bool lhsNonNegative = valueStaticallyNonNegative(call.operand(0));
  bool rhsNonNegative = valueStaticallyNonNegative(call.operand(1));
  if (lhsNonNegative && rhsNonNegative) {
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
          activeValNames.insert(storeOp->operand(0).printName());
        }
      }

      for (int i = (int) body->ops().size() - 1; i >= 0; i--) {
        auto *child = body->ops()[i].get();
        if (!child || child->isErased())
          continue;
        if (child->resultCount() > 0) {
          std::string resName = child->result().printName();
          if (activeValNames.count(resName) > 0) {
            if (child->name() == "memref.load") {
              loads.push_back(child);
            } else if (child->name() == "arith.addi" || child->name() == "arith.addf" ||
                       child->name() == "arith.subi" || child->name() == "arith.subf" ||
                       child->name() == "rv_machine.addw" || child->name() == "arm_machine.add" ||
                       child->name() == "arith.muli" || child->name() == "arith.mulf") {
              ariths.push_back(child);
              for (int opIdx = 0; opIdx < child->operandCount(); opIdx++) {
                activeValNames.insert(child->operand(opIdx).printName());
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
              if (storeOp->operand(2).printName() != loadOp->operand(1).printName()) {
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
        vecLoads[loadOp->result().printName()] = vread.result();
      }

      std::map<std::string, Value> vecAriths;
      for (auto *arithOp : ariths) {
        std::vector<Value> vecOperands;
        for (int i = 0; i < arithOp->operandCount(); i++) {
          Value operand = arithOp->operand(i);
          if (vecLoads.count(operand.printName())) {
            vecOperands.push_back(vecLoads[operand.printName()]);
          } else if (vecAriths.count(operand.printName())) {
            vecOperands.push_back(vecAriths[operand.printName()]);
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
        vecAriths[arithOp->result().printName()] = varith.result();
      }

      for (auto *storeOp : stores) {
        Value val = storeOp->operand(0);
        Value base = storeOp->operand(1);
        Value index = storeOp->operand(2);
        std::string storeSym = symbolAttr(storeOp->attr("symbol"));
        if (storeSym.empty())
          storeSym = symbolAttr(storeOp->attr("sym_name"));

        Value vecVal;
        if (vecLoads.count(val.printName())) {
          vecVal = vecLoads[val.printName()];
        } else if (vecAriths.count(val.printName())) {
          vecVal = vecAriths[val.printName()];
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
