#include "SelfMLIR.h"

#include <algorithm>
#include <cstdint>
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
  for (const auto &uses : resultUses) {
    for (const auto &use : uses) {
      if (use.owner && !use.owner->isErased())
        return;
    }
  }
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
  return opName == "affine.yield" ||
         opName == "scf.yield" || opName == "scf.return" ||
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
                                    return op->isErased() && op->resultCount() == 0 &&
                                           op->getRegions().empty();
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
  if (op.isErased())
    return;
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
  if (op.isErased())
    return;
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
      size_t lastLiveOp = block->ops().size();
      for (size_t i = block->ops().size(); i > 0; i--) {
        Operation *candidate = block->ops()[i - 1].get();
        if (candidate && !candidate->isErased()) {
          lastLiveOp = i - 1;
          break;
        }
      }
      for (size_t i = 0; i < block->ops().size(); i++) {
        auto &child = *block->ops()[i];
        if (child.isErased())
          continue;
        if (child.getBlock() != block.get())
          fail("child block owner mismatch");
        if (child.isTerminator() && i != lastLiveOp)
          fail("terminator is not the final block operation");
        if (!child.isTerminator() && i == lastLiveOp &&
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




} // namespace sys::mlir
