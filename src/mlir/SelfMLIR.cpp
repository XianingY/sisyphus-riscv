#include "SelfMLIR.h"

#include "../hir/HIROps.h"
#include "../parse/ASTNode.h"
#include "../parse/Type.h"
#include "../utils/DynamicCast.h"

#include <algorithm>
#include <cctype>
#include <iostream>
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

BlockArgument::BlockArgument(Block *owner, unsigned index, Type type,
                             Location loc, std::string name):
  owner(owner), index(index), argType(type), argLoc(loc), argName(std::move(name)) {
  if (argName.empty())
    argName = "arg" + std::to_string(index);
}

Value BlockArgument::value() {
  return Value::blockArgument(this);
}

Operation::Operation(std::string name, std::vector<Value> operands,
                     std::vector<Type> results,
                     std::map<std::string, Attribute> attrs,
                     Location loc):
  opName(std::move(name)), operands(std::move(operands)),
  resultTypes(std::move(results)), attributes(std::move(attrs)), opLoc(loc) {}

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

void replaceAllUses(Module &module, Value oldValue, Value newValue) {
  for (auto *op : walk(module)) {
    for (int i = 0; i < op->operandCount(); i++)
      if (op->operand(i) == oldValue)
        op->setOperand(i, newValue);
  }
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
  return false;
}

static void eraseMarked(Module &module) {
  for (auto *op : walk(module))
    for (auto &region : op->getRegions())
      for (auto &block : region->getBlocks())
        block->eraseMarkedOperations();
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

namespace {

Type mapHIRType(Context &ctx, sys::hir::TypeKind type) {
  switch (type) {
  case sys::hir::TypeKind::Int:
    return ctx.i(32);
  case sys::hir::TypeKind::Float:
    return ctx.f(32);
  case sys::hir::TypeKind::Void:
    return ctx.noneType();
  case sys::hir::TypeKind::Pointer:
  case sys::hir::TypeKind::Array:
    return ctx.memref({-1}, ctx.i(32));
  case sys::hir::TypeKind::Function:
    return ctx.function({}, {});
  case sys::hir::TypeKind::Unknown:
    return ctx.i(32);
  }
  return ctx.i(32);
}

std::string arithOpName(const sys::hir::Op *op) {
  const bool isFloat = op && op->type == sys::hir::TypeKind::Float;
  const std::string sym = op ? op->symbol : "";
  if (sym == "+") return isFloat ? "arith.addf" : "arith.addi";
  if (sym == "-") {
    if (op && op->children.size() == 1)
      return isFloat ? "arith.negf" : "arith.negi";
    return isFloat ? "arith.subf" : "arith.subi";
  }
  if (sym == "*") return isFloat ? "arith.mulf" : "arith.muli";
  if (sym == "/") return isFloat ? "arith.divf" : "arith.divi";
  if (sym == "%") return "arith.remi";
  if (sym == "&&") return "arith.andi";
  if (sym == "||") return "arith.ori";
  if (sym == "!") return "arith.noti";
  if (sym == "i2f") return "arith.sitofp";
  if (sym == "f2i") return "arith.fptosi";
  return isFloat ? "arith.unknownf" : "arith.unknowni";
}

std::string cmpPredicate(const std::string &sym) {
  if (sym == "==") return "eq";
  if (sym == "!=") return "ne";
  if (sym == "<=") return "sle";
  if (sym == "<") return "slt";
  return sym.empty() ? "unknown" : sym;
}

Location locFor(Context &ctx, const sys::hir::Op *op) {
  // The current HIR does not retain source spans, so every production import
  // still receives an explicit location object instead of a missing loc.
  if (!op || !op->origin)
    return ctx.unknownLoc();
  return ctx.loc("sysy", 0, 0);
}

int countHIR(const sys::hir::Op *op) {
  if (!op)
    return 0;
  int total = 1;
  for (const auto &child : op->children)
    total += countHIR(child.get());
  return total;
}

class HIRImporter {
  Context &ctx;

  static bool blockEndsWithTerminator(const Block &block) {
    return !block.ops().empty() && block.ops().back()->isTerminator();
  }

  Value materializeUnknown(Builder &builder, const sys::hir::Op *op) {
    auto type = mapHIRType(ctx, op ? op->type : sys::hir::TypeKind::Int);
    auto &unknown = builder.create("sysy.unknown_value", {}, {type}, {},
                                   locFor(ctx, op));
    return unknown.result();
  }

  Value emitExpr(const sys::hir::Op *op, Builder &builder) {
    if (!op)
      return materializeUnknown(builder, op);

    switch (op->kind) {
    case sys::hir::OpKind::ConstInt: {
      auto &constant = builder.create(
          "arith.constant", {}, {ctx.i(32)},
          {{"value", ctx.integerAttr(op->hasIntValue ? op->intValue : 0, ctx.i(32))}},
          locFor(ctx, op));
      return constant.result();
    }
    case sys::hir::OpKind::ConstFloat: {
      auto &constant = builder.create(
          "arith.constant", {}, {ctx.f(32)},
          {{"value", ctx.stringAttr(op->hasFloatValue ? std::to_string(op->floatValue) : "0.0")}},
          locFor(ctx, op));
      return constant.result();
    }
    case sys::hir::OpKind::Load: {
      std::vector<Value> indices;
      for (const auto &child : op->children)
        indices.push_back(emitExpr(child.get(), builder));
      auto attrs = std::map<std::string, Attribute>{{"symbol", ctx.stringAttr(op->symbol)}};
      auto &load = builder.create("memref.load", indices, {mapHIRType(ctx, op->type)},
                                  attrs, locFor(ctx, op));
      return load.result();
    }
    case sys::hir::OpKind::Arith: {
      std::vector<Value> operands;
      for (const auto &child : op->children)
        operands.push_back(emitExpr(child.get(), builder));
      auto &arith = builder.create(arithOpName(op), operands,
                                   {mapHIRType(ctx, op->type)}, {},
                                   locFor(ctx, op));
      return arith.result();
    }
    case sys::hir::OpKind::Cmp: {
      std::vector<Value> operands;
      for (const auto &child : op->children)
        operands.push_back(emitExpr(child.get(), builder));
      auto &cmp = builder.create(
          "arith.cmpi", operands, {ctx.i(32)},
          {{"predicate", ctx.stringAttr(cmpPredicate(op->symbol))}},
          locFor(ctx, op));
      return cmp.result();
    }
    case sys::hir::OpKind::Call: {
      std::vector<Value> operands;
      for (const auto &child : op->children)
        operands.push_back(emitExpr(child.get(), builder));
      std::vector<Type> results;
      if (op->type != sys::hir::TypeKind::Void)
        results.push_back(mapHIRType(ctx, op->type));
      auto &call = builder.create("sysy.call", operands, results,
                                  {{"callee", ctx.stringAttr(op->symbol)}},
                                  locFor(ctx, op));
      return call.resultCount() ? call.result() : materializeUnknown(builder, op);
    }
    default:
      emitStmt(op, builder);
      return materializeUnknown(builder, op);
    }
  }

  void emitRegionFromBlock(const sys::hir::Op *op, Region &region) {
    auto &block = region.addBlock();
    Builder nested(ctx, &block);
    if (op && op->kind == sys::hir::OpKind::Block) {
      for (const auto &child : op->children)
        emitStmt(child.get(), nested);
    } else if (op) {
      emitStmt(op, nested);
    }
    if (!blockEndsWithTerminator(block))
      nested.create("scf.yield", {}, {}, {}, locFor(ctx, op));
  }

  void emitFunc(const sys::hir::Op *op, Builder &builder) {
    auto fnType = ctx.function({}, op && op->type != sys::hir::TypeKind::Void
                                     ? std::vector<Type>{mapHIRType(ctx, op->type)}
                                     : std::vector<Type>{});
    auto &func = builder.create(
        "sysy.func", {}, {},
        {{"sym_name", ctx.stringAttr(op ? op->symbol : "")},
         {"type", ctx.stringAttr(fnType.str())}},
        locFor(ctx, op), 1);
    auto &entry = func.getRegions()[0]->addBlock();
    if (op && op->origin) {
      if (auto *fn = dyn_cast<FnDeclNode>(op->origin)) {
        for (const auto &argName : fn->args)
          entry.addArgument(ctx.i(32), locFor(ctx, op), argName);
      }
    }
    Builder bodyBuilder(ctx, &entry);
    if (op && !op->children.empty())
      emitStmt(op->children[0].get(), bodyBuilder);
    if (!blockEndsWithTerminator(entry))
      bodyBuilder.create("scf.yield", {}, {}, {}, locFor(ctx, op));
  }

public:
  explicit HIRImporter(Context &ctx): ctx(ctx) {}

  void emitStmt(const sys::hir::Op *op, Builder &builder) {
    if (!op)
      return;

    switch (op->kind) {
    case sys::hir::OpKind::Module:
      for (const auto &child : op->children)
        emitStmt(child.get(), builder);
      return;
    case sys::hir::OpKind::Func:
      emitFunc(op, builder);
      return;
    case sys::hir::OpKind::Block:
      for (const auto &child : op->children)
        emitStmt(child.get(), builder);
      return;
    case sys::hir::OpKind::VarDecl: {
      std::vector<Value> operands;
      if (!op->children.empty() && op->children[0])
        operands.push_back(emitExpr(op->children[0].get(), builder));
      std::map<std::string, Attribute> attrs{{"symbol", ctx.stringAttr(op->symbol)}};
      if (!op->arrayDims.empty()) {
        std::string dims;
        for (size_t i = 0; i < op->arrayDims.size(); i++) {
          if (i)
            dims += "x";
          dims += std::to_string(op->arrayDims[i]);
        }
        attrs["shape"] = ctx.stringAttr(dims);
      }
      builder.create("memref.alloca", operands,
                     {mapHIRType(ctx, op->type)}, attrs, locFor(ctx, op));
      return;
    }
    case sys::hir::OpKind::Store: {
      std::vector<Value> operands;
      for (const auto &child : op->children)
        operands.push_back(emitExpr(child.get(), builder));
      builder.create("memref.store", operands, {},
                     {{"symbol", ctx.stringAttr(op->symbol)}}, locFor(ctx, op));
      return;
    }
    case sys::hir::OpKind::If: {
      Value cond = op->children.empty() ? materializeUnknown(builder, op)
                                        : emitExpr(op->children[0].get(), builder);
      auto &ifOp = builder.create("scf.if", {cond}, {}, {}, locFor(ctx, op),
                                  op->children.size() >= 3 ? 2 : 1);
      if (op->children.size() >= 2)
        emitRegionFromBlock(op->children[1].get(), *ifOp.getRegions()[0]);
      if (op->children.size() >= 3)
        emitRegionFromBlock(op->children[2].get(), *ifOp.getRegions()[1]);
      return;
    }
    case sys::hir::OpKind::While: {
      Value cond = op->children.empty() ? materializeUnknown(builder, op)
                                        : emitExpr(op->children[0].get(), builder);
      auto &whileOp = builder.create("scf.while", {cond}, {}, {}, locFor(ctx, op), 1);
      if (op->children.size() >= 2)
        emitRegionFromBlock(op->children[1].get(), *whileOp.getRegions()[0]);
      return;
    }
    case sys::hir::OpKind::For: {
      std::vector<Value> operands;
      for (size_t i = 0; i < op->children.size() && i < 3; i++)
        operands.push_back(emitExpr(op->children[i].get(), builder));
      auto &forOp = builder.create("scf.for", operands, {}, {}, locFor(ctx, op), 1);
      if (op->children.size() >= 4)
        emitRegionFromBlock(op->children[3].get(), *forOp.getRegions()[0]);
      return;
    }
    case sys::hir::OpKind::Return: {
      std::vector<Value> operands;
      for (const auto &child : op->children)
        if (child)
          operands.push_back(emitExpr(child.get(), builder));
      builder.create("sysy.return", operands, {}, {}, locFor(ctx, op));
      return;
    }
    case sys::hir::OpKind::Break:
      builder.create("sysy.break", {}, {}, {}, locFor(ctx, op));
      return;
    case sys::hir::OpKind::Continue:
      builder.create("sysy.continue", {}, {}, {}, locFor(ctx, op));
      return;
    case sys::hir::OpKind::ConstInt:
    case sys::hir::OpKind::ConstFloat:
    case sys::hir::OpKind::Load:
    case sys::hir::OpKind::Arith:
    case sys::hir::OpKind::Cmp:
    case sys::hir::OpKind::Call:
      (void) emitExpr(op, builder);
      return;
    case sys::hir::OpKind::Unknown:
      builder.create("sysy.unknown", {}, {}, {}, locFor(ctx, op));
      return;
    }
  }
};

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

} // namespace

static const char *kCoreRules =
  "rule fold_addi_zero arith.addi addi-zero 10\n"
  "rule fold_muli_one arith.muli muli-one 9\n"
  "rule fold_select_same arith.select select-same 8\n";

std::unique_ptr<Module> lowerFromHIR(Context &ctx, const sys::hir::Module &hirModule,
                                     const std::string &target,
                                     ProductionStats *stats) {
  if (stats)
    stats->hirOps = countHIR(hirModule.root.get());
  auto module = std::make_unique<Module>(ctx);
  auto &top = module->body().getBlocks()[0];
  Builder builder(ctx, top.get());
  (void) target;
  HIRImporter importer(ctx);
  if (hirModule.root)
    importer.emitStmt(hirModule.root.get(), builder);
  if (stats)
    stats->mlirOpsBefore = (int) walk(*module).size();
  return module;
}

bool runProductionGate(const sys::hir::Module &hirModule, const std::string &target,
                       ProductionStats &stats, std::ostream *dump) {
  stats = ProductionStats();
  Context ctx;
  auto module = lowerFromHIR(ctx, hirModule, target, &stats);
  auto before = verify(*module);
  stats.verifyBefore = before.ok;
  if (!before.ok) {
    stats.error = before.errors.empty() ? "self-MLIR verification failed before canonicalization"
                                        : before.errors.front();
    if (dump)
      print(*module, *dump);
    return false;
  }

  std::vector<std::string> parseErrors;
  auto rules = parseDRR(kCoreRules, parseErrors);
  if (!parseErrors.empty()) {
    stats.error = parseErrors.front();
    return false;
  }
  auto rewriteStats = applyGreedyPatterns(*module, rules);
  stats.rewrites = rewriteStats.rewrites;

  auto after = verify(*module);
  stats.verifyAfter = after.ok;
  if (!after.ok) {
    stats.error = after.errors.empty() ? "self-MLIR verification failed after canonicalization"
                                       : after.errors.front();
    if (dump)
      print(*module, *dump);
    return false;
  }

  auto convStats = convertDialects(*module, productionTarget(target), targetPatterns(target));
  stats.conversionVisited = convStats.visited;
  stats.conversionLegal = convStats.legal;
  stats.conversionConverted = convStats.converted;
  stats.conversionFailed = convStats.failed;
  stats.conversionRollbacks = convStats.rollbacks;
  stats.mlirOpsAfter = (int) walk(*module).size();
  if (dump)
    print(*module, *dump);
  if (convStats.failed) {
    stats.error = "self-MLIR dialect conversion failed for target " + target;
    return false;
  }
  return true;
}

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

int runCoreSelfTest(std::ostream &os) {
  Context ctx;
  auto i32a = ctx.i(32);
  auto i32b = ctx.i(32);
  auto locA = ctx.loc("sample.sy", 1, 1);
  auto locB = ctx.loc("sample.sy", 1, 1);
  bool uniqued = i32a == i32b && locA == locB;
  Module module = buildSample(ctx);
  auto before = verify(module);
  std::vector<std::string> parseErrors;
  auto rules = parseDRR(kCoreRules, parseErrors);
  auto stats = applyGreedyPatterns(module, rules);
  auto after = verify(module);
  os << "[self-mlir-core] uniqued=" << (uniqued ? 1 : 0)
     << " types=" << ctx.typeCount()
     << " attrs=" << ctx.attrCount()
     << " locs=" << ctx.locationCount()
     << " block-args=1"
     << " rules=" << stats.rules
     << " rewrites=" << stats.rewrites
     << " verify-before=" << (before.ok ? 1 : 0)
     << " verify-after=" << (after.ok ? 1 : 0)
     << " parse-errors=" << parseErrors.size() << "\n";
  print(module, os);
  return uniqued && before.ok && after.ok && parseErrors.empty() && stats.rewrites == 1 ? 0 : 1;
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

void dumpSample(std::ostream &os) {
  Context ctx;
  Module module = buildSample(ctx);
  print(module, os);
}

} // namespace sys::mlir
