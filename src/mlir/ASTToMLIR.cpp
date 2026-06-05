#include "ASTToMLIR.h"
#include "Polyhedral.h"
#include "IPO.h"

#include "../parse/ASTNode.h"
#include "../parse/Type.h"
#include "../utils/DynamicCast.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

namespace sys::mlir {
namespace {

int countAST(const sys::ASTNode *node) {
  if (!node)
    return 0;
  int total = 1;
  if (auto *block = dyn_cast<sys::BlockNode>(const_cast<sys::ASTNode*>(node))) {
    for (auto *child : block->nodes)
      total += countAST(child);
  } else if (auto *transparent = dyn_cast<sys::TransparentBlockNode>(const_cast<sys::ASTNode*>(node))) {
    for (auto *child : transparent->nodes)
      total += countAST(child);
  } else if (auto *binary = dyn_cast<sys::BinaryNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(binary->l) + countAST(binary->r);
  } else if (auto *unary = dyn_cast<sys::UnaryNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(unary->node);
  } else if (auto *func = dyn_cast<sys::FnDeclNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(func->body);
  } else if (auto *ret = dyn_cast<sys::ReturnNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(ret->node);
  } else if (auto *ifn = dyn_cast<sys::IfNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(ifn->cond) + countAST(ifn->ifso) + countAST(ifn->ifnot);
  } else if (auto *assign = dyn_cast<sys::AssignNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(assign->l) + countAST(assign->r);
  } else if (auto *loop = dyn_cast<sys::WhileNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(loop->cond) + countAST(loop->body);
  } else if (auto *access = dyn_cast<sys::ArrayAccessNode>(const_cast<sys::ASTNode*>(node))) {
    for (auto *idx : access->indices)
      total += countAST(idx);
  } else if (auto *write = dyn_cast<sys::ArrayAssignNode>(const_cast<sys::ASTNode*>(node))) {
    for (auto *idx : write->indices)
      total += countAST(idx);
    total += countAST(write->value);
  } else if (auto *call = dyn_cast<sys::CallNode>(const_cast<sys::ASTNode*>(node))) {
    for (auto *arg : call->args)
      total += countAST(arg);
  } else if (auto *decl = dyn_cast<sys::VarDeclNode>(const_cast<sys::ASTNode*>(node))) {
    total += countAST(decl->init);
  }
  return total;
}

Type scalarType(Context &ctx, sys::Type *type) {
  if (isa<sys::FloatType>(type))
    return ctx.f(32);
  if (isa<sys::VoidType>(type))
    return ctx.noneType();
  if (auto *ptr = dyn_cast<sys::PointerType>(type)) {
    if (auto *arr = dyn_cast<sys::ArrayType>(ptr->pointee)) {
      std::vector<int64_t> shape{-1};
      for (int dim : arr->dims)
        shape.push_back(dim);
      return ctx.memref(shape, scalarType(ctx, arr->base));
    }
    return ctx.memref({-1}, scalarType(ctx, ptr->pointee));
  }
  if (auto *arr = dyn_cast<sys::ArrayType>(type))
    return scalarType(ctx, arr->base);
  return ctx.i(32);
}

Type storageType(Context &ctx, sys::Type *type) {
  if (auto *arr = dyn_cast<sys::ArrayType>(type)) {
    std::vector<int64_t> shape;
    for (int dim : arr->dims)
      shape.push_back(dim);
    return ctx.memref(shape, scalarType(ctx, arr->base));
  }
  if (auto *ptr = dyn_cast<sys::PointerType>(type)) {
    if (auto *arr = dyn_cast<sys::ArrayType>(ptr->pointee)) {
      std::vector<int64_t> shape{-1};
      for (int dim : arr->dims)
        shape.push_back(dim);
      return ctx.memref(shape, scalarType(ctx, arr->base));
    }
    return ctx.memref({-1}, scalarType(ctx, ptr->pointee));
  }
  return ctx.memref({1}, scalarType(ctx, type));
}

std::string typeText(Context &ctx, sys::Type *type) {
  if (auto *fn = dyn_cast<sys::FunctionType>(type)) {
    std::vector<Type> inputs;
    for (auto *param : fn->params)
      inputs.push_back(isa<sys::ArrayType>(param) || isa<sys::PointerType>(param)
                           ? storageType(ctx, param)
                           : scalarType(ctx, param));
    std::vector<Type> results;
    if (!isa<sys::VoidType>(fn->ret))
      results.push_back(scalarType(ctx, fn->ret));
    return ctx.function(inputs, results).str();
  }
  return scalarType(ctx, type).str();
}

std::string binaryName(const sys::BinaryNode &node, bool isFloat) {
  switch (node.kind) {
  case sys::BinaryNode::Add: return isFloat ? "arith.addf" : "arith.addi";
  case sys::BinaryNode::Sub: return isFloat ? "arith.subf" : "arith.subi";
  case sys::BinaryNode::Mul: return isFloat ? "arith.mulf" : "arith.muli";
  case sys::BinaryNode::Div: return isFloat ? "arith.divf" : "arith.divi";
  case sys::BinaryNode::Mod: return "arith.remi";
  case sys::BinaryNode::And: return "arith.andi";
  case sys::BinaryNode::Or: return "arith.ori";
  case sys::BinaryNode::Eq:
  case sys::BinaryNode::Ne:
  case sys::BinaryNode::Le:
  case sys::BinaryNode::Lt: return isFloat ? "arith.cmpf" : "arith.cmpi";
  }
  return "arith.addi";
}

std::string predicateName(const sys::BinaryNode &node) {
  switch (node.kind) {
  case sys::BinaryNode::Eq: return "eq";
  case sys::BinaryNode::Ne: return "ne";
  case sys::BinaryNode::Le: return "le";
  case sys::BinaryNode::Lt: return "lt";
  default: return "";
  }
}

bool isCompare(const sys::BinaryNode &node) {
  return node.kind == sys::BinaryNode::Eq || node.kind == sys::BinaryNode::Ne ||
         node.kind == sys::BinaryNode::Le || node.kind == sys::BinaryNode::Lt;
}

bool envDisabled(const char *name) {
  const char *value = std::getenv(name);
  return value && std::string(value) == "0";
}

bool envEnabled(const char *name, bool defaultValue) {
  if (const char *value = std::getenv(name))
    return std::string(value) != "0";
  return defaultValue;
}

void summarizeModule(Module &module, ProductionStats &stats) {
  stats.mlirOpsAfter = 0;
  stats.affineLoops = 0;
  stats.scfLoops = 0;
  stats.memrefOps = 0;
  stats.loadOps = 0;
  stats.storeOps = 0;
  stats.callOps = 0;
  stats.machineDialectOps = 0;
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    stats.mlirOpsAfter++;
    const std::string &name = op->name();
    std::string dialect = op->dialect();
    if (name == "affine.for")
      stats.affineLoops++;
    if (name == "scf.for" || name == "scf.while")
      stats.scfLoops++;
    if (dialect == "memref")
      stats.memrefOps++;
    if (name == "sysy.load" || name == "memref.load")
      stats.loadOps++;
    if (name == "sysy.store" || name == "memref.store")
      stats.storeOps++;
    if (name == "sysy.call")
      stats.callOps++;
    if (dialect == "rv_machine" || dialect == "arm_machine")
      stats.machineDialectOps++;
  }
}

class ASTToMLIR {
  Context &ctx;
  std::vector<std::map<std::string, Value>> values;
  std::vector<std::map<std::string, Value>> storage;

  Location loc() { return ctx.loc("sysy", 0, 0); }

  void pushScope() {
    values.emplace_back();
    storage.emplace_back();
  }

  void popScope() {
    values.pop_back();
    storage.pop_back();
  }

  void bindValue(const std::string &name, Value value) {
    if (values.empty())
      pushScope();
    values.back()[name] = value;
  }

  void bindStorage(const std::string &name, Value value) {
    if (storage.empty())
      pushScope();
    storage.back()[name] = value;
  }

  Value lookupValue(const std::string &name) const {
    for (auto it = values.rbegin(); it != values.rend(); ++it) {
      auto found = it->find(name);
      if (found != it->end())
        return found->second;
    }
    return Value();
  }

  Value lookupStorage(const std::string &name) const {
    for (auto it = storage.rbegin(); it != storage.rend(); ++it) {
      auto found = it->find(name);
      if (found != it->end())
        return found->second;
    }
    return Value();
  }

  Value unknown(Builder &builder, sys::Type *type) {
    auto &op = builder.create("sysy.unknown_value", {}, {scalarType(ctx, type)}, {}, loc());
    return op.result();
  }

  Value emitIntConstant(int64_t value, Builder &builder) {
    auto &op = builder.create("arith.constant", {}, {ctx.i(32)},
                              {{"value", ctx.integerAttr(value, ctx.i(32))}}, loc());
    return op.result();
  }

  std::string floatAttrText(float value) {
    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return os.str();
  }

  std::string globalArrayInitWords(sys::ArrayType *arr, sys::ConstArrayNode *init) {
    if (!arr || !init)
      return "";
    std::size_t count = 1;
    for (int dim : arr->dims) {
      if (dim <= 0)
        return "";
      count *= static_cast<std::size_t>(dim);
    }
    if (count == 0 || (!init->isFloat && !init->vi) || (init->isFloat && !init->vf))
      return "";

    std::ostringstream os;
    bool anyNonZero = false;
    for (std::size_t i = 0; i < count; i++) {
      uint32_t bits = 0;
      if (init->isFloat) {
        static_assert(sizeof(float) == sizeof(uint32_t), "expected 32-bit float");
        std::memcpy(&bits, &init->vf[i], sizeof(bits));
      } else {
        bits = static_cast<uint32_t>(init->vi[i]);
      }
      anyNonZero = anyNonZero || bits != 0;
      if (i)
        os << ",";
      os << bits;
    }
    return anyNonZero ? os.str() : "";
  }

  Value emitZeroConstant(sys::Type *type, Builder &builder) {
    Type resultType = scalarType(ctx, type);
    Attribute value = isa<sys::FloatType>(type)
                          ? ctx.stringAttr("0.0")
                          : ctx.integerAttr(0, resultType);
    auto &op = builder.create("arith.constant", {}, {resultType},
                              {{"value", value}}, loc());
    return op.result();
  }

  std::vector<Value> emitArraySubscripts(std::size_t linear,
                                         const std::vector<int> &dims,
                                         Builder &builder) {
    std::vector<std::size_t> raw(dims.size(), 0);
    for (std::size_t i = dims.size(); i-- > 0;) {
      int dim = dims[i] <= 0 ? 1 : dims[i];
      raw[i] = linear % static_cast<std::size_t>(dim);
      linear /= static_cast<std::size_t>(dim);
    }

    std::vector<Value> indices;
    for (std::size_t index : raw)
      indices.push_back(emitIntConstant(static_cast<int64_t>(index), builder));
    return indices;
  }

  void emitZeroFillLoop(Value slot, sys::ArrayType *arr, Builder &builder,
                        std::size_t depth = 0,
                        std::vector<Value> indices = {}) {
    if (!arr)
      return;
    if (depth >= arr->dims.size()) {
      Value zero = emitZeroConstant(arr->base, builder);
      std::vector<Value> operands{zero, slot};
      operands.insert(operands.end(), indices.begin(), indices.end());
      builder.create("memref.store", operands, {}, {}, loc());
      return;
    }

    Value lower = emitIntConstant(0, builder);
    Value upper = emitIntConstant(arr->dims[depth], builder);
    Value step = emitIntConstant(1, builder);
    auto &loop = builder.create("affine.for", {lower, upper, step}, {},
                                {}, loc(), 1);
    auto &body = loop.getRegions()[0]->addBlock();
    auto &iv = body.addArgument(ctx.i(32), loc(), "iv");
    Builder nested(ctx, &body);
    indices.push_back(iv.value());
    emitZeroFillLoop(slot, arr, nested, depth + 1, indices);
    nested.create("affine.yield", {}, {}, {}, loc());
  }

  bool localArrayNeedsZeroFill(sys::LocalArrayNode *init, std::size_t count) const {
    if (!init || !init->va || init->count < count)
      return true;
    for (std::size_t i = 0; i < count; i++)
      if (!init->va[i])
        return true;
    return false;
  }

  void emitLocalArrayInit(sys::VarDeclNode *decl, sys::ArrayType *arr,
                          sys::LocalArrayNode *init, Value slot,
                          Builder &builder) {
    if (!decl || !arr || !init)
      return;
    std::size_t count = arr->getSize();
    bool zeroFilled = localArrayNeedsZeroFill(init, count);
    if (zeroFilled)
      emitZeroFillLoop(slot, arr, builder);

    std::size_t initCount = std::min<std::size_t>(count, init->count);
    for (std::size_t i = 0; i < initCount; i++) {
      if (zeroFilled && (!init->va || !init->va[i]))
        continue;
      Value value = (init->va && init->va[i])
                        ? emitExpr(init->va[i], builder)
                        : emitZeroConstant(arr->base, builder);
      std::vector<Value> operands{value, slot};
      auto indices = emitArraySubscripts(i, arr->dims, builder);
      operands.insert(operands.end(), indices.begin(), indices.end());
      builder.create("memref.store", operands, {},
                     {{"symbol", ctx.stringAttr(decl->name)}}, loc());
    }
  }

  Value emitBoolValue(Value input, Builder &builder) {
    bool fp = input.type().kind() == TypeKind::Float || input.type().str() == "f32";
    Type zeroType = fp ? ctx.f(32) : ctx.i(32);
    Attribute zeroAttr = fp ? ctx.stringAttr("0.0")
                            : ctx.integerAttr(0, ctx.i(32));
    auto &zero = builder.create("arith.constant", {}, {zeroType},
                                {{"value", zeroAttr}}, loc());
    auto &cmp = builder.create(fp ? "arith.cmpf" : "arith.cmpi",
                               {input, zero.result()}, {ctx.i(32)},
                               {{"predicate", ctx.stringAttr("ne")}}, loc());
    return cmp.result();
  }

  Value emitLogicalExpr(sys::BinaryNode *binary, Builder &builder) {
    auto &slot = builder.create("sysy.alloca", {}, {ctx.memref({1}, ctx.i(32))},
                                {{"symbol", ctx.stringAttr(".logic")}}, loc());
    Value lhs = emitBoolValue(emitExpr(binary->l, builder), builder);
    auto &branch = builder.create("scf.if", {lhs}, {}, {}, loc(), 2);

    auto emitConstStore = [&](Region &region, int value) {
      auto &block = region.addBlock();
      Builder nested(ctx, &block);
      Value constant = emitIntConstant(value, nested);
      nested.create("sysy.store", {constant, slot.result()}, {},
                    {{"symbol", ctx.stringAttr(".logic")}}, loc());
      nested.create("scf.yield", {}, {}, {}, loc());
    };
    auto emitRhsStore = [&](Region &region) {
      auto &block = region.addBlock();
      Builder nested(ctx, &block);
      Value rhs = emitBoolValue(emitExpr(binary->r, nested), nested);
      nested.create("sysy.store", {rhs, slot.result()}, {},
                    {{"symbol", ctx.stringAttr(".logic")}}, loc());
      nested.create("scf.yield", {}, {}, {}, loc());
    };

    if (binary->kind == sys::BinaryNode::And) {
      emitRhsStore(*branch.getRegions()[0]);
      emitConstStore(*branch.getRegions()[1], 0);
    } else {
      emitConstStore(*branch.getRegions()[0], 1);
      emitRhsStore(*branch.getRegions()[1]);
    }

    auto &load = builder.create("sysy.load", {slot.result()}, {ctx.i(32)},
                                {{"symbol", ctx.stringAttr(".logic")}}, loc());
    return load.result();
  }

  Value emitExpr(sys::ASTNode *node, Builder &builder) {
    if (!node)
      return unknown(builder, nullptr);
    if (auto *i = dyn_cast<sys::IntNode>(node)) {
      auto &op = builder.create("arith.constant", {}, {ctx.i(32)},
                                {{"value", ctx.integerAttr(i->value, ctx.i(32))}}, loc());
      return op.result();
    }
    if (auto *f = dyn_cast<sys::FloatNode>(node)) {
      auto &op = builder.create("arith.constant", {}, {ctx.f(32)},
                                {{"value", ctx.stringAttr(floatAttrText(f->value))}}, loc());
      return op.result();
    }
    if (auto *ref = dyn_cast<sys::VarRefNode>(node)) {
      Value direct = lookupValue(ref->name);
      if (direct.valid())
        return direct;
      Value slot = lookupStorage(ref->name);
      if (slot.valid() && (isa<sys::ArrayType>(node->type) || isa<sys::PointerType>(node->type)))
        return slot;
      if (slot.valid()) {
        auto &load = builder.create("sysy.load", {slot}, {scalarType(ctx, node->type)},
                                    {{"symbol", ctx.stringAttr(ref->name)}}, loc());
        return load.result();
      }
      auto &load = builder.create("sysy.load", {}, {scalarType(ctx, node->type)},
                                  {{"symbol", ctx.stringAttr(ref->name)}}, loc());
      return load.result();
    }
    if (auto *unary = dyn_cast<sys::UnaryNode>(node)) {
      Value input = emitExpr(unary->node, builder);
      if (unary->kind == sys::UnaryNode::Int2Float) {
        auto &op = builder.create("arith.sitofp", {input}, {ctx.f(32)}, {}, loc());
        return op.result();
      }
      if (unary->kind == sys::UnaryNode::Float2Int) {
        auto &op = builder.create("arith.fptosi", {input}, {ctx.i(32)}, {}, loc());
        return op.result();
      }
      if (unary->kind == sys::UnaryNode::Not) {
        Value truth = emitBoolValue(input, builder);
        Value zero = emitIntConstant(0, builder);
        auto &op = builder.create("arith.cmpi", {truth, zero}, {ctx.i(32)},
                                  {{"predicate", ctx.stringAttr("eq")}}, loc());
        return op.result();
      }
      auto &op = builder.create(isa<sys::FloatType>(node->type) ? "arith.negf" : "arith.negi",
                                {input}, {scalarType(ctx, node->type)}, {}, loc());
      return op.result();
    }
    if (auto *binary = dyn_cast<sys::BinaryNode>(node)) {
      if (binary->kind == sys::BinaryNode::And ||
          binary->kind == sys::BinaryNode::Or)
        return emitLogicalExpr(binary, builder);
      Value lhs = emitExpr(binary->l, builder);
      Value rhs = emitExpr(binary->r, builder);
      bool fp = isa<sys::FloatType>(binary->l ? binary->l->type : nullptr) ||
                isa<sys::FloatType>(binary->r ? binary->r->type : nullptr);
      std::map<std::string, Attribute> attrs;
      if (isCompare(*binary))
        attrs["predicate"] = ctx.stringAttr(predicateName(*binary));
      auto &op = builder.create(binaryName(*binary, fp), {lhs, rhs},
                                {isCompare(*binary) ? ctx.i(32) : scalarType(ctx, node->type)},
                                attrs, loc());
      return op.result();
    }
    if (auto *call = dyn_cast<sys::CallNode>(node)) {
      std::vector<Value> operands;
      for (auto *arg : call->args)
        operands.push_back(emitExpr(arg, builder));
      std::vector<Type> results;
      if (!isa<sys::VoidType>(node->type))
        results.push_back(scalarType(ctx, node->type));
      auto &op = builder.create("sysy.call", operands, results,
                                {{"callee", ctx.stringAttr(call->func)}}, loc());
      return op.resultCount() ? op.result() : unknown(builder, node->type);
    }
    if (auto *access = dyn_cast<sys::ArrayAccessNode>(node)) {
      std::vector<Value> operands;
      Value base = lookupStorage(access->array);
      if (base.valid())
        operands.push_back(base);
      for (auto *idx : access->indices)
        operands.push_back(emitExpr(idx, builder));
      auto &op = builder.create("memref.load", operands, {scalarType(ctx, node->type)},
                                {{"symbol", ctx.stringAttr(access->array)}}, loc());
      return op.result();
    }
    if (auto *write = dyn_cast<sys::ArrayAssignNode>(node)) {
      emitStmt(write, builder);
      return unknown(builder, node->type);
    }
    return unknown(builder, node->type);
  }

  void emitRegion(sys::ASTNode *node, Region &region) {
    auto &block = region.addBlock();
    Builder nested(ctx, &block);
    pushScope();
    emitStmt(node, nested);
    if (block.ops().empty() || !block.ops().back()->isTerminator())
      nested.create("scf.yield", {}, {}, {}, loc());
    popScope();
  }

  bool insertionBlockTerminated(Builder &builder) const {
    auto *block = builder.getInsertionBlock();
    return block && !block->ops().empty() && block->ops().back()->isTerminator();
  }

  void emitStmt(sys::ASTNode *node, Builder &builder) {
    if (!node)
      return;
    if (auto *block = dyn_cast<sys::BlockNode>(node)) {
      pushScope();
      for (auto *child : block->nodes) {
        if (insertionBlockTerminated(builder))
          break;
        emitStmt(child, builder);
      }
      popScope();
      return;
    }
    if (auto *transparent = dyn_cast<sys::TransparentBlockNode>(node)) {
      for (auto *decl : transparent->nodes) {
        if (insertionBlockTerminated(builder))
          break;
        emitStmt(decl, builder);
      }
      return;
    }
    if (auto *func = dyn_cast<sys::FnDeclNode>(node)) {
      auto *fnTy = dyn_cast<sys::FunctionType>(func->type);
      auto &op = builder.create("sysy.func", {}, {},
                                {{"sym_name", ctx.stringAttr(func->name)},
                                 {"type", ctx.stringAttr(typeText(ctx, func->type))}},
                                loc(), 1);
      auto &entry = op.getRegions()[0]->addBlock();
      Builder body(ctx, &entry);
      pushScope();
      for (size_t i = 0; i < func->args.size(); i++) {
        sys::Type *argTy = fnTy && i < fnTy->params.size() ? fnTy->params[i] : nullptr;
        auto &arg = entry.addArgument(isa<sys::ArrayType>(argTy) || isa<sys::PointerType>(argTy)
                                          ? storageType(ctx, argTy)
                                          : scalarType(ctx, argTy),
                                      loc(), func->args[i]);
        if (isa<sys::ArrayType>(argTy) || isa<sys::PointerType>(argTy)) {
          bindStorage(func->args[i], arg.value());
        } else {
          auto &slot = body.create("sysy.alloca", {}, {storageType(ctx, argTy)},
                                   {{"symbol", ctx.stringAttr(func->args[i])}}, loc());
          body.create("sysy.store", {arg.value(), slot.result()}, {},
                      {{"symbol", ctx.stringAttr(func->args[i])}}, loc());
          bindStorage(func->args[i], slot.result());
        }
      }
      emitStmt(func->body, body);
      if (entry.ops().empty() || !entry.ops().back()->isTerminator())
        body.create("scf.yield", {}, {}, {}, loc());
      popScope();
      return;
    }
    if (auto *decl = dyn_cast<sys::VarDeclNode>(node)) {
      std::map<std::string, Attribute> attrs{{"symbol", ctx.stringAttr(decl->name)}};
      if (auto *arr = dyn_cast<sys::ArrayType>(decl->type)) {
        std::string shape;
        for (size_t i = 0; i < arr->dims.size(); i++) {
          if (i)
            shape += "x";
          shape += std::to_string(arr->dims[i]);
        }
        attrs["shape"] = ctx.stringAttr(shape);
        if (decl->global) {
          if (auto *init = dyn_cast<sys::ConstArrayNode>(decl->init)) {
            std::string words = globalArrayInitWords(arr, init);
            if (!words.empty())
              attrs["init_words"] = ctx.stringAttr(words);
          }
        }
      }
      auto &slot = builder.create(decl->global ? "sysy.global" : "sysy.alloca",
                                  {}, {storageType(ctx, decl->type)}, attrs, loc());
      bindStorage(decl->name, slot.result());
      if (!decl->global) {
        if (auto *arr = dyn_cast<sys::ArrayType>(decl->type)) {
          if (decl->init) {
            if (auto *init = dyn_cast<sys::LocalArrayNode>(decl->init))
              emitLocalArrayInit(decl, arr, init, slot.result(), builder);
          }
        }
      }
      if (decl->init && !isa<sys::ConstArrayNode>(decl->init) &&
          !isa<sys::LocalArrayNode>(decl->init)) {
        Value value = emitExpr(decl->init, builder);
        builder.create("sysy.store", {value, slot.result()}, {},
                       {{"symbol", ctx.stringAttr(decl->name)}}, loc());
      }
      return;
    }
    if (auto *ret = dyn_cast<sys::ReturnNode>(node)) {
      std::vector<Value> operands;
      if (ret->node)
        operands.push_back(emitExpr(ret->node, builder));
      builder.create("sysy.return", operands, {}, {}, loc());
      return;
    }
    if (auto *ifn = dyn_cast<sys::IfNode>(node)) {
      Value cond = emitExpr(ifn->cond, builder);
      auto &op = builder.create("scf.if", {cond}, {}, {}, loc(), ifn->ifnot ? 2 : 1);
      emitRegion(ifn->ifso, *op.getRegions()[0]);
      if (ifn->ifnot)
        emitRegion(ifn->ifnot, *op.getRegions()[1]);
      return;
    }
    if (auto *loop = dyn_cast<sys::WhileNode>(node)) {
      auto &op = builder.create("scf.while", {}, {}, {}, loc(), 2);
      auto &beforeBlock = op.getRegions()[0]->addBlock();
      Builder beforeBuilder(ctx, &beforeBlock);
      Value cond = emitExpr(loop->cond, beforeBuilder);
      beforeBuilder.create("scf.condition", {cond}, {}, {}, loc());
      emitRegion(loop->body, *op.getRegions()[1]);
      return;
    }
    if (auto *assign = dyn_cast<sys::AssignNode>(node)) {
      Value rhs = emitExpr(assign->r, builder);
      if (auto *ref = dyn_cast<sys::VarRefNode>(assign->l)) {
        Value slot = lookupStorage(ref->name);
        if (slot.valid())
          builder.create("sysy.store", {rhs, slot}, {}, {{"symbol", ctx.stringAttr(ref->name)}}, loc());
        else
          bindValue(ref->name, rhs);
        return;
      }
      if (auto *access = dyn_cast<sys::ArrayAccessNode>(assign->l)) {
        std::vector<Value> operands{rhs};
        Value base = lookupStorage(access->array);
        if (base.valid())
          operands.push_back(base);
        for (auto *idx : access->indices)
          operands.push_back(emitExpr(idx, builder));
        builder.create("memref.store", operands, {},
                       {{"symbol", ctx.stringAttr(access->array)}}, loc());
        return;
      }
      builder.create("sysy.store", {rhs}, {}, {}, loc());
      return;
    }
    if (auto *write = dyn_cast<sys::ArrayAssignNode>(node)) {
      std::vector<Value> operands{emitExpr(write->value, builder)};
      Value base = lookupStorage(write->array);
      if (base.valid())
        operands.push_back(base);
      for (auto *idx : write->indices)
        operands.push_back(emitExpr(idx, builder));
      builder.create("memref.store", operands, {},
                     {{"symbol", ctx.stringAttr(write->array)}}, loc());
      return;
    }
    if (isa<sys::BreakNode>(node)) {
      builder.create("sysy.break", {}, {}, {}, loc());
      return;
    }
    if (isa<sys::ContinueNode>(node)) {
      builder.create("sysy.continue", {}, {}, {}, loc());
      return;
    }
    if (isa<sys::EmptyNode>(node))
      return;
    (void) emitExpr(node, builder);
  }

public:
  explicit ASTToMLIR(Context &ctx): ctx(ctx) {}

  std::unique_ptr<Module> run(const sys::ASTNode &ast, ProductionStats *stats) {
    if (stats)
      stats->hirOps = countAST(&ast);
    auto module = std::make_unique<Module>(ctx);
    auto &top = module->body().getBlocks()[0];
    Builder builder(ctx, top.get());
    pushScope();
    emitStmt(const_cast<sys::ASTNode*>(&ast), builder);
    popScope();
    if (stats)
      stats->mlirOpsBefore = (int) walk(*module).size();
    return module;
  }
};

std::vector<ConversionPattern> targetPatterns(const std::string &target) {
  const std::string prefix = target == "arm" ? "arm_machine." : "rv_machine.";
  if (target == "arm") {
    return {
      {"arith.constant", prefix + "mov"}, {"arith.addi", prefix + "add"},
      {"arith.subi", prefix + "sub"}, {"arith.muli", prefix + "mul"},
      {"arith.divi", prefix + "sdiv"}, {"arith.remi", prefix + "srem"},
      {"arith.andi", prefix + "and"}, {"arith.ori", prefix + "orr"},
      {"arith.xori", prefix + "eor"},
      {"arith.noti", prefix + "not"}, {"arith.cmpi", prefix + "cmp"},
      {"arith.cmpf", prefix + "fcmp"}, {"arith.addf", prefix + "fadd"},
      {"arith.subf", prefix + "fsub"}, {"arith.mulf", prefix + "fmul"},
      {"arith.divf", prefix + "fdiv"}, {"arith.negf", prefix + "fneg"},
      {"arith.negi", prefix + "neg"}, {"arith.sitofp", prefix + "scvtf"},
      {"arith.fptosi", prefix + "fcvtzs"}, {"arith.select", prefix + "select"},
      {"vector.transfer_read", "arm_machine.ld1"},
      {"vector.transfer_write", "arm_machine.st1"},
      {"vector.splat", "arm_machine.dup"},
    };
  }
  return {
    {"arith.constant", prefix + "li"}, {"arith.addi", prefix + "addw"},
    {"arith.subi", prefix + "subw"}, {"arith.muli", prefix + "mulw"},
    {"arith.divi", prefix + "divw"}, {"arith.remi", prefix + "remw"},
    {"arith.andi", prefix + "and"}, {"arith.ori", prefix + "or"},
    {"arith.xori", prefix + "xor"},
    {"arith.noti", prefix + "seqz"}, {"arith.cmpi", prefix + "cmp"},
    {"arith.cmpf", prefix + "fcmp"}, {"arith.addf", prefix + "fadd"},
    {"arith.subf", prefix + "fsub"}, {"arith.mulf", prefix + "fmul"},
    {"arith.divf", prefix + "fdiv"}, {"arith.negf", prefix + "fneg"},
    {"arith.negi", prefix + "neg"}, {"arith.sitofp", prefix + "fcvt_s_w"},
    {"arith.fptosi", prefix + "fcvt_w_s"}, {"arith.select", prefix + "select"},
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

static const char *kASTCoreRules =
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

} // namespace

std::unique_ptr<Module> lowerFromAST(Context &ctx, const sys::ASTNode &ast,
                                     const std::string &target,
                                     ProductionStats *stats) {
  (void) target;
  return ASTToMLIR(ctx).run(ast, stats);
}

std::unique_ptr<Module> runProductionGateFromAST(Context &ctx, const sys::ASTNode &ast, const std::string &target,
                                                 const OptimizationConfig &config,
                                                 ProductionStats &stats, std::ostream *dump) {
  (void) ctx;
  stats = ProductionStats();
  auto module = lowerFromAST(ctx, ast, target, &stats);
  OptimizationConfig effective = config;
  if (envDisabled("SISY_ENABLE_SELF_WORKLIST"))
    effective.enableDRRWorklist = false;
  if (envDisabled("SISY_ENABLE_SELF_LINEAR_SCAN"))
    effective.enableLinearScan = false;
  if (envDisabled("SISY_ENABLE_SELF_SCHED"))
    effective.enableScheduler = false;
  if (envDisabled("SISY_ENABLE_SELF_INLINE"))
    effective.enableInline = false;
  if (envDisabled("SISY_ENABLE_SELF_ROT_HELPER"))
    effective.enableRotateHelper = false;
  if (envDisabled("SISY_ENABLE_SELF_POW2_STRENGTH"))
    effective.enablePow2Strength = false;
  if (envDisabled("SISY_ENABLE_SELF_STENCIL_PEEL"))
    effective.enableStencilPeel = false;
  if (envDisabled("SISY_ENABLE_SELF_ADDR_IV"))
    effective.enableLoopAddressIV = false;
  if (envDisabled("SISY_ENABLE_SELF_POLY_TILE"))
    effective.enableLoopTiling = false;
  if (envDisabled("SISY_ENABLE_SELF_POLY_PERMUTE"))
    effective.enableLoopInterchange = false;
  if (envEnabled("SISY_ENABLE_SELF_TILE", effective.enableLoopTiling))
    effective.enableLoopTiling = true;
  if (envDisabled("SISY_ENABLE_SELF_TILE"))
    effective.enableLoopTiling = false;

  if (!envDisabled("SISY_ENABLE_SELF_ADAPTIVE_GATE") &&
      effective.level == OptimizationConfig::Level::O1) {
    stats.adaptiveLevel = "o1-default";
    int initialOps = 0;
    int loopOps = 0;
    int callOps = 0;
    int memOps = 0;
    int funcs = 0;
    for (auto *op : walk(*module)) {
      if (!op || op->isErased())
        continue;
      initialOps++;
      if (op->name() == "scf.while" || op->name() == "affine.for")
        loopOps++;
      else if (op->name() == "sysy.call")
        callOps++;
      else if (op->name() == "memref.load" || op->name() == "memref.store" ||
               op->name() == "sysy.load" || op->name() == "sysy.store")
        memOps++;
      else if (op->name() == "sysy.func")
        funcs++;
    }

    bool giantUnit = stats.hirOps > 25000 || initialOps > 90000 || funcs > 300;
    if (giantUnit) {
      stats.adaptiveLevel = "o1-fast-giant";
      effective.enableAffine = false;
      effective.enableLoopTiling = false;
      effective.enableLoopFusion = false;
      effective.enableLoopInterchange = false;
      effective.enableStencilPeel = false;
      effective.enableScheduler = false;
      if (callOps > 2000)
        effective.enableInline = false;
      stats.opt.walksEliminated++;
    } else if (target == "riscv" && loopOps >= 6 && memOps >= 12 &&
               initialOps < 6000) {
      stats.adaptiveLevel = "o1-compute-hot";
      effective.enableScheduler = true;
      effective.enableStencilPeel = true;
      effective.enableLoopAddressIV = true;
      effective.inlineThreshold = std::max(effective.inlineThreshold, 360);
      effective.lateInlineThreshold = std::max(effective.lateInlineThreshold, 360);
    }
  }

  // 1. AST lowering
  if (effective.enableGlobalOpt) {
    runGlobalOpt(*module, &stats.opt);
    runReadonlyGlobalScalarPropagation(*module, &stats.opt);
  }

  // 2. High-level structure recovery and polyhedral preparation
  if (effective.enableAffine && !envDisabled("SISY_ENABLE_SELF_AFFINE_OPT")) {
    if (envEnabled("SISY_ENABLE_SELF_CONTINUE_WRAP", true))
      runContinueToIfWrap(*module);
    if (!envDisabled("SISY_ENABLE_SELF_RAISE_AFFINE"))
      stats.opt.affineWorklistItems += runRaiseToAffine(*module);
    if (envEnabled("SISY_ENABLE_SELF_IMPERFECT_PROMOTION", false))
      runImperfectLoopPromotion(*module);
  }
  if (effective.enableStencilPeel)
    runStencilPeelingAndUnroll(*module, &stats.opt);
  runIfStoreSelectPromotion(*module, &stats.opt);
  if (effective.level != OptimizationConfig::Level::O0) {
    runLoopInvariantCodeMotion(*module, &stats.opt);
    runLocalCSE(*module, &stats.opt);
  }

  // 3. Global straight-line optimizations.
  if (effective.enableGlobalOpt) {
    runGlobalOpt(*module, &stats.opt);
    runReadonlyGlobalScalarPropagation(*module, &stats.opt);
  }
  if (effective.enableMemoryOpt)
    runMemoryOpt(*module, &stats.opt);
  if (effective.enableLoopAddressIV)
    runLoopRepeatReduction(*module, &stats.opt);
  if (effective.enableLoopAddressIV)
    runLoopAddressIV(*module, &stats.opt);
  if (effective.level != OptimizationConfig::Level::O0)
    runClosedFormDivReduction(*module, &stats.opt);
  if (effective.level != OptimizationConfig::Level::O0)
    runLocalCSE(*module, &stats.opt);
  if (effective.enableProvenBitwise)
    runProvenBitwiseHelper(*module, &stats.opt);
  if (effective.enableRotateHelper)
    runRotateHelperFold(*module, &stats.opt);
  if (effective.level != OptimizationConfig::Level::O0) {
    runSignedPow2RemainderRewrite(*module, target, &stats.opt);
    runLocalCSE(*module, &stats.opt);
  }

  // 4. Conservative IPO runs after structural helper recognition so it does
  // not destroy recognizable bitwise helper shapes.
  if (effective.level != OptimizationConfig::Level::O0 && effective.enableInline) {
    runPureFunctionDeduction(*module);
    int inlineCallsBefore = stats.opt.inlineCalls;
    runInlining(*module, effective.inlineThreshold, &stats.opt);
    runIPCP(*module, &stats.opt);
    if (effective.enableMemoryOpt && stats.opt.inlineCalls > inlineCallsBefore)
      runMemoryOpt(*module, &stats.opt, true);
    if (effective.enableStencilPeel && stats.opt.inlineCalls > inlineCallsBefore) {
      runStencilPeelingAndUnroll(*module, &stats.opt);
      if (effective.enableMemoryOpt)
        runMemoryOpt(*module, &stats.opt, true);
    }
    if (effective.enableLoopAddressIV && stats.opt.inlineCalls > inlineCallsBefore)
      runLoopRepeatReduction(*module, &stats.opt);
    runClosedFormDivReduction(*module, &stats.opt);
    runLoopInvariantCodeMotion(*module, &stats.opt);
    runLocalCSE(*module, &stats.opt);
    if (effective.enableLoopAddressIV && stats.opt.inlineCalls > inlineCallsBefore)
      runLoopAddressIV(*module, &stats.opt);
  }

  // 5. Re-run structural recovery after IPO.  Helper calls such as idx() and
  // digit extraction routines often block the first affine raise; once they
  // are inlined, the surrounding hot loops can finally enter the affine and
  // polyhedral pipeline.
  if (effective.level != OptimizationConfig::Level::O0) {
    if (effective.enableStencilPeel)
      runStencilPeelingAndUnroll(*module, &stats.opt);
    if (effective.enableAffine && !envDisabled("SISY_ENABLE_SELF_AFFINE_OPT")) {
      if (!envDisabled("SISY_ENABLE_SELF_RAISE_AFFINE")) {
        int phase2 = runRaiseToAffine(*module);
        stats.opt.affineWorklistItems += phase2;
        stats.opt.phase2AffineRaises += phase2;
      }
      if (envEnabled("SISY_ENABLE_SELF_IMPERFECT_PROMOTION", false))
        runImperfectLoopPromotion(*module);
      if (effective.enableLoopTiling)
        runLoopTiling(*module, &stats.opt);
      if (effective.enableLoopTiling)
        runDiagonalTransposeTiling(*module, &stats.opt);
      if (effective.enableLoopFusion && !envDisabled("SISY_ENABLE_SELF_LOOP_FUSION"))
        runAffineLoopFusion(*module);
      if (effective.enableLoopInterchange && !envDisabled("SISY_ENABLE_SELF_LOOP_INTERCHANGE"))
        runPolyhedralLoopPermutation(*module, &stats.opt);
    }
    if (effective.enableStencilPeel) {
      runLoopRangePeeling(*module, &stats.opt);
      runStencilPeelingAndUnroll(*module, &stats.opt);
    }
    runIfStoreSelectPromotion(*module, &stats.opt);
    runMemrefLinearization(*module, &stats.opt);
    runLoopInvariantCodeMotion(*module, &stats.opt);
    runClosedFormDivReduction(*module, &stats.opt);
    runLocalCSE(*module, &stats.opt);
    runLoopInvariantCodeMotion(*module, &stats.opt);
    if (effective.enableMemoryOpt)
      runMemoryOpt(*module, &stats.opt, true);
    if (effective.enableLoopAddressIV) {
      runLoopRepeatReduction(*module, &stats.opt);
      runLoopAddressIV(*module, &stats.opt);
    }
    if (effective.enableProvenBitwise)
      runProvenBitwiseHelper(*module, &stats.opt);
    if (effective.enableRotateHelper)
      runRotateHelperFold(*module, &stats.opt);
    runSignedPow2RemainderRewrite(*module, target, &stats.opt);
    runLocalCSE(*module, &stats.opt);
  }

  if (target == "riscv" && effective.level != OptimizationConfig::Level::O0)
    runAccumulatorRecursiveMemoization(*module, &stats.opt);

  if (effective.enableScheduler)
    runLoopLocalScheduler(*module, &stats.opt);
  if (effective.enableAffine)
    collectAffineNestSummary(*module, &stats.opt);
  if (std::getenv("SISY_ENABLE_RVV"))
    runLoopVectorization(*module);
  if (target == "riscv" && effective.level != OptimizationConfig::Level::O0) {
    for (Operation *op : walk(*module)) {
      if (op && !op->isErased() && op->name() == "sysy.func")
        op->setAttr("structural_kernel_eligible", ctx.boolAttr(true));
    }
  }
  auto before = verify(*module);
  stats.verifyBefore = before.ok;
  if (!before.ok) {
    stats.error = before.errors.empty() ? "self-MLIR AST verification failed before canonicalization"
                                        : before.errors.front();
    if (dump)
      print(*module, *dump);
    return nullptr;
  }

  if (effective.enableDRR) {
    std::vector<std::string> parseErrors;
    auto rules = parseDRR(kASTCoreRules, parseErrors);
    if (!parseErrors.empty()) {
      stats.error = parseErrors.front();
      return nullptr;
    }
    auto rewriteStats = applyGreedyPatterns(*module, rules, effective.enableDRRWorklist);
    stats.rewrites = rewriteStats.rewrites;
    stats.opt.worklistRewrites += rewriteStats.rewrites;
    stats.opt.walksEliminated += rewriteStats.walksEliminated;
  }

  auto after = verify(*module);
  stats.verifyAfter = after.ok;
  if (!after.ok) {
    stats.error = after.errors.empty() ? "self-MLIR AST verification failed after canonicalization"
                                       : after.errors.front();
    if (dump)
      print(*module, *dump);
    return nullptr;
  }

  auto convStats = convertDialects(*module, productionTarget(target), targetPatterns(target));
  stats.conversionVisited = convStats.visited;
  stats.conversionLegal = convStats.legal;
  stats.conversionConverted = convStats.converted;
  stats.conversionFailed = convStats.failed;
  stats.conversionRollbacks = convStats.rollbacks;
  if (effective.enablePow2Strength)
    runParityProductCompareStrength(*module, &stats.opt);
  runMachineDeadCodeElim(*module, &stats.opt);
  summarizeModule(*module, stats);
  if (dump) {
    *dump << "===== self-MLIR production =====\n";
    print(*module, *dump);
  }
  if (convStats.failed) {
    stats.error = "self-MLIR AST dialect conversion failed for target " + target;
    return nullptr;
  }
  return module;
}

} // namespace sys::mlir
