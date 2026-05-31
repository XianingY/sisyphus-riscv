#include "ASTToMLIR.h"
#include "Polyhedral.h"
#include "IPO.h"

#include "../parse/ASTNode.h"
#include "../parse/Type.h"
#include "../utils/DynamicCast.h"

#include <map>
#include <cstdlib>
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
  if (auto *ptr = dyn_cast<sys::PointerType>(type))
    return ctx.memref({-1}, scalarType(ctx, ptr->pointee));
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
  if (auto *ptr = dyn_cast<sys::PointerType>(type))
    return ctx.memref({-1}, scalarType(ctx, ptr->pointee));
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

void summarizeModule(Module &module, ProductionStats &stats) {
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
                                {{"value", ctx.stringAttr(std::to_string(f->value))}}, loc());
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
        auto &op = builder.create("arith.noti", {input}, {ctx.i(32)}, {}, loc());
        return op.result();
      }
      auto &op = builder.create(isa<sys::FloatType>(node->type) ? "arith.negf" : "arith.negi",
                                {input}, {scalarType(ctx, node->type)}, {}, loc());
      return op.result();
    }
    if (auto *binary = dyn_cast<sys::BinaryNode>(node)) {
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
      }
      auto &slot = builder.create(decl->global ? "sysy.global" : "sysy.alloca",
                                  {}, {storageType(ctx, decl->type)}, attrs, loc());
      bindStorage(decl->name, slot.result());
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
                                                 ProductionStats &stats, std::ostream *dump) {
  stats = ProductionStats();
  auto module = lowerFromAST(ctx, ast, target, &stats);

  // 1. AST lowering
  runGlobalOpt(*module, &stats.opt);

  // 2. High-level structure recovery and polyhedral preparation
  if (!envDisabled("SISY_ENABLE_SELF_AFFINE_OPT")) {
    runRaiseToAffine(*module);
    runAffineLoopFusion(*module);
    runAffineLoopInterchange(*module);
  }

  // 3. Inter-Procedural Optimizations (IPO)
  runPureFunctionDeduction(*module);
  runInlining(*module);
  runIPCP(*module);

  // 4. Global straight-line optimizations
  runGlobalOpt(*module, &stats.opt);
  runMemoryOpt(*module, &stats.opt);
  if (std::getenv("SISY_ENABLE_RVV"))
    runLoopVectorization(*module);
  auto before = verify(*module);
  stats.verifyBefore = before.ok;
  if (!before.ok) {
    stats.error = before.errors.empty() ? "self-MLIR AST verification failed before canonicalization"
                                        : before.errors.front();
    if (dump)
      print(*module, *dump);
    return nullptr;
  }

  std::vector<std::string> parseErrors;
  auto rules = parseDRR(kASTCoreRules, parseErrors);
  if (!parseErrors.empty()) {
    stats.error = parseErrors.front();
    return nullptr;
  }
  auto rewriteStats = applyGreedyPatterns(*module, rules);
  stats.rewrites = rewriteStats.rewrites;

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
  stats.mlirOpsAfter = (int) walk(*module).size();
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
