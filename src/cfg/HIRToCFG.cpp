#include "HIRToCFG.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "../hir/HIRBuilder.h"
#include "../utils/DynamicCast.h"

namespace sys::cfg {

namespace {

size_t productDims(const std::vector<int> &dims) {
  if (dims.empty())
    return 1;
  size_t prod = 1;
  for (int dim : dims)
    prod *= (size_t) std::max(dim, 1);
  return prod;
}

size_t typeSize(Type *ty);

size_t scalarSize(Type *ty) {
  if (!ty)
    return 4;
  if (isa<IntType>(ty) || isa<FloatType>(ty))
    return 4;
  if (isa<PointerType>(ty))
    return 8;
  if (isa<VoidType>(ty))
    return 0;
  if (isa<FunctionType>(ty))
    return 8;
  if (isa<ArrayType>(ty))
    return 8;
  return 4;
}

hir::TypeKind mapElementType(Type *ty) {
  if (!ty)
    return hir::TypeKind::Unknown;
  if (isa<IntType>(ty))
    return hir::TypeKind::Int;
  if (isa<FloatType>(ty))
    return hir::TypeKind::Float;
  if (auto ptr = dyn_cast<PointerType>(ty))
    return mapElementType(ptr->pointee);
  if (auto arr = dyn_cast<ArrayType>(ty))
    return mapElementType(arr->base);
  return hir::TypeKind::Unknown;
}

size_t typeSize(Type *ty) {
  if (!ty)
    return 4;
  if (auto arr = dyn_cast<ArrayType>(ty))
    return typeSize(arr->base) * productDims(arr->dims);
  return scalarSize(ty);
}

std::vector<int> typeDims(Type *ty) {
  if (!ty)
    return {};
  if (auto arr = dyn_cast<ArrayType>(ty))
    return arr->dims;
  if (auto ptr = dyn_cast<PointerType>(ty)) {
    if (auto arr = dyn_cast<ArrayType>(ptr->pointee)) {
      // Parameter arrays like int a[][N][M] are represented as pointer-to-array.
      // Keep an unknown leading dimension so stride of the first index uses
      // known trailing dims (N, M, ...).
      std::vector<int> dims = arr->dims;
      dims.insert(dims.begin(), 0);
      return dims;
    }
  }
  return {};
}

std::vector<size_t> computeStrideBytes(const std::vector<int> &dims, size_t elemSize) {
  if (dims.empty())
    return {};
  std::vector<size_t> strides(dims.size(), elemSize ? elemSize : 4);
  for (size_t i = 0; i < dims.size(); i++) {
    size_t stride = elemSize ? elemSize : 4;
    for (size_t j = i + 1; j < dims.size(); j++)
      stride *= (size_t) std::max(dims[j], 1);
    strides[i] = stride;
  }
  return strides;
}

SymbolInfo buildSymbolInfo(const std::string &name, Type *ty, bool isGlobal, bool isParam, bool isMutable) {
  SymbolInfo info;
  info.name = name;
  info.type = hir::mapType(ty);
  info.elementType = mapElementType(ty);
  info.dims = typeDims(ty);
  info.isGlobal = isGlobal;
  info.isParam = isParam;
  info.isMutable = isMutable;
  info.baseKind = isGlobal ? MemoryBaseKind::Global : (isParam ? MemoryBaseKind::Param : MemoryBaseKind::Local);
  info.elemSize = 4;
  if (info.elementType == hir::TypeKind::Float || info.elementType == hir::TypeKind::Int)
    info.elemSize = 4;
  else if (info.type == hir::TypeKind::Pointer || info.type == hir::TypeKind::Array)
    info.elemSize = 8;
  info.storageSize = typeSize(ty);
  if (info.storageSize == 0)
    info.storageSize = (info.type == hir::TypeKind::Pointer || info.type == hir::TypeKind::Array) ? 8 : 4;
  info.strideBytes = computeStrideBytes(info.dims, info.elemSize);
  return info;
}

std::string formatFloat(double value) {
  std::ostringstream oss;
  // Preserve enough decimal digits to round-trip through strtof() in
  // CFG->legacy lowering without introducing avoidable ULP drift.
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

struct FuncLoweringState {
  int tempId = 0;
  int blockId = 0;
};

class Lowerer {
  const hir::Module &hirModule;
  std::vector<std::string> &errors;
  std::unordered_map<const hir::Op*, std::string> resolvedSymbols;
  std::unordered_map<std::string, int> declCounts;
  std::unordered_set<std::string> globalSymbols;

public:
  explicit Lowerer(const hir::Module &hirModule, std::vector<std::string> &errors):
    hirModule(hirModule), errors(errors) {}

  Module run() {
    Module cfgModule;
    cfgModule.originAst = hirModule.originAst;

    if (!hirModule.root || hirModule.root->kind != hir::OpKind::Module) {
      errors.push_back("hir->cfg: invalid HIR root");
      return cfgModule;
    }

    collectGlobals(*hirModule.root, cfgModule);

    for (const auto &child : hirModule.root->children) {
      if (!child)
        continue;
      if (child->kind != hir::OpKind::Func)
        continue;
      cfgModule.funcs.push_back(lowerFunc(*child, cfgModule));
    }

    if (cfgModule.funcs.empty())
      errors.push_back("hir->cfg: no function found");
    return cfgModule;
  }

private:
  std::string getResolvedSymbol(const hir::Op *op) const {
    if (!op)
      return "";
    auto it = resolvedSymbols.find(op);
    if (it != resolvedSymbols.end())
      return it->second;
    return op->symbol;
  }

  void resolveSymbolsRec(const hir::Op *op,
                         std::vector<std::unordered_map<std::string, std::string>> &scopes,
                         std::unordered_map<std::string, int> &counters) {
    if (!op)
      return;

    bool pushed = false;
    bool lexicalBlock = false;
    if (op->kind == hir::OpKind::Block) {
      // TransparentBlockNode is frequently used as a declaration wrapper
      // (e.g. `int a = ...;`) and should not create a nested lexical scope
      // by itself. But mixed-content transparent blocks do represent real
      // statement scopes and must isolate shadowed symbols.
      bool transparent = op->origin && isa<TransparentBlockNode>(op->origin);
      if (!transparent) {
        lexicalBlock = true;
      } else {
        bool declWrapper = !op->children.empty();
        for (const auto &child : op->children) {
          if (!child || child->kind != hir::OpKind::VarDecl) {
            declWrapper = false;
            break;
          }
        }
        lexicalBlock = !declWrapper;
      }
    }
    if (lexicalBlock || op->kind == hir::OpKind::For) {
      scopes.emplace_back();
      pushed = true;
    }

    if (op->kind == hir::OpKind::VarDecl && !op->symbol.empty()) {
      bool collides = globalSymbols.count(op->symbol) > 0;
      for (auto it = scopes.rbegin(); !collides && it != scopes.rend(); ++it)
        collides = it->count(op->symbol) > 0;
      bool needRename = declCounts[op->symbol] > 1 || collides;
      std::string renamed = op->symbol;
      if (needRename) {
        int id = counters[op->symbol]++;
        renamed = op->symbol + "$" + std::to_string(id);
      }
      scopes.back()[op->symbol] = renamed;
      resolvedSymbols[op] = renamed;
    } else if ((op->kind == hir::OpKind::Load || op->kind == hir::OpKind::Store) && !op->symbol.empty()) {
      std::string resolved = op->symbol;
      for (auto it = scopes.rbegin(); it != scopes.rend(); ++it) {
        auto found = it->find(op->symbol);
        if (found != it->end()) {
          resolved = found->second;
          break;
        }
      }
      resolvedSymbols[op] = resolved;
    }

    for (const auto &child : op->children)
      resolveSymbolsRec(child.get(), scopes, counters);

    if (pushed)
      scopes.pop_back();
  }

  void countVarDecls(const hir::Op *op) {
    if (!op)
      return;
    if (op->kind == hir::OpKind::VarDecl && !op->symbol.empty())
      declCounts[op->symbol]++;
    for (const auto &child : op->children)
      countVarDecls(child.get());
  }

  void buildResolvedSymbols(const hir::Op &funcOp) {
    resolvedSymbols.clear();
    declCounts.clear();
    globalSymbols.clear();
    std::vector<std::unordered_map<std::string, std::string>> scopes;
    scopes.emplace_back();
    std::unordered_map<std::string, int> counters;

    if (hirModule.root) {
      std::function<void(const hir::Op*)> collectGlobalsRec = [&](const hir::Op *node) {
        if (!node)
          return;
        if (node->kind == hir::OpKind::Func)
          return;
        if (node->kind == hir::OpKind::VarDecl && !node->symbol.empty())
          globalSymbols.insert(node->symbol);
        for (const auto &child : node->children)
          collectGlobalsRec(child.get());
      };
      collectGlobalsRec(hirModule.root.get());
    }

    if (!funcOp.children.empty())
      countVarDecls(funcOp.children[0].get());

    if (auto *fn = dyn_cast<FnDeclNode>(funcOp.origin)) {
      for (const auto &arg : fn->args)
        scopes.back()[arg] = arg;
    }
    if (!funcOp.children.empty())
      resolveSymbolsRec(funcOp.children[0].get(), scopes, counters);
  }

  static bool isTerminated(const Func &func, int bid) {
    if (bid < 0 || bid >= (int) func.blocks.size())
      return true;
    const auto &insts = func.blocks[bid].insts;
    if (insts.empty())
      return false;
    return isTerminator(insts.back().kind);
  }

  static void normalizePhiOrder(Func &func) {
    for (auto &bb : func.blocks) {
      if (bb.insts.empty())
        continue;
      std::vector<Inst> phis;
      std::vector<Inst> rest;
      phis.reserve(bb.insts.size());
      rest.reserve(bb.insts.size());
      for (auto &inst : bb.insts) {
        if (inst.kind == OpKind::Phi)
          phis.push_back(std::move(inst));
        else
          rest.push_back(std::move(inst));
      }
      std::vector<Inst> merged;
      merged.reserve(phis.size() + rest.size());
      for (auto &inst : phis)
        merged.push_back(std::move(inst));
      for (auto &inst : rest)
        merged.push_back(std::move(inst));
      bb.insts.swap(merged);
    }
  }

  static hir::TypeKind inferExprType(const hir::Op *op) {
    if (!op)
      return hir::TypeKind::Unknown;
    if (op->kind == hir::OpKind::Cmp)
      return hir::TypeKind::Int;
    return op->type;
  }

  static size_t typeBytes(hir::TypeKind kind) {
    switch (kind) {
    case hir::TypeKind::Int:
    case hir::TypeKind::Float:
      return 4;
    case hir::TypeKind::Pointer:
    case hir::TypeKind::Array:
    case hir::TypeKind::Function:
      return 8;
    case hir::TypeKind::Void:
      return 0;
    case hir::TypeKind::Unknown:
      return 4;
    }
    return 4;
  }

  int newBlock(Func &func, FuncLoweringState &st, const std::string &prefix) {
    Block bb;
    bb.name = prefix + "." + std::to_string(st.blockId++);
    func.blocks.push_back(std::move(bb));
    return (int) func.blocks.size() - 1;
  }

  static void emit(Func &func, int bid, const Inst &inst) {
    if (bid < 0 || bid >= (int) func.blocks.size())
      return;
    func.blocks[bid].insts.push_back(inst);
  }

  void ensureTerminator(Func &func, int bid) {
    if (bid < 0 || bid >= (int) func.blocks.size())
      return;
    if (isTerminated(func, bid))
      return;
    Inst ret;
    ret.kind = OpKind::Ret;
    ret.args.push_back("#0");
    emit(func, bid, ret);
  }

  std::string newTemp(FuncLoweringState &st) {
    return "%t" + std::to_string(st.tempId++);
  }

  static std::unordered_map<std::string, SymbolInfo> toSymbolMap(const Func &func, const Module &module) {
    std::unordered_map<std::string, SymbolInfo> map;
    for (const auto &sym : module.globals)
      map[sym.name] = sym;
    for (const auto &sym : func.params)
      map[sym.name] = sym;
    for (const auto &sym : func.locals)
      map[sym.name] = sym;
    return map;
  }

  static void addStoreSym(std::set<std::string> *stores, const std::string &sym) {
    if (!stores || sym.empty())
      return;
    stores->insert(sym);
  }

  void collectGlobals(const hir::Op &root, Module &cfgModule) {
    std::unordered_set<std::string> seen;
    std::function<void(const hir::Op*)> visit = [&](const hir::Op *op) {
      if (!op)
        return;
      if (op->kind == hir::OpKind::Func)
        return;
      if (op->kind == hir::OpKind::VarDecl && !op->symbol.empty() && !seen.count(op->symbol)) {
        const auto *origin = dyn_cast<VarDeclNode>(op->origin);
        Type *ty = origin ? origin->type : op->origin ? op->origin->type : nullptr;
        auto info = buildSymbolInfo(op->symbol, ty, true, false, origin ? origin->mut : true);

        if (origin && origin->init) {
          if (auto i = dyn_cast<IntNode>(origin->init)) {
            info.hasIntInit = true;
            info.intInit = i->value;
          } else if (auto f = dyn_cast<FloatNode>(origin->init)) {
            info.hasFloatInit = true;
            info.floatInit = f->value;
          } else if (auto arr = dyn_cast<ConstArrayNode>(origin->init)) {
            size_t elems = productDims(info.dims);
            if (elems == 0)
              elems = 1;
            if (arr->isFloat) {
              if (arr->vf)
                info.floatArrayInit.assign(arr->vf, arr->vf + elems);
            } else {
              if (arr->vi)
                info.intArrayInit.assign(arr->vi, arr->vi + elems);
            }
          }
        }
        seen.insert(op->symbol);
        cfgModule.globals.push_back(std::move(info));
      }
      for (const auto &child : op->children)
        visit(child.get());
    };

    visit(&root);
  }

  void collectLocalsRec(const hir::Op *op, std::vector<SymbolInfo> &locals, std::unordered_set<std::string> &seen) {
    if (!op)
      return;
    std::string sym = getResolvedSymbol(op);
    if (op->kind == hir::OpKind::VarDecl && !sym.empty() && !seen.count(sym)) {
      const auto *origin = op->origin ? dyn_cast<VarDeclNode>(op->origin) : nullptr;
      Type *ty = origin ? origin->type : op->origin ? op->origin->type : nullptr;
      locals.push_back(buildSymbolInfo(sym, ty, false, false, origin ? origin->mut : true));
      seen.insert(sym);
    }
    for (const auto &child : op->children)
      collectLocalsRec(child.get(), locals, seen);
  }

  std::string lowerExpr(const hir::Op *op, Func &func, FuncLoweringState &st, int &cur,
                        const std::unordered_map<std::string, SymbolInfo> &symbols) {
    if (!op)
      return "#0";

    switch (op->kind) {
    case hir::OpKind::ConstInt:
      if (op->hasIntValue)
        return "#" + std::to_string(op->intValue);
      return "#0";
    case hir::OpKind::ConstFloat:
      if (op->hasFloatValue)
        return "f#" + formatFloat(op->floatValue);
      return "f#0.0";
    case hir::OpKind::Load: {
      Inst inst;
      inst.kind = OpKind::Load;
      inst.type = op->type;
      inst.symbol = getResolvedSymbol(op);
      auto it = symbols.find(inst.symbol);
      if (it != symbols.end()) {
        inst.elementType = it->second.elementType;
        bool indexed = !op->children.empty();
        attachMemorySemantics(inst, &it->second, op->children.size());
        if (indexed) {
          bool partialIndex = !it->second.dims.empty() && op->children.size() < it->second.dims.size();
          if (partialIndex) {
            inst.type = hir::TypeKind::Pointer;
            inst.memSize = 8;
            inst.producesAddress = true;
          } else {
            inst.type = it->second.elementType;
            inst.memSize = it->second.elemSize;
          }
        } else if (it->second.type == hir::TypeKind::Array || it->second.type == hir::TypeKind::Pointer) {
          inst.type = hir::TypeKind::Pointer;
          inst.memSize = 8;
          inst.producesAddress = true;
        } else {
          inst.memSize = std::max((size_t) 4, it->second.storageSize);
        }
      } else {
        inst.memSize = typeBytes(inst.type);
      }
      inst.result = newTemp(st);
      for (const auto &idx : op->children)
        inst.args.push_back(lowerExpr(idx.get(), func, st, cur, symbols));
      emit(func, cur, inst);
      return inst.result;
    }
    case hir::OpKind::Call: {
      Inst inst;
      inst.kind = OpKind::Call;
      inst.type = op->type;
      inst.calleeRetType = op->type;
      inst.symbol = op->symbol;
      for (const auto &arg : op->children) {
        inst.calleeArgTypes.push_back(inferExprType(arg.get()));
        inst.args.push_back(lowerExpr(arg.get(), func, st, cur, symbols));
      }
      if (op->type != hir::TypeKind::Void)
        inst.result = newTemp(st);
      emit(func, cur, inst);
      return inst.result.empty() ? "#0" : inst.result;
    }
    case hir::OpKind::Arith:
      if (op->children.size() == 2 && (op->symbol == "&&" || op->symbol == "||")) {
        bool isAnd = op->symbol == "&&";
        int rhsId = newBlock(func, st, isAnd ? "sc.and.rhs" : "sc.or.rhs");
        int shortId = newBlock(func, st, isAnd ? "sc.and.short" : "sc.or.short");
        int mergeId = newBlock(func, st, isAnd ? "sc.and.merge" : "sc.or.merge");

        auto lhsCond = normalizeCond(op->children[0].get(), func, st, cur, symbols);
        Inst cbr;
        cbr.kind = OpKind::CondBr;
        cbr.args = { lhsCond };
        cbr.targets = isAnd ? std::vector<int> { rhsId, shortId } : std::vector<int> { shortId, rhsId };
        emit(func, cur, cbr);

        int rhsEnd = rhsId;
        auto rhsCond = normalizeCond(op->children[1].get(), func, st, rhsEnd, symbols);
        if (!isTerminated(func, rhsEnd)) {
          Inst br;
          br.kind = OpKind::Br;
          br.targets.push_back(mergeId);
          emit(func, rhsEnd, br);
        }

        if (!isTerminated(func, shortId)) {
          Inst br;
          br.kind = OpKind::Br;
          br.targets.push_back(mergeId);
          emit(func, shortId, br);
        }

        Inst phi;
        phi.kind = OpKind::Phi;
        phi.type = hir::TypeKind::Int;
        phi.result = newTemp(st);
        phi.phiPreds = { rhsEnd, shortId };
        phi.args = { rhsCond, isAnd ? "#0" : "#1" };
        emit(func, mergeId, phi);

        cur = mergeId;
        return phi.result;
      }
      [[fallthrough]];
    case hir::OpKind::Cmp: {
      Inst inst;
      inst.kind = (op->kind == hir::OpKind::Cmp) ? OpKind::Cmp : OpKind::Arith;
      inst.type = (op->kind == hir::OpKind::Cmp) ? hir::TypeKind::Int : op->type;
      if (op->kind == hir::OpKind::Cmp && !op->children.empty())
        inst.elementType = inferExprType(op->children[0].get());
      else
        inst.elementType = op->type;
      inst.symbol = op->symbol;
      inst.result = newTemp(st);
      for (const auto &child : op->children)
        inst.args.push_back(lowerExpr(child.get(), func, st, cur, symbols));
      emit(func, cur, inst);
      return inst.result;
    }
    default:
      break;
    }

    if (!op->children.empty())
      return lowerExpr(op->children[0].get(), func, st, cur, symbols);
    return "#0";
  }

  static std::string defaultToken(hir::TypeKind type) {
    return type == hir::TypeKind::Float ? "f#0.0" : "#0";
  }

  void attachMemorySemantics(Inst &inst, const SymbolInfo *sym, size_t indexCount, bool flattenedLinearIndex = false) {
    if (!sym)
      return;
    inst.baseKind = sym->baseKind;
    inst.accessRank = (int) indexCount;
    inst.strideBytes.clear();
    if (indexCount == 0)
      return;
    if (flattenedLinearIndex) {
      size_t elemSize = sym->elemSize ? sym->elemSize : 4;
      inst.strideBytes.push_back(elemSize);
      return;
    }
    size_t usable = std::min(indexCount, sym->strideBytes.size());
    inst.strideBytes.insert(inst.strideBytes.end(), sym->strideBytes.begin(), sym->strideBytes.begin() + usable);
  }

  std::string lowerASTExpr(ASTNode *node, hir::TypeKind fallbackType, Func &func,
                           FuncLoweringState &st, int &cur,
                           const std::unordered_map<std::string, SymbolInfo> &symbols) {
    if (!node)
      return defaultToken(fallbackType);

    hir::Builder builder;
    auto exprModule = builder.build(node);
    if (!exprModule.root || exprModule.root->children.empty() || !exprModule.root->children[0]) {
      errors.push_back("hir->cfg: cannot lower array init expression");
      return defaultToken(fallbackType);
    }
    return lowerExpr(exprModule.root->children[0].get(), func, st, cur, symbols);
  }

  std::string normalizeCond(const hir::Op *cond, Func &func, FuncLoweringState &st, int &cur,
                            const std::unordered_map<std::string, SymbolInfo> &symbols) {
    auto value = lowerExpr(cond, func, st, cur, symbols);
    auto ty = inferExprType(cond);
    if (ty == hir::TypeKind::Int || ty == hir::TypeKind::Unknown)
      return value;

    Inst cmp;
    cmp.kind = OpKind::Cmp;
    cmp.type = hir::TypeKind::Int;
    cmp.symbol = "!=";
    cmp.result = newTemp(st);
    cmp.args.push_back(value);
    cmp.args.push_back(ty == hir::TypeKind::Float ? "f#0.0" : "#0");
    emit(func, cur, cmp);
    return cmp.result;
  }

  void emitCondBranch(const hir::Op *cond, Func &func, FuncLoweringState &st, int cur,
                      const std::unordered_map<std::string, SymbolInfo> &symbols,
                      int trueTarget, int falseTarget) {
    if (!cond) {
      Inst br;
      br.kind = OpKind::Br;
      br.targets.push_back(falseTarget);
      emit(func, cur, br);
      return;
    }

    if (cond->kind == hir::OpKind::Arith && cond->children.size() == 2) {
      if (cond->symbol == "&&") {
        int rhsId = newBlock(func, st, "sc.and.rhs");
        emitCondBranch(cond->children[0].get(), func, st, cur, symbols, rhsId, falseTarget);
        emitCondBranch(cond->children[1].get(), func, st, rhsId, symbols, trueTarget, falseTarget);
        return;
      }
      if (cond->symbol == "||") {
        int rhsId = newBlock(func, st, "sc.or.rhs");
        emitCondBranch(cond->children[0].get(), func, st, cur, symbols, trueTarget, rhsId);
        emitCondBranch(cond->children[1].get(), func, st, rhsId, symbols, trueTarget, falseTarget);
        return;
      }
    }

    int at = cur;
    auto condVal = normalizeCond(cond, func, st, at, symbols);
    Inst cbr;
    cbr.kind = OpKind::CondBr;
    cbr.args.push_back(condVal);
    cbr.targets = { trueTarget, falseTarget };
    emit(func, at, cbr);
  }

  void emitLocalArrayInit(const std::string &symbol, const VarDeclNode *var, Func &func, FuncLoweringState &st, int &cur,
                          const std::unordered_map<std::string, SymbolInfo> &symbols,
                          std::set<std::string> *stores) {
    if (!var || !var->init)
      return;
    auto local = dyn_cast<LocalArrayNode>(var->init);
    auto arrTy = dyn_cast<ArrayType>(var->type);
    if (!local || !arrTy)
      return;

    auto it = symbols.find(symbol);
    if (it == symbols.end()) {
      errors.push_back("hir->cfg: local array init symbol not found: " + symbol);
      return;
    }
    if (it->second.dims.empty()) {
      errors.push_back("hir->cfg: local array init target is not array: " + symbol);
      return;
    }

    size_t declaredSize = productDims(it->second.dims);
    size_t astSize = productDims(arrTy->dims);
    if (declaredSize == 0)
      declaredSize = 1;
    if (astSize == 0)
      astSize = 1;
    if (declaredSize != astSize) {
      errors.push_back("hir->cfg: local array init dimension mismatch for " + symbol);
    }

    size_t elemSize = it->second.elemSize ? it->second.elemSize : 4;
    if (declaredSize <= 65536) {
      for (size_t i = 0; i < declaredSize; i++) {
        Inst zstore;
        zstore.kind = OpKind::Store;
        zstore.symbol = symbol;
        zstore.args = { "#" + std::to_string(i), defaultToken(it->second.elementType) };
        zstore.type = it->second.elementType;
        zstore.memSize = elemSize;
        attachMemorySemantics(zstore, &it->second, 1, /*flattenedLinearIndex=*/ true);
        emit(func, cur, zstore);
      }
      addStoreSym(stores, symbol);
    } else {
      int condId = newBlock(func, st, "arr.init.cond");
      int bodyId = newBlock(func, st, "arr.init.body");
      int exitId = newBlock(func, st, "arr.init.exit");

      Inst jump;
      jump.kind = OpKind::Br;
      jump.targets.push_back(condId);
      emit(func, cur, jump);

      std::string idxPhi = newTemp(st);
      std::string idxNext = newTemp(st);

      Inst phi;
      phi.kind = OpKind::Phi;
      phi.type = hir::TypeKind::Int;
      phi.result = idxPhi;
      phi.phiPreds = { cur, bodyId };
      phi.args = { "#0", idxNext };
      emit(func, condId, phi);

      Inst cmp;
      cmp.kind = OpKind::Cmp;
      cmp.type = hir::TypeKind::Int;
      cmp.symbol = "<";
      cmp.result = newTemp(st);
      cmp.args = { idxPhi, "#" + std::to_string(declaredSize) };
      emit(func, condId, cmp);

      Inst cbr;
      cbr.kind = OpKind::CondBr;
      cbr.args = { cmp.result };
      cbr.targets = { bodyId, exitId };
      emit(func, condId, cbr);

      Inst zstore;
      zstore.kind = OpKind::Store;
      zstore.symbol = symbol;
      zstore.args = { idxPhi, defaultToken(it->second.elementType) };
      zstore.type = it->second.elementType;
      zstore.memSize = elemSize;
      attachMemorySemantics(zstore, &it->second, 1, /*flattenedLinearIndex=*/ true);
      emit(func, bodyId, zstore);

      Inst add;
      add.kind = OpKind::Arith;
      add.type = hir::TypeKind::Int;
      add.symbol = "+";
      add.result = idxNext;
      add.args = { idxPhi, "#1" };
      emit(func, bodyId, add);

      Inst back;
      back.kind = OpKind::Br;
      back.targets.push_back(condId);
      emit(func, bodyId, back);

      cur = exitId;
      addStoreSym(stores, symbol);
    }

    for (size_t i = 0; i < astSize; i++) {
      ASTNode *elem = (local->va && i < astSize) ? local->va[i] : nullptr;
      if (!elem)
        continue;
      std::string token = lowerASTExpr(elem, it->second.elementType, func, st, cur, symbols);
      Inst store;
      store.kind = OpKind::Store;
      store.symbol = symbol;
      store.args = { "#" + std::to_string(i), token };
      store.type = it->second.elementType;
      store.memSize = elemSize;
      attachMemorySemantics(store, &it->second, 1, /*flattenedLinearIndex=*/ true);
      emit(func, cur, store);
      addStoreSym(stores, symbol);
    }
  }

  int lowerStmt(const hir::Op *op, Func &func, FuncLoweringState &st, int cur,
                int breakTarget, int continueTarget,
                const std::unordered_map<std::string, SymbolInfo> &symbols,
                std::set<std::string> *stores) {
    if (!op)
      return cur;

    if (isTerminated(func, cur)) {
      int cont = newBlock(func, st, "dead");
      cur = cont;
    }

    switch (op->kind) {
    case hir::OpKind::Block: {
      int at = cur;
      for (const auto &child : op->children)
        at = lowerStmt(child.get(), func, st, at, breakTarget, continueTarget, symbols, stores);
      return at;
    }
    case hir::OpKind::VarDecl: {
      std::string sym = getResolvedSymbol(op);
      auto it = symbols.find(sym);
      bool isArray = it != symbols.end() && !it->second.dims.empty();
      if (isArray) {
        auto *var = dyn_cast<VarDeclNode>(op->origin);
        emitLocalArrayInit(sym, var, func, st, cur, symbols, stores);
        return cur;
      }
      if (!op->children.empty()) {
        auto value = lowerExpr(op->children[0].get(), func, st, cur, symbols);
        Inst store;
        store.kind = OpKind::Store;
        store.symbol = sym;
        store.args.push_back(value);
        if (it != symbols.end()) {
          store.type = it->second.type;
          store.memSize = std::max((size_t) 4, it->second.storageSize);
          attachMemorySemantics(store, &it->second, 0);
        }
        emit(func, cur, store);
        addStoreSym(stores, sym);
      }
      return cur;
    }
    case hir::OpKind::Store: {
      if (op->children.empty())
        return cur;
      Inst store;
      store.kind = OpKind::Store;
      store.symbol = getResolvedSymbol(op);
      for (size_t i = 0; i + 1 < op->children.size(); i++)
        store.args.push_back(lowerExpr(op->children[i].get(), func, st, cur, symbols));
      store.args.push_back(lowerExpr(op->children.back().get(), func, st, cur, symbols));
      auto it = symbols.find(store.symbol);
      if (it != symbols.end()) {
        bool indexed = store.args.size() > 1;
        store.type = indexed ? it->second.elementType : it->second.type;
        store.memSize = indexed ? it->second.elemSize : std::max((size_t) 4, it->second.storageSize);
        attachMemorySemantics(store, &it->second, indexed ? store.args.size() - 1 : 0);
      }
      emit(func, cur, store);
      addStoreSym(stores, store.symbol);
      return cur;
    }
    case hir::OpKind::Call:
    case hir::OpKind::Arith:
    case hir::OpKind::Cmp:
    case hir::OpKind::Load:
      (void) lowerExpr(op, func, st, cur, symbols);
      return cur;
    case hir::OpKind::Return: {
      Inst ret;
      ret.kind = OpKind::Ret;
      if (!op->children.empty())
        ret.args.push_back(lowerExpr(op->children[0].get(), func, st, cur, symbols));
      else
        ret.args.push_back("#0");
      emit(func, cur, ret);
      return cur;
    }
    case hir::OpKind::Break: {
      if (breakTarget < 0) {
        errors.push_back("hir->cfg: break outside loop");
        return cur;
      }
      Inst br;
      br.kind = OpKind::Br;
      br.targets.push_back(breakTarget);
      emit(func, cur, br);
      return cur;
    }
    case hir::OpKind::Continue: {
      if (continueTarget < 0) {
        errors.push_back("hir->cfg: continue outside loop");
        return cur;
      }
      Inst br;
      br.kind = OpKind::Br;
      br.targets.push_back(continueTarget);
      emit(func, cur, br);
      return cur;
    }
    case hir::OpKind::If: {
      if (op->children.size() < 2) {
        errors.push_back("hir->cfg: malformed if op");
        return cur;
      }
      int thenId = newBlock(func, st, "if.then");
      int elseId = newBlock(func, st, "if.else");
      int mergeId = newBlock(func, st, "if.merge");

      emitCondBranch(op->children[0].get(), func, st, cur, symbols, thenId, elseId);

      std::set<std::string> thenStores;
      int thenEnd = lowerStmt(op->children[1].get(), func, st, thenId, breakTarget, continueTarget, symbols, &thenStores);
      if (!isTerminated(func, thenEnd)) {
        Inst br;
        br.kind = OpKind::Br;
        br.targets.push_back(mergeId);
        emit(func, thenEnd, br);
      }

      std::set<std::string> elseStores;
      int elseEnd = elseId;
      if (op->children.size() >= 3)
        elseEnd = lowerStmt(op->children[2].get(), func, st, elseId, breakTarget, continueTarget, symbols, &elseStores);
      if (!isTerminated(func, elseEnd)) {
        Inst br;
        br.kind = OpKind::Br;
        br.targets.push_back(mergeId);
        emit(func, elseEnd, br);
      }

      if (stores) {
        stores->insert(thenStores.begin(), thenStores.end());
        stores->insert(elseStores.begin(), elseStores.end());
      }
      return mergeId;
    }
    case hir::OpKind::While: {
      if (op->children.size() < 2) {
        errors.push_back("hir->cfg: malformed while op");
        return cur;
      }
      int condId = newBlock(func, st, "while.cond");
      int bodyId = newBlock(func, st, "while.body");
      int exitId = newBlock(func, st, "while.exit");

      Inst jump;
      jump.kind = OpKind::Br;
      jump.targets.push_back(condId);
      emit(func, cur, jump);

      emitCondBranch(op->children[0].get(), func, st, condId, symbols, bodyId, exitId);

      std::set<std::string> bodyStores;
      int bodyEnd = lowerStmt(op->children[1].get(), func, st, bodyId, exitId, condId, symbols, &bodyStores);
      if (!isTerminated(func, bodyEnd)) {
        Inst back;
        back.kind = OpKind::Br;
        back.targets.push_back(condId);
        emit(func, bodyEnd, back);
      }

      for (const auto &sym : bodyStores)
        addStoreSym(stores, sym);
      return exitId;
    }
    case hir::OpKind::For: {
      if (op->children.size() < 4) {
        errors.push_back("hir->cfg: malformed for op");
        return cur;
      }
      cur = lowerStmt(op->children[0].get(), func, st, cur, breakTarget, continueTarget, symbols, stores);

      int condId = newBlock(func, st, "for.cond");
      int bodyId = newBlock(func, st, "for.body");
      int stepId = newBlock(func, st, "for.step");
      int exitId = newBlock(func, st, "for.exit");

      Inst toCond;
      toCond.kind = OpKind::Br;
      toCond.targets.push_back(condId);
      emit(func, cur, toCond);

      emitCondBranch(op->children[1].get(), func, st, condId, symbols, bodyId, exitId);

      int bodyEnd = lowerStmt(op->children[3].get(), func, st, bodyId, exitId, stepId, symbols, stores);
      if (!isTerminated(func, bodyEnd)) {
        Inst toStep;
        toStep.kind = OpKind::Br;
        toStep.targets.push_back(stepId);
        emit(func, bodyEnd, toStep);
      }

      int stepEnd = lowerStmt(op->children[2].get(), func, st, stepId, exitId, stepId, symbols, stores);
      if (!isTerminated(func, stepEnd)) {
        Inst back;
        back.kind = OpKind::Br;
        back.targets.push_back(condId);
        emit(func, stepEnd, back);
      }
      return exitId;
    }
    default:
      return cur;
    }
  }

  Func lowerFunc(const hir::Op &funcOp, const Module &cfgModule) {
    Func func;
    FuncLoweringState st;
    func.name = funcOp.symbol.empty() ? "anonymous" : funcOp.symbol;

    buildResolvedSymbols(funcOp);

    if (auto *fn = dyn_cast<FnDeclNode>(funcOp.origin)) {
      if (auto *fnTy = dyn_cast<FunctionType>(fn->type)) {
        func.returnType = hir::mapType(fnTy->ret);
        for (size_t i = 0; i < fn->args.size(); i++) {
          Type *argTy = i < fnTy->params.size() ? fnTy->params[i] : nullptr;
          func.params.push_back(buildSymbolInfo(fn->args[i], argTy, false, true, true));
        }
      }
    }

    std::unordered_set<std::string> seen;
    for (const auto &param : func.params)
      seen.insert(param.name);
    if (!funcOp.children.empty())
      collectLocalsRec(funcOp.children[0].get(), func.locals, seen);

    func.entry = newBlock(func, st, "entry");

    auto symbols = toSymbolMap(func, cfgModule);

    int cur = func.entry;
    if (!funcOp.children.empty())
      cur = lowerStmt(funcOp.children[0].get(), func, st, cur, -1, -1, symbols, nullptr);
    normalizePhiOrder(func);
    ensureTerminator(func, cur);

    for (int bid = 0; bid < (int) func.blocks.size(); bid++)
      ensureTerminator(func, bid);
    return func;
  }
};

}  // namespace

Module lowerFromHIR(const hir::Module &hirModule, std::vector<std::string> &errors) {
  Lowerer lowering(hirModule, errors);
  return lowering.run();
}

}  // namespace sys::cfg
