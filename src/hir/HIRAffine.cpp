#include "HIRAffine.h"

#include <algorithm>

namespace sys::hir::affine {

namespace {

void normalize(Expr &expr) {
  for (auto it = expr.coeffs.begin(); it != expr.coeffs.end();) {
    if (it->second == 0)
      it = expr.coeffs.erase(it);
    else
      ++it;
  }
}

Expr invalidExpr() {
  return Expr{};
}

Expr constantExpr(int64_t value) {
  Expr expr;
  expr.valid = true;
  expr.constant = value;
  return expr;
}

Expr symbolExpr(const std::string &symbol) {
  if (symbol.empty())
    return invalidExpr();
  Expr expr;
  expr.valid = true;
  expr.coeffs[symbol] = 1;
  return expr;
}

Expr addExpr(Expr lhs, const Expr &rhs, int64_t sign = 1) {
  if (!lhs.valid || !rhs.valid)
    return invalidExpr();
  lhs.constant += sign * rhs.constant;
  for (const auto &[sym, coeff] : rhs.coeffs)
    lhs.coeffs[sym] += sign * coeff;
  normalize(lhs);
  return lhs;
}

Expr scaleExpr(Expr expr, int64_t factor) {
  if (!expr.valid)
    return invalidExpr();
  expr.constant *= factor;
  for (auto &[_, coeff] : expr.coeffs)
    coeff *= factor;
  normalize(expr);
  return expr;
}

bool isConstInt(const Op *op, int64_t &value) {
  if (!op || op->kind != OpKind::ConstInt || !op->hasIntValue)
    return false;
  value = op->intValue;
  return true;
}

const Op *unwrapSingleDecl(const Op *op) {
  if (!op)
    return nullptr;
  if (op->kind == OpKind::Block && op->children.size() == 1)
    return unwrapSingleDecl(op->children[0].get());
  return op;
}

bool isScalarLoad(const Op *op, const std::string &symbol) {
  return op && op->kind == OpKind::Load && op->symbol == symbol && op->children.empty();
}

bool isConstIntValue(const Op *op, int64_t value) {
  int64_t actual = 0;
  return isConstInt(op, actual) && actual == value;
}

bool matchStepStore(const Op *op, const std::string &iv, int64_t expectedStep) {
  op = unwrapSingleDecl(op);
  if (!op || op->kind != OpKind::Store || op->symbol != iv || op->children.size() != 1)
    return false;
  const Op *rhs = op->children[0].get();
  if (!rhs || rhs->kind != OpKind::Arith || rhs->symbol != "+" || rhs->children.size() != 2)
    return false;
  return isScalarLoad(rhs->children[0].get(), iv) &&
         isConstIntValue(rhs->children[1].get(), expectedStep);
}

void collectArrayAccessesImpl(const Op *op, std::vector<Access> &out) {
  if (!op)
    return;

  if ((op->kind == OpKind::Load || op->kind == OpKind::Store) &&
      !op->symbol.empty() && !op->children.empty()) {
    const bool isStore = op->kind == OpKind::Store;
    const size_t indexCount = isStore ? op->children.size() - 1 : op->children.size();
    Access access;
    access.op = op;
    access.base = op->symbol;
    access.isStore = isStore;
    access.indices.reserve(indexCount);
    bool valid = true;
    for (size_t i = 0; i < indexCount; i++) {
      Expr idx = analyzeExpr(op->children[i].get());
      valid = valid && idx.valid;
      access.indices.push_back(std::move(idx));
    }
    if (valid)
      out.push_back(std::move(access));
  }

  for (const auto &child : op->children)
    collectArrayAccessesImpl(child.get(), out);
}

Expr renameExpr(Expr expr, const std::unordered_map<std::string, std::string> &renames) {
  if (!expr.valid)
    return invalidExpr();
  std::map<std::string, int64_t> renamed;
  for (const auto &[sym, coeff] : expr.coeffs) {
    auto it = renames.find(sym);
    renamed[it == renames.end() ? sym : it->second] += coeff;
  }
  expr.coeffs = std::move(renamed);
  normalize(expr);
  return expr;
}

bool exprEqual(const Expr &lhs, const Expr &rhs) {
  return lhs.valid && rhs.valid && lhs.constant == rhs.constant && lhs.coeffs == rhs.coeffs;
}

Expr diffExpr(const Expr &lhs, const Expr &rhs) {
  return addExpr(lhs, rhs, -1);
}

bool accessPairSameIteration(const Access &a, const Access &b,
                             const std::unordered_map<std::string, std::string> &renames) {
  if (a.indices.size() != b.indices.size())
    return false;
  for (size_t i = 0; i < a.indices.size(); i++) {
    if (!exprEqual(a.indices[i], renameExpr(b.indices[i], renames)))
      return false;
  }
  return true;
}

bool exprMentionsAny(const Expr &expr, const std::unordered_set<std::string> &symbols) {
  if (!expr.valid)
    return true;
  for (const auto &[sym, _] : expr.coeffs)
    if (symbols.count(sym))
      return true;
  return false;
}

bool accessPairProvablyDisjoint(const Access &a, const Access &b,
                                const std::unordered_set<std::string> &loopIVs) {
  if (a.indices.size() != b.indices.size())
    return false;

  for (size_t i = 0; i < a.indices.size(); i++) {
    if (!a.indices[i].valid || !b.indices[i].valid)
      return false;
    if (exprMentionsAny(a.indices[i], loopIVs) || exprMentionsAny(b.indices[i], loopIVs))
      continue;
    Expr diff = diffExpr(a.indices[i], b.indices[i]);
    if (diff.valid && diff.coeffs.empty() && diff.constant != 0)
      return true;
  }
  return false;
}

void collectScalarLoads(const Op *op, std::unordered_set<std::string> &loads) {
  if (!op)
    return;
  if (op->kind == OpKind::Load && op->children.empty() && !op->symbol.empty())
    loads.insert(op->symbol);
  for (const auto &child : op->children)
    collectScalarLoads(child.get(), loads);
}

bool writesAnyScalar(const Op *op, const std::unordered_set<std::string> &symbols) {
  if (!op)
    return false;
  const bool scalarStore = op->kind == OpKind::Store && op->children.size() == 1;
  const bool scalarDecl = op->kind == OpKind::VarDecl;
  if ((scalarStore || scalarDecl) && symbols.count(op->symbol))
    return true;
  for (const auto &child : op->children)
    if (writesAnyScalar(child.get(), symbols))
      return true;
  return false;
}

}  // namespace

Expr analyzeExpr(const Op *op) {
  if (!op)
    return invalidExpr();

  if (op->kind == OpKind::ConstInt && op->hasIntValue)
    return constantExpr(op->intValue);

  if (op->kind == OpKind::Load && op->children.empty())
    return symbolExpr(op->symbol);

  if (op->kind != OpKind::Arith || op->children.size() != 2)
    return invalidExpr();

  const Op *lhsOp = op->children[0].get();
  const Op *rhsOp = op->children[1].get();
  Expr lhs = analyzeExpr(lhsOp);
  Expr rhs = analyzeExpr(rhsOp);

  if (op->symbol == "+")
    return addExpr(std::move(lhs), rhs);
  if (op->symbol == "-")
    return addExpr(std::move(lhs), rhs, -1);
  if (op->symbol == "*") {
    int64_t lhsConst = 0;
    int64_t rhsConst = 0;
    if (isConstInt(lhsOp, lhsConst))
      return scaleExpr(std::move(rhs), lhsConst);
    if (isConstInt(rhsOp, rhsConst))
      return scaleExpr(std::move(lhs), rhsConst);
  }

  return invalidExpr();
}

bool matchCanonicalLoop(const Op *op, CanonicalLoop &loop) {
  if (!op || op->kind != OpKind::While || op->children.size() < 2)
    return false;
  const Op *cond = op->children[0].get();
  const Op *body = op->children[1].get();
  if (!cond || cond->kind != OpKind::Cmp || cond->symbol != "<" || cond->children.size() != 2)
    return false;
  const Op *lhs = cond->children[0].get();
  if (!lhs || lhs->kind != OpKind::Load || !lhs->children.empty() || lhs->symbol.empty())
    return false;
  if (!body || body->kind != OpKind::Block || body->children.empty())
    return false;
  const Op *step = body->children.back().get();
  if (!matchStepStore(step, lhs->symbol, 1))
    return false;

  loop.iv = lhs->symbol;
  loop.bound = cond->children[1].get();
  loop.body = body;
  loop.step = step;
  return true;
}

std::vector<Access> collectArrayAccesses(const Op *op) {
  std::vector<Access> accesses;
  collectArrayAccessesImpl(op, accesses);
  return accesses;
}

bool exprUsesAny(const Expr &expr, const std::unordered_set<std::string> &symbols) {
  return exprMentionsAny(expr, symbols);
}

bool opWritesAnyScalarUsedBy(const Op *op, const Op *expr) {
  std::unordered_set<std::string> loads;
  collectScalarLoads(expr, loads);
  return writesAnyScalar(op, loads);
}

bool fusionMemorySafe(const Op *loopAOp, const Op *loopBOp) {
  CanonicalLoop loopA;
  CanonicalLoop loopB;
  if (!matchCanonicalLoop(loopAOp, loopA) || !matchCanonicalLoop(loopBOp, loopB))
    return false;

  std::vector<Access> aAccesses = collectArrayAccesses(loopA.body);
  std::vector<Access> bAccesses = collectArrayAccesses(loopB.body);
  std::unordered_map<std::string, std::string> renameBToA = {{loopB.iv, loopA.iv}};
  std::unordered_set<std::string> loopIVs = {loopA.iv, loopB.iv};

  for (const Access &a : aAccesses) {
    for (const Access &b : bAccesses) {
      if (a.base != b.base)
        continue;
      if (!a.isStore && !b.isStore)
        continue;
      if (a.indices.size() != b.indices.size())
        return false;

      if (accessPairSameIteration(a, b, renameBToA))
        continue;
      if (accessPairProvablyDisjoint(a, b, loopIVs))
        continue;

      return false;
    }
  }

  return true;
}

}  // namespace sys::hir::affine
