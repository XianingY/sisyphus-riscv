#include "HIRAffine.h"

#include "../utils/presburger/BasicSet.h"

#include <algorithm>
#include <optional>

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

  const bool isArrayLoad = op->kind == OpKind::Load && !op->children.empty();
  const bool isArrayStore = op->kind == OpKind::Store && op->children.size() > 1;
  if ((isArrayLoad || isArrayStore) && !op->symbol.empty()) {
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

enum class ReorderedDep {
  No,
  May,
  Unknown,
};

using CoeffMap = std::map<std::string, int64_t>;

void addSeparatedExpr(CoeffMap &coeffs, int64_t &constant, const Expr &expr,
                      int64_t sign, const std::string &iv,
                      const std::string &separatedIV) {
  constant += sign * expr.constant;
  for (const auto &[sym, coeff] : expr.coeffs) {
    const std::string &target = sym == iv ? separatedIV : sym;
    coeffs[target] += sign * coeff;
  }
}

bool makePresburgerRow(const CoeffMap &coeffs, int64_t constant,
                       std::vector<int> &row) {
  row.assign(3, 0);
  for (const auto &[sym, coeff] : coeffs) {
    if (coeff == 0)
      continue;
    if (sym == "__a_iter") {
      row[0] += static_cast<int>(coeff);
    } else if (sym == "__b_iter") {
      row[1] += static_cast<int>(coeff);
    } else {
      return false;
    }
  }
  row[2] = static_cast<int>(constant);
  return true;
}

bool setFixedValue(std::optional<int64_t> &slot, int64_t value) {
  if (slot.has_value())
    return *slot == value;
  slot = value;
  return true;
}

ReorderedDep solveDifferenceEqualities(const std::vector<std::vector<int>> &equalities) {
  std::optional<int64_t> fixedA;
  std::optional<int64_t> fixedB;
  std::optional<int64_t> fixedDiff;

  for (const auto &row : equalities) {
    if (row.size() != 3)
      return ReorderedDep::Unknown;
    const int64_t ca = row[0];
    const int64_t cb = row[1];
    const int64_t c = row[2];

    if (ca == 0 && cb == 0) {
      if (c != 0)
        return ReorderedDep::No;
      continue;
    }
    if (ca == 1 && cb == -1) {
      if (!setFixedValue(fixedDiff, -c))
        return ReorderedDep::No;
      continue;
    }
    if (ca == -1 && cb == 1) {
      if (!setFixedValue(fixedDiff, c))
        return ReorderedDep::No;
      continue;
    }
    if (ca == 1 && cb == 0) {
      if (!setFixedValue(fixedA, -c))
        return ReorderedDep::No;
      continue;
    }
    if (ca == -1 && cb == 0) {
      if (!setFixedValue(fixedA, c))
        return ReorderedDep::No;
      continue;
    }
    if (ca == 0 && cb == 1) {
      if (!setFixedValue(fixedB, -c))
        return ReorderedDep::No;
      continue;
    }
    if (ca == 0 && cb == -1) {
      if (!setFixedValue(fixedB, c))
        return ReorderedDep::No;
      continue;
    }
    return ReorderedDep::Unknown;
  }

  if (fixedA && *fixedA < 0)
    return ReorderedDep::No;
  if (fixedB && *fixedB < 0)
    return ReorderedDep::No;

  if (fixedA && fixedB) {
    const int64_t actualDiff = *fixedA - *fixedB;
    if (fixedDiff && *fixedDiff != actualDiff)
      return ReorderedDep::No;
    return actualDiff >= 1 ? ReorderedDep::May : ReorderedDep::No;
  }

  if (fixedDiff) {
    if (*fixedDiff < 1)
      return ReorderedDep::No;
    if (fixedA)
      return *fixedA - *fixedDiff >= 0 ? ReorderedDep::May : ReorderedDep::No;
    if (fixedB)
      return *fixedB + *fixedDiff >= 0 ? ReorderedDep::May : ReorderedDep::No;
    return ReorderedDep::May;
  }

  if (fixedA)
    return *fixedA >= 1 ? ReorderedDep::May : ReorderedDep::No;
  if (fixedB)
    return *fixedB >= 0 ? ReorderedDep::May : ReorderedDep::No;
  return ReorderedDep::May;
}

ReorderedDep reorderedDependenceViaPresburger(const Access &a, const Access &b,
                                              const std::string &aIV,
                                              const std::string &bIV,
                                              bool &queried) {
  queried = false;
  if (a.indices.size() != b.indices.size())
    return ReorderedDep::May;

  pres::BasicSet set;
  // BasicSet uses non-negative variables. Add explicit lower-bound rows too so
  // the constructed relation is self-documenting: a_iter >= 0, b_iter >= 0.
  set.addConstraint({1, 0, 0});
  set.addConstraint({0, 1, 0});

  std::vector<std::vector<int>> equalityRows;
  for (size_t i = 0; i < a.indices.size(); i++) {
    CoeffMap coeffs;
    int64_t constant = 0;
    addSeparatedExpr(coeffs, constant, a.indices[i], 1, aIV, "__a_iter");
    addSeparatedExpr(coeffs, constant, b.indices[i], -1, bIV, "__b_iter");

    std::vector<int> row;
    if (!makePresburgerRow(coeffs, constant, row))
      return ReorderedDep::Unknown;
    equalityRows.push_back(row);
    set.addConstraint(row);
    for (int &value : row)
      value = -value;
    set.addConstraint(row);
  }

  // Fusion changes the relative order only for B(iter_b) before A(iter_a)
  // when iter_a > iter_b. If no overlapping access pair exists under this
  // constraint, the fusion cannot reverse a real memory dependence.
  set.addConstraint({1, -1, -1});
  queried = true;
  return solveDifferenceEqualities(equalityRows);
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

bool hasAffineArrayAccessUsing(const Op *op, const std::string &symbol, int minRank) {
  std::vector<Access> accesses = collectArrayAccesses(op);
  std::unordered_set<std::string> symbols = {symbol};
  for (const Access &access : accesses) {
    if ((int) access.indices.size() < minRank)
      continue;
    for (const Expr &idx : access.indices)
      if (exprMentionsAny(idx, symbols))
        return true;
  }
  return false;
}

bool fusionMemorySafe(const Op *loopAOp, const Op *loopBOp) {
  return fusionMemorySafePresburger(loopAOp, loopBOp).safe;
}

PresburgerFusionResult fusionMemorySafePresburger(const Op *loopAOp, const Op *loopBOp) {
  PresburgerFusionResult result;
  CanonicalLoop loopA;
  CanonicalLoop loopB;
  if (!matchCanonicalLoop(loopAOp, loopA) || !matchCanonicalLoop(loopBOp, loopB))
    return result;

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

      bool queried = false;
      ReorderedDep dep =
          reorderedDependenceViaPresburger(a, b, loopA.iv, loopB.iv, queried);
      if (queried)
        result.queries++;
      if (dep == ReorderedDep::No) {
        result.noReorderedDependence++;
        continue;
      }
      if (dep == ReorderedDep::May) {
        result.mayReorderedDependence++;
        return result;
      }

      result.unknown++;
      if (a.indices.size() != b.indices.size())
        return result;
      if (accessPairSameIteration(a, b, renameBToA))
        continue;
      if (accessPairProvablyDisjoint(a, b, loopIVs))
        continue;

      return result;
    }
  }

  result.safe = true;
  return result;
}

}  // namespace sys::hir::affine
