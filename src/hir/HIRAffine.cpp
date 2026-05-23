#include "HIRAffine.h"

#include "../utils/presburger/BasicSet.h"

#include <algorithm>
#include <limits>
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

std::optional<int64_t> constIntValue(const Op *op) {
  int64_t value = 0;
  if (!isConstInt(op, value))
    return std::nullopt;
  return value;
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
  return (isScalarLoad(rhs->children[0].get(), iv) &&
          isConstIntValue(rhs->children[1].get(), expectedStep)) ||
         (isConstIntValue(rhs->children[0].get(), expectedStep) &&
          isScalarLoad(rhs->children[1].get(), iv));
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

int64_t abs64(int64_t value) {
  return value < 0 ? -value : value;
}

int64_t gcd64(int64_t a, int64_t b) {
  a = abs64(a);
  b = abs64(b);
  while (b != 0) {
    int64_t next = a % b;
    a = b;
    b = next;
  }
  return a;
}

ReorderedDep solveDifferenceEqualities(const std::vector<std::vector<int>> &equalities) {
  std::optional<int64_t> fixedA;
  std::optional<int64_t> fixedB;
  std::optional<int64_t> fixedDiff;

  for (const auto &row : equalities) {
    if (row.size() != 3)
      return ReorderedDep::Unknown;
    int64_t ca = row[0];
    int64_t cb = row[1];
    int64_t c = row[2];

    const int64_t coeffGcd = gcd64(ca, cb);
    if (coeffGcd > 1) {
      if (c % coeffGcd != 0)
        return ReorderedDep::No;
      ca /= coeffGcd;
      cb /= coeffGcd;
      c /= coeffGcd;
    }

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

  if (fixedA && fixedB) {
    const int64_t actualDiff = *fixedA - *fixedB;
    if (fixedDiff && *fixedDiff != actualDiff)
      return ReorderedDep::No;
    return actualDiff >= 1 ? ReorderedDep::May : ReorderedDep::No;
  }

  if (fixedDiff) {
    if (*fixedDiff < 1)
      return ReorderedDep::No;
    return ReorderedDep::May;
  }

  return ReorderedDep::May;
}

ReorderedDep reorderedDependenceViaPresburger(const Access &a, const Access &b,
                                              const std::string &aIV,
                                              const std::string &bIV,
                                              std::optional<int64_t> aBegin,
                                              std::optional<int64_t> aEnd,
                                              std::optional<int64_t> bBegin,
                                              std::optional<int64_t> bEnd,
                                              bool &queried) {
  queried = false;
  if (a.indices.size() != b.indices.size())
    return ReorderedDep::May;

  pres::BasicSet set;

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
  if (aBegin)
    set.addConstraint({1, 0, static_cast<int>(-*aBegin)});
  if (aEnd)
    set.addConstraint({-1, 0, static_cast<int>(*aEnd - 1)});
  if (bBegin)
    set.addConstraint({0, 1, static_cast<int>(-*bBegin)});
  if (bEnd)
    set.addConstraint({0, -1, static_cast<int>(*bEnd - 1)});

  queried = true;
  if (solveDifferenceEqualities(equalityRows) == ReorderedDep::No)
    return ReorderedDep::No;
  if (set.empty())
    return ReorderedDep::No;
  return ReorderedDep::May;
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

const Op *initValue(const Op *op) {
  op = unwrapSingleDecl(op);
  if (!op || op->children.empty())
    return nullptr;
  return op->children[0].get();
}

struct VarOrder {
  std::vector<std::string> names;
  std::unordered_map<std::string, int> index;

  int add(const std::string &name) {
    auto it = index.find(name);
    if (it != index.end())
      return it->second;
    int next = static_cast<int>(names.size());
    names.push_back(name);
    index[name] = next;
    return next;
  }

  int size() const { return static_cast<int>(names.size()); }
};

bool appendRowFromCoeffs(const CoeffMap &coeffs, int64_t constant,
                         const VarOrder &vars, std::vector<int> &row) {
  if (constant < std::numeric_limits<int>::min() ||
      constant > std::numeric_limits<int>::max())
    return false;
  row.assign(vars.size() + 1, 0);
  for (const auto &[sym, coeff] : coeffs) {
    if (coeff == 0)
      continue;
    if (coeff < std::numeric_limits<int>::min() ||
        coeff > std::numeric_limits<int>::max())
      return false;
    auto it = vars.index.find(sym);
    if (it == vars.index.end())
      return false;
    int idx = it->second;
    int64_t next = static_cast<int64_t>(row[idx]) + coeff;
    if (next < std::numeric_limits<int>::min() ||
        next > std::numeric_limits<int>::max())
      return false;
    row[idx] = static_cast<int>(next);
  }
  row.back() = static_cast<int>(constant);
  return true;
}

bool addConstraint(pres::BasicSet &set, const VarOrder &vars,
                   const CoeffMap &coeffs, int64_t constant) {
  std::vector<int> row;
  if (!appendRowFromCoeffs(coeffs, constant, vars, row))
    return false;
  set.addConstraint(row);
  return true;
}

void collectExprSymbols(const Expr &expr, std::unordered_set<std::string> &out) {
  if (!expr.valid)
    return;
  for (const auto &[sym, _] : expr.coeffs)
    out.insert(sym);
}

void addExprToCoeffs(CoeffMap &coeffs, int64_t &constant, const Expr &expr,
                     int64_t sign, const std::string &outerIV,
                     const std::string &innerIV, const std::string &outerVar,
                     const std::string &innerVar,
                     const std::unordered_map<std::string, std::string> &paramRenames) {
  constant += sign * expr.constant;
  for (const auto &[sym, coeff] : expr.coeffs) {
    std::string mapped;
    if (sym == outerIV)
      mapped = outerVar;
    else if (sym == innerIV)
      mapped = innerVar;
    else {
      auto it = paramRenames.find(sym);
      mapped = (it == paramRenames.end()) ? sym : it->second;
    }
    coeffs[mapped] += sign * coeff;
  }
}

void addExprToCoeffsND(CoeffMap &coeffs, int64_t &constant, const Expr &expr,
                       int64_t sign, const std::vector<std::string> &ivs,
                       const std::vector<std::string> &iterVars,
                       const std::unordered_map<std::string, std::string> &paramRenames) {
  constant += sign * expr.constant;
  for (const auto &[sym, coeff] : expr.coeffs) {
    std::string mapped;
    for (size_t i = 0; i < ivs.size(); i++) {
      if (sym == ivs[i]) {
        mapped = iterVars[i];
        break;
      }
    }
    if (mapped.empty()) {
      auto it = paramRenames.find(sym);
      mapped = (it == paramRenames.end()) ? sym : it->second;
    }
    coeffs[mapped] += sign * coeff;
  }
}

bool makeOrderRow(const VarOrder &vars,
                  const std::string &i1, const std::string &j1,
                  const std::string &i2, const std::string &j2,
                  int coeffI1, int coeffJ1, int coeffI2, int coeffJ2,
                  int constant, std::vector<int> &row) {
  row.assign(vars.size() + 1, 0);
  row[vars.index.at(i1)] = coeffI1;
  row[vars.index.at(j1)] = coeffJ1;
  row[vars.index.at(i2)] = coeffI2;
  row[vars.index.at(j2)] = coeffJ2;
  row.back() = constant;
  return true;
}

bool hasInterchangeViolation(const pres::BasicSet &base, const VarOrder &vars,
                             const std::string &i1, const std::string &j1,
                             const std::string &i2, const std::string &j2) {
  std::vector<int> row;
  std::vector<std::vector<int>> origA;
  std::vector<std::vector<int>> origB;
  std::vector<std::vector<int>> violC;
  std::vector<std::vector<int>> violD;

  // Original order case A: i1 < i2  -> -i1 + i2 - 1 >= 0
  makeOrderRow(vars, i1, j1, i2, j2, -1, 0, 1, 0, -1, row);
  origA.push_back(row);

  // Original order case B: i1 == i2, j1 <= j2
  makeOrderRow(vars, i1, j1, i2, j2, 1, 0, -1, 0, 0, row);
  origB.push_back(row);
  makeOrderRow(vars, i1, j1, i2, j2, -1, 0, 1, 0, 0, row);
  origB.push_back(row);
  makeOrderRow(vars, i1, j1, i2, j2, 0, -1, 0, 1, 0, row);
  origB.push_back(row);

  // Violation case C: j1 > j2 -> j1 - j2 - 1 >= 0
  makeOrderRow(vars, i1, j1, i2, j2, 0, 1, 0, -1, -1, row);
  violC.push_back(row);

  // Violation case D: j1 == j2, i1 > i2
  makeOrderRow(vars, i1, j1, i2, j2, 0, 1, 0, -1, 0, row);
  violD.push_back(row);
  makeOrderRow(vars, i1, j1, i2, j2, 0, -1, 0, 1, 0, row);
  violD.push_back(row);
  makeOrderRow(vars, i1, j1, i2, j2, 1, 0, -1, 0, -1, row);
  violD.push_back(row);

  auto hasSolution = [&](const std::vector<std::vector<int>> &rows) {
    pres::BasicSet test = base;
    for (const auto &constraint : rows)
      test.addConstraint(constraint);
    return !test.empty();
  };

  for (const auto &orig : {origA, origB}) {
    for (const auto &viol : {violC, violD}) {
      std::vector<std::vector<int>> combined = orig;
      combined.insert(combined.end(), viol.begin(), viol.end());
      if (hasSolution(combined))
        return true;
    }
  }
  return false;
}

enum class DeltaConstraintResult {
  NoSolution,
  Applied,
  Unknown,
};

bool setDelta(std::vector<std::optional<int64_t>> &deltas, size_t dim, int64_t value) {
  if (deltas[dim])
    return *deltas[dim] == value;
  deltas[dim] = value;
  return true;
}

DeltaConstraintResult applySameDimDeltaEquality(
    const CoeffMap &coeffs, int64_t constant,
    const std::vector<std::string> &xVars,
    const std::vector<std::string> &yVars,
    std::vector<std::optional<int64_t>> &deltas) {
  std::vector<std::pair<std::string, int64_t>> nonzero;
  for (const auto &[sym, coeff] : coeffs) {
    if (coeff != 0)
      nonzero.push_back({sym, coeff});
  }

  if (nonzero.empty())
    return constant == 0 ? DeltaConstraintResult::Applied
                         : DeltaConstraintResult::NoSolution;
  if (nonzero.size() != 2)
    return DeltaConstraintResult::Unknown;

  auto findDim = [&](const std::string &sym,
                     const std::vector<std::string> &vars) -> std::optional<size_t> {
    for (size_t i = 0; i < vars.size(); i++)
      if (vars[i] == sym)
        return i;
    return std::nullopt;
  };

  std::optional<size_t> xDim;
  std::optional<size_t> yDim;
  int64_t xCoeff = 0;
  int64_t yCoeff = 0;
  for (const auto &[sym, coeff] : nonzero) {
    if (auto dim = findDim(sym, xVars)) {
      if (xDim)
        return DeltaConstraintResult::Unknown;
      xDim = *dim;
      xCoeff = coeff;
      continue;
    }
    if (auto dim = findDim(sym, yVars)) {
      if (yDim)
        return DeltaConstraintResult::Unknown;
      yDim = *dim;
      yCoeff = coeff;
      continue;
    }
    return DeltaConstraintResult::Unknown;
  }

  if (!xDim || !yDim || *xDim != *yDim)
    return DeltaConstraintResult::Unknown;

  std::optional<int64_t> delta;
  if (xCoeff == 1 && yCoeff == -1)
    delta = -constant;
  else if (xCoeff == -1 && yCoeff == 1)
    delta = constant;
  else
    return DeltaConstraintResult::Unknown;

  if (!setDelta(deltas, *xDim, *delta))
    return DeltaConstraintResult::NoSolution;
  return DeltaConstraintResult::Applied;
}

enum class DeltaRequirement {
  Any,
  Eq,
  Neg,
  Pos,
};

bool mergeRequirement(std::vector<DeltaRequirement> &requirements, size_t dim,
                      DeltaRequirement requirement) {
  DeltaRequirement &slot = requirements[dim];
  if (slot == requirement)
    return true;
  if (slot == DeltaRequirement::Any) {
    slot = requirement;
    return true;
  }
  if (requirement == DeltaRequirement::Any)
    return true;
  return false;
}

bool deltaCanSatisfy(const std::optional<int64_t> &delta,
                     DeltaRequirement requirement) {
  if (!delta)
    return true;
  switch (requirement) {
  case DeltaRequirement::Any:
    return true;
  case DeltaRequirement::Eq:
    return *delta == 0;
  case DeltaRequirement::Neg:
    return *delta < 0;
  case DeltaRequirement::Pos:
    return *delta > 0;
  }
  return false;
}

bool hasPermutationViolationFromDeltas(const std::vector<std::optional<int64_t>> &deltas,
                                       const std::vector<int> &permutation) {
  const size_t depth = deltas.size();
  for (size_t oldDiff = 0; oldDiff < depth; oldDiff++) {
    for (size_t newDiff = 0; newDiff < depth; newDiff++) {
      std::vector<DeltaRequirement> requirements(depth, DeltaRequirement::Any);
      bool ok = true;
      for (size_t d = 0; d < oldDiff; d++)
        ok = ok && mergeRequirement(requirements, d, DeltaRequirement::Eq);
      ok = ok && mergeRequirement(requirements, oldDiff, DeltaRequirement::Neg);
      for (size_t p = 0; p < newDiff; p++)
        ok = ok && mergeRequirement(requirements, permutation[p], DeltaRequirement::Eq);
      ok = ok && mergeRequirement(requirements, permutation[newDiff], DeltaRequirement::Pos);
      if (!ok)
        continue;

      bool satisfiable = true;
      for (size_t d = 0; d < depth; d++) {
        if (!deltaCanSatisfy(deltas[d], requirements[d])) {
          satisfiable = false;
          break;
        }
      }
      if (satisfiable)
        return true;
    }
  }
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
  return fusionMemorySafePresburger(loopAOp, loopBOp, nullptr, nullptr);
}

PresburgerFusionResult fusionMemorySafePresburger(const Op *loopAOp, const Op *loopBOp,
                                                  const Op *initAOp, const Op *initBOp) {
  PresburgerFusionResult result;
  CanonicalLoop loopA;
  CanonicalLoop loopB;
  if (!matchCanonicalLoop(loopAOp, loopA) || !matchCanonicalLoop(loopBOp, loopB))
    return result;

  std::optional<int64_t> initA = constIntValue(initAOp);
  std::optional<int64_t> initB = constIntValue(initBOp);
  std::optional<int64_t> boundA = constIntValue(loopA.bound);
  std::optional<int64_t> boundB = constIntValue(loopB.bound);
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
          reorderedDependenceViaPresburger(a, b, loopA.iv, loopB.iv,
                                           initA, boundA, initB, boundB, queried);
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

PresburgerInterchangeResult interchangeMemorySafePresburger(const Op *outerLoopOp,
                                                            const Op *innerLoopOp,
                                                            const Op *outerInitOp,
                                                            const Op *innerInitOp) {
  PresburgerInterchangeResult result;
  CanonicalLoop outer;
  CanonicalLoop inner;
  if (!matchCanonicalLoop(outerLoopOp, outer) || !matchCanonicalLoop(innerLoopOp, inner))
    return result;

  const Op *outerInitExpr = initValue(outerInitOp);
  const Op *innerInitExpr = initValue(innerInitOp);
  if (!outerInitExpr || !innerInitExpr) {
    result.unknown++;
    return result;
  }

  Expr outerInit = analyzeExpr(outerInitExpr);
  Expr innerInit = analyzeExpr(innerInitExpr);
  Expr outerBound = analyzeExpr(outer.bound);
  Expr innerBound = analyzeExpr(inner.bound);
  if (!outerInit.valid || !innerInit.valid || !outerBound.valid || !innerBound.valid) {
    result.unknown++;
    return result;
  }

  std::unordered_set<std::string> ivs = {outer.iv, inner.iv};
  if (exprUsesAny(outerInit, ivs) || exprUsesAny(innerInit, ivs) ||
      exprUsesAny(outerBound, ivs) || exprUsesAny(innerBound, ivs)) {
    result.unknown++;
    return result;
  }

  std::vector<Access> accesses = collectArrayAccesses(inner.body);
  if (accesses.empty()) {
    result.safe = true;
    return result;
  }

  std::unordered_set<std::string> params;
  collectExprSymbols(outerInit, params);
  collectExprSymbols(innerInit, params);
  collectExprSymbols(outerBound, params);
  collectExprSymbols(innerBound, params);
  for (const auto &access : accesses)
    for (const auto &idx : access.indices)
      collectExprSymbols(idx, params);
  params.erase(outer.iv);
  params.erase(inner.iv);

  std::unordered_map<std::string, std::string> paramRenames;
  std::vector<std::string> paramList(params.begin(), params.end());
  std::sort(paramList.begin(), paramList.end());
  for (const auto &sym : paramList)
    paramRenames.emplace(sym, "__p_" + sym);

  const std::string kI1 = "__i1";
  const std::string kJ1 = "__j1";
  const std::string kI2 = "__i2";
  const std::string kJ2 = "__j2";
  VarOrder vars;
  vars.add(kI1);
  vars.add(kJ1);
  vars.add(kI2);
  vars.add(kJ2);
  for (const auto &sym : paramList)
    vars.add(paramRenames.at(sym));

  auto addBounds = [&](pres::BasicSet &set,
                       const std::string &iVar, const std::string &jVar,
                       const std::string &outerVar, const std::string &innerVar) -> bool {
    // i >= init, i <= bound - 1
    CoeffMap coeffs;
    int64_t constant = 0;
    coeffs[iVar] = 1;
    addExprToCoeffs(coeffs, constant, outerInit, -1,
                    outer.iv, inner.iv, outerVar, innerVar, paramRenames);
    if (!addConstraint(set, vars, coeffs, constant))
      return false;
    coeffs.clear();
    constant = -1;
    coeffs[iVar] = -1;
    addExprToCoeffs(coeffs, constant, outerBound, 1,
                    outer.iv, inner.iv, outerVar, innerVar, paramRenames);
    if (!addConstraint(set, vars, coeffs, constant))
      return false;

    // j >= init, j <= bound - 1
    coeffs.clear();
    constant = 0;
    coeffs[jVar] = 1;
    addExprToCoeffs(coeffs, constant, innerInit, -1,
                    outer.iv, inner.iv, outerVar, innerVar, paramRenames);
    if (!addConstraint(set, vars, coeffs, constant))
      return false;
    coeffs.clear();
    constant = -1;
    coeffs[jVar] = -1;
    addExprToCoeffs(coeffs, constant, innerBound, 1,
                    outer.iv, inner.iv, outerVar, innerVar, paramRenames);
    if (!addConstraint(set, vars, coeffs, constant))
      return false;

    return true;
  };

  for (size_t i = 0; i < accesses.size(); i++) {
    for (size_t j = i + 1; j < accesses.size(); j++) {
      const Access &a = accesses[i];
      const Access &b = accesses[j];
      if (a.base != b.base)
        continue;
      if (!a.isStore && !b.isStore)
        continue;
      if (a.indices.size() != b.indices.size()) {
        result.unknown++;
        return result;
      }

      pres::BasicSet base;
      if (!addBounds(base, kI1, kJ1, kI1, kJ1) ||
          !addBounds(base, kI2, kJ2, kI2, kJ2)) {
        result.unknown++;
        return result;
      }

      for (size_t dim = 0; dim < a.indices.size(); dim++) {
        CoeffMap coeffs;
        int64_t constant = 0;
        addExprToCoeffs(coeffs, constant, a.indices[dim], 1,
                        outer.iv, inner.iv, kI1, kJ1, paramRenames);
        addExprToCoeffs(coeffs, constant, b.indices[dim], -1,
                        outer.iv, inner.iv, kI2, kJ2, paramRenames);
        if (!addConstraint(base, vars, coeffs, constant)) {
          result.unknown++;
          return result;
        }
        for (auto &entry : coeffs)
          entry.second = -entry.second;
        if (!addConstraint(base, vars, coeffs, -constant)) {
          result.unknown++;
          return result;
        }
      }

      result.queries++;
      if (hasInterchangeViolation(base, vars, kI1, kJ1, kI2, kJ2)) {
        result.mayViolatingDependence++;
        return result;
      }
      result.noViolatingDependence++;
    }
  }

  result.safe = true;
  return result;
}

PresburgerInterchangeResult permutationMemorySafePresburger(
    const std::vector<const Op*> &loopOps,
    const std::vector<const Op*> &initOps,
    const std::vector<int> &permutation) {
  PresburgerInterchangeResult result;
  const size_t depth = loopOps.size();
  if (depth == 0 || depth != initOps.size() || depth != permutation.size()) {
    result.unknown++;
    return result;
  }

  std::vector<CanonicalLoop> loops(depth);
  std::vector<std::string> ivs;
  std::vector<Expr> initExprs;
  std::vector<Expr> boundExprs;
  ivs.reserve(depth);
  initExprs.reserve(depth);
  boundExprs.reserve(depth);

  for (size_t i = 0; i < depth; i++) {
    if (!matchCanonicalLoop(loopOps[i], loops[i])) {
      result.unknown++;
      return result;
    }
    const Op *initExprOp = initValue(initOps[i]);
    if (!initExprOp) {
      result.unknown++;
      return result;
    }
    ivs.push_back(loops[i].iv);
    initExprs.push_back(analyzeExpr(initExprOp));
    boundExprs.push_back(analyzeExpr(loops[i].bound));
    if (!initExprs.back().valid || !boundExprs.back().valid) {
      result.unknown++;
      return result;
    }
  }

  std::unordered_set<std::string> ivSet(ivs.begin(), ivs.end());
  for (size_t i = 0; i < depth; i++) {
    if (exprUsesAny(initExprs[i], ivSet) || exprUsesAny(boundExprs[i], ivSet)) {
      result.unknown++;
      return result;
    }
  }

  std::vector<int> sortedPerm = permutation;
  std::sort(sortedPerm.begin(), sortedPerm.end());
  for (size_t i = 0; i < depth; i++) {
    if (sortedPerm[i] != (int) i) {
      result.unknown++;
      return result;
    }
  }

  std::vector<Access> accesses = collectArrayAccesses(loops.back().body);
  if (accesses.empty()) {
    result.safe = true;
    return result;
  }

  std::unordered_set<std::string> params;
  for (size_t i = 0; i < depth; i++) {
    collectExprSymbols(initExprs[i], params);
    collectExprSymbols(boundExprs[i], params);
  }
  for (const auto &access : accesses)
    for (const auto &idx : access.indices)
      collectExprSymbols(idx, params);
  for (const std::string &iv : ivs)
    params.erase(iv);

  std::unordered_map<std::string, std::string> paramRenames;
  std::vector<std::string> paramList(params.begin(), params.end());
  std::sort(paramList.begin(), paramList.end());
  for (const auto &sym : paramList)
    paramRenames.emplace(sym, "__p_" + sym);

  std::vector<std::string> xVars;
  std::vector<std::string> yVars;
  xVars.reserve(depth);
  yVars.reserve(depth);
  for (size_t i = 0; i < depth; i++) {
    xVars.push_back("__x" + std::to_string(i));
    yVars.push_back("__y" + std::to_string(i));
  }

  for (const Access &a : accesses) {
    for (const Access &b : accesses) {
      if (a.base != b.base)
        continue;
      if (!a.isStore && !b.isStore)
        continue;
      if (a.indices.size() != b.indices.size()) {
        result.unknown++;
        return result;
      }

      // First use a same-dimension dependence direction solver. It covers the
      // common affine loop-nest cases (`A[i][j][k+c]`) without invoking the
      // heavier feasibility engine for every access pair.
      std::vector<std::optional<int64_t>> deltas(depth);
      bool noSolution = false;

      for (size_t dim = 0; dim < a.indices.size(); dim++) {
        CoeffMap coeffs;
        int64_t constant = 0;
        addExprToCoeffsND(coeffs, constant, a.indices[dim], 1,
                          ivs, xVars, paramRenames);
        addExprToCoeffsND(coeffs, constant, b.indices[dim], -1,
                          ivs, yVars, paramRenames);
        DeltaConstraintResult applied =
            applySameDimDeltaEquality(coeffs, constant, xVars, yVars, deltas);
        if (applied == DeltaConstraintResult::NoSolution) {
          noSolution = true;
          break;
        }
        if (applied == DeltaConstraintResult::Unknown) {
          result.unknown++;
          return result;
        }
      }
      if (noSolution) {
        result.noViolatingDependence++;
        continue;
      }

      result.queries++;
      if (hasPermutationViolationFromDeltas(deltas, permutation)) {
        result.mayViolatingDependence++;
        return result;
      }
      result.noViolatingDependence++;
    }
  }

  result.safe = true;
  return result;
}

}  // namespace sys::hir::affine
