#include "Analysis.h"

#include <map>
#include <set>
#include <unordered_set>

using namespace sys;

namespace {

int clampToInt(long long v) {
  if (v > INT_MAX)
    return INT_MAX;
  if (v < INT_MIN)
    return INT_MIN;
  return (int) v;
}

struct LinearExpr {
  long long constant = 0;
  std::map<Op*, long long> terms;
  bool valid = true;
};

void mergeLinear(LinearExpr &dst, const LinearExpr &src, long long scale = 1) {
  if (!dst.valid || !src.valid)
    return;
  dst.constant += src.constant * scale;
  for (auto [op, coeff] : src.terms)
    dst.terms[op] += coeff * scale;
}

bool collectLinearOffset(Op *op, LinearExpr &out, std::unordered_set<Op*> &visiting) {
  if (!out.valid)
    return false;
  if (!op) {
    out.valid = false;
    return false;
  }
  if (visiting.count(op)) {
    out.valid = false;
    return false;
  }

  if (isa<IntOp>(op)) {
    out.constant += V(op);
    return true;
  }

  if (op->getResultType() != Value::i32) {
    out.valid = false;
    return false;
  }

  visiting.insert(op);
  bool ok = true;

  if (isa<AddIOp>(op) || isa<SubIOp>(op)) {
    LinearExpr lhs, rhs;
    ok &= collectLinearOffset(op->DEF(0), lhs, visiting);
    ok &= collectLinearOffset(op->DEF(1), rhs, visiting);
    if (ok) {
      mergeLinear(out, lhs, 1);
      mergeLinear(out, rhs, isa<AddIOp>(op) ? 1 : -1);
    }
  } else if (isa<MinusOp>(op)) {
    LinearExpr x;
    ok &= collectLinearOffset(op->DEF(), x, visiting);
    if (ok)
      mergeLinear(out, x, -1);
  } else if (isa<MulIOp>(op)) {
    Op *lhs = op->DEF(0), *rhs = op->DEF(1);
    if (isa<IntOp>(lhs) || isa<IntOp>(rhs)) {
      int factor = isa<IntOp>(lhs) ? V(lhs) : V(rhs);
      Op *other = isa<IntOp>(lhs) ? rhs : lhs;
      LinearExpr x;
      ok &= collectLinearOffset(other, x, visiting);
      if (ok)
        mergeLinear(out, x, factor);
    } else {
      out.terms[op] += 1;
    }
  } else if (isa<PhiOp>(op)) {
    out.terms[op] += 1;
  } else {
    out.terms[op] += 1;
  }

  visiting.erase(op);
  if (!ok)
    out.valid = false;
  return out.valid;
}

bool flattenAddressExpr(Op *op,
                        Op *&base,
                        LinearExpr &offset,
                        bool &touched,
                        std::unordered_set<Op*> &visitingAddr) {
  if (!op)
    return false;
  if (visitingAddr.count(op))
    return false;
  if (!isa<AddLOp>(op)) {
    base = op;
    return true;
  }

  visitingAddr.insert(op);
  touched = true;
  bool ok = flattenAddressExpr(op->DEF(0), base, offset, touched, visitingAddr);
  if (ok) {
    std::unordered_set<Op*> visiting;
    ok = collectLinearOffset(op->DEF(1), offset, visiting);
  }
  visitingAddr.erase(op);
  return ok && offset.valid;
}

bool hasMeaningfulRewrite(AddLOp *op, Op *base, const LinearExpr &expr, bool touched) {
  if (!touched)
    return false;
  if (base != op->DEF(0))
    return true;

  long long nonZeroTerms = 0;
  Op *single = nullptr;
  long long coeff = 0;
  for (auto [term, c] : expr.terms) {
    if (!c)
      continue;
    nonZeroTerms++;
    single = term;
    coeff = c;
  }
  if (expr.constant != 0)
    return true;
  if (nonZeroTerms != 1)
    return true;
  return !(single == op->DEF(1) && coeff == 1);
}

Op *buildOffsetValue(Builder &builder, const LinearExpr &expr) {
  Op *acc = nullptr;
  auto append = [&](Op *term) {
    if (!acc)
      acc = term;
    else
      acc = builder.create<AddIOp>(std::vector<Value> { acc, term });
  };

  for (auto [term, coeff] : expr.terms) {
    if (!coeff)
      continue;
    Op *scaled = term;
    if (coeff != 1) {
      auto c = builder.create<IntOp>({ new IntAttr(clampToInt(coeff)) });
      scaled = builder.create<MulIOp>({ term, c });
    }
    append(scaled);
  }
  if (expr.constant != 0) {
    auto c = builder.create<IntOp>({ new IntAttr(clampToInt(expr.constant)) });
    append(c);
  }
  if (!acc)
    acc = builder.create<IntOp>({ new IntAttr(0) });
  return acc;
}

}  // namespace

void ArrayStrideAnalysis::run() {
  auto funcs = collectFuncs();
  Builder builder;

  for (auto func : funcs) {
    auto addls = func->findAll<AddLOp>();
    for (auto op : addls) {
      auto addl = cast<AddLOp>(op);
      if (!addl || !addl->getParent())
        continue;
      Op *base = nullptr;
      LinearExpr expr;
      bool touched = false;
      std::unordered_set<Op*> visitingAddr;
      if (!flattenAddressExpr(addl, base, expr, touched, visitingAddr))
        continue;
      if (!base || !expr.valid)
        continue;

      for (auto it = expr.terms.begin(); it != expr.terms.end();) {
        if (it->second == 0)
          it = expr.terms.erase(it);
        else
          ++it;
      }

      if (!hasMeaningfulRewrite(addl, base, expr, touched))
        continue;

      builder.setBeforeOp(addl);
      if (expr.terms.empty() && expr.constant == 0) {
        addl->replaceAllUsesWith(base);
        addl->erase();
      } else {
        auto *offset = buildOffsetValue(builder, expr);
        builder.replace<AddLOp>(addl, std::vector<Value> { base, offset });
      }
      rewritten++;
    }
  }
}
