#include "SmtProver.h"
#include "../codegen/Attrs.h"

#include <unordered_map>

using namespace sys;
using namespace sys::smt_prover;

namespace {

class Lowerer {
  ::smt::BvExprContext &ctx;
  std::unordered_map<Op *, ::smt::BvExpr *> memo;
  bool failed = false;
  int depth = 0;

  ::smt::BvExpr *bin(::smt::BvExpr::Type t, Op *op) {
    auto l = lower(op->DEF(0));
    auto r = lower(op->DEF(1));
    if (!l || !r) return nullptr;
    return ctx.create(t, l, r);
  }

public:
  Lowerer(::smt::BvExprContext &ctx) : ctx(ctx) {}

  ::smt::BvExpr *lower(Op *op) {
    if (!op || failed || depth > 128) { failed = true; return nullptr; }
    auto it = memo.find(op);
    if (it != memo.end()) return it->second;
    depth++;

    ::smt::BvExpr *e = nullptr;
    if (isa<IntOp>(op)) {
      e = ctx.create(::smt::BvExpr::Const, V(op));
    } else if (isa<GetArgOp>(op)) {
      e = ctx.create(::smt::BvExpr::Var, "arg" + std::to_string(V(op)));
    } else if (isa<AddIOp>(op)) {
      e = bin(::smt::BvExpr::Add, op);
    } else if (isa<SubIOp>(op)) {
      e = bin(::smt::BvExpr::Sub, op);
    } else if (isa<MulIOp>(op)) {
      e = bin(::smt::BvExpr::Mul, op);
    } else if (isa<AndIOp>(op)) {
      e = bin(::smt::BvExpr::And, op);
    } else if (isa<OrIOp>(op)) {
      e = bin(::smt::BvExpr::Or, op);
    } else if (isa<XorIOp>(op)) {
      e = bin(::smt::BvExpr::Xor, op);
    } else if (isa<LShiftOp>(op)) {
      // RHS must be an IntOp because Lsh in BvExpr takes an int amount.
      auto rhs = op->DEF(1);
      if (rhs && isa<IntOp>(rhs)) {
        auto l = lower(op->DEF(0));
        if (l) e = ctx.create(::smt::BvExpr::Lsh, l, V(rhs));
      }
    } else if (isa<RShiftOp>(op)) {
      auto rhs = op->DEF(1);
      if (rhs && isa<IntOp>(rhs)) {
        auto l = lower(op->DEF(0));
        if (l) e = ctx.create(::smt::BvExpr::Rsh, l, V(rhs));
      }
    } else if (isa<MinusOp>(op)) {
      auto l = lower(op->DEF(0));
      if (l) e = ctx.create(::smt::BvExpr::Minus, l);
    } else if (isa<NotOp>(op)) {
      auto l = lower(op->DEF(0));
      if (l) e = ctx.create(::smt::BvExpr::Not, l);
    }

    depth--;
    if (!e) { failed = true; return nullptr; }
    memo[op] = e;
    return e;
  }

  bool ok() const { return !failed; }
};

}  // namespace

Result sys::smt_prover::tryProveEqualI32(Op *a, Op *b) {
  if (!a || !b) return Result::Unknown;
  if (a == b) return Result::Equal;
  if (a->getResultType() != Value::i32 || b->getResultType() != Value::i32)
    return Result::Unknown;

  ::smt::BvExprContext ctx;
  Lowerer lo(ctx);
  auto la = lo.lower(a);
  auto lb = lo.lower(b);
  if (!la || !lb || !lo.ok())
    return Result::Unknown;

  ::smt::BvSolver solver;
  auto ne = ctx.create(::smt::BvExpr::Ne, la, lb);
  bool sat = solver.infer(ne);
  return sat ? Result::NotEqual : Result::Equal;
}

bool sys::smt_prover::tryProveNonZeroI32(Op *cond) {
  if (!cond || cond->getResultType() != Value::i32) return false;
  ::smt::BvExprContext ctx;
  Lowerer lo(ctx);
  auto lc = lo.lower(cond);
  if (!lc || !lo.ok()) return false;
  ::smt::BvSolver solver;
  auto zero = ctx.create(::smt::BvExpr::Const, 0);
  auto eq = ctx.create(::smt::BvExpr::Eq, lc, zero);
  // If `cond == 0` is unsatisfiable, the condition is non-zero for all inputs.
  return !solver.infer(eq);
}

bool sys::smt_prover::tryProveZeroI32(Op *cond) {
  if (!cond || cond->getResultType() != Value::i32) return false;
  ::smt::BvExprContext ctx;
  Lowerer lo(ctx);
  auto lc = lo.lower(cond);
  if (!lc || !lo.ok()) return false;
  ::smt::BvSolver solver;
  auto zero = ctx.create(::smt::BvExpr::Const, 0);
  auto ne = ctx.create(::smt::BvExpr::Ne, lc, zero);
  // If `cond != 0` is unsatisfiable, the condition is zero for all inputs.
  return !solver.infer(ne);
}
