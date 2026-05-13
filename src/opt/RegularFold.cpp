#include "Passes.h"
#include "../utils/Matcher.h"

#include <cstdlib>
#include <cstring>
#include <optional>

using namespace sys;

// Try to resolve an Op to a compile-time integer constant.
// Handles two cases:
//   1. The Op is an IntOp (literal constant).
//   2. The Op is a LoadOp whose address is a GetGlobalOp pointing to a
//      scalar (size == 1) integer global with a known initializer.
//      This covers patterns like `const int base = 16;` which the frontend
//      emits as a global variable rather than inlining the constant.
static std::optional<int> tryGetConstantValue(
    Op *op, const std::map<std::string, GlobalOp*> &gMap) {
  if (isa<IntOp>(op))
    return V(op);

  // Pattern: load(getglobal(<name>))
  if (isa<LoadOp>(op)) {
    auto addr = op->DEF(0);
    if (!isa<GetGlobalOp>(addr))
      return std::nullopt;
    auto name = NAME(addr);
    auto it = gMap.find(name);
    if (it == gMap.end())
      return std::nullopt;
    auto glob = it->second;
    auto iarr = glob->get<IntArrayAttr>();
    if (!iarr || iarr->size != 1)
      return std::nullopt;
    return iarr->vi[0];
  }

  return std::nullopt;
}

static bool hasDirectStoreToScalarGlobal(
    const std::string &name,
    ModuleOp *module) {
  for (auto store : module->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    auto addr = store->DEF(1);
    if (!addr || !isa<GetGlobalOp>(addr))
      continue;
    if (NAME(addr) == name)
      return true;
  }
  return false;
}

#define INT(op) isa<IntOp>(op)

static Rule rules[] = {
  // Addition
  "(change (add x 0) x)",
  "(change (add 'a 'b) (!add 'a 'b))",
  "(change (add 'a x) (add x 'a))",
  "(change (add x (minus y)) (sub x y))",
  "(change (add (minus x) y) (sub y x))",
  "(change (add x (sub y x)) y)",
  "(change (add (add x 'a) 'b) (add x (!add 'a 'b)))",
  "(change (add (sub x 'a) 'b) (add x (!sub 'b 'a)))",
  "(change (add (mul x 'a) (mul x 'b)) (mul x (!add 'a 'b)))",
  "(change (add (mul x 'a) x) (mul (!add 'a 1) x))",
  "(change (add x (mul x 'a)) (mul (!add 'a 1) x))",
  "(change (add (mul x 'a) (mul y 'a)) (mul (add x y) 'a))",
  "(change (add (div 'a x) (div 'b x)) (div (!add 'a 'b) x))",
  "(change (add (div x 'a) (div y 'a)) (div (add x y) 'a))",
  "(change (add x x) (mul x 2))",

  // Addition (64-bit)
  "(change (addl x 0) x)",
  "(change (addl 'a 'b) (!add 'a 'b))",
  "(change (addl 'a x) (addl x 'a))",
  "(change (addl (addl x 'a) 'b) (addl x (!add 'a 'b)))",

  // FP Addition
  "(change (fadd *a *b) (?add *a *b))",
  "(change (fadd x *0) x)",
  "(change (fadd *a x) (fadd x *a))",
  "(change (fadd x (fminus y)) (fsub x y))",
  "(change (fadd (fminus x) y) (fsub y x))",

  // Subtraction
  "(change (sub x 0) x)",
  "(change (sub x x) 0)",
  "(change (sub 0 x) (minus x))",
  "(change (sub 'a 'b) (!sub 'a 'b))",
  "(change (sub (add x y) x) y)",
  "(change (sub (add x y) y) x)",
  "(change (sub x (add x y)) (minus y))",
  "(change (sub y (add x y)) (minus x))",
  "(change (sub x (sub x y)) y)",
  "(change (sub (add x y) (sub x z)) (add y z))",
  "(change (sub (add x y) (add x z)) (sub y z))",
  "(change (sub (add x 'a) 'b) (add x (!sub 'a 'b)))",
  "(change (sub (sub x 'a) 'b) (sub x (!add 'a 'b)))",
  "(change (sub x (minus y)) (add x y))",
  "(change (sub (mul x 'a) (mul x 'b)) (mul x (!sub 'a 'b)))",
  "(change (add (mul x 'a) x) (mul (!sub 'a 1) x))",
  "(change (sub (mul x 'a) (mul y 'a)) (mul (sub x y) 'a))",
  "(change (sub (div 'a x) (div 'b x)) (div (!sub 'a 'b) x))",
  "(change (sub (div x 'a) (div y 'a)) (div (sub x y) 'a))",

  // Subtraction (64-bit)
  "(change (subl x 0) x)",
  "(change (subl (addl x 'a) x) 'a)",

  // FP subtraction
  "(change (fsub *a *b) (?sub *a *b))",
  "(change (fsub x *0) x)",
  "(change (fsub x (fminus y)) (fadd x y))",

  // Multiplication
  "(change (mul x 0) 0)",
  "(change (mul x 1) x)",
  "(change (mul x -1) (minus x))",
  "(change (mul 'a 'b) (!mul 'a 'b))",
  "(change (mul 'a x) (mul x 'a))",
  "(change (mul (mul x 'a) 'b) (mul x (!mul 'a 'b)))",
  "(change (mul (add x 'a) 'b) (add (mul x 'b) (!mul 'a 'b)))",
  "(change (mul (sub x 'a) 'b) (sub (mul x 'b) (!mul 'a 'b)))",

  // Long multiplication
  "(change (mull x 1) x)",

  // FP multiplication
  "(change (fmul *a *b) (?mul *a *b))",
  "(change (fmul x *0) *0)",
  "(change (fmul x *1) x)",
  "(change (fmul x *2) (fadd x x))",
  "(change (fmul x *-1) (fminus x))",
  "(change (fmul *a x) (fmul x *a))",

  // Division
  "(change (div 0 x) 0)",
  "(change (div x 1) x)",
  "(change (div x -1) (minus x))",
  "(change (div x x) 1)",
  "(change (div 'a 'b) (!div 'a 'b))",
  "(change (div (mul x 'a) 'b) (!only-if (!eq (!mod 'a 'b) 0) (mul x (!div 'a 'b))))",
  "(change (div (div x 'a) 'b) (!only-if (!ne (!mul 'a 'b) -1) (div x (!mul 'a 'b))))",

  // Word division (32-bit) - same chain folding as 64-bit
  "(change (divw (divw x 'a) 'b) (!only-if (!ne (!mul 'a 'b) -1) (divw x (!mul 'a 'b))))",

  // Long division
  "(change (divl x 1) x)",

  // FP Division
  "(change (fdiv *a *b) (?div *a *b))",
  "(change (fdiv *0 x) *0)",
  "(change (fdiv x *1) x)",
  "(change (fdiv x *-1) (fminus x))",
  "(change (fdiv x x) *1)",

  // Modulus
  "(change (mod x 1) 0)",
  "(change (mod x x) 0)",
  "(change (mod 0 x) 0)",
  "(change (mod 'a 'b) (!mod 'a 'b))",

  // Shift
  "(change (lshift 'a 'b) (!lsh 'a 'b))",
  "(change (rshift 'a 'b) (!rsh 'a 'b))",

  // Minus
  "(change (minus 'a) (!sub 0 'a))",
  "(change (minus (add x 'a)) (sub (!sub 0 'a) x))",
  "(change (minus (sub x y)) (sub y x))",
  "(change (minus (minus x)) x)",

  // FP Minus
  "(change (fminus *a) (?sub *0 *a))",
  "(change (fminus (fadd x *a)) (fsub (?sub *0 *a) x))",
  "(change (fminus (fsub x y)) (fsub y x))",
  "(change (fminus (fminus x)) x)",

  // Equality
  "(change (eq x x) 1)",
  "(change (eq 'a 'b) (!eq 'a 'b))",
  "(change (eq 'a x) (eq x 'a))",
  "(change (eq (add x 'a) 'b) (eq x (!sub 'b 'a)))",
  "(change (eq (sub x 'a) 'b) (eq x (!add 'b 'a)))",
  "(change (eq (mul x 'a) 'b) (!only-if (!eq 0 (!mod 'b 'a)) (eq x (!div 'b 'a))))",
  "(change (eq (mul x 'a) 'b) (!only-if (!ne 0 (!mod 'b 'a)) 0))",
  "(change (eq (div x 'a) 'b) (!only-if (!gt 'a 0) (and (lt x (!mul (!add 'b 1) 'a)) (ge x (!mul 'b 'a)))))",
  "(change (eq (mod x 'a) 'b) (!only-if (!le 'a 'b) 0))",
  "(change (eq (mod x 2) 1) (and (ge x 0) (mod x 2)))",
  "(change (eq x 0) (not x))",
  "(change (eq (not x) 0) (snz x))",

  // FP Equality
  "(change (feq x x) 1)",
  "(change (feq *a *b) (!feq *a *b))",
  "(change (feq *a x) (feq x *a))",

  // Less than or equal
  "(change (le x x) 1)",
  "(change (le 'a 'b) (!le 'a 'b))",
  "(change (le x 'a) (lt x (!add 'a 1)))",
  "(change (le 'a x) (lt (!sub 'a 1) x))",

  // FP Less than or equal
  "(change (fle x x) 1)",
  "(change (fle *a *b) (!fle *a *b))",

  // Less than
  "(change (lt x x) 0)",
  "(change (lt 'a 'b) (!lt 'a 'b))",
  "(change (lt (mul x 'a) 'b) (!only-if (!and (!and (!gt 'a 0) (!gt 'b 0)) (!eq (!mod 'b 'a) 0)) (lt x (!div 'b 'a))))",
  "(change (lt (mul x 'a) 'b) (!only-if (!and (!and (!gt 'a 0) (!gt 'b 0)) (!ne (!mod 'b 'a) 0)) (lt x (!add (!div 'b 'a) 1))))",
  "(change (lt (div x 'a) 'b) (!only-if (!and (!gt 'a 0) (!gt 'b 0)) (lt x (!mul 'b 'a))))",
  "(change (lt 'b (add x 'a)) (lt (!sub 'b 'a) x))",
  "(change (lt 'b (sub x 'a)) (lt (!add 'b 'a) x))",
  "(change (lt 'b (sub 'a x)) (gt (!sub 'a 'b) x))",
  "(change (lt (sub 'a x) 'b) (gt x (!sub 'a 'b)))",
  "(change (lt 'b (mul x 'a)) (!only-if (!and (!gt 'a 0) (!gt 'b 0)) (lt (!div 'b 'a) x)))",
  "(change (lt 'b (div x 'a)) (!only-if (!and (!gt 'a 0) (!gt 'b 0)) (le (!mul 'a (!add 'b 1)) x)))",
  "(change (lt x (add x 'a)) (!lt 0 'a))",
  "(change (lt x (sub 'a x)) (lt (mul x 2) 'a))",
  
  // FP Less than
  "(change (flt x x) 0)",
  "(change (flt *a *b) (!flt *a *b))",

  // Not
  "(change (not 'a) (!not 'a))",
  "(change (not (snz x)) (not x))",
  "(change (not (eq x y)) (ne x y))",
  "(change (not (lt x y)) (ge x y))",
  "(change (not (le x y)) (gt x y))",
  "(change (not (ne x y)) (eq x y))",
  "(change (not (not x)) (snz x))",
  "(change (not (or x y)) (and (not x) (not y)))",
  "(change (not (and x y)) (or (not x) (not y)))",

  // FP Not
  "(change (not *a) (!feq *a *0))",
  "(change (not (feq x y)) (fne x y))",
  "(change (not (flt x y)) (flt y x))",
  "(change (not (fle x y)) (fle y x))",
  "(change (not (fne x y)) (feq x y))",

  // Not equal
  "(change (ne 'a 'b) (!ne 'a 'b))",
  "(change (ne 'a x) (ne x 'a))",
  "(change (ne (add x 'a) 'b) (ne x (!sub 'b 'a)))",
  "(change (ne (sub x 'a) 'b) (ne x (!add 'b 'a)))",
  "(change (ne (mod x 2) 1) (or (lt x 0) (not (mod x 2))))",
  "(change (ne x 0) (snz x))",
  "(change (ne (not x) 0) (not x))",

  // FP Not equal
  "(change (fne *a *b) (!fne *a *b))",

  // Set not zero
  "(change (snz 0) 0)",
  "(change (snz 'a) (!only-if (!ne 'a 0) 1))",
  "(change (snz (mod x 2)) (mod x 2))",
  "(change (snz (eq x y)) (eq x y))",
  "(change (snz (le x y)) (le x y))",
  "(change (snz (lt x y)) (lt x y))",
  "(change (snz (ne x y)) (ne x y))",
  "(change (snz (not x)) (not x))",

  // FP Set not zero
  "(change (snz *a) (!fne *a *0))",

  // And
  "(change (and x 0) 0)",
  "(change (and 'a 'b) (!and 'a 'b))",
  "(change (and x x) x)",
  "(change (and x -1) x)",
  "(change (and -1 x) x)",

  // Or
  "(change (or x 0) x)",
  "(change (or 0 x) x)",
  "(change (or x x) x)",
  "(change (or 'a 'b) (!or 'a 'b))",
  "(change (or x -1) -1)",
  "(change (or -1 x) -1)",

  // Xor
  "(change (xor x 0) x)",
  "(change (xor 0 x) x)",
  "(change (xor x x) 0)",
  "(change (xor 'a 'b) (!xor 'a 'b))",

  // Select
  "(change (select 1 x y) x)",
  "(change (select 0 x y) y)",
  "(change (select c 1 0) (snz c))",
  "(change (select c 0 1) (not c))",
  "(change (select c x x) x)",
  "(change (select c (eq x y) 0) (and (eq x y) c))",
  "(change (select c (ne x y) 0) (and (ne x y) c))",
  "(change (select c (lt x y) 0) (and (lt x y) c))",
  "(change (select c (le x y) 0) (and (le x y) c))",
  "(change (select c (eq x y) 1) (or (eq x y) (not c)))",
  "(change (select c (ne x y) 1) (or (ne x y) (not c)))",
  "(change (select c (lt x y) 1) (or (lt x y) (not c)))",
  "(change (select c (le x y) 1) (or (le x y) (not c)))",
  "(change (select c 1 (eq x y)) (or (eq x y) c))",
  "(change (select c 1 (ne x y)) (or (ne x y) c))",
  "(change (select c 1 (lt x y)) (or (lt x y) c))",
  "(change (select c 1 (le x y)) (or (le x y) c))",

  // float -> int
  "(change (f2i *a) (!cvt *a))",

  // int -> float
  "(change (i2f 'a) (?cvt 'a))",
};

std::map<std::string, int> RegularFold::stats() {
  return {
    { "folded-ops", foldedTotal }
  };
}

void removePhiOperand(Op *phi, BasicBlock *from) {
  auto ops = phi->getOperands();
  std::vector<Attr*> attrs;
  for (auto attr : phi->getAttrs())
    attrs.push_back(attr->clone());

  phi->removeAllOperands();
  // This deletes attributes if their refcnt goes to zero.
  // That's why we cloned above.
  phi->removeAllAttributes();

  for (size_t i = 0; i < ops.size(); i++) {
    auto pred = FROM(attrs[i]);
    if (from == pred)
      continue;

    // Only preserve the operands that aren't from dead blocks.
    phi->pushOperand(ops[i]);
    phi->add<FromAttr>(pred);
  }
}

void tidyPhi(BasicBlock *bb, BasicBlock *from) {
  auto phis = bb->getPhis();
  for (auto phi : phis)
    removePhiOperand(phi, from);
}

// This pass works on both structured control flow and flattened cfg.
int RegularFold::runImpl(Region *region) {
  int folded = 0;
  for (auto bb : region->getBlocks()) {
    auto ops = bb->getOps();
    for (auto op : ops) {
      // Match each rule.
      bool success = false;
      for (auto &rule : rules) {
        success = rule.rewrite(op);
        if (success) {
          folded++;
          break;
        }
      }

      // Recursively fold regions.
      if (!success) {
        for (auto r : op->getRegions())
          folded += runImpl(r);
      }
    }
  }
  return folded;
}

void RegularFold::run() {
  auto funcs = collectFuncs();
  int folded;

  bool enableIrDivStrength = true;
  if (const char *env = std::getenv("SISY_ENABLE_IR_DIV_STRENGTH"))
    enableIrDivStrength = env[0] && std::strcmp(env, "0") != 0;

  // Build a map of global name → GlobalOp for constant-global detection.
  // We use this to resolve loads from scalar const globals (e.g. `const int base = 16`).
  auto gMap = getGlobalMap();

  do {
    folded = 0;
    for (auto func : funcs) {
      auto region = func->getRegion();
      folded += runImpl(region);
    }

    // Also, run some extra folds.
    Builder builder;

    runRewriter([&](LoadOp *op) {
      if (op->getResultType() != Value::i32 || op->getOperandCount() != 1)
        return false;
      auto addr = op->DEF(0);
      if (!addr || !isa<GetGlobalOp>(addr))
        return false;

      auto name = NAME(addr);
      auto it = gMap.find(name);
      if (it == gMap.end())
        return false;
      auto glob = it->second;
      auto iarr = glob->get<IntArrayAttr>();
      if (!iarr || iarr->size != 1)
        return false;
      if (hasDirectStoreToScalarGlobal(name, module))
        return false;

      folded++;
      builder.replace<IntOp>(op, { new IntAttr(iarr->vi[0]) });
      return true;
    });

    runRewriter([&](BranchOp *op) {
      auto cond = op->DEF();

      // (br (snz x)) becomes (br x)
      // Do note that "set-not-zero" of float cannot be fold.
      if (isa<SetNotZeroOp>(cond) && cond->DEF()->getResultType() != Value::f32) {
        folded++;
        auto def = cond->DEF();
        builder.replace<BranchOp>(op, { def }, op->getAttrs());
        return true;
      }
      
      // (br (not x) >bb >bb2) becomes (br x >bb2 >bb)
      if (isa<NotOp>(cond) && cond->DEF()->getResultType() != Value::f32) {
        folded++;
        auto def = cond->DEF();
        builder.replace<BranchOp>(op, { def }, { new TargetAttr(ELSE(op)), new ElseAttr(TARGET(op)) });
        return true;
      }

      if (!isa<IntOp>(cond))
        return false;
      
      if (V(cond) == 0) {
        folded++;
        tidyPhi(TARGET(op), op->getParent());
        builder.replace<GotoOp>(op, { new TargetAttr(ELSE(op)) });
        return false;
      }

      // V(cond) != 0
      folded++;
      tidyPhi(ELSE(op), op->getParent());
      builder.replace<GotoOp>(op, { new TargetAttr(TARGET(op)) });
      return false;
    });

    runRewriter([&](DivFOp *op) {
      auto y = op->DEF(1);
      if (isa<FloatOp>(y) && F(y) == 2) {
        folded++;
        builder.setBeforeOp(op);
        auto half = builder.create<FloatOp>({ new FloatAttr(0.5) });
        builder.replace<MulFOp>(op, { op->getOperand(0), half });
        return false;
      }
      return false;
    });

    // Signed integer division by a compile-time power-of-2 constant:
    //   x / 2^n  →  (x + ((x >> 31) & (2^n - 1))) >> n
    // This is the standard "round toward zero" transformation for signed division.
    // The non-negative case (x >= 0) is handled more precisely by RangeAwareFold;
    // this rule covers the general signed case.
    if (enableIrDivStrength) {
      runRewriter([&](DivIOp *op) {
        auto x = op->DEF(0);
        auto y = op->DEF(1);

        // Divisor must be a compile-time integer constant (or a load from a
        // scalar const global, e.g. `const int base = 16`).
        auto maybeDiv = tryGetConstantValue(y, gMap);
        if (!maybeDiv)
          return false;

        int divisor = *maybeDiv;
        // Must be a positive power of 2 greater than 1.
        if (divisor <= 1 || __builtin_popcount((unsigned)divisor) != 1)
          return false;

        int n = __builtin_ctz((unsigned)divisor);  // log2(divisor)
        int mask = divisor - 1;                    // 2^n - 1

        // Generate: sign = x >> 31  (arithmetic, propagates sign bit)
        //           bias = sign & mask
        //           result = (x + bias) >> n
        builder.setBeforeOp(op);
        auto c31  = builder.create<IntOp>({ new IntAttr(31) });
        auto sign = builder.create<RShiftOp>({ x, c31 });
        auto cmask = builder.create<IntOp>({ new IntAttr(mask) });
        auto bias = builder.create<AndIOp>({ sign, cmask });
        auto xbias = builder.create<AddIOp>({ x, bias });
        auto cn   = builder.create<IntOp>({ new IntAttr(n) });
        builder.replace<RShiftOp>(op, { xbias, cn });

        folded++;
        return false;
      });

      // Signed integer remainder by a compile-time power-of-2 constant,
      // when Range analysis confirms x >= 0:
      //   x % 2^n  →  x & (2^n - 1)
      // This is only valid for non-negative x: for negative x, C semantics give
      // a negative remainder, but bitwise AND always gives a non-negative result.
      runRewriter([&](ModIOp *op) {
        auto x = op->DEF(0);
        auto y = op->DEF(1);

        // Divisor must be a compile-time integer constant (or a load from a
        // scalar const global, e.g. `const int base = 16`).
        auto maybeMod = tryGetConstantValue(y, gMap);
        if (!maybeMod)
          return false;

        int divisor = *maybeMod;
        // Must be a positive power of 2.
        if (divisor <= 0 || __builtin_popcount((unsigned)divisor) != 1)
          return false;

        // Range analysis must confirm x >= 0 (lower bound >= 0).
        if (!x->has<RangeAttr>())
          return false;
        auto [low, high] = RANGE(x);
        if (low < 0)
          return false;

        int mask = divisor - 1;  // 2^n - 1

        builder.setBeforeOp(op);
        auto cmask = builder.create<IntOp>({ new IntAttr(mask) });
        builder.replace<AndIOp>(op, { x, cmask });

        folded++;
        return false;
      });
    }

    foldedTotal += folded;
  } while (folded);
}
