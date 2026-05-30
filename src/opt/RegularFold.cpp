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
static bool addressMentionsScalarGlobal(const std::string &name, Op *addr) {
  if (!addr)
    return false;
  if (isa<GetGlobalOp>(addr))
    return NAME(addr) == name;
  auto alias = addr->find<AliasAttr>();
  if (!alias || alias->unknown)
    return false;
  for (auto &[base, _] : alias->location)
    if (base && isa<GlobalOp>(base) && NAME(base) == name)
      return true;
  return false;
}

static bool scalarGlobalEscapesToCall(const std::string &name, ModuleOp *module) {
  for (auto call : module->findAll<CallOp>()) {
    for (auto operand : call->getOperands()) {
      auto def = operand.defining;
      if (!def || def->getResultType() != Value::i64)
        continue;
      if (addressMentionsScalarGlobal(name, def))
        return true;
    }
  }
  return false;
}

static bool mayTouchScalarGlobal(const std::string &name, Op *addr, bool escaped) {
  if (!addr)
    return false;
  if (addressMentionsScalarGlobal(name, addr))
    return true;
  auto alias = addr->find<AliasAttr>();
  return escaped && (!alias || alias->unknown);
}

static bool hasStoreToScalarGlobal(
    const std::string &name,
    ModuleOp *module) {
  bool escaped = scalarGlobalEscapesToCall(name, module);
  for (auto store : module->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    if (mayTouchScalarGlobal(name, store->DEF(1), escaped))
      return true;
  }

  return escaped;
}

static std::optional<int> immutableScalarGlobalValue(
    const std::string &name,
    const std::map<std::string, GlobalOp*> &gMap,
    ModuleOp *module) {
  auto it = gMap.find(name);
  if (it == gMap.end())
    return std::nullopt;
  auto glob = it->second;
  if (!glob->has<ConstAttr>())
    return std::nullopt;
  auto iarr = glob->get<IntArrayAttr>();
  if (!iarr || iarr->size != 1)
    return std::nullopt;
  if (hasStoreToScalarGlobal(name, module))
    return std::nullopt;
  if (!iarr->vi)
    return 0;
  return iarr->vi[0];
}

static std::optional<std::string> zeroOffsetGlobalName(Op *addr) {
  if (!addr)
    return std::nullopt;
  if (isa<GetGlobalOp>(addr))
    return NAME(addr);
  // AliasAttr stores raw Op* bases. Pass cleanup may release detached address
  // computations between fold rounds, so consulting old alias bases here can
  // dereference freed IR nodes. Keep this scalar-global fold conservative.
  return std::nullopt;
}

static std::optional<int> tryGetConstantValue(
    Op *op, const std::map<std::string, GlobalOp*> &gMap, ModuleOp *module) {
  if (isa<IntOp>(op))
    return V(op);

  // Pattern: load(getglobal(<name>)) or an equivalent zero-offset alias.
  if (isa<LoadOp>(op)) {
    auto name = zeroOffsetGlobalName(op->DEF(0));
    if (!name)
      return std::nullopt;
    return immutableScalarGlobalValue(*name, gMap, module);
  }

  return std::nullopt;
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
  // Avoid reassociation patterns that cancel repeated non-trivial SSA values
  // across load/store-heavy loops.  They are profitable, but stale cleanup after
  // loop rotation can otherwise leave dangling operands in downstream RV passes.
  "(change (sub (add x 'a) 'b) (add x (!sub 'a 'b)))",
  "(change (sub (sub x 'a) 'b) (sub x (!add 'a 'b)))",
  "(change (sub x (minus y)) (add x y))",
  "(change (sub (mul x 'a) (mul x 'b)) (mul x (!sub 'a 'b)))",
  "(change (sub (mul x 'a) x) (mul (!sub 'a 1) x))",
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
  "(change (lshift x 0) x)",
  "(change (rshift x 0) x)",
  "(change (lshift 0 x) 0)",
  "(change (rshift 0 x) 0)",
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
      if (!op->has<ImpureAttr>()) {
        for (auto &rule : rules) {
          success = rule.rewrite(op);
          if (success) {
            folded++;
            break;
          }
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

  int rounds = 0;
  do {
    currentRound = rounds;
    if (rounds++ > 1000) {
      std::cerr << "[RegularFold] warning: algebraic folding did not converge after 1000 rounds! Breaking to prevent infinite loop." << std::endl;
      break;
    }
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
      auto name = zeroOffsetGlobalName(addr);
      if (!name)
        return false;

      auto value = immutableScalarGlobalValue(*name, gMap, module);
      if (!value)
        return false;

      folded++;
      builder.replace<IntOp>(op, { new IntAttr(*value) });
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

    auto constantInt = [&](Op *op) -> std::optional<int> {
      return tryGetConstantValue(op, gMap, module);
    };

    auto productParity = [&](Op *op) -> Op * {
      auto mod = dyn_cast<ModIOp>(op);
      if (!mod || mod->getOperandCount() != 2)
        return nullptr;
      auto divisor = constantInt(mod->DEF(1));
      if (!divisor || (*divisor != 2 && *divisor != -2))
        return nullptr;
      auto mul = dyn_cast<MulIOp>(mod->DEF(0));
      if (!mul || mul->getOperandCount() != 2)
        return nullptr;

      builder.setBeforeOp(op);
      auto oneA = builder.create<IntOp>(std::vector<Attr*>{ new IntAttr(1) });
      auto oneB = builder.create<IntOp>(std::vector<Attr*>{ new IntAttr(1) });
      auto lhsBit = builder.create<AndIOp>(std::vector<Value>{ mul->DEF(0), oneA });
      auto rhsBit = builder.create<AndIOp>(std::vector<Value>{ mul->DEF(1), oneB });
      return builder.create<AndIOp>(std::vector<Value>{ lhsBit, rhsBit });
    };

    auto foldProductParityCompare = [&](Op *op, bool isEq) -> bool {
      if (op->getOperandCount() != 2)
        return false;
      auto lhs = op->DEF(0);
      auto rhs = op->DEF(1);

      Op *modSide = nullptr;
      Op *constSide = nullptr;
      if (auto bit = productParity(lhs)) {
        modSide = bit;
        constSide = rhs;
      } else if (auto bit = productParity(rhs)) {
        modSide = bit;
        constSide = lhs;
      } else {
        return false;
      }

      auto constant = constantInt(constSide);
      if (!constant || (*constant != 0 && *constant != 1))
        return false;

      builder.setBeforeOp(op);
      if ((*constant == 1) == isEq) {
        op->replaceAllUsesWith(modSide);
        op->erase();
      } else {
        auto zero = builder.create<IntOp>(std::vector<Attr*>{ new IntAttr(0) });
        builder.replace<EqOp>(op, std::vector<Value>{ modSide, zero });
      }
      folded++;
      return true;
    };

    // Product parity is independent of signed overflow in the full product:
    // (x * y) is odd iff both operands are odd.  Fold comparisons against
    // `% +/-2` to low-bit tests so later if-conversion and backend lowering
    // never need to materialize a multiply just to ask an even/odd question.
    runRewriter([&](EqOp *op) { return foldProductParityCompare(op, true); });
    runRewriter([&](NeOp *op) { return foldProductParityCompare(op, false); });
    runRewriter([&](NotOp *op) {
      if (op->getOperandCount() != 1)
        return false;
      auto bit = productParity(op->DEF(0));
      if (!bit)
        return false;
      builder.setBeforeOp(op);
      auto zero = builder.create<IntOp>(std::vector<Attr*>{ new IntAttr(0) });
      builder.replace<EqOp>(op, std::vector<Value>{ bit, zero });
      folded++;
      return true;
    });
    runRewriter([&](SetNotZeroOp *op) {
      if (op->getOperandCount() != 1)
        return false;
      auto bit = productParity(op->DEF(0));
      if (!bit)
        return false;
      op->replaceAllUsesWith(bit);
      op->erase();
      folded++;
      return true;
    });
    runRewriter([&](BranchOp *op) {
      if (op->getOperandCount() != 1 || !op->has<TargetAttr>() || !op->has<ElseAttr>())
        return false;
      auto bit = productParity(op->DEF(0));
      if (!bit)
        return false;
      builder.replace<BranchOp>(op, std::vector<Value>{ bit }, op->getAttrs());
      folded++;
      return true;
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
        auto maybeDiv = tryGetConstantValue(y, gMap, module);
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

      // Signed integer remainder by a compile-time power-of-2 constant.
      //
      // For non-negative x, x % 2^n is just x & (2^n - 1).  For the general
      // signed case, preserve SysY/C round-toward-zero semantics by using the
      // same biased quotient as the division fold:
      //   q = (x + ((x >> 31) & (2^n - 1))) >> n
      //   r = x - (q << n)
      // This removes expensive remw without assuming that the source is a
      // bitwise helper or that x is non-negative.
      runRewriter([&](ModIOp *op) {
        auto x = op->DEF(0);
        auto y = op->DEF(1);

        // Divisor must be a compile-time integer constant (or a load from a
        // scalar const global, e.g. `const int base = 16`).
        auto maybeMod = tryGetConstantValue(y, gMap, module);
        if (!maybeMod)
          return false;

        int divisor = *maybeMod;
        if (divisor < 0)
          divisor = -divisor;

        // Must be a positive power of 2.
        if (divisor <= 0 || __builtin_popcount((unsigned)divisor) != 1)
          return false;

        if (divisor == 1) {
          builder.replace<IntOp>(op, { new IntAttr(0) });
          folded++;
          return true;
        }

        builder.setBeforeOp(op);
        int shift = __builtin_ctz((unsigned)divisor);
        int mask = divisor - 1;

        if (x->has<RangeAttr>()) {
          auto [low, high] = RANGE(x);
          if (low >= 0) {
            auto cmask = builder.create<IntOp>({ new IntAttr(mask) });
            builder.replace<AndIOp>(op, { x, cmask });
            folded++;
            return false;
          }
        }

        auto c31 = builder.create<IntOp>({ new IntAttr(31) });
        auto sign = builder.create<RShiftOp>({ x, c31 });
        auto cmask = builder.create<IntOp>({ new IntAttr(mask) });
        auto bias = builder.create<AndIOp>({ sign, cmask });
        auto biased = builder.create<AddIOp>({ x, bias });
        auto cshift = builder.create<IntOp>({ new IntAttr(shift) });
        auto quotient = builder.create<RShiftOp>({ biased, cshift });
        auto product = builder.create<LShiftOp>({ quotient, cshift });
        builder.replace<SubIOp>(op, { x, product });

        folded++;
        return false;
      });
    }

    foldedTotal += folded;
  } while (folded);
}
