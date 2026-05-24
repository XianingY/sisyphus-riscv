#include "RvPasses.h"
#include "RvDupPasses.h"
#include <cstdlib>
#include <cstring>

using namespace sys::rv;
using namespace sys;

std::map<std::string, int> InstCombine::stats() {
  return {
    { "combined-instructions", combined }
  };
}

bool inRange(Op *op) {
  auto attr = op->get<IntAttr>();
  return attr->value >= -2048 && attr->value <= 2047;
}

bool inRange(int x) {
  return x >= -2048 && x <= 2047;
}

bool inShamt32(int x) {
  return x >= 0 && x < 32;
}

bool inShamt64(int x) {
  return x >= 0 && x < 64;
}

int combineAndImm(int a, int b) {
  return a & b;
}

int combineOrImm(int a, int b) {
  return a | b;
}

int combineXorImm(int a, int b) {
  return a ^ b;
}

bool sameVsetvliInput(Op *a, Op *b) {
  if (a == b)
    return true;
  if (!a || !b)
    return false;
  return isa<LiOp>(a) && isa<LiOp>(b) && V(a) == V(b);
}

bool clobbersVectorConfig(Op *op) {
  return isa<sys::rv::CallOp>(op) || isa<RetOp>(op) || isa<JOp>(op) ||
         isa<BeqOp>(op) || isa<BneOp>(op) || isa<BltOp>(op) ||
         isa<BgeOp>(op) || isa<BleOp>(op) || isa<BgtOp>(op);
}

static bool getenvEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0)
    return false;
  return true;
}

void InstCombine::run() {
  Builder builder;
  const bool enableLiZeroCombine = getenvEnabled("SISY_RV_ENABLE_LI_ZERO_COMBINE", false);

  runRewriter([&](AddOp *op) {
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    if (isa<LiOp>(x) && inRange(x)) {
      combined++;
      builder.replace<AddiOp>(op, { y }, { x->get<IntAttr>() });
      return true;
    }

    if (isa<LiOp>(y) && inRange(y)) {
      combined++;
      builder.replace<AddiOp>(op, { x }, { y->get<IntAttr>() });
      return true;
    }

    return false;
  });

  runRewriter([&](AddwOp *op) {
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    if (isa<LiOp>(x) && inRange(x)) {
      combined++;
      builder.replace<AddiwOp>(op, { y }, { x->get<IntAttr>() });
      return true;
    }

    if (isa<LiOp>(y) && inRange(y)) {
      combined++;
      builder.replace<AddiwOp>(op, { x }, { y->get<IntAttr>() });
      return true;
    }

    return false;
  });

  runRewriter([&](SubOp *op) {
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    // Note we can't fold on `x`.

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = -V(y);
      if (!inRange(val))
        return false;

      combined++;
      builder.replace<AddiOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SubwOp *op) {
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    // Note we can't fold on `x`.

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = -V(y);
      if (!inRange(val))
        return false;

      combined++;
      builder.replace<AddiwOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SllwOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_SHIFT_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = V(y);
      if (!inShamt32(val))
        return false;

      combined++;
      builder.replace<SlliwOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SrawOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_SHIFT_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = V(y);
      if (!inShamt32(val))
        return false;

      combined++;
      builder.replace<SraiwOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SrlwOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_SHIFT_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = V(y);
      if (!inShamt32(val))
        return false;

      combined++;
      builder.replace<SrliwOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SraOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_SHIFT_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = V(y);
      if (!inShamt64(val))
        return false;

      combined++;
      builder.replace<SraiOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SrlOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_SHIFT_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = V(y);
      if (!inShamt64(val))
        return false;

      combined++;
      builder.replace<SrliOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](SllOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_SHIFT_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;

    if (isa<LiOp>(y) && inRange(y)) {
      auto val = V(y);
      if (!inShamt64(val))
        return false;

      combined++;
      builder.replace<SlliOp>(op, { x }, { new IntAttr(val) });
      return true;
    }

    return false;
  });

  runRewriter([&](OrOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_LOGIC_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    if (isa<LiOp>(x) && inRange(x)) {
      combined++;
      builder.replace<OriOp>(op, { y }, { x->get<IntAttr>() });
      return true;
    }

    if (isa<LiOp>(y) && inRange(y)) {
      combined++;
      builder.replace<OriOp>(op, { x }, { y->get<IntAttr>() });
      return true;
    }

    return false;
  });

  runRewriter([&](XorOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_LOGIC_IMM_COMBINE", true))
      return false;
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    if (isa<LiOp>(x) && inRange(x)) {
      combined++;
      builder.replace<XoriOp>(op, { y }, { x->get<IntAttr>() });
      return true;
    }

    if (isa<LiOp>(y) && inRange(y)) {
      combined++;
      builder.replace<XoriOp>(op, { x }, { y->get<IntAttr>() });
      return true;
    }

    return false;
  });

  runRewriter([&](AndOp *op) {
    auto x = op->getOperand(0).defining;
    auto y = op->getOperand(1).defining;
    if (isa<LiOp>(x) && inRange(x)) {
      combined++;
      builder.replace<AndiOp>(op, { y }, { x->get<IntAttr>() });
      return true;
    }

    if (isa<LiOp>(y) && inRange(y)) {
      combined++;
      builder.replace<AndiOp>(op, { x }, { y->get<IntAttr>() });
      return true;
    }

    return false;
  });

  runRewriter([&](StoreOp *op) {
    auto value = op->getOperand(0);
    auto addr = op->getOperand(1).defining;
    if (isa<AddiOp>(addr)) {
      auto offset = V(addr);
      auto &currentOffset = V(op);
      if (inRange(offset + currentOffset)) {
        currentOffset += offset;
        auto base = addr->getOperand();
        builder.replace<StoreOp>(op, { value, base }, op->getAttrs());
        return true;
      }
    }
    return false;
  });

  runRewriter([&](LoadOp *op) {
    auto addr = op->getOperand(0).defining;
    if (isa<AddiOp>(addr)) {
      auto offset = V(addr);
      auto &currentOffset = V(op);
      if (inRange(offset + currentOffset)) {
        currentOffset += offset;
        auto base = addr->getOperand();
        builder.replace<LoadOp>(op, op->getResultType(), { base }, op->getAttrs());
        return true;
      }
    }
    return false;
  });

  runRewriter([&](AddiOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_IMM_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<AddiOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inRange(value))
      return false;
    combined++;
    builder.replace<AddiOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](AddiwOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_IMM_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<AddiwOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inRange(value))
      return false;
    combined++;
    builder.replace<AddiwOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](AndiOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_IMM_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<AndiOp>(base))
      return false;
    int value = combineAndImm(V(base), V(op));
    if (!inRange(value))
      return false;
    combined++;
    builder.replace<AndiOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](OriOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_IMM_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<OriOp>(base))
      return false;
    int value = combineOrImm(V(base), V(op));
    if (!inRange(value))
      return false;
    combined++;
    builder.replace<OriOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](XoriOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_IMM_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<XoriOp>(base))
      return false;
    int value = combineXorImm(V(base), V(op));
    if (!inRange(value))
      return false;
    combined++;
    builder.replace<XoriOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](SlliOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_SHIFT_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<SlliOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inShamt64(value))
      return false;
    combined++;
    builder.replace<SlliOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](SrliOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_SHIFT_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<SrliOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inShamt64(value))
      return false;
    combined++;
    builder.replace<SrliOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](SraiOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_SHIFT_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<SraiOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inShamt64(value))
      return false;
    combined++;
    builder.replace<SraiOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](SlliwOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_SHIFT_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<SlliwOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inShamt32(value))
      return false;
    combined++;
    builder.replace<SlliwOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](SrliwOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_SHIFT_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<SrliwOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inShamt32(value))
      return false;
    combined++;
    builder.replace<SrliwOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](SraiwOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_CHAIN_SHIFT_COMBINE", true))
      return false;
    auto base = op->getOperand(0).defining;
    if (!isa<SraiwOp>(base))
      return false;
    int value = V(base) + V(op);
    if (!inShamt32(value))
      return false;
    combined++;
    builder.replace<SraiwOp>(op, { base->getOperand() }, { new IntAttr(value) });
    return true;
  });

  runRewriter([&](VsetvliOp *op) {
    if (!getenvEnabled("SISY_RV_ENABLE_REDUNDANT_VSETVLI_DCE", true))
      return false;
    auto vl = op->getOperand(0).defining;
    for (auto runner = op->prevOp(); runner; runner = runner->prevOp()) {
      if (isa<PhiOp>(runner) || clobbersVectorConfig(runner))
        return false;
      if (!isa<VsetvliOp>(runner))
        continue;
      if (!sameVsetvliInput(runner->getOperand(0).defining, vl))
        return false;
      combined++;
      op->erase();
      return true;
    }
    return false;
  });

  RvDCE(module).run();

  runRewriter([&](SlliwOp *op) {
    (void) op;
    // `slliw rd, rs, 0` is not a pure move on RV64: it sign-extends the
    // low 32 bits. Keep it so i32 values stay ABI-canonicalized.
    return false;
  });

  runRewriter([&](SrliwOp *op) {
    (void) op;
    // `srliw` also truncates to 32 bits before producing the result.
    return false;
  });

  runRewriter([&](SraiwOp *op) {
    (void) op;
    // `sraiw rd, rs, 0` is the standard 32-bit sign-extension idiom.
    return false;
  });

  runRewriter([&](SlliOp *op) {
    if (V(op) == 0) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter([&](SrliOp *op) {
    if (V(op) == 0) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter([&](SraiOp *op) {
    if (V(op) == 0) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter([&](AndiOp *op) {
    if (V(op) == -1) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter([&](OriOp *op) {
    if (V(op) == 0) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter([&](XoriOp *op) {
    if (V(op) == 0) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter([&](AddiwOp *op) {
    if (V(op) == 0) {
      // `addiw rd, rs, 0` sign-extends the low 32 bits on RV64; it is the
      // canonical sext.w idiom, not a plain move.
      return false;
    }
    //   %op   = addiw %1 <V>
    //   %addr = add %2 %op
    //   %x    = load %def <V2>
    // becomes
    //   %addr = add %1 %2
    //   %x    = load %def <V2 + V>
    auto uses = op->getUses();
    for (auto add : uses) {
      if (!isa<AddOp>(add))
        continue;

      // Make sure all consumers of this `add` is load or store,
      // and they're changeable.
      auto vi = V(op);
      bool good = true;
      for (auto x : add->getUses()) {
        if (!isa<StoreOp>(x) && !isa<LoadOp>(x) || !inRange(V(x) + vi)) {
          good = false;
          break;
        }
      }
      if (!good)
        continue;

      add->replaceOperand(op, op->DEF());
      for (auto x : add->getUses())
        V(x) += vi;
    }
    return false;
  });

  runRewriter([&](AddiOp *op) {
    if (V(op) == 0) {
      op->replaceAllUsesWith(op->getOperand().defining);
      op->erase();
      return true;
    }
    //   %op   = addi %1 <V>
    //   %addr = add %op %2
    //   %x    = load %def <V2>
    // becomes
    //   %addr = add %1 %2
    //   %x    = load %def <V2 + V>
    auto uses = op->getUses();
    for (auto add : uses) {
      if (!isa<AddOp>(add))
        continue;

      // Make sure all consumers of this `add` is load or store,
      // and they're changeable.
      auto vi = V(op);
      bool good = true;
      for (auto x : add->getUses()) {
        if (!isa<StoreOp>(x) && !isa<LoadOp>(x) || !inRange(V(x) + vi)) {
          good = false;
          break;
        }
      }
      if (!good)
        continue;

      add->replaceOperand(op, op->DEF());
      for (auto x : add->getUses())
        V(x) += vi;
    }
    return false;
  });

  runRewriter([&](PhiOp *op) {
    // Remove phi with a single operand.
    if (op->getOperands().size() == 1) {
      auto def = op->getOperand().defining;
      op->replaceAllUsesWith(def);
      op->erase();
      return true;
    }
    return false;
  });

  // Replace (snez x) where x is always between 0 and 1.
  runRewriter([&](SnezOp *op) {
    auto def = op->DEF(0);
    if (isa<SltOp>(def) || isa<SltiOp>(def) || isa<SnezOp>(def) || isa<SeqzOp>(def)) {
      op->replaceAllUsesWith(def);
      op->erase();
    }
    return false;
  });

  runRewriter([&](BeqOp *op) {
    auto def = op->DEF(0);
    auto zero = op->DEF(1);
    // Replace `beq (seqz x), zero` with `bne x, zero`.
    if (isa<SeqzOp>(def) && isa<ReadRegOp>(zero) && REG(zero) == Reg::zero) {
      builder.replace<BneOp>(op, { def->DEF(0), zero }, op->getAttrs());
      return true;
    }
    // Replace `beq (snez x), zero` with `beq x, zero`.
    if (isa<SnezOp>(def) && isa<ReadRegOp>(zero) && REG(zero) == Reg::zero) {
      builder.replace<BeqOp>(op, { def->DEF(0), zero }, op->getAttrs());
      return true;
    }
    return false;
  });
  
  runRewriter([&](BneOp *op) {
    auto def = op->DEF(0);
    auto zero = op->DEF(1);
    // Replace `bne (seqz x), zero` with `beq x, zero`.
    if (isa<SeqzOp>(def) && isa<ReadRegOp>(zero) && REG(zero) == Reg::zero) {
      builder.replace<BeqOp>(op, { def->DEF(0), zero }, op->getAttrs());
      return true;
    }
    // Replace `bne (snez x), zero` with `bne x, zero`.
    if (isa<SnezOp>(def) && isa<ReadRegOp>(zero) && REG(zero) == Reg::zero) {
      builder.replace<BneOp>(op, { def->DEF(0), zero }, op->getAttrs());
      return true;
    }
    return false;
  });
  
  // Only run this after all int-related fold completes.
  // Rewrite `li a0, 0` into reading from `zero`.
  runRewriter([&](LiOp *op) {
    if (!enableLiZeroCombine)
      return false;
    if (V(op) == 0)
      builder.replace<ReadRegOp>(op, op->getResultType(), { new RegAttr(Reg::zero)} );
    
    return false;
  });
}
