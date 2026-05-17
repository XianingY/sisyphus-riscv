#include "RvPasses.h"
#include "Regs.h"
#include <cstdlib>
#include <cstring>

using namespace sys::rv;
using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback = true) {
  const char *v = std::getenv(name);
  if (!v || !v[0])
    return fallback;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0)
    return false;
  return true;
}

std::vector<Value::Type> getArgTypes(FuncOp *funcOp) {
  int argcnt = funcOp->get<ArgCountAttr>()->count;
  if (auto argTypes = funcOp->find<ArgTypesAttr>()) {
    if ((int) argTypes->types.size() == argcnt)
      return argTypes->types;
  }

  std::vector<Value::Type> types(argcnt, Value::i32);
  for (auto getarg : funcOp->findAll<GetArgOp>()) {
    int idx = V(getarg);
    if (idx >= 0 && idx < argcnt)
      types[idx] = getarg->getResultType();
  }
  return types;
}

int getStackArgIndex(const std::vector<Value::Type> &types, int index) {
  int intCount = 0;
  int fpCount = 0;
  int stackCount = 0;

  for (int i = 0; i <= index; i++) {
    bool fp = types[i] == Value::f32;
    int &regCount = fp ? fpCount : intCount;
    if (regCount < 8) {
      if (i == index)
        return -1;
      regCount++;
      continue;
    }

    if (i == index)
      return stackCount;
    stackCount++;
  }

  return -1;
}

template <typename BranchTy, typename InverseBranchTy>
bool rotateFallthroughLoop(BranchTy *op, Builder &builder) {
  if (!op->template has<TargetAttr>() || op->template has<ElseAttr>())
    return false;
  auto header = op->getParent();
  auto exit = op->template get<TargetAttr>()->bb;

  auto &blocks = header->getParent()->getBlocks();
  auto headerIt = blocks.begin();
  for (; headerIt != blocks.end() && *headerIt != header; ++headerIt) {}
  if (headerIt == blocks.end())
    return false;

  if (headerIt != blocks.begin()) {
    auto prevIt = headerIt;
    --prevIt;
    auto prev = *prevIt;
    if (prev->getOpCount() == 0)
      return false;
    auto prevTerm = prev->getLastOp();
    if (!isa<JOp>(prevTerm) && !isa<RetOp>(prevTerm))
      return false;
  }

  auto bodyIt = headerIt;
  ++bodyIt;
  if (bodyIt == blocks.end())
    return false;
  auto body = *bodyIt;
  auto afterBodyIt = bodyIt;
  ++afterBodyIt;

  if (!exit || body == exit)
    return false;
  if (afterBodyIt == blocks.end() || *afterBodyIt != exit)
    return false;
  auto bodyTerm = body->getLastOp();
  if (!bodyTerm || !isa<JOp>(bodyTerm) ||
      bodyTerm->template get<TargetAttr>()->bb != header)
    return false;
  if (!op->template has<RsAttr>() || !op->template has<Rs2Attr>())
    return false;

  body->moveBefore(header);
  bodyTerm->erase();
  builder.replace<InverseBranchTy>(op, std::vector<Attr*> {
    op->template get<RsAttr>(),
    op->template get<Rs2Attr>(),
    new TargetAttr(body),
  });
  return true;
}

}

#define CREATE_MV(fp, rd, rs) \
  if (!fp) \
    builder.create<MvOp>({ RDC(rd), RSC(rs) }); \
  else \
    builder.create<FmvOp>({ RDC(rd), RSC(rs) });

#define REPLACE_BRANCH(T1, T2) \
  REPLACE_BRANCH_IMPL(T1, T2); \
  REPLACE_BRANCH_IMPL(T2, T1)

// Say the before is `blt`, then we might see
//   blt %1 %2 <target = bb1> <else = bb2>
// which means `if (%1 < %2) goto bb1 else goto bb2`.
//
// If the next block is just <bb1>, then we flip it to bge, and make the target <bb2>.
// if the next block is <bb2>, then we make the target <bb2>.
// otherwise, make the target <bb1>, and add another `j <bb2>`.
#define REPLACE_BRANCH_IMPL(BeforeTy, AfterTy) \
  runRewriter(funcOp, [&](BeforeTy *op) { \
    if (!op->has<ElseAttr>()) \
      return false; \
    auto &target = TARGET(op); \
    auto ifnot = ELSE(op); \
    auto me = op->getParent(); \
    /* If there's no "next block", then give up */ \
    if (me == me->getParent()->getLastBlock()) { \
      GENERATE_J; \
      END_REPLACE; \
    } \
    if (me->nextBlock() == target) { \
      builder.replace<AfterTy>(op, { \
        op->get<RsAttr>(), \
        op->get<Rs2Attr>(), \
        new TargetAttr(ifnot), \
      }); \
      return true; \
    } \
    if (me->nextBlock() == ifnot) { \
      /* No changes needed. */\
      return false; \
    } \
    GENERATE_J; \
    END_REPLACE; \
  })

// Don't touch `target`.
#define GENERATE_J \
  builder.setAfterOp(op); \
  builder.create<JOp>({ new TargetAttr(ifnot) })

#define END_REPLACE \
  op->remove<ElseAttr>(); \
  return true
int RegAlloc::latePeephole(Op *funcOp) {
  Builder builder;
  auto inRange12 = [](int x) { return x > -2048 && x < 2048; };
  auto readsReg = [](Op *op, Reg reg) -> bool {
    return (op->has<RsAttr>() && RS(op) == reg) ||
           (op->has<Rs2Attr>() && RS2(op) == reg) ||
           (op->has<RegAttr>() && REG(op) == reg);
  };
  auto definesReg = [](Op *op, Reg reg) -> bool {
    return op->has<RdAttr>() && RD(op) == reg;
  };
  auto isFrameBase = [](Reg reg) -> bool {
    return reg == Reg::sp || reg == Reg::s0;
  };
  auto canTwoHopFold = [&](AddiOp *op) -> bool {
    if (!op->has<RdAttr>() || !op->has<RsAttr>())
      return false;
    if (isFrameBase(RS(op)) || isFrameBase(RD(op)))
      return false;
    return true;
  };
  auto canEraseAddrTmp = [&](AddiOp *op, Op *folded) -> bool {
    Reg rd = RD(op);
    Op *cur = folded;
    while (!cur->atBack()) {
      cur = cur->nextOp();
      if (readsReg(cur, rd))
        return false;
      if (definesReg(cur, rd))
        return true;
    }
    return true;
  };

  int converted = 0;

  runRewriter(funcOp, [&](LaOp *op) {
    auto prev = op->prevOp();
    if (!prev || !isa<LaOp>(prev))
      return false;
    if (NAME(prev) != NAME(op))
      return false;

    converted++;
    if (RD(prev) != RD(op)) {
      builder.setBeforeOp(op);
      builder.create<MvOp>({ RDC(RD(op)), RSC(RD(prev)) });
    }
    op->erase();
    return true;
  });

  runRewriter(funcOp, [&](LiOp *op) {
    auto prev = op->prevOp();
    if (!prev || !isa<LiOp>(prev))
      return false;
    if (!prev->has<RdAttr>() || !prev->has<IntAttr>() ||
        !op->has<RdAttr>() || !op->has<IntAttr>())
      return false;
    if (RD(prev) != RD(op) || V(prev) != V(op))
      return false;

    converted++;
    op->erase();
    return true;
  });

  runRewriter(funcOp, [&](StoreOp *op) {
    if (op->atBack())
      return false;

    //   sw a0, N(addr)
    //   lw a1, N(addr)
    // becomes
    //   sw a0, N(addr)
    //   mv a1, a0
    auto next = op->nextOp();
    if (isa<LoadOp>(next) &&
        RS(next) == RS2(op) && V(next) == V(op) && SIZE(next) == SIZE(op)) {
      converted++;
      builder.setBeforeOp(next);
      CREATE_MV(isFP(RD(next)), RD(next), RS(op));
      next->erase();
      return false;
    }

    return false;
  });

  if (envEnabled("SISY_RV_ENABLE_LOAD_LOAD_CSE", true)) {
    for (auto bb : funcOp->getRegion()->getBlocks()) {
      Op *op = bb->getFirstOp();
      while (op) {
        Op *next = op->atBack() ? nullptr : op->nextOp();
        Op *after = (next && !next->atBack()) ? next->nextOp() : nullptr;

        if (next && isa<LoadOp>(op) && isa<LoadOp>(next) &&
            op->has<RdAttr>() && op->has<RsAttr>() && op->has<IntAttr>() &&
            next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
            RS(op) == RS(next) && V(op) == V(next) && SIZE(op) == SIZE(next) &&
            op->getResultType() == next->getResultType() &&
            RD(op) != RS(next)) {
          converted++;
          builder.setBeforeOp(next);
          CREATE_MV(isFP(RD(next)), RD(next), RD(op));
          next->erase();
          op = after;
          continue;
        }

        if (next && isa<FldOp>(op) && isa<FldOp>(next) &&
            op->has<RdAttr>() && op->has<RsAttr>() && op->has<IntAttr>() &&
            next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
            RS(op) == RS(next) && V(op) == V(next) &&
            RD(op) != RS(next)) {
          converted++;
          builder.setBeforeOp(next);
          CREATE_MV(/*fp=*/true, RD(next), RD(op));
          next->erase();
          op = after;
          continue;
        }

        op = next;
      }
    }
  }

  runRewriter(funcOp, [&](FmvdxOp *op) {
    if (op->atBack())
      return false;

    auto next = op->nextOp();
    if (isa<FmvxdOp>(next) && RS(next) == RD(op)) {
      converted++;
      builder.setBeforeOp(next);
      CREATE_MV(isFP(RD(next)), RD(next), RS(op));
      next->erase();
      return false;
    }
    
    return false;
  });

  // Fold:
  //   addi rd, rs, c0
  //   lw/ld rd, c1(rd)
  // into:
  //   lw/ld rd, c0+c1(rs)
  runRewriter(funcOp, [&](AddiOp *op) {
    if (op->atBack())
      return false;
    if (!op->has<RdAttr>() || !op->has<RsAttr>() || !op->has<IntAttr>())
      return false;
    // Never fold stack-pointer updates into memory offsets.
    // `sp` arithmetic is frame-structure critical across prologue/epilogue.
    if (RD(op) == Reg::sp)
      return false;
    if (isFrameBase(RS(op)) || isFrameBase(RD(op)))
      return false;

    auto next = op->nextOp();
    int offset = V(op);
    if (isa<AddiOp>(next) &&
        next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
        RS(next) == RS(op) && V(next) == offset &&
        RD(next) != RD(op) &&
        RD(op) != Reg::sp && RD(next) != Reg::sp &&
        canTwoHopFold(op) && canTwoHopFold(cast<AddiOp>(next))) {
      converted++;
      builder.replace<MvOp>(next, { RDC(RD(next)), RSC(RD(op)) });
      return true;
    }

    if (isa<AddiOp>(next) &&
        canTwoHopFold(op) &&
        op->getUses().size() == 1 &&
        next->getUses().size() == 1 &&
        next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
        RD(next) != Reg::sp &&
        RS(next) == RD(op)) {
      auto mem = next->nextOp();
      int extra = V(next);
      bool folded = false;
      if (isa<LoadOp>(mem) && mem->has<RdAttr>() && mem->has<RsAttr>() && mem->has<IntAttr>() &&
          RS(mem) == RD(next) && inRange12(V(mem) + offset + extra) && canEraseAddrTmp(op, mem)) {
        RS(mem) = RS(op);
        V(mem) += offset + extra;
        folded = true;
      } else if (isa<FldOp>(mem) && mem->has<RdAttr>() && mem->has<RsAttr>() && mem->has<IntAttr>() &&
                 RS(mem) == RD(next) && inRange12(V(mem) + offset + extra) && canEraseAddrTmp(op, mem)) {
        RS(mem) = RS(op);
        V(mem) += offset + extra;
        folded = true;
      } else if (isa<StoreOp>(mem) && mem->has<RsAttr>() && mem->has<Rs2Attr>() && mem->has<IntAttr>() &&
                 RS2(mem) == RD(next) && RS(mem) != RD(next) &&
                 inRange12(V(mem) + offset + extra) && canEraseAddrTmp(op, mem)) {
        RS2(mem) = RS(op);
        V(mem) += offset + extra;
        folded = true;
      } else if (isa<FsdOp>(mem) && mem->has<RsAttr>() && mem->has<Rs2Attr>() && mem->has<IntAttr>() &&
                 RS2(mem) == RD(next) && RS(mem) != RD(next) &&
                 inRange12(V(mem) + offset + extra) && canEraseAddrTmp(op, mem)) {
        RS2(mem) = RS(op);
        V(mem) += offset + extra;
        folded = true;
      }
      if (folded) {
        converted++;
        next->erase();
        op->erase();
        return true;
      }
    }
    if (isa<LoadOp>(next) && next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
        RS(next) == RD(op) && RD(next) == RD(op) && inRange12(V(next) + offset) &&
        canEraseAddrTmp(op, next)) {
      converted++;
      RS(next) = RS(op);
      V(next) += offset;
      op->erase();
      return true;
    }
    if (isa<FldOp>(next) && next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
        RS(next) == RD(op) && RD(next) == RD(op) && inRange12(V(next) + offset) &&
        canEraseAddrTmp(op, next)) {
      converted++;
      RS(next) = RS(op);
      V(next) += offset;
      op->erase();
      return true;
    }
    if (isa<StoreOp>(next) && next->has<RsAttr>() && next->has<Rs2Attr>() && next->has<IntAttr>() &&
        RS2(next) == RD(op) && RS(next) != RD(op) && inRange12(V(next) + offset) &&
        canEraseAddrTmp(op, next)) {
      converted++;
      RS2(next) = RS(op);
      V(next) += offset;
      op->erase();
      return true;
    }
    if (isa<MvOp>(next) && !next->atBack() &&
        canTwoHopFold(op) &&
        op->getUses().size() == 1 &&
        RS(next) == RD(op)) {
      auto mem = next->nextOp();
      bool folded = false;
      if (isa<LoadOp>(mem) && mem->has<RsAttr>() && mem->has<IntAttr>() &&
          RS(mem) == RD(next) && inRange12(V(mem) + offset) && canEraseAddrTmp(op, mem)) {
        RS(mem) = RS(op);
        V(mem) += offset;
        folded = true;
      } else if (isa<FldOp>(mem) && mem->has<RsAttr>() && mem->has<IntAttr>() &&
                 RS(mem) == RD(next) && inRange12(V(mem) + offset) && canEraseAddrTmp(op, mem)) {
        RS(mem) = RS(op);
        V(mem) += offset;
        folded = true;
      } else if (isa<StoreOp>(mem) && mem->has<RsAttr>() && mem->has<Rs2Attr>() && mem->has<IntAttr>() &&
                 RS2(mem) == RD(next) && RS(mem) != RD(next) &&
                 inRange12(V(mem) + offset) && canEraseAddrTmp(op, mem)) {
        RS2(mem) = RS(op);
        V(mem) += offset;
        folded = true;
      }
      if (folded) {
        converted++;
        next->erase();
        op->erase();
        return true;
      }
    }
    if (isa<MvOp>(next) && !next->atBack() &&
        canTwoHopFold(op) &&
        op->getUses().size() == 1 &&
        next->getUses().size() == 1 &&
        RS(next) == RD(op)) {
      auto mid = next->nextOp();
      if (!isa<MvOp>(mid) || mid->atBack() || RS(mid) != RD(next))
        return false;
      auto mem = mid->nextOp();
      bool folded = false;
      if (isa<LoadOp>(mem) && mem->has<RsAttr>() && mem->has<IntAttr>() &&
          RS(mem) == RD(mid) && inRange12(V(mem) + offset) && canEraseAddrTmp(op, mem)) {
        RS(mem) = RS(op);
        V(mem) += offset;
        folded = true;
      } else if (isa<FldOp>(mem) && mem->has<RsAttr>() && mem->has<IntAttr>() &&
                 RS(mem) == RD(mid) && inRange12(V(mem) + offset) && canEraseAddrTmp(op, mem)) {
        RS(mem) = RS(op);
        V(mem) += offset;
        folded = true;
      } else if (isa<StoreOp>(mem) && mem->has<RsAttr>() && mem->has<Rs2Attr>() && mem->has<IntAttr>() &&
                 RS2(mem) == RD(mid) && RS(mem) != RD(mid) &&
                 inRange12(V(mem) + offset) && canEraseAddrTmp(op, mem)) {
        RS2(mem) = RS(op);
        V(mem) += offset;
        folded = true;
      }
      if (folded) {
        converted++;
        mid->erase();
        next->erase();
        op->erase();
        return true;
      }
    }
    return false;
  });

  bool changed;
  std::vector<Op*> stores;
  do {
    changed = false;
    // This modifies the content of stores, so cannot run in a rewriter.
    stores = funcOp->findAll<StoreOp>();
    for (auto op : stores) {
      if (op == op->getParent()->getLastOp())
        continue;
      auto next = op->nextOp();

      //   sw zero, N(sp)
      //   sw zero, N+4(sp)
      // becomes
      //   sd zero, N(sp)
      // only when N is a multiple of 8.
      //
      // We know `sp` is 16-aligned, but we don't know for other registers.
      // That's why we fold it only for `sp`.
      if (isa<StoreOp>(next) &&
          RS(op) == Reg::zero && RS2(op) == Reg::sp &&
          RS(next) == Reg::zero && RS2(next) == Reg::sp &&
          V(op) % 8 == 0 && SIZE(op) == 4 &&
          V(next) == V(op) + 4 && SIZE(next) == 4) {
        converted++;
        changed = true;

        auto offset = V(op);
        builder.replace<StoreOp>(op, {
          RSC(Reg::zero),
          RS2C(Reg::sp),
          new IntAttr(offset),
          new SizeAttr(8),
        });
        next->erase();
        break;
      }

      // Similarly:
      //   sw zero, N(sp)
      //   sw zero, N-4(sp)
      // becomes
      //   sd zero, N-4(sp)
      // only when N-4 is a multiple of 8.
      if (isa<StoreOp>(next) &&
          RS(op) == Reg::zero && RS2(op) == Reg::sp &&
          RS(next) == Reg::zero && RS2(next) == Reg::sp &&
          V(op) % 8 == 4 && SIZE(op) == 4 &&
          V(next) == V(op) - 4 && SIZE(next) == 4) {
        converted++;
        changed = true;

        auto offset = V(op);
        builder.replace<StoreOp>(op, {
          RSC(Reg::zero),
          RS2C(Reg::sp),
          new IntAttr(offset - 4),
          new SizeAttr(8),
        });
        next->erase();
        break;
      }
    }
  } while (changed);

  // Eliminate useless MvOp.
  runRewriter(funcOp, [&](MvOp *op) {
    if (RD(op) == RS(op) || RD(op) == Reg::zero) {
      converted++;
      op->erase();
      return true;
    }

    // mv  a0, a1    <-- op
    // mv  a1, a0    <-- mv2
    // We can delete the second operation, `mv2`.
    if (op != op->getParent()->getLastOp()) {
      auto mv2 = op->nextOp();
      if (isa<MvOp>(mv2) && RD(op) == RS(mv2) && RS(op) == RD(mv2)) {
        // We can't erase `mv2` because it might be explored afterwards.
        // We need to change the content of mv2 and erase this one.
        op->erase();
        std::swap(RD(mv2), RS(mv2));
        converted++;
        return true;
      }
    }
    if (!op->atBack()) {
      auto next = op->nextOp();
      if (isa<MvOp>(next) &&
          RS(next) == RD(op) &&
          op->getUses().size() == 1) {
        RS(next) = RS(op);
        converted++;
        op->erase();
        return true;
      }
      if (definesReg(next, RD(op)) && !readsReg(next, RD(op))) {
        converted++;
        op->erase();
        return true;
      }

      // Propagate a single-use move into the following arithmetic op.
      // This must substitute the move source, not the arithmetic destination:
      //   mv t0, a0; addw a1, t0, t0  ->  addw a1, a0, a0
      // Replacing with RD(next) corrupts cases where RD(next) is unrelated.
      if (envEnabled("SISY_RV_ENABLE_MOVE_ARITH_PEEPHOLE", true) &&
          (isa<MulwOp>(next) || isa<AddwOp>(next) || isa<SubwOp>(next)) &&
          op->getUses().size() == 1 &&
          *op->getUses().begin() == next) {
        Reg mvSrc = RS(op);
        Reg mvDst = RD(op);

        bool changed = false;
        if (RS(next) == mvDst) {
          RS(next) = mvSrc;
          changed = true;
        }
        if (RS2(next) == mvDst) {
          RS2(next) = mvSrc;
          changed = true;
        }
        if (changed) {
          converted++;
          op->erase();
          return true;
        }
      }
    }
    return false;
  });

  runRewriter(funcOp, [&](FmvOp *op) {
    if (RD(op) == RS(op)) {
      converted++;
      op->erase();
      return true;
    }
    if (!op->atBack()) {
      auto next = op->nextOp();
      if (isa<FmvOp>(next) &&
          RS(next) == RD(op) &&
          op->getUses().size() == 1) {
        RS(next) = RS(op);
        converted++;
        op->erase();
        return true;
      }
      if (definesReg(next, RD(op)) && !readsReg(next, RD(op))) {
        converted++;
        op->erase();
        return true;
      }
    }
    return false;
  });

  runRewriter(funcOp, [&](LiOp *op) {
    if (RD(op) == Reg::zero) {
      converted++;
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter(funcOp, [&](AddiOp *op) {
    if (RD(op) == Reg::zero) {
      converted++;
      op->erase();
      return true;
    }
    return false;
  });

  return converted;
}

void RegAlloc::tidyup(Region *region) {
  Builder builder;
  auto funcOp = region->getParent();
  region->updatePreds();

  // Replace blocks with only a single `j` as terminator.
  std::map<BasicBlock*, BasicBlock*> jumpTo;
  for (auto bb : region->getBlocks()) {
    if (bb->getOpCount() == 1 && isa<JOp>(bb->getLastOp())) {
      auto target = bb->getLastOp()->get<TargetAttr>()->bb;
      jumpTo[bb] = target;
    }
  }

  // Calculate jump-to closure.
  bool changed;
  do {
    changed = false;
    for (auto [k, v] : jumpTo) {
      if (jumpTo.count(v)) {
        jumpTo[k] = jumpTo[v];
        changed = true;
      }
    }
  } while (changed);

  for (auto bb : region->getBlocks()) { 
    if (bb->getOpCount() == 0)
      continue;
    auto term = bb->getLastOp();
    if (auto target = term->find<TargetAttr>()) {
      if (jumpTo.count(target->bb))
        target->bb = jumpTo[target->bb];
    }

    if (auto ifnot = term->find<ElseAttr>()) {
      if (jumpTo.count(ifnot->bb))
        ifnot->bb = jumpTo[ifnot->bb];
    }
  }

  // Erase all those single-j's.
  region->updatePreds();
  for (auto [bb, v] : jumpTo)
    bb->erase();
  
  // After lowering, combine sequential blocks into one.
  // Simpler than simplify-cfg, because no phis could remain now.
  do {
    changed = false;
    const auto &bbs = region->getBlocks();
    for (auto bb : bbs) {
      if (bb->succs.size() != 1)
        continue;

      auto succ = *bb->succs.begin();
      if (succ->preds.size() != 1)
        continue;

      // Remove the jump to `succ`.
      auto term = bb->getLastOp();
      if (isa<JOp>(term))
        term->erase();
      
      // All successors of `succ` now have pred `bb`.
      // `bb` also regard them as successors.
      for (auto s : succ->succs) {
        s->preds.erase(succ);
        s->preds.insert(bb);
        bb->succs.insert(s);
      }
      // Remove `succ` from the successors of `bb`.
      bb->succs.erase(succ);

      // Then move all instruction in `succ` to `bb`.
      auto ops = succ->getOps();
      for (auto op : ops)
        op->moveToEnd(bb);

      succ->forceErase();
      changed = true;
      break;
    }
  } while (changed);

  // Now branches are still having both TargetAttr and ElseAttr.
  // Replace them (perform split when necessary), so that they only have one target.
  runRewriter(funcOp, [&](BltOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<JOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BgeOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<JOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BleOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<JOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BgtOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<JOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BeqOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<JOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BneOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<JOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });

  REPLACE_BRANCH(BltOp, BgeOp);
  REPLACE_BRANCH(BeqOp, BneOp);
  REPLACE_BRANCH(BleOp, BgtOp);

  {
    auto blocks = region->getBlocks();
    for (auto bb : blocks) {
      if (!bb || bb->getOpCount() == 0)
        continue;
      auto term = bb->getLastOp();
      if (auto op = dyn_cast<BgeOp>(term))
        changed |= rotateFallthroughLoop<BgeOp, BltOp>(op, builder);
      else if (auto op = dyn_cast<BltOp>(term))
        changed |= rotateFallthroughLoop<BltOp, BgeOp>(op, builder);
      else if (auto op = dyn_cast<BgtOp>(term))
        changed |= rotateFallthroughLoop<BgtOp, BleOp>(op, builder);
      else if (auto op = dyn_cast<BleOp>(term))
        changed |= rotateFallthroughLoop<BleOp, BgtOp>(op, builder);
      else if (auto op = dyn_cast<BeqOp>(term))
        changed |= rotateFallthroughLoop<BeqOp, BneOp>(op, builder);
      else if (auto op = dyn_cast<BneOp>(term))
        changed |= rotateFallthroughLoop<BneOp, BeqOp>(op, builder);
    }
  }

  auto inRange12 = [](int x) { return x > -2048 && x < 2048; };
  auto legalizeLargeMemOffsets = [&]() {
    runRewriter(funcOp, [&](sys::rv::LoadOp *op) {
      if (inRange12(V(op)))
        return false;

      Reg base = RS(op);
      Reg dst = RD(op);
      Reg tmp = (base == spillReg2 || dst == spillReg2) ? spillReg : spillReg2;

      builder.setBeforeOp(op);
      builder.create<LiOp>({ RDC(tmp), new IntAttr(V(op)) });
      builder.create<AddOp>({ RDC(tmp), RSC(tmp), RS2C(base) });
      builder.replace<sys::rv::LoadOp>(op, op->getResultType(), {
        RDC(dst), RSC(tmp), new IntAttr(0), new SizeAttr(SIZE(op))
      });
      return true;
    });

    runRewriter(funcOp, [&](sys::rv::StoreOp *op) {
      if (inRange12(V(op)))
        return false;

      Reg src = RS(op);
      Reg base = RS2(op);
      Reg tmp = (src == spillReg2 || base == spillReg2) ? spillReg : spillReg2;

      builder.setBeforeOp(op);
      builder.create<LiOp>({ RDC(tmp), new IntAttr(V(op)) });
      builder.create<AddOp>({ RDC(tmp), RSC(tmp), RS2C(base) });
      builder.replace<sys::rv::StoreOp>(op, {
        RSC(src), RS2C(tmp), new IntAttr(0), new SizeAttr(SIZE(op))
      });
      return true;
    });

    runRewriter(funcOp, [&](FldOp *op) {
      if (inRange12(V(op)))
        return false;

      Reg base = RS(op);
      Reg dst = RD(op);
      Reg tmp = (base == spillReg2 || dst == spillReg2) ? spillReg : spillReg2;

      builder.setBeforeOp(op);
      builder.create<LiOp>({ RDC(tmp), new IntAttr(V(op)) });
      builder.create<AddOp>({ RDC(tmp), RSC(tmp), RS2C(base) });
      builder.replace<FldOp>(op, { RDC(dst), RSC(tmp), new IntAttr(0) });
      return true;
    });

    runRewriter(funcOp, [&](FsdOp *op) {
      if (inRange12(V(op)))
        return false;

      Reg src = RS(op);
      Reg base = RS2(op);
      Reg tmp = (src == spillReg2 || base == spillReg2) ? spillReg : spillReg2;

      builder.setBeforeOp(op);
      builder.create<LiOp>({ RDC(tmp), new IntAttr(V(op)) });
      builder.create<AddOp>({ RDC(tmp), RSC(tmp), RS2C(base) });
      builder.replace<FsdOp>(op, { RSC(src), RS2C(tmp), new IntAttr(0) });
      return true;
    });
  };

  legalizeLargeMemOffsets();

  int converted;
  do {
    converted = latePeephole(funcOp);
    convertedTotal += converted;
  } while (converted);

  legalizeLargeMemOffsets();

  // A second CFG cleanup can expose more branch-to-next opportunities
  // after late peephole and mem-offset legalization.
  do {
    changed = false;
    const auto &bbs = region->getBlocks();
    for (auto bb : bbs) {
      if (bb->succs.size() != 1)
        continue;

      auto succ = *bb->succs.begin();
      if (succ->preds.size() != 1)
        continue;

      auto term = bb->getLastOp();
      if (isa<JOp>(term))
        term->erase();

      for (auto s : succ->succs) {
        s->preds.erase(succ);
        s->preds.insert(bb);
        bb->succs.insert(s);
      }
      bb->succs.erase(succ);

      auto ops = succ->getOps();
      for (auto op : ops)
        op->moveToEnd(bb);

      succ->forceErase();
      changed = true;
      break;
    }
  } while (changed);

  // Also, eliminate useless JOp.
  runRewriter(funcOp, [&](JOp *op) {
    auto &target = TARGET(op);
    auto me = op->getParent();
    if (me == me->getParent()->getLastBlock())
      return false;

    if (me->nextBlock() == target) {
      op->erase();
      return true;
    }
    return false;
  });
}

#define CREATE_STORE(addr, offset) \
  if (isFP(reg)) \
    builder.create<FsdOp>({ RSC(reg), RS2C(addr), new IntAttr(offset) }); \
  else \
    builder.create<StoreOp>({ RSC(reg), RS2C(addr), new IntAttr(offset), new SizeAttr(8) });

void save(Builder builder, const std::vector<Reg> &regs, int offset) {
  using sys::rv::StoreOp;

  for (auto reg : regs) {
    offset -= 8;
    if (offset < 2048)
      CREATE_STORE(Reg::sp, offset)
    else {
      // li   t6, offset
      // addi t6, t6, sp
      // sd   reg, 0(t6)
      // Use a dedicated scratch that never appears in callee-saved restore list.
      Reg scratch = Reg::t6;
      if (scratch == reg)
        scratch = Reg::t5;
      builder.create<LiOp>({ RDC(scratch), new IntAttr(offset) });
      builder.create<AddOp>({ RDC(scratch), RSC(scratch), RS2C(Reg::sp) });
      CREATE_STORE(scratch, 0);
    }
  }
}

#define CREATE_LOAD(addr, offset) \
  if (isFP(reg)) \
    builder.create<FldOp>({ RDC(reg), RSC(addr), new IntAttr(offset) }); \
  else \
    builder.create<LoadOp>(ty, { RDC(reg), RSC(addr), new IntAttr(offset), new SizeAttr(8) });

namespace {

bool fitsImm12(int x) {
  return x > -2048 && x < 2048;
}

void materializeSpAddr(Builder &builder, Reg tmp, int offset) {
  if (fitsImm12(offset))
    builder.create<AddiOp>({ RDC(tmp), RSC(Reg::sp), new IntAttr(offset) });
  else {
    builder.create<LiOp>({ RDC(tmp), new IntAttr(offset) });
    builder.create<AddOp>({ RDC(tmp), RSC(tmp), RS2C(Reg::sp) });
  }
}

void emitStackLoad(Builder &builder, Reg dst, Value::Type ty, int offset) {
  int size = ty == Value::i64 ? 8 : 4;
  if (fitsImm12(offset)) {
    builder.create<sys::rv::LoadOp>(ty, {
      RDC(dst),
      RSC(Reg::sp),
      new IntAttr(offset),
      new SizeAttr(size)
    });
    return;
  }

  Reg scratch = (dst == spillReg2 ? spillReg : spillReg2);
  materializeSpAddr(builder, scratch, offset);
  builder.create<sys::rv::LoadOp>(ty, {
    RDC(dst),
    RSC(scratch),
    new IntAttr(0),
    new SizeAttr(size)
  });
}

void emitSpAdjust(Builder &builder, int delta) {
  while (delta != 0) {
    int step = delta;
    if (step > 2047)
      step = 2047;
    if (step < -2047)
      step = -2047;
    builder.create<AddiOp>({ RDC(Reg::sp), RSC(Reg::sp), new IntAttr(step) });
    delta -= step;
  }
}

}

void load(Builder builder, const std::vector<Reg> &regs, int offset) {
  using sys::rv::LoadOp;

  for (auto reg : regs) {
    offset -= 8;
    auto isFloat = isFP(reg);
    Value::Type ty = isFloat ? Value::f32 : Value::i64;
    if (offset < 2048)
      CREATE_LOAD(Reg::sp, offset)
    else {
      // li   s11, offset
      // addi s11, s11, sp
      // ld   reg, 0(s11)
      Reg scratch = Reg::t6;
      if (scratch == reg)
        scratch = Reg::t5;
      builder.create<LiOp>({ RDC(scratch), new IntAttr(offset) });
      builder.create<AddOp>({ RDC(scratch), RSC(scratch), RS2C(Reg::sp) });
      CREATE_LOAD(scratch, 0);
    }
  }
}

void RegAlloc::proEpilogue(FuncOp *funcOp, bool isLeaf) {
  Builder builder;
  auto usedRegs = usedRegisters[funcOp];
  auto region = funcOp->getRegion();

  // Preserve return address if this calls another function.
  std::vector<Reg> preserve;
  for (auto x : usedRegs) {
    if (calleeSaved.count(x))
      preserve.push_back(x);
  }
  if (!isLeaf)
    preserve.push_back(Reg::ra);

  // If there's a SubSpOp, then it must be at the top of the first block.
  int &offset = STACKOFF(funcOp);
  offset += 8 * preserve.size();

  // Round op to the nearest multiple of 16.
  // This won't be entered in the special case where offset == 0.
  if (offset % 16 != 0)
    offset = offset / 16 * 16 + 16;

  // Add function prologue, preserving the regs.
  auto entry = region->getFirstBlock();
  builder.setToBlockStart(entry);
  if (offset != 0)
    builder.create<SubSpOp>({ new IntAttr(offset) });
  
  save(builder, preserve, offset);

  // Similarly add function epilogue.
  if (offset != 0) {
    auto rets = funcOp->findAll<RetOp>();
    auto bb = region->appendBlock();
    for (auto ret : rets)
      builder.replace<JOp>(ret, { new TargetAttr(bb) });

    builder.setToBlockStart(bb);

    load(builder, preserve, offset);
    builder.create<SubSpOp>({ new IntAttr(-offset) });
    builder.create<RetOp>();
  }

  // Caller preserved registers are marked correctly as interfered,
  // because of the placeholders.

  // Deal with remaining GetArg.
  // The arguments passed by registers have already been eliminated.
  // Now all remaining ones are passed on stack; sort them according to index.
  auto remainingGets = funcOp->findAll<GetArgOp>();
  std::sort(remainingGets.begin(), remainingGets.end(), [](Op *a, Op *b) {
    return V(a) < V(b);
  });
  std::map<Op*, int> argOffsets;
  auto argTypes = getArgTypes(cast<FuncOp>(funcOp));

  for (auto op : remainingGets) {
    int stackIndex = getStackArgIndex(argTypes, V(op));
    if (stackIndex >= 0)
      argOffsets[op] = stackIndex * 8;
  }

  runRewriter(funcOp, [&](GetArgOp *op) {
    // `sp + offset` is the base pointer.
    // We read past the base pointer (starting from 0):
    //    <arg 8> bp + 0
    //    <arg 9> bp + 8
    // ...
    assert(argOffsets.count(op));
    int myoffset = offset + argOffsets[op];
    Value::Type loadTy = argTypes[V(op)];
    builder.setBeforeOp(op);
    emitStackLoad(builder, RD(op), loadTy, myoffset);
    auto created = op->prevOp();
    if (created)
      op->replaceAllUsesWith(created);
    op->erase();
    return false;
  });

  //   subsp <4>
  // becomes
  //   addi <rd = sp> <rs = sp> <-4>
  runRewriter(funcOp, [&](SubSpOp *op) {
    builder.setBeforeOp(op);
    emitSpAdjust(builder, -V(op));
    op->erase();
    return true;
  });
}
