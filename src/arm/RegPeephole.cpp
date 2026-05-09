#include "ArmPasses.h"
#include "Regs.h"

using namespace sys::arm;
using namespace sys;

namespace {

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

}

#define REPLACE_BRANCH(T1, T2, ...) \
  REPLACE_BRANCH_IMPL(T1, T2, __VA_ARGS__); \
  REPLACE_BRANCH_IMPL(T2, T1, __VA_ARGS__)

// Say the before is `blt`, then we might see
//   blt %1 %2 <target = bb1> <else = bb2>
// which means `if (%1 < %2) goto bb1 else goto bb2`.
//
// If the next block is just <bb1>, then we flip it to bge, and make the target <bb2>.
// if the next block is <bb2>, then we make the target <bb2>.
// otherwise, make the target <bb1>, and add another `j <bb2>`.
#define REPLACE_BRANCH_IMPL(BeforeTy, AfterTy, ...) \
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
        new TargetAttr(ifnot), \
        __VA_ARGS__ \
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
  builder.create<BOp>({ new TargetAttr(ifnot) })

#define END_REPLACE \
  op->remove<ElseAttr>(); \
  return true

#define UNARY_BRANCH RSC(RS(op))
#define BINARY_BRANCH RSC(RS(op)), RS2C(RS2(op))

#define CREATE_MV(fp, rd, rs) \
  if (!fp) \
    builder.create<MovROp>({ RDC(rd), RSC(rs) }); \
  else \
    builder.create<FmovOp>({ RDC(rd), RSC(rs) });

int RegAlloc::latePeephole(Op *funcOp) {
  Builder builder;
  constexpr int kLargeFrameFoldThreshold = 4096;
  const bool armLargeFrame =
    funcOp->has<StackOffsetAttr>() && STACKOFF(funcOp) >= kLargeFrameFoldThreshold;
  auto inSignedUnscaled = [](int x) { return x >= -256 && x <= 255; };
  auto inUnsignedScaled = [](int x, int scale) {
    return x >= 0 && x % scale == 0 && x / scale <= 4095;
  };
  auto validMemOffset = [&](Op *mem, int x) -> bool {
    int scale = 0;
    if (isa<LdrXOp>(mem) || isa<StrXOp>(mem) || isa<LdrDOp>(mem) || isa<StrDOp>(mem))
      scale = 8;
    if (isa<LdrWOp>(mem) || isa<StrWOp>(mem) || isa<LdrFOp>(mem) || isa<StrFOp>(mem))
      scale = 4;
    if (scale == 0)
      return false;
    return inSignedUnscaled(x) || inUnsignedScaled(x, scale);
  };
  auto readsReg = [](Op *op, Reg reg) -> bool {
    return (op->has<RsAttr>() && RS(op) == reg) ||
           (op->has<Rs2Attr>() && RS2(op) == reg) ||
           (op->has<Rs3Attr>() && RS3(op) == reg) ||
           (op->has<RegAttr>() && REG(op) == reg);
  };
  auto definesReg = [](Op *op, Reg reg) -> bool {
    return op->has<RdAttr>() && RD(op) == reg;
  };
  auto isFrameBase = [](Reg reg) -> bool {
    return reg == Reg::sp || reg == Reg::x29;
  };
  auto canTwoHopFold = [&](AddXIOp *op) -> bool {
    if (!op->has<RdAttr>() || !op->has<RsAttr>())
      return false;
    if (armLargeFrame)
      return false;
    if (isFrameBase(RS(op)) || isFrameBase(RD(op)))
      return false;
    return true;
  };
  auto canEraseAddrTmp = [&](AddXIOp *op, Op *folded) -> bool {
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

  runRewriter(funcOp, [&](AdrOp *op) {
    auto prev = op->prevOp();
    if (!prev || !isa<AdrOp>(prev))
      return false;
    if (NAME(prev) != NAME(op))
      return false;

    converted++;
    if (RD(prev) != RD(op)) {
      builder.setBeforeOp(op);
      builder.create<MovROp>({ RDC(RD(op)), RSC(RD(prev)) });
    }
    op->erase();
    return true;
  });

  runRewriter(funcOp, [&](StrWOp *op) {
    if (op == op->getParent()->getLastOp())
      return false;
    if (!op->has<RsAttr>() || !op->has<Rs2Attr>() || !op->has<SizeAttr>() || !op->has<IntAttr>())
      return false;

    //   sw a0, N(addr)
    //   lw a1, N(addr)
    // becomes
    //   sw a0, N(addr)
    //   mv a1, a0
    auto next = op->nextOp();
    if (isa<LdrWOp>(next) &&
        next->has<RsAttr>() && next->has<RdAttr>() && next->has<SizeAttr>() && next->has<IntAttr>() &&
        RS(next) == RS2(op) && V(next) == V(op) && SIZE(next) == SIZE(op)) {
      converted++;
      builder.setBeforeOp(next);
      CREATE_MV(isFP(RD(next)), RD(next), RS(op));
      next->erase();
      return true;
    }

    return false;
  });

  runRewriter(funcOp, [&](StrXOp *op) {
    if (op == op->getParent()->getLastOp())
      return false;
    if (!op->has<RsAttr>() || !op->has<Rs2Attr>() || !op->has<IntAttr>())
      return false;

    auto next = op->nextOp();
    if (isa<LdrXOp>(next) &&
        next->has<RsAttr>() && next->has<RdAttr>() && next->has<IntAttr>() &&
        RS(next) == RS2(op) && V(next) == V(op)) {
      converted++;
      builder.setBeforeOp(next);
      CREATE_MV(false, RD(next), RS(op));
      next->erase();
      return true;
    }

    return false;
  });

  runRewriter(funcOp, [&](StrFOp *op) {
    if (op == op->getParent()->getLastOp())
      return false;
    if (!op->has<RsAttr>() || !op->has<Rs2Attr>() || !op->has<IntAttr>())
      return false;

    auto next = op->nextOp();
    if (isa<LdrFOp>(next) &&
        next->has<RsAttr>() && next->has<RdAttr>() && next->has<IntAttr>() &&
        RS(next) == RS2(op) && V(next) == V(op)) {
      converted++;
      builder.setBeforeOp(next);
      CREATE_MV(true, RD(next), RS(op));
      next->erase();
      return true;
    }

    return false;
  });

  runRewriter(funcOp, [&](StrDOp *op) {
    if (op == op->getParent()->getLastOp())
      return false;
    if (!op->has<RsAttr>() || !op->has<Rs2Attr>() || !op->has<IntAttr>())
      return false;

    auto next = op->nextOp();
    if (isa<LdrDOp>(next) &&
        next->has<RsAttr>() && next->has<RdAttr>() && next->has<IntAttr>() &&
        RS(next) == RS2(op) && V(next) == V(op)) {
      converted++;
      builder.setBeforeOp(next);
      CREATE_MV(true, RD(next), RS(op));
      next->erase();
      return true;
    }

    return false;
  });

  bool changed;
  std::vector<Op*> stores;
  do {
    changed = false;
    // This modifies the content of stores, so cannot run in a rewriter.
    stores = funcOp->findAll<StrWOp>();
    for (auto op : stores) {
      if (op == op->getParent()->getLastOp())
        continue;
      auto next = op->nextOp();

      //   str wzr, N(sp)
      //   str wzr, N+4(sp)
      // becomes
      //   str xzr, N(sp)
      // only when N is a multiple of 8.
      //
      // We know `sp` is 16-aligned, but we don't know for other registers.
      // That's why we fold it only for `sp`.
      if (isa<StrWOp>(next) &&
          RS(op) == Reg::xzr && RS2(op) == Reg::sp &&
          RS(next) == Reg::xzr && RS2(next) == Reg::sp &&
          V(op) % 8 == 0 && V(next) == V(op) + 4) {
        converted++;
        changed = true;

        auto offset = V(op);
        builder.replace<StrXOp>(op, {
          RSC(Reg::xzr),
          RS2C(Reg::sp),
          new IntAttr(offset),
        });
        next->erase();
        break;
      }
      
      if (isa<StrWOp>(next) &&
          RS(op) == Reg::xzr && RS2(op) == Reg::sp &&
          RS(next) == Reg::xzr && RS2(next) == Reg::sp &&
          V(op) % 8 == 4 && V(next) == V(op) - 4) {
        converted++;
        changed = true;

        auto offset = V(op);
        builder.replace<StrWOp>(op, {
          RSC(Reg::xzr),
          RS2C(Reg::sp),
          new IntAttr(offset - 4),
        });
        next->erase();
        break;
      }
    }
  } while (changed);

  runRewriter(funcOp, [&](MulVOp *op) {
    if (op == op->getParent()->getLastOp())
      return false;

    //   mul v0, v1, v2
    //   add v3, v3, v0 (or in the opposite order)
    // becomes 
    //   mla v3, v1, v2
    // This isn't doable in inst-select because it modifies v3 inplace.
    auto next = op->nextOp();
    if (isa<AddVOp>(next) &&
       (RS(next) == RD(next) && RS2(next) == RD(op)
     || RS2(next) == RD(next) && RS(next) == RD(op))) {
      converted++;
      builder.setBeforeOp(next);
      builder.replace<MlaVOp>(next, { RDC(RD(next)), RSC(RS(op)), RS2C(RS2(op)) });
      op->erase();
    }

    return false;
  });

  // Fold:
  //   add xN, xM, #c0
  //   ldr/str ..., [xN, #c1]
  // into:
  //   ldr/str ..., [xM, #c0+c1]
  runRewriter(funcOp, [&](AddXIOp *op) {
    if (op->atBack())
      return false;
    if (!op->has<RdAttr>() || !op->has<RsAttr>() || !op->has<IntAttr>())
      return false;
    // Never fold stack-pointer update into mem offsets.
    // `sp` updates are frame-structure critical and cross-block/implicit uses
    // (calls, ret paths) are not fully visible to this local peephole.
    if (RD(op) == Reg::sp)
      return false;
    if (isFrameBase(RS(op)) || isFrameBase(RD(op)))
      return false;

    auto next = op->nextOp();
    int offset = V(op);
    if (isa<AddXIOp>(next) &&
        next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
        RS(next) == RS(op) && V(next) == offset &&
        RD(next) != RD(op) &&
        RD(op) != Reg::sp && RD(next) != Reg::sp &&
        canTwoHopFold(op) && canTwoHopFold(cast<AddXIOp>(next))) {
      converted++;
      builder.replace<MovROp>(next, { RDC(RD(next)), RSC(RD(op)) });
      return true;
    }

    if (isa<AddXIOp>(next) &&
        canTwoHopFold(op) &&
        op->getUses().size() == 1 &&
        next->getUses().size() == 1 &&
        next->has<RdAttr>() && next->has<RsAttr>() && next->has<IntAttr>() &&
        RD(next) != Reg::sp &&
        RS(next) == RD(op)) {
      auto mem = next->nextOp();
      int extra = V(next);
      bool folded = false;
      if (isa<LdrXOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrWOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrFOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrDOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrXOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrWOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrFOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrDOp>(mem)) {
        int nextOffset = V(mem) + offset + extra;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      }
      if (folded) {
        converted++;
        next->erase();
        op->erase();
        return true;
      }
    }

    if (isa<MovROp>(next) && !next->atBack() &&
        canTwoHopFold(op) &&
        op->getUses().size() == 1 &&
        RS(next) == RD(op)) {
      auto mem = next->nextOp();
      bool folded = false;
      if (isa<LdrXOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrWOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrFOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrDOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(next) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrXOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrWOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrFOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrDOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(next) && RS(mem) != RD(next) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      }

      if (folded) {
        converted++;
        next->erase();
        op->erase();
        return true;
      }
    }
    if (isa<MovROp>(next) && !next->atBack() &&
        canTwoHopFold(op) &&
        op->getUses().size() == 1 &&
        next->getUses().size() == 1 &&
        RS(next) == RD(op)) {
      auto mid = next->nextOp();
      if (!isa<MovROp>(mid) || mid->atBack() || RS(mid) != RD(next))
        return false;
      auto mem = mid->nextOp();
      bool folded = false;
      if (isa<LdrXOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(mid) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrWOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(mid) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrFOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(mid) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<LdrDOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS(mem) == RD(mid) && validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrXOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(mid) && RS(mem) != RD(mid) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrWOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(mid) && RS(mem) != RD(mid) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrFOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(mid) && RS(mem) != RD(mid) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      } else if (isa<StrDOp>(mem)) {
        int nextOffset = V(mem) + offset;
        if (RS2(mem) == RD(mid) && RS(mem) != RD(mid) &&
            validMemOffset(mem, nextOffset) && canEraseAddrTmp(op, mem)) {
          RS2(mem) = RS(op);
          V(mem) = nextOffset;
          folded = true;
        }
      }

      if (folded) {
        converted++;
        mid->erase();
        next->erase();
        op->erase();
        return true;
      }
    }


    if (isa<LdrXOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS(next) == RD(op) && validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<LdrWOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS(next) == RD(op) && validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<LdrFOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS(next) == RD(op) && validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<LdrDOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS(next) == RD(op) && validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<StrXOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS2(next) == RD(op) && RS(next) != RD(op) &&
          validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS2(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<StrWOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS2(next) == RD(op) && RS(next) != RD(op) &&
          validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS2(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<StrFOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS2(next) == RD(op) && RS(next) != RD(op) &&
          validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS2(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    if (isa<StrDOp>(next)) {
      int nextOffset = V(next) + offset;
      if (RS2(next) == RD(op) && RS(next) != RD(op) &&
          validMemOffset(next, nextOffset) && canEraseAddrTmp(op, next)) {
        converted++;
        RS2(next) = RS(op);
        V(next) = nextOffset;
        op->erase();
        return true;
      }
    }
    return false;
  });

  // Eliminate useless MovROp.
  runRewriter(funcOp, [&](MovROp *op) {
    if (RD(op) == RS(op) || RD(op) == Reg::xzr) {
      converted++;
      op->erase();
      return true;
    }
    if (!op->atBack()) {
      auto next = op->nextOp();
      if (isa<MovROp>(next) &&
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

  runRewriter(funcOp, [&](FmovOp *op) {
    if (RD(op) == RS(op)) {
      converted++;
      op->erase();
      return true;
    }
    if (!op->atBack()) {
      auto next = op->nextOp();
      if (isa<FmovOp>(next) &&
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

  runRewriter(funcOp, [&](MovIOp *op) {
    if (RD(op) == Reg::xzr) {
      converted++;
      op->erase();
      return true;
    }
    if (!op->atFront()) {
      auto prev = op->prevOp();
      if (isa<MovIOp>(prev) &&
          prev->has<RdAttr>() && prev->has<IntAttr>() &&
          RD(prev) == RD(op) && V(prev) == V(op)) {
        converted++;
        op->erase();
        return true;
      }
    }
    return false;
  });

  runRewriter(funcOp, [&](MovkOp *op) {
    if (RD(op) == Reg::xzr) {
      converted++;
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter(funcOp, [&](MovnOp *op) {
    if (RD(op) == Reg::xzr) {
      converted++;
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter(funcOp, [&](AddXIOp *op) {
    if (RD(op) == Reg::xzr) {
      converted++;
      op->erase();
      return true;
    }
    return false;
  });

  runRewriter(funcOp, [&](AddWIOp *op) {
    if (RD(op) == Reg::xzr) {
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

  int converted;
  int rounds = 0;
  do {
    converted = latePeephole(funcOp);
    convertedTotal += converted;
    rounds++;
    if (peepholeRounds > 0 && rounds >= peepholeRounds)
      break;
  } while (converted);

  // Replace blocks with only a single `j` as terminator.
  std::map<BasicBlock*, BasicBlock*> jumpTo;
  for (auto bb : region->getBlocks()) {
    if (bb->getOpCount() == 1 && isa<BOp>(bb->getLastOp())) {
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
  // This helps eliminate branch chains left by earlier rewrites.
  do {
    changed = false;
    const auto &bbs = region->getBlocks();
    for (auto bb : bbs) {
      if (bb->getOpCount() == 0)
        continue;
      if (bb->succs.size() != 1)
        continue;

      auto succ = *bb->succs.begin();
      if (!succ || succ == bb)
        continue;
      if (succ->preds.size() != 1)
        continue;

      auto term = bb->getLastOp();
      if (isa<BOp>(term))
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

  // Now branches are still having both TargetAttr and ElseAttr.
  // Replace them (perform split when necessary), so that they only have one target.
  runRewriter(funcOp, [&](BltOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BgtOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BleOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BgeOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BeqOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](BneOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](CbzOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });
  runRewriter(funcOp, [&](CbnzOp *op) {
    if (op->has<ElseAttr>() && TARGET(op) == ELSE(op)) {
      builder.replace<BOp>(op, { new TargetAttr(TARGET(op)) });
      return true;
    }
    return false;
  });

  REPLACE_BRANCH(BltOp, BgeOp, BINARY_BRANCH);
  REPLACE_BRANCH(BleOp, BgtOp, BINARY_BRANCH);
  REPLACE_BRANCH(BeqOp, BneOp, BINARY_BRANCH);
  REPLACE_BRANCH(CbzOp, CbnzOp, UNARY_BRANCH);

  // Also, eliminate useless BOp.
  runRewriter(funcOp, [&](BOp *op) {
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

namespace {

bool fitsAddImm12(int imm) {
  return imm >= -4095 && imm <= 4095;
}

bool fitsMemImm8(int imm) {
  return imm >= 0 && imm <= 32760 && imm % 8 == 0;
}

bool fitsMemImm4(int imm) {
  return imm >= 0 && imm <= 16380 && imm % 4 == 0;
}

bool fitsPairImm8(int imm) {
  return imm >= -512 && imm <= 504 && imm % 8 == 0;
}

void emitSpAdjust(Builder &builder, int delta) {
  while (delta != 0) {
    int step = delta;
    if (step > 4095)
      step = 4095;
    if (step < -4095)
      step = -4095;
    builder.create<AddXIOp>({ RDC(Reg::sp), RSC(Reg::sp), new IntAttr(step) });
    delta -= step;
  }
}

void materializeSpAddr(Builder &builder, Reg tmp, int offset) {
  if (fitsAddImm12(offset)) {
    builder.create<AddXIOp>({ RDC(tmp), RSC(Reg::sp), new IntAttr(offset) });
    return;
  }
  builder.create<MovIOp>({ RDC(tmp), new IntAttr(offset) });
  builder.create<AddXOp>({ RDC(tmp), RSC(Reg::sp), RS2C(tmp) });
}

void emitStackStore64(Builder &builder, Reg rs, bool fp, int offset) {
  if (fitsMemImm8(offset)) {
    if (fp)
      builder.create<StrDOp>({ RSC(rs), RS2C(Reg::sp), new IntAttr(offset) });
    else
      builder.create<StrXOp>({ RSC(rs), RS2C(Reg::sp), new IntAttr(offset) });
    return;
  }

  materializeSpAddr(builder, spillReg2, offset);
  if (fp)
    builder.create<StrDOp>({ RSC(rs), RS2C(spillReg2), new IntAttr(0) });
  else
    builder.create<StrXOp>({ RSC(rs), RS2C(spillReg2), new IntAttr(0) });
}

void emitStackLoad64(Builder &builder, Reg rd, bool fp, int offset) {
  if (fitsMemImm8(offset)) {
    if (fp)
      builder.create<LdrDOp>({ RDC(rd), RSC(Reg::sp), new IntAttr(offset) });
    else
      builder.create<LdrXOp>({ RDC(rd), RSC(Reg::sp), new IntAttr(offset) });
    return;
  }

  materializeSpAddr(builder, spillReg2, offset);
  if (fp)
    builder.create<LdrDOp>({ RDC(rd), RSC(spillReg2), new IntAttr(0) });
  else
    builder.create<LdrXOp>({ RDC(rd), RSC(spillReg2), new IntAttr(0) });
}

void emitStackGetArgLoad(Builder &builder, Reg rd, bool fp, int offset) {
  if ((fp && fitsMemImm4(offset)) || (!fp && fitsMemImm8(offset))) {
    if (fp)
      builder.create<LdrFOp>({ RDC(rd), RSC(Reg::sp), new IntAttr(offset) });
    else
      builder.create<LdrXOp>({ RDC(rd), RSC(Reg::sp), new IntAttr(offset) });
    return;
  }

  materializeSpAddr(builder, spillReg2, offset);
  if (fp)
    builder.create<LdrFOp>({ RDC(rd), RSC(spillReg2), new IntAttr(0) });
  else
    builder.create<LdrXOp>({ RDC(rd), RSC(spillReg2), new IntAttr(0) });
}

}

void save(Builder builder, const std::vector<Reg> &regs, int offset) {
  for (int i = 0; i < int(regs.size()) / 2 * 2; i += 2) {
    Reg r1 = regs[i];
    Reg r2 = regs[i + 1];
    offset -= 16;
    bool fp1 = isFP(r1), fp2 = isFP(r2);
    if (fp1 == fp2 && fitsPairImm8(offset)) {
      if (fp1)
        builder.create<StpDOp>({ RSC(r1), RS2C(r2), RS3C(Reg::sp), new IntAttr(offset) });
      else
        builder.create<StpXOp>({ RSC(r1), RS2C(r2), RS3C(Reg::sp), new IntAttr(offset) });
      continue;
    }
    emitStackStore64(builder, r2, fp2, offset);
    emitStackStore64(builder, r1, fp1, offset + 8);
  }
  if (regs.size() & 1) {
    Reg reg = regs.back();
    offset -= 8;
    emitStackStore64(builder, reg, isFP(reg), offset);
  }
}

void load(Builder builder, const std::vector<Reg> &regs, int offset) {
  for (int i = 0; i < int(regs.size()) / 2 * 2; i += 2) {
    Reg r1 = regs[i];
    Reg r2 = regs[i + 1];
    offset -= 16;
    bool fp1 = isFP(r1), fp2 = isFP(r2);
    if (fp1 == fp2 && fitsPairImm8(offset)) {
      if (fp1)
        builder.create<LdpDOp>({ RSC(r1), RS2C(r2), RS3C(Reg::sp), new IntAttr(offset) });
      else
        builder.create<LdpXOp>({ RSC(r1), RS2C(r2), RS3C(Reg::sp), new IntAttr(offset) });
      continue;
    }
    emitStackLoad64(builder, r2, fp2, offset);
    emitStackLoad64(builder, r1, fp1, offset + 8);
  }
  if (regs.size() & 1) {
    Reg reg = regs.back();
    offset -= 8;
    emitStackLoad64(builder, reg, isFP(reg), offset);
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
    preserve.push_back(Reg::x30);

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
      builder.replace<BOp>(ret, { new TargetAttr(bb) });

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
    builder.setBeforeOp(op);
    bool fp = isFP(RD(op));
    emitStackGetArgLoad(builder, RD(op), fp, myoffset);
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
