#include "ArmPasses.h"
#include "Regs.h"
#include "../backend/shared/RegAllocHotness.h"
#include <algorithm>
#include <unordered_set>

using namespace sys;
using namespace sys::arm;

namespace {

class SpilledRdAttr : public AttrImpl<SpilledRdAttr, ARMLINE + 4194304> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRdAttr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rd = " + std::to_string(offset) + (fp ? "f" : "") + ">"; }
  SpilledRdAttr *clone() override { return new SpilledRdAttr(fp, offset, ref); }
};

class SpilledRsAttr : public AttrImpl<SpilledRsAttr, ARMLINE + 4194304> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRsAttr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rs = " + std::to_string(offset) + (fp ? "f" : "") + ">"; }
  SpilledRsAttr *clone() override { return new SpilledRsAttr(fp, offset, ref); }
};

class SpilledRs2Attr : public AttrImpl<SpilledRs2Attr, ARMLINE + 4194304> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRs2Attr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rs2 = " + std::to_string(offset) + + (fp ? "f" : "") + ">"; }
  SpilledRs2Attr *clone() override { return new SpilledRs2Attr(fp, offset, ref); }
};

class SpilledRs3Attr : public AttrImpl<SpilledRs3Attr, ARMLINE + 4194304> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRs3Attr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rs3 = " + std::to_string(offset) + + (fp ? "f" : "") + ">"; }
  SpilledRs3Attr *clone() override { return new SpilledRs3Attr(fp, offset, ref); }
};

bool fpreg(Value::Type ty) {
  return ty == Value::f32 || ty == Value::f128;
}

bool fitsAddImm12(int imm) {
  return imm >= -4095 && imm <= 4095;
}

bool fitsMemImm32(int imm) {
  return imm >= 0 && imm <= 16380 && imm % 4 == 0;
}

bool fitsMemImm64(int imm) {
  return imm >= 0 && imm <= 32760 && imm % 8 == 0;
}

void materializeSpAddr(Builder &builder, Reg tmp, int offset) {
  if (fitsAddImm12(offset)) {
    builder.create<AddXIOp>({ RDC(tmp), RSC(Reg::sp), new IntAttr(offset) });
    return;
  }
  builder.create<MovIOp>({ RDC(tmp), new IntAttr(offset) });
  builder.create<AddXOp>({ RDC(tmp), RSC(Reg::sp), RS2C(tmp) });
}

void emitStackStore(Builder &builder, Reg src, bool fp, int offset, Reg addrTmp) {
  if ((fp && fitsMemImm32(offset)) || (!fp && fitsMemImm64(offset))) {
    if (fp)
      builder.create<StrFOp>({ RSC(src), RS2C(Reg::sp), new IntAttr(offset) });
    else
      builder.create<StrXOp>({ RSC(src), RS2C(Reg::sp), new IntAttr(offset) });
    return;
  }

  materializeSpAddr(builder, addrTmp, offset);
  if (fp)
    builder.create<StrFOp>({ RSC(src), RS2C(addrTmp), new IntAttr(0) });
  else
    builder.create<StrXOp>({ RSC(src), RS2C(addrTmp), new IntAttr(0) });
}

void emitStackLoad(Builder &builder, Reg dst, bool fp, int offset, Reg addrTmp) {
  if ((fp && fitsMemImm32(offset)) || (!fp && fitsMemImm64(offset))) {
    if (fp)
      builder.create<LdrFOp>({ RDC(dst), RSC(Reg::sp), new IntAttr(offset) });
    else
      builder.create<LdrXOp>({ RDC(dst), RSC(Reg::sp), new IntAttr(offset) });
    return;
  }

  materializeSpAddr(builder, addrTmp, offset);
  if (fp)
    builder.create<LdrFOp>({ RDC(dst), RSC(addrTmp), new IntAttr(0) });
  else
    builder.create<LdrXOp>({ RDC(dst), RSC(addrTmp), new IntAttr(0) });
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

struct ArgPlacement {
  bool fp;
  int regIndex;
  int stackIndex;
};

ArgPlacement getArgPlacement(const std::vector<Value::Type> &types, int index) {
  int intCount = 0;
  int fpCount = 0;
  int stackCount = 0;

  for (int i = 0; i <= index; i++) {
    bool fp = types[i] == Value::f32;
    int &regCount = fp ? fpCount : intCount;
    if (regCount < 8) {
      if (i == index)
        return { fp, regCount, -1 };
      regCount++;
      continue;
    }

    if (i == index)
      return { fp, -1, stackCount };
    stackCount++;
  }

  return { false, -1, -1 };
}

}

std::map<std::string, int> RegAlloc::stats() {
  return {
    { "spilled", spilled },
    { "peepholed", convertedTotal },
  };
}

// This doesn't invalidate the `Op*` itself, which is crucial.
#define LOWER(Ty, Body) \
  runRewriter(funcOp, [&](Ty *op) { \
    if (op->getOperands().size() == 0) \
      return false; \
    Body \
    op->removeAllOperands(); \
    return true; \
  });

// If assignment does not contain the value, then it doesn't clash with any other one.
// In that case just give it a random register.
#define ADD_ATTR(Index, AttrTy) \
  auto v##Index = op->getOperand(Index).defining; \
  if (!spillOffset.count(v##Index)) \
    op->add<AttrTy>(getReg(v##Index)); \
  else \
    op->add<Spilled##AttrTy> GET_SPILLED_ARGS(v##Index);

#define GET_SPILLED_ARGS(op) \
  (fpreg(op->getResultType()), spillOffset[op], op)

#define NULLARY
#define UNARY ADD_ATTR(0, RsAttr)
#define BINARY UNARY ADD_ATTR(1, Rs2Attr)
#define TERNARY BINARY ADD_ATTR(2, Rs3Attr)

#define CREATE_MV(fp, rd, rs) \
  if (!fp) \
    builder.create<MovROp>({ RDC(rd), RSC(rs) }); \
  else \
    builder.create<FmovOp>({ RDC(rd), RSC(rs) });

// Used in constructing interference graph.
namespace {

struct Event {
  int timestamp;
  bool start;
  Op *op;
};

}

// Defined in rv/RegAlloc.cpp.
void dumpInterf(Region *region, const std::unordered_map<Op*, std::unordered_set<Op*>> &interf);

void RegAlloc::runImpl(Region *region, bool isLeaf) {
  const Reg *order = isLeaf ? leafOrder : normalOrder;
  const Reg *orderf = isLeaf ? leafOrderf : normalOrderf;
  const int regcount = isLeaf ? leafRegCnt : normalRegCnt;
  const int regcountf = isLeaf ? leafRegCntf : normalRegCntf;

  int opCount = 0;
  for (auto bb : region->getBlocks()) { opCount += bb->getOps().size(); }
  bool localFastMode = fastMode || (opCount > 3000);

  Builder builder;
  
  std::unordered_map<Op*, Reg> assignment;

  auto funcOp = region->getParent();

  // First of all, add 35 precolored placeholders before each call.
  // This denotes that a call clobbers those registers.
  // Note clones and joins will also be emitted as calls.
  std::vector<Op*> clobbering = funcOp->findAll<BlOp>();
  auto clones = funcOp->findAll<CloneOp>();
  auto joins = funcOp->findAll<JoinOp>();
  for (auto cl : clones)
    clobbering.push_back(cl);
  for (auto j : joins)
    clobbering.push_back(j);

  for (auto op : clobbering) {
    std::vector<Op*> writes;
    for (auto runner = op->prevOp(); runner && isa<WriteRegOp>(runner); runner = runner->prevOp())
      writes.push_back(runner);

    // `writes` are in backward order.
    // We need to insert a placeholder for everything after this writereg in the vector.
    for (int i = 0; i < int(writes.size()) - 1; i++) {
      builder.setBeforeOp(writes[i]);
      for (int j = i + 1; j < writes.size(); j++) {
        auto reg = REG(writes[j]);
        auto placeholder = builder.create<PlaceHolderOp>();
        assignment[placeholder] = reg;
        if (isFP(reg))
          placeholder->setResultType(Value::f32);
      }
    }
    
    builder.setBeforeOp(op);
    for (auto reg : callerSaved) {
      auto placeholder = builder.create<PlaceHolderOp>();
      assignment[placeholder] = reg;
      // Make floating point respect the placeholders.
      if (isFP(reg))
        placeholder->setResultType(Value::f32);
    }
  };

  // Similarly, add placeholders around each GetArg.
  // First create placeholders for a0-a7.
  builder.setToRegionStart(region);
  std::vector<Value> argHolders, fargHolders;
  auto argcnt = funcOp->get<ArgCountAttr>()->count;
  auto argTypes = getArgTypes(cast<FuncOp>(funcOp));
  for (int i = 0; i < 8; i++) {
    auto placeholder = builder.create<PlaceHolderOp>();
    assignment[placeholder] = argRegs[i];
    argHolders.push_back(placeholder);

    auto fplaceholder = builder.create<PlaceHolderOp>();
    assignment[fplaceholder] = fargRegs[i];
    fargHolders.push_back(fplaceholder);
  }

  auto rawGets = funcOp->findAll<GetArgOp>();
  for (auto it = rawGets.begin(); it != rawGets.end();) {
    if ((*it)->getUses().empty()) {
      (*it)->erase();
      it = rawGets.erase(it);
      continue;
    }
    ++it;
  }
  // We might find some getArgs missing by DCE, so it's not necessarily consecutive.
  std::vector<Op*> getArgs;
  getArgs.resize(argcnt);
  for (auto x : rawGets) {
    if (V(x) >= 0 && V(x) < argcnt)
      getArgs[V(x)] = x;
  }

  auto entry = region->getFirstBlock();
  for (size_t i = 0; i < getArgs.size(); i++) {
    // A missing argument.
    if (!getArgs[i])
      continue;

    Op *op = getArgs[i];
    auto ty = op->getResultType();
    auto placement = getArgPlacement(argTypes, (int) i);

    if (fpreg(ty) && placement.regIndex >= 0) {
      op->moveToStart(entry);
      builder.setBeforeOp(op);
      builder.create<PlaceHolderOp>({ fargHolders[placement.regIndex] });
      builder.replace<ReadRegOp>(op, Value::f32, { new RegAttr(fargRegs[placement.regIndex]) });
      continue;
    }
    if (!fpreg(ty) && placement.regIndex >= 0) {
      op->moveToStart(entry);
      builder.setBeforeOp(op);
      builder.create<PlaceHolderOp>({ argHolders[placement.regIndex] });
      builder.replace<ReadRegOp>(op, Value::i64, { new RegAttr(argRegs[placement.regIndex]) });
      continue;
    }
    // Spilled to stack; don't do anything.
  }

  // Don't forget to update the types of ReadRegOp.
  runRewriter(funcOp, [&](ReadRegOp *op) {
    if (isFP(REG(op)))
      op->setResultType(Value::f32);
    return false;
  });

  region->updateLiveness();

  // Build a coarse block hotness model for spill heuristics:
  // back-edge blocks are likely loop bodies; call-like blocks are hotter.
  auto bbWeight = sys::backend::shared::computeBlockHotness(region, [](Op *op) {
    return isa<BlOp>(op) || isa<CloneOp>(op) || isa<JoinOp>(op);
  });
  const int callPenaltyWeight = std::max(64, callPenalty);
  const int crossCallRiskThreshold = loopHotBoost > 1 ? 0 : 1;
  const bool limitPrefer = preferBudget >= 0;
  int preferCount = 0;

  std::unordered_map<Op*, int> spillOffset;
  int currentOffset = STACKOFF(funcOp);
  int highest = 0;
  if (!localFastMode) {
    // Interference graph.
    std::unordered_map<Op*, std::unordered_set<Op*>> interf, spillInterf;
    std::unordered_map<Op*, long long> spillWeight;
    std::unordered_map<Op*, int> callSpan;

    // Values of readreg, or operands of writereg, or phis (mvs), are prioritzed.
    std::unordered_map<Op*, int> priority;
    // The `key` is preferred to have the same value as `value`.
    std::unordered_map<Op*, Op*> prefer;
    // Maps a phi to its operands.
    std::unordered_map<Op*, std::vector<Op*>> phiOperand;

  int currentPriority = 2;
  for (auto bb : region->getBlocks()) {
    int localWeight = bbWeight[bb];
    if (localWeight > 1 && loopHotBoost > 1)
      localWeight *= loopHotBoost;
    auto bumpPriority = [&](Op *x, int v) {
      auto it = priority.find(x);
      if (it == priority.end() || it->second < v)
        priority[x] = v;
    };
    // Scan through the block and see the place where the value's last used.
    std::unordered_map<Op*, int> lastUsed, defined;
    const auto &ops = bb->getOps();
    auto it = ops.end();
    for (int i = (int) ops.size() - 1; i >= 0; i--) {
      auto op = *--it;
      spillWeight[op] += localWeight;
      for (auto v : op->getOperands()) {
        if (!lastUsed.count(v.defining))
          lastUsed[v.defining] = i;
        // Read pressure matters more than pure definition count.
        spillWeight[v.defining] += 2LL * localWeight;
      }
      defined[op] = i;

      // Even though the op is not used, it still lives in the instruction that defines it.
      // Actually this should be eliminated with DCE, but we need to take care of it.
      if (!lastUsed.count(op))
        lastUsed[op] = i + 1;

      // Precolor.
      if (isa<WriteRegOp>(op)) {
        assignment[op] = REG(op);
        priority[op] = 1;
      }
      if (isa<ReadRegOp>(op))
        priority[op] = 1;
      
      if (isa<MovIOp>(op) && (V(op) <= 32767 && V(op) >= -32768))
        priority[op] = -2;

      if (isa<PhiOp>(op)) {
        priority[op] = currentPriority + 1;
        for (auto x : op->getOperands()) {
          priority[x.defining] = currentPriority;
          if (!localFastMode && (!limitPrefer || preferCount < preferBudget)) {
            prefer[x.defining] = op;
            preferCount++;
          }
          phiOperand[op].push_back(x.defining);
        }
        currentPriority += 2;
      }

      // Preserve copy chains to reduce move pressure after allocation.
      if (!localFastMode && (isa<MovROp>(op) || isa<FmovOp>(op)) && op->getOperandCount() == 1) {
        auto src = op->DEF(0);
        if (!limitPrefer || preferCount < preferBudget) {
          prefer[op] = src;
          preferCount++;
        }
        bumpPriority(op, currentPriority + 2);
        bumpPriority(src, currentPriority + 1);
      }
    }

    // For all liveOuts, they are last-used at place size().
    // If they aren't defined in this block, then `defined[op]` will be zero, which is intended.
    for (auto op : bb->getLiveOut())
      lastUsed[op] = ops.size();

    // Penalize long-lived values and values spanning call-like operations.
    std::vector<int> callPrefix;
    if (!localFastMode) {
      callPrefix.assign(ops.size() + 1, 0);
      int callIdx = 0;
      for (auto liveOp : ops) {
        bool isCallLike = isa<BlOp>(liveOp) || isa<CloneOp>(liveOp) || isa<JoinOp>(liveOp);
        callPrefix[callIdx + 1] = callPrefix[callIdx] + (isCallLike ? 1 : 0);
        callIdx++;
      }
    }
    for (auto [op, last] : lastUsed) {
      int def = defined[op];
      if (def >= last)
        continue;
      int span = last - def;
      spillWeight[op] += 3LL * localWeight * span;

      if (!localFastMode) {
        int l = std::max(0, def + 1);
        int r = std::min<int>(ops.size(), last);
        int callCount = callPrefix[r] - callPrefix[l];
        callSpan[op] = std::max(callSpan[op], callCount);
        spillWeight[op] += (long long) callPenaltyWeight * localWeight * callCount;
      }
    }

    // We use event-driven approach to optimize it into O(n log n + E).
    std::vector<Event> events;
    for (auto [op, v] : lastUsed) {
      // Don't push empty live range. It's not handled properly.
      if (defined[op] == v)
        continue;
      
      events.push_back(Event { defined[op], true, op });
      events.push_back(Event { v, false, op });
    }

    // Sort with ascending time (i.e. instruction count).
    std::sort(events.begin(), events.end(), [](Event a, Event b) {
      // For the same timestamp, we first set END events as inactive, then deal with START events.
      return a.timestamp == b.timestamp ? (!a.start && b.start) : a.timestamp < b.timestamp;
    });

    std::unordered_set<Op*> active;
    for (const auto& event : events) {
      auto op = event.op;
      // Jumps will never interfere.
      if (isa<BOp>(op))
        continue;

      if (event.start) {
        for (Op* activeOp : active) {
          // FP and int are using different registers.
          // However, they use the same stack frame.
          if (fpreg(activeOp->getResultType()) ^ fpreg(op->getResultType())) {
            spillInterf[op].insert(activeOp);
            spillInterf[activeOp].insert(op);
            continue;
          }

          interf[op].insert(activeOp);
          interf[activeOp].insert(op);
        }
        active.insert(op);
      } else
        active.erase(op);
    }
  }

    std::vector<Op*> ops;
    std::unordered_set<Op*> visitedOps;
    for (auto [k, v] : interf)
      if (!visitedOps.count(k)) {
        ops.push_back(k);
        visitedOps.insert(k);
      }
    // Even though registers in `priority` might not be colliding,
    // we still allocate them here to respect their preference.
    for (auto [k, v] : priority)
      if (!visitedOps.count(k)) {
        ops.push_back(k);
        visitedOps.insert(k);
      }

    // Sort by **descending** degree.
    std::sort(ops.begin(), ops.end(), [&](Op *a, Op *b) {
      auto pa = priority[a];
      auto pb = priority[b];
      if (pa != pb)
        return pa > pb;
      if (spillWeight[a] != spillWeight[b])
        return spillWeight[a] > spillWeight[b];
      if (interf[a].size() != interf[b].size())
        return interf[a].size() > interf[b].size();
      return a < b;
    });

    auto isAllocatable = [&](Reg reg, bool fp) {
      auto rcnt = fp ? regcountf : regcount;
      auto rorder = fp ? orderf : order;
      for (int i = 0; i < rcnt; i++) {
        if (rorder[i] == reg)
          return true;
      }
      return false;
    };

    for (auto op : ops) {
    // Do not allocate colored instructions.
    if (assignment.count(op))
      continue;

    std::unordered_set<Reg> bad, unpreferred;

    for (auto v : interf[op]) {
      // In the whole function, `sp` and `zero` are read-only.
      if (assignment.count(v) && assignment[v] != Reg::sp && assignment[v] != Reg::xzr)
        bad.insert(assignment[v]);
    }

    if (isa<PhiOp>(op)) {
      // Dislike everything that might interfere with phi's operands.
      const auto &operands = phiOperand[op];
      for (auto x : operands) {
        for (auto v : interf[x]) {
          if (assignment.count(v) && assignment[v] != Reg::sp && assignment[v] != Reg::xzr)
            unpreferred.insert(assignment[v]);
        }
      }
    }

    if (prefer.count(op)) {
      auto ref = prefer[op];
      // Try to allocate the same register as `ref`.
      if (assignment.count(ref) && !bad.count(assignment[ref])) {
        Reg preferredReg = assignment[ref];
        if (!isAllocatable(preferredReg, fpreg(op->getResultType())))
          goto skip_prefer_assign;
        bool crossCallRisk =
          (callSpan[op] > crossCallRiskThreshold || callSpan[ref] > crossCallRiskThreshold) &&
          callerSaved.count(preferredReg);
        if (!crossCallRisk) {
          assignment[op] = preferredReg;
          continue;
        }
      }
    }
skip_prefer_assign:

    // See if there's any preferred registers.
    int preferred = -1;
    for (auto use : op->getUses()) {
      if (isa<WriteRegOp>(use)) {
        auto reg = REG(use);
        if (!bad.count(reg) && isAllocatable(reg, fpreg(op->getResultType()))) {
          preferred = (int) reg;
          break;
        }
      }
    }
    if (isa<ReadRegOp>(op)) {
      auto reg = REG(op);
      if (!bad.count(reg) && isAllocatable(reg, fpreg(op->getResultType())))
        preferred = (int) reg;
    }

    if (preferred != -1) {
      assignment[op] = (Reg) preferred;
      continue;
    }

    auto rcnt = !fpreg(op->getResultType()) ? regcount : regcountf;
    auto rorder = !fpreg(op->getResultType()) ? order : orderf;
    auto tryAssign = [&](bool onlyCallee, bool allowUnpreferred) -> bool {
      for (int i = 0; i < rcnt; i++) {
        Reg cand = rorder[i];
        if (onlyCallee && !calleeSaved.count(cand))
          continue;
        if (bad.count(cand))
          continue;
        if (!allowUnpreferred && unpreferred.count(cand))
          continue;
        assignment[op] = cand;
        return true;
      }
      return false;
    };

    // Values spanning call-like ops tend to thrash if placed in caller-saved regs.
    bool preferCallee = !isLeaf && callSpan[op] > 0;
    if (!(preferCallee && tryAssign(/*onlyCallee=*/ true, /*allowUnpreferred=*/ false)))
      tryAssign(/*onlyCallee=*/ false, /*allowUnpreferred=*/ false);

    // We have excluded too much. Try again with weaker constraints.
    if (!assignment.count(op) && unpreferred.size()) {
      if (!(preferCallee && tryAssign(/*onlyCallee=*/ true, /*allowUnpreferred=*/ true)))
        tryAssign(/*onlyCallee=*/ false, /*allowUnpreferred=*/ true);
    }

    if (assignment.count(op))
      continue;

    spilled++;
    // Spilled. Try to see all spill offsets of conflicting ops.
    int desired = currentOffset;
    std::unordered_set<int> conflict;
    for (auto v : interf[op]) {
      if (!spillOffset.count(v))
        continue;

      conflict.insert(spillOffset[v]);
    }
    for (auto v : spillInterf[op]) {
      if (!spillOffset.count(v))
        continue;

      conflict.insert(spillOffset[v]);
    }

    // Try find a space.
    while (conflict.count(desired))
      desired += 8;

    spillOffset[op] = desired;

    // Update `highest`, which will indicate the size allocated.
      if (desired > highest)
        highest = desired;
    }
  } else {
    // Huge-module fast path: avoid O(N^2) interference construction in regalloc.
    // Conservatively spill all allocatable SSA values to keep correctness.
    int nextOffset = currentOffset;
    for (auto bb : region->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (!op || assignment.count(op) || spillOffset.count(op))
          continue;
        if (op->getResultType() == Value::unit)
          continue;
        spillOffset[op] = nextOffset;
        highest = nextOffset;
        nextOffset += 8;
        spilled++;
      }
    }
  }

  // Keep spilled values in stack slots consistently.
  // The previous "1/2-spill shortcut" bound spilled values to dedicated regs
  // (x28/x29, v31/v30), which can alias with temporary address regs in later
  // lowering and break liveness assumptions in some loops.
  
  // If possible, map some offsets to floating-point registers.
  if (spillOffset.size()) {
    // Try to reuse floating-point registers for spilling.
    std::unordered_set<Reg> used;
    for (auto [op, x] : assignment) {
      if (isa<PlaceHolderOp>(op))
        continue;
      used.insert(x);
    }

    std::unordered_map<int, Reg> fpmv;
    auto off = STACKOFF(funcOp);
    for (auto reg : leafOrderf) {
      if (highest <= off)
        break;
      if (used.count(reg) || (!isLeaf && !calleeSaved.count(reg)))
        continue;

      fpmv[highest] = reg;
      highest -= 8;
    }

    for (auto &[_, offset] : spillOffset) {
      if (fpmv.count(offset))
        offset = -int(fpmv[offset]);
    }
  }

  // Allocate more stack space for it.
  if (spillOffset.size())
    STACKOFF(funcOp) = highest + 8;

  const auto getReg = [&](Op *op) {
    return assignment.count(op) ? assignment[op] :
      fpreg(op->getResultType()) ? orderf[0] : order[0];
  };

  // Convert all operands to registers.
  LOWER(MsubWOp, TERNARY);
  LOWER(MsubXOp, TERNARY);
  LOWER(MaddWOp, TERNARY);
  LOWER(MaddXOp, TERNARY);
  LOWER(FmaddOp, TERNARY);
  LOWER(FmsubOp, TERNARY);
  LOWER(CselNeZOp, TERNARY);
  LOWER(CselLtZOp, TERNARY);
  LOWER(StrWROp, TERNARY);
  LOWER(StrXROp, TERNARY);
  LOWER(StrFROp, TERNARY);

  LOWER(AddXOp, BINARY);
  LOWER(AddWOp, BINARY);
  LOWER(AddVOp, BINARY);
  LOWER(AddFVOp, BINARY);
  LOWER(SubWOp, BINARY);
  LOWER(SubXOp, BINARY);
  LOWER(SubVOp, BINARY);
  LOWER(SubFVOp, BINARY);
  LOWER(MulWOp, BINARY);
  LOWER(MulXOp, BINARY);
  LOWER(MulVOp, BINARY);
  LOWER(MulFVOp, BINARY);
  LOWER(SdivWOp, BINARY);
  LOWER(SdivXOp, BINARY);
  LOWER(AndOp, BINARY);
  LOWER(OrOp, BINARY);
  LOWER(EorOp, BINARY);
  LOWER(StrWOp, BINARY);
  LOWER(StrXOp, BINARY);
  LOWER(StrFOp, BINARY);
  LOWER(StrWPOp, BINARY);
  LOWER(StrXPOp, BINARY);
  LOWER(StrFPOp, BINARY);
  LOWER(St1Op, BINARY);
  LOWER(LslWOp, BINARY);
  LOWER(LsrWOp, BINARY);
  LOWER(LslXOp, BINARY);
  LOWER(LsrXOp, BINARY);
  LOWER(AsrWOp, BINARY);
  LOWER(AsrXOp, BINARY);
  LOWER(FaddOp, BINARY);
  LOWER(FsubOp, BINARY);
  LOWER(FmulOp, BINARY);
  LOWER(FdivOp, BINARY);
  LOWER(AddWLOp, BINARY);
  LOWER(AddXLOp, BINARY);
  LOWER(AddWROp, BINARY);
  LOWER(AddWAROp, BINARY);
  LOWER(AddXROp, BINARY);
  LOWER(LdrWROp, BINARY);
  LOWER(LdrXROp, BINARY);
  LOWER(LdrFROp, BINARY);
  LOWER(SmullOp, BINARY);
  
  LOWER(BltOp, BINARY);
  LOWER(BleOp, BINARY);
  LOWER(BeqOp, BINARY);
  LOWER(BneOp, BINARY);
  LOWER(BgtOp, BINARY);
  LOWER(BgeOp, BINARY);
  LOWER(CsetLeOp, BINARY);
  LOWER(CsetLtOp, BINARY);
  LOWER(CsetNeOp, BINARY);
  LOWER(CsetEqOp, BINARY);
  LOWER(CsetLeFOp, BINARY);
  LOWER(CsetLtFOp, BINARY);
  LOWER(CsetNeFOp, BINARY);
  LOWER(CsetEqFOp, BINARY);
  LOWER(CsetEqTstOp, BINARY);
  LOWER(CsetNeTstOp, BINARY);
  LOWER(CnegLtZOp, BINARY);
  
  LOWER(LdrWOp, UNARY);
  LOWER(LdrXOp, UNARY);
  LOWER(LdrFOp, UNARY);
  LOWER(LdrWPOp, UNARY);
  LOWER(LdrXPOp, UNARY);
  LOWER(LdrFPOp, UNARY);
  LOWER(Ld1Op, UNARY);
  LOWER(AddXIOp, UNARY);
  LOWER(AddWIOp, UNARY);
  LOWER(NegOp, UNARY);
  LOWER(AndIOp, UNARY);
  LOWER(OrIOp, UNARY);
  LOWER(EorIOp, UNARY);
  LOWER(LslWIOp, UNARY);
  LOWER(LsrWIOp, UNARY);
  LOWER(LslXIOp, UNARY);
  LOWER(LsrXIOp, UNARY);
  LOWER(AsrWIOp, UNARY);
  LOWER(AsrXIOp, UNARY);
  LOWER(CbzOp, UNARY);
  LOWER(CbnzOp, UNARY);
  LOWER(ScvtfOp, UNARY);
  LOWER(FcvtzsOp, UNARY);
  LOWER(FmovWOp, UNARY);
  LOWER(NegOp, UNARY);
  LOWER(FnegOp, UNARY);
  LOWER(CsetEqFcmpZOp, UNARY);
  LOWER(CsetNeFcmpZOp, UNARY);
  LOWER(DupOp, UNARY);
  LOWER(DupFOp, UNARY);

  // Note that some ops are dealt with later.
  // We can't remove all operands here.
  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<BlOp>(op) || isa<RetOp>(op))
        op->removeAllOperands();

      if (isa<PlaceHolderOp>(op)) {
        // All users of it are phis.
        for (auto use : op->getUses()) {
          assert(isa<PhiOp>(use));
          for (int i = 0; i < use->getOperandCount(); i++) {
            if (use->DEF(i) == op) {
              use->removeOperand(i);
              use->removeAttribute(i);
              break;
            }
          }
        }
        op->removeAllOperands();
      }
    }
  }

  // Remove placeholders inserted previously.
  auto holders = funcOp->findAll<PlaceHolderOp>();
  for (auto holder : holders)
    holder->erase();

  //   writereg %1, <reg = a0>
  // becomes
  //   mv a0, assignment[%1]
  // As RdAttr is supplied, though `assignment[]` won't have the new op recorded, it's fine.
  runRewriter(funcOp, [&](WriteRegOp *op) {
    builder.setBeforeOp(op);
    CREATE_MV(isFP(REG(op)), REG(op), getReg(op->DEF(0)));
    auto mv = op->prevOp();

    if (spillOffset.count(op->DEF(0))) {
      mv->remove<RsAttr>();
      mv->add<SpilledRsAttr> GET_SPILLED_ARGS(op->DEF(0));
    }

    op->erase();
    return false;
  });

  //   readreg %1, <reg = a0>
  // becomes
  //   mv assignment[%1], a0
  runRewriter(funcOp, [&](ReadRegOp *op) {
    builder.setBeforeOp(op);
    CREATE_MV(isFP(REG(op)), getReg(op), REG(op));
    auto mv = op->prevOp();
    assignment[mv] = getReg(op);

    if (spillOffset.count(op)) {
      mv->remove<RdAttr>();
      mv->add<SpilledRdAttr> GET_SPILLED_ARGS(op);
      spillOffset[mv] = spillOffset[op];
    }

    // We can't directly erase it because it might get used by phi's later.
    op->replaceAllUsesWith(mv);
    op->erase();
    return false;
  });

  // Finally, after everything has been erased:
  // Destruct phi.

  // This contains all phis to be removed.
  std::vector<Op*> allPhis;
  auto bbs = region->getBlocks();

  // Split critical edges.
  for (auto bb : bbs) {
    if (bb->succs.size() <= 1)
      continue;
    if (bb->getOpCount() == 0)
      continue;

    // Note that we need to split even if there's no phi in one of the blocks.
    // This is because the registers of branch operation can be clobbered if that's not done.
    // Consider:
    //   b %1, <bb1>, <bb2>
    // bb1:
    //   %3 = phi ...
    // It is entirely possible for %3 to have the same register as %1.
    
    auto edge1 = region->insertAfter(bb);
    auto edge2 = region->insertAfter(bb);
    auto bbTerm = bb->getLastOp();
    if (!bbTerm)
      continue;

    // Create edge for target branch.
    auto target = bbTerm->get<TargetAttr>();
    auto oldTarget = target->bb;
    target->bb = edge1;

    builder.setToBlockEnd(edge1);
    builder.create<BOp>({ new TargetAttr(oldTarget) });

    // Create edge for else branch.
    auto ifnot = bbTerm->get<ElseAttr>();
    auto oldElse = ifnot->bb;
    ifnot->bb = edge2;

    builder.setToBlockEnd(edge2);
    builder.create<BOp>({ new TargetAttr(oldElse) });

    // Rename the blocks of the phis.
    for (auto succ : bb->succs) {
      for (auto phis : succ->getPhis()) {
        for (auto attr : phis->getAttrs()) {
          auto from = cast<FromAttr>(attr);
          if (from->bb != bb)
            continue;
          if (succ == oldTarget)
            from->bb = edge1;
          if (succ == oldElse)
            from->bb = edge2;
        }
      }
    }
  }

#define SOFFSET(op, Ty) ((Reg) (-(op)->get<Spilled##Ty##Attr>()->offset-1000))
#define SPILLABLE(op, Ty) (op->has<Ty##Attr>() ? op->get<Ty##Attr>()->reg : SOFFSET(op, Ty))

  // Detect circular copies and calculate a correct order.
  std::unordered_map<BasicBlock*, std::vector<std::pair<Reg, Reg>>> moveMap;
  std::unordered_map<BasicBlock*, std::map<std::pair<Reg, Reg>, Op*>> revMap;
  for (auto bb : bbs) {
    auto phis = bb->getPhis();

    std::vector<Op*> moves;
    for (auto phi : phis) {
      auto &ops = phi->getOperands();
      auto &attrs = phi->getAttrs();
      for (size_t i = 0; i < ops.size(); i++) {
        auto bb = FROM(attrs[i]);
        if (!bb || bb->getParent() != region || bb->getOpCount() == 0)
          continue;
        auto term = bb->getLastOp();
        if (!term)
          continue;
        builder.setBeforeOp(term);
        auto def = ops[i].defining;
        Op *mv;
        if (fpreg(phi->getResultType())) {
          mv = builder.create<FmovOp>({
            new ImpureAttr,
            spillOffset.count(phi) ? (Attr*) new SpilledRdAttr GET_SPILLED_ARGS(phi) : RDC(getReg(phi)),
            spillOffset.count(def) ? (Attr*) new SpilledRsAttr GET_SPILLED_ARGS(def) : RSC(getReg(def))
          });
        } else {
          mv = builder.create<MovROp>({
            new ImpureAttr,
            spillOffset.count(phi) ? (Attr*) new SpilledRdAttr GET_SPILLED_ARGS(phi) : RDC(getReg(phi)),
            spillOffset.count(def) ? (Attr*) new SpilledRsAttr GET_SPILLED_ARGS(def) : RSC(getReg(def))
          });
        }
        moves.push_back(mv);
      }
    }

    std::copy(phis.begin(), phis.end(), std::back_inserter(allPhis));

    for (auto mv : moves) {
      auto dst = SPILLABLE(mv, Rd);
      auto src = SPILLABLE(mv, Rs);
      if (src == dst) {
        mv->erase();
        continue;
      }

      auto parent = mv->getParent();
      moveMap[parent].emplace_back(dst, src);
      revMap[parent][{ dst, src }] = mv;
    }
  }

  for (const auto &[bb, mvs] : moveMap) {
    std::unordered_map<Reg, Reg> moveGraph;
    for (auto [dst, src] : mvs)
      moveGraph[dst] = src;

    // Detect cycles.
    std::set<Reg> visited, visiting;
    std::vector<std::pair<Reg, Reg>> sorted;
    // Cycle headers.
    std::vector<Reg> headers;
    // All members in the cycle under a certain header.
    std::unordered_map<Reg, std::vector<Reg>> members;
    // All nodes that are in a certain cycle.
    std::unordered_set<Reg> inCycle;

    // Do a topological sort; it will decide whether there's a cycle.
    std::function<void(Reg)> dfs = [&](Reg node) {
      visiting.insert(node);
      Reg src = moveGraph[node];

      if (visiting.count(src))
        // A node is visited twice. Here's a cycle.
        headers.push_back(node);
      else if (!visited.count(src) && moveGraph.count(src))
        dfs(src);
    
      visiting.erase(node);
      visited.insert(node);
      sorted.emplace_back(node, src);
    };

    for (auto [dst, src] : mvs) {
      if (!visited.count(dst))
        dfs(dst);
    }

    std::reverse(sorted.begin(), sorted.end());

    // Fill in record of cycles.
    for (auto header : headers) {
      Reg runner = header;
      do {
        members[header].push_back(runner);
        runner = moveGraph[runner];
      } while (runner != header);

      for (auto member : members[header])
        inCycle.insert(member);
    }

    // Move sorted phis so that they're in the correct order.
    if (bb->getOpCount() == 0)
      continue;
    Op* term = bb->getLastOp();
    if (!term)
      continue;

    std::unordered_set<Reg> emitted;

    for (auto [dst, src] : sorted) {
      if (dst == src || emitted.count(dst) || inCycle.count(dst))
        continue;

      revMap[bb][{ dst, src }]->moveBefore(term);
      emitted.insert(dst);
    }

    if (members.empty())
      continue;

    // Now emit all cycles.
    for (auto header : headers) {
      assert(!members[header].empty());

      // Move the header's value to temp.
      Reg headerSrc = moveGraph[header];
      auto mv = revMap[bb][{ header, headerSrc }];
      bool fp = isFP(header);
      Reg tmp = fp ? fspillReg2 : spillReg2;
      RD(mv) = tmp;
      mv->moveBefore(term);

      // For the rest of the cycle, perform the moves in order.
      Reg curr = headerSrc;
      while (curr != header) {
        Reg nextSrc = moveGraph[curr];
        revMap[bb][{ curr, nextSrc }]->moveBefore(term);
        curr = nextSrc;
      }

      // Move from temp into the header.
      builder.setBeforeOp(term);
      CREATE_MV(fp, header, tmp);
    }
  }

  // Erase all phi's properly. There might be cross-reference across blocks.
  // So we need to remove all operands first.
  for (auto phi : allPhis)
    phi->removeAllOperands();

  for (auto phi : allPhis) {
    phi->erase();
  }

  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (hasRd(op) && !op->has<RdAttr>() && !op->has<SpilledRdAttr>()) {
        if (!spillOffset.count(op))
          op->add<RdAttr>(getReg(op));
        else
          op->add<SpilledRdAttr> GET_SPILLED_ARGS(op);
      }
    }
  }

  // Deal with spilled variables.
  std::vector<Op*> remove;
  for (auto bb : region->getBlocks()) {
    int delta = 0;
    for (auto op : bb->getOps()) {
      // We might encounter spilling around calls.
      // For example:
      //   addi sp, sp, -192    ; setting up 24 extra arguments
      //   mv a0, ...
      //   ld s11, OFFSET(sp)   ; !!! ADJUST HERE
      //
      // That's why we need an extra "delta".
      // No need for dominance analysis etc. because the SubSp is well-bracketed inside a block.
      if (isa<SubSpOp>(op)) {
        delta += V(op);
        continue;
      }

      if (auto rd = op->find<SpilledRdAttr>()) {
        // We will rematerialize them later.
        if (isa<MovIOp>(rd->ref) || isa<AdrOp>(rd->ref)) {
          remove.push_back(op);
          continue;
        }

        int offset = delta + rd->offset;
        bool fp = rd->fp;
        auto reg = fp ? fspillReg : spillReg;

        builder.setAfterOp(op);
        if (offset < delta)
          builder.create<FmovXOp>({ RDC(Reg(delta - offset)), RSC(reg) });
        else
          emitStackStore(builder, reg, fp, offset, spillReg2);
        op->add<RdAttr>(reg);
      }

      if (auto rs = op->find<SpilledRsAttr>()) {
        int offset = delta + rs->offset;
        bool fp = rs->fp;
        auto reg = fp ? fspillReg : spillReg;

        builder.setBeforeOp(op);
        // Rematerialized.
        auto ref = rs->ref;
        if (isa<MovIOp>(ref))
          builder.create<MovIOp>({ RDC(reg), new IntAttr(V(ref)) });
        else if (isa<AdrOp>(ref))
          builder.create<AdrOp>({ RDC(reg), new NameAttr(NAME(ref)) });
        else if (offset < delta)
          builder.create<FmovDOp>({ RDC(reg), RSC(Reg(delta - offset)) });
        else
          emitStackLoad(builder, reg, fp, offset, spillReg2);
        op->add<RsAttr>(reg);
      }

      if (auto rs2 = op->find<SpilledRs2Attr>()) {
        int offset = delta + rs2->offset;
        bool fp = rs2->fp;
        auto reg = fp ? fspillReg2 : spillReg2;

        builder.setBeforeOp(op);
        auto ref = rs2->ref;
        if (isa<MovIOp>(ref))
          builder.create<MovIOp>({ RDC(reg), new IntAttr(V(ref)) });
        else if (isa<AdrOp>(ref))
          builder.create<AdrOp>({ RDC(reg), new NameAttr(NAME(ref)) });
        else if (offset < delta)
          builder.create<FmovDOp>({ RDC(reg), RSC(Reg(delta - offset)) });
        else
          emitStackLoad(builder, reg, fp, offset, spillReg3);
        op->add<Rs2Attr>(reg);
      }

      if (auto rs3 = op->find<SpilledRs3Attr>()) {
        int offset = delta + rs3->offset;
        bool fp = rs3->fp;
        auto reg = fp ? fspillReg3 : spillReg3;

        builder.setBeforeOp(op);
        auto ref = rs3->ref;
        if (isa<MovIOp>(ref))
          builder.create<MovIOp>({ RDC(reg), new IntAttr(V(ref)) });
        else if (isa<AdrOp>(ref))
          builder.create<AdrOp>({ RDC(reg), new NameAttr(NAME(ref)) });
        else if (offset < delta)
          builder.create<FmovDOp>({ RDC(reg), RSC(Reg(delta - offset)) });
        else
          emitStackLoad(builder, reg, fp, offset, spillReg2);
        op->add<Rs3Attr>(reg);
      }
    }
  }

  for (auto op : remove)
    op->erase();
}

void RegAlloc::run() {
  auto funcs = collectFuncs();
  fnMap = getFunctionMap();
  std::set<FuncOp*> leaves;

  for (auto func : funcs) {
    auto calls = func->findAll<BlOp>();
    if (calls.size() == 0)
      leaves.insert(func);
    runImpl(func->getRegion(), calls.size() == 0);
  }

  // Have a look at what registers are used inside each function.
  for (auto func : funcs) {
    auto &set = usedRegisters[func];
    for (auto bb : func->getRegion()->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (op->has<RdAttr>())
          set.insert(op->get<RdAttr>()->reg);
        if (op->has<RsAttr>())
          set.insert(op->get<RsAttr>()->reg);
        if (op->has<Rs2Attr>())
          set.insert(op->get<Rs2Attr>()->reg);
        if (op->has<Rs3Attr>())
          set.insert(op->get<Rs3Attr>()->reg);
      }
    }
  }

  for (auto func : funcs) {
    proEpilogue(func, leaves.count(func));
    tidyup(func->getRegion());
  }
}
