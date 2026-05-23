#include "RvPasses.h"
#include "Regs.h"
#include "../backend/shared/RegAllocHotness.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace sys;
using namespace sys::rv;

namespace {

bool envEnabled(const char *name, bool fallback = true) {
  const char *v = std::getenv(name);
  if (!v || !v[0])
    return fallback;
  if (std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0)
    return false;
  return true;
}

int envInt(const char *name, int fallback, int minValue, int maxValue) {
  const char *v = std::getenv(name);
  if (!v || !v[0])
    return fallback;
  char *end = nullptr;
  long parsed = std::strtol(v, &end, 10);
  if ((end && *end) || parsed < minValue || parsed > maxValue)
    return fallback;
  return (int) parsed;
}

class SpilledRdAttr : public AttrImpl<SpilledRdAttr, RVLINE + 2097152> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRdAttr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rd-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">"; }
  SpilledRdAttr *clone() override { return new SpilledRdAttr(fp, offset, ref); }
};

class SpilledRsAttr : public AttrImpl<SpilledRsAttr, RVLINE + 2097152> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRsAttr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rs-spilled = " + std::to_string(offset) + (fp ? "f" : "") + ">"; }
  SpilledRsAttr *clone() override { return new SpilledRsAttr(fp, offset, ref); }
};

class SpilledRs2Attr : public AttrImpl<SpilledRs2Attr, RVLINE + 2097152> {
public:
  bool fp;
  int offset;
  Op *ref;

  SpilledRs2Attr(bool fp, int offset, Op *ref): fp(fp), offset(offset), ref(ref) {}

  std::string toString() override { return "<rs2-spilled = " + std::to_string(offset) + + (fp ? "f" : "") + ">"; }
  SpilledRs2Attr *clone() override { return new SpilledRs2Attr(fp, offset, ref); }
};

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

void emitStackStore(Builder &builder, Reg src, bool fp, int offset) {
  if (isVector(src)) {
    materializeSpAddr(builder, spillReg2, offset);
    builder.create<Vse32Op>({ RSC(src), RS2C(spillReg2) });
    return;
  }

  if (fitsImm12(offset)) {
    if (fp)
      builder.create<FsdOp>({ RSC(src), RS2C(Reg::sp), new IntAttr(offset) });
    else
      builder.create<sys::rv::StoreOp>({ RSC(src), RS2C(Reg::sp), new IntAttr(offset), new SizeAttr(8) });
    return;
  }

  materializeSpAddr(builder, spillReg2, offset);
  if (fp)
    builder.create<FsdOp>({ RSC(src), RS2C(spillReg2), new IntAttr(0) });
  else
    builder.create<sys::rv::StoreOp>({ RSC(src), RS2C(spillReg2), new IntAttr(0), new SizeAttr(8) });
}

void emitStackLoad(Builder &builder, Reg dst, Value::Type ty, int offset, Reg addrTmp) {
  if (isVector(dst)) {
    materializeSpAddr(builder, addrTmp, offset);
    builder.create<Vle32Op>(ty, { RDC(dst), RSC(addrTmp) });
    return;
  }

  if (fitsImm12(offset)) {
    builder.create<sys::rv::LoadOp>(ty, { RDC(dst), RSC(Reg::sp), new IntAttr(offset), new SizeAttr(8) });
    return;
  }

  materializeSpAddr(builder, addrTmp, offset);
  builder.create<sys::rv::LoadOp>(ty, { RDC(dst), RSC(addrTmp), new IntAttr(0), new SizeAttr(8) });
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

bool shareRegClass(Value::Type t1, Value::Type t2) {
  if (vecreg(t1) && vecreg(t2)) return true;
  if (fpreg(t1) && fpreg(t2)) return true;
  if (!vecreg(t1) && !fpreg(t1) && !vecreg(t2) && !fpreg(t2)) return true;
  return false;
}

bool splitCandidateType(Value::Type ty) {
  return ty == Value::i32 || ty == Value::i64 || ty == Value::f32;
}

bool splitCandidateValue(Op *op) {
  if (!op)
    return false;
  if (!splitCandidateType(op->getResultType()))
    return false;
  if (isa<PlaceHolderOp>(op) || isa<JOp>(op) || isa<RetOp>(op) ||
      isa<BranchOp>(op) || isa<WriteRegOp>(op) || isa<sys::rv::CallOp>(op))
    return false;
  // Constants and labels are cheap to rematerialize at their use sites; splitting
  // them only creates extra copy pressure.
  if (isa<LiOp>(op) || isa<LaOp>(op))
    return false;
  return true;
}

bool blockHasCall(BasicBlock *bb) {
  for (auto *op : bb->getOps())
    if (isa<sys::rv::CallOp>(op))
      return true;
  return false;
}

Op *firstNonPhi(BasicBlock *bb) {
  for (auto op : bb->getOps())
    if (!isa<PhiOp>(op))
      return op;
  return nullptr;
}

int replaceUsesAfterCopy(BasicBlock *bb, Op *copy, Op *oldValue, Op *stopBefore) {
  int replaced = 0;
  (void) bb;
  for (auto op = copy->nextOp(); op; op = op->nextOp()) {
    if (op == stopBefore)
      break;
    for (int i = 0; i < op->getOperandCount(); i++) {
      if (op->getOperand(i).defining != oldValue)
        continue;
      op->setOperand(i, copy);
      replaced++;
    }
  }
  return replaced;
}

int splitHotLiveRanges(Region *region,
                       const std::unordered_map<BasicBlock*, int> &bbWeight) {
  if (!envEnabled("SISY_RV_ENABLE_LIVE_RANGE_SPLIT", true))
    return 0;

  const int threshold =
      envInt("SISY_RV_LIVE_RANGE_SPLIT_HOT_THRESHOLD", 64, 1, 1000000000);
  const int maxSplitsPerBlock =
      envInt("SISY_RV_LIVE_RANGE_SPLIT_MAX_PER_BLOCK", 256, 1, 256);

  Builder builder;
  int splits = 0;

  auto usedInBlockBody = [](Op *value, BasicBlock *bb) -> bool {
    if (!value || !bb)
      return false;
    for (auto op : bb->getOps()) {
      if (isa<PhiOp>(op))
        continue;
      for (auto operand : op->getOperands()) {
        if (operand.defining == value)
          return true;
      }
    }
    return false;
  };

  for (auto bb : region->getBlocks()) {
    auto weightIt = bbWeight.find(bb);
    int weight = weightIt == bbWeight.end() ? 1 : weightIt->second;
    if (weight < threshold || bb->getOpCount() == 0)
      continue;
    if (envEnabled("SISY_RV_LIVE_RANGE_SPLIT_REQUIRE_CALL", false) &&
        !blockHasCall(bb))
      continue;

    Op *insertBefore = firstNonPhi(bb);
    if (!insertBefore || isa<JOp>(insertBefore) || isa<RetOp>(insertBefore))
      continue;

    std::vector<Op*> liveIns;
    liveIns.reserve(bb->getLiveIn().size());

    for (auto value : bb->getLiveIn()) {
      if (!splitCandidateValue(value))
        continue;
      if (value->getParent() == bb)
        continue;
      if (!usedInBlockBody(value, bb))
        continue;
      liveIns.push_back(value);
    }

    std::sort(liveIns.begin(), liveIns.end(), [&](Op *a, Op *b) {
      if (a->getUses().size() != b->getUses().size())
        return a->getUses().size() > b->getUses().size();
      return a < b;
    });

    int blockSplits = 0;
    for (auto value : liveIns) {
      if (blockSplits >= maxSplitsPerBlock)
        break;

      builder.setBeforeOp(insertBefore);
      Op *copy = nullptr;
      if (value->getResultType() == Value::f32)
        copy = builder.create<FmvOp>({ value });
      else
        copy = builder.create<MvOp>({ value });
      copy->setResultType(value->getResultType());

      int replaced = replaceUsesAfterCopy(bb, copy, value, nullptr);
      if (replaced == 0) {
        copy->removeAllOperands();
        copy->erase();
        continue;
      }

      splits++;
      blockSplits++;
    }
  }

  return splits;
}

}

std::map<std::string, int> RegAlloc::stats() {
  return {
    { "spilled", spilled },
    { "peepholed", convertedTotal },
    { "max-block-hotness", maxBlockHotness },
    { "live-range-splits", liveRangeSplits },
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

#define BINARY ADD_ATTR(0, RsAttr) ADD_ATTR(1, Rs2Attr)
#define UNARY ADD_ATTR(0, RsAttr)

#define CREATE_MV(fp, rd, rs) \
  if (!fp) \
    builder.create<MvOp>({ RDC(rd), RSC(rs) }); \
  else \
    builder.create<FmvOp>({ RDC(rd), RSC(rs) });

// Implemented in OpBase.cpp.
std::string getValueNumber(Value value);

// For debug purposes
void dumpInterf(Region *region, const std::unordered_map<Op*, std::unordered_set<Op*>> &interf) {
  region->dump(std::cerr, /*depth=*/1);
  std::cerr << "\n\n===== interference graph =====\n\n";
  for (auto [k, v] : interf) {
    std::cerr << getValueNumber(k->getResult()) << ": ";
    for (auto op : v)
      std::cerr << getValueNumber(op->getResult()) << " ";
    std::cerr << "\n";
  }
}

void dumpAssignment(Region *region, const std::unordered_map<Op*, Reg> &assignment) {
  region->dump(std::cerr, /*depth=*/1);
  std::cerr << "\n\n===== assignment =====\n\n";
  for (auto [k, v] : assignment) {
    std::cerr << getValueNumber(k->getResult()) << " = " << showReg(v) << "\n";
  }
}

// Used in constructing interference graph.
struct Event {
  int timestamp;
  bool start;
  Op *op;
};

void RegAlloc::runImpl(Region *region, bool isLeaf) {
  const Reg *order = isLeaf ? leafOrder : normalOrder;
  const Reg *orderf = isLeaf ? leafOrderf : normalOrderf;
  const Reg *orderv = isLeaf ? leafOrderv : normalOrderv;
  const int regcount = isLeaf ? leafRegCnt : normalRegCnt;
  const int regcountf = isLeaf ? leafRegCntf : normalRegCntf;
  const int regcountv = isLeaf ? leafRegCntv : normalRegCntv;

  int opCount = 0;
  for (auto bb : region->getBlocks()) { opCount += bb->getOps().size(); }
  bool localFastMode = fastMode || (opCount > 3000);

  Builder builder;
  
  std::unordered_map<Op*, Reg> assignment;

  auto funcOp = region->getParent();

  // First of all, add 35 precolored placeholders before each call.
  // This denotes that a CallOp clobbers those registers.
  runRewriter(funcOp, [&](CallOp *op) {
    // Make sure arguments don't conflict.
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

    return false;
  });

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

  BasicBlock *entry = region->getFirstBlock();
  for (size_t i = 0; i < getArgs.size(); i++) {
    // A missing argument.
    if (!getArgs[i])
      continue;

    Op *op = getArgs[i];
    auto ty = op->getResultType();
    auto placement = getArgPlacement(argTypes, (int) i);

    // It is necessary to put those GetArgs to the front.
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
      builder.replace<ReadRegOp>(op, ty, { new RegAttr(argRegs[placement.regIndex]) });
      continue;
    }
    // Spilled to stack; don't do anything.
  }

  region->updateLiveness();

  // Build a coarse block hotness model for spill heuristics:
  // back-edge blocks are likely loop bodies; call-heavy blocks are also hot.
  auto bbWeight = sys::backend::shared::computeBlockHotness(region, [](Op *op) {
    return isa<CallOp>(op);
  });
  for (auto &[_, weight] : bbWeight)
    maxBlockHotness = std::max(maxBlockHotness, weight);

  int splits = localFastMode ? 0 : splitHotLiveRanges(region, bbWeight);
  if (splits) {
    liveRangeSplits += splits;
    region->updateLiveness();
  }

  std::unordered_map<Op*, int> spillOffset;
  int currentOffset = STACKOFF(funcOp);
  if (currentOffset % 8 != 0)
    currentOffset = currentOffset / 8 * 8 + 8;
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
      
      if (isa<LiOp>(op))
        priority[op] = (V(op) <= 2047 && V(op) >= -2048) ? -3 : -1;
      if (isa<LaOp>(op))
        priority[op] = op->getUses().size() <= 1 ? -1 : 100000;

      if (isa<PhiOp>(op)) {
        priority[op] = currentPriority + 1;
        for (auto x : op->getOperands()) {
          priority[x.defining] = currentPriority;
          if (!localFastMode)
            prefer[x.defining] = op;
          phiOperand[op].push_back(x.defining);
        }
        currentPriority += 2;
      }

      // Preserve copy chains to reduce move pressure after allocation.
      if (!localFastMode && (isa<MvOp>(op) || isa<FmvOp>(op)) && op->getOperandCount() == 1) {
        auto src = op->DEF(0);
        prefer[op] = src;
        bumpPriority(op, currentPriority + 2);
        bumpPriority(src, currentPriority + 1);
      }
    }

    // For all liveOuts, they are last-used at place size().
    // If they aren't defined in this block, then `defined[op]` will be zero, which is intended.
    for (auto op : bb->getLiveOut())
      lastUsed[op] = ops.size();

    // Penalize long-lived values and values spanning calls.
    std::vector<int> callPrefix;
    if (!localFastMode) {
      callPrefix.assign(ops.size() + 1, 0);
      int callIdx = 0;
      for (auto liveOp : ops) {
        callPrefix[callIdx + 1] = callPrefix[callIdx] + (isa<CallOp>(liveOp) ? 1 : 0);
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
        spillWeight[op] += 96LL * localWeight * callCount;
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
      if (isa<JOp>(op))
        continue;

      if (event.start) {
        for (Op* activeOp : active) {
          // Different register classes use different register files.
          // However, they are using the same stack,
          // so that must be taken into account when spilling.
          if (!shareRegClass(activeOp->getResultType(), op->getResultType())) {
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

    for (auto op : ops) {
    // Do not allocate colored instructions.
    if (assignment.count(op))
      continue;

    std::unordered_set<Reg> bad, unpreferred;

    for (auto v : interf[op]) {
      // In the whole function, `sp` and `zero` are read-only.
      if (assignment.count(v) && assignment[v] != Reg::sp && assignment[v] != Reg::zero)
        bad.insert(assignment[v]);
    }

    if (isa<PhiOp>(op)) {
      // Dislike everything that might interfere with phi's operands.
      const auto &operands = phiOperand[op];
      for (auto x : operands) {
        for (auto v : interf[x]) {
          if (assignment.count(v) && assignment[v] != Reg::sp && assignment[v] != Reg::zero)
            unpreferred.insert(assignment[v]);
        }
      }
    }

    if (prefer.count(op)) {
      auto ref = prefer[op];
      // Try to allocate the same register as `ref`.
      if (assignment.count(ref) && !bad.count(assignment[ref])) {
        Reg preferredReg = assignment[ref];
        bool crossCallRisk =
          (callSpan[op] > 1 || callSpan[ref] > 1) && callerSaved.count(preferredReg);
        if (!crossCallRisk) {
          assignment[op] = preferredReg;
          continue;
        }
      }
    }

    // See if there's any preferred registers.
    int preferred = -1;
    for (auto use : op->getUses()) {
      if (isa<WriteRegOp>(use)) {
        auto reg = REG(use);
        if (!bad.count(reg)) {
          preferred = (int) reg;
          break;
        }
      }
    }
    if (isa<ReadRegOp>(op)) {
      auto reg = REG(op);
      // `readreg sp` is a snapshot value in SSA.
      // Mapping it back to physical `sp` breaks semantics after later SubSpOp
      // adjustments (value should remain the old stack pointer).
      if (reg != Reg::sp && !bad.count(reg))
        preferred = (int) reg;
    }

    if (preferred != -1) {
      assignment[op] = (Reg) preferred;
      continue;
    }

    auto rcnt = vecreg(op->getResultType()) ? regcountv : (!fpreg(op->getResultType()) ? regcount : regcountf);
    auto rorder = vecreg(op->getResultType()) ? orderv : (!fpreg(op->getResultType()) ? order : orderf);
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

    bool preferCallee = !isLeaf && callSpan[op] > 0;
    bool assigned = false;
    if (preferCallee)
      assigned = tryAssign(/*onlyCallee=*/ true, /*allowUnpreferred=*/ false);
    else
      assigned = tryAssign(/*onlyCallee=*/ false, /*allowUnpreferred=*/ false);

    // We have excluded too much. Try it again.
    if (!assigned && unpreferred.size()) {
      if (preferCallee)
        assigned = tryAssign(/*onlyCallee=*/ true, /*allowUnpreferred=*/ true);
      else
        assigned = tryAssign(/*onlyCallee=*/ false, /*allowUnpreferred=*/ true);
    }

    if (assignment.count(op))
      continue;

    spilled++;
    // Spilled. Try to see all spill offsets of conflicting ops.
    int desired = currentOffset;
    if (vecreg(op->getResultType())) {
      if (desired % 16 != 0)
        desired = (desired / 16 + 1) * 16;
    }
    std::unordered_set<int> conflict;

    // Consider both `interf` (of the same register type)
    // and `spillInterf` (of different register type).
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
    if (vecreg(op->getResultType())) {
      while (conflict.count(desired) || conflict.count(desired + 8))
        desired += 16;
    } else {
      while (conflict.count(desired))
        desired += 8;
    }

    spillOffset[op] = desired;

    // Update `highest`, which will indicate the size allocated.
    if (vecreg(op->getResultType())) {
      if (desired + 8 > highest)
        highest = desired + 8;
    } else {
      if (desired > highest)
        highest = desired;
    }
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
        if (vecreg(op->getResultType())) {
          if (nextOffset % 16 != 0)
            nextOffset = (nextOffset / 16 + 1) * 16;
          spillOffset[op] = nextOffset;
          highest = nextOffset + 8;
          nextOffset += 16;
        } else {
          spillOffset[op] = nextOffset;
          highest = nextOffset;
          nextOffset += 8;
        }
        spilled++;
      }
    }
  }

  // Only a single register is spilled. Let's use s10.
  // Fast mode keeps spill semantics simple and uniform (stack-only).
  if (!localFastMode && highest == currentOffset) {
    for (auto [op, _] : spillOffset)
      assignment[op] = fpreg(op->getResultType()) ? fspillReg : spillReg;
    spillOffset.clear();
  }

  // Only 2 registers are spilled. Let's use s10 and s11..
  if (!localFastMode && highest == currentOffset + 8) {
    for (auto [op, offset] : spillOffset) {
      auto fp = fpreg(op->getResultType());
      assignment[op] = (offset == currentOffset) ? (fp ? fspillReg : spillReg)
                                                 : (fp ? fspillReg2 : spillReg2);
    }
    spillOffset.clear();
  }

  // Mapping integer spill slots to FP registers is fragile around large
  // argument frames and dynamic call-frame deltas. Keep it opt-in.
  if (!localFastMode && spillOffset.size() && envEnabled("SISY_RV_SPILL_TO_FP", false)) {
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
      vecreg(op->getResultType()) ? orderv[0] : (fpreg(op->getResultType()) ? orderf[0] : order[0]);
  };

  // Convert all operands to registers.
  LOWER(AddOp, BINARY);
  LOWER(AddwOp, BINARY);
  LOWER(SubOp, BINARY);
  LOWER(SubwOp, BINARY);
  LOWER(MulwOp, BINARY);
  LOWER(MulhOp, BINARY);
  LOWER(MulhuOp, BINARY);
  LOWER(MulOp, BINARY);
  LOWER(DivwOp, BINARY);
  LOWER(DivOp, BINARY);
  LOWER(RemwOp, BINARY);
  LOWER(RemOp, BINARY);
  LOWER(BneOp, BINARY);
  LOWER(BeqOp, BINARY);
  LOWER(BltOp, BINARY);
  LOWER(BgeOp, BINARY);
  LOWER(StoreOp, BINARY);
  LOWER(AndOp, BINARY);
  LOWER(OrOp, BINARY);
  LOWER(XorOp, BINARY);
  LOWER(SltOp, BINARY);
  LOWER(FaddOp, BINARY);
  LOWER(FsubOp, BINARY);
  LOWER(FmulOp, BINARY);
  LOWER(FdivOp, BINARY);
  LOWER(FeqOp, BINARY);
  LOWER(FltOp, BINARY);
  LOWER(FleOp, BINARY);
  LOWER(VaddvvOp, BINARY);
  LOWER(VsubvvOp, BINARY);
  LOWER(VmulvvOp, BINARY);
  LOWER(VfaddvvOp, BINARY);
  LOWER(VfsubvvOp, BINARY);
  LOWER(VfmulvvOp, BINARY);
  LOWER(Vse32Op, BINARY);
  LOWER(SllwOp, BINARY);
  LOWER(SrlwOp, BINARY);
  LOWER(SrawOp, BINARY);
  LOWER(SraOp, BINARY);
  LOWER(SllOp, BINARY);
  LOWER(SrlOp, BINARY);
  
  LOWER(LoadOp, UNARY);
  LOWER(Vle32Op, UNARY);
  LOWER(VmvvxOp, UNARY);
  LOWER(VfmvvfOp, UNARY);
  LOWER(VsetvliOp, UNARY);
  LOWER(AddiwOp, UNARY);
  LOWER(AddiOp, UNARY);
  LOWER(SlliwOp, UNARY);
  LOWER(SrliwOp, UNARY);
  LOWER(SraiwOp, UNARY);
  LOWER(SraiOp, UNARY);
  LOWER(SlliOp, UNARY);
  LOWER(SrliOp, UNARY);
  LOWER(SeqzOp, UNARY);
  LOWER(SnezOp, UNARY);
  LOWER(SltiOp, UNARY);
  LOWER(AndiOp, UNARY);
  LOWER(OriOp, UNARY);
  LOWER(XoriOp, UNARY);
  LOWER(FcvtswOp, UNARY);
  LOWER(FcvtwsRtzOp, UNARY);
  LOWER(FmvwxOp, UNARY);
  LOWER(MvOp, UNARY);
  LOWER(FmvOp, UNARY);

  // Note that some ops are dealt with later.
  // We can't remove all operands here.
  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<PlaceHolderOp>(op) || isa<CallOp>(op) || isa<RetOp>(op))
        op->removeAllOperands();
    }
  }

  // Remove placeholders inserted previously.
  // We cannot directly erase that, otherwise `new`s might reuse the memory,
  // so that a newly constructed op might accidentally fall in `spillOffset`;
  // this means all ops must be erased at the end of the function.
  auto holders = funcOp->findAll<PlaceHolderOp>();
  for (auto holder : holders)
    holder->erase();

  //   writereg %1, <reg = a0>
  // becomes
  //   mv a0, assignment[%1]
  // As RdAttr is supplied, though `assignment[]` won't have the new op recorded, it's fine.
  runRewriter(funcOp, [&](WriteRegOp *op) {
    builder.setBeforeOp(op);
    auto src = op->DEF(0);
    auto dst = REG(op);

    // Rematerialize immediates directly at the write site to avoid relying on
    // long-lived constant carrier registers that may be coalesced aggressively.
    if (!isFP(dst) && isa<LiOp>(src)) {
      builder.create<LiOp>({ RDC(dst), new IntAttr(V(src)) });
      op->erase();
      return false;
    }
    if (!isFP(dst) && isa<LaOp>(src)) {
      builder.create<LaOp>({ RDC(dst), new NameAttr(NAME(src)) });
      op->erase();
      return false;
    }

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

  // Split edges.
  for (auto bb : bbs) {
    // If a block has multiple successors with phi, then we split the edges. As an example:
    // 
    // bb0:
    //   %0 = ...
    //   %1 = ...
    //   br %1 <bb1> <bb2>
    // bb1:
    //   phi %2, %0, ...
    // bb2:
    //   phi %3, %0, ...
    //
    // If we naively create a move at the end of bb0, then it's wrong.
    // We need to rewrite it into
    //
    // bb0:
    //   br %1 <bb3> <bb4>
    // bb3:
    //   j bb1
    // bb4:
    //   j bb2
    // ...
    //
    // To actually make it work.
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
    builder.create<JOp>({ new TargetAttr(oldTarget) });

    // Create edge for else branch.
    auto ifnot = bbTerm->get<ElseAttr>();
    auto oldElse = ifnot->bb;
    ifnot->bb = edge2;

    builder.setToBlockEnd(edge2);
    builder.create<JOp>({ new TargetAttr(oldElse) });

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

  // Don't forget that register 0 and offset 0 are the same.
  // We only need to guarantee that SOFFSET doesn't collide with existing regs.
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
        builder.setBeforeOp(term);
        auto def = ops[i].defining;
        Op *mv;
        if (fpreg(phi->getResultType())) {
          mv = builder.create<FmvOp>({
            new ImpureAttr,
            spillOffset.count(phi) ? (Attr*) new SpilledRdAttr GET_SPILLED_ARGS(phi) : RDC(getReg(phi)),
            spillOffset.count(def) ? (Attr*) new SpilledRsAttr GET_SPILLED_ARGS(def) : RSC(getReg(def))
          });
        } else {
          mv = builder.create<MvOp>({
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

  // Erase all phi's properly. There might be cross-reference across blocks,
  // so we need to remove all operands first.
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
        if (isa<LiOp>(rd->ref) || isa<LaOp>(rd->ref)) {
          remove.push_back(op);
          continue;
        }

        int offset = delta + rd->offset;
        bool fp = rd->fp;
        auto reg = vecreg(rd->ref->getResultType()) ? vspillReg : (fp ? fspillReg : spillReg);

        builder.setAfterOp(op);
        if (offset < delta)
          builder.create<FmvdxOp>({ RDC(Reg(delta - offset)), RSC(reg) });
        else
          emitStackStore(builder, reg, fp, offset);
        op->add<RdAttr>(reg);
      }

      if (auto rs = op->find<SpilledRsAttr>()) {
        int offset = delta + rs->offset;
        bool fp = rs->fp;
        bool vec = vecreg(rs->ref->getResultType());
        auto reg = vec ? vspillReg : (fp ? fspillReg : spillReg);
        auto ldty = vec ? Value::i128 : (fp ? Value::f32 : Value::i64);

        builder.setBeforeOp(op);
        // Rematerialized.
        auto ref = rs->ref;
        if (isa<LiOp>(ref))
          builder.create<LiOp>({ RDC(reg), new IntAttr(V(ref)) });
        else if (isa<LaOp>(ref))
          builder.create<LaOp>({ RDC(reg), new NameAttr(NAME(ref)) });
        else if (offset < delta)
          builder.create<FmvxdOp>({ RDC(reg), RSC(Reg(delta - offset)) });
        else
          emitStackLoad(builder, reg, ldty, offset, spillReg);
        op->add<RsAttr>(reg);
      }

      if (auto rs2 = op->find<SpilledRs2Attr>()) {
        int offset = delta + rs2->offset;
        bool fp = rs2->fp;
        bool vec = vecreg(rs2->ref->getResultType());
        auto reg = vec ? vspillReg2 : (fp ? fspillReg2 : spillReg2);
        auto ldty = vec ? Value::i128 : (fp ? Value::f32 : Value::i64);

        builder.setBeforeOp(op);
        // Rematerialized.
        auto ref = rs2->ref;
        if (isa<LiOp>(ref))
          builder.create<LiOp>({ RDC(reg), new IntAttr(V(ref)) });
        else if (isa<LaOp>(ref))
          builder.create<LaOp>({ RDC(reg), new NameAttr(NAME(ref)) });
        else if (offset < delta)
          builder.create<FmvxdOp>({ RDC(reg), RSC(Reg(delta - offset)) });
        else
          emitStackLoad(builder, reg, ldty, offset, spillReg2);
        op->add<Rs2Attr>(reg);
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
    auto calls = func->findAll<sys::rv::CallOp>();
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
      }
    }
  }

  for (auto func : funcs) {
    proEpilogue(func, leaves.count(func));
    if (envEnabled("SISY_RV_ENABLE_TIDYUP", true))
      tidyup(func->getRegion());
  }
}
