#include "ArmPasses.h"
#include "ArmOps.h"
#include "../opt/Analysis.h"
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <list>
#include <climits>
#include <functional>

using namespace sys;
using namespace sys::arm;

// Pre-RA Instruction Scheduling for ARM ops (mirrors src/rv/Schedule.cpp).
//
// Reorders instructions within a basic block to hide pipeline latencies on
// Cortex-A class out-of-order cores:
//   - GPR load (ldr/ldur): ~3 cycles to result available
//   - FP load (ldr d/s)  : ~5 cycles
//   - Integer multiply   : ~3-4 cycles
//   - Integer divide     : ~12-20 cycles (variable)
//   - FP mul/madd        : ~4 cycles
//   - FP div/sqrt        : ~10+ cycles
//   - Most ALU           : 1 cycle
//
// Uses list scheduling with a goodness heuristic that prefers:
//   1. Producers of high-latency operations (start them early)
//   2. Operations that consume high-latency results that are now ready
//   3. Stable original ordering as a tie-break
//
// Respects:
//   - Memory dependencies (load-after-store / store-after-* / store-store)
//   - Register dependencies (data flow within the block)
//   - Pinned ops (calls, branches, write_reg/read_reg pairs, sp adjustment)
//
// The pass is intentionally conservative: any block containing a pinned
// non-terminator op (call, write_reg, sp-update, ...) is left untouched so
// implicit ABI / register constraints remain in place.

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0]) return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

int envInt(const char *name, int fallback, int minValue, int maxValue) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0]) return fallback;
  char *end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if ((end && *end) || parsed < minValue || parsed > maxValue)
    return fallback;
  return (int) parsed;
}

bool isLoadGPR(Op *op) {
  return isa<LdrWOp>(op) || isa<LdrXOp>(op) ||
         isa<LdrWPOp>(op) || isa<LdrXPOp>(op) ||
         isa<LdrWROp>(op) || isa<LdrXROp>(op);
}

bool isLoadFP(Op *op) {
  return isa<LdrFOp>(op) || isa<LdrDOp>(op) ||
         isa<LdrFPOp>(op) || isa<LdrFROp>(op) ||
         isa<Ld1Op>(op);
}

bool isLoad(Op *op) { return isLoadGPR(op) || isLoadFP(op); }

bool isStore(Op *op) {
  return isa<StrWOp>(op) || isa<StrXOp>(op) || isa<StrFOp>(op) || isa<StrDOp>(op) ||
         isa<StrWPOp>(op) || isa<StrXPOp>(op) || isa<StrFPOp>(op) ||
         isa<StrWROp>(op) || isa<StrXROp>(op) || isa<StrFROp>(op) ||
         isa<St1Op>(op);
}

// Latency model for ARM ops: cycles until the result becomes available
// to a dependent consumer. Numbers are tuned for Cortex-A55/A72-class
// cores, which is the typical contest evaluator profile.
int latency(Op *op) {
  if (isLoadFP(op)) return 5;
  if (isLoadGPR(op)) return 3;

  if (isa<MulWOp>(op) || isa<MulXOp>(op) ||
      isa<MaddWOp>(op) || isa<MaddXOp>(op) ||
      isa<MsubWOp>(op) || isa<MsubXOp>(op))
    return 3;

  if (isa<SdivWOp>(op) || isa<SdivXOp>(op) || isa<UdivWOp>(op))
    return 12;

  if (isa<FmulOp>(op) || isa<FmaddOp>(op) || isa<FmsubOp>(op) ||
      isa<FaddOp>(op) || isa<FsubOp>(op))
    return 4;
  if (isa<FdivOp>(op))
    return 10;

  if (isa<arm::MulVOp>(op) || isa<arm::MulFVOp>(op) || isa<arm::MlaVOp>(op))
    return 4;
  if (isa<arm::AddVOp>(op) || isa<arm::SubVOp>(op) ||
      isa<arm::AddFVOp>(op) || isa<arm::SubFVOp>(op))
    return 3;

  if (isa<ScvtfOp>(op) || isa<FcvtzsOp>(op) || isa<FmovOp>(op))
    return 3;

  return 1;
}

// Pinned ops cannot be moved (they have side effects, terminate the block,
// or carry implicit ABI / register constraints we must preserve in order).
bool isPinned(Op *op) {
  return isa<BlOp>(op) ||                           // call
         isa<RetOp>(op) ||                          // return
         isa<BOp>(op) ||                            // unconditional jump
         isa<BeqOp>(op) || isa<BneOp>(op) ||
         isa<BltOp>(op) || isa<BgeOp>(op) ||
         isa<BleOp>(op) || isa<BgtOp>(op) ||
         isa<BmiOp>(op) || isa<BplOp>(op) ||
         isa<CbzOp>(op) || isa<CbnzOp>(op) ||
         isa<WriteRegOp>(op) || isa<ReadRegOp>(op) ||
         isa<SubSpOp>(op) ||                        // stack-pointer mutation
         isa<PlaceHolderOp>(op) ||
         isa<arm::CloneOp>(op) || isa<arm::JoinOp>(op) || isa<arm::WakeOp>(op);
}

// Returns the address operand of a memory op. For loads (LdrXOp etc.) the
// address is the first operand; for stores it is the second (operand 0 is
// the value being stored). Vector Ld1/St1 follow the same convention as
// their scalar counterparts in this IR.
Op *getMemAddr(Op *op) {
  if (isLoad(op))
    return op->getOperandCount() > 0 ? op->DEF(0) : nullptr;
  if (isStore(op))
    return op->getOperandCount() > 1 ? op->DEF(1) : nullptr;
  return nullptr;
}

// May-alias check for ARM-level addresses. Leverages AliasAttr if it has
// been propagated onto the load/store op (the front-end attaches it
// pre-lowering and the ARM lowering preserves it where possible). Falls
// back to identity comparison of address SSA values, which is conservative
// (different SSA values are assumed to alias unless metadata says otherwise).
bool addressMayAlias(Op *memOp1, Op *memOp2) {
  if (!memOp1 || !memOp2) return true;
  if (memOp1 == memOp2) return true;

  auto alias1 = memOp1->find<AliasAttr>();
  auto alias2 = memOp2->find<AliasAttr>();
  if (alias1 && alias2 && !alias1->unknown && !alias2->unknown) {
    if (alias1->neverAlias(alias2))
      return false;
  }

  Op *addr1 = getMemAddr(memOp1);
  Op *addr2 = getMemAddr(memOp2);
  if (!addr1 || !addr2) return true;
  if (addr1 == addr2) return true;
  return true;
}

} // namespace

namespace sys { namespace arm {

std::map<std::string, int> Schedule::stats() {
  return {
    { "reordered", reordered },
    { "critical-path-nodes", criticalPathNodes },
    { "critical-path-max-height", criticalPathMaxHeight },
    { "height-weight", heightWeight },
  };
}

void Schedule::runImpl(BasicBlock *bb) {
  if (!bb || bb->getOpCount() == 0)
    return;

  auto term = bb->getLastOp();
  if (!term)
    return;

  // Don't reorder blocks with calls or other pinned non-terminator ops at
  // all. This also covers FP calls and implicit ABI constraints without
  // needing a separate whole-block FP bailout.
  for (auto op : bb->getOps()) {
    if (op == term) continue;
    if (isa<PhiOp>(op)) continue;
    if (isPinned(op)) return;
  }

  // Build dependence DAG.
  // For each op, count how many predecessors it depends on.
  // Memory ordering: stores can't move past stores or aliased loads/stores.
  std::vector<Op*> stores, loads;
  std::unordered_map<Op*, std::vector<Op*>> memDeps; // op → ops it must come after

  for (auto op : bb->getOps()) {
    if (op == term || isa<PhiOp>(op)) continue;

    if (isLoad(op)) {
      for (auto s : stores) {
        if (addressMayAlias(op, s))
          memDeps[op].push_back(s);
      }
      loads.push_back(op);
    } else if (isStore(op)) {
      for (auto s : stores) {
        if (addressMayAlias(op, s))
          memDeps[op].push_back(s);
      }
      for (auto l : loads) {
        if (addressMayAlias(op, l))
          memDeps[op].push_back(l);
      }
      stores.push_back(op);
    }
  }

  std::unordered_map<Op*, int> degree;
  std::unordered_map<Op*, std::vector<Op*>> users;
  std::vector<Op*> orderedSchedulable;
  std::unordered_set<Op*> schedulable;

  for (auto op : bb->getOps()) {
    if (op == term || isa<PhiOp>(op)) continue;
    orderedSchedulable.push_back(op);
    schedulable.insert(op);
  }

  for (auto op : orderedSchedulable) {
    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      if (def && schedulable.count(def)) {
        degree[op]++;
        users[def].push_back(op);
      }
    }
    for (auto pred : memDeps[op]) {
      if (schedulable.count(pred)) {
        degree[op]++;
        users[pred].push_back(op);
      }
    }
  }

  // Compute critical-path height for each node:
  //   height[U] = latency(U) + max( { height[V] for all users V of U } )
  std::unordered_map<Op*, int> height;
  std::function<int(Op*)> computeHeight = [&](Op *op) -> int {
    auto it = height.find(op);
    if (it != height.end())
      return it->second;
    int maxUserHeight = 0;
    auto userIt = users.find(op);
    if (userIt != users.end()) {
      for (auto user : userIt->second)
        maxUserHeight = std::max(maxUserHeight, computeHeight(user));
    }
    int h = latency(op) + maxUserHeight;
    height[op] = h;
    return h;
  };

  for (auto op : orderedSchedulable)
    computeHeight(op);
  criticalPathNodes += (int) height.size();
  for (auto &[_, h] : height)
    criticalPathMaxHeight = std::max(criticalPathMaxHeight, h);

  std::list<Op*> ready;
  for (auto op : orderedSchedulable) {
    if (degree[op] == 0)
      ready.push_back(op);
  }

  std::unordered_map<Op*, int> issueTime;
  int currentTime = 0;
  int origCount = (int) schedulable.size();

  std::unordered_map<Op*, int> origPos;
  int pos = 0;
  for (auto op : orderedSchedulable)
    origPos[op] = pos++;

  // Goodness function: higher is better.
  // Priorities:
  //   1. Critical-path height (drive longest dependency chain first)
  //   2. Bonus for high-latency ops (start them early to hide latency)
  //   3. Penalty if any operand is still in latency window (we'd stall)
  //   4. Stable tie-break by original position
  auto goodness = [&](Op *op) -> int {
    int score = 0;

    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      auto it = issueTime.find(def);
      if (it != issueTime.end()) {
        int gap = currentTime - it->second;
        int needed = latency(def);
        if (gap < needed)
          score -= (needed - gap) * 4;
      }
    }

    score += height[op] * heightWeight;

    int lat = latency(op);
    if (lat >= 3) score += lat * 2;

    score -= origPos[op] / 100;
    return score;
  };

  std::vector<Op*> newOrder;
  newOrder.reserve(origCount);

  while (!ready.empty()) {
    auto bestIt = ready.begin();
    int bestScore = goodness(*bestIt);
    for (auto it = std::next(ready.begin()); it != ready.end(); ++it) {
      int s = goodness(*it);
      if (s > bestScore) {
        bestScore = s;
        bestIt = it;
      }
    }
    Op *op = *bestIt;
    ready.erase(bestIt);

    issueTime[op] = currentTime;
    currentTime++;
    newOrder.push_back(op);

    for (auto user : users[op]) {
      if (--degree[user] == 0)
        ready.push_back(user);
    }
  }

  if ((int) newOrder.size() != origCount)
    return; // sanity: leave the block unchanged if something went wrong

  bool changedOrder = false;
  for (int i = 0; i < origCount; i++) {
    if (orderedSchedulable[i] != newOrder[i]) {
      changedOrder = true;
      break;
    }
  }
  if (!changedOrder)
    return;

  // Apply the new ordering: move each scheduled op to be immediately before
  // the terminator, in the order they appear in newOrder. Since moveBefore()
  // inserts before the target, calling it in order rebuilds the sequence.
  for (auto op : newOrder)
    op->moveBefore(term);

  reordered++;
}

void Schedule::run() {
  if (!envEnabled("SISY_ARM_ENABLE_SCHEDULE", true))
    return;

  heightWeight = envInt("SISY_ARM_SCHEDULE_HEIGHT_WEIGHT", 3, 0, 100);

  for (auto func : collectFuncs()) {
    auto region = func->getRegion();
    for (auto bb : region->getBlocks())
      runImpl(bb);
  }
}

}} // namespace sys::arm
