#include "RvPasses.h"
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
using namespace sys::rv;

// Pre-RA Instruction Scheduling for RV ops.
//
// Reorders instructions within basic blocks to hide pipeline latencies:
//   - Load: ~3 cycles to result available
//   - Multiply: ~3 cycles
//   - Divide/Remainder: ~30+ cycles (variable)
//   - Most ALU: 1 cycle
//
// Uses list scheduling with a goodness heuristic that prefers:
//   1. Producers of high-latency operations (start them early)
//   2. Operations that consume high-latency results that are now ready
//   3. Pure address computations (small) over loads (delay loads)
//
// Respects:
//   - Memory dependencies (load-after-store, store-after-load, store-after-store)
//   - Register dependencies (data flow)
//   - Pinned operations (calls with side effects, write_reg/read_reg pairs)

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

// Latency model for RV ops (cycles until result is available)
// Tuned for U74/C906-class pipelines: FP add has 3-cycle latency (not 1),
// and FP mul is 4 cycles.  Integer mul is 3 cycles on M-extension.
int latency(Op *op) {
  if (isa<rv::LoadOp>(op) || isa<FldOp>(op))
    return 3;
  if (isa<MulOp>(op) || isa<MulwOp>(op) || isa<MulhOp>(op) || isa<MulhuOp>(op))
    return 3;
  if (isa<DivOp>(op) || isa<DivwOp>(op) || isa<RemOp>(op) || isa<RemwOp>(op))
    return 20;
  if (isa<FmulOp>(op) || isa<FdivOp>(op))
    return 4;
  // FP add/sub have ~3 cycle latency on typical RV FPU pipelines (FPU is
  // usually pipelined but not fully bypassed).  Modeling this prevents the
  // scheduler from back-to-backing fadd/fsub after their producers.
  if (isa<FaddOp>(op) || isa<FsubOp>(op))
    return 3;
  return 1;
}

bool isStore(Op *op) {
  return isa<rv::StoreOp>(op) || isa<FsdOp>(op);
}

bool isLoad(Op *op) {
  return isa<rv::LoadOp>(op) || isa<FldOp>(op);
}

// Get the address operand of a memory op.
// LoadOp: address is the only / first operand
// StoreOp: operand 0 is value, operand 1 is address
Op *getMemAddr(Op *op) {
  if (isa<rv::LoadOp>(op) || isa<FldOp>(op))
    return op->getOperandCount() > 0 ? op->DEF(0) : nullptr;
  if (isa<rv::StoreOp>(op) || isa<FsdOp>(op))
    return op->getOperandCount() > 1 ? op->DEF(1) : nullptr;
  return nullptr;
}

// Pinned ops cannot be moved (they have side effects or implicit register constraints).
bool isPinned(Op *op) {
  return isa<rv::CallOp>(op) ||
         isa<RetOp>(op) ||
         isa<JOp>(op) ||
         isa<BneOp>(op) || isa<BeqOp>(op) ||
         isa<BltOp>(op) || isa<BgeOp>(op) ||
         isa<BleOp>(op) || isa<BgtOp>(op) ||
         isa<WriteRegOp>(op) || isa<ReadRegOp>(op) ||
         isa<VsetvliOp>(op) || isa<VsetvliResultOp>(op) ||
         isa<PlaceHolderOp>(op);
}

bool touchesFloat(Op *op) {
  if (op->getResultType() == Value::f32)
    return true;
  for (auto operand : op->getOperands()) {
    if (operand.defining && operand.defining->getResultType() == Value::f32)
      return true;
  }
  return false;
}

// May-alias check for RV-level addresses.
// Leverages AliasAttr if present on the load/store ops themselves.
// AliasAttr is computed pre-lowering and copied to rv::LoadOp/StoreOp during rv::Lower.
bool addressMayAlias(Op *memOp1, Op *memOp2) {
  if (!memOp1 || !memOp2) return true;
  if (memOp1 == memOp2) return true;

  // Check if both ops have AliasAttr
  auto alias1 = memOp1->find<AliasAttr>();
  auto alias2 = memOp2->find<AliasAttr>();

  if (alias1 && alias2 && !alias1->unknown && !alias2->unknown) {
    // If both have known alias info, use neverAlias check
    if (alias1->neverAlias(alias2))
      return false;
  }

  // Fallback: check if address operands are the same SSA value
  Op *addr1 = getMemAddr(memOp1);
  Op *addr2 = getMemAddr(memOp2);
  if (!addr1 || !addr2) return true;
  if (addr1 == addr2) return true;

  // Different address SSA values with no alias info → conservative
  return true;
}

} // namespace

namespace sys { namespace rv {

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

  // Collect all schedulable ops (skip phis, terminator, and pinned ops at boundaries).
  auto term = bb->getLastOp();
  if (!term)
    return;

  // Don't reorder blocks with pinned ops (calls, branches, etc.) in the
  // schedulable region.  However, calls that appear only at the very end of
  // the block (after all other ops) are treated as part of the terminator
  // region and do not prevent scheduling the preceding ops.  This is safe
  // because the call's clobbered registers are already modeled by
  // placeholders in RegAlloc, and moving earlier ops before the call does
  // not change the call's semantics.
  {
    bool hasEarlyPinned = false;
    bool seenNonPhiNonTerm = false;
    for (auto op : bb->getOps()) {
      if (op == term || isa<PhiOp>(op)) continue;
      if (isPinned(op)) {
        // If we have already seen a non-pinned, non-phi op, then this
        // pinned op is "early" (interspersed with schedulable work).
        if (seenNonPhiNonTerm) { hasEarlyPinned = true; break; }
      } else {
        seenNonPhiNonTerm = true;
      }
    }
    if (hasEarlyPinned) return;
  }

  // Build dependence DAG.
  // For each op, count how many predecessors it depends on (degree).
  // Memory ordering: stores can't move past stores or aliased loads/stores.
  std::vector<Op*> stores, loads;
  std::unordered_map<Op*, std::vector<Op*>> memDeps; // op → ops it must come after

  for (auto op : bb->getOps()) {
    if (op == term || isa<PhiOp>(op) || isPinned(op)) continue;

    if (isLoad(op)) {
      // Load must come after any aliased store
      for (auto s : stores) {
        if (addressMayAlias(op, s))
          memDeps[op].push_back(s);
      }
      loads.push_back(op);
    } else if (isStore(op)) {
      // Store must come after aliased prior stores and prior loads
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

  // Compute degree (number of unmet dependencies).
  std::unordered_map<Op*, int> degree;
  std::unordered_map<Op*, std::vector<Op*>> users; // reverse map: op → ops that depend on it
  std::vector<Op*> orderedSchedulable;
  std::unordered_set<Op*> schedulable;

  for (auto op : bb->getOps()) {
    if (op == term || isa<PhiOp>(op) || isPinned(op)) continue;
    orderedSchedulable.push_back(op);
    schedulable.insert(op);
  }

  for (auto op : orderedSchedulable) {
    // Data dependencies: op depends on its operand definitions (if in this block)
    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      if (def && schedulable.count(def)) {
        degree[op]++;
        users[def].push_back(op);
      }
    }
    // Memory dependencies
    for (auto pred : memDeps[op]) {
      if (schedulable.count(pred)) {
        degree[op]++;
        users[pred].push_back(op);
      }
    }
  }

  // Compute the critical-path height of each node.
  // height[U] = latency(U) + max( { height[V] for all users V of U } )
  std::unordered_map<Op*, int> height;
  std::function<int(Op*)> computeHeight = [&](Op *op) -> int {
    auto it = height.find(op);
    if (it != height.end())
      return it->second;

    int maxUserHeight = 0;
    auto userIt = users.find(op);
    if (userIt != users.end()) {
      for (auto user : userIt->second) {
        maxUserHeight = std::max(maxUserHeight, computeHeight(user));
      }
    }

    int h = latency(op) + maxUserHeight;
    height[op] = h;
    return h;
  };

  for (auto op : orderedSchedulable) {
    computeHeight(op);
  }
  criticalPathNodes += (int) height.size();
  for (auto &[_, h] : height)
    criticalPathMaxHeight = std::max(criticalPathMaxHeight, h);

  // Initialize ready list with degree-0 ops.
  std::list<Op*> ready;
  for (auto op : orderedSchedulable) {
    if (degree[op] == 0)
      ready.push_back(op);
  }

  // Track when each op was scheduled (for latency-aware goodness).
  std::unordered_map<Op*, int> issueTime;
  int currentTime = 0;
  int origCount = schedulable.size();

  // Compute the original position of each op (to detect reordering).
  std::unordered_map<Op*, int> origPos;
  int pos = 0;
  for (auto op : orderedSchedulable)
    origPos[op] = pos++;

  // Goodness function: higher is better.
  // Priorities:
  //   1. Operations whose operands are still in latency window get penalized
  //      (we want to wait for them)
  //   2. High-latency ops (loads, mul, div) get priority (start them early)
  //   3. Ready-to-issue loads get extra bonus (prefetch early)
  //   4. Stores prefer to be placed when their value is ready
  //   5. Tie-break by original position to keep stable ordering
  auto goodness = [&](Op *op) -> int {
    int score = 0;

    // Penalty: if any operand was scheduled recently, this op stalls.
    bool allOperandsReady = true;
    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      auto it = issueTime.find(def);
      if (it != issueTime.end()) {
        int gap = currentTime - it->second;
        int needed = latency(def);
        if (gap < needed) {
          // Each cycle short of latency costs us
          score -= (needed - gap) * 4;
          allOperandsReady = false;
        }
      }
    }

    // Critical-path height priority.
    score += height[op] * heightWeight;

    // Bonus: starting a high-latency op early benefits the schedule.
    int lat = latency(op);
    if (lat >= 3) score += lat * 2;

    // Extra bonus for loads whose address is ready: issuing them early
    // hides load latency and gives the prefetcher more time.  This is
    // especially important for matrix loops where address computation
    // (addi/slli) completes early but the load result is on the critical
    // path.
    if (isLoad(op) && allOperandsReady)
      score += 3;

    // Prefer earlier original position for stability.
    score -= origPos[op] / 100;

    return score;
  };

  // Output ordering.
  std::vector<Op*> newOrder;
  newOrder.reserve(origCount);

  while (!ready.empty()) {
    // Find best op in ready list.
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

    // Wake up dependents.
    for (auto user : users[op]) {
      if (--degree[user] == 0)
        ready.push_back(user);
    }
  }

  // Sanity check: did we schedule everything?
  if ((int) newOrder.size() != origCount)
    return; // leave the block unchanged if something went wrong

  bool changedOrder = false;
  for (int i = 0; i < origCount; i++) {
    if (orderedSchedulable[i] != newOrder[i]) {
      changedOrder = true;
      break;
    }
  }
  if (!changedOrder)
    return;

  // Apply the new ordering: move ops in newOrder to be right before term.
  // We always apply (even if it happens to match the original) — moveBefore
  // is idempotent for already-positioned ops.

  // Move each scheduled op to be immediately before the terminator,
  // in the order they appear in newOrder. Since moveBefore() inserts before
  // the target, calling it in order builds the sequence.
  for (auto op : newOrder)
    op->moveBefore(term);

  reordered++;
}

void Schedule::run() {
  if (!envEnabled("SISY_RV_ENABLE_SCHEDULE", true))
    return;

  heightWeight = envInt("SISY_RV_SCHEDULE_HEIGHT_WEIGHT", 5, 0, 100);

  for (auto func : collectFuncs()) {
    auto region = func->getRegion();
    for (auto bb : region->getBlocks()) {
      runImpl(bb);
    }
  }
}

}} // namespace sys::rv
