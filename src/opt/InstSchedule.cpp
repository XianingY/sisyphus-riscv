#include "LowerPasses.h"
#include "Analysis.h"
#include <climits>
#include <cstdlib>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace sys;

namespace {

int instScheduleMaxOps() {
  const char *raw = std::getenv("SISY_INST_SCHEDULE_MAX_OPS");
  if (!raw || !raw[0])
    return 256;
  char *end = nullptr;
  long parsed = std::strtol(raw, &end, 10);
  if (!end || *end || parsed < 0 || parsed > 100000)
    return 256;
  return (int) parsed;
}

bool validateLocalSchedule(
    BasicBlock *bb,
    const std::unordered_set<Op*> &scheduled,
    const std::unordered_map<Op*, std::vector<Op*>> &memDeps) {
  if (!bb)
    return false;

  std::unordered_map<Op*, int> position;
  int index = 0;
  for (auto op : bb->getOps())
    position[op] = index++;

  for (auto op : scheduled) {
    auto opIt = position.find(op);
    if (opIt == position.end())
      return false;
    int opPos = opIt->second;

    for (auto operand : op->getOperands()) {
      Op *def = operand.defining;
      if (!def || def == op)
        return false;
      if (def->getParent() != bb || isa<PhiOp>(def))
        continue;
      auto defIt = position.find(def);
      if (defIt == position.end())
        return false;
      if (defIt->second >= opPos)
        return false;
    }

    auto depsIt = memDeps.find(op);
    if (depsIt == memDeps.end())
      continue;
    for (auto dep : depsIt->second) {
      if (!dep || dep->getParent() != bb)
        continue;
      auto depIt = position.find(dep);
      if (depIt == position.end())
        return false;
      if (depIt->second >= opPos)
        return false;
    }
  }

  return true;
}

void restoreLocalOrder(const std::vector<Op*> &order, Op *term) {
  for (auto op : order)
    op->moveBefore(term);
}

}

void InstSchedule::runImpl(BasicBlock *bb) {
  if (!bb || bb->getOpCount() == 0)
    return;

  // If there are local arrays, then there are probably initialization.
  // We'd like to keep those stores in order to achieve better cache performance.
  if (isa<AllocaOp>(bb->getFirstOp()))
    return;
  
  for (auto op : bb->getOps()) {
    // We can't reschedule if there is any pinned operations.
    // TODO: Perhaps there's some way to mitigate this?
    if (isa<CallOp>(op) && op->has<ImpureAttr>() || isa<CloneOp>(op) || isa<JoinOp>(op) || isa<WakeOp>(op))
      return;
  }

  auto term = bb->getLastOp();

  std::vector<Op*> allowedOrder;
  std::unordered_set<Op*> allowed;
  for (auto op : bb->getOps()) {
    if (op == term || isa<PhiOp>(op))
      continue;
    allowedOrder.push_back(op);
    allowed.insert(op);
  }
  if (allowedOrder.empty())
    return;
  int maxOps = instScheduleMaxOps();
  if (maxOps > 0 && (int) allowedOrder.size() > maxOps)
    return;

  // First, we need to build a dependence graph between loads/stores.
  // Keep memory dependencies in a side table. Mutating load/store operands for
  // scheduling is fragile: if the scheduler cannot order a block, the IR is
  // left with artificial operands and later passes see malformed SSA.
  std::vector<Op*> stores, loads;
  std::unordered_map<Op*, std::vector<Op*>> memDeps;
  for (auto op : allowedOrder) {
    // Check against store, but no need to check loads.
    if (isa<LoadOp>(op)) {
      for (auto store : stores) {
        if (mayAlias(op->DEF(), store->DEF(1)))
          memDeps[op].push_back(store);
      }

      loads.push_back(op);
      continue;
    }

    // Check both stores and loads.
    if (isa<StoreOp>(op)) {
      for (auto store : stores) {
        if (mayAlias(op->DEF(1), store->DEF(1)))
          memDeps[op].push_back(store);
      }

      for (auto load : loads) {
        if (mayAlias(op->DEF(1), load->DEF()))
          memDeps[op].push_back(load);
      }

      stores.push_back(op);
    }
  }

  // Now do a list scheduling.

  // The amount of ops that this one is waiting for.
  std::unordered_map<Op*, int> degree;
  std::unordered_map<Op*, std::vector<Op*>> users;
  std::unordered_map<Op*, int> time;
  int index = 0;

  auto phis = bb->getPhis();
  std::unordered_set<Op*> operands;
  for (const auto phi : phis) {
    for (auto operand : phi->getOperands())
      operands.insert(operand.defining);
  }

  auto goodness = [&](Op *op) -> int {
    int result = 0;

    // Delay constants whenever possible.
    if (isa<IntOp>(op) || isa<FloatOp>(op) || isa<GetGlobalOp>(op))
      return -3000;

    // If this is a phi's operand of this block, then delay it as far as possible.
    if (operands.count(op))
      return -5000;

    // The numbers are somewhat arbitrary; we don't have a good understanding of BOOM.
    for (int i = 0; i < op->getOperandCount(); i++) {
      auto def = op->DEF(i);

      // Wait 2 instructions for load.
      auto defTime = time.find(def);
      if (defTime != time.end() && isa<LoadOp>(def) && index - defTime->second <= 2) {
        result--;
      }

      // Keep a small gap after integer multiplies when another ready op can
      // fill it. This is a scheduling heuristic only; dependencies still
      // control correctness.
      if (defTime != time.end() && (isa<MulIOp>(def) || isa<MulLOp>(def)) &&
          index - defTime->second <= 3) {
        result--;
      }
    }

    if (result < 0)
      return result;

    // Emit loads early.
    if (isa<LoadOp>(op))
      result += 8;
    
    // If the instruction is too far from its operands, then emit it early.
    // This is to lower the amount of live-range conflicts.
    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      // If the operand is live-in, then no need.
      auto defTime = time.find(def);
      if (defTime != time.end())
        result = std::max(result, (index - defTime->second) / 3);
    }
    return result;
  };
  // The priority queue pops the LARGEST element by default.
  std::list<Op*> ready;

  for (auto op : allowedOrder) {
    std::unordered_set<Op*> preds;
    for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      if (def && allowed.count(def))
        preds.insert(def);
    }
    for (auto pred : memDeps[op]) {
      if (pred && allowed.count(pred))
        preds.insert(pred);
    }
    degree[op] = (int) preds.size();
    for (auto pred : preds)
      users[pred].push_back(op);
    if (!degree[op])
      ready.push_back(op);
  }

  std::vector<Op*> newOrder;
  newOrder.reserve(allowedOrder.size());
  while (!ready.empty()) {
    // Find the best element.
    decltype(ready)::iterator it;
    int best = INT_MIN;
    for (auto i = ready.begin(); i != ready.end(); i++) {
      auto good = goodness(*i);
      if (good > best) {
        it = i;
        best = good;
      }
    }
    Op *op = *it;
    ready.erase(it);

    newOrder.push_back(op);
    time[op] = index++;

    for (auto use : users[op]) {
      if (--degree[use] == 0)
        ready.push_back(use);
    }
  }

  if (newOrder.size() != allowedOrder.size())
    return;

  bool changed = false;
  for (size_t i = 0; i < newOrder.size(); i++) {
    if (newOrder[i] != allowedOrder[i]) {
      changed = true;
      break;
    }
  }
  if (!changed)
    return;

  std::unordered_set<Op*> scheduled(newOrder.begin(), newOrder.end());
  for (auto op : newOrder)
    op->moveBefore(term);

  if (!validateLocalSchedule(bb, scheduled, memDeps)) {
    restoreLocalOrder(allowedOrder, term);
    return;
  }
}

void InstSchedule::run() {
  Alias(module).run();

  auto funcs = collectFuncs();
  for (auto func : funcs) {
    auto region = func->getRegion();
    region->updateLiveness();

    for (auto bb : region->getBlocks())
      runImpl(bb);
  }
}
