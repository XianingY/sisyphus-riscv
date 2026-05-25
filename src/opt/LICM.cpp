#include "LoopPasses.h"
#include "MemorySSA.h"

#include <optional>

using namespace sys;

std::map<std::string, int> LICM::stats() {
  return {
    { "hoisted", hoisted }
  };
}

namespace {

// Pinned operations cannot move.
// Note that it's slightly different from GCM.cpp in that Load is not pinned.
#define PINNED(Ty) || isa<Ty>(op)
bool pinned(Op *op) {
  return (isa<CallOp>(op) && op->has<ImpureAttr>())
    PINNED(ReturnOp)
    PINNED(BranchOp)
    PINNED(GotoOp)
    PINNED(AllocaOp);
}


bool noAlias(Op *load, const std::vector<Op*> &stores) {
  auto addr = load->DEF();
  // Sometimes we don't have stores while unable to analyze loads.
  // It's safe to say no alias anyway.
  if (!stores.size())
    return true;

  while (isa<PhiOp>(addr)) {
    // Conservatively assume alias.
    if (addr->getOperandCount() >= 2)
      return false;

    addr = addr->DEF();
  }

  for (auto store : stores) {
    if (mayAlias(addr, store))
      return false;
  }
  return true;
}

bool noAliasWithMemorySSA(Op *load, LoopInfo *loop, MemorySSA *mssa, const std::vector<Op*> &stores, bool impure) {
  (void)impure;
  if (!mssa)
    return noAlias(load, stores);

  auto clobbers = mssa->getClobberingDefs(load);
  if (clobbers.empty())
    return noAlias(load, stores);

  auto addr = load->DEF();
  for (auto *access : clobbers) {
    auto *def = access->op;
    if (!def || !loop->contains(def->getParent()))
      continue;
    if (isa<StoreOp>(def) && addr && !mayAlias(addr, def->DEF(1)))
      continue;
    if (def && loop->contains(def->getParent()))
      return false;
  }

  return noAlias(load, stores);
}

bool knownNonNegative(Op *op, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);

  if (isa<IntOp>(op))
    return V(op) >= 0;

  if (auto range = op->find<RangeAttr>())
    return range->range.first >= 0;

  if (isa<AddIOp>(op) || isa<AddLOp>(op))
    return knownNonNegative(op->DEF(0), seen) && knownNonNegative(op->DEF(1), seen);

  if (isa<ModIOp>(op) && op->getOperandCount() == 2) {
    if (auto divisor = dyn_cast<IntOp>(op->DEF(1)); divisor && V(divisor) > 0)
      return knownNonNegative(op->DEF(0), seen);
  }

  if (isa<PhiOp>(op)) {
    bool hasNonSelf = false;
    for (auto operand : op->getOperands()) {
      auto incoming = operand.defining;
      if (incoming == op)
        continue;
      if (auto add = dyn_cast<AddIOp>(incoming)) {
        if ((add->DEF(0) == op && knownNonNegative(add->DEF(1), seen)) ||
            (add->DEF(1) == op && knownNonNegative(add->DEF(0), seen)))
          continue;
      }
      if (auto add = dyn_cast<AddLOp>(incoming)) {
        if ((add->DEF(0) == op && knownNonNegative(add->DEF(1), seen)) ||
            (add->DEF(1) == op && knownNonNegative(add->DEF(0), seen)))
          continue;
      }
      hasNonSelf = true;
      if (!knownNonNegative(incoming, seen))
        return false;
    }
    return hasNonSelf;
  }

  return false;
}

bool knownNonNegative(Op *op) {
  std::set<Op*> seen;
  return knownNonNegative(op, seen);
}

std::optional<long long> evalAtLoopEntry(Op *op, LoopInfo *info, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return std::nullopt;
  seen.insert(op);

  if (isa<IntOp>(op))
    return V(op);

  if (isa<PhiOp>(op) && op->getParent() == info->header && info->preheader)
    return evalAtLoopEntry(Op::getPhiFrom(op, info->preheader), info, seen);

  if (isa<MinusOp>(op)) {
    auto value = evalAtLoopEntry(op->DEF(), info, seen);
    if (value)
      return -*value;
    return std::nullopt;
  }

  if (op->getOperandCount() != 2)
    return std::nullopt;

  auto lhs = evalAtLoopEntry(op->DEF(0), info, seen);
  auto rhs = evalAtLoopEntry(op->DEF(1), info, seen);
  if (!lhs || !rhs)
    return std::nullopt;

  if (isa<AddIOp>(op) || isa<AddLOp>(op))
    return *lhs + *rhs;
  if (isa<SubIOp>(op) || isa<SubLOp>(op))
    return *lhs - *rhs;
  if (isa<MulIOp>(op) || isa<MulLOp>(op))
    return *lhs * *rhs;
  return std::nullopt;
}

std::optional<bool> evalCondAtLoopEntry(Op *op, LoopInfo *info) {
  if (!op || op->getOperandCount() != 2)
    return std::nullopt;
  std::set<Op*> seenLhs;
  std::set<Op*> seenRhs;
  auto lhs = evalAtLoopEntry(op->DEF(0), info, seenLhs);
  auto rhs = evalAtLoopEntry(op->DEF(1), info, seenRhs);
  if (!lhs || !rhs)
    return std::nullopt;

  if (isa<LtOp>(op))
    return *lhs < *rhs;
  if (isa<LeOp>(op))
    return *lhs <= *rhs;
  if (isa<EqOp>(op))
    return *lhs == *rhs;
  if (isa<NeOp>(op))
    return *lhs != *rhs;
  return std::nullopt;
}

bool branchEdgeDominates(BasicBlock *edge, BasicBlock *bb) {
  return edge && bb && (edge == bb || edge->dominates(bb));
}

bool dominatingBranchProvesPositive(Op *value, BasicBlock *useBlock) {
  if (!value || !useBlock)
    return false;

  auto region = useBlock->getParent();
  for (auto bb : region->getBlocks()) {
    auto branch = dyn_cast<BranchOp>(bb->getLastOp());
    if (!branch || branch->getOperandCount() != 1 || !branch->has<TargetAttr>() ||
        !branch->has<ElseAttr>())
      continue;

    auto cond = branch->DEF(0);
    if (!cond || cond->getOperandCount() != 2)
      continue;

    auto lhs = cond->DEF(0);
    auto rhs = cond->DEF(1);

    // On a dominated true edge of `x < value`, any non-negative x proves
    // value > 0.  This is common in nests such as:
    //   while (i < n) { ... while (k < n) load(A[i][i]) ... }
    // and lets LICM speculate loop-invariant loads from the inner loop without
    // assuming anything about test-case dimensions.
    if (isa<LtOp>(cond) && rhs == value &&
        branchEdgeDominates(TARGET(branch), useBlock) && knownNonNegative(lhs))
      return true;

    // Symmetric false edge: !(value < c) gives value >= c.
    if (isa<LtOp>(cond) && lhs == value && isa<IntOp>(rhs) && V(rhs) > 0 &&
        branchEdgeDominates(ELSE(branch), useBlock))
      return true;

    if (isa<LeOp>(cond) && lhs == value && isa<IntOp>(rhs) && V(rhs) >= 0 &&
        branchEdgeDominates(ELSE(branch), useBlock))
      return true;
  }
  return false;
}

bool loopKnownToRunOnce(LoopInfo *info) {
  if (!info || !info->header || !info->preheader)
    return false;

  if (info->start && info->stop && info->step &&
      isa<IntOp>(info->start) && isa<IntOp>(info->stop) && isa<IntOp>(info->step)) {
    int step = V(info->step);
    if (step > 0 && V(info->start) < V(info->stop))
      return true;
    if (step < 0 && V(info->start) > V(info->stop))
      return true;
  }

  if (info->start && info->stop && info->step &&
      isa<IntOp>(info->start) && V(info->start) == 0 &&
      isa<IntOp>(info->step) && V(info->step) > 0 &&
      dominatingBranchProvesPositive(info->stop, info->preheader))
    return true;

  auto branch = dyn_cast<BranchOp>(info->header->getLastOp());
  if (!branch || branch->getOperandCount() != 1 || !branch->has<TargetAttr>() ||
      !branch->has<ElseAttr>())
    return false;
  auto cond = evalCondAtLoopEntry(branch->DEF(0), info);
  if (!cond)
    return false;
  auto taken = *cond ? TARGET(branch) : ELSE(branch);
  return info->contains(taken);
}

bool canSpeculateLoadFromBlock(LoopInfo *info, BasicBlock *bb, bool speculativeLoads) {
  if (speculativeLoads)
    return true;

  // Non-rotated loops may skip the body entirely.  Hoisting a load from such a
  // body is only safe when the canonical loop is known to execute once and has
  // no side exits that could bypass the load in the first iteration.
  if (!loopKnownToRunOnce(info))
    return false;
  if (info->exitings.size() != 1 || *info->exitings.begin() != info->header)
    return false;
  if (bb == info->header)
    return true;
  for (auto latch : info->latches)
    if (!bb->dominates(latch))
      return false;
  return true;
}

// Traces `op` up back, and determines if everything between `info` and `outer` are invariant.
bool variant(Op *op, LoopInfo *info, LoopInfo *outer, std::set<Op*> &visited) {
  if (visited.count(op))
    return false; // Cycle assumes invariant unless proven otherwise
  visited.insert(op);

  // These will be marked as variant, but they don't affect anything.
  if (isa<BranchOp>(op) || isa<GotoOp>(op) || isa<ReturnOp>(op) || isa<AllocaOp>(op))
    return false;

  if (op->has<VariantAttr>())
    return true;

  if (isa<PhiOp>(op)) {
    auto parent = op->getParent();
    // For header phis, the only concern would be preheader.
    if (parent == info->header)
      return variant(Op::getPhiFrom(op, info->preheader), info, outer, visited);
  }

  for (auto operand : op->getOperands()) {
    auto def = operand.defining;
    auto parent = def->getParent();
    if (parent->dominates(outer->header) && parent != outer->header)
      continue;
    // Values defined between `outer` and `info` are part of the outer loop
    // but outside the inner loop — skip them for subloop hoisting decisions.
    if (parent->dominates(info->header) && parent != info->header)
      continue;
    if (variant(def, info, outer, visited))
      return true;
  }
  return false;
}

}

// This also hoists ops besides giving variant.
void LICM::hoistVariant(LoopInfo *info, BasicBlock *bb, bool hoistable, bool speculativeLoads) {
  std::vector<Op*> invariant;

  for (auto op : bb->getOps()) {
    if (isa<LoadOp>(op) || isa<BranchOp>(op))
      hoistable = false;

    if (op->has<VariantAttr>())
      continue;

    if (pinned(op) || isa<PhiOp>(op) ||
        (isa<LoadOp>(op) && (!canSpeculateLoadFromBlock(info, bb, speculativeLoads) ||
                             !noAliasWithMemorySSA(op, info, this->mssa, stores, impure) || impure))
        // When a store only writes a loop-invariant value to loop-invariant address,
        // and it doesn't follow any load, then it's safe to hoist it out.
        || (isa<StoreOp>(op) && (!hoistable || impure || op->DEF(0)->has<VariantAttr>() || op->DEF(1)->has<VariantAttr>())))
      op->add<VariantAttr>();
    else for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      if (def->has<VariantAttr>()) {
        op->add<VariantAttr>();
        break;
      }
    }

    if (!op->has<VariantAttr>())
      invariant.push_back(op);
  }

  // Now hoist everything to preheader.
  hoisted += invariant.size();
  auto term = info->preheader->getLastOp();
  for (auto op : invariant)
    op->moveBefore(term);

  for (auto child : domtree[bb]) {
    if (info->contains(child))
      hoistVariant(info, child, hoistable, speculativeLoads);
  }
}

void LICM::markVariant(LoopInfo *info, BasicBlock *bb, bool hoistable) {
  for (auto op : bb->getOps()) {
    if (isa<LoadOp>(op) || isa<BranchOp>(op))
      hoistable = false;

    if (op->has<VariantAttr>())
      continue;

    if (pinned(op) || (isa<LoadOp>(op) && (!noAliasWithMemorySSA(op, info, this->mssa, stores, impure) || impure))
        || (isa<StoreOp>(op) && (!hoistable || impure || op->DEF(0)->has<VariantAttr>() || op->DEF(1)->has<VariantAttr>())))
      op->add<VariantAttr>();
    else for (auto operand : op->getOperands()) {
      auto def = operand.defining;
      if (def->has<VariantAttr>()) {
        op->add<VariantAttr>();
        break;
      }
    }
  }

  for (auto child : domtree[bb]) {
    if (info->contains(child))
      markVariant(info, child, hoistable);
  }
}

bool LICM::updateStores(LoopInfo *info, bool *storeHoistable) {
  auto preheader = info->preheader;
  if (!preheader)
    return false;

  bool rotated = true;
  for (auto latch : info->latches) {
    auto term = latch->getLastOp();
    if (!isa<BranchOp>(term))
      rotated = false;
  }
  if (storeHoistable)
    *storeHoistable = rotated;

  // Record all stores in the loop.
  stores.clear();
  impure = false;
  for (auto bb : info->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<StoreOp>(op))
        stores.push_back(op->DEF(1));
      if (isa<CallOp>(op) && op->has<ImpureAttr>())
        impure = true;
    }
  }
  return true;
}

void LICM::runImpl(LoopInfo *info) {
  // Hoist internal loops first;
  // otherwise we risk hoisting inner-loop's variants out.
  for (auto subloop : info->subloops)
    runImpl(subloop);

  bool storeHoistable = false;
  if (!updateStores(info, &storeHoistable))
    return;

  // Mark invariants inside the loop, and try hoisting it out.
  // We must traverse through domtree to preserve def-use chain.
  auto header = info->header;
  hoistVariant(info, header, storeHoistable, storeHoistable);
}

bool LICM::hoistSubloop(LoopInfo *outer) {
  bool storeHoistable = false;
  if (!updateStores(outer, &storeHoistable) || !storeHoistable)
    return false;

  for (auto bb : outer->getBlocks()) {
    // All single-operand phis that refer to things outside this loop can be folded.
    // This won't break LCSSA.
    auto ops = bb->getOps();
    for (auto op : ops) {
      if (!isa<PhiOp>(op) || op->getOperandCount() != 1)
        continue;

      auto def = op->DEF();
      if (def->getParent()->dominates(outer->preheader)) {
        op->replaceAllUsesWith(def);
        op->erase();
      }
    }
  }

  auto phis = outer->header->getPhis();
  for (auto phi : phis)
    phi->add<VariantAttr>();

  markVariant(outer, outer->header, /*hoistable=*/true);

  for (auto loop : outer->subloops) {
    if (loop->exits.size() > 1 || loop->latches.size() > 1)
      continue;

    // Check for rotated.
    auto latch = loop->getLatch();
    if (!isa<BranchOp>(latch->getLastOp()))
      continue;

    bool good = true;
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        std::set<Op*> visited;
        if (variant(op, loop, outer, visited)) {
          good = false;
          break;
        }
      }
    }
    if (!good)
      continue;

    Builder builder;

    // Redirect preheader to the hoisted loop's header.
    auto preheader = outer->preheader;
    auto region = preheader->getParent();
    auto prterm = preheader->getLastOp();
    builder.replace<GotoOp>(prterm, { new TargetAttr(loop->header) });

    // The previous preheader should now get to exit.
    auto exit = loop->getExit();
    prterm = loop->preheader->getLastOp();
    builder.replace<GotoOp>(prterm, { new TargetAttr(exit) });
    
    // Hoist the loop out.
    // Move every block before outer's header.
    for (auto bb : loop->getBlocks())
      bb->moveBefore(outer->header);

    // Create outer loop's new preheader as loop exit.
    auto newexit = region->insert(outer->header);
    auto latchterm = latch->getLastOp();

    if (TARGET(latchterm) == exit)
      TARGET(latchterm) = newexit;
    if (ELSE(latchterm) == exit)
      ELSE(latchterm) = newexit;

    // The phi's at original exit should be moved to the new exit.
    auto phis = exit->getPhis();
    for (auto phi : phis)
      phi->moveToStart(newexit);

    // New exit should go to outer loop header.
    builder.setToBlockEnd(newexit);
    builder.create<GotoOp>({ new TargetAttr(outer->header) });

    // For both header and exit, rewire preheader to outer's old preheader.
    auto rewire = loop->header->getPhis();
    rewire.reserve(rewire.size() + phis.size());
    std::copy(phis.begin(), phis.end(), std::back_inserter(rewire));

    for (auto phi : rewire) {
      phi->remove<VariantAttr>();
      for (auto attr : phi->getAttrs()) {
        if (FROM(attr) == loop->preheader) {
          FROM(attr) = outer->preheader;
          break;
        }
      }
    }

    // For outer loop's header, it's preheader is now from newexit.
    rewire = outer->header->getPhis();
    for (auto phi : rewire) {
      phi->remove<VariantAttr>();
      for (auto attr : phi->getAttrs()) {
        if (FROM(attr) == outer->preheader) {
          FROM(attr) = newexit;
          break;
        }
      }
    }
    return true;
  }
  return false;
}

void LICM::run() {
  LoopAnalysis loop(module);
  loop.run();
  auto forests = loop.getResult();

  auto funcs = collectFuncs();
  
  for (auto func : funcs) {
    auto region = func->getRegion();
    domtree = getDomTree(region);

    MemorySSA mssaInstance(region);
    mssaInstance.build();
    this->mssa = &mssaInstance;

    const auto &forest = forests[func];
    for (auto info : forest.getLoops()) {
      // Only call for top-level loops.
      if (!info->getParent())
        runImpl(info);
    }

    // Remove VariantAttr's attached.
    // That's necessary because phi's cannot have attrs other than FromAttr.
    for (auto bb : region->getBlocks()) {
      for (auto op : bb->getOps())
        op->remove<VariantAttr>();
    }
  }

  for (auto func : funcs) {
    auto region = func->getRegion();
    domtree = getDomTree(region);

    MemorySSA mssaInstance(region);
    mssaInstance.build();
    this->mssa = &mssaInstance;

    auto forest = forests[func];
    bool changed;
    int loopCount = 0;
    do {
      changed = false;
      for (auto info : forest.getLoops()) {
        // Only call for top-level loops.
        if (!info->getParent() && hoistSubloop(info)) {
          forest = loop.runImpl(region);
          domtree = getDomTree(region);
          mssaInstance.build();
          changed = true;
          break;
        }
      }

      for (auto bb : region->getBlocks()) {
        for (auto op : bb->getOps())
          op->remove<VariantAttr>();
      }

      if (loopCount++ > 100) {
        std::cerr << "LICM: Loop threshold exceeded (100) in subloop hoisting in " << func->getName() << ". Breaking to prevent infinite loop!" << std::endl;
        break;
      }
    } while (changed);
  }
}
