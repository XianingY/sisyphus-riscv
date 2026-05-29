#include "RvPasses.h"
#include "../opt/Analysis.h"

#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

using namespace sys;
using namespace sys::rv;

namespace {

bool envEnabled(const char *n, bool d) {
  const char *r = std::getenv(n);
  if (!r || !r[0]) return d;
  return std::strcmp(r, "0") != 0 && std::strcmp(r, "false") != 0;
}

int envInt(const char *n, int d, int lo, int hi) {
  const char *r = std::getenv(n);
  if (!r || !r[0]) return d;
  char *e = nullptr;
  long v = std::strtol(r, &e, 10);
  if ((e && *e) || v < lo || v > hi) return d;
  return (int) v;
}

bool isLoadOp(Op *op) { return isa<rv::LoadOp>(op) || isa<FldOp>(op); }
bool isStoreOp(Op *op) { return isa<rv::StoreOp>(op) || isa<FsdOp>(op); }

bool isPinnedOp(Op *op) {
  return isa<rv::CallOp>(op) || isa<RetOp>(op) || isa<JOp>(op) ||
         isa<BneOp>(op) || isa<BeqOp>(op) ||
         isa<BltOp>(op) || isa<BgeOp>(op) ||
         isa<BleOp>(op) || isa<BgtOp>(op) ||
         isa<WriteRegOp>(op) || isa<ReadRegOp>(op) ||
         isa<SubSpOp>(op) ||
         isa<PlaceHolderOp>(op);
}

bool addressNeverAlias(Op *a, Op *b) {
  if (!a || !b || a == b) return false;
  auto aa = a->find<AliasAttr>();
  auto bb = b->find<AliasAttr>();
  if (aa && bb && !aa->unknown && !bb->unknown && aa->neverAlias(bb))
    return true;
  return false;
}

int liveOutPressure(BasicBlock *bb) {
  int p = 0;
  for (auto op : bb->getOps()) {
    bool escapes = false;
    for (auto user : op->getUses())
      if (user->getParent() != bb) { escapes = true; break; }
    if (escapes) p++;
  }
  return p;
}

}  // namespace

void sys::rv::SuperblockSchedule::run() {
  if (!envEnabled("SISY_RV_ENABLE_SUPERBLOCK", true))
    return;

  int budget = envInt("SISY_RV_SUPERBLOCK_PRESSURE_BUDGET", 20, 1, 64);

  for (auto func : collectFuncs()) {
    auto region = func->getRegion();
    if (!region) continue;

    std::vector<BasicBlock*> blocks(region->getBlocks().begin(),
                                    region->getBlocks().end());

    for (auto a : blocks) {
      if (a->succs.size() != 1) { rejectedShape++; continue; }
      BasicBlock *b = *a->succs.begin();
      if (b == a || b->preds.size() != 1) { rejectedShape++; continue; }
      if (BlockFrequency::isCold(a) || BlockFrequency::isCold(b)) continue;

      Op *aTerm = a->getLastOp();
      if (!aTerm || !isPinnedOp(aTerm)) { rejectedShape++; continue; }

      std::unordered_set<Op*> definedInB;
      for (auto op : b->getOps())
        definedInB.insert(op);

      int basePressure = liveOutPressure(a);

      for (auto op : std::vector<Op*>(b->getOps().begin(), b->getOps().end())) {
        if (isa<PhiOp>(op)) continue;
        if (isPinnedOp(op) || isStoreOp(op)) break;
        if (!isLoadOp(op)) continue;
        candidates++;

        // All operands must be defined outside B.
        bool depsOK = true;
        for (auto operand : op->getOperands()) {
          if (operand.defining && definedInB.count(operand.defining)) {
            depsOK = false;
            break;
          }
        }
        if (!depsOK) { rejectedDeps++; continue; }

        // No may-alias store/call sits before this load in B (we walk in order
        // and break on the first store/pinned, so anything we see here is OK).

        // Estimate pressure delta: hoisting adds the load's result to A's
        // live-out.  If we exceed the budget, stop.
        if (basePressure + 1 > budget) { rejectedPressure++; continue; }

        // Also conservatively check that no prior may-alias store exists in
        // A between hoist target and terminator: scan A backwards from term.
        bool aliasOK = true;
        for (auto it = a->getOps().rbegin(); it != a->getOps().rend(); ++it) {
          Op *ao = *it;
          if (ao == aTerm) continue;
          if (isStoreOp(ao) && !addressNeverAlias(op, ao)) {
            aliasOK = false;
            break;
          }
          if (isa<rv::CallOp>(ao)) { aliasOK = false; break; }
        }
        if (!aliasOK) { rejectedAlias++; continue; }

        // Move op to be right before A's terminator.
        op->moveBefore(aTerm);
        definedInB.erase(op);
        hoisted++;
        basePressure++;
      }
    }
  }
}
