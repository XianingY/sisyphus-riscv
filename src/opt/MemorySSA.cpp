#include "MemorySSA.h"
#include "Analysis.h"
#include <algorithm>
#include <functional>
#include <vector>

using namespace sys;

MemoryAccess *MemorySSA::createAccess(MemoryAccessKind kind, Op *op, BasicBlock *bb) {
  auto access = std::make_unique<MemoryAccess>();
  access->kind = kind;
  access->op = op;
  access->bb = bb;
  auto *ptr = access.get();
  accessPool.push_back(std::move(access));
  if (op) {
    opToAccess[op] = ptr;
  }
  return ptr;
}

void MemorySSA::build() {
  accessPool.clear();
  opToAccess.clear();
  blockMemoryPhi.clear();
  blockExitDefs.clear();

  region->updateDoms();
  DomTree tree;
  for (auto bb : region->getBlocks()) {
    if (auto idom = bb->getIdom())
      tree[idom].push_back(bb);
  }
  
  // Phase 1: Initialize all MemoryAccess nodes in each basic block.
  // We insert a virtual MemoryPhi at the beginning of basic blocks with multiple predecessors.
  for (auto bb : region->getBlocks()) {
    if (bb->preds.size() > 1) {
      auto *phi = createAccess(MemoryAccessKind::Phi, nullptr, bb);
      blockMemoryPhi[bb] = phi;
    }
    
    for (auto op : bb->getOps()) {
      if (isa<StoreOp>(op) || (isa<CallOp>(op) && op->has<ImpureAttr>())) {
        createAccess(MemoryAccessKind::Def, op, bb);
      } else if (isa<LoadOp>(op) || (isa<CallOp>(op) && !op->has<ImpureAttr>())) {
        createAccess(MemoryAccessKind::Use, op, bb);
      }
    }
  }

  // Phase 2: Dual-pass Iterative Dataflow Resolution to wire up definingAccess chains.
  auto getIncomingDef = [&](BasicBlock *bb) -> MemoryAccess* {
    if (blockExitDefs.count(bb)) return blockExitDefs[bb];
    if (blockMemoryPhi.count(bb)) return blockMemoryPhi[bb];
    return nullptr;
  };

  bool changed = true;
  int iterations = 0;
  constexpr int kMaxIterations = 16;
  
  while (changed && iterations++ < kMaxIterations) {
    changed = false;
    for (auto bb : region->getBlocks()) {
      MemoryAccess *activeDef = nullptr;
      
      // If block starts with a Phi, it becomes the initial active definition inside the block.
      if (blockMemoryPhi.count(bb)) {
        activeDef = blockMemoryPhi[bb];
        
        // Accumulate defining defs from predecessor blocks.
        for (auto *pred : bb->preds) {
          auto *predDef = getIncomingDef(pred);
          if (predDef) {
            if (activeDef->definingAccess != predDef) {
              activeDef->definingAccess = predDef;
              changed = true;
            }
            if (std::find(activeDef->phiIncoming.begin(), activeDef->phiIncoming.end(), predDef) == activeDef->phiIncoming.end()) {
              activeDef->phiIncoming.push_back(predDef);
              changed = true;
            }
          }
        }
      }

      for (auto op : bb->getOps()) {
        if (!opToAccess.count(op)) continue;
        auto *access = opToAccess[op];
        if (access->kind == MemoryAccessKind::Use) {
          if (access->definingAccess != activeDef) {
            access->definingAccess = activeDef;
            changed = true;
          }
        } else if (access->kind == MemoryAccessKind::Def) {
          if (access->definingAccess != activeDef) {
            access->definingAccess = activeDef;
            changed = true;
          }
          activeDef = access; // Def updates the active definition
        }
      }
      
      if (blockExitDefs[bb] != activeDef) {
        blockExitDefs[bb] = activeDef;
        changed = true;
      }
    }
  }
}

MemoryAccess *MemorySSA::getAccess(Op *op) {
  return opToAccess.count(op) ? opToAccess[op] : nullptr;
}

MemoryAccess *MemorySSA::getMemoryPhi(BasicBlock *bb) {
  return blockMemoryPhi.count(bb) ? blockMemoryPhi[bb] : nullptr;
}

MemoryAccess *MemorySSA::getReachingDef(MemoryAccess *useAccess) {
  if (!useAccess || useAccess->kind != MemoryAccessKind::Use) return nullptr;

  auto *def = useAccess->definingAccess;
  while (def && def->kind == MemoryAccessKind::Phi) {
    def = def->definingAccess; // Transitively follow defining access through Phi nodes
  }
  return def;
}

std::vector<MemoryAccess*> MemorySSA::getClobberingDefs(Op *memoryOp) {
  std::vector<MemoryAccess*> result;
  auto *access = getAccess(memoryOp);
  if (!access)
    return result;

  Op *addr = nullptr;
  if (isa<LoadOp>(memoryOp))
    addr = memoryOp->DEF();
  else if (isa<StoreOp>(memoryOp))
    addr = memoryOp->DEF(1);
  bool impureMemory = isa<CallOp>(memoryOp) && memoryOp->has<ImpureAttr>();
  if (!addr && !impureMemory)
    return result;

  std::unordered_set<MemoryAccess*> visited;
  std::function<void(MemoryAccess*)> visit = [&](MemoryAccess *cur) {
    if (!cur || !visited.insert(cur).second)
      return;
    if (cur->kind == MemoryAccessKind::Phi) {
      for (auto *incoming : cur->phiIncoming)
        visit(incoming);
      return;
    }
    if (cur->kind != MemoryAccessKind::Def)
      return visit(cur->definingAccess);

    Op *defOp = cur->op;
    bool clobbers = false;
    if (isa<CallOp>(defOp) && defOp->has<ImpureAttr>()) {
      clobbers = true;
    } else if (addr && isa<StoreOp>(defOp)) {
      clobbers = mayAlias(addr, defOp->DEF(1));
    }

    if (clobbers)
      result.push_back(cur);
    else
      visit(cur->definingAccess);
  };
  visit(access->definingAccess);
  return result;
}
