#include "CleanupPasses.h"
#include "Analysis.h"
#include "../codegen/Attrs.h"
#include <unordered_set>

using namespace sys;

namespace {

bool normalizePhiIncoming(Region *region) {
  bool changed = false;

  for (auto bb : region->getBlocks()) {
    auto phis = bb->getPhis();
    for (auto phi : phis) {
      while (phi->getOperandCount() > (int) phi->getAttrs().size()) {
        phi->removeOperand(phi->getOperandCount() - 1);
        changed = true;
      }
      while ((int) phi->getAttrs().size() > phi->getOperandCount()) {
        phi->removeAttribute((int) phi->getAttrs().size() - 1);
        changed = true;
      }

      std::unordered_set<BasicBlock*> seen;
      for (int i = phi->getOperandCount() - 1; i >= 0; i--) {
        auto attr = phi->getAttrs()[i];
        bool keep = isa<FromAttr>(attr);
        BasicBlock *from = keep ? FROM(attr) : nullptr;
        if (keep) {
          if (!from || from->getParent() != region || !bb->preds.count(from) || seen.count(from))
            keep = false;
        }

        if (!keep) {
          phi->removeOperand(i);
          phi->removeAttribute(i);
          changed = true;
          continue;
        }
        seen.insert(from);
      }
    }
  }

  return changed;
}

}

std::map<std::string, int> DCE::stats() {
  return {
    { "removed-ops", elimOp },
    { "removed-basic-blocks", elimBB },
    { "removed-funcs", elimFn },
  };
}

bool DCE::isImpure(Op *op) {
  if (isa<StoreOp>(op) || isa<VScaleStoreOp>(op) ||
      isa<ReturnOp>(op) || isa<GetArgOp>(op) ||
      isa<BranchOp>(op) || isa<GotoOp>(op) ||
      isa<ProceedOp>(op) || isa<BreakOp>(op) ||
      isa<ContinueOp>(op) || isa<ForOp>(op) ||
      isa<CloneOp>(op) || isa<JoinOp>(op) ||
      isa<WakeOp>(op) || isa<IfOp>(op))
    return true;

  if (isa<CallOp>(op)) {
    auto name = NAME(op);
    if (isExtern(name))
      return true;
    auto it = fnMap.find(name);
    if (it == fnMap.end() || !it->second)
      return true;
    return it->second->has<ImpureAttr>();
  }

  return op->has<ImpureAttr>();
}

bool DCE::markImpure(Region *region) {
  bool impure = false;
  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      bool opImpure = false;
      if (isImpure(op))
        opImpure = true;
      for (auto r : op->getRegions())
        opImpure |= markImpure(r);

      if (opImpure && !op->has<ImpureAttr>()) {
        impure = true;
        op->add<ImpureAttr>();
      }
    }
  }
  return impure;
}

void DCE::runOnRegion(Region *region) {
  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (!op->has<ImpureAttr>() && op->getUses().size() == 0)
        removeable.push_back(op);
      else for (auto r : op->getRegions())
        runOnRegion(r);
    }
  }
}

void DCE::run() {
  auto funcs = collectFuncs();
  fnMap = getFunctionMap();
  
  for (auto func : funcs)
    markImpure(func->getRegion());
  
  // Remove unused variables.
  do {
    removeable.clear();
    for (auto func : funcs)
      runOnRegion(func->getRegion());

    std::vector<Op*> toErase;
    toErase.reserve(removeable.size());
    for (auto op : removeable) {
      // Some ops may become live again via use/def rewrites while we are
      // collecting candidates. Skip them defensively instead of asserting.
      if (!op->getUses().empty())
        continue;
      op->removeAllOperands();
      toErase.push_back(op);
    }

    elimOp += toErase.size();
    // Erase users before defs when possible to keep use-lists consistent.
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it)
      (*it)->erase();
  } while (removeable.size());

  // Remove unused phi's. 
  // They might be cyclically referencing each other, but not used elsewhere.
  std::vector<Op*> unused;
  runRewriter([&](PhiOp *op) {
    if (!op->getOperandCount())
      return false;
    
    std::vector<Op*> worklist { op };
    std::set<Op*> visited;

    while (!worklist.empty()) {
      auto *op = worklist.back();
      worklist.pop_back();
      visited.insert(op);

      for (auto *use : op->getUses()) {
        if (visited.count(use))
          continue;
        
        if (isa<PhiOp>(use))
          worklist.push_back(use);
        else
          return false; 
      }
    }

    // Here all phi's in worklist are dead.
    for (auto phi : visited) {
      phi->removeAllOperands();
      unused.push_back(phi);
    }
    return true;
  });

  for (auto dead : unused)
    dead->erase();

  // Remove unused functions.
  bool changed;
  do {
    CallGraph(module).run();

    changed = false;
    for (auto func : funcs) {
      const auto &name = NAME(func);
      // main() might not be used, but it must be preserved.
      if (name == "main")
        continue;

      // If a function is only used by itself, then it's also unused.
      const auto &callers = CALLER(func);
      if (!callers.size() || (callers.size() == 1 && callers[0] == name)) {
        func->erase();
        changed = true;
        elimFn++;
      }
    }

    if (changed)
      funcs = collectFuncs();
  } while (changed);

  if (!elimBlocks)
    return;
  
  do {
    changed = false;
    
    for (auto func : funcs) {
      auto region = func->getRegion();
      region->updatePreds();
      if (normalizePhiIncoming(region)) {
        changed = true;
        region->updatePreds();
      }
      std::set<BasicBlock*> toRemove;
      auto entry = region->getFirstBlock();

      std::set<BasicBlock*> reachable;
      std::vector<BasicBlock*> queue { entry };
      while (!queue.empty()) {
        auto bb = queue.back();
        queue.pop_back();
        
        if (reachable.count(bb))
          continue;
        reachable.insert(bb);
        for (auto succ : bb->succs)
          queue.push_back(succ);
      }

      for (auto bb : region->getBlocks()) {
        if (!reachable.count(bb))
          toRemove.insert(bb);
      }

      elimBB += toRemove.size();
      if (toRemove.size())
        changed = true;

      // Remove all operands first, to avoid inter-dependency between blocks.
      for (auto bb : toRemove) {
        for (auto op : bb->getOps())
          op->removeAllOperands();
      }

      // Remove phi nodes that take these dead blocks as input.
      for (auto bb : region->getBlocks()) {
        auto phis = bb->getPhis();
        for (auto phi : phis) {
          auto ops = phi->getOperands();
          std::vector<Attr*> attrs;
          for (auto attr : phi->getAttrs())
            attrs.push_back(attr->clone());

          phi->removeAllOperands();
          // This deletes attributes if their refcnt goes to zero.
          // That's why we cloned above.
          phi->removeAllAttributes();

          for (size_t i = 0; i < ops.size(); i++) {
            auto from = FROM(attrs[i]);
            if (toRemove.count(ops[i].defining->getParent()) || toRemove.count(from))
              continue;

            // Only preserve the operands that aren't from dead blocks.
            phi->pushOperand(ops[i]);
            phi->add<FromAttr>(from);
          }

          // The added attributes are also copies.
          // No need to preserve the ones in this vector.
          for (auto attr : attrs)
            delete attr;
        }
      }

      // Do the real removal.
      for (auto bb : toRemove)
        bb->forceErase();

      region->updatePreds();
      if (normalizePhiIncoming(region)) {
        changed = true;
        region->updatePreds();
      }
    }
  } while (changed);
}
