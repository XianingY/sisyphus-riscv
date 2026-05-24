#include "Passes.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"
#include <iterator>
#include <vector>

using namespace sys;

std::map<std::string, int> Mem2Reg::stats() {
  return {
    { "lowered-alloca", count },
    { "missed-alloca", missed },
  };
}

namespace {

bool hasTerminator(BasicBlock *bb) {
  if (!bb || bb->getOpCount() == 0)
    return false;
  auto *term = bb->getLastOp();
  return term && (isa<ReturnOp>(term) || term->has<TargetAttr>() || term->has<ElseAttr>());
}

bool sanitizeForMem2Reg(Region *region) {
  if (!region)
    return false;
  region->updateDoms();
  auto *entry = region->getFirstBlock();
  bool changed = false;
  bool ok = true;

  for (auto *bb : region->getBlocks()) {
    if (!bb)
      continue;
    if (bb == entry || bb->dominatedBy(entry)) {
      if (!hasTerminator(bb))
        ok = false;
      continue;
    }

    // Unreachable phis are irrelevant to promotion and can corrupt DF walk.
    auto phis = bb->getPhis();
    for (auto *phi : phis) {
      phi->erase();
      changed = true;
    }
  }

  if (changed)
    region->updatePreds();
  return ok;
}

struct AllocaLiveness {
  std::map<BasicBlock*, bool> liveIn;
};

bool isLoadFromAlloca(Op *op, Op *alloca) {
  auto load = dyn_cast<LoadOp>(op);
  return load && load->getOperand().defining == alloca;
}

bool isStoreToAlloca(Op *op, Op *alloca) {
  auto store = dyn_cast<StoreOp>(op);
  return store && store->getOperand(1).defining == alloca;
}

AllocaLiveness computeAllocaLiveness(Region *region, Op *alloca) {
  AllocaLiveness result;
  std::map<BasicBlock*, bool> liveOut;
  std::map<BasicBlock*, bool> useBeforeDef;
  std::map<BasicBlock*, bool> def;
  std::vector<BasicBlock*> blocks;
  std::copy(region->begin(), region->end(), std::back_inserter(blocks));

  for (auto *bb : blocks) {
    bool seenDef = false;
    for (auto *op : bb->getOps()) {
      if (isLoadFromAlloca(op, alloca) && !seenDef)
        useBeforeDef[bb] = true;
      if (isStoreToAlloca(op, alloca)) {
        seenDef = true;
        def[bb] = true;
      }
    }
  }

  bool changed;
  do {
    changed = false;
    for (auto it = blocks.rbegin(); it != blocks.rend(); ++it) {
      auto *bb = *it;
      bool out = false;
      for (auto *succ : bb->succs) {
        if (result.liveIn[succ]) {
          out = true;
          break;
        }
      }

      bool in = useBeforeDef[bb] || (out && !def[bb]);
      if (result.liveIn[bb] != in || liveOut[bb] != out) {
        result.liveIn[bb] = in;
        liveOut[bb] = out;
        changed = true;
      }
    }
  } while (changed);

  return result;
}

Value materializePhiIncomingOnEdge(BasicBlock *pred, BasicBlock *succ, Value value, Builder &builder) {
  auto *def = value.defining;
  if (!def || def->getParent() != succ || !isa<PhiOp>(def))
    return value;

  auto *term = pred->getLastOp();
  if (!term)
    return value;

  // Preserve edge-copy semantics for phi destruction; later folds must not
  // collapse this back into a same-block phi dependency.
  builder.setBeforeOp(term);
  switch (def->getResultType()) {
  case Value::f32: {
    auto *zero = builder.create<FloatOp>({ new FloatAttr(0) });
    return builder.create<AddFOp>({ value, zero }, { new ImpureAttr });
  }
  case Value::i64: {
    auto *zero = builder.create<IntOp>({ new IntAttr(0) });
    return builder.create<AddLOp>({ value, zero }, { new ImpureAttr });
  }
  case Value::i32: {
    auto *zero = builder.create<IntOp>({ new IntAttr(0) });
    return builder.create<AddIOp>({ value, zero }, { new ImpureAttr });
  }
  default:
    return value;
  }
}

}

// See explanation at https://longfangsong.github.io/en/mem2reg-made-simple/
void Mem2Reg::runImpl(FuncOp *func) {
  converted.clear();
  visited.clear();
  phiFrom.clear();
  domtree.clear();

  auto region = func->getRegion();
  if (!sanitizeForMem2Reg(region))
    return;
  region->updateDoms();
  region->updateDomFront();
  domtree = getDomTree(region);
  auto entry = region->getFirstBlock();

  Builder builder;

  // We need to put PhiOp at places where a StoreOp doesn't dominate,
  // because it means at least 2 possible values.
  auto allocas = func->findAll<AllocaOp>();
  for (auto alloca : allocas) {
    bool good = true;

    // If the alloca is used for, as an example, AddOp, then
    // it's an array and can't be promoted to registers.
    for (auto use : alloca->getUses()) {
      // Current SSA renaming walks reachable domtree from entry. If a use
      // lives in an unreachable block, partial promotion can leave stale
      // users behind. Keep such allocas in memory form.
      if (!use->getParent()->dominatedBy(entry)) {
        good = false;
        break;
      }
      if (!isa<LoadOp>(use) && !isa<StoreOp>(use)) {
        good = false;
        break;
      }
      // If the alloca is used as a value in a StoreOp, then it has to be an array.
      if (isa<StoreOp>(use) && use->DEF(0) == alloca) {
        good = false;
        break;
      }
      // Promotion assumes store values are in SSA form and available on all paths
      // reaching the store block. If the incoming value's defining block does not
      // dominate the store block, keeping memory form is safer.
      if (auto store = dyn_cast<StoreOp>(use)) {
        auto valueDef = store->DEF(0);
        if (valueDef && !store->getParent()->dominatedBy(valueDef->getParent())) {
          good = false;
          break;
        }
      }
    }

    if (!good) {
      missed++;
      continue;
    }
    count++;
    converted.insert(alloca);
    auto liveness = computeAllocaLiveness(region, alloca);

    // Now find all blocks where stores reside in. Use set to de-duplicate.
    std::set<BasicBlock*> bbs;
    for (auto use : alloca->getUses()) {
      if (isa<StoreOp>(use))
        bbs.insert(use->getParent());
    }

    std::vector<BasicBlock*> worklist;
    std::copy(bbs.begin(), bbs.end(), std::back_inserter(worklist));

    std::set<BasicBlock*> visited;

    while (!worklist.empty()) {
      auto bb = worklist.back();
      worklist.pop_back();

      for (auto dom : bb->getDominanceFrontier()) {
        if (!liveness.liveIn[dom])
          continue;
        if (visited.count(dom))
          continue;
        visited.insert(dom);

        // Insert a PhiOp at the dominance frontier of each StoreOp, as described above.
        // The PhiOp is broken; we only record which AllocaOp it's from.
        // We'll fill it in later.
        builder.setToBlockStart(dom);
        auto phi = builder.create<PhiOp>();
        phiFrom[phi] = alloca;
        worklist.push_back(dom);
      }
    }
  }

  fillPhi(func->getRegion()->getFirstBlock(), {});

  for (auto alloca : converted) {
    // Be conservative: if any user remains, promotion was partial and erasing
    // the alloca would corrupt use-def chains.
    if (!alloca->getUses().empty())
      continue;
    alloca->erase();
  }
}

void Mem2Reg::fillPhi(BasicBlock *bb, SymbolTable symbols) {
  if (visited.count(bb))
    return;
  visited.insert(bb);

  Builder builder;

  std::vector<Op*> removed;
  for (auto op : bb->getOps()) {
    // Loads are now ordinary reads.
    if (auto load = dyn_cast<LoadOp>(op)) {
      auto alloca = load->getOperand().defining;
      if (!converted.count(alloca))
        continue;
  
      if (!symbols.count(alloca)) {
        builder.setBeforeOp(load);
        bool fp = alloca->has<FPAttr>();
        symbols[alloca] = fp
          ? (Op*) builder.create<FloatOp>({ new FloatAttr(0) })
          : (Op*) builder.create<IntOp>({ new IntAttr(0) });
      }
      
      load->replaceAllUsesWith(symbols[alloca].defining);
      removed.push_back(load);
    }
    
    // Stores are now mutating symbol table.
    if (auto store = dyn_cast<StoreOp>(op)) {
      auto value = store->getOperand(0);
      auto alloca = store->getOperand(1).defining;
      if (!converted.count(alloca))
        continue;
      symbols[alloca] = value;

      removed.push_back(store);
    }

    if (auto phi = dyn_cast<PhiOp>(op)) {
      if (!phiFrom.count(phi))
        continue;
      auto alloca = phiFrom[phi];
      symbols[alloca] = phi;
    }
  }

  for (auto succ : bb->succs) {
    auto phis = succ->getPhis();
    for (auto op : phis) {
      auto phi = cast<PhiOp>(op);
      auto it = phiFrom.find(phi);
      if (it == phiFrom.end())
        continue;
      auto alloca = it->second;

      // We meet a PhiOp. This means the promoted register might hold value `symbols[alloca]` when it reaches here.
      // So this PhiOp should have that value as operand as well.
      Value value;
      
      // It doesn't have an initial value from this path.
      // It's acceptable (for example a variable defined only in a loop)
      // Treat it as zero from this branch.
      if (!symbols.count(alloca)) {
        // Create a zero at the back of the incoming edge.
        auto term = bb->getLastOp();
        builder.setBeforeOp(term);
        bool fp = alloca->has<FPAttr>();
        value = fp
          ? (Value) builder.create<FloatOp>({ new FloatAttr(0) })
          : (Value) builder.create<IntOp>({ new IntAttr(0) });
      } else
        value = symbols[alloca];

      value = materializePhiIncomingOnEdge(bb, succ, value, builder);
      op->pushOperand(value);
      op->add<FromAttr>(bb);
    }
  }

  for (auto x : removed)
    x->erase();
  
  for (auto child : domtree[bb])
    fillPhi(child, symbols);
}

void Mem2Reg::run() {
  auto funcs = collectFuncs();
  for (auto func : funcs)
    runImpl(func);
}
