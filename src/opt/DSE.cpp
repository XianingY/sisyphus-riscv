#include "CleanupPasses.h"
#include "Analysis.h"
#include <algorithm>
#include <deque>
#include <unordered_set>

using namespace sys;

std::map<std::string, int> DSE::stats() {
  return {
    { "removed-stores", elim },
    { "sunk-stores", sunk },
  };
}

namespace {

bool sameStoredValueForSink(Op *a, Op *b) {
  if (a == b)
    return true;
  if (!a || !b || a->opid != b->opid)
    return false;
  if (isa<IntOp>(a))
    return V(a) == V(b);
  if (isa<FloatOp>(a))
    return F(a) == F(b);
  return false;
}

bool sameStoreAddressForSink(StoreOp *a, StoreOp *b) {
  if (!a || !b)
    return false;
  if (a->has<SizeAttr>() && b->has<SizeAttr>() && SIZE(a) != SIZE(b))
    return false;
  return a->DEF(1) == b->DEF(1) || mustAlias(a->DEF(1), b->DEF(1));
}

StoreOp *trailingStoreToJoin(BasicBlock *pred, BasicBlock *join) {
  if (!pred || pred->getOpCount() == 0)
    return nullptr;
  auto term = pred->getLastOp();
  if (!isa<GotoOp>(term) || !term->has<TargetAttr>() || TARGET(term) != join)
    return nullptr;
  auto prev = term->prevOp();
  if (!prev)
    return nullptr;
  return dyn_cast<StoreOp>(prev);
}

bool valueAvailableAt(Op *value, BasicBlock *bb) {
  if (!value)
    return false;
  if (isa<IntOp>(value) || isa<FloatOp>(value))
    return true;
  auto parent = value->getParent();
  return parent && parent->dominates(bb);
}

bool valueAvailableOnEdge(Op *value, BasicBlock *pred) {
  if (!value || !pred)
    return false;
  if (isa<IntOp>(value) || isa<FloatOp>(value))
    return true;
  auto parent = value->getParent();
  return parent && parent->dominates(pred);
}

Op *materializeSinkValue(Builder &builder, Op *value) {
  if (isa<IntOp>(value))
    return builder.create<IntOp>({ new IntAttr(V(value)) });
  if (isa<FloatOp>(value))
    return builder.create<FloatOp>({ new FloatAttr(F(value)) });
  return value;
}

void setAfterPhis(Builder &builder, BasicBlock *bb) {
  auto phis = bb->getPhis();
  if (phis.empty())
    builder.setToBlockStart(bb);
  else
    builder.setAfterOp(phis.back());
}

Op *materializeStoreSinkValue(Builder &builder, BasicBlock *bb,
                              const std::vector<StoreOp*> &stores) {
  auto firstValue = stores.front()->DEF(0);
  bool sameValue = true;
  for (auto store : stores) {
    if (!sameStoredValueForSink(firstValue, store->DEF(0))) {
      sameValue = false;
      break;
    }
  }

  if (sameValue) {
    if (!valueAvailableAt(firstValue, bb))
      return nullptr;
    return materializeSinkValue(builder, firstValue);
  }

  std::vector<Value> values;
  std::vector<Attr*> from;
  values.reserve(stores.size());
  from.reserve(stores.size());

  for (auto pred : bb->preds) {
    StoreOp *storeForPred = nullptr;
    for (auto store : stores) {
      auto term = store->nextOp();
      if (term && isa<GotoOp>(term) && term->has<TargetAttr>() &&
          TARGET(term) == bb && store->getParent() == pred) {
        storeForPred = store;
        break;
      }
    }
    if (!storeForPred)
      return nullptr;
    auto value = storeForPred->DEF(0);
    if (value->getResultType() != firstValue->getResultType())
      return nullptr;
    if (!valueAvailableOnEdge(value, pred))
      return nullptr;
    values.push_back(value);
    from.push_back(new FromAttr(pred));
  }

  return builder.create<PhiOp>(values, from);
}

}

void DSE::runImpl(Region *region) {
  region->updateDoms();

  Builder builder;
  for (auto bb : region->getBlocks()) {
    if (!bb || bb->preds.size() < 2)
      continue;

    std::vector<StoreOp*> stores;
    bool good = true;
    for (auto pred : bb->preds) {
      auto store = trailingStoreToJoin(pred, bb);
      if (!store) {
        good = false;
        break;
      }
      stores.push_back(store);
    }
    if (!good || stores.empty())
      continue;

    auto first = stores.front();
    if (!valueAvailableAt(first->DEF(1), bb))
      continue;

    bool sameAddress = true;
    for (auto store : stores) {
      if (!sameStoreAddressForSink(first, store)) {
        sameAddress = false;
        break;
      }
    }
    if (!sameAddress)
      continue;

    setAfterPhis(builder, bb);
    auto value = materializeStoreSinkValue(builder, bb, stores);
    if (!value)
      continue;
    std::vector<Attr*> storeAttrs;
    if (first->has<SizeAttr>())
      storeAttrs.push_back(new SizeAttr(SIZE(first)));
    builder.create<StoreOp>({ value, first->DEF(1) }, storeAttrs);
    for (auto store : stores)
      store->erase();
    sunk += static_cast<int>(stores.size()) - 1;
  }

  used.clear();
  // Use a dataflow approach.
  std::map<BasicBlock*, std::set<Op*>> in, out;
  const auto &bbs = region->getBlocks();
  std::deque<BasicBlock*> worklist(bbs.begin(), bbs.end());

  // Note that this is a FORWARD dataflow, rather than backward (as in updateLiveness()).
  // That's what ate up my whole afternoon.
  while (!worklist.empty()) {
    BasicBlock *bb = worklist.front();
    worklist.pop_front();

    // liveIn[bb] = \bigcup liveOut[pred]
    std::set<Op*> live;
    for (auto pred : bb->preds)
      live.insert(out[pred].begin(), out[pred].end());

    auto oldOut = out[bb];
    auto curLive = live;

    for (auto op : bb->getOps()) {
      if (isa<LoadOp>(op)) {
        for (auto *store : curLive) {
          auto addr = store->getOperand(1).defining;
          if (mayAlias(addr, op->DEF()))
            used[store] = true;
        }
      }

      if (isa<StoreOp>(op)) {
        std::vector<Op*> killed;
        for (auto store : curLive) {
          auto addr = store->getOperand(1).defining;
          // If this op stores to the exact same place as one of the `live` ops,
          // then that store is no longer live.
          if (mustAlias(addr, op->DEF(1)) || addr == op->DEF(1))
            killed.push_back(store);
        }
        for (auto kill : killed)
          curLive.erase(kill);

        // Don't forget to add this store to live set.
        curLive.insert(op);
      }
    }

    // Update if `out` changed.
    if (curLive != out[bb]) {
      out[bb] = curLive;
      for (auto succ : bb->succs)
        worklist.push_back(succ);
    }
  }


  auto funcOp = region->getParent();

  // Find allocas used by calls.
  // Stores to them shouldn't be eliminated. (This is also a kind of escape analysis.)
  auto calls = funcOp->findAll<CallOp>();
  std::set<Op*> outref;
  for (auto call : calls) {
    for (auto operand : call->getOperands()) {
      auto def = operand.defining;
      if (auto attr = def->find<AliasAttr>()) {
        for (auto &[base, offsets] : attr->location) {
          if (isa<AllocaOp>(base))
            outref.insert(base);
        }
      }
    }
  }

  // Eliminate unused stores.
  auto allStores = funcOp->findAll<StoreOp>();
  for (auto *store : allStores) {
    if (used[store])
      continue;

    auto addr = store->getOperand(1).defining;
    if (!addr->has<AliasAttr>())
      continue;
    
    auto alias = ALIAS(addr);
    // Never eliminate unknown things.
    bool canElim = !alias->unknown;
    for (auto [base, _] : alias->location) {
      // We can only eliminate if this stores to a local variable.
      // If parent of `base` is a ModuleOp (i.e. global), or another function,
      // then it's not a candidate of removal.
      if (base->getParentOp() != funcOp) {
        canElim = false;
        break;
      }
      // Don't remove stores to escaped locals.
      if (outref.count(base)) {
        canElim = false;
        break;
      }
    }

    if (!canElim)
      continue;

    elim++;
    store->erase();
  }
}

void DSE::removeUnread(Op *op, const std::vector<Op*> &gets) {
  std::unordered_set<Op*> addrs;
  std::vector<Op*> queue;
  const auto &name = NAME(op);
  for (auto x : gets) {
    if (NAME(x) == name)
      queue.push_back(x);
  }

  std::vector<Op*> stores;
  while (!queue.empty()) {
    auto back = queue.back();
    queue.pop_back();

    if (addrs.count(back))
      continue;
    addrs.insert(back);

    for (auto use : back->getUses()) {
      // Cannot remove.
      if (isa<LoadOp>(use) || isa<CallOp>(use))
        return;
      if (isa<StoreOp>(use)) {
        stores.push_back(use);
        continue;
      }

      // Might be a phi or an AddL. Anyway, must mark them.
      queue.push_back(use);
    }
  }

  // Hasn't been read; all stores can be removed.
  for (auto x : stores)
    x->erase();
}

void DSE::run() {
  ArrayStrideAnalysis(module).run();
  Alias(module).run();
  
  auto funcs = collectFuncs();

  for (auto func : funcs)
    runImpl(func->getRegion());

  // Look at globals that have only been stored but not read.
  auto globs = collectGlobals();
  auto gets = module->findAll<GetGlobalOp>();
  for (auto glob : globs)
    removeUnread(glob, gets);

  // Look at a load-store pattern like this:
  //   %a = load %1
  //   store %a, %1
  // The second store is useless.
  // Only do this on each basic block to avoid complicated analysis.  Keep the
  // scan alias-aware so unrelated memory ops do not hide the redundancy.
  auto loads = module->findAll<LoadOp>();
  std::vector<Op*> remove;
  auto rememberRemove = [&](Op *op) {
    if (std::find(remove.begin(), remove.end(), op) == remove.end())
      remove.push_back(op);
  };

  for (auto load : loads) {
    auto loadAddr = load->DEF();
    for (auto runner = load->nextOp(); runner; runner = runner->nextOp()) {
      if (isa<CallOp>(runner))
        break;
      if (isa<LoadOp>(runner)) {
        if (mayAlias(runner->DEF(), loadAddr))
          break;
        continue;
      }
      if (isa<StoreOp>(runner)) {
        auto addr = runner->DEF(1), value = runner->DEF(0);
        if (value == load && (addr == loadAddr || mustAlias(addr, loadAddr))) {
          rememberRemove(runner);
          continue;
        }
        if (mayAlias(addr, loadAddr))
          break;
        continue;
      }
    }
  }

  // Look at a store-store pattern like this:
  //   store %a, %1
  // <no load/call in between>
  //   store %b, %1
  // The first store is useless.
  auto stores = module->findAll<StoreOp>();
  for (auto store : stores) {
    auto storeAddr = store->DEF(1);
    for (auto runner = store->nextOp(); runner; runner = runner->nextOp()) {
      if (isa<CallOp>(runner))
        break;
      if (isa<LoadOp>(runner)) {
        if (mayAlias(runner->DEF(), storeAddr))
          break;
        continue;
      }
      if (isa<StoreOp>(runner)) {
        auto runnerAddr = runner->DEF(1);
        if (runnerAddr == storeAddr || mustAlias(runnerAddr, storeAddr)) {
          rememberRemove(store);
          continue;
        }
        if (mayAlias(runnerAddr, storeAddr))
          break;
        continue;
      }
    }
  }
  
  elim += remove.size();
  for (auto op : remove)
    op->erase();
}
