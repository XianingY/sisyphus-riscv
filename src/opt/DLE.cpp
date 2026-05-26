#include "CleanupPasses.h"
#include "MemorySSA.h"

using namespace sys;

#define STORE_ADDR(op) op->DEF(1)
#define LOAD_ADDR(op)  op->DEF()

namespace {

bool sameStoredValue(Op *a, Op *b) {
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

bool sameMemoryValue(StoreOp *a, StoreOp *b) {
  if (!a || !b)
    return false;
  if (!sameStoredValue(a->DEF(0), b->DEF(0)))
    return false;
  return mustAlias(STORE_ADDR(a), STORE_ADDR(b)) || STORE_ADDR(a) == STORE_ADDR(b);
}

bool sameLoadStoreSlot(LoadOp *load, StoreOp *store) {
  if (!load || !store)
    return false;
  if (store->DEF(0)->getResultType() != load->getResultType())
    return false;
  if (store->has<SizeAttr>() && load->has<SizeAttr>() && SIZE(store) != SIZE(load))
    return false;
  auto loadAddr = LOAD_ADDR(load);
  auto storeAddr = STORE_ADDR(store);
  return loadAddr == storeAddr || mustAlias(loadAddr, storeAddr);
}

std::vector<StoreOp*> commonStores(const std::set<BasicBlock*> &preds,
                                   const std::map<BasicBlock*, std::vector<StoreOp*>> &out) {
  std::vector<StoreOp*> common;
  bool first = true;
  for (auto pred : preds) {
    auto it = out.find(pred);
    const std::vector<StoreOp*> empty;
    const auto &stores = it == out.end() ? empty : it->second;
    if (first) {
      common = stores;
      first = false;
      continue;
    }

    std::vector<StoreOp*> next;
    for (auto candidate : common) {
      bool found = false;
      for (auto store : stores) {
        if (sameMemoryValue(candidate, store)) {
          found = true;
          break;
        }
      }
      if (found)
        next.push_back(candidate);
    }
    common = std::move(next);
  }
  return common;
}

bool callCanTouchUnpassedGlobals(CallOp *call) {
  if (!call)
    return true;
  // SysY runtime calls can only touch user arrays through pointer arguments.
  // If no pointer to a global is passed, keeping reaching global stores across
  // timer/input/output calls enables ordinary store-to-load forwarding without
  // pretending that the runtime has hidden access to compiler globals.
  return !isExtern(NAME(call));
}

void retainStoresNotClobberedByCall(std::vector<StoreOp*> &stores, CallOp *call) {
  std::set<Op*> toclear;
  bool clearUnpassedGlobals = callCanTouchUnpassedGlobals(call);
  for (auto operand : call->getOperands()) {
    auto def = operand.defining;
    if (!def->has<AliasAttr>())
      continue;
    auto alias = ALIAS(def);
    if (alias->unknown) {
      stores.clear();
      return;
    }

    for (auto [base, _] : alias->location)
      toclear.insert(base);
  }

  std::vector<StoreOp*> remaining;
  for (auto store : stores) {
    auto x = STORE_ADDR(store);
    if (!x->has<AliasAttr>())
      continue;

    bool good = true;
    for (auto [base, _] : ALIAS(x)->location) {
      if (toclear.count(base) ||
          (clearUnpassedGlobals && (isa<GetGlobalOp>(base) || isa<GlobalOp>(base)))) {
        good = false;
        break;
      }
    }
    if (good)
      remaining.push_back(store);
  }
  stores = std::move(remaining);
}

void retainStoresNotClobberedByCall(std::vector<Op*> &stores, CallOp *call) {
  std::set<Op*> toclear;
  bool clearUnpassedGlobals = callCanTouchUnpassedGlobals(call);
  for (auto operand : call->getOperands()) {
    auto def = operand.defining;
    if (!def->has<AliasAttr>())
      continue;
    auto alias = ALIAS(def);
    if (alias->unknown) {
      stores.clear();
      return;
    }

    for (auto [base, _] : alias->location)
      toclear.insert(base);
  }

  std::vector<Op*> remaining;
  for (auto store : stores) {
    auto x = STORE_ADDR(store);
    if (!x->has<AliasAttr>())
      continue;

    bool good = true;
    for (auto [base, _] : ALIAS(x)->location) {
      if (toclear.count(base) ||
          (clearUnpassedGlobals && (isa<GetGlobalOp>(base) || isa<GlobalOp>(base)))) {
        good = false;
        break;
      }
    }
    if (good)
      remaining.push_back(store);
  }
  stores = std::move(remaining);
}

bool storeMayWriteExternally(FuncOp *func, StoreOp *store);

bool functionMayWriteMemory(FuncOp *func, const std::map<std::string, FuncOp*> &fnMap,
                            std::set<FuncOp*> &visiting,
                            std::map<FuncOp*, bool> &memo) {
  if (!func)
    return true;
  if (memo.count(func))
    return memo[func];
  if (visiting.count(func))
    return false;
  visiting.insert(func);

  bool mayWrite = false;
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (auto store = dyn_cast<StoreOp>(op); store && storeMayWriteExternally(func, store)) {
        mayWrite = true;
        break;
      }
      if (isa<CloneOp>(op) || isa<JoinOp>(op) || isa<WakeOp>(op)) {
        mayWrite = true;
        break;
      }
      if (auto call = dyn_cast<CallOp>(op)) {
        const auto &name = NAME(call);
        if (isExtern(name)) {
          mayWrite = true;
          break;
        }
        auto it = fnMap.find(name);
        if (it == fnMap.end() || functionMayWriteMemory(it->second, fnMap, visiting, memo)) {
          mayWrite = true;
          break;
        }
      }
    }
    if (mayWrite)
      break;
  }

  visiting.erase(func);
  memo[func] = mayWrite;
  return mayWrite;
}

bool callMayWriteMemory(CallOp *call, const std::map<std::string, FuncOp*> &fnMap,
                        std::map<FuncOp*, bool> &memo) {
  if (!call)
    return true;
  const auto &name = NAME(call);
  if (isExtern(name))
    return true;
  auto it = fnMap.find(name);
  if (it == fnMap.end())
    return true;
  std::set<FuncOp*> visiting;
  return functionMayWriteMemory(it->second, fnMap, visiting, memo);
}

bool storeMayWriteExternally(FuncOp *func, StoreOp *store) {
  if (!func || !store)
    return true;
  auto addr = STORE_ADDR(store);
  if (!addr || !addr->has<AliasAttr>())
    return true;
  auto alias = ALIAS(addr);
  if (alias->unknown)
    return true;
  for (auto &[base, _] : alias->location) {
    if (!isa<AllocaOp>(base))
      return true;
    if (base->getParentOp<FuncOp>() != func)
      return true;
  }
  return false;
}

Op *materializeDominatingValue(Builder &builder, LoadOp *load, Op *value) {
  if (!load || !value)
    return nullptr;

  builder.setBeforeOp(load);
  if (isa<IntOp>(value))
    return builder.create<IntOp>({ new IntAttr(V(value)) });
  if (isa<FloatOp>(value))
    return builder.create<FloatOp>({ new FloatAttr(F(value)) });

  auto valueBlock = value->getParent();
  auto loadBlock = load->getParent();
  if (valueBlock && loadBlock && valueBlock->dominates(loadBlock))
    return value;
  return nullptr;
}

bool valueAvailableOnEdge(Op *value, BasicBlock *pred) {
  if (!value || !pred)
    return false;
  if (isa<IntOp>(value) || isa<FloatOp>(value))
    return true;
  auto parent = value->getParent();
  return parent && parent->dominates(pred);
}

void setAfterPhis(Builder &builder, BasicBlock *bb) {
  auto phis = bb->getPhis();
  if (phis.empty())
    builder.setToBlockStart(bb);
  else
    builder.setAfterOp(phis.back());
}

StoreOp *findUnambiguousReachingStore(const std::vector<StoreOp*> &stores, LoadOp *load) {
  StoreOp *matched = nullptr;
  auto loadAddr = LOAD_ADDR(load);
  for (auto store : stores) {
    auto storeAddr = STORE_ADDR(store);
    if (mustAlias(loadAddr, storeAddr) || loadAddr == storeAddr) {
      if (store->DEF(0)->getResultType() != load->getResultType())
        return nullptr;
      if (store->has<SizeAttr>() && load->has<SizeAttr>() && SIZE(store) != SIZE(load))
        return nullptr;
      if (matched && matched != store)
        return nullptr;
      matched = store;
      continue;
    }

    if (mayAlias(loadAddr, storeAddr))
      return nullptr;
  }
  return matched;
}

Op *materializeEdgePhi(Builder &builder, BasicBlock *bb,
                       const std::vector<std::pair<BasicBlock*, StoreOp*>> &incoming) {
  if (incoming.size() < 2)
    return nullptr;

  std::vector<Value> values;
  std::vector<Attr*> from;
  values.reserve(incoming.size());
  from.reserve(incoming.size());

  for (auto [pred, store] : incoming) {
    auto value = store->DEF(0);
    if (!valueAvailableOnEdge(value, pred))
      return nullptr;
    values.push_back(value);
    from.push_back(new FromAttr(pred));
  }

  setAfterPhis(builder, bb);
  return builder.create<PhiOp>(values, from);
}

Op *tryMaterializeReachingStorePhi(Builder &builder, BasicBlock *bb, LoadOp *load,
                                   const std::map<BasicBlock*, std::vector<StoreOp*>> &availableOut) {
  if (!bb || !load || bb->preds.size() < 2)
    return nullptr;

  std::vector<std::pair<BasicBlock*, StoreOp*>> incoming;
  incoming.reserve(bb->preds.size());

  for (auto pred : bb->preds) {
    auto it = availableOut.find(pred);
    if (it == availableOut.end())
      return nullptr;
    auto store = findUnambiguousReachingStore(it->second, load);
    if (!store)
      return nullptr;
    incoming.emplace_back(pred, store);
  }

  return materializeEdgePhi(builder, bb, incoming);
}

Op *tryForwardMemorySSALoad(Builder &builder, MemorySSA &mssa, LoadOp *load) {
  if (!load)
    return nullptr;

  auto clobbers = mssa.getClobberingDefs(load);
  std::set<MemoryAccess*> unique(clobbers.begin(), clobbers.end());
  if (unique.size() != 1)
    return nullptr;

  auto *access = *unique.begin();
  auto store = access ? dyn_cast<StoreOp>(access->op) : nullptr;
  if (!sameLoadStoreSlot(load, store))
    return nullptr;

  auto storeBlock = store->getParent();
  auto loadBlock = load->getParent();
  if (!storeBlock || !loadBlock || !storeBlock->dominates(loadBlock))
    return nullptr;

  return materializeDominatingValue(builder, load, store->DEF(0));
}

}

std::map<std::string, int> DLE::stats() {
  return {
    { "removed-loads", elim },
    { "memory-ssa-forwarded", memorySsaForwarded },
    { "readonly-calls-retained", readonlyCallsRetained },
  };
}

void DLE::runImpl(Region *region) {
  // First have a simple, context-insensitive approach to deal with load-after-store.
  std::map<Op*, Op*> replacement;
  Builder builder;
  std::map<FuncOp*, bool> memoryWriteMemo;

  for (auto bb : region->getBlocks()) {
    std::vector<Op*> liveStore;
    auto ops = bb->getOps();

    for (auto op : ops) {
      if (isa<StoreOp>(op)) {
        std::vector<Op*> newStore { op };
        for (auto x : liveStore) {
          if (neverAlias(STORE_ADDR(x), STORE_ADDR(op)))
            newStore.push_back(x);
        }
        liveStore = std::move(newStore);
        continue;
      }
      
      // This call might invalidate all living stores.
      // Check its arguments to see what's been passed into it,
      // and clear the store of those things.
      // Moreover, clear the stores of globals.
      if (auto call = dyn_cast<CallOp>(op); call && call->has<ImpureAttr>()) {
        if (!callMayWriteMemory(call, fnMap, memoryWriteMemo)) {
          readonlyCallsRetained++;
          continue;
        }
        retainStoresNotClobberedByCall(liveStore, call);
        continue;
      }

      if (isa<LoadOp>(op)) {
        // Replaces the loaded value with the init value of store.
        for (auto x : liveStore) {
          auto init = x->getOperand(0).defining;
          auto storeAddr = x->getOperand(1).defining;
          auto loadAddr = op->getOperand().defining;
          if (mustAlias(LOAD_ADDR(op), STORE_ADDR(x)) || storeAddr == loadAddr) {
            op->replaceAllUsesWith(init);
            op->erase();
            elim++;
            break;
          }
        }
      }
    }
  }

  region->updateDoms();
  std::map<BasicBlock*, std::vector<StoreOp*>> availableOut;
  std::vector<BasicBlock*> storeWorklist(region->getBlocks().begin(), region->getBlocks().end());

  while (!storeWorklist.empty()) {
    BasicBlock *bb = storeWorklist.back();
    storeWorklist.pop_back();

    auto liveStore = commonStores(bb->preds, availableOut);

    for (auto op : bb->getOps()) {
      if (auto store = dyn_cast<StoreOp>(op)) {
        std::vector<StoreOp*> next { store };
        for (auto x : liveStore) {
          if (neverAlias(STORE_ADDR(x), STORE_ADDR(store)))
            next.push_back(x);
        }
        liveStore = std::move(next);
        continue;
      }

      if (auto call = dyn_cast<CallOp>(op); call && call->has<ImpureAttr>()) {
        if (!callMayWriteMemory(call, fnMap, memoryWriteMemo)) {
          readonlyCallsRetained++;
          continue;
        }
        retainStoresNotClobberedByCall(liveStore, call);
        continue;
      }
    }

    if (liveStore != availableOut[bb]) {
      availableOut[bb] = liveStore;
      for (auto succ : bb->succs)
        storeWorklist.push_back(succ);
    }
  }

  for (auto bb : region->getBlocks()) {
    auto liveStore = commonStores(bb->preds, availableOut);
    auto ops = bb->getOps();
    bool cleanPredPrefix = true;

    for (auto op : ops) {
      if (auto store = dyn_cast<StoreOp>(op)) {
        cleanPredPrefix = false;
        std::vector<StoreOp*> next { store };
        for (auto x : liveStore) {
          if (neverAlias(STORE_ADDR(x), STORE_ADDR(store)))
            next.push_back(x);
        }
        liveStore = std::move(next);
        continue;
      }

      if (auto call = dyn_cast<CallOp>(op); call && call->has<ImpureAttr>()) {
        if (!callMayWriteMemory(call, fnMap, memoryWriteMemo)) {
          readonlyCallsRetained++;
          continue;
        }
        cleanPredPrefix = false;
        retainStoresNotClobberedByCall(liveStore, call);
        continue;
      }

      if (auto load = dyn_cast<LoadOp>(op)) {
        bool replaced = false;
        for (auto store : liveStore) {
          if (mustAlias(LOAD_ADDR(load), STORE_ADDR(store)) || LOAD_ADDR(load) == STORE_ADDR(store)) {
            auto replacement = materializeDominatingValue(builder, load, store->DEF(0));
            if (!replacement)
              continue;
            load->replaceAllUsesWith(replacement);
            load->erase();
            elim++;
            replaced = true;
            break;
          }
        }

        if (!replaced && cleanPredPrefix) {
          auto replacement = tryMaterializeReachingStorePhi(builder, bb, load, availableOut);
          if (replacement) {
            load->replaceAllUsesWith(replacement);
            load->erase();
            elim++;
          }
        }
      }
    }
  }

  region->updateDoms();
  MemorySSA memorySSA(region);
  memorySSA.build();
  for (auto bb : region->getBlocks()) {
    auto ops = bb->getOps();
    for (auto op : ops) {
      auto load = dyn_cast<LoadOp>(op);
      if (!load)
        continue;
      auto replacement = tryForwardMemorySSALoad(builder, memorySSA, load);
      if (!replacement)
        continue;
      load->replaceAllUsesWith(replacement);
      load->erase();
      elim++;
      memorySsaForwarded++;
    }
  }

  std::map<BasicBlock*, std::set<Op*>> liveIn;
  std::map<BasicBlock*, std::set<Op*>> liveOut;

  const auto &blocks = region->getBlocks();
  std::vector<BasicBlock*> worklist(blocks.begin(), blocks.end());

  while (!worklist.empty()) {
    BasicBlock *bb = worklist.back();
    worklist.pop_back();

    // Live in should be the INTERSECTION of all live out.
    // Unlike liveness analysis, it isn't union here.
    std::set<Op*> newLiveIn;

    bool firstPred = true;
    for (auto pred : bb->preds) {
      if (firstPred) {
        newLiveIn = liveOut[pred];
        firstPred = false;
      } else {
        // Compute intersection.
        std::set<Op*> temp;
        for (Op* op : newLiveIn) {
          if (liveOut[pred].count(op))
            temp.insert(op);
        }
        newLiveIn = std::move(temp);
      }
    }

    if (newLiveIn != liveIn[bb])
      liveIn[bb] = newLiveIn;

    std::set<Op*> live = liveIn[bb];

    auto ops = bb->getOps();
    for (auto op : ops) {
      if (isa<StoreOp>(op)) {
        // Kill all loads in `live` that might alias with the store.
        Op *storeAddr = STORE_ADDR(op);

        for (auto it = live.begin(); it != live.end(); ) {
          Op *load = *it;
          Op *loadAddr = LOAD_ADDR(load);
          if (mayAlias(storeAddr, loadAddr))
            it = live.erase(it);
          else
            ++it;
        }
      }

      // Note that there might be some redundancy, but it doesn't matter.
      // Also, we're pulling `load` in `live` rather than the address. That makes it easier for rewriting.
      if (isa<LoadOp>(op)) {
        // Check if something is exactly the value of `addr`, or must alias with it.
        // Note that the value might not `mustAlias` with itself; 
        // for example it might have `<alloca %1, -1>` which doesn't mustAlias with anything.
        Op *addr = LOAD_ADDR(op);
        bool replaced = false;
        for (auto load : live) {
          if (mustAlias(LOAD_ADDR(load), addr) || load->getOperand().defining == addr) {
            op->replaceAllUsesWith(load);
            op->erase();
            elim++;
            replaced = true;
            break;
          }
        }

        if (!replaced)
          live.insert(op);
      }
    }

    if (live != liveOut[bb]) {
      liveOut[bb] = live;

      for (auto succ : bb->succs)
        worklist.push_back(succ);
    }
  }
}

void DLE::run() {
  auto funcs = collectFuncs();
  fnMap = getFunctionMap();

  for (auto func : funcs)
    runImpl(func->getRegion());
}
