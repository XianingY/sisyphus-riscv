#include "CleanupPasses.h"

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

void retainStoresNotClobberedByCall(std::vector<StoreOp*> &stores, CallOp *call) {
  std::set<Op*> toclear;
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
      if (toclear.count(base) || isa<GetGlobalOp>(base) || isa<GlobalOp>(base)) {
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
      if (toclear.count(base) || isa<GetGlobalOp>(base) || isa<GlobalOp>(base)) {
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

}

std::map<std::string, int> DLE::stats() {
  return {
    { "removed-loads", elim },
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

    for (auto op : ops) {
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

      if (auto load = dyn_cast<LoadOp>(op)) {
        for (auto store : liveStore) {
          if (mustAlias(LOAD_ADDR(load), STORE_ADDR(store)) || LOAD_ADDR(load) == STORE_ADDR(store)) {
            auto replacement = materializeDominatingValue(builder, load, store->DEF(0));
            if (!replacement)
              continue;
            load->replaceAllUsesWith(replacement);
            load->erase();
            elim++;
            break;
          }
        }
      }
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
