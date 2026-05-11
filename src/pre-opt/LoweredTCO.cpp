#include "PrePasses.h"

#include <vector>

using namespace sys;

std::map<std::string, int> LoweredTCO::stats() {
  return {
    { "tail-calls-removed", removed },
    { "tail-calls-rejected", rejected },
  };
}

namespace {

bool isI32OnlyArgs(FuncOp *func) {
  auto *types = func->find<ArgTypesAttr>();
  if (!types)
    return false;
  for (auto ty : types->types) {
    if (ty != Value::i32)
      return false;
  }
  return true;
}

bool findReturnSlot(FuncOp *func, BasicBlock *&exitBlock, Op *&retSlot) {
  exitBlock = nullptr;
  retSlot = nullptr;

  for (auto *bb : func->getRegion()->getBlocks()) {
    if (bb->getOpCount() != 2)
      continue;
    auto *load = dyn_cast<LoadOp>(bb->getFirstOp());
    auto *ret = dyn_cast<ReturnOp>(bb->getLastOp());
    if (!load || !ret || ret->getOperandCount() != 1 || ret->DEF() != load)
      continue;
    exitBlock = bb;
    retSlot = load->DEF();
    return true;
  }

  return false;
}

bool collectArgSlots(FuncOp *func, std::vector<Op*> &slots,
                     std::vector<Op*> &getargs, std::vector<Op*> &stores) {
  int argc = func->get<ArgCountAttr>()->count;
  slots.assign(argc, nullptr);
  getargs.assign(argc, nullptr);
  stores.assign(argc, nullptr);

  for (auto *getarg : func->findAll<GetArgOp>()) {
    if (!getarg->has<IntAttr>())
      return false;
    int index = V(getarg);
    if (index < 0 || index >= argc)
      return false;
    if (getarg->getResultType() != Value::i32)
      return false;

    Op *initStore = nullptr;
    Op *slot = nullptr;
    for (auto *use : getarg->getUses()) {
      auto *store = dyn_cast<StoreOp>(use);
      if (!store || store->getOperandCount() != 2 || store->DEF(0) != getarg)
        return false;
      if (initStore)
        return false;
      initStore = store;
      slot = store->DEF(1);
    }
    if (!initStore || !slot || !isa<AllocaOp>(slot))
      return false;

    getargs[index] = getarg;
    stores[index] = initStore;
    slots[index] = slot;
  }

  for (int i = 0; i < argc; i++) {
    if (!slots[i] || !getargs[i] || !stores[i])
      return false;
  }
  return true;
}

bool isTailSelfCall(FuncOp *func, Op *retSlot, BasicBlock *exitBlock, Op *call) {
  if (!isa<CallOp>(call) || NAME(call) != NAME(func))
    return false;
  if (call->getUses().size() != 1)
    return false;

  auto *store = dyn_cast<StoreOp>(call->nextOp());
  if (!store || store->getOperandCount() != 2 || store->DEF(0) != call || store->DEF(1) != retSlot)
    return false;

  auto *go = dyn_cast<GotoOp>(store->nextOp());
  if (!go || TARGET(go) != exitBlock || go != store->getParent()->getLastOp())
    return false;

  return true;
}

void moveEntryState(FuncOp *func, BasicBlock *oldEntry,
                    const std::vector<Op*> &getargs, const std::vector<Op*> &stores) {
  auto *region = func->getRegion();
  auto *entry = region->insert(oldEntry);

  std::vector<Op*> allocas = func->findAll<AllocaOp>();
  for (auto *alloca : allocas)
    alloca->moveToEnd(entry);

  for (size_t i = 0; i < getargs.size(); i++) {
    getargs[i]->moveToEnd(entry);
    stores[i]->moveToEnd(entry);
  }

  Builder builder;
  builder.setToBlockEnd(entry);
  builder.create<GotoOp>({ new TargetAttr(oldEntry) });
}

} // namespace

bool LoweredTCO::runImpl(FuncOp *func) {
  int argc = func->get<ArgCountAttr>()->count;
  if (argc <= 0 || argc >= 16 || !isI32OnlyArgs(func)) {
    rejected++;
    return false;
  }

  auto *region = func->getRegion();
  auto *oldEntry = region->getFirstBlock();
  if (!oldEntry->getPhis().empty()) {
    rejected++;
    return false;
  }

  BasicBlock *exitBlock = nullptr;
  Op *retSlot = nullptr;
  if (!findReturnSlot(func, exitBlock, retSlot)) {
    rejected++;
    return false;
  }

  std::vector<Op*> argSlots, getargs, initStores;
  if (!collectArgSlots(func, argSlots, getargs, initStores)) {
    rejected++;
    return false;
  }

  std::vector<CallOp*> tailCalls;
  for (auto *call : func->findAll<CallOp>()) {
    if (isTailSelfCall(func, retSlot, exitBlock, call))
      tailCalls.push_back(cast<CallOp>(call));
  }
  if (tailCalls.empty())
    return false;

  Builder builder;
  int localRemoved = 0;
  for (auto *call : tailCalls) {
    if (call->getOperandCount() != argc) {
      rejected++;
      continue;
    }

    auto *store = call->nextOp();
    auto *go = store->nextOp();

    builder.setBeforeOp(call);
    for (int i = 0; i < argc; i++)
      builder.create<StoreOp>({ call->getOperand(i), argSlots[i] }, { new SizeAttr(4) });

    store->erase();
    call->erase();
    builder.replace<GotoOp>(go, { new TargetAttr(oldEntry) });
    removed++;
    localRemoved++;
  }

  if (localRemoved > 0) {
    moveEntryState(func, oldEntry, getargs, initStores);
    region->updatePreds();
    return true;
  }
  return false;
}

void LoweredTCO::run() {
  for (auto *func : collectFuncs())
    runImpl(func);
}
