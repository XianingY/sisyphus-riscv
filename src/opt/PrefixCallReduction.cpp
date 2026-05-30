#include "LoopPasses.h"
#include "AnalysisManager.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

bool isInt(Op *op, int value) {
  return op && isa<IntOp>(op) && V(op) == value;
}

Op *stripSinglePhi(Op *op) {
  std::set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1 && !seen.count(op)) {
    seen.insert(op);
    op = op->DEF(0);
  }
  return op;
}

std::pair<Op*, Op*> phiIncomingByLatch(Op *phi, BasicBlock *latch) {
  Op *fromLatch = nullptr;
  Op *fromOther = nullptr;
  const auto &ops = phi->getOperands();
  const auto &attrs = phi->getAttrs();
  if (ops.size() != attrs.size())
    return { nullptr, nullptr };
  for (int i = 0; i < ops.size(); i++) {
    auto from = dyn_cast<FromAttr>(attrs[i]);
    if (!from)
      continue;
    if (from->bb == latch)
      fromLatch = ops[i].defining;
    else
      fromOther = ops[i].defining;
  }
  return { fromOther, fromLatch };
}

bool conditionIsPositiveCounter(Op *cond, Op *&counter) {
  if (!cond)
    return false;
  if (isa<PhiOp>(cond) && cond->getResultType() == Value::i32) {
    counter = cond;
    return true;
  }
  if ((isa<LtOp>(cond) || isa<NeOp>(cond)) && cond->getOperandCount() == 2) {
    if (isInt(cond->DEF(0), 0) && isa<PhiOp>(cond->DEF(1))) {
      counter = cond->DEF(1);
      return true;
    }
    if (isa<NeOp>(cond) && isa<PhiOp>(cond->DEF(0)) && isInt(cond->DEF(1), 0)) {
      counter = cond->DEF(0);
      return true;
    }
  }
  return false;
}

bool isDecrementByOne(Op *value, Op *counter) {
  if (!value || !counter || value->getOperandCount() != 2)
    return false;
  if (isa<SubIOp>(value) || isa<SubLOp>(value))
    return value->DEF(0) == counter && isInt(value->DEF(1), 1);
  if (isa<AddIOp>(value) || isa<AddLOp>(value))
    return (value->DEF(0) == counter && isInt(value->DEF(1), -1)) ||
           (value->DEF(1) == counter && isInt(value->DEF(0), -1));
  return false;
}

bool exprUsesValue(Op *op, Op *needle, std::set<Op*> &seen) {
  if (!op || !needle)
    return false;
  if (op == needle)
    return true;
  if (seen.count(op))
    return false;
  seen.insert(op);
  for (auto operand : op->getOperands())
    if (exprUsesValue(operand.defining, needle, seen))
      return true;
  return false;
}

bool exprUsesLoopLoadOrCall(Op *op, const std::set<BasicBlock*> &blocks,
                            std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (!blocks.count(op->getParent()))
    return false;
  if (isa<LoadOp>(op) || isa<CallOp>(op))
    return true;
  for (auto operand : op->getOperands())
    if (exprUsesLoopLoadOrCall(operand.defining, blocks, seen))
      return true;
  return false;
}

bool functionHasNoObservableWrites(FuncOp *func,
                                   const std::map<std::string, FuncOp*> &fnMap,
                                   std::set<std::string> &visiting);

bool callHasNoObservableWrites(CallOp *call,
                               const std::map<std::string, FuncOp*> &fnMap,
                               std::set<std::string> &visiting) {
  if (!call)
    return false;
  auto name = NAME(call);
  if (isExtern(name))
    return false;
  auto it = fnMap.find(name);
  if (it == fnMap.end() || !it->second)
    return false;
  return functionHasNoObservableWrites(it->second, fnMap, visiting);
}

bool functionHasNoObservableWrites(FuncOp *func,
                                   const std::map<std::string, FuncOp*> &fnMap,
                                   std::set<std::string> &visiting) {
  if (!func)
    return false;
  auto name = NAME(func);
  if (isExtern(name) || visiting.count(name))
    return false;
  visiting.insert(name);
  for (auto store : func->findAll<StoreOp>()) {
    (void) store;
    visiting.erase(name);
    return false;
  }
  for (auto store : func->findAll<VScaleStoreOp>()) {
    (void) store;
    visiting.erase(name);
    return false;
  }
  if (!func->findAll<CloneOp>().empty() || !func->findAll<JoinOp>().empty() ||
      !func->findAll<WakeOp>().empty()) {
    visiting.erase(name);
    return false;
  }
  for (auto nested : func->findAll<CallOp>()) {
    if (!callHasNoObservableWrites(cast<CallOp>(nested), fnMap, visiting)) {
      visiting.erase(name);
      return false;
    }
  }
  visiting.erase(name);
  return true;
}

bool bodyHasOnlyFinalValueSideEffects(const std::set<BasicBlock*> &blocks,
                                      BasicBlock *header, Op *counter,
                                      const std::map<std::string, FuncOp*> &fnMap) {
  bool hasStore = false;
  bool hasReadLikeOp = false;
  for (auto bb : blocks) {
    if (bb == header)
      continue;
    for (auto op : bb->getOps()) {
      if (auto call = dyn_cast<CallOp>(op)) {
        std::set<std::string> visiting;
        if (call->has<ImpureAttr>() &&
            !callHasNoObservableWrites(call, fnMap, visiting))
          return false;
        if (isExtern(NAME(call)))
          return false;
        hasReadLikeOp = true;
      }
      if (isa<LoadOp>(op))
        hasReadLikeOp = true;
      if (auto store = dyn_cast<StoreOp>(op)) {
        hasStore = true;
        std::set<Op*> seen;
        if (exprUsesValue(store->DEF(1), counter, seen))
          return false;
        seen.clear();
        if (exprUsesLoopLoadOrCall(store->DEF(0), blocks, seen))
          return false;
      }
      if (isa<ReturnOp>(op) || isa<CloneOp>(op) || isa<JoinOp>(op) ||
          isa<WakeOp>(op))
        return false;
    }
  }
  if (hasStore && hasReadLikeOp)
    return false;
  return true;
}

bool nonCounterPhisDoNotFeedBody(BasicBlock *header, Op *counter,
                                 const std::set<BasicBlock*> &blocks) {
  for (auto phi : header->getPhis()) {
    if (phi == counter)
      continue;
    for (auto use : phi->getUses()) {
      if (blocks.count(use->getParent()) && use->getParent() != header)
        return false;
    }
  }
  return true;
}

} // namespace

std::map<std::string, int> PrefixCallReduction::stats() {
  return {
    { "candidates", candidates },
    { "collapsed", collapsed },
    { "reject-counter", rejectedCounter },
    { "reject-phi", rejectedPhi },
    { "reject-shape", rejectedShape },
    { "reject-side-effect", rejectedSideEffect },
  };
}

bool PrefixCallReduction::runImpl(LoopInfo *loop,
                                  const std::map<std::string, FuncOp*> &fnMap) {
  candidates++;
  if (!loop || loop->latches.size() != 1 || loop->exits.size() != 1) {
    rejectedShape++;
    return false;
  }

  std::set<BasicBlock*> blocks(loop->getBlocks().begin(), loop->getBlocks().end());
  if (blocks.size() != 2) {
    rejectedShape++;
    return false;
  }

  auto header = loop->header;
  auto term = header ? header->getLastOp() : nullptr;
  if (!term || !isa<BranchOp>(term) || term->getOperandCount() == 0 ||
      !term->has<TargetAttr>() || !term->has<ElseAttr>()) {
    rejectedShape++;
    return false;
  }

  Op *counter = nullptr;
  if (!conditionIsPositiveCounter(term->DEF(0), counter) || !counter ||
      counter->getParent() != header || counter->getResultType() != Value::i32) {
    rejectedCounter++;
    return false;
  }

  auto latch = loop->getLatch();
  auto [start, latchVal] = phiIncomingByLatch(counter, latch);
  if (!start || !latchVal || !isDecrementByOne(latchVal, counter)) {
    rejectedCounter++;
    return false;
  }
  auto rawStart = stripSinglePhi(start);
  if (rawStart && isa<IntOp>(rawStart) && V(rawStart) <= 0) {
    rejectedCounter++;
    return false;
  }

  if (!bodyHasOnlyFinalValueSideEffects(blocks, header, counter, fnMap)) {
    rejectedSideEffect++;
    return false;
  }
  if (!nonCounterPhisDoNotFeedBody(header, counter, blocks)) {
    rejectedPhi++;
    return false;
  }

  Builder builder;
  builder.setBeforeOp(term);
  auto one = builder.create<IntOp>({ new IntAttr(1) });
  builder.setBeforeOp(latchVal);
  auto zero = builder.create<IntOp>({ new IntAttr(0) });

  std::vector<Op*> oldUses(counter->getUses().begin(), counter->getUses().end());
  for (auto use : oldUses) {
    if (!blocks.count(use->getParent()) || use->getParent() == header)
      continue;
    if (use == latchVal)
      continue;
    use->replaceOperand(counter, one);
  }
  latchVal->replaceAllUsesWith(zero);
  collapsed++;
  return true;
}

void PrefixCallReduction::run() {
  if (!envEnabled("SISY_ENABLE_PREFIX_CALL_REDUCTION", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
  std::map<std::string, FuncOp*> fnMap;
  for (auto func : collectFuncs())
    fnMap[NAME(func)] = func;

  std::map<FuncOp*, LoopForest> localForests;
  std::map<FuncOp*, LoopForest> *forests = nullptr;
  std::unique_ptr<LoopAnalysis> localAnalysis;
  if (context() && context()->enabled()) {
    forests = &context()->analysis().getLoopForests();
  } else {
    localAnalysis = std::make_unique<LoopAnalysis>(module);
    localAnalysis->run();
    localForests = localAnalysis->getResult();
    forests = &localForests;
  }

  for (auto &[func, forest] : *forests) {
    (void) func;
    for (auto loop : forest.getLoops())
      runImpl(loop, fnMap);
  }
}

PreservedAnalyses PrefixCallReduction::run(PassContext &ctx) {
  activeContext = &ctx;
  int before = collapsed;
  run();
  activeContext = nullptr;
  return collapsed == before ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
