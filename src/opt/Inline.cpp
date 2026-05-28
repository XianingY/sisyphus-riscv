#include "Passes.h"
#include "Analysis.h"
#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

int recursiveInlined = 0;

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

int envIntClamped(const char *name, int fallback, int low, int high) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  char *end = nullptr;
  long parsed = std::strtol(env, &end, 10);
  if (!end || *end || parsed < low || parsed > high)
    return fallback;
  return (int) parsed;
}

int valueSize(Value::Type ty) {
  return ty == Value::i64 ? 8 : 4;
}

}

std::map<std::string, int> Inline::stats() {
  return {
    { "inlined-functions", inlined },
    { "recursive-inlined", recursiveInlined }
  };
}

bool isRecursive(Op *op) {
  const auto &callers = CALLER(op);
  const auto &name = NAME(op);
  return std::find(callers.begin(), callers.end(), name) != callers.end();
}

namespace {

void doInline(Op *call, Region *fnRegion) {
  Builder builder;

  std::vector<BasicBlock*> srcBlocks;
  for (auto src : fnRegion->getBlocks())
    srcBlocks.push_back(src);

  // Maps old Op to new Op.
  std::map<Op*, Op*> cloneMap;
  std::map<BasicBlock*, BasicBlock*> retargetMap;
  auto bb = call->getParent();
  auto callRegion = bb->getParent();
  auto end = callRegion->insertAfter(bb);
  bb->splitOpsAfter(end, call);

  // Current situation:
  // bb0:
  //    ....
  // bb1:
  //    call
  //    ....
  // Insert a proper amount of blocks between `bb` and `end`.
  std::vector<BasicBlock*> body;
  for ([[maybe_unused]] auto src : srcBlocks)
    body.push_back(callRegion->insert(end));

  builder.setToBlockEnd(bb);
  // The address of return value. SysY returns scalars here; keep the slot size
  // tied to the call result so recursive inlining stays correct for i64 IR.
  auto retSize = valueSize(call->getResultType());
  auto addr = builder.create<AllocaOp>({ new SizeAttr(retSize) });

  // Copy the operations block by block (phase 1: clone without remapping).
  int i = 0;
  for (auto bb : srcBlocks) {
    builder.setToBlockStart(body[i]);
    retargetMap[bb] = body[i];
    i++;

    for (auto op : bb->getOps()) {
      auto shallow = builder.copy(op);
      cloneMap[op] = shallow;
    }
  }

  // Phase 2: remap operands now that all clones exist in cloneMap.
  for (auto bb : srcBlocks) {
    for (auto op : bb->getOps()) {
      auto clone = cloneMap[op];
      auto operands = clone->getOperands();
      clone->removeAllOperands();
      for (auto operand : operands) {
        auto def = operand.defining;
        clone->pushOperand(cloneMap.count(def) ? cloneMap[def]->getResult() : operand);
      }
    }
  }

  // Connect the blocks together.
  assert(body.size());
  builder.setToBlockEnd(bb);
  builder.create<GotoOp>({ new TargetAttr(body[0]) });

  // Rewrite operations.
  for (auto [_, v] : cloneMap) {
    // Rewire jump targets.
    // As it's a shallow copy, we need to create a new one to avoid affecting the original version.
    if (auto attr = v->find<TargetAttr>()) {
      assert(retargetMap.count(attr->bb));
      attr->bb = retargetMap[attr->bb];
    }
    if (auto attr = v->find<ElseAttr>()) {
      assert(retargetMap.count(attr->bb));
      attr->bb = retargetMap[attr->bb];
    }

    if (isa<GetArgOp>(v)) {
      auto i = V(v);
      auto def = call->getOperand(i).defining;
      v->replaceAllUsesWith(def);
      v->erase();
      continue;
    }

    if (isa<ReturnOp>(v)) {
      if (v->getOperands().size() == 0) {
        builder.replace<GotoOp>(v, { new TargetAttr(end) });
        continue;
      }

      auto ret = v->getOperand().defining;
      builder.setBeforeOp(v);
      builder.create<StoreOp>({ ret, addr }, { new SizeAttr(retSize) });
      builder.replace<GotoOp>(v, { new TargetAttr(end) });
      continue;
    }
  }

  builder.setBeforeOp(call);
  auto load = builder.create<LoadOp>(call->getResultType(), { addr }, { new SizeAttr(retSize) });
  call->replaceAllUsesWith(load);
  call->erase();
}

struct InlineShape {
  int opcount = 0;
  int callcount = 0;
  int selfCalls = 0;
};

bool exprReferencesGlobal(Op *op, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (isa<GetGlobalOp>(op))
    return true;
  for (auto operand : op->getOperands())
    if (exprReferencesGlobal(operand.defining, seen))
      return true;
  return false;
}

bool writesGlobalMemory(FuncOp *func) {
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    std::set<Op*> seen;
    if (exprReferencesGlobal(store->DEF(1), seen))
      return true;
  }
  return false;
}

bool readsOnlyImmutableScalarGlobals(FuncOp *func) {
  auto gets = func->findAll<GetGlobalOp>();
  if (gets.empty())
    return true;

  std::map<std::string, GlobalOp*> gMap;
  auto module = func->getParentOp<ModuleOp>();
  for (auto op : module->getRegion()->getFirstBlock()->getOps())
    if (auto glob = dyn_cast<GlobalOp>(op); glob && glob->has<NameAttr>())
      gMap[NAME(glob)] = glob;
  for (auto get : gets) {
    auto it = gMap.find(NAME(get));
    if (it == gMap.end())
      return false;
    auto global = it->second;
    if (!global->has<DimensionAttr>() || DIM(global).size() != 1 || DIM(global)[0] != 1)
      return false;
    if (!global->find<IntArrayAttr>())
      return false;
  }
  return true;
}

InlineShape analyzeInlineShape(FuncOp *func) {
  InlineShape shape;
  const auto &self = NAME(func);
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      shape.opcount++;
      if (isa<CallOp>(op)) {
        shape.callcount++;
        if (NAME(op) == self)
          shape.selfCalls++;
      }
    }
  }
  return shape;
}

bool canInlineRecursive(FuncOp *func) {
  if (!envEnabled("SISY_ENABLE_RECURSIVE_INLINE", true))
    return false;
  if (!func->has<ArgCountAttr>())
    return false;
  if (func->get<ArgCountAttr>()->count >= 8)
    return false;
  if (func->has<ImpureAttr>() || writesGlobalMemory(func) || !readsOnlyImmutableScalarGlobals(func))
    return false;

  auto shape = analyzeInlineShape(func);
  if (shape.opcount >= envIntClamped("SISY_RECURSIVE_INLINE_MAX_OPS", 96, 16, 256))
    return false;
  if (shape.selfCalls == 0 || shape.selfCalls > 2)
    return false;

  return true;
}

}

// This pass must run before Mem2Reg but after FlattenCFG.
void Inline::run() {
  CallGraph(module).run();
  
  Builder builder;

  fnMap = getFunctionMap();
  std::set<FuncOp*> recursive;

  runRewriter([&](CallOp *call) {
    const auto &fname = NAME(call);
    if (isExtern(fname))
      return false;

    if (!fnMap.count(fname)) {
      std::cerr << "unknown function: " << fname << "\n";
      assert(false);
    }

    FuncOp *func = fnMap[fname];
    if (writesGlobalMemory(func))
      return false;

    // Don't inline overly large functions.
    auto fnRegion = func->getRegion();
    int opcount = 0;
    int callcount = 0;
    for (auto bb : fnRegion->getBlocks())
      for (auto op : bb->getOps()) {
        opcount++;
        if (isa<CallOp>(op))
          callcount++;
      }
    int localThreshold = threshold;
    if (envEnabled("SISY_INLINE_USE_FN_SUMMARY", false)) {
      if (auto s = func->find<FunctionSummaryAttr>()) {
        if (s->pure && s->norecurse && s->argWriteMask == 0) {
          int boost = envIntClamped("SISY_INLINE_PURE_BOOST", 2, 1, 8);
          localThreshold = threshold * boost;
        }
      }
    }
    if (opcount >= localThreshold)
      return false;
    // Functions with internal calls near threshold tend to bloat recursive workloads.
    if (callcount > 0 && opcount * 5 >= localThreshold * 4)
      return false;

    // Don't inline recursive functions here, otherwise this rewriter will loop forever.
    // Deal with them later.
    if (isRecursive(func)) {
      recursive.insert(func);
      return false;
    }

    doInline(call, fnRegion);
    inlined++;
    return true;
  });

  if (envEnabled("SISY_ENABLE_RECURSIVE_INLINE", true)) {
    int maxDepth = envIntClamped("SISY_RECURSIVE_INLINE_DEPTH", 2, 1, 4);
    int maxCalls = envIntClamped("SISY_RECURSIVE_INLINE_BUDGET", 2, 1, 256);
    int usedCalls = 0;

    for (int depth = 0; depth < maxDepth && usedCalls < maxCalls; depth++) {
      std::vector<Op*> recursiveCalls;
      for (auto func : collectFuncs()) {
        if (!recursive.count(func) && !isRecursive(func))
          continue;
        if (!canInlineRecursive(func))
          continue;

        const auto &fname = NAME(func);
        for (auto call : module->findAll<CallOp>()) {
          if (NAME(call) != fname)
            continue;
          if (call->getOperandCount() != func->get<ArgCountAttr>()->count)
            continue;
          auto parent = call->getParentOp();
          if (!parent || !isa<FuncOp>(parent))
            continue;
          if (parent == func)
            continue;
          recursiveCalls.push_back(call);
          if ((int)recursiveCalls.size() + usedCalls >= maxCalls)
            break;
        }
        if ((int)recursiveCalls.size() + usedCalls >= maxCalls)
          break;
      }

      if (recursiveCalls.empty())
        break;

      int changedThisRound = 0;
      for (auto call : recursiveCalls) {
        const auto &fname = NAME(call);
        if (!fnMap.count(fname))
          continue;
        auto func = fnMap[fname];
        if (!canInlineRecursive(func))
          continue;

        doInline(call, func->getRegion());
        recursiveInlined++;
        inlined++;
        usedCalls++;
        changedThisRound++;
        if (usedCalls >= maxCalls)
          break;
      }
      if (changedThisRound == 0)
        break;
    }
  }

  // New alloca's have been introduced. Move them to the top.
  auto funcs = collectFuncs();
  
  for (auto func : funcs) {
    auto allocas = func->findAll<AllocaOp>();
    auto region = func->getRegion();
    auto begin = region->getFirstBlock();
    
    // It's possible we've inlined a function into another one without alloca's.
    // In that case we must create a new block for it.
    if (!isa<AllocaOp>(begin->getFirstOp())) {
      auto last = begin;
      begin = region->insert(begin);
      builder.setToBlockEnd(begin);
      builder.create<GotoOp>({ new TargetAttr(last) });
    }

    for (auto alloca : allocas)
      alloca->moveBefore(begin->getLastOp());
  }
}
