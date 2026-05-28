#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <queue>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

// Trace a value back to the GetArgOp it most directly derives from (if any).
// Handles load-from-alloca-that-was-stored-with-arg, simple address arithmetic,
// and arg flowing through scalar stores.  Bounded to avoid pathological cases.
GetArgOp *traceToArg(Op *op, int depth = 0) {
  if (!op || depth > 16)
    return nullptr;
  if (auto a = dyn_cast<GetArgOp>(op))
    return a;
  // Address arithmetic: AddL/AddI/SubL/SubI preserve the base argument.
  if (isa<AddLOp>(op) || isa<AddIOp>(op) ||
      isa<SubLOp>(op) || isa<SubIOp>(op)) {
    if (auto l = traceToArg(op->DEF(0), depth + 1)) return l;
    if (auto r = traceToArg(op->DEF(1), depth + 1)) return r;
    return nullptr;
  }
  // Load from an alloca slot that was written with a GetArg.
  if (auto load = dyn_cast<LoadOp>(op)) {
    Op *slot = load->DEF(0);
    if (slot && isa<AllocaOp>(slot)) {
      for (auto user : slot->getUses()) {
        if (auto store = dyn_cast<StoreOp>(user)) {
          if (store->DEF(1) == slot) {
            if (auto a = traceToArg(store->DEF(0), depth + 1))
              return a;
          }
        }
      }
    }
  }
  return nullptr;
}

bool callsSelfDirectly(FuncOp *func) {
  const auto &name = NAME(func);
  for (auto call : func->findAll<CallOp>())
    if (NAME(call) == name)
      return true;
  return false;
}

}  // namespace

void FunctionSummary::run() {
  if (!envEnabled("SISY_ENABLE_FN_SUMMARY", true))
    return;

  auto funcs = collectFuncs();
  auto fnMap = getFunctionMap();

  // Build callgraph adjacency: caller -> direct callees (FuncOp*).
  std::map<FuncOp*, std::set<FuncOp*>> succs;
  for (auto func : funcs) {
    auto &out = succs[func];
    for (auto call : func->findAll<CallOp>()) {
      const auto &n = NAME(call);
      auto it = fnMap.find(n);
      if (it != fnMap.end())
        out.insert(it->second);
    }
  }

  // norecurse: BFS from each function over callgraph; if we ever revisit `func`
  // it can recurse.  Bounded BFS keeps this cheap for SysY-sized modules.
  auto canRecurse = [&](FuncOp *func) {
    std::set<FuncOp*> visited;
    std::queue<FuncOp*> q;
    for (auto s : succs[func]) q.push(s);
    while (!q.empty()) {
      auto cur = q.front(); q.pop();
      if (cur == func) return true;
      if (!visited.insert(cur).second) continue;
      for (auto s : succs[cur]) q.push(s);
    }
    return false;
  };

  for (auto func : funcs) {
    func->remove<FunctionSummaryAttr>();
    func->add<FunctionSummaryAttr>();
    auto *summary = func->find<FunctionSummaryAttr>();
    bool impure = func->has<ImpureAttr>();

    bool hasGlobalAccess = !func->findAll<GetGlobalOp>().empty();
    bool hasClone = !func->findAll<CloneOp>().empty();
    bool hasJoin = !func->findAll<JoinOp>().empty();
    bool hasWake = !func->findAll<WakeOp>().empty();
    bool hasCall = !func->findAll<CallOp>().empty();

    // Compute arg masks by scanning Load/Store memory ops.
    uint64_t argRead = 0, argWrite = 0;
    bool unknownStore = false;
    for (auto op : func->findAll<StoreOp>()) {
      Op *dst = op->DEF(1);
      if (!dst) { unknownStore = true; continue; }
      // Store to a local AllocaOp is private; not observable.
      if (isa<AllocaOp>(dst))
        continue;
      if (auto arg = traceToArg(dst)) {
        int idx = V(arg);
        if (idx >= 0 && idx < 64)
          argWrite |= (uint64_t)1 << idx;
        else
          unknownStore = true;
      } else {
        unknownStore = true;
      }
    }
    for (auto op : func->findAll<LoadOp>()) {
      Op *src = op->DEF(0);
      if (!src) continue;
      if (isa<AllocaOp>(src))
        continue;
      if (auto arg = traceToArg(src)) {
        int idx = V(arg);
        if (idx >= 0 && idx < 64)
          argRead |= (uint64_t)1 << idx;
      }
    }

    summary->argReadMask = argRead;
    summary->argWriteMask = argWrite;

    // readonly: no observable store path (no impure, no global write, no
    // unknown destination, no thread-spawning, no arg-write).
    bool maybeReadonly = !impure && !hasClone && !hasJoin && !hasWake &&
                         !unknownStore && argWrite == 0;
    // Calls into other functions: only readonly if every callee is readonly.
    // First-pass approximation: require no calls at all.  A future iteration
    // can propagate transitively once we have a fixpoint.
    if (maybeReadonly && hasCall) maybeReadonly = false;
    summary->readonly = maybeReadonly;

    // pure: readonly + no global reads + no arg reads.
    summary->pure = summary->readonly && !hasGlobalAccess && argRead == 0;

    summary->norecurse = !canRecurse(func);

    if (summary->pure) pureCount++;
    if (summary->readonly) readonlyCount++;
    if (summary->norecurse) norecurseCount++;
    if (argRead) argReadCount++;
    if (argWrite) argWriteCount++;
  }
}
