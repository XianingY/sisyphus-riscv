#include "Passes.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <vector>

using namespace sys;

namespace {

bool envEnabled(const char *n, bool d) {
  const char *r = std::getenv(n);
  if (!r || !r[0]) return d;
  return std::strcmp(r, "0") != 0 && std::strcmp(r, "false") != 0;
}

}  // namespace

void IPConstProp::run() {
  if (!envEnabled("SISY_ENABLE_IPCP", false))
    return;

  auto funcs = collectFuncs();
  auto fnMap = getFunctionMap();

  // For each function, gather the set of integer constants observed at each
  // argument index across all (non-recursive) callsites.
  for (auto func : funcs) {
    const auto &fname = NAME(func);
    if (fname == "main") continue;
    if (!func->has<ArgCountAttr>()) continue;
    int argc = func->get<ArgCountAttr>()->count;
    if (argc <= 0) continue;

    // Map arg index -> uniform constant (unset means no callsite yet, or
    // non-uniform).
    std::vector<std::optional<int>> uniform(argc);
    std::vector<bool> killed(argc, false);
    int totalCalls = 0;

    for (auto call : module->findAll<CallOp>()) {
      if (NAME(call) != fname) continue;
      auto callerFunc = call->getParentOp<FuncOp>();
      if (callerFunc == func) continue;  // recursion
      if (call->getOperandCount() != (size_t) argc) continue;
      totalCalls++;
      for (int i = 0; i < argc; i++) {
        if (killed[i]) continue;
        auto def = call->getOperand(i).defining;
        if (!isa<IntOp>(def)) { killed[i] = true; rejectedNonConst++; continue; }
        int v = V(def);
        if (!uniform[i]) uniform[i] = v;
        else if (*uniform[i] != v) killed[i] = true;
      }
    }
    callsitesConsidered += totalCalls;
    if (totalCalls == 0) continue;

    // Replace GetArgOp(i) for arg indices with a confirmed uniform value.
    bool anyReplaced = false;
    Builder builder;
    for (auto bb : func->getRegion()->getBlocks()) {
      for (auto op : std::vector<Op*>(bb->getOps().begin(), bb->getOps().end())) {
        auto getArg = dyn_cast<GetArgOp>(op);
        if (!getArg) continue;
        int idx = V(getArg);
        if (idx < 0 || idx >= argc || killed[idx] || !uniform[idx]) continue;
        if (op->getResultType() != Value::i32) {
          rejectedNotI32++;
          continue;
        }
        builder.setBeforeOp(op);
        auto k = builder.create<IntOp>({ new IntAttr(*uniform[idx]) });
        op->replaceAllUsesWith(k);
        op->erase();
        argsReplaced++;
        anyReplaced = true;
      }
    }
    if (anyReplaced) functionsTouched++;
  }
}
