#include "Passes.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

void collectGlobals(Op *op, std::set<std::string> &out, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto global = dyn_cast<GetGlobalOp>(op)) {
    out.insert(NAME(global));
    return;
  }
  for (auto operand : op->getOperands())
    collectGlobals(operand.defining, out, seen);
}

std::set<std::string> globalsIn(Op *op) {
  std::set<std::string> result;
  std::set<Op*> seen;
  collectGlobals(op, result, seen);
  return result;
}

void markAll(Op *op, std::set<std::string> &live) {
  auto names = globalsIn(op);
  live.insert(names.begin(), names.end());
}

} // namespace

std::map<std::string, int> DeadGlobalStore::stats() {
  return {
    { "dead-globals", deadGlobals },
    { "removed-stores", removed },
  };
}

void DeadGlobalStore::run() {
  if (!envEnabled("SISY_ENABLE_DEAD_GLOBAL_STORE", true))
    return;

  std::set<std::string> globals;
  for (auto glob : module->findAll<GlobalOp>())
    if (glob->has<NameAttr>())
      globals.insert(NAME(glob));

  std::set<std::string> live;
  for (auto load : module->findAll<LoadOp>())
    markAll(load->DEF(0), live);

  for (auto call : module->findAll<CallOp>())
    for (auto operand : call->getOperands())
      markAll(operand.defining, live);

  for (auto ret : module->findAll<ReturnOp>())
    for (auto operand : ret->getOperands())
      markAll(operand.defining, live);

  for (auto store : module->findAll<StoreOp>())
    // The stored value can itself be a global pointer; the destination is
    // handled below and does not by itself make the global live.
    markAll(store->DEF(0), live);

  std::set<std::string> dead;
  for (const auto &name : globals)
    if (!live.count(name))
      dead.insert(name);
  deadGlobals = dead.size();
  if (dead.empty())
    return;

  auto stores = module->findAll<StoreOp>();
  for (auto op : stores) {
    auto store = cast<StoreOp>(op);
    if (store->getOperandCount() < 2)
      continue;
    auto dest = globalsIn(store->DEF(1));
    if (dest.size() != 1 || !dead.count(*dest.begin()))
      continue;
    store->erase();
    removed++;
  }
}
