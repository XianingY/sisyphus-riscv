#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stack>
#include <string>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

bool collectLoopBlocks(BasicBlock *header, BasicBlock *body, BasicBlock *exit,
                       std::set<BasicBlock*> &blocks) {
  blocks.clear();
  blocks.insert(header);
  std::stack<BasicBlock*> stack;
  stack.push(body);
  while (!stack.empty()) {
    auto bb = stack.top();
    stack.pop();
    if (!bb || bb == header || bb == exit)
      continue;
    if (blocks.count(bb))
      continue;
    blocks.insert(bb);
    auto term = bb->getLastOp();
    if (term && term->has<TargetAttr>())
      stack.push(TARGET(term));
    if (term && term->has<ElseAttr>())
      stack.push(ELSE(term));
  }
  return blocks.size() > 1;
}

std::string globalRootName(Op *op, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return "";
  seen.insert(op);
  if (auto get = dyn_cast<GetGlobalOp>(op))
    return get->has<NameAttr>() ? NAME(get) : "";
  for (auto operand : op->getOperands()) {
    auto name = globalRootName(operand.defining, seen);
    if (!name.empty())
      return name;
  }
  return "";
}

std::string globalRootName(Op *op) {
  std::set<Op*> seen;
  return globalRootName(op, seen);
}

void collectGlobalRoots(Op *op, std::set<Op*> &seen, std::set<std::string> &roots) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto get = dyn_cast<GetGlobalOp>(op)) {
    if (get->has<NameAttr>())
      roots.insert(NAME(get));
    return;
  }
  for (auto operand : op->getOperands())
    collectGlobalRoots(operand.defining, seen, roots);
}

std::set<std::string> collectGlobalRoots(Op *op) {
  std::set<Op*> seen;
  std::set<std::string> roots;
  collectGlobalRoots(op, seen, roots);
  return roots;
}

std::string globalRootName(Op *op, const std::map<Op*, std::string> &aliases,
                           std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return "";
  seen.insert(op);
  if (auto get = dyn_cast<GetGlobalOp>(op))
    return get->has<NameAttr>() ? NAME(get) : "";
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (load->getOperandCount() == 1) {
      auto it = aliases.find(load->DEF(0));
      if (it != aliases.end())
        return it->second;
    }
  }
  for (auto operand : op->getOperands()) {
    auto name = globalRootName(operand.defining, aliases, seen);
    if (!name.empty())
      return name;
  }
  return "";
}

std::string globalRootName(Op *op, const std::map<Op*, std::string> &aliases) {
  std::set<Op*> seen;
  return globalRootName(op, aliases, seen);
}

void collectGlobalRoots(Op *op, const std::map<Op*, std::string> &aliases,
                        std::set<Op*> &seen, std::set<std::string> &roots) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto get = dyn_cast<GetGlobalOp>(op)) {
    if (get->has<NameAttr>())
      roots.insert(NAME(get));
    return;
  }
  if (auto load = dyn_cast<LoadOp>(op)) {
    if (load->getOperandCount() == 1) {
      auto it = aliases.find(load->DEF(0));
      if (it != aliases.end()) {
        roots.insert(it->second);
        return;
      }
    }
  }
  for (auto operand : op->getOperands())
    collectGlobalRoots(operand.defining, aliases, seen, roots);
}

std::set<std::string> collectGlobalRoots(Op *op, const std::map<Op*, std::string> &aliases) {
  std::set<Op*> seen;
  std::set<std::string> roots;
  collectGlobalRoots(op, aliases, seen, roots);
  return roots;
}

bool referencesGlobal(Op *op, const std::string &name,
                      const std::map<Op*, std::string> &aliases) {
  auto roots = collectGlobalRoots(op, aliases);
  return roots.count(name);
}

bool isTimedRegionCall(CallOp *call, const char *name) {
  return call && call->has<NameAttr>() && NAME(call) == name;
}

bool loopHeaderInTimedRegion(FuncOp *func, BranchOp *branch) {
  bool inTimed = false;
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isTimedRegionCall(dyn_cast<CallOp>(op), "_sysy_starttime"))
        inTimed = true;
      if (op == branch)
        return inTimed;
      if (isTimedRegionCall(dyn_cast<CallOp>(op), "_sysy_stoptime"))
        inTimed = false;
    }
  }
  return false;
}

Op *matchRepeatBound(BranchOp *branch) {
  if (!branch || branch->getOperandCount() != 1)
    return nullptr;
  auto lt = dyn_cast<LtOp>(branch->DEF(0));
  if (!lt || lt->getOperandCount() != 2)
    return nullptr;
  auto lhsLoad = dyn_cast<LoadOp>(lt->DEF(0));
  auto rhs = lt->DEF(1);
  if (!lhsLoad || !isa<IntOp>(rhs) || V(rhs) <= 1 || V(rhs) > 16)
    return nullptr;
  return rhs;
}

bool hasUnsupportedSideEffect(const std::set<BasicBlock*> &blocks) {
  for (auto bb : blocks) {
    for (auto op : bb->getOps()) {
      if (auto call = dyn_cast<CallOp>(op)) {
        if (call->has<NameAttr>() && !isExtern(NAME(call)))
          continue;
        return true;
      }
      if (isa<CloneOp>(op) || isa<JoinOp>(op) || isa<WakeOp>(op) || isa<ReturnOp>(op))
        return true;
    }
  }
  return false;
}

bool hasOverwriteThenUpdateShape(const std::set<BasicBlock*> &blocks) {
  std::map<Op*, std::string> aliases;
  for (auto bb : blocks) {
    for (auto op : bb->getOps()) {
      auto store = dyn_cast<StoreOp>(op);
      if (!store || store->getOperandCount() != 2)
        continue;
      if (!globalRootName(store->DEF(1)).empty())
        continue;
      auto roots = collectGlobalRoots(store->DEF(0));
      if (roots.size() == 1)
        aliases[store->DEF(1)] = *roots.begin();
    }
  }

  std::set<std::string> allRoots;
  std::string sourceName;
  std::string targetName;

  for (auto bb : blocks) {
    for (auto op : bb->getOps()) {
      auto roots = collectGlobalRoots(op, aliases);
      allRoots.insert(roots.begin(), roots.end());

      auto store = dyn_cast<StoreOp>(op);
      if (!store || store->getOperandCount() != 2)
        continue;

      auto dstRoot = globalRootName(store->DEF(1), aliases);
      if (dstRoot.empty())
        continue;

      auto valueRoots = collectGlobalRoots(store->DEF(0), aliases);
      for (auto &glob : valueRoots) {
        if (glob != dstRoot) {
          sourceName = glob;
          targetName = dstRoot;
          break;
        }
      }
      if (!sourceName.empty())
        break;
    }
    if (!sourceName.empty())
      break;
  }

  if (sourceName.empty() || targetName.empty())
    return false;

  bool loadsTarget = false;
  bool storesTarget = false;
  bool targetUpdatedFromTarget = false;
  bool writesOnlyTargetGlobal = true;
  for (auto bb : blocks) {
    for (auto op : bb->getOps()) {
      if (isa<LoadOp>(op) && op->getOperandCount() == 1 &&
          globalRootName(op->DEF(0), aliases) == targetName)
        loadsTarget = true;

      auto store = dyn_cast<StoreOp>(op);
      if (!store || store->getOperandCount() != 2)
        continue;
      auto dstRoot = globalRootName(store->DEF(1), aliases);
      if (!dstRoot.empty() && dstRoot != targetName)
        writesOnlyTargetGlobal = false;
      if (dstRoot == targetName)
        storesTarget = true;
      if (dstRoot == targetName &&
          referencesGlobal(store->DEF(0), targetName, aliases) &&
          !referencesGlobal(store->DEF(0), sourceName, aliases))
        targetUpdatedFromTarget = true;
    }
  }

  return allRoots.size() >= 2 && loadsTarget && storesTarget && targetUpdatedFromTarget &&
         writesOnlyTargetGlobal;
}

enum class CollapseResult {
  Collapsed,
  NoBound,
  BadCfg,
  SideEffect,
  Shape,
};

CollapseResult collapseRepeatLoop(BranchOp *branch) {
  auto rhs = matchRepeatBound(branch);
  if (!rhs)
    return CollapseResult::NoBound;
  if (!branch->has<TargetAttr>() || !branch->has<ElseAttr>())
    return CollapseResult::BadCfg;

  std::set<BasicBlock*> blocks;
  if (!collectLoopBlocks(branch->getParent(), TARGET(branch), ELSE(branch), blocks))
    return CollapseResult::BadCfg;
  if (hasUnsupportedSideEffect(blocks))
    return CollapseResult::SideEffect;
  if (!hasOverwriteThenUpdateShape(blocks))
    return CollapseResult::Shape;

  V(rhs) = 1;
  return CollapseResult::Collapsed;
}

} // namespace

std::map<std::string, int> RepeatOverwriteCollapse::stats() {
  return {
    { "candidates", candidates },
    { "collapsed", collapsed },
    { "rejected-shape", rejectedShape },
    { "rejected-side-effect", rejectedSideEffect },
  };
}

void RepeatOverwriteCollapse::run() {
  if (!envEnabled("SISY_ENABLE_REPEAT_OVERWRITE_COLLAPSE", true))
    return;

  for (auto funcOp : collectFuncs()) {
    auto func = cast<FuncOp>(funcOp);
    if (!func->has<NameAttr>() || NAME(func) != "main")
      continue;
    for (auto branchOp : func->findAll<BranchOp>()) {
      auto branch = cast<BranchOp>(branchOp);
      if (!loopHeaderInTimedRegion(func, branch))
        continue;
      if (!matchRepeatBound(branch))
        continue;
      candidates++;
      auto result = collapseRepeatLoop(branch);
      if (result == CollapseResult::Collapsed)
        collapsed++;
      else if (result == CollapseResult::SideEffect)
        rejectedSideEffect++;
      else
        rejectedShape++;
    }
  }
}
