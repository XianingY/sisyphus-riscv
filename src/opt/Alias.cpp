#include "Analysis.h"

#include <optional>
#include <unordered_map>
#include <unordered_set>

using namespace sys;

namespace {

int clampToInt(long long v) {
  if (v > INT_MAX)
    return INT_MAX;
  if (v < INT_MIN)
    return INT_MIN;
  return (int) v;
}

std::optional<long long> evalConstIntExpr(Op *op,
                                          std::unordered_map<Op*, std::optional<long long>> &memo,
                                          std::unordered_set<Op*> &visiting) {
  if (!op)
    return std::nullopt;
  if (memo.count(op))
    return memo[op];
  if (visiting.count(op))
    return std::nullopt;
  visiting.insert(op);

  std::optional<long long> value;
  if (isa<IntOp>(op))
    value = V(op);
  else if (isa<MinusOp>(op)) {
    auto x = evalConstIntExpr(op->DEF(), memo, visiting);
    if (x)
      value = -*x;
  } else if (isa<AddIOp>(op) || isa<SubIOp>(op) || isa<MulIOp>(op)) {
    auto x = evalConstIntExpr(op->DEF(0), memo, visiting);
    auto y = evalConstIntExpr(op->DEF(1), memo, visiting);
    if (x && y) {
      if (isa<AddIOp>(op))
        value = *x + *y;
      else if (isa<SubIOp>(op))
        value = *x - *y;
      else
        value = *x * *y;
    }
  } else if (isa<PhiOp>(op) && op->getOperandCount() > 0) {
    auto first = evalConstIntExpr(op->DEF(0), memo, visiting);
    if (first) {
      bool same = true;
      for (int i = 1; i < op->getOperandCount(); i++) {
        auto c = evalConstIntExpr(op->DEF(i), memo, visiting);
        if (!c || *c != *first) {
          same = false;
          break;
        }
      }
      if (same)
        value = first;
    }
  }

  visiting.erase(op);
  memo[op] = value;
  return value;
}

}

static void postorder(BasicBlock *current,
                      DomTree &tree,
                      std::vector<BasicBlock*> &order,
                      std::unordered_set<BasicBlock*> &visiting,
                      std::unordered_set<BasicBlock*> &visited) {
  if (!current || visited.count(current) || visiting.count(current))
    return;
  visiting.insert(current);
  for (auto bb : tree[current])
    postorder(bb, tree, order, visiting, visited);
  visiting.erase(current);
  visited.insert(current);
  order.push_back(current);
}

void Alias::runImpl(Region *region) {
  // Run local analysis over RPO of the dominator tree.

  // First calculate RPO.
  DomTree tree = getDomTree(region);

  BasicBlock *entry = region->getFirstBlock();
  std::vector<BasicBlock*> rpo;
  std::unordered_set<BasicBlock*> visiting;
  std::unordered_set<BasicBlock*> visited;
  postorder(entry, tree, rpo, visiting, visited);
  std::reverse(rpo.begin(), rpo.end());

  // Then traverse the CFG in that order.
  // This should guarantee definition comes before all uses.
  for (auto bb : rpo) {
    std::unordered_map<Op*, std::optional<long long>> constMemo;
    std::unordered_set<Op*> constVisiting;
    for (auto op : bb->getOps()) {
      if (isa<AllocaOp>(op)) {
        op->remove<AliasAttr>();
        op->add<AliasAttr>(op, 0);
        continue;
      }

      if (isa<GetGlobalOp>(op)) {
        op->remove<AliasAttr>();
        op->add<AliasAttr>(gMap[NAME(op)], 0);
        continue;
      }
      
      if (isa<AddLOp>(op)) {
        op->remove<AliasAttr>();
        auto x = op->getOperand(0).defining;
        auto y = op->getOperand(1).defining;
        if (!x->has<AliasAttr>() && !y->has<AliasAttr>()) {
          op->add<AliasAttr>(/*unknown*/);
          continue;
        }

        if (!x->has<AliasAttr>())
          std::swap(x, y);

        // Now `x` is the address and `y` is the offset. 
        // Note this swap won't affect the original op.
        auto alias = ALIAS(x)->clone();
        auto constOffset = evalConstIntExpr(y, constMemo, constVisiting);
        if (constOffset) {
          auto delta = clampToInt(*constOffset);
          for (auto &[_, offset] : alias->location) {
            for (auto &value : offset) {
              if (value != -1)
                value += delta;
            }
          }
        } else {
          // Unknown offset. Set all offsets to -1.
          for (auto &[_, offset] : alias->location)
            offset = { -1 };
        }

        if (ALIAS(x)->unknown)
          op->add<AliasAttr>(/*unknown*/);
        else
          op->add<AliasAttr>(alias->location);
        delete alias;
        continue;
      }
    }
  }
}

// This has better precision after Mem2Reg, because less `int**` is possible.
// Before Mem2Reg, we can store the address of an array in an alloca. 
// Some nested address patterns are still not fully eliminated after this pass.
// Moreover, it's more useful when all unnecessary alloca's have been removed.
//
// In addition, remember to update the information after Globalize and Localize.
void Alias::run() {
  auto funcs = collectFuncs();
  gMap = getGlobalMap();

  for (auto func : funcs)
    runImpl(func->getRegion());

  // Now do a dataflow analysis on call graph.
  auto fnMap = getFunctionMap();
  std::vector<FuncOp*> worklist;
  for (auto [_, v] : fnMap)
    worklist.push_back(v);

  while (!worklist.empty()) {
    auto func = worklist.back();
    worklist.pop_back();

    // Update local alias info.
    runImpl(func->getRegion());

    // Find all CallOps from the function.
    auto calls = func->findAll<CallOp>();

    for (auto call : calls) {
      const auto &name = NAME(call);
      bool changed = false;

      // Propagate alias info for each argument.
      if (isExtern(name))
        continue;

      runRewriter(fnMap[name], [&](GetArgOp *op) {
        int index = V(op);
        auto def = call->getOperand(index).defining;
        if (!def->has<AliasAttr>())
          return false;

        auto defLoc = ALIAS(def);
        if (op->has<AliasAttr>())
          changed |= ALIAS(op)->addAll(defLoc);
        else {
          op->add<AliasAttr>(*defLoc);
          changed = true;
        }
        
        // Do it only once.
        return false;
      });

      if (changed)
        worklist.push_back(fnMap[name]);
    }
  }
}
