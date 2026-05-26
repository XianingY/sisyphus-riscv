#include "HIRCanonicalize.h"

#include "../utils/DynamicCast.h"

#include <cmath>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sys::hir {

namespace {

bool isConstInt(const Op *op) {
  return op && op->kind == OpKind::ConstInt && op->hasIntValue;
}

bool isConstFloat(const Op *op) {
  return op && op->kind == OpKind::ConstFloat && op->hasFloatValue;
}

void rewriteToConstInt(Op *op, long long value) {
  op->kind = OpKind::ConstInt;
  op->type = TypeKind::Int;
  op->traits = defaultTraits(OpKind::ConstInt);
  op->hasIntValue = true;
  op->intValue = value;
  op->hasFloatValue = false;
  op->floatValue = 0.0;
  op->children.clear();
  op->symbol.clear();
}

void rewriteToConstFloat(Op *op, double value) {
  op->kind = OpKind::ConstFloat;
  op->type = TypeKind::Float;
  op->traits = defaultTraits(OpKind::ConstFloat);
  op->hasFloatValue = true;
  op->floatValue = value;
  op->hasIntValue = false;
  op->intValue = 0;
  op->children.clear();
  op->symbol.clear();
}

bool takeTruthyBranch(Op *cond) {
  if (isConstInt(cond))
    return cond->intValue != 0;
  if (isConstFloat(cond))
    return cond->floatValue != 0.0;
  return true;
}

std::unique_ptr<Op> cloneOp(const Op *op) {
  if (!op)
    return nullptr;
  auto out = std::make_unique<Op>(op->kind, op->origin);
  out->type = op->type;
  out->traits = op->traits;
  out->symbol = op->symbol;
  out->hasIntValue = op->hasIntValue;
  out->intValue = op->intValue;
  out->hasFloatValue = op->hasFloatValue;
  out->floatValue = op->floatValue;
  out->arrayDims = op->arrayDims;
  for (const auto &child : op->children)
    out->children.push_back(cloneOp(child.get()));
  return out;
}

struct SimpleAffineFunction {
  std::vector<std::string> params;
  const Op *expr = nullptr;
};

bool affineExprUsesOnlyParams(const Op *op,
                              const std::unordered_set<std::string> &params,
                              int &budget) {
  if (!op || --budget < 0)
    return false;
  if (op->kind == OpKind::ConstInt)
    return true;
  if (op->kind == OpKind::Load && op->children.empty())
    return params.count(op->symbol) != 0;
  if (op->kind == OpKind::Arith && op->children.size() == 2 &&
      (op->symbol == "+" || op->symbol == "-" || op->symbol == "*")) {
    // This is ordinary expression inlining, not semantic recognition: the
    // callee body is copied verbatim with actual arguments substituted.  Later
    // affine analyses may still reject symbolic products such as r * n.
    return affineExprUsesOnlyParams(op->children[0].get(), params, budget) &&
           affineExprUsesOnlyParams(op->children[1].get(), params, budget);
  }
  return false;
}

bool findSimpleReturnExpr(const Op *func, SimpleAffineFunction &out) {
  if (!func || func->kind != OpKind::Func || !func->origin)
    return false;
  auto *fn = dyn_cast<FnDeclNode>(func->origin);
  if (!fn || fn->args.empty() || func->children.size() != 1)
    return false;
  const Op *body = func->children[0].get();
  if (!body || body->kind != OpKind::Block || body->children.size() != 1)
    return false;
  const Op *ret = body->children[0].get();
  if (!ret || ret->kind != OpKind::Return || ret->children.size() != 1)
    return false;

  std::unordered_set<std::string> params(fn->args.begin(), fn->args.end());
  int budget = 24;
  if (!affineExprUsesOnlyParams(ret->children[0].get(), params, budget))
    return false;

  out.params = fn->args;
  out.expr = ret->children[0].get();
  return true;
}

std::unique_ptr<Op> cloneReplacingParams(
    const Op *op,
    const std::unordered_map<std::string, const Op*> &replacements) {
  if (!op)
    return nullptr;
  if (op->kind == OpKind::Load && op->children.empty()) {
    auto it = replacements.find(op->symbol);
    if (it != replacements.end())
      return cloneOp(it->second);
  }
  auto out = std::make_unique<Op>(op->kind, op->origin);
  out->type = op->type;
  out->traits = op->traits;
  out->symbol = op->symbol;
  out->hasIntValue = op->hasIntValue;
  out->intValue = op->intValue;
  out->hasFloatValue = op->hasFloatValue;
  out->floatValue = op->floatValue;
  out->arrayDims = op->arrayDims;
  for (const auto &child : op->children)
    out->children.push_back(cloneReplacingParams(child.get(), replacements));
  return out;
}

}  // namespace

CanonStats Canonicalizer::run(Module &module) {
  CanonStats stats;
  if (!module.root)
    return stats;

  bool changed = true;
  while (changed) {
    changed = false;
    changed = inlineSimpleAffineCalls(module, stats) || changed;
    std::vector<Op*> stack = { module.root.get() };
    while (!stack.empty()) {
      Op *op = stack.back();
      stack.pop_back();
      changed = foldConstExpr(op, stats) || changed;
      changed = simplifyStructuredControl(op, stats) || changed;
      for (auto it = op->children.rbegin(); it != op->children.rend(); ++it)
        if (it->get())
          stack.push_back(it->get());
    }
  }
  return stats;
}

bool Canonicalizer::inlineSimpleAffineCalls(Module &module, CanonStats &stats) {
  if (!module.root)
    return false;

  std::unordered_map<std::string, SimpleAffineFunction> funcs;
  for (const auto &child : module.root->children) {
    SimpleAffineFunction fn;
    if (findSimpleReturnExpr(child.get(), fn))
      funcs.emplace(child->symbol, fn);
  }
  if (funcs.empty())
    return false;

  bool changed = false;
  std::function<void(std::unique_ptr<Op>&)> visit = [&](std::unique_ptr<Op> &slot) {
    if (!slot)
      return;
    for (auto &child : slot->children)
      visit(child);

    if (slot->kind != OpKind::Call)
      return;
    auto it = funcs.find(slot->symbol);
    if (it == funcs.end())
      return;
    const SimpleAffineFunction &fn = it->second;
    if (slot->children.size() != fn.params.size())
      return;

    std::unordered_map<std::string, const Op*> replacements;
    for (size_t i = 0; i < fn.params.size(); i++)
      replacements.emplace(fn.params[i], slot->children[i].get());
    slot = cloneReplacingParams(fn.expr, replacements);
    stats.simpleAffineCallsInlined++;
    changed = true;
  };

  for (auto &child : module.root->children)
    visit(child);
  return changed;
}

bool Canonicalizer::foldConstExpr(Op *op, CanonStats &stats) {
  if (!op)
    return false;

  if (op->kind != OpKind::Arith && op->kind != OpKind::Cmp)
    return false;
  if (op->children.size() < 2)
    return false;

  Op *lhs = op->children[0].get();
  Op *rhs = op->children[1].get();
  if (!lhs || !rhs)
    return false;

  if (isConstInt(lhs) && isConstInt(rhs)) {
    long long l = lhs->intValue;
    long long r = rhs->intValue;
    bool changed = true;
    if (op->kind == OpKind::Cmp) {
      if (op->symbol == "==")
        rewriteToConstInt(op, l == r);
      else if (op->symbol == "!=")
        rewriteToConstInt(op, l != r);
      else if (op->symbol == "<")
        rewriteToConstInt(op, l < r);
      else if (op->symbol == "<=")
        rewriteToConstInt(op, l <= r);
      else
        changed = false;
    } else {
      if (op->symbol == "+")
        rewriteToConstInt(op, l + r);
      else if (op->symbol == "-")
        rewriteToConstInt(op, l - r);
      else if (op->symbol == "*")
        rewriteToConstInt(op, l * r);
      else if (op->symbol == "/" && r != 0)
        rewriteToConstInt(op, l / r);
      else if (op->symbol == "%" && r != 0)
        rewriteToConstInt(op, l % r);
      else if (op->symbol == "&&")
        rewriteToConstInt(op, (l != 0) && (r != 0));
      else if (op->symbol == "||")
        rewriteToConstInt(op, (l != 0) || (r != 0));
      else
        changed = false;
    }
    if (changed)
      stats.constFolded++;
    return changed;
  }

  if (isConstFloat(lhs) && isConstFloat(rhs)) {
    double l = lhs->floatValue;
    double r = rhs->floatValue;
    bool changed = true;
    if (op->kind == OpKind::Cmp) {
      if (op->symbol == "==")
        rewriteToConstInt(op, l == r);
      else if (op->symbol == "!=")
        rewriteToConstInt(op, l != r);
      else if (op->symbol == "<")
        rewriteToConstInt(op, l < r);
      else if (op->symbol == "<=")
        rewriteToConstInt(op, l <= r);
      else
        changed = false;
    } else {
      if (op->symbol == "+")
        rewriteToConstFloat(op, l + r);
      else if (op->symbol == "-")
        rewriteToConstFloat(op, l - r);
      else if (op->symbol == "*")
        rewriteToConstFloat(op, l * r);
      else if (op->symbol == "/" && std::fabs(r) > 0.0)
        rewriteToConstFloat(op, l / r);
      else
        changed = false;
    }
    if (changed)
      stats.constFolded++;
    return changed;
  }
  return false;
}

bool Canonicalizer::simplifyStructuredControl(Op *op, CanonStats &stats) {
  if (!op)
    return false;

  if (op->kind == OpKind::If && !op->children.empty()) {
    Op *cond = op->children[0].get();
    if (isConstInt(cond) || isConstFloat(cond)) {
      bool truthy = takeTruthyBranch(cond);
      std::unique_ptr<Op> selected;
      if (truthy && op->children.size() >= 2)
        selected = std::move(op->children[1]);
      else if (!truthy && op->children.size() >= 3)
        selected = std::move(op->children[2]);

      op->kind = OpKind::Block;
      op->traits = defaultTraits(OpKind::Block);
      op->symbol.clear();
      op->children.clear();
      if (selected) {
        if (selected->kind == OpKind::Block) {
          for (auto &child : selected->children)
            op->children.push_back(std::move(child));
        } else {
          op->children.push_back(std::move(selected));
        }
      }
      stats.deadBranchesEliminated++;
      return true;
    }
  }

  if (op->kind == OpKind::While && !op->children.empty()) {
    Op *cond = op->children[0].get();
    if ((isConstInt(cond) && cond->intValue == 0) ||
        (isConstFloat(cond) && cond->floatValue == 0.0)) {
      op->kind = OpKind::Block;
      op->traits = defaultTraits(OpKind::Block);
      op->symbol.clear();
      op->children.clear();
      stats.deadBranchesEliminated++;
      return true;
    }
  }

  return false;
}

}  // namespace sys::hir
