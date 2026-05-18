#include "HIRPolyhedral.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sys::hir {

namespace {

constexpr int kJamFactor = 4;

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

std::unique_ptr<Op> makeConstInt(long long value) {
  auto op = std::make_unique<Op>(OpKind::ConstInt);
  op->type = TypeKind::Int;
  op->hasIntValue = true;
  op->intValue = value;
  return op;
}

std::unique_ptr<Op> makeLoad(const std::string &symbol) {
  auto op = std::make_unique<Op>(OpKind::Load);
  op->type = TypeKind::Int;
  op->symbol = symbol;
  return op;
}

std::unique_ptr<Op> makeArith(const std::string &symbol,
                              std::unique_ptr<Op> lhs,
                              std::unique_ptr<Op> rhs) {
  auto op = std::make_unique<Op>(OpKind::Arith);
  op->type = TypeKind::Int;
  op->symbol = symbol;
  op->children.push_back(std::move(lhs));
  op->children.push_back(std::move(rhs));
  return op;
}

std::unique_ptr<Op> makeCmp(const std::string &symbol,
                            std::unique_ptr<Op> lhs,
                            std::unique_ptr<Op> rhs) {
  auto op = std::make_unique<Op>(OpKind::Cmp);
  op->type = TypeKind::Int;
  op->symbol = symbol;
  op->children.push_back(std::move(lhs));
  op->children.push_back(std::move(rhs));
  return op;
}

std::unique_ptr<Op> makeStore(const std::string &symbol, std::unique_ptr<Op> value) {
  auto op = std::make_unique<Op>(OpKind::Store);
  op->symbol = symbol;
  op->children.push_back(std::move(value));
  return op;
}

std::unique_ptr<Op> makeVarDecl(const std::string &symbol, std::unique_ptr<Op> value) {
  auto op = std::make_unique<Op>(OpKind::VarDecl);
  op->type = TypeKind::Int;
  op->symbol = symbol;
  if (value)
    op->children.push_back(std::move(value));
  return op;
}

std::unique_ptr<Op> makeBlock() {
  return std::make_unique<Op>(OpKind::Block);
}

std::unique_ptr<Op> makeWhile(std::unique_ptr<Op> cond, std::unique_ptr<Op> body) {
  auto op = std::make_unique<Op>(OpKind::While);
  op->children.push_back(std::move(cond));
  op->children.push_back(std::move(body));
  return op;
}

std::unique_ptr<Op> makeIndexWithOffset(const std::string &iv, int offset) {
  if (offset == 0)
    return makeLoad(iv);
  return makeArith("+", makeLoad(iv), makeConstInt(offset));
}

bool isScalarLoad(const Op *op, const std::string &symbol) {
  return op && op->kind == OpKind::Load && op->symbol == symbol && op->children.empty();
}

bool isConstIntValue(const Op *op, long long value) {
  return op && op->kind == OpKind::ConstInt && op->hasIntValue && op->intValue == value;
}

const Op *unwrapSingleDecl(const Op *op) {
  if (!op)
    return nullptr;
  if (op->kind == OpKind::Block && op->children.size() == 1)
    return unwrapSingleDecl(op->children[0].get());
  return op;
}

Op *unwrapSingleDecl(Op *op) {
  return const_cast<Op*>(unwrapSingleDecl(static_cast<const Op*>(op)));
}

struct CanonicalLoop {
  std::string iv;
  const Op *bound = nullptr;
  Op *body = nullptr;
  Op *step = nullptr;
};

bool matchStepStore(Op *op, const std::string &iv, int expectedStep) {
  op = unwrapSingleDecl(op);
  if (!op || op->kind != OpKind::Store || op->symbol != iv || op->children.size() != 1)
    return false;
  Op *rhs = op->children[0].get();
  if (!rhs || rhs->kind != OpKind::Arith || rhs->symbol != "+" || rhs->children.size() != 2)
    return false;
  return isScalarLoad(rhs->children[0].get(), iv) &&
         isConstIntValue(rhs->children[1].get(), expectedStep);
}

bool matchCanonicalWhile(Op *op, CanonicalLoop &loop) {
  if (!op || op->kind != OpKind::While || op->children.size() < 2)
    return false;
  Op *cond = op->children[0].get();
  Op *body = op->children[1].get();
  if (!cond || cond->kind != OpKind::Cmp || cond->symbol != "<" || cond->children.size() != 2)
    return false;
  Op *lhs = cond->children[0].get();
  if (!lhs || lhs->kind != OpKind::Load || !lhs->children.empty() || lhs->symbol.empty())
    return false;
  if (!body || body->kind != OpKind::Block || body->children.empty())
    return false;
  Op *step = body->children.back().get();
  if (!matchStepStore(step, lhs->symbol, 1))
    return false;
  loop.iv = lhs->symbol;
  loop.bound = cond->children[1].get();
  loop.body = body;
  loop.step = step;
  return true;
}

bool matchLoopInit(const Op *op, const std::string &iv) {
  op = unwrapSingleDecl(op);
  if (!op)
    return false;
  if (op->kind == OpKind::Store && op->symbol == iv && op->children.size() == 1)
    return true;
  if (op->kind == OpKind::VarDecl && op->symbol == iv && op->children.size() <= 1)
    return true;
  return false;
}

bool hasControlOrCall(const Op *op) {
  if (!op)
    return false;
  switch (op->kind) {
  case OpKind::If:
  case OpKind::For:
  case OpKind::Call:
  case OpKind::Return:
  case OpKind::Break:
  case OpKind::Continue:
    return true;
  default:
    break;
  }
  for (const auto &child : op->children)
    if (hasControlOrCall(child.get()))
      return true;
  return false;
}

bool exprUsesScalar(const Op *op, const std::string &symbol) {
  if (!op)
    return false;
  if (isScalarLoad(op, symbol))
    return true;
  for (const auto &child : op->children)
    if (exprUsesScalar(child.get(), symbol))
      return true;
  return false;
}

bool isDirectScalarExpr(const Op *op, const std::string &symbol) {
  return isScalarLoad(op, symbol);
}

bool arrayIndexUsesOnlyDirectIV(const Op *op, const std::string &iv) {
  if (!op)
    return true;
  if ((op->kind == OpKind::Load || op->kind == OpKind::Store) && op->children.size() > 1) {
    size_t indexCount = op->kind == OpKind::Store ? op->children.size() - 1 : op->children.size();
    for (size_t i = 0; i < indexCount; i++) {
      const Op *idx = op->children[i].get();
      if (exprUsesScalar(idx, iv) && !isDirectScalarExpr(idx, iv))
        return false;
    }
  }
  for (const auto &child : op->children)
    if (!arrayIndexUsesOnlyDirectIV(child.get(), iv))
      return false;
  return true;
}

bool isAdditiveReductionUpdate(const Op *store, const std::string &acc) {
  if (!store || store->kind != OpKind::Store || store->symbol != acc || store->children.size() != 1)
    return false;
  const Op *rhs = store->children[0].get();
  if (!rhs || rhs->kind != OpKind::Arith || rhs->symbol != "+" || rhs->children.size() != 2)
    return false;
  return isScalarLoad(rhs->children[0].get(), acc) || isScalarLoad(rhs->children[1].get(), acc);
}

bool containsArrayAccessTo(const Op *op, const std::string &symbol) {
  if (!op)
    return false;
  if ((op->kind == OpKind::Load || op->kind == OpKind::Store) &&
      op->symbol == symbol && !op->children.empty())
    return true;
  for (const auto &child : op->children)
    if (containsArrayAccessTo(child.get(), symbol))
      return true;
  return false;
}

bool sameArrayDestination(const Op *lhs, const Op *rhs) {
  if (!lhs || !rhs || lhs->kind != OpKind::Store || rhs->kind != OpKind::Store)
    return false;
  if (lhs->symbol != rhs->symbol || lhs->children.empty() || rhs->children.empty())
    return false;
  size_t lhsIndexCount = lhs->children.size() - 1;
  size_t rhsIndexCount = rhs->children.size() - 1;
  if (lhsIndexCount != rhsIndexCount)
    return false;
  for (size_t i = 0; i < lhsIndexCount; i++) {
    const Op *a = lhs->children[i].get();
    const Op *b = rhs->children[i].get();
    if (!a || !b || a->kind != b->kind || a->symbol != b->symbol ||
        a->hasIntValue != b->hasIntValue || a->intValue != b->intValue ||
        a->children.size() != b->children.size())
      return false;
  }
  return true;
}

std::unique_ptr<Op> makeLoadFromStoreDestination(const Op *store) {
  if (!store || store->kind != OpKind::Store || store->children.empty())
    return nullptr;
  auto load = std::make_unique<Op>(OpKind::Load, store->origin);
  load->type = TypeKind::Int;
  load->symbol = store->symbol;
  for (size_t i = 0; i + 1 < store->children.size(); i++)
    load->children.push_back(cloneOp(store->children[i].get()));
  return load;
}

std::unique_ptr<Op> cloneReplacingScalarLoad(const Op *op, const std::string &scalar,
                                             const Op *replacement) {
  if (!op)
    return nullptr;
  if (isScalarLoad(op, scalar))
    return cloneOp(replacement);

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
    out->children.push_back(cloneReplacingScalarLoad(child.get(), scalar, replacement));
  return out;
}

std::unique_ptr<Op> cloneReplacing(const Op *op,
                                   const std::unordered_map<std::string, std::string> &scalarRenames,
                                   const std::unordered_map<std::string, int> &ivOffsets) {
  if (!op)
    return nullptr;
  if (op->kind == OpKind::Load && op->children.empty()) {
    auto scalarIt = scalarRenames.find(op->symbol);
    if (scalarIt != scalarRenames.end())
      return makeLoad(scalarIt->second);
    auto ivIt = ivOffsets.find(op->symbol);
    if (ivIt != ivOffsets.end())
      return makeIndexWithOffset(op->symbol, ivIt->second);
  }

  auto out = std::make_unique<Op>(op->kind, op->origin);
  out->type = op->type;
  out->traits = op->traits;
  out->symbol = op->symbol;
  auto scalarIt = scalarRenames.find(out->symbol);
  if ((out->kind == OpKind::Store || out->kind == OpKind::VarDecl) && scalarIt != scalarRenames.end())
    out->symbol = scalarIt->second;
  out->hasIntValue = op->hasIntValue;
  out->intValue = op->intValue;
  out->hasFloatValue = op->hasFloatValue;
  out->floatValue = op->floatValue;
  out->arrayDims = op->arrayDims;
  for (const auto &child : op->children)
    out->children.push_back(cloneReplacing(child.get(), scalarRenames, ivOffsets));
  return out;
}

struct ReductionPattern {
  std::string j;
  std::string k;
  std::string acc;
  const Op *jBound = nullptr;
  const Op *kInit = nullptr;
  const Op *accInit = nullptr;
  Op *kWhile = nullptr;
  Op *accUpdate = nullptr;
  Op *destStore = nullptr;
  Op *jStep = nullptr;
};

bool matchVarInit(const Op *op, std::string &symbol) {
  op = unwrapSingleDecl(op);
  if (!op)
    return false;
  if (op->kind == OpKind::VarDecl && !op->symbol.empty() && op->children.size() <= 1) {
    symbol = op->symbol;
    return true;
  }
  if (op->kind == OpKind::Store && !op->symbol.empty() && op->children.size() == 1) {
    symbol = op->symbol;
    return true;
  }
  return false;
}

const Op *initValue(const Op *op) {
  op = unwrapSingleDecl(op);
  if (!op || op->children.empty())
    return nullptr;
  return op->children[0].get();
}

bool matchReductionPattern(Op *jWhile, ReductionPattern &pat) {
  CanonicalLoop jLoop;
  if (!matchCanonicalWhile(jWhile, jLoop))
    return false;
  pat.j = jLoop.iv;
  pat.jBound = jLoop.bound;
  pat.jStep = jLoop.step;

  Op *body = jLoop.body;
  if (!body || body->children.size() != 5)
    return false;
  if (hasControlOrCall(body))
    return false;

  std::string kSym;
  std::string accSym;
  if (!matchVarInit(body->children[0].get(), kSym))
    return false;
  if (!matchVarInit(body->children[1].get(), accSym))
    return false;

  CanonicalLoop kLoop;
  if (!matchCanonicalWhile(body->children[2].get(), kLoop) || kLoop.iv != kSym)
    return false;
  Op *kBody = kLoop.body;
  if (!kBody || kBody->children.size() != 2)
    return false;

  Op *accUpdate = unwrapSingleDecl(kBody->children[0].get());
  if (!accUpdate || accUpdate->kind != OpKind::Store || accUpdate->symbol != accSym ||
      accUpdate->children.size() != 1)
    return false;
  if (!isAdditiveReductionUpdate(accUpdate, accSym))
    return false;
  if (!arrayIndexUsesOnlyDirectIV(accUpdate, jLoop.iv))
    return false;
  if (!matchStepStore(kBody->children[1].get(), kSym, 1))
    return false;

  Op *destStore = unwrapSingleDecl(body->children[3].get());
  if (!destStore || destStore->kind != OpKind::Store || destStore->children.size() < 2)
    return false;
  if (!arrayIndexUsesOnlyDirectIV(destStore, jLoop.iv))
    return false;
  const Op *storedValue = destStore->children.back().get();
  if (!isScalarLoad(storedValue, accSym))
    return false;
  if (!matchStepStore(body->children[4].get(), jLoop.iv, 1))
    return false;

  pat.k = kSym;
  pat.acc = accSym;
  pat.kInit = body->children[0].get();
  pat.accInit = body->children[1].get();
  pat.kWhile = body->children[2].get();
  pat.accUpdate = accUpdate;
  pat.destStore = destStore;
  return true;
}

std::unique_ptr<Op> makeJamStep(const std::string &iv) {
  return makeStore(iv, makeArith("+", makeLoad(iv), makeConstInt(kJamFactor)));
}

}  // namespace

PolyhedralStats PolyhedralOptimizer::run(Module &module) {
  PolyhedralStats stats;
  if (!module.root)
    return stats;
  optimizeBlock(module.root.get(), stats);
  return stats;
}

bool PolyhedralOptimizer::optimizeBlock(Op *block, PolyhedralStats &stats) {
  if (!block)
    return false;

  bool changed = false;
  for (auto &child : block->children)
    changed = optimizeBlock(child.get(), stats) || changed;

  if (block->kind != OpKind::Block && block->kind != OpKind::Module)
    return changed;

  for (size_t i = 0; i + 1 < block->children.size(); i++) {
    if (tryReductionJam(block, i, stats)) {
      changed = true;
      i = 0;
    }
  }
  return changed;
}

bool PolyhedralOptimizer::tryReductionJam(Op *block, size_t initIndex, PolyhedralStats &stats) {
  if (!block || initIndex + 1 >= block->children.size())
    return false;

  Op *whileOp = block->children[initIndex + 1].get();
  ReductionPattern pat;
  if (!matchReductionPattern(whileOp, pat)) {
    stats.rejected++;
    return false;
  }
  if (!matchLoopInit(block->children[initIndex].get(), pat.j)) {
    stats.rejected++;
    return false;
  }

  const std::string prefix = "__poly_" + pat.acc + "_" + std::to_string(uniqueId++);
  std::vector<std::string> accNames;
  accNames.reserve(kJamFactor);
  for (int lane = 0; lane < kJamFactor; lane++)
    accNames.push_back(prefix + "_" + std::to_string(lane));

  auto vecBody = makeBlock();
  for (int lane = 0; lane < kJamFactor; lane++) {
    vecBody->children.push_back(makeVarDecl(accNames[lane], cloneOp(initValue(pat.accInit))));
  }

  vecBody->children.push_back(cloneOp(pat.kInit));

  CanonicalLoop kLoop;
  if (!matchCanonicalWhile(pat.kWhile, kLoop))
    return false;

  auto kBody = makeBlock();
  for (int lane = 0; lane < kJamFactor; lane++) {
    std::unordered_map<std::string, std::string> scalarRenames = { { pat.acc, accNames[lane] } };
    std::unordered_map<std::string, int> ivOffsets = { { pat.j, lane } };
    auto laneUpdate = cloneReplacing(pat.accUpdate, scalarRenames, ivOffsets);
    if (!laneUpdate || laneUpdate->kind != OpKind::Store)
      return false;
    laneUpdate->symbol = accNames[lane];
    kBody->children.push_back(std::move(laneUpdate));
  }
  kBody->children.push_back(cloneOp(kLoop.step));

  auto kWhile = makeWhile(cloneOp(pat.kWhile->children[0].get()), std::move(kBody));
  vecBody->children.push_back(std::move(kWhile));

  for (int lane = 0; lane < kJamFactor; lane++) {
    std::unordered_map<std::string, std::string> scalarRenames = { { pat.acc, accNames[lane] } };
    std::unordered_map<std::string, int> ivOffsets = { { pat.j, lane } };
    vecBody->children.push_back(cloneReplacing(pat.destStore, scalarRenames, ivOffsets));
  }
  vecBody->children.push_back(makeJamStep(pat.j));

  auto vecCond = makeCmp("<",
                         makeArith("+", makeLoad(pat.j), makeConstInt(kJamFactor - 1)),
                         cloneOp(pat.jBound));
  auto vecWhile = makeWhile(std::move(vecCond), std::move(vecBody));
  auto tailWhile = cloneOp(whileOp);

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == initIndex + 1) {
      replacement.push_back(std::move(vecWhile));
      replacement.push_back(std::move(tailWhile));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.reductionJammed++;
  return true;
}

}  // namespace sys::hir
