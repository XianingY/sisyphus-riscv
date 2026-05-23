#include "HIRPolyhedral.h"

#include "HIRAffine.h"
#include "../backend/riscv/RiscvParams.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sys::hir {

namespace {

struct LoopBodyMetrics {
  int scalarIntDefs = 0;
  int scalarFloatDefs = 0;
  std::unordered_set<std::string> arrayReadStreams;
  std::unordered_set<std::string> arrayWriteStreams;
};

void analyzeLoopBody(const Op *op, LoopBodyMetrics &metrics) {
  if (!op) return;

  if (op->kind == OpKind::VarDecl && op->arrayDims.empty()) {
    if (op->type == TypeKind::Float) {
      metrics.scalarFloatDefs++;
    } else {
      metrics.scalarIntDefs++;
    }
  }

  const bool isArrayLoad = (op->kind == OpKind::Load && !op->children.empty());
  const bool isArrayStore = (op->kind == OpKind::Store && op->children.size() > 1);
  if (isArrayLoad && !op->symbol.empty()) {
    metrics.arrayReadStreams.insert(op->symbol);
  }
  if (isArrayStore && !op->symbol.empty()) {
    metrics.arrayWriteStreams.insert(op->symbol);
  }

  for (const auto &child : op->children) {
    analyzeLoopBody(child.get(), metrics);
  }
}

TypeKind detectMainType(const Op *op) {
  if (!op) return TypeKind::Int;
  if (op->type == TypeKind::Float) return TypeKind::Float;
  if (op->kind == OpKind::ConstFloat) return TypeKind::Float;
  for (const auto &child : op->children) {
    if (detectMainType(child.get()) == TypeKind::Float)
      return TypeKind::Float;
  }
  return TypeKind::Int;
}

int computeOptimalJamFactor(const Op *innerBody, TypeKind mainType) {
  const char *envRaw = std::getenv("SISY_HIR_JAM_FACTOR");
  if (envRaw && envRaw[0]) {
    int val = std::atoi(envRaw);
    if (val >= 2 && val <= 16) {
      return val;
    }
  }
  using namespace sys::backend::riscv;

  LoopBodyMetrics metrics;
  analyzeLoopBody(innerBody, metrics);

  int activeStreams = metrics.arrayReadStreams.size() + metrics.arrayWriteStreams.size();
  if (activeStreams == 0) activeStreams = 1;

  int usableRegs = (mainType == TypeKind::Float) ? kUsableFPRs : kUsableGPRs;
  int activeScalars = (mainType == TypeKind::Float) ? metrics.scalarFloatDefs : metrics.scalarIntDefs;

  int bestFactor = 4;
  for (int factor : {8, 4, 2}) {
    int estimatedPressure = (factor * activeStreams) + activeScalars;
    if (estimatedPressure <= usableRegs) {
      bestFactor = factor;
      break;
    }
  }
  return bestFactor;
}

int computeOptimalTileSize(TypeKind mainType) {
  const char *envRaw = std::getenv("SISY_HIR_TILE_SIZE");
  if (envRaw && envRaw[0]) {
    int val = std::atoi(envRaw);
    if (val >= 4) {
      return val;
    }
  }
  using namespace sys::backend::riscv;

  int elementSize = (mainType == TypeKind::Float || mainType == TypeKind::Int) ? 4 : 8;
  int elementsPerCacheLine = kCacheLineSize / elementSize;
  
  return elementsPerCacheLine * 2;
}

constexpr int kJamFactor = 4;
constexpr int kDefaultHirTileSize = 32;

bool hirEnvEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

int hirEnvInt(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  int v = std::atoi(raw);
  return v > 0 ? v : fallback;
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

std::unique_ptr<Op> makeArrayStoreLike(const Op *store, std::unique_ptr<Op> value) {
  if (!store || store->kind != OpKind::Store || store->children.empty())
    return nullptr;
  auto out = std::make_unique<Op>(OpKind::Store, store->origin);
  out->symbol = store->symbol;
  out->type = store->type;
  out->traits = store->traits;
  out->arrayDims = store->arrayDims;
  for (size_t i = 0; i + 1 < store->children.size(); i++)
    out->children.push_back(cloneOp(store->children[i].get()));
  out->children.push_back(std::move(value));
  return out;
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
  return (isScalarLoad(rhs->children[0].get(), iv) &&
          isConstIntValue(rhs->children[1].get(), expectedStep)) ||
         (isConstIntValue(rhs->children[0].get(), expectedStep) &&
          isScalarLoad(rhs->children[1].get(), iv));
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
  const bool isArrayLoad = op->kind == OpKind::Load && !op->children.empty();
  const bool isArrayStore = op->kind == OpKind::Store && op->children.size() > 1;
  if ((isArrayLoad || isArrayStore) && op->symbol == symbol)
    return true;
  for (const auto &child : op->children)
    if (containsArrayAccessTo(child.get(), symbol))
      return true;
  return false;
}

void collectArrayAccessSymbols(const Op *op, std::unordered_set<std::string> &symbols) {
  if (!op)
    return;
  const bool isArrayLoad = op->kind == OpKind::Load && !op->children.empty();
  const bool isArrayStore = op->kind == OpKind::Store && op->children.size() > 1;
  if ((isArrayLoad || isArrayStore) && !op->symbol.empty())
    symbols.insert(op->symbol);
  for (const auto &child : op->children)
    collectArrayAccessSymbols(child.get(), symbols);
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
  const Op *kBound = nullptr;
  Op *kWhile = nullptr;
  Op *kStep = nullptr;
  Op *kReductionStmt = nullptr;
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

bool hasUnsafeReductionControl(const Op *op) {
  if (!op)
    return false;
  switch (op->kind) {
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
    if (hasUnsafeReductionControl(child.get()))
      return true;
  return false;
}

bool containsWhile(const Op *op) {
  if (!op)
    return false;
  if (op->kind == OpKind::While)
    return true;
  for (const auto &child : op->children)
    if (containsWhile(child.get()))
      return true;
  return false;
}

Op *findAdditiveReductionUpdate(Op *stmt, const std::string &acc) {
  stmt = unwrapSingleDecl(stmt);
  if (!stmt)
    return nullptr;
  if (stmt->kind == OpKind::Store)
    return isAdditiveReductionUpdate(stmt, acc) ? stmt : nullptr;
  if (stmt->kind != OpKind::If || stmt->children.size() != 2)
    return nullptr;
  Op *thenBlock = stmt->children[1].get();
  if (!thenBlock || thenBlock->kind != OpKind::Block || thenBlock->children.size() != 1)
    return nullptr;
  return findAdditiveReductionUpdate(thenBlock->children[0].get(), acc);
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
  if (hasUnsafeReductionControl(body))
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

  Op *reductionStmt = unwrapSingleDecl(kBody->children[0].get());
  Op *accUpdate = findAdditiveReductionUpdate(reductionStmt, accSym);
  if (!accUpdate)
    return false;
  if (!arrayIndexUsesOnlyDirectIV(reductionStmt, jLoop.iv))
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
  pat.kBound = kLoop.bound;
  pat.kWhile = body->children[2].get();
  pat.kStep = kLoop.step;
  pat.kReductionStmt = reductionStmt;
  pat.accUpdate = accUpdate;
  pat.destStore = destStore;
  return true;
}

bool strictReductionInterchangeLegal(const ReductionPattern &pat,
                                     const std::unordered_set<std::string> &globalArrays) {
  if (!pat.destStore || pat.destStore->children.size() < 2)
    return false;
  if (!pat.accUpdate || pat.accUpdate->children.size() != 1 || !pat.kReductionStmt)
    return false;
  if (!globalArrays.count(pat.destStore->symbol))
    return false;

  // Strict mode: do not interchange an in-place reduction. If the destination
  // array is read while computing the reduction, swapping k outside j changes
  // the order of visible writes (many_mat_cal has exactly this hazard).
  if (containsArrayAccessTo(pat.kReductionStmt, pat.destStore->symbol))
    return false;

  std::unordered_set<std::string> rhsArrays;
  collectArrayAccessSymbols(pat.kReductionStmt, rhsArrays);
  for (const auto &symbol : rhsArrays)
    if (!globalArrays.count(symbol))
      return false;

  if (exprUsesScalar(pat.destStore, pat.k))
    return false;
  return true;
}

std::unique_ptr<Op> makeJamStep(const std::string &iv, int factor) {
  return makeStore(iv, makeArith("+", makeLoad(iv), makeConstInt(factor)));
}

std::unique_ptr<Op> makeReductionInitLoop(const ReductionPattern &pat) {
  auto body = makeBlock();
  auto init = cloneOp(initValue(pat.accInit));
  if (!init)
    init = makeConstInt(0);
  auto initStore = makeArrayStoreLike(pat.destStore, std::move(init));
  if (!initStore)
    return nullptr;
  body->children.push_back(std::move(initStore));
  body->children.push_back(cloneOp(pat.jStep));
  auto cond = makeCmp("<", makeLoad(pat.j), cloneOp(pat.jBound));
  return makeWhile(std::move(cond), std::move(body));
}

std::unique_ptr<Op> makeReductionLmsStore(const ReductionPattern &pat) {
  if (!pat.accUpdate || !pat.destStore || pat.accUpdate->children.size() != 1)
    return nullptr;
  auto oldValue = makeLoadFromStoreDestination(pat.destStore);
  if (!oldValue)
    return nullptr;
  auto nextValue = cloneReplacingScalarLoad(pat.accUpdate->children[0].get(),
                                            pat.acc,
                                            oldValue.get());
  if (!nextValue)
    return nullptr;
  return makeArrayStoreLike(pat.destStore, std::move(nextValue));
}

std::unique_ptr<Op> cloneReductionStmtToDestination(const Op *op,
                                                    const ReductionPattern &pat) {
  if (!op)
    return nullptr;
  if (isScalarLoad(op, pat.acc))
    return makeLoadFromStoreDestination(pat.destStore);
  if (op->kind == OpKind::Store && op->symbol == pat.acc && op->children.size() == 1) {
    auto oldValue = makeLoadFromStoreDestination(pat.destStore);
    if (!oldValue)
      return nullptr;
    auto nextValue = cloneReplacingScalarLoad(op->children[0].get(), pat.acc,
                                              oldValue.get());
    if (!nextValue)
      return nullptr;
    return makeArrayStoreLike(pat.destStore, std::move(nextValue));
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
    out->children.push_back(cloneReductionStmtToDestination(child.get(), pat));
  return out;
}

std::unique_ptr<Op> makeReductionLmsJLoop(const ReductionPattern &pat) {
  auto body = makeBlock();
  auto lms = (pat.kReductionStmt == pat.accUpdate)
                 ? makeReductionLmsStore(pat)
                 : cloneReductionStmtToDestination(pat.kReductionStmt, pat);
  if (!lms)
    return nullptr;
  body->children.push_back(std::move(lms));
  body->children.push_back(cloneOp(pat.jStep));
  auto cond = makeCmp("<", makeLoad(pat.j), cloneOp(pat.jBound));
  return makeWhile(std::move(cond), std::move(body));
}

std::unique_ptr<Op> makeReductionInterchangedKLoop(const ReductionPattern &pat,
                                                   const Op *jInit) {
  if (!jInit || !pat.kWhile || !pat.kStep)
    return nullptr;
  auto body = makeBlock();
  body->children.push_back(cloneOp(jInit));
  auto jLoop = makeReductionLmsJLoop(pat);
  if (!jLoop)
    return nullptr;
  body->children.push_back(std::move(jLoop));
  body->children.push_back(cloneOp(pat.kStep));
  auto cond = cloneOp(pat.kWhile->children[0].get());
  if (!cond)
    return nullptr;
  return makeWhile(std::move(cond), std::move(body));
}

// ===========================================================================
// Loop Tiling Helpers
// ===========================================================================

// Check whether an Op tree contains any control-flow altering ops (Call,
// Return, Break, Continue) that would make loop tiling unsafe.
bool tilingSafeBody(const Op *op) {
  if (!op)
    return true;
  switch (op->kind) {
  case OpKind::Call:
  case OpKind::Return:
  case OpKind::Break:
  case OpKind::Continue:
    return false;
  default:
    break;
  }
  for (const auto &child : op->children)
    if (!tilingSafeBody(child.get()))
      return false;
  return true;
}

// Returns true if the two bound expressions are syntactically identical
// (conservative but sufficient for most cases).
bool boundsEqual(const Op *a, const Op *b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  if (a->kind != b->kind)
    return false;
  if (a->hasIntValue && b->hasIntValue)
    return a->intValue == b->intValue;
  if (a->hasFloatValue && b->hasFloatValue)
    return a->floatValue == b->floatValue;
  if (a->symbol != b->symbol)
    return false;
  if (a->children.size() != b->children.size())
    return false;
  for (size_t i = 0; i < a->children.size(); i++)
    if (!boundsEqual(a->children[i].get(), b->children[i].get()))
      return false;
  return true;
}

bool isArrayStore(const Op *op) {
  op = unwrapSingleDecl(op);
  return op && op->kind == OpKind::Store && op->children.size() > 1 &&
         !op->symbol.empty();
}

Op *asArrayStore(Op *op) {
  op = unwrapSingleDecl(op);
  return isArrayStore(op) ? op : nullptr;
}

Op *asScalarVarDecl(Op *op) {
  op = unwrapSingleDecl(op);
  return op && op->kind == OpKind::VarDecl && op->arrayDims.empty() &&
         op->children.size() == 1 ? op : nullptr;
}

bool sameArrayAddress(const Op *store, const Op *load) {
  if (!store || !load || store->kind != OpKind::Store ||
      load->kind != OpKind::Load || store->symbol != load->symbol)
    return false;
  if (store->children.size() != load->children.size() + 1)
    return false;
  for (size_t i = 0; i < load->children.size(); i++)
    if (!boundsEqual(store->children[i].get(), load->children[i].get()))
      return false;
  return true;
}

// Two loops can be fused when B does not depend on scalars exclusively defined by A.
bool collectDefinedScalars(const Op *block, std::unordered_set<std::string> &syms) {
  if (!block)
    return true;
  for (const auto &child : block->children) {
    const Op *op = child.get();
    if (!op)
      continue;
    if ((op->kind == OpKind::VarDecl || op->kind == OpKind::Store) &&
        !op->symbol.empty() && op->arrayDims.empty())
      syms.insert(op->symbol);
    collectDefinedScalars(op, syms);
  }
  return true;
}

void collectTopLevelInitializedScalars(const Op *block, std::unordered_set<std::string> &syms) {
  if (!block || block->kind != OpKind::Block)
    return;
  for (const auto &child : block->children) {
    const Op *op = unwrapSingleDecl(child.get());
    if (!op || op->symbol.empty())
      continue;
    if (op->kind == OpKind::VarDecl) {
      if (op->children.empty() || !exprUsesScalar(op->children[0].get(), op->symbol))
        syms.insert(op->symbol);
      continue;
    }
    if (op->kind == OpKind::Store && op->children.size() == 1) {
      if (!exprUsesScalar(op->children[0].get(), op->symbol))
        syms.insert(op->symbol);
    }
  }
}

bool bodyUsesAnyOf(const Op *op, const std::unordered_set<std::string> &syms) {
  if (!op)
    return false;
  if (op->kind == OpKind::Load && op->children.empty() && syms.count(op->symbol))
    return true;
  for (const auto &child : op->children)
    if (bodyUsesAnyOf(child.get(), syms))
      return true;
  return false;
}

bool bodyWritesScalar(const Op *op, const std::string &symbol) {
  if (!op)
    return false;
  if ((op->kind == OpKind::VarDecl ||
       (op->kind == OpKind::Store && op->children.size() == 1)) &&
      op->symbol == symbol)
    return true;
  for (const auto &child : op->children)
    if (bodyWritesScalar(child.get(), symbol))
      return true;
  return false;
}

bool bodyWritesNonLoopScalar(const Op *op, const std::unordered_set<std::string> &loopIVs) {
  if (!op)
    return false;
  const bool scalarDecl = op->kind == OpKind::VarDecl && op->arrayDims.empty();
  const bool scalarStore = op->kind == OpKind::Store && op->children.size() == 1;
  if ((scalarDecl || scalarStore) && !op->symbol.empty() && !loopIVs.count(op->symbol))
    return true;
  for (const auto &child : op->children)
    if (bodyWritesNonLoopScalar(child.get(), loopIVs))
      return true;
  return false;
}

bool blockExceptLastWritesScalar(const Op *block, const std::string &symbol) {
  if (!block || block->kind != OpKind::Block || block->children.empty())
    return false;
  for (size_t i = 0; i + 1 < block->children.size(); i++)
    if (bodyWritesScalar(block->children[i].get(), symbol))
      return true;
  return false;
}

std::unique_ptr<Op> cloneBlockWithoutLast(const Op *block) {
  if (!block || block->kind != OpKind::Block || block->children.empty())
    return nullptr;
  auto out = makeBlock();
  for (size_t i = 0; i + 1 < block->children.size(); i++)
    out->children.push_back(cloneOp(block->children[i].get()));
  return out;
}

bool collectCanonicalLoopIVs(const Op *op, std::unordered_set<std::string> &ivs) {
  if (!op)
    return true;
  if (op->kind == OpKind::While) {
    affine::CanonicalLoop loop;
    if (!affine::matchCanonicalLoop(op, loop))
      return false;
    ivs.insert(loop.iv);
    return collectCanonicalLoopIVs(loop.body, ivs);
  }
  for (const auto &child : op->children)
    if (!collectCanonicalLoopIVs(child.get(), ivs))
      return false;
  return true;
}

const Op *additiveDeltaExpr(const Op *store, const std::string &acc) {
  if (!store || store->kind != OpKind::Store || store->symbol != acc ||
      store->children.size() != 1)
    return nullptr;
  const Op *rhs = store->children[0].get();
  if (!rhs || rhs->kind != OpKind::Arith || rhs->symbol != "+" ||
      rhs->children.size() != 2)
    return nullptr;
  if (isScalarLoad(rhs->children[0].get(), acc))
    return rhs->children[1].get();
  if (isScalarLoad(rhs->children[1].get(), acc))
    return rhs->children[0].get();
  return nullptr;
}

bool repeatBodyLegalImpl(const Op *op, const std::unordered_set<std::string> &loopIVs,
                         const std::string &repeatIV, std::string &acc,
                         int &accUpdates) {
  if (!op)
    return true;

  switch (op->kind) {
  case OpKind::Call:
  case OpKind::Return:
  case OpKind::Break:
  case OpKind::Continue:
  case OpKind::If:
  case OpKind::For:
    return false;
  default:
    break;
  }

  if (op->kind == OpKind::Store && op->children.size() > 1) {
    // Repeating a loop body once is only equivalent when the repeated body is
    // read-only with respect to arrays. Scalar reductions are handled below.
    return false;
  }

  if (op->kind == OpKind::VarDecl && !op->symbol.empty() && op->arrayDims.empty()) {
    if (!loopIVs.count(op->symbol))
      return false;
  }

  if (op->kind == OpKind::Store && op->children.size() == 1 && !op->symbol.empty()) {
    if (loopIVs.count(op->symbol)) {
      if (op->symbol == repeatIV && !matchStepStore(const_cast<Op*>(op), repeatIV, 1))
        return false;
    } else {
      const Op *delta = additiveDeltaExpr(op, acc.empty() ? op->symbol : acc);
      if (!delta)
        return false;
      if (!acc.empty() && op->symbol != acc)
        return false;
      if (exprUsesScalar(delta, repeatIV))
        return false;
      acc = op->symbol;
      accUpdates++;
    }
  }

  for (const auto &child : op->children)
    if (!repeatBodyLegalImpl(child.get(), loopIVs, repeatIV, acc, accUpdates))
      return false;
  return true;
}

bool repeatBodyLegal(const Op *body, const std::string &repeatIV, std::string &acc) {
  if (!body || body->kind != OpKind::Block || body->children.empty())
    return false;
  std::unordered_set<std::string> loopIVs = {repeatIV};
  if (!collectCanonicalLoopIVs(body, loopIVs))
    return false;
  int accUpdates = 0;
  if (!repeatBodyLegalImpl(body, loopIVs, repeatIV, acc, accUpdates))
    return false;
  return !acc.empty() && accUpdates > 0;
}

bool blockExceptLastUsesScalar(const Op *block, const std::string &symbol) {
  if (!block || block->kind != OpKind::Block || block->children.empty())
    return false;
  for (size_t i = 0; i + 1 < block->children.size(); i++)
    if (exprUsesScalar(block->children[i].get(), symbol))
      return true;
  return false;
}

bool hasUnsafeAffineScanControl(const Op *op) {
  if (!op)
    return false;
  switch (op->kind) {
  case OpKind::Call:
  case OpKind::Return:
  case OpKind::Break:
  case OpKind::Continue:
    return true;
  default:
    break;
  }
  for (const auto &child : op->children)
    if (hasUnsafeAffineScanControl(child.get()))
      return true;
  return false;
}

int countArrayAccessOps(const Op *op) {
  if (!op)
    return 0;
  int count = 0;
  const bool isArrayLoad = op->kind == OpKind::Load && !op->children.empty();
  const bool isArrayStore = op->kind == OpKind::Store && op->children.size() > 1;
  if ((isArrayLoad || isArrayStore) && !op->symbol.empty())
    count++;
  for (const auto &child : op->children)
    count += countArrayAccessOps(child.get());
  return count;
}

Op *findSingleDirectInnerWhile(Op *body) {
  if (!body || body->kind != OpKind::Block || body->children.size() < 2)
    return nullptr;
  Op *inner = nullptr;
  for (size_t i = 0; i + 1 < body->children.size(); i++) {
    Op *candidate = unwrapSingleDecl(body->children[i].get());
    if (!candidate || candidate->kind != OpKind::While)
      continue;
    if (inner)
      return nullptr;
    inner = candidate;
  }
  return inner;
}

bool accessMentions(const affine::Access &access, const std::string &symbol) {
  for (const auto &idx : access.indices)
    if (idx.coeffs.count(symbol))
      return true;
  return false;
}

bool accessMentionsPair(const affine::Access &access, const std::string &a,
                        const std::string &b) {
  bool hasA = false;
  bool hasB = false;
  for (const auto &idx : access.indices) {
    hasA = hasA || idx.coeffs.count(a);
    hasB = hasB || idx.coeffs.count(b);
  }
  return hasA && hasB;
}

bool isMatmulLikeNest(const std::vector<CanonicalLoop> &loops,
                      const std::vector<affine::Access> &accesses) {
  if (loops.size() < 3)
    return false;
  const std::string &i = loops[0].iv;
  const std::string &j = loops[1].iv;
  const std::string &k = loops[2].iv;
  bool hasDestStore = false;
  bool hasIKLoad = false;
  bool hasKJLoad = false;
  for (const auto &access : accesses) {
    if (access.indices.size() < 2)
      continue;
    if (access.isStore && accessMentionsPair(access, i, j) && !accessMentions(access, k))
      hasDestStore = true;
    if (!access.isStore && accessMentionsPair(access, i, k))
      hasIKLoad = true;
    if (!access.isStore && accessMentionsPair(access, k, j))
      hasKJLoad = true;
  }
  return hasDestStore && hasIKLoad && hasKJLoad;
}

bool fusionWithinCacheBudget(const Op *bodyA, const Op *bodyB) {
  using namespace sys::backend::riscv;

  std::unordered_set<std::string> arraysA, arraysB;
  collectArrayAccessSymbols(bodyA, arraysA);
  collectArrayAccessSymbols(bodyB, arraysB);

  std::unordered_set<std::string> allArrays;
  for (const auto &s : arraysA) allArrays.insert(s);
  for (const auto &s : arraysB) allArrays.insert(s);

  // Each active array stream needs cache-line residency.  Conservatively
  // assume 3 cache lines per stream at steady state.
  int activeStreams = (int)allArrays.size();
  int estimatedFootprint = activeStreams * kCacheLineSize * 3;

  return estimatedFootprint <= kL1DataCacheSize;
}

}  // namespace

PolyhedralStats PolyhedralOptimizer::run(Module &module) {
  PolyhedralStats stats;
  if (!module.root)
    return stats;
  hirTileSize = hirEnvInt("SISY_HIR_TILE_SIZE", kDefaultHirTileSize);
  if (hirTileSize < 4)
    hirTileSize = 4;
  hirJamFactor = hirEnvInt("SISY_HIR_JAM_FACTOR", kJamFactor);
  if (hirJamFactor < 2)
    hirJamFactor = 2;
  if (hirJamFactor > 16)
    hirJamFactor = 16;
  globalArrays.clear();
  for (const auto &child : module.root->children) {
    const Op *decl = unwrapSingleDecl(child.get());
    if (decl && decl->kind == OpKind::VarDecl && !decl->symbol.empty() &&
        !decl->arrayDims.empty())
      globalArrays.insert(decl->symbol);
  }
  optimizeBlock(module.root.get(), stats);
  return stats;
}

bool PolyhedralOptimizer::optimizeBlock(Op *block, PolyhedralStats &stats) {
  if (!block)
    return false;

  bool changed = false;
  const bool isBlockLike = block->kind == OpKind::Block || block->kind == OpKind::Module;
  if (isBlockLike) {
    for (auto &child : block->children)
      if (child && child->kind == OpKind::While)
        scanAffineNest(child.get(), stats);

    // Try fully affine 3D interchange before visiting nested blocks. Some
    // lower-level reduction rewrites intentionally change matrix-like nests
    // into a different shape, which would otherwise hide the original
    // dependence directions from the generic permutation legality check.
    if (hirEnvEnabled("SISY_HIR_ENABLE_INTERCHANGE", true)) {
      for (size_t i = 0; i < block->children.size(); i++) {
        if (block->children[i] && block->children[i]->kind == OpKind::While &&
            tryLoopInterchange3D(block, i, stats)) {
          changed = true;
          i = 0;
        }
      }
    }
  }

  for (auto &child : block->children)
    changed = optimizeBlock(child.get(), stats) || changed;

  if (block->kind != OpKind::Block && block->kind != OpKind::Module)
    return changed;

  for (size_t i = 0; i < block->children.size(); i++) {
    if (tryReductionInterchange(block, i, stats)) {
      changed = true;
      i = 0;
      continue;
    }
    if (tryReductionJam(block, i, stats)) {
      changed = true;
      i = 0;
    }
  }

  for (size_t i = 0; i < block->children.size(); i++) {
    if (block->children[i] && block->children[i]->kind == OpKind::While &&
        tryRepeatInvariantReduction(block, i, stats)) {
      changed = true;
      i = 0;
    }
  }

  for (size_t i = 0; i < block->children.size(); i++) {
    if (hirEnvEnabled("SISY_HIR_ENABLE_INTERCHANGE", true)) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryLoopInterchange(block, i, stats)) {
          changed = true;
          i = 0;
          continue;
        }
      }
    }
    if (hirEnvEnabled("SISY_HIR_ENABLE_UNROLL_JAM", true)) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryLoopUnrollJam(block, i, stats)) {
          changed = true;
          i = 0;
          continue;
        }
      }
    }
    // Loop tiling: strip-mine loops that have an inner while.
    // HIR tiling rewrites loop structure before CFG construction. Keep it
    // opt-in until the transform can prove that loop-exit IV values remain
    // identical for users after the loop.
    if (hirEnvEnabled("SISY_HIR_ENABLE_TILING", false)) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryLoopTiling(block, i, stats)) {
          changed = true;
          i = 0;
          continue;
        }
      }
    }
    // Loop fusion: merge adjacent canonical while-loops with equal bounds.
    if (hirEnvEnabled("SISY_HIR_ENABLE_FUSION", true)) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryLoopFusion(block, i, stats)) {
          changed = true;
          i = 0;
          continue;
        }
      }
    }
  }
  return changed;
}

void PolyhedralOptimizer::scanAffineNest(Op *op, PolyhedralStats &stats) {
  stats.affineNestCandidates++;

  std::vector<CanonicalLoop> loops;
  Op *current = op;
  for (int depth = 0; depth < 3; depth++) {
    CanonicalLoop loop;
    if (!matchCanonicalWhile(current, loop)) {
      if (depth == 0) {
        stats.affineNestRejectedShape++;
        return;
      }
      break;
    }
    loops.push_back(loop);
    current = findSingleDirectInnerWhile(loop.body);
    if (!current)
      break;
  }

  if (loops.size() < 2) {
    stats.affineNestRejectedShape++;
    return;
  }
  if (hasUnsafeAffineScanControl(op)) {
    stats.affineNestRejectedControl++;
    return;
  }

  int rawAccesses = countArrayAccessOps(op);
  std::vector<affine::Access> accesses = affine::collectArrayAccesses(op);
  if (rawAccesses != (int) accesses.size()) {
    stats.affineNestRejectedAccess++;
    return;
  }

  if (loops.size() >= 2)
    stats.affineNestPerfect2D++;
  if (loops.size() >= 3)
    stats.affineNestPerfect3D++;
  if (isMatmulLikeNest(loops, accesses))
    stats.matmulLikeCandidates++;
}

// ===========================================================================
// Loop Interchange / Unroll-and-Jam (Presburger-based)
// ===========================================================================

bool PolyhedralOptimizer::tryLoopInterchange3D(Op *block, size_t idx,
                                               PolyhedralStats &stats) {
  if (!block || idx == 0 || idx >= block->children.size())
    return false;

  Op *outerWhile = block->children[idx].get();
  CanonicalLoop outer;
  if (!matchCanonicalWhile(outerWhile, outer)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }

  Op *outerInitOp = block->children[idx - 1].get();
  if (!matchLoopInit(outerInitOp, outer.iv)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectInit++;
    return false;
  }

  if (!outer.body || outer.body->kind != OpKind::Block || outer.body->children.size() != 3) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }

  Op *middleInitOp = outer.body->children[0].get();
  Op *middleWhile = unwrapSingleDecl(outer.body->children[1].get());
  if (!middleWhile || middleWhile->kind != OpKind::While) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }

  CanonicalLoop middle;
  if (!matchCanonicalWhile(middleWhile, middle)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }
  if (!matchLoopInit(middleInitOp, middle.iv)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectInit++;
    return false;
  }

  if (!middle.body || middle.body->kind != OpKind::Block || middle.body->children.size() != 3) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }

  Op *innerInitOp = middle.body->children[0].get();
  Op *innerWhile = unwrapSingleDecl(middle.body->children[1].get());
  if (!innerWhile || innerWhile->kind != OpKind::While) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }

  CanonicalLoop inner;
  if (!matchCanonicalWhile(innerWhile, inner)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }
  if (!matchLoopInit(innerInitOp, inner.iv)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectInit++;
    return false;
  }

  if (!tilingSafeBody(inner.body) || containsWhile(inner.body) ||
      blockExceptLastWritesScalar(inner.body, inner.iv) ||
      bodyWritesScalar(inner.body, middle.iv) ||
      bodyWritesScalar(inner.body, outer.iv)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectControl++;
    return false;
  }

  std::unordered_set<std::string> loopIVs = {outer.iv, middle.iv, inner.iv};
  if (bodyWritesNonLoopScalar(inner.body, loopIVs)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectControl++;
    return false;
  }

  int rawAccesses = countArrayAccessOps(inner.body);
  std::vector<affine::Access> accesses = affine::collectArrayAccesses(inner.body);
  if (rawAccesses != (int) accesses.size()) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectAccess++;
    return false;
  }
  if (!isMatmulLikeNest({outer, middle, inner}, accesses)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectShape++;
    return false;
  }

  const Op *outerInitVal = initValue(outerInitOp);
  const Op *middleInitVal = initValue(middleInitOp);
  const Op *innerInitVal = initValue(innerInitOp);
  if (!outerInitVal || !middleInitVal || !innerInitVal) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectInit++;
    return false;
  }

  affine::Expr outerInitExpr = affine::analyzeExpr(outerInitVal);
  affine::Expr middleInitExpr = affine::analyzeExpr(middleInitVal);
  affine::Expr innerInitExpr = affine::analyzeExpr(innerInitVal);
  affine::Expr outerBoundExpr = affine::analyzeExpr(outer.bound);
  affine::Expr middleBoundExpr = affine::analyzeExpr(middle.bound);
  affine::Expr innerBoundExpr = affine::analyzeExpr(inner.bound);
  if (!outerInitExpr.valid || !middleInitExpr.valid || !innerInitExpr.valid ||
      !outerBoundExpr.valid || !middleBoundExpr.valid || !innerBoundExpr.valid) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectBounds++;
    return false;
  }

  if (affine::exprUsesAny(outerInitExpr, loopIVs) ||
      affine::exprUsesAny(middleInitExpr, loopIVs) ||
      affine::exprUsesAny(innerInitExpr, loopIVs) ||
      affine::exprUsesAny(outerBoundExpr, loopIVs) ||
      affine::exprUsesAny(middleBoundExpr, loopIVs) ||
      affine::exprUsesAny(innerBoundExpr, loopIVs)) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectBounds++;
    return false;
  }

  affine::PresburgerInterchangeResult dep =
      affine::permutationMemorySafePresburger(
          {outerWhile, middleWhile, innerWhile},
          {outerInitOp, middleInitOp, innerInitOp},
          {0, 2, 1});
  stats.presburgerInterchangeQueries += dep.queries;
  stats.presburgerInterchangeNoDeps += dep.noViolatingDependence;
  stats.presburgerInterchangeMayDeps += dep.mayViolatingDependence;
  stats.presburgerInterchangeUnknown += dep.unknown;
  if (!dep.safe) {
    stats.interchange3DRejected++;
    stats.interchange3DRejectMemory++;
    return false;
  }

  auto newJBody = makeBlock();
  for (size_t i = 0; i + 1 < inner.body->children.size(); i++)
    newJBody->children.push_back(cloneOp(inner.body->children[i].get()));
  newJBody->children.push_back(cloneOp(middle.step));
  auto newJWhile = makeWhile(cloneOp(middleWhile->children[0].get()), std::move(newJBody));

  auto newKBody = makeBlock();
  newKBody->children.push_back(cloneOp(middleInitOp));
  newKBody->children.push_back(std::move(newJWhile));
  newKBody->children.push_back(cloneOp(inner.step));
  auto newKWhile = makeWhile(cloneOp(innerWhile->children[0].get()), std::move(newKBody));

  auto newOuterBody = makeBlock();
  newOuterBody->children.push_back(cloneOp(innerInitOp));
  newOuterBody->children.push_back(std::move(newKWhile));
  newOuterBody->children.push_back(cloneOp(outer.step));
  auto newOuterWhile = makeWhile(cloneOp(outerWhile->children[0].get()), std::move(newOuterBody));

  block->children[idx] = std::move(newOuterWhile);
  stats.interchange3DApplied++;
  return true;
}

bool PolyhedralOptimizer::tryLoopInterchange(Op *block, size_t idx,
                                             PolyhedralStats &stats) {
  if (!block || idx == 0 || idx >= block->children.size())
    return false;

  Op *outerWhile = block->children[idx].get();
  CanonicalLoop outer;
  if (!matchCanonicalWhile(outerWhile, outer)) {
    stats.interchangeRejected++;
    stats.interchangeRejectShape++;
    return false;
  }

  Op *outerInitOp = block->children[idx - 1].get();
  if (!matchLoopInit(outerInitOp, outer.iv)) {
    stats.interchangeRejected++;
    stats.interchangeRejectInit++;
    return false;
  }

  if (!outer.body || outer.body->kind != OpKind::Block || outer.body->children.size() != 3) {
    stats.interchangeRejected++;
    stats.interchangeRejectShape++;
    return false;
  }

  Op *innerWhile = unwrapSingleDecl(outer.body->children[1].get());
  if (!innerWhile || innerWhile->kind != OpKind::While) {
    stats.interchangeRejected++;
    stats.interchangeRejectShape++;
    return false;
  }

  CanonicalLoop inner;
  if (!matchCanonicalWhile(innerWhile, inner)) {
    stats.interchangeRejected++;
    stats.interchangeRejectShape++;
    return false;
  }

  Op *innerInitOp = outer.body->children[0].get();
  if (!matchLoopInit(innerInitOp, inner.iv)) {
    stats.interchangeRejected++;
    stats.interchangeRejectInit++;
    return false;
  }

  if (!tilingSafeBody(inner.body) || containsWhile(inner.body) ||
      blockExceptLastWritesScalar(inner.body, inner.iv) ||
      bodyWritesScalar(inner.body, outer.iv)) {
    stats.interchangeRejected++;
    stats.interchangeRejectControl++;
    return false;
  }
  std::unordered_set<std::string> definedScalars;
  collectDefinedScalars(inner.body, definedScalars);
  definedScalars.erase(inner.iv);
  definedScalars.erase(outer.iv);
  if (!definedScalars.empty()) {
    stats.interchangeRejected++;
    stats.interchangeRejectControl++;
    return false;
  }

  int rawAccesses = countArrayAccessOps(inner.body);
  std::vector<affine::Access> accesses = affine::collectArrayAccesses(inner.body);
  if (rawAccesses != (int) accesses.size()) {
    stats.interchangeRejected++;
    stats.interchangeRejectAccess++;
    return false;
  }

  const Op *outerInitVal = initValue(outerInitOp);
  const Op *innerInitVal = initValue(innerInitOp);
  if (!outerInitVal || !innerInitVal) {
    stats.interchangeRejected++;
    stats.interchangeRejectInit++;
    return false;
  }

  affine::Expr outerInitExpr = affine::analyzeExpr(outerInitVal);
  affine::Expr innerInitExpr = affine::analyzeExpr(innerInitVal);
  affine::Expr outerBoundExpr = affine::analyzeExpr(outer.bound);
  affine::Expr innerBoundExpr = affine::analyzeExpr(inner.bound);
  if (!outerInitExpr.valid || !innerInitExpr.valid ||
      !outerBoundExpr.valid || !innerBoundExpr.valid) {
    stats.interchangeRejected++;
    stats.interchangeRejectBounds++;
    return false;
  }

  std::unordered_set<std::string> loopIVs = {outer.iv, inner.iv};
  if (affine::exprUsesAny(outerInitExpr, loopIVs) ||
      affine::exprUsesAny(innerInitExpr, loopIVs) ||
      affine::exprUsesAny(outerBoundExpr, loopIVs) ||
      affine::exprUsesAny(innerBoundExpr, loopIVs)) {
    stats.interchangeRejected++;
    stats.interchangeRejectBounds++;
    return false;
  }

  affine::PresburgerInterchangeResult dep =
      affine::interchangeMemorySafePresburger(outerWhile, innerWhile,
                                              outerInitOp, innerInitOp);
  stats.presburgerInterchangeQueries += dep.queries;
  stats.presburgerInterchangeNoDeps += dep.noViolatingDependence;
  stats.presburgerInterchangeMayDeps += dep.mayViolatingDependence;
  stats.presburgerInterchangeUnknown += dep.unknown;
  if (!dep.safe) {
    stats.interchangeRejected++;
    stats.interchangeRejectMemory++;
    return false;
  }

  auto newInnerBody = makeBlock();
  for (size_t i = 0; i + 1 < inner.body->children.size(); i++)
    newInnerBody->children.push_back(cloneOp(inner.body->children[i].get()));
  newInnerBody->children.push_back(cloneOp(outer.step));
  auto newInnerWhile =
      makeWhile(cloneOp(outerWhile->children[0].get()), std::move(newInnerBody));

  auto newOuterBody = makeBlock();
  newOuterBody->children.push_back(makeStore(outer.iv, cloneOp(outerInitVal)));
  newOuterBody->children.push_back(std::move(newInnerWhile));
  newOuterBody->children.push_back(cloneOp(inner.step));
  auto newOuterWhile =
      makeWhile(cloneOp(innerWhile->children[0].get()), std::move(newOuterBody));

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(cloneOp(innerInitOp));
      replacement.push_back(std::move(newOuterWhile));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.interchangeApplied++;
  return true;
}

bool PolyhedralOptimizer::tryLoopUnrollJam(Op *block, size_t idx,
                                           PolyhedralStats &stats) {
  if (!block || idx == 0 || idx >= block->children.size())
    return false;

  Op *outerWhile = block->children[idx].get();
  CanonicalLoop outer;
  if (!matchCanonicalWhile(outerWhile, outer)) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectShape++;
    return false;
  }

  Op *outerInitOp = block->children[idx - 1].get();
  if (!matchLoopInit(outerInitOp, outer.iv)) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectInit++;
    return false;
  }

  if (!outer.body || outer.body->kind != OpKind::Block || outer.body->children.size() != 3) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectShape++;
    return false;
  }

  Op *innerWhile = unwrapSingleDecl(outer.body->children[1].get());
  if (!innerWhile || innerWhile->kind != OpKind::While) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectShape++;
    return false;
  }

  CanonicalLoop inner;
  if (!matchCanonicalWhile(innerWhile, inner)) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectShape++;
    return false;
  }

  Op *innerInitOp = outer.body->children[0].get();
  if (!matchLoopInit(innerInitOp, inner.iv)) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectInit++;
    return false;
  }

  if (!tilingSafeBody(inner.body) || containsWhile(inner.body) ||
      blockExceptLastWritesScalar(inner.body, inner.iv) ||
      bodyWritesScalar(inner.body, outer.iv)) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectControl++;
    return false;
  }
  std::unordered_set<std::string> definedScalars;
  collectDefinedScalars(inner.body, definedScalars);
  definedScalars.erase(inner.iv);
  definedScalars.erase(outer.iv);
  if (!definedScalars.empty()) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectControl++;
    return false;
  }

  int rawAccesses = countArrayAccessOps(inner.body);
  std::vector<affine::Access> accesses = affine::collectArrayAccesses(inner.body);
  if (rawAccesses != (int) accesses.size()) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectAccess++;
    return false;
  }

  const Op *outerInitVal = initValue(outerInitOp);
  const Op *innerInitVal = initValue(innerInitOp);
  if (!outerInitVal || !innerInitVal) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectInit++;
    return false;
  }

  affine::Expr outerInitExpr = affine::analyzeExpr(outerInitVal);
  affine::Expr innerInitExpr = affine::analyzeExpr(innerInitVal);
  affine::Expr outerBoundExpr = affine::analyzeExpr(outer.bound);
  affine::Expr innerBoundExpr = affine::analyzeExpr(inner.bound);
  if (!outerInitExpr.valid || !innerInitExpr.valid ||
      !outerBoundExpr.valid || !innerBoundExpr.valid) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectBounds++;
    return false;
  }

  std::unordered_set<std::string> loopIVs = {outer.iv, inner.iv};
  if (affine::exprUsesAny(outerInitExpr, loopIVs) ||
      affine::exprUsesAny(innerInitExpr, loopIVs) ||
      affine::exprUsesAny(outerBoundExpr, loopIVs) ||
      affine::exprUsesAny(innerBoundExpr, loopIVs)) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectBounds++;
    return false;
  }

  affine::PresburgerInterchangeResult dep =
      affine::interchangeMemorySafePresburger(outerWhile, innerWhile,
                                              outerInitOp, innerInitOp);
  stats.presburgerInterchangeQueries += dep.queries;
  stats.presburgerInterchangeNoDeps += dep.noViolatingDependence;
  stats.presburgerInterchangeMayDeps += dep.mayViolatingDependence;
  stats.presburgerInterchangeUnknown += dep.unknown;
  if (!dep.safe) {
    stats.unrollJamRejected++;
    stats.unrollJamRejectMemory++;
    return false;
  }

  int dynamicJamFactor = computeOptimalJamFactor(inner.body, detectMainType(inner.body));

  auto jamInnerBody = makeBlock();
  for (int lane = 0; lane < dynamicJamFactor; lane++) {
    std::unordered_map<std::string, std::string> scalarRenames;
    std::unordered_map<std::string, int> ivOffsets = { { outer.iv, lane } };
    for (size_t i = 0; i + 1 < inner.body->children.size(); i++) {
      jamInnerBody->children.push_back(
          cloneReplacing(inner.body->children[i].get(), scalarRenames, ivOffsets));
    }
  }
  jamInnerBody->children.push_back(cloneOp(inner.step));

  auto jamInnerWhile =
      makeWhile(cloneOp(innerWhile->children[0].get()), std::move(jamInnerBody));

  auto jamOuterBody = makeBlock();
  jamOuterBody->children.push_back(cloneOp(innerInitOp));
  jamOuterBody->children.push_back(std::move(jamInnerWhile));
  jamOuterBody->children.push_back(makeJamStep(outer.iv, dynamicJamFactor));

  auto jamCond = makeCmp("<",
                         makeArith("+", makeLoad(outer.iv),
                                   makeConstInt(dynamicJamFactor - 1)),
                         cloneOp(outer.bound));
  auto jamOuterWhile = makeWhile(std::move(jamCond), std::move(jamOuterBody));
  auto tailWhile = cloneOp(outerWhile);

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(std::move(jamOuterWhile));
      replacement.push_back(std::move(tailWhile));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.unrollJammed++;
  return true;
}

// ===========================================================================
// Loop Tiling Implementation
// ===========================================================================
//
// Strip-mines the outer loop of any loop nest that has:
//   - A canonical unit-step while form: while (iv < bound) { ...; iv++; }
//   - At least one inner while loop in the body
//   - No break/continue/call in the body at any depth
//
// Transformation:
//   iv_init; while (iv < N): [... inner loops ...]; iv++
// →
//   iv_init; __tile = iv; while (__tile < N):
//     __tile_stop = __tile + T; if (N < __tile_stop) { __tile_stop = N; }
//     iv = __tile;
//     while (iv < __tile_stop): [... inner loops ...]; iv++
//     __tile = __tile + T

bool PolyhedralOptimizer::tryLoopTiling(Op *block, size_t idx,
                                         PolyhedralStats &stats) {
  if (!block || idx >= block->children.size())
    return false;

  Op *whileOp = block->children[idx].get();
  CanonicalLoop outer;
  if (!matchCanonicalWhile(whileOp, outer)) {
    stats.tilingRejected++;
    stats.tilingRejectShape++;
    return false;
  }

  // Body must be tiling-safe (no break/continue/call at any depth).
  if (!tilingSafeBody(outer.body)) {
    stats.tilingRejected++;
    stats.tilingRejectControl++;
    return false;
  }
  if (affine::opWritesAnyScalarUsedBy(outer.body, outer.bound)) {
    stats.tilingRejected++;
    stats.tilingRejectBoundWrite++;
    return false;
  }
  if (!affine::hasAffineArrayAccessUsing(outer.body, outer.iv, 2)) {
    stats.tilingRejected++;
    stats.tilingRejectAffineAccess++;
    return false;
  }

  // Must have at least one inner while loop (otherwise tiling does nothing).
  bool hasInnerWhile = false;
  for (const auto &child : outer.body->children)
    if (child && child->kind == OpKind::While) { hasInnerWhile = true; break; }
  if (!hasInnerWhile) {
    stats.tilingRejected++;
    stats.tilingRejectNoInner++;
    return false;
  }

  // Don't tile if there's already a tile IV variable with our prefix (idempotency guard).
  if (outer.iv.rfind("__hir_tile_", 0) == 0) {
    stats.tilingRejected++;
    stats.tilingRejectIdempotent++;
    return false;
  }

  int dynamicTileSize = computeOptimalTileSize(detectMainType(outer.body));

  const std::string tileIV = "__hir_tile_" + outer.iv + "_" + std::to_string(uniqueId);
  const std::string stopVar = "__hir_tile_stop_" + outer.iv + "_" + std::to_string(uniqueId);
  uniqueId++;

  // --- Build the tile while body ---
  //   int __tile_stop = __tile + T;
  //   if (N < __tile_stop) { __tile_stop = N; }
  //   iv = __tile;          // re-init original IV for inner loop
  //   while (iv < __tile_stop): [...original body...]
  //   __tile = __tile + T;
  auto tileBody = makeBlock();

  // stopVar = tileIV + T
  tileBody->children.push_back(
    makeVarDecl(stopVar, makeArith("+", makeLoad(tileIV), makeConstInt(dynamicTileSize))));

  // if (bound < stopVar) { stopVar = bound; }
  {
    auto ifCond = makeCmp("<", cloneOp(outer.bound), makeLoad(stopVar));
    auto thenBlk = makeBlock();
    thenBlk->children.push_back(makeStore(stopVar, cloneOp(outer.bound)));
    auto ifOp = std::make_unique<Op>(OpKind::If);
    ifOp->children.push_back(std::move(ifCond));
    ifOp->children.push_back(std::move(thenBlk));
    tileBody->children.push_back(std::move(ifOp));
  }

  // iv = tileIV  (reset outer IV at the start of each tile)
  tileBody->children.push_back(makeStore(outer.iv, makeLoad(tileIV)));

  // Clone the original while with condition bound replaced by stopVar.
  auto innerCond = makeCmp("<", makeLoad(outer.iv), makeLoad(stopVar));
  auto innerBody = cloneOp(outer.body);
  tileBody->children.push_back(makeWhile(std::move(innerCond), std::move(innerBody)));

  // tileIV += T
  tileBody->children.push_back(
    makeStore(tileIV, makeArith("+", makeLoad(tileIV), makeConstInt(dynamicTileSize))));

  // --- Build the tile while ---
  auto tileCond = makeCmp("<", makeLoad(tileIV), cloneOp(outer.bound));
  auto tileWhile = makeWhile(std::move(tileCond), std::move(tileBody));
  // Tile IV init: start from the loop's current IV value. Loops such as
  // stencil kernels often begin at 1, and forcing tileIV to 0 would introduce
  // out-of-bounds accesses before the original iteration domain.
  auto tileInit = makeVarDecl(tileIV, makeLoad(outer.iv));

  // --- Splice into block ---
  // Insert tileInit + tileWhile at position idx, replacing the original while.
  // The original outer IV init (VarDecl/Store) before idx is kept intact
  // (it just becomes dead-assigned; DCE will clean it up later).
  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(std::move(tileInit));
      replacement.push_back(std::move(tileWhile));
      // Skip the original while (don't push block->children[idx]).
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.tilingApplied++;
  return true;
}


// ===========================================================================
// Loop Fusion Implementation
// ===========================================================================
//
// Fuse two adjacent canonical while-loops A and B that iterate over the
// same range, when B does not depend on scalars defined only in A:
//
//   i = 0; while (i < N): bodyA; i++
//   j = 0; while (j < N): bodyB[j renamed to i]; j++
//
// Becomes:
//   i = 0; while (i < N): bodyA; bodyB[j→i]; i++

bool PolyhedralOptimizer::tryLoopFusion(Op *block, size_t idx,
                                         PolyhedralStats &stats) {
  if (!block || idx + 1 >= block->children.size())
    return false;

  Op *whileA = block->children[idx].get();
  if (!whileA || whileA->kind != OpKind::While)
    return false;

  bool hasBInit = false;
  size_t bInitIdx = idx + 1;
  size_t bWhileIdx = idx + 1;
  if (block->children[bWhileIdx] && block->children[bWhileIdx]->kind != OpKind::While &&
      idx + 2 < block->children.size() &&
      block->children[idx + 2] && block->children[idx + 2]->kind == OpKind::While) {
    hasBInit = true;
    bWhileIdx = idx + 2;
  }

  Op *whileB = block->children[bWhileIdx].get();
  if (!whileB)
    return false;

  CanonicalLoop loopA, loopB;
  if (!matchCanonicalWhile(whileA, loopA) || !matchCanonicalWhile(whileB, loopB)) {
    stats.fusionRejected++;
    stats.fusionRejectShape++;
    return false;
  }
  if (hasBInit) {
    if (!matchLoopInit(block->children[bInitIdx].get(), loopB.iv)) {
      stats.fusionRejected++;
      stats.fusionRejectInit++;
      return false;
    }
    if (idx == 0 || !matchLoopInit(block->children[idx - 1].get(), loopA.iv)) {
      stats.fusionRejected++;
      stats.fusionRejectInit++;
      return false;
    }
    if (!boundsEqual(initValue(block->children[idx - 1].get()),
                     initValue(block->children[bInitIdx].get()))) {
      stats.fusionRejected++;
      stats.fusionRejectInit++;
      return false;
    }
  }

  // Bounds must be equal.
  if (!boundsEqual(loopA.bound, loopB.bound)) {
    stats.fusionRejected++;
    stats.fusionRejectBounds++;
    return false;
  }

  // Safety: no control/call ops in either body.
  if (!tilingSafeBody(loopA.body) || !tilingSafeBody(loopB.body)) {
    stats.fusionRejected++;
    stats.fusionRejectControl++;
    return false;
  }
  if (containsWhile(loopA.body) != containsWhile(loopB.body)) {
    stats.fusionRejected++;
    stats.fusionRejectControl++;
    return false;
  }

  // Check that loop B's body does not depend on scalars exclusively defined
  // by loop A's body (other than the IV itself, which we will rename).
  std::unordered_set<std::string> aDefinedScalars;
  collectDefinedScalars(loopA.body, aDefinedScalars);
  // Remove loopA's IV — it's fine if B uses it (renamed to A's IV).
  aDefinedScalars.erase(loopA.iv);
  std::unordered_set<std::string> bInitializedScalars;
  collectTopLevelInitializedScalars(loopB.body, bInitializedScalars);
  for (const auto &sym : bInitializedScalars)
    aDefinedScalars.erase(sym);

  if (bodyUsesAnyOf(loopB.body, aDefinedScalars)) {
    stats.fusionRejected++;
    stats.fusionRejectScalar++;
    return false;
  }
  const Op *initA = hasBInit && idx > 0 ? initValue(block->children[idx - 1].get()) : nullptr;
  const Op *initB = hasBInit ? initValue(block->children[bInitIdx].get()) : nullptr;
  affine::PresburgerFusionResult fusionDep =
      affine::fusionMemorySafePresburger(whileA, whileB, initA, initB);
  stats.presburgerFusionQueries += fusionDep.queries;
  stats.presburgerFusionNoDeps += fusionDep.noReorderedDependence;
  stats.presburgerFusionMayDeps += fusionDep.mayReorderedDependence;
  stats.presburgerFusionUnknown += fusionDep.unknown;
  if (!fusionDep.safe) {
    stats.fusionRejected++;
    stats.fusionRejectMemory++;
    return false;
  }

  // Cache-line budget gate: avoid fusion if combined working set exceeds L1.
  if (!fusionWithinCacheBudget(loopA.body, loopB.body)) {
    stats.fusionRejected++;
    stats.fusionRejectMemory++;
    return false;
  }

  // Fuse: clone loop B's body, renaming its IV to loop A's IV, then append
  // to loop A's body (before the step).
  // Loop B's step store is the last child of its body; skip it.
  std::unordered_map<std::string, std::string> renames = { { loopB.iv, loopA.iv } };
  std::unordered_map<std::string, int> noOffsets;

  // Build the fused body: loopA's body statements + loopB's body statements
  // (excluding loopB's step, since loopA's step covers both).
  auto &aBodyChildren = loopA.body->children;
  // Insert all loopB body children (except last = loopB's step) before loopA's step.
  // loopA's step is aBodyChildren.back().
  size_t insertPos = aBodyChildren.size() - 1; // before loopA's step
  std::vector<std::unique_ptr<Op>> bStatements;
  auto &bBodyChildren = loopB.body->children;
  for (size_t i = 0; i + 1 < bBodyChildren.size(); i++) { // skip last (step)
    bStatements.push_back(cloneReplacing(bBodyChildren[i].get(), renames, noOffsets));
  }
  // Also need to declare loopB's IV as an alias (we just renamed it, so no decl needed
  // if loopB.iv was already declared before the while). We need to handle the case where
  // the loop B init (before the while) declared loopB.iv. After fusion, loop B's init
  // becomes dead. We keep it for safety (it's just an extra assignment to a dead var).

  for (auto &stmt : bStatements)
    aBodyChildren.insert(aBodyChildren.begin() + insertPos++, std::move(stmt));

  if (hasBInit) {
    block->children.erase(block->children.begin() + bInitIdx,
                          block->children.begin() + bWhileIdx + 1);
  } else {
    block->children.erase(block->children.begin() + idx + 1);
  }

  stats.fusionApplied++;
  forwardArrayStoreLoads(loopA.body, stats);
  for (size_t nested = 0; loopA.body && nested < loopA.body->children.size(); nested++) {
    if (tryLoopFusion(loopA.body, nested, stats))
      nested = static_cast<size_t>(-1);
  }
  return true;
}

bool PolyhedralOptimizer::forwardArrayStoreLoads(Op *block, PolyhedralStats &stats) {
  if (!block || block->kind != OpKind::Block)
    return false;

  bool changed = false;
  for (size_t i = 0; i + 1 < block->children.size(); i++) {
    Op *store = asArrayStore(block->children[i].get());
    Op *decl = asScalarVarDecl(block->children[i + 1].get());
    if (!store || !decl)
      continue;
    Op *init = decl->children[0].get();
    if (!init || init->kind != OpKind::Load || init->children.empty())
      continue;
    if (!sameArrayAddress(store, init))
      continue;

    decl->children[0] = cloneOp(store->children.back().get());
    stats.forwardedArrayStoreLoads++;
    changed = true;
  }

  return changed;
}

bool PolyhedralOptimizer::tryRepeatInvariantReduction(Op *block, size_t idx,
                                                       PolyhedralStats &stats) {
  if (!block || idx >= block->children.size() || idx == 0)
    return false;

  Op *whileOp = block->children[idx].get();
  CanonicalLoop loop;
  if (!matchCanonicalWhile(whileOp, loop)) {
    stats.repeatRejected++;
    stats.repeatRejectShape++;
    return false;
  }
  if (!matchLoopInit(block->children[idx - 1].get(), loop.iv)) {
    stats.repeatRejected++;
    stats.repeatRejectInit++;
    return false;
  }
  const Op *start = initValue(block->children[idx - 1].get());
  if (!isConstIntValue(start, 0)) {
    stats.repeatRejected++;
    stats.repeatRejectInit++;
    return false;
  }
  if (exprUsesScalar(loop.bound, loop.iv) ||
      blockExceptLastUsesScalar(loop.body, loop.iv) ||
      affine::opWritesAnyScalarUsedBy(loop.body, loop.bound)) {
    stats.repeatRejected++;
    stats.repeatRejectBound++;
    return false;
  }

  std::string acc;
  if (!repeatBodyLegal(loop.body, loop.iv, acc)) {
    stats.repeatRejected++;
    stats.repeatRejectLegal++;
    return false;
  }

  auto bodyOnce = cloneBlockWithoutLast(loop.body);
  if (!bodyOnce) {
    stats.repeatRejected++;
    stats.repeatRejectClone++;
    return false;
  }

  const std::string baseVar = "__hir_repeat_base_" + acc + "_" + std::to_string(uniqueId++);
  auto thenBlock = makeBlock();
  thenBlock->children.push_back(std::move(bodyOnce));
  auto delta = makeArith("-", makeLoad(acc), makeLoad(baseVar));
  auto scaled = makeArith("*", std::move(delta), cloneOp(loop.bound));
  thenBlock->children.push_back(makeStore(acc,
      makeArith("+", makeLoad(baseVar), std::move(scaled))));
  thenBlock->children.push_back(makeStore(loop.iv, cloneOp(loop.bound)));

  auto ifOp = std::make_unique<Op>(OpKind::If);
  ifOp->children.push_back(makeCmp("<", makeConstInt(0), cloneOp(loop.bound)));
  ifOp->children.push_back(std::move(thenBlock));

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(makeVarDecl(baseVar, makeLoad(acc)));
      replacement.push_back(std::move(ifOp));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.repeatReduced++;
  return true;
}

bool PolyhedralOptimizer::tryReductionInterchange(Op *block, size_t initIndex,
                                                  PolyhedralStats &stats) {
  if (!block || initIndex + 1 >= block->children.size())
    return false;

  Op *whileOp = block->children[initIndex + 1].get();
  ReductionPattern pat;
  if (!matchReductionPattern(whileOp, pat))
    return false;
  if (!matchLoopInit(block->children[initIndex].get(), pat.j))
    return false;
  if (!strictReductionInterchangeLegal(pat, globalArrays))
    return false;

  auto initLoop = makeReductionInitLoop(pat);
  auto swappedKLoop = makeReductionInterchangedKLoop(pat, block->children[initIndex].get());
  if (!initLoop || !swappedKLoop) {
    stats.rejected++;
    return false;
  }

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 2);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == initIndex) {
      replacement.push_back(std::move(block->children[i]));
      replacement.push_back(std::move(initLoop));
      replacement.push_back(cloneOp(pat.kInit));
      replacement.push_back(std::move(swappedKLoop));
      i++; // skip the original j loop
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }

  block->children = std::move(replacement);
  stats.reductionInterchanged++;
  if (pat.kReductionStmt != pat.accUpdate)
    stats.conditionalReductionInterchanged++;
  return true;
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
  if (pat.kReductionStmt != pat.accUpdate) {
    stats.rejected++;
    return false;
  }

  int dynamicJamFactor = computeOptimalJamFactor(pat.kWhile, detectMainType(pat.kWhile));

  const std::string prefix = "__poly_" + pat.acc + "_" + std::to_string(uniqueId++);
  std::vector<std::string> accNames;
  accNames.reserve(dynamicJamFactor);
  for (int lane = 0; lane < dynamicJamFactor; lane++)
    accNames.push_back(prefix + "_" + std::to_string(lane));

  auto vecBody = makeBlock();
  for (int lane = 0; lane < dynamicJamFactor; lane++) {
    vecBody->children.push_back(makeVarDecl(accNames[lane], cloneOp(initValue(pat.accInit))));
  }

  vecBody->children.push_back(cloneOp(pat.kInit));

  CanonicalLoop kLoop;
  if (!matchCanonicalWhile(pat.kWhile, kLoop))
    return false;

  auto kBody = makeBlock();
  for (int lane = 0; lane < dynamicJamFactor; lane++) {
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

  for (int lane = 0; lane < dynamicJamFactor; lane++) {
    std::unordered_map<std::string, std::string> scalarRenames = { { pat.acc, accNames[lane] } };
    std::unordered_map<std::string, int> ivOffsets = { { pat.j, lane } };
    vecBody->children.push_back(cloneReplacing(pat.destStore, scalarRenames, ivOffsets));
  }
  vecBody->children.push_back(makeJamStep(pat.j, dynamicJamFactor));

  auto vecCond = makeCmp("<",
                         makeArith("+", makeLoad(pat.j), makeConstInt(dynamicJamFactor - 1)),
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
