#include "HIRPolyhedral.h"

#include "HIRAffine.h"
#include "../backend/riscv/RiscvParams.h"
#include "HIRPolyDetail.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sys::hir::detail {



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
  for (int factor : {8, 7, 6, 4, 2}) {
    // Each jam lane keeps the active memory streams plus address/update
    // temporaries alive. Modeling those temporaries keeps the transform from
    // choosing a wide jam that immediately spills or stretches scheduling on
    // the target backend.
    int estimatedPressure = factor * (activeStreams + 1) + activeScalars + 2;
    if (estimatedPressure <= usableRegs) {
      bestFactor = factor;
      break;
    }
  }
  return bestFactor;
}

static int clampPow2(int n, int lo, int hi);

ReductionTilePlan computeReductionTilePlan(const Op *innerBody,
                                           TypeKind mainType,
                                           bool needsScratch) {
  using namespace sys::backend::riscv;

  ReductionTilePlan plan;
  plan.needsScratch = needsScratch;
  if (mainType != TypeKind::Int)
    return plan;

  LoopBodyMetrics metrics;
  analyzeLoopBody(innerBody, metrics);
  int readStreams = static_cast<int>(metrics.arrayReadStreams.size());
  int writeStreams = static_cast<int>(metrics.arrayWriteStreams.size());
  int basePressure = 7 + readStreams + writeStreams +
                     metrics.scalarIntDefs;
  int usableRegs = kUsableGPRs;

  int requestedNr = hirEnvInt("SISY_HIR_MICRO_NR", 0);
  if (requestedNr >= 2 && requestedNr <= 8) {
    plan.nr = requestedNr;
  } else {
    for (int candidate : {4, 2}) {
      // Each output lane holds an accumulator and one transient operand;
      // panel cursors and common inputs are accounted for in basePressure.
      if (basePressure + 2 * candidate <= usableRegs - 2) {
        plan.nr = candidate;
        break;
      }
    }
  }
  if (plan.nr < 2)
    return plan;

  const int elementBytes = 4;
  const int streamCount = std::max(2, readStreams + writeStreams);
  const int panelBudget = kL1DataCacheSize / 4;
  int cacheKc = panelBudget / std::max(elementBytes * streamCount * plan.nr, 1);
  cacheKc = clampPow2(cacheKc, 8, 64);
  plan.kc = hirEnvInt("SISY_HIR_MICRO_KC", cacheKc);
  if (plan.kc < 2) {
    plan.nr = 0;
    plan.kc = 0;
    return plan;
  }
  plan.nc = plan.nr;
  return plan;
}

// Round `n` down to the nearest power-of-two in [lo, hi]. Used to keep tile
// sizes vectorization-friendly (multiples of 4 lanes) and cache-aligned.
static int clampPow2(int n, int lo, int hi) {
  if (n < lo) return lo;
  if (n > hi) return hi;
  int p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

// Cache-aware tile size. Considers:
//   1. SISY_HIR_TILE_SIZE env override (highest priority).
//   2. Per-element-type byte size (4 for i32/f32, 8 for i64/f64).
//   3. Number of distinct array streams touched in the inner body — each
//      stream lives in its own cache way, so total tile footprint is
//      tile * elemSize * streams.
//   4. A fixed L1 budget (half of kL1DataCacheSize to leave room for stack
//      slots, scalars, and conflict misses).
//   5. Backend usable vector lane count, so the tile size stays a clean
//      multiple of 4 lanes when fixed-128-bit SIMD eventually consumes it.
//
// The caller passes an optional inner-body Op. When non-null we walk it to
// collect array-stream counts; otherwise we fall back to the type-only path.
int computeOptimalTileSize(TypeKind mainType, const Op *innerBody) {
  const char *envRaw = std::getenv("SISY_HIR_TILE_SIZE");
  if (envRaw && envRaw[0]) {
    int val = std::atoi(envRaw);
    if (val >= 4) {
      return val;
    }
  }
  using namespace sys::backend::riscv;

  const int elementSize = (mainType == TypeKind::Float || mainType == TypeKind::Int) ? 4 : 8;
  const int elementsPerCacheLine = kCacheLineSize / elementSize;

  // Fall back to the original heuristic when we have no body context.
  if (!innerBody)
    return elementsPerCacheLine * 2;

  LoopBodyMetrics metrics;
  analyzeLoopBody(innerBody, metrics);
  int streams = (int) (metrics.arrayReadStreams.size() + metrics.arrayWriteStreams.size());
  if (streams <= 0) streams = 1;

  // Reserve half of L1 for non-tile data: stack, scalars, conflict misses.
  const int cacheBudgetBytes = kL1DataCacheSize / 2;
  const int perIterFootprint = elementSize * streams;
  const int budgetFitTile = cacheBudgetBytes / std::max(1, perIterFootprint);

  // Lower bound: at least one cache line per stream so prefetchers can ramp.
  // Upper bound: 256 (avoid pathological tile sizes that bury the outer loop).
  const int loBound = std::max(4, elementsPerCacheLine);
  const int hiBound = 256;

  int candidate = clampPow2(budgetFitTile, loBound, hiBound);

  // Further halve when the stream count is high (>=4), since real workloads
  // commonly bring in extra associativity pressure from index registers and
  // small lookup tables not visible in the metrics walk.
  if (streams >= 4 && candidate > loBound)
    candidate /= 2;

  return candidate;
}

// Backward-compatible overload (older callers had no body context).
int computeOptimalTileSize(TypeKind mainType) {
  return computeOptimalTileSize(mainType, nullptr);
}


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

std::unique_ptr<Op> makeIf(std::unique_ptr<Op> cond,
                           std::unique_ptr<Op> thenBlock,
                           std::unique_ptr<Op> elseBlock) {
  auto op = std::make_unique<Op>(OpKind::If);
  op->children.push_back(std::move(cond));
  op->children.push_back(std::move(thenBlock));
  if (elseBlock)
    op->children.push_back(std::move(elseBlock));
  return op;
}

std::unique_ptr<Op> makeLogicalAnd(std::unique_ptr<Op> lhs,
                                   std::unique_ptr<Op> rhs) {
  return makeArith("&&", std::move(lhs), std::move(rhs));
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

bool isScalarLoadAny(const Op *op, std::string &symbol) {
  if (!op || op->kind != OpKind::Load || !op->children.empty() || op->symbol.empty())
    return false;
  symbol = op->symbol;
  return true;
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

bool matchContinueStepThenBlock(const Op *op, const std::string &iv) {
  op = unwrapSingleDecl(op);
  if (!op || op->kind != OpKind::Block || op->children.empty())
    return false;

  bool sawStep = false;
  bool sawContinue = false;
  for (const auto &child : op->children) {
    const Op *stmt = unwrapSingleDecl(child.get());
    if (!stmt)
      return false;
    if (!sawStep && matchStepStore(const_cast<Op*>(stmt), iv, 1)) {
      sawStep = true;
      continue;
    }
    if (stmt->kind == OpKind::Continue) {
      sawContinue = true;
      continue;
    }
    return false;
  }
  return sawStep && sawContinue;
}

bool matchPrefixSkipGuard(const Op *op, const std::string &iv,
                          std::string &limitScalar) {
  op = unwrapSingleDecl(op);
  if (!op || op->kind != OpKind::If || op->children.size() < 2)
    return false;
  const Op *cond = op->children[0].get();
  if (!cond || cond->kind != OpKind::Cmp || cond->symbol != "<" ||
      cond->children.size() != 2)
    return false;
  const Op *lhs = cond->children[0].get();
  const Op *rhs = cond->children[1].get();
  if (!rhs || !isScalarLoad(rhs, iv))
    return false;
  std::string scalar;
  if (!isScalarLoadAny(lhs, scalar) || scalar.empty() || scalar == iv)
    return false;
  if (!matchContinueStepThenBlock(op->children[1].get(), iv))
    return false;
  limitScalar = scalar;
  return true;
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

  // Do not interchange an in-place reduction. If the destination array is
  // read while computing the reduction, swapping k outside j changes the
  // order of visible writes.  Order-preserving transforms such as
  // reduction unroll-and-jam may still handle these loops.
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

void collectScalarInitializers(const Op *op,
                               std::unordered_map<std::string, const Op*> &inits) {
  if (!op)
    return;
  const Op *unwrapped = unwrapSingleDecl(op);
  if (unwrapped && (unwrapped->kind == OpKind::VarDecl ||
                    unwrapped->kind == OpKind::Store) &&
      !unwrapped->symbol.empty() && unwrapped->children.size() == 1) {
    inits[unwrapped->symbol] = unwrapped->children[0].get();
  }
  for (const auto &child : op->children)
    collectScalarInitializers(child.get(), inits);
}

bool matchSpatialKernelMinusPad(const Op *expr, std::string &spatial,
                                std::string &kernel, std::string &pad) {
  if (!expr || expr->kind != OpKind::Arith || expr->symbol != "-" ||
      expr->children.size() != 2)
    return false;
  if (!isScalarLoadAny(expr->children[1].get(), pad))
    return false;
  const Op *sum = expr->children[0].get();
  if (!sum || sum->kind != OpKind::Arith || sum->symbol != "+" ||
      sum->children.size() != 2)
    return false;
  std::string a, b;
  if (!isScalarLoadAny(sum->children[0].get(), a) ||
      !isScalarLoadAny(sum->children[1].get(), b))
    return false;
  spatial = a;
  kernel = b;
  return true;
}

bool flattenAndTerms(const Op *expr, std::vector<const Op*> &terms) {
  if (!expr)
    return false;
  if (expr->kind == OpKind::Arith && expr->symbol == "&&" &&
      expr->children.size() == 2) {
    return flattenAndTerms(expr->children[0].get(), terms) &&
           flattenAndTerms(expr->children[1].get(), terms);
  }
  terms.push_back(expr);
  return true;
}


bool matchStencilBoundsIf(const Op *ifOp,
                          const std::unordered_map<std::string, const Op*> &inits,
                          const std::string &colIv,
                          StencilBounds &out) {
  if (!ifOp || ifOp->kind != OpKind::If || ifOp->children.size() < 2)
    return false;

  std::vector<const Op*> terms;
  if (!flattenAndTerms(ifOp->children[0].get(), terms) || terms.size() < 4)
    return false;

  std::vector<std::string> boundedSyms;
  std::unordered_map<std::string, const Op*> upperBounds;
  for (const Op *term : terms) {
    if (!term || term->kind != OpKind::Cmp)
      continue;
    std::string candidate;
    const Op *bound = nullptr;
    if (term->symbol == "<=" && term->children.size() == 2 &&
        isConstIntValue(term->children[0].get(), 0) &&
        isScalarLoadAny(term->children[1].get(), candidate)) {
      boundedSyms.push_back(candidate);
      continue;
    }
    if (term->symbol == "<" && term->children.size() == 2 &&
        isScalarLoadAny(term->children[0].get(), candidate)) {
      bound = term->children[1].get();
      upperBounds[candidate] = bound;
    }
  }

  for (const std::string &a : boundedSyms) {
    if (!upperBounds.count(a) || !inits.count(a))
      continue;
    std::string aSpatial, aKernel, aPad;
    if (!matchSpatialKernelMinusPad(inits.at(a), aSpatial, aKernel, aPad))
      continue;
    for (const std::string &b : boundedSyms) {
      if (a == b || !upperBounds.count(b) || !inits.count(b))
        continue;
      std::string bSpatial, bKernel, bPad;
      if (!matchSpatialKernelMinusPad(inits.at(b), bSpatial, bKernel, bPad))
        continue;
      if (aPad != bPad || !boundsEqual(upperBounds[a], upperBounds[b]))
        continue;

      const bool aIsCol = aSpatial == colIv;
      const bool bIsCol = bSpatial == colIv;
      if (aIsCol == bIsCol)
        continue;

      out.guardedIf = ifOp;
      out.pad = aPad;
      out.bound = upperBounds[a];
      if (aIsCol) {
        out.colSpatial = aSpatial;
        out.colKernel = aKernel;
        out.rowSpatial = bSpatial;
        out.rowKernel = bKernel;
      } else {
        out.rowSpatial = aSpatial;
        out.rowKernel = aKernel;
        out.colSpatial = bSpatial;
        out.colKernel = bKernel;
      }
      return true;
    }
  }
  return false;
}

bool findStencilBoundsIf(const Op *op,
                         const std::unordered_map<std::string, const Op*> &inits,
                         const std::string &colIv,
                         StencilBounds &out) {
  if (!op)
    return false;
  if (matchStencilBoundsIf(op, inits, colIv, out))
    return true;
  for (const auto &child : op->children)
    if (findStencilBoundsIf(child.get(), inits, colIv, out))
      return true;
  return false;
}

std::unique_ptr<Op> cloneDroppingStencilGuard(const Op *op, const Op *guardedIf) {
  if (!op)
    return nullptr;
  if (op == guardedIf && op->kind == OpKind::If && op->children.size() >= 2)
    return cloneOp(op->children[1].get());
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
    out->children.push_back(cloneDroppingStencilGuard(child.get(), guardedIf));
  return out;
}

std::unique_ptr<Op> makeInteriorCond(const StencilBounds &bounds) {
  auto rowLower = makeCmp("<=", makeLoad(bounds.pad), makeLoad(bounds.rowSpatial));
  auto rowUpper = makeCmp("<", makeLoad(bounds.rowSpatial),
                          makeArith("-", cloneOp(bounds.bound), makeLoad(bounds.pad)));
  auto colLower = makeCmp("<=", makeLoad(bounds.pad), makeLoad(bounds.colSpatial));
  auto colUpper = makeCmp("<", makeLoad(bounds.colSpatial),
                          makeArith("-", cloneOp(bounds.bound), makeLoad(bounds.pad)));
  return makeLogicalAnd(
      makeLogicalAnd(std::move(rowLower), std::move(rowUpper)),
      makeLogicalAnd(std::move(colLower), std::move(colUpper)));
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

bool exprLoadsArray(const Op *op, const std::string &symbol) {
  if (!op)
    return false;
  if (op->kind == OpKind::Load && !op->children.empty() && op->symbol == symbol)
    return true;
  for (const auto &child : op->children)
    if (exprLoadsArray(child.get(), symbol))
      return true;
  return false;
}

bool collectOverwriteStores(const Op *op, const std::string &repeatIV,
                            std::unordered_set<std::string> &overwritten,
                            bool &sawArrayStore) {
  if (!op)
    return true;
  if (exprUsesScalar(op, repeatIV))
    return false;
  switch (op->kind) {
  case OpKind::Call:
  case OpKind::Return:
  case OpKind::Break:
  case OpKind::Continue:
    return false;
  default:
    break;
  }
  if (op->kind == OpKind::Store && op->children.size() > 1 && !op->symbol.empty()) {
    const Op *rhs = op->children.back().get();
    if (exprLoadsArray(rhs, op->symbol))
      return false;
    overwritten.insert(op->symbol);
    sawArrayStore = true;
  }
  for (const auto &child : op->children)
    if (!collectOverwriteStores(child.get(), repeatIV, overwritten, sawArrayStore))
      return false;
  return true;
}

bool collectCalleeStores(const Op *func, std::unordered_set<std::string> &stores,
                         std::unordered_set<std::string> &visiting);

bool collectCalleeStoresImpl(const Op *op, std::unordered_set<std::string> &stores,
                             std::unordered_set<std::string> &visiting) {
  if (!op)
    return true;
  switch (op->kind) {
  case OpKind::Call:
    // Keep this analysis intraprocedural unless the call target was already
    // resolved by collectCalleeStores. Unknown nested calls may have visible
    // side effects, so do not remove repeated executions around them.
    return false;
  case OpKind::Return:
  case OpKind::Break:
  case OpKind::Continue:
    return op->kind == OpKind::Return;
  default:
    break;
  }
  if (op->kind == OpKind::Store && op->children.size() > 1 && !op->symbol.empty())
    stores.insert(op->symbol);
  for (const auto &child : op->children)
    if (!collectCalleeStoresImpl(child.get(), stores, visiting))
      return false;
  return true;
}

bool collectCalleeStores(const Op *func, std::unordered_set<std::string> &stores,
                         std::unordered_set<std::string> &visiting) {
  if (!func || func->kind != OpKind::Func || func->symbol.empty())
    return false;
  if (visiting.count(func->symbol))
    return false;
  visiting.insert(func->symbol);
  bool ok = true;
  for (const auto &child : func->children)
    ok = ok && collectCalleeStoresImpl(child.get(), stores, visiting);
  visiting.erase(func->symbol);
  return ok;
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
  for (const auto &idx : access.indices) {
    if (idx.coeffs.count(symbol))
      return true;
    for (const auto &[_, coeff] : idx.coeffs)
      if (coeff.symbols.count(symbol))
        return true;
  }
  return false;
}

bool accessMentionsPair(const affine::Access &access, const std::string &a,
                        const std::string &b) {
  bool hasA = false;
  bool hasB = false;
  for (const auto &idx : access.indices) {
    hasA = hasA || idx.coeffs.count(a);
    hasB = hasB || idx.coeffs.count(b);
    for (const auto &[_, coeff] : idx.coeffs) {
      hasA = hasA || coeff.symbols.count(a);
      hasB = hasB || coeff.symbols.count(b);
    }
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

bool writesScalarHere(const Op *op, const std::string &symbol) {
  op = unwrapSingleDecl(op);
  if (!op)
    return false;
  return (op->kind == OpKind::Store || op->kind == OpKind::VarDecl) &&
         op->symbol == symbol && op->children.size() <= 1;
}

bool scalarUsedBeforeRedef(const Op *block, size_t startIdx,
                           const std::string &symbol) {
  if (!block || block->kind != OpKind::Block)
    return true;
  for (size_t i = startIdx; i < block->children.size(); i++) {
    const Op *stmt = block->children[i].get();
    if (writesScalarHere(stmt, symbol))
      return false;
    if (exprUsesScalar(stmt, symbol))
      return true;
  }
  return false;
}

}  // namespace sys::hir::detail
