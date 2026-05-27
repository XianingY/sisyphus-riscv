// HIRPolyDetail.h
//
// Internal helpers and pattern-matching primitives shared by the polyhedral
// pass implementations split out of HIRPolyhedral.cpp. Intentionally kept
// inside namespace `sys::hir::detail` so it stays a private compile-time
// detail of the HIR polyhedral pass family. Do not include from anywhere
// outside src/hir/HIRPoly*.cpp.

#pragma once

#include "HIROps.h"
#include "HIRAffine.h"

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sys::hir::detail {

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int kJamFactor = 4;
constexpr int kDefaultHirTileSize = 32;

// ---------------------------------------------------------------------------
// Shared structs
// ---------------------------------------------------------------------------
struct LoopBodyMetrics {
  int scalarIntDefs = 0;
  int scalarFloatDefs = 0;
  std::unordered_set<std::string> arrayReadStreams;
  std::unordered_set<std::string> arrayWriteStreams;
};

struct CanonicalLoop {
  std::string iv;
  const Op *bound = nullptr;
  Op *body = nullptr;
  Op *step = nullptr;
};

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

struct ReductionTilePlan {
  int mr = 1;
  int nr = 0;
  int kc = 0;
  int nc = 0;
  bool needsScratch = true;
};

struct StencilBounds {
  std::string rowSpatial;
  std::string colSpatial;
  std::string rowKernel;
  std::string colKernel;
  std::string pad;
  const Op *bound = nullptr;
  const Op *guardedIf = nullptr;
};

// ---------------------------------------------------------------------------
// Helper function declarations (definitions live in HIRPolyDetail.cpp)
// ---------------------------------------------------------------------------
void analyzeLoopBody(const Op *op, LoopBodyMetrics &metrics);
TypeKind detectMainType(const Op *op);
int computeOptimalJamFactor(const Op *innerBody, TypeKind mainType);
int computeOptimalTileSize(TypeKind mainType);
int computeOptimalTileSize(TypeKind mainType, const Op *innerBody);
ReductionTilePlan computeReductionTilePlan(const Op *innerBody,
                                           TypeKind mainType,
                                           bool needsScratch);

bool hirEnvEnabled(const char *name, bool fallback);
int hirEnvInt(const char *name, int fallback);

std::unique_ptr<Op> cloneOp(const Op *op);
std::unique_ptr<Op> makeConstInt(long long value);
std::unique_ptr<Op> makeLoad(const std::string &symbol);
std::unique_ptr<Op> makeArith(const std::string &symbol,
                              std::unique_ptr<Op> lhs,
                              std::unique_ptr<Op> rhs);
std::unique_ptr<Op> makeCmp(const std::string &symbol,
                            std::unique_ptr<Op> lhs,
                            std::unique_ptr<Op> rhs);
std::unique_ptr<Op> makeStore(const std::string &symbol,
                              std::unique_ptr<Op> value);
std::unique_ptr<Op> makeArrayStoreLike(const Op *store,
                                       std::unique_ptr<Op> value);
std::unique_ptr<Op> makeVarDecl(const std::string &symbol,
                                std::unique_ptr<Op> value);
std::unique_ptr<Op> makeBlock();
std::unique_ptr<Op> makeWhile(std::unique_ptr<Op> cond,
                              std::unique_ptr<Op> body);
std::unique_ptr<Op> makeIf(std::unique_ptr<Op> cond,
                           std::unique_ptr<Op> thenBlock,
                           std::unique_ptr<Op> elseBlock);
std::unique_ptr<Op> makeLogicalAnd(std::unique_ptr<Op> lhs,
                                   std::unique_ptr<Op> rhs);
std::unique_ptr<Op> makeIndexWithOffset(const std::string &iv, int offset);

bool isScalarLoad(const Op *op, const std::string &symbol);
bool isConstIntValue(const Op *op, long long value);
bool isScalarLoadAny(const Op *op, std::string &symbol);
const Op *unwrapSingleDecl(const Op *op);
Op *unwrapSingleDecl(Op *op);

bool matchStepStore(Op *op, const std::string &iv, int expectedStep);
bool matchContinueStepThenBlock(const Op *op, const std::string &iv);
bool matchPrefixSkipGuard(const Op *op, const std::string &iv,
                          std::string &limitScalar);
bool matchCanonicalWhile(Op *op, CanonicalLoop &loop);
bool matchLoopInit(const Op *op, const std::string &iv);

bool exprUsesScalar(const Op *op, const std::string &symbol);
bool isDirectScalarExpr(const Op *op, const std::string &symbol);
bool arrayIndexUsesOnlyDirectIV(const Op *op, const std::string &iv);
bool isAdditiveReductionUpdate(const Op *store, const std::string &acc);
bool containsArrayAccessTo(const Op *op, const std::string &symbol);
void collectArrayAccessSymbols(const Op *op,
                               std::unordered_set<std::string> &symbols);

std::unique_ptr<Op> makeLoadFromStoreDestination(const Op *store);
std::unique_ptr<Op> cloneReplacingScalarLoad(const Op *op,
                                             const std::string &scalar,
                                             const Op *replacement);
std::unique_ptr<Op> cloneReplacing(
    const Op *op,
    const std::unordered_map<std::string, std::string> &scalarRenames,
    const std::unordered_map<std::string, int> &ivOffsets);

bool matchVarInit(const Op *op, std::string &symbol);
const Op *initValue(const Op *op);
bool hasUnsafeReductionControl(const Op *op);
bool containsWhile(const Op *op);
Op *findAdditiveReductionUpdate(Op *stmt, const std::string &acc);
bool matchReductionPattern(Op *jWhile, ReductionPattern &pat);
bool strictReductionInterchangeLegal(
    const ReductionPattern &pat,
    const std::unordered_set<std::string> &globalArrays);

std::unique_ptr<Op> makeJamStep(const std::string &iv, int factor);
std::unique_ptr<Op> makeReductionInitLoop(const ReductionPattern &pat);
std::unique_ptr<Op> makeReductionLmsStore(const ReductionPattern &pat);
std::unique_ptr<Op> cloneReductionStmtToDestination(const Op *op,
                                                    const ReductionPattern &pat);
std::unique_ptr<Op> makeReductionLmsJLoop(const ReductionPattern &pat);
std::unique_ptr<Op> makeReductionInterchangedKLoop(const ReductionPattern &pat,
                                                   const Op *jInitOp);

bool tilingSafeBody(const Op *op);
bool boundsEqual(const Op *a, const Op *b);
void collectScalarInitializers(const Op *op,
                               std::unordered_map<std::string, const Op *> &inits);
bool matchSpatialKernelMinusPad(const Op *expr, std::string &spatial,
                                std::string &kernel, std::string &pad);
bool flattenAndTerms(const Op *expr, std::vector<const Op *> &terms);
bool matchStencilBoundsIf(
    const Op *ifOp,
    const std::unordered_map<std::string, const Op *> &scalarInits,
    const std::string &expectedColIv, StencilBounds &bounds);
bool findStencilBoundsIf(
    const Op *op,
    const std::unordered_map<std::string, const Op *> &scalarInits,
    const std::string &expectedColIv, StencilBounds &bounds);
std::unique_ptr<Op> cloneDroppingStencilGuard(const Op *op,
                                              const Op *guardedIf);
std::unique_ptr<Op> makeInteriorCond(const StencilBounds &bounds);

bool isArrayStore(const Op *op);
Op *asArrayStore(Op *op);
Op *asScalarVarDecl(Op *op);
bool sameArrayAddress(const Op *store, const Op *load);
bool collectDefinedScalars(const Op *block,
                           std::unordered_set<std::string> &syms);
void collectTopLevelInitializedScalars(const Op *block,
                                       std::unordered_set<std::string> &syms);
bool bodyUsesAnyOf(const Op *op,
                   const std::unordered_set<std::string> &syms);
bool bodyWritesScalar(const Op *op, const std::string &symbol);
bool bodyWritesNonLoopScalar(const Op *op,
                             const std::unordered_set<std::string> &loopIVs);
bool blockExceptLastWritesScalar(const Op *block, const std::string &symbol);
std::unique_ptr<Op> cloneBlockWithoutLast(const Op *block);
bool collectCanonicalLoopIVs(const Op *op,
                             std::unordered_set<std::string> &ivs);
const Op *additiveDeltaExpr(const Op *store, const std::string &acc);
bool repeatBodyLegalImpl(const Op *op,
                         const std::unordered_set<std::string> &loopIVs,
                         const std::string &repeatIV, std::string &acc,
                         int &accUpdates);
bool repeatBodyLegal(const Op *body, const std::string &repeatIV,
                     std::string &acc);
bool exprLoadsArray(const Op *op, const std::string &symbol);
bool collectOverwriteStores(const Op *op, const std::string &repeatIV,
                            std::unordered_set<std::string> &overwritten,
                            bool &sawArrayStore);
bool collectCalleeStores(const Op *func,
                         std::unordered_set<std::string> &stores,
                         std::unordered_set<std::string> &visiting);
bool blockExceptLastUsesScalar(const Op *block, const std::string &symbol);
bool hasUnsafeAffineScanControl(const Op *op);
int countArrayAccessOps(const Op *op);
Op *findSingleDirectInnerWhile(Op *body);
bool accessMentions(const affine::Access &access, const std::string &symbol);
bool accessMentionsPair(const affine::Access &access, const std::string &a,
                        const std::string &b);
bool isMatmulLikeNest(const std::vector<CanonicalLoop> &loops,
                      const std::vector<affine::Access> &accesses);
bool fusionWithinCacheBudget(const Op *bodyA, const Op *bodyB);
bool writesScalarHere(const Op *op, const std::string &symbol);
bool scalarUsedBeforeRedef(const Op *block, size_t startIdx,
                           const std::string &symbol);

}  // namespace sys::hir::detail
