#include "HIRPolyhedral.h"
#include "HIRPolyDetail.h"

#include "HIRAffine.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sys::hir {

using namespace sys::hir::detail;

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

namespace {

bool containsCarriedScalarStore(const Op *op, const std::string &iv) {
  if (!op)
    return false;
  if (op->kind == OpKind::Store && op->children.size() == 1 &&
      op->symbol != iv)
    return true;
  for (const auto &child : op->children)
    if (containsCarriedScalarStore(child.get(), iv))
      return true;
  return false;
}

void collectLocalScalarDecls(const Op *op, std::unordered_set<std::string> &syms) {
  if (!op)
    return;
  if (op->kind == OpKind::VarDecl && op->arrayDims.empty() &&
      !op->symbol.empty())
    syms.insert(op->symbol);
  for (const auto &child : op->children)
    collectLocalScalarDecls(child.get(), syms);
}

}  // namespace

bool PolyhedralOptimizer::tryInnermostPartialUnroll(
    Op *block, size_t idx, PolyhedralStats &stats) {
  if (!block || (block->kind != OpKind::Block && block->kind != OpKind::Module) ||
      idx >= block->children.size()) {
    stats.partialUnrollRejected++;
    stats.partialUnrollRejectShape++;
    return false;
  }

  Op *whileOp = block->children[idx].get();
  if (!monotoneTightenedLoops.count(whileOp))
    return false;
  if (partialUnrollRemainders.count(whileOp))
    return false;

  CanonicalLoop loop;
  if (!matchCanonicalWhile(whileOp, loop) || !loop.body ||
      loop.body->kind != OpKind::Block || loop.body->children.size() < 2 ||
      containsWhile(loop.body)) {
    stats.partialUnrollRejected++;
    stats.partialUnrollRejectShape++;
    return false;
  }

  if (!tilingSafeBody(loop.body) ||
      blockExceptLastWritesScalar(loop.body, loop.iv) ||
      containsCarriedScalarStore(loop.body, loop.iv)) {
    stats.partialUnrollRejected++;
    stats.partialUnrollRejectControl++;
    return false;
  }

  // This transform only duplicates adjacent iterations in their original
  // order; unlike interchange it does not need an affine dependence proof.
  // Requiring affine recovery here would reject linearized multidimensional
  // addresses with symbolic row strides, including triangular transpose.
  if (countArrayAccessOps(loop.body) < 2) {
    stats.partialUnrollRejected++;
    stats.partialUnrollRejectAccess++;
    return false;
  }

  std::unordered_set<std::string> localScalars;
  collectLocalScalarDecls(loop.body, localScalars);
  localScalars.erase(loop.iv);

  constexpr int kFactor = 2;
  auto unrolledBody = makeBlock();
  for (int lane = 0; lane < kFactor; lane++) {
    std::unordered_map<std::string, std::string> scalarRenames;
    if (lane != 0) {
      for (const auto &scalar : localScalars) {
        scalarRenames[scalar] =
            "__hir_partial_" + std::to_string(uniqueId) + "_" +
            std::to_string(lane) + "_" + scalar;
      }
    }
    std::unordered_map<std::string, int> ivOffsets = {{loop.iv, lane}};
    for (size_t stmt = 0; stmt + 1 < loop.body->children.size(); stmt++) {
      unrolledBody->children.push_back(
          cloneReplacing(loop.body->children[stmt].get(), scalarRenames, ivOffsets));
    }
  }
  uniqueId++;
  unrolledBody->children.push_back(makeJamStep(loop.iv, kFactor));

  auto unrolledCond =
      makeCmp("<",
              makeArith("+", makeLoad(loop.iv), makeConstInt(kFactor - 1)),
              cloneOp(loop.bound));
  auto unrolledWhile = makeWhile(std::move(unrolledCond), std::move(unrolledBody));
  auto remainderWhile = cloneOp(whileOp);
  partialUnrollRemainders.insert(remainderWhile.get());

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(std::move(unrolledWhile));
      replacement.push_back(std::move(remainderWhile));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.partialUnrolled++;
  return true;
}

}  // namespace sys::hir
