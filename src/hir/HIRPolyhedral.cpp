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

#include "HIRPolyDetail.h"
namespace sys::hir {

using namespace sys::hir::detail;


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
  functions.clear();
  for (const auto &child : module.root->children) {
    const Op *decl = unwrapSingleDecl(child.get());
    if (decl && decl->kind == OpKind::VarDecl && !decl->symbol.empty() &&
        !decl->arrayDims.empty())
      globalArrays.insert(decl->symbol);
    if (child && child->kind == OpKind::Func && !child->symbol.empty())
      functions[child->symbol] = child.get();
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
      continue;
    }
    if (block->children[i] && block->children[i]->kind == OpKind::While &&
        tryDeadOverwriteRepeat(block, i, stats)) {
      changed = true;
      i = 0;
    }
  }

  for (size_t i = 0; i < block->children.size(); i++) {
    if (block->children[i] && block->children[i]->kind == OpKind::While &&
        tryMonotoneGuardBoundTightening(block, i, stats)) {
      changed = true;
      i = 0;
      continue;
    }
    // The interior/boundary dispatcher is a general stencil transform for
    // convolution-like affine loops. It is still opt-in because cloning the
    // fast path can increase backend register pressure on large kernels.
    if (hirEnvEnabled("SISY_HIR_ENABLE_STENCIL_INTERIOR",
                      hirEnvEnabled("SISY_HIR_ENABLE_ADVANCED_CONV2D", false))) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryStencilInteriorDispatch(block, i, stats)) {
          changed = true;
          i = 0;
          continue;
        }
      }
    }
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

bool PolyhedralOptimizer::tryStencilInteriorDispatch(Op *block, size_t idx,
                                                     PolyhedralStats &stats) {
  if (!block || idx >= block->children.size()) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }

  Op *whileOp = block->children[idx].get();
  CanonicalLoop colLoop;
  if (!matchCanonicalWhile(whileOp, colLoop) || !colLoop.body ||
      colLoop.body->kind != OpKind::Block || colLoop.body->children.size() < 2) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }

  // Idempotency guard: after dispatch the loop body is a single if whose
  // branches contain the original step.
  if (colLoop.body->children.size() == 1 &&
      colLoop.body->children[0] &&
      colLoop.body->children[0]->kind == OpKind::If) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }

  std::unordered_map<std::string, const Op*> scalarInits;
  collectScalarInitializers(colLoop.body, scalarInits);

  StencilBounds bounds;
  if (!findStencilBoundsIf(colLoop.body, scalarInits, colLoop.iv, bounds)) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectBounds++;
    return false;
  }

  if (bounds.rowSpatial.empty() || bounds.colSpatial != colLoop.iv ||
      bounds.pad.empty() || !bounds.bound || !bounds.guardedIf) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectBounds++;
    return false;
  }

  // The fast path is valid only if the spatial row is invariant within this
  // column loop. Otherwise the interior guard would not cover every cloned use.
  if (bodyWritesScalar(colLoop.body, bounds.rowSpatial) ||
      bodyWritesScalar(colLoop.body, bounds.pad)) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectBounds++;
    return false;
  }

  auto fastBody = cloneDroppingStencilGuard(colLoop.body, bounds.guardedIf);
  auto slowBody = cloneOp(colLoop.body);
  if (!fastBody || !slowBody) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }

  auto dispatchedBody = makeBlock();
  dispatchedBody->children.push_back(
      makeIf(makeInteriorCond(bounds), std::move(fastBody), std::move(slowBody)));

  auto newWhile = makeWhile(cloneOp(whileOp->children[0].get()), std::move(dispatchedBody));
  block->children[idx] = std::move(newWhile);
  stats.stencilInteriorDispatched++;
  return true;
}

bool PolyhedralOptimizer::tryMonotoneGuardBoundTightening(
    Op *block, size_t idx, PolyhedralStats &stats) {
  if (!block || block->kind != OpKind::Block || idx >= block->children.size()) {
    stats.monotoneGuardRejected++;
    stats.monotoneGuardRejectShape++;
    return false;
  }

  Op *whileOp = block->children[idx].get();
  CanonicalLoop loop;
  if (!matchCanonicalWhile(whileOp, loop) || !loop.body ||
      loop.body->kind != OpKind::Block || loop.body->children.size() < 2) {
    stats.monotoneGuardRejected++;
    stats.monotoneGuardRejectShape++;
    return false;
  }

  std::string limitScalar;
  if (!matchPrefixSkipGuard(loop.body->children.front().get(), loop.iv, limitScalar)) {
    stats.monotoneGuardRejected++;
    stats.monotoneGuardRejectShape++;
    return false;
  }

  // The transform shortens the loop and removes the continue.  It is only
  // semantics-preserving when the guard scalar is invariant inside the loop,
  // and when the final value of the shortened induction variable is not used
  // before being redefined.  In the original loop the IV would finish at the
  // old bound; after tightening it finishes at min(oldBound, limit + 1).
  if (bodyWritesScalar(loop.body, limitScalar) ||
      scalarUsedBeforeRedef(block, idx + 1, loop.iv)) {
    stats.monotoneGuardRejected++;
    stats.monotoneGuardRejectUse++;
    return false;
  }

  const std::string stopVar =
      "__hir_guard_stop_" + loop.iv + "_" + std::to_string(uniqueId++);

  auto stopInit = makeVarDecl(
      stopVar,
      makeArith("+", makeLoad(limitScalar), makeConstInt(1)));

  auto clampThen = makeBlock();
  clampThen->children.push_back(makeStore(stopVar, cloneOp(loop.bound)));
  auto clampIf = makeIf(
      makeCmp("<", cloneOp(loop.bound), makeLoad(stopVar)),
      std::move(clampThen),
      nullptr);

  auto newBody = makeBlock();
  for (size_t i = 1; i < loop.body->children.size(); i++)
    newBody->children.push_back(cloneOp(loop.body->children[i].get()));
  auto newWhile =
      makeWhile(makeCmp("<", makeLoad(loop.iv), makeLoad(stopVar)),
                std::move(newBody));

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 2);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(std::move(stopInit));
      replacement.push_back(std::move(clampIf));
      replacement.push_back(std::move(newWhile));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.monotoneGuardTightened++;
  return true;
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

bool PolyhedralOptimizer::tryDeadOverwriteRepeat(Op *block, size_t idx,
                                                  PolyhedralStats &stats) {
  if (!block || idx >= block->children.size() || idx == 0)
    return false;

  Op *whileOp = block->children[idx].get();
  CanonicalLoop loop;
  if (!matchCanonicalWhile(whileOp, loop) || !loop.body ||
      loop.body->kind != OpKind::Block || loop.body->children.size() < 3) {
    stats.overwriteRepeatRejected++;
    return false;
  }
  if (!matchLoopInit(block->children[idx - 1].get(), loop.iv)) {
    stats.overwriteRepeatRejected++;
    return false;
  }
  const Op *start = initValue(block->children[idx - 1].get());
  if (!isConstIntValue(start, 0) || !loop.bound ||
      loop.bound->kind != OpKind::ConstInt || !loop.bound->hasIntValue ||
      loop.bound->intValue <= 0) {
    stats.overwriteRepeatRejected++;
    return false;
  }
  if (exprUsesScalar(loop.bound, loop.iv) ||
      blockExceptLastUsesScalar(loop.body, loop.iv) ||
      affine::opWritesAnyScalarUsedBy(loop.body, loop.bound)) {
    stats.overwriteRepeatRejected++;
    return false;
  }

  std::unordered_set<std::string> overwritten;
  bool sawArrayStore = false;
  size_t firstCall = loop.body->children.size();
  for (size_t i = 0; i + 1 < loop.body->children.size(); i++) {
    Op *stmt = unwrapSingleDecl(loop.body->children[i].get());
    if (stmt && stmt->kind == OpKind::Call) {
      firstCall = i;
      break;
    }
    if (!collectOverwriteStores(loop.body->children[i].get(), loop.iv,
                                overwritten, sawArrayStore)) {
      stats.overwriteRepeatRejected++;
      return false;
    }
  }
  if (!sawArrayStore || overwritten.empty() || firstCall >= loop.body->children.size() - 1) {
    stats.overwriteRepeatRejected++;
    return false;
  }

  for (size_t i = firstCall; i + 1 < loop.body->children.size(); i++) {
    Op *stmt = unwrapSingleDecl(loop.body->children[i].get());
    if (!stmt || exprUsesScalar(stmt, loop.iv)) {
      stats.overwriteRepeatRejected++;
      return false;
    }
    if (stmt->kind == OpKind::Call) {
      auto calleeIt = functions.find(stmt->symbol);
      if (calleeIt == functions.end()) {
        stats.overwriteRepeatRejected++;
        return false;
      }
      std::unordered_set<std::string> stores;
      std::unordered_set<std::string> visiting;
      if (!collectCalleeStores(calleeIt->second, stores, visiting)) {
        stats.overwriteRepeatRejected++;
        return false;
      }
      for (const auto &sym : stores) {
        if (!overwritten.count(sym)) {
          stats.overwriteRepeatRejected++;
          return false;
        }
      }
      continue;
    }
    std::unordered_set<std::string> stores;
    bool localArrayStore = false;
    if (!collectOverwriteStores(stmt, loop.iv, stores, localArrayStore)) {
      stats.overwriteRepeatRejected++;
      return false;
    }
    for (const auto &sym : stores) {
      if (!overwritten.count(sym)) {
        stats.overwriteRepeatRejected++;
        return false;
      }
    }
  }

  auto bodyOnce = cloneBlockWithoutLast(loop.body);
  if (!bodyOnce) {
    stats.overwriteRepeatRejected++;
    return false;
  }

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 1);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(std::move(bodyOnce));
      replacement.push_back(makeStore(loop.iv, cloneOp(loop.bound)));
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }
  block->children = std::move(replacement);
  stats.overwriteRepeatReduced++;
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
