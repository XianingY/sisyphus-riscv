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
}  // namespace sys::hir
