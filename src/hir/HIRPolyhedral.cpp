#include "HIRPolyhedral.h"

#include "HIRAffine.h"
#include "../backend/riscv/RiscvParams.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "HIRPolyDetail.h"
namespace sys::hir {

using namespace sys::hir::detail;

namespace {

constexpr const char *kStencilSplitLoopMarker = "__hir_stencil_split_loop";

bool termVariesInLoop(const Op *op, const Op *body) {
  if (!op)
    return false;
  if (op->kind == OpKind::Call)
    return true;
  if (op->kind == OpKind::Load) {
    if (!op->children.empty())
      return true;
    if (bodyWritesScalar(body, op->symbol))
      return true;
  }
  for (const auto &child : op->children)
    if (termVariesInLoop(child.get(), body))
      return true;
  return false;
}

std::unique_ptr<Op> makeConjunction(const std::vector<const Op *> &terms) {
  if (terms.empty())
    return nullptr;
  auto result = cloneOp(terms.front());
  for (size_t i = 1; i < terms.size(); i++)
    result = makeLogicalAnd(std::move(result), cloneOp(terms[i]));
  return result;
}

bool findSplitGuard(const Op *op, const Op *loopBody, const Op *&guard,
                    std::vector<const Op *> &invariant,
                    std::vector<const Op *> &varying) {
  if (!op)
    return false;
  if (op->kind == OpKind::If && op->children.size() == 2) {
    std::vector<const Op *> terms;
    if (flattenAndTerms(op->children[0].get(), terms) && terms.size() >= 2) {
      for (const Op *term : terms) {
        if (termVariesInLoop(term, loopBody))
          varying.push_back(term);
        else
          invariant.push_back(term);
      }
      if (!invariant.empty() && !varying.empty()) {
        guard = op;
        return true;
      }
      invariant.clear();
      varying.clear();
    }
  }
  for (const auto &child : op->children)
    if (findSplitGuard(child.get(), loopBody, guard, invariant, varying))
      return true;
  return false;
}

bool guardDominatesObservableWork(const Op *op, const Op *guard,
                                  const Op *step) {
  if (!op)
    return true;
  if (op == guard)
    return true;
  switch (op->kind) {
  case OpKind::Block:
  case OpKind::Arith:
  case OpKind::Cmp:
  case OpKind::ConstInt:
  case OpKind::ConstFloat:
  case OpKind::Load:
    break;
  case OpKind::VarDecl:
    if (!op->arrayDims.empty())
      return false;
    break;
  case OpKind::Store:
    if (op != step)
      return false;
    break;
  default:
    return false;
  }
  for (const auto &child : op->children)
    if (!guardDominatesObservableWork(child.get(), guard, step))
      return false;
  return true;
}

std::unique_ptr<Op> cloneReplacingGuardCond(const Op *op, const Op *guard,
                                            const Op *newCondition) {
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
  for (size_t i = 0; i < op->children.size(); i++) {
    if (op == guard && i == 0)
      out->children.push_back(cloneOp(newCondition));
    else
      out->children.push_back(
          cloneReplacingGuardCond(op->children[i].get(), guard, newCondition));
  }
  return out;
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
  globalArrayDims.clear();
  monotoneTightenedLoops.clear();
  partialUnrollRemainders.clear();
  functions.clear();
  for (const auto &child : module.root->children) {
    const Op *decl = unwrapSingleDecl(child.get());
    if (decl && decl->kind == OpKind::VarDecl && !decl->symbol.empty() &&
        !decl->arrayDims.empty()) {
      globalArrays.insert(decl->symbol);
      globalArrayDims[decl->symbol] = decl->arrayDims;
    }
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

    // Stencil border peeling must run before recursive descent. Otherwise a
    // nested guard-hoist pass can split the full boundary conjunction inside
    // the kernel loop before this parent spatial loop has a chance to create
    // the left/interior/right regions.
    if (hirEnvEnabled("SISY_HIR_ENABLE_STENCIL_INTERIOR",
                      hirEnvEnabled("SISY_HIR_ENABLE_ADVANCED_CONV2D", false))) {
      for (size_t i = 0; i < block->children.size(); i++) {
        if (block->children[i] && block->children[i]->kind == OpKind::While &&
            tryStencilInteriorDispatch(block, i, stats)) {
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
    if (tryReductionRowPrivatize(block, i, stats)) {
      changed = true;
      i = 0;
      continue;
    }
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
    // The interior/boundary dispatcher is a general stencil transform for
    // guarded affine loops. Run it before generic guard splitting so it can see
    // the full conjunction of lower/upper bounds and peel the long contiguous
    // interior range out of the boundary path.
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
    if (hirEnvEnabled("SISY_HIR_ENABLE_INVARIANT_GUARD_HOIST", true) &&
        block->children[i] && block->children[i]->kind == OpKind::While &&
        tryInvariantGuardHoist(block, i, stats)) {
      changed = true;
      i = 0;
      continue;
    }
    if (block->children[i] && block->children[i]->kind == OpKind::While &&
        tryMonotoneGuardBoundTightening(block, i, stats)) {
      changed = true;
      i = 0;
      continue;
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
    // Partial unroll preserves the original iteration order.  It is useful
    // after guarded triangular loops have been tightened into a straight
    // affine inner loop, and does not require dependence permutation.
    if (hirEnvEnabled("SISY_HIR_ENABLE_PARTIAL_UNROLL", true)) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryInnermostPartialUnroll(block, i, stats)) {
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
    // Loop distribution / fission: split a single canonical loop body into
    // two adjacent loops when the two halves access disjoint arrays and
    // scalars. Default off — enables more aggressive vectorization on
    // mixed-stream loops (e.g. FFT butterfly, conv with bias) at the cost
    // of doubling the loop overhead. Toggle via SISY_HIR_ENABLE_DISTRIBUTE.
    if (hirEnvEnabled("SISY_HIR_ENABLE_DISTRIBUTE", false)) {
      if (block->children[i] && block->children[i]->kind == OpKind::While) {
        if (tryLoopDistribute(block, i, stats)) {
          changed = true;
          i = 0;
          continue;
        }
      }
    }
  }
  return changed;
}

bool PolyhedralOptimizer::tryInvariantGuardHoist(Op *block, size_t idx,
                                                 PolyhedralStats &stats) {
  if (!block || idx >= block->children.size())
    return false;

  CanonicalLoop loop;
  Op *whileOp = block->children[idx].get();
  if (!matchCanonicalWhile(whileOp, loop) || !loop.body ||
      loop.body->kind != OpKind::Block) {
    stats.invariantGuardRejected++;
    return false;
  }

  const Op *guard = nullptr;
  std::vector<const Op *> invariantTerms;
  std::vector<const Op *> varyingTerms;
  if (!findSplitGuard(loop.body, loop.body, guard, invariantTerms, varyingTerms))
    return false;

  // If the guard is false, the original loop may only have computed local
  // temporaries and advanced its now-dead IV. Otherwise skipping its
  // iterations would remove observable effects.
  if (!guardDominatesObservableWork(loop.body, guard, loop.step) ||
      scalarUsedBeforeRedef(block, idx + 1, loop.iv)) {
    stats.invariantGuardRejected++;
    return false;
  }

  auto invariantCond = makeConjunction(invariantTerms);
  auto varyingCond = makeConjunction(varyingTerms);
  if (!invariantCond || !varyingCond) {
    stats.invariantGuardRejected++;
    return false;
  }

  auto innerBody =
      cloneReplacingGuardCond(loop.body, guard, varyingCond.get());
  if (!innerBody) {
    stats.invariantGuardRejected++;
    return false;
  }
  auto thenBody = makeBlock();
  thenBody->children.push_back(
      makeWhile(cloneOp(whileOp->children[0].get()), std::move(innerBody)));
  block->children[idx] =
      makeIf(std::move(invariantCond), std::move(thenBody), nullptr);
  stats.invariantGuardHoisted++;
  return true;
}

bool PolyhedralOptimizer::tryStencilInteriorDispatch(Op *block, size_t idx,
                                                     PolyhedralStats &stats) {
  if (!block || idx >= block->children.size()) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }

  Op *whileOp = block->children[idx].get();
  if (whileOp->symbol == kStencilSplitLoopMarker) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }
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
  auto leftSlowBody = cloneOp(colLoop.body);
  auto rightSlowBody = cloneOp(colLoop.body);
  if (!fastBody || !slowBody || !leftSlowBody || !rightSlowBody) {
    stats.stencilInteriorRejected++;
    stats.stencilInteriorRejectShape++;
    return false;
  }

  if (hirEnvEnabled("SISY_HIR_STENCIL_SPLIT_COLUMNS", true)) {
    // Peel the guarded stencil loop into:
    //   left boundary:   original guarded body while c < min(bound, pad)
    //   interior:        guard-free body      while c < bound - pad
    //   right boundary:  original guarded body while c < bound
    //
    // This is the affine/border-peeling form used by production stencil
    // optimizers.  It removes the per-pixel interior branch from the long
    // contiguous middle region while preserving the original guarded body for
    // both borders.  The final value of the column IV may differ when
    // bound < pad, so require that it is not observed before a redefinition.
    if (scalarUsedBeforeRedef(block, idx + 1, colLoop.iv)) {
      stats.stencilInteriorRejected++;
      stats.stencilInteriorRejectBounds++;
      return false;
    }

    auto leftCond = makeLogicalAnd(
        cloneOp(whileOp->children[0].get()),
        makeCmp("<", makeLoad(colLoop.iv), makeLoad(bounds.pad)));
    auto middleStop =
        makeArith("-", cloneOp(bounds.bound), makeLoad(bounds.pad));
    auto middleCond =
        makeCmp("<", makeLoad(colLoop.iv), std::move(middleStop));
    auto rightCond = cloneOp(whileOp->children[0].get());
    if (!leftCond || !middleCond || !rightCond) {
      stats.stencilInteriorRejected++;
      stats.stencilInteriorRejectShape++;
      return false;
    }

    auto rowLower = makeCmp("<=", makeLoad(bounds.pad), makeLoad(bounds.rowSpatial));
    auto rowUpper = makeCmp("<", makeLoad(bounds.rowSpatial),
                            makeArith("-", cloneOp(bounds.bound),
                                      makeLoad(bounds.pad)));
    auto rowInterior = makeLogicalAnd(std::move(rowLower), std::move(rowUpper));
    if (!rowInterior) {
      stats.stencilInteriorRejected++;
      stats.stencilInteriorRejectShape++;
      return false;
    }

    auto middleBody = makeBlock();
    middleBody->children.push_back(
        makeIf(std::move(rowInterior), std::move(fastBody), std::move(slowBody)));

    std::vector<std::unique_ptr<Op>> replacement;
    auto leftLoop = makeWhile(std::move(leftCond), std::move(leftSlowBody));
    auto middleLoop = makeWhile(std::move(middleCond), std::move(middleBody));
    auto rightLoop = makeWhile(std::move(rightCond), std::move(rightSlowBody));
    leftLoop->symbol = kStencilSplitLoopMarker;
    middleLoop->symbol = kStencilSplitLoopMarker;
    rightLoop->symbol = kStencilSplitLoopMarker;
    replacement.push_back(std::move(leftLoop));
    replacement.push_back(makeStore(colLoop.iv, makeLoad(bounds.pad)));
    replacement.push_back(std::move(middleLoop));
    replacement.push_back(std::move(rightLoop));

    auto first = block->children.begin() + static_cast<std::ptrdiff_t>(idx);
    first = block->children.erase(first);
    block->children.insert(first,
                           std::make_move_iterator(replacement.begin()),
                           std::make_move_iterator(replacement.end()));
    stats.stencilInteriorDispatched++;
    return true;
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
  Op *tightenedWhile = newWhile.get();

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
  monotoneTightenedLoops.insert(tightenedWhile);
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
bool PolyhedralOptimizer::tryLoopDistribute(Op *block, size_t idx,
                                             PolyhedralStats &stats) {
  if (!block || idx >= block->children.size())
    return false;
  Op *whileOp = block->children[idx].get();
  if (!whileOp || whileOp->kind != OpKind::While)
    return false;

  CanonicalLoop loop;
  if (!matchCanonicalWhile(whileOp, loop) || !loop.body ||
      loop.body->kind != OpKind::Block) {
    stats.loopDistributionRejected++;
    stats.loopDistributionRejectShape++;
    return false;
  }

  // Need at least two non-step statements in the body to consider splitting.
  auto &bodyChildren = loop.body->children;
  if (bodyChildren.size() < 3) {
    stats.loopDistributionRejected++;
    stats.loopDistributionRejectShape++;
    return false;
  }

  // Conservative: bail on any control flow / calls / nested loops in the body.
  // tilingSafeBody already rejects calls, breaks, returns, etc.
  if (!tilingSafeBody(loop.body) || containsWhile(loop.body)) {
    stats.loopDistributionRejected++;
    stats.loopDistributionRejectControl++;
    return false;
  }

  // For correctness when re-emitting the loop twice, we require an init
  // statement at children[idx-1] of the form `iv = start` so the second
  // copy can be preceded by a re-init.
  if (idx == 0 || !matchLoopInit(block->children[idx - 1].get(), loop.iv)) {
    stats.loopDistributionRejected++;
    stats.loopDistributionRejectShape++;
    return false;
  }

  const size_t numStmts = bodyChildren.size() - 1;  // exclude step at end
  if (bodyChildren[numStmts].get() != loop.step &&
      !matchStepStore(bodyChildren[numStmts].get(), loop.iv, 1)) {
    stats.loopDistributionRejected++;
    stats.loopDistributionRejectShape++;
    return false;
  }

  // Try each candidate split point K (1..numStmts-1). For each, test that
  // the two halves access disjoint arrays and disjoint scalars (besides
  // the loop IV). This is a conservative legality check that guarantees
  // no flow / anti / output dependence across the split — same iteration
  // and loop-carried alike.
  size_t bestK = 0;
  for (size_t K = 1; K < numStmts; K++) {
    // Array symbol disjointness.
    std::unordered_set<std::string> arr1, arr2;
    for (size_t i = 0; i < K; i++)
      collectArrayAccessSymbols(bodyChildren[i].get(), arr1);
    for (size_t i = K; i < numStmts; i++)
      collectArrayAccessSymbols(bodyChildren[i].get(), arr2);

    bool overlap = false;
    for (const auto &s : arr1) {
      if (arr2.count(s)) { overlap = true; break; }
    }
    if (overlap) continue;

    // Scalar def disjointness (plus IV — fine, both halves use IV).
    std::unordered_set<std::string> scal1, scal2;
    for (size_t i = 0; i < K; i++)
      collectDefinedScalars(bodyChildren[i].get(), scal1);
    scal1.erase(loop.iv);
    for (size_t i = K; i < numStmts; i++)
      collectDefinedScalars(bodyChildren[i].get(), scal2);
    scal2.erase(loop.iv);

    // No half may consume scalars defined by the other half.
    bool scalarFlow = false;
    for (size_t i = K; i < numStmts; i++) {
      if (bodyUsesAnyOf(bodyChildren[i].get(), scal1)) { scalarFlow = true; break; }
    }
    if (scalarFlow) continue;
    for (size_t i = 0; i < K; i++) {
      if (bodyUsesAnyOf(bodyChildren[i].get(), scal2)) { scalarFlow = true; break; }
    }
    if (scalarFlow) continue;

    bestK = K;
    break;
  }

  if (bestK == 0) {
    stats.loopDistributionRejected++;
    stats.loopDistributionRejectNoSplit++;
    return false;
  }

  // Build two new while ops, each with its own copy of the body half plus
  // the original step. The condition op is cloned for each.
  auto cloneStep = [&]() { return cloneOp(bodyChildren[numStmts].get()); };
  auto buildWhile = [&](size_t lo, size_t hi) -> std::unique_ptr<Op> {
    auto newBody = makeBlock();
    for (size_t i = lo; i < hi; i++)
      newBody->children.push_back(cloneOp(bodyChildren[i].get()));
    newBody->children.push_back(cloneStep());
    return makeWhile(cloneOp(whileOp->children[0].get()), std::move(newBody));
  };

  auto while1 = buildWhile(0, bestK);
  auto while2 = buildWhile(bestK, numStmts);
  // Re-init the IV between the two loops so loop2 starts from the same
  // value as loop1 originally did.
  auto reInit = cloneOp(block->children[idx - 1].get());

  // Replace block->children[idx] (the original while) with: while1, reInit, while2.
  block->children[idx] = std::move(while1);
  block->children.insert(block->children.begin() + idx + 1, std::move(while2));
  block->children.insert(block->children.begin() + idx + 1, std::move(reInit));

  stats.loopDistributionApplied++;
  return true;
}

}  // namespace sys::hir
