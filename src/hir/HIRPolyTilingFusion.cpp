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

  // Don't tile if this is already the inner loop of a tiled loop.
  if (outer.bound && outer.bound->kind == OpKind::Load && outer.bound->symbol.find("__hir_tile_stop_") == 0) {
    stats.tilingRejected++;
    stats.tilingRejectIdempotent++;
    return false;
  }

  int dynamicTileSize = computeOptimalTileSize(detectMainType(outer.body), outer.body);

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

}  // namespace sys::hir
