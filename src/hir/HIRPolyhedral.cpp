#include "HIRPolyhedral.h"

#include "HIRAffine.h"
#include "../backend/riscv/RiscvParams.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
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
constexpr const char *kRowStencilSplitLoopMarker = "__hir_row_stencil_split_loop";

int countOpsForPressure(const Op *op) {
  if (!op)
    return 0;
  int count = 1;
  for (const auto &child : op->children)
    count += countOpsForPressure(child.get());
  return count;
}

bool findDirectCanonicalChildLoop(Op *body, CanonicalLoop &loop, Op *&whileOp) {
  if (!body || body->kind != OpKind::Block)
    return false;
  for (auto &child : body->children) {
    if (!child || child->kind != OpKind::While)
      continue;
    CanonicalLoop candidate;
    if (matchCanonicalWhile(child.get(), candidate)) {
      loop = candidate;
      whileOp = child.get();
      return true;
    }
  }
  return false;
}

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

bool isArrayLoad2(const Op *op, const std::string &array,
                  const std::string &row, const std::string &col) {
  return op && op->kind == OpKind::Load && op->symbol == array &&
         op->children.size() == 2 &&
         isScalarLoad(op->children[0].get(), row) &&
         isScalarLoad(op->children[1].get(), col);
}

bool isArrayStore2(const Op *op, const std::string &array,
                   const std::string &row, const std::string &col) {
  op = unwrapSingleDecl(op);
  return op && op->kind == OpKind::Store && op->symbol == array &&
         op->children.size() == 3 &&
         isScalarLoad(op->children[0].get(), row) &&
         isScalarLoad(op->children[1].get(), col);
}

bool isAddAssignArrayLoad(const Op *op, const std::string &acc,
                          const std::string &array,
                          const std::string &row,
                          const std::string &col) {
  op = unwrapSingleDecl(op);
  if (!op || op->kind != OpKind::Store || op->symbol != acc ||
      op->children.size() != 1)
    return false;
  const Op *rhs = op->children[0].get();
  if (!rhs || rhs->kind != OpKind::Arith || rhs->symbol != "+" ||
      rhs->children.size() != 2)
    return false;
  return (isScalarLoad(rhs->children[0].get(), acc) &&
          isArrayLoad2(rhs->children[1].get(), array, row, col)) ||
         (isScalarLoad(rhs->children[1].get(), acc) &&
          isArrayLoad2(rhs->children[0].get(), array, row, col));
}

bool matchAnyScalarInit(const Op *op, std::string &symbol) {
  op = unwrapSingleDecl(op);
  if (!op || op->symbol.empty())
    return false;
  if (op->kind == OpKind::Store && op->children.size() == 1) {
    symbol = op->symbol;
    return true;
  }
  if (op->kind == OpKind::VarDecl && op->children.size() <= 1) {
    symbol = op->symbol;
    return true;
  }
  return false;
}

bool blockHasArrayAccessTo(const Op *op, const std::string &array) {
  if (!op)
    return false;
  const bool isArrayLoad = op->kind == OpKind::Load && !op->children.empty();
  const bool isArrayStore = op->kind == OpKind::Store && op->children.size() > 1;
  if ((isArrayLoad || isArrayStore) && op->symbol == array)
    return true;
  for (const auto &child : op->children)
    if (blockHasArrayAccessTo(child.get(), array))
      return true;
  return false;
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

std::unique_ptr<Op> cloneRowBodyWithStencilColumnSplit(
    const Op *rowBody, const Op *colWhileOp, const CanonicalLoop &colLoop,
    const StencilBounds &bounds) {
  if (!rowBody || rowBody->kind != OpKind::Block)
    return nullptr;

  auto out = makeBlock();
  for (const auto &child : rowBody->children) {
    if (child.get() != colWhileOp) {
      out->children.push_back(cloneOp(child.get()));
      continue;
    }

    auto leftBody = cloneOp(colLoop.body);
    auto centerBody = cloneDroppingStencilGuard(colLoop.body, bounds.guardedIf);
    auto rightBody = cloneOp(colLoop.body);
    if (!leftBody || !centerBody || !rightBody)
      return nullptr;

    auto leftCond = makeLogicalAnd(
        cloneOp(colWhileOp->children[0].get()),
        makeCmp("<", makeLoad(colLoop.iv), makeLoad(bounds.pad)));
    auto centerStop =
        makeArith("-", cloneOp(bounds.bound), makeLoad(bounds.pad));
    auto centerCond =
        makeCmp("<", makeLoad(colLoop.iv), std::move(centerStop));
    auto rightCond = cloneOp(colWhileOp->children[0].get());
    if (!leftCond || !centerCond || !rightCond)
      return nullptr;

    auto leftLoop = makeWhile(std::move(leftCond), std::move(leftBody));
    auto centerLoop = makeWhile(std::move(centerCond), std::move(centerBody));
    auto rightLoop = makeWhile(std::move(rightCond), std::move(rightBody));
    leftLoop->symbol = kRowStencilSplitLoopMarker;
    centerLoop->symbol = kRowStencilSplitLoopMarker;
    rightLoop->symbol = kRowStencilSplitLoopMarker;

    out->children.push_back(std::move(leftLoop));
    out->children.push_back(makeStore(colLoop.iv, makeLoad(bounds.pad)));
    out->children.push_back(std::move(centerLoop));
    out->children.push_back(std::move(rightLoop));
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

    if (hirEnvEnabled("SISY_HIR_ENABLE_ROW_STENCIL_INTERIOR", false)) {
      for (size_t i = 0; i < block->children.size(); i++) {
        if (block->children[i] && block->children[i]->kind == OpKind::While &&
            tryRowStencilInteriorDispatch(block, i, stats)) {
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

    if (hirEnvEnabled("SISY_HIR_ENABLE_TRANSPOSE_FORWARDING", false))
      changed = forwardTransposeLoads(block, stats) || changed;
  }

  for (auto &child : block->children)
    changed = optimizeBlock(child.get(), stats) || changed;

  if (block->kind != OpKind::Block && block->kind != OpKind::Module)
    return changed;

  for (size_t i = 0; i < block->children.size(); i++) {
    if (tryReductionMicroTile(block, i, stats)) {
      changed = true;
      i = 0;
      continue;
    }
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
    if (block->children[i] && block->children[i]->kind == OpKind::While &&
        tryMatmulTailCollapse(block, i, stats)) {
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
    // Now enabled by default for affine nests; pass `SISY_HIR_ENABLE_TILING=0`
    // to bisect. The transform preserves the original loop-exit IV value
    // (see tryLoopTiling).
    if (hirEnvEnabled("SISY_HIR_ENABLE_TILING", true)) {
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

bool PolyhedralOptimizer::tryRowStencilInteriorDispatch(Op *block, size_t idx,
                                                        PolyhedralStats &stats) {
  if (!block || idx >= block->children.size()) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectShape++;
    return false;
  }

  Op *rowWhileOp = block->children[idx].get();
  if (!rowWhileOp || rowWhileOp->kind != OpKind::While ||
      rowWhileOp->symbol == kRowStencilSplitLoopMarker ||
      rowWhileOp->symbol == kStencilSplitLoopMarker) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectShape++;
    return false;
  }

  CanonicalLoop rowLoop;
  if (!matchCanonicalWhile(rowWhileOp, rowLoop) || !rowLoop.body ||
      rowLoop.body->kind != OpKind::Block || rowLoop.body->children.size() < 2) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectShape++;
    return false;
  }

  Op *colWhileOp = nullptr;
  CanonicalLoop colLoop;
  if (!findDirectCanonicalChildLoop(rowLoop.body, colLoop, colWhileOp) ||
      !colLoop.body || colLoop.body->kind != OpKind::Block ||
      colWhileOp->symbol == kRowStencilSplitLoopMarker ||
      colWhileOp->symbol == kStencilSplitLoopMarker) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectShape++;
    return false;
  }

  std::unordered_map<std::string, const Op*> scalarInits;
  collectScalarInitializers(rowLoop.body, scalarInits);

  StencilBounds bounds;
  if (!findStencilBoundsIf(colLoop.body, scalarInits, colLoop.iv, bounds)) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectBounds++;
    return false;
  }

  if (bounds.rowSpatial != rowLoop.iv || bounds.colSpatial != colLoop.iv ||
      bounds.pad.empty() || !bounds.bound || !bounds.guardedIf) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectBounds++;
    return false;
  }

  if (bodyWritesScalar(rowLoop.body, bounds.pad) ||
      scalarUsedBeforeRedef(block, idx + 1, rowLoop.iv)) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectBounds++;
    return false;
  }

  // Avoid the old column-level dispatcher failure mode: if cloning the slow
  // and fast bodies would create a large live range fan-out, leave the nest in
  // its original guarded form.  The limit is intentionally conservative and
  // tunable for A/B testing.
  const int bodyOps = countOpsForPressure(rowLoop.body);
  const int colOps = countOpsForPressure(colLoop.body);
  const int pressureBudget =
      hirEnvInt("SISY_HIR_ROW_STENCIL_PRESSURE_BUDGET", 200);
  if (bodyOps + 2 * colOps > pressureBudget) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectPressure++;
    stats.guardedScopRejectPressure++;
    return false;
  }

  auto topBody = cloneOp(rowLoop.body);
  auto middleBody =
      cloneRowBodyWithStencilColumnSplit(rowLoop.body, colWhileOp, colLoop, bounds);
  auto bottomBody = cloneOp(rowLoop.body);
  if (!topBody || !middleBody || !bottomBody) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectShape++;
    return false;
  }

  if (!hirEnvEnabled("SISY_HIR_ROW_STENCIL_THREE_LOOPS", false)) {
    auto rowLower =
        makeCmp("<=", makeLoad(bounds.pad), makeLoad(rowLoop.iv));
    auto rowUpper = makeCmp("<", makeLoad(rowLoop.iv),
                            makeArith("-", cloneOp(bounds.bound),
                                      makeLoad(bounds.pad)));
    auto rowInterior = makeLogicalAnd(std::move(rowLower), std::move(rowUpper));
    if (!rowInterior) {
      stats.rowStencilInteriorRejected++;
      stats.rowStencilRejectShape++;
      return false;
    }

    auto guardedBody = makeBlock();
    guardedBody->children.push_back(
        makeIf(std::move(rowInterior), std::move(middleBody), std::move(topBody)));
    auto guardedLoop =
        makeWhile(cloneOp(rowWhileOp->children[0].get()), std::move(guardedBody));
    guardedLoop->symbol = kRowStencilSplitLoopMarker;
    block->children[idx] = std::move(guardedLoop);
    stats.rowStencilInteriorDispatched++;
    stats.guardedScopApplied++;
    return true;
  }

  auto topCond = makeLogicalAnd(
      cloneOp(rowWhileOp->children[0].get()),
      makeCmp("<", makeLoad(rowLoop.iv), makeLoad(bounds.pad)));
  auto middleStop =
      makeArith("-", cloneOp(bounds.bound), makeLoad(bounds.pad));
  auto middleCond =
      makeCmp("<", makeLoad(rowLoop.iv), std::move(middleStop));
  auto bottomCond = cloneOp(rowWhileOp->children[0].get());
  if (!topCond || !middleCond || !bottomCond) {
    stats.rowStencilInteriorRejected++;
    stats.rowStencilRejectShape++;
    return false;
  }

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(4);
  auto topLoop = makeWhile(std::move(topCond), std::move(topBody));
  auto middleLoop = makeWhile(std::move(middleCond), std::move(middleBody));
  auto bottomLoop = makeWhile(std::move(bottomCond), std::move(bottomBody));
  topLoop->symbol = kRowStencilSplitLoopMarker;
  middleLoop->symbol = kRowStencilSplitLoopMarker;
  bottomLoop->symbol = kRowStencilSplitLoopMarker;
  replacement.push_back(std::move(topLoop));
  replacement.push_back(makeStore(rowLoop.iv, makeLoad(bounds.pad)));
  replacement.push_back(std::move(middleLoop));
  replacement.push_back(std::move(bottomLoop));

  auto first = block->children.begin() + static_cast<std::ptrdiff_t>(idx);
  first = block->children.erase(first);
  block->children.insert(first,
                         std::make_move_iterator(replacement.begin()),
                         std::make_move_iterator(replacement.end()));
  stats.rowStencilInteriorDispatched++;
  stats.guardedScopApplied++;
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
  affine::AffineNest nest;
  bool haveNest = affine::collectAffineNest(op, nest, hirEnvInt("SISY_HIR_AFFINE_SCAN_DEPTH", 5),
                                            /*allowGuards=*/ true);
  if (haveNest && std::getenv("SISY_DUMP_AFFINE_NEST")) {
    std::cerr << "[affine-nest] loops=" << nest.loops.size()
              << " accesses=" << nest.accesses.size()
              << " memory=" << nest.memory.size()
              << " guards=" << nest.guards.size()
              << " imperfect=" << (nest.imperfect ? 1 : 0)
              << " symbolic=" << (nest.hasSymbolicAccesses ? 1 : 0)
              << "\n";
  }

  std::vector<CanonicalLoop> loops;
  Op *current = op;
  const int maxScanDepth = std::max(3, hirEnvInt("SISY_HIR_AFFINE_SCAN_DEPTH", 5));
  for (int depth = 0; depth < maxScanDepth; depth++) {
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
  std::unordered_set<std::string> loopIVs;
  for (const auto &loop : loops)
    loopIVs.insert(loop.iv);
  if (hasUnsafeAffineScanControl(op)) {
    std::vector<affine::Access> guardedAccesses =
        affine::collectArrayAccesses(op, loopIVs);
    int rawAccesses = countArrayAccessOps(op);
    if (rawAccesses > 0 && rawAccesses == (int)guardedAccesses.size()) {
      stats.guardedScopCandidates++;
      for (const auto &access : guardedAccesses) {
        for (const auto &idx : access.indices) {
          if (affine::hasSymbolicCoefficients(idx)) {
            stats.guardedScopSymbolic++;
            stats.symbolicAffineAccesses++;
            break;
          }
        }
      }
    }
    stats.affineNestRejectedControl++;
    return;
  }

  int rawAccesses = countArrayAccessOps(op);
  std::vector<affine::Access> accesses = affine::collectArrayAccesses(op, loopIVs);
  if (rawAccesses != (int) accesses.size()) {
    stats.affineNestRejectedAccess++;
    return;
  }
  for (const auto &access : accesses) {
    for (const auto &idx : access.indices) {
      if (affine::hasSymbolicCoefficients(idx)) {
        stats.symbolicAffineAccesses++;
        break;
      }
    }
  }

  if (loops.size() >= 2)
    stats.affineNestPerfect2D++;
  if (loops.size() >= 3)
    stats.affineNestPerfect3D++;
  if (isMatmulLikeNest(loops, accesses))
    stats.matmulLikeCandidates++;
}

bool PolyhedralOptimizer::tryMatmulTailCollapse(Op *block, size_t idx,
                                                PolyhedralStats &stats) {
  if (!hirEnvEnabled("SISY_HIR_ENABLE_MATMUL_TAIL_COLLAPSE", true))
    return false;
  if (!block || block->kind != OpKind::Block || idx == 0 ||
      idx + 4 >= block->children.size()) {
    return false;
  }

  auto reject = [&](const char *why, int *counter) {
    if (std::getenv("SISY_HIR_TAIL_DEBUG"))
      std::cerr << "[matmul-tail] reject " << why << " at idx " << idx << "\n";
    stats.matmulTailRejected++;
    if (counter)
      (*counter)++;
    return false;
  };

  Op *minLoopOp = block->children[idx].get();
  Op *transposeInit = block->children[idx + 1].get();
  Op *transposeLoopOp = block->children[idx + 2].get();
  Op *sumInit = block->children[idx + 3].get();
  Op *sumLoopOp = block->children[idx + 4].get();

  CanonicalLoop minOuter, transposeOuter, sumOuter;
  if (!matchCanonicalWhile(minLoopOp, minOuter) ||
      !matchCanonicalWhile(transposeLoopOp, transposeOuter) ||
      !matchCanonicalWhile(sumLoopOp, sumOuter))
    return reject("outer-shape", &stats.matmulTailRejectShape);
  if (!matchLoopInit(block->children[idx - 1].get(), minOuter.iv) ||
      !matchLoopInit(transposeInit, transposeOuter.iv) ||
      !matchLoopInit(sumInit, sumOuter.iv))
    return reject("outer-init", &stats.matmulTailRejectShape);
  if (minOuter.iv != transposeOuter.iv || minOuter.iv != sumOuter.iv ||
      !boundsEqual(minOuter.bound, transposeOuter.bound) ||
      !boundsEqual(minOuter.bound, sumOuter.bound))
    return reject("outer-compatible", &stats.matmulTailRejectShape);

  Op *minBody = minOuter.body;
  if (!minBody || minBody->kind != OpKind::Block ||
      minBody->children.size() != 6)
    return reject("min-body-size", &stats.matmulTailRejectShape);

  std::string jSym;
  if (!matchAnyScalarInit(minBody->children[0].get(), jSym))
    return reject("j-init", &stats.matmulTailRejectShape);
  Op *tempDecl = unwrapSingleDecl(minBody->children[1].get());
  if (!tempDecl || tempDecl->symbol.empty() ||
      (tempDecl->kind != OpKind::VarDecl && tempDecl->kind != OpKind::Store) ||
      tempDecl->children.size() != 1)
    return reject("temp-decl", &stats.matmulTailRejectShape);
  std::string tempSym = tempDecl->symbol;

  CanonicalLoop minInner;
  if (!matchCanonicalWhile(minBody->children[2].get(), minInner) ||
      minInner.iv != jSym || !boundsEqual(minInner.bound, minOuter.bound))
    return reject("min-inner", &stats.matmulTailRejectShape);
  Op *minInnerBody = minInner.body;
  if (!minInnerBody || minInnerBody->kind != OpKind::Block ||
      minInnerBody->children.size() != 2 ||
      !matchStepStore(minInnerBody->children[1].get(), jSym, 1))
    return reject("min-inner-body", &stats.matmulTailRejectShape);
  Op *minIf = unwrapSingleDecl(minInnerBody->children[0].get());
  if (!minIf || minIf->kind != OpKind::If || minIf->children.size() != 2)
    return reject("min-if", &stats.matmulTailRejectShape);
  Op *minCond = minIf->children[0].get();
  if (!minCond || minCond->kind != OpKind::Cmp || minCond->symbol != "<" ||
      minCond->children.size() != 2 ||
      !isScalarLoad(minCond->children[1].get(), tempSym))
    return reject("min-cond", &stats.matmulTailRejectShape);
  const Op *rowLoad = minCond->children[0].get();
  if (!rowLoad || rowLoad->kind != OpKind::Load || rowLoad->children.size() != 2 ||
      !isScalarLoad(rowLoad->children[0].get(), minOuter.iv) ||
      !isScalarLoad(rowLoad->children[1].get(), jSym))
    return reject("row-load", &stats.matmulTailRejectShape);
  std::string array = rowLoad->symbol;
  Op *minThen = minIf->children[1].get();
  if (!minThen || minThen->kind != OpKind::Block || minThen->children.size() != 1)
    return reject("min-then", &stats.matmulTailRejectShape);
  Op *tempStore = unwrapSingleDecl(minThen->children[0].get());
  if (!tempStore || tempStore->kind != OpKind::Store || tempStore->symbol != tempSym ||
      tempStore->children.size() != 1 ||
      !isArrayLoad2(tempStore->children[0].get(), array, minOuter.iv, jSym))
    return reject("temp-store", &stats.matmulTailRejectShape);

  std::string fillJ;
  if (!matchAnyScalarInit(minBody->children[3].get(), fillJ) || fillJ != jSym)
    return reject("fill-init", &stats.matmulTailRejectShape);
  CanonicalLoop fillInner;
  if (!matchCanonicalWhile(minBody->children[4].get(), fillInner) ||
      fillInner.iv != jSym || !boundsEqual(fillInner.bound, minOuter.bound))
    return reject("fill-inner", &stats.matmulTailRejectShape);
  Op *fillBody = fillInner.body;
  if (!fillBody || fillBody->kind != OpKind::Block ||
      fillBody->children.size() != 2 ||
      !matchStepStore(fillBody->children[1].get(), jSym, 1))
    return reject("fill-body", &stats.matmulTailRejectShape);
  Op *fillStore = unwrapSingleDecl(fillBody->children[0].get());
  if (!isArrayStore2(fillStore, array, minOuter.iv, jSym) ||
      !isScalarLoad(fillStore->children[2].get(), tempSym))
    return reject("fill-store", &stats.matmulTailRejectShape);
  if (!matchStepStore(minBody->children[5].get(), minOuter.iv, 1))
    return reject("min-outer-step", &stats.matmulTailRejectShape);

  auto matchTransposeLoop = [&]() -> bool {
    Op *body = transposeOuter.body;
    if (!body || body->kind != OpKind::Block || body->children.size() < 3)
      return false;
    std::string tj;
    if (!matchAnyScalarInit(body->children[0].get(), tj) || tj != jSym)
      return false;
    size_t whileIdx = body->children.size() == 4 ? 2 : 1;
    CanonicalLoop inner;
    if (!matchCanonicalWhile(body->children[whileIdx].get(), inner) ||
        inner.iv != jSym || !boundsEqual(inner.bound, minOuter.bound))
      return false;
    if (!matchStepStore(body->children.back().get(), transposeOuter.iv, 1))
      return false;
    Op *innerBody = inner.body;
    if (!innerBody || innerBody->kind != OpKind::Block ||
        innerBody->children.size() != 2 ||
        !matchStepStore(innerBody->children[1].get(), jSym, 1))
      return false;
    Op *store = unwrapSingleDecl(innerBody->children[0].get());
    if (!isArrayStore2(store, array, transposeOuter.iv, jSym))
      return false;
    const Op *rhs = store->children[2].get();
    const Op *src = nullptr;
    if (rhs && rhs->kind == OpKind::Arith && rhs->symbol == "-" &&
        rhs->children.size() == 1)
      src = rhs->children[0].get();
    return isArrayLoad2(src, array, jSym, transposeOuter.iv);
  };
  if (!matchTransposeLoop())
    return reject("transpose-loop", &stats.matmulTailRejectShape);

  std::string sumSym;
  auto matchSumLoop = [&]() -> bool {
    Op *body = sumOuter.body;
    if (!body || body->kind != OpKind::Block || body->children.size() < 3)
      return false;
    std::string sj;
    if (!matchAnyScalarInit(body->children[0].get(), sj) || sj != jSym)
      return false;
    size_t whileIdx = body->children.size() == 4 ? 2 : 1;
    CanonicalLoop inner;
    if (!matchCanonicalWhile(body->children[whileIdx].get(), inner) ||
        inner.iv != jSym || !boundsEqual(inner.bound, minOuter.bound))
      return false;
    if (!matchStepStore(body->children.back().get(), sumOuter.iv, 1))
      return false;
    Op *innerBody = inner.body;
    if (!innerBody || innerBody->kind != OpKind::Block ||
        innerBody->children.size() != 2 ||
        !matchStepStore(innerBody->children[1].get(), jSym, 1))
      return false;
    Op *sumStore = unwrapSingleDecl(innerBody->children[0].get());
    if (!sumStore || sumStore->kind != OpKind::Store ||
        sumStore->children.size() != 1 || sumStore->symbol.empty())
      return false;
    if (!isAddAssignArrayLoad(sumStore, sumStore->symbol, array, sumOuter.iv, jSym))
      return false;
    sumSym = sumStore->symbol;
    return true;
  };
  if (!matchSumLoop())
    return reject("sum-loop", &stats.matmulTailRejectShape);

  for (size_t after = idx + 5; after < block->children.size(); after++) {
    if (blockHasArrayAccessTo(block->children[after].get(), array))
      return reject("later-use", &stats.matmulTailRejectUse);
  }

  auto newBody = makeBlock();
  newBody->children.push_back(cloneOp(minBody->children[0].get()));
  newBody->children.push_back(cloneOp(minBody->children[1].get()));
  newBody->children.push_back(cloneOp(minBody->children[2].get()));
  newBody->children.push_back(makeStore(
      sumSym, makeArith("-", makeLoad(sumSym), makeLoad(tempSym))));
  newBody->children.push_back(cloneOp(minBody->children[5].get()));
  auto collapsed = makeWhile(cloneOp(minLoopOp->children[0].get()),
                             std::move(newBody));

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() - 4);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == idx) {
      replacement.push_back(std::move(collapsed));
      i += 4;
      continue;
    }
    replacement.push_back(std::move(block->children[i]));
  }
  block->children = std::move(replacement);
  stats.matmulTailCollapsed++;
  return true;
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
