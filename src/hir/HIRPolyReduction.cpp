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
