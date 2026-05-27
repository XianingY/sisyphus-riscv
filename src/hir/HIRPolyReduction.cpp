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

namespace {

std::unique_ptr<Op> makeArrayLoadAt(const std::string &symbol,
                                    std::unique_ptr<Op> index,
                                    TypeKind type = TypeKind::Int) {
  auto op = std::make_unique<Op>(OpKind::Load);
  op->type = type;
  op->symbol = symbol;
  op->children.push_back(std::move(index));
  return op;
}

std::unique_ptr<Op> makeArrayStoreAt(const std::string &symbol,
                                     std::unique_ptr<Op> index,
                                     std::unique_ptr<Op> value,
                                     TypeKind type = TypeKind::Int) {
  auto op = std::make_unique<Op>(OpKind::Store);
  op->type = type;
  op->symbol = symbol;
  op->children.push_back(std::move(index));
  op->children.push_back(std::move(value));
  return op;
}

bool isDirectLoadOf(const Op *op, const std::string &symbol) {
  return op && op->kind == OpKind::Load && op->children.empty() &&
         op->symbol == symbol;
}

std::unique_ptr<Op> makeSyntheticArrayDecl(const std::string &symbol,
                                           int elements, TypeKind elementType) {
  auto op = std::make_unique<Op>(OpKind::VarDecl);
  op->type = elementType == TypeKind::Float ? TypeKind::Float : TypeKind::Int;
  op->symbol = symbol;
  op->arrayDims.push_back(elements);
  return op;
}

std::unique_ptr<Op> cloneReductionStmtToScratch(const Op *op,
                                                const ReductionPattern &pat,
                                                const std::string &scratch) {
  if (!op)
    return nullptr;
  if (isScalarLoad(op, pat.acc))
    return makeArrayLoadAt(scratch, makeLoad(pat.j), op->type);
  if (op->kind == OpKind::Store && op->symbol == pat.acc && op->children.size() == 1) {
    auto oldValue = makeArrayLoadAt(scratch, makeLoad(pat.j), op->type);
    auto nextValue = cloneReplacingScalarLoad(op->children[0].get(), pat.acc,
                                              oldValue.get());
    if (!nextValue)
      return nullptr;
    return makeArrayStoreAt(scratch, makeLoad(pat.j), std::move(nextValue),
                            op->type);
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
    out->children.push_back(cloneReductionStmtToScratch(child.get(), pat, scratch));
  return out;
}

}  // namespace

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

bool PolyhedralOptimizer::tryReductionRowPrivatize(Op *block, size_t initIndex,
                                                   PolyhedralStats &stats) {
  if (!block || initIndex + 1 >= block->children.size())
    return false;

  if (!hirEnvEnabled("SISY_HIR_ENABLE_REDUCTION_PRIVATIZE", true))
    return false;

  Op *whileOp = block->children[initIndex + 1].get();
  ReductionPattern pat;
  if (!matchReductionPattern(whileOp, pat))
    return false;
  if (!matchLoopInit(block->children[initIndex].get(), pat.j))
    return false;
  if (!pat.destStore || pat.destStore->children.size() != 3 ||
      !pat.accUpdate || !pat.kReductionStmt)
    return false;

  auto dimsIt = globalArrayDims.find(pat.destStore->symbol);
  if (dimsIt == globalArrayDims.end() || dimsIt->second.size() != 2)
    return false;

  int jDim = -1;
  for (int dim = 0; dim < 2; dim++) {
    if (isDirectLoadOf(pat.destStore->children[dim].get(), pat.j))
      jDim = dim;
  }
  if (jDim < 0)
    return false;

  const int scratchElems = dimsIt->second[jDim];
  constexpr int kMaxScratchElems = 1 << 16;
  if (scratchElems <= 0 || scratchElems > kMaxScratchElems)
    return false;

  // This is scalar expansion for reductions that are already legal to
  // interchange.  In-place recurrences keep the existing order-preserving jam
  // path; row buffering them is legal only for narrower shapes and was slower
  // on cache-sized integer kernels.
  if (!strictReductionInterchangeLegal(pat, globalArrays)) {
    bool fail = true;
    if (exprUsesScalar(pat.destStore, pat.k)) {
      fail = true;
    } else {
      fail = false;
      std::vector<affine::Access> accesses = affine::collectArrayAccesses(pat.kReductionStmt);
      for (const auto &acc : accesses) {
        if (!globalArrays.count(acc.base)) { fail = true; break; }
        if (acc.base == pat.destStore->symbol) {
          if (acc.indices.size() != pat.destStore->children.size() - 1) { fail = true; break; }
          if (acc.indices.size() > (size_t)jDim) {
            const affine::Expr &jExpr = acc.indices[jDim];
            if (jExpr.coeffs.size() != 1 || jExpr.coeffs.count(pat.j) == 0 || jExpr.coeffs.at(pat.j) != 1 || jExpr.constant != 0) { fail = true; break; }
          } else { fail = true; break; }
        }
      }
    }
    if (fail)
      return false;
  }

  const std::string scratch =
      "__poly_rowbuf_" + pat.destStore->symbol + "_" + std::to_string(uniqueId++);

  auto scratchDecl = makeSyntheticArrayDecl(
      scratch, scratchElems,
      pat.destStore->type == TypeKind::Float ? TypeKind::Float : TypeKind::Int);

  auto initBody = makeBlock();
  auto initValueClone = cloneOp(initValue(pat.accInit));
  if (!initValueClone)
    initValueClone = makeConstInt(0);
  initBody->children.push_back(
      makeArrayStoreAt(scratch, makeLoad(pat.j), std::move(initValueClone),
                       pat.destStore->type));
  initBody->children.push_back(cloneOp(pat.jStep));
  auto initLoop = makeWhile(makeCmp("<", makeLoad(pat.j), cloneOp(pat.jBound)),
                            std::move(initBody));

  CanonicalLoop kLoop;
  if (!matchCanonicalWhile(pat.kWhile, kLoop))
    return false;

  auto innerJBody = makeBlock();
  auto scratchUpdate = cloneReductionStmtToScratch(pat.kReductionStmt, pat, scratch);
  if (!scratchUpdate)
    return false;
  innerJBody->children.push_back(std::move(scratchUpdate));
  innerJBody->children.push_back(cloneOp(pat.jStep));
  auto innerJLoop = makeWhile(makeCmp("<", makeLoad(pat.j), cloneOp(pat.jBound)),
                              std::move(innerJBody));

  auto kBody = makeBlock();
  kBody->children.push_back(cloneOp(block->children[initIndex].get()));
  kBody->children.push_back(std::move(innerJLoop));
  kBody->children.push_back(cloneOp(pat.kStep));
  auto kWhile = makeWhile(cloneOp(pat.kWhile->children[0].get()), std::move(kBody));

  auto commitBody = makeBlock();
  auto scratchLoad = makeArrayLoadAt(scratch, makeLoad(pat.j), pat.destStore->type);
  auto commitStore = makeArrayStoreLike(pat.destStore, std::move(scratchLoad));
  if (!commitStore)
    return false;
  commitBody->children.push_back(std::move(commitStore));
  commitBody->children.push_back(cloneOp(pat.jStep));
  auto commitLoop = makeWhile(makeCmp("<", makeLoad(pat.j), cloneOp(pat.jBound)),
                              std::move(commitBody));
  auto commitJInit = cloneOp(block->children[initIndex].get());
  if (!commitJInit)
    return false;

  std::vector<std::unique_ptr<Op>> replacement;
  replacement.reserve(block->children.size() + 5);
  for (size_t i = 0; i < block->children.size(); i++) {
    if (i == initIndex) {
      replacement.push_back(std::move(scratchDecl));
      replacement.push_back(std::move(block->children[i]));
      replacement.push_back(std::move(initLoop));
      replacement.push_back(cloneOp(pat.kInit));
      replacement.push_back(std::move(kWhile));
      replacement.push_back(std::move(commitJInit));
      replacement.push_back(std::move(commitLoop));
      i++; // skip original j loop
    } else {
      replacement.push_back(std::move(block->children[i]));
    }
  }

  block->children = std::move(replacement);
  stats.reductionPrivatized++;
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
