#include "AffineNestAnalysis.h"

#include "HIRAffine.h"

#include <algorithm>
#include <unordered_set>

namespace sys::hir::affine {
namespace {

int accessRank(const MemoryAccess &access) {
  return (int) access.indices.size();
}

bool hasArrayReductionLikeFlow(const AffineNest &nest) {
  std::unordered_set<std::string> reads;
  for (const auto &access : nest.memory) {
    if (!access.base.empty() && !access.isWrite)
      reads.insert(access.base);
  }
  for (const auto &access : nest.memory) {
    if (!access.base.empty() && access.isWrite && reads.count(access.base))
      return true;
  }
  for (const auto &name : nest.effects.arrayWrites) {
    if (nest.effects.arrayReads.count(name))
      return true;
  }
  return false;
}

int estimatePressure(const AffineNest &nest) {
  int estimate = (int) nest.domains.size();
  estimate += (int) nest.effects.scalarReads.size();
  estimate += (int) nest.effects.scalarWrites.size();
  estimate += (int) nest.memory.size();
  if (hasArrayReductionLikeFlow(nest))
    estimate += 4;
  if (nest.effects.hasCall)
    estimate += 8;
  return estimate;
}

void mergeNest(AffineNestAnalysisSummary &summary, const AffineNest &nest) {
  summary.nests++;
  summary.maxDepth = std::max(summary.maxDepth, (int) nest.domains.size());
  if (!nest.imperfect && nest.domains.size() >= 2)
    summary.perfect2D++;
  if (!nest.imperfect && nest.domains.size() >= 3)
    summary.perfect3D++;
  if (nest.effects.hasCall)
    summary.nestsWithCalls++;
  if (!nest.effects.arrayWrites.empty())
    summary.nestsWithStores++;
  if (nest.hasSymbolicAccesses)
    summary.nestsWithSymbolicAccesses++;
  if (hasArrayReductionLikeFlow(nest))
    summary.nestsWithReductions++;

  for (const auto &access : nest.memory) {
    summary.maxAccessRank = std::max(summary.maxAccessRank, accessRank(access));
    if (access.isAffine)
      summary.affineAccesses++;
    else
      summary.nonAffineAccesses++;
    if (access.isContiguous)
      summary.contiguousAccesses++;
    if (access.isStrided)
      summary.stridedAccesses++;
  }

  summary.pressureEstimateMax =
      std::max(summary.pressureEstimateMax, estimatePressure(nest));
}

void walk(const Op *op, AffineNestAnalysisSummary &summary) {
  if (!op)
    return;
  summary.opsVisited++;

  AffineNest nest;
  if (collectAffineNest(op, nest, /*maxDepth=*/ 6, /*allowGuards=*/ true))
    mergeNest(summary, nest);

  for (const auto &child : op->children)
    walk(child.get(), summary);
}

}  // namespace

AffineNestAnalysisSummary analyzeAffineNests(const Module &module) {
  AffineNestAnalysisSummary summary;
  walk(module.root.get(), summary);
  return summary;
}

}  // namespace sys::hir::affine
