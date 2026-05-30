#include "AnalysisManager.h"
#include "Analysis.h"

#include <cstdlib>
#include <iostream>

using namespace sys;

namespace {

DomTree buildDomTreeFor(Region *region) {
  region->updateDoms();
  DomTree tree;
  for (auto bb : region->getBlocks())
    if (auto idom = bb->getIdom())
      tree[idom].push_back(bb);
  return tree;
}

LegacyAffineNestSummary buildLegacyAffineNestSummary(ModuleOp *module) {
  LegacyAffineNestSummary summary;
  for (auto op : module->findAll<ForOp>())
    (void) op, summary.loops++;
  for (auto op : module->findAll<WhileOp>())
    (void) op, summary.loops++;
  for (auto op : module->findAll<LoadOp>())
    (void) op, summary.loads++;
  for (auto op : module->findAll<StoreOp>())
    (void) op, summary.stores++;
  for (auto op : module->findAll<BranchOp>())
    (void) op, summary.branches++;
  return summary;
}

} // namespace

AnalysisManager::AnalysisManager(ModuleOp *module, bool enabled, bool dump):
  module(module), enabled(enabled), dump(dump) {}

void AnalysisManager::noteInvalid(AnalysisCacheStat &stat,
                                  const std::string &reason) {
  stat.version++;
  stat.invalidateReason = reason;
}

DomTree &AnalysisManager::getDomTree(Region *region) {
  auto &entry = doms[region];
  if (enabled && entry.valid) {
    entry.stat.hitCount++;
    return entry.tree;
  }
  entry.tree = buildDomTreeFor(region);
  entry.valid = true;
  entry.stat.buildCount++;
  return entry.tree;
}

std::map<FuncOp*, LoopForest> &AnalysisManager::getLoopForests() {
  if (enabled && loopsValid && loopAnalysis) {
    loopStat.hitCount++;
    return loopAnalysis->getResultRef();
  }
  loopAnalysis = std::make_unique<LoopAnalysis>(module);
  loopAnalysis->run();
  loopsValid = true;
  loopStat.buildCount++;
  return loopAnalysis->getResultRef();
}

LoopForest &AnalysisManager::getLoopForest(FuncOp *func) {
  return getLoopForests()[func];
}

MemorySSA &AnalysisManager::getMemorySSA(Region *region) {
  auto &entry = memories[region];
  if (enabled && entry.valid && entry.mssa && !entry.mssa->isDirty()) {
    entry.stat.hitCount++;
    return *entry.mssa;
  }
  entry.mssa = std::make_unique<MemorySSA>(region);
  entry.mssa->build();
  entry.valid = true;
  entry.stat.buildCount++;
  return *entry.mssa;
}

void AnalysisManager::ensureAlias() {
  if (enabled && aliasValid) {
    aliasStat.hitCount++;
    return;
  }
  Alias(module).run();
  aliasValid = true;
  aliasStat.buildCount++;
}

void AnalysisManager::ensureBlockFrequency() {
  if (enabled && blockFrequencyValid) {
    blockFrequencyStat.hitCount++;
    return;
  }
  BlockFrequency(module).run();
  blockFrequencyValid = true;
  blockFrequencyStat.buildCount++;
}

DataLayout &AnalysisManager::getDataLayout() {
  if (enabled && dataLayoutValid && dataLayout) {
    dataLayoutStat.hitCount++;
    return *dataLayout;
  }
  dataLayout = std::make_unique<DataLayout>(8);
  dataLayoutValid = true;
  dataLayoutStat.buildCount++;
  return *dataLayout;
}

LegacyAffineNestSummary &AnalysisManager::getLegacyAffineNestSummary() {
  if (enabled && affineNestValid && affineNest) {
    affineNestStat.hitCount++;
    return *affineNest;
  }
  affineNest = std::make_unique<LegacyAffineNestSummary>(
      buildLegacyAffineNestSummary(module));
  affineNestValid = true;
  affineNestStat.buildCount++;
  return *affineNest;
}

MemRefAliasAnalysis &AnalysisManager::getMemRefAlias() {
  if (enabled && memRefAliasValid && memRefAlias) {
    memRefAliasStat.hitCount++;
    return *memRefAlias;
  }
  ensureAlias();
  memRefAlias = std::make_unique<MemRefAliasAnalysis>(module, getDataLayout());
  memRefAlias->build();
  memRefAliasValid = true;
  memRefAliasStat.buildCount++;
  return *memRefAlias;
}

void AnalysisManager::invalidate(const PreservedAnalyses &pa,
                                 const std::string &passName) {
  if (!enabled)
    return;
  std::string reason = passName;
  if (!pa.preserves(PreservedAnalyses::DomTreeAnalysis)) {
    for (auto &[_, cache] : doms) {
      if (cache.valid)
        noteInvalid(cache.stat, reason);
      cache.valid = false;
    }
  }
  if (!pa.preserves(PreservedAnalyses::LoopAnalysisResult)) {
    if (loopsValid)
      noteInvalid(loopStat, reason);
    loopsValid = false;
    loopAnalysis.reset();
  }
  if (!pa.preserves(PreservedAnalyses::MemorySSAAnalysis)) {
    for (auto &[_, cache] : memories) {
      if (cache.valid)
        noteInvalid(cache.stat, reason);
      cache.valid = false;
      if (cache.mssa)
        cache.mssa->markDirty(reason);
    }
  }
  if (!pa.preserves(PreservedAnalyses::AliasAnalysisResult)) {
    if (aliasValid)
      noteInvalid(aliasStat, reason);
    aliasValid = false;
  }
  if (!pa.preserves(PreservedAnalyses::BlockFrequencyAnalysis)) {
    if (blockFrequencyValid)
      noteInvalid(blockFrequencyStat, reason);
    blockFrequencyValid = false;
  }
  if (!pa.preserves(PreservedAnalyses::DataLayoutAnalysis)) {
    if (dataLayoutValid)
      noteInvalid(dataLayoutStat, reason);
    dataLayoutValid = false;
    dataLayout.reset();
  }
  if (!pa.preserves(PreservedAnalyses::AffineNestAnalysis)) {
    if (affineNestValid)
      noteInvalid(affineNestStat, reason);
    affineNestValid = false;
    affineNest.reset();
  }
  if (!pa.preserves(PreservedAnalyses::MemRefAliasAnalysis)) {
    if (memRefAliasValid)
      noteInvalid(memRefAliasStat, reason);
    memRefAliasValid = false;
    memRefAlias.reset();
  }
}

void AnalysisManager::invalidateMemory(Region *region,
                                       const std::string &reason) {
  auto it = memories.find(region);
  if (it == memories.end())
    return;
  if (it->second.valid)
    noteInvalid(it->second.stat, reason);
  it->second.valid = false;
  if (it->second.mssa)
    it->second.mssa->markDirty(reason);
}

void AnalysisManager::dumpStats(std::ostream &os) const {
  if (!dump)
    return;
  auto dumpOne = [&](const std::string &name, const AnalysisCacheStat &s) {
    os << "[analysis-cache] " << name
       << " builds=" << s.buildCount
       << " hits=" << s.hitCount
       << " version=" << s.version;
    if (!s.invalidateReason.empty())
      os << " last-invalidated-by=" << s.invalidateReason;
    os << "\n";
  };
  dumpOne("loop", loopStat);
  dumpOne("alias", aliasStat);
  dumpOne("block-frequency", blockFrequencyStat);
  dumpOne("data-layout", dataLayoutStat);
  dumpOne("affine-nest", affineNestStat);
  dumpOne("memref-alias", memRefAliasStat);
  int idx = 0;
  for (const auto &[_, cache] : doms)
    dumpOne("domtree." + std::to_string(idx++), cache.stat);
  idx = 0;
  for (const auto &[_, cache] : memories)
    dumpOne("memoryssa." + std::to_string(idx++), cache.stat);
}
