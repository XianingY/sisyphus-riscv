#ifndef SISY_ANALYSIS_MANAGER_H
#define SISY_ANALYSIS_MANAGER_H

#include "Pass.h"
#include "LoopPasses.h"
#include "MemorySSA.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace sys {

struct AnalysisCacheStat {
  int version = 0;
  int buildCount = 0;
  int hitCount = 0;
  std::string invalidateReason;
};

class AnalysisManager {
  ModuleOp *module;
  bool enabled;
  bool dump;

  struct DomCache {
    DomTree tree;
    bool valid = false;
    AnalysisCacheStat stat;
  };
  struct MemoryCache {
    std::unique_ptr<MemorySSA> mssa;
    bool valid = false;
    AnalysisCacheStat stat;
  };

  std::unordered_map<Region*, DomCache> doms;
  std::unordered_map<Region*, MemoryCache> memories;
  std::unique_ptr<LoopAnalysis> loopAnalysis;
  bool loopsValid = false;
  AnalysisCacheStat loopStat;
  bool aliasValid = false;
  AnalysisCacheStat aliasStat;
  bool blockFrequencyValid = false;
  AnalysisCacheStat blockFrequencyStat;

  void noteInvalid(AnalysisCacheStat &stat, const std::string &reason);

public:
  AnalysisManager(ModuleOp *module, bool enabled, bool dump);

  bool isEnabled() const { return enabled; }
  DomTree &getDomTree(Region *region);
  LoopForest &getLoopForest(FuncOp *func);
  std::map<FuncOp*, LoopForest> &getLoopForests();
  MemorySSA &getMemorySSA(Region *region);
  void ensureAlias();
  void ensureBlockFrequency();

  void invalidate(const PreservedAnalyses &pa, const std::string &passName);
  void invalidateMemory(Region *region, const std::string &reason);
  void dumpStats(std::ostream &os) const;
};

class PassContext {
  AnalysisManager *manager;

public:
  explicit PassContext(AnalysisManager *manager): manager(manager) {}

  AnalysisManager &analysis() { return *manager; }
  bool enabled() const { return manager && manager->isEnabled(); }
};

} // namespace sys

#endif
