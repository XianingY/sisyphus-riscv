#ifndef PIPELINE_PROFILES_H
#define PIPELINE_PROFILES_H

#include <cstddef>
#include <string>

#include "Options.h"
#include "../opt/PassManager.h"

namespace sys::pipeline {

enum class CoreProfile {
  O0,
  O1,
  O2,
};

enum class FrontendProfile {
  Legacy,
  Dialect,
};

struct PipelineMetrics {
  size_t moduleOpCount = 0;
  size_t blockCount = 0;
  size_t cfgEdgeCount = 0;
  size_t phiCount = 0;
  size_t callLikeCount = 0;
  size_t getArgCount = 0;
  int maxGetArgArity = 0;
  int maxLoopDepth = 0;

  // Forwarded from the HIR polyhedral stage (see PolyhedralStats). Zero
  // when the frontend ran without HIR polyhedral or no transformation
  // fired. Used by selectPlan / appendCoreO1 to condition downstream
  // cleanup intensity.
  int hirPolyApplied = 0;          // sum of all transforms that fired
  int hirPolyTilingApplied = 0;
  int hirPolyAffineRejected = 0;   // shape+control+access rejection sum
};

struct PipelinePlan {
  FrontendProfile frontendProfile;
  CoreProfile coreProfile;
  bool aggressive;
  bool enableO2Experimental;
  bool enableO2Heavy;
  int o2LoopRounds;
  bool largeModuleMode;
  bool hugeModuleMode;
  bool backendFastMode;
  bool armTimeoutSafeMode;
  int armInstCombineRounds;
  int armPeepholeRounds;
  int armRegAllocCallPenalty;
  int armRegAllocLoopBoost;
  int armRegAllocPreferBudget;
  PipelineMetrics metrics;
  bool useArmBackend;
  bool useRvBackend;
};

PipelinePlan selectPlan(const Options &opts, PipelineMetrics metrics = {});
PipelinePlan configurePipeline(PassManager &pm, const Options &opts, PipelineMetrics metrics = {});
std::string formatPlan(const PipelinePlan &plan);

}

#endif
