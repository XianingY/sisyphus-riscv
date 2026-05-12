#include "PipelineProfiles.h"

#include <cstdlib>
#include <cstring>
#include <sstream>

#include "../opt/Passes.h"
#include "../opt/LoopPasses.h"
#include "../opt/CleanupPasses.h"
#include "../opt/LowerPasses.h"
#include "../opt/SMTPasses.h"
#include "../opt/Analysis.h"
#include "../pre-opt/PrePasses.h"
#include "../pre-opt/PreLoopPasses.h"
#include "../pre-opt/PreAnalysis.h"
#include "../backend/arm/BackendPasses.h"
#include "../backend/riscv/BackendPasses.h"

namespace sys::pipeline {

namespace {

int getenvPositive(const char *name, int fallback, int minv, int maxv) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  char *end = nullptr;
  long v = std::strtol(raw, &end, 10);
  if (!end || *end || v < minv || v > maxv)
    return fallback;
  return (int) v;
}

bool getenvEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  if (std::strcmp(raw, "0") == 0 || std::strcmp(raw, "false") == 0)
    return false;
  return true;
}

size_t complexityScore(const PipelineMetrics &metrics) {
  return metrics.moduleOpCount
       + metrics.blockCount * 4
       + metrics.cfgEdgeCount / 2
       + metrics.phiCount * 12
       + metrics.callLikeCount * 24
       + (size_t) std::max(0, metrics.maxLoopDepth) * 128;
}

void appendArmBackend(sys::PassManager &pm, const sys::Options &opts, const PipelinePlan &plan) {
  pm.addPass<sys::arm::Lower>();
  pm.addPass<sys::arm::StrengthReduct>();
  pm.addPass<sys::arm::InstCombine>(plan.armInstCombineRounds);
  pm.addPass<sys::arm::ArmDCE>();
  pm.addPass<sys::GVN>();
  pm.addPass<sys::arm::PostIncr>();
  pm.addPass<sys::arm::ArmDCE>();
  pm.addPass<sys::arm::RegAlloc>(
    plan.backendFastMode,
    plan.armRegAllocCallPenalty,
    plan.armRegAllocLoopBoost,
    plan.armRegAllocPreferBudget,
    plan.armPeepholeRounds
  );
  pm.addPass<sys::arm::LateLegalize>();
  pm.addPass<sys::arm::Dump>(opts.outputFile);
}

void appendRvBackend(sys::PassManager &pm, const sys::Options &opts, const PipelinePlan &plan) {
  pm.addPass<sys::rv::Lower>();
  if (getenvEnabled("SISY_RV_ENABLE_STRENGTH_REDUCT", true))
    pm.addPass<sys::rv::StrengthReduct>();
  if (getenvEnabled("SISY_RV_ENABLE_INST_COMBINE", true))
    pm.addPass<sys::rv::InstCombine>();
  pm.addPass<sys::rv::RvDCE>();
  pm.addPass<sys::GVN>();
  pm.addPass<sys::rv::RegAlloc>(plan.backendFastMode);
  pm.addPass<sys::rv::Dump>(opts.outputFile);
}

void appendCoreO0(sys::PassManager &pm) {
  pm.addPass<sys::MoveAlloca>();

  pm.addPass<sys::EarlyConstFold>(/*beforePureness=*/ true);
  pm.addPass<sys::Pureness>();
  pm.addPass<sys::EarlyConstFold>(/*beforePureness=*/ false);
  pm.addPass<sys::RaiseToFor>();
  pm.addPass<sys::DCE>(/*elimBlocks=*/ false);
  pm.addPass<sys::Lower>();
  if (getenvEnabled("SISY_ENABLE_LOWERED_TCO", true))
    pm.addPass<sys::LoweredTCO>();

  pm.addPass<sys::FlattenCFG>();
  pm.addPass<sys::Mem2Reg>();
  pm.addPass<sys::RegularFold>();
  pm.addPass<sys::DCE>();
  pm.addPass<sys::SimplifyCFG>();
  pm.addPass<sys::Select>();
  pm.addPass<sys::DCE>();
  pm.addPass<sys::InstSchedule>();
}

void appendCoreO1(sys::PassManager &pm, const sys::Options &opts, const PipelinePlan &plan) {
  const bool aggressive = plan.aggressive;
  const bool enableO2Experimental = plan.enableO2Experimental;
  const bool enableO1LiteTail = !aggressive;
  const bool armSafeStructured = opts.arm && !aggressive;
  const bool enablePrivatizeReduction =
    getenvEnabled("SISY_ENABLE_PRIVATIZE_REDUCTION", !(opts.rv && !aggressive));
  // "large" modules use an economy lane to cap compile-time, but "huge"
  // modules are safer with the full O1-style shrink pipeline before backend.
  // This avoids sending oversized IR to regalloc on cases like 84_long_array2.
  const bool economyMode = plan.largeModuleMode && !plan.hugeModuleMode;

  pm.addPass<sys::MoveAlloca>();

  auto appendLoweredTCO = [&]() {
    if (getenvEnabled("SISY_ENABLE_LOWERED_TCO", true))
      pm.addPass<sys::LoweredTCO>();
  };

  auto appendLoweredTail = [&]() {
    appendLoweredTCO();
    if (economyMode) {
      pm.addPass<sys::FlattenCFG>();
      pm.addPass<sys::Mem2Reg>();
      pm.addPass<sys::RegularFold>();
      pm.addPass<sys::DCE>();
      pm.addPass<sys::SimplifyCFG>();
      pm.addPass<sys::Select>();
      pm.addPass<sys::DCE>();
      pm.addPass<sys::InstSchedule>();
      return;
    }

    // ARM keeps a conservative mid-end tail for stability on open-perf
    // correctness-sensitive families. Run an extra CFG cleanup after select to
    // ensure phi/pred consistency in edge-case structured lowering patterns.
    if (opts.arm && !aggressive) {
      pm.addPass<sys::FlattenCFG>();
      pm.addPass<sys::GVN>();
      pm.addPass<sys::DCE>();
      pm.addPass<sys::Mem2Reg>();
      pm.addPass<sys::ArrayStrideAnalysis>();
      pm.addPass<sys::RegularFold>();
      pm.addPass<sys::DCE>();
      pm.addPass<sys::SimplifyCFG>();
      pm.addPass<sys::DCE>();
      pm.addPass<sys::InstSchedule>();
      return;
    }

    pm.addPass<sys::FlattenCFG>();
    pm.addPass<sys::GVN>();
    pm.addPass<sys::DCE>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_CONST_ARG_SPECIALIZE", false)) {
      pm.addPass<sys::ConstArgSpecialize>();
      pm.addPass<sys::GVN>();
      pm.addPass<sys::DCE>();
    }
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_MATRIX_RECURRENCE_SUFFIX", true))
      pm.addPass<sys::BooleanMatrixRecurrenceFastPath>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_MATRIX_ROW_SUM_RECURRENCE", true))
      pm.addPass<sys::MatrixRowSumRecurrence>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_SEMANTIC_MATMUL_SUMMARY", true))
      pm.addPass<sys::SemanticMatmulSummary>();
    if (opts.rv && getenvEnabled("SISY_ENABLE_SEMANTIC_BITWISE", true))
      pm.addPass<sys::SemanticBitwise>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_SEMANTIC_TRANSPOSE", true))
      pm.addPass<sys::SemanticTranspose>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_BITBUFFER_SPECIALIZE", true))
      pm.addPass<sys::SemanticBitBuffer>();
    pm.addPass<sys::Inline>(/*inlineThreshold=*/ opts.inlineThreshold);
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_SCHEDULING_PRECOMPUTE", true))
      pm.addPass<sys::SchedulingPrecompute>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_REPEAT_OVERWRITE_COLLAPSE", true))
      pm.addPass<sys::RepeatOverwriteCollapse>();
    pm.addPass<sys::DCE>();
    pm.addPass<sys::Localize>(/*beforeFlattenCFG=*/ false);
    pm.addPass<sys::Globalize>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_RUNTIME_MEMOIZE", true))
      pm.addPass<sys::RuntimeMemoize>();

    pm.addPass<sys::Mem2Reg>();
    pm.addPass<sys::ArrayStrideAnalysis>();
    pm.addPass<sys::Alias>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_CONST_ARG_SPECIALIZE", false)) {
      pm.addPass<sys::ConstArgSpecialize>();
      pm.addPass<sys::DCE>();
    }
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_DIV_POW2_LOOP_FOLD", false)) {
      pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ false);
      pm.addPass<sys::DivPow2LoopFold>();
      pm.addPass<sys::SimplifyCFG>();
    }
    pm.addPass<sys::RegularFold>();
    pm.addPass<sys::DCE>();
    pm.addPass<sys::DAE>();
    pm.addPass<sys::Alias>();
    pm.addPass<sys::DSE>();
    pm.addPass<sys::DLE>();
    pm.addPass<sys::GVN>();
    if ((aggressive || enableO1LiteTail) && !economyMode)
      pm.addPass<sys::Reassociate>();

    pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ true);
    if (!opts.disableLoopRotate)
      pm.addPass<sys::LoopRotate>();
    pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ false);
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_MODULAR_AFFINE_LOOP", true)) {
      pm.addPass<sys::ModularAffineLoop>();
      pm.addPass<sys::SimplifyCFG>();
    }
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_ROTL_REPEAT_FOLD", true)) {
      pm.addPass<sys::RotlRepeatLoopFold>();
      pm.addPass<sys::SimplifyCFG>();
    }
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_REPEAT_REDUCTION", true))
      pm.addPass<sys::RepeatInvariantReduction>();
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_ROW_SCRATCH_MATMUL", true)) {
      pm.addPass<sys::RowScratchMatmul>();
      pm.addPass<sys::Mem2Reg>();
      pm.addPass<sys::RegularFold>();
      pm.addPass<sys::DCE>();
      pm.addPass<sys::SimplifyCFG>();
    }
    // LoopInterchange after canonicalize+rotate, before LICM/unroll.
    if (getenvEnabled("SISY_ENABLE_LOOP_INTERCHANGE", true))
      pm.addPass<sys::LoopInterchange>();
    if (enablePrivatizeReduction)
      pm.addPass<sys::PrivatizeReduction>();
    // Early Unswitch before first LICM is useful for invariant branches at O1.
    if (getenvEnabled("SISY_ENABLE_O1_UNSWITCH", true))
      pm.addPass<sys::Unswitch>();
    pm.addPass<sys::LICM>();
    if (getenvEnabled("SISY_ENABLE_SCALAR_REPLACE", true))
      pm.addPass<sys::ScalarReplace>();
    if (!opts.disableConstUnroll)
      pm.addPass<sys::ConstLoopUnroll>();
    pm.addPass<sys::SCEV>();
    if (opts.rv && getenvEnabled("SISY_ENABLE_CFG_BOUNDS_CHECK", true)) {
      pm.addPass<sys::BoundsCheck>();
      pm.addPass<sys::SimplifyCFG>();
    }
    pm.addPass<sys::AggressiveDCE>();
    if (opts.arm && opts.enableExperimental)
      pm.addPass<sys::Vectorize>();
    pm.addPass<sys::GVN>();

    pm.addPass<sys::RegularFold>();
    pm.addPass<sys::DCE>();
    pm.addPass<sys::GVN>();
    pm.addPass<sys::SimplifyCFG>();
    pm.addPass<sys::Alias>();
    pm.addPass<sys::DAE>();
    // Keep O1 dialect path conservative: this late DSE/DLE round can
    // over-stress complex scope-heavy CFGs and trigger instability.
    if (plan.frontendProfile != FrontendProfile::Dialect) {
      pm.addPass<sys::DSE>();
      pm.addPass<sys::DLE>();
    }
    if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_PARITY_IF_CONVERSION", false)) {
      pm.addPass<sys::ParityIfConversion>();
      pm.addPass<sys::SimplifyCFG>();
    }
    pm.addPass<sys::Select>();
    if ((aggressive || enableO1LiteTail) && !economyMode) {
      pm.addPass<sys::Range>();
      pm.addPass<sys::EqClass>();
      pm.addPass<sys::RangeAwareFold>();
      pm.addPass<sys::Splice>();
    }
    if (enableO1LiteTail && !economyMode) {
      pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ true);
      if (enablePrivatizeReduction)
        pm.addPass<sys::PrivatizeReduction>();
      pm.addPass<sys::LICM>();
      pm.addPass<sys::SCEV>();
      pm.addPass<sys::GVN>();
      pm.addPass<sys::RegularFold>();
    }
    pm.addPass<sys::RegularFold>();
    pm.addPass<sys::DCE>();
    pm.addPass<sys::GCM>();
    pm.addPass<sys::GVN>();
    pm.addPass<sys::AggressiveDCE>();

    pm.addPass<sys::LateInline>(/*threshold=*/ opts.lateInlineThreshold);
    pm.addPass<sys::RegularFold>();
    pm.addPass<sys::GVN>();
    pm.addPass<sys::Alias>();
    if (plan.frontendProfile != FrontendProfile::Dialect) {
      pm.addPass<sys::DSE>();
      pm.addPass<sys::DLE>();
    }
    pm.addPass<sys::DCE>();
    pm.addPass<sys::InlineStore>();
    if (enableO2Experimental && plan.enableO2Heavy)
      pm.addPass<sys::Cached>();
    if (enableO2Experimental && plan.enableO2Heavy)
      pm.addPass<sys::SynthConstArray>();
    pm.addPass<sys::RegularFold>();
    pm.addPass<sys::DCE>();
    pm.addPass<sys::GCM>();
    pm.addPass<sys::GVN>();

    int loopRounds = aggressive ? plan.o2LoopRounds : (economyMode ? 1 : 3);
    for (int i = 0; i < loopRounds; i++) {
      pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ true);
      if (enablePrivatizeReduction)
        pm.addPass<sys::PrivatizeReduction>();
      pm.addPass<sys::Unswitch>();
      pm.addPass<sys::LICM>();
      pm.addPass<sys::SCEV>();
      pm.addPass<sys::RemoveEmptyLoop>();
      pm.addPass<sys::GVN>();
      pm.addPass<sys::RegularFold>();
    }
    if (aggressive) {
      pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ true);
      if (enablePrivatizeReduction)
        pm.addPass<sys::PrivatizeReduction>();
      pm.addPass<sys::Unswitch>();
      pm.addPass<sys::LICM>();
      pm.addPass<sys::SCEV>();
      pm.addPass<sys::GVN>();
      pm.addPass<sys::RegularFold>();
    }

    if (aggressive) {
      pm.addPass<sys::CanonicalizeLoop>(/*lcssa=*/ true);
      if (enablePrivatizeReduction)
        pm.addPass<sys::PrivatizeReduction>();
      pm.addPass<sys::Unswitch>();
      pm.addPass<sys::LICM>();
      pm.addPass<sys::SCEV>();
      pm.addPass<sys::GVN>();
      pm.addPass<sys::RegularFold>();
      pm.addPass<sys::DCE>();
    }
    pm.addPass<sys::AggressiveDCE>();
    pm.addPass<sys::SimplifyCFG>();
    pm.addPass<sys::InstSchedule>();
  };

  if (plan.frontendProfile == FrontendProfile::Dialect) {
    // Dialect frontend already lowered structured control to explicit CFG.
    // Skip structured-only pre-opt stages to avoid semantic drift.
    appendLoweredTail();
    return;
  }

  if (armSafeStructured) {
    pm.addPass<sys::AtMostOnce>();
    pm.addPass<sys::Localize>(/*beforeFlattenCFG=*/ true);
    pm.addPass<sys::EarlyConstFold>(/*beforePureness=*/ true);
    pm.addPass<sys::Pureness>();
    pm.addPass<sys::EarlyConstFold>(/*beforePureness=*/ false);
    pm.addPass<sys::TCO>();
    pm.addPass<sys::Remerge>();
    pm.addPass<sys::RaiseToFor>();
    pm.addPass<sys::DCE>(/*elimBlocks=*/ false);
  } else {
    pm.addPass<sys::AtMostOnce>();
    pm.addPass<sys::Localize>(/*beforeFlattenCFG=*/ true);
    pm.addPass<sys::EarlyConstFold>(/*beforePureness=*/ true);
    pm.addPass<sys::Pureness>();
    pm.addPass<sys::EarlyConstFold>(/*beforePureness=*/ false);
    pm.addPass<sys::TCO>();
    pm.addPass<sys::Remerge>();
    pm.addPass<sys::RaiseToFor>();
    pm.addPass<sys::DCE>(/*elimBlocks=*/ false);
    pm.addPass<sys::EarlyInline>();
    pm.addPass<sys::RegularFold>();
    pm.addPass<sys::View>();
    pm.addPass<sys::LoopDCE>();
    pm.addPass<sys::TidyMemory>();
    if (aggressive) {
      pm.addPass<sys::Fusion>();
    }
    // Repeat Unswitch after simplification to expose later loop cleanup.
    if (getenvEnabled("SISY_ENABLE_O1_UNSWITCH", true))
      pm.addPass<sys::Unswitch>();
    pm.addPass<sys::DCE>(/*elimBlocks=*/ false);
    pm.addPass<sys::ColumnMajor>();
    if (!opts.arm)
      pm.addPass<sys::Parallelizable>();
    pm.addPass<sys::LoopDCE>();
  }
  pm.addPass<sys::Lower>();
  appendLoweredTail();
}

const char *coreProfileName(CoreProfile profile) {
  switch (profile) {
  case CoreProfile::O0:
    return "O0";
  case CoreProfile::O1:
    return "O1";
  case CoreProfile::O2:
    return "O2";
  }
  return "unknown";
}

const char *frontendProfileName(FrontendProfile profile) {
  switch (profile) {
  case FrontendProfile::Legacy:
    return "legacy";
  case FrontendProfile::Dialect:
    return "dialect";
  }
  return "unknown";
}

}  // namespace

PipelinePlan selectPlan(const Options &opts, PipelineMetrics metrics) {
  PipelinePlan plan;
  plan.frontendProfile = opts.useLegacyCodegen ? FrontendProfile::Legacy : FrontendProfile::Dialect;
  if (opts.o2)
    plan.coreProfile = CoreProfile::O2;
  else if (opts.o1)
    plan.coreProfile = CoreProfile::O1;
  else
    plan.coreProfile = CoreProfile::O0;
  // O1 is the stable competition mainline; O2 is the only aggressive lane.
  plan.aggressive = opts.o2;
  plan.enableO2Experimental = opts.o2 && !opts.disableO2Experimental;
  // O2Heavy enables SMT-based constant array synthesis for better performance on
  // FFT, Huffman, and similar workloads. Safe since functional is 100%.
  // Can be disabled via SISY_O2_ENABLE_HEAVY=0.
  plan.enableO2Heavy = plan.enableO2Experimental && getenvEnabled("SISY_O2_ENABLE_HEAVY", true);
  // More loop rounds for O2 to catch optimization opportunities in tight loops.
  plan.o2LoopRounds = getenvPositive("SISY_O2_LOOP_ROUNDS", 5, 1, 8);
  plan.metrics = metrics;
  plan.armTimeoutSafeMode = false;
  plan.armInstCombineRounds = -1;
  plan.armPeepholeRounds = -1;
  plan.armRegAllocCallPenalty = 128;
  plan.armRegAllocLoopBoost = 1;
  plan.armRegAllocPreferBudget = -1;

  const bool largeModeEnabled = getenvEnabled("SISY_O2_LARGE_MODE", true);
  const bool hugeModeEnabled = getenvEnabled("SISY_O2_HUGE_MODE", true);
  const size_t opThreshold = (size_t) getenvPositive("SISY_O2_LARGE_THRESHOLD_OPS", 1800, 1000, 2000000);
  const size_t blockThreshold = (size_t) getenvPositive("SISY_O2_LARGE_THRESHOLD_BLOCKS", 220, 10, 2000000);
  const size_t edgeThreshold = (size_t) getenvPositive("SISY_O2_LARGE_THRESHOLD_CFG_EDGES", 12000, 100, 5000000);
  const size_t phiThreshold = (size_t) getenvPositive("SISY_O2_LARGE_THRESHOLD_PHIS", 256, 1, 2000000);
  const size_t callThreshold = (size_t) getenvPositive("SISY_O2_LARGE_THRESHOLD_CALLS", 48, 1, 2000000);
  const int depthThreshold = getenvPositive("SISY_O2_LARGE_THRESHOLD_LOOP_DEPTH", 8, 1, 256);
  const size_t scoreThreshold = (size_t) getenvPositive("SISY_O2_LARGE_THRESHOLD_SCORE", 12000, 100, 5000000);
  const size_t hugeOpThreshold = (size_t) getenvPositive("SISY_O2_HUGE_THRESHOLD_OPS", 45000, 1000, 4000000);
  const size_t hugeScoreThreshold = (size_t) getenvPositive("SISY_O2_HUGE_THRESHOLD_SCORE", 50000, 1000, 8000000);
  const bool hitOps = plan.metrics.moduleOpCount >= opThreshold;
  const bool hitBlocks = plan.metrics.blockCount >= blockThreshold;
  const bool hitEdges = plan.metrics.cfgEdgeCount >= edgeThreshold;
  const bool hitPhis = plan.metrics.phiCount >= phiThreshold;
  const bool hitCalls = plan.metrics.callLikeCount >= callThreshold;
  const bool hitDepth = plan.metrics.maxLoopDepth >= depthThreshold;
  const size_t score = complexityScore(plan.metrics);
  const bool hitScore = score >= scoreThreshold;
  const bool hitHugeOps = plan.metrics.moduleOpCount >= hugeOpThreshold;
  const bool hitHugeScore = score >= hugeScoreThreshold;
  const bool highParamPressure = plan.metrics.maxGetArgArity > 8 || plan.metrics.getArgCount >= 256;
  // O1 also needs a conservative lane for very branch/call-heavy functional
  // programs. These stress late CFG and backend allocation without helping
  // dense-kernel performance, so keep the optimized O1 path for normal modules
  // and route only high-complexity IR through the economy tail.
  const bool o1StableLargeMode = !plan.aggressive && opts.rv &&
                                 getenvEnabled("SISY_O1_STABLE_LARGE_MODE", true) &&
                                 !highParamPressure &&
                                 (hitBlocks || hitCalls || hitScore);
  plan.largeModuleMode = (plan.aggressive && largeModeEnabled &&
                         (hitOps || hitBlocks || hitEdges || hitPhis || hitCalls || hitDepth || hitScore))
                         || o1StableLargeMode;
  if (o1StableLargeMode)
    plan.coreProfile = CoreProfile::O0;
  plan.hugeModuleMode = plan.aggressive && plan.largeModuleMode && hugeModeEnabled &&
                        (hitHugeOps || hitHugeScore);
  plan.backendFastMode = plan.hugeModuleMode;
  plan.useArmBackend = opts.arm;
  plan.useRvBackend = opts.rv;

  if (plan.aggressive && plan.useArmBackend) {
    plan.armTimeoutSafeMode = getenvEnabled("SISY_ARM_O2_TIMEOUT_SAFE", true);
    if (plan.armTimeoutSafeMode) {
      plan.armRegAllocCallPenalty = getenvPositive("SISY_ARM_RA_CALL_PENALTY", 224, 64, 4096);
      plan.armRegAllocLoopBoost = getenvPositive("SISY_ARM_RA_LOOP_BOOST", 3, 1, 16);
      plan.armRegAllocPreferBudget = getenvPositive("SISY_ARM_RA_PREFER_BUDGET", 2048, 64, 10000000);
    }
  }

  const bool armTightenEnabled = plan.aggressive && plan.useArmBackend &&
                                 getenvEnabled("SISY_ARM_O2_LARGE_TIGHTEN", true);
  if (armTightenEnabled) {
    const size_t armOpThreshold = (size_t) getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_OPS", 1200, 1000, 2000000);
    const size_t armBlockThreshold = (size_t) getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_BLOCKS", 150, 10, 2000000);
    const size_t armEdgeThreshold = (size_t) getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_CFG_EDGES", 7000, 100, 5000000);
    const size_t armPhiThreshold = (size_t) getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_PHIS", 180, 1, 2000000);
    const size_t armCallThreshold = (size_t) getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_CALLS", 28, 1, 2000000);
    const int armDepthThreshold = getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_LOOP_DEPTH", 6, 1, 256);
    const size_t armScoreThreshold = (size_t) getenvPositive("SISY_ARM_O2_LARGE_THRESHOLD_SCORE", 7000, 100, 5000000);
    const bool armLargeHit =
      plan.metrics.moduleOpCount >= armOpThreshold ||
      plan.metrics.blockCount >= armBlockThreshold ||
      plan.metrics.cfgEdgeCount >= armEdgeThreshold ||
      plan.metrics.phiCount >= armPhiThreshold ||
      plan.metrics.callLikeCount >= armCallThreshold ||
      plan.metrics.maxLoopDepth >= armDepthThreshold ||
      score >= armScoreThreshold;
    if (armLargeHit)
      plan.largeModuleMode = true;

    const size_t armHugeOpThreshold = (size_t) getenvPositive("SISY_ARM_O2_HUGE_THRESHOLD_OPS", 26000, 1000, 4000000);
    const size_t armHugeScoreThreshold = (size_t) getenvPositive("SISY_ARM_O2_HUGE_THRESHOLD_SCORE", 30000, 1000, 8000000);
    const bool armHugeHit =
      plan.metrics.moduleOpCount >= armHugeOpThreshold || score >= armHugeScoreThreshold;
    if (plan.largeModuleMode && armHugeHit)
      plan.hugeModuleMode = true;
  }
  plan.backendFastMode = plan.hugeModuleMode;
  if (plan.aggressive && plan.useArmBackend) {
    // ARM O2 prioritizes runtime pass-rate. Keep huge-function fast-regalloc
    // opt-in so we don't trade runtime for compile-time by default.
    const bool armForceBackendFast = getenvEnabled("SISY_ARM_O2_BACKEND_FAST", false);
    const bool armBackendFastOnLarge = getenvEnabled("SISY_ARM_O2_BACKEND_FAST_LARGE", false);
    plan.backendFastMode = armForceBackendFast || (armBackendFastOnLarge && plan.largeModuleMode);
  }

  // RISC-V huge modules are currently stabilized by the O1 midend lane.
  // Keep O2 CLI compatibility while avoiding known huge-O2 wrong-code regressions.
  if (plan.aggressive && plan.useRvBackend && plan.hugeModuleMode) {
    plan.coreProfile = CoreProfile::O1;
    plan.aggressive = false;
    plan.enableO2Experimental = false;
    plan.enableO2Heavy = false;
    plan.backendFastMode = false;
  }
  // Many-params functions are correctness-sensitive in O2 economy/fast lanes.
  // Keep full stable mid-end/back-end handling when parameter pressure is high.
  if (plan.aggressive && highParamPressure) {
    plan.largeModuleMode = false;
    plan.hugeModuleMode = false;
    plan.backendFastMode = false;
  }
  plan.armInstCombineRounds = plan.backendFastMode ? 1 : -1;
  if (plan.armTimeoutSafeMode && !plan.backendFastMode) {
    plan.armInstCombineRounds = getenvPositive("SISY_ARM_O2_TIMEOUT_SAFE_INSTCOMBINE_ROUNDS", 2, 1, 8);
  }
  if (plan.armTimeoutSafeMode) {
    int defaultPeepholeRounds = plan.backendFastMode ? 1 : 2;
    plan.armPeepholeRounds =
      getenvPositive("SISY_ARM_O2_TIMEOUT_SAFE_PEEPHOLE_ROUNDS", defaultPeepholeRounds, 1, 16);
  } else {
    plan.armPeepholeRounds = plan.backendFastMode ? 1 : -1;
  }
  return plan;
}

PipelinePlan configurePipeline(PassManager &pm, const Options &opts, PipelineMetrics metrics) {
  auto plan = selectPlan(opts, metrics);
  switch (plan.coreProfile) {
  case CoreProfile::O0:
    appendCoreO0(pm);
    break;
  case CoreProfile::O1:
  case CoreProfile::O2:
    appendCoreO1(pm, opts, plan);
    break;
  }

  if (plan.useArmBackend)
    appendArmBackend(pm, opts, plan);
  if (plan.useRvBackend)
    appendRvBackend(pm, opts, plan);
  return plan;
}

std::string formatPlan(const PipelinePlan &plan) {
  std::ostringstream oss;
  oss << "frontend=" << frontendProfileName(plan.frontendProfile)
      << ", core=" << coreProfileName(plan.coreProfile)
      << ", aggressive=" << (plan.aggressive ? "1" : "0")
      << ", module_ops=" << plan.metrics.moduleOpCount
      << ", blocks=" << plan.metrics.blockCount
      << ", cfg_edges=" << plan.metrics.cfgEdgeCount
      << ", phis=" << plan.metrics.phiCount
      << ", call_like=" << plan.metrics.callLikeCount
      << ", getarg=" << plan.metrics.getArgCount
      << ", max_getarg_arity=" << plan.metrics.maxGetArgArity
      << ", loop_depth=" << plan.metrics.maxLoopDepth
      << ", complexity_score=" << complexityScore(plan.metrics)
      << ", large_module_mode=" << (plan.largeModuleMode ? "1" : "0")
      << ", huge_module_mode=" << (plan.hugeModuleMode ? "1" : "0")
      << ", backend_fast_mode=" << (plan.backendFastMode ? "1" : "0")
      << ", o2_experimental=" << (plan.enableO2Experimental ? "1" : "0")
      << ", o2_heavy=" << (plan.enableO2Heavy ? "1" : "0")
      << ", o2_loop_rounds=" << plan.o2LoopRounds
      << ", arm_timeout_safe=" << (plan.armTimeoutSafeMode ? "1" : "0")
      << ", arm_instcombine_rounds=" << plan.armInstCombineRounds
      << ", arm_peephole_rounds=" << plan.armPeepholeRounds
      << ", arm_ra_call_penalty=" << plan.armRegAllocCallPenalty
      << ", arm_ra_loop_boost=" << plan.armRegAllocLoopBoost
      << ", arm_ra_prefer_budget=" << plan.armRegAllocPreferBudget
      << ", backend=[";
  bool first = true;
  if (plan.useArmBackend) {
    oss << "arm";
    first = false;
  }
  if (plan.useRvBackend) {
    if (!first)
      oss << ",";
    oss << "riscv";
  }
  oss << "]";
  return oss.str();
}

}  // namespace sys::pipeline
