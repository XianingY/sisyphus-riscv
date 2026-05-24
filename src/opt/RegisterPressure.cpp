#include "LoopPasses.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

using namespace sys;

namespace {

// ===========================================================================
// Register Pressure Analysis for Loop Unroll Factor Selection
// ===========================================================================

struct LiveValue {
  Op* value;
  int firstUse;   // Instruction index of first use
  int lastUse;    // Instruction index of last use
  int liveLength; // lastUse - firstUse
};

class RegisterPressureAnalyzer {
public:
  explicit RegisterPressureAnalyzer(LoopInfo* loop) : loop(loop), maxPressure(0) {}

  // Estimate max register pressure during unroll factor N
  int estimatePressure(int unrollFactor) {
    if (!loop)
      return 0;

    std::vector<Op*> loopOps;
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        loopOps.push_back(op);
      }
    }

    if (loopOps.empty())
      return 0;

    // For each value, compute live range
    std::vector<LiveValue> liveValues;
    int instCount = 0;
    std::unordered_set<Op*> defs;

    for (size_t i = 0; i < loopOps.size(); i++) {
      Op* op = loopOps[i];
      defs.insert(op);

      // Track uses
      for (int j = 0; j < op->getOperandCount(); j++) {
        Op* use = op->DEF(j);
        if (defs.count(use)) {
          LiveValue lv;
          lv.value = use;
          lv.firstUse = -1;  // computed later
          lv.lastUse = i;
          lv.liveLength = 0;
          // In a real implementation, would track all uses
        }
      }

      instCount++;
    }

    // After unroll, effective loop body size = instCount * unrollFactor
    // Estimate register pressure as: unroll_factor * avg_live_values
    // Conservative: assume average 4 live values per instruction
    int avgLiveValues = 4;
    return unrollFactor * avgLiveValues;
  }

  // Check if unroll factor is safe (won't cause excessive spilling)
  bool isSafeUnrollFactor(int unrollFactor, int maxRegisterCount = 32) {
    int pressure = estimatePressure(unrollFactor);
    // Reserve some registers for caller/callee saves and temps
    int availableRegs = maxRegisterCount - 8;
    return pressure <= availableRegs;
  }

  // Get recommended unroll factor
  int getRecommendedUnrollFactor(int defaultFactor = 4, int maxFactor = 16) {
    // Find largest safe factor
    for (int factor = maxFactor; factor >= 1; factor--) {
      if (isSafeUnrollFactor(factor)) {
        return factor;
      }
    }
    return 1;
  }

private:
  LoopInfo* loop;
  int maxPressure;
};

}  // namespace

// Export function for unroll pass
int calculateAdaptiveUnrollFactor(LoopInfo* loop, int defaultFactor) {
  if (!loop)
    return 1;

  RegisterPressureAnalyzer analyzer(loop);

  // Check loop depth: nested loops get smaller factors
  int depth = 1;
  auto parent = loop->getParent();
  while (parent) {
    depth++;
    parent = parent->getParent();
  }

  // Reduce factor for nested loops (depth 1 = default, depth 2 = factor/2, etc)
  int maxFactor = std::max(1, defaultFactor / depth);
  return analyzer.getRecommendedUnrollFactor(defaultFactor, maxFactor);
}

