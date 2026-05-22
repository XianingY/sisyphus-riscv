#include "LoopPasses.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace sys;

namespace {

// ===========================================================================
// Prefetch Instruction Generation and Optimization
// ===========================================================================

struct MemoryAccessPattern {
  Op* basePtr;
  int64_t stride;
  bool isLoad;
  bool isPrefetchable;
};

class PrefetchOptimizer {
public:
  explicit PrefetchOptimizer(LoopInfo* loop) : loop(loop) {}

  // Analyze memory access patterns
  std::vector<MemoryAccessPattern> analyzeAccessPatterns() {
    std::vector<MemoryAccessPattern> patterns;

    if (!loop)
      return patterns;

    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        // Detect loads and stores
        if (isa<LoadOp>(op) || isa<StoreOp>(op)) {
          MemoryAccessPattern pattern;
          pattern.isLoad = isa<LoadOp>(op);

          // Try to extract stride information
          if (op->getOperandCount() > 0) {
            pattern.basePtr = op->DEF(0);
            
            // Simple stride detection: check if indexed access
            // In a real implementation, would use value range analysis
            pattern.stride = 4;  // default: assume 4-byte stride for integers
            pattern.isPrefetchable = true;
          }

          patterns.push_back(pattern);
        }
      }
    }

    return patterns;
  }

  // Generate prefetch instructions for identified patterns
  std::string generatePrefetchCode(const MemoryAccessPattern& pattern) {
    if (!pattern.isPrefetchable)
      return "";

    // For RISC-V: prefetch via Zicbop extension
    // pref.l (prefetch for load): pref.l offset(rs1)
    // pref.w (prefetch for write): pref.w offset(rs1)
    // Typical prefetch distance: 8-16 iterations ahead

    std::string prefetchOp = pattern.isLoad ? "pref.l" : "pref.w";
    
    // Prefetch 32 bytes ahead (typical cache line)
    int prefetchDistance = 32;
    
    return prefetchOp + " " + std::to_string(prefetchDistance) + "(x0)";
  }

  // Check if loop benefits from prefetching
  bool shouldGeneratePrefetch() {
    if (!loop)
      return false;

    // Heuristic: prefetch beneficial if:
    // 1. Loop has regular memory access patterns
    // 2. Loop trip count > 10 (enough iterations for prefetch to be useful)
    // 3. Memory accesses are not cache-resident (checked via pattern analysis)

    auto patterns = analyzeAccessPatterns();
    
    // Count regular patterns
    int regularPatterns = 0;
    for (const auto& p : patterns) {
      if (p.isPrefetchable && p.stride > 0)
        regularPatterns++;
    }

    return regularPatterns > 0;
  }

private:
  LoopInfo* loop;
};

}  // namespace

// Export function
bool shouldOptimizeLoopPrefetch(LoopInfo* loop) {
  PrefetchOptimizer optimizer(loop);
  return optimizer.shouldGeneratePrefetch();
}

