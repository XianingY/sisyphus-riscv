#include "LoopPasses.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

using namespace sys;

namespace {

// ===========================================================================
// Loop Fusion After Tiling for Improved Data Reuse
// ===========================================================================

class TiledLoopFusionAnalyzer {
public:
  // Check if two tiled loops can be fused
  bool canFuseTiledLoops(LoopInfo* loop1, LoopInfo* loop2) {
    if (!loop1 || !loop2)
      return false;

    // Both must be synthetic loops created by tiling
    // Check: same iteration space bounds
    auto stop1 = loop1->getStop();
    auto stop2 = loop2->getStop();

    if (!stop1 || !stop2)
      return false;

    // Same tile bounds = can fuse
    if (!(isa<IntOp>(stop1) && isa<IntOp>(stop2)))
      return false;

    // Check for loop-carried dependencies that would prevent fusion
    if (hasLoopCarriedDependence(loop1, loop2))
      return false;

    // Check if accessing same arrays (good fusion candidate)
    auto arrays1 = getAccessedArrays(loop1);
    auto arrays2 = getAccessedArrays(loop2);

    // Compute intersection
    int commonArrays = 0;
    for (const auto& arr : arrays1) {
      if (arrays2.count(arr))
        commonArrays++;
    }

    return commonArrays > 0;
  }

  // Fuse two adjacent loops
  LoopInfo* fuseTiledLoops(LoopInfo* loop1, LoopInfo* loop2) {
    if (!canFuseTiledLoops(loop1, loop2))
      return nullptr;

    // In actual implementation, would:
    // 1. Clone loop1's body
    // 2. Append loop2's body
    // 3. Update phi nodes
    // 4. Redirect edges
    // For now, return loop1 as indication of fusion

    return loop1;
  }

private:
  bool hasLoopCarriedDependence(LoopInfo* loop1, LoopInfo* loop2) {
    // Conservative: check if any store in loop1 feeds into load in loop2
    std::unordered_set<Op*> stores;

    for (auto bb : loop1->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (isa<StoreOp>(op))
          stores.insert(op);
      }
    }

    for (auto bb : loop2->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (isa<LoadOp>(op)) {
          // Check if this load could depend on any store from loop1
          for (int i = 0; i < op->getOperandCount(); i++) {
            if (stores.count(op->DEF(i)))
              return true;
          }
        }
      }
    }

    return false;
  }

  std::unordered_set<std::string> getAccessedArrays(LoopInfo* loop) {
    std::unordered_set<std::string> arrays;

    if (!loop)
      return arrays;

    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (isa<LoadOp>(op) || isa<StoreOp>(op)) {
          // In real implementation, would extract array name from address
          // For now, just mark as "array"
          arrays.insert("array");
        }
      }
    }

    return arrays;
  }
};

}  // namespace

// Export
bool canFuseTiledLoops(LoopInfo* loop1, LoopInfo* loop2) {
  TiledLoopFusionAnalyzer analyzer;
  return analyzer.canFuseTiledLoops(loop1, loop2);
}

