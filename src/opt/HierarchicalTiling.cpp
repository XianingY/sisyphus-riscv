#include "LoopPasses.h"
#include <cmath>
#include <algorithm>

using namespace sys;

namespace {

// ===========================================================================
// Hierarchical Tiling for L1, L2, L3 Cache Levels
// ===========================================================================

struct TileConfig {
  int l1TileSize;  // L1: typically 4x4 to 8x8 for floats
  int l2TileSize;  // L2: typically 32x32 to 64x64
  int l3TileSize;  // L3: typically 128x128+
};

class HierarchicalTiler {
public:
  HierarchicalTiler() {
    // Default cache configuration (tuned for modern processors)
    // Adjust based on target architecture (RISC-V vs ARM)
    l1CacheSize = 32 * 1024;      // 32 KB L1 cache
    l2CacheSize = 256 * 1024;     // 256 KB L2 cache
    l3CacheSize = 8 * 1024 * 1024; // 8 MB L3 cache
  }

  // Get optimal tile size for a given cache level and element width
  int computeTileSize(int cacheSize, int elemWidth, int numArrays) {
    // Cache utilization target: 80% of cache
    int usableCache = (cacheSize * 80) / 100;

    // For a 2D tiling of N x N blocks with W-byte elements
    // Memory = N * N * W * numArrays
    // Solve for N: N = sqrt(usableCache / (W * numArrays))

    if (elemWidth <= 0 || numArrays <= 0)
      return 8;  // conservative default

    double tileArea = usableCache / (double)(elemWidth * numArrays);
    int tileSize = std::max(4, (int)std::sqrt(tileArea));

    // Round down to multiple of 4 for good alignment
    tileSize = (tileSize / 4) * 4;

    return tileSize;
  }

  // Generate hierarchical tiling configuration for matrix operation
  TileConfig getTilingConfig(int matrixSize, int elemWidth) {
    TileConfig config;

    // L1 tile: fit in L1 cache
    config.l1TileSize = computeTileSize(l1CacheSize, elemWidth, 3);
    config.l1TileSize = std::min(config.l1TileSize, 8);  // Cap at 8x8

    // L2 tile: fit in L2 cache, larger than L1
    config.l2TileSize = computeTileSize(l2CacheSize, elemWidth, 3);
    config.l2TileSize = std::max(config.l2TileSize, config.l1TileSize * 4);

    // L3 tile: fit in L3 cache, even larger
    config.l3TileSize = computeTileSize(l3CacheSize, elemWidth, 3);
    config.l3TileSize = std::max(config.l3TileSize, config.l2TileSize * 2);

    return config;
  }

  // Estimate actual cache misses for a tiling configuration
  int estimateMissRatio(int tileSize, int matrixSize, int elemWidth) {
    // Simplified model: misses occur at tile boundaries
    int tilesPerDimension = (matrixSize + tileSize - 1) / tileSize;
    
    // Rough estimation: ~1% miss rate if good locality
    int missesPerIter = (tilesPerDimension * tilesPerDimension) / 100;
    
    return std::max(1, missesPerIter);
  }

private:
  int l1CacheSize;
  int l2CacheSize;
  int l3CacheSize;
};

}  // namespace

// Export
TileConfig getHierarchicalTilingConfig(int matrixSize, int elemWidth) {
  HierarchicalTiler tiler;
  return tiler.getTilingConfig(matrixSize, elemWidth);
}

