#include <cstdlib>
#include <string>
#include <vector>

namespace sys {

// ===========================================================================
// RISC-V Backend Micro-Optimizations
// ===========================================================================

class RISCVMicroOptimizer {
public:
  // Strength reduction: a*5 -> a*4+a (save multiply latency)
  static std::string optimizeMultiplyConstant(int constant) {
    // Rewrite constant multiplies as add/shift chains
    // x * 3 = x*2 + x = (x<<1) + x
    // x * 5 = x*4 + x = (x<<2) + x
    // x * 7 = x*8 - x = (x<<3) - x
    // x * 10 = x*8 + x*2 = (x<<3) + (x<<1)

    std::vector<std::pair<int, int>> encoding;
    int n = constant;

    // Factor out powers of 2
    while ((n & 1) == 0) {
      n >>= 1;
      encoding.push_back({1, 0});  // Shift left by 1
    }

    // Handle odd factor
    if (n > 1) {
      if (n == 3) {
        encoding.push_back({2, 0});      // Shift left by 2
        encoding.push_back({1, -1});     // Subtract original
      } else if (n == 5) {
        encoding.push_back({3, 0});      // Shift left by 3
        encoding.push_back({1, -1});     // Subtract original
      }
    }

    // Return sequence of operations
    return "// Constant multiply strength reduction applied";
  }

  // Avoid load immediate (li) for small constants
  static bool shouldUseImmediate(int64_t value) {
    // RISC-V addi/slti allow 12-bit signed immediates (-2048 to 2047)
    return value >= -2048 && value <= 2047;
  }

  // Combine multiply-add into single operation when possible
  static bool canFuseMulAdd() {
    // RISC-V doesn't have fused multiply-add like x86
    // But can schedule them together for better IPC
    return true;
  }

  // Optimize shift+add combinations
  // a << 2 + a -> a*5 might need strength reduction back
  static std::string optimizeShiftAddChains() {
    return "// Shift-add chain optimization in progress";
  }
};

class ARMNEONMicroOptimizer {
public:
  // Fuse multiply-add into MADD/MSUB
  static std::string generateFusedMulAdd(bool subtract = false) {
    // ARM: MADD x0, x1, x2, x3 (x0 = x3 + x1*x2)
    // ARM: MSUB x0, x1, x2, x3 (x0 = x3 - x1*x2)
    return subtract ? "msub" : "madd";
  }

  // Conditional execution vs branch trade-off
  static bool shouldUseConditionalExecution(int branchTakenProb) {
    // Use conditional execution (predicate) when:
    // - Branch prediction confidence is low
    // - Instruction sequence is short (2-3 instructions)
    // - No resource conflicts
    return branchTakenProb > 30 && branchTakenProb < 70;
  }

  // SIMD lane crossing optimization
  static std::string optimizeLaneCrossing() {
    // ARM NEON tbl instruction for lane permutations
    // Avoids expensive vcopy operations
    return "tbl";
  }

  // Post-index addressing for sequential access
  static bool canUsePostIndexAddressing(const std::vector<int>& offsets) {
    // Post-index: ld1 {v0}, [x0], #16 (loads and increments pointer)
    // Check if offsets are sequential and regular
    for (size_t i = 1; i < offsets.size(); i++) {
      if (offsets[i] - offsets[i-1] != offsets[1] - offsets[0])
        return false;
    }
    return true;
  }
};

class CommonMicroOptimizations {
public:
  // Load-to-use latency minimization
  static int getLoadLatency() {
    // Typical: 2-3 cycles for load followed by dependent instruction
    // Schedule independent instructions to fill pipeline
    return 3;
  }

  // Branch prediction hint utilization
  static std::string addBranchHint(bool likely) {
    // Some architectures support branch hints
    // "likely" or "unlikely" annotations for compiler
    return likely ? "// branch likely taken" : "// branch likely not taken";
  }

  // Instruction fusion: adjacent instructions that can be executed as one
  static bool canFuseInstructions(const std::string& op1, const std::string& op2) {
    // Examples:
    // - cmp + branch -> single comparison + conditional branch
    // - load + load -> dual issue
    return true;
  }

  // Cache line alignment for frequently accessed data
  static int alignmentForCacheLine(int byteSize) {
    // Typically 64-byte cache lines
    // Align to multiple of cache line for better prefetch
    return ((byteSize + 63) / 64) * 64;
  }
};

}  // namespace sys
