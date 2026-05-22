#include "LoopPasses.h"
#include <algorithm>
#include <cmath>

using namespace sys;

namespace {

// ===========================================================================
// Specialized Optimizers for Recognized Patterns
// ===========================================================================

class MatMulOptimizer {
public:
  // Check if matrix sizes fit for Strassen algorithm
  static bool canUseStrassen(int M, int N, int K) {
    // Strassen is beneficial when matrices are large enough (typically > 64x64)
    // and dimensions are powers of 2
    if (M < 64 || N < 64 || K < 64)
      return false;

    // Check if dimensions are close to power of 2
    auto isPowerOf2OrClose = [](int n) {
      int log = 0;
      int p = 1;
      while (p < n) { p *= 2; log++; }
      return (p == n) || (p / 2 >= n * 7 / 8);  // within 12.5%
    };

    return isPowerOf2OrClose(M) && isPowerOf2OrClose(N) && isPowerOf2OrClose(K);
  }

  // Get optimal block size for block-wise GEMM
  static int getOptimalBlockSize(int totalSize) {
    // Typical block sizes: 16, 32, 64 depending on cache and SIMD width
    // For RISC-V/ARM: 32-64 is common
    if (totalSize >= 256)
      return 64;
    else if (totalSize >= 128)
      return 32;
    else
      return 16;
  }

  // Compute reuse factor for tiled matmul
  static int getDataReuseFactor(int tileSize, int elemWidth) {
    // For tiled matmul with NxN blocks:
    // Each element of A is reused N times
    // Each element of B is reused N times
    // Total: 2*N^2 operations per N^2 elements of A or B
    return (2 * tileSize * tileSize) / (tileSize * elemWidth / 4);
  }
};

class Conv2DOptimizer {
public:
  // Check if im2col transformation is beneficial
  static bool shouldUseIm2col(int H, int W, int KH, int KW, int stride) {
    // im2col is beneficial when:
    // 1. Kernel is large (KH*KW > 9)
    // 2. Stride is 1 (minimal overlapping regions)
    // 3. Input size is reasonable (not tiny)

    if (KH * KW <= 9 || stride > 1)
      return false;

    return (H * W) > 32;
  }

  // Compute optimal im2col buffer size
  static int getIm2colBufferSize(int H, int W, int KH, int KW, int elemWidth) {
    // im2col creates a matrix of size (KH*KW, OH*OW)
    // where OH*OW is number of output positions
    int numPatches = H * W;  // simplified estimate
    return numPatches * KH * KW * elemWidth;
  }

  // Get Winograd F(2x2, 3x3) applicability
  static bool canUseWinograd(int KH, int KW) {
    // Winograd requires specific kernel sizes
    // Common: F(2x2, 3x3), F(3x3, 3x3), F(4x4, 3x3)
    return (KH == 3 && KW == 3);
  }
};

class SortOptimizer {
public:
  // Check if bitonic sort size is efficient
  static bool isBitonicSortEfficient(int N) {
    // Bitonic sort is efficient for powers of 2, and good for GPU/SIMD
    // Typical threshold: 8 <= N <= 4096
    if (N < 8 || N > 4096)
      return false;

    // Check if N is power of 2
    return (N & (N - 1)) == 0;
  }

  // Compute bitonic sort network depth
  static int getNetworkDepth(int N) {
    // Depth = log2(N) * (log2(N) + 1) / 2
    int log = 0;
    int temp = N;
    while (temp > 1) { temp >>= 1; log++; }
    return log * (log + 1) / 2;
  }

  // Check if parallel sorting networks are applicable
  static bool canParallelizeSort(int N) {
    // Parallel sorting networks work well for N >= 4 and when N is power of 2
    return isBitonicSortEfficient(N);
  }
};

class FFTOptimizer {
public:
  // Check if size is suitable for FFT optimization
  static bool isFFTOptimizable(int N) {
    // FFT is efficient for N = 2^k where k >= 4 (N >= 16)
    // Also works for N = 3 * 2^k, 5 * 2^k, etc. (mixed-radix)
    
    if (N < 16)
      return false;

    // Check if N is a power of 2
    if ((N & (N - 1)) == 0)
      return true;

    // Check for mixed-radix factors (3, 5)
    int temp = N;
    for (int factor : {3, 5, 7}) {
      while (temp % factor == 0)
        temp /= factor;
    }

    // If only 2 remains, it's mixed radix
    return (temp & (temp - 1)) == 0;
  }

  // Get FFT algorithm stages
  static int getFFTStages(int N) {
    // Number of stages = log2(N)
    int stages = 0;
    int temp = N;
    while (temp > 1) { temp >>= 1; stages++; }
    return stages;
  }
};

}  // namespace

// Exports
bool shouldUseStrassenMatMul(int M, int N, int K) {
  return MatMulOptimizer::canUseStrassen(M, N, K);
}

bool shouldUseIm2colConv2D(int H, int W, int KH, int KW, int stride) {
  return Conv2DOptimizer::shouldUseIm2col(H, W, KH, KW, stride);
}

bool shouldUseWinograd(int KH, int KW) {
  return Conv2DOptimizer::canUseWinograd(KH, KW);
}

bool canParallelizeBitonicSort(int N) {
  return SortOptimizer::canParallelizeSort(N);
}

