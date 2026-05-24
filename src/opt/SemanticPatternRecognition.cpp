#include "LoopPasses.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

using namespace sys;

namespace {

// ===========================================================================
// Semantic Pattern Recognition: MatMul, Conv2D, Sort, FFT
// ===========================================================================

enum class SemanticPattern {
  Unknown,
  MatrixMultiply,
  Convolution2D,
  BitonicSort,
  FastFourierTransform
};

struct PatternInfo {
  SemanticPattern pattern;
  int nestingDepth;
};

class SemanticPatternAnalyzer {
public:
  explicit SemanticPatternAnalyzer(LoopInfo* loop) : loop(loop) {}

  // Detect if this is a matrix multiplication pattern
  bool isMatrixMultiply() {
    if (!loop || loop->subloops.size() < 2)
      return false;

    // Pattern: nested loop with accumulation
    // Check innermost loop for multiply-accumulate

    LoopInfo* innermost = loop;
    while (!innermost->subloops.empty()) {
      innermost = innermost->subloops[0];
    }

    int accumulateOps = 0;
    for (auto bb : innermost->getBlocks()) {
      for (auto op : bb->getOps()) {
        if ((isa<MulIOp>(op) || isa<MulFOp>(op) ||
             isa<AddIOp>(op) || isa<AddFOp>(op)) &&
            op->getOperandCount() == 2) {
          accumulateOps++;
        }
      }
    }

    return accumulateOps >= 2;
  }

  // Detect if this is a 2D convolution pattern
  bool isConvolution2D() {
    if (!loop || loop->subloops.size() < 3)
      return false;

    // Pattern: 4+ nested loops with offset array accesses
    int nestLevel = 0;
    LoopInfo* temp = loop;
    while (!temp->subloops.empty()) {
      nestLevel++;
      temp = temp->subloops[0];
    }

    return nestLevel >= 4;
  }

  // Detect if this is a bitonic sort pattern
  bool isBitonicSort() {
    if (!loop)
      return false;

    // Pattern: compare-based sorting (many comparison ops)
    int compares = 0;
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (isa<LtOp>(op) || isa<LeOp>(op)) {
          compares++;
        }
      }
    }

    return compares > 5;
  }

  // Detect FFT patterns
  bool isFastFourierTransform() {
    if (!loop || loop->subloops.size() < 1)
      return false;

    // Pattern: complex multiply operations
    int complexMuls = 0;
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        if ((isa<MulFOp>(op) || isa<AddFOp>(op) || 
             isa<SubFOp>(op)) && op->getOperandCount() == 2) {
          complexMuls++;
        }
      }
    }

    return complexMuls > 10;
  }

  PatternInfo getPattern() {
    PatternInfo info;
    
    // Calculate nesting depth
    int depth = 1;
    LoopInfo* temp = loop;
    while (temp && !temp->subloops.empty()) {
      depth++;
      temp = temp->subloops[0];
    }
    info.nestingDepth = depth;

    if (isMatrixMultiply()) {
      info.pattern = SemanticPattern::MatrixMultiply;
    } else if (isConvolution2D()) {
      info.pattern = SemanticPattern::Convolution2D;
    } else if (isBitonicSort()) {
      info.pattern = SemanticPattern::BitonicSort;
    } else if (isFastFourierTransform()) {
      info.pattern = SemanticPattern::FastFourierTransform;
    } else {
      info.pattern = SemanticPattern::Unknown;
    }

    return info;
  }

private:
  LoopInfo* loop;
};

}  // namespace

// Export recognition functions
bool isMatMulPattern(LoopInfo* loop) {
  SemanticPatternAnalyzer analyzer(loop);
  return analyzer.isMatrixMultiply();
}

bool isConv2DPattern(LoopInfo* loop) {
  SemanticPatternAnalyzer analyzer(loop);
  return analyzer.isConvolution2D();
}

bool isSortPattern(LoopInfo* loop) {
  SemanticPatternAnalyzer analyzer(loop);
  return analyzer.isBitonicSort();
}

bool isFFTPattern(LoopInfo* loop) {
  SemanticPatternAnalyzer analyzer(loop);
  return analyzer.isFastFourierTransform();
}

