#ifndef ADVANCED_CONV2D_TRANSFORM_H
#define ADVANCED_CONV2D_TRANSFORM_H

#include "Passes.h"
#include "LoopPasses.h"
#include "../codegen/CodeGen.h"

#include <map>
#include <string>

namespace sys {

class AdvancedConv2DTransformPass : public Pass {
  int conv2dDetected = 0;
  int winogradApplied = 0;
  int im2colApplied = 0;
  int rejectedShape = 0;
  int routedFallback = 0;

  struct ConvParams {
    int H = 0;
    int W = 0;
    int KH = 0;
    int KW = 0;
    int stride = 1;
    Op *inputArray = nullptr;
    Op *outputArray = nullptr;
    Op *kernelArray = nullptr;
  };

  bool extractParams(LoopInfo *loop, ConvParams &params);
  bool runWinograd(FuncOp *func, LoopInfo *loop, const ConvParams &params);
  bool runIm2col(FuncOp *func, LoopInfo *loop, const ConvParams &params);

public:
  explicit AdvancedConv2DTransformPass(ModuleOp *module) : Pass(module) {}

  std::string name() override { return "advanced-conv2d-transform"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

} // namespace sys

#endif // ADVANCED_CONV2D_TRANSFORM_H
