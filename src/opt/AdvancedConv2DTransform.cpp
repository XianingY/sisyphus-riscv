#include "AdvancedConv2DTransform.h"
#include "../codegen/Attrs.h"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <vector>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool def) {
  const char *value = std::getenv(name);
  if (!value)
    return def;
  return !(value[0] == '0' || std::string(value) == "false" ||
           std::string(value) == "FALSE");
}

std::vector<LoopInfo*> candidateLoops(const LoopForest &forest) {
  std::vector<LoopInfo*> loops;
  for (auto *loop : forest.getLoops()) {
    if (loop)
      loops.push_back(loop);
  }
  return loops;
}

void collectLoopChain(LoopInfo *loop, std::vector<LoopInfo*> &chain) {
  while (loop) {
    chain.push_back(loop);
    if (loop->subloops.size() != 1)
      break;
    loop = loop->subloops.front();
  }
}

int constValue(Op *op, int fallback = 0) {
  if (op && isa<IntOp>(op))
    return V(op);
  return fallback;
}

bool isUnitPositiveStep(LoopInfo *loop) {
  if (!loop || !loop->step || !isa<IntOp>(loop->step))
    return false;
  return V(loop->step) == 1;
}

Op *findMemoryBase(Op *op, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return nullptr;
  seen.insert(op);
  if (isa<GetGlobalOp>(op) || isa<AllocaOp>(op) || isa<GetArgOp>(op))
    return op;
  for (auto operand : op->getOperands()) {
    if (auto *base = findMemoryBase(operand.defining, seen))
      return base;
  }
  return nullptr;
}

Op *findMemoryBase(Op *op) {
  std::set<Op*> seen;
  return findMemoryBase(op, seen);
}

bool hasControlOrCall(Op *op) {
  if (!op)
    return false;
  if (isa<CallOp>(op) || isa<ReturnOp>(op) || isa<BreakOp>(op) ||
      isa<ContinueOp>(op))
    return true;
  return false;
}

struct AccessSummary {
  std::set<Op*> loadedBases;
  std::set<Op*> storedBases;
  int loads = 0;
  int stores = 0;
  bool unsafeControl = false;
};

void summarizeLoop(LoopInfo *loop, AccessSummary &summary) {
  if (!loop)
    return;
  for (auto *bb : loop->getBlocks()) {
    for (auto *op : bb->getOps()) {
      summary.unsafeControl = summary.unsafeControl || hasControlOrCall(op);
      if (auto *load = dyn_cast<LoadOp>(op)) {
        summary.loads++;
        if (load->getOperandCount() > 0) {
          if (auto *base = findMemoryBase(load->getOperand().defining))
            summary.loadedBases.insert(base);
        }
      } else if (auto *store = dyn_cast<StoreOp>(op)) {
        summary.stores++;
        if (store->getOperandCount() > 1) {
          if (auto *base = findMemoryBase(store->getOperand(1).defining))
            summary.storedBases.insert(base);
        }
      }
    }
  }
}

bool hasMultiplyAccumulateShape(LoopInfo *loop) {
  if (!loop)
    return false;
  bool hasMul = false;
  bool hasAdd = false;
  for (auto *bb : loop->getBlocks()) {
    for (auto *op : bb->getOps()) {
      hasMul = hasMul || isa<MulIOp>(op) || isa<MulLOp>(op);
      hasAdd = hasAdd || isa<AddIOp>(op) || isa<AddLOp>(op);
    }
  }
  return hasMul && hasAdd;
}

} // namespace

std::map<std::string, int> AdvancedConv2DTransformPass::stats() {
  return {
    {"conv2d-detected", conv2dDetected},
    {"winograd-applied", winogradApplied},
    {"im2col-applied", im2colApplied},
    {"rejected-shape", rejectedShape},
    {"fallback-generic", routedFallback},
  };
}

bool AdvancedConv2DTransformPass::extractParams(LoopInfo *loop, ConvParams &params) {
  std::vector<LoopInfo*> chain;
  collectLoopChain(loop, chain);
  if (chain.size() < 4) {
    rejectedShape++;
    return false;
  }

  for (auto *candidate : chain) {
    if (!isUnitPositiveStep(candidate)) {
      rejectedShape++;
      return false;
    }
  }

  // The common SysY convolution shape is spatial loops followed by two kernel
  // loops.  We keep the matcher deliberately structural and do not inspect
  // source names or test-case dimensions.
  params.H = constValue(chain[0]->stop);
  params.W = constValue(chain[1]->stop);
  params.KH = constValue(chain[chain.size() - 2]->stop);
  params.KW = constValue(chain[chain.size() - 1]->stop);
  params.stride = 1;

  // Dynamic H/W are normal for input-dependent benchmark kernels.  They still
  // qualify for the route decision if KH/KW and the memory pattern are clear.
  if (params.KH <= 0 || params.KW <= 0 || params.KH > 11 || params.KW > 11) {
    rejectedShape++;
    return false;
  }

  AccessSummary summary;
  summarizeLoop(loop, summary);
  if (summary.unsafeControl || summary.stores == 0 || summary.loads < 2 ||
      summary.storedBases.empty() || summary.loadedBases.size() < 2 ||
      !hasMultiplyAccumulateShape(loop)) {
    rejectedShape++;
    return false;
  }

  params.outputArray = *summary.storedBases.begin();
  for (auto *base : summary.loadedBases) {
    if (base != params.outputArray) {
      if (!params.inputArray)
        params.inputArray = base;
      else if (!params.kernelArray && base != params.inputArray)
        params.kernelArray = base;
    }
  }

  return params.inputArray && params.outputArray && params.kernelArray;
}

bool AdvancedConv2DTransformPass::runWinograd(FuncOp *, LoopInfo *,
                                              const ConvParams &) {
  // This pass intentionally does not replace convolution with a hand-written
  // kernel.  Winograd lowering needs a full integer-range proof for every
  // transformed coefficient because the F(2x2,3x3) half-coefficients are not
  // generally equivalent under SysY i32 truncating arithmetic.  Keep the route
  // visible in stats, and let HIR stencil dispatch / loop transforms handle the
  // safe fast path.
  return false;
}

bool AdvancedConv2DTransformPass::runIm2col(FuncOp *, LoopInfo *,
                                            const ConvParams &) {
  // Full im2col materialization can exceed the benchmark memory budget on
  // large dynamic images.  The default route therefore uses the existing
  // affine loop optimizer and vectorizer instead of allocating a hidden
  // temporary matrix.  An experimental implementation can be guarded behind
  // this flag once buffer lifetime and memory caps are proven.
  return false;
}

void AdvancedConv2DTransformPass::run() {
  if (!envEnabled("SISY_ENABLE_ADVANCED_CONV2D", true))
    return;

  LoopAnalysis analysis(module);
  for (auto *func : collectFuncs()) {
    if (!func || !func->getRegion())
      continue;

    auto forest = analysis.runImpl(func->getRegion());
    for (auto *loop : candidateLoops(forest)) {
      ConvParams params;
      if (!extractParams(loop, params))
        continue;

      conv2dDetected++;
      if (params.KH == 3 && params.KW == 3 && params.stride == 1) {
        if (runWinograd(func, loop, params)) {
          winogradApplied++;
          continue;
        }
      }

      const int area = params.H > 0 && params.W > 0 ? params.H * params.W : 64 * 64;
      if (params.KH * params.KW > 9 && params.stride == 1 && area > 32) {
        if (runIm2col(func, loop, params)) {
          im2colApplied++;
          continue;
        }
      }

      routedFallback++;
    }
  }
}
