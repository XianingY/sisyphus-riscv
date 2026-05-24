#include "LoopPasses.h"
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstdlib>

using namespace sys;

namespace {

// ===========================================================================
// Software Pipelining: Modulo Scheduling for Fixed-Trip Loops
// ===========================================================================

struct ScheduleStage {
  std::vector<Op*> prologue;
  std::vector<Op*> kernel;
  std::vector<Op*> epilogue;
};

class SoftwarePipelineScheduler {
public:
  explicit SoftwarePipelineScheduler(LoopInfo* loop) : loop(loop), stages(0) {}

  // Check if loop is suitable for pipelining
  bool isCandidateLoop() {
    if (!loop || !loop->header)
      return false;

    // Must have predictable trip count
    auto stop = loop->getStop();
    if (!stop || !isa<IntOp>(stop))
      return false;

    // Should not have function calls
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        if (isa<CallOp>(op))
          return false;
      }
    }

    // Should have clear recurrence pattern
    return hasRecurrencePattern();
  }

  // Generate pipelined schedule
  ScheduleStage generateSchedule(int initiation_interval = 2) {
    ScheduleStage result;

    if (!isCandidateLoop())
      return result;

    std::vector<Op*> loopOps;
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        loopOps.push_back(op);
      }
    }

    // Stage 1: Prologue (executes once before main kernel)
    for (size_t i = 0; i < std::min((size_t)initiation_interval, loopOps.size()); i++) {
      result.prologue.push_back(loopOps[i]);
    }

    // Stage 2: Kernel (steady state, repeats every initiation_interval cycles)
    // In steady state, we typically execute ops from multiple iterations in flight
    stages = initiation_interval;
    int kernelSize = std::min((int)loopOps.size(), initiation_interval);
    for (int i = 0; i < kernelSize; i++) {
      result.kernel.push_back(loopOps[i % loopOps.size()]);
    }

    // Stage 3: Epilogue (completes remaining iterations)
    // Number of epilogue iterations = (trip_count - 1) mod initiation_interval
    int remainingOps = loopOps.size() % initiation_interval;
    if (remainingOps != 0) {
      for (int i = 0; i < remainingOps; i++) {
        result.epilogue.push_back(loopOps[i]);
      }
    }

    return result;
  }

  int getStages() const { return stages; }

private:
  LoopInfo* loop;
  int stages;

  bool hasRecurrencePattern() {
    // Check if loop body contains accumulation or similar pattern
    for (auto bb : loop->getBlocks()) {
      for (auto op : bb->getOps()) {
        // Accumulation: op = op + expr
        if (isa<AddIOp>(op) || isa<AddFOp>(op) ||
            isa<MulIOp>(op) || isa<MulFOp>(op)) {
          return true;
        }
      }
    }
    return false;
  }
};

}  // namespace

// Export for other passes
bool isSoftwarePipelineCandidate(LoopInfo* loop) {
  SoftwarePipelineScheduler scheduler(loop);
  return scheduler.isCandidateLoop();
}

