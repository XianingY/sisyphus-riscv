#pragma once

#include "LoopPasses.h"
#include <unordered_map>

namespace sys {

struct LoopDependenceGraph {
  std::vector<Op*> operations;
  struct DependenceEdge {
    Op* source;
    Op* sink;
    int type;  // 0=Flow, 1=Anti, 2=Output
    int distance;
    bool loopCarried;
  };
  std::vector<DependenceEdge> edges;
  std::unordered_map<Op*, int> opIndex;
};

class LoopDependenceAnalyzer : public LoopPass {
public:
  int parallelizableLoops = 0;
  int loopCarriedDeps = 0;
  int flowDeps = 0;
  int antiDeps = 0;
  int outputDeps = 0;

  std::unordered_map<LoopInfo*, LoopDependenceGraph> loopDependenceGraphs;

  std::map<std::string, int> stats() override;
  Pass::Result run(Module& module) override;
};

}  // namespace sys
