#pragma once

#include "LoopPasses.h"

namespace sys {

class SLPVectorizeEnhanced : public LoopPass {
public:
  int verticalPacks = 0;
  int contiguousPacks = 0;
  int maskedPacks = 0;
  int rejected = 0;

  std::map<std::string, int> stats() override;
  Pass::Result run(Module& module) override;

private:
  void runImpl(LoopInfo* info);
};

}  // namespace sys
