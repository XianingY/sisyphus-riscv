#ifndef RV_PASSES_H
#define RV_PASSES_H

#include "../opt/Pass.h"
#include "RvAttrs.h"
#include "RvOps.h"
#include "../codegen/Ops.h"
#include "../codegen/Attrs.h"
#include "../codegen/CodeGen.h"

namespace sys {

namespace rv {

class Lower : public Pass {
public:
  Lower(ModuleOp *module): Pass(module) {}
  
  std::string name() override { return "rv-lower"; };
  std::map<std::string, int> stats() override { return {}; };
  void run() override;
};

class StrengthReduct : public Pass {
  int convertedTotal = 0;

  int runImpl();
public:
  StrengthReduct(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "strength-reduction"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class InstCombine : public Pass {
  int combined = 0;
public:
  InstCombine(ModuleOp *module): Pass(module) {}

  std::string name() override { return "rv-inst-combine"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

// Pre-RA list scheduler for RV ops.
// Reorders instructions within basic blocks to hide load/multiply latency
// while respecting register-pressure constraints.
class Schedule : public Pass {
  int reordered = 0;
  int criticalPathNodes = 0;
  int criticalPathMaxHeight = 0;
  int heightWeight = 3;

  void runImpl(BasicBlock *bb);
public:
  Schedule(ModuleOp *module): Pass(module) {}

  std::string name() override { return "rv-schedule"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class RegAlloc : public Pass {
  int spilled = 0;
  int convertedTotal = 0;
  int maxBlockHotness = 0;
  int liveRangeSplits = 0;
  bool fastMode;

  std::map<FuncOp*, std::set<Reg>> usedRegisters;
  std::map<std::string, FuncOp*> fnMap;

  void runImpl(Region *region, bool isLeaf);
  // Create both prologue and epilogue of a function.
  void proEpilogue(FuncOp *funcOp, bool isLeaf);
  int latePeephole(Op *funcOp);
  void tidyup(Region *region);
public:
  RegAlloc(ModuleOp *module, bool fastMode = false): Pass(module), fastMode(fastMode) {}

  std::string name() override { return "rv-regalloc"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

// Dumps the output.
class Dump : public Pass {
  std::string out;

  void dump(std::ostream &os);
public:
  Dump(ModuleOp *module, const std::string &out): Pass(module), out(out) {}

  std::string name() override { return "rv-dump"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

}

}

#endif
