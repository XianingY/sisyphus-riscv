#ifndef ARM_PASSES_H
#define ARM_PASSES_H

#include "../opt/Pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Ops.h"
#include "../codegen/Attrs.h"
#include "ArmOps.h"
#include "ArmAttrs.h"

namespace sys::arm {

class Lower : public Pass {
public:
  Lower(ModuleOp *module): Pass(module) {}
  
  std::string name() override { return "arm-lower"; };
  std::map<std::string, int> stats() override { return {}; };
  void run() override;
};

class StrengthReduct : public Pass {
  int convertedTotal = 0;

  int runImpl();
public:
  StrengthReduct(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "arm-strength-reduct"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class InstCombine : public Pass {
  int combined = 0;
  int maxRounds;
public:
  InstCombine(ModuleOp *module, int maxRounds = -1): Pass(module), maxRounds(maxRounds) {}

  std::string name() override { return "arm-inst-combine"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

// The only difference with opt/DCE is that `isImpure` behaves differently.
class ArmDCE : public Pass {
  std::vector<Op*> removeable;
  int elim = 0;

  bool isImpure(Op *op);
  void markImpure(Region *region);
  void runOnRegion(Region *region);
public:
  ArmDCE(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "arm-dce"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class RegAlloc : public Pass {
  int spilled = 0;
  int convertedTotal = 0;
  int maxBlockHotness = 0;
  int liveRangeSplits = 0;
  bool fastMode;
  int callPenalty;
  int loopHotBoost;
  int preferBudget;
  int peepholeRounds;

  std::map<FuncOp*, std::set<Reg>> usedRegisters;
  std::map<std::string, FuncOp*> fnMap;

  void runImpl(Region *region, bool isLeaf);
  void proEpilogue(FuncOp *funcOp, bool isLeaf);
  int latePeephole(Op *funcOp);
  void tidyup(Region *region);
public:

  RegAlloc(ModuleOp *module,
           bool fastMode = false,
           int callPenalty = 128,
           int loopHotBoost = 1,
           int preferBudget = -1,
           int peepholeRounds = -1):
    Pass(module),
    fastMode(fastMode),
    callPenalty(callPenalty),
    loopHotBoost(loopHotBoost),
    preferBudget(preferBudget),
    peepholeRounds(peepholeRounds) {}

  std::string name() override { return "arm-regalloc"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

// Pre-RA list scheduler for ARM ops.
// Reorders instructions within basic blocks to hide load/multiply/divide
// latency on Cortex-A class cores. Mirrors the structure of sys::rv::Schedule
// but uses an ARM-specific latency table and op classification.
class Schedule : public Pass {
  int reordered = 0;
  int criticalPathNodes = 0;
  int criticalPathMaxHeight = 0;
  int heightWeight = 3;

  void runImpl(BasicBlock *bb);
public:
  Schedule(ModuleOp *module): Pass(module) {}

  std::string name() override { return "arm-schedule"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class LateLegalize : public Pass {
public:
  LateLegalize(ModuleOp *module): Pass(module) {}

  std::string name() override { return "arm-late-legalize"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

// Dumps the output.
class Dump : public Pass {
  std::string out;

  void dump(std::ostream &os);
  void dumpBody(Region *region, std::ostream &os);
  void dumpOp(Op *op, std::ostream &os);
public:
  Dump(ModuleOp *module, const std::string &out): Pass(module), out(out) {}

  std::string name() override { return "arm-dump"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

};

#endif
