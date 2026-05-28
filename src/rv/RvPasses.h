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

class Legalize : public Pass {
  int legal = 0;
  int promoted = 0;
  int split = 0;
  int expanded = 0;
  int custom = 0;
  int illegal = 0;
  int verifierErrors = 0;
public:
  Legalize(ModuleOp *module): Pass(module) {}

  std::string name() override { return "rv-legalize"; }
  std::map<std::string, int> stats() override;
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
  int rematerialized = 0;
  int spillLoads = 0;
  int spillStores = 0;
  int dynamicSplits = 0;
  int coalescedCopies = 0;
  int splitBailouts = 0;
  int spillAfterSplit = 0;
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

// Pressure-aware superblock scheduling v1.
//
// Hoists pure / memory-load operations from a successor block B into its
// single predecessor A across hot edges, when:
//   - A's only successor is B and B's only predecessor is A (single-entry
//     superblock fragment), reducing aliasing concerns to one block,
//   - the candidate's defining operands all dominate the tail of A,
//   - no may-alias store, call, or other pinned op sits between the candidate
//     and A's terminator,
//   - estimated live-out pressure of A stays within the budget.
//
// Stats expose how many candidates were considered and rejected for each
// reason.  Disabled by default; enable via SISY_RV_ENABLE_SUPERBLOCK=1.
class SuperblockSchedule : public Pass {
  int hoisted = 0;
  int rejectedShape = 0;
  int rejectedAlias = 0;
  int rejectedPressure = 0;
  int rejectedDeps = 0;
  int candidates = 0;

public:
  SuperblockSchedule(ModuleOp *module): Pass(module) {}

  std::string name() override { return "rv-superblock-schedule"; }
  std::map<std::string, int> stats() override {
    return {
      { "hoisted", hoisted },
      { "candidates", candidates },
      { "rejected-shape", rejectedShape },
      { "rejected-alias", rejectedAlias },
      { "rejected-pressure", rejectedPressure },
      { "rejected-deps", rejectedDeps },
    };
  }
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
