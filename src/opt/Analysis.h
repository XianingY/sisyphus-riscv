#ifndef ANALYSIS_H
#define ANALYSIS_H

#include "Passes.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

namespace sys {

// Analysis pass.
// Detects whether a function is pure. If it isn't, give it an ImpureAttr.
class Pureness : public Pass {
  // Maps a function to all functions that it might call.
  std::map<FuncOp*, std::set<FuncOp*>> callGraph;

public:
  Pureness(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "pureness"; };
  std::map<std::string, int> stats() override { return {}; }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Puts CallerAttr to each function.
class CallGraph : public Pass {
public:
  CallGraph(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "call-graph"; };
  std::map<std::string, int> stats() override { return {}; }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Gives an AliasAttr to values, if they are addresses.
class Alias : public Pass {
  std::map<std::string, GlobalOp*> gMap;
  int iterationsTotal = 0;
  int maxIterationsHit = 0;
  int degradedToUnknown = 0;
  void runImpl(Region *region);
public:
  Alias(ModuleOp *module): Pass(module) {}

  std::string name() override { return "alias"; };
  std::map<std::string, int> stats() override {
    return {
      { "iterations", iterationsTotal },
      { "max-iterations-hit", maxIterationsHit },
      { "degraded-to-unknown", degradedToUnknown },
    };
  }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Canonicalize array/pointer address expressions into stable base+stride form.
// This prepares lower-risk wins for Alias/DSE/DLE/GVN without changing semantics.
class ArrayStrideAnalysis : public Pass {
  int rewritten = 0;

public:
  ArrayStrideAnalysis(ModuleOp *module): Pass(module) {}

  std::string name() override { return "array-stride-analysis"; }
  std::map<std::string, int> stats() override {
    return {
      { "rewritten", rewritten }
    };
  }
  void run() override;
};

// Integer range analysis.
class Range : public Pass {
  // The set of all loop headers in a function.
  // We should apply widening at these blocks, otherwise it would take forever to converge.
  std::set<BasicBlock*> headers;

  // Reorder the blocks so that they have a single exit.
  void postdom(Region *region);
  // Split a single operation into two for comparison branches.
  void split(Region *region);
  // Give RangeAttr to operations.
  void analyze(Region *region);
public:
  Range(ModuleOp *module): Pass(module) {}

  std::string name() override { return "range"; }
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

// Lightweight equality-class propagation.
// Attaches EqClassAttr to integer SSA values for analysis-driven folds.
class EqClass : public Pass {
  int classes = 0;

public:
  EqClass(ModuleOp *module): Pass(module) {}

  std::string name() override { return "eqclass"; }
  std::map<std::string, int> stats() override {
    return {
      { "classes", classes }
    };
  }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Computes a richer FunctionSummaryAttr (pure / readonly / norecurse +
// argRead/argWrite bit masks) on every FuncOp.  Conservative: a property is
// only set when proven.  Existing ImpureAttr is preserved; this pass never
// removes it.  Killable via SISY_ENABLE_FN_SUMMARY=0.
class FunctionSummary : public Pass {
  int pureCount = 0;
  int readonlyCount = 0;
  int norecurseCount = 0;
  int argReadCount = 0;
  int argWriteCount = 0;

public:
  FunctionSummary(ModuleOp *module): Pass(module) {}

  std::string name() override { return "function-summary"; }
  std::map<std::string, int> stats() override {
    return {
      { "pure", pureCount },
      { "readonly", readonlyCount },
      { "norecurse", norecurseCount },
      { "arg-read", argReadCount },
      { "arg-write", argWriteCount },
    };
  }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Optional thin-summary dumper.  When SISY_THIN_SUMMARY_DUMP=<path> is set,
// emits a structured text summary of each function (name, arity, ImpureAttr,
// FunctionSummaryAttr fields, direct callees, estimated op count) usable as
// the seed of a future cross-module thin-link pipeline (IPCP / IPDCE / inline
// ranking).  When the env var is unset the pass is a no-op and produces no
// IR or behavior change.
class ThinSummary : public Pass {
  int emittedFunctions = 0;
  int importedFunctions = 0;
  int linkedFunctions = 0;

public:
  ThinSummary(ModuleOp *module): Pass(module) {}

  std::string name() override { return "thin-summary"; }
  std::map<std::string, int> stats() override {
    return {
      { "emitted", emittedFunctions },
      { "imported", importedFunctions },
      { "linked", linkedFunctions },
    };
  }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Static block-frequency / branch-probability analysis.
//
// Computes a per-BasicBlock frequency using loop-nest depth as the primary
// signal (header back-edges detected via dominance).  When SISY_PROFILE is
// set to a readable file of `funcname blockindex freq` lines, those entries
// override the static estimate.  Stored in a singleton side-table to avoid
// modifying BasicBlock; future consumers (RegAlloc spill weight, scheduler,
// block layout) can call BlockFrequency::freqOf(bb).
//
// The pass never modifies IR.  Kill switch: SISY_ENABLE_BFI=0.
class BlockFrequency : public Pass {
  int hotBlocks = 0;
  int coldBlocks = 0;
  int profileEntries = 0;
  int maxLoopDepth = 0;

public:
  BlockFrequency(ModuleOp *module): Pass(module) {}

  std::string name() override { return "block-frequency"; }
  std::map<std::string, int> stats() override {
    return {
      { "hot-blocks", hotBlocks },
      { "cold-blocks", coldBlocks },
      { "profile-entries", profileEntries },
      { "max-loop-depth", maxLoopDepth },
    };
  }
  void run() override;
  PreservedAnalyses run(PassContext &context) override;

  // Returns a relative frequency for `bb`.  1.0 == straight-line entry-level
  // estimate.  Returns 1.0 when no analysis has been run for this block.
  static double freqOf(BasicBlock *bb);
  static bool isCold(BasicBlock *bb);
  static void clear();
};

// Mark functions that are called at most once.
class AtMostOnce : public Pass {
public:
  AtMostOnce(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "at-most-once"; };
  std::map<std::string, int> stats() override { return {}; }
  PreservedAnalyses run(PassContext &context) override;
  void run() override;
};

// Utility for clearing transient EqClassAttr metadata.
void removeEqClass(Region *region);

}

#endif
