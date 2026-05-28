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
  void run() override;
};

// Puts CallerAttr to each function.
class CallGraph : public Pass {
public:
  CallGraph(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "call-graph"; };
  std::map<std::string, int> stats() override { return {}; }
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
  void run() override;
};

// Mark functions that are called at most once.
class AtMostOnce : public Pass {
public:
  AtMostOnce(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "at-most-once"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

// Utility for clearing transient EqClassAttr metadata.
void removeEqClass(Region *region);

}

#endif
