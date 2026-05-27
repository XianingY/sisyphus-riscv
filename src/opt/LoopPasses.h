#ifndef LOOP_PASSES_H
#define LOOP_PASSES_H

#include "Pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

#include <unordered_set>
#include <set>

// The whole content of this file should be run after Mem2Reg.
namespace sys {

class MemorySSA;

// We will take the LLVM terminology:
//   _Header_ is the only block of loop entry;
//   _Preheader_ is the only block in header.preds;
//   _Latch_ is a block with backedge;
//   _Exiting Block_ is a block that jumps out of the loop. 
//   _Exit Block_ is a block that isn't in the loop but is a target of an exitING block.
//
// Note the difference of the last two terms.
class LoopInfo {
public:
  std::vector<LoopInfo*> subloops;
  std::set<BasicBlock*> exitings;
  std::set<BasicBlock*> exits;
  std::set<BasicBlock*> latches;
  std::set<BasicBlock*> bbs;

  BasicBlock *preheader = nullptr;
  BasicBlock *header;
  LoopInfo *parent = nullptr;
  // Induction variable. Though there might be multiple, we only preserve the first encountered.
  Op *induction = nullptr;
  Op *start = nullptr, *stop = nullptr;
  Op *step = nullptr;

  const auto &getBlocks() const { return bbs; }
  auto getParent() const { return parent; }
  auto getLatch() const { assert(latches.size() == 1); return *latches.begin(); }
  auto getExit() const { assert(exits.size() == 1); return *exits.begin(); }
  auto getInduction() const { return induction; }
  auto getStart() const { return start; }
  auto getStop() const { return stop; }
  auto getStepOp() const { return step; }
  int getStep() const { return V(step); }

  bool contains(const BasicBlock *bb) const { return bbs.count(const_cast<BasicBlock*>(bb)); }

  // Note that this relies on a previous dump of the parent op,
  // otherwise the numbers of blocks are meaningless.
  void dump(std::ostream &os = std::cerr);
};

inline std::ostream &operator<<(std::ostream &os, LoopInfo *info) {
  info->dump(os);
  return os;
}

// Each loop structure is a tree.
// Multiple loops become a forest.
class LoopForest {
  std::vector<LoopInfo*> loops;
  std::map<BasicBlock*, LoopInfo*> loopMap;

  friend class LoopAnalysis;
public:
  const auto &getLoops() const { return loops; }

  LoopInfo *getInfo(BasicBlock *header) { return loopMap[header]; }

  void dump(std::ostream &os = std::cerr);
};

class LoopAnalysis : public Pass {
  std::map<FuncOp*, LoopForest> info;

public:
  LoopAnalysis(ModuleOp *module): Pass(module) {}
  ~LoopAnalysis();

  std::string name() override { return "loop-analysis"; }
  std::map<std::string, int> stats() override { return {}; }
  LoopForest runImpl(Region *region);
  void run() override;
  void reset() { info = {}; }

  auto getResult() { return info; }
};

// Canonicalize loops. Ensures:
//   1) A single preheader;
//   2) In LCSSA, if it's constructed with `lcssa = true`.
class CanonicalizeLoop : public Pass {
  void canonicalize(LoopInfo *loop);
  void runImpl(Region *region, LoopForest forest);

  bool lcssa;
public:
  CanonicalizeLoop(ModuleOp *module, bool lcssa): Pass(module), lcssa(lcssa) {}

  std::string name() override { return "canonicalize-loop"; }
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

class LoopRotate : public Pass {
  int rotated = 0;
  std::set<BasicBlock*> skipHeaders;
  bool allowCanonicalizedHeaders = false;

  void runImpl(LoopInfo *info);
public:
  LoopRotate(ModuleOp *module, bool allowCanonicalizedHeaders = false):
    Pass(module), allowCanonicalizedHeaders(allowCanonicalizedHeaders) {}

  std::string name() override { return "loop-rotate"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class ConstLoopUnroll : public Pass {
  std::map<Op*, Op*> phiMap;
  std::map<Op*, Op*> exitlatch;
  int unrolled = 0;
  int factorUnrolled = 0;
  int rotatedForUnroll = 0;

  // Returns true if changed.
  bool runImpl(LoopInfo *info);
  // Returns the new latch.
  // Starts insertion after `bb`, and duplicate `info` a total of `unroll` times.
  // This only unrolls constant loops.
  BasicBlock *copyLoop(LoopInfo *info, BasicBlock *bb, int unroll);
  // Attempt factor unroll for non-constant trip-count loops.
  // When `loop` has a runtime trip-count and a small, side-effect-free body,
  // clone the loop body as a remainder loop and convert the original loop
  // into a "main loop" that iterates in steps of (factor * original_step).
  // The cloned loop handles the tail iterations (trip % factor).
  //
  // Returns true if the transformation was applied.
  bool tryFactorUnroll(LoopInfo *loop, int factor);
public:
  ConstLoopUnroll(ModuleOp *module): Pass(module) {}

  std::string name() override { return "const-loop-unroll"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class LoopUnswitch : public Pass {
  int unswitched = 0;
public:
  LoopUnswitch(ModuleOp *module): Pass(module) {}

  std::string name() override { return "loop-unswitch"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class SCEV : public Pass {
  int expanded = 0;

  // All addresses stored inside current loop.
  // We need this because we need to find variants as well,
  // but it's slightly different from what LICM requires.
  std::vector<Op*> stores;
  std::unordered_map<Op*, Op*> start;
  std::unordered_set<Op*> nochange;

  DomTree domtree;
  bool impure;

  void rewrite(BasicBlock *bb, LoopInfo *info);
  void runImpl(LoopInfo *info);
  void discardIv(LoopInfo *info);
  // Replace usages after the loop when possible.
  void replaceAfter(LoopInfo *info);
public:
  SCEV(ModuleOp *module): Pass(module) {}

  std::string name() override { return "scev"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class LICM : public Pass {
  int hoisted = 0;
  DomTree domtree;
  // All addresses stored inside current loop.
  std::vector<Op*> stores;
  // Whether the current function has an impure call.
  bool impure;
  MemorySSA *mssa = nullptr;

  // A store is hoistable when no branch or load has been met.
  void hoistVariant(LoopInfo *info, BasicBlock *bb, bool hoistable, bool speculativeLoads);
  void markVariant(LoopInfo *info, BasicBlock *bb, bool hoistable);
  void runImpl(LoopInfo *info);
  bool hoistSubloop(LoopInfo *outer);

  // Find out all stores in the loop and update `stores`.
  // `storeHoistable` is false for non-rotated loops where stores may not run.
  bool updateStores(LoopInfo *info, bool *storeHoistable = nullptr);
public:
  LICM(ModuleOp *module): Pass(module) {}

  std::string name() override { return "licm"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class LoopInterchange : public Pass {
  int detected = 0;
  int interchanged = 0;

  // Check if the loop nest is a perfect nest (inner is only content)
  bool isPerfectNest(LoopInfo* outer, LoopInfo* inner);
  // Check if two loops can be safely interchanged
  bool canInterchange(LoopInfo* outer, LoopInfo* inner);
  void runImpl(LoopInfo* info);
public:
  LoopInterchange(ModuleOp *module): Pass(module) {}

  std::string name() override { return "loop-interchange"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Loop tiling (strip-mining) for cache locality improvement.
// Wraps a loop with a tile-level outer loop iterating in steps of T.
class LoopTiling : public Pass {
  int candidates = 0;
  int tiled = 0;
  int rejectedShape = 0;
  int rejectedProfit = 0;
public:
  LoopTiling(ModuleOp *module): Pass(module) {}

  std::string name() override { return "loop-tiling"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Reduction-aware loop interchange.
// Transforms i-j-k loops with a reduction in j (storing temp to A[g(j)])
// into i-k-j by distributing the reduction.
class ReductionInterchange : public Pass {
  int candidates = 0;
  int interchanged = 0;
  int rejectedShape = 0;
public:
  ReductionInterchange(ModuleOp *module): Pass(module) {}

  std::string name() override { return "reduction-interchange"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Promotes loop-invariant load/store patterns to scalar registers.
// For loops where the same address is repeatedly loaded, computed on, and stored,
// this pass hoists the load to the preheader, replaces in-loop accesses with a phi,
// and sinks the store to the exit block.
class ScalarReplace : public Pass {
  int detected = 0;
  int promoted = 0;
  int arraysScalarized = 0;
  int arrayElements = 0;
  int arrayAccesses = 0;

  void runImpl(LoopInfo *info);
  void runArrayScalarization(FuncOp *func);
public:
  ScalarReplace(ModuleOp *module): Pass(module) {}

  std::string name() override { return "scalar-replace"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class RemoveEmptyLoop : public Pass {
  int removed = 0;

  bool runImpl(LoopInfo *info);
public:
  RemoveEmptyLoop(ModuleOp *module): Pass(module) {}

  std::string name() override { return "remove-empty-loop"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class RepeatInvariantReduction : public Pass {
  int visited = 0;
  int reduced = 0;
  int rejected = 0;
  int badShape = 0;
  int badCfg = 0;
  int noShape = 0;
  int impureBody = 0;
  int badInductionUse = 0;

  bool runImpl(LoopInfo *info);
public:
  RepeatInvariantReduction(ModuleOp *module): Pass(module) {}

  std::string name() override { return "repeat-invariant-reduction"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Collapses bounded affine modular update loops such as:
//   while (i < n) {
//     *p = (*p + c) % m;
//     i++;
//   }
// into a single update of *p. This is intentionally narrow because it
// rewrites loops with stores.
class ModularAffineLoop : public Pass {
  int visited = 0;
  int folded = 0;
  int rejected = 0;

  bool runImpl(LoopInfo *info);
public:
  ModularAffineLoop(ModuleOp *module): Pass(module) {}

  std::string name() override { return "modular-affine-loop"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class DivPow2LoopFold : public Pass {
  int visited = 0;
  int folded = 0;
  int rejected = 0;

  bool runImpl(LoopInfo *info);
public:
  DivPow2LoopFold(ModuleOp *module): Pass(module) {}

  std::string name() override { return "div-pow2-loop-fold"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Collapses pure repeat-doubling loops such as:
//   while (n != 0) {
//     x = x * 2;
//     n = n - 1;
//   }
// into x * ((0 <= n && n < 32) ? (1 << n) : 0). This targets lowered
// tail-recursive array helpers without changing loops that have side effects.
class Pow2RepeatLoopFold : public Pass {
  int visited = 0;
  int folded = 0;
  int rejected = 0;
  int badCfg = 0;
  int badBranch = 0;
  int badCounter = 0;
  int badValue = 0;
  int impureBody = 0;

  bool runImpl(LoopInfo *info);
public:
  Pow2RepeatLoopFold(ModuleOp *module): Pass(module) {}

  std::string name() override { return "pow2-repeat-loop-fold"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class RotlRepeatLoopFold : public Pass {
  int visited = 0;
  int folded = 0;
  int rejected = 0;

  bool runImpl(FuncOp *func, LoopInfo *info);
public:
  RotlRepeatLoopFold(ModuleOp *module): Pass(module) {}

  std::string name() override { return "rotl-repeat-loop-fold"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class Vectorize : public Pass {
  std::unordered_map<Op*, Op*> base;
  
  Op *findBase(Op *op);
  void runImpl(LoopInfo *info);
public:
  Vectorize(ModuleOp *module): Pass(module) {}

  std::string name() override { return "vectorize"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class Splice : public Pass {
  void runImpl(LoopInfo *loop);
public:
  Splice(ModuleOp *module): Pass(module) {}

  std::string name() override { return "splice"; }
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

class PrivatizeReduction : public Pass {
  int privatized = 0;
  void runImpl(LoopInfo *info);
public:
  PrivatizeReduction(ModuleOp *module): Pass(module) {}

  std::string name() override { return "privatize-reduction"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class BoundsCheck : public Pass {
  int branchesEliminated = 0;
  int boundsProved = 0;
  int boundsRejected = 0;

  void runImpl(LoopInfo *info);
public:
  BoundsCheck(ModuleOp *module): Pass(module) {}

  std::string name() override { return "bounds-check"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

}

#endif
