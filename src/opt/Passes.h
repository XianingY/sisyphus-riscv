#ifndef PASSES_H
#define PASSES_H

#include "Pass.h"
#include "../codegen/CodeGen.h"
#include "../codegen/Attrs.h"

#include <set>

namespace sys {

// Converts alloca's to SSA values.
// This must run on flattened CFG, otherwise `break` and `continue` are hard to deal with.
class Mem2Reg : public Pass {
  int count = 0;  // Total converted count
  int missed = 0; // Unconvertible alloca's

  // Maps AllocaOp* to Value (the real value of this alloca).
  using SymbolTable = std::map<Op*, Value>;

  void runImpl(FuncOp *func);
  void fillPhi(BasicBlock *bb, SymbolTable symbols);
  
  // Maps phi to alloca.
  std::map<Op*, Op*> phiFrom;
  std::set<BasicBlock*> visited;
  // Allocas we're going to convert in the pass.
  std::set<Op*> converted;
  DomTree domtree;
public:
  Mem2Reg(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "mem2reg"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

// Global value numbering.
class GVN : public Pass {
  int elim = 0;

  using SymbolTable = std::unordered_map<Op*, int>;
  using Domtree = std::map<BasicBlock*, std::vector<BasicBlock*>>;

  // The number of each Op.
  SymbolTable symbols;

  struct Expr {
    int id;
    std::vector<int> operands;

    // Attributes
    int vi = 0;
    float vf = 0;
    std::string name;

    bool operator<(const Expr &other) const;
  };
  std::map<Expr, int> exprNum;
  std::map<int, Op*> numOp;
  // The current number.
  int num = 1;

  class SemanticScope {
    GVN &pass;
    SymbolTable symbols;
    std::map<Expr, int> exprNum;
    std::map<int, Op*> numOp;
  public:
    SemanticScope(GVN &pass):
      pass(pass), symbols(pass.symbols), exprNum(pass.exprNum), numOp(pass.numOp) {}
    ~SemanticScope() {
      pass.symbols = symbols;
      pass.exprNum = exprNum;
      pass.numOp = numOp;
    }
  };

  // Dominator-based Value Numbering Technique. See Briggs.
  void dvnt(BasicBlock *bb, Domtree &domtree);
public:
  GVN(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "gvn"; };
  std::map<std::string, int> stats() override;
  void run() override;
  void runImpl(Region *region);
};

class Inline : public Pass {
  int inlined = 0;

  // Do not inline functions with Op count > `threshold`.
  int threshold;
  std::map<std::string, FuncOp*> fnMap;
public:
  Inline(ModuleOp *module, int threshold): Pass(module), threshold(threshold) {}
    
  std::string name() override { return "inline"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class ConstArgSpecialize : public Pass {
  int cloned = 0;
  int retargeted = 0;
  int skippedBudget = 0;

public:
  ConstArgSpecialize(ModuleOp *module): Pass(module) {}

  std::string name() override { return "const-arg-specialize"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class DeadGlobalStore : public Pass {
  int removed = 0;
  int deadGlobals = 0;
public:
  DeadGlobalStore(ModuleOp *module): Pass(module) {}

  std::string name() override { return "dead-global-store"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class RuntimeMemoize : public Pass {
  int candidates = 0;
  int memoized = 0;
  int entryChecks = 0;
  int callEpochBumps = 0;
  int rejectedImpure = 0;
  int rejectedArgShape = 0;
  int rejectedTypes = 0;
  int rejectedCalls = 0;
  int rejectedStores = 0;
  int rejectedOps = 0;
  int rejectedReturn = 0;

public:
  RuntimeMemoize(ModuleOp *module): Pass(module) {}

  std::string name() override { return "runtime-memoize"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class RowScratchMatmul : public Pass {
  int candidates = 0;
  int replaced = 0;
  int rejectedShape = 0;
public:
  RowScratchMatmul(ModuleOp *module): Pass(module) {}

  std::string name() override { return "row-scratch-matmul"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class ZeroFactorStoreLoop : public Pass {
  int candidates = 0;
  int replaced = 0;
  int affineReplaced = 0;
  int rejectedShape = 0;
public:
  ZeroFactorStoreLoop(ModuleOp *module): Pass(module) {}

  std::string name() override { return "zero-factor-store-loop"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Globalizes local arrays.
class Globalize : public Pass {
  void runImpl(Region *region);
public:
  Globalize(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "globalize"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

// Global code motion.
class LoopInfo;
class LoopForest;
class GCM : public Pass {
  std::set<Op*> visited;
  DomTree tree;
  // This is the depth on dominator tree.
  std::map<BasicBlock*, int> depth;
  // This is the depth in loop forest.
  std::map<BasicBlock*, int> loopDepth;
  std::map<Op*, BasicBlock*> scheduled;

  void updateDepth(BasicBlock *bb, int dep);
  void updateLoopDepth(LoopInfo *info, int dep);

  void scheduleEarly(BasicBlock *entry, Op *op);
  void scheduleLate(Op *op);

  // Lowest common ancestor.
  BasicBlock *lca(BasicBlock *a, BasicBlock *b);

  void runImpl(Region *region, const LoopForest &forest);
public:
  GCM(ModuleOp *module): Pass(module) {}

  std::string name() override { return "gcm"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

// Folds a wide range of expressions.
class RegularFold : public Pass {
  int foldedTotal = 0;
  int currentRound = 0;

  int runImpl(Region *region);
public:
  RegularFold(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "regular-fold"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class FunctionEquivalence : public Pass {
  int classified = 0;
  int replaced = 0;
  bool allowModMul;
public:
  FunctionEquivalence(ModuleOp *module, bool allowModMul = false):
    Pass(module), allowModMul(allowModMul) {}

  std::string name() override { return "function-equivalence"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class StructuralModMul : public Pass {
  int classified = 0;
  int replaced = 0;
  int guarded = 0;

public:
  StructuralModMul(ModuleOp *module): Pass(module) {}

  std::string name() override { return "structural-modmul"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

class StructuralBitwise : public Pass {
  int classified = 0;
  int replaced = 0;
  int guarded = 0;

public:
  StructuralBitwise(ModuleOp *module): Pass(module) {}

  std::string name() override { return "structural-bitwise"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Convert short parity-controlled accumulator branches into SelectOp. This is
// deliberately narrower than the generic Select pass because it may hoist
// loads out of the taken block.
class ParityIfConversion : public Pass {
  int converted = 0;
  int rejected = 0;
public:
  ParityIfConversion(ModuleOp *module): Pass(module) {}

  std::string name() override { return "parity-if-conversion"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class LateInline : public Pass {
  int inlined = 0;
  int threshold;
public:
  LateInline(ModuleOp *module, int threshold): Pass(module), threshold(threshold) {}
    
  std::string name() override { return "late-inline"; };
  std::map<std::string, int> stats() override;
  void run() override;
};

class HoistConstArray : public Pass {
  int hoisted = 0;
  
  void attemptHoist(Op *op);
public:
  HoistConstArray(ModuleOp *module): Pass(module) {}

  std::string name() override { return "hoist-const-array"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Inline constant stores to globals.
class InlineStore : public Pass {
  int inlined = 0;
  
  void attemptHoist(Op *op);
public:
  InlineStore(ModuleOp *module): Pass(module) {}

  std::string name() override { return "inline-store"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Recognize `if-else` patterns and raise to SelectOp.
class Select : public Pass {
  int raised = 0;
public:
  Select(ModuleOp *module): Pass(module) {}

  std::string name() override { return "select"; }
  std::map<std::string, int> stats() override;
  void run() override;
};

// Specialize functions according to argument signs.
class Specialize : public Pass {
  bool specialize();
public:
  Specialize(ModuleOp *module): Pass(module) {}

  std::string name() override { return "specialize"; }
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

class Verify : public Pass {
public:
  Verify(ModuleOp *module): Pass(module) {}
    
  std::string name() override { return "verify"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

// Precompute some values for recursive functions.
class Cached : public Pass {
public:
  Cached(ModuleOp *module): Pass(module) {}
  
  std::string name() override { return "cached"; };
  std::map<std::string, int> stats() override { return {}; }
  void run() override;
};

}

#endif
