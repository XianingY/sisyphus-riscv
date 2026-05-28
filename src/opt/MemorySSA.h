#ifndef SISY_MEMORY_SSA_H
#define SISY_MEMORY_SSA_H

#include "../codegen/Ops.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <vector>

namespace sys {

// Memory access node type
enum class MemoryAccessKind {
  Use,  // Corresponds to LoadOp or Pure CallOp
  Def,  // Corresponds to StoreOp or Impure CallOp
  Phi   // Virtual Memory Phi node at basic block boundaries
};

struct MemoryAccess {
  MemoryAccessKind kind;
  Op *op = nullptr;               // Associated original IR Op (nullptr for Phi)
  BasicBlock *bb = nullptr;       // Block where it resides
  MemoryAccess *definingAccess = nullptr; // Preceding defining MemoryDef or MemoryPhi
  std::vector<MemoryAccess*> phiIncoming; // For Phi nodes, all incoming definitions from predecessors
};

class MemorySSA {
private:
  Region *region;
  std::vector<std::unique_ptr<MemoryAccess>> accessPool;
  std::unordered_map<Op*, MemoryAccess*> opToAccess;
  std::unordered_map<BasicBlock*, MemoryAccess*> blockMemoryPhi;

  // Stores the latest MemoryDef/MemoryPhi when exiting a basic block
  std::unordered_map<BasicBlock*, MemoryAccess*> blockExitDefs;
  bool dirty = false;
  std::string dirtyReason;

public:
  MemorySSA(Region *r) : region(r) {}
  
  void build();
  bool isDirty() const { return dirty; }
  void markDirty(const std::string &reason = "unknown");
  MemoryAccess *getAccess(Op *op);
  MemoryAccess *getMemoryPhi(BasicBlock *bb);
  
  // Reaching Def API: search recursively for the physical defining Def (Store/Def)
  MemoryAccess *getReachingDef(MemoryAccess *useAccess);
  std::vector<MemoryAccess*> getClobberingDefs(Op *memoryOp);

  void eraseMemoryOp(Op *op);
  void insertMemoryUse(Op *op);
  void insertMemoryDef(Op *op);
  void moveMemoryOpBefore(Op *op, Op *before);
  void replaceMemoryOp(Op *oldOp, Op *replacement);

private:
  MemoryAccess *createAccess(MemoryAccessKind kind, Op *op, BasicBlock *bb);
};

} // namespace sys

#endif // SISY_MEMORY_SSA_H
