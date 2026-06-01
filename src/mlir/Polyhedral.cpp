#include "Polyhedral.h"
#include <iostream>
#include <set>
#include <string>

namespace sys::mlir {

static std::string stringAttr(Attribute attr) {
  if (!attr)
    return "";
  std::string text = attr.str();
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
    return text.substr(1, text.size() - 2);
  return text;
}

// Helper to find the original alloca for a loaded value, navigating back through basic block ops.
static Operation* findAllocaForLoad(Operation *loadOp) {
  if (loadOp->name() != "sysy.load" && loadOp->name() != "memref.load") return nullptr;
  if (loadOp->operandCount() == 0) return nullptr;
  Operation *allocOp = loadOp->operand(0).getDefiningOp();
  if (allocOp && (allocOp->name() == "sysy.alloca" || allocOp->name() == "memref.alloca")) {
    return allocOp;
  }
  return nullptr;
}

static void dropRegionOperandUses(Region &region) {
  for (auto &block : region.getBlocks()) {
    for (auto &owned : block->ops()) {
      owned->dropAllOperands();
      for (auto &nested : owned->getRegions())
        dropRegionOperandUses(*nested);
    }
  }
}

static bool blockInRegionTree(Block *block, Region *region) {
  if (!block || !region)
    return false;
  for (auto &childBlock : region->getBlocks()) {
    if (childBlock.get() == block)
      return true;
    for (auto &op : childBlock->ops()) {
      for (auto &nested : op->getRegions()) {
        if (blockInRegionTree(block, nested.get()))
          return true;
      }
    }
  }
  return false;
}

static bool valueDefinedInRegionTree(Value value, Region *region) {
  if (!value.valid())
    return false;
  if (value.isOperationResult())
    return blockInRegionTree(value.getDefiningOp()->getBlock(), region);
  if (auto *arg = value.getBlockArgument())
    return blockInRegionTree(arg->getOwner(), region);
  return false;
}

static bool opOperandsDependOnRegion(Operation *op, Region *region) {
  if (!op)
    return false;
  for (auto value : op->getOperands()) {
    if (valueDefinedInRegionTree(value, region))
      return true;
  }
  return false;
}

static bool isLoadFromAlloca(Operation *op, Operation *allocaOp) {
  return op && (op->name() == "sysy.load" || op->name() == "memref.load") &&
         op->operandCount() > 0 && op->operand(0).getDefiningOp() == allocaOp;
}

static bool isStoreToAlloca(Operation *op, Operation *allocaOp) {
  return op && op->name() == "sysy.store" && op->operandCount() >= 2 &&
         op->operand(1).getDefiningOp() == allocaOp;
}

static bool regionHasStoreToAlloca(Region *region, Operation *allocaOp,
                                   Operation *exceptStore) {
  if (!region)
    return false;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (op != exceptStore && isStoreToAlloca(op, allocaOp))
        return true;
      for (auto &nested : op->getRegions()) {
        if (regionHasStoreToAlloca(nested.get(), allocaOp, exceptStore))
          return true;
      }
    }
  }
  return false;
}

static bool regionHasCallTo(Region *region, const std::string &callee) {
  if (!region || callee.empty())
    return false;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (op->name() == "sysy.call" && stringAttr(op->attr("callee")) == callee)
        return true;
      for (auto &nested : op->getRegions()) {
        if (regionHasCallTo(nested.get(), callee))
          return true;
      }
    }
  }
  return false;
}

static bool regionHasAnyCall(Region *region) {
  if (!region)
    return false;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (op->name() == "sysy.call")
        return true;
      for (auto &nested : op->getRegions()) {
        if (regionHasAnyCall(nested.get()))
          return true;
      }
    }
  }
  return false;
}

static bool regionHasLoopControl(Region *region) {
  if (!region)
    return false;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (op->name() == "sysy.break" || op->name() == "sysy.continue" ||
          op->name() == "sysy.return")
        return true;
      for (auto &nested : op->getRegions()) {
        if (regionHasLoopControl(nested.get()))
          return true;
      }
    }
  }
  return false;
}

static Operation *enclosingFunction(Operation *op) {
  if (!op)
    return nullptr;
  Block *block = op->getBlock();
  while (block) {
    Region *region = block->getRegion();
    if (!region)
      break;
    Operation *parent = region->getParent();
    if (!parent)
      break;
    if (parent->name() == "sysy.func")
      return parent;
    block = parent->getBlock();
  }
  return nullptr;
}

static bool opTreeHasLoadFromAlloca(Operation *op, Operation *allocaOp) {
  if (!op || op->isErased())
    return false;
  if (isLoadFromAlloca(op, allocaOp))
    return true;
  for (auto &nested : op->getRegions()) {
    for (auto &block : nested->getBlocks()) {
      for (auto &owned : block->ops()) {
        if (opTreeHasLoadFromAlloca(owned.get(), allocaOp))
          return true;
      }
    }
  }
  return false;
}

static bool opTreeHasStoreToAlloca(Operation *op, Operation *allocaOp) {
  if (!op || !allocaOp || op->isErased())
    return false;
  if (isStoreToAlloca(op, allocaOp))
    return true;
  for (auto &nested : op->getRegions()) {
    if (regionHasStoreToAlloca(nested.get(), allocaOp, nullptr))
      return true;
  }
  return false;
}

static bool bodyReadsIVAfterStepStore(Block *body, Operation *allocaOp,
                                      Operation *ivStore) {
  if (!body)
    return false;
  bool pastStepStore = false;
  for (auto &owned : body->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op == ivStore) {
      pastStepStore = true;
      continue;
    }
    if (pastStepStore && opTreeHasLoadFromAlloca(op, allocaOp))
      return true;
  }
  return false;
}

static bool blockStoresAllocaBetween(Block *block, Operation *begin,
                                     Operation *end, Operation *allocaOp) {
  if (!block || !begin || !end || !allocaOp)
    return false;
  bool inRange = false;
  for (auto &owned : block->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op == begin) {
      inRange = true;
      continue;
    }
    if (op == end)
      return false;
    if (inRange && opTreeHasStoreToAlloca(op, allocaOp))
      return true;
  }
  return false;
}

static bool blockReadsAllocaAfter(Block *block, Operation *anchor, Operation *allocaOp) {
  if (!block || !anchor || !allocaOp)
    return false;
  bool pastAnchor = false;
  for (auto &owned : block->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op == anchor) {
      pastAnchor = true;
      continue;
    }
    if (pastAnchor && opTreeHasLoadFromAlloca(op, allocaOp))
      return true;
  }
  return false;
}

static void replaceLoadsFromAllocaInRegion(Module &module, Region *region,
                                           Operation *allocaOp,
                                           Value replacement) {
  if (!region)
    return;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (isLoadFromAlloca(op, allocaOp)) {
        replaceAllUses(module, op->result(), replacement);
        op->markErased();
        continue;
      }
      for (auto &nested : op->getRegions())
        replaceLoadsFromAllocaInRegion(module, nested.get(), allocaOp,
                                       replacement);
    }
  }
}

static bool tryRaiseWhileToAffine(Module &module, Operation *op) {
  if (!op || op->isErased() || op->name() != "scf.while")
    return false;
  if (op->getRegions().size() < 2)
    return false;
  if (op->getRegions()[0]->getBlocks().empty() || op->getRegions()[1]->getBlocks().empty())
    return false;

  Block *condBlock = op->getRegions()[0]->getBlocks()[0].get();
  Block *body = op->getRegions()[1]->getBlocks()[0].get();

  if (condBlock->ops().empty() || condBlock->ops().back()->name() != "scf.condition")
    return false;
  Operation *condOp = condBlock->ops().back().get();
  if (condOp->operandCount() == 0)
    return false;

  Operation *cmpOp = condOp->operand(0).getDefiningOp();
  if (!cmpOp)
    return false;
  if (cmpOp->name() != "arith.cmpi" && cmpOp->name() != "rv_machine.cmp" &&
      cmpOp->name() != "arm_machine.cmp")
    return false;
  if (stringAttr(cmpOp->attr("predicate")) != "lt")
    return false;

  Operation *ivLoad = cmpOp->operand(0).getDefiningOp();
  Operation *boundOp = cmpOp->operand(1).getDefiningOp();
  if (!ivLoad || !boundOp)
    return false;

  Operation *allocaOp = findAllocaForLoad(ivLoad);
  if (!allocaOp)
    return false;

  Operation *stepOp = nullptr;
  Operation *ivStore = nullptr;
  for (auto &owned : body->ops()) {
    auto *child = owned.get();
    if (child->name() == "sysy.store" && child->operandCount() >= 2 &&
        child->operand(1).getDefiningOp() == allocaOp) {
      ivStore = child;
      stepOp = child->operand(0).getDefiningOp();
    }
  }

  if (!stepOp || !ivStore)
    return false;
  if (stepOp->name() != "arith.addi" && stepOp->name() != "rv_machine.addw" &&
      stepOp->name() != "arm_machine.add")
    return false;

  Operation *startStore = nullptr;
  Block *parentBlock = op->getBlock();
  if (!parentBlock)
    return false;
  for (auto &sibling : parentBlock->ops()) {
    if (sibling.get() == op)
      break;
    if (sibling->name() == "sysy.store" && sibling->operandCount() >= 2 &&
        sibling->operand(1).getDefiningOp() == allocaOp) {
      startStore = sibling.get();
    }
  }
  if (!startStore)
    return false;
  if (blockStoresAllocaBetween(parentBlock, startStore, op, allocaOp))
    return false;

  Value startVal = startStore->operand(0);
  Value boundVal = cmpOp->operand(1);

  Value stepVal;
  if (stepOp->operand(0).getDefiningOp() &&
      findAllocaForLoad(stepOp->operand(0).getDefiningOp()) == allocaOp) {
    stepVal = stepOp->operand(1);
  } else if (stepOp->operandCount() > 1 && stepOp->operand(1).getDefiningOp() &&
             findAllocaForLoad(stepOp->operand(1).getDefiningOp()) == allocaOp) {
    stepVal = stepOp->operand(0);
  }
  if (!stepVal.valid())
    return false;

  size_t insertIdx = 0;
  for (size_t i = 0; i < parentBlock->ops().size(); i++) {
    if (parentBlock->ops()[i].get() == op) {
      insertIdx = i;
      break;
    }
  }

  Region *condRegion = op->getRegions()[0].get();
  Region *bodyRegion = op->getRegions()[1].get();
  if (regionHasLoopControl(bodyRegion))
    return false;
  if (regionHasAnyCall(bodyRegion))
    return false;
  if (Operation *func = enclosingFunction(op)) {
    if (regionHasCallTo(bodyRegion, stringAttr(func->attr("sym_name"))))
      return false;
  }
  if (regionHasStoreToAlloca(bodyRegion, allocaOp, ivStore))
    return false;
  if (bodyReadsIVAfterStepStore(body, allocaOp, ivStore))
    return false;
  if (blockReadsAllocaAfter(parentBlock, op, allocaOp))
    return false;
  if ((blockInRegionTree(boundOp->getBlock(), condRegion) &&
       opOperandsDependOnRegion(boundOp, condRegion)) ||
      (blockInRegionTree(boundOp->getBlock(), bodyRegion) &&
       opOperandsDependOnRegion(boundOp, bodyRegion)))
    return false;

  Operation *stepValOp = stepVal.getDefiningOp();
  if (stepValOp &&
      ((blockInRegionTree(stepValOp->getBlock(), condRegion) &&
        opOperandsDependOnRegion(stepValOp, condRegion)) ||
       (blockInRegionTree(stepValOp->getBlock(), bodyRegion) &&
        opOperandsDependOnRegion(stepValOp, bodyRegion))))
    return false;

  if (boundOp->getBlock() == condBlock || boundOp->getBlock() == body) {
    auto hoisted = boundOp->getBlock()->takeOperation(boundOp);
    hoisted->setBlock(parentBlock);
    parentBlock->insertOperation(insertIdx, std::move(hoisted));
    insertIdx++;
  }

  if (stepValOp && (stepValOp->getBlock() == condBlock || stepValOp->getBlock() == body)) {
    auto hoisted = stepValOp->getBlock()->takeOperation(stepValOp);
    hoisted->setBlock(parentBlock);
    parentBlock->insertOperation(insertIdx, std::move(hoisted));
    insertIdx++;
  }

  op->rename("affine.for");
  if (op->operandCount() > 0)
    op->setOperand(0, startVal);
  else
    op->addOperand(startVal);
  if (op->operandCount() > 1)
    op->setOperand(1, boundVal);
  else
    op->addOperand(boundVal);
  if (op->operandCount() > 2)
    op->setOperand(2, stepVal);
  else
    op->addOperand(stepVal);

  dropRegionOperandUses(*op->getRegions()[0]);
  op->getRegions().erase(op->getRegions().begin());

  auto &ivArg = body->addArgument(module.context().i(32), op->loc(), "iv");
  ivStore->markErased();
  replaceLoadsFromAllocaInRegion(module, op->getRegions()[0].get(),
                                 allocaOp, ivArg.value());

  if (!body->ops().empty() && body->ops().back()->name() == "scf.yield")
    body->ops().back()->rename("affine.yield");
  eraseMarked(module);
  return true;
}

static void enqueueWhile(std::vector<Operation*> &worklist,
                         std::set<Operation*> &queued,
                         Operation *op) {
  if (!op || op->isErased() || op->name() != "scf.while")
    return;
  if (queued.insert(op).second)
    worklist.push_back(op);
}

static void enqueueWhileOpsInBlock(std::vector<Operation*> &worklist,
                                   std::set<Operation*> &queued,
                                   Block *block) {
  if (!block)
    return;
  for (auto &owned : block->ops()) {
    enqueueWhile(worklist, queued, owned.get());
    for (auto &region : owned->getRegions()) {
      for (auto &childBlock : region->getBlocks())
        enqueueWhileOpsInBlock(worklist, queued, childBlock.get());
    }
  }
}

int runRaiseToAffine(Module &module) {
  std::vector<Operation*> worklist;
  std::set<Operation*> queued;
  for (auto *op : walk(module))
    enqueueWhile(worklist, queued, op);

  int processed = 0;
  size_t head = 0;
  while (head < worklist.size()) {
    Operation *op = worklist[head++];
    queued.erase(op);
    if (!op || op->isErased())
      continue;
    processed++;
    Block *parentBlock = op->getBlock();
    if (!tryRaiseWhileToAffine(module, op))
      continue;
    enqueueWhileOpsInBlock(worklist, queued, parentBlock);
    for (auto &region : op->getRegions()) {
      for (auto &block : region->getBlocks())
        enqueueWhileOpsInBlock(worklist, queued, block.get());
    }
  }
  eraseMarked(module);
  return processed;
}

void runAffineLoopTiling(Module &module) {
  (void) module;
  // Tile perfect loop nests
}

void runAffineLoopFusion(Module &module) {
  for (auto *op : walk(module)) {
    if (!op || op->isErased()) continue;
    // Iterate over blocks to find adjacent affine.for operations
    for (auto &region : op->getRegions()) {
      for (auto &block : region->getBlocks()) {
        Operation *prevFor = nullptr;

        // We have to iterate carefully because we are modifying the block.
        // Wait, walk() over ops() gives unique_ptrs.
        // Better to collect them first.
        std::vector<Operation*> blockOps;
        for (auto &owned : block->ops()) {
          if (!owned->isErased()) blockOps.push_back(owned.get());
        }

        for (Operation *child : blockOps) {
          if (child->isErased()) continue;
          if (child->name() == "affine.for") {
            if (prevFor) {
              // Check if bounds match
              bool match = (prevFor->operandCount() == child->operandCount() && prevFor->operandCount() >= 3);
              if (match) {
                for (size_t i = 0; i < 3; i++) {
                  if (prevFor->operand(i) != child->operand(i)) { match = false; break; }
                }
              }
              if (match) {
                // Fuse child into prevFor
                Block *prevBody = prevFor->getRegions()[0]->getBlocks()[0].get();
                Block *childBody = child->getRegions()[0]->getBlocks()[0].get();

                // Erase affine.yield from prevBody
                if (!prevBody->ops().empty() && prevBody->ops().back()->name() == "affine.yield") {
                  prevBody->ops().back()->markErased();
                }

                // Replace child's IV with prevFor's IV
                Value prevIV = prevBody->args()[0]->value();
                Value childIV = childBody->args()[0]->value();
                replaceAllUses(module, childIV, prevIV);

                // Move all ops from childBody to prevBody
                std::vector<Operation*> toMove;
                for (auto &childOp : childBody->ops()) {
                  if (!childOp->isErased()) toMove.push_back(childOp.get());
                }

                for (Operation *mOp : toMove) {
                  auto moved = childBody->takeOperation(mOp);
                  moved->setBlock(prevBody);
                  prevBody->insertOperation(prevBody->ops().size(), std::move(moved));
                }

                child->markErased();
                // keep prevFor as prevFor to potentially fuse more
              } else {
                prevFor = child;
              }
            } else {
              prevFor = child;
            }
          } else {
            // Any other operation breaks fusion chain
            prevFor = nullptr;
          }
        }
      }
    }
  }
}

void runAffineLoopInterchange(Module &module) {
  // Find perfectly nested affine.for loops
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "affine.for") continue;

    Block *body = op->getRegions()[0]->getBlocks()[0].get();

    // Check if the only operations inside are another affine.for and an affine.yield
    Operation *inner = nullptr;
    Operation *yield = nullptr;
    int opCount = 0;
    for (auto &child : body->ops()) {
      if (child->isErased()) continue;
      opCount++;
      if (child->name() == "affine.for") inner = child.get();
      else if (child->name() == "affine.yield") yield = child.get();
    }

    if (opCount != 2 || !inner || !yield) continue;

    // Perfect loop nest found! Interchange them by swapping their bounds and steps.
    std::vector<Value> outerOps;
    for (size_t i = 0; i < op->operandCount(); i++) outerOps.push_back(op->operand(i));

    std::vector<Value> innerOps;
    for (size_t i = 0; i < inner->operandCount(); i++) innerOps.push_back(inner->operand(i));

    // Safety check
    if (outerOps.size() < 3 || innerOps.size() < 3) continue;

    // Outer gets inner's operands
    op->setOperand(0, innerOps[0]);
    op->setOperand(1, innerOps[1]);
    op->setOperand(2, innerOps[2]);

    // Inner gets outer's operands
    inner->setOperand(0, outerOps[0]);
    inner->setOperand(1, outerOps[1]);
    inner->setOperand(2, outerOps[2]);
  }
}

}
