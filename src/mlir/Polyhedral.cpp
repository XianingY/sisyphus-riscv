#include "Polyhedral.h"
#include <iostream>
#include <set>

namespace sys::mlir {

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

void runRaiseToAffine(Module &module) {
  std::vector<Operation*> toErase;
  for (auto *op : walk(module)) {
    if (!op || op->isErased()) continue;
    if (op->name() == "scf.while") {
      if (op->getRegions().size() < 2) continue;
      if (op->getRegions()[0]->getBlocks().empty() || op->getRegions()[1]->getBlocks().empty()) continue;

      Block *condBlock = op->getRegions()[0]->getBlocks()[0].get();
      Block *body = op->getRegions()[1]->getBlocks()[0].get();

      if (condBlock->ops().empty() || condBlock->ops().back()->name() != "scf.condition") continue;
      Operation *condOp = condBlock->ops().back().get();
      if (condOp->operandCount() == 0) continue;

      Operation *cmpOp = condOp->operand(0).getDefiningOp();
      if (!cmpOp) continue;
      if (cmpOp->name() != "arith.cmpi" && cmpOp->name() != "rv_machine.cmp" && cmpOp->name() != "arm_machine.cmp") continue;

      // Need an induction variable and a bound.
      Operation *ivLoad = cmpOp->operand(0).getDefiningOp();
      Operation *boundOp = cmpOp->operand(1).getDefiningOp();

      // Allow reversed operands for cmp (bound < iv). Not fully implemented yet, just assume IV < bound.
      if (!ivLoad || !boundOp) continue;

      Operation *allocaOp = findAllocaForLoad(ivLoad);
      if (!allocaOp) continue;

      // Found a candidate IV alloca. Now look for the step inside the loop body.
      Operation *stepOp = nullptr;
      Operation *ivStore = nullptr;
      for (auto &owned : body->ops()) {
        auto *child = owned.get();
        if (child->name() == "sysy.store" && child->operandCount() >= 2) {
          if (child->operand(1).getDefiningOp() == allocaOp) {
            ivStore = child;
            stepOp = child->operand(0).getDefiningOp();
          }
        }
      }

      if (!stepOp || !ivStore) continue;
      if (stepOp->name() != "arith.addi" && stepOp->name() != "rv_machine.addw" && stepOp->name() != "arm_machine.add") continue;

      // Need to find the start value. This is the last store to the alloca before the while loop.
      Operation *startStore = nullptr;
      Block *parentBlock = op->getBlock();
      for (auto &sibling : parentBlock->ops()) {
        if (sibling.get() == op) break;
        if (sibling->name() == "sysy.store" && sibling->operandCount() >= 2) {
          if (sibling->operand(1).getDefiningOp() == allocaOp) {
            startStore = sibling.get();
          }
        }
      }
      if (!startStore) continue;

      // Extract values:
      Value startVal = startStore->operand(0);
      Value boundVal = cmpOp->operand(1);

      // Find step value (must be constant for affine)
      Value stepVal;
      if (stepOp->operand(0).getDefiningOp() && findAllocaForLoad(stepOp->operand(0).getDefiningOp()) == allocaOp) {
        stepVal = stepOp->operand(1);
      } else if (stepOp->operandCount() > 1 && stepOp->operand(1).getDefiningOp() && findAllocaForLoad(stepOp->operand(1).getDefiningOp()) == allocaOp) {
        stepVal = stepOp->operand(0);
      }
      if (!stepVal.valid()) continue;

      size_t insertIdx = 0;
      for (size_t i = 0; i < parentBlock->ops().size(); i++) {
        if (parentBlock->ops()[i].get() == op) { insertIdx = i; break; }
      }

      if (boundOp->getBlock() == condBlock || boundOp->getBlock() == body) {
        auto hoisted = boundOp->getBlock()->takeOperation(boundOp);
        hoisted->setBlock(parentBlock);
        parentBlock->insertOperation(insertIdx, std::move(hoisted));
        insertIdx++;
      }

      Operation *stepValOp = stepVal.getDefiningOp();
      if (stepValOp && (stepValOp->getBlock() == condBlock || stepValOp->getBlock() == body)) {
        auto hoisted = stepValOp->getBlock()->takeOperation(stepValOp);
        hoisted->setBlock(parentBlock);
        parentBlock->insertOperation(insertIdx, std::move(hoisted));
        insertIdx++;
      }

      // Replace scf.while with affine.for in place
      op->rename("affine.for");
      if (op->operandCount() > 0) op->setOperand(0, startVal);
      else op->addOperand(startVal);
      if (op->operandCount() > 1) op->setOperand(1, boundVal);
      else op->addOperand(boundVal);
      if (op->operandCount() > 2) op->setOperand(2, stepVal);
      else op->addOperand(stepVal);

      // Erase the condBlock (we only need the body)
      op->getRegions().erase(op->getRegions().begin());

      // Add block argument for IV
      auto &ivArg = body->addArgument(module.context().i(32), op->loc(), "iv");

      // Rewrite body
      for (auto &owned : body->ops()) {
        if (owned.get() == ivStore) {
          owned->markErased();
          continue;
        }
        if (owned->name() == "sysy.load" && owned->operand(0).getDefiningOp() == allocaOp) {
          replaceAllUses(module, owned->result(), ivArg.value());
          owned->markErased();
          continue;
        }
      }

      if (!body->ops().empty() && body->ops().back()->name() == "scf.yield") {
        body->ops().back()->rename("affine.yield");
      }
    }
  }
  eraseMarked(module);
}

void runAffineLoopTiling(Module &module) {
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
