#include "Polyhedral.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

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
                                   const std::vector<Operation*> &exceptStores) {
  if (!region)
    return false;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (isStoreToAlloca(op, allocaOp) &&
          std::find(exceptStores.begin(), exceptStores.end(), op) == exceptStores.end())
        return true;
      for (auto &nested : op->getRegions()) {
        if (regionHasStoreToAlloca(nested.get(), allocaOp, exceptStores))
          return true;
      }
    }
  }
  return false;
}

static bool regionHasStoreToAlloca(Region *region, Operation *allocaOp,
                                   Operation *exceptStore) {
  std::vector<Operation*> excepts;
  if (exceptStore)
    excepts.push_back(exceptStore);
  return regionHasStoreToAlloca(region, allocaOp, excepts);
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

static void collectStoresToAllocaInRegion(Region *region, Operation *allocaOp,
                                          std::vector<Operation*> &out) {
  if (!region)
    return;
  for (auto &block : region->getBlocks()) {
    for (auto &owned : block->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased())
        continue;
      if (op->name() == "sysy.store" && op->operandCount() >= 2 &&
          op->operand(1).getDefiningOp() == allocaOp) {
        out.push_back(op);
      }
      for (auto &nested : op->getRegions()) {
        collectStoresToAllocaInRegion(nested.get(), allocaOp, out);
      }
    }
  }
}

static bool constantIntegerValue(Value v, int64_t &out);

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

  Region *bodyRegion = op->getRegions()[1].get();
  std::vector<Operation*> allStores;
  collectStoresToAllocaInRegion(bodyRegion, allocaOp, allStores);

  if (allStores.empty())
    return false;

  Operation *ivStore = allStores[0];
  Operation *stepOp = ivStore->operand(0).getDefiningOp();

  if (!stepOp)
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

  int64_t stepValConst = 0;
  bool stepIsConst = constantIntegerValue(stepVal, stepValConst);

  for (Operation *store : allStores) {
    Operation *currStep = store->operand(0).getDefiningOp();
    if (!currStep) return false;
    if (currStep->name() != "arith.addi" && currStep->name() != "rv_machine.addw" &&
        currStep->name() != "arm_machine.add")
      return false;

    Value currStepVal;
    if (currStep->operand(0).getDefiningOp() &&
        findAllocaForLoad(currStep->operand(0).getDefiningOp()) == allocaOp) {
      currStepVal = currStep->operand(1);
    } else if (currStep->operandCount() > 1 && currStep->operand(1).getDefiningOp() &&
               findAllocaForLoad(currStep->operand(1).getDefiningOp()) == allocaOp) {
      currStepVal = currStep->operand(0);
    }
    if (!currStepVal.valid())
      return false;
    if (currStepVal != stepVal) {
      int64_t currConst = 0;
      if (!stepIsConst || !constantIntegerValue(currStepVal, currConst) || currConst != stepValConst) {
        return false;
      }
    }
  }

  size_t insertIdx = 0;
  for (size_t i = 0; i < parentBlock->ops().size(); i++) {
    if (parentBlock->ops()[i].get() == op) {
      insertIdx = i;
      break;
    }
  }

  Region *condRegion = op->getRegions()[0].get();
  if (regionHasLoopControl(bodyRegion))
    return false;
  if (regionHasAnyCall(bodyRegion))
    return false;
  if (Operation *func = enclosingFunction(op)) {
    if (regionHasCallTo(bodyRegion, stringAttr(func->attr("sym_name"))))
      return false;
  }
  if (regionHasStoreToAlloca(bodyRegion, allocaOp, allStores))
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
  for (Operation *store : allStores) {
    store->markErased();
  }
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

static bool constantIntegerValue(Value v, int64_t &out);

void runAffineLoopTiling(Module &module) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_TILE");
  if (enabled && std::string(enabled) == "0")
    return;

  Context &ctx = module.context();

  // Tile size from env or default 32
  int64_t tileSize = 32;
  const char *ts = std::getenv("SISY_SELF_TILE_SIZE");
  if (ts) {
    try {
      tileSize = std::stoll(ts);
      if (tileSize <= 0) tileSize = 32;
    } catch (...) {
      tileSize = 32;
    }
  }

  auto constIntValue = [&](Value v, int64_t &out) -> bool {
    if (!v.valid() || !v.isOperationResult()) return false;
    Operation *def = v.getDefiningOp();
    if (!def) return false;
    if (def->name() != "arith.constant") return false;
    Attribute a = def->attr("value");
    if (!a) return false;
    std::string s = stringAttr(a);
    try {
      out = std::stoll(s);
      return true;
    } catch (...) {
      return false;
    }
  };

  auto isNestedUnder = [](Block *block, Operation *ancestor) -> bool {
    if (!block || !ancestor) return false;
    Block *currBlock = block;
    while (currBlock) {
      Region *reg = currBlock->getRegion();
      if (!reg) break;
      Operation *parentOp = reg->getParent();
      if (parentOp == ancestor) return true;
      currBlock = parentOp ? parentOp->getBlock() : nullptr;
    }
    return false;
  };

  std::function<void(Operation*, Operation*, Block*, std::size_t&)> hoistToParent =
    [&](Operation *nestedOp, Operation *loopOp, Block *targetBlock, std::size_t &idx) {
      if (!nestedOp || nestedOp->isErased()) return;
      Block *currBlock = nestedOp->getBlock();
      if (!currBlock || !isNestedUnder(currBlock, loopOp)) return;

      for (int i = 0; i < nestedOp->operandCount(); ++i) {
        Value val = nestedOp->operand(i);
        if (val.isOperationResult()) {
          Operation *def = val.getDefiningOp();
          if (def) hoistToParent(def, loopOp, targetBlock, idx);
        }
      }

      auto hoisted = currBlock->takeOperation(nestedOp);
      if (!hoisted) return;
      hoisted->setBlock(targetBlock);
      targetBlock->insertOperation(idx, std::move(hoisted));
      idx++;
    };

  // Walk top-level ops and look for perfect 2-level affine.for nests.
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "affine.for") continue;
    // get outer body block
    if (op->getRegions().empty()) continue;
    Region *outerReg = op->getRegions()[0].get();
    if (outerReg->getBlocks().empty()) continue;

    // find first affine.for anywhere inside outer region
    Operation *inner = nullptr;
    std::function<Operation*(Block*)> findInBlock = [&](Block *blk) -> Operation* {
      if (!blk) return nullptr;
      for (auto &owned : blk->ops()) {
        Operation *cand = owned.get();
        if (!cand || cand->isErased()) continue;
        if (cand->name() == "affine.for") return cand;
        for (auto &r : cand->getRegions()) {
          for (auto &b : r->getBlocks()) {
            Operation *res = findInBlock(b.get());
            if (res) return res;
          }
        }
      }
      return nullptr;
    };
    for (auto &b : outerReg->getBlocks()) {
      inner = findInBlock(b.get());
      if (inner) break;
    }
    // conservative legality: skip if inner region contains calls (side-effects)
    if (inner && !inner->getRegions().empty()) {
      Region *r = inner->getRegions()[0].get();
      if (regionHasAnyCall(r))
        continue;
    }
    if (!inner) {
      // debug: no inner affine.for found in outer
      continue;
    }

    // require both loops have lower/upper/step
    if (op->operandCount() < 3 || inner->operandCount() < 3) continue;

    int64_t outerLowerC, outerStepC;
    int64_t innerLowerC, innerStepC;
    // require constant lower bounds and step==1 for conservative strip-mining
    if (!constIntValue(op->operand(0), outerLowerC)) continue;
    if (!constIntValue(op->operand(2), outerStepC)) continue;
    if (!constIntValue(inner->operand(0), innerLowerC)) continue;
    if (!constIntValue(inner->operand(2), innerStepC)) continue;
    if (outerStepC != 1 || innerStepC != 1) continue;

    // Prepare insertion at the position of outer in its parent block
    Block *parent = op->getBlock();
    if (!parent) continue;
    std::size_t insertIndex = 0;
    bool found = false;
    for (std::size_t i = 0; i < parent->ops().size(); ++i) {
      if (parent->ops()[i].get() == op) { insertIndex = i; found = true; break; }
    }
    if (!found) continue;

    // Hoist bounds first, before inserting any new loop operations!
    Value outerUpperVal = op->operand(1);
    int64_t outerConstVal = 0;
    if (constantIntegerValue(outerUpperVal, outerConstVal)) {
      auto newConst = std::make_unique<Operation>(
          "arith.constant", std::vector<Value>{}, std::vector<Type>{ctx.i(32)},
          std::map<std::string, Attribute>{{"value", ctx.integerAttr(outerConstVal, ctx.i(32))}}, op->loc());
      Operation &newConstOp = parent->insertOperation(insertIndex, std::move(newConst));
      insertIndex++;
      outerUpperVal = newConstOp.result();
    } else if (outerUpperVal.isOperationResult()) {
      Operation *def = outerUpperVal.getDefiningOp();
      if (def) {
        hoistToParent(def, op, parent, insertIndex);
      }
    }

    Value innerUpperVal = inner->operand(1);
    int64_t innerConstVal = 0;
    if (constantIntegerValue(innerUpperVal, innerConstVal)) {
      // Handled later when creating it inside midBody
    } else if (innerUpperVal.isOperationResult()) {
      Operation *def = innerUpperVal.getDefiningOp();
      if (def) {
        hoistToParent(def, op, parent, insertIndex);
      }
    }

    // Create tile size constant in parent
    Attribute tileAttr = ctx.integerAttr(tileSize, ctx.i(32));
    auto tileConst = std::make_unique<Operation>("arith.constant", std::vector<Value>{}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{{"value", tileAttr}}, op->loc());
    Operation &tileConstOp = parent->insertOperation(insertIndex, std::move(tileConst));
    insertIndex++;

    // create tile loop: affine.for(lower, upper, tileSize)
    std::vector<Value> tileOperands{op->operand(0), outerUpperVal, tileConstOp.result()};
    auto tileLoopOpPtr = std::make_unique<Operation>("affine.for", tileOperands, std::vector<Type>{}, std::map<std::string, Attribute>{}, op->loc());
    tileLoopOpPtr->addRegion();
    Operation &tileLoopOp = parent->insertOperation(insertIndex, std::move(tileLoopOpPtr));
    insertIndex++;

    // tile loop body block and argument
    Block &tileBody = tileLoopOp.getRegions()[0]->addBlock();
    BlockArgument &tileArg = tileBody.addArgument(ctx.i(32), op->loc(), "tv");

    // inside tile body compute end = tv + tileSize
    // arith.addi
    auto addEndPtr = std::make_unique<Operation>("arith.addi", std::vector<Value>{tileArg.value(), tileConstOp.result()}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{}, op->loc());
    Operation &addEnd = tileBody.addOperation(std::move(addEndPtr));

    // cmp = arith.cmpi(end, outerUpper, {predicate: "gt"})
    auto cmpPtr = std::make_unique<Operation>("arith.cmpi", std::vector<Value>{addEnd.result(), outerUpperVal}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{{"predicate", ctx.stringAttr("gt")}}, op->loc());
    Operation &cmp = tileBody.addOperation(std::move(cmpPtr));

    // select = arith.select(cmp, outerUpper, end)
    auto selPtr = std::make_unique<Operation>("arith.select", std::vector<Value>{cmp.result(), outerUpperVal, addEnd.result()}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{}, op->loc());
    Operation &sel = tileBody.addOperation(std::move(selPtr));

    // create middle loop (iterate original outer IV within tile): affine.for(tv, sel, outer.step)
    std::vector<Value> midOperands{tileArg.value(), sel.result(), op->operand(2)};
    auto midLoopPtr = std::make_unique<Operation>("affine.for", midOperands, std::vector<Type>{}, std::map<std::string, Attribute>{}, op->loc());
    midLoopPtr->addRegion();
    Operation &midLoop = tileBody.addOperation(std::move(midLoopPtr));

    Block &midBody = midLoop.getRegions()[0]->addBlock();
    BlockArgument &midIv = midBody.addArgument(ctx.i(32), op->loc(), "j");

    // create tiled inner loop (2D tiling): create tile constant for inner
    auto innerTileConst = std::make_unique<Operation>("arith.constant", std::vector<Value>{}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{{"value", ctx.integerAttr(tileSize, ctx.i(32))}}, inner->loc());
    Operation &innerTileConstOp = midBody.addOperation(std::move(innerTileConst));

    if (constantIntegerValue(innerUpperVal, innerConstVal)) {
      auto newConst = std::make_unique<Operation>(
          "arith.constant", std::vector<Value>{}, std::vector<Type>{ctx.i(32)},
          std::map<std::string, Attribute>{{"value", ctx.integerAttr(innerConstVal, ctx.i(32))}}, inner->loc());
      Operation &newConstOp = midBody.insertOperation(0, std::move(newConst));
      innerUpperVal = newConstOp.result();
    }

    // tile loop for inner: affine.for(inner_lower, inner_upper, tileSize)
    std::vector<Value> tileYOperands{inner->operand(0), innerUpperVal, innerTileConstOp.result()};
    auto tileLoopYPtr = std::make_unique<Operation>("affine.for", tileYOperands, std::vector<Type>{}, std::map<std::string, Attribute>{}, inner->loc());
    tileLoopYPtr->addRegion();
    Operation &tileLoopY = midBody.addOperation(std::move(tileLoopYPtr));

    // tile body and arg
    Block &tileYBody = tileLoopY.getRegions()[0]->addBlock();
    BlockArgument &tileYArg = tileYBody.addArgument(ctx.i(32), inner->loc(), "tv_y");

    // endY = tv_y + tileSize
    auto addEndYPtr = std::make_unique<Operation>("arith.addi", std::vector<Value>{tileYArg.value(), innerTileConstOp.result()}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{}, inner->loc());
    Operation &addEndY = tileYBody.addOperation(std::move(addEndYPtr));

    auto cmpYPtr = std::make_unique<Operation>("arith.cmpi", std::vector<Value>{addEndY.result(), innerUpperVal}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{{"predicate", ctx.stringAttr("gt")}}, inner->loc());
    Operation &cmpY = tileYBody.addOperation(std::move(cmpYPtr));

    auto selYPtr = std::make_unique<Operation>("arith.select", std::vector<Value>{cmpY.result(), innerUpperVal, addEndY.result()}, std::vector<Type>{ctx.i(32)}, std::map<std::string, Attribute>{}, inner->loc());
    Operation &selY = tileYBody.addOperation(std::move(selYPtr));

    // inner mid loop: iterate k from tv_y to selY
    std::vector<Value> innerMidOperands{tileYArg.value(), selY.result(), inner->operand(2)};
    auto innerMidPtr = std::make_unique<Operation>("affine.for", innerMidOperands, std::vector<Type>{}, std::map<std::string, Attribute>{}, inner->loc());
    innerMidPtr->addRegion();
    Operation &innerMid = tileYBody.addOperation(std::move(innerMidPtr));

    Block &innerMidBody = innerMid.getRegions()[0]->addBlock();
    BlockArgument &newKiv = innerMidBody.addArgument(ctx.i(32), inner->loc(), "k");

    // move original inner body ops into innerMidBody
    Region *oldInnerReg = inner->getRegions()[0].get();
    if (!oldInnerReg->getBlocks().empty()) {
      Block *oldInnerBody = oldInnerReg->getBlocks()[0].get();
      std::vector<Operation*> toMove;
      for (auto &owned : oldInnerBody->ops()) {
        Operation *o = owned.get();
        if (!o || o->isErased()) continue;
        toMove.push_back(o);
      }
      for (Operation *m : toMove) {
        auto moved = oldInnerBody->takeOperation(m);
        innerMidBody.addOperation(std::move(moved));
      }
    }

    // Replace uses of old IVs: outer iv -> midIv, inner iv -> newKiv
    // old outer IV
    if (!op->getRegions().empty() && !op->getRegions()[0]->getBlocks().empty()) {
      Block *oldOuterBody = op->getRegions()[0]->getBlocks()[0].get();
      if (!oldOuterBody->args().empty()) {
        Value oldOuterIv = oldOuterBody->args()[0]->value();
        replaceAllUses(module, oldOuterIv, midIv.value());
      }
    }
    // old inner IV
    if (!inner->getRegions().empty() && !inner->getRegions()[0]->getBlocks().empty()) {
      Block *oldInnerBody = inner->getRegions()[0]->getBlocks()[0].get();
      if (!oldInnerBody->args().empty()) {
        Value oldInnerIv = oldInnerBody->args()[0]->value();
        replaceAllUses(module, oldInnerIv, newKiv.value());
      }
    }

    // mark old loops erased
    inner->markErased();
    op->markErased();
    // increment stat (attempt to find stats elsewhere; use module-level side-effect via a global? For now nothing)
    // We'll rely on collectAffineNestSummary to recount after transformations
  }
  eraseMarked(module);
}

void runAffineLoopFusion(Module &module) {
  struct FusionAccess {
    std::string baseKey;
    bool store = false;
  };
  auto singleBlock = [](Operation *op) -> Block* {
    if (!op || op->getRegions().size() != 1 ||
        op->getRegions()[0]->getBlocks().size() != 1)
      return nullptr;
    return op->getRegions()[0]->getBlocks()[0].get();
  };
  std::function<bool(Operation*)> unsafeTree = [&](Operation *op) -> bool {
    if (!op || op->isErased())
      return false;
    const std::string &name = op->name();
    if (name == "sysy.call" || name == "sysy.return" ||
        name == "sysy.break" || name == "sysy.continue" ||
        name == "scf.if" || name == "scf.while" || name == "scf.for" ||
        name == "sysy.alloca" || name == "memref.alloca")
      return true;
    for (auto &region : op->getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (unsafeTree(child.get()))
            return true;
    return false;
  };
  std::function<void(Operation*, std::vector<FusionAccess>&)> collectAccesses =
      [&](Operation *op, std::vector<FusionAccess> &out) {
        if (!op || op->isErased())
          return;
        if ((op->name() == "memref.load" || op->name() == "sysy.load") &&
            op->operandCount() >= 1) {
          out.push_back({op->operand(0).identityKey(), false});
        } else if ((op->name() == "memref.store" || op->name() == "sysy.store") &&
                   op->operandCount() >= 2) {
          out.push_back({op->operand(1).identityKey(), true});
        }
        for (auto &region : op->getRegions())
          for (auto &block : region->getBlocks())
            for (auto &child : block->ops())
              collectAccesses(child.get(), out);
      };
  auto fusionSafe = [&](Operation *prevFor, Operation *childFor) -> bool {
    if (!prevFor || !childFor || prevFor->operandCount() < 3 ||
        childFor->operandCount() < 3)
      return false;
    Block *prevBody = singleBlock(prevFor);
    Block *childBody = singleBlock(childFor);
    if (!prevBody || !childBody || prevBody->args().empty() ||
        childBody->args().empty())
      return false;
    for (auto &owned : prevBody->ops()) {
      if (!owned || owned->isErased() || owned->name() == "affine.yield")
        continue;
      if (unsafeTree(owned.get()))
        return false;
    }
    for (auto &owned : childBody->ops()) {
      if (!owned || owned->isErased() || owned->name() == "affine.yield")
        continue;
      if (unsafeTree(owned.get()))
        return false;
    }

    std::vector<FusionAccess> prevAccesses;
    std::vector<FusionAccess> childAccesses;
    for (auto &owned : prevBody->ops())
      collectAccesses(owned.get(), prevAccesses);
    for (auto &owned : childBody->ops())
      collectAccesses(owned.get(), childAccesses);

    // Fusion changes execution from all prev iterations followed by all child
    // iterations to prev(i); child(i).  Keep only cases where writes in one
    // loop cannot alias any access in the other loop.  Read/read sharing is
    // harmless; producer-consumer fusion can be added later with exact
    // distance-vector checks.
    for (const auto &a : prevAccesses) {
      for (const auto &b : childAccesses) {
        if (a.baseKey != b.baseKey)
          continue;
        if (a.store || b.store)
          return false;
      }
    }
    return true;
  };
  std::function<void(Operation*, Value, Value)> replaceValueInTree =
      [&](Operation *op, Value oldValue, Value newValue) {
        if (!op || op->isErased())
          return;
        for (int i = 0; i < op->operandCount(); i++) {
          if (op->operand(i) == oldValue)
            op->setOperand(i, newValue);
        }
        for (auto &region : op->getRegions())
          for (auto &block : region->getBlocks())
            for (auto &child : block->ops())
              replaceValueInTree(child.get(), oldValue, newValue);
      };

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
                if (!fusionSafe(prevFor, child)) {
                  prevFor = child;
                  continue;
                }
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

                // Move all ops from childBody to prevBody
                std::vector<Operation*> toMove;
                for (auto &childOp : childBody->ops()) {
                  if (!childOp->isErased()) toMove.push_back(childOp.get());
                }

                for (Operation *mOp : toMove) {
                  replaceValueInTree(mOp, childIV, prevIV);
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

static int operationIndexInBlock(Block &block, Operation *needle) {
  for (size_t i = 0; i < block.ops().size(); i++) {
    if (block.ops()[i].get() == needle)
      return (int) i;
  }
  return -1;
}

static bool constantIntegerValue(Value v, int64_t &out) {
  if (!v.valid() || !v.isOperationResult()) return false;
  Operation *def = v.getDefiningOp();
  if (!def) return false;
  if (def->name() != "arith.constant") return false;
  Attribute a = def->attr("value");
  if (!a) return false;
  std::string s = stringAttr(a);
  try {
    out = std::stoll(s);
    return true;
  } catch (...) {
    return false;
  }
}

static void eliminateContinuesInBlock(Block &block, Module &module) {
  auto &ops = block.ops();
  for (size_t i = 0; i < ops.size(); i++) {
    Operation *op = ops[i].get();
    if (!op || op->isErased()) continue;

    for (auto &reg : op->getRegions()) {
      for (auto &b : reg->getBlocks()) {
        eliminateContinuesInBlock(*b, module);
      }
    }

    if (op->name() == "scf.if") {
      bool hasContinueInThen = false;
      Region *thenRegion = op->getRegions()[0].get();
      if (!thenRegion->getBlocks().empty()) {
        for (auto &owned : thenRegion->getBlocks()[0]->ops()) {
          if (owned && !owned->isErased() && owned->name() == "sysy.continue") {
            hasContinueInThen = true;
            break;
          }
        }
      }

      if (hasContinueInThen) {
        Block *thenBlock = thenRegion->getBlocks()[0].get();
        for (auto &owned : thenBlock->ops()) {
          if (owned && !owned->isErased() && owned->name() == "sysy.continue") {
            owned->markErased();
          }
        }
        thenBlock->eraseMarkedOperations();

        if (op->getRegions().size() < 2) {
          Region &elseRegion = op->addRegion();
          elseRegion.addBlock();
        }
        Block *elseBlock = op->getRegions()[1]->getBlocks()[0].get();

        size_t lastIdx = ops.size() - 1;
        while (lastIdx > i && ops[lastIdx]->isTerminator()) {
          lastIdx--;
        }

        for (size_t j = i + 1; j <= lastIdx; ) {
          if (j < ops.size()) {
            Operation *toMove = ops[j].get();
            auto taken = block.takeOperation(toMove);
            elseBlock->addOperation(std::move(taken));
          } else {
            break;
          }
        }

        eliminateContinuesInBlock(*elseBlock, module);
        break;
      }
    }
  }
  block.eraseMarkedOperations();
}

static void collectOpsInBlock(Block &block, std::vector<Operation*> &out) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased()) continue;
    out.push_back(op);
    for (auto &reg : op->getRegions()) {
      for (auto &b : reg->getBlocks()) {
        collectOpsInBlock(*b, out);
      }
    }
  }
}

void runContinueToIfWrap(Module &module) {
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "sysy.func") {
      for (auto &reg : op->getRegions()) {
        for (auto &block : reg->getBlocks()) {
          eliminateContinuesInBlock(*block, module);
        }
      }
    }
  }
}

void runImperfectLoopPromotion(Module &module) {
  Context &ctx = module.context();
  std::vector<Operation*> loops;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "affine.for") {
      loops.push_back(op);
    }
  }

  for (Operation *loop_j : loops) {
    if (!loop_j || loop_j->isErased()) continue;
    if (loop_j->getRegions().empty() || loop_j->getRegions()[0]->getBlocks().empty()) continue;
    Block *body_j = loop_j->getRegions()[0]->getBlocks()[0].get();

    Operation *init_store = nullptr;
    Operation *loop_k = nullptr;
    Operation *load_sum = nullptr;
    Operation *writeback_store = nullptr;

    for (auto &owned : body_j->ops()) {
      Operation *child = owned.get();
      if (!child || child->isErased()) continue;
      if (child->name() == "sysy.store" && child->operandCount() >= 2) {
        int64_t val = 0;
        if (constantIntegerValue(child->operand(0), val) && val == 0) {
          init_store = child;
        }
      } else if (child->name() == "affine.for") {
        loop_k = child;
      } else if (child->name() == "sysy.load") {
        load_sum = child;
      } else if (child->name() == "memref.store") {
        writeback_store = child;
      }
    }

    if (!init_store || !loop_k || !load_sum || !writeback_store) continue;

    Value sum_slot = init_store->operand(1);
    if (load_sum->operand(0) != sum_slot) continue;
    if (writeback_store->operand(0) != load_sum->result()) continue;

    Value array_base = writeback_store->operand(1);
    std::vector<Value> array_indices;
    for (int i = 2; i < writeback_store->operandCount(); i++) {
      array_indices.push_back(writeback_store->operand(i));
    }

    Block *parentBlock = loop_j->getBlock();
    int insertIdx = operationIndexInBlock(*parentBlock, loop_j);
    if (insertIdx < 0) continue;

    Location loc = loop_j->loc();

    auto init_loop_op = std::make_unique<Operation>(
        "affine.for", std::vector<Value>{loop_j->operand(0), loop_j->operand(1), loop_j->operand(2)},
        std::vector<Type>{}, std::map<std::string, Attribute>{}, loc);
    Region &init_reg = init_loop_op->addRegion();
    Block &init_body = init_reg.addBlock();

    auto &j_arg = init_body.addArgument(ctx.i(32), loc, "j");

    auto zero_const = std::make_unique<Operation>(
        "arith.constant", std::vector<Value>{}, std::vector<Type>{ctx.i(32)},
        std::map<std::string, Attribute>{{"value", ctx.integerAttr(0, ctx.i(32))}}, loc);
    Operation &zero_op = init_body.addOperation(std::move(zero_const));

    Value old_j_iv = body_j->args()[0]->value();
    std::vector<Value> new_indices;
    for (Value idx : array_indices) {
      if (idx == old_j_iv) {
        new_indices.push_back(j_arg.value());
      } else {
        new_indices.push_back(idx);
      }
    }

    std::vector<Value> store_ops{zero_op.result(), array_base};
    store_ops.insert(store_ops.end(), new_indices.begin(), new_indices.end());
    auto mem_store = std::make_unique<Operation>(
        "memref.store", store_ops, std::vector<Type>{},
        writeback_store->attrs(), loc);
    init_body.addOperation(std::move(mem_store));

    auto yield_op = std::make_unique<Operation>("affine.yield", std::vector<Value>{}, std::vector<Type>{}, std::map<std::string, Attribute>{}, loc);
    init_body.addOperation(std::move(yield_op));

    parentBlock->insertOperation(insertIdx, std::move(init_loop_op));
    insertIdx++;

    init_store->markErased();
    load_sum->markErased();
    writeback_store->markErased();
    body_j->eraseMarkedOperations();

    Block *body_k = loop_k->getRegions()[0]->getBlocks()[0].get();

    std::vector<Operation*> k_ops;
    collectOpsInBlock(*body_k, k_ops);

    for (Operation *op_in_k : k_ops) {
      if (op_in_k->name() == "sysy.load" && op_in_k->operand(0) == sum_slot) {
        std::vector<Value> load_ops{array_base};
        load_ops.insert(load_ops.end(), array_indices.begin(), array_indices.end());

        auto new_load = std::make_unique<Operation>(
            "memref.load", load_ops, std::vector<Type>{ctx.i(32)},
            std::map<std::string, Attribute>{}, op_in_k->loc());

        int op_idx = operationIndexInBlock(*body_k, op_in_k);
        Operation &inserted_load = body_k->insertOperation(op_idx, std::move(new_load));

        replaceAllUses(module, op_in_k->result(), inserted_load.result());
        op_in_k->markErased();
      } else if (op_in_k->name() == "sysy.store" && op_in_k->operandCount() >= 2 && op_in_k->operand(1) == sum_slot) {
        std::vector<Value> store_ops{op_in_k->operand(0), array_base};
        store_ops.insert(store_ops.end(), array_indices.begin(), array_indices.end());

        auto new_store = std::make_unique<Operation>(
            "memref.store", store_ops, std::vector<Type>{},
            std::map<std::string, Attribute>{}, op_in_k->loc());

        int op_idx = operationIndexInBlock(*body_k, op_in_k);
        body_k->insertOperation(op_idx, std::move(new_store));
        op_in_k->markErased();
      }
    }
    body_k->eraseMarkedOperations();
  }
  eraseMarked(module);
}

}
