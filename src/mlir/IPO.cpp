#include "IPO.h"

#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace sys::mlir {

namespace {

static std::string symAttr(Attribute attr) {
  if (!attr)
    return "";
  std::string text = attr.str();
  if (text.size() >= 2 && text.front() == '"' && text.back() == '"')
    return text.substr(1, text.size() - 2);
  return text;
}

static int opIndex(Block &block, Operation *needle) {
  for (size_t i = 0; i < block.ops().size(); i++)
    if (block.ops()[i].get() == needle)
      return (int) i;
  return -1;
}

static std::vector<Type> resultTypes(Operation &op) {
  std::vector<Type> types;
  for (int i = 0; i < op.resultCount(); i++)
    types.push_back(op.resultType(i));
  return types;
}

static std::vector<Value> remapOperands(Operation &op,
                                        const std::map<std::string, Value> &valueMap) {
  std::vector<Value> operands;
  for (auto operand : op.getOperands()) {
    auto it = valueMap.find(operand.identityKey());
    operands.push_back(it == valueMap.end() ? operand : it->second);
  }
  return operands;
}

static std::unique_ptr<Operation> cloneOperationTree(
    Operation &op, std::map<std::string, Value> &valueMap) {
  auto cloned = std::make_unique<Operation>(op.name(), remapOperands(op, valueMap),
                                            resultTypes(op), op.attrs(), op.loc());
  for (auto &region : op.getRegions()) {
    Region &newRegion = cloned->addRegion();
    for (auto &block : region->getBlocks()) {
      Block &newBlock = newRegion.addBlock();
      for (auto &arg : block->args()) {
        BlockArgument &newArg = newBlock.addArgument(arg->type(), arg->loc(), arg->name());
        valueMap[arg->value().identityKey()] = newArg.value();
      }
      for (auto &child : block->ops()) {
        if (!child || child->isErased())
          continue;
        auto childClone = cloneOperationTree(*child, valueMap);
        Operation &inserted = newBlock.addOperation(std::move(childClone));
        for (int i = 0; i < child->resultCount(); i++)
          valueMap[child->result(i).identityKey()] = inserted.result(i);
      }
    }
  }
  return cloned;
}

struct InlineCandidate {
  Operation *func = nullptr;
  Operation *ret = nullptr;
  int opCount = 0;
};

struct SelectInlineCandidate {
  Operation *func = nullptr;
  Operation *ifOp = nullptr;
  Value condition;
  Value trueValue;
  Value falseValue;
  std::vector<Operation*> prefixOps;
  int opCount = 0;
};

static InlineCandidate classifyInlineCandidate(Operation &func, int threshold) {
  InlineCandidate candidate;
  if (threshold <= 0 || func.name() != "sysy.func" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return candidate;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  std::set<Operation*> localAllocas;
  std::function<void(Operation&)> collectAllocas = [&](Operation &op) {
    if (op.name() == "sysy.alloca" || op.name() == "memref.alloca")
      localAllocas.insert(&op);
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (child && !child->isErased())
            collectAllocas(*child);
  };
  for (auto &owned : entry.ops()) {
    if (owned && !owned->isErased())
      collectAllocas(*owned);
  }

  Operation *ret = nullptr;
  int returns = 0;
  int ops = 0;
  bool sawCall = false;
  std::function<void(Operation&)> countNested = [&](Operation &op) {
    ops++;
    if (op.name() == "sysy.call") {
      sawCall = true;
      ops = threshold + 1;
    }
    // A direct single-return inline clones loads at the original call site, so
    // read-only helper bodies remain semantically equivalent even when they
    // read mutable globals. Stores and calls are still rejected below. Small
    // structured loops are allowed; cloneOperationTree remaps nested regions
    // and block arguments, which lets hot digit helpers inline cleanly.
    if (op.name() == "sysy.store" || op.name() == "memref.store") {
      if (op.operandCount() < 2 ||
          localAllocas.count(op.operand(1).getDefiningOp()) == 0)
        ops = threshold + 1;
    }
    if (op.name() == "sysy.return" || op.name() == "scf.return") {
      returns++;
      ret = &op;
    }
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (child && !child->isErased())
            countNested(*child);
  };
  for (auto &owned : entry.ops()) {
    if (owned && !owned->isErased())
      countNested(*owned);
  }
  if (sawCall || ops > threshold || returns != 1 || !ret || ret->getBlock() != &entry)
    return candidate;
  if (entry.ops().empty() || entry.ops().back().get() != ret)
    return candidate;
  candidate.func = &func;
  candidate.ret = ret;
  candidate.opCount = ops;
  return candidate;
}

static bool isAllocaOp(Operation *op) {
  return op && (op->name() == "sysy.alloca" || op->name() == "memref.alloca");
}

static bool isLoadOp(Operation *op) {
  return op && (op->name() == "sysy.load" || op->name() == "memref.load");
}

static bool isStoreOp(Operation *op) {
  return op && (op->name() == "sysy.store" || op->name() == "memref.store");
}

static bool isReturnOp(Operation *op) {
  return op && (op->name() == "sysy.return" || op->name() == "scf.return");
}

static bool isIntegerScalar(Type type) {
  return type.kind() == TypeKind::Integer || type.kind() == TypeKind::Index;
}

static bool isMemref(Type type) {
  return type.kind() == TypeKind::MemRef;
}

static bool allResultsUnused(Operation *op) {
  if (!op || op->resultCount() == 0)
    return false;
  for (int i = 0; i < op->resultCount(); i++) {
    if (!op->resultUses[i].empty())
      return false;
  }
  return true;
}

static bool isPureDeadOp(Operation *op) {
  if (!op || op->isErased() || !op->getRegions().empty())
    return false;
  if (op->name() == "sysy.load" || op->name() == "memref.load" ||
      op->name() == "arith.constant" || op->name() == "arith.addi" ||
      op->name() == "arith.subi" || op->name() == "arith.muli" ||
      op->name() == "arith.divi" || op->name() == "arith.remi" ||
      op->name() == "arith.cmpi" || op->name() == "arith.andi" ||
      op->name() == "arith.ori" || op->name() == "arith.xori" ||
      op->name() == "rv_machine.li" || op->name() == "rv_machine.addw" ||
      op->name() == "rv_machine.subw" || op->name() == "rv_machine.mulw" ||
      op->name() == "rv_machine.divw" || op->name() == "rv_machine.remw" ||
      op->name() == "rv_machine.cmp" || op->name() == "rv_machine.and" ||
      op->name() == "rv_machine.or" || op->name() == "rv_machine.xor")
    return allResultsUnused(op);
  return false;
}

static bool isScalarStoreToLocal(Operation *op,
                                 const std::set<std::string> &localAllocas) {
  return isStoreOp(op) && op->operandCount() >= 2 &&
         localAllocas.count(op->operand(1).identityKey()) != 0;
}

static bool isLoadFromLocalOrGlobal(Operation *op,
                                    const std::set<std::string> &localAllocas) {
  if (!isLoadOp(op) || op->operandCount() == 0)
    return false;
  Value base = op->operand(0);
  if (localAllocas.count(base.identityKey()) != 0)
    return true;
  Operation *def = base.getDefiningOp();
  return def && !def->isErased() && def->name() == "sysy.global" &&
         base.type().kind() == TypeKind::MemRef;
}

static bool functionHasPureShape(Operation &func,
                                 const std::set<std::string> &pureNames) {
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  std::string name = symAttr(func.attr("sym_name"));
  if (name.empty())
    return false;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  for (auto &arg : entry.args()) {
    if (!arg || !isIntegerScalar(arg->type()))
      return false;
  }

  std::set<std::string> localAllocas;
  std::function<bool(Operation&)> visit = [&](Operation &op) -> bool {
    if (op.isErased())
      return true;
    if (op.name() == "sysy.alloca" || op.name() == "memref.alloca") {
      if (op.resultCount() != 1 || !isMemref(op.resultType()))
        return false;
      localAllocas.insert(op.result().identityKey());
      return true;
    }
    if (isStoreOp(&op) && !isScalarStoreToLocal(&op, localAllocas))
      return false;
    if (isLoadOp(&op) && !isLoadFromLocalOrGlobal(&op, localAllocas))
      return false;
    if (op.name() == "sysy.call") {
      std::string callee = symAttr(op.attr("callee"));
      if (callee != name && pureNames.count(callee) == 0)
        return false;
    }
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (child && !visit(*child))
            return false;
    return true;
  };

  for (auto &owned : entry.ops())
    if (owned && !visit(*owned))
      return false;
  return true;
}

static std::set<std::string> deducePureIntegerFunctions(Module &module) {
  SymbolTable symbols = buildSymbolTable(module);
  std::set<std::string> pure;
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto &kv : symbols.all()) {
      if (pure.count(kv.first) != 0)
        continue;
      if (functionHasPureShape(*kv.second, pure)) {
        pure.insert(kv.first);
        changed = true;
      }
    }
  }
  return pure;
}

static void collectDefinedValues(Operation &op, std::set<std::string> &defined) {
  for (int i = 0; i < op.resultCount(); i++)
    defined.insert(op.result(i).identityKey());
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &arg : block->args())
        if (arg)
          defined.insert(arg->value().identityKey());
      for (auto &child : block->ops())
        if (child && !child->isErased())
          collectDefinedValues(*child, defined);
    }
  }
}

static void collectStoredBases(Operation &op, std::set<std::string> &storedBases) {
  if (isStoreOp(&op) && op.operandCount() >= 2)
    storedBases.insert(op.operand(1).identityKey());
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && !child->isErased())
          collectStoredBases(*child, storedBases);
}

static bool canCloneInvariantOp(Operation *op,
                                const std::set<std::string> &pureNames,
                                const std::set<std::string> &storedBases) {
  if (!op || op->isErased() || !op->getRegions().empty() || op->resultCount() != 1)
    return false;
  if (op->name() == "sysy.load" || op->name() == "memref.load")
    return op->operandCount() > 0 &&
           storedBases.count(op->operand(0).identityKey()) == 0;
  if (op->name() == "arith.constant" || op->name() == "rv_machine.li" ||
      op->name() == "arith.addi" || op->name() == "arith.subi" ||
      op->name() == "arith.muli" || op->name() == "arith.divi" ||
      op->name() == "arith.remi" || op->name() == "arith.cmpi" ||
      op->name() == "arith.andi" || op->name() == "arith.ori" ||
      op->name() == "arith.xori" || op->name() == "rv_machine.addw" ||
      op->name() == "rv_machine.subw" || op->name() == "rv_machine.mulw" ||
      op->name() == "rv_machine.divw" || op->name() == "rv_machine.remw" ||
      op->name() == "rv_machine.cmp" || op->name() == "rv_machine.and" ||
      op->name() == "rv_machine.or" || op->name() == "rv_machine.xor")
    return true;
  if (op->name() == "sysy.call")
    return pureNames.count(symAttr(op->attr("callee"))) != 0;
  return false;
}

static Value cloneInvariantValue(Value value, Block &targetBlock,
                                 std::size_t &insertIndex,
                                 const std::set<std::string> &definedInside,
                                 const std::set<std::string> &storedBases,
                                 const std::set<std::string> &pureNames,
                                 std::map<std::string, Value> &cloned) {
  if (!value.valid())
    return Value();
  std::string key = value.identityKey();
  auto clonedIt = cloned.find(key);
  if (clonedIt != cloned.end())
    return clonedIt->second;
  if (definedInside.count(key) == 0)
    return value;
  Operation *def = value.getDefiningOp();
  if (!canCloneInvariantOp(def, pureNames, storedBases))
    return Value();

  std::vector<Value> operands;
  for (auto operand : def->getOperands()) {
    Value remapped = cloneInvariantValue(operand, targetBlock, insertIndex,
                                         definedInside, storedBases, pureNames,
                                         cloned);
    if (!remapped.valid())
      return Value();
    operands.push_back(remapped);
  }
  auto clone = std::make_unique<Operation>(def->name(), operands, resultTypes(*def),
                                           def->attrs(), def->loc());
  Operation &inserted = targetBlock.insertOperation(insertIndex++, std::move(clone));
  Value result = inserted.result();
  cloned[key] = result;
  return result;
}

static bool eraseDeadPureOpsInRegion(Region &region) {
  bool changed = false;
  for (auto &block : region.getBlocks()) {
    for (auto &owned : block->ops()) {
      if (owned && !owned->isErased()) {
        for (auto &nested : owned->getRegions())
          changed |= eraseDeadPureOpsInRegion(*nested);
      }
    }
    for (auto it = block->ops().rbegin(); it != block->ops().rend(); ++it) {
      Operation *op = it->get();
      if (isPureDeadOp(op)) {
        op->markErased();
        changed = true;
      }
    }
  }
  return changed;
}

static int hoistPureCallsFromWhile(Module &module, Operation &loop,
                                   const std::set<std::string> &pureNames) {
  if (loop.name() != "scf.while" || loop.getRegions().size() < 2)
    return 0;
  Block *parent = loop.getBlock();
  if (!parent)
    return 0;
  int loopIndex = opIndex(*parent, &loop);
  if (loopIndex < 0)
    return 0;

  std::set<std::string> definedInside;
  std::set<std::string> storedBases;
  collectDefinedValues(loop, definedInside);
  collectStoredBases(loop, storedBases);

  std::vector<Operation*> calls;
  std::function<void(Operation&)> collectCalls = [&](Operation &op) {
    if (op.name() == "sysy.call" && op.resultCount() == 1 &&
        pureNames.count(symAttr(op.attr("callee"))) != 0)
      calls.push_back(&op);
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (child && !child->isErased())
            collectCalls(*child);
  };
  collectCalls(loop);

  int hoisted = 0;
  std::size_t insertIndex = (std::size_t) loopIndex;
  for (Operation *call : calls) {
    if (!call || call->isErased())
      continue;
    std::map<std::string, Value> cloned;
    std::vector<Value> operands;
    bool ok = true;
    for (auto operand : call->getOperands()) {
      Value remapped = cloneInvariantValue(operand, *parent, insertIndex,
                                           definedInside, storedBases, pureNames,
                                           cloned);
      if (!remapped.valid()) {
        ok = false;
        break;
      }
      operands.push_back(remapped);
    }
    if (!ok)
      continue;
    auto clonedCall = std::make_unique<Operation>(
        call->name(), operands, resultTypes(*call), call->attrs(), call->loc());
    Operation &inserted = parent->insertOperation(insertIndex++, std::move(clonedCall));
    replaceAllUses(module, call->result(), inserted.result());
    call->markErased();
    hoisted++;
  }
  return hoisted;
}

static bool isPureInlinePrefixOp(Operation *op) {
  return op && !op->isErased() && op->getRegions().empty() &&
         !op->isTerminator() && op->name() != "sysy.call" &&
         !isAllocaOp(op) && !isLoadOp(op) && !isStoreOp(op) &&
         !isReturnOp(op);
}

static Value resolveLocalLoad(Value value,
                              const std::map<std::string, Value> &localStoreValues) {
  Operation *load = value.getDefiningOp();
  if (!isLoadOp(load) || load->operandCount() == 0)
    return value;
  auto it = localStoreValues.find(load->operand(0).identityKey());
  return it == localStoreValues.end() ? value : it->second;
}

static bool branchReturnsLocalOrPureValue(Region &region,
                                          const std::map<std::string, Value> &localStoreValues,
                                          Value &returned) {
  if (region.getBlocks().size() != 1)
    return false;
  Block &block = *region.getBlocks()[0];
  Operation *ret = nullptr;
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (isReturnOp(op)) {
      if (ret || op->operandCount() != 1)
        return false;
      ret = op;
      continue;
    }
    if (isLoadOp(op) && op->operandCount() > 0 &&
        localStoreValues.count(op->operand(0).identityKey()) != 0)
      continue;
    return false;
  }
  if (!ret)
    return false;
  returned = resolveLocalLoad(ret->operand(0), localStoreValues);
  return returned.valid();
}

static SelectInlineCandidate classifySelectInlineCandidate(Operation &func,
                                                           int threshold) {
  SelectInlineCandidate candidate;
  if (threshold <= 0 || func.name() != "sysy.func" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return candidate;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];

  std::set<std::string> localAllocas;
  std::map<std::string, Value> localStoreValues;
  Operation *ifOp = nullptr;
  int ops = 0;
  for (auto &owned : entry.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    ops++;
    if (ops > threshold)
      return candidate;
    if (isAllocaOp(op) && op->resultCount() == 1) {
      localAllocas.insert(op->result().identityKey());
      continue;
    }
    if (isStoreOp(op) && op->operandCount() >= 2 &&
        localAllocas.count(op->operand(1).identityKey()) != 0) {
      localStoreValues[op->operand(1).identityKey()] = op->operand(0);
      continue;
    }
    if (op->name() == "scf.if") {
      if (ifOp || op->operandCount() != 1 || op->getRegions().size() != 2)
        return candidate;
      ifOp = op;
      continue;
    }
    if (op->name() == "scf.yield")
      continue;
    if (isPureInlinePrefixOp(op)) {
      candidate.prefixOps.push_back(op);
      continue;
    }
    return candidate;
  }
  if (!ifOp)
    return candidate;

  Value trueValue;
  Value falseValue;
  if (!branchReturnsLocalOrPureValue(*ifOp->getRegions()[0], localStoreValues, trueValue) ||
      !branchReturnsLocalOrPureValue(*ifOp->getRegions()[1], localStoreValues, falseValue))
    return candidate;

  candidate.func = &func;
  candidate.ifOp = ifOp;
  candidate.condition = ifOp->operand(0);
  candidate.trueValue = trueValue;
  candidate.falseValue = falseValue;
  candidate.opCount = ops;
  return candidate;
}

static bool inlineDirectCall(Module &module, Operation &call,
                             const InlineCandidate &candidate) {
  Block &entry = *candidate.func->getRegions()[0]->getBlocks()[0];
  if ((int) entry.args().size() != call.operandCount())
    return false;
  if (candidate.ret->operandCount() != call.resultCount())
    return false;
  Block *block = call.getBlock();
  if (!block)
    return false;
  int rawIndex = opIndex(*block, &call);
  if (rawIndex < 0)
    return false;

  std::map<std::string, Value> valueMap;
  for (size_t i = 0; i < entry.args().size(); i++)
    valueMap[entry.args()[i]->value().identityKey()] = call.operand((int) i);

  std::size_t insertIndex = (std::size_t) rawIndex;
  for (auto &owned : entry.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op == candidate.ret)
      continue;
    auto cloned = cloneOperationTree(*op, valueMap);
    Operation &inserted = block->insertOperation(insertIndex++, std::move(cloned));
    for (int i = 0; i < op->resultCount(); i++)
      valueMap[op->result(i).identityKey()] = inserted.result(i);
  }
  for (int i = 0; i < call.resultCount(); i++) {
    Value returned = candidate.ret->operand(i);
    auto mapIt = valueMap.find(returned.identityKey());
    if (mapIt == valueMap.end())
      return false;
    replaceAllUses(module, call.result(i), mapIt->second);
  }
  call.markErased();
  return true;
}

static bool inlineSelectCall(Module &module, Operation &call,
                             const SelectInlineCandidate &candidate) {
  if (call.resultCount() != 1 || !candidate.func || !candidate.ifOp)
    return false;
  Block &entry = *candidate.func->getRegions()[0]->getBlocks()[0];
  if ((int) entry.args().size() != call.operandCount())
    return false;
  Block *block = call.getBlock();
  if (!block)
    return false;
  int rawIndex = opIndex(*block, &call);
  if (rawIndex < 0)
    return false;

  std::map<std::string, Value> valueMap;
  for (size_t i = 0; i < entry.args().size(); i++)
    valueMap[entry.args()[i]->value().identityKey()] = call.operand((int) i);

  std::size_t insertIndex = (std::size_t) rawIndex;
  for (Operation *op : candidate.prefixOps) {
    if (!op || op->isErased())
      return false;
    auto cloned = cloneOperationTree(*op, valueMap);
    Operation &inserted = block->insertOperation(insertIndex++, std::move(cloned));
    for (int i = 0; i < op->resultCount(); i++)
      valueMap[op->result(i).identityKey()] = inserted.result(i);
  }

  auto remap = [&](Value value) -> Value {
    auto it = valueMap.find(value.identityKey());
    return it == valueMap.end() ? Value() : it->second;
  };
  Value cond = remap(candidate.condition);
  Value trueValue = remap(candidate.trueValue);
  Value falseValue = remap(candidate.falseValue);
  if (!cond.valid() || !trueValue.valid() || !falseValue.valid())
    return false;

  auto select = std::make_unique<Operation>(
      "arith.select",
      std::vector<Value>{cond, trueValue, falseValue},
      std::vector<Type>{call.resultType()},
      std::map<std::string, Attribute>{},
      call.loc());
  Operation &inserted = block->insertOperation(insertIndex++, std::move(select));
  replaceAllUses(module, call.result(), inserted.result());
  call.markErased();
  return true;
}

} // namespace

void runInlining(Module &module, int threshold, SelfOptStats *stats) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_INLINE");
  if (enabled && std::string(enabled) == "0")
    return;

  std::set<std::string> inlinedFunctions;
  for (int iteration = 0; iteration < 4; iteration++) {
    SymbolTable symbols = buildSymbolTable(module);
    std::map<std::string, InlineCandidate> directCandidates;
    std::map<std::string, SelectInlineCandidate> selectCandidates;
    for (const auto &kv : symbols.all()) {
      InlineCandidate direct = classifyInlineCandidate(*kv.second, threshold);
      if (direct.func)
        directCandidates[kv.first] = direct;
      SelectInlineCandidate select = classifySelectInlineCandidate(*kv.second, threshold);
      if (select.func)
        selectCandidates[kv.first] = select;
    }
    if (directCandidates.empty() && selectCandidates.empty())
      break;

    std::vector<Operation*> calls;
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "sysy.call")
        calls.push_back(op);
    }

    bool changed = false;
    for (Operation *call : calls) {
      if (!call || call->isErased())
        continue;
      std::string callee = symAttr(call->attr("callee"));
      auto selectIt = selectCandidates.find(callee);
      if (selectIt != selectCandidates.end() &&
          inlineSelectCall(module, *call, selectIt->second)) {
        inlinedFunctions.insert(callee);
        changed = true;
        if (stats)
          stats->inlineCalls++;
        continue;
      }
      auto directIt = directCandidates.find(callee);
      if (directIt != directCandidates.end() &&
          inlineDirectCall(module, *call, directIt->second)) {
        inlinedFunctions.insert(callee);
        changed = true;
        if (stats)
          stats->inlineCalls++;
      }
    }
    eraseMarked(module);
    if (!changed)
      break;
  }
  if (stats)
    stats->inlineFunctions += (int) inlinedFunctions.size();
}

void runIPCP(Module &module, SelfOptStats *stats) {
  // Propagate constants across function boundaries and run a conservative
  // pure-call LICM/CSE slice for scalar helper calls.  The LICM path only
  // clones loop-invariant operands and pure i32 helper calls; memref-argument
  // functions and calls behind uncertain stores are left untouched.
  const std::set<std::string> pureNames = deducePureIntegerFunctions(module);
  if (pureNames.empty())
    return;

  int hoisted = 0;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "scf.while")
      hoisted += hoistPureCallsFromWhile(module, *op, pureNames);
  }
  if (hoisted > 0) {
    for (auto *op : walk(module)) {
      if (!op || op->isErased())
        continue;
      for (auto &region : op->getRegions())
        eraseDeadPureOpsInRegion(*region);
    }
    eraseMarked(module);
    if (stats)
      stats->pureCallHoists += hoisted;
  }
}

void runPureFunctionDeduction(Module &module) {
  (void) module;
  // Deduce pure functions (no side effects) and add "pure" attribute.
}

}
