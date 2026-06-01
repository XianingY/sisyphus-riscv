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
  std::function<void(Operation&)> countNested = [&](Operation &op) {
    ops++;
    if (op.name() == "sysy.call" || op.name() == "scf.while" ||
        op.name() == "scf.for" || op.name() == "affine.for")
      ops = threshold + 1;
    // A direct single-return inline clones loads at the original call site, so
    // read-only helper bodies remain semantically equivalent even when they
    // read mutable globals. Stores, calls, and loops are still rejected below.
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
  if (ops > threshold || returns != 1 || !ret || ret->getBlock() != &entry)
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

void runIPCP(Module &module) {
  (void) module;
  // Propagate constants across function boundaries.
}

void runPureFunctionDeduction(Module &module) {
  (void) module;
  // Deduce pure functions (no side effects) and add "pure" attribute.
}

}
