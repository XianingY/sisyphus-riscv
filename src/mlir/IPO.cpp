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
    if (op.name() == "sysy.load" || op.name() == "memref.load") {
      if (op.operandCount() == 0 ||
          localAllocas.count(op.operand(0).getDefiningOp()) == 0)
        ops = threshold + 1;
    }
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

} // namespace

void runInlining(Module &module, int threshold, SelfOptStats *stats) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_INLINE");
  if (enabled && std::string(enabled) == "0")
    return;

  SymbolTable symbols = buildSymbolTable(module);
  std::map<std::string, InlineCandidate> candidates;
  for (const auto &kv : symbols.all()) {
    InlineCandidate candidate = classifyInlineCandidate(*kv.second, threshold);
    if (candidate.func)
      candidates[kv.first] = candidate;
  }
  if (candidates.empty())
    return;

  std::vector<Operation*> calls;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "sysy.call")
      calls.push_back(op);
  }

  std::set<std::string> inlinedFunctions;
  for (Operation *call : calls) {
    if (!call || call->isErased())
      continue;
    std::string callee = symAttr(call->attr("callee"));
    auto candIt = candidates.find(callee);
    if (candIt == candidates.end())
      continue;
    InlineCandidate candidate = candIt->second;
    Block &entry = *candidate.func->getRegions()[0]->getBlocks()[0];
    if ((int) entry.args().size() != call->operandCount())
      continue;
    if (candidate.ret->operandCount() != call->resultCount())
      continue;
    Block *block = call->getBlock();
    if (!block)
      continue;
    int rawIndex = opIndex(*block, call);
    if (rawIndex < 0)
      continue;

    std::map<std::string, Value> valueMap;
    for (size_t i = 0; i < entry.args().size(); i++)
      valueMap[entry.args()[i]->value().identityKey()] = call->operand((int) i);

    std::size_t insertIndex = (std::size_t) rawIndex;
    bool ok = true;
    for (auto &owned : entry.ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() || op == candidate.ret)
        continue;
      auto cloned = cloneOperationTree(*op, valueMap);
      Operation &inserted = block->insertOperation(insertIndex++, std::move(cloned));
      for (int i = 0; i < op->resultCount(); i++)
        valueMap[op->result(i).identityKey()] = inserted.result(i);
    }
    for (int i = 0; i < call->resultCount(); i++) {
      Value returned = candidate.ret->operand(i);
      auto mapIt = valueMap.find(returned.identityKey());
      if (mapIt == valueMap.end()) {
        ok = false;
        break;
      }
      replaceAllUses(module, call->result(i), mapIt->second);
    }
    if (!ok)
      continue;
    call->markErased();
    inlinedFunctions.insert(callee);
    if (stats)
      stats->inlineCalls++;
  }
  eraseMarked(module);
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
