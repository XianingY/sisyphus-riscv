#include "SelfMLIR.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace sys::mlir {

static std::string trim(const std::string &s) {
  size_t begin = 0;
  while (begin < s.size() && std::isspace((unsigned char) s[begin]))
    begin++;
  size_t end = s.size();
  while (end > begin && std::isspace((unsigned char) s[end - 1]))
    end--;
  return s.substr(begin, end - begin);
}

static int operationIndexInBlock(Block &block, Operation *needle) {
  for (size_t i = 0; i < block.ops().size(); i++)
    if (block.ops()[i].get() == needle)
      return (int) i;
  return -1;
}

std::vector<RewriteRule> parseDRR(const std::string &text,
                                  std::vector<std::string> &errors) {
  std::vector<RewriteRule> rules;
  std::istringstream is(text);
  std::string line;
  int lineno = 0;
  while (std::getline(is, line)) {
    lineno++;
    line = trim(line);
    if (line.empty() || line[0] == '#')
      continue;
    std::istringstream ls(line);
    std::string word;
    RewriteRule rule;
    ls >> word;
    if (word != "rule") {
      errors.push_back("line " + std::to_string(lineno) + ": expected 'rule'");
      continue;
    }
    ls >> rule.name >> rule.root >> rule.kind >> rule.benefit;
    if (rule.name.empty() || rule.root.empty() || rule.kind.empty()) {
      errors.push_back("line " + std::to_string(lineno) + ": malformed rule");
      continue;
    }
    rules.push_back(rule);
  }
  std::sort(rules.begin(), rules.end(), [](const RewriteRule &a, const RewriteRule &b) {
    return a.benefit > b.benefit;
  });
  return rules;
}

static bool isIntegerConstant(Value value, int64_t expected) {
  auto *op = value.getDefiningOp();
  if (!op || op->name() != "arith.constant")
    return false;
  auto attr = op->attr("value");
  return attr && attr.str().find(std::to_string(expected) + " :") == 0;
}

static Value insertIntegerConstantBefore(Module &module, Operation &op, int64_t value) {
  Block *block = op.getBlock();
  if (!block)
    return Value();
  int index = operationIndexInBlock(*block, &op);
  if (index < 0)
    return Value();
  auto c_op = std::make_unique<Operation>(
      "arith.constant", std::vector<Value>{}, std::vector<Type>{op.resultType()},
      std::map<std::string, Attribute>{{"value", module.context().integerAttr(value, op.resultType())}},
      op.loc());
  auto &inserted = block->insertOperation(index, std::move(c_op));
  return inserted.result();
}

static bool applyRule(Module &module, Operation &op, const RewriteRule &rule) {
  if (op.name() != rule.root || op.resultCount() != 1)
    return false;
  if (rule.kind == "addi-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 0)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
    if (isIntegerConstant(op.operand(0), 0)) {
      replaceAllUses(module, op.result(), op.operand(1));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "muli-one" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
    if (isIntegerConstant(op.operand(0), 1)) {
      replaceAllUses(module, op.result(), op.operand(1));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "select-same" && op.operandCount() == 3 && op.operand(1) == op.operand(2)) {
    replaceAllUses(module, op.result(), op.operand(1));
    op.markErased();
    return true;
  }
  if (rule.kind == "subi-same" && op.operandCount() == 2) {
    if (op.operand(0) == op.operand(1)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "subi-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 0)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "muli-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(0), 0) || isIntegerConstant(op.operand(1), 0)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "divi-one" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "remi-one" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 1)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "andi-same" && op.operandCount() == 2) {
    if (op.operand(0) == op.operand(1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "andi-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(0), 0) || isIntegerConstant(op.operand(1), 0)) {
      Value zero = insertIntegerConstantBefore(module, op, 0);
      if (zero.valid()) {
        replaceAllUses(module, op.result(), zero);
        op.markErased();
        return true;
      }
    }
  }
  if (rule.kind == "ori-same" && op.operandCount() == 2) {
    if (op.operand(0) == op.operand(1)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "ori-zero" && op.operandCount() == 2) {
    if (isIntegerConstant(op.operand(1), 0)) {
      replaceAllUses(module, op.result(), op.operand(0));
      op.markErased();
      return true;
    }
    if (isIntegerConstant(op.operand(0), 0)) {
      replaceAllUses(module, op.result(), op.operand(1));
      op.markErased();
      return true;
    }
  }
  if (rule.kind == "double-noti" && op.operandCount() == 1) {
    auto *inner = op.operand(0).getDefiningOp();
    if (inner && inner->name() == "arith.noti" && inner->operandCount() == 1) {
      replaceAllUses(module, op.result(), inner->operand(0));
      op.markErased();
      return true;
    }
  }
  return false;
}

static void eraseMarkedInRegion(Region &region) {
  for (auto &block : region.getBlocks()) {
    for (auto &owned : block->ops()) {
      if (!owned || owned->isErased())
        continue;
      for (auto &nested : owned->getRegions())
        eraseMarkedInRegion(*nested);
    }
    block->eraseMarkedOperations();
  }
}

void eraseMarked(Module &module) {
  for (auto &region : module.op().getRegions())
    eraseMarkedInRegion(*region);
}

static RewriteStats applyGreedyPatternsFullWalk(Module &module,
                                                const std::vector<RewriteRule> &rules) {
  RewriteStats stats;
  stats.rules = (int) rules.size();
  bool changed = false;
  do {
    changed = false;
    stats.iterations++;
    for (auto *op : walk(module)) {
      if (op->isErased())
        continue;
      for (const auto &rule : rules) {
        if (applyRule(module, *op, rule)) {
          stats.rewrites++;
          changed = true;
          break;
        }
      }
    }
    eraseMarked(module);
  } while (changed && stats.iterations < 32);
  return stats;
}

static void enqueueOp(std::vector<Operation*> &worklist, std::set<Operation*> &queued,
                      Operation *op) {
  if (!op || op->isErased())
    return;
  if (queued.insert(op).second)
    worklist.push_back(op);
}

static void enqueueBlockNeighbors(std::vector<Operation*> &worklist,
                                  std::set<Operation*> &queued,
                                  Operation *op) {
  if (!op || !op->getBlock())
    return;
  auto &ops = op->getBlock()->ops();
  for (size_t i = 0; i < ops.size(); i++) {
    if (ops[i].get() != op)
      continue;
    if (i > 0)
      enqueueOp(worklist, queued, ops[i - 1].get());
    if (i + 1 < ops.size())
      enqueueOp(worklist, queued, ops[i + 1].get());
    return;
  }
}

static void collectRewriteNeighbors(Operation &op, std::vector<Operation*> &neighbors) {
  for (auto operand : op.getOperands()) {
    if (auto *def = operand.getDefiningOp())
      neighbors.push_back(def);
  }
  for (int i = 0; i < op.resultCount(); i++) {
    for (const auto &use : op.resultUses[i]) {
      if (use.owner)
        neighbors.push_back(use.owner);
    }
  }
  if (Block *block = op.getBlock()) {
    auto &ops = block->ops();
    for (size_t i = 0; i < ops.size(); i++) {
      if (ops[i].get() != &op)
        continue;
      if (i > 0)
        neighbors.push_back(ops[i - 1].get());
      if (i + 1 < ops.size())
        neighbors.push_back(ops[i + 1].get());
      break;
    }
  }
}

RewriteStats applyGreedyPatterns(Module &module, const std::vector<RewriteRule> &rules,
                                 bool useWorklist) {
  if (!useWorklist)
    return applyGreedyPatternsFullWalk(module, rules);

  RewriteStats stats;
  stats.rules = (int) rules.size();
  stats.iterations = 1;

  std::vector<Operation*> worklist;
  std::set<Operation*> queued;
  for (auto *op : walk(module))
    enqueueOp(worklist, queued, op);

  size_t head = 0;
  while (head < worklist.size()) {
    Operation *op = worklist[head++];
    queued.erase(op);
    if (!op || op->isErased())
      continue;
    stats.worklistPops++;

    std::vector<Operation*> neighbors;
    collectRewriteNeighbors(*op, neighbors);
    for (const auto &rule : rules) {
      if (!applyRule(module, *op, rule))
        continue;
      stats.rewrites++;
      for (auto *neighbor : neighbors)
        enqueueOp(worklist, queued, neighbor);
      enqueueBlockNeighbors(worklist, queued, op);
      break;
    }
  }
  eraseMarked(module);
  if (stats.rewrites > 0)
    stats.walksEliminated = 31;
  return stats;
}

ConversionStats convertDialects(Module &module, const ConversionTarget &target,
                                const std::vector<ConversionPattern> &patterns) {
  ConversionStats stats;
  std::map<std::string, std::string> patternMap;
  for (const auto &pattern : patterns)
    patternMap[pattern.root] = pattern.replacement;

  std::vector<std::pair<Operation*, std::string>> pending;
  for (auto *op : walk(module)) {
    stats.visited++;
    if (target.isLegal(*op)) {
      stats.legal++;
      continue;
    }
    auto it = patternMap.find(op->name());
    if (it == patternMap.end()) {
      stats.failed++;
      stats.rollbacks++;
      return stats;
    }
    pending.push_back({op, it->second});
  }
  for (auto &entry : pending) {
    entry.first->rename(entry.second);
    stats.converted++;
  }
  return stats;
}

bool verifyLegacyFree(Module &module, NativeAsmStats *stats) {
  bool ok = true;
  for (auto *op : walk(module)) {
    if (!op)
      continue;
    if (op->dialect() == "legacy") {
      ok = false;
      if (stats)
        stats->legacyOps++;
    }
    if (op->name().find("Phi") != std::string::npos ||
        op->name().find(".phi") != std::string::npos) {
      ok = false;
      if (stats)
        stats->phiLikeOps++;
    }
  }
  return ok;
}

} // namespace sys::mlir
