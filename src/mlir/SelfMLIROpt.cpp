#include "SelfMLIRInternal.h"
#include "Polyhedral.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace sys::mlir {

struct ModuleIndex {
  std::vector<Operation*> ops;
  std::vector<Operation*> globals;
  std::vector<Operation*> functions;
  std::map<std::string, std::vector<Operation*>> byName;
  std::map<std::string, std::vector<Operation*>> symbolAccesses;
  std::map<std::string, std::set<Operation*>> symbolFunctions;
  std::map<std::string, int> valueUseCounts;
};

static Operation *enclosingFunction(Operation *op) {
  if (!op)
    return nullptr;
  Block *currBlock = op->getBlock();
  while (currBlock) {
    Region *region = currBlock->getRegion();
    if (!region)
      break;
    Operation *parentOp = region->getParent();
    if (!parentOp)
      break;
    if (parentOp->name() == "sysy.func")
      return parentOp;
    currBlock = parentOp->getBlock();
  }
  return nullptr;
}

static ModuleIndex buildModuleIndex(Module &module) {
  ModuleIndex index;
  index.ops = walk(module);
  for (auto *op : index.ops) {
    if (!op || op->isErased())
      continue;
    index.byName[op->name()].push_back(op);
    if (op->name() == "sysy.global")
      index.globals.push_back(op);
    if (op->name() == "sysy.func")
      index.functions.push_back(op);
    for (auto operand : op->getOperands()) {
      if (operand.valid())
        index.valueUseCounts[valueKey(operand)]++;
    }
    if (op->name() == "sysy.load" || op->name() == "sysy.store" ||
        op->name() == "memref.load" || op->name() == "memref.store") {
      std::string symbol = symbolAttr(op->attr("symbol"));
      if (symbol.empty())
        symbol = symbolAttr(op->attr("sym_name"));
      if (!symbol.empty()) {
        index.symbolAccesses[symbol].push_back(op);
        if (auto *func = enclosingFunction(op))
          index.symbolFunctions[symbol].insert(func);
      }
    }
  }
  return index;
}

void runGlobalOpt(Module &module, SelfOptStats *stats) {
  const bool promoteGlobals = envEnabled("SISY_ENABLE_SELF_GLOBAL_PROMOTE", false);
  int64_t promoteLimit = 4096;
  if (const char *value = std::getenv("SISY_SELF_GLOBAL_PROMOTE_MAX_BYTES")) {
    try {
      promoteLimit = std::max<int64_t>(0, std::stoll(value));
    } catch (...) {
      promoteLimit = 4096;
    }
  }

  ModuleIndex index = buildModuleIndex(module);
  if (stats && index.globals.size() > 1)
    stats->walksEliminated += (int) index.globals.size() - 1;

  for (auto *global : index.globals) {
    std::string gName = symbolAttr(global->attr("symbol"));
    if (gName.empty())
      gName = symbolAttr(global->attr("sym_name"));
    if (gName.empty())
      continue;

    auto accessIt = index.symbolAccesses.find(gName);
    const std::vector<Operation*> emptyAccesses;
    const std::vector<Operation*> &accesses =
        accessIt == index.symbolAccesses.end() ? emptyAccesses : accessIt->second;
    auto funcIt = index.symbolFunctions.find(gName);
    const std::set<Operation*> emptyFunctions;
    const std::set<Operation*> &functions =
        funcIt == index.symbolFunctions.end() ? emptyFunctions : funcIt->second;

    if (accesses.empty()) {
      bool hasDirectUse = global->resultCount() > 0 &&
                          index.valueUseCounts[valueKey(global->result())] > 0;
      if (global->resultCount() == 0 || !hasDirectUse) {
        global->markErased();
        if (stats)
          stats->globalsErased++;
      }
      continue;
    }

    if (promoteGlobals && functions.size() == 1 &&
        memrefAllocationBytes(global->resultType()) <= promoteLimit) {
      Operation *func = *functions.begin();
      auto &regions = func->getRegions();
      if (!regions.empty() && !regions[0]->getBlocks().empty()) {
        auto &entry = *regions[0]->getBlocks()[0];
        Type storageType = global->resultType();
        auto &slot = entry.insertOperation(0, std::make_unique<Operation>(
            "sysy.alloca", std::vector<Value>{}, std::vector<Type>{storageType},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(gName)}},
            global->loc()));

        int remainingDirectUses = global->resultCount() > 0
                                      ? index.valueUseCounts[valueKey(global->result())]
                                      : 0;
        for (auto *op : accesses) {
          if (op->operandCount() > 0) {
            if (global->resultCount() > 0 && op->operand(0) == global->result())
              remainingDirectUses = std::max(0, remainingDirectUses - 1);
            op->setOperand(0, slot.result());
          } else {
            op->addOperand(slot.result());
          }
        }
        if (stats)
          stats->globalsPromoted++;
        bool hasDirectUse = global->resultCount() > 0 && remainingDirectUses > 0;
        if (global->resultCount() == 0 || !hasDirectUse) {
          global->markErased();
          if (stats)
            stats->globalsErased++;
        }
      }
    }
  }
}

void runReadonlyGlobalScalarPropagation(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_READONLY_GLOBALS", true))
    return;
  if (module.body().getBlocks().empty())
    return;

  struct ScalarGlobal {
    Value value;
    std::string symbol;
    Attribute initAttr;
    bool hasInit = false;
    bool invalid = false;
  };

  std::map<std::string, ScalarGlobal> globalsByKey;
  std::map<std::string, std::string> keyBySymbol;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.global" ||
        op->resultCount() == 0 || !isScalarWordMemref(op->resultType()))
      continue;
    if (op->resultType().str().find("xi32") == std::string::npos)
      continue;
    std::string symbol = symbolAttr(op->attr("symbol"));
    if (symbol.empty())
      symbol = symbolAttr(op->attr("sym_name"));
    if (symbol.empty())
      continue;
    std::string key = valueKey(op->result());
    ScalarGlobal info;
    info.value = op->result();
    info.symbol = symbol;
    globalsByKey[key] = info;
    keyBySymbol[symbol] = key;
  }
  if (globalsByKey.empty())
    return;

  auto zeroIndices = [](Operation &op) {
    int start = -1;
    if (op.name() == "sysy.load" || op.name() == "memref.load")
      start = 1;
    else if (op.name() == "sysy.store" || op.name() == "memref.store")
      start = 2;
    if (start < 0)
      return false;
    for (int i = start; i < op.operandCount(); i++) {
      int64_t index = 0;
      if (!constantIntegerValue(op.operand(i), index) || index != 0)
        return false;
    }
    return true;
  };

  auto candidateKeyForAccess = [&](Operation &op, bool store) -> std::string {
    int baseIndex = store ? 1 : 0;
    if (op.operandCount() > baseIndex) {
      std::string key = valueKey(op.operand(baseIndex));
      if (globalsByKey.count(key) != 0)
        return key;
    }
    std::string symbol = symbolAttr(op.attr("symbol"));
    if (symbol.empty())
      symbol = symbolAttr(op.attr("sym_name"));
    auto it = keyBySymbol.find(symbol);
    return it == keyBySymbol.end() ? "" : it->second;
  };

  Block *moduleBlock = module.body().getBlocks()[0].get();
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    bool isStore = op->name() == "sysy.store" || op->name() == "memref.store";
    if (!isStore)
      continue;
    std::string key = candidateKeyForAccess(*op, true);
    if (key.empty())
      continue;
    auto &global = globalsByKey[key];
    bool isModuleInit = op->getBlock() == moduleBlock && zeroIndices(*op);
    int64_t ignored = 0;
    auto *def = op->operandCount() > 0 ? op->operand(0).getDefiningOp() : nullptr;
    if (isModuleInit && !global.hasInit && op->operandCount() > 0 &&
        constantIntegerValue(op->operand(0), ignored) && def && def->attr("value")) {
      global.initAttr = def->attr("value");
      global.hasInit = true;
      continue;
    }
    global.invalid = true;
  }

  for (auto &kv : globalsByKey) {
    auto uses = usesOf(module, kv.second.value);
    for (const auto &use : uses) {
      Operation *owner = use.owner;
      if (!owner || owner->isErased())
        continue;
      bool allowedLoad = (owner->name() == "sysy.load" || owner->name() == "memref.load") &&
                         use.operandIndex == 0;
      bool allowedStore = (owner->name() == "sysy.store" || owner->name() == "memref.store") &&
                          use.operandIndex == 1;
      if (!allowedLoad && !allowedStore) {
        kv.second.invalid = true;
        break;
      }
    }
  }

  int replaced = 0;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() ||
        (op->name() != "sysy.load" && op->name() != "memref.load") ||
        op->resultCount() != 1 || !hasScalarHome(op->resultType()))
      continue;
    std::string key = candidateKeyForAccess(*op, false);
    if (key.empty())
      continue;
    auto it = globalsByKey.find(key);
    if (it == globalsByKey.end() || it->second.invalid || !zeroIndices(*op))
      continue;
    Block *block = op->getBlock();
    if (!block)
      continue;
    int index = operationIndexInBlock(*block, op);
    if (index < 0)
      continue;
    Attribute valueAttr = it->second.hasInit
                              ? it->second.initAttr
                              : module.context().integerAttr(0, op->resultType());
    auto constant = std::make_unique<Operation>(
        "arith.constant", std::vector<Value>{}, std::vector<Type>{op->resultType()},
        std::map<std::string, Attribute>{{"value", valueAttr}}, op->loc());
    auto &inserted = block->insertOperation((std::size_t) index, std::move(constant));
    replaceAllUses(module, op->result(), inserted.result());
    op->markErased();
    replaced++;
  }
  if (replaced > 0) {
    if (stats)
      stats->readonlyGlobalConstants += replaced;
    eraseMarked(module);
  }
}

static std::string memoryLocationKey(Operation &op);

static std::string memoryExprKey(Value value, int depth = 0) {
  if (!value.valid())
    return "";
  if (depth > 8)
    return "v:" + valueKey(value);
  int64_t imm = 0;
  if (constantIntegerValue(value, imm))
    return "c:" + std::to_string(imm);

  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return "v:" + valueKey(value);

  if ((def->name() == "sysy.load" || def->name() == "memref.load") &&
      def->resultCount() == 1) {
    std::string loc = memoryLocationKey(*def);
    if (!loc.empty())
      return "load(" + loc + ")";
  }

  auto binaryExpr = [&](const char *kind, bool commutative) -> std::string {
    if (def->operandCount() != 2)
      return "";
    std::string lhs = memoryExprKey(def->operand(0), depth + 1);
    std::string rhs = memoryExprKey(def->operand(1), depth + 1);
    if (lhs.empty() || rhs.empty())
      return "";
    if (commutative && rhs < lhs)
      std::swap(lhs, rhs);
    return std::string(kind) + "(" + lhs + "," + rhs + ")";
  };

  if (def->name() == "arith.addi" || def->name() == "rv_machine.addw" ||
      def->name() == "arm_machine.add")
    return binaryExpr("add", true);
  if (def->name() == "arith.muli" || def->name() == "rv_machine.mulw" ||
      def->name() == "arm_machine.mul")
    return binaryExpr("mul", true);
  if (def->name() == "arith.subi" || def->name() == "rv_machine.subw" ||
      def->name() == "arm_machine.sub")
    return binaryExpr("sub", false);

  return "v:" + valueKey(value);
}

static bool memoryKeyDependsOnLocation(const std::string &key,
                                       const std::string &location) {
  if (key.empty() || location.empty())
    return false;
  return key.find("load(" + location + ")") != std::string::npos;
}

static std::string memoryLocationKey(Operation &op) {
  auto baseKey = [&]() -> std::string {
    int baseIndex = -1;
    if (op.name() == "sysy.load" || op.name() == "memref.load")
      baseIndex = 0;
    else if (op.name() == "sysy.store" || op.name() == "memref.store")
      baseIndex = 1;
    if (baseIndex >= 0 && op.operandCount() > baseIndex)
      return "v:" + valueKey(op.operand(baseIndex));
    std::string sym = symbolAttr(op.attr("symbol"));
    if (sym.empty())
      sym = symbolAttr(op.attr("sym_name"));
    return sym.empty() ? "" : "s:" + sym;
  };

  std::string key = baseKey();
  if (key.empty())
    return "";
  if (op.name() == "memref.load") {
    for (int i = 1; i < op.operandCount(); i++)
      key += "," + memoryExprKey(op.operand(i));
  } else if (op.name() == "memref.store") {
    for (int i = 2; i < op.operandCount(); i++)
      key += "," + memoryExprKey(op.operand(i));
  }
  return key;
}

static std::string memoryBaseKey(Operation &op) {
  if (op.name() == "sysy.load" || op.name() == "memref.load") {
    if (op.operandCount() > 0)
      return "v:" + valueKey(op.operand(0));
  } else if (op.name() == "sysy.store" || op.name() == "memref.store") {
    if (op.operandCount() > 1)
      return "v:" + valueKey(op.operand(1));
  }
  std::string sym = symbolAttr(op.attr("symbol"));
  if (sym.empty())
    sym = symbolAttr(op.attr("sym_name"));
  return sym.empty() ? "" : "s:" + sym;
}

static bool memoryKeyMayShareBase(const std::string &locationKey,
                                  const std::string &baseKey) {
  if (locationKey.empty() || baseKey.empty())
    return false;
  return locationKey == baseKey || locationKey.rfind(baseKey + ",", 0) == 0;
}

static bool isLocalAllocaOp(Operation *op) {
  return op && (op->name() == "sysy.alloca" || op->name() == "memref.alloca") &&
         op->resultCount() == 1;
}

static void collectAllocaUseInfo(Operation &op, const std::set<std::string> &allocas,
                                 std::set<std::string> &loaded,
                                 std::set<std::string> &escaped) {
  if (op.isErased())
    return;
  auto classifyOperand = [&](int index, Value operand) {
    std::string key = valueKey(operand);
    if (allocas.count(key) == 0)
      return;
    bool baseLoad = (op.name() == "sysy.load" || op.name() == "memref.load") &&
                    index == 0;
    bool baseStore = (op.name() == "sysy.store" || op.name() == "memref.store") &&
                     index == 1;
    if (baseLoad)
      loaded.insert(key);
    else if (!baseStore)
      escaped.insert(key);
  };
  for (int i = 0; i < op.operandCount(); i++)
    classifyOperand(i, op.operand(i));
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child)
          collectAllocaUseInfo(*child, allocas, loaded, escaped);
      }
    }
  }
}

static void eraseDeadLocalStoresInFunction(Operation &func, SelfOptStats *stats) {
  if (func.getRegions().empty() || func.getRegions()[0]->getBlocks().empty())
    return;
  std::vector<Operation*> nestedOps;
  std::function<void(Operation&)> collect = [&](Operation &op) {
    nestedOps.push_back(&op);
    for (auto &region : op.getRegions()) {
      for (auto &block : region->getBlocks()) {
        for (auto &child : block->ops()) {
          if (child && !child->isErased())
            collect(*child);
        }
      }
    }
  };
  collect(func);

  std::set<std::string> allocas;
  for (auto *op : nestedOps) {
    if (isLocalAllocaOp(op))
      allocas.insert(valueKey(op->result()));
  }
  if (allocas.empty())
    return;

  std::set<std::string> loaded;
  std::set<std::string> escaped;
  collectAllocaUseInfo(func, allocas, loaded, escaped);

  std::set<std::string> deadAllocas;
  for (const auto &key : allocas) {
    if (loaded.count(key) == 0 && escaped.count(key) == 0)
      deadAllocas.insert(key);
  }
  if (deadAllocas.empty())
    return;

  for (auto *op : nestedOps) {
    if (!op || op->isErased())
      continue;
    if ((op->name() == "sysy.store" || op->name() == "memref.store") &&
        op->operandCount() >= 2 &&
        deadAllocas.count(valueKey(op->operand(1))) != 0) {
      op->markErased();
      if (stats)
        stats->memoryRemovedStores++;
    } else if (isLocalAllocaOp(op) && deadAllocas.count(valueKey(op->result())) != 0) {
      op->markErased();
    }
  }
}

static void runBlockMemoryOpt(Module &module, Block &block, SelfOptStats *stats) {
  if (stats)
    stats->memoryBlocks++;
  std::map<std::string, Value> activeStores;
  std::map<std::string, Operation*> activeStoreOps;
  std::map<std::string, std::string> loadOrigins;

  auto loadOriginOf = [&](Value value) -> std::string {
    auto it = loadOrigins.find(valueKey(value));
    return it == loadOrigins.end() ? "" : it->second;
  };
  auto invalidateLoadOriginsAfterStore = [&](const std::string &storeKey,
                                             const std::string &base,
                                             Value storedValue) {
    std::string storedOrigin = loadOriginOf(storedValue);
    std::vector<std::string> toErase;
    for (const auto &kv : loadOrigins) {
      const std::string &origin = kv.second;
      if (origin == storeKey ||
          (!base.empty() && memoryKeyMayShareBase(origin, base))) {
        // If a potentially aliasing store writes back the exact value that was
        // originally loaded from this location, the old loaded value remains a
        // valid representation even when the two expressions alias. Otherwise
        // the load-origin fact is no longer safe.
        if (storedOrigin != origin)
          toErase.push_back(kv.first);
      }
    }
    for (const auto &key : toErase)
      loadOrigins.erase(key);
  };
  auto invalidateFactsDependingOnLocation = [&](const std::string &locationKey) {
    if (locationKey.empty())
      return;
    std::vector<std::string> activeToErase;
    for (const auto &kv : activeStores) {
      if (memoryKeyDependsOnLocation(kv.first, locationKey))
        activeToErase.push_back(kv.first);
    }
    for (const auto &key : activeToErase) {
      activeStores.erase(key);
      activeStoreOps.erase(key);
    }

    std::vector<std::string> originsToErase;
    for (const auto &kv : loadOrigins) {
      if (memoryKeyDependsOnLocation(kv.second, locationKey))
        originsToErase.push_back(kv.first);
    }
    for (const auto &key : originsToErase)
      loadOrigins.erase(key);
  };

  for (auto &owned : block.ops()) {
    auto &op = *owned;
    if (op.isErased())
      continue;

    if (!op.getRegions().empty()) {
      activeStores.clear();
      activeStoreOps.clear();
      loadOrigins.clear();
      continue;
    }

    if (op.name() == "sysy.call") {
      std::vector<std::string> keysToInvalidate;
      for (const auto &kv : activeStores) {
        keysToInvalidate.push_back(kv.first);
      }
      for (const auto &k : keysToInvalidate) {
        activeStores.erase(k);
        activeStoreOps.erase(k);
      }
      loadOrigins.clear();
      continue;
    }

    if (op.name() == "sysy.store" || op.name() == "memref.store") {
      std::string key = memoryLocationKey(op);
      if (key.empty())
        continue;

      std::string base = memoryBaseKey(op);
      if (op.operandCount() >= 1 && loadOriginOf(op.operand(0)) == key) {
        op.markErased();
        if (stats)
          stats->memoryRemovedStores++;
        continue;
      }
      bool scalarLocationStore =
          op.operandCount() == 2 && isScalarWordMemref(op.operand(1).type());
      if (scalarLocationStore)
        invalidateFactsDependingOnLocation(key);
      if (op.operandCount() >= 1)
        invalidateLoadOriginsAfterStore(key, base, op.operand(0));

      std::vector<std::string> keysToInvalidate;
      for (const auto &kv : activeStores) {
        if (kv.first != key && !base.empty() &&
            (kv.first == base || kv.first.rfind(base + ",", 0) == 0)) {
          keysToInvalidate.push_back(kv.first);
        }
      }
      for (const auto &k : keysToInvalidate) {
        activeStores.erase(k);
        activeStoreOps.erase(k);
      }

      if (activeStores.count(key) > 0) {
        auto *prevOp = activeStoreOps[key];
        if (prevOp && !prevOp->isErased()) {
          prevOp->markErased();
          if (stats)
            stats->memoryRemovedStores++;
        }
      }

      Value storedVal = op.operand(0);
      activeStores[key] = storedVal;
      activeStoreOps[key] = &op;
    }

    if (op.name() == "sysy.load" || op.name() == "memref.load") {
      std::string key = memoryLocationKey(op);
      if (key.empty())
        continue;

      if (activeStores.count(key) > 0) {
        Value storedVal = activeStores[key];
        replaceAllUses(module, op.result(), storedVal);
        op.markErased();
        if (stats)
          stats->memoryForwardedLoads++;
      } else {
        loadOrigins[valueKey(op.result())] = key;
      }
    }
  }
}

static void runMemoryOptInRegion(Module &module, Region &region, SelfOptStats *stats) {
  for (auto &block : region.getBlocks()) {
    runBlockMemoryOpt(module, *block, stats);
    for (auto &owned : block->ops()) {
      if (!owned || owned->isErased())
        continue;
      for (auto &nested : owned->getRegions())
        runMemoryOptInRegion(module, *nested, stats);
    }
  }
}

void runMemoryOpt(Module &module, SelfOptStats *stats, bool enableDeadLocalStores) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_MEMOPT");
  if (enabled && std::string(enabled) == "0")
    return;
  for (auto *op : walk(module)) {
    if (op && !op->isErased() && op->name() == "sysy.func") {
      for (auto &region : op->getRegions()) {
        runMemoryOptInRegion(module, *region, stats);
      }
      if (enableDeadLocalStores)
        eraseDeadLocalStoresInFunction(*op, stats);
    }
  }
  eraseMarked(module);
}

namespace {

static bool isMLIRLoadOp(Operation *op) {
  return op && (op->name() == "sysy.load" || op->name() == "memref.load");
}

static bool isMLIRStoreOp(Operation *op) {
  return op && (op->name() == "sysy.store" || op->name() == "memref.store");
}

static bool blockIsNestedInOp(Block &block, const std::string &name) {
  Block *curr = &block;
  while (curr) {
    Region *region = curr->getRegion();
    Operation *parent = region ? region->getParent() : nullptr;
    if (!parent)
      return false;
    if (parent->name() == name)
      return true;
    curr = parent->getBlock();
  }
  return false;
}

static bool loadFromSlot(Operation *op, Value slot) {
  return isMLIRLoadOp(op) && op->operandCount() > 0 && op->operand(0) == slot;
}

static bool storeToSlot(Operation *op, Value slot) {
  return isMLIRStoreOp(op) && op->operandCount() >= 2 && op->operand(1) == slot;
}

static std::vector<Type> resultTypesOf(Operation &op) {
  std::vector<Type> types;
  for (int i = 0; i < op.resultCount(); i++)
    types.push_back(op.resultType(i));
  return types;
}

static std::vector<Value> remapOperandsForClone(
    Operation &op, const std::map<std::string, Value> &valueMap) {
  std::vector<Value> operands;
  operands.reserve((std::size_t) op.operandCount());
  for (auto operand : op.getOperands()) {
    auto it = valueMap.find(valueKey(operand));
    operands.push_back(it == valueMap.end() ? operand : it->second);
  }
  return operands;
}

static std::string memAccessKey(Operation &op) {
  if (isMLIRLoadOp(&op)) {
    if (op.operandCount() == 0)
      return "";
    std::string key = op.name() + "|" + op.resultType().str() + "|" +
                      valueKey(op.operand(0));
    for (int i = 1; i < op.operandCount(); i++)
      key += "|" + valueKey(op.operand(i));
    return key;
  }
  if (isMLIRStoreOp(&op)) {
    if (op.operandCount() < 2)
      return "";
    std::string key = op.name() + "|" + op.operand(0).type().str() + "|" +
                      valueKey(op.operand(1));
    for (int i = 2; i < op.operandCount(); i++)
      key += "|" + valueKey(op.operand(i));
    return key;
  }
  return "";
}

static std::string memAccessBaseKey(Operation &op) {
  if (isMLIRLoadOp(&op) && op.operandCount() > 0)
    return valueKey(op.operand(0));
  if (isMLIRStoreOp(&op) && op.operandCount() >= 2)
    return valueKey(op.operand(1));
  return "";
}

static bool opTreeHasUnsafeLoopUnrollControl(Operation &op, Value ivSlot,
                                             std::vector<Operation*> &ivStores) {
  if (op.isErased())
    return false;
  if (op.name() == "sysy.call" || op.name() == "sysy.return" ||
      op.name() == "scf.return" || op.name() == "sysy.break" ||
      op.name() == "sysy.continue" || op.name() == "scf.if")
    return true;
  if (storeToSlot(&op, ivSlot))
    ivStores.push_back(&op);
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && opTreeHasUnsafeLoopUnrollControl(*child, ivSlot, ivStores))
          return true;
  return false;
}

static bool addOneFromSlot(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased() ||
      (op->name() != "arith.addi" && op->name() != "rv_machine.addw" &&
       op->name() != "arm_machine.add") ||
      op->operandCount() != 2)
    return false;
  int64_t imm = 0;
  Operation *load = op->operand(0).getDefiningOp();
  if (loadFromSlot(load, slot) && constantIntegerValue(op->operand(1), imm) && imm == 1)
    return true;
  load = op->operand(1).getDefiningOp();
  return loadFromSlot(load, slot) && constantIntegerValue(op->operand(0), imm) && imm == 1;
}

static Operation *findConditionOp(Region &region) {
  if (region.getBlocks().empty())
    return nullptr;
  for (auto &owned : region.getBlocks()[0]->ops())
    if (owned && !owned->isErased() && owned->name() == "scf.condition")
      return owned.get();
  return nullptr;
}

struct ConstantWhileInfo {
  bool valid = false;
  Value ivSlot;
  int64_t init = 0;
  int64_t bound = 0;
  int64_t tripCount = 0;
  Operation *stepStore = nullptr;
};

static ConstantWhileInfo classifySmallConstantWhile(Operation &loop) {
  ConstantWhileInfo info;
  if (loop.name() != "scf.while" || loop.getRegions().size() != 2 ||
      loop.getRegions()[0]->getBlocks().size() != 1 ||
      loop.getRegions()[1]->getBlocks().size() != 1)
    return info;
  Operation *condition = findConditionOp(*loop.getRegions()[0]);
  if (!condition || condition->operandCount() != 1)
    return info;
  Operation *cmp = condition->operand(0).getDefiningOp();
  if (!cmp || cmp->isErased() ||
      (cmp->name() != "arith.cmpi" && cmp->name() != "rv_machine.cmp" &&
       cmp->name() != "arm_machine.cmp") ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "lt")
    return info;
  Operation *load = cmp->operand(0).getDefiningOp();
  if (!isMLIRLoadOp(load) || load->operandCount() == 0 ||
      !isScalarWordMemref(load->operand(0).type()))
    return info;
  int64_t bound = 0;
  if (!constantIntegerValue(cmp->operand(1), bound))
    return info;
  Value ivSlot = load->operand(0);

  Block *parent = loop.getBlock();
  if (!parent)
    return info;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return info;
  bool foundInit = false;
  int64_t init = 0;
  for (int i = loopIndex - 1; i >= 0; i--) {
    Operation *prev = parent->ops()[(std::size_t) i].get();
    if (!prev || prev->isErased())
      continue;
    if (storeToSlot(prev, ivSlot) && constantIntegerValue(prev->operand(0), init)) {
      foundInit = true;
      break;
    }
    if (prev->name() == "scf.while" || prev->name() == "affine.for" ||
        prev->name() == "sysy.call")
      break;
  }
  if (!foundInit)
    return info;

  Block &body = *loop.getRegions()[1]->getBlocks()[0];
  std::vector<Operation*> ivStores;
  for (auto &owned : body.ops()) {
    if (owned && opTreeHasUnsafeLoopUnrollControl(*owned, ivSlot, ivStores))
      return info;
  }
  if (ivStores.size() != 1 || ivStores[0]->getBlock() != &body ||
      !addOneFromSlot(ivStores[0]->operand(0), ivSlot))
    return info;

  int64_t trip = bound - init;
  if (trip < 0 || trip > 7)
    return info;
  info.valid = true;
  info.ivSlot = ivSlot;
  info.init = init;
  info.bound = bound;
  info.tripCount = trip;
  info.stepStore = ivStores[0];
  return info;
}

static std::unique_ptr<Operation> cloneForUnrolledIteration(
    Module &module, Operation &op, std::map<std::string, Value> &valueMap,
    const std::set<Operation*> &skipOps, Value ivSlot, int64_t ivValue) {
  if (skipOps.count(&op) || op.isErased())
    return nullptr;

  bool replaceIvLoad = loadFromSlot(&op, ivSlot) && op.resultCount() == 1 &&
                       isI32Like(op.resultType());
  std::string clonedName = replaceIvLoad ? "arith.constant" : op.name();
  std::vector<Value> operands = replaceIvLoad ? std::vector<Value>{}
                                              : remapOperandsForClone(op, valueMap);
  std::map<std::string, Attribute> attrs = op.attrs();
  if (replaceIvLoad)
    attrs = {{"value", module.context().integerAttr(ivValue, op.resultType())}};
  auto cloned = std::make_unique<Operation>(clonedName, operands, resultTypesOf(op),
                                            attrs, op.loc());
  for (auto &region : op.getRegions()) {
    Region &newRegion = cloned->addRegion();
    for (auto &block : region->getBlocks()) {
      Block &newBlock = newRegion.addBlock();
      for (auto &arg : block->args()) {
        BlockArgument &newArg = newBlock.addArgument(arg->type(), arg->loc(), arg->name());
        valueMap[valueKey(arg->value())] = newArg.value();
      }
      for (auto &child : block->ops()) {
        if (!child || child->isErased())
          continue;
        auto childClone = cloneForUnrolledIteration(module, *child, valueMap,
                                                    skipOps, ivSlot, ivValue);
        if (!childClone)
          continue;
        Operation &inserted = newBlock.addOperation(std::move(childClone));
        for (int i = 0; i < child->resultCount(); i++)
          valueMap[valueKey(child->result(i))] = inserted.result(i);
      }
    }
  }
  return cloned;
}

static bool unrollSmallConstantWhile(Module &module, Operation &loop,
                                     const ConstantWhileInfo &info) {
  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;
  Block &body = *loop.getRegions()[1]->getBlocks()[0];
  std::vector<Operation*> bodyOps;
  for (auto &owned : body.ops())
    if (owned && !owned->isErased())
      bodyOps.push_back(owned.get());

  std::set<Operation*> skipOps;
  skipOps.insert(info.stepStore);
  for (Operation *op : bodyOps)
    if (op && op->name() == "scf.yield")
      skipOps.insert(op);

  std::size_t insertIndex = (std::size_t) loopIndex;
  for (int64_t iter = 0; iter < info.tripCount; iter++) {
    std::map<std::string, Value> valueMap;
    int64_t iv = info.init + iter;
    for (Operation *op : bodyOps) {
      if (!op || skipOps.count(op))
        continue;
      auto cloned = cloneForUnrolledIteration(module, *op, valueMap, skipOps,
                                              info.ivSlot, iv);
      if (!cloned)
        continue;
      Operation &inserted = parent->insertOperation(insertIndex++, std::move(cloned));
      for (int i = 0; i < op->resultCount(); i++)
        valueMap[valueKey(op->result(i))] = inserted.result(i);
    }
  }
  loop.markErased();
  return true;
}

static void runLoopAddressIVInRegion(Module &module, Region &region, SelfOptStats *stats);

static void runLoopAddressIVInBlock(Module &module, Block &block, SelfOptStats *stats) {
  std::map<std::string, Value> loadCache;
  std::map<std::string, std::set<std::string>> keysByBase;
  bool scalarSlotCSE = envEnabled("SISY_ENABLE_SELF_SCALAR_LOAD_CSE", true) &&
                       blockIsNestedInOp(block, "scf.while");

  auto invalidateAll = [&]() {
    loadCache.clear();
    keysByBase.clear();
  };
  auto cacheLoad = [&](const std::string &key, const std::string &base, Value value) {
    loadCache[key] = value;
    keysByBase[base].insert(key);
  };
  auto invalidateBaseAfterStore = [&](const std::string &base, Value storedValue) {
    auto baseIt = keysByBase.find(base);
    if (baseIt == keysByBase.end())
      return;
    std::vector<std::string> eraseKeys;
    for (const auto &key : baseIt->second) {
      auto cacheIt = loadCache.find(key);
      if (cacheIt == loadCache.end() || cacheIt->second != storedValue)
        eraseKeys.push_back(key);
    }
    for (const auto &key : eraseKeys) {
      loadCache.erase(key);
      baseIt->second.erase(key);
    }
    if (baseIt->second.empty())
      keysByBase.erase(baseIt);
  };

  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    bool cacheableLoad = op->name() == "memref.load" ||
                         (scalarSlotCSE && op->name() == "sysy.load");
    if (cacheableLoad && op->resultCount() == 1 &&
        !isMemrefType(op->resultType())) {
      std::string key = memAccessKey(*op);
      std::string base = memAccessBaseKey(*op);
      auto it = loadCache.find(key);
      if (!key.empty() && it != loadCache.end() && it->second.valid()) {
        replaceAllUses(module, op->result(), it->second);
        op->markErased();
        if (stats) {
          stats->loopAddressCSE++;
          stats->addrIvRewrites++;
        }
        continue;
      }
      if (!key.empty() && !base.empty())
        cacheLoad(key, base, op->result());
      continue;
    }

    bool cacheInvalidatingStore = op->name() == "memref.store" ||
                                  (scalarSlotCSE && op->name() == "sysy.store");
    if (cacheInvalidatingStore && op->operandCount() >= 2) {
      std::string key = memAccessKey(*op);
      std::string base = memAccessBaseKey(*op);
      auto cached = loadCache.find(key);
      if (!key.empty() && cached != loadCache.end() && cached->second == op->operand(0)) {
        op->markErased();
        if (stats) {
          stats->memoryRemovedStores++;
          stats->loopAddressCSE++;
          stats->addrIvRewrites++;
        }
        continue;
      }
      if (!base.empty())
        invalidateBaseAfterStore(base, op->operand(0));
      continue;
    }

    if (op->name() == "sysy.call") {
      invalidateAll();
      continue;
    }

    if (!op->getRegions().empty()) {
      for (auto &region : op->getRegions())
        runLoopAddressIVInRegion(module, *region, stats);
      invalidateAll();
    }
  }
}

static void runLoopAddressIVInRegion(Module &module, Region &region, SelfOptStats *stats) {
  for (auto &block : region.getBlocks())
    runLoopAddressIVInBlock(module, *block, stats);
}

static bool opTreeUsesValue(Operation &op, Value needle) {
  if (op.isErased())
    return false;
  for (auto operand : op.getOperands())
    if (operand == needle)
      return true;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && opTreeUsesValue(*child, needle))
          return true;
  return false;
}

static bool opHasDirectValueUse(Operation &op, Value needle) {
  for (auto operand : op.getOperands())
    if (operand == needle)
      return true;
  return false;
}

static bool opResultIsUnused(Operation &op) {
  for (int i = 0; i < op.resultCount(); i++) {
    if ((int) op.resultUses.size() > i && !op.resultUses[(std::size_t) i].empty())
      return false;
  }
  return true;
}

static bool isPureDeadIvBump(Operation &op, Value iv) {
  return op.getRegions().empty() && opHasDirectValueUse(op, iv) && opResultIsUnused(op) &&
         (op.name() == "arith.addi" || op.name() == "rv_machine.addw" ||
          op.name() == "arm_machine.add");
}

static void collectLocalAllocas(Operation &op, std::set<std::string> &allocas) {
  if (op.isErased())
    return;
  if ((op.name() == "sysy.alloca" || op.name() == "memref.alloca") &&
      op.resultCount() == 1)
    allocas.insert(valueKey(op.result()));
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child)
          collectLocalAllocas(*child, allocas);
}

static bool collectRepeatReductionStores(Operation &op,
                                         const std::set<std::string> &localAllocas,
                                         std::map<std::string, Value> &outerStores) {
  if (op.isErased())
    return true;
  if (op.name() == "sysy.call" || op.name() == "sysy.return" ||
      op.name() == "scf.return" || op.name() == "sysy.break" ||
      op.name() == "sysy.continue" || op.name() == "memref.store" ||
      op.name() == "scf.if" || op.name() == "arith.divi" ||
      op.name() == "arith.remi" || op.name() == "rv_machine.divw" ||
      op.name() == "rv_machine.remw" || op.name() == "arm_machine.sdiv")
    return false;
  if (op.name() == "sysy.store") {
    if (op.operandCount() < 2 || !isScalarWordMemref(op.operand(1).type()))
      return false;
    std::string slot = valueKey(op.operand(1));
    if (!localAllocas.count(slot))
      outerStores[slot] = op.operand(1);
  }
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && !collectRepeatReductionStores(*child, localAllocas, outerStores))
          return false;
  return true;
}

static bool valueDependsOnSlot(Value value, Value slot, int depth = 0) {
  if (!value.valid() || depth > 24)
    return false;
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased())
    return false;
  if (loadFromSlot(op, slot))
    return true;
  for (auto operand : op->getOperands())
    if (valueDependsOnSlot(operand, slot, depth + 1))
      return true;
  return false;
}

static bool valueUsesOnlyAllowedRepeatOps(Value value, Value accumulator,
                                          int depth = 0) {
  if (!value.valid() || depth > 48)
    return false;
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased())
    return true;
  if (loadFromSlot(op, accumulator))
    return true;
  if (op->getRegions().size() != 0)
    return false;
  const std::string &name = op->name();
  if (name != "arith.constant" && name != "arith.addi" &&
      name != "arith.subi" && name != "arith.muli" &&
      name != "rv_machine.li" && name != "rv_machine.addw" &&
      name != "rv_machine.subw" && name != "rv_machine.mulw" &&
      name != "rv_machine.neg" && name != "arm_machine.mov" &&
      name != "arm_machine.add" && name != "arm_machine.sub" &&
      name != "arm_machine.mul" && name != "memref.load" &&
      name != "sysy.load")
    return false;
  for (auto operand : op->getOperands())
    if (!valueUsesOnlyAllowedRepeatOps(operand, accumulator, depth + 1))
      return false;
  return true;
}

static bool storeIsAccumulatorAdd(Operation &store, Value accumulator) {
  if (!storeToSlot(&store, accumulator) || store.operandCount() < 1)
    return false;
  Value stored = store.operand(0);
  Operation *add = stored.getDefiningOp();
  if (!add || add->isErased() ||
      (add->name() != "arith.addi" && add->name() != "rv_machine.addw" &&
       add->name() != "arm_machine.add") ||
      add->operandCount() != 2)
    return false;

  bool hasAccumulatorLoad = false;
  bool incrementIndependent = false;
  for (int i = 0; i < 2; i++) {
    Value operand = add->operand(i);
    Operation *load = operand.getDefiningOp();
    if (loadFromSlot(load, accumulator)) {
      hasAccumulatorLoad = true;
      continue;
    }
    if (!valueDependsOnSlot(operand, accumulator) &&
        valueUsesOnlyAllowedRepeatOps(operand, accumulator))
      incrementIndependent = true;
  }
  return hasAccumulatorLoad && incrementIndependent;
}

static bool opTreeAccumulatorStoresAreLinearAdds(Operation &op, Value accumulator) {
  if (op.isErased())
    return true;
  if (storeToSlot(&op, accumulator) && !storeIsAccumulatorAdd(op, accumulator))
    return false;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && !opTreeAccumulatorStoresAreLinearAdds(*child, accumulator))
          return false;
  return true;
}

static bool opTreeLoadsFromSlot(Operation &op, Value slot) {
  if (op.isErased())
    return false;
  if (loadFromSlot(&op, slot))
    return true;
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && opTreeLoadsFromSlot(*child, slot))
          return true;
  return false;
}

static bool slotLoadedAfter(Operation &loop, Value slot) {
  Block *block = loop.getBlock();
  if (!block)
    return true;
  bool seen = false;
  for (auto &owned : block->ops()) {
    if (!owned || owned->isErased())
      continue;
    if (owned.get() == &loop) {
      seen = true;
      continue;
    }
    if (seen && opTreeLoadsFromSlot(*owned, slot))
      return true;
  }
  return false;
}

struct RepeatReductionInfo {
  bool valid = false;
  Value lower;
  Value upper;
  Value step;
  Value accumulatorSlot;
  std::set<Operation*> skipOps;
};

struct LinearModRecurrenceInfo {
  bool valid = false;
  Value lower;
  Value upper;
  Value slot;
  int64_t increment = 0;
  int64_t modulus = 0;
  int64_t maxTrip = 0;
};

static bool positiveUpperBound(Value value, int64_t &bound) {
  int64_t imm = 0;
  if (constantIntegerValue(value, imm) && imm >= 0) {
    bound = imm;
    return true;
  }
  Operation *op = value.getDefiningOp();
  if (!op || op->isErased() || op->operandCount() != 2)
    return false;
  if (op->name() != "arith.remi" && op->name() != "rv_machine.remw" &&
      op->name() != "arm_machine.srem")
    return false;
  if (!constantIntegerValue(op->operand(1), imm) || imm <= 0)
    return false;
  bound = imm - 1;
  return true;
}

static bool constantValueOperand(Value value, int64_t &imm) {
  return constantIntegerValue(value, imm);
}

static bool classifyLinearModStore(Operation &store, Value &slot,
                                   int64_t &increment, int64_t &modulus) {
  if (!isMLIRStoreOp(&store) || store.operandCount() < 2 ||
      !isScalarWordMemref(store.operand(1).type()))
    return false;
  Operation *rem = store.operand(0).getDefiningOp();
  if (!rem || rem->isErased() ||
      (rem->name() != "arith.remi" && rem->name() != "rv_machine.remw" &&
       rem->name() != "arm_machine.srem") ||
      rem->operandCount() != 2)
    return false;
  int64_t mod = 0;
  if (!constantValueOperand(rem->operand(1), mod) || mod <= 0)
    return false;
  Operation *add = rem->operand(0).getDefiningOp();
  if (!add || add->isErased() ||
      (add->name() != "arith.addi" && add->name() != "rv_machine.addw" &&
       add->name() != "arm_machine.add") ||
      add->operandCount() != 2)
    return false;
  Value candidateSlot = store.operand(1);
  int64_t inc = 0;
  Operation *load = add->operand(0).getDefiningOp();
  if (loadFromSlot(load, candidateSlot) &&
      constantValueOperand(add->operand(1), inc)) {
    slot = candidateSlot;
    increment = inc;
    modulus = mod;
    return inc > 0;
  }
  load = add->operand(1).getDefiningOp();
  if (loadFromSlot(load, candidateSlot) &&
      constantValueOperand(add->operand(0), inc)) {
    slot = candidateSlot;
    increment = inc;
    modulus = mod;
    return inc > 0;
  }
  return false;
}

static LinearModRecurrenceInfo classifyLinearModRecurrenceLoop(Operation &loop) {
  LinearModRecurrenceInfo info;
  if (loop.name() != "affine.for" || loop.operandCount() < 3 ||
      loop.getRegions().size() != 1 || loop.getRegions()[0]->getBlocks().size() != 1)
    return info;
  int64_t lower = 0;
  int64_t step = 0;
  if (!constantIntegerValue(loop.operand(0), lower) || lower != 0 ||
      !constantIntegerValue(loop.operand(2), step) || step != 1)
    return info;
  int64_t maxTrip = 0;
  if (!positiveUpperBound(loop.operand(1), maxTrip) || maxTrip <= 0)
    return info;

  Block &body = *loop.getRegions()[0]->getBlocks()[0];
  if (body.args().empty())
    return info;
  Value iv = body.args()[0]->value();
  Value slot;
  int64_t increment = 0;
  int64_t modulus = 0;
  int stores = 0;
  for (auto &owned : body.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op->name() == "affine.yield")
      continue;
    if (opTreeUsesValue(*op, iv) && !isPureDeadIvBump(*op, iv))
      return info;
    if (op->name() == "sysy.call" || op->name() == "sysy.return" ||
        op->name() == "scf.return" || op->name() == "sysy.break" ||
        op->name() == "sysy.continue" || op->name() == "memref.store" ||
        op->name() == "scf.if" || op->name() == "scf.while" ||
        op->name() == "affine.for")
      return info;
    if (isMLIRStoreOp(op)) {
      Value storeSlot;
      int64_t storeIncrement = 0;
      int64_t storeModulus = 0;
      if (!classifyLinearModStore(*op, storeSlot, storeIncrement, storeModulus))
        return info;
      if (stores == 0) {
        slot = storeSlot;
        increment = storeIncrement;
        modulus = storeModulus;
      } else if (storeSlot != slot || storeIncrement != increment ||
                 storeModulus != modulus) {
        return info;
      }
      stores++;
    }
  }
  if (stores != 1 || !slot.valid() || modulus <= 0 || increment <= 0)
    return info;
  if (maxTrip > 1000000 || increment > 1000000)
    return info;
  if (increment > (std::numeric_limits<int32_t>::max() - (modulus - 1)) / maxTrip)
    return info;

  info.valid = true;
  info.lower = loop.operand(0);
  info.upper = loop.operand(1);
  info.slot = slot;
  info.increment = increment;
  info.modulus = modulus;
  info.maxTrip = maxTrip;
  return info;
}

static RepeatReductionInfo classifyRepeatReductionLoop(Operation &loop) {
  RepeatReductionInfo info;
  if (loop.name() != "affine.for" || loop.operandCount() < 3 ||
      loop.getRegions().size() != 1 || loop.getRegions()[0]->getBlocks().size() != 1)
    return info;
  int64_t lower = 0;
  int64_t step = 0;
  if (!constantIntegerValue(loop.operand(0), lower) ||
      !constantIntegerValue(loop.operand(2), step) || step != 1)
    return info;
  Block &body = *loop.getRegions()[0]->getBlocks()[0];
  if (body.args().empty())
    return info;
  Value iv = body.args()[0]->value();

  std::vector<Operation*> bodyOps;
  std::set<std::string> localAllocas;
  for (auto &owned : body.ops()) {
    if (!owned || owned->isErased())
      continue;
    bodyOps.push_back(owned.get());
    collectLocalAllocas(*owned, localAllocas);
  }

  for (Operation *op : bodyOps) {
    if (!op)
      continue;
    if (op->name() == "affine.yield") {
      info.skipOps.insert(op);
      continue;
    }
    if (!opTreeUsesValue(*op, iv))
      continue;
    if (isPureDeadIvBump(*op, iv)) {
      info.skipOps.insert(op);
      continue;
    }
    return info;
  }

  std::map<std::string, Value> outerStores;
  for (Operation *op : bodyOps) {
    if (!op || info.skipOps.count(op))
      continue;
    if (!collectRepeatReductionStores(*op, localAllocas, outerStores))
      return info;
  }
  Value accumulatorSlot;
  if (outerStores.size() != 1)
    return info;
  for (const auto &kv : outerStores)
    accumulatorSlot = kv.second;
  if (!accumulatorSlot.valid() ||
      accumulatorSlot.type().str().find("xi32") == std::string::npos ||
      !slotLoadedAfter(loop, accumulatorSlot))
    return info;
  for (Operation *op : bodyOps) {
    if (!op || info.skipOps.count(op))
      continue;
    if (!opTreeAccumulatorStoresAreLinearAdds(*op, accumulatorSlot))
      return info;
  }
  info.valid = true;
  info.lower = loop.operand(0);
  info.upper = loop.operand(1);
  info.step = loop.operand(2);
  info.accumulatorSlot = accumulatorSlot;
  return info;
}

static Operation &appendOp(Block &block, const std::string &name,
                           const std::vector<Value> &operands,
                           const std::vector<Type> &results,
                           const std::map<std::string, Attribute> &attrs,
                           Location loc, int regionCount = 0) {
  auto op = std::make_unique<Operation>(name, operands, results, attrs, loc);
  for (int i = 0; i < regionCount; i++)
    op->addRegion();
  return block.addOperation(std::move(op));
}

static Operation &insertOp(Block &block, std::size_t &index,
                           const std::string &name,
                           const std::vector<Value> &operands,
                           const std::vector<Type> &results,
                           const std::map<std::string, Attribute> &attrs,
                           Location loc, int regionCount = 0) {
  auto op = std::make_unique<Operation>(name, operands, results, attrs, loc);
  for (int i = 0; i < regionCount; i++)
    op->addRegion();
  return block.insertOperation(index++, std::move(op));
}

static bool applyLinearModRecurrenceLoop(Module &module, Operation &loop,
                                         const LinearModRecurrenceInfo &info) {
  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;

  Context &ctx = module.context();
  Type i32 = ctx.i(32);
  Location loc = loop.loc();
  std::size_t insertIndex = (std::size_t) loopIndex;
  Operation &start = insertOp(*parent, insertIndex, "sysy.load",
                              {info.slot}, {i32}, {}, loc);
  Operation &zero = insertOp(*parent, insertIndex, "arith.constant", {}, {i32},
                             {{"value", ctx.integerAttr(0, i32)}}, loc);
  Operation &modConst = insertOp(*parent, insertIndex, "arith.constant", {}, {i32},
                                 {{"value", ctx.integerAttr(info.modulus, i32)}}, loc);
  Operation &incConst = insertOp(*parent, insertIndex, "arith.constant", {}, {i32},
                                 {{"value", ctx.integerAttr(info.increment, i32)}}, loc);

  Operation &tripPositive = insertOp(
      *parent, insertIndex, "arith.cmpi", {info.lower, info.upper}, {i32},
      {{"predicate", ctx.stringAttr("lt")}}, loc);
  Operation &startNonNegative = insertOp(
      *parent, insertIndex, "arith.cmpi", {zero.result(), start.result()}, {i32},
      {{"predicate", ctx.stringAttr("le")}}, loc);
  Operation &startCanonical = insertOp(
      *parent, insertIndex, "arith.cmpi", {start.result(), modConst.result()}, {i32},
      {{"predicate", ctx.stringAttr("lt")}}, loc);
  Operation &guardA = insertOp(*parent, insertIndex, "arith.andi",
                               {tripPositive.result(), startNonNegative.result()},
                               {i32}, {}, loc);
  Operation &guard = insertOp(*parent, insertIndex, "arith.andi",
                              {guardA.result(), startCanonical.result()},
                              {i32}, {}, loc);

  auto ifOp = std::make_unique<Operation>("scf.if", std::vector<Value>{guard.result()},
                                          std::vector<Type>{},
                                          std::map<std::string, Attribute>{}, loc);
  Region &thenRegion = ifOp->addRegion();
  Block &thenBlock = thenRegion.addBlock();
  Value trip = info.upper;
  int64_t lowerImm = 0;
  if (constantIntegerValue(info.lower, lowerImm) && lowerImm != 0) {
    Operation &lowerConst = appendOp(
        thenBlock, "arith.constant", {}, {i32},
        {{"value", ctx.integerAttr(lowerImm, i32)}}, loc);
    Operation &tripOp = appendOp(thenBlock, "arith.subi",
                                 {info.upper, lowerConst.result()}, {i32}, {}, loc);
    trip = tripOp.result();
  }
  Operation &scaled = appendOp(thenBlock, "arith.muli",
                               {incConst.result(), trip}, {i32}, {}, loc);
  Operation &advanced = appendOp(thenBlock, "arith.addi",
                                 {start.result(), scaled.result()}, {i32}, {}, loc);
  Operation &folded = appendOp(thenBlock, "arith.remi",
                               {advanced.result(), modConst.result()}, {i32}, {}, loc);
  appendOp(thenBlock, "sysy.store", {folded.result(), info.slot}, {}, {}, loc);
  appendOp(thenBlock, "scf.yield", {}, {}, {}, loc);

  Region &elseRegion = ifOp->addRegion();
  Block &elseBlock = elseRegion.addBlock();
  std::map<std::string, Value> valueMap;
  std::set<Operation*> skipOps;
  auto cloned = cloneForUnrolledIteration(module, loop, valueMap, skipOps, Value(), 0);
  if (!cloned)
    return false;
  elseBlock.addOperation(std::move(cloned));
  appendOp(elseBlock, "scf.yield", {}, {}, {}, loc);

  parent->insertOperation(insertIndex, std::move(ifOp));
  loop.markErased();
  return true;
}

static bool applyRepeatReductionLoop(Module &module, Operation &loop,
                                     const RepeatReductionInfo &info) {
  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;
  Block &body = *loop.getRegions()[0]->getBlocks()[0];
  std::vector<Operation*> bodyOps;
  for (auto &owned : body.ops())
    if (owned && !owned->isErased())
      bodyOps.push_back(owned.get());

  Context &ctx = module.context();
  Location loc = loop.loc();
  auto cmp = std::make_unique<Operation>(
      "arith.cmpi", std::vector<Value>{info.lower, info.upper},
      std::vector<Type>{ctx.i(32)},
      std::map<std::string, Attribute>{{"predicate", ctx.stringAttr("lt")}},
      loc);
  Operation &cmpOp = parent->insertOperation((std::size_t) loopIndex, std::move(cmp));

  auto ifOp = std::make_unique<Operation>(
      "scf.if", std::vector<Value>{cmpOp.result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc);
  Region &thenRegion = ifOp->addRegion();
  Block &thenBlock = thenRegion.addBlock();

  Operation &start = appendOp(thenBlock, "sysy.load",
                              {info.accumulatorSlot}, {ctx.i(32)}, {}, loc);
  std::map<std::string, Value> valueMap;
  for (Operation *op : bodyOps) {
    if (!op || info.skipOps.count(op))
      continue;
    auto cloned = cloneForUnrolledIteration(module, *op, valueMap, info.skipOps,
                                            Value(), 0);
    if (!cloned)
      continue;
    Operation &inserted = thenBlock.addOperation(std::move(cloned));
    for (int i = 0; i < op->resultCount(); i++)
      valueMap[valueKey(op->result(i))] = inserted.result(i);
  }
  Operation &end = appendOp(thenBlock, "sysy.load",
                            {info.accumulatorSlot}, {ctx.i(32)}, {}, loc);
  Operation &delta = appendOp(thenBlock, "arith.subi",
                              {end.result(), start.result()}, {ctx.i(32)}, {}, loc);
  Value trip = info.upper;
  int64_t lowerImm = 0;
  if (constantIntegerValue(info.lower, lowerImm) && lowerImm != 0) {
    Operation &lowerConst = appendOp(thenBlock, "arith.constant", {}, {ctx.i(32)},
                                    {{"value", ctx.integerAttr(lowerImm, ctx.i(32))}}, loc);
    Operation &tripOp = appendOp(thenBlock, "arith.subi",
                                 {info.upper, lowerConst.result()}, {ctx.i(32)}, {}, loc);
    trip = tripOp.result();
  }
  Operation &scaled = appendOp(thenBlock, "arith.muli",
                               {delta.result(), trip}, {ctx.i(32)}, {}, loc);
  Operation &finalValue = appendOp(thenBlock, "arith.addi",
                                   {start.result(), scaled.result()}, {ctx.i(32)}, {}, loc);
  appendOp(thenBlock, "sysy.store",
           {finalValue.result(), info.accumulatorSlot}, {}, {}, loc);
  appendOp(thenBlock, "scf.yield", {}, {}, {}, loc);

  parent->insertOperation((std::size_t) loopIndex + 1, std::move(ifOp));
  loop.markErased();
  return true;
}

static bool isScalarLocalSlot(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return false;
  if (def->name() != "sysy.alloca" && def->name() != "memref.alloca")
    return false;
  return value.type().str().find("memref<1xi32") != std::string::npos;
}

static bool isSelectSpeculatableOp(Operation *op) {
  if (!op || op->isErased() || op->isTerminator() || !op->getRegions().empty())
    return false;
  if (isMLIRLoadOp(op))
    return op->operandCount() > 0 && isScalarLocalSlot(op->operand(0));
  const std::string &name = op->name();
  return name == "arith.constant" || name == "arith.addi" ||
         name == "arith.subi" || name == "arith.muli" ||
         name == "arith.cmpi" || name == "rv_machine.li" ||
         name == "rv_machine.addw" || name == "rv_machine.subw" ||
         name == "rv_machine.mulw" || name == "rv_machine.cmp" ||
         name == "rv_machine.and" || name == "rv_machine.or" ||
         name == "rv_machine.xor" || name == "rv_machine.neg" ||
         name == "rv_machine.seqz";
}

static bool promoteIfStoresToSelects(Module &module, Operation &ifOp,
                                     SelfOptStats *stats) {
  if (ifOp.name() != "scf.if" || ifOp.operandCount() != 1 ||
      ifOp.getRegions().size() != 1)
    return false;
  if (ifOp.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block *parent = ifOp.getBlock();
  if (!parent)
    return false;
  int ifIndex = operationIndexInBlock(*parent, &ifOp);
  if (ifIndex < 0)
    return false;

  Block &thenBlock = *ifOp.getRegions()[0]->getBlocks()[0];
  std::vector<Operation*> pureOps;
  std::vector<std::pair<Value, Value>> stores;
  std::set<std::string> storeSlots;
  for (auto &owned : thenBlock.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.yield")
      continue;
    if (isMLIRStoreOp(op)) {
      if (op->operandCount() < 2 || !isScalarLocalSlot(op->operand(1)))
        return false;
      std::string slotKey = valueKey(op->operand(1));
      if (!storeSlots.insert(slotKey).second)
        return false;
      stores.push_back({op->operand(1), op->operand(0)});
      continue;
    }
    if (!isSelectSpeculatableOp(op))
      return false;
    pureOps.push_back(op);
  }
  if (stores.empty())
    return false;

  for (Operation *op : pureOps) {
    if (isMLIRLoadOp(op) && op->operandCount() > 0 &&
        storeSlots.count(valueKey(op->operand(0))) != 0)
      return false;
  }
  std::set<std::string> pureResults;
  for (Operation *op : pureOps) {
    for (int i = 0; i < op->resultCount(); i++)
      pureResults.insert(valueKey(op->result(i)));
  }
  for (const auto &store : stores) {
    Operation *def = store.second.getDefiningOp();
    if (def && def->getBlock() == &thenBlock &&
        pureResults.count(valueKey(store.second)) == 0)
      return false;
  }

  std::map<std::string, Value> valueMap;
  std::size_t insertIndex = (std::size_t) ifIndex;
  for (Operation *op : pureOps) {
    auto cloned = std::make_unique<Operation>(
        op->name(), remapOperandsForClone(*op, valueMap),
        resultTypesOf(*op), op->attrs(), op->loc());
    Operation &inserted = parent->insertOperation(insertIndex++, std::move(cloned));
    for (int i = 0; i < op->resultCount(); i++)
      valueMap[valueKey(op->result(i))] = inserted.result(i);
  }

  Context &ctx = module.context();
  int promoted = 0;
  for (const auto &store : stores) {
    Value slot = store.first;
    Value trueValue = store.second;
    auto mapped = valueMap.find(valueKey(trueValue));
    if (mapped != valueMap.end())
      trueValue = mapped->second;
    Operation *trueDef = trueValue.getDefiningOp();
    if (trueDef && trueDef->getBlock() == &thenBlock)
      return false;

    Operation &oldValue = parent->insertOperation(
        insertIndex++,
        std::make_unique<Operation>("sysy.load", std::vector<Value>{slot},
                                    std::vector<Type>{ctx.i(32)},
                                    std::map<std::string, Attribute>{},
                                    ifOp.loc()));
    Operation &select = parent->insertOperation(
        insertIndex++,
        std::make_unique<Operation>("arith.select",
                                    std::vector<Value>{ifOp.operand(0), trueValue,
                                                       oldValue.result()},
                                    std::vector<Type>{ctx.i(32)},
                                    std::map<std::string, Attribute>{},
                                    ifOp.loc()));
    parent->insertOperation(
        insertIndex++,
        std::make_unique<Operation>("sysy.store",
                                    std::vector<Value>{select.result(), slot},
                                    std::vector<Type>{},
                                    std::map<std::string, Attribute>{},
                                    ifOp.loc()));
    promoted++;
  }

  ifOp.markErased();
  if (stats)
    stats->raisedSelects += promoted;
  return promoted > 0;
}

} // namespace

void runIfStoreSelectPromotion(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_SELECT_PROMOTION", true))
    return;
  std::vector<Operation*> ifs;
  for (auto *op : walk(module))
    if (op && !op->isErased() && op->name() == "scf.if")
      ifs.push_back(op);
  bool changed = false;
  for (Operation *op : ifs) {
    if (!op || op->isErased())
      continue;
    changed |= promoteIfStoresToSelects(module, *op, stats);
  }
  if (changed)
    eraseMarked(module);
}

void runStencilPeelingAndUnroll(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_STENCIL_PEEL", true))
    return;
  for (int round = 0; round < 4; round++) {
    std::vector<Operation*> loops;
    for (auto *op : walk(module))
      if (op && !op->isErased() && op->name() == "scf.while")
        loops.push_back(op);
    bool changed = false;
    for (Operation *loop : loops) {
      if (!loop || loop->isErased())
        continue;
      ConstantWhileInfo info = classifySmallConstantWhile(*loop);
      if (!info.valid)
        continue;
      if (unrollSmallConstantWhile(module, *loop, info)) {
        changed = true;
        if (stats) {
          stats->kernelUnrolls++;
          stats->interiorPeels += info.tripCount > 0 ? 1 : 0;
        }
      }
    }
    eraseMarked(module);
    if (!changed)
      break;
  }
}

static bool tileOpHasName(Operation *op, std::initializer_list<const char*> names) {
  if (!op)
    return false;
  for (const char *name : names) {
    if (op->name() == name)
      return true;
  }
  return false;
}

static Block *tileSingleBlock(Operation &op) {
  if (op.getRegions().size() != 1 || op.getRegions()[0]->getBlocks().size() != 1)
    return nullptr;
  return op.getRegions()[0]->getBlocks()[0].get();
}

static Value tileFirstBlockArg(Operation &op) {
  Block *block = tileSingleBlock(op);
  if (!block || block->args().empty())
    return Value();
  return block->args()[0]->value();
}

static bool tileOpTreeHasAnyName(Operation &op, std::initializer_list<const char*> names) {
  if (tileOpHasName(&op, names))
    return true;
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child && tileOpTreeHasAnyName(*child, names))
          return true;
      }
    }
  }
  return false;
}

static bool tileBlockHasNestedLoop(Block &block) {
  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased())
      continue;
    if (tileOpHasName(owned.get(), {"affine.for", "scf.while", "scf.for"}))
      return true;
    for (auto &region : owned->getRegions()) {
      for (auto &childBlock : region->getBlocks()) {
        if (tileBlockHasNestedLoop(*childBlock))
          return true;
      }
    }
  }
  return false;
}

static bool tileBlockUnsafeForStripMining(Block &block) {
  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased())
      continue;
    if (tileOpHasName(owned.get(), {"sysy.call", "sysy.return", "sysy.break",
                                    "sysy.continue", "scf.if", "scf.while",
                                    "scf.for", "affine.for", "scf.condition",
                                    "sysy.alloca", "memref.alloca"}))
      return true;
    for (auto &region : owned->getRegions()) {
      for (auto &childBlock : region->getBlocks()) {
        if (tileBlockUnsafeForStripMining(*childBlock))
          return true;
      }
    }
  }
  return false;
}

static Operation *tileSingleDirectAffineChild(Block &block) {
  Operation *found = nullptr;
  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased() || owned->name() != "affine.for")
      continue;
    if (found)
      return nullptr;
    found = owned.get();
  }
  return found;
}

static Value tileAppendIntConstant(Module &module, Block &block, int64_t value,
                                   Location loc) {
  Operation &op = appendOp(
      block, "rv_machine.li", {}, {module.context().i(32)},
      {{"value", module.context().integerAttr(value, module.context().i(32))}},
      loc);
  return op.result();
}

static Value tileMaterializeValue(Module &module, Block &block, Value value,
                                  Location loc) {
  int64_t imm = 0;
  if (constantIntegerValue(value, imm))
    return tileAppendIntConstant(module, block, imm, loc);
  Operation *def = value.getDefiningOp();
  if (def && !def->isErased() && def->resultCount() == 1 &&
      tileOpHasName(def, {"sysy.load", "memref.load", "rv_machine.li",
                          "arith.constant", "arith.addi", "arith.subi",
                          "rv_machine.addw", "rv_machine.subw"})) {
    Operation &clone = appendOp(block, def->name(), def->getOperands(),
                                resultTypesOf(*def), def->attrs(), loc);
    return clone.result();
  }
  return value;
}

static Operation &tileAppendAffineLoop(Module &module, Block &block, Value lower,
                                       Value upper, Value step, Location loc,
                                       const std::string &ivName) {
  Operation &loop = appendOp(block, "affine.for", {lower, upper, step}, {},
                             {}, loc, 1);
  loop.getRegions()[0]->addBlock().addArgument(module.context().i(32), loc, ivName);
  return loop;
}

static Operation &tileInsertAffineLoop(Module &module, Block &block, std::size_t &index,
                                       Value lower, Value upper, Value step,
                                       Location loc, const std::string &ivName) {
  Operation &loop = insertOp(block, index, "affine.for", {lower, upper, step},
                             {}, {}, loc, 1);
  loop.getRegions()[0]->addBlock().addArgument(module.context().i(32), loc, ivName);
  return loop;
}

static Value tileAppendMinValue(Module &module, Block &block, Value lhs, Value rhs,
                                Location loc) {
  Operation &cmp = appendOp(
      block, "rv_machine.cmp", {lhs, rhs}, {module.context().i(32)},
      {{"predicate", module.context().stringAttr("gt")}}, loc);
  Operation &sel = appendOp(block, "arith.select", {cmp.result(), rhs, lhs},
                            {module.context().i(32)}, {}, loc);
  return sel.result();
}

static bool tileSame2DIndices(Operation *op, Value first, Value second,
                              int firstOperandIndex) {
  return op && op->operandCount() > firstOperandIndex + 1 &&
         op->operand(firstOperandIndex) == first &&
         op->operand(firstOperandIndex + 1) == second;
}

struct RowReductionInfo {
  Operation *outer = nullptr;
  Operation *jLoop = nullptr;
  Operation *kLoop = nullptr;
  Operation *finalStore = nullptr;
  Operation *lhsLoad = nullptr;
  Operation *rhsLoad = nullptr;
  bool valid = false;
};

struct ConditionalRowReductionInfo {
  Operation *outer = nullptr;
  Operation *jLoop = nullptr;
  Operation *kLoop = nullptr;
  Operation *ifOp = nullptr;
  Operation *finalStore = nullptr;
  Operation *product = nullptr;
  Value accumulatorSlot;
  Value parityInvariant;
  bool parityFastPath = false;
  std::vector<Operation*> conditionPrefixOps;
  bool valid = false;
};

struct TileAccumulatorStoreInfo {
  Operation *finalStore = nullptr;
  Value slot;
};

struct PolyAccessInfo {
  Value base;
  bool store = false;
  bool hasOuterIndex = false;
  bool hasInnerIndex = false;
  bool lastIndexOuter = false;
  bool lastIndexInner = false;
};

struct PolyNestInfo {
  Operation *inner = nullptr;
  Block *outerBody = nullptr;
  Block *innerBody = nullptr;
};

static bool tileIsLoadFromSlot(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  return tileOpHasName(op, {"sysy.load", "memref.load"}) &&
         op->operandCount() > 0 && op->operand(0) == slot;
}

static Value tileLoadedScalarSlot(Value value) {
  Operation *op = value.getDefiningOp();
  if (!tileOpHasName(op, {"sysy.load", "memref.load"}) ||
      op->operandCount() == 0 || !isScalarWordMemref(op->operand(0).type()))
    return Value();
  return op->operand(0);
}

static bool tileIsStoreToSlot(Operation *op, Value slot) {
  return tileOpHasName(op, {"sysy.store", "memref.store"}) &&
         op->operandCount() >= 2 && op->operand(1) == slot;
}

static bool tileIsLoadOfSlot(Value value, Value slot) {
  return tileIsLoadFromSlot(value, slot);
}

static bool tileOpTreeStoresSlot(Operation &op, Value slot) {
  if (tileIsStoreToSlot(&op, slot))
    return true;
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child && tileOpTreeStoresSlot(*child, slot))
          return true;
      }
    }
  }
  return false;
}

static bool tileValueIsSlotPlusConst(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  if (!tileOpHasName(op, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
      op->operandCount() != 2)
    return false;
  auto isLoad = [&](Value v) { return tileIsLoadFromSlot(v, slot); };
  int64_t imm = 0;
  return (isLoad(op->operand(0)) && constantIntegerValue(op->operand(1), imm)) ||
         (isLoad(op->operand(1)) && constantIntegerValue(op->operand(0), imm));
}

static bool tileValueIsSlotPlusProduct(Value value, Value slot, Operation *&product) {
  product = nullptr;
  Operation *op = value.getDefiningOp();
  if (!tileOpHasName(op, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
      op->operandCount() != 2)
    return false;
  auto isProduct = [](Value v) {
    Operation *def = v.getDefiningOp();
    return tileOpHasName(def, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) &&
           def->operandCount() == 2 && def->resultCount() == 1;
  };
  if (tileIsLoadOfSlot(op->operand(0), slot) && isProduct(op->operand(1))) {
    product = op->operand(1).getDefiningOp();
    return true;
  }
  if (tileIsLoadOfSlot(op->operand(1), slot) && isProduct(op->operand(0))) {
    product = op->operand(0).getDefiningOp();
    return true;
  }
  return false;
}

static bool tileValueTreeUsesValue(Value value, Value needle, int depth = 0) {
  if (!value.valid() || !needle.valid() || depth > 12)
    return false;
  if (value == needle)
    return true;
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return false;
  for (auto operand : def->getOperands())
    if (tileValueTreeUsesValue(operand, needle, depth + 1))
      return true;
  return false;
}

static bool tileBlockHasZeroStoreToSlot(Block &block, Value slot) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || !tileIsStoreToSlot(op, slot))
      continue;
    int64_t init = 0;
    if (constantIntegerValue(op->operand(0), init) && init == 0)
      return true;
  }
  return false;
}

static std::vector<TileAccumulatorStoreInfo>
tileFindFinalStoresLoadedFromScalarSlot(Block &block, Value iIv, Value jIv) {
  std::vector<TileAccumulatorStoreInfo> result;
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() != "memref.store" ||
        op->operandCount() < 4 || !tileSame2DIndices(op, iIv, jIv, 2))
      continue;
    Value slot = tileLoadedScalarSlot(op->operand(0));
    if (!slot.valid())
      continue;
    bool duplicate = false;
    for (const auto &existing : result) {
      if (existing.finalStore == op && existing.slot == slot) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate)
      result.push_back({op, slot});
  }
  return result;
}

static bool tileStripAndOne(Value value, Value &inner) {
  Operation *op = value.getDefiningOp();
  if (!tileOpHasName(op, {"arith.andi", "rv_machine.and", "arm_machine.and"}) ||
      op->operandCount() != 2)
    return false;
  int64_t imm = 0;
  if (constantIntegerValue(op->operand(0), imm) && imm == 1) {
    inner = op->operand(1);
    return true;
  }
  if (constantIntegerValue(op->operand(1), imm) && imm == 1) {
    inner = op->operand(0);
    return true;
  }
  return false;
}

static bool tileParityOperandsFromOddExpr(Value value, Value &lhs, Value &rhs) {
  Value stripped;
  if (tileStripAndOne(value, stripped))
    value = stripped;
  Operation *op = value.getDefiningOp();
  if (tileOpHasName(op, {"arith.remi", "rv_machine.remw", "arm_machine.srem"}) &&
      op->operandCount() == 2) {
    int64_t divisor = 0;
    if (!constantIntegerValue(op->operand(1), divisor) || divisor != 2)
      return false;
    Operation *mul = op->operand(0).getDefiningOp();
    if (!tileOpHasName(mul, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) ||
        mul->operandCount() != 2)
      return false;
    lhs = mul->operand(0);
    rhs = mul->operand(1);
    return true;
  }
  if (tileOpHasName(op, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
      op->operandCount() == 2) {
    lhs = op->operand(0);
    rhs = op->operand(1);
    return true;
  }
  return false;
}

static bool tileMatchParityEvenCondition(Value condition, Value jIv,
                                         Value &invariantOperand) {
  invariantOperand = Value();
  Operation *condOp = condition.getDefiningOp();
  Value oddExpr;
  if (tileOpHasName(condOp, {"rv_machine.seqz"}) && condOp->operandCount() == 1) {
    oddExpr = condOp->operand(0);
  } else if (tileOpHasName(condOp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) &&
             condOp->operandCount() >= 2 && symbolAttr(condOp->attr("predicate")) == "eq") {
    int64_t imm = 0;
    if (constantIntegerValue(condOp->operand(0), imm) && imm == 0)
      oddExpr = condOp->operand(1);
    else if (constantIntegerValue(condOp->operand(1), imm) && imm == 0)
      oddExpr = condOp->operand(0);
  }
  if (!oddExpr.valid())
    return false;
  Value lhs, rhs;
  if (!tileParityOperandsFromOddExpr(oddExpr, lhs, rhs))
    return false;
  bool lhsUsesJ = tileValueTreeUsesValue(lhs, jIv);
  bool rhsUsesJ = tileValueTreeUsesValue(rhs, jIv);
  if (lhsUsesJ == rhsUsesJ)
    return false;
  invariantOperand = lhsUsesJ ? rhs : lhs;
  return invariantOperand.valid();
}

static void tileReplaceLoadsFromSlot(Module &module, Operation &op, Value slot,
                                     Value replacement, int &count) {
  if (tileOpHasName(&op, {"sysy.load", "memref.load"}) &&
      op.operandCount() > 0 && op.operand(0) == slot && op.resultCount() == 1) {
    replaceAllUses(module, op.result(), replacement);
    op.markErased();
    count++;
    return;
  }
  for (auto &region : op.getRegions()) {
    for (auto &block : region->getBlocks()) {
      for (auto &child : block->ops()) {
        if (child && !child->isErased())
          tileReplaceLoadsFromSlot(module, *child, slot, replacement, count);
      }
    }
  }
}

static void tileCollectMemrefLoadBases(Operation &op, std::set<std::string> &bases) {
  if (tileOpHasName(&op, {"memref.load"}) && op.operandCount() > 0)
    bases.insert(valueKey(op.operand(0)));
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child && !child->isErased())
          tileCollectMemrefLoadBases(*child, bases);
}

static void tileCollectMemrefLoadBases(Block &block, std::set<std::string> &bases) {
  for (auto &owned : block.ops())
    if (owned && !owned->isErased())
      tileCollectMemrefLoadBases(*owned, bases);
}

static bool tileBlockHasStoreOutsideSlot(Block &block, Value slot) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (tileOpHasName(op, {"sysy.store", "memref.store"}) &&
        !tileIsStoreToSlot(op, slot))
      return true;
    for (auto &region : op->getRegions())
      for (auto &childBlock : region->getBlocks())
        if (tileBlockHasStoreOutsideSlot(*childBlock, slot))
          return true;
  }
  return false;
}

static bool cacheWhileLoopIvLoads(Module &module, Operation &loop,
                                  SelfOptStats *stats) {
  if (loop.name() != "scf.while" || loop.getRegions().size() < 2 ||
      loop.getRegions()[0]->getBlocks().empty() ||
      loop.getRegions()[1]->getBlocks().empty())
    return false;
  Block &cond = *loop.getRegions()[0]->getBlocks()[0];
  Block &body = *loop.getRegions()[1]->getBlocks()[0];
  if (tileBlockHasNestedLoop(body))
    return false;
  if (cond.ops().empty() || cond.ops().back()->name() != "scf.condition" ||
      cond.ops().back()->operandCount() == 0)
    return false;
  Operation *cmp = cond.ops().back()->operand(0).getDefiningOp();
  if (!tileOpHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() < 2)
    return false;
  Operation *condLoad = cmp->operand(0).getDefiningOp();
  if (!tileOpHasName(condLoad, {"sysy.load", "memref.load"}) ||
      condLoad->operandCount() == 0 || condLoad->resultCount() != 1)
    return false;
  Value slot = condLoad->operand(0);

  int stepIndex = -1;
  for (std::size_t i = 0; i < body.ops().size(); i++) {
    Operation *op = body.ops()[i].get();
    if (!op || op->isErased())
      continue;
    if (tileIsStoreToSlot(op, slot)) {
      if (!tileValueIsSlotPlusConst(op->operand(0), slot))
        return false;
      if (stepIndex >= 0)
        return false;
      stepIndex = (int) i;
    } else if (tileOpTreeStoresSlot(*op, slot)) {
      return false;
    }
  }
  if (stepIndex <= 0)
    return false;

  std::size_t insertIndex = 0;
  Operation &cached = insertOp(body, insertIndex, condLoad->name(), {slot},
                               resultTypesOf(*condLoad), condLoad->attrs(),
                               loop.loc());
  stepIndex++;
  int replaced = 0;
  for (int i = 1; i < stepIndex && i < (int) body.ops().size(); i++) {
    Operation *op = body.ops()[(std::size_t) i].get();
    if (!op || op->isErased())
      continue;
    tileReplaceLoadsFromSlot(module, *op, slot, cached.result(), replaced);
  }
  if (replaced == 0) {
    cached.markErased();
    return false;
  }
  if (stats) {
    stats->loopAddressCSE += replaced;
    stats->addrIvRewrites++;
  }
  return true;
}

static bool polyIsMemrefAccess(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "memref.load" || op->name() == "memref.store");
}

static Value polyAccessBase(Operation *op) {
  if (!polyIsMemrefAccess(op))
    return Value();
  return op->operand(op->name() == "memref.store" ? 1 : 0);
}

static int polyAccessIndexStart(Operation *op) {
  return op && op->name() == "memref.store" ? 2 : 1;
}

static void polyCollectAccesses(Operation &op, Value outerIv, Value innerIv,
                                std::vector<PolyAccessInfo> &accesses) {
  if (op.isErased())
    return;
  if (polyIsMemrefAccess(&op)) {
    int start = polyAccessIndexStart(&op);
    if (op.operandCount() > start) {
      Value last = op.operand(op.operandCount() - 1);
      bool hasOuter = false;
      bool hasInner = false;
      for (int i = start; i < op.operandCount(); i++) {
        hasOuter = hasOuter || op.operand(i) == outerIv;
        hasInner = hasInner || op.operand(i) == innerIv;
      }
      accesses.push_back({polyAccessBase(&op), op.name() == "memref.store",
                          hasOuter, hasInner, last == outerIv, last == innerIv});
    }
  }
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child)
          polyCollectAccesses(*child, outerIv, innerIv, accesses);
}

static bool polyNoStoreAliasHazard(const std::vector<PolyAccessInfo> &accesses) {
  std::set<std::string> storeBases;
  for (const auto &access : accesses) {
    if (!access.store)
      continue;
    if (!access.hasOuterIndex || !access.hasInnerIndex)
      return false;
    std::string key = valueKey(access.base);
    if (storeBases.count(key) != 0)
      return false;
    storeBases.insert(key);
  }
  for (const auto &access : accesses) {
    if (!access.store && storeBases.count(valueKey(access.base)) != 0)
      return false;
  }
  return true;
}

static bool polyScalarSlotHasLoad(Module &module, Value slot) {
  if (!slot.valid())
    return true;
  for (Operation *op : walk(module)) {
    if (!op || op->isErased() ||
        !tileOpHasName(op, {"sysy.load", "memref.load"}) ||
        op->operandCount() == 0)
      continue;
    if (op->operand(0) == slot)
      return true;
  }
  return false;
}

static bool polyTransparentOuterOp(Module &module, Operation *op) {
  if (!op || op->isErased() || op->name() == "affine.yield")
    return true;
  if (op->name() == "affine.for")
    return true;
  if (!op->getRegions().empty())
    return false;
  if (tileOpHasName(op, {"sysy.call", "sysy.return", "sysy.break",
                         "sysy.continue", "scf.if", "scf.while",
                         "scf.for", "scf.condition", "memref.load",
                         "sysy.load"}))
    return false;
  if (tileOpHasName(op, {"sysy.store", "memref.store"})) {
    if (op->operandCount() < 2 || !isScalarWordMemref(op->operand(1).type()))
      return false;
    return !polyScalarSlotHasLoad(module, op->operand(1));
  }
  for (int i = 0; i < op->resultCount(); i++) {
    int64_t imm = 0;
    if (!constantIntegerValue(op->result(i), imm))
      return false;
  }
  return true;
}

static bool polyClassify2DNest(Module &module, Operation &outer,
                               PolyNestInfo &info) {
  info = PolyNestInfo();
  info.outerBody = tileSingleBlock(outer);
  if (!info.outerBody || info.outerBody->args().empty())
    return false;
  for (auto &owned : info.outerBody->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() == "affine.yield")
      continue;
    if (op->name() == "affine.for") {
      if (info.inner)
        return false;
      info.inner = op;
      continue;
    }
    if (!polyTransparentOuterOp(module, op))
      return false;
  }
  if (!info.inner)
    return false;
  info.innerBody = tileSingleBlock(*info.inner);
  return info.innerBody && !info.innerBody->args().empty() &&
         !tileBlockUnsafeForStripMining(*info.innerBody);
}

static Value polyMaterializeLoopOperand(Module &module, Block &block,
                                        std::size_t &index, Value value,
                                        Block *oldOuterBody, Location loc) {
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != oldOuterBody)
    return value;
  int64_t imm = 0;
  if (!constantIntegerValue(value, imm))
    return Value();
  Operation &constant = insertOp(
      block, index, "rv_machine.li", {}, {module.context().i(32)},
      {{"value", module.context().integerAttr(imm, module.context().i(32))}},
      loc);
  return constant.result();
}

static bool applyPolyLoopInterchange(Module &module, Operation &outer,
                                     const PolyNestInfo &nest) {
  Operation &inner = *nest.inner;
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;
  Block *oldOuterBody = nest.outerBody;
  Block *oldInnerBody = nest.innerBody;
  if (!oldOuterBody || !oldInnerBody || oldOuterBody->args().empty() ||
      oldInnerBody->args().empty())
    return false;

  Location loc = outer.loc();
  std::size_t insertIndex = (std::size_t) outerIndex;
  Value innerLower = polyMaterializeLoopOperand(module, *parent, insertIndex,
                                                inner.operand(0), oldOuterBody, loc);
  Value innerUpper = polyMaterializeLoopOperand(module, *parent, insertIndex,
                                                inner.operand(1), oldOuterBody, loc);
  Value innerStep = polyMaterializeLoopOperand(module, *parent, insertIndex,
                                               inner.operand(2), oldOuterBody, loc);
  if (!innerLower.valid() || !innerUpper.valid() || !innerStep.valid())
    return false;
  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             innerLower, innerUpper, innerStep,
                                             loc, "poly_j");
  Block &newOuterBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value newOuterIv = newOuterBody.args()[0]->value();
  Operation &newInner = tileAppendAffineLoop(module, newOuterBody,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "poly_i");
  Block &newInnerBody = *newInner.getRegions()[0]->getBlocks()[0];
  Value newInnerIv = newInnerBody.args()[0]->value();

  std::map<std::string, Value> valueMap;
  valueMap[valueKey(oldOuterBody->args()[0]->value())] = newInnerIv;
  valueMap[valueKey(oldInnerBody->args()[0]->value())] = newOuterIv;
  for (auto &owned : oldOuterBody->ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op == &inner)
      continue;
    for (int i = 0; i < op->resultCount(); i++) {
      int64_t imm = 0;
      if (constantIntegerValue(op->result(i), imm))
        valueMap[valueKey(op->result(i))] =
            tileAppendIntConstant(module, newInnerBody, imm, loc);
    }
  }
  std::set<Operation*> skipOps;
  for (auto &owned : oldInnerBody->ops()) {
    if (!owned || owned->isErased() || owned->isTerminator())
      continue;
    auto cloned = cloneForUnrolledIteration(module, *owned, valueMap, skipOps,
                                            Value(), 0);
    if (!cloned)
      continue;
    Operation &inserted = newInnerBody.addOperation(std::move(cloned));
    for (int i = 0; i < owned->resultCount(); i++)
      valueMap[valueKey(owned->result(i))] = inserted.result(i);
  }
  appendOp(newInnerBody, "affine.yield", {}, {}, {}, loc);
  appendOp(newOuterBody, "affine.yield", {}, {}, {}, loc);
  outer.markErased();
  return true;
}

void runPolyhedralLoopPermutation(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_POLY_PERMUTE", true))
    return;
  std::vector<Operation*> loops;
  for (auto *op : walk(module))
    if (op && !op->isErased() && op->name() == "affine.for")
      loops.push_back(op);

  bool changed = false;
  for (Operation *outer : loops) {
    if (!outer || outer->isErased() || outer->operandCount() < 3)
      continue;
    PolyNestInfo nest;
    if (!polyClassify2DNest(module, *outer, nest))
      continue;
    Operation *inner = nest.inner;
    if (!inner || inner->operandCount() < 3)
      continue;
    int64_t outerStep = 0;
    int64_t innerStep = 0;
    if (!constantIntegerValue(outer->operand(2), outerStep) || outerStep != 1 ||
        !constantIntegerValue(inner->operand(2), innerStep) || innerStep != 1)
      continue;

    if (stats)
      stats->polyNests++;
    std::vector<PolyAccessInfo> accesses;
    Value outerIv = nest.outerBody->args()[0]->value();
    Value innerIv = nest.innerBody->args()[0]->value();
    for (auto &owned : nest.innerBody->ops())
      if (owned && !owned->isErased())
        polyCollectAccesses(*owned, outerIv, innerIv, accesses);
    if (accesses.empty())
      continue;
    if (!polyNoStoreAliasHazard(accesses))
      continue;
    if (stats)
      stats->polyDepsProved++;

    int innerStrideScore = 0;
    int outerStrideScore = 0;
    for (const auto &access : accesses) {
      innerStrideScore += access.lastIndexInner ? 2 : 0;
      outerStrideScore += access.lastIndexOuter ? 2 : 0;
      if (access.store) {
        innerStrideScore += access.lastIndexInner ? 1 : 0;
        outerStrideScore += access.lastIndexOuter ? 1 : 0;
      }
    }
    if (outerStrideScore <= innerStrideScore)
      continue;
    if (applyPolyLoopInterchange(module, *outer, nest)) {
      changed = true;
      if (stats) {
        stats->polyPermutations++;
        stats->imperfectInterchanges++;
      }
    }
  }
  if (changed)
    eraseMarked(module);
}

static bool parityOpHasName(Operation *op, std::initializer_list<const char*> names) {
  if (!op || op->isErased())
    return false;
  for (const char *name : names)
    if (op->name() == name)
      return true;
  return false;
}

static int parityLiveUseCount(Module &module, Value value, Operation **onlyUser = nullptr) {
  int count = 0;
  Operation *last = nullptr;
  for (const auto &use : usesOf(module, value)) {
    if (!use.owner || use.owner->isErased())
      continue;
    count++;
    last = use.owner;
  }
  if (onlyUser)
    *onlyUser = last;
  return count;
}

static std::string parityConstOpForCmp(const std::string &cmpName) {
  if (cmpName == "rv_machine.cmp")
    return "rv_machine.li";
  if (cmpName == "arm_machine.cmp")
    return "arm_machine.mov";
  return "arith.constant";
}

static std::string parityAndOpForCmp(const std::string &cmpName) {
  if (cmpName == "rv_machine.cmp")
    return "rv_machine.and";
  if (cmpName == "arm_machine.cmp")
    return "arm_machine.and";
  return "arith.andi";
}

static std::string parityNotOpForCmp(const std::string &cmpName) {
  if (cmpName == "rv_machine.cmp")
    return "rv_machine.seqz";
  if (cmpName == "arm_machine.cmp")
    return "arm_machine.not";
  return "arith.noti";
}

static bool parityPureDeadOp(Operation *op) {
  if (!op || op->isErased() || op->resultCount() == 0)
    return false;
  for (int i = 0; i < op->resultCount(); i++) {
    if (!op->resultUses[i].empty())
      return false;
  }
  return parityOpHasName(op, {"arith.constant", "rv_machine.li", "arm_machine.mov",
                              "arith.andi", "rv_machine.and", "arm_machine.and"});
}

void runParityProductCompareStrength(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_POW2_STRENGTH", true))
    return;
  std::vector<Operation*> cmps;
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (parityOpHasName(op, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}))
      cmps.push_back(op);
  }

  bool changed = false;
  for (Operation *cmp : cmps) {
    if (!cmp || cmp->isErased() || cmp->operandCount() != 2 || cmp->resultCount() != 1)
      continue;
    std::string pred = symbolAttr(cmp->attr("predicate"));
    if (pred != "eq" && pred != "ne")
      continue;
    Value remValue;
    Value zeroValue;
    int64_t zero = 0;
    if (constantIntegerValue(cmp->operand(1), zero) && zero == 0) {
      remValue = cmp->operand(0);
      zeroValue = cmp->operand(1);
    } else if (constantIntegerValue(cmp->operand(0), zero) && zero == 0) {
      remValue = cmp->operand(1);
      zeroValue = cmp->operand(0);
    } else {
      continue;
    }

    Operation *rem = remValue.getDefiningOp();
    if (!parityOpHasName(rem, {"arith.remi", "rv_machine.remw", "arm_machine.srem"}) ||
        rem->operandCount() != 2 || rem->resultCount() != 1)
      continue;
    int64_t divisor = 0;
    if (!constantIntegerValue(rem->operand(1), divisor) || divisor != 2)
      continue;
    Operation *mul = rem->operand(0).getDefiningOp();
    if (!parityOpHasName(mul, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) ||
        mul->operandCount() != 2 || mul->resultCount() != 1)
      continue;

    Operation *only = nullptr;
    if (parityLiveUseCount(module, rem->result(), &only) != 1 || only != cmp)
      continue;
    if (parityLiveUseCount(module, mul->result(), &only) != 1 || only != rem)
      continue;

    Block *block = cmp->getBlock();
    if (!block)
      continue;
    int cmpIndex = operationIndexInBlock(*block, cmp);
    if (cmpIndex < 0)
      continue;
    std::size_t insertIndex = (std::size_t) cmpIndex;
    Type i32 = cmp->resultType();
    auto oneOp = std::make_unique<Operation>(
        parityConstOpForCmp(cmp->name()), std::vector<Value>{}, std::vector<Type>{i32},
        std::map<std::string, Attribute>{{"value", module.context().integerAttr(1, i32)}},
        cmp->loc());
    Operation &one = block->insertOperation(insertIndex++, std::move(oneOp));
    auto andPairOp = std::make_unique<Operation>(
        parityAndOpForCmp(cmp->name()),
        std::vector<Value>{mul->operand(0), mul->operand(1)}, std::vector<Type>{i32},
        std::map<std::string, Attribute>{}, cmp->loc());
    Operation &andPair = block->insertOperation(insertIndex++, std::move(andPairOp));
    auto lowBitOp = std::make_unique<Operation>(
        parityAndOpForCmp(cmp->name()),
        std::vector<Value>{andPair.result(), one.result()}, std::vector<Type>{i32},
        std::map<std::string, Attribute>{}, cmp->loc());
    Operation &lowBit = block->insertOperation(insertIndex++, std::move(lowBitOp));
    Value replacement = lowBit.result();
    if (pred == "eq") {
      auto notOp = std::make_unique<Operation>(
          parityNotOpForCmp(cmp->name()), std::vector<Value>{lowBit.result()},
          std::vector<Type>{i32}, std::map<std::string, Attribute>{}, cmp->loc());
      Operation &inserted = block->insertOperation(insertIndex++, std::move(notOp));
      replacement = inserted.result();
    }

    Operation *zeroDef = zeroValue.getDefiningOp();
    Operation *divisorDef = rem->operand(1).getDefiningOp();
    replaceAllUses(module, cmp->result(), replacement);
    cmp->markErased();
    rem->markErased();
    mul->markErased();
    if (parityPureDeadOp(zeroDef))
      zeroDef->markErased();
    if (parityPureDeadOp(divisorDef))
      divisorDef->markErased();
    changed = true;
    if (stats)
      stats->pow2StrengthReductions++;
  }
  if (changed)
    eraseMarked(module);
}

static bool machineDcePureOp(Operation *op) {
  if (!op || op->isErased() || op->resultCount() == 0 || !op->getRegions().empty())
    return false;
  for (int i = 0; i < op->resultCount(); i++) {
    if (!op->resultUses[i].empty())
      return false;
  }
  const std::string &name = op->name();
  return name == "arith.constant" || name == "arith.addi" ||
         name == "arith.subi" || name == "arith.muli" ||
         name == "arith.divi" || name == "arith.remi" ||
         name == "arith.andi" || name == "arith.ori" ||
         name == "arith.xori" || name == "arith.noti" ||
         name == "arith.cmpi" || name == "rv_machine.li" ||
         name == "rv_machine.addw" || name == "rv_machine.subw" ||
         name == "rv_machine.mulw" || name == "rv_machine.divw" ||
         name == "rv_machine.remw" || name == "rv_machine.and" ||
         name == "rv_machine.or" || name == "rv_machine.xor" ||
         name == "rv_machine.seqz" || name == "rv_machine.neg" ||
         name == "rv_machine.cmp" || name == "arm_machine.mov" ||
         name == "arm_machine.add" || name == "arm_machine.sub" ||
         name == "arm_machine.mul" || name == "arm_machine.sdiv" ||
         name == "arm_machine.srem" || name == "arm_machine.and" ||
         name == "arm_machine.orr" || name == "arm_machine.eor" ||
         name == "arm_machine.not" || name == "arm_machine.neg" ||
         name == "arm_machine.cmp";
}

void runMachineDeadCodeElim(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_MACHINE_DCE", true))
    return;
  bool changed = true;
  int removed = 0;
  while (changed) {
    changed = false;
    for (auto *op : walk(module)) {
      if (!machineDcePureOp(op))
        continue;
      op->markErased();
      changed = true;
      removed++;
    }
    if (changed)
      eraseMarked(module);
  }
  if (stats && removed > 0)
    stats->worklistRewrites += removed;
}

static bool classifyRowBufferedReduction(Operation &outer, RowReductionInfo &info) {
  if (outer.name() != "affine.for" || outer.operandCount() < 3)
    return false;
  int64_t step = 0;
  if (!constantIntegerValue(outer.operand(2), step) || step != 1)
    return false;
  Block *outerBody = tileSingleBlock(outer);
  if (!outerBody || outerBody->args().empty())
    return false;
  Operation *jLoop = tileSingleDirectAffineChild(*outerBody);
  if (!jLoop || jLoop->operandCount() < 3)
    return false;
  if (tileOpTreeHasAnyName(*jLoop, {"sysy.call", "sysy.return", "sysy.break",
                                    "sysy.continue", "scf.if", "scf.while"}))
    return false;
  int64_t jStep = 0;
  if (!constantIntegerValue(jLoop->operand(2), jStep) || jStep != 1)
    return false;
  Block *jBody = tileSingleBlock(*jLoop);
  if (!jBody || jBody->args().empty())
    return false;
  Operation *kLoop = tileSingleDirectAffineChild(*jBody);
  if (!kLoop || kLoop->operandCount() < 3)
    return false;
  int64_t kStep = 0;
  if (!constantIntegerValue(kLoop->operand(2), kStep) || kStep != 1)
    return false;
  Block *kBody = tileSingleBlock(*kLoop);
  if (!kBody || kBody->args().empty() || tileBlockHasNestedLoop(*kBody))
    return false;

  Value iIv = tileFirstBlockArg(outer);
  Value jIv = tileFirstBlockArg(*jLoop);
  Value kIv = tileFirstBlockArg(*kLoop);

  std::vector<TileAccumulatorStoreInfo> accumulatorCandidates =
      tileFindFinalStoresLoadedFromScalarSlot(*jBody, iIv, jIv);
  if (accumulatorCandidates.empty())
    return false;

  Operation *finalStore = nullptr;
  Operation *sumStore = nullptr;
  Operation *lhsLoad = nullptr;
  Operation *rhsLoad = nullptr;
  for (const auto &candidate : accumulatorCandidates) {
    finalStore = candidate.finalStore;
    Value sumSlot = candidate.slot;
    sumStore = nullptr;
    lhsLoad = nullptr;
    rhsLoad = nullptr;

    if (!finalStore || !tileBlockHasZeroStoreToSlot(*jBody, sumSlot))
      continue;

    for (auto &owned : kBody->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() || !tileIsStoreToSlot(op, sumSlot))
        continue;
      Operation *add = op->operand(0).getDefiningOp();
      if (!tileOpHasName(add, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
          add->operandCount() != 2)
        continue;
      Value product;
      if (tileIsLoadFromSlot(add->operand(0), sumSlot))
        product = add->operand(1);
      else if (tileIsLoadFromSlot(add->operand(1), sumSlot))
        product = add->operand(0);
      else
        continue;
      Operation *mul = product.getDefiningOp();
      if (!tileOpHasName(mul, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) ||
          mul->operandCount() != 2)
        continue;
      Operation *a = mul->operand(0).getDefiningOp();
      Operation *b = mul->operand(1).getDefiningOp();
      if (!tileOpHasName(a, {"memref.load"}) || !tileOpHasName(b, {"memref.load"}))
        continue;
      bool aIsLhs = tileSame2DIndices(a, iIv, kIv, 1) &&
                    tileSame2DIndices(b, kIv, jIv, 1);
      bool bIsLhs = tileSame2DIndices(b, iIv, kIv, 1) &&
                    tileSame2DIndices(a, kIv, jIv, 1);
      if (!aIsLhs && !bIsLhs)
        continue;
      lhsLoad = aIsLhs ? a : b;
      rhsLoad = aIsLhs ? b : a;
      sumStore = op;
    }
    if (sumStore && lhsLoad && rhsLoad)
      break;
  }
  if (!sumStore || !lhsLoad || !rhsLoad)
    return false;
  MemrefInfo outInfo = parseMemrefInfo(finalStore->operand(1).type());
  if (!outInfo.valid || outInfo.shape.size() < 2 || outInfo.shape[1] <= 0)
    return false;

  info.outer = &outer;
  info.jLoop = jLoop;
  info.kLoop = kLoop;
  info.finalStore = finalStore;
  info.lhsLoad = lhsLoad;
  info.rhsLoad = rhsLoad;
  info.valid = true;
  return true;
}

static bool classifyConditionalRowReduction(Operation &outer,
                                            ConditionalRowReductionInfo &info) {
  if (outer.name() != "affine.for" || outer.operandCount() < 3)
    return false;

  struct LoopTriple {
    Operation *iLoop = nullptr;
    Operation *jLoop = nullptr;
    Operation *kLoop = nullptr;
  };
  auto parentAffineLoop = [](Operation *op) -> Operation* {
    if (!op || !op->getBlock() || !op->getBlock()->getRegion())
      return nullptr;
    Operation *parent = op->getBlock()->getRegion()->getParent();
    return parent && parent->name() == "affine.for" ? parent : nullptr;
  };
  auto directAffineChild = [](Operation *loop) -> Operation* {
    Block *body = loop ? tileSingleBlock(*loop) : nullptr;
    return body ? tileSingleDirectAffineChild(*body) : nullptr;
  };
  auto singleChildIs = [&](Operation *parent, Operation *child) {
    return parent && child && directAffineChild(parent) == child;
  };
  auto stepOne = [](Operation *loop) {
    int64_t step = 0;
    return loop && loop->operandCount() >= 3 &&
           constantIntegerValue(loop->operand(2), step) && step == 1;
  };
  auto purePostIfOp = [](Operation *op) {
    return op && op->getRegions().empty() &&
           !tileOpHasName(op, {"sysy.load", "sysy.store", "memref.load",
                               "memref.store", "sysy.call", "sysy.return",
                               "sysy.break", "sysy.continue", "scf.if",
                               "scf.while", "scf.for", "affine.for",
                               "scf.condition"});
  };

  std::vector<LoopTriple> triples;
  auto addTriple = [&](Operation *iLoop, Operation *jLoop, Operation *kLoop) {
    if (!iLoop || !jLoop || !kLoop)
      return;
    for (const auto &existing : triples)
      if (existing.iLoop == iLoop && existing.jLoop == jLoop &&
          existing.kLoop == kLoop)
        return;
    triples.push_back({iLoop, jLoop, kLoop});
  };

  Operation *child = directAffineChild(&outer);
  addTriple(&outer, child, directAffineChild(child));
  Operation *parent = parentAffineLoop(&outer);
  if (singleChildIs(parent, &outer))
    addTriple(parent, &outer, directAffineChild(&outer));
  Operation *grandparent = parentAffineLoop(parent);
  if (singleChildIs(parent, &outer) && singleChildIs(grandparent, parent))
    addTriple(grandparent, parent, &outer);

  for (const auto &triple : triples) {
    Operation *outerLoop = triple.iLoop;
    Operation *jLoop = triple.jLoop;
    Operation *kLoop = triple.kLoop;
    if (!stepOne(outerLoop) || !stepOne(jLoop) || !stepOne(kLoop))
      continue;
    Block *outerBody = tileSingleBlock(*outerLoop);
    Block *jBody = tileSingleBlock(*jLoop);
    Block *kBody = tileSingleBlock(*kLoop);
    if (!outerBody || outerBody->args().empty() || !jBody ||
        jBody->args().empty() || !kBody || kBody->args().empty() ||
        tileBlockHasNestedLoop(*kBody))
      continue;

    Value iIv = tileFirstBlockArg(*outerLoop);
    Value jIv = tileFirstBlockArg(*jLoop);
    Value kIv = tileFirstBlockArg(*kLoop);

    std::vector<TileAccumulatorStoreInfo> accumulatorCandidates =
        tileFindFinalStoresLoadedFromScalarSlot(*jBody, iIv, jIv);
    if (accumulatorCandidates.empty())
      continue;

    for (const auto &candidate : accumulatorCandidates) {
      Operation *finalStore = candidate.finalStore;
      Value sumSlot = candidate.slot;
      if (!finalStore || !tileBlockHasZeroStoreToSlot(*jBody, sumSlot))
        continue;

      Operation *ifOp = nullptr;
      std::vector<Operation*> conditionPrefixOps;
      bool prefixValid = true;
      bool seenIf = false;
      for (auto &owned : kBody->ops()) {
        Operation *op = owned.get();
        if (!op || op->isErased() || op->name() == "affine.yield")
          continue;
        if (op->name() == "scf.if") {
          if (ifOp) {
            prefixValid = false;
            break;
          }
          ifOp = op;
          seenIf = true;
        } else {
          if (seenIf) {
            if (!purePostIfOp(op)) {
              prefixValid = false;
              break;
            }
            continue;
          }
          if (!op->getRegions().empty()) {
            prefixValid = false;
            break;
          }
          conditionPrefixOps.push_back(op);
        }
      }
      if (!prefixValid || !ifOp || ifOp->operandCount() != 1 ||
          ifOp->getRegions().size() != 1 ||
          ifOp->getRegions()[0]->getBlocks().size() != 1)
        continue;
      Block &thenBlock = *ifOp->getRegions()[0]->getBlocks()[0];
      if (tileBlockHasNestedLoop(thenBlock) ||
          tileBlockHasStoreOutsideSlot(thenBlock, sumSlot))
        continue;

      Operation *product = nullptr;
      Operation *sumStore = nullptr;
      for (auto &owned : thenBlock.ops()) {
        Operation *op = owned.get();
        if (!op || op->isErased() || op->name() == "scf.yield")
          continue;
        if (!tileIsStoreToSlot(op, sumSlot))
          continue;
        Operation *candidateProduct = nullptr;
        if (!tileValueIsSlotPlusProduct(op->operand(0), sumSlot, candidateProduct))
          continue;
        product = candidateProduct;
        sumStore = op;
        break;
      }
      if (!sumStore || !product)
        continue;

      std::set<std::string> loadBases;
      for (Operation *op : conditionPrefixOps)
        tileCollectMemrefLoadBases(*op, loadBases);
      tileCollectMemrefLoadBases(thenBlock, loadBases);
      std::string outputBase = valueKey(finalStore->operand(1));
      if (loadBases.count(outputBase) != 0)
        continue;

      bool productUsesI = tileValueTreeUsesValue(product->result(), iIv);
      bool productUsesJ = tileValueTreeUsesValue(product->result(), jIv);
      bool productUsesK = tileValueTreeUsesValue(product->result(), kIv);
      if (!productUsesJ || !productUsesK)
        continue;

      info.outer = outerLoop;
      info.jLoop = jLoop;
      info.kLoop = kLoop;
      info.ifOp = ifOp;
      info.finalStore = finalStore;
      info.product = product;
      info.accumulatorSlot = sumSlot;
      Value parityInvariant;
      if (envEnabled("SISY_ENABLE_SELF_PARITY_BIFURCATION", true) &&
          tileMatchParityEvenCondition(info.ifOp->operand(0), jIv, parityInvariant)) {
        info.parityFastPath = true;
        info.parityInvariant = parityInvariant;
      } else {
        info.parityFastPath = false;
        info.parityInvariant = Value();
      }
      info.conditionPrefixOps = conditionPrefixOps;
      info.valid = true;
      (void) productUsesI;
      return true;
    }
  }
  return false;
}

static Operation *cloneSimpleIntoBlock(Module &module, Block &block, Operation &op,
                                       std::map<std::string, Value> &valueMap) {
  std::set<Operation*> skipOps;
  auto cloned = cloneForUnrolledIteration(module, op, valueMap, skipOps,
                                          Value(), 0);
  if (!cloned)
    return nullptr;
  Operation &inserted = block.addOperation(std::move(cloned));
  for (int i = 0; i < op.resultCount(); i++)
    valueMap[valueKey(op.result(i))] = inserted.result(i);
  return &inserted;
}

static bool ensureClonedValue(Module &module, Block &block, Value value,
                              std::map<std::string, Value> &valueMap,
                              Block *allowedSourceBlock) {
  if (!value.valid())
    return false;
  if (valueMap.count(valueKey(value)) != 0)
    return true;
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased() || def->getBlock() != allowedSourceBlock ||
      def->getRegions().size() != 0)
    return true;
  for (auto operand : def->getOperands()) {
    if (!ensureClonedValue(module, block, operand, valueMap, allowedSourceBlock))
      return false;
  }
  return cloneSimpleIntoBlock(module, block, *def, valueMap) != nullptr;
}

static bool applyConditionalRowBufferedReduction(
    Module &module, const ConditionalRowReductionInfo &info, SelfOptStats *stats) {
  Operation &outer = *info.outer;
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;

  Context &ctx = module.context();
  Location loc = outer.loc();
  Type i32 = ctx.i(32);
  MemrefInfo outInfo = parseMemrefInfo(info.finalStore->operand(1).type());
  if (!outInfo.valid || outInfo.shape.size() < 2 || outInfo.shape[1] <= 0)
    return false;
  int64_t rowElements = outInfo.shape[1];

  std::size_t insertIndex = (std::size_t) outerIndex;
  Operation &rowBuf = insertOp(
      *parent, insertIndex, "sysy.alloca", {}, {ctx.memref({rowElements}, i32)},
      {{"symbol", ctx.stringAttr(".tile_cond_rowbuf")}}, loc);
  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "i");
  Block &outerBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value newI = outerBody.args()[0]->value();
  Value zeroJLower = tileMaterializeValue(module, outerBody,
                                          info.jLoop->operand(0), loc);
  Value zeroJUpper = tileMaterializeValue(module, outerBody,
                                          info.jLoop->operand(1), loc);
  Value zeroJStep = tileMaterializeValue(module, outerBody,
                                         info.jLoop->operand(2), loc);

  Operation &zeroLoop = tileAppendAffineLoop(module, outerBody, zeroJLower,
                                             zeroJUpper, zeroJStep, loc, "j");
  {
    Block &body = *zeroLoop.getRegions()[0]->getBlocks()[0];
    Value newJ = body.args()[0]->value();
    Value zero = tileAppendIntConstant(module, body, 0, loc);
    appendOp(body, "memref.store", {zero, rowBuf.result(), newJ}, {}, {}, loc);
    appendOp(body, "affine.yield", {}, {}, {}, loc);
  }

  Block *oldOuterBody = tileSingleBlock(*info.outer);
  Block *oldJBody = tileSingleBlock(*info.jLoop);
  Block *oldKBody = tileSingleBlock(*info.kLoop);
  if (!oldOuterBody || !oldJBody || !oldKBody || oldOuterBody->args().empty() ||
      oldJBody->args().empty() || oldKBody->args().empty())
    return false;
  Value oldI = oldOuterBody->args()[0]->value();
  Value oldJ = oldJBody->args()[0]->value();
  Value oldK = oldKBody->args()[0]->value();

  Value kLower = tileMaterializeValue(module, outerBody, info.kLoop->operand(0), loc);
  Value kUpper = tileMaterializeValue(module, outerBody, info.kLoop->operand(1), loc);
  Value kStep = tileMaterializeValue(module, outerBody, info.kLoop->operand(2), loc);
  Operation &kLoop = tileAppendAffineLoop(module, outerBody, kLower, kUpper,
                                          kStep, loc, "k");
  {
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value newK = kBody.args()[0]->value();
    std::map<std::string, Value> kValueMap;
    kValueMap[valueKey(oldI)] = newI;
    kValueMap[valueKey(oldK)] = newK;

    Block &oldThen = *info.ifOp->getRegions()[0]->getBlocks()[0];
    for (int operandIdx = 0; operandIdx < info.product->operandCount(); operandIdx++) {
      Value operand = info.product->operand(operandIdx);
      Operation *def = operand.getDefiningOp();
      if (!def || def->isErased() || def->getBlock() != &oldThen ||
          opTreeUsesValue(*def, oldJ))
        continue;
      if (!ensureClonedValue(module, kBody, operand, kValueMap, &oldThen))
        return false;
    }

    auto appendAccumulate = [&](Block &target, Value newJ,
                                std::map<std::string, Value> valueMap) -> bool {
      Operation &oldAcc = appendOp(target, "memref.load",
                                   {rowBuf.result(), newJ}, {i32}, {}, loc);
      for (auto operand : info.product->getOperands()) {
        if (!ensureClonedValue(module, target, operand, valueMap, &oldThen))
          return false;
      }
      if (!cloneSimpleIntoBlock(module, target, *info.product, valueMap))
        return false;
      auto productIt = valueMap.find(valueKey(info.product->result()));
      if (productIt == valueMap.end() || !productIt->second.valid())
        return false;
      Operation &sum = appendOp(target, "rv_machine.addw",
                                {oldAcc.result(), productIt->second}, {i32},
                                {}, loc);
      appendOp(target, "memref.store", {sum.result(), rowBuf.result(), newJ},
               {}, {}, loc);
      return true;
    };

    auto appendReductionJLoop =
        [&](Block &container, bool guarded,
            const std::map<std::string, Value> &incomingMap) -> bool {
      Value innerJLower = tileMaterializeValue(module, container,
                                               info.jLoop->operand(0), loc);
      Value innerJUpper = tileMaterializeValue(module, container,
                                               info.jLoop->operand(1), loc);
      Value innerJStep = tileMaterializeValue(module, container,
                                              info.jLoop->operand(2), loc);
      Operation &jLoop = tileAppendAffineLoop(module, container, innerJLower,
                                              innerJUpper, innerJStep, loc, "j");
      Block &jBody = *jLoop.getRegions()[0]->getBlocks()[0];
      Value newJ = jBody.args()[0]->value();
      std::map<std::string, Value> valueMap = incomingMap;
      valueMap[valueKey(oldJ)] = newJ;
      if (!guarded) {
        if (!appendAccumulate(jBody, newJ, valueMap))
          return false;
        appendOp(jBody, "affine.yield", {}, {}, {}, loc);
        return true;
      }

      for (Operation *op : info.conditionPrefixOps) {
        if (!cloneSimpleIntoBlock(module, jBody, *op, valueMap))
          return false;
      }
      auto condIt = valueMap.find(valueKey(info.ifOp->operand(0)));
      if (condIt == valueMap.end() || !condIt->second.valid())
        return false;

      Operation &newIf = appendOp(jBody, "scf.if", {condIt->second}, {}, {},
                                  info.ifOp->loc(), 1);
      Block &thenBody = newIf.getRegions()[0]->addBlock();
      if (!appendAccumulate(thenBody, newJ, valueMap))
        return false;
      appendOp(thenBody, "scf.yield", {}, {}, {}, loc);
      appendOp(jBody, "affine.yield", {}, {}, {}, loc);
      return true;
    };

    bool useParityFastPath = info.parityFastPath;
    Value parityInvariantValue;
    if (useParityFastPath) {
      if (!ensureClonedValue(module, kBody, info.parityInvariant, kValueMap,
                             oldKBody)) {
        useParityFastPath = false;
      } else {
        auto invariantIt = kValueMap.find(valueKey(info.parityInvariant));
        if (invariantIt == kValueMap.end() || !invariantIt->second.valid())
          useParityFastPath = false;
        else
          parityInvariantValue = invariantIt->second;
      }
    }

    if (useParityFastPath) {
      Value one = tileAppendIntConstant(module, kBody, 1, loc);
      Operation &odd = appendOp(kBody, "rv_machine.and",
                                {parityInvariantValue, one}, {i32}, {}, loc);
      Operation &even = appendOp(kBody, "rv_machine.seqz",
                                 {odd.result()}, {i32}, {}, loc);
      Operation &fastIf = appendOp(kBody, "scf.if", {even.result()}, {}, {},
                                   info.ifOp->loc(), 2);
      Block &fastThen = fastIf.getRegions()[0]->addBlock();
      if (!appendReductionJLoop(fastThen, false, kValueMap))
        return false;
      appendOp(fastThen, "scf.yield", {}, {}, {}, loc);

      Block &fastElse = fastIf.getRegions()[1]->addBlock();
      if (!appendReductionJLoop(fastElse, true, kValueMap))
        return false;
      appendOp(fastElse, "scf.yield", {}, {}, {}, loc);
    } else {
      if (!appendReductionJLoop(kBody, true, kValueMap))
        return false;
    }
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);
  }

  Value writeJLower = tileMaterializeValue(module, outerBody,
                                           info.jLoop->operand(0), loc);
  Value writeJUpper = tileMaterializeValue(module, outerBody,
                                           info.jLoop->operand(1), loc);
  Value writeJStep = tileMaterializeValue(module, outerBody,
                                          info.jLoop->operand(2), loc);
  Operation &writeLoop = tileAppendAffineLoop(module, outerBody, writeJLower,
                                              writeJUpper, writeJStep, loc, "j");
  {
    Block &body = *writeLoop.getRegions()[0]->getBlocks()[0];
    Value newJ = body.args()[0]->value();
    Operation &val = appendOp(body, "memref.load", {rowBuf.result(), newJ},
                              {i32}, {}, loc);
    appendOp(body, "memref.store",
             {val.result(), info.finalStore->operand(1), newI, newJ},
             {}, info.finalStore->attrs(), loc);
    appendOp(body, "affine.yield", {}, {}, {}, loc);
  }
  appendOp(outerBody, "affine.yield", {}, {}, {}, loc);

  outer.markErased();
  if (stats) {
    stats->rowBufferedReductions++;
    stats->reductionBlocks++;
    stats->loopTiles++;
    stats->polyTiles++;
    stats->imperfectInterchanges++;
  }
  return true;
}

static bool applyRowBufferedReduction(Module &module, const RowReductionInfo &info,
                                      SelfOptStats *stats) {
  Operation &outer = *info.outer;
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;
  Context &ctx = module.context();
  Location loc = outer.loc();
  Type i32 = ctx.i(32);
  MemrefInfo outInfo = parseMemrefInfo(info.finalStore->operand(1).type());
  int64_t rowElements = outInfo.shape.size() >= 2 && outInfo.shape[1] > 0
                            ? outInfo.shape[1]
                            : 1024;

  std::size_t insertIndex = (std::size_t) outerIndex;
  Operation &rowBuf = insertOp(
      *parent, insertIndex, "sysy.alloca", {}, {ctx.memref({rowElements}, i32)},
      {{"symbol", ctx.stringAttr(".tile_rowbuf")}}, loc);
  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "i");
  Block &outerBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value iIv = outerBody.args()[0]->value();
  Value jLower = tileMaterializeValue(module, outerBody, info.jLoop->operand(0), loc);
  Value jUpper = tileMaterializeValue(module, outerBody, info.jLoop->operand(1), loc);
  Value jStep = tileMaterializeValue(module, outerBody, info.jLoop->operand(2), loc);
  Value kLower = tileMaterializeValue(module, outerBody, info.kLoop->operand(0), loc);
  Value kUpper = tileMaterializeValue(module, outerBody, info.kLoop->operand(1), loc);
  Value kStep = tileMaterializeValue(module, outerBody, info.kLoop->operand(2), loc);

  Operation &zeroLoop = tileAppendAffineLoop(module, outerBody, jLower, jUpper,
                                             jStep, loc, "j");
  {
    Block &body = *zeroLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = body.args()[0]->value();
    Value zero = tileAppendIntConstant(module, body, 0, loc);
    appendOp(body, "memref.store", {zero, rowBuf.result(), jIv}, {}, {}, loc);
    appendOp(body, "affine.yield", {}, {}, {}, loc);
  }

  Operation &kLoop = tileAppendAffineLoop(module, outerBody, kLower, kUpper,
                                          kStep, loc, "k");
  {
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value kIv = kBody.args()[0]->value();
    Operation &lhs = appendOp(kBody, "memref.load",
                              {info.lhsLoad->operand(0), iIv, kIv}, {i32},
                              info.lhsLoad->attrs(), loc);
    Operation &jLoop = tileAppendAffineLoop(module, kBody, jLower, jUpper,
                                            jStep, loc, "j");
    Block &jBody = *jLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = jBody.args()[0]->value();
    Operation &old = appendOp(jBody, "memref.load", {rowBuf.result(), jIv},
                              {i32}, {}, loc);
    Operation &rhs = appendOp(jBody, "memref.load",
                              {info.rhsLoad->operand(0), kIv, jIv}, {i32},
                              info.rhsLoad->attrs(), loc);
    Operation &prod = appendOp(jBody, "rv_machine.mulw",
                               {lhs.result(), rhs.result()}, {i32}, {}, loc);
    Operation &sum = appendOp(jBody, "rv_machine.addw",
                              {old.result(), prod.result()}, {i32}, {}, loc);
    appendOp(jBody, "memref.store", {sum.result(), rowBuf.result(), jIv},
             {}, {}, loc);
    appendOp(jBody, "affine.yield", {}, {}, {}, loc);
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);
  }

  Operation &writeLoop = tileAppendAffineLoop(module, outerBody, jLower, jUpper,
                                              jStep, loc, "j");
  {
    Block &body = *writeLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = body.args()[0]->value();
    Operation &val = appendOp(body, "memref.load", {rowBuf.result(), jIv},
                              {i32}, {}, loc);
    appendOp(body, "memref.store",
             {val.result(), info.finalStore->operand(1), iIv, jIv},
             {}, info.finalStore->attrs(), loc);
    appendOp(body, "affine.yield", {}, {}, {}, loc);
  }
  appendOp(outerBody, "affine.yield", {}, {}, {}, loc);

  outer.markErased();
  if (stats) {
    stats->rowBufferedReductions++;
    stats->reductionBlocks++;
    stats->loopTiles++;
    stats->polyTiles++;
    stats->imperfectInterchanges++;
  }
  return true;
}

static Value tileAppendAddI32(Module &module, Block &block, Value lhs, Value rhs,
                              Location loc) {
  Operation &op = appendOp(block, "rv_machine.addw", {lhs, rhs},
                           {module.context().i(32)}, {}, loc);
  return op.result();
}

static Value tileAppendSubI32(Module &module, Block &block, Value lhs, Value rhs,
                              Location loc) {
  Operation &op = appendOp(block, "rv_machine.subw", {lhs, rhs},
                           {module.context().i(32)}, {}, loc);
  return op.result();
}

static Value tileAppendRemI32(Module &module, Block &block, Value lhs, Value rhs,
                              Location loc) {
  Operation &op = appendOp(block, "rv_machine.remw", {lhs, rhs},
                           {module.context().i(32)}, {}, loc);
  return op.result();
}

static Value tileAppendLaneIndex(Module &module, Block &block, Value base,
                                 int lane, Location loc) {
  if (lane == 0)
    return base;
  Value laneConst = tileAppendIntConstant(module, block, lane, loc);
  return tileAppendAddI32(module, block, base, laneConst, loc);
}

static void appendScalarZeroStore(Module &module, Block &block, Value slot,
                                  Location loc) {
  Value zero = tileAppendIntConstant(module, block, 0, loc);
  appendOp(block, "sysy.store", {zero, slot}, {}, {}, loc);
}

static Value appendScalarLoad(Module &module, Block &block, Value slot,
                              Location loc) {
  Operation &op = appendOp(block, "sysy.load", {slot}, {module.context().i(32)},
                           {}, loc);
  return op.result();
}

static void appendScalarStore(Block &block, Value value, Value slot,
                              Location loc) {
  appendOp(block, "sysy.store", {value, slot}, {}, {}, loc);
}

static bool applyRegisterBlockedReduction(Module &module,
                                          const RowReductionInfo &info,
                                          SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_REDUCTION_REG", true))
    return false;
  Operation &outer = *info.outer;
  Block *parent = outer.getBlock();
  if (!parent)
    return false;
  int outerIndex = operationIndexInBlock(*parent, &outer);
  if (outerIndex < 0)
    return false;

  Context &ctx = module.context();
  Location loc = outer.loc();
  Type i32 = ctx.i(32);
  constexpr int kBlock = 4;
  std::size_t insertIndex = (std::size_t) outerIndex;

  std::vector<Value> accSlots;
  accSlots.reserve(kBlock);
  for (int lane = 0; lane < kBlock; lane++) {
    Operation &slot = insertOp(
        *parent, insertIndex, "sysy.alloca", {}, {ctx.memref({1}, i32)},
        {{"symbol", ctx.stringAttr(".tile_acc" + std::to_string(lane))},
         {"scalar_promote", ctx.stringAttr("forced")}},
        loc);
    accSlots.push_back(slot.result());
  }

  Operation &newOuter = tileInsertAffineLoop(module, *parent, insertIndex,
                                             outer.operand(0), outer.operand(1),
                                             outer.operand(2), loc, "i");
  Block &outerBody = *newOuter.getRegions()[0]->getBlocks()[0];
  Value iIv = outerBody.args()[0]->value();
  Value jLower = tileMaterializeValue(module, outerBody, info.jLoop->operand(0), loc);
  Value jUpper = tileMaterializeValue(module, outerBody, info.jLoop->operand(1), loc);
  Value kLower = tileMaterializeValue(module, outerBody, info.kLoop->operand(0), loc);
  Value kUpper = tileMaterializeValue(module, outerBody, info.kLoop->operand(1), loc);
  Value kStep = tileMaterializeValue(module, outerBody, info.kLoop->operand(2), loc);
  Value four = tileAppendIntConstant(module, outerBody, kBlock, loc);
  Value delta = tileAppendSubI32(module, outerBody, jUpper, jLower, loc);
  Value rem = tileAppendRemI32(module, outerBody, delta, four, loc);
  Value mainUpper = tileAppendSubI32(module, outerBody, jUpper, rem, loc);

  Operation &jBlockLoop = tileAppendAffineLoop(module, outerBody, jLower,
                                               mainUpper, four, loc, "jb");
  {
    Block &jbBody = *jBlockLoop.getRegions()[0]->getBlocks()[0];
    Value jbIv = jbBody.args()[0]->value();
    std::vector<Value> lanes;
    lanes.reserve(kBlock);
    for (int lane = 0; lane < kBlock; lane++) {
      appendScalarZeroStore(module, jbBody, accSlots[lane], loc);
      lanes.push_back(tileAppendLaneIndex(module, jbBody, jbIv, lane, loc));
    }

    Operation &kLoop = tileAppendAffineLoop(module, jbBody, kLower, kUpper,
                                            kStep, loc, "k");
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value kIv = kBody.args()[0]->value();
    Operation &lhs = appendOp(kBody, "memref.load",
                              {info.lhsLoad->operand(0), iIv, kIv}, {i32},
                              info.lhsLoad->attrs(), loc);
    for (int lane = 0; lane < kBlock; lane++) {
      Value old = appendScalarLoad(module, kBody, accSlots[lane], loc);
      Operation &rhs = appendOp(kBody, "memref.load",
                                {info.rhsLoad->operand(0), kIv, lanes[lane]},
                                {i32}, info.rhsLoad->attrs(), loc);
      Operation &prod = appendOp(kBody, "rv_machine.mulw",
                                 {lhs.result(), rhs.result()}, {i32}, {}, loc);
      Operation &sum = appendOp(kBody, "rv_machine.addw",
                                {old, prod.result()}, {i32}, {}, loc);
      appendScalarStore(kBody, sum.result(), accSlots[lane], loc);
    }
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);

    for (int lane = 0; lane < kBlock; lane++) {
      Value val = appendScalarLoad(module, jbBody, accSlots[lane], loc);
      appendOp(jbBody, "memref.store",
               {val, info.finalStore->operand(1), iIv, lanes[lane]},
               {}, info.finalStore->attrs(), loc);
    }
    appendOp(jbBody, "affine.yield", {}, {}, {}, loc);
  }

  Operation &tailLoop = tileAppendAffineLoop(module, outerBody, mainUpper, jUpper,
                                             tileMaterializeValue(module, outerBody,
                                                                  info.jLoop->operand(2), loc),
                                             loc, "j");
  {
    Block &tailBody = *tailLoop.getRegions()[0]->getBlocks()[0];
    Value jIv = tailBody.args()[0]->value();
    appendScalarZeroStore(module, tailBody, accSlots[0], loc);
    Operation &kLoop = tileAppendAffineLoop(module, tailBody, kLower, kUpper,
                                            kStep, loc, "k");
    Block &kBody = *kLoop.getRegions()[0]->getBlocks()[0];
    Value kIv = kBody.args()[0]->value();
    Operation &lhs = appendOp(kBody, "memref.load",
                              {info.lhsLoad->operand(0), iIv, kIv}, {i32},
                              info.lhsLoad->attrs(), loc);
    Value old = appendScalarLoad(module, kBody, accSlots[0], loc);
    Operation &rhs = appendOp(kBody, "memref.load",
                              {info.rhsLoad->operand(0), kIv, jIv}, {i32},
                              info.rhsLoad->attrs(), loc);
    Operation &prod = appendOp(kBody, "rv_machine.mulw",
                               {lhs.result(), rhs.result()}, {i32}, {}, loc);
    Operation &sum = appendOp(kBody, "rv_machine.addw",
                              {old, prod.result()}, {i32}, {}, loc);
    appendScalarStore(kBody, sum.result(), accSlots[0], loc);
    appendOp(kBody, "affine.yield", {}, {}, {}, loc);
    Value val = appendScalarLoad(module, tailBody, accSlots[0], loc);
    appendOp(tailBody, "memref.store",
             {val, info.finalStore->operand(1), iIv, jIv},
             {}, info.finalStore->attrs(), loc);
    appendOp(tailBody, "affine.yield", {}, {}, {}, loc);
  }

  appendOp(outerBody, "affine.yield", {}, {}, {}, loc);
  outer.markErased();
  if (stats) {
    stats->reductionBlocks++;
    stats->loopTiles++;
    stats->polyTiles++;
    stats->reductionRegs += kBlock;
    stats->imperfectInterchanges++;
  }
  return true;
}

static bool stripMineInnermostAffineLoop(Module &module, Operation &loop,
                                         int64_t tileSize, SelfOptStats *stats) {
  (void) stats;
  if (loop.name() != "affine.for" || loop.operandCount() < 3 ||
      loop.getRegions().size() != 1)
    return false;
  int64_t step = 0;
  if (!constantIntegerValue(loop.operand(2), step) || step != 1)
    return false;
  Block *body = tileSingleBlock(loop);
  if (!body || body->args().empty() || tileBlockHasNestedLoop(*body) ||
      tileBlockUnsafeForStripMining(*body))
    return false;
  bool hasMemrefAccess = false;
  for (auto &owned : body->ops()) {
    if (!owned || owned->isErased())
      continue;
    if (tileOpTreeHasAnyName(*owned, {"memref.load", "memref.store"})) {
      hasMemrefAccess = true;
      break;
    }
  }
  if (!hasMemrefAccess)
    return false;
  int64_t lowerImm = 0;
  int64_t upperImm = 0;
  if (!constantIntegerValue(loop.operand(0), lowerImm) ||
      !constantIntegerValue(loop.operand(1), upperImm))
    return false;
  if (upperImm - lowerImm <= tileSize)
    return false;

  Block *parent = loop.getBlock();
  if (!parent)
    return false;
  int loopIndex = operationIndexInBlock(*parent, &loop);
  if (loopIndex < 0)
    return false;

  Context &ctx = module.context();
  Location loc = loop.loc();
  std::size_t insertIndex = (std::size_t) loopIndex;
  Operation &tileConst = insertOp(
      *parent, insertIndex, "rv_machine.li", {}, {ctx.i(32)},
      {{"value", ctx.integerAttr(tileSize, ctx.i(32))}}, loc);
  Value tileStep = tileConst.result();
  Operation &tileLoop = tileInsertAffineLoop(module, *parent, insertIndex,
                                             loop.operand(0), loop.operand(1),
                                             tileStep, loc, "tile");
  Block &tileBody = *tileLoop.getRegions()[0]->getBlocks()[0];
  Value tileIv = tileBody.args()[0]->value();
  Operation &tileEndRaw = appendOp(tileBody, "rv_machine.addw",
                                   {tileIv, tileStep}, {ctx.i(32)}, {}, loc);
  Value tileEnd = tileAppendMinValue(module, tileBody, tileEndRaw.result(),
                                     loop.operand(1), loc);
  Operation &inner = tileAppendAffineLoop(module, tileBody, tileIv, tileEnd,
                                          loop.operand(2), loc, "iv");
  Block &innerBody = *inner.getRegions()[0]->getBlocks()[0];
  Value oldIv = body->args()[0]->value();
  Value newIv = innerBody.args()[0]->value();
  std::map<std::string, Value> valueMap;
  valueMap[valueKey(oldIv)] = newIv;
  std::set<Operation*> skipOps;
  for (auto &owned : body->ops()) {
    if (!owned || owned->isErased())
      continue;
    auto cloned = cloneForUnrolledIteration(module, *owned, valueMap, skipOps,
                                            Value(), 0);
    if (!cloned)
      continue;
    Operation &inserted = innerBody.addOperation(std::move(cloned));
    for (int i = 0; i < owned->resultCount(); i++)
      valueMap[valueKey(owned->result(i))] = inserted.result(i);
  }
  appendOp(tileBody, "affine.yield", {}, {}, {}, loc);
  loop.markErased();
  return true;
}

static void runSafeLoopTiling(Module &module, SelfOptStats *stats) {
  int64_t tileSize = 32;
  if (const char *ts = std::getenv("SISY_SELF_TILE_SIZE")) {
    try {
      tileSize = std::stoll(ts);
    } catch (...) {
      tileSize = 32;
    }
    if (tileSize <= 1)
      tileSize = 32;
  }

  std::vector<Operation*> whileLoops;
  if (envEnabled("SISY_ENABLE_SELF_WHILE_IV_CACHE", true)) {
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "scf.while")
        whileLoops.push_back(op);
    }
    for (Operation *loop : whileLoops) {
      if (loop && !loop->isErased())
        cacheWhileLoopIvLoads(module, *loop, stats);
    }
    eraseMarked(module);
  }

  std::vector<Operation*> loops;
  if (envEnabled("SISY_ENABLE_SELF_ROW_BUFFER_TILE", true)) {
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "affine.for")
        loops.push_back(op);
    }
    for (Operation *loop : loops) {
      if (!loop || loop->isErased())
        continue;
      RowReductionInfo info;
      if (classifyRowBufferedReduction(*loop, info)) {
        if (!applyRegisterBlockedReduction(module, info, stats))
          applyRowBufferedReduction(module, info, stats);
      } else {
        ConditionalRowReductionInfo conditionalInfo;
        if (classifyConditionalRowReduction(*loop, conditionalInfo)) {
          applyConditionalRowBufferedReduction(module, conditionalInfo, stats);
        } else if (stats) {
          stats->tileSkippedShape++;
        }
      }
    }
    eraseMarked(module);
  }

  if (envEnabled("SISY_ENABLE_SELF_STRIP_TILE",
                 envEnabled("SISY_ENABLE_SELF_POLY_TILE", false))) {
    loops.clear();
    for (auto *op : walk(module)) {
      if (op && !op->isErased() && op->name() == "affine.for")
        loops.push_back(op);
    }
    for (Operation *loop : loops) {
      if (!loop || loop->isErased())
        continue;
      if (stripMineInnermostAffineLoop(module, *loop, tileSize, stats)) {
        if (stats) {
          stats->stencilTiles++;
          stats->polyTiles++;
        }
      } else if (stats) {
        stats->tileSkippedShape++;
      }
    }
    eraseMarked(module);
  }
}

void runLoopTiling(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_TILE", true))
    return;
  if (envEnabled("SISY_ENABLE_SELF_TILE_LEGACY", false)) {
    runAffineLoopTiling(module);
    return;
  }
  runSafeLoopTiling(module, stats);
}

void runLoopRepeatReduction(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_REPEAT_REDUCTION", true))
    return;
  std::vector<Operation*> loops;
  for (auto *op : walk(module))
    if (op && !op->isErased() && op->name() == "affine.for")
      loops.push_back(op);
  bool changed = false;
  for (Operation *loop : loops) {
    if (!loop || loop->isErased())
      continue;
    LinearModRecurrenceInfo modInfo = classifyLinearModRecurrenceLoop(*loop);
    if (modInfo.valid && applyLinearModRecurrenceLoop(module, *loop, modInfo)) {
      changed = true;
      if (stats)
        stats->addrIvRewrites++;
      continue;
    }
    RepeatReductionInfo info = classifyRepeatReductionLoop(*loop);
    if (!info.valid)
      continue;
    if (applyRepeatReductionLoop(module, *loop, info)) {
      changed = true;
      if (stats)
        stats->imperfectInterchanges++;
    }
  }
  if (changed)
    eraseMarked(module);
}

void runLoopAddressIV(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_ADDR_IV", true))
    return;
  runLoopAddressIVInRegion(module, module.body(), stats);
  eraseMarked(module);
}

static bool isSchedulableLoad(Operation *op) {
  return op && !op->isErased() && op->getRegions().empty() &&
         (op->name() == "sysy.load" || op->name() == "memref.load") &&
         op->resultCount() > 0;
}

static bool isPureArithmeticForSchedule(Operation *op) {
  return op && !op->isErased() && op->getRegions().empty() &&
         op->dialect() == "arith" && !op->isTerminator();
}

static bool opUsesResultOf(Operation *user, Operation *def) {
  if (!user || !def)
    return false;
  for (auto operand : user->getOperands()) {
    if (operand.getDefiningOp() == def)
      return true;
  }
  return false;
}

static void runLoopLocalSchedulerInRegion(Region &region, SelfOptStats *stats);

static void runLoopLocalSchedulerInBlock(Block &block, SelfOptStats *stats) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 1; i < block.ops().size(); i++) {
      Operation *load = block.ops()[i].get();
      Operation *prev = block.ops()[i - 1].get();
      if (!isSchedulableLoad(load) || !isPureArithmeticForSchedule(prev))
        continue;
      if (opUsesResultOf(load, prev))
        continue;
      auto moved = block.takeOperation(load);
      block.insertOperation(i - 1, std::move(moved));
      if (stats)
        stats->schedulerMoves++;
      changed = true;
      if (i > 1)
        i -= 2;
    }
  }

  for (auto &owned : block.ops()) {
    if (!owned || owned->isErased())
      continue;
    for (auto &nested : owned->getRegions())
      runLoopLocalSchedulerInRegion(*nested, stats);
  }
}

static void runLoopLocalSchedulerInRegion(Region &region, SelfOptStats *stats) {
  for (auto &block : region.getBlocks())
    runLoopLocalSchedulerInBlock(*block, stats);
}

void runLoopLocalScheduler(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_SCHED", true))
    return;
  for (auto &region : module.op().getRegions())
    runLoopLocalSchedulerInRegion(*region, stats);
}

namespace {

enum class ProvenBitwiseKind {
  None,
  And,
  Or,
  Xor,
};

struct ProvenBitwiseFunction {
  ProvenBitwiseKind kind = ProvenBitwiseKind::None;
  Operation *func = nullptr;
};

static std::vector<Operation*> collectNestedOps(Operation &root) {
  std::vector<Operation*> ops;
  std::function<void(Operation&)> rec = [&](Operation &op) {
    ops.push_back(&op);
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          rec(*child);
  };
  rec(root);
  return ops;
}

static bool opHasName(Operation *op, std::initializer_list<const char*> names) {
  if (!op)
    return false;
  for (const char *name : names)
    if (op->name() == name)
      return true;
  return false;
}

static bool isConst(Value value, int64_t expected) {
  int64_t actual = 0;
  return constantIntegerValue(value, actual) && actual == expected;
}

static bool isLoadFromSlot(Value value, Value slot) {
  auto *op = value.getDefiningOp();
  return opHasName(op, {"sysy.load", "memref.load"}) &&
         op->operandCount() > 0 && op->operand(0) == slot;
}

static bool isStoreToSlot(Operation *op, Value slot) {
  return opHasName(op, {"sysy.store", "memref.store"}) &&
         op->operandCount() >= 2 && op->operand(1) == slot;
}

static bool isBinaryWithConst(Value value, const char *arithName,
                              const char *rvName, const char *armName,
                              Value slot, int64_t constant,
                              bool commutative = false) {
  auto *op = value.getDefiningOp();
  if (!opHasName(op, {arithName, rvName, armName}) || op->operandCount() != 2)
    return false;
  if (isLoadFromSlot(op->operand(0), slot) && isConst(op->operand(1), constant))
    return true;
  return commutative && isLoadFromSlot(op->operand(1), slot) &&
         isConst(op->operand(0), constant);
}

static bool isSubSlotByOne(Value value, Value slot) {
  auto *op = value.getDefiningOp();
  if (!opHasName(op, {"arith.subi", "rv_machine.subw", "arm_machine.sub"}) ||
      op->operandCount() != 2)
    return false;
  return isLoadFromSlot(op->operand(0), slot) && isConst(op->operand(1), 1);
}

static bool isEqBitToOne(Value value, const std::set<std::string> &bitSlots) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "eq")
    return false;
  auto matches = [&](Value maybeBit, Value maybeOne) {
    if (isConst(maybeOne, 1) && bitSlots.count(valueKey(maybeBit)) != 0)
      return true;
    auto *load = maybeBit.getDefiningOp();
    return isConst(maybeOne, 1) && opHasName(load, {"sysy.load", "memref.load"}) &&
           load->operandCount() > 0 && bitSlots.count(valueKey(load->operand(0))) != 0;
  };
  return matches(cmp->operand(0), cmp->operand(1)) ||
         matches(cmp->operand(1), cmp->operand(0));
}

static bool isTruthOfEqBitToOne(Value value, const std::set<std::string> &bitSlots) {
  if (isEqBitToOne(value, bitSlots))
    return true;
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "ne")
    return false;
  return (isEqBitToOne(cmp->operand(0), bitSlots) && isConst(cmp->operand(1), 0)) ||
         (isEqBitToOne(cmp->operand(1), bitSlots) && isConst(cmp->operand(0), 0));
}

static bool isOrOfEqBitToOne(Value value, const std::set<std::string> &bitSlots) {
  auto *op = value.getDefiningOp();
  if (!opHasName(op, {"arith.ori", "rv_machine.or", "arm_machine.orr"}) ||
      op->operandCount() != 2)
    return false;
  return isEqBitToOne(op->operand(0), bitSlots) &&
         isEqBitToOne(op->operand(1), bitSlots);
}

static bool isNeBetweenBitLoads(Value value, const std::set<std::string> &bitSlots) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "ne")
    return false;
  for (int i = 0; i < 2; i++) {
    if (bitSlots.count(valueKey(cmp->operand(i))) != 0)
      continue;
    auto *load = cmp->operand(i).getDefiningOp();
    if (!opHasName(load, {"sysy.load", "memref.load"}) || load->operandCount() == 0 ||
        bitSlots.count(valueKey(load->operand(0))) == 0)
      return false;
  }
  return true;
}

static bool isResultPlusPowerStore(Operation *store, Value resultSlot, Value powerSlot) {
  if (!isStoreToSlot(store, resultSlot))
    return false;
  auto *add = store->operand(0).getDefiningOp();
  if (!opHasName(add, {"arith.addi", "rv_machine.addw", "arm_machine.add"}) ||
      add->operandCount() != 2)
    return false;
  return (isLoadFromSlot(add->operand(0), resultSlot) && isLoadFromSlot(add->operand(1), powerSlot)) ||
         (isLoadFromSlot(add->operand(1), resultSlot) && isLoadFromSlot(add->operand(0), powerSlot));
}

static bool blockHasResultPlusPowerStore(Block &block, Value resultSlot, Value powerSlot) {
  for (auto &owned : block.ops()) {
    if (owned && !owned->isErased() &&
        isResultPlusPowerStore(owned.get(), resultSlot, powerSlot))
      return true;
  }
  return false;
}

static bool blockStoresConstToSlot(Block &block, Value slot, int64_t constant) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (op && !op->isErased() && isStoreToSlot(op, slot) &&
        isConst(op->operand(0), constant))
      return true;
  }
  return false;
}

static bool blockStoresTruthToSlot(Block &block, Value slot,
                                   const std::set<std::string> &bitSlots) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (op && !op->isErased() && isStoreToSlot(op, slot) &&
        isTruthOfEqBitToOne(op->operand(0), bitSlots))
      return true;
  }
  return false;
}

static bool isShortCircuitLogicIf(Operation *op,
                                  const std::set<std::string> &bitSlots,
                                  Value &slot, ProvenBitwiseKind &kind) {
  if (!op || op->name() != "scf.if" || op->operandCount() != 1 ||
      op->getRegions().size() != 2 ||
      op->getRegions()[0]->getBlocks().empty() ||
      op->getRegions()[1]->getBlocks().empty() ||
      !isTruthOfEqBitToOne(op->operand(0), bitSlots))
    return false;
  Block &thenBlock = *op->getRegions()[0]->getBlocks()[0];
  Block &elseBlock = *op->getRegions()[1]->getBlocks()[0];
  auto considerStore = [&](Block &block) -> Value {
    for (auto &owned : block.ops()) {
      Operation *store = owned.get();
      if (store && !store->isErased() && isStoreToSlot(store, store->operandCount() >= 2
                                                                  ? store->operand(1)
                                                                  : Value()))
        return store->operand(1);
    }
    return Value();
  };
  Value thenSlot = considerStore(thenBlock);
  Value elseSlot = considerStore(elseBlock);
  if (!thenSlot.valid() || thenSlot != elseSlot)
    return false;
  if (blockStoresTruthToSlot(thenBlock, thenSlot, bitSlots) &&
      blockStoresConstToSlot(elseBlock, thenSlot, 0)) {
    slot = thenSlot;
    kind = ProvenBitwiseKind::And;
    return true;
  }
  if (blockStoresConstToSlot(thenBlock, thenSlot, 1) &&
      blockStoresTruthToSlot(elseBlock, thenSlot, bitSlots)) {
    slot = thenSlot;
    kind = ProvenBitwiseKind::Or;
    return true;
  }
  return false;
}

static bool isShortCircuitOrIf(Operation *op, const std::set<std::string> &bitSlots,
                               Value resultSlot, Value powerSlot) {
  if (!op || op->name() != "scf.if" || op->operandCount() != 1 ||
      op->getRegions().size() != 2 ||
      op->getRegions()[0]->getBlocks().empty() ||
      op->getRegions()[1]->getBlocks().empty())
    return false;
  if (!isEqBitToOne(op->operand(0), bitSlots))
    return false;

  Block &outerThen = *op->getRegions()[0]->getBlocks()[0];
  Block &outerElse = *op->getRegions()[1]->getBlocks()[0];
  if (!blockHasResultPlusPowerStore(outerThen, resultSlot, powerSlot))
    return false;

  Operation *innerIf = nullptr;
  int innerIfCount = 0;
  for (auto &owned : outerElse.ops()) {
    Operation *child = owned.get();
    if (!child || child->isErased() || child->name() == "scf.yield")
      continue;
    if (child->name() == "scf.if") {
      innerIf = child;
      innerIfCount++;
    }
  }
  if (innerIfCount != 1 || !innerIf || innerIf->operandCount() != 1 ||
      innerIf->getRegions().empty() || innerIf->getRegions()[0]->getBlocks().empty())
    return false;
  if (!isEqBitToOne(innerIf->operand(0), bitSlots))
    return false;
  Block &innerThen = *innerIf->getRegions()[0]->getBlocks()[0];
  return blockHasResultPlusPowerStore(innerThen, resultSlot, powerSlot);
}

static bool valueStaticallyNonNegativeImpl(Value value, std::set<std::string> &visiting,
                                           int depth, SelfOptStats *stats) {
  if (!value.valid() || depth > 8)
    return false;
  std::string key = valueKey(value);
  if (!visiting.insert(key).second)
    return false;
  int64_t c = 0;
  if (constantIntegerValue(value, c)) {
    visiting.erase(key);
    return c >= 0;
  }
  auto *op = value.getDefiningOp();
  if (!op) {
    visiting.erase(key);
    return false;
  }
  auto prove = [&](Value operand) {
    return valueStaticallyNonNegativeImpl(operand, visiting, depth + 1, stats);
  };
  bool proven = false;
  if (opHasName(op, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp",
                     "arith.noti", "rv_machine.seqz", "arm_machine.not"})) {
    proven = true;
  } else if (opHasName(op, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
      op->operandCount() == 2) {
    int64_t mask = 0;
    proven = (constantIntegerValue(op->operand(0), mask) && mask >= 0) ||
             (constantIntegerValue(op->operand(1), mask) && mask >= 0) ||
             prove(op->operand(0)) || prove(op->operand(1));
  } else if (opHasName(op, {"arith.ori", "rv_machine.or", "arm_machine.orr",
                            "arith.xori", "rv_machine.xor", "arm_machine.eor"}) &&
             op->operandCount() == 2) {
    proven = prove(op->operand(0)) && prove(op->operand(1));
  }
  visiting.erase(key);
  if (proven && stats)
    stats->bitwiseStaticProofs++;
  return proven;
}

static bool valueStaticallyNonNegative(Value value, SelfOptStats *stats = nullptr) {
  std::set<std::string> visiting;
  return valueStaticallyNonNegativeImpl(value, visiting, 0, stats);
}

static ProvenBitwiseFunction classifyProvenBitwiseFunction(Operation &func,
                                                           SelfOptStats *stats) {
  ProvenBitwiseFunction result;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return result;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  if (entry.args().size() != 2)
    return result;

  auto allOps = collectNestedOps(func);
  int loopCount = 0;
  Operation *loop = nullptr;
  for (auto *op : allOps) {
    if (!op || op == &func || op->isErased())
      continue;
    if (op->name() == "sysy.call") {
      if (stats)
        stats->bitwiseRejectImpure++;
      return result;
    }
    if (op->name() == "memref.load" || op->name() == "memref.store")
      return result;
    if (op->name() == "scf.while" || op->name() == "affine.for") {
      loopCount++;
      loop = op;
    }
  }
  if (loopCount != 1 || !loop || loop->name() != "scf.while" ||
      loop->getRegions().size() < 2 || loop->getRegions()[1]->getBlocks().empty())
    return result;

  std::map<std::string, int64_t> initConstants;
  std::set<std::string> paramSlots;
  for (auto &owned : entry.ops()) {
    auto *op = owned.get();
    if (!op || op == loop)
      break;
    if (!opHasName(op, {"sysy.store"}) || op->operandCount() < 2)
      continue;
    int64_t init = 0;
    if (constantIntegerValue(op->operand(0), init))
      initConstants[valueKey(op->operand(1))] = init;
    if (op->operand(0).isBlockArgument())
      paramSlots.insert(valueKey(op->operand(1)));
  }
  if (paramSlots.size() != 2)
    return result;

  Value resultSlot;
  for (auto &owned : entry.ops()) {
    auto *op = owned.get();
    if (!opHasName(op, {"sysy.return"}) || op->operandCount() == 0)
      continue;
    auto *load = op->operand(0).getDefiningOp();
    auto initIt = load && load->operandCount() > 0
                      ? initConstants.find(valueKey(load->operand(0)))
                      : initConstants.end();
    if (opHasName(load, {"sysy.load"}) && load->operandCount() > 0 &&
        initIt != initConstants.end() && initIt->second == 0) {
      resultSlot = load->operand(0);
      break;
    }
  }
  if (!resultSlot.valid())
    return result;

  Value lenSlot, powerSlot;
  for (const auto &kv : initConstants) {
    if (kv.second == 32) {
      for (auto &owned : entry.ops()) {
        if (owned && owned->resultCount() && valueKey(owned->result()) == kv.first)
          lenSlot = owned->result();
      }
    } else if (kv.second == 1) {
      for (auto &owned : entry.ops()) {
        if (owned && owned->resultCount() && valueKey(owned->result()) == kv.first)
          powerSlot = owned->result();
      }
    }
  }
  if (!lenSlot.valid() || !powerSlot.valid())
    return result;

  Block &body = *loop->getRegions()[1]->getBlocks()[0];
  std::set<std::string> bitSlots;
  std::set<std::string> bitRefs;
  std::set<std::string> paramsWithRem;
  std::set<std::string> paramsWithDiv;
  std::map<std::string, ProvenBitwiseKind> logicSlotKinds;
  bool doublesPower = false;
  bool decrementsLen = false;
  bool updatesResult = false;
  ProvenBitwiseKind condKind = ProvenBitwiseKind::None;

  for (auto &owned : body.ops()) {
    auto *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (isStoreToSlot(op, powerSlot) &&
        isBinaryWithConst(op->operand(0), "arith.muli", "rv_machine.mulw", "arm_machine.mul",
                          powerSlot, 2, true))
      doublesPower = true;
    if (isStoreToSlot(op, lenSlot) && isSubSlotByOne(op->operand(0), lenSlot))
      decrementsLen = true;
    if (opHasName(op, {"sysy.store"}) && op->operandCount() >= 2) {
      for (const auto &paramKey : paramSlots) {
        Value paramSlot;
        for (auto &entryOp : entry.ops()) {
          if (entryOp && entryOp->resultCount() && valueKey(entryOp->result()) == paramKey)
            paramSlot = entryOp->result();
        }
        if (!paramSlot.valid())
          continue;
        if (isBinaryWithConst(op->operand(0), "arith.remi", "rv_machine.remw",
                              "arm_machine.srem", paramSlot, 2)) {
          bitSlots.insert(valueKey(op->operand(1)));
          bitRefs.insert(valueKey(op->operand(1)));
          bitRefs.insert(valueKey(op->operand(0)));
          paramsWithRem.insert(paramKey);
        }
        if (isStoreToSlot(op, paramSlot) &&
            isBinaryWithConst(op->operand(0), "arith.divi", "rv_machine.divw",
                              "arm_machine.sdiv", paramSlot, 2))
          paramsWithDiv.insert(paramKey);
      }
    }
    Value logicSlot;
    ProvenBitwiseKind logicKind = ProvenBitwiseKind::None;
    if (isShortCircuitLogicIf(op, bitRefs, logicSlot, logicKind))
      logicSlotKinds[valueKey(logicSlot)] = logicKind;

    if (op->name() == "scf.if" && op->operandCount() == 1 &&
        op->getRegions().size() == 1 && !op->getRegions()[0]->getBlocks().empty()) {
      auto *condOp = op->operand(0).getDefiningOp();
      if (opHasName(condOp, {"arith.andi", "rv_machine.and", "arm_machine.and"}) &&
          condOp->operandCount() == 2 &&
          isEqBitToOne(condOp->operand(0), bitRefs) &&
          isEqBitToOne(condOp->operand(1), bitRefs)) {
        condKind = ProvenBitwiseKind::And;
      } else if (isOrOfEqBitToOne(op->operand(0), bitRefs)) {
        condKind = ProvenBitwiseKind::Or;
      } else if (isNeBetweenBitLoads(op->operand(0), bitRefs)) {
        condKind = ProvenBitwiseKind::Xor;
      } else {
        auto *load = op->operand(0).getDefiningOp();
        if (opHasName(load, {"sysy.load", "memref.load"}) && load->operandCount() > 0) {
          auto logicIt = logicSlotKinds.find(valueKey(load->operand(0)));
          if (logicIt != logicSlotKinds.end())
            condKind = logicIt->second;
        }
      }
      Block &thenBlock = *op->getRegions()[0]->getBlocks()[0];
      for (auto &thenOp : thenBlock.ops())
        if (thenOp && isResultPlusPowerStore(thenOp.get(), resultSlot, powerSlot))
          updatesResult = true;
    } else if (op->name() == "scf.if" &&
               isShortCircuitOrIf(op, bitRefs, resultSlot, powerSlot)) {
      condKind = ProvenBitwiseKind::Or;
      updatesResult = true;
    }
  }

  if (paramsWithRem.size() != 2 || paramsWithDiv.size() != 2 ||
      bitSlots.size() != 2 || !doublesPower || !decrementsLen ||
      !updatesResult || condKind == ProvenBitwiseKind::None)
    return result;

  result.kind = condKind;
  result.func = &func;
  if (stats)
    stats->bitwiseCandidates++;
  return result;
}

static Operation *insertOp(Block &block, std::size_t &index,
                           std::unique_ptr<Operation> op) {
  Operation &inserted = block.insertOperation(index, std::move(op));
  index++;
  return &inserted;
}

static Operation *insertConstant(Module &module, Block &block, std::size_t &index,
                                 int64_t value, Type type, Location loc) {
  return insertOp(block, index, std::make_unique<Operation>(
      "arith.constant", std::vector<Value>{}, std::vector<Type>{type},
      std::map<std::string, Attribute>{{"value", module.context().integerAttr(value, type)}},
      loc));
}

static Operation *insertBinary(Block &block, std::size_t &index, const std::string &name,
                               Value lhs, Value rhs, Type type, Location loc) {
  return insertOp(block, index, std::make_unique<Operation>(
      name, std::vector<Value>{lhs, rhs}, std::vector<Type>{type},
      std::map<std::string, Attribute>{}, loc));
}

static Operation *insertCmp(Module &module, Block &block, std::size_t &index,
                            Value lhs, Value rhs, const std::string &pred,
                            Location loc) {
  return insertOp(block, index, std::make_unique<Operation>(
      "arith.cmpi", std::vector<Value>{lhs, rhs}, std::vector<Type>{module.context().i(32)},
      std::map<std::string, Attribute>{{"predicate", module.context().stringAttr(pred)}},
      loc));
}

static std::string bitwiseOpName(ProvenBitwiseKind kind) {
  switch (kind) {
  case ProvenBitwiseKind::And:
    return "arith.andi";
  case ProvenBitwiseKind::Or:
    return "arith.ori";
  case ProvenBitwiseKind::Xor:
    return "arith.xori";
  default:
    return "";
  }
}

static void lowerCallWithGuard(Module &module, Operation &call,
                               ProvenBitwiseKind kind, SelfOptStats *stats) {
  Block *block = call.getBlock();
  if (!block || call.operandCount() != 2 || call.resultCount() != 1)
    return;
  int callIndex = operationIndexInBlock(*block, &call);
  if (callIndex < 0)
    return;
  std::size_t index = (std::size_t) callIndex;
  Type type = call.resultType();
  Location loc = call.loc();
  std::string directName = bitwiseOpName(kind);
  if (directName.empty())
    return;

  bool lhsNonNegative = valueStaticallyNonNegative(call.operand(0), stats);
  bool rhsNonNegative = valueStaticallyNonNegative(call.operand(1), stats);
  bool directSafe = lhsNonNegative && rhsNonNegative;
  if (directSafe) {
    Operation *direct = insertBinary(*block, index, directName, call.operand(0), call.operand(1), type, loc);
    replaceAllUses(module, call.result(), direct->result());
    call.markErased();
    if (stats)
      stats->bitwiseRewrittenCalls++;
    return;
  }

  Operation *slot = insertOp(*block, index, std::make_unique<Operation>(
      "sysy.alloca", std::vector<Value>{},
      std::vector<Type>{module.context().memref({1}, type)},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  Operation *zeroA = insertConstant(module, *block, index, 0, type, loc);
  Operation *lhsOk = insertCmp(module, *block, index, zeroA->result(), call.operand(0), "le", loc);
  Operation *zeroB = insertConstant(module, *block, index, 0, type, loc);
  Operation *rhsOk = insertCmp(module, *block, index, zeroB->result(), call.operand(1), "le", loc);
  Operation *guard = insertBinary(*block, index, "arith.andi", lhsOk->result(), rhsOk->result(),
                                  module.context().i(32), loc);

  auto ifOp = std::make_unique<Operation>(
      "scf.if", std::vector<Value>{guard->result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc);
  ifOp->addRegion();
  ifOp->addRegion();
  Operation *ifPtr = insertOp(*block, index, std::move(ifOp));
  Block &thenBlock = ifPtr->getRegions()[0]->addBlock();
  Operation &direct = thenBlock.addOperation(std::make_unique<Operation>(
      directName, std::vector<Value>{call.operand(0), call.operand(1)}, std::vector<Type>{type},
      std::map<std::string, Attribute>{}, loc));
  thenBlock.addOperation(std::make_unique<Operation>(
      "sysy.store", std::vector<Value>{direct.result(), slot->result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  thenBlock.addOperation(std::make_unique<Operation>(
      "scf.yield", std::vector<Value>{}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc));

  Block &elseBlock = ifPtr->getRegions()[1]->addBlock();
  Operation &fallback = elseBlock.addOperation(std::make_unique<Operation>(
      "sysy.call", std::vector<Value>{call.operand(0), call.operand(1)}, std::vector<Type>{type},
      call.attrs(), loc));
  elseBlock.addOperation(std::make_unique<Operation>(
      "sysy.store", std::vector<Value>{fallback.result(), slot->result()}, std::vector<Type>{},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  elseBlock.addOperation(std::make_unique<Operation>(
      "scf.yield", std::vector<Value>{}, std::vector<Type>{},
      std::map<std::string, Attribute>{}, loc));

  Operation *load = insertOp(*block, index, std::make_unique<Operation>(
      "sysy.load", std::vector<Value>{slot->result()}, std::vector<Type>{type},
      std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(".proven_bitwise")}},
      loc));
  replaceAllUses(module, call.result(), load->result());
  call.markErased();
  if (stats)
    stats->bitwiseGuardedCalls++;
}

enum class RotateHelperKind {
  None,
  Left,
  Right,
};

struct RotateHelperFunction {
  RotateHelperKind kind = RotateHelperKind::None;
  std::map<int64_t, int64_t> factors;
};

static int64_t denseRotateMaxShift(const RotateHelperFunction &info) {
  if (info.kind == RotateHelperKind::None || info.factors.empty())
    return 0;
  int64_t maxShift = info.factors.rbegin()->first;
  if (maxShift <= 0 || maxShift > 30)
    return 0;
  for (int64_t shift = 1; shift <= maxShift; shift++) {
    auto it = info.factors.find(shift);
    if (it == info.factors.end() || it->second != (int64_t(1) << shift))
      return 0;
  }
  return maxShift;
}

static bool isValueOrLoadFromSlot(Value value, Value slot, Value arg) {
  if (value == arg)
    return true;
  return slot.valid() && isLoadFromSlot(value, slot);
}

static bool isEqToConst(Value value, Value slot, Value arg, int64_t &constant) {
  auto *cmp = value.getDefiningOp();
  if (!opHasName(cmp, {"arith.cmpi", "rv_machine.cmp", "arm_machine.cmp"}) ||
      cmp->operandCount() != 2 || symbolAttr(cmp->attr("predicate")) != "eq")
    return false;
  auto matches = [&](Value maybeValue, Value maybeConst) {
    return isValueOrLoadFromSlot(maybeValue, slot, arg) &&
           constantIntegerValue(maybeConst, constant);
  };
  return matches(cmp->operand(0), cmp->operand(1)) ||
         matches(cmp->operand(1), cmp->operand(0));
}

static Operation *singleReturnInRegion(Operation *op) {
  if (!op || op->getRegions().empty() || op->getRegions()[0]->getBlocks().empty())
    return nullptr;
  Operation *ret = nullptr;
  for (auto &owned : op->getRegions()[0]->getBlocks()[0]->ops()) {
    Operation *child = owned.get();
    if (!child || child->isErased() || child->name() == "scf.yield")
      continue;
    if (opHasName(child, {"sysy.return", "scf.return"})) {
      if (ret)
        return nullptr;
      ret = child;
      continue;
    }
    if (ret)
      return nullptr;
    if (!opHasName(child, {"sysy.load", "memref.load", "arith.constant",
                           "arith.addi", "arith.subi", "arith.muli",
                           "arith.divi", "arith.remi", "arith.cmpi",
                           "rv_machine.li", "rv_machine.addw",
                           "rv_machine.subw", "rv_machine.mulw",
                           "rv_machine.divw", "rv_machine.remw",
                           "rv_machine.cmp", "arm_machine.mov",
                           "arm_machine.add", "arm_machine.sub",
                           "arm_machine.mul", "arm_machine.sdiv",
                           "arm_machine.srem", "arm_machine.cmp"}))
      return nullptr;
  }
  return ret;
}

static RotateHelperFunction classifyRotateHelperFunction(Operation &func) {
  RotateHelperFunction info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  if (entry.args().size() != 2)
    return info;
  Value xArg = entry.args()[0]->value();
  Value nArg = entry.args()[1]->value();
  Value xSlot;
  Value nSlot;
  for (auto &owned : entry.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || !isStoreToSlot(op, op->operandCount() >= 2 ? op->operand(1) : Value()))
      continue;
    if (op->operandCount() >= 2 && op->operand(0) == xArg)
      xSlot = op->operand(1);
    if (op->operandCount() >= 2 && op->operand(0) == nArg)
      nSlot = op->operand(1);
  }

  bool sawCase = false;
  bool finalReturnsX = false;
  RotateHelperKind kind = RotateHelperKind::None;
  std::map<int64_t, int64_t> factors;
  for (auto &owned : entry.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.if") {
      if (op->operandCount() != 1)
        return {};
      int64_t shift = 0;
      if (!isEqToConst(op->operand(0), nSlot, nArg, shift) || shift < 1 || shift > 30)
        return {};
      Operation *ret = singleReturnInRegion(op);
      if (!ret || ret->operandCount() != 1)
        return {};
      Operation *arith = ret->operand(0).getDefiningOp();
      if (!arith || arith->operandCount() != 2)
        return {};
      int64_t factor = 0;
      bool xFirst = isValueOrLoadFromSlot(arith->operand(0), xSlot, xArg) &&
                    constantIntegerValue(arith->operand(1), factor);
      bool xSecond = isValueOrLoadFromSlot(arith->operand(1), xSlot, xArg) &&
                     constantIntegerValue(arith->operand(0), factor);
      RotateHelperKind thisKind = RotateHelperKind::None;
      if (opHasName(arith, {"arith.muli", "rv_machine.mulw", "arm_machine.mul"}) &&
          (xFirst || xSecond)) {
        thisKind = RotateHelperKind::Left;
      } else if (opHasName(arith, {"arith.divi", "rv_machine.divw", "arm_machine.sdiv"}) &&
                 xFirst) {
        thisKind = RotateHelperKind::Right;
      } else {
        return {};
      }
      if (factor != (int64_t(1) << shift))
        return {};
      if (kind == RotateHelperKind::None)
        kind = thisKind;
      else if (kind != thisKind)
        return {};
      factors[shift] = factor;
      sawCase = true;
      continue;
    }
    if (opHasName(op, {"sysy.return", "scf.return"})) {
      if (op->operandCount() != 1 || !isValueOrLoadFromSlot(op->operand(0), xSlot, xArg))
        return {};
      finalReturnsX = true;
    }
  }

  if (sawCase && finalReturnsX && kind != RotateHelperKind::None) {
    info.kind = kind;
    info.factors = std::move(factors);
  }
  return info;
}

} // namespace

void runProvenBitwiseHelper(Module &module, SelfOptStats *stats) {
  const char *enabled = std::getenv("SISY_ENABLE_SELF_PROVEN_BITWISE");
  if (enabled && std::string(enabled) == "0")
    return;

  std::map<std::string, ProvenBitwiseFunction> classified;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.func")
      continue;
    auto info = classifyProvenBitwiseFunction(*op, stats);
    if (info.kind == ProvenBitwiseKind::None)
      continue;
    std::string symbol = symbolAttr(op->attr("sym_name"));
    if (!symbol.empty())
      classified[symbol] = info;
  }
  if (classified.empty())
    return;

  std::vector<Operation*> calls;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.call" ||
        op->operandCount() != 2 || op->resultCount() != 1)
      continue;
    std::string callee = symbolAttr(op->attr("callee"));
    if (classified.count(callee))
      calls.push_back(op);
  }

  for (auto *call : calls) {
    if (!call || call->isErased())
      continue;
    std::string callee = symbolAttr(call->attr("callee"));
    auto it = classified.find(callee);
    if (it == classified.end())
      continue;
    lowerCallWithGuard(module, *call, it->second.kind, stats);
  }
  eraseMarked(module);
}

void runRotateHelperFold(Module &module, SelfOptStats *stats) {
  if (!envEnabled("SISY_ENABLE_SELF_ROT_HELPER", true))
    return;
  std::map<std::string, RotateHelperFunction> helpers;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.func")
      continue;
    auto info = classifyRotateHelperFunction(*op);
    if (info.kind == RotateHelperKind::None)
      continue;
    std::string symbol = symbolAttr(op->attr("sym_name"));
    if (!symbol.empty())
      helpers[symbol] = std::move(info);
  }
  if (helpers.empty())
    return;

  std::vector<Operation*> calls;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.call" ||
        op->operandCount() != 2 || op->resultCount() != 1)
      continue;
    std::string callee = symbolAttr(op->attr("callee"));
    if (helpers.count(callee) != 0)
      calls.push_back(op);
  }

  for (Operation *call : calls) {
    if (!call || call->isErased())
      continue;
    auto helperIt = helpers.find(symbolAttr(call->attr("callee")));
    if (helperIt == helpers.end())
      continue;
    int64_t shift = 0;
    if (!constantIntegerValue(call->operand(1), shift))
      shift = std::numeric_limits<int64_t>::min();
    Block *block = call->getBlock();
    if (!block)
      continue;
    int indexRaw = operationIndexInBlock(*block, call);
    if (indexRaw < 0)
      continue;
    if (shift == std::numeric_limits<int64_t>::min()) {
      int64_t maxShift = denseRotateMaxShift(helperIt->second);
      if (maxShift <= 0)
        continue;
      auto replacement = std::make_unique<Operation>(
          "sysy.rotate_helper",
          std::vector<Value>{call->operand(0), call->operand(1)},
          std::vector<Type>{call->resultType()},
          std::map<std::string, Attribute>{
              {"direction", module.context().stringAttr(
                                helperIt->second.kind == RotateHelperKind::Left ? "left"
                                                                                 : "right")},
              {"max_shift", module.context().integerAttr(maxShift,
                                                          module.context().i(32))}},
          call->loc());
      replaceOperation(module, *call, std::move(replacement));
      if (stats)
        stats->rotHelperFolds++;
      continue;
    }
    if (shift <= 0 || helperIt->second.factors.count(shift) == 0) {
      replaceAllUses(module, call->result(), call->operand(0));
      call->markErased();
      if (stats)
        stats->rotHelperFolds++;
      continue;
    }
    std::size_t index = (std::size_t) indexRaw;
    int64_t factor = helperIt->second.factors[shift];
    Operation *factorOp = insertConstant(module, *block, index, factor, call->resultType(), call->loc());
    const std::string opName = helperIt->second.kind == RotateHelperKind::Left
                                   ? "arith.muli"
                                   : "arith.divi";
    Operation *folded = insertBinary(*block, index, opName, call->operand(0),
                                     factorOp->result(), call->resultType(), call->loc());
    replaceAllUses(module, call->result(), folded->result());
    call->markErased();
    if (stats)
      stats->rotHelperFolds++;
  }
  eraseMarked(module);
}

void collectAffineNestSummary(Module &module, SelfOptStats *stats) {
  if (!stats)
    return;
  stats->affineSummaryLoops = 0;
  stats->affineSummaryMemoryOps = 0;
  stats->affineSummarySideEffects = 0;
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "affine.for" || op->name() == "scf.for" || op->name() == "scf.while")
      stats->affineSummaryLoops++;
    if (op->name() == "memref.load" || op->name() == "memref.store" ||
        op->name() == "sysy.load" || op->name() == "sysy.store")
      stats->affineSummaryMemoryOps++;
    if (op->name() == "sysy.call")
      stats->affineSummarySideEffects++;
    if (op->name() == "affine.for" && op->operandCount() >= 3) {
      int64_t step = 1;
      if (constantIntegerValue(op->operand(2), step) && step != 1)
        stats->loopTiles++;
    }
  }
}

void runLoopVectorization(Module &module) {
  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.while" || op->name() == "scf.for" || op->name() == "affine.for") {
      Block *body = nullptr;
      if (!op->getRegions().empty() && !op->getRegions()[0]->getBlocks().empty()) {
        body = op->getRegions()[0]->getBlocks()[0].get();
      }
      if (!body || body->ops().empty())
        continue;

      std::vector<Operation*> loads;
      std::vector<Operation*> stores;
      std::vector<Operation*> ariths;

      for (auto &owned : body->ops()) {
        auto *child = owned.get();
        if (!child || child->isErased())
          continue;
        if (child->name() == "memref.store") {
          stores.push_back(child);
        }
      }

      if (stores.empty())
        continue;

      std::set<std::string> activeValNames;
      for (auto *storeOp : stores) {
        if (storeOp->operandCount() >= 1) {
          activeValNames.insert(valueKey(storeOp->operand(0)));
        }
      }

      for (int i = (int) body->ops().size() - 1; i >= 0; i--) {
        auto *child = body->ops()[i].get();
        if (!child || child->isErased())
          continue;
        if (child->resultCount() > 0) {
          std::string resName = valueKey(child->result());
          if (activeValNames.count(resName) > 0) {
            if (child->name() == "memref.load") {
              loads.push_back(child);
            } else if (child->name() == "arith.addi" || child->name() == "arith.addf" ||
                       child->name() == "arith.subi" || child->name() == "arith.subf" ||
                       child->name() == "rv_machine.addw" || child->name() == "arm_machine.add" ||
                       child->name() == "arith.muli" || child->name() == "arith.mulf") {
              ariths.push_back(child);
              for (int opIdx = 0; opIdx < child->operandCount(); opIdx++) {
                activeValNames.insert(valueKey(child->operand(opIdx)));
              }
            }
          }
        }
      }

      std::reverse(loads.begin(), loads.end());
      std::reverse(ariths.begin(), ariths.end());

      if (loads.empty())
        continue;

      bool ok = true;
      for (auto *l : loads) {
        if (l->operandCount() < 2) { ok = false; break; }
      }
      for (auto *s : stores) {
        if (s->operandCount() < 3) { ok = false; break; }
      }
      if (!ok)
        continue;

      bool hasRAW = false;
      for (auto *storeOp : stores) {
        std::string storeBase = symbolAttr(storeOp->attr("symbol"));
        if (storeBase.empty())
          storeBase = symbolAttr(storeOp->attr("sym_name"));
        for (auto *loadOp : loads) {
          std::string loadBase = symbolAttr(loadOp->attr("symbol"));
          if (loadBase.empty())
            loadBase = symbolAttr(loadOp->attr("sym_name"));
          if (storeBase == loadBase) {
            if (storeOp->operandCount() >= 3 && loadOp->operandCount() >= 2) {
              if (valueKey(storeOp->operand(2)) != valueKey(loadOp->operand(1))) {
                hasRAW = true;
                break;
              }
            } else {
              hasRAW = true;
              break;
            }
          }
        }
        if (hasRAW)
          break;
      }
      if (hasRAW)
        continue;

      std::size_t insertIdx = body->ops().size() - 1;

      int64_t lanes = 4;
      Type elemType = loads[0]->resultType();
      Type vecType = module.context().vector(elemType, lanes);

      std::map<std::string, Value> vecLoads;
      for (auto *loadOp : loads) {
        Value base = loadOp->operand(0);
        Value index = loadOp->operand(1);
        std::string loadSym = symbolAttr(loadOp->attr("symbol"));
        if (loadSym.empty())
          loadSym = symbolAttr(loadOp->attr("sym_name"));
        auto &vread = body->insertOperation(insertIdx++, std::make_unique<Operation>(
            "vector.transfer_read", std::vector<Value>{base, index}, std::vector<Type>{vecType},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(loadSym)}},
            loadOp->loc()));
        vecLoads[valueKey(loadOp->result())] = vread.result();
      }

      std::map<std::string, Value> vecAriths;
      for (auto *arithOp : ariths) {
        std::vector<Value> vecOperands;
        for (int i = 0; i < arithOp->operandCount(); i++) {
          Value operand = arithOp->operand(i);
          std::string operandKey = valueKey(operand);
          if (vecLoads.count(operandKey)) {
            vecOperands.push_back(vecLoads[operandKey]);
          } else if (vecAriths.count(operandKey)) {
            vecOperands.push_back(vecAriths[operandKey]);
          } else {
            auto &splat = body->insertOperation(insertIdx++, std::make_unique<Operation>(
                "vector.splat", std::vector<Value>{operand}, std::vector<Type>{vecType},
                std::map<std::string, Attribute>{}, arithOp->loc()));
            vecOperands.push_back(splat.result());
          }
        }
        auto &varith = body->insertOperation(insertIdx++, std::make_unique<Operation>(
            arithOp->name(), vecOperands, std::vector<Type>{vecType},
            arithOp->attrs(), arithOp->loc()));
        vecAriths[valueKey(arithOp->result())] = varith.result();
      }

      for (auto *storeOp : stores) {
        Value val = storeOp->operand(0);
        Value base = storeOp->operand(1);
        Value index = storeOp->operand(2);
        std::string storeSym = symbolAttr(storeOp->attr("symbol"));
        if (storeSym.empty())
          storeSym = symbolAttr(storeOp->attr("sym_name"));

        Value vecVal;
        std::string valKey = valueKey(val);
        if (vecLoads.count(valKey)) {
          vecVal = vecLoads[valKey];
        } else if (vecAriths.count(valKey)) {
          vecVal = vecAriths[valKey];
        } else {
          auto &splat = body->insertOperation(insertIdx++, std::make_unique<Operation>(
              "vector.splat", std::vector<Value>{val}, std::vector<Type>{vecType},
              std::map<std::string, Attribute>{}, storeOp->loc()));
          vecVal = splat.result();
        }

        body->insertOperation(insertIdx++, std::make_unique<Operation>(
            "vector.transfer_write", std::vector<Value>{vecVal, base, index}, std::vector<Type>{},
            std::map<std::string, Attribute>{{"symbol", module.context().stringAttr(storeSym)}},
            storeOp->loc()));
      }

      for (auto &owned : body->ops()) {
        auto *child = owned.get();
        if (!child || child->isErased())
          continue;
        if ((child->name() == "rv_machine.li" || child->name() == "arith.constant" || child->name() == "arm_machine.mov") &&
            parseIntegerAttr(child->attr("value")) == 1) {
          child->setAttr("value", module.context().integerAttr(lanes, module.context().i(32)));
        }
      }

      for (auto *opToErase : loads) {
        for (int i = 0; i < opToErase->operandCount(); i++) {
          opToErase->setOperand(i, Value());
        }
        opToErase->markErased();
      }
      for (auto *opToErase : ariths) {
        for (int i = 0; i < opToErase->operandCount(); i++) {
          opToErase->setOperand(i, Value());
        }
        opToErase->markErased();
      }
      for (auto *opToErase : stores) {
        for (int i = 0; i < opToErase->operandCount(); i++) {
          opToErase->setOperand(i, Value());
        }
        opToErase->markErased();
      }
    }
  }
  eraseMarked(module);
}

} // namespace sys::mlir
