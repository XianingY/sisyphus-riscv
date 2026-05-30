#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

using namespace sys;

namespace {

enum class BitKind {
  And,
  Or,
  Xor,
};

struct BitCandidate {
  FuncOp *func = nullptr;
  BitKind kind = BitKind::And;
};

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

bool same(Op *a, Op *b) {
  return a && b && a == b;
}

bool isInt(Op *op, int value) {
  return op && isa<IntOp>(op) && V(op) == value;
}

std::string suffix(BitKind kind) {
  if (kind == BitKind::And)
    return "and";
  if (kind == BitKind::Or)
    return "or";
  return "xor";
}

std::string helperName(const std::string &name, BitKind kind) {
  return "__sisy_struct_bitwise_" + suffix(kind) + "_" + name;
}

bool hasFunctionNamed(ModuleOp *module, const std::string &name) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == name)
      return true;
  return false;
}

bool hasI32Args(FuncOp *func, int argc) {
  if (!func->has<ArgCountAttr>() || func->get<ArgCountAttr>()->count != argc)
    return false;
  if (auto types = func->find<ArgTypesAttr>()) {
    if ((int) types->types.size() != argc)
      return false;
    for (auto ty : types->types)
      if (ty != Value::i32)
        return false;
  }
  return true;
}

std::vector<Op*> allOps(FuncOp *func) {
  std::vector<Op*> ops;
  for (auto region : func->getRegions())
    for (auto block : region->getBlocks())
      for (auto op : block->getOps())
        ops.push_back(op);
  return ops;
}

bool isLoadFrom(Op *op, Op *slot) {
  auto load = dyn_cast<LoadOp>(op);
  return load && load->getOperandCount() == 1 && same(load->DEF(0), slot);
}

bool isStoreTo(StoreOp *store, Op *slot) {
  return store && store->getOperandCount() == 2 && same(store->DEF(1), slot);
}

bool storeToSlot(FuncOp *func, Op *slot, bool (*pred)(Op*)) {
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, slot) && pred(store->DEF(0)))
      return true;
  }
  return false;
}

bool valueIsZero(Op *op) {
  return isInt(op, 0);
}

bool valueIsOne(Op *op) {
  return isInt(op, 1);
}

bool valueIsThirtyTwo(Op *op) {
  return isInt(op, 32);
}

bool findInit(FuncOp *func, Op *slot, bool (*pred)(Op*)) {
  return storeToSlot(func, slot, pred);
}

bool isModuloByTwoOfSlot(Op *op, Op *slot) {
  auto mod = dyn_cast<ModIOp>(op);
  return mod && mod->getOperandCount() == 2 && isLoadFrom(mod->DEF(0), slot) &&
         isInt(mod->DEF(1), 2);
}

bool isDivByTwoOfSlot(Op *op, Op *slot) {
  auto div = dyn_cast<DivIOp>(op);
  return div && div->getOperandCount() == 2 && isLoadFrom(div->DEF(0), slot) &&
         isInt(div->DEF(1), 2);
}

bool isSubOneOfSlot(Op *op, Op *slot) {
  auto sub = dyn_cast<SubIOp>(op);
  return sub && sub->getOperandCount() == 2 && isLoadFrom(sub->DEF(0), slot) &&
         isInt(sub->DEF(1), 1);
}

bool isMulTwoOfSlot(Op *op, Op *slot) {
  auto mul = dyn_cast<MulIOp>(op);
  if (!mul || mul->getOperandCount() != 2)
    return false;
  return (isLoadFrom(mul->DEF(0), slot) && isInt(mul->DEF(1), 2)) ||
         (isLoadFrom(mul->DEF(1), slot) && isInt(mul->DEF(0), 2));
}

bool isAddSlots(Op *op, Op *lhsSlot, Op *rhsSlot) {
  auto add = dyn_cast<AddIOp>(op);
  if (!add || add->getOperandCount() != 2)
    return false;
  return (isLoadFrom(add->DEF(0), lhsSlot) && isLoadFrom(add->DEF(1), rhsSlot)) ||
         (isLoadFrom(add->DEF(1), lhsSlot) && isLoadFrom(add->DEF(0), rhsSlot));
}

bool hasDivUpdate(FuncOp *func, Op *slot) {
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, slot) && isDivByTwoOfSlot(store->DEF(0), slot))
      return true;
  }
  return false;
}

bool hasCounterUpdate(FuncOp *func, Op *slot) {
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, slot) && isSubOneOfSlot(store->DEF(0), slot))
      return true;
  }
  return false;
}

bool hasPowerUpdate(FuncOp *func, Op *slot) {
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, slot) && isMulTwoOfSlot(store->DEF(0), slot))
      return true;
  }
  return false;
}

bool hasAccUpdate(FuncOp *func, Op *accSlot, Op *powerSlot) {
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, accSlot) && isAddSlots(store->DEF(0), accSlot, powerSlot))
      return true;
  }
  return false;
}

std::set<BasicBlock*> accUpdateBlocks(FuncOp *func, Op *accSlot, Op *powerSlot) {
  std::set<BasicBlock*> blocks;
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, accSlot) && isAddSlots(store->DEF(0), accSlot, powerSlot))
      blocks.insert(store->getParent());
  }
  return blocks;
}

Op *findBitSlot(FuncOp *func, Op *argSlot, const std::set<Op*> &slots) {
  Op *result = nullptr;
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (store->getOperandCount() != 2 || !slots.count(store->DEF(1)))
      continue;
    if (!isModuloByTwoOfSlot(store->DEF(0), argSlot))
      continue;
    if (result && result != store->DEF(1))
      return nullptr;
    result = store->DEF(1);
  }
  return result;
}

bool opIsEqSlotOne(Op *op, Op *slot) {
  auto eq = dyn_cast<EqOp>(op);
  if (!eq || eq->getOperandCount() != 2)
    return false;
  return (isLoadFrom(eq->DEF(0), slot) && isInt(eq->DEF(1), 1)) ||
         (isLoadFrom(eq->DEF(1), slot) && isInt(eq->DEF(0), 1));
}

bool opIsNeSlots(Op *op, Op *lhsSlot, Op *rhsSlot) {
  auto ne = dyn_cast<NeOp>(op);
  if (!ne || ne->getOperandCount() != 2)
    return false;
  return (isLoadFrom(ne->DEF(0), lhsSlot) && isLoadFrom(ne->DEF(1), rhsSlot)) ||
         (isLoadFrom(ne->DEF(1), lhsSlot) && isLoadFrom(ne->DEF(0), rhsSlot));
}

bool usedByBranch(FuncOp *func, Op *cond) {
  for (auto op : func->findAll<BranchOp>()) {
    auto branch = cast<BranchOp>(op);
    if (branch->getOperandCount() == 1 && same(branch->DEF(0), cond))
      return true;
  }
  return false;
}

bool hasEqBranchForSlot(FuncOp *func, Op *slot) {
  for (auto op : allOps(func))
    if (opIsEqSlotOne(op, slot) && usedByBranch(func, op))
      return true;
  return false;
}

struct EqBranch {
  BasicBlock *block = nullptr;
  Op *slot = nullptr;
  BasicBlock *target = nullptr;
  BasicBlock *otherwise = nullptr;
};

std::vector<EqBranch> collectEqBranches(FuncOp *func, Op *bit0Slot, Op *bit1Slot) {
  std::vector<EqBranch> result;
  for (auto op : func->findAll<BranchOp>()) {
    auto branch = cast<BranchOp>(op);
    if (branch->getOperandCount() != 1)
      continue;
    auto cond = branch->DEF(0);
    Op *slot = nullptr;
    if (opIsEqSlotOne(cond, bit0Slot))
      slot = bit0Slot;
    else if (opIsEqSlotOne(cond, bit1Slot))
      slot = bit1Slot;
    else
      continue;
    result.push_back(EqBranch{ branch->getParent(), slot, TARGET(branch), ELSE(branch) });
  }
  return result;
}

bool classifyAndOr(FuncOp *func, Op *bit0Slot, Op *bit1Slot, Op *accSlot,
                   Op *powerSlot, BitKind &kind) {
  auto updates = accUpdateBlocks(func, accSlot, powerSlot);
  if (updates.size() != 1)
    return false;
  auto update = *updates.begin();
  auto eqs = collectEqBranches(func, bit0Slot, bit1Slot);
  if (eqs.size() != 2)
    return false;

  auto targetsUpdate = [&](Op *slot) {
    for (auto &br : eqs)
      if (br.slot == slot && br.target == update)
        return true;
    return false;
  };

  // OR short-circuit shape:
  //   if (bit0 == 1) update;
  //   else if (bit1 == 1) update;
  if (targetsUpdate(bit0Slot) && targetsUpdate(bit1Slot)) {
    kind = BitKind::Or;
    return true;
  }

  if (update->preds.size() != 1)
    return false;
  auto second = *update->preds.begin();
  EqBranch *secondBranch = nullptr;
  EqBranch *firstBranch = nullptr;
  for (auto &br : eqs) {
    if (br.block == second && br.target == update)
      secondBranch = &br;
  }
  if (!secondBranch)
    return false;
  for (auto &br : eqs) {
    if (br.slot != secondBranch->slot && br.target == second)
      firstBranch = &br;
  }
  if (!firstBranch)
    return false;

  // AND short-circuit shape:
  //   if (bit0 == 1)
  //     if (bit1 == 1) update;
  kind = BitKind::And;
  return true;
}

bool hasNeBranchForSlots(FuncOp *func, Op *lhsSlot, Op *rhsSlot) {
  for (auto op : allOps(func))
    if (opIsNeSlots(op, lhsSlot, rhsSlot) && usedByBranch(func, op))
      return true;
  return false;
}

bool branchOnCounter(FuncOp *func, Op *counterSlot) {
  for (auto op : func->findAll<BranchOp>()) {
    auto branch = cast<BranchOp>(op);
    if (branch->getOperandCount() == 1 && isLoadFrom(branch->DEF(0), counterSlot))
      return true;
  }
  return false;
}

bool returnFromAcc(FuncOp *func, Op *accSlot, const std::set<Op*> &slots) {
  auto returns = func->findAll<ReturnOp>();
  if (returns.size() != 1)
    return false;

  auto ret = cast<ReturnOp>(returns[0]);
  if (ret->getOperandCount() != 1)
    return false;
  auto value = ret->DEF(0);
  if (isLoadFrom(value, accSlot))
    return true;

  auto load = dyn_cast<LoadOp>(value);
  if (!load || load->getOperandCount() != 1 || !slots.count(load->DEF(0)))
    return false;
  Op *retSlot = load->DEF(0);
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (isStoreTo(store, retSlot) && isLoadFrom(store->DEF(0), accSlot))
      return true;
  }
  return false;
}

bool hasOnlySupportedOps(FuncOp *func, const std::set<Op*> &slots) {
  int count = 0;
  for (auto op : allOps(func)) {
    count++;
    if (count > 140)
      return false;
    if (auto store = dyn_cast<StoreOp>(op)) {
      if (store->getOperandCount() != 2 || !slots.count(store->DEF(1)))
        return false;
      continue;
    }
    if (auto load = dyn_cast<LoadOp>(op)) {
      if (load->getOperandCount() != 1 || !slots.count(load->DEF(0)))
        return false;
      continue;
    }
    if (isa<AllocaOp>(op) || isa<GotoOp>(op) || isa<GetArgOp>(op) ||
        isa<IntOp>(op) || isa<BranchOp>(op) || isa<ModIOp>(op) ||
        isa<DivIOp>(op) || isa<EqOp>(op) || isa<NeOp>(op) ||
        isa<AddIOp>(op) || isa<MulIOp>(op) || isa<SubIOp>(op) ||
        isa<ReturnOp>(op))
      continue;
    return false;
  }
  return true;
}

bool allArithmeticFitsRoles(FuncOp *func, Op *arg0Slot, Op *arg1Slot,
                            Op *bit0Slot, Op *bit1Slot, Op *counterSlot,
                            Op *powerSlot, Op *accSlot, BitKind kind) {
  for (auto op : allOps(func)) {
    if (isa<ModIOp>(op)) {
      if (!isModuloByTwoOfSlot(op, arg0Slot) && !isModuloByTwoOfSlot(op, arg1Slot))
        return false;
    } else if (isa<DivIOp>(op)) {
      if (!isDivByTwoOfSlot(op, arg0Slot) && !isDivByTwoOfSlot(op, arg1Slot))
        return false;
    } else if (isa<MulIOp>(op)) {
      if (!isMulTwoOfSlot(op, powerSlot))
        return false;
    } else if (isa<SubIOp>(op)) {
      if (!isSubOneOfSlot(op, counterSlot))
        return false;
    } else if (isa<AddIOp>(op)) {
      if (!isAddSlots(op, accSlot, powerSlot))
        return false;
    } else if (isa<EqOp>(op)) {
      if (kind != BitKind::And && kind != BitKind::Or)
        return false;
      if (!opIsEqSlotOne(op, bit0Slot) && !opIsEqSlotOne(op, bit1Slot))
        return false;
    } else if (isa<NeOp>(op)) {
      if (kind != BitKind::Xor || !opIsNeSlots(op, bit0Slot, bit1Slot))
        return false;
    }
  }
  return true;
}

std::optional<BitCandidate> classify(FuncOp *func) {
  if (!hasI32Args(func, 2))
    return std::nullopt;
  if (!func->findAll<CallOp>().empty() || !func->findAll<GetGlobalOp>().empty())
    return std::nullopt;

  auto allocaOps = func->findAll<AllocaOp>();
  if (allocaOps.size() < 6 || allocaOps.size() > 10)
    return std::nullopt;
  std::set<Op*> slots(allocaOps.begin(), allocaOps.end());
  if (!hasOnlySupportedOps(func, slots))
    return std::nullopt;

  Op *argSlot[2] = { nullptr, nullptr };
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    auto arg = dyn_cast<GetArgOp>(store->DEF(0));
    if (!arg)
      continue;
    int idx = V(arg);
    if (idx < 0 || idx > 1 || !slots.count(store->DEF(1)))
      return std::nullopt;
    if (argSlot[idx] && argSlot[idx] != store->DEF(1))
      return std::nullopt;
    argSlot[idx] = store->DEF(1);
  }
  if (!argSlot[0] || !argSlot[1] || argSlot[0] == argSlot[1])
    return std::nullopt;
  if (!hasDivUpdate(func, argSlot[0]) || !hasDivUpdate(func, argSlot[1]))
    return std::nullopt;

  Op *bit0Slot = findBitSlot(func, argSlot[0], slots);
  Op *bit1Slot = findBitSlot(func, argSlot[1], slots);
  if (!bit0Slot || !bit1Slot || bit0Slot == bit1Slot)
    return std::nullopt;

  Op *counterSlot = nullptr;
  Op *powerSlot = nullptr;
  Op *accSlot = nullptr;
  for (auto slot : slots) {
    if (!counterSlot && findInit(func, slot, valueIsThirtyTwo) &&
        hasCounterUpdate(func, slot) && branchOnCounter(func, slot))
      counterSlot = slot;
    if (!powerSlot && findInit(func, slot, valueIsOne) && hasPowerUpdate(func, slot))
      powerSlot = slot;
  }
  if (!counterSlot || !powerSlot)
    return std::nullopt;

  for (auto slot : slots) {
    if (slot == counterSlot || slot == powerSlot || slot == argSlot[0] ||
        slot == argSlot[1] || slot == bit0Slot || slot == bit1Slot)
      continue;
    if (findInit(func, slot, valueIsZero) && hasAccUpdate(func, slot, powerSlot)) {
      accSlot = slot;
      break;
    }
  }
  if (!accSlot || !returnFromAcc(func, accSlot, slots))
    return std::nullopt;

  BitKind kind;
  if (hasNeBranchForSlots(func, bit0Slot, bit1Slot)) {
    kind = BitKind::Xor;
  } else if (hasEqBranchForSlot(func, bit0Slot) && hasEqBranchForSlot(func, bit1Slot)) {
    if (!classifyAndOr(func, bit0Slot, bit1Slot, accSlot, powerSlot, kind))
      return std::nullopt;
  } else {
    return std::nullopt;
  }

  if (!allArithmeticFitsRoles(func, argSlot[0], argSlot[1], bit0Slot, bit1Slot,
                              counterSlot, powerSlot, accSlot, kind))
    return std::nullopt;

  // The loop computes true bitwise operations only for non-negative operands.
  // Replacement sites that are not statically non-negative go through a helper
  // with a runtime non-negative guard and the original function as fallback.
  return BitCandidate{ func, kind };
}

bool provenNonNegative(Op *op) {
  if (!op)
    return false;
  if (isa<IntOp>(op))
    return V(op) >= 0;
  if (!op->has<RangeAttr>())
    return false;
  auto [low, high] = RANGE(op);
  (void) high;
  return low >= 0;
}

Op *buildBitOp(Builder &builder, BitKind kind, Op *lhs, Op *rhs) {
  if (kind == BitKind::And)
    return builder.create<AndIOp>(std::vector<Value>{ lhs, rhs });
  if (kind == BitKind::Or)
    return builder.create<OrIOp>(std::vector<Value>{ lhs, rhs });
  return builder.create<XorIOp>(std::vector<Value>{ lhs, rhs });
}

void ensureHelper(ModuleOp *module, const BitCandidate &candidate) {
  auto name = helperName(NAME(candidate.func), candidate.kind);
  if (hasFunctionNamed(module, name))
    return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(name),
    new ArgCountAttr(2),
    new ArgTypesAttr({ Value::i32, Value::i32 })
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto slow = region->appendBlock();
  auto fast = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto lhs = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  auto rhs = builder.create<GetArgOp>(Value::i32, { new IntAttr(1) });
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto lhsNeg = builder.create<LtOp>(std::vector<Value>{ lhs, zero });
  auto rhsNeg = builder.create<LtOp>(std::vector<Value>{ rhs, zero });
  auto anyNeg = builder.create<OrIOp>(std::vector<Value>{ lhsNeg, rhsNeg });
  builder.create<BranchOp>(std::vector<Value>{ anyNeg },
                           { new TargetAttr(slow), new ElseAttr(fast) });

  builder.setToBlockEnd(slow);
  auto fallback = builder.create<CallOp>(Value::i32, std::vector<Value>{ lhs, rhs },
                                         { new NameAttr(NAME(candidate.func)) });
  builder.create<ReturnOp>(std::vector<Value>{ fallback });

  builder.setToBlockEnd(fast);
  auto result = buildBitOp(builder, candidate.kind, lhs, rhs);
  builder.create<ReturnOp>(std::vector<Value>{ result });
}

bool runBitwiseLowering(ModuleOp *module, int &classified, int &guarded,
                        int &replaced) {
  CallGraph(module).run();

  std::map<std::string, BitCandidate> candidates;
  for (auto op : module->findAll<FuncOp>()) {
    auto func = cast<FuncOp>(op);
    if (isExtern(NAME(func)))
      continue;
    auto candidate = classify(func);
    if (!candidate)
      continue;
    candidates[NAME(func)] = *candidate;
    classified++;
  }
  if (candidates.empty())
    return false;

  bool changed = false;
  Builder builder;
  auto calls = module->findAll<CallOp>();
  for (auto call : calls) {
    auto it = candidates.find(NAME(call));
    if (it == candidates.end() || call->getResultType() != Value::i32 ||
        call->getOperandCount() != 2)
      continue;

    auto parent = call->getParentOp<FuncOp>();
    if (parent == it->second.func ||
        NAME(parent) == helperName(NAME(it->second.func), it->second.kind))
      continue;

    builder.setBeforeOp(call);
    if (provenNonNegative(call->DEF(0)) && provenNonNegative(call->DEF(1))) {
      auto replacement = buildBitOp(builder, it->second.kind, call->DEF(0), call->DEF(1));
      call->replaceAllUsesWith(replacement);
      call->erase();
      replaced++;
      changed = true;
      continue;
    }

    ensureHelper(module, it->second);
    NAME(call) = helperName(NAME(it->second.func), it->second.kind);
    guarded++;
    replaced++;
    changed = true;
  }
  return changed;
}

} // namespace

std::map<std::string, int> StructuralBitwise::stats() {
  return {
    { "classified", classified },
    { "guarded-calls", guarded },
    { "replaced-calls", replaced },
  };
}

void StructuralBitwise::run() {
  if (!envEnabled("SISY_ENABLE_STRUCTURAL_BITWISE", false))
    return;

  runBitwiseLowering(module, classified, guarded, replaced);
}

std::map<std::string, int> ProvenBitwiseHelper::stats() {
  return {
    { "classified", classified },
    { "guarded-calls", guarded },
    { "replaced-calls", replaced },
  };
}

void ProvenBitwiseHelper::run() {
  if (!envEnabled("SISY_ENABLE_PROVEN_BITWISE_HELPER", true))
    return;
  runBitwiseLowering(module, classified, guarded, replaced);
}
