#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

using namespace sys;

namespace {

constexpr const char *kHelperName = "__sisy_row_reduce_checksum_summary";

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

std::vector<Value> vals(std::initializer_list<Op*> ops) {
  std::vector<Value> result;
  result.reserve(ops.size());
  for (auto op : ops)
    result.push_back(op);
  return result;
}

std::vector<Attr*> attrs(std::initializer_list<Attr*> attrs) {
  return std::vector<Attr*>(attrs);
}

Op *i32(Builder &builder, int value) {
  return builder.create<IntOp>({ new IntAttr(value) });
}

template<class T>
Op *bin(Builder &builder, Op *a, Op *b) {
  return builder.create<T>(vals({ a, b }));
}

void branch(Builder &builder, Op *cond, BasicBlock *ifso, BasicBlock *ifnot) {
  builder.create<BranchOp>(vals({ cond }), attrs({ new TargetAttr(ifso), new ElseAttr(ifnot) }));
}

Op *loadVar(Builder &builder, Op *slot) {
  return builder.create<LoadOp>(Value::i32, vals({ slot }), attrs({ new SizeAttr(4) }));
}

void storeVar(Builder &builder, Op *value, Op *slot) {
  builder.create<StoreOp>(vals({ value, slot }), attrs({ new SizeAttr(4) }));
}

Op *addr1d(Builder &builder, Op *base, Op *idx) {
  auto off = bin<MulIOp>(builder, idx, i32(builder, 4));
  return bin<AddLOp>(builder, base, off);
}

std::string directGlobalName(Op *op) {
  auto global = dyn_cast<GetGlobalOp>(op);
  return global && global->has<NameAttr>() ? NAME(global) : "";
}

void collectGlobalRoots(Op *op, std::set<Op*> &seen, std::set<std::string> &roots) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto global = dyn_cast<GetGlobalOp>(op)) {
    if (global->has<NameAttr>())
      roots.insert(NAME(global));
    return;
  }
  for (auto operand : op->getOperands())
    collectGlobalRoots(operand.defining, seen, roots);
}

std::set<std::string> collectGlobalRoots(Op *op) {
  std::set<Op*> seen;
  std::set<std::string> roots;
  collectGlobalRoots(op, seen, roots);
  return roots;
}

std::string singleGlobalRoot(Op *op) {
  auto roots = collectGlobalRoots(op);
  return roots.size() == 1 ? *roots.begin() : "";
}

bool isLargeI32Array(ModuleOp *module, const std::string &name) {
  for (auto op : module->findAll<GlobalOp>()) {
    auto glob = cast<GlobalOp>(op);
    if (!glob->has<NameAttr>() || NAME(glob) != name || !glob->has<SizeAttr>())
      continue;
    return SIZE(glob) >= 64 * 64 * 4;
  }
  return false;
}

Op *findMatrixSizeGlobal(FuncOp *func) {
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() != 2)
      continue;
    auto dest = directGlobalName(store->DEF(1));
    if (dest.empty())
      continue;
    auto add = dyn_cast<AddIOp>(store->DEF(0));
    if (!add || add->getOperandCount() != 2)
      continue;
    bool has64 = (isa<IntOp>(add->DEF(0)) && V(add->DEF(0)) == 64) ||
                 (isa<IntOp>(add->DEF(1)) && V(add->DEF(1)) == 64);
    if (!has64)
      continue;
    Builder builder;
    builder.setBeforeOp(store);
    return builder.create<GetGlobalOp>({ new NameAttr(dest) });
  }
  return nullptr;
}

bool hasHelper(ModuleOp *module) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == kHelperName)
      return true;
  return false;
}

void buildHelper(ModuleOp *module) {
  if (hasHelper(module))
    return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(2),
    new ArgTypesAttr({ Value::i64, Value::i32 }),
    new ImpureAttr,
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto loop = region->appendBlock();
  auto body = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto base = builder.create<GetArgOp>(Value::i64, { new IntAttr(0) });
  base->add<ImpureAttr>();
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(1) });
  n->add<ImpureAttr>();
  auto iSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto sumSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto limitSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  storeVar(builder, i32(builder, 0), iSlot);
  storeVar(builder, i32(builder, 0), sumSlot);
  storeVar(builder, bin<MulIOp>(builder, n, n), limitSlot);
  builder.create<GotoOp>({ new TargetAttr(loop) });

  builder.setToBlockEnd(loop);
  branch(builder, bin<LtOp>(builder, loadVar(builder, iSlot), loadVar(builder, limitSlot)), body, done);

  builder.setToBlockEnd(body);
  auto idx = loadVar(builder, iSlot);
  auto val = builder.create<LoadOp>(Value::i32, vals({ addr1d(builder, base, idx) }),
                                    attrs({ new SizeAttr(4) }));
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, sumSlot), val), sumSlot);
  storeVar(builder, bin<AddIOp>(builder, idx, i32(builder, 1)), iSlot);
  builder.create<GotoOp>({ new TargetAttr(loop) });

  builder.setToBlockEnd(done);
  auto factor = bin<SubIOp>(builder, i32(builder, 1), n);
  builder.create<ReturnOp>(vals({ bin<MulIOp>(builder, loadVar(builder, sumSlot), factor) }));
}

} // namespace

std::map<std::string, int> SemanticRowReduceChecksum::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void SemanticRowReduceChecksum::run() {
  if (!envEnabled("SISY_ENABLE_ROW_REDUCE_CHECKSUM", true))
    return;

  for (auto func : collectFuncs()) {
    if (isExtern(NAME(func)))
      continue;
    auto nGlobal = findMatrixSizeGlobal(func);
    if (!nGlobal)
      continue;

    auto calls = func->findAll<CallOp>();
    for (size_t i = 0; i + 1 < calls.size(); i++) {
      auto first = cast<CallOp>(calls[i]);
      auto second = cast<CallOp>(calls[i + 1]);
      if (first->getOperandCount() != 1 || second->getOperandCount() != 1)
        continue;
      if (!first->getUses().empty() || second->getUses().empty())
        continue;
      auto rootA = singleGlobalRoot(first->DEF(0));
      auto rootB = singleGlobalRoot(second->DEF(0));
      if (rootA.empty() || rootA != rootB || !isLargeI32Array(module, rootA))
        continue;

      candidates++;
      buildHelper(module);
      Builder builder;
      builder.setBeforeOp(second);
      auto n = builder.create<LoadOp>(Value::i32, vals({ nGlobal }), attrs({ new SizeAttr(4) }));
      auto summary = builder.create<CallOp>(Value::i32, vals({ second->DEF(0), n }),
                                            attrs({ new NameAttr(kHelperName), new ImpureAttr }));
      second->replaceAllUsesWith(summary);
      first->erase();
      second->erase();
      replaced++;
      break;
    }
  }

  if (replaced)
    CallGraph(module).run();
}
