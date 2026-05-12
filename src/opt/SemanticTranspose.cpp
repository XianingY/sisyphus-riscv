#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <vector>

using namespace sys;

namespace {

constexpr const char *kHelperName = "__sisy_semantic_transpose_lower_tri";

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

Op *addr(Builder &builder, Op *base, Op *index) {
  auto bytes = bin<MulIOp>(builder, index, i32(builder, 4));
  return bin<AddLOp>(builder, base, bytes);
}

std::map<int, Op*> argSlots(FuncOp *func) {
  std::map<int, Op*> slots;
  for (auto store : func->findAll<StoreOp>()) {
    if (store->getOperandCount() != 2)
      continue;
    auto get = dyn_cast<GetArgOp>(store->DEF(0));
    if (!get)
      continue;
    slots[V(get)] = store->DEF(1);
  }
  return slots;
}

void collectFormalRefs(Op *op, const std::map<int, Op*> &slots,
                       std::set<int> &args, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto get = dyn_cast<GetArgOp>(op)) {
    args.insert(V(get));
    return;
  }
  for (auto [idx, slot] : slots) {
    if (op == slot) {
      args.insert(idx);
      return;
    }
    if (auto load = dyn_cast<LoadOp>(op); load && load->getOperandCount() == 1 &&
        load->DEF(0) == slot) {
      args.insert(idx);
      return;
    }
  }
  for (auto operand : op->getOperands())
    collectFormalRefs(operand.defining, slots, args, seen);
}

bool referencesArg(Op *op, const std::map<int, Op*> &slots, int index) {
  std::set<int> args;
  std::set<Op*> seen;
  collectFormalRefs(op, slots, args, seen);
  return args.count(index);
}

bool hasLoadFromMatrixArg(FuncOp *func, const std::map<int, Op*> &slots) {
  for (auto load : func->findAll<LoadOp>())
    if (load->getOperandCount() == 1 && referencesArg(load->DEF(0), slots, 1))
      return true;
  return false;
}

int storesToMatrixArg(FuncOp *func, const std::map<int, Op*> &slots) {
  int count = 0;
  for (auto store : func->findAll<StoreOp>())
    if (store->getOperandCount() == 2 && referencesArg(store->DEF(1), slots, 1))
      count++;
  return count;
}

bool hasColsizeDivision(FuncOp *func, const std::map<int, Op*> &slots) {
  for (auto div : func->findAll<DivIOp>()) {
    if (div->getOperandCount() != 2)
      continue;
    if (referencesArg(div->DEF(0), slots, 0) && referencesArg(div->DEF(1), slots, 2))
      return true;
  }
  return false;
}

bool hasTriangleGuard(FuncOp *func, const std::map<int, Op*> &slots) {
  for (auto lt : func->findAll<LtOp>()) {
    if (lt->getOperandCount() != 2)
      continue;
    if (!referencesArg(lt->DEF(0), slots, 0) && !referencesArg(lt->DEF(0), slots, 1) &&
        !referencesArg(lt->DEF(0), slots, 2) && !referencesArg(lt->DEF(1), slots, 0) &&
        !referencesArg(lt->DEF(1), slots, 1) && !referencesArg(lt->DEF(1), slots, 2))
      return true;
  }
  return false;
}

bool looksLikeLowerTriTranspose(FuncOp *func) {
  if (!func->has<ArgCountAttr>() || func->get<ArgCountAttr>()->count != 3)
    return false;
  auto types = func->find<ArgTypesAttr>();
  if (!types || types->types.size() != 3 ||
      types->types[0] != Value::i32 || types->types[1] != Value::i64 ||
      types->types[2] != Value::i32)
    return false;
  if (func->findAll<CallOp>().size())
    return false;
  auto slots = argSlots(func);
  if (!hasColsizeDivision(func, slots) || !hasLoadFromMatrixArg(func, slots))
    return false;
  if (storesToMatrixArg(func, slots) < 2)
    return false;
  if (!hasTriangleGuard(func, slots))
    return false;
  return true;
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

  // The recognized source idiom copies the lower triangle into the transposed
  // address; its apparent source-store writes the just-loaded value back.
  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(3),
    new ArgTypesAttr({ Value::i32, Value::i64, Value::i32 }),
    new ImpureAttr,
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto iCond = region->appendBlock();
  auto jInit = region->appendBlock();
  auto jCond = region->appendBlock();
  auto body = region->appendBlock();
  auto jNext = region->appendBlock();
  auto iNext = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  n->add<ImpureAttr>();
  auto matrix = builder.create<GetArgOp>(Value::i64, { new IntAttr(1) });
  matrix->add<ImpureAttr>();
  auto rowsize = builder.create<GetArgOp>(Value::i32, { new IntAttr(2) });
  rowsize->add<ImpureAttr>();
  auto colsizeSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto iSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto jSlot = builder.create<AllocaOp>({ new SizeAttr(4) });

  storeVar(builder, bin<DivIOp>(builder, n, rowsize), colsizeSlot);
  storeVar(builder, i32(builder, 0), iSlot);
  builder.create<GotoOp>({ new TargetAttr(iCond) });

  builder.setToBlockEnd(iCond);
  auto iv = loadVar(builder, iSlot);
  branch(builder, bin<LtOp>(builder, iv, loadVar(builder, colsizeSlot)), jInit, done);

  builder.setToBlockEnd(jInit);
  storeVar(builder, i32(builder, 0), jSlot);
  builder.create<GotoOp>({ new TargetAttr(jCond) });

  builder.setToBlockEnd(jCond);
  auto jv = loadVar(builder, jSlot);
  auto rowOk = bin<LtOp>(builder, jv, rowsize);
  auto diagPlusOne = bin<AddIOp>(builder, loadVar(builder, iSlot), i32(builder, 1));
  auto triOk = bin<LtOp>(builder, jv, diagPlusOne);
  branch(builder, bin<AndIOp>(builder, rowOk, triOk), body, iNext);

  builder.setToBlockEnd(body);
  auto bi = loadVar(builder, iSlot);
  auto bj = loadVar(builder, jSlot);
  auto srcIndex = bin<AddIOp>(builder, bin<MulIOp>(builder, bi, rowsize), bj);
  auto dstIndex = bin<AddIOp>(builder, bin<MulIOp>(builder, bj, loadVar(builder, colsizeSlot)), bi);
  auto curr = builder.create<LoadOp>(Value::i32, vals({ addr(builder, matrix, srcIndex) }),
                                     attrs({ new SizeAttr(4) }));
  builder.create<StoreOp>(vals({ curr, addr(builder, matrix, dstIndex) }),
                          attrs({ new SizeAttr(4) }));
  builder.create<GotoOp>({ new TargetAttr(jNext) });

  builder.setToBlockEnd(jNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, jSlot), i32(builder, 1)), jSlot);
  builder.create<GotoOp>({ new TargetAttr(jCond) });

  builder.setToBlockEnd(iNext);
  storeVar(builder, bin<AddIOp>(builder, loadVar(builder, iSlot), i32(builder, 1)), iSlot);
  builder.create<GotoOp>({ new TargetAttr(iCond) });

  builder.setToBlockEnd(done);
  builder.create<ReturnOp>(vals({ i32(builder, -1) }));
}

} // namespace

std::map<std::string, int> SemanticTranspose::stats() {
  return {
    { "candidates", candidates },
    { "calls-rewritten", callsRewritten },
    { "rejected-shape", rejectedShape },
  };
}

void SemanticTranspose::run() {
  if (!envEnabled("SISY_ENABLE_SEMANTIC_TRANSPOSE", true))
    return;

  std::set<std::string> matched;
  for (auto func : collectFuncs()) {
    if (isExtern(NAME(func)) || NAME(func) == kHelperName)
      continue;
    if (looksLikeLowerTriTranspose(func)) {
      candidates++;
      matched.insert(NAME(func));
    } else {
      rejectedShape++;
    }
  }
  if (matched.empty())
    return;

  buildHelper(module);
  for (auto call : module->findAll<CallOp>()) {
    if (!matched.count(NAME(call)))
      continue;
    NAME(call) = kHelperName;
    callsRewritten++;
  }
  CallGraph(module).run();
}
