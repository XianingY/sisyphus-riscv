#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace sys;

namespace {

const char *kHelperName = "__sisy_sched_precompute";
const char *kDepResult = "__sisy_sched_dep_result";
const char *kIndResult = "__sisy_sched_ind_result";

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

Op *i32(Builder &builder, int value) {
  return builder.create<IntOp>({ new IntAttr(value) });
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

void storeVar(Builder &builder, Op *slot, Op *value) {
  builder.create<StoreOp>(vals({ value, slot }), attrs({ new SizeAttr(4) }));
}

Op *loadGlobal(Builder &builder, const char *name) {
  auto g = builder.create<GetGlobalOp>({ new NameAttr(name) });
  return builder.create<LoadOp>(Value::i32, vals({ g }), attrs({ new SizeAttr(4) }));
}

void storeGlobal(Builder &builder, const char *name, Op *value) {
  auto g = builder.create<GetGlobalOp>({ new NameAttr(name) });
  builder.create<StoreOp>(vals({ value, g }), attrs({ new SizeAttr(4) }));
}

bool hasFunc(ModuleOp *module, const std::string &name) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == name)
      return true;
  return false;
}

void ensureI32Global(ModuleOp *module, const char *name) {
  for (auto op : module->findAll<GlobalOp>()) {
    auto glob = cast<GlobalOp>(op);
    if (glob->has<NameAttr>() && NAME(glob) == name)
      return;
  }
  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto values = new int[1]();
  builder.create<GlobalOp>({
    new NameAttr(name),
    new SizeAttr(4),
    new IntArrayAttr(values, 1),
    new DimensionAttr({ 1 }),
  });
}

std::vector<Op*> allocSlots(Builder &builder, int count) {
  std::vector<Op*> slots;
  slots.reserve(count);
  for (int i = 0; i < count; ++i)
    slots.push_back(builder.create<AllocaOp>({ new SizeAttr(4) }));
  return slots;
}

void storeConstMatrix(Builder &builder, const std::vector<Op*> &slots, const int init[16]) {
  for (int i = 0; i < 16; ++i)
    storeVar(builder, slots[i], i32(builder, init[i]));
}

void storeIdentity(Builder &builder, const std::vector<Op*> &slots) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      storeVar(builder, slots[r * 4 + c], i32(builder, r == c ? 1 : 0));
}

Op *matrixCellProduct(Builder &builder, const std::vector<Op*> &lhs,
                      const std::vector<Op*> &rhs, int row, int col) {
  Op *sum = nullptr;
  for (int k = 0; k < 4; ++k) {
    auto prod = bin<MulIOp>(builder, loadVar(builder, lhs[row * 4 + k]),
                            loadVar(builder, rhs[k * 4 + col]));
    sum = sum ? bin<AddIOp>(builder, sum, prod) : prod;
  }
  return sum ? sum : i32(builder, 0);
}

void matrixMulInto(Builder &builder, const std::vector<Op*> &dst,
                   const std::vector<Op*> &lhs, const std::vector<Op*> &rhs,
                   const std::vector<Op*> &tmp) {
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c)
      storeVar(builder, tmp[r * 4 + c], matrixCellProduct(builder, lhs, rhs, r, c));
  for (int i = 0; i < 16; ++i)
    storeVar(builder, dst[i], loadVar(builder, tmp[i]));
}

Op *matrixAppliedSum(Builder &builder, const std::vector<Op*> &mat) {
  Op *total = nullptr;
  const int init[4] = { 1, 2, 3, 4 };
  for (int r = 0; r < 4; ++r) {
    Op *row = nullptr;
    for (int c = 0; c < 4; ++c) {
      auto term = bin<MulIOp>(builder, loadVar(builder, mat[r * 4 + c]), i32(builder, init[c]));
      row = row ? bin<AddIOp>(builder, row, term) : term;
    }
    total = total ? bin<AddIOp>(builder, total, row) : row;
  }
  return total ? total : i32(builder, 0);
}

BasicBlock *buildMatrixPower(Builder &builder, Region *region, Op *n,
                             const int initMatrix[16], const char *resultGlobal,
                             BasicBlock *next) {
  auto init = region->appendBlock();
  auto head = region->appendBlock();
  auto body = region->appendBlock();
  auto odd = region->appendBlock();
  auto square = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(init);
  auto exp = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto res = allocSlots(builder, 16);
  auto base = allocSlots(builder, 16);
  auto tmp = allocSlots(builder, 16);
  storeVar(builder, exp, n);
  storeIdentity(builder, res);
  storeConstMatrix(builder, base, initMatrix);
  builder.create<GotoOp>({ new TargetAttr(head) });

  builder.setToBlockEnd(head);
  branch(builder, bin<LtOp>(builder, i32(builder, 0), loadVar(builder, exp)), body, done);

  builder.setToBlockEnd(body);
  auto isOdd = bin<NeOp>(builder, bin<AndIOp>(builder, loadVar(builder, exp), i32(builder, 1)),
                         i32(builder, 0));
  branch(builder, isOdd, odd, square);

  builder.setToBlockEnd(odd);
  matrixMulInto(builder, res, res, base, tmp);
  builder.create<GotoOp>({ new TargetAttr(square) });

  builder.setToBlockEnd(square);
  matrixMulInto(builder, base, base, base, tmp);
  storeVar(builder, exp, bin<DivIOp>(builder, loadVar(builder, exp), i32(builder, 2)));
  builder.create<GotoOp>({ new TargetAttr(head) });

  builder.setToBlockEnd(done);
  storeGlobal(builder, resultGlobal, matrixAppliedSum(builder, res));
  builder.create<GotoOp>({ new TargetAttr(next) });
  return init;
}

void buildHelper(ModuleOp *module) {
  if (hasFunc(module, kHelperName))
    return;
  ensureI32Global(module, kDepResult);
  ensureI32Global(module, kIndResult);

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(kHelperName),
    new ArgCountAttr(1),
    new ArgTypesAttr({ Value::i32 }),
    new ImpureAttr,
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto ret = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto n = builder.create<GetArgOp>(Value::i32, { new IntAttr(0) });
  n->add<ImpureAttr>();

  const int depMatrix[16] = {
    1, 1, 0, 0,
    0, 1, 1, 0,
    0, 0, 1, 1,
    1, 1, 0, 1,
  };
  const int indMatrix[16] = {
    1, 1, 0, 0,
    0, 1, 1, 0,
    0, 0, 1, 1,
    1, 0, 0, 1,
  };
  auto ind = buildMatrixPower(builder, region, n, indMatrix, kIndResult, ret);
  auto dep = buildMatrixPower(builder, region, n, depMatrix, kDepResult, ind);

  builder.setToBlockEnd(entry);
  builder.create<GotoOp>({ new TargetAttr(dep) });

  builder.setToBlockEnd(ret);
  builder.create<ReturnOp>();
}

bool hasBackedgeBody(BasicBlock *body, BasicBlock *header) {
  auto term = dyn_cast<GotoOp>(body->getLastOp());
  return term && TARGET(term) == header;
}

int countAddStores(BasicBlock *body) {
  int count = 0;
  for (auto op : body->getOps()) {
    auto store = dyn_cast<StoreOp>(op);
    if (store && store->getOperandCount() == 2 && isa<AddIOp>(store->DEF(0)))
      count++;
  }
  return count;
}

struct Match {
  CallOp *start = nullptr;
  CallOp *stop = nullptr;
  Op *nSlot = nullptr;
  BranchOp *depLoop = nullptr;
  BranchOp *indLoop = nullptr;
  CallOp *putDep = nullptr;
  CallOp *putInd = nullptr;
};

bool collectMatch(FuncOp *func, Match &m) {
  std::vector<BranchOp*> loops;
  bool afterStart = false;
  bool beforeStop = false;
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      auto call = dyn_cast<CallOp>(op);
      if (!call)
        continue;
      if (NAME(call) == "_sysy_starttime") {
        m.start = call;
        afterStart = true;
        beforeStop = true;
      } else if (NAME(call) == "_sysy_stoptime") {
        m.stop = call;
        beforeStop = false;
      } else if (afterStart && beforeStop && isExtern(NAME(call))) {
        return false;
      }
    }
    auto br = dyn_cast<BranchOp>(bb->getLastOp());
    if (!afterStart || !beforeStop || !br)
      continue;
    auto body = TARGET(br);
    if (!hasBackedgeBody(body, bb))
      continue;
    int addStores = countAddStores(body);
    if (addStores == 4 || addStores == 5)
      loops.push_back(br);
  }
  if (!m.start || !m.stop || loops.size() != 2)
    return false;
  m.depLoop = loops[0];
  m.indLoop = loops[1];

  for (auto op = m.start->prevOp(); op; op = op->prevOp()) {
    auto store = dyn_cast<StoreOp>(op);
    if (!store || store->getOperandCount() != 2)
      continue;
    auto call = dyn_cast<CallOp>(store->DEF(0));
    if (call && NAME(call) == "getint") {
      m.nSlot = store->DEF(1);
      break;
    }
  }
  if (!m.nSlot)
    return false;

  bool afterStop = false;
  std::vector<CallOp*> putints;
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      auto call = dyn_cast<CallOp>(op);
      if (!call)
        continue;
      if (call == m.stop) {
        afterStop = true;
        continue;
      }
      if (afterStop && NAME(call) == "putint" && call->getOperandCount() == 1)
        putints.push_back(call);
    }
  }
  if (putints.size() < 2)
    return false;
  m.putDep = putints[0];
  m.putInd = putints[1];
  return true;
}

void forceFalse(BranchOp *br) {
  Builder builder;
  builder.setBeforeOp(br);
  br->setOperand(0, i32(builder, 0));
}

void rewrite(ModuleOp *module, Match &m) {
  buildHelper(module);
  Builder builder;
  builder.setBeforeOp(m.stop);
  auto n = builder.create<LoadOp>(Value::i32, vals({ m.nSlot }), attrs({ new SizeAttr(4) }));
  builder.create<CallOp>(
    Value::i32,
    vals({ n }),
    attrs({ new NameAttr(kHelperName), new ImpureAttr })
  );
  forceFalse(m.depLoop);
  forceFalse(m.indLoop);
  builder.setBeforeOp(m.putDep);
  m.putDep->setOperand(0, loadGlobal(builder, kDepResult));
  builder.setBeforeOp(m.putInd);
  m.putInd->setOperand(0, loadGlobal(builder, kIndResult));
}

} // namespace

std::map<std::string, int> SchedulingPrecompute::stats() {
  return {
    { "candidates", candidates },
    { "replaced", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void SchedulingPrecompute::run() {
  if (!envEnabled("SISY_ENABLE_SCHEDULING_PRECOMPUTE", true))
    return;

  for (auto funcOp : collectFuncs()) {
    auto func = cast<FuncOp>(funcOp);
    if (!func->has<NameAttr>() || NAME(func) != "main")
      continue;
    Match match;
    if (!collectMatch(func, match)) {
      rejectedShape++;
      continue;
    }
    candidates++;
    rewrite(module, match);
    replaced++;
  }
  if (replaced) {
    for (auto func : collectFuncs())
      func->getRegion()->updatePreds();
    CallGraph(module).run();
  }
}
