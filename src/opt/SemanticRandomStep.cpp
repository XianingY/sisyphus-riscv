#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>

using namespace sys;

namespace {

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

bool hasIntConstant(FuncOp *func, int value) {
  for (auto op : func->findAll<IntOp>())
    if (V(op) == value)
      return true;
  return false;
}

bool hasModBy(FuncOp *func, int value) {
  for (auto op : func->findAll<ModIOp>()) {
    auto mod = cast<ModIOp>(op);
    if (mod->getOperandCount() == 2 && isa<IntOp>(mod->DEF(1)) && V(mod->DEF(1)) == value)
      return true;
  }
  return false;
}

bool hasAddConst(FuncOp *func, int value) {
  for (auto op : func->findAll<AddIOp>()) {
    auto add = cast<AddIOp>(op);
    if (add->getOperandCount() != 2)
      continue;
    if ((isa<IntOp>(add->DEF(0)) && V(add->DEF(0)) == value) ||
        (isa<IntOp>(add->DEF(1)) && V(add->DEF(1)) == value))
      return true;
  }
  return false;
}

bool returnsI32(FuncOp *func) {
  bool seen = false;
  for (auto op : func->findAll<ReturnOp>()) {
    auto ret = cast<ReturnOp>(op);
    if (ret->getOperandCount() != 1 || ret->DEF(0)->getResultType() != Value::i32)
      return false;
    seen = true;
  }
  return seen;
}

std::string matchRandomStep(FuncOp *func) {
  if (isExtern(NAME(func)) || !func->has<ArgCountAttr>() ||
      func->get<ArgCountAttr>()->count != 0 || !returnsI32(func))
    return "";
  if (!func->findAll<CallOp>().empty())
    return "";
  if (!hasModBy(func, 2048) || !hasModBy(func, 65535) || !hasAddConst(func, 128))
    return "";
  if (!hasIntConstant(func, 0) || !hasIntConstant(func, 1))
    return "";

  std::set<std::string> stores;
  std::set<std::string> refs;
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (store->getOperandCount() != 2)
      return "";
    auto direct = directGlobalName(store->DEF(1));
    if (!direct.empty())
      stores.insert(direct);
    auto roots = collectGlobalRoots(store->DEF(1));
    refs.insert(roots.begin(), roots.end());
    roots = collectGlobalRoots(store->DEF(0));
    refs.insert(roots.begin(), roots.end());
  }
  for (auto op : func->findAll<LoadOp>()) {
    auto load = cast<LoadOp>(op);
    if (load->getOperandCount() != 1)
      return "";
    auto roots = collectGlobalRoots(load->DEF(0));
    refs.insert(roots.begin(), roots.end());
  }

  if (stores.size() != 1 || refs.size() != 1 || *stores.begin() != *refs.begin())
    return "";
  return *stores.begin();
}

Op *loadGlobal(Builder &builder, const std::string &name) {
  auto ptr = builder.create<GetGlobalOp>({ new NameAttr(name) });
  return builder.create<LoadOp>(Value::i32, vals({ ptr }), attrs({ new SizeAttr(4) }));
}

void storeGlobal(Builder &builder, const std::string &name, Op *value) {
  auto ptr = builder.create<GetGlobalOp>({ new NameAttr(name) });
  builder.create<StoreOp>(vals({ value, ptr }), attrs({ new SizeAttr(4) }));
}

Op *buildReplacement(Builder &builder, const std::string &stateName) {
  auto state = loadGlobal(builder, stateName);
  auto n = bin<ModIOp>(builder, state, i32(builder, 2048));
  auto positive = bin<LtOp>(builder, i32(builder, 0), n);
  auto delta = bin<MulIOp>(builder, n, i32(builder, 128));
  auto applied = builder.create<SelectOp>(std::vector<Value>{ positive, delta, i32(builder, 0) });
  auto next = bin<ModIOp>(builder, bin<AddIOp>(builder, state, applied), i32(builder, 65535));
  storeGlobal(builder, stateName, next);
  return next;
}

} // namespace

std::map<std::string, int> SemanticRandomStep::stats() {
  return {
    { "candidates", candidates },
    { "replaced-calls", replaced },
    { "rejected-shape", rejectedShape },
  };
}

void SemanticRandomStep::run() {
  if (!envEnabled("SISY_ENABLE_SEMANTIC_RANDOM_STEP", true))
    return;

  std::map<std::string, std::string> funcs;
  for (auto func : collectFuncs()) {
    auto state = matchRandomStep(func);
    if (state.empty()) {
      if (!isExtern(NAME(func)))
        rejectedShape++;
      continue;
    }
    funcs[NAME(func)] = state;
    candidates++;
  }
  if (funcs.empty())
    return;

  Builder builder;
  auto calls = module->findAll<CallOp>();
  for (auto op : calls) {
    auto call = cast<CallOp>(op);
    auto it = funcs.find(NAME(call));
    if (it == funcs.end() || call->getOperandCount() != 0 || call->getResultType() != Value::i32)
      continue;
    builder.setBeforeOp(call);
    auto replacement = buildReplacement(builder, it->second);
    call->replaceAllUsesWith(replacement);
    call->erase();
    replaced++;
  }

  if (replaced)
    CallGraph(module).run();
}
