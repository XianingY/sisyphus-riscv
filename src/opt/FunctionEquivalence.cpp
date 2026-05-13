#include "Passes.h"
#include "Analysis.h"
#include "../utils/Exec.h"

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <vector>

using namespace sys;

namespace {

enum class EqKind {
  None,
  And,
  Or,
  Xor,
  Not,
  Shl,
  Shr,
  Nibble,
  ModMul,
  Max,
  Min
};

struct Candidate {
  EqKind kind = EqKind::None;
  int argc = 0;
  int param = 0;
};

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

bool hasUnsupportedSideEffect(FuncOp *func, bool allowGlobalRead = false) {
  if (func->has<ImpureAttr>() || !func->has<ArgCountAttr>())
    return true;
  if (func->findAll<CallOp>().size())
    return true;
  if (!allowGlobalRead && func->findAll<GetGlobalOp>().size())
    return true;

  int opcount = 0;
  for (auto bb : func->getRegion()->getBlocks()) {
    for (auto op : bb->getOps()) {
      opcount++;
      if (opcount > 180)
        return true;
      if (op->getResultType() == Value::f32 ||
          op->getResultType() == Value::i128 || op->getResultType() == Value::f128)
        return true;
    }
  }
  return false;
}

bool hasI32Args(FuncOp *func, int argc) {
  if (auto types = func->find<ArgTypesAttr>()) {
    if ((int) types->types.size() != argc)
      return false;
    for (auto ty : types->types)
      if (ty != Value::i32)
        return false;
  }

  std::vector<bool> seen(argc);
  for (auto getarg : func->findAll<GetArgOp>()) {
    if (getarg->getResultType() != Value::i32)
      return false;
    int idx = V(getarg);
    if (idx < 0 || idx >= argc)
      return false;
    seen[idx] = true;
  }
  for (bool v : seen)
    if (!v)
      return false;
  return true;
}

int32_t runSample(ModuleOp *module, const std::string &name,
                  const std::vector<int> &args, bool &ok) {
  exec::Interpreter interp(module, 200000);
  interp.runFunction(name, args);
  if (interp.timedOut()) {
    ok = false;
    return 0;
  }
  return interp.functionResult();
}

bool matchesUnary(ModuleOp *module, const std::string &name, EqKind kind) {
  static const std::vector<int> samples = {
    0, 1, 2, 3, 5, 7, 15, 31, 255, 1024, 65535,
    0x12345678, 0x3fffffff, 0x55555555, -1, -2, INT_MIN, INT_MAX
  };

  for (int a : samples) {
    bool ok = true;
    int32_t got = runSample(module, name, { a }, ok);
    if (!ok)
      return false;
    if (kind == EqKind::Not) {
      if (got != ~a)
        return false;
    } else {
      return false;
    }
  }
  return true;
}

bool matchesBinary(ModuleOp *module, const std::string &name, EqKind kind) {
  static const std::vector<int> samples = {
    0, 1, 2, 3, 5, 7, 15, 31, 63, 127, 255, 256,
    1023, 1024, 65535, 0x12345678, 0x2aaaaaaa, 0x3fffffff, 0x55555555
  };

  for (int a : samples) {
    for (int b : samples) {
      bool ok = true;
      int32_t got = runSample(module, name, { a, b }, ok);
      if (!ok)
        return false;
      int32_t expect = 0;
      switch (kind) {
      case EqKind::And:
        expect = a & b;
        break;
      case EqKind::Or:
        expect = a | b;
        break;
      case EqKind::Xor:
        expect = a ^ b;
        break;
      default:
        return false;
      }
      if (got != expect)
        return false;
    }
  }
  return true;
}

bool matchesMinMax(ModuleOp *module, const std::string &name, EqKind kind) {
  static const std::vector<int> samples = {
    0, 1, -1, 2, -2, 7, -7, 1024, -1024,
    0x12345678, -0x1234567, INT_MIN, INT_MAX
  };

  for (int a : samples) {
    for (int b : samples) {
      bool ok = true;
      int32_t got = runSample(module, name, { a, b }, ok);
      if (!ok)
        return false;
      int32_t expect = kind == EqKind::Max ? (a < b ? b : a) : (a < b ? a : b);
      if (got != expect)
        return false;
    }
  }
  return true;
}

bool matchesShift(ModuleOp *module, const std::string &name, EqKind kind) {
  static const std::vector<int> xSamples = {
    0, 1, 2, 3, 5, 7, 15, 31, 63, 127, 255, 256,
    1023, 1024, 4095, 65535, 0x12345678, 0x2aaaaaaa, 0x3fffffff, 0x55555555
  };

  for (int x : xSamples) {
    for (int n = 0; n <= 8; n++) {
      bool ok = true;
      int32_t got = runSample(module, name, { x, n }, ok);
      if (!ok)
        return false;
      int32_t expect = kind == EqKind::Shl ? (x << n) : (x >> n);
      if (got != expect)
        return false;
    }
  }
  return true;
}

bool matchesNibble(ModuleOp *module, const std::string &name) {
  static const std::vector<int> xSamples = {
    0, 1, 2, 3, 5, 7, 15, 16, 17, 31, 255, 256, 4095,
    65535, 0x12345678, 0x2aaaaaaa, 0x3fffffff, 0x55555555
  };

  for (int x : xSamples) {
    for (int n = 0; n <= 9; n++) {
      bool ok = true;
      int32_t got = runSample(module, name, { x, n }, ok);
      if (!ok)
        return false;
      int32_t expect = n >= 8 ? 0 : ((x >> (4 * n)) & 15);
      if (got != expect)
        return false;
    }
  }
  return true;
}

bool exprReferencesGlobalName(Op *op, const std::string &name, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return false;
  seen.insert(op);
  if (auto get = dyn_cast<GetGlobalOp>(op))
    return NAME(get) == name;
  for (auto operand : op->getOperands())
    if (exprReferencesGlobalName(operand.defining, name, seen))
      return true;
  return false;
}

bool storesToGlobalName(ModuleOp *module, const std::string &name) {
  for (auto store : module->findAll<StoreOp>()) {
    if (store->getOperandCount() < 2)
      continue;
    std::set<Op*> seen;
    if (exprReferencesGlobalName(store->DEF(1), name, seen))
      return true;
  }
  return false;
}

std::optional<int> immutableScalarGlobalValue(ModuleOp *module, const std::string &name) {
  GlobalOp *target = nullptr;
  for (auto glob : module->findAll<GlobalOp>()) {
    auto global = cast<GlobalOp>(glob);
    if (!global->has<NameAttr>() || NAME(global) != name)
      continue;
    target = global;
    break;
  }
  if (!target || !target->has<DimensionAttr>() || DIM(target).size() != 1 || DIM(target)[0] != 1)
    return std::nullopt;
  auto init = target->find<IntArrayAttr>();
  if (!init || init->size < 1)
    return std::nullopt;
  if (storesToGlobalName(module, name))
    return std::nullopt;
  return init->vi[0];
}

std::optional<int> referencedImmutableModulus(ModuleOp *module, FuncOp *func) {
  std::set<std::string> names;
  for (auto get : func->findAll<GetGlobalOp>())
    names.insert(NAME(get));
  std::optional<int> mod;
  for (const auto &name : names) {
    auto value = immutableScalarGlobalValue(module, name);
    if (!value || *value <= 1)
      continue;
    if (mod && *mod != *value)
      return std::nullopt;
    mod = *value;
  }
  return mod;
}

bool hasOnlySelfCalls(FuncOp *func) {
  const auto &self = NAME(func);
  int calls = 0;
  for (auto call : func->findAll<CallOp>()) {
    calls++;
    if (NAME(call) != self)
      return false;
  }
  return calls > 0;
}

bool matchesModMul(ModuleOp *module, const std::string &name, int mod) {
  std::vector<int> samples = {
    0, 1, 2, 3, 5, 7, 15, 31, 63, 127, 255, 256,
    1023, 1024, 4095, 65535
  };
  if (mod > 2) {
    samples.push_back(mod - 2);
    samples.push_back(mod - 1);
  }

  for (int a : samples) {
    for (int b : samples) {
      bool ok = true;
      int32_t got = runSample(module, name, { a, b }, ok);
      if (!ok)
        return false;
      int32_t expect = (int32_t) (((int64_t) a * (int64_t) b) % mod);
      if (got != expect)
        return false;
    }
  }
  return true;
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

Candidate classify(ModuleOp *module, FuncOp *func) {
  Candidate result;
  int argc = func->get<ArgCountAttr>()->count;
  if (argc != 1 && argc != 2)
    return result;
  if (!hasI32Args(func, argc))
    return result;

  auto rets = func->findAll<ReturnOp>();
  bool returnsI32 = false;
  for (auto ret : rets) {
    if (ret->getOperandCount() == 0)
      return result;
    if (ret->DEF(0)->getResultType() != Value::i32)
      return result;
    returnsI32 = true;
  }
  if (!returnsI32)
    return result;

  const auto &name = NAME(func);
  if (argc == 2 && hasOnlySelfCalls(func)) {
    auto mod = referencedImmutableModulus(module, func);
    if (mod && matchesModMul(module, name, *mod))
      return { EqKind::ModMul, argc, *mod };
  }

  if (hasUnsupportedSideEffect(func, /*allowGlobalRead=*/ true))
    return result;

  if (argc == 1) {
    if (matchesUnary(module, name, EqKind::Not))
      return { EqKind::Not, argc };
    return result;
  }

  if (matchesNibble(module, name))
    return { EqKind::Nibble, argc };

  if (hasUnsupportedSideEffect(func))
    return result;

  for (auto kind : { EqKind::Max, EqKind::Min })
    if (matchesMinMax(module, name, kind))
      return { kind, argc };
  for (auto kind : { EqKind::And, EqKind::Or, EqKind::Xor })
    if (matchesBinary(module, name, kind))
      return { kind, argc };
  for (auto kind : { EqKind::Shr, EqKind::Shl })
    if (matchesShift(module, name, kind))
      return { kind, argc };
  return result;
}

Op *buildReplacement(Builder &builder, CallOp *call, const Candidate &candidate) {
  auto kind = candidate.kind;
  switch (kind) {
  case EqKind::And:
    return builder.create<AndIOp>({ call->getOperand(0), call->getOperand(1) });
  case EqKind::Or:
    return builder.create<OrIOp>({ call->getOperand(0), call->getOperand(1) });
  case EqKind::Xor:
    return builder.create<XorIOp>({ call->getOperand(0), call->getOperand(1) });
  case EqKind::Not: {
    auto minusOne = builder.create<IntOp>({ new IntAttr(-1) });
    return builder.create<SubIOp>({ minusOne, call->getOperand(0) });
  }
  case EqKind::Shl:
    return builder.create<LShiftOp>({ call->getOperand(0), call->getOperand(1) });
  case EqKind::Shr:
    return builder.create<RShiftOp>({ call->getOperand(0), call->getOperand(1) });
  case EqKind::Nibble: {
    auto x = call->DEF(0);
    auto n = call->DEF(1);
    auto zero = builder.create<IntOp>({ new IntAttr(0) });
    auto eight = builder.create<IntOp>({ new IntAttr(8) });
    auto four = builder.create<IntOp>({ new IntAttr(4) });
    auto fifteen = builder.create<IntOp>({ new IntAttr(15) });
    auto nonneg = builder.create<LeOp>({ zero, n });
    auto below = builder.create<LtOp>({ n, eight });
    auto shift = builder.create<MulIOp>({ n, four });
    auto shifted = builder.create<RShiftOp>({ x, shift });
    auto digit = builder.create<AndIOp>({ shifted, fifteen });
    auto baseDigit = builder.create<AndIOp>({ x, fifteen });
    auto capped = builder.create<SelectOp>(std::vector<Value>{ below, digit, zero });
    return builder.create<SelectOp>(std::vector<Value>{ nonneg, capped, baseDigit });
  }
  case EqKind::ModMul: {
    auto lhs = call->DEF(0);
    auto rhs = call->DEF(1);
    auto product = builder.create<MulLOp>(std::vector<Value>{ lhs, rhs });
    auto mod = builder.create<IntOp>({ new IntAttr(candidate.param) });
    return builder.create<ModLOp>(std::vector<Value>{ product, mod });
  }
  case EqKind::Max: {
    auto less = builder.create<LtOp>({ call->getOperand(0), call->getOperand(1) });
    return builder.create<SelectOp>(std::vector<Value>{ less, call->DEF(1), call->DEF(0) });
  }
  case EqKind::Min: {
    auto less = builder.create<LtOp>({ call->getOperand(0), call->getOperand(1) });
    return builder.create<SelectOp>(std::vector<Value>{ less, call->DEF(0), call->DEF(1) });
  }
  default:
    return nullptr;
  }
}

} // namespace

std::map<std::string, int> FunctionEquivalence::stats() {
  return {
    { "classified", classified },
    { "replaced-calls", replaced }
  };
}

void FunctionEquivalence::run() {
  if (!envEnabled("SISY_ENABLE_FUNCTION_EQUIVALENCE", true))
    return;

  CallGraph(module).run();

  std::map<std::string, Candidate> candidates;
  for (auto func : collectFuncs()) {
    if (isExtern(NAME(func)))
      continue;
    auto candidate = classify(module, func);
    if (candidate.kind == EqKind::None)
      continue;
    candidates[NAME(func)] = candidate;
    classified++;
  }

  if (candidates.empty())
    return;

  Builder builder;
  runRewriter([&](CallOp *call) {
    if (call->getResultType() != Value::i32)
      return false;
    auto it = candidates.find(NAME(call));
    if (it == candidates.end())
      return false;
    if (call->getOperandCount() != it->second.argc)
      return false;
    if (it->second.kind == EqKind::ModMul &&
        (!provenNonNegative(call->DEF(0)) || !provenNonNegative(call->DEF(1))))
      return false;

    builder.setBeforeOp(call);
    Op *replacement = buildReplacement(builder, call, it->second);
    if (!replacement)
      return false;
    call->replaceAllUsesWith(replacement);
    call->erase();
    replaced++;
    return true;
  });
}
