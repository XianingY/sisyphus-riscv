#include "Exec.h"
#include "../codegen/Attrs.h"
#include <cstdlib>
#include <cstring>

using namespace sys;
using namespace sys::exec;

#define sys_unreachable(x) \
  do { std::cerr << x << "\n"; std::abort(); } while (0)

Interpreter::Interpreter(ModuleOp *module, size_t explicitStepLimit) {
  if (explicitStepLimit > 0) {
    stepLimit = explicitStepLimit;
  } else if (const char *env = std::getenv("SISY_EXEC_STEP_LIMIT")) {
    long long parsed = atoll(env);
    if (parsed > 0)
      stepLimit = (size_t) parsed;
  }

  auto region = module->getRegion();
  auto block = region->getFirstBlock();
  for (auto op : block->getOps()) {
    if (isa<GlobalOp>(op)) {
      const auto &name = NAME(op);
      int size = SIZE(op) / 4;
      
      if (auto intArr = op->find<IntArrayAttr>()) {
        int *vp = new int[size];
        if (intArr->vi)
          memcpy(vp, intArr->vi, size * 4);
        else
          memset(vp, 0, size * 4);
        globalMap[name] = Value { .vi = (intptr_t) vp };
        addRange(globalRanges, (intptr_t) vp, (size_t) SIZE(op));
      }
      if (auto fpArr = op->find<FloatArrayAttr>()) {
        float *vfp = new float[size];
        if (fpArr->vf)
          memcpy(vfp, fpArr->vf, size * 4);
        else
          memset(vfp, 0, size * 4);
        globalMap[name] = Value { .vi = (intptr_t) vfp };
        fpGlobals.insert(name);
        addRange(globalRanges, (intptr_t) vfp, (size_t) SIZE(op));
      }
      continue;
    }

    if (isa<FuncOp>(op)) {
      fnMap[NAME(op)] = op;
      continue;
    }
    
    sys_unreachable("unexpected top level op: " << op);
  }
}

Interpreter::~Interpreter() {
  for (const auto &[name, ptr] : globalMap) {
    // Hopefully this won't violate strict aliasing rule.
    if (fpGlobals.count(name))
      delete[] ((float*) ptr.vi);
    else
      delete[] ((int*) ptr.vi);
  }
}

intptr_t Interpreter::eval(Op *op) {
  if (!value.count(op))
    sys_unreachable("undefined op" << op);
  if (op->getResultType() == sys::Value::f32)
    sys_unreachable("op of float type: " << op);
  return value[op].vi;
}

float Interpreter::evalf(Op *op) {
  if (!value.count(op))
    sys_unreachable("undefined op" << op);
  if (op->getResultType() != sys::Value::f32)
    sys_unreachable("op of non-float type: " << op);
  return value[op].vf;
}

void Interpreter::store(Op *op, intptr_t v) {
  value[op] = Value { .vi = v };
}

void Interpreter::store(Op *op, float v) {
  value[op] = Value { .vf = v };
}

void Interpreter::addRange(std::vector<MemoryRange> &ranges, intptr_t addr, size_t size) {
  if (!addr || !size)
    return;

  uintptr_t begin = (uintptr_t) addr;
  uintptr_t end = begin + size;
  if (end < begin)
    return;
  ranges.push_back(MemoryRange { begin, end });
}

bool Interpreter::isAddressValid(intptr_t addr, size_t size) const {
  if (!addr || !size)
    return false;

  uintptr_t begin = (uintptr_t) addr;
  uintptr_t end = begin + size;
  if (end < begin)
    return false;
  if (lastValidCached && begin >= lastValidBegin && end <= lastValidEnd)
    return true;

  auto inRanges = [this, begin, end](const std::vector<MemoryRange> &ranges) {
    for (const auto &range : ranges) {
      if (begin >= range.begin && end <= range.end) {
        lastValidCached = true;
        lastValidBegin = range.begin;
        lastValidEnd = range.end;
        return true;
      }
    }
    return false;
  };

  return inRanges(globalRanges) || inRanges(stackRanges);
}

size_t Interpreter::getAccessSize(Op *op, bool isLoad) {
  auto &cache = isLoad ? loadSizeCache : storeSizeCache;
  auto it = cache.find(op);
  if (it != cache.end())
    return it->second;

  if (auto sizeAttr = op->find<SizeAttr>())
    return cache[op] = sizeAttr->value;

  if (isLoad) {
    if (op->getResultType() == sys::Value::f32)
      return cache[op] = 4;
    if (op->getResultType() == sys::Value::i64)
      return cache[op] = 8;
    if (op->getResultType() == sys::Value::i128)
      return cache[op] = 16;
    return cache[op] = 4;
  }

  auto defTy = op->DEF(0)->getResultType();
  if (defTy == sys::Value::f32)
    return cache[op] = 4;
  if (defTy == sys::Value::i64)
    return cache[op] = 8;
  if (defTy == sys::Value::i128)
    return cache[op] = 16;
  return cache[op] = 4;
}

// The registers are in fact 64-bit.
#define EXEC_BINARY(Ty, sign) \
  case Ty::id: \
    store(op, (intptr_t) (int32_t) ((int32_t) eval(op->DEF(0)) sign (int32_t) eval(op->DEF(1)))); \
    break

#define EXEC_BINARY_L(Ty, sign) \
  case Ty::id: \
    store(op, (intptr_t) ((eval(op->DEF(0)) sign eval(op->DEF(1))))); \
    break

#define EXEC_BINARY_F(Ty, sign) \
  case Ty::id: \
    store(op, evalf(op->DEF(0)) sign evalf(op->DEF(1))); \
    break

#define EXEC_BINARY_FCOMP(Ty, sign) \
  case Ty::id: \
    store(op, (intptr_t) (evalf(op->DEF(0)) sign evalf(op->DEF(1)))); \
    break

#define EXEC_UNARY(Ty, sign) \
  case Ty::id: \
    store(op, (intptr_t) (int32_t) (sign (int32_t) eval(op->DEF()))); \
    break

#define EXEC_UNARY_F(Ty, sign) \
  case Ty::id: \
    store(op, sign evalf(op->DEF())); \
    break

// Defined in Pass.cpp
namespace sys {
  bool isExtern(const std::string &name);
}

void Interpreter::exec(Op *op) {
  switch (op->opid) {
  case IntOp::id:
    store(op, (intptr_t) V(op));
    break;
  case FloatOp::id:
    store(op, F(op));
    break;
  case I2FOp::id:
    store(op, (float) eval(op->DEF()));
    break;
  case F2IOp::id:
    store(op, (intptr_t) evalf(op->DEF()));
    break;
  EXEC_BINARY(AddIOp, +);
  EXEC_BINARY(SubIOp, -);
  EXEC_BINARY(MulIOp, *);
  case DivIOp::id: {
    auto rhs = (int32_t) eval(op->DEF(1));
    if (rhs == 0) {
      executionTimedOut = true;
      store(op, (intptr_t) 0);
      break;
    }
    store(op, (intptr_t) (int32_t) ((int32_t) eval(op->DEF(0)) / rhs));
    break;
  }
  case ModIOp::id: {
    auto rhs = (int32_t) eval(op->DEF(1));
    if (rhs == 0) {
      executionTimedOut = true;
      store(op, (intptr_t) 0);
      break;
    }
    store(op, (intptr_t) (int32_t) ((int32_t) eval(op->DEF(0)) % rhs));
    break;
  }
  EXEC_BINARY(EqOp, ==);
  EXEC_BINARY(NeOp, !=);
  EXEC_BINARY(LtOp, <);
  EXEC_BINARY(LeOp, <=);
  EXEC_BINARY(AndIOp, &);
  EXEC_BINARY(OrIOp, |);
  EXEC_BINARY(XorIOp, ^);
  EXEC_BINARY(LShiftOp, <<);
  
  EXEC_BINARY_L(AddLOp, +);
  EXEC_BINARY_L(SubLOp, -);
  EXEC_BINARY_L(MulLOp, *);
  case DivLOp::id: {
    auto rhs = eval(op->DEF(1));
    if (rhs == 0) {
      executionTimedOut = true;
      store(op, (intptr_t) 0);
      break;
    }
    store(op, (intptr_t) (eval(op->DEF(0)) / rhs));
    break;
  }
  case ModLOp::id: {
    auto rhs = eval(op->DEF(1));
    if (rhs == 0) {
      executionTimedOut = true;
      store(op, (intptr_t) 0);
      break;
    }
    store(op, (intptr_t) (eval(op->DEF(0)) % rhs));
    break;
  }
  EXEC_BINARY_L(LShiftLOp, <<);
  EXEC_BINARY_L(RShiftLOp, >>);

  EXEC_BINARY_F(AddFOp, +);
  EXEC_BINARY_F(SubFOp, -);
  EXEC_BINARY_F(MulFOp, *);
  EXEC_BINARY_F(DivFOp, /);
  EXEC_BINARY_FCOMP(EqFOp, ==);
  EXEC_BINARY_FCOMP(LeFOp, <=);
  EXEC_BINARY_FCOMP(LtFOp, <);
  EXEC_BINARY_FCOMP(NeFOp, !=);

  EXEC_UNARY(NotOp, !);
  EXEC_UNARY(SetNotZeroOp, !!);
  EXEC_UNARY(MinusOp, -);

  EXEC_UNARY_F(MinusFOp, -);
  case RShiftOp::id: {
    auto x = (int32_t) eval(op->DEF(0));
    auto y = (int32_t) eval(op->DEF(1));
    store(op, (intptr_t) (int32_t) (x >> y));
    break;
  }
  case PhiOp::id: {
    const auto &ops = op->getOperands();
    const auto &attrs = op->getAttrs();
    bool success = false;
    for (int i = 0; i < ops.size(); i++) {
      if (FROM(attrs[i]) == prev) {
        value[op] = value[ops[i].defining];
        success = true;
        break;
      }
    }
    if (!success)
      sys_unreachable("undef phi: coming from " << bbmap[prev] << ", current place is " << bbmap[op->getParent()]);
    break;
  }
  case GetGlobalOp::id: {
    const auto &name = NAME(op);
    if (!globalMap.count(name))
      sys_unreachable("unknown global: " << name);

    value[op] = globalMap[name];
    break;
  }
  case LoadOp::id: {
    size_t size = getAccessSize(op, /*isLoad=*/true);
    bool fp = op->getResultType() == sys::Value::f32;
    intptr_t addr = eval(op->DEF());
    if (!isAddressValid(addr, size)) {
      if (fp)
        store(op, 0.0f);
      else
        store(op, (intptr_t) 0);
      break;
    }
    if (fp)
      store(op, *(float*) addr);
    else if (size == 4)
      store(op, (intptr_t) *(int*) addr);
    else if (size == 8)
      store(op, *(intptr_t*) addr);
    else if (size == 16) {
      intptr_t lower = 0;
      memcpy(&lower, (void*) addr, sizeof(lower));
      store(op, lower);
    }
    else
      sys_unreachable("unsupported load size " << size);
    break;
  }
  case StoreOp::id: {
    size_t size = getAccessSize(op, /*isLoad=*/false);
    intptr_t addr = eval(op->DEF(1));
    if (!isAddressValid(addr, size))
      break;
    Op *def = op->DEF(0);
    bool fp = def->getResultType() == sys::Value::f32;
    if (fp)
      *(float*) addr = evalf(def);
    else if (size == 4)
      *(int*) addr = eval(def);
    else if (size == 8)
      *(intptr_t*) addr = eval(def);
    else if (size == 16) {
      auto lower = eval(def);
      memcpy((void*) addr, &lower, sizeof(lower));
    }
    else
      sys_unreachable("unsupported store size " << size);
    break;
  }
  case SelectOp::id: {
    Op *cond = op->DEF(0);
    store(op, eval(cond) ? eval(op->DEF(1)) : eval(op->DEF(2)));
    break;
  }
  default:
    sys_unreachable("unknown op type: " << ip);
  }
}

Interpreter::Value Interpreter::applyExtern(const std::string &name, const std::vector<Value> &callArgs) {
  if (name == "getint") {
    int x; inbuf >> x;
    return Value { .vi = x };
  }
  if (name == "getch") {
    char x = inbuf.get();
    return Value { .vi = x };
  }
  if (name == "getfloat") {
    std::string x; inbuf >> x;
    return Value { .vf = strtof(x.c_str(), nullptr) };
  }
  if (name == "getarray") {
    int n; inbuf >> n;
    // Some public inputs contain unsigned values outside signed int range.
    unsigned *ptr = (unsigned*) callArgs[0].vi;
    for (int i = 0; i < n; i++)
      inbuf >> ptr[i];
    return Value { .vi = n };
  }
  if (name == "getfarray") {
    int n; inbuf >> n;
    float *ptr = (float*) callArgs[0].vi;
    std::string x;
    for (int i = 0; i < n; i++) {
      inbuf >> x;
      ptr[i] = strtof(x.c_str(), nullptr);
    }
    return Value { .vi = n };
  }
  if (name == "putint") {
    intptr_t v = callArgs[0].vi;
    // Direct cast of `(int) v` is implementation-defined.
    outbuf << (int) (unsigned) v;
    return Value();
  }
  if (name == "putch") {
    outbuf << (char) callArgs[0].vi;
    return Value();
  }
  if (name == "putfloat") {
    outbuf << callArgs[0].vf;
    return Value();
  }
  if (name == "putfarray") {
    int n = callArgs[0].vi;
    float *ptr = (float*) callArgs[1].vi;
    outbuf << n << ":";
    for (int i = 0; i < n; i++) {
      outbuf << " " << ptr[i];
    }
    outbuf << "\n";
    return Value();
  }
  if (name == "putarray") {
    int n = callArgs[0].vi;
    int *ptr = (int*) callArgs[1].vi;
    outbuf << n << ":";
    for (int i = 0; i < n; i++)
      outbuf << " " << ptr[i];
    outbuf << "\n";
    return Value();
  }
  if (name == "_sysy_starttime" || name == "_sysy_stoptime")
    return Value();
  sys_unreachable("unknown extern function: " << name);
}

Interpreter::Value Interpreter::execf(Region *region, const std::vector<Value> &fnArgs) {
  size_t frameMark = stackRanges.size();
  auto entry = region->getFirstBlock();
  ip = entry->getFirstOp();
  while (!executionTimedOut && !isa<ReturnOp>(ip)) {
    if (stepCount++ >= stepLimit) {
      executionTimedOut = true;
      break;
    }
    switch (ip->opid) {
    case GotoOp::id: {
      auto dest = TARGET(ip);
      prev = ip->getParent();
      ip = dest->getFirstOp();
      break;
    }
    case BranchOp::id: {
      auto def = ip->DEF(0);
      auto dest = eval(def) ? TARGET(ip) : ELSE(ip);
      prev = ip->getParent();
      ip = dest->getFirstOp();
      break;
    }
    // Note that we need the stack space to live long enough,
    // till we exit this interpreted function.
    case AllocaOp::id: {
      size_t size = SIZE(ip);
      void *space = alloca(size);
      addRange(stackRanges, (intptr_t) space, size);
      store(ip, (intptr_t) space);
      ip = ip->nextOp();
      break;
    }
    case CallOp::id: {
      const auto &name = NAME(ip);
      auto operands = ip->getOperands();
      std::vector<Value> callArgs;
      callArgs.reserve(operands.size());
      for (auto operand : operands)
        callArgs.push_back(value[operand.defining]);

      bool utilized = false;
      switch (cache_type) {
      case 3: {
        if (callArgs.size() >= 3) {
          auto i = callArgs[0].vi, j = callArgs[1].vi, k = callArgs[2].vi;
          if (i < CACHE_3_N && j < CACHE_3_N && k < CACHE_3_N && i >= 0 && j >= 0 && k >= 0) {
            value[ip] = { ((cache_3_ptr) cache)[i][j][k] };
            utilized = true;
          }
        }
        break;
      }
      case 2: {
        if (callArgs.size() >= 2) {
          auto i = callArgs[0].vi, j = callArgs[1].vi;
          if (i < CACHE_2_N && j < CACHE_2_N && i >= 0 && j >= 0) {
            value[ip] = { ((cache_2_ptr) cache)[i][j] };
            utilized = true;
          }
        }
        break;
      }
      default:
        break;
      }
      if (utilized) {
        ip = ip->nextOp();
        break;
      }

      if (isExtern(name)) {
        value[ip] = applyExtern(name, callArgs);
        ip = ip->nextOp();
      }
      else {
        Op *call = ip;
        Value v;
        {
          SemanticScope scope(*this);
          v = execf(fnMap[name]->getRegion(), callArgs);
        }
        value[call] = v;
        ip = call->nextOp();
      }
      break;
    }
    case GetArgOp::id: {
      int argIndex = V(ip);
      if (argIndex < 0 || argIndex >= (int) fnArgs.size())
        sys_unreachable("getarg out of range: " << argIndex);
      value[ip] = fnArgs[argIndex];
      ip = ip->nextOp();
      break;
    }
    default:
      exec(ip);
      ip = ip->nextOp();
      break;
    }
  }
  // Now `ip` is a ReturnOp.
  Value ret = Value { .vi = 0 };
  if (!executionTimedOut && ip->getOperandCount()) {
    auto def = ip->DEF(0);
    ret = value[def];
  }
  stackRanges.resize(frameMark);
  return ret;
}

void Interpreter::run(std::istream &input) {
  stepCount = 0;
  executionTimedOut = false;
  inbuf << std::hexfloat << input.rdbuf();
  outbuf << std::hexfloat;
  auto exit = execf(fnMap["main"]->getRegion(), {});
  retcode = exit.vi;
}

void Interpreter::runFunction(const std::string &func, const std::vector<int> &args) {
  stepCount = 0;
  executionTimedOut = false;
  lastValidCached = false;
  std::vector<Value> values;
  values.reserve(args.size());
  for (auto x : args)
    values.push_back(Value { .vi = x });
  auto exit = execf(fnMap[func]->getRegion(), values);
  if (cache) {
    if (cache_type == 3) {
      if (args.size() >= 3 && args[0] >= 0 && args[1] >= 0 && args[2] >= 0 &&
          args[0] < CACHE_3_N && args[1] < CACHE_3_N && args[2] < CACHE_3_N) {
        auto x = (cache_3_ptr) cache;
        x[args[0]][args[1]][args[2]] = exit.vi;
      }
    }
    if (cache_type == 2) {
      if (args.size() >= 2 && args[0] >= 0 && args[1] >= 0 &&
          args[0] < CACHE_2_N && args[1] < CACHE_2_N) {
        auto x = (cache_2_ptr) cache;
        x[args[0]][args[1]] = exit.vi;
      }
    }
  }
  retcode = exit.vi;
}
