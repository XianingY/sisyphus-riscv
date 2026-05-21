#include "CFGToLegacy.h"

#include "CFGLegality.h"

#include "../codegen/Attrs.h"
#include "../codegen/CodeGen.h"
#include "../utils/DynamicCast.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sys::cfg {

namespace {

Value::Type toValueType(hir::TypeKind type) {
  switch (type) {
  case hir::TypeKind::Float:
    return Value::f32;
  case hir::TypeKind::Pointer:
  case hir::TypeKind::Array:
  case hir::TypeKind::Function:
    return Value::i64;
  case hir::TypeKind::Void:
    return Value::i128;
  case hir::TypeKind::Int:
  case hir::TypeKind::Unknown:
    return Value::i32;
  }
  return Value::i32;
}

size_t defaultSize(hir::TypeKind type) {
  switch (type) {
  case hir::TypeKind::Float:
  case hir::TypeKind::Int:
    return 4;
  case hir::TypeKind::Pointer:
  case hir::TypeKind::Array:
  case hir::TypeKind::Function:
    return 8;
  case hir::TypeKind::Void:
    return 0;
  case hir::TypeKind::Unknown:
    return 4;
  }
  return 4;
}

size_t productDims(const std::vector<int> &dims) {
  if (dims.empty())
    return 1;
  size_t prod = 1;
  for (int dim : dims)
    prod *= (size_t) std::max(dim, 1);
  return prod;
}

bool isArrayLike(const SymbolInfo &sym) {
  return !sym.dims.empty() || sym.type == hir::TypeKind::Array || sym.type == hir::TypeKind::Pointer;
}

std::string normalizeCallee(const std::string &name) {
  if (name == "starttime")
    return "_sysy_starttime";
  if (name == "stoptime")
    return "_sysy_stoptime";
  return name;
}

struct SymbolState {
  SymbolInfo info;
  Value slot;
  bool pointerSlot = false;
};

struct PendingPhi {
  int bid = -1;
  const Inst *inst = nullptr;
  Op *placeholder = nullptr;
};

struct FunctionState {
  const Func *cfgFunc = nullptr;
  FuncOp *funcOp = nullptr;
  int cfgBlockCount = 0;
  std::vector<BasicBlock*> blocks;
  BasicBlock *exitBlock = nullptr;
  Value retSlot;
  bool hasReturnValue = false;
  size_t retSize = 4;
  std::unordered_map<std::string, Value> values;
  std::unordered_map<std::string, SymbolState> symbols;
  std::unordered_map<BasicBlock*, std::unordered_map<std::string, Value>> edgeLoads;
  std::vector<PendingPhi> pendingPhi;
};

class LegacyLowerer {
  const Module &cfgModule;
  std::vector<std::string> &errors;

  std::unique_ptr<ModuleOp> module;
  Builder builder;
  BasicBlock *topBlock = nullptr;
  std::unordered_map<std::string, Value> globals;
  std::unordered_map<std::string, SymbolInfo> globalInfo;

public:
  explicit LegacyLowerer(const Module &cfgModule, std::vector<std::string> &errors):
    cfgModule(cfgModule), errors(errors) {}

  std::unique_ptr<ModuleOp> run() {
    module = std::make_unique<ModuleOp>();
    topBlock = module->createFirstBlock();
    builder.setToBlockEnd(topBlock);

    indexGlobals();
    lowerGlobals();

    for (const auto &cfgFunc : cfgModule.funcs) {
      if (!lowerFunction(cfgFunc))
        return nullptr;
    }

    return std::move(module);
  }

private:
  void addError(const std::string &msg) {
    errors.push_back("cfg->legacy: " + msg);
  }

  void indexGlobals() {
    globalInfo.clear();
    for (const auto &sym : cfgModule.globals)
      globalInfo[sym.name] = sym;
  }

  void lowerGlobals() {
    builder.setToBlockEnd(topBlock);
    for (const auto &sym : cfgModule.globals) {
      std::vector<int> dims = sym.dims.empty() ? std::vector<int>{1} : sym.dims;
      size_t elemCount = productDims(dims);
      if (elemCount == 0)
        elemCount = 1;
      size_t elemSize = sym.elemSize ? sym.elemSize : defaultSize(sym.elementType);
      if (!elemSize)
        elemSize = 4;
      size_t storageSize = sym.storageSize ? sym.storageSize : (elemSize * elemCount);

      Value g;
      if (sym.elementType == hir::TypeKind::Float) {
        float *data = nullptr;
        if (!sym.floatArrayInit.empty()) {
          data = new float[elemCount];
          std::fill(data, data + elemCount, 0.0f);
          size_t n = std::min(elemCount, sym.floatArrayInit.size());
          for (size_t i = 0; i < n; i++)
            data[i] = sym.floatArrayInit[i];
        } else if (sym.hasFloatInit) {
          data = new float[elemCount];
          std::fill(data, data + elemCount, 0.0f);
          data[0] = (float) sym.floatInit;
        }

        g = builder.create<GlobalOp>({
          new SizeAttr(storageSize),
          new FloatArrayAttr(data, (int) elemCount),
          new NameAttr(sym.name),
          new DimensionAttr(dims),
        });
        g.defining->add<FPAttr>();
        if (!sym.isMutable)
          g.defining->add<ConstAttr>();
      } else {
        int *data = nullptr;
        if (!sym.intArrayInit.empty()) {
          data = new int[elemCount];
          std::fill(data, data + elemCount, 0);
          size_t n = std::min(elemCount, sym.intArrayInit.size());
          for (size_t i = 0; i < n; i++)
            data[i] = sym.intArrayInit[i];
        } else if (sym.hasIntInit) {
          data = new int[elemCount];
          std::fill(data, data + elemCount, 0);
          data[0] = (int) sym.intInit;
        }

        g = builder.create<GlobalOp>({
          new SizeAttr(storageSize),
          new IntArrayAttr(data, (int) elemCount),
          new NameAttr(sym.name),
          new DimensionAttr(dims),
        });
        if (!sym.isMutable)
          g.defining->add<ConstAttr>();
      }
      globals[sym.name] = g;
    }
  }

  bool lowerFunction(const Func &cfgFunc) {
    builder.setToBlockEnd(topBlock);

    std::vector<Value::Type> argTypes;
    argTypes.reserve(cfgFunc.params.size());
    for (const auto &param : cfgFunc.params)
      argTypes.push_back(isArrayLike(param) ? Value::i64 : toValueType(param.type));

    auto *funcOp = builder.create<FuncOp>({
      new NameAttr(cfgFunc.name),
      new ArgCountAttr((int) cfgFunc.params.size()),
      new ArgTypesAttr(argTypes)
    });

    int bbCount = std::max(1, (int) cfgFunc.blocks.size());
    funcOp->createFirstBlock();
    for (int i = 1; i < bbCount; i++)
      funcOp->getRegion()->appendBlock();

    if (!cfgFunc.params.empty()) {
      bool impure = false;
      for (const auto &param : cfgFunc.params)
        impure = impure || isArrayLike(param);
      if (impure)
        funcOp->add<ImpureAttr>();
    }

    FunctionState st;
    st.cfgFunc = &cfgFunc;
    st.funcOp = funcOp;
    st.cfgBlockCount = bbCount;
    st.blocks.reserve(bbCount);
    for (auto bb : funcOp->getRegion()->getBlocks())
      st.blocks.push_back(bb);
    st.exitBlock = funcOp->getRegion()->appendBlock();

    initializeFunctionSymbols(st);
    if (!lowerFunctionBody(st))
      return false;
    if (!finalizePhis(st))
      return false;
    emitFunctionExit(st);

    return true;
  }

  SymbolInfo fallbackLocal(const std::string &name, hir::TypeKind type) {
    SymbolInfo info;
    info.name = name;
    info.type = type == hir::TypeKind::Unknown ? hir::TypeKind::Int : type;
    info.elementType = info.type;
    info.elemSize = defaultSize(info.type);
    if (!info.elemSize)
      info.elemSize = 4;
    info.storageSize = info.elemSize;
    return info;
  }

  SymbolState &ensureLocalSymbol(FunctionState &st, const std::string &name, hir::TypeKind fallbackType) {
    auto it = st.symbols.find(name);
    if (it != st.symbols.end())
      return it->second;

    SymbolInfo info = fallbackLocal(name, fallbackType);
    builder.setToBlockStart(st.blocks[0]);
    auto slot = builder.create<AllocaOp>({ new SizeAttr(std::max((size_t) 4, info.storageSize)) });
    if (info.type == hir::TypeKind::Float)
      slot->add<FPAttr>();
    SymbolState state;
    state.info = info;
    state.slot = slot;
    state.pointerSlot = false;
    auto [pos, _] = st.symbols.emplace(name, state);
    return pos->second;
  }

  void allocateParamSymbol(FunctionState &st, const SymbolInfo &param, int index) {
    builder.setToBlockEnd(st.blocks[0]);

    Value::Type argTy = toValueType(param.type);
    if (isArrayLike(param))
      argTy = Value::i64;

    auto arg = builder.create<GetArgOp>(argTy, { new IntAttr(index) });
    // Keep argument position information stable for backend ABI lowering.
    // If unused getarg ops are dropped early, stack-arg offset mapping for
    // mixed/large signatures can become ambiguous.
    arg->add<ImpureAttr>();
    SymbolState sym;
    sym.info = param;

    if (isArrayLike(param)) {
      auto slot = builder.create<AllocaOp>({ new SizeAttr(8) });
      builder.create<StoreOp>({ arg, slot }, { new SizeAttr(8) });
      sym.slot = slot;
      sym.pointerSlot = true;
    } else {
      size_t sz = std::max((size_t) 4, param.storageSize ? param.storageSize : defaultSize(param.type));
      auto slot = builder.create<AllocaOp>({ new SizeAttr(sz) });
      builder.create<StoreOp>({ arg, slot }, { new SizeAttr(sz) });
      if (param.type == hir::TypeKind::Float)
        slot->add<FPAttr>();
      sym.slot = slot;
      sym.pointerSlot = false;
    }

    st.symbols[param.name] = sym;
  }

  void allocateLocalSymbol(FunctionState &st, const SymbolInfo &local) {
    if (local.name.empty() || st.symbols.count(local.name))
      return;

    builder.setToBlockEnd(st.blocks[0]);
    SymbolState sym;
    sym.info = local;

    if (isArrayLike(local)) {
      size_t elemCount = productDims(local.dims);
      if (!elemCount)
        elemCount = 1;
      size_t elemSize = local.elemSize ? local.elemSize : defaultSize(local.elementType);
      if (!elemSize)
        elemSize = 4;
      size_t totalSize = local.storageSize ? local.storageSize : elemCount * elemSize;
      if (!totalSize)
        totalSize = elemSize;

      auto base = builder.create<AllocaOp>({ new SizeAttr(totalSize) });
      if (local.elementType == hir::TypeKind::Float)
        base->add<FPAttr>();
      if (!local.dims.empty())
        base->add<DimensionAttr>(local.dims);

      auto slot = builder.create<AllocaOp>({ new SizeAttr(8) });
      builder.create<StoreOp>({ base, slot }, { new SizeAttr(8) });

      sym.slot = slot;
      sym.pointerSlot = true;
    } else {
      size_t sz = std::max((size_t) 4, local.storageSize ? local.storageSize : defaultSize(local.type));
      auto slot = builder.create<AllocaOp>({ new SizeAttr(sz) });
      if (local.type == hir::TypeKind::Float)
        slot->add<FPAttr>();
      sym.slot = slot;
      sym.pointerSlot = false;

      if (local.hasIntInit || local.hasFloatInit) {
        Value init;
        if (local.hasFloatInit)
          init = builder.create<FloatOp>({ new FloatAttr((float) local.floatInit) });
        else
          init = builder.create<IntOp>({ new IntAttr((int) local.intInit) });
        builder.create<StoreOp>({ init, slot }, { new SizeAttr(sz) });
      }
    }

    st.symbols[local.name] = sym;
  }

  void initializeFunctionSymbols(FunctionState &st) {
    if (!st.cfgFunc)
      return;

    for (size_t i = 0; i < st.cfgFunc->params.size(); i++)
      allocateParamSymbol(st, st.cfgFunc->params[i], (int) i);
    for (const auto &local : st.cfgFunc->locals)
      allocateLocalSymbol(st, local);

    st.hasReturnValue = st.cfgFunc->returnType != hir::TypeKind::Void;
    if (st.hasReturnValue) {
      builder.setToBlockEnd(st.blocks[0]);
      st.retSize = std::max((size_t) 4, defaultSize(st.cfgFunc->returnType));
      st.retSlot = builder.create<AllocaOp>({ new SizeAttr(st.retSize) });
      if (st.cfgFunc->returnType == hir::TypeKind::Float)
        st.retSlot.defining->add<FPAttr>();

      Value init;
      if (st.cfgFunc->returnType == hir::TypeKind::Float)
        init = builder.create<FloatOp>({ new FloatAttr(0.0f) });
      else
        init = builder.create<IntOp>({ new IntAttr(0) });
      builder.create<StoreOp>({ init, st.retSlot }, { new SizeAttr(st.retSize) });
    }
  }

  bool parseIntToken(const std::string &token, long long &value) {
    if (token.empty() || token[0] != '#')
      return false;
    char *end = nullptr;
    long long parsed = std::strtoll(token.c_str() + 1, &end, 10);
    if (!end || *end)
      return false;
    value = parsed;
    return true;
  }

  bool parseFloatToken(const std::string &token, float &value) {
    if (token.rfind("f#", 0) != 0)
      return false;
    char *end = nullptr;
    float parsed = std::strtof(token.c_str() + 2, &end);
    if (!end || *end)
      return false;
    value = parsed;
    return true;
  }

  Value materializeImmediate(FunctionState &st, int bid, const std::string &token, hir::TypeKind expectedType) {
    builder.setToBlockEnd(st.blocks[bid]);

    long long iv = 0;
    if (parseIntToken(token, iv)) {
      auto c = builder.create<IntOp>({ new IntAttr((int) iv) });
      return c;
    }

    float fv = 0.0f;
    if (parseFloatToken(token, fv)) {
      auto c = builder.create<FloatOp>({ new FloatAttr(fv) });
      return c;
    }

    if (expectedType == hir::TypeKind::Float)
      return builder.create<FloatOp>({ new FloatAttr(0.0f) });
    return builder.create<IntOp>({ new IntAttr(0) });
  }

  void setBeforeTerminatorIfPresent(FunctionState &st, int bid) {
    auto *bb = st.blocks[bid];
    if (!bb || bb->getOpCount() == 0) {
      builder.setToBlockEnd(st.blocks[bid]);
      return;
    }
    auto *term = bb->getLastOp();
    if (isa<GotoOp>(term) || isa<BranchOp>(term) || isa<ReturnOp>(term)) {
      builder.setBeforeOp(term);
      return;
    }
    builder.setToBlockEnd(st.blocks[bid]);
  }

  Value materializeImmediateBeforeTerminator(FunctionState &st, int bid, const std::string &token,
                                             hir::TypeKind expectedType) {
    setBeforeTerminatorIfPresent(st, bid);

    long long iv = 0;
    if (parseIntToken(token, iv))
      return builder.create<IntOp>({ new IntAttr((int) iv) });

    float fv = 0.0f;
    if (parseFloatToken(token, fv))
      return builder.create<FloatOp>({ new FloatAttr(fv) });

    if (expectedType == hir::TypeKind::Float)
      return builder.create<FloatOp>({ new FloatAttr(0.0f) });
    return builder.create<IntOp>({ new IntAttr(0) });
  }

  Value resolveToken(FunctionState &st, int bid, const std::string &token, hir::TypeKind expectedType, bool allowEdgePlaceholder) {
    if (token.empty()) {
      addError("empty value token in @" + st.cfgFunc->name);
      return materializeImmediate(st, bid, "#0", expectedType);
    }

    if (token[0] == '#' || token.rfind("f#", 0) == 0)
      return materializeImmediate(st, bid, token, expectedType);

    if (token[0] == '$') {
      if (!allowEdgePlaceholder) {
        addError("unexpected edge placeholder token '" + token + "' in non-phi op");
        return materializeImmediate(st, bid, "#0", expectedType);
      }
      return Value();
    }

    auto it = st.values.find(token);
    if (it != st.values.end())
      return it->second;

    addError("undefined SSA token '" + token + "' in @" + st.cfgFunc->name);
    return materializeImmediate(st, bid, "#0", expectedType);
  }

  SymbolInfo lookupSymbolInfo(FunctionState &st, const std::string &name, hir::TypeKind fallbackType) {
    auto it = st.symbols.find(name);
    if (it != st.symbols.end())
      return it->second.info;
    auto git = globalInfo.find(name);
    if (git != globalInfo.end())
      return git->second;
    return fallbackLocal(name, fallbackType);
  }

  Value localAddress(FunctionState &st, int bid, const std::string &name, hir::TypeKind fallbackType) {
    auto &sym = ensureLocalSymbol(st, name, fallbackType);
    if (!sym.pointerSlot)
      return sym.slot;

    builder.setToBlockEnd(st.blocks[bid]);
    return builder.create<LoadOp>(Value::i64, { sym.slot }, { new SizeAttr(8) });
  }

  Value symbolBaseAddress(FunctionState &st, int bid, const std::string &name, hir::TypeKind fallbackType) {
    // Function-local symbols must shadow globals with the same name.
    if (st.symbols.count(name))
      return localAddress(st, bid, name, fallbackType);

    auto git = globals.find(name);
    if (git != globals.end()) {
      builder.setToBlockEnd(st.blocks[bid]);
      return builder.create<GetGlobalOp>({ new NameAttr(name) });
    }
    return localAddress(st, bid, name, fallbackType);
  }

  Value buildIndexedAddress(FunctionState &st, int bid, const SymbolInfo &info, Value base,
                            const std::vector<std::string> &indices,
                            const std::vector<size_t> *explicitStrides = nullptr,
                            bool linearizedScalarIndex = false) {
    if (indices.empty())
      return base;

    size_t elemSize = info.elemSize ? info.elemSize : defaultSize(info.elementType);
    if (!elemSize)
      elemSize = 4;

    std::vector<size_t> strides(indices.size(), elemSize);
    if (explicitStrides && !explicitStrides->empty()) {
      size_t usable = std::min(indices.size(), explicitStrides->size());
      for (size_t i = 0; i < usable; i++)
        strides[i] = (*explicitStrides)[i];
    } else if (!info.dims.empty() && !(linearizedScalarIndex && indices.size() == 1)) {
      for (size_t i = 0; i < indices.size(); i++) {
        size_t stride = elemSize;
        for (size_t j = i + 1; j < info.dims.size(); j++)
          stride *= (size_t) std::max(info.dims[j], 1);
        strides[i] = stride;
      }
    }

    builder.setToBlockEnd(st.blocks[bid]);
    Value addr = base;
    for (size_t i = 0; i < indices.size(); i++) {
      auto idx = resolveToken(st, bid, indices[i], hir::TypeKind::Int, false);
      auto stride = builder.create<IntOp>({ new IntAttr((int) strides[i]) });
      auto offset = builder.create<MulLOp>({ idx, stride });
      addr = builder.create<AddLOp>({ addr, offset });
    }
    return addr;
  }

  size_t loadStoreSize(const Inst &inst, const SymbolInfo &sym, bool indexed) {
    if (inst.memSize)
      return inst.memSize;
    if (indexed)
      return sym.elemSize ? sym.elemSize : defaultSize(sym.elementType);
    if (sym.storageSize)
      return sym.storageSize;
    return defaultSize(sym.type);
  }

  bool lowerInst(FunctionState &st, int bid, const Inst &inst) {
    auto *bb = st.blocks[bid];
    (void) bb;

    switch (inst.kind) {
    case OpKind::Nop:
      return true;
    case OpKind::Phi: {
      builder.setToBlockStart(st.blocks[bid]);
      auto ph = builder.create<PhiOp>();
      if (!inst.result.empty())
        st.values[inst.result] = ph;
      st.pendingPhi.push_back(PendingPhi { bid, &inst, ph });
      return true;
    }
    case OpKind::Arith:
    case OpKind::Cmp:
      return lowerArithCmp(st, bid, inst);
    case OpKind::Load:
      return lowerLoad(st, bid, inst);
    case OpKind::Store:
      return lowerStore(st, bid, inst);
    case OpKind::Call:
      return lowerCall(st, bid, inst);
    case OpKind::Ret:
      return lowerRet(st, bid, inst);
    case OpKind::Br:
      return lowerBr(st, bid, inst);
    case OpKind::CondBr:
      return lowerCondBr(st, bid, inst);
    }
    return true;
  }

  Value forceIntValue(FunctionState &st, int bid, Value value) {
    if (!value.defining)
      return materializeImmediate(st, bid, "#0", hir::TypeKind::Int);
    if (value.defining->getResultType() == Value::f32) {
      builder.setToBlockEnd(st.blocks[bid]);
      auto z = builder.create<FloatOp>({ new FloatAttr(0.0f) });
      return builder.create<NeFOp>({ value, z });
    }
    return value;
  }

  bool lowerArithCmp(FunctionState &st, int bid, const Inst &inst) {
    builder.setToBlockEnd(st.blocks[bid]);

    std::vector<Value> ops;
    ops.reserve(inst.args.size());
    for (const auto &token : inst.args)
      ops.push_back(resolveToken(st, bid, token, inst.type, false));

    Op *def = nullptr;
    const bool isCmp = inst.kind == OpKind::Cmp;
    auto valueIsFloat = [](Value v) {
      return v.defining && v.defining->getResultType() == Value::f32;
    };
    bool cmpFloat = inst.elementType == hir::TypeKind::Float;
    bool arithFloat = inst.type == hir::TypeKind::Float;
    for (auto v : ops) {
      cmpFloat = cmpFloat || valueIsFloat(v);
      arithFloat = arithFloat || valueIsFloat(v);
    }
    const bool isFloat = isCmp ? cmpFloat : arithFloat;

    if (inst.symbol == "f2i" && ops.size() == 1)
      def = builder.create<F2IOp>({ ops[0] });
    else if (inst.symbol == "i2f" && ops.size() == 1)
      def = builder.create<I2FOp>({ ops[0] });
    else if (inst.symbol == "!" && ops.size() == 1)
      def = builder.create<NotOp>({ forceIntValue(st, bid, ops[0]) });
    else if (inst.symbol == "-" && ops.size() == 1)
      def = isFloat ? (Op*) builder.create<MinusFOp>({ ops[0] }) : (Op*) builder.create<MinusOp>({ ops[0] });
    else if (ops.size() == 2) {
      Value lhs = ops[0], rhs = ops[1];
      if (inst.symbol == "&&") {
        lhs = builder.create<SetNotZeroOp>({ forceIntValue(st, bid, lhs) });
        rhs = builder.create<SetNotZeroOp>({ forceIntValue(st, bid, rhs) });
        def = builder.create<AndIOp>({ lhs, rhs });
      } else if (inst.symbol == "||") {
        lhs = builder.create<SetNotZeroOp>({ forceIntValue(st, bid, lhs) });
        rhs = builder.create<SetNotZeroOp>({ forceIntValue(st, bid, rhs) });
        def = builder.create<OrIOp>({ lhs, rhs });
      } else if (isCmp) {
        if (isFloat) {
          if (inst.symbol == "==") def = builder.create<EqFOp>({ lhs, rhs });
          else if (inst.symbol == "!=") def = builder.create<NeFOp>({ lhs, rhs });
          else if (inst.symbol == "<") def = builder.create<LtFOp>({ lhs, rhs });
          else if (inst.symbol == "<=") def = builder.create<LeFOp>({ lhs, rhs });
        } else {
          if (inst.symbol == "==") def = builder.create<EqOp>({ lhs, rhs });
          else if (inst.symbol == "!=") def = builder.create<NeOp>({ lhs, rhs });
          else if (inst.symbol == "<") def = builder.create<LtOp>({ lhs, rhs });
          else if (inst.symbol == "<=") def = builder.create<LeOp>({ lhs, rhs });
        }
      } else {
        if (isFloat) {
          if (inst.symbol == "+") def = builder.create<AddFOp>({ lhs, rhs });
          else if (inst.symbol == "-") def = builder.create<SubFOp>({ lhs, rhs });
          else if (inst.symbol == "*") def = builder.create<MulFOp>({ lhs, rhs });
          else if (inst.symbol == "/") def = builder.create<DivFOp>({ lhs, rhs });
          else if (inst.symbol == "%") def = builder.create<ModFOp>({ lhs, rhs });
        } else {
          if (inst.symbol == "+") def = builder.create<AddIOp>({ lhs, rhs });
          else if (inst.symbol == "-") def = builder.create<SubIOp>({ lhs, rhs });
          else if (inst.symbol == "*") def = builder.create<MulIOp>({ lhs, rhs });
          else if (inst.symbol == "/") def = builder.create<DivIOp>({ lhs, rhs });
          else if (inst.symbol == "%") def = builder.create<ModIOp>({ lhs, rhs });
        }
      }
    }

    if (!def) {
      addError("unsupported arith/cmp op '" + inst.symbol + "' in @" + st.cfgFunc->name);
      def = builder.create<IntOp>({ new IntAttr(0) });
    }

    if (!inst.result.empty())
      st.values[inst.result] = def;
    return true;
  }

  bool lowerLoad(FunctionState &st, int bid, const Inst &inst) {
    builder.setToBlockEnd(st.blocks[bid]);

    auto sym = lookupSymbolInfo(st, inst.symbol, inst.type);
    auto base = symbolBaseAddress(st, bid, inst.symbol, inst.type);
    std::vector<std::string> indices = inst.args;

    Value out;
    bool pointerResult = inst.producesAddress || inst.type == hir::TypeKind::Pointer || inst.type == hir::TypeKind::Array;
    if (pointerResult && isArrayLike(sym)) {
      if (indices.empty())
        out = base;
      else
        out = buildIndexedAddress(st, bid, sym, base, indices, &inst.strideBytes);
    } else {
      bool indexed = !indices.empty();
      const bool linearizedScalarIndex =
        indexed && indices.size() == 1 && sym.dims.size() > 1 &&
        inst.type == sym.elementType &&
        (inst.memSize == 0 || inst.memSize == (sym.elemSize ? sym.elemSize : defaultSize(sym.elementType)));
      auto addr = buildIndexedAddress(st, bid, sym, base, indices, &inst.strideBytes, linearizedScalarIndex);
      auto ty = indexed ? toValueType(sym.elementType) : toValueType(inst.type);
      if (ty == Value::i128)
        ty = Value::i32;
      size_t sz = loadStoreSize(inst, sym, indexed);
      if (!sz)
        sz = 4;
      out = builder.create<LoadOp>(ty, { addr }, { new SizeAttr(sz) });
    }

    if (!inst.result.empty())
      st.values[inst.result] = out;
    return true;
  }

  bool lowerStore(FunctionState &st, int bid, const Inst &inst) {
    if (inst.args.empty()) {
      addError("store with empty operands in @" + st.cfgFunc->name);
      return true;
    }

    builder.setToBlockEnd(st.blocks[bid]);

    auto sym = lookupSymbolInfo(st, inst.symbol, inst.type);
    std::vector<std::string> indices;
    indices.reserve(inst.args.size() > 0 ? inst.args.size() - 1 : 0);
    for (size_t i = 0; i + 1 < inst.args.size(); i++)
      indices.push_back(inst.args[i]);

    auto value = resolveToken(st, bid, inst.args.back(), sym.elementType, false);

    auto base = symbolBaseAddress(st, bid, inst.symbol, inst.type);
    bool indexed = !indices.empty();
    const bool linearizedScalarIndex =
      indexed && indices.size() == 1 && sym.dims.size() > 1 &&
      (inst.memSize == 0 || inst.memSize == (sym.elemSize ? sym.elemSize : defaultSize(sym.elementType)));
    auto addr = indexed ? buildIndexedAddress(st, bid, sym, base, indices, &inst.strideBytes, linearizedScalarIndex) : base;

    size_t sz = loadStoreSize(inst, sym, indexed);
    if (!sz)
      sz = defaultSize(sym.elementType);
    if (!sz)
      sz = 4;
    builder.create<StoreOp>({ value, addr }, { new SizeAttr(sz) });
    return true;
  }

  bool lowerCall(FunctionState &st, int bid, const Inst &inst) {
    builder.setToBlockEnd(st.blocks[bid]);

    std::vector<Value> callArgs;
    callArgs.reserve(inst.args.size());
    for (size_t i = 0; i < inst.args.size(); i++) {
      hir::TypeKind expected = i < inst.calleeArgTypes.size() ? inst.calleeArgTypes[i] : hir::TypeKind::Int;
      callArgs.push_back(resolveToken(st, bid, inst.args[i], expected, false));
    }

    hir::TypeKind retKind = inst.calleeRetType == hir::TypeKind::Unknown ? inst.type : inst.calleeRetType;
    Value::Type retTy = toValueType(retKind);
    if (retTy == Value::i128)
      retTy = Value::i32;

    auto *call = builder.create<CallOp>(retTy, callArgs, {
      new NameAttr(normalizeCallee(inst.symbol))
    });
    // Dialect frontend does not run the legacy pureness inference stage.
    // Be conservative: treat calls as impure to avoid invalid CSE/hoisting
    // across side-effect boundaries (e.g. getint/putint and user calls).
    call->add<ImpureAttr>();

    if (!inst.result.empty())
      st.values[inst.result] = call;
    return true;
  }

  bool lowerRet(FunctionState &st, int bid, const Inst &inst) {
    builder.setToBlockEnd(st.blocks[bid]);
    if (st.hasReturnValue && !inst.args.empty()) {
      auto value = resolveToken(st, bid, inst.args[0], st.cfgFunc->returnType, false);
      builder.create<StoreOp>({ value, st.retSlot }, { new SizeAttr(st.retSize) });
    }
    builder.create<GotoOp>({ new TargetAttr(st.exitBlock) });
    return true;
  }

  bool lowerBr(FunctionState &st, int bid, const Inst &inst) {
    if (inst.targets.size() != 1) {
      addError("br requires one target in @" + st.cfgFunc->name);
      return true;
    }
    int tgt = inst.targets[0];
    if (tgt < 0 || tgt >= st.cfgBlockCount) {
      addError("br target out of range in @" + st.cfgFunc->name);
      return true;
    }
    builder.setToBlockEnd(st.blocks[bid]);
    builder.create<GotoOp>({ new TargetAttr(st.blocks[tgt]) });
    return true;
  }

  bool lowerCondBr(FunctionState &st, int bid, const Inst &inst) {
    if (inst.targets.size() != 2) {
      addError("cond_br requires two targets in @" + st.cfgFunc->name);
      return true;
    }
    if (inst.args.empty()) {
      addError("cond_br missing condition in @" + st.cfgFunc->name);
      return true;
    }

    int t = inst.targets[0], f = inst.targets[1];
    if (t < 0 || t >= st.cfgBlockCount || f < 0 || f >= st.cfgBlockCount) {
      addError("cond_br target out of range in @" + st.cfgFunc->name);
      return true;
    }

    auto cond = resolveToken(st, bid, inst.args[0], hir::TypeKind::Int, false);
    cond = forceIntValue(st, bid, cond);

    builder.setToBlockEnd(st.blocks[bid]);
    builder.create<BranchOp>({ cond }, {
      new TargetAttr(st.blocks[t]),
      new ElseAttr(st.blocks[f]),
    });
    return true;
  }

  bool lowerFunctionBody(FunctionState &st) {
    if (!st.cfgFunc)
      return false;

    for (int bid = 0; bid < (int) st.cfgFunc->blocks.size(); bid++) {
      const auto &cfgBB = st.cfgFunc->blocks[bid];
      for (const auto &inst : cfgBB.insts) {
        if (!lowerInst(st, bid, inst))
          return false;
      }
    }

    return true;
  }

  void emitFunctionExit(FunctionState &st) {
    builder.setToBlockEnd(st.exitBlock);
    if (st.hasReturnValue) {
      auto retTy = toValueType(st.cfgFunc->returnType);
      if (retTy == Value::i128)
        retTy = Value::i32;
      auto ret = builder.create<LoadOp>(retTy, { st.retSlot }, { new SizeAttr(st.retSize) });
      builder.create<ReturnOp>({ ret });
      return;
    }
    builder.create<ReturnOp>();
  }

  Value ensureEdgeSymbolLoad(FunctionState &st, int predBid, const std::string &symbol, hir::TypeKind typeHint) {
    auto *pred = st.blocks[predBid];
    auto &cache = st.edgeLoads[pred];
    auto it = cache.find(symbol);
    if (it != cache.end())
      return it->second;

    SymbolInfo sym = lookupSymbolInfo(st, symbol, typeHint);

    if (pred->getOpCount() == 0) {
      addError("phi predecessor has empty block in @" + st.cfgFunc->name);
      return materializeImmediate(st, predBid, "#0", typeHint);
    }

    auto *term = pred->getLastOp();
    builder.setBeforeOp(term);

    auto base = symbolBaseAddress(st, predBid, symbol, typeHint);
    auto loadTy = toValueType(sym.type == hir::TypeKind::Unknown ? typeHint : sym.type);
    if (loadTy == Value::i128)
      loadTy = Value::i32;
    size_t sz = sym.storageSize ? sym.storageSize : defaultSize(sym.type);
    if (!sz)
      sz = 4;

    auto val = (Value) builder.create<LoadOp>(loadTy, { base }, { new SizeAttr(sz) });
    cache[symbol] = val;
    return val;
  }

  bool finalizePhis(FunctionState &st) {
    for (const auto &pend : st.pendingPhi) {
      const Inst &inst = *pend.inst;
      if (!pend.placeholder) {
        addError("internal null phi placeholder in @" + st.cfgFunc->name);
        continue;
      }
      if (inst.args.size() != inst.phiPreds.size()) {
        addError("phi args/preds mismatch in @" + st.cfgFunc->name);
        continue;
      }

      std::vector<Value> incoming;
      std::vector<Attr*> from;
      incoming.reserve(inst.args.size());
      from.reserve(inst.args.size());

      for (size_t i = 0; i < inst.args.size(); i++) {
        int pred = inst.phiPreds[i];
        if (pred < 0 || pred >= st.cfgBlockCount) {
          addError("phi predecessor out of range in @" + st.cfgFunc->name);
          continue;
        }

        const std::string &token = inst.args[i];
        Value value;
        if (!token.empty() && token[0] == '$') {
          auto at = token.find('@');
          std::string symbol = (at == std::string::npos) ? token.substr(1) : token.substr(1, at - 1);
          value = ensureEdgeSymbolLoad(st, pred, symbol, inst.type);
        } else if (!token.empty() && (token[0] == '#' || token.rfind("f#", 0) == 0)) {
          // Phi incoming constants must be inserted before predecessor
          // terminators; placing them at block end creates unreachable defs.
          value = materializeImmediateBeforeTerminator(st, pred, token, inst.type);
        } else {
          value = resolveToken(st, pred, token, inst.type, true);
          if (!value.defining)
            value = materializeImmediateBeforeTerminator(st, pred, "#0", inst.type);
        }

        incoming.push_back(value);
        from.push_back(new FromAttr(st.blocks[pred]));
      }

      builder.setBeforeOp(pend.placeholder);
      auto *phi = builder.create<PhiOp>(incoming, from);
      pend.placeholder->replaceAllUsesWith(phi);
      pend.placeholder->erase();

      if (!inst.result.empty())
        st.values[inst.result] = phi;
    }

    Op::release();
    return true;
  }
};

}  // namespace

bool verifyCFGToLegacyLegality(const Module &cfgModule, ModuleOp *legacyModule, std::vector<std::string> &errors) {
  bool ok = true;

  std::vector<std::string> local;
  if (!verifyCFGLegalSet(cfgModule, local))
    ok = false;
  errors.insert(errors.end(), local.begin(), local.end());

  if (!legacyModule) {
    errors.push_back("cfg->legacy legality: legacy module is null");
    ok = false;
  }

  for (const auto &func : cfgModule.funcs) {
    for (const auto &bb : func.blocks) {
      for (const auto &inst : bb.insts) {
        if ((inst.kind == OpKind::Load || inst.kind == OpKind::Store) && inst.memSize == 0) {
          errors.push_back("cfg->legacy legality: load/store without mem size in @" + func.name);
          ok = false;
        }
        if (inst.kind == OpKind::Phi && inst.args.size() != inst.phiPreds.size()) {
          errors.push_back("cfg->legacy legality: phi incoming mismatch in @" + func.name);
          ok = false;
        }
        if ((inst.kind == OpKind::Br && inst.targets.size() != 1) ||
            (inst.kind == OpKind::CondBr && inst.targets.size() != 2)) {
          errors.push_back("cfg->legacy legality: malformed branch in @" + func.name);
          ok = false;
        }
      }
    }
  }

  return ok;
}

std::unique_ptr<ModuleOp> lowerToLegacyIR(const Module &cfgModule, std::vector<std::string> &errors) {
  LegacyLowerer lowering(cfgModule, errors);
  auto module = lowering.run();
  if (!module)
    return nullptr;

  if (!verifyCFGToLegacyLegality(cfgModule, module.get(), errors))
    return nullptr;
  return module;
}

}  // namespace sys::cfg
