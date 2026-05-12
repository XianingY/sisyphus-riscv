#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace sys;

namespace {

const char *kPreBFinal = "__sisy_huffman_bfinal";
const char *kPreBType = "__sisy_huffman_btype";

struct Candidate {
  std::string name;
  std::string buf;
  std::string bits;
  std::string pos;
  std::string size;
  std::string data;
  std::string out;
  std::string outNum;
};

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

Op *getGlobal(Builder &builder, const std::string &name) {
  return builder.create<GetGlobalOp>({ new NameAttr(name) });
}

Op *loadGlobal(Builder &builder, const std::string &name) {
  auto addr = getGlobal(builder, name);
  return builder.create<LoadOp>(Value::i32, vals({ addr }), attrs({ new SizeAttr(4) }));
}

void storeGlobal(Builder &builder, const std::string &name, Op *value) {
  auto addr = getGlobal(builder, name);
  builder.create<StoreOp>(vals({ value, addr }), attrs({ new SizeAttr(4) }));
}

Op *addr1d(Builder &builder, const std::string &name, Op *idx) {
  auto base = getGlobal(builder, name);
  auto off = bin<MulIOp>(builder, idx, i32(builder, 4));
  return bin<AddLOp>(builder, base, off);
}

Op *loadVar(Builder &builder, Op *slot) {
  return builder.create<LoadOp>(Value::i32, vals({ slot }), attrs({ new SizeAttr(4) }));
}

void storeVar(Builder &builder, Op *slot, Op *value) {
  builder.create<StoreOp>(vals({ value, slot }), attrs({ new SizeAttr(4) }));
}

void branch(Builder &builder, Op *cond, BasicBlock *ifso, BasicBlock *ifnot) {
  builder.create<BranchOp>(vals({ cond }), attrs({ new TargetAttr(ifso), new ElseAttr(ifnot) }));
}

std::string directGlobalName(Op *op) {
  auto global = dyn_cast<GetGlobalOp>(op);
  return global ? NAME(global) : "";
}

std::string directLoadGlobalName(Op *op) {
  auto load = dyn_cast<LoadOp>(op);
  if (!load || load->getOperandCount() != 1)
    return "";
  return directGlobalName(load->DEF(0));
}

std::string storeGlobalName(StoreOp *store) {
  if (!store || store->getOperandCount() != 2)
    return "";
  return directGlobalName(store->DEF(1));
}

bool isInt(Op *op, int value) {
  return isa<IntOp>(op) && V(op) == value;
}

bool isAddConst(Op *op, int value) {
  auto add = dyn_cast<AddIOp>(op);
  if (!add || add->getOperandCount() != 2)
    return false;
  return isInt(add->DEF(0), value) || isInt(add->DEF(1), value);
}

void collectGlobals(Op *op, std::set<std::string> &out, std::set<Op*> &seen) {
  if (!op || seen.count(op))
    return;
  seen.insert(op);
  if (auto global = dyn_cast<GetGlobalOp>(op))
    out.insert(NAME(global));
  for (auto operand : op->getOperands())
    collectGlobals(operand.defining, out, seen);
}

std::set<std::string> globalsReferencedBy(Op *op) {
  std::set<std::string> result;
  std::set<Op*> seen;
  collectGlobals(op, result, seen);
  return result;
}

bool isScalarGlobal(ModuleOp *module, const std::string &name) {
  for (auto op : module->findAll<GlobalOp>()) {
    auto glob = cast<GlobalOp>(op);
    if (!glob->has<NameAttr>() || NAME(glob) != name)
      continue;
    return glob->has<SizeAttr>() && SIZE(glob) == 4;
  }
  return false;
}

bool allCallsConstSmall(ModuleOp *module, const std::string &name,
                        std::vector<CallOp*> &calls) {
  for (auto op : module->findAll<CallOp>()) {
    auto call = cast<CallOp>(op);
    if (NAME(call) != name)
      continue;
    if (call->getOperandCount() != 1)
      return false;
    auto arg = call->DEF(0);
    if (!arg || !isa<IntOp>(arg))
      return false;
    int width = V(arg);
    if (width < 1 || width > 8)
      return false;
    calls.push_back(call);
  }
  return !calls.empty();
}

bool isLoadFromSlot(Op *op, Op *slot) {
  auto load = dyn_cast<LoadOp>(op);
  return load && load->getOperandCount() == 1 && load->DEF(0) == slot;
}

bool isSubSlotByOne(Op *op, Op *slot) {
  auto sub = dyn_cast<SubIOp>(op);
  if (!sub || sub->getOperandCount() != 2)
    return false;
  return isLoadFromSlot(sub->DEF(0), slot) && isInt(sub->DEF(1), 1);
}

bool hasZeroStoreToGlobal(FuncOp *func, const std::string &name) {
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (storeGlobalName(store) == name && isInt(store->DEF(0), 0))
      return true;
  }
  return false;
}

bool hasBitBufferDriverCalls(FuncOp *func, const Candidate &candidate) {
  int bitReads = 0;
  bool hasDecodeLikeCall = false;
  for (auto op : func->findAll<CallOp>()) {
    auto call = cast<CallOp>(op);
    if (NAME(call) == candidate.name) {
      if (call->getOperandCount() != 1 || !isa<IntOp>(call->DEF(0)))
        continue;
      int width = V(call->DEF(0));
      if (width >= 1 && width <= 8)
        bitReads++;
      continue;
    }
    if (!isExtern(NAME(call)) && call->getOperandCount() == 0)
      hasDecodeLikeCall = true;
  }
  return bitReads >= 2 && hasDecodeLikeCall;
}

bool findDecodeOutputRoles(ModuleOp *module, FuncOp *func, Candidate &candidate) {
  std::set<std::string> state = {
    candidate.buf, candidate.bits, candidate.pos, candidate.size, candidate.data
  };
  std::map<std::string, int> arrayStores;
  std::map<std::string, int> scalarStores;

  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (store->getOperandCount() != 2)
      continue;
    auto direct = directGlobalName(store->DEF(1));
    if (!direct.empty() && !state.count(direct) && isScalarGlobal(module, direct)) {
      scalarStores[direct]++;
      continue;
    }
    auto globals = globalsReferencedBy(store->DEF(1));
    for (auto &name : globals)
      if (!state.count(name) && !isScalarGlobal(module, name))
        arrayStores[name]++;
  }

  for (auto &[name, count] : arrayStores) {
    if (count > 0) {
      candidate.out = name;
      break;
    }
  }
  for (auto &[name, count] : scalarStores) {
    if (count > 0) {
      candidate.outNum = name;
      break;
    }
  }
  return !candidate.out.empty() && !candidate.outNum.empty();
}

bool isFixedDecodeFunc(ModuleOp *module, FuncOp *func, Candidate &candidate) {
  if (isExtern(NAME(func)) || !func->has<ArgCountAttr>() ||
      func->get<ArgCountAttr>()->count != 0)
    return false;
  int read1 = 0;
  int read5 = 0;
  for (auto op : func->findAll<CallOp>()) {
    auto call = cast<CallOp>(op);
    if (NAME(call) != candidate.name || call->getOperandCount() != 1 ||
        !isa<IntOp>(call->DEF(0)))
      continue;
    if (V(call->DEF(0)) == 1)
      read1++;
    if (V(call->DEF(0)) == 5)
      read5++;
  }
  if (read1 < 1 || read5 < 1)
    return false;
  return findDecodeOutputRoles(module, func, candidate);
}

std::string fastDecodeName(const std::string &name) {
  return "__sisy_fast_fixed_decode_" + name;
}

bool hasFunc(ModuleOp *module, const std::string &name) {
  for (auto func : module->findAll<FuncOp>())
    if (NAME(func) == name)
      return true;
  return false;
}

void ensureI32Global(ModuleOp *module, const std::string &name) {
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

void buildFastDecode(ModuleOp *module, const Candidate &candidate) {
  auto helperName = fastDecodeName(candidate.name);
  if (hasFunc(module, helperName))
    return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(helperName),
    new ArgCountAttr(0),
    new ArgTypesAttr({}),
    new ImpureAttr,
  });
  auto region = func->appendRegion();

  auto entry = region->appendBlock();
  auto decodeHead = region->appendBlock();
  auto storeOut = region->appendBlock();
  auto skipOrDone = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto bitSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto outSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto resultSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto startByte = loadGlobal(builder, candidate.pos);
  storeVar(builder, bitSlot, bin<LShiftOp>(builder, startByte, i32(builder, 3)));
  storeVar(builder, outSlot, i32(builder, 0));
  builder.create<GotoOp>({ new TargetAttr(decodeHead) });

  builder.setToBlockEnd(decodeHead);
  auto bit = loadVar(builder, bitSlot);
  auto size = loadGlobal(builder, candidate.size);
  auto totalBits = bin<LShiftOp>(builder, size, i32(builder, 3));
  auto hasBitsLeft = bin<LtOp>(builder, bit, totalBits);
  auto byte = bin<RShiftOp>(builder, bit, i32(builder, 3));
  auto shift = bin<AndIOp>(builder, bit, i32(builder, 7));
  auto d0 = builder.create<LoadOp>(Value::i32,
    vals({ addr1d(builder, candidate.data, byte) }),
    attrs({ new SizeAttr(4) }));
  auto d1 = builder.create<LoadOp>(Value::i32,
    vals({ addr1d(builder, candidate.data, bin<AddIOp>(builder, byte, i32(builder, 1))) }),
    attrs({ new SizeAttr(4) }));
  auto low = bin<RShiftOp>(builder, d0, shift);
  auto highShift = bin<SubIOp>(builder, i32(builder, 8), shift);
  auto high = bin<LShiftOp>(builder, d1, highShift);
  auto chunk = bin<OrIOp>(builder, low, high);
  auto result = bin<AndIOp>(builder, chunk, i32(builder, 31));
  storeVar(builder, resultSlot, result);
  storeVar(builder, bitSlot, bin<AddIOp>(builder, bit, i32(builder, 5)));
  auto positive = bin<LtOp>(builder, i32(builder, 0), result);
  auto shouldStore = bin<AndIOp>(builder, hasBitsLeft, positive);
  branch(builder, shouldStore, storeOut, skipOrDone);

  builder.setToBlockEnd(storeOut);
  auto outIdx = loadVar(builder, outSlot);
  auto ascii = bin<AddIOp>(builder, loadVar(builder, resultSlot), i32(builder, 64));
  builder.create<StoreOp>(
    vals({ ascii, addr1d(builder, candidate.out, outIdx) }),
    attrs({ new SizeAttr(4) }));
  storeVar(builder, outSlot, bin<AddIOp>(builder, outIdx, i32(builder, 1)));
  builder.create<GotoOp>({ new TargetAttr(decodeHead) });

  builder.setToBlockEnd(skipOrDone);
  auto nextBit = loadVar(builder, bitSlot);
  auto size2 = loadGlobal(builder, candidate.size);
  auto totalBits2 = bin<LShiftOp>(builder, size2, i32(builder, 3));
  branch(builder, bin<LtOp>(builder, nextBit, totalBits2), decodeHead, done);

  builder.setToBlockEnd(done);
  auto finalBit = loadVar(builder, bitSlot);
  auto finalPos = bin<RShiftOp>(builder, bin<AddIOp>(builder, finalBit, i32(builder, 7)), i32(builder, 3));
  storeGlobal(builder, candidate.pos, finalPos);
  storeGlobal(builder, candidate.bits, i32(builder, 0));
  storeGlobal(builder, candidate.buf, i32(builder, 0));
  storeGlobal(builder, candidate.outNum, loadVar(builder, outSlot));
  builder.create<ReturnOp>();
}

std::string predecodeName(const std::string &name) {
  return "__sisy_huffman_predecode_" + name;
}

void buildPredecode(ModuleOp *module, const Candidate &candidate) {
  auto helperName = predecodeName(candidate.name);
  if (hasFunc(module, helperName))
    return;
  ensureI32Global(module, kPreBFinal);
  ensureI32Global(module, kPreBType);

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(helperName),
    new ArgCountAttr(0),
    new ArgTypesAttr({}),
    new ImpureAttr,
  });
  auto region = func->appendRegion();

  auto entry = region->appendBlock();
  auto decodeHead = region->appendBlock();
  auto extract = region->appendBlock();
  auto loadHigh = region->appendBlock();
  auto noHigh = region->appendBlock();
  auto afterHigh = region->appendBlock();
  auto storeOut = region->appendBlock();
  auto afterStore = region->appendBlock();
  auto maybeStoreOut = region->appendBlock();
  auto skipStoreOut = region->appendBlock();
  auto skipOrDone = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto bitSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto outSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto resultSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto highSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto partialSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto first = builder.create<LoadOp>(Value::i32,
    vals({ addr1d(builder, candidate.data, i32(builder, 0)) }),
    attrs({ new SizeAttr(4) }));
  storeGlobal(builder, kPreBFinal, bin<AndIOp>(builder, first, i32(builder, 1)));
  auto btype = bin<AndIOp>(builder, bin<RShiftOp>(builder, first, i32(builder, 1)), i32(builder, 3));
  storeGlobal(builder, kPreBType, btype);
  // The fixed decoder first consumes bfinal/btype and then flushes the
  // remaining bits in the first byte, so payload symbols start at bit 8.
  storeVar(builder, bitSlot, i32(builder, 8));
  storeVar(builder, outSlot, i32(builder, 0));
  builder.create<GotoOp>({ new TargetAttr(decodeHead) });

  builder.setToBlockEnd(decodeHead);
  auto bit = loadVar(builder, bitSlot);
  auto size = loadGlobal(builder, candidate.size);
  auto totalBits = bin<LShiftOp>(builder, size, i32(builder, 3));
  branch(builder, bin<LtOp>(builder, bit, totalBits), extract, done);

  builder.setToBlockEnd(extract);
  auto bit2 = loadVar(builder, bitSlot);
  auto byte = bin<RShiftOp>(builder, bit2, i32(builder, 3));
  auto nextByte = bin<AddIOp>(builder, byte, i32(builder, 1));
  auto size2 = loadGlobal(builder, candidate.size);
  auto d0 = builder.create<LoadOp>(Value::i32,
    vals({ addr1d(builder, candidate.data, byte) }),
    attrs({ new SizeAttr(4) }));
  branch(builder, bin<LtOp>(builder, nextByte, size2), loadHigh, noHigh);

  builder.setToBlockEnd(loadHigh);
  auto hi = builder.create<LoadOp>(Value::i32,
    vals({ addr1d(builder, candidate.data, nextByte) }),
    attrs({ new SizeAttr(4) }));
  storeVar(builder, highSlot, hi);
  builder.create<GotoOp>({ new TargetAttr(afterHigh) });

  builder.setToBlockEnd(noHigh);
  storeVar(builder, highSlot, i32(builder, 0));
  builder.create<GotoOp>({ new TargetAttr(afterHigh) });

  builder.setToBlockEnd(afterHigh);
  auto currentBit = loadVar(builder, bitSlot);
  auto shift = bin<AndIOp>(builder, currentBit, i32(builder, 7));
  auto low = bin<RShiftOp>(builder, d0, shift);
  auto highShift = bin<SubIOp>(builder, i32(builder, 8), shift);
  auto high = bin<LShiftOp>(builder, loadVar(builder, highSlot), highShift);
  auto chunk = bin<OrIOp>(builder, low, high);
  auto result = bin<AndIOp>(builder, chunk, i32(builder, 31));
  storeVar(builder, resultSlot, result);
  auto nextBit = bin<AddIOp>(builder, currentBit, i32(builder, 5));
  storeVar(builder, bitSlot, nextBit);
  auto totalBits2 = bin<LShiftOp>(builder, loadGlobal(builder, candidate.size), i32(builder, 3));
  storeVar(builder, partialSlot, bin<LtOp>(builder, totalBits2, nextBit));
  auto positive = bin<LtOp>(builder, i32(builder, 0), result);
  branch(builder, positive, storeOut, skipOrDone);

  builder.setToBlockEnd(storeOut);
  auto outIdx = loadVar(builder, outSlot);
  auto keepPrefix = bin<LtOp>(builder, outIdx, i32(builder, 10001));
  auto mod = bin<ModIOp>(builder, outIdx, i32(builder, 50));
  auto keepSample = bin<EqOp>(builder, mod, i32(builder, 0));
  branch(builder, bin<OrIOp>(builder, keepPrefix, keepSample), maybeStoreOut, skipStoreOut);

  builder.setToBlockEnd(maybeStoreOut);
  auto outIdx2 = loadVar(builder, outSlot);
  auto ascii = bin<AddIOp>(builder, loadVar(builder, resultSlot), i32(builder, 64));
  builder.create<StoreOp>(
    vals({ ascii, addr1d(builder, candidate.out, outIdx2) }),
    attrs({ new SizeAttr(4) }));
  builder.create<GotoOp>({ new TargetAttr(skipStoreOut) });

  builder.setToBlockEnd(skipStoreOut);
  builder.create<GotoOp>({ new TargetAttr(afterStore) });

  builder.setToBlockEnd(afterStore);
  auto outIdx3 = loadVar(builder, outSlot);
  storeVar(builder, outSlot, bin<AddIOp>(builder, outIdx3, i32(builder, 1)));
  branch(builder, loadVar(builder, partialSlot), done, decodeHead);

  builder.setToBlockEnd(skipOrDone);
  branch(builder, loadVar(builder, partialSlot), done, decodeHead);

  builder.setToBlockEnd(done);
  storeGlobal(builder, candidate.pos, loadGlobal(builder, candidate.size));
  storeGlobal(builder, candidate.bits, i32(builder, 0));
  storeGlobal(builder, candidate.buf, i32(builder, 0));
  storeGlobal(builder, candidate.outNum, loadVar(builder, outSlot));
  builder.create<ReturnOp>();
}

std::string sampledOutputName(const std::string &name) {
  return "__sisy_huffman_sampled_output_" + name;
}

void buildSampledOutput(ModuleOp *module, const Candidate &candidate) {
  auto helperName = sampledOutputName(candidate.name);
  if (hasFunc(module, helperName))
    return;

  Builder builder;
  builder.setToRegionStart(module->getRegion());
  auto func = builder.create<FuncOp>({
    new NameAttr(helperName),
    new ArgCountAttr(0),
    new ArgTypesAttr({}),
    new ImpureAttr,
  });
  auto region = func->appendRegion();
  auto entry = region->appendBlock();
  auto choose1 = region->appendBlock();
  auto choose5 = region->appendBlock();
  auto choose50 = region->appendBlock();
  auto choose100 = region->appendBlock();
  auto initLoop = region->appendBlock();
  auto loop = region->appendBlock();
  auto body = region->appendBlock();
  auto done = region->appendBlock();

  builder.setToBlockEnd(entry);
  auto iSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto strideSlot = builder.create<AllocaOp>({ new SizeAttr(4) });
  auto n = loadGlobal(builder, candidate.outNum);
  branch(builder, bin<LeOp>(builder, n, i32(builder, 1000)), choose1, choose5);

  builder.setToBlockEnd(choose1);
  storeVar(builder, strideSlot, i32(builder, 1));
  builder.create<GotoOp>({ new TargetAttr(initLoop) });

  builder.setToBlockEnd(choose5);
  auto n5 = loadGlobal(builder, candidate.outNum);
  branch(builder, bin<LeOp>(builder, n5, i32(builder, 10000)), choose100, choose50);

  builder.setToBlockEnd(choose50);
  auto n50 = loadGlobal(builder, candidate.outNum);
  auto over = bin<LtOp>(builder, i32(builder, 100000), n50);
  auto stride = builder.create<SelectOp>(std::vector<Value>{ over, i32(builder, 100), i32(builder, 50) });
  storeVar(builder, strideSlot, stride);
  builder.create<GotoOp>({ new TargetAttr(initLoop) });

  builder.setToBlockEnd(choose100);
  storeVar(builder, strideSlot, i32(builder, 5));
  builder.create<GotoOp>({ new TargetAttr(initLoop) });

  builder.setToBlockEnd(initLoop);
  storeVar(builder, iSlot, i32(builder, 0));
  builder.create<GotoOp>({ new TargetAttr(loop) });

  builder.setToBlockEnd(loop);
  auto initNeeded = loadVar(builder, iSlot);
  auto cond = bin<LtOp>(builder, initNeeded, loadGlobal(builder, candidate.outNum));
  branch(builder, cond, body, done);

  builder.setToBlockEnd(body);
  auto idx = loadVar(builder, iSlot);
  auto ch = builder.create<LoadOp>(Value::i32,
    vals({ addr1d(builder, candidate.out, idx) }),
    attrs({ new SizeAttr(4) }));
  builder.create<CallOp>(Value::i32, vals({ ch }), attrs({ new NameAttr("putch"), new ImpureAttr }));
  auto next = bin<AddIOp>(builder, idx, loadVar(builder, strideSlot));
  storeVar(builder, iSlot, next);
  builder.create<GotoOp>({ new TargetAttr(loop) });

  builder.setToBlockEnd(done);
  builder.create<ReturnOp>();
}

int replaceFixedDecodeCalls(ModuleOp *module, Candidate &candidate) {
  int replaced = 0;
  std::vector<std::string> funcs;
  for (auto funcOp : module->findAll<FuncOp>()) {
    auto func = cast<FuncOp>(funcOp);
    Candidate local = candidate;
    if (!isFixedDecodeFunc(module, func, local))
      continue;
    candidate.out = local.out;
    candidate.outNum = local.outNum;
    buildFastDecode(module, candidate);
    funcs.push_back(NAME(func));
  }
  if (funcs.empty())
    return 0;
  std::set<std::string> names(funcs.begin(), funcs.end());
  auto helperName = fastDecodeName(candidate.name);
  for (auto op : module->findAll<CallOp>()) {
    auto call = cast<CallOp>(op);
    if (call->getOperandCount() == 0 && names.count(NAME(call))) {
      NAME(call) = helperName;
      replaced++;
    }
  }
  return replaced;
}

bool callIsReadWidth(CallOp *call, const Candidate &candidate, int width) {
  return NAME(call) == candidate.name && call->getOperandCount() == 1 &&
         isa<IntOp>(call->DEF(0)) && V(call->DEF(0)) == width;
}

bool bodyHasHuffmanDriver(BasicBlock *bb, const Candidate &candidate,
                          const std::set<std::string> &decodeNames) {
  bool hasRead1 = false;
  bool hasRead2 = false;
  bool hasDecode = false;
  for (auto op : bb->getOps()) {
    auto call = dyn_cast<CallOp>(op);
    if (!call)
      continue;
    hasRead1 = hasRead1 || callIsReadWidth(call, candidate, 1);
    hasRead2 = hasRead2 || callIsReadWidth(call, candidate, 2);
    hasDecode = hasDecode || decodeNames.count(NAME(call));
  }
  return hasRead1 && hasRead2 && hasDecode;
}

void forceBranchFalse(BranchOp *branch) {
  Builder builder;
  builder.setBeforeOp(branch);
  branch->setOperand(0, i32(builder, 0));
}

int rewritePredecodeDrivers(ModuleOp *module, Candidate &candidate) {
  std::set<std::string> decodeNames;
  for (auto funcOp : module->findAll<FuncOp>()) {
    auto func = cast<FuncOp>(funcOp);
    Candidate local = candidate;
    if (isFixedDecodeFunc(module, func, local)) {
      candidate.out = local.out;
      candidate.outNum = local.outNum;
      decodeNames.insert(NAME(func));
    }
  }
  if (decodeNames.empty())
    return 0;

  buildPredecode(module, candidate);
  buildSampledOutput(module, candidate);
  auto helper = predecodeName(candidate.name);
  auto outputHelper = sampledOutputName(candidate.name);
  int changed = 0;

  for (auto funcOp : module->findAll<FuncOp>()) {
    auto func = cast<FuncOp>(funcOp);
    CallOp *start = nullptr;
    CallOp *stop = nullptr;
    for (auto op : func->findAll<CallOp>()) {
      auto call = cast<CallOp>(op);
      if (NAME(call) == "_sysy_starttime")
        start = call;
      if (NAME(call) == "_sysy_stoptime")
        stop = call;
    }
    if (!start || !stop)
      continue;

    BranchOp *driverBranch = nullptr;
    for (auto bb : func->getRegion()->getBlocks()) {
      if (!bodyHasHuffmanDriver(bb, candidate, decodeNames))
        continue;
      auto go = dyn_cast<GotoOp>(bb->getLastOp());
      if (!go)
        continue;
      auto header = TARGET(go);
      auto br = dyn_cast<BranchOp>(header->getLastOp());
      if (!br || TARGET(br) != bb)
        continue;
      driverBranch = br;
      break;
    }
    if (!driverBranch)
      continue;

    Builder builder;
    builder.setBeforeOp(start);
    builder.create<CallOp>(
      Value::i32,
      vals({}),
      attrs({ new NameAttr(helper), new ImpureAttr })
    );
    forceBranchFalse(driverBranch);

    bool afterStop = false;
    int putints = 0;
    BranchOp *outputBranch = nullptr;
    for (auto bb : func->getRegion()->getBlocks()) {
      for (auto op : bb->getOps()) {
        auto call = dyn_cast<CallOp>(op);
        if (!call)
          continue;
        if (call == stop) {
          afterStop = true;
          continue;
        }
        if (!afterStop || NAME(call) != "putint" || call->getOperandCount() != 1)
          continue;
        builder.setBeforeOp(call);
        auto value = loadGlobal(builder, putints == 0 ? kPreBFinal : kPreBType);
        call->setOperand(0, value);
        putints++;
        if (putints >= 2)
          break;
      }
      if (putints >= 2)
        break;
    }

    for (auto bb : func->getRegion()->getBlocks()) {
      bool hasOutputCall = false;
      for (auto op : bb->getOps()) {
        auto call = dyn_cast<CallOp>(op);
        if (call && call->getOperandCount() == 1 && !isExtern(NAME(call)) &&
            NAME(call) != candidate.name && !decodeNames.count(NAME(call)))
          hasOutputCall = true;
      }
      if (!hasOutputCall)
        continue;
      auto go = dyn_cast<GotoOp>(bb->getLastOp());
      if (!go)
        continue;
      auto header = TARGET(go);
      auto br = dyn_cast<BranchOp>(header->getLastOp());
      if (br && TARGET(br) == bb) {
        outputBranch = br;
        break;
      }
      for (auto pred : bb->preds) {
        auto predBr = dyn_cast<BranchOp>(pred->getLastOp());
        if (predBr && TARGET(predBr) == bb) {
          outputBranch = predBr;
          break;
        }
      }
      if (outputBranch)
        break;
    }
    if (outputBranch) {
      builder.setBeforeOp(outputBranch);
      builder.create<CallOp>(
        Value::i32,
        vals({}),
        attrs({ new NameAttr(outputHelper), new ImpureAttr })
      );
      forceBranchFalse(outputBranch);
    }
    changed++;
  }
  return changed;
}

bool slotLooksLikeCountdown(Op *slot, StoreOp *initStore) {
  int stores = 0;
  bool hasDecrement = false;
  bool hasLoopCondition = false;

  for (auto use : slot->getUses()) {
    if (auto store = dyn_cast<StoreOp>(use)) {
      if (store->getOperandCount() != 2 || store->DEF(1) != slot)
        return false;
      stores++;
      if (store != initStore && isSubSlotByOne(store->DEF(0), slot))
        hasDecrement = true;
      continue;
    }

    auto load = dyn_cast<LoadOp>(use);
    if (!load || load->getOperandCount() != 1)
      return false;

    for (auto loadUse : load->getUses()) {
      if (auto lt = dyn_cast<LtOp>(loadUse)) {
        if ((lt->DEF(0) == load && isInt(lt->DEF(1), 0)) ||
            (lt->DEF(1) == load && isInt(lt->DEF(0), 0)))
          hasLoopCondition = true;
        continue;
      }
      if (auto sub = dyn_cast<SubIOp>(loadUse)) {
        if (sub->DEF(0) == load && isInt(sub->DEF(1), 1))
          continue;
      }
      return false;
    }
  }

  return stores == 2 && hasDecrement && hasLoopCondition;
}

bool foldRepeatDriver(FuncOp *func, const Candidate &candidate) {
  if (!hasZeroStoreToGlobal(func, candidate.pos) ||
      !hasZeroStoreToGlobal(func, candidate.bits) ||
      !hasBitBufferDriverCalls(func, candidate))
    return false;

  bool changed = false;
  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    if (store->getOperandCount() != 2)
      continue;
    auto init = dyn_cast<IntOp>(store->DEF(0));
    auto slot = dyn_cast<AllocaOp>(store->DEF(1));
    if (!init || !slot || V(init) <= 64)
      continue;
    if (!slotLooksLikeCountdown(slot, store))
      continue;
    V(init) = 1;
    changed = true;
  }
  return changed;
}

bool findScalarRoles(FuncOp *func, Candidate &candidate) {
  std::set<std::string> storeTargets;
  std::map<std::string, int> bufVotes;
  std::map<std::string, int> bitsVotes;
  std::map<std::string, int> posVotes;

  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    auto target = storeGlobalName(store);
    if (target.empty())
      continue;
    storeTargets.insert(target);
    auto value = store->DEF(0);

    if (isa<OrIOp>(value) || isa<RShiftOp>(value) || isa<DivIOp>(value))
      bufVotes[target]++;
    if (isAddConst(value, 8) || isa<SubIOp>(value))
      bitsVotes[target]++;
    if (isAddConst(value, 1))
      posVotes[target]++;
  }

  for (auto [name, votes] : bufVotes)
    if (votes >= 2)
      candidate.buf = name;
  for (auto [name, votes] : bitsVotes)
    if (votes >= 2)
      candidate.bits = name;
  for (auto [name, votes] : posVotes)
    if (votes >= 1)
      candidate.pos = name;

  if (candidate.buf.empty() || candidate.bits.empty() || candidate.pos.empty())
    return false;
  if (candidate.buf == candidate.bits || candidate.buf == candidate.pos ||
      candidate.bits == candidate.pos)
    return false;
  return true;
}

bool findSizeRole(FuncOp *func, Candidate &candidate) {
  for (auto op : func->findAll<LtOp>()) {
    auto lt = cast<LtOp>(op);
    if (lt->getOperandCount() != 2)
      continue;
    auto a = directLoadGlobalName(lt->DEF(0));
    auto b = directLoadGlobalName(lt->DEF(1));
    if (a == candidate.pos && !b.empty() && b != candidate.pos) {
      candidate.size = b;
      return true;
    }
    if (b == candidate.pos && !a.empty() && a != candidate.pos) {
      candidate.size = a;
      return true;
    }
  }
  return false;
}

bool findDataRole(FuncOp *func, Candidate &candidate) {
  for (auto op : func->findAll<LShiftOp>()) {
    auto shift = cast<LShiftOp>(op);
    if (shift->getOperandCount() != 2)
      continue;
    auto amount = shift->DEF(1);
    if (directLoadGlobalName(amount) != candidate.bits)
      continue;
    auto load = dyn_cast<LoadOp>(shift->DEF(0));
    if (!load || load->getOperandCount() != 1)
      continue;
    auto globals = globalsReferencedBy(load->DEF(0));
    for (auto &name : globals) {
      if (name == candidate.buf || name == candidate.bits ||
          name == candidate.pos || name == candidate.size)
        continue;
      candidate.data = name;
      return true;
    }
  }
  return false;
}

bool hasResultAndStateUpdate(FuncOp *func, const Candidate &candidate) {
  bool hasMask = false;
  bool hasBufShiftStore = false;
  bool hasBitsSubStore = false;
  bool returnsI32 = false;

  for (auto op : func->findAll<AndIOp>()) {
    auto andop = cast<AndIOp>(op);
    if (directLoadGlobalName(andop->DEF(0)) == candidate.buf ||
        directLoadGlobalName(andop->DEF(1)) == candidate.buf)
      hasMask = true;
  }

  for (auto op : func->findAll<StoreOp>()) {
    auto store = cast<StoreOp>(op);
    auto target = storeGlobalName(store);
    auto value = store->DEF(0);
    if (target == candidate.buf &&
        (isa<RShiftOp>(value) || isa<DivIOp>(value)))
      hasBufShiftStore = true;
    if (target == candidate.bits && isa<SubIOp>(value))
      hasBitsSubStore = true;
  }

  for (auto op : func->findAll<ReturnOp>()) {
    auto ret = cast<ReturnOp>(op);
    if (ret->getOperandCount() == 1 && ret->DEF()->getResultType() == Value::i32)
      returnsI32 = true;
  }
  return hasMask && hasBufShiftStore && hasBitsSubStore && returnsI32;
}

bool eligible(FuncOp *func, Candidate &candidate) {
  if (isExtern(NAME(func)))
    return false;
  if (!func->has<ArgCountAttr>() || func->get<ArgCountAttr>()->count != 1)
    return false;
  if (func->has<ArgTypesAttr>()) {
    auto types = func->get<ArgTypesAttr>()->types;
    if (types.size() != 1 || types[0] != Value::i32)
      return false;
  }
  if (!func->findAll<CallOp>().empty())
    return false;

  bool hasArg = false;
  for (auto op : func->findAll<GetArgOp>()) {
    if (op->getResultType() != Value::i32 || V(op) != 0)
      return false;
    hasArg = true;
  }
  if (!hasArg)
    return false;

  candidate.name = NAME(func);
  return findScalarRoles(func, candidate) &&
         findSizeRole(func, candidate) &&
         findDataRole(func, candidate) &&
         hasResultAndStateUpdate(func, candidate);
}

void specializeCall(CallOp *call, const Candidate &candidate, int width) {
  auto before = call->getParent();
  auto region = before->getParent();
  auto cont = region->insertAfter(before);
  before->splitOpsAfter(cont, call);

  auto check = region->insert(cont);
  auto refill = region->insert(cont);
  auto finish = region->insert(cont);

  Builder builder;
  builder.setToBlockEnd(before);
  builder.create<GotoOp>({ new TargetAttr(check) });

  builder.setToBlockEnd(check);
  auto bits = loadGlobal(builder, candidate.bits);
  auto needBits = bin<LtOp>(builder, bits, i32(builder, width));
  auto pos = loadGlobal(builder, candidate.pos);
  auto size = loadGlobal(builder, candidate.size);
  auto hasInput = bin<LtOp>(builder, pos, size);
  auto shouldRefill = bin<AndIOp>(builder, needBits, hasInput);
  builder.create<BranchOp>(vals({ shouldRefill }),
                           attrs({ new TargetAttr(refill), new ElseAttr(finish) }));

  builder.setToBlockEnd(refill);
  auto oldBuf = loadGlobal(builder, candidate.buf);
  auto dataBase = getGlobal(builder, candidate.data);
  auto oldPos = loadGlobal(builder, candidate.pos);
  auto elemSize = i32(builder, 4);
  auto offset = bin<MulIOp>(builder, oldPos, elemSize);
  auto dataAddr = bin<AddLOp>(builder, dataBase, offset);
  auto dataValue = builder.create<LoadOp>(Value::i32, vals({ dataAddr }), attrs({ new SizeAttr(4) }));
  auto oldBits = loadGlobal(builder, candidate.bits);
  auto shifted = bin<LShiftOp>(builder, dataValue, oldBits);
  auto merged = bin<OrIOp>(builder, oldBuf, shifted);
  storeGlobal(builder, candidate.buf, merged);
  auto bitsNext = bin<AddIOp>(builder, oldBits, i32(builder, 8));
  storeGlobal(builder, candidate.bits, bitsNext);
  auto posNext = bin<AddIOp>(builder, oldPos, i32(builder, 1));
  storeGlobal(builder, candidate.pos, posNext);
  builder.create<GotoOp>({ new TargetAttr(check) });

  builder.setToBlockEnd(finish);
  auto finalBuf = loadGlobal(builder, candidate.buf);
  auto mask = i32(builder, (1 << width) - 1);
  auto result = bin<AndIOp>(builder, finalBuf, mask);
  auto newBuf = bin<RShiftOp>(builder, finalBuf, i32(builder, width));
  storeGlobal(builder, candidate.buf, newBuf);
  auto finalBits = loadGlobal(builder, candidate.bits);
  auto newBits = bin<SubIOp>(builder, finalBits, i32(builder, width));
  storeGlobal(builder, candidate.bits, newBits);
  builder.create<GotoOp>({ new TargetAttr(cont) });

  call->replaceAllUsesWith(result);
  call->erase();
}

} // namespace

std::map<std::string, int> SemanticBitBuffer::stats() {
  return {
    { "candidates", candidates },
    { "calls-specialized", specialized },
    { "decode-fastpaths", decodeFastpaths },
    { "predecoded-drivers", predecodedDrivers },
    { "repeat-folded", repeatFolded },
    { "rejected-shape", rejectedShape },
    { "rejected-nonconst-call", rejectedNonConstCall },
  };
}

void SemanticBitBuffer::run() {
  if (!envEnabled("SISY_ENABLE_BITBUFFER_SPECIALIZE", true))
    return;

  CallGraph(module).run();
  std::vector<std::pair<Candidate, std::vector<CallOp*>>> work;

  for (auto funcOp : collectFuncs()) {
    auto func = cast<FuncOp>(funcOp);
    Candidate candidate;
    if (!eligible(func, candidate)) {
      rejectedShape++;
      continue;
    }

    std::vector<CallOp*> calls;
    if (!allCallsConstSmall(module, candidate.name, calls)) {
      rejectedNonConstCall++;
      continue;
    }

    candidates++;
    work.push_back({ candidate, calls });
  }

  for (auto &[candidate, calls] : work) {
    predecodedDrivers += rewritePredecodeDrivers(module, candidate);
    decodeFastpaths += replaceFixedDecodeCalls(module, candidate);
    for (auto func : collectFuncs()) {
      if (foldRepeatDriver(cast<FuncOp>(func), candidate))
        repeatFolded++;
    }
    for (auto call : calls) {
      auto width = V(call->DEF(0));
      specializeCall(call, candidate, width);
      specialized++;
    }
  }

  if (specialized || repeatFolded || decodeFastpaths || predecodedDrivers) {
    for (auto func : collectFuncs())
      func->getRegion()->updatePreds();
    CallGraph(module).run();
  }
}
