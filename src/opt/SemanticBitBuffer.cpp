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

struct Candidate {
  std::string name;
  std::string buf;
  std::string bits;
  std::string pos;
  std::string size;
  std::string data;
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

  if (specialized || repeatFolded) {
    for (auto func : collectFuncs())
      func->getRegion()->updatePreds();
    CallGraph(module).run();
  }
}
