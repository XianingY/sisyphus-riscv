#include "LoopPasses.h"

#include <tuple>
#include <unordered_map>

using namespace sys;

std::map<std::string, int> AddressRecurrence::stats() {
  return {
    { "phi-addresses", phiAddresses },
    { "rewritten-addresses", rewrittenAddresses },
  };
}

namespace {

struct I32Affine {
  int coeff = 0;
  int constant = 0;
  Op *invariant = nullptr;
  bool valid = false;
};

struct AddrAffine {
  Op *base = nullptr;
  Op *invariant = nullptr;
  int coeff = 0;
  int constant = 0;
  bool valid = false;
};

struct HeaderIv {
  Op *phi = nullptr;
  Value start;
  int step = 0;
  bool valid = false;
};

Op *peel(Op *op) {
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1)
    op = op->DEF();
  return op;
}

bool dependsOn(Op *op, Op *iv, std::set<Op*> &seen) {
  op = peel(op);
  iv = peel(iv);
  if (!op || !iv)
    return false;
  if (op == iv)
    return true;
  if (seen.count(op))
    return false;
  seen.insert(op);
  for (auto operand : op->getOperands())
    if (dependsOn(operand.defining, iv, seen))
      return true;
  return false;
}

bool flattenI32(Op *op, Op *iv, I32Affine &out) {
  op = peel(op);
  iv = peel(iv);
  if (!op || !iv)
    return false;
  std::set<Op*> seen;
  if (!dependsOn(op, iv, seen)) {
    if (auto *i = dyn_cast<IntOp>(op))
      out = { 0, V(i), nullptr, true };
    else
      out = { 0, 0, op, true };
    return true;
  }
  if (op == iv) {
    out = { 1, 0, nullptr, true };
    return true;
  }
  if (auto *i = dyn_cast<IntOp>(op)) {
    out = { 0, V(i), nullptr, true };
    return true;
  }
  if (auto *add = dyn_cast<AddIOp>(op)) {
    I32Affine lhs, rhs;
    if (!flattenI32(add->DEF(0), iv, lhs) || !flattenI32(add->DEF(1), iv, rhs))
      return false;
    if (lhs.invariant && rhs.invariant)
      return false;
    out = { lhs.coeff + rhs.coeff, lhs.constant + rhs.constant,
            lhs.invariant ? lhs.invariant : rhs.invariant, true };
    return true;
  }
  if (auto *sub = dyn_cast<SubIOp>(op)) {
    I32Affine lhs, rhs;
    if (!flattenI32(sub->DEF(0), iv, lhs) || !flattenI32(sub->DEF(1), iv, rhs))
      return false;
    if (rhs.invariant || (lhs.invariant && rhs.invariant))
      return false;
    out = { lhs.coeff - rhs.coeff, lhs.constant - rhs.constant, lhs.invariant, true };
    return true;
  }
  if (auto *mul = dyn_cast<MulIOp>(op)) {
    if (peel(mul->DEF(0)) == iv && isa<IntOp>(mul->DEF(1))) {
      out = { V(mul->DEF(1)), 0, nullptr, true };
      return true;
    }
    if (peel(mul->DEF(1)) == iv && isa<IntOp>(mul->DEF(0))) {
      out = { V(mul->DEF(0)), 0, nullptr, true };
      return true;
    }
    I32Affine lhs, rhs;
    if (flattenI32(mul->DEF(0), iv, lhs) && lhs.valid && lhs.coeff == 0 && !lhs.invariant &&
        flattenI32(mul->DEF(1), iv, rhs) && rhs.valid) {
      out = { rhs.coeff * lhs.constant, rhs.constant * lhs.constant, rhs.invariant, true };
      return true;
    }
    if (flattenI32(mul->DEF(1), iv, rhs) && rhs.valid && rhs.coeff == 0 && !rhs.invariant &&
        flattenI32(mul->DEF(0), iv, lhs) && lhs.valid) {
      out = { lhs.coeff * rhs.constant, lhs.constant * rhs.constant, lhs.invariant, true };
      return true;
    }
  }
  return false;
}

bool flattenAddr(Op *op, Op *iv, AddrAffine &out) {
  op = peel(op);
  iv = peel(iv);
  if (!op || !iv)
    return false;
  if (isa<GetGlobalOp>(op) || isa<AllocaOp>(op)) {
    out = { op, nullptr, 0, 0, true };
    return true;
  }
  if (auto *add = dyn_cast<AddLOp>(op)) {
    AddrAffine lhs;
    I32Affine rhs;
    if (flattenAddr(add->DEF(0), iv, lhs) && flattenI32(add->DEF(1), iv, rhs)) {
      if (lhs.invariant && rhs.invariant)
        return false;
      out = { lhs.base, lhs.invariant ? lhs.invariant : rhs.invariant,
              lhs.coeff + rhs.coeff, lhs.constant + rhs.constant, true };
      return true;
    }
    if (flattenAddr(add->DEF(1), iv, lhs) && flattenI32(add->DEF(0), iv, rhs)) {
      if (lhs.invariant && rhs.invariant)
        return false;
      out = { lhs.base, lhs.invariant ? lhs.invariant : rhs.invariant,
              lhs.coeff + rhs.coeff, lhs.constant + rhs.constant, true };
      return true;
    }
  }
  return false;
}

HeaderIv findHeaderIv(LoopInfo *loop) {
  HeaderIv result;
  if (!loop || !loop->preheader || loop->latches.size() != 1)
    return result;
  auto *latch = loop->getLatch();

  for (auto *phi : loop->header->getPhis()) {
    int preCount = 0;
    Value start;
    int step = 0;
    bool good = true;
    for (int i = 0; i < phi->getOperandCount(); i++) {
      auto *from = FROM(phi->getAttrs()[i]);
      if (from == loop->preheader) {
        preCount++;
        start = phi->getOperand(i);
        continue;
      }
      if (from != latch) {
        good = false;
        break;
      }
      auto *def = phi->DEF(i);
      auto *add = dyn_cast<AddIOp>(peel(def));
      if (!add) {
        good = false;
        break;
      }
      Op *other = nullptr;
      if (peel(add->DEF(0)) == peel(phi))
        other = add->DEF(1);
      else if (peel(add->DEF(1)) == peel(phi))
        other = add->DEF(0);
      else {
        good = false;
        break;
      }
      if (!isa<IntOp>(other) || V(other) <= 0) {
        good = false;
        break;
      }
      step = V(other);
    }
    if (good && preCount == 1 && step > 0)
      return { phi, start, step, true };
  }
  return result;
}

Op *cloneInvariantExpr(Builder &builder, LoopInfo *loop, Op *op,
                       std::unordered_map<Op*, Op*> &cache) {
  op = peel(op);
  if (!op)
    return nullptr;
  if (!op->getParent() || !loop->contains(op->getParent()))
    return op;
  if (cache.count(op))
    return cache[op];
  auto *copied = builder.copy(op);
  cache[op] = copied;
  auto operands = copied->getOperands();
  copied->removeAllOperands();
  for (auto operand : operands) {
    auto *def = operand.defining;
    auto *mapped = cloneInvariantExpr(builder, loop, def, cache);
    copied->pushOperand(mapped ? mapped->getResult() : operand);
  }
  return copied;
}

Value buildI32Offset(Builder &builder, LoopInfo *loop, Value ivValue, const I32Affine &affine) {
  Value offset;
  bool haveOffset = false;
  if (affine.coeff != 0) {
    offset = ivValue;
    if (affine.coeff != 1) {
      auto c = builder.create<IntOp>({ new IntAttr(affine.coeff) });
      offset = builder.create<MulIOp>(std::vector<Value>{ offset, c });
    }
    haveOffset = true;
  }
  if (affine.constant != 0 || !haveOffset) {
    auto c = builder.create<IntOp>({ new IntAttr(affine.constant) });
    offset = haveOffset ? Value(builder.create<AddIOp>(std::vector<Value>{ offset, c })) : Value(c);
    haveOffset = true;
  }
  if (affine.invariant) {
    std::unordered_map<Op*, Op*> cache;
    auto *inv = cloneInvariantExpr(builder, loop, affine.invariant, cache);
    offset = haveOffset ? Value(builder.create<AddIOp>(std::vector<Value>{ offset, inv })) : Value(inv);
  }
  return offset;
}

} // namespace

void AddressRecurrence::runImpl(LoopInfo *loop) {
  for (auto *sub : loop->subloops)
    runImpl(sub);

  if (!loop->preheader || loop->latches.size() != 1)
    return;
  auto ivInfo = findHeaderIv(loop);
  if (!ivInfo.valid)
    return;

  auto *iv = ivInfo.phi;
  auto *header = loop->header;
  auto *latch = loop->getLatch();

  using Key = std::tuple<Op*, Op*, int, int>;
  std::map<Key, std::vector<Op*>> groups;
  for (auto *bb : loop->getBlocks()) {
    for (auto *op : bb->getOps()) {
      Op *addr = nullptr;
      if (isa<LoadOp>(op))
        addr = op->DEF(0);
      else if (isa<StoreOp>(op))
        addr = op->DEF(1);
      else
        continue;
      if (!addr || !addr->getParent() || !loop->contains(addr->getParent()))
        continue;
      AddrAffine a;
      if (!flattenAddr(addr, iv, a) || !a.valid || !a.base || a.coeff == 0)
        continue;
      groups[{ a.base, a.invariant, a.coeff, a.constant }].push_back(addr);
    }
  }

  if (groups.empty())
    return;

  Builder builder;
  std::unordered_map<Op*, Op*> replacement;
  int step = ivInfo.step;

  for (auto &[key, addrs] : groups) {
    auto [base, invariant, coeff, constant] = key;
    builder.setBeforeOp(loop->preheader->getLastOp());
    I32Affine affine { coeff, constant, invariant, true };
    auto startOffset = buildI32Offset(builder, loop, ivInfo.start, affine);
    auto startAddr = builder.create<AddLOp>(std::vector<Value>{ base, startOffset });

    builder.setToBlockStart(header);
    auto phi = builder.create<PhiOp>(std::vector<Value>{ startAddr }, { new FromAttr(loop->preheader) });

    builder.setBeforeOp(latch->getLastOp());
    auto stride = builder.create<IntOp>({ new IntAttr(coeff * step) });
    auto nextAddr = builder.create<AddLOp>(std::vector<Value>{ phi, stride });
    phi->pushOperand(nextAddr);
    phi->add<FromAttr>(latch);

    for (auto *addr : addrs)
      replacement[addr] = phi;
    phiAddresses++;
  }

  for (auto &[addr, phi] : replacement) {
    addr->replaceAllUsesWith(phi);
    rewrittenAddresses++;
  }
}

void AddressRecurrence::run() {
  LoopAnalysis analysis(module);
  analysis.run();
  auto forests = analysis.getResult();

  for (auto *func : collectFuncs()) {
    auto &forest = forests[func];
    for (auto *loop : forest.getLoops()) {
      if (!loop->getParent())
        runImpl(loop);
    }
    func->getRegion()->updatePreds();
  }
}
