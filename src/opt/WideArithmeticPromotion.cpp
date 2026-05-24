#include "Passes.h"
#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

bool provenNonNegativeImpl(Op *op, std::set<Op*> &visiting);

bool selfNonNegativeIncrement(Op *op, Op *phi, std::set<Op*> &visiting) {
  auto add = dyn_cast<AddIOp>(op);
  if (!add || add->getOperandCount() != 2)
    return false;
  auto lhs = add->DEF(0);
  auto rhs = add->DEF(1);
  if (lhs == phi && provenNonNegativeImpl(rhs, visiting))
    return true;
  if (rhs == phi && provenNonNegativeImpl(lhs, visiting))
    return true;
  return false;
}

bool provenNonNegativePhi(PhiOp *phi, std::set<Op*> &visiting) {
  if (!phi || phi->getOperandCount() == 0)
    return false;
  bool hasSelfStep = false;
  for (auto operand : phi->getOperands()) {
    auto incoming = operand.defining;
    if (incoming == phi)
      return false;
    if (selfNonNegativeIncrement(incoming, phi, visiting)) {
      hasSelfStep = true;
      continue;
    }
    if (!provenNonNegativeImpl(incoming, visiting))
      return false;
  }
  return hasSelfStep || phi->has<RangeAttr>();
}

bool provenNonNegativeImpl(Op *op, std::set<Op*> &visiting) {
  if (!op)
    return false;
  if (isa<IntOp>(op))
    return V(op) >= 0;
  if (!op->has<RangeAttr>())
    {
      if (visiting.count(op))
        return false;
      visiting.insert(op);
      bool result = false;
      if (auto phi = dyn_cast<PhiOp>(op))
        result = provenNonNegativePhi(phi, visiting);
      else if (auto add = dyn_cast<AddIOp>(op))
        result = provenNonNegativeImpl(add->DEF(0), visiting) &&
                 provenNonNegativeImpl(add->DEF(1), visiting);
      else if (auto mod = dyn_cast<ModIOp>(op))
        result = isa<IntOp>(mod->DEF(1)) && V(mod->DEF(1)) > 0 &&
                 provenNonNegativeImpl(mod->DEF(0), visiting);
      visiting.erase(op);
      return result;
    }
  auto [low, high] = RANGE(op);
  (void)high;
  if (low >= 0)
    return true;
  if (visiting.count(op))
    return false;
  visiting.insert(op);
  bool result = false;
  if (auto phi = dyn_cast<PhiOp>(op))
    result = provenNonNegativePhi(phi, visiting);
  visiting.erase(op);
  return result;
}

bool provenNonNegative(Op *op) {
  std::set<Op*> visiting;
  return provenNonNegativeImpl(op, visiting);
}

} // namespace

std::map<std::string, int> WideArithmeticPromotion::stats() {
  return {
    { "candidates", candidates },
    { "promoted", promoted },
    { "rejected-range", rejectedRange },
    { "rejected-shape", rejectedShape },
  };
}

void WideArithmeticPromotion::run() {
  if (!envEnabled("SISY_ENABLE_WIDE_ARITH_PROMOTION", true))
    return;

  Builder builder;
  runRewriter([&](ModIOp *op) {
    if (op->getOperandCount() != 2 || !isa<IntOp>(op->DEF(1))) {
      rejectedShape++;
      return false;
    }
    int mod = V(op->DEF(1));
    if (mod <= 1) {
      rejectedShape++;
      return false;
    }

    auto mul = dyn_cast<MulIOp>(op->DEF(0));
    if (!mul || mul->getOperandCount() != 2) {
      rejectedShape++;
      return false;
    }
    candidates++;

    auto lhs = mul->DEF(0);
    auto rhs = mul->DEF(1);
    if (!provenNonNegative(lhs) || !provenNonNegative(rhs)) {
      rejectedRange++;
      return false;
    }

    builder.setBeforeOp(op);
    auto product = builder.create<MulLOp>(std::vector<Value>{ lhs, rhs });
    auto modulus = builder.create<IntOp>({ new IntAttr(mod) });
    auto reduced = builder.create<ModLOp>(std::vector<Value>{ product, modulus });
    op->replaceAllUsesWith(reduced);
    op->erase();
    promoted++;
    return true;
  });
}
