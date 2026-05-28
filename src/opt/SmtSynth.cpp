#include "CleanupPasses.h"
#include "SmtProver.h"

#include <cstdlib>
#include <cstring>
#include <vector>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

bool isSynthRoot(Op *op) {
  return op && op->getResultType() == Value::i32 &&
         (isa<AddIOp>(op) || isa<SubIOp>(op) || isa<MulIOp>(op) ||
          isa<AndIOp>(op) || isa<OrIOp>(op) || isa<XorIOp>(op));
}

void eraseIfUnused(Op *op) {
  if (op && op->getUses().empty())
    op->erase();
}

}  // namespace

void SmtSynth::run() {
  if (!envEnabled("SISY_ENABLE_SMT_SYNTH", true))
    return;

  Builder builder;
  std::vector<Op*> ops;
  for (auto *func : collectFuncs()) {
    for (auto *region : func->getRegions())
      for (auto *bb : region->getBlocks())
        for (auto *op : bb->getOps())
          ops.push_back(op);
  }
  for (auto *op : ops) {
    if (!isSynthRoot(op) || op->getOperandCount() < 1)
      continue;

    considered++;
    std::vector<Op*> candidates;
    for (auto operand : op->getOperands()) {
      if (operand.defining && operand.defining->getResultType() == Value::i32)
        candidates.push_back(operand.defining);
    }

    builder.setBeforeOp(op);
    std::vector<Op*> scratch;
    scratch.push_back(builder.create<IntOp>({ new IntAttr(0) }));
    scratch.push_back(builder.create<IntOp>({ new IntAttr(1) }));
    scratch.push_back(builder.create<IntOp>({ new IntAttr(-1) }));
    candidates.insert(candidates.end(), scratch.begin(), scratch.end());

    Op *winner = nullptr;
    for (auto *candidate : candidates) {
      if (!candidate || candidate == op)
        continue;
      auto result = smt_prover::tryProveEqualI32(op, candidate);
      if (result == smt_prover::Result::Unknown) {
        unknown++;
        continue;
      }
      if (result == smt_prover::Result::Equal) {
        proved++;
        winner = candidate;
        break;
      }
    }

    if (winner) {
      op->replaceAllUsesWith(winner);
      op->erase();
      replaced++;
      for (auto *tmp : scratch)
        if (tmp != winner)
          eraseIfUnused(tmp);
    } else {
      for (auto *tmp : scratch)
        eraseIfUnused(tmp);
    }
  }
}
