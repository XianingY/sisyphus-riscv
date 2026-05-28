#include "CleanupPasses.h"
#include "SmtProver.h"

#include <cstdlib>
#include <cstring>

using namespace sys;

namespace {

bool envEnabled(const char *n, bool d) {
  const char *r = std::getenv(n);
  if (!r || !r[0]) return d;
  return std::strcmp(r, "0") != 0 && std::strcmp(r, "false") != 0;
}

void removePhiIncomingFromLocal(BasicBlock *target, BasicBlock *from) {
  if (!target || !from) return;
  for (auto *phi : target->getPhis()) {
    auto ops = phi->getOperands();
    std::vector<Attr*> attrs;
    for (auto *attr : phi->getAttrs())
      attrs.push_back(attr->clone());

    phi->removeAllOperands();
    phi->removeAllAttributes();

    for (size_t i = 0; i < ops.size() && i < attrs.size(); i++) {
      if (isa<FromAttr>(attrs[i]) && FROM(attrs[i]) == from)
        continue;
      if (!isa<FromAttr>(attrs[i]))
        continue;
      phi->pushOperand(ops[i]);
      phi->add<FromAttr>(FROM(attrs[i]));
    }

    for (auto *attr : attrs)
      delete attr;
  }
}

}  // namespace

void SmtBranchPrune::run() {
  if (!envEnabled("SISY_ENABLE_SMT_PATH_PRUNE", false))
    return;

  Builder builder;
  for (auto func : collectFuncs()) {
    auto region = func->getRegion();
    region->updatePreds();
    for (auto bb : region->getBlocks()) {
      if (bb->getOpCount() == 0) continue;
      auto br = dyn_cast<BranchOp>(bb->getLastOp());
      if (!br) continue;
      considered++;

      Op *cond = br->DEF(0);
      // Skip if cond is already a constant — SCCP/DCE handle that.
      if (isa<IntOp>(cond)) continue;

      bool isNonZero = smt_prover::tryProveNonZeroI32(cond);
      bool isZero = !isNonZero && smt_prover::tryProveZeroI32(cond);
      if (!isNonZero && !isZero) {
        unsupported++;
        continue;
      }

      auto taken = isNonZero ? TARGET(br) : ELSE(br);
      auto dead = isNonZero ? ELSE(br) : TARGET(br);
      if (!taken || !dead) continue;
      if (taken == dead) continue;
      removePhiIncomingFromLocal(dead, bb);
      builder.replace<GotoOp>(br, { new TargetAttr(taken) });
      folded++;
    }
    region->updatePreds();
  }
}
