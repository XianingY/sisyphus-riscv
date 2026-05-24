#include "LoopPasses.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}

bool sameValue(Op *a, Op *b) {
  return a && b && a == b;
}

bool isIntConst(Op *op, int value) {
  return op && isa<IntOp>(op) && V(op) == value;
}

bool intConst(Op *op, int &value) {
  if (!op || !isa<IntOp>(op))
    return false;
  value = V(op);
  return true;
}

Op *stripSinglePhi(Op *op) {
  std::set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1 && !seen.count(op)) {
    seen.insert(op);
    op = op->DEF(0);
  }
  return op;
}

bool sameStrippedValue(Op *a, Op *b) {
  return stripSinglePhi(a) == stripSinglePhi(b);
}

bool signedPow2RemainderUpper(Op *op, int &upper) {
  op = stripSinglePhi(op);
  auto *sub = dyn_cast<SubIOp>(op);
  if (!sub || sub->getOperandCount() != 2)
    return false;

  Op *x = stripSinglePhi(sub->DEF(0));
  auto *product = dyn_cast<LShiftOp>(stripSinglePhi(sub->DEF(1)));
  if (!x || !product || product->getOperandCount() != 2)
    return false;

  int shift = 0;
  if (!intConst(stripSinglePhi(product->DEF(1)), shift) || shift <= 0 || shift >= 31)
    return false;

  auto *quotient = dyn_cast<RShiftOp>(stripSinglePhi(product->DEF(0)));
  if (!quotient || quotient->getOperandCount() != 2 ||
      !isIntConst(stripSinglePhi(quotient->DEF(1)), shift))
    return false;

  auto *biased = dyn_cast<AddIOp>(stripSinglePhi(quotient->DEF(0)));
  if (!biased || biased->getOperandCount() != 2)
    return false;

  Op *bias = nullptr;
  if (sameStrippedValue(biased->DEF(0), x))
    bias = stripSinglePhi(biased->DEF(1));
  else if (sameStrippedValue(biased->DEF(1), x))
    bias = stripSinglePhi(biased->DEF(0));
  else
    return false;

  auto *andOp = dyn_cast<AndIOp>(bias);
  if (!andOp || andOp->getOperandCount() != 2)
    return false;

  Op *sign = nullptr;
  int mask = 0;
  if (intConst(stripSinglePhi(andOp->DEF(0)), mask))
    sign = stripSinglePhi(andOp->DEF(1));
  else if (intConst(stripSinglePhi(andOp->DEF(1)), mask))
    sign = stripSinglePhi(andOp->DEF(0));
  else
    return false;

  if (mask != ((1 << shift) - 1))
    return false;

  auto *signShift = dyn_cast<RShiftOp>(sign);
  if (!signShift || signShift->getOperandCount() != 2 ||
      !sameStrippedValue(signShift->DEF(0), x) ||
      !isIntConst(stripSinglePhi(signShift->DEF(1)), 31))
    return false;

  upper = mask;
  return true;
}

bool simpleIncrement(Op *op, Op *phi) {
  if (!op || !isa<AddIOp>(op) || op->getOperandCount() != 2)
    return false;
  return (op->DEF(0) == phi && isIntConst(op->DEF(1), 1)) ||
         (op->DEF(1) == phi && isIntConst(op->DEF(0), 1));
}

bool stopUpperBound(Op *stop, int &upper) {
  if (!stop)
    return false;
  if (isa<IntOp>(stop)) {
    upper = V(stop);
    return upper >= 0;
  }
  if (auto range = stop->find<RangeAttr>()) {
    upper = range->range.second;
    return upper >= 0 && upper < INT_MAX;
  }
  if (isa<ModIOp>(stop) && stop->getOperandCount() == 2 && isa<IntOp>(stop->DEF(1))) {
    int mod = V(stop->DEF(1));
    if (mod <= 0)
      return false;
    upper = mod - 1;
    return true;
  }
  if (signedPow2RemainderUpper(stop, upper))
    return true;
  return false;
}

bool headerHasOnlyLoopControl(BasicBlock *header, Op *induction, Op *cond, Op *branch) {
  for (auto op : header->getOps()) {
    if (isa<PhiOp>(op) || op == cond || op == branch)
      continue;
    return false;
  }
  for (auto phi : header->getPhis()) {
    if (phi != induction)
      return false;
  }
  return true;
}

bool findInduction(LoopInfo *loop, Op *&induction, Op *&stop, Op *&incr,
                   BasicBlock *&body, BasicBlock *&exit) {
  if (!loop || !loop->preheader || loop->latches.size() != 1 || loop->exits.size() != 1)
    return false;
  if (loop->getBlocks().size() != 2)
    return false;

  auto header = loop->header;
  auto branch = dyn_cast<BranchOp>(header->getLastOp());
  if (!branch || !branch->has<TargetAttr>() || !branch->has<ElseAttr>() ||
      branch->getOperandCount() != 1)
    return false;
  auto cond = branch->DEF(0);
  if (!cond || !isa<LtOp>(cond) || cond->getOperandCount() != 2)
    return false;

  induction = cond->DEF(0);
  stop = cond->DEF(1);
  if (!induction || !isa<PhiOp>(induction) || induction->getParent() != header ||
      induction->getResultType() != Value::i32)
    return false;

  body = TARGET(branch);
  exit = ELSE(branch);
  if (!loop->contains(body) || loop->contains(exit) || body == header)
    return false;
  if (body != loop->getLatch())
    return false;
  if (!headerHasOnlyLoopControl(header, induction, cond, branch))
    return false;

  auto gotoBack = dyn_cast<GotoOp>(body->getLastOp());
  if (!gotoBack || !gotoBack->has<TargetAttr>() || TARGET(gotoBack) != header)
    return false;

  bool sawStart = false;
  bool sawLatch = false;
  const auto &ops = induction->getOperands();
  const auto &attrs = induction->getAttrs();
  if (ops.size() != 2 || attrs.size() != 2)
    return false;
  for (int i = 0; i < 2; i++) {
    auto from = dyn_cast<FromAttr>(attrs[i]);
    if (!from)
      return false;
    auto def = ops[i].defining;
    if (from->bb == loop->preheader && isIntConst(stripSinglePhi(def), 0)) {
      sawStart = true;
    } else if (from->bb == body && simpleIncrement(def, induction)) {
      incr = def;
      sawLatch = true;
    }
  }
  return sawStart && sawLatch && incr;
}

struct ModUpdate {
  Op *addr = nullptr;
  int addConst = 0;
  int modulus = 0;
};

bool matchModUpdateBody(BasicBlock *body, Op *induction, Op *incr, ModUpdate &shape) {
  std::vector<Op*> ops;
  for (auto op : body->getOps())
    if (!isa<GotoOp>(op))
      ops.push_back(op);

  if (ops.size() != 9)
    return false;

  auto load = dyn_cast<LoadOp>(ops[0]);
  auto add = dyn_cast<AddIOp>(ops[2]);
  auto storeAdd = dyn_cast<StoreOp>(ops[3]);
  auto mod = dyn_cast<ModIOp>(ops[5]);
  auto storeMod = dyn_cast<StoreOp>(ops[6]);
  if (!load || !isa<IntOp>(ops[1]) || !add || !storeAdd || !isa<IntOp>(ops[4]) ||
      !mod || !storeMod || !isa<IntOp>(ops[7]) || ops[8] != incr)
    return false;

  auto addr = load->DEF(0);
  if (!addr || !sameValue(storeAdd->DEF(1), addr) || !sameValue(storeMod->DEF(1), addr))
    return false;

  Op *constant = nullptr;
  if (add->DEF(0) == load && isa<IntOp>(add->DEF(1)))
    constant = add->DEF(1);
  else if (add->DEF(1) == load && isa<IntOp>(add->DEF(0)))
    constant = add->DEF(0);
  else
    return false;

  if (storeAdd->DEF(0) != add)
    return false;
  if (mod->DEF(0) != add || !isa<IntOp>(mod->DEF(1)))
    return false;
  if (storeMod->DEF(0) != mod)
    return false;

  int c = V(constant);
  int m = V(mod->DEF(1));
  if (c == 0 || m <= 1)
    return false;

  for (auto op : ops) {
    if (op == load || op == add || op == storeAdd || op == mod || op == storeMod ||
        op == incr || isa<IntOp>(op))
      continue;
    return false;
  }

  // The induction variable may only control the increment in this block.
  for (auto use : induction->getUses())
    if (use->getParent() == body && use != incr)
      return false;

  shape.addr = addr;
  shape.addConst = c;
  shape.modulus = m;
  return true;
}

} // namespace

std::map<std::string, int> ModularAffineLoop::stats() {
  return {
    { "visited", visited },
    { "folded", folded },
    { "rejected", rejected },
  };
}

bool ModularAffineLoop::runImpl(LoopInfo *loop) {
  visited++;

  Op *induction = nullptr;
  Op *stop = nullptr;
  Op *incr = nullptr;
  BasicBlock *body = nullptr;
  BasicBlock *exit = nullptr;
  if (!findInduction(loop, induction, stop, incr, body, exit)) {
    rejected++;
    return false;
  }
  if (exit->getPhis().size()) {
    rejected++;
    return false;
  }

  ModUpdate update;
  if (!matchModUpdateBody(body, induction, incr, update)) {
    rejected++;
    return false;
  }

  int upper = 0;
  if (!stopUpperBound(stop, upper)) {
    rejected++;
    return false;
  }
  long long maxDelta = 1LL * update.addConst * upper;
  if (maxDelta > INT_MAX || maxDelta < INT_MIN) {
    rejected++;
    return false;
  }

  auto preterm = loop->preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || !preterm->has<TargetAttr>() || TARGET(preterm) != loop->header) {
    rejected++;
    return false;
  }

  Builder builder;
  builder.setBeforeOp(preterm);
  auto old = builder.create<LoadOp>(Value::i32, { update.addr }, { new SizeAttr(4) });
  auto zero = builder.create<IntOp>({ new IntAttr(0) });
  auto positive = builder.create<LtOp>({ zero, stop });
  auto trip = builder.create<SelectOp>(std::vector<Value>{ positive, stop, zero });
  auto step = builder.create<IntOp>({ new IntAttr(update.addConst) });
  auto delta = builder.create<MulIOp>(std::vector<Value>{ trip, step });
  auto sum = builder.create<AddIOp>(std::vector<Value>{ old, delta });
  auto mod = builder.create<IntOp>({ new IntAttr(update.modulus) });
  auto foldedValue = builder.create<ModIOp>(std::vector<Value>{ sum, mod });
  builder.create<StoreOp>({ foldedValue, update.addr }, { new SizeAttr(4) });
  builder.replace<GotoOp>(preterm, { new TargetAttr(exit) });

  // The original 2-block loop is now unreachable. Leaving it in place keeps a
  // self-contained dead SCC in the region, which later CFG-sensitive passes can
  // still walk and trip over. Remove it eagerly instead of relying on later DCE.
  auto detachBlock = [](BasicBlock *bb) {
    auto ops = bb->getOps();
    for (auto op : ops)
      op->removeAllOperands();
  };
  detachBlock(loop->header);
  detachBlock(body);
  body->preds.clear();
  body->succs.clear();
  loop->header->preds.clear();
  loop->header->succs.clear();
  body->forceErase();
  loop->header->forceErase();

  folded++;
  return true;
}

void ModularAffineLoop::run() {
  if (!envEnabled("SISY_ENABLE_MODULAR_AFFINE_LOOP", true))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  LoopAnalysis analysis(module);
  analysis.run();
  for (auto [func, forest] : analysis.getResult()) {
    (void) func;
    for (auto loop : forest.getLoops())
      runImpl(loop);
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}
