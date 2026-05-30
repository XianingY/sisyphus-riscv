#include "LoopPasses.h"
#include "../utils/Matcher.h"

#include <cstdlib>
#include <cstring>
#include <set>

using namespace sys;

static bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

static void postorder(LoopInfo *loop, std::vector<LoopInfo*> &loops) {
  for (auto subloop : loop->subloops)
    loops.push_back(subloop);
  loops.push_back(loop);
}

std::map<std::string, int> LoopRotate::stats() {
  return {
    { "candidates", candidates },
    { "rotated-loops", rotated },
    { "reject-shape", rejectShape },
    { "reject-phi", rejectPhi },
    { "reject-iv", rejectIV },
    { "guard-split", guardSplit }
  };
}

static bool inferInductionForRotate(LoopInfo *info) {
  if (!info || info->getInduction() || info->latches.size() != 1 ||
      !info->header || !info->preheader)
    return info && info->getInduction();

  auto latch = info->getLatch();
  auto phis = info->header->getPhis();
  Rule br("(br (lt i x))"), brle("(br (le i x))");
  Rule addi("(add x y)");

  for (auto phi : phis) {
    auto start = Op::getPhiFrom(phi, info->preheader);
    auto latchVal = Op::getPhiFrom(phi, latch);
    if (!start || !latchVal || !addi.match(latchVal, { { "x", phi } }))
      continue;
    auto step = addi.extract("y");
    if (!isa<IntOp>(step) || V(step) <= 0)
      continue;

    auto term = info->header->getLastOp();
    Op *stop = nullptr;
    if (br.match(term, { { "i", phi } }))
      stop = br.extract("x");
    else if (brle.match(term, { { "i", phi } }))
      stop = brle.extract("x");
    if (!stop)
      continue;

    info->induction = phi;
    info->start = start;
    info->stop = stop;
    info->step = step;
    return true;
  }
  return false;
}

static BasicBlock *effectiveUseBlock(Op *value, Op *use) {
  if (!isa<PhiOp>(use))
    return use->getParent();
  for (size_t i = 0; i < use->getOperands().size(); i++) {
    if (use->getOperand(i).defining == value)
      return cast<FromAttr>(use->getAttrs()[i])->bb;
  }
  return use->getParent();
}

void LoopRotate::runImpl(LoopInfo *info) {
  if (!allowCanonicalizedHeaders && skipHeaders.count(info->header))
    return;
  candidates++;
  if (info->latches.size() != 1 || info->exits.size() != 1 ||
      info->exitings.size() != 1) {
    rejectShape++;
    return;
  }
  if (!inferInductionForRotate(info)) {
    rejectIV++;
    return;
  }
  bool hasStore = false;
  bool hasCall = false;
  for (auto bb : info->getBlocks()) {
    for (auto op : bb->getOps()) {
      hasStore |= isa<StoreOp>(op);
      hasCall |= isa<CallOp>(op);
    }
  }
  if (hasStore) {
    if (!envEnabled("SISY_ENABLE_LOOP_ROTATE_STORES", false)) {
      rejectShape++;
      return;
    }
    if (hasCall) {
      rejectShape++;
      return;
    }
    if (info->getBlocks().size() > 2) {
      rejectShape++;
      return;
    }
    if (!info->getExit()->getPhis().empty()) {
      rejectPhi++;
      return;
    }
    if (auto func = dyn_cast<FuncOp>(info->header->getParent()->getParent())) {
      for (auto call : func->findAll<CallOp>()) {
        if (!isExtern(NAME(call))) {
          rejectShape++;
          return;
        }
      }
    }
  }

  auto exit = info->getExit();

  auto induction = info->getInduction();
  auto header = info->header;
  // We only rotate canonicalized loop in form `for (int i = %0; i < x; i += 'b)`
  // Here `x` is an SSA value defined outside loop.
  // Check header condition for this.
  auto term = header->getLastOp();
  Rule br("(br (lt i x))"), brle("(br (le i x))");
  bool le = false;
  if (!br.match(term, { { "i", induction } })) {
    if (!brle.match(term, { { "i", induction } })) {
      rejectShape++;
      return;
    }
    le = true;
  }
  if (ELSE(term) != exit || !info->contains(TARGET(term)) ||
      *info->exitings.begin() != header) {
    rejectShape++;
    return;
  }

  auto latch = info->getLatch();
  if (latch == header) {
    rejectShape++;
    return;
  }
  auto latchterm = latch->getLastOp();
  if (!isa<GotoOp>(latchterm) || !latchterm->has<TargetAttr>() ||
      TARGET(latchterm) != header) {
    rejectShape++;
    return;
  }

  // Now replace the preheader's condition with (%0 < %1).
  Builder builder;

  auto preheader = info->preheader;
  auto preterm = preheader->getLastOp();
  if (!isa<GotoOp>(preterm) || !preterm->has<TargetAttr>() ||
      TARGET(preterm) != header) {
    rejectShape++;
    return;
  }
  
  auto upper = le ? brle.extract("x") : br.extract("x");
  auto upperFrom = upper->getParent();
  if (!upperFrom->dominates(header) || upperFrom == header) {
    if (!isa<IntOp>(upper)) {
      rejectIV++;
      return;
    }

    // We can hoist that constant out of loop.
    upper->moveBefore(preterm);
  }

  auto headerPhis = header->getPhis();
  std::map<Op*, Op*> valueMap, initMap;
  // Map each phi to the value from latch.
  // When we update references outside the loop, we'll change phi to that value instead.
  for (auto phi : headerPhis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    
    if (attrs.size() != 2 || ops.size() != 2) {
      rejectPhi++;
      return;
    }
    
    auto bb1 = cast<FromAttr>(attrs[0])->bb;
    if (bb1 == latch) {
      valueMap[phi] = ops[0].defining;
      initMap[phi] = ops[1].defining;
    }

    auto bb2 = cast<FromAttr>(attrs[1])->bb;
    if (bb2 == latch){
      valueMap[phi] = ops[1].defining;
      initMap[phi] = ops[0].defining;
    }

    if (!valueMap.count(phi) ||
        !((bb1 == latch && bb2 == preheader) ||
          (bb2 == latch && bb1 == preheader))) {
      rejectPhi++;
      return;
    }
  }
  if (!valueMap.count(induction)) {
    rejectIV++;
    return;
  }

  std::set<Op*> exitPhiSet;
  // Be conservative: only rotate when exit phis fed from header are direct
  // header-phi values that we can rewrite consistently.
  auto exitPhis = exit->getPhis();
  for (auto phi : exitPhis)
    exitPhiSet.insert(phi);
  for (auto phi : exitPhis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    bool hasHeaderIncoming = false;
    for (size_t i = 0; i < ops.size(); i++) {
      if (cast<FromAttr>(attrs[i])->bb != header)
        continue;
      hasHeaderIncoming = true;
      auto def = ops[i].defining;
      if (!valueMap.count(def) || !initMap.count(def)) {
        rejectPhi++;
        return;
      }
    }
    if (!hasHeaderIncoming) {
      rejectPhi++;
      return;
    }
  }

  std::map<Op*, std::vector<Op*>> directOutsideUses;
  for (auto phi : headerPhis) {
    std::vector<Op*> uses(phi->getUses().begin(), phi->getUses().end());
    for (auto use : uses) {
      if (exitPhiSet.count(use))
        continue;
      auto useBlock = effectiveUseBlock(phi, use);
      if (info->contains(useBlock))
        continue;
      // Phi edge uses outside the loop require edge-specific repair.  Keep the
      // default store-rotation path focused on ordinary LCSSA direct users.
      if (isa<PhiOp>(use)) {
        rejectPhi++;
        return;
      }
      if (!exit->dominates(use->getParent())) {
        rejectPhi++;
        return;
      }
      directOutsideUses[phi].push_back(use);
    }
  }
  if (!directOutsideUses.empty()) {
    for (auto pred : exit->preds) {
      if (pred != header) {
        rejectPhi++;
        return;
      }
    }
  }

  // Split the old preheader into a zero-trip guard plus a single-edge
  // loop preheader.  This keeps LoopAnalysis/LICM in LoopSimplify-style form:
  // guard -> prebody -> header, and guard -> exit.
  auto region = header->getParent();
  auto prebody = region->insert(header);
  builder.setToBlockEnd(prebody);
  builder.create<GotoOp>({ new TargetAttr(header) });

  for (auto phi : headerPhis) {
    for (auto attr : phi->getAttrs()) {
      if (cast<FromAttr>(attr)->bb == preheader)
        cast<FromAttr>(attr)->bb = prebody;
    }
  }

  builder.setBeforeOp(preterm);
  Value cmp;
  if (le)
    cmp = builder.create<LeOp>({ (Value) info->getStart(), upper });
  else
    cmp = builder.create<LtOp>({ (Value) info->getStart(), upper });
  builder.replace<BranchOp>(preterm, { cmp }, { new TargetAttr(prebody), new ElseAttr(exit) });

  // Replace the branch at header with a goto.
  auto target = TARGET(term);
  builder.replace<GotoOp>(term, { new TargetAttr(target) });

  // Replace the latch's terminator with a branch.
  // This time we should compare the increased induction variable with the upper bound,
  // i.e. the value from latch.
  builder.setBeforeOp(latchterm);
  if (le)
    cmp = builder.create<LeOp>({ valueMap[induction], upper->getResult() });
  else
    cmp = builder.create<LtOp>({ valueMap[induction], upper->getResult() });
  builder.replace<BranchOp>(latchterm, { cmp }, { new TargetAttr(header), new ElseAttr(exit) });

  // Fix phi nodes at exit.
  auto phis = exit->getPhis();
  for (auto phi : phis) {
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    for (size_t i = 0; i < ops.size(); i++) {
      auto &from = cast<FromAttr>(attrs[i])->bb;
      auto def = ops[i].defining;
      if (from == header) {
        from = latch;
        if (valueMap.count(def))
          phi->setOperand(i, valueMap[def]);
        
        phi->pushOperand(initMap.count(def) ? initMap[def] : def);
        phi->add<FromAttr>(preheader);
        break;
      }
    }
  }

  // Header phis may be used directly after the loop.  After rotation the exit
  // has a zero-trip edge from the guard, so those uses need an explicit final
  // value phi: latch value for executed loops, initial value for zero-trip.
  for (auto &[headerPhi, uses] : directOutsideUses) {
    builder.setToBlockStart(exit);
    auto finalPhi = builder.create<PhiOp>();
    finalPhi->pushOperand(valueMap[headerPhi]);
    finalPhi->add<FromAttr>(latch);
    finalPhi->pushOperand(initMap[headerPhi]);
    finalPhi->add<FromAttr>(preheader);
    for (auto use : uses) {
      for (size_t i = 0; i < use->getOperands().size(); i++) {
        if (use->getOperand(i).defining == headerPhi)
          use->setOperand(i, finalPhi);
      }
    }
  }

  guardSplit++;
  rotated++;
}

void LoopRotate::run() {
  Builder builder;
  LoopAnalysis loop(module);
  loop.run();
  auto info = loop.getResult();
  skipHeaders.clear();

  auto funcs = collectFuncs();
  for (auto func : funcs) {
    const auto &forest = info[func];
    for (auto top : forest.getLoops()) {
      std::vector<LoopInfo*> loops;
      postorder(top, loops);
      for (auto l : loops) {
        if (l->latches.size() != 1 || l->exits.size() != 1)
          skipHeaders.insert(l->header);
      }
    }
  }

  // Make sure each loop have a single latch.
  // Similar to Canonicalize::run().
  for (auto func : funcs) {
    LoopForest forest = info[func];

    for (auto loop : forest.getLoops()) {
      auto header = loop->header;
      if (loop->latches.size() == 1)
        continue;

      const auto &latches = loop->latches;
      auto region = header->getParent();
      auto latch = region->insert(*--latches.end());

      // Reconnect old latches to the new latch.
      for (auto old : latches) {
        auto term = old->getLastOp();
        if (term->has<TargetAttr>() && TARGET(term) == header)
          TARGET(term) = latch;
        if (term->has<ElseAttr>() && ELSE(term) == header)
          ELSE(term) = latch;
      }

      // Rewire backedge phi's at header to latch.
      auto phis = header->getPhis();
      for (auto phi : phis) {
        std::vector<std::pair<Op*, BasicBlock*>> forwarded, preserved;
        for (size_t i = 0; i < phi->getOperands().size(); i++) {
          auto from = cast<FromAttr>(phi->getAttrs()[i])->bb;
          if (latches.count(from))
            forwarded.push_back({ phi->getOperand(i).defining, from });
          else
            preserved.push_back({ phi->getOperand(i).defining, from });
        }

        // These form a new phi at the latch.
        if (forwarded.size()) {
          builder.setToBlockEnd(latch);
          Op *newPhi = builder.create<PhiOp>();
          for (auto [def, from] : forwarded) {
            newPhi->pushOperand(def);
            newPhi->add<FromAttr>(from);
          }

          // Remove all forwarded operands, and push a { newPhi, latch } pair.
          phi->removeAllOperands();
          phi->removeAllAttributes();

          if (!preserved.size()) {
            phi->replaceAllUsesWith(newPhi);
            phi->erase();
          } else {
            for (auto [def, from] : preserved) {
              phi->pushOperand(def);
              phi->add<FromAttr>(from);
            }
            phi->pushOperand(newPhi);
            phi->add<FromAttr>(latch);
          }
        }
      }

      // Wire latch to the header.
      builder.setToBlockEnd(latch);
      builder.create<GotoOp>({ new TargetAttr(header) });
    }
  }

  loop.run();
  info = loop.getResult();
  for (auto func : funcs) {
    const auto &forest = info[func];
    for (auto toploop : forest.getLoops()) {
      std::vector<LoopInfo*> loops;
      postorder(toploop, loops);

      for (auto loop : loops)
        runImpl(loop);
    }
  }
}
