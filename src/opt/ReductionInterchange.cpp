#include "LoopPasses.h"
#include <cstdlib>
#include <cstring>
#include <set>
#include <map>

using namespace sys;

// ReductionInterchange Pass
//
// Transforms the canonical "outer reduction" pattern (such as matrix
// multiplication's i-j-k loop) by swapping the middle and innermost loops
// while distributing the reduction.
//
// Pattern (j-k where j has reduction):
//
//   for j:
//     int temp = 0;          // init in j body, before k
//     for k:
//       temp = temp + f(j, k);  // pure reduction in k
//     A[g(j)] = temp;        // store result, address depends on j (not k)
//
// Transforms to:
//
//   for j: A[g(j)] = 0;       // init loop
//   for k:
//     for j:
//       A[g(j)] = A[g(j)] + f(j, k);  // load-modify-store, j-k swapped
//
// This improves cache locality when f(j, k) accesses memory with k as outer
// stride and j as inner stride (e.g., B[k][j]).
//
// Phase 1 (this version): Pattern detection only — counts candidates without
// transforming. Used to verify the matcher works correctly before
// implementing the transformation.

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0]) return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

// Check if an op uses `target` (transitively, in-loop only)
bool usesValue(Op *op, Op *target, LoopInfo *loop, std::set<Op*> &seen) {
  if (!op || op == target) return op == target;
  if (seen.count(op)) return false;
  seen.insert(op);
  for (auto operand : op->getOperands()) {
    auto def = operand.defining;
    if (def == target) return true;
    if (def && def->getParent() && loop->contains(def->getParent())) {
      if (usesValue(def, target, loop, seen)) return true;
    }
  }
  return false;
}

bool usesValue(Op *op, Op *target, LoopInfo *loop) {
  std::set<Op*> seen;
  return usesValue(op, target, loop, seen);
}

// Strip single-input phis (degenerate, just forward their input)
Op *stripPhis(Op *op) {
  std::set<Op*> seen;
  while (op && isa<PhiOp>(op) && op->getOperandCount() == 1 && !seen.count(op)) {
    seen.insert(op);
    op = op->DEF(0);
  }
  return op;
}

// Find the reduction pattern in inner loop's header.
// Returns the reduction phi if found, nullptr otherwise.
//
// Pattern: phi at inner header where:
//   - outer-incoming value is IntOp(0)
//   - inner-incoming value is AddIOp(phi, X) where X is computed from
//     the inner IV and j (the outer IV)
Op *findReductionPhi(LoopInfo *inner, LoopInfo *outer) {
  if (!inner || !outer) return nullptr;
  auto innerHeader = inner->header;
  auto innerLatch = inner->getLatch();
  auto innerPreheader = inner->preheader;
  if (!innerHeader || !innerLatch || !innerPreheader) return nullptr;

  for (auto phi : innerHeader->getPhis()) {
    if (phi->getResultType() != Value::i32) continue;

    Op *initVal = nullptr;
    Op *latchVal = nullptr;
    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();
    for (int i = 0; i < (int) attrs.size(); i++) {
      auto from = dyn_cast<FromAttr>(attrs[i]);
      if (!from) continue;
      if (from->bb == innerPreheader) initVal = ops[i].defining;
      else if (from->bb == innerLatch) latchVal = ops[i].defining;
    }
    if (!initVal || !latchVal) continue;

    // initVal must be IntOp(0) (after stripping degenerate phis)
    auto initRaw = stripPhis(initVal);
    if (!isa<IntOp>(initRaw) || V(initRaw) != 0) continue;

    // Skip if this is the inner IV phi (latchVal would be add(phi, 1))
    if (isa<AddIOp>(latchVal) && latchVal->getOperandCount() == 2) {
      auto a = latchVal->DEF(0);
      auto b = latchVal->DEF(1);
      bool isUnitInc = (a == phi && isa<IntOp>(b) && V(b) == 1) ||
                       (b == phi && isa<IntOp>(a) && V(a) == 1);
      if (isUnitInc) continue; // this is the IV, skip
    }

    // latchVal must be (transitively) reachable from phi through additions.
    // Pattern A: latchVal = AddIOp(phi, expr)
    // Pattern B: latchVal = phi(then=AddIOp(phi, expr), else=phi) (conditional accumulation)
    bool isReduction = false;
    if (isa<AddIOp>(latchVal) && latchVal->getOperandCount() == 2) {
      auto a = latchVal->DEF(0);
      auto b = latchVal->DEF(1);
      if (a == phi || b == phi) isReduction = true;
    } else if (isa<PhiOp>(latchVal)) {
      // Check if any incoming is AddIOp(phi, ...) and others are phi (no-op path)
      bool hasAdd = false;
      bool hasPhiOnly = false;
      for (auto operand : latchVal->getOperands()) {
        auto def = operand.defining;
        if (def == phi) {
          hasPhiOnly = true; // skip path keeps phi unchanged
        } else if (isa<AddIOp>(def) && def->getOperandCount() == 2) {
          auto a = def->DEF(0);
          auto b = def->DEF(1);
          if (a == phi || b == phi) hasAdd = true;
        }
      }
      if (hasAdd && hasPhiOnly) isReduction = true;
      else if (hasAdd) isReduction = true;
    }
    if (!isReduction) continue;

    return phi;
  }
  return nullptr;
}

// Find the LCSSA phi or direct use that captures the reduction's exit value
// (the value of the reduction phi after the inner loop completes).
// Returns the value-producing op (could be the reduction phi itself if
// the loop is rotated and the latchVal is exactly the same as the phi
// after the last iteration; or an LCSSA phi at the exit block).
Op *findReductionExitValue(Op *redPhi, LoopInfo *inner) {
  if (!redPhi || !inner) return nullptr;
  auto exit = inner->getExit();
  if (!exit) return nullptr;

  // Look for a phi at the exit block whose only incoming is from the
  // inner loop's latch and whose value is redPhi (LCSSA).
  for (auto phi : exit->getPhis()) {
    for (auto operand : phi->getOperands()) {
      if (operand.defining == redPhi)
        return phi;
    }
    // Also check if it's the latchVal directly
    auto latchVal = redPhi->DEF(redPhi->getOperandCount() - 1); // approximation
    (void) latchVal;
  }

  return nullptr;
}

// Find the store that writes the reduction result to memory (in the j body,
// after the k loop).
// Returns the StoreOp if found.
//
// The store's value should be the reduction exit value.
// The store's address should depend on j but NOT on k (the inner IV).
Op *findReductionStore(Op *redPhi, Op *redExitVal, LoopInfo *outer, LoopInfo *inner) {
  if (!redPhi || !outer || !inner) return nullptr;
  auto innerIV = inner->getInduction();
  if (!innerIV) return nullptr;

  // The candidate is a StoreOp inside outer (but outside inner) whose
  // stored value uses the reduction exit value.
  for (auto bb : outer->getBlocks()) {
    if (inner->contains(bb)) continue;
    for (auto op : bb->getOps()) {
      if (!isa<StoreOp>(op) || op->getOperandCount() < 2) continue;
      auto storedVal = op->DEF(0);
      auto storedAddr = op->DEF(1);
      // The stored value must use the reduction exit (or directly be redPhi
      // at a single-iteration exit).
      bool valuesMatch = (storedVal == redExitVal) || (storedVal == redPhi);
      if (!valuesMatch) {
        // Try transitive: storedVal could be a derived value
        std::set<Op*> seen;
        valuesMatch = usesValue(storedVal, redExitVal ? redExitVal : redPhi, outer, seen);
      }
      if (!valuesMatch) continue;

      // The store address must not depend on the inner IV.
      std::set<Op*> seen2;
      if (usesValue(storedAddr, innerIV, outer, seen2)) continue;

      return op;
    }
  }
  return nullptr;
}

// Check if a 2-level nest (outer=j, inner=k) matches the reduction pattern.
struct ReductionPattern {
  LoopInfo *outer;       // j loop
  LoopInfo *inner;       // k loop
  Op *redPhi;            // the reduction phi at k's header
  Op *redExitVal;        // the value that escapes the k loop (LCSSA phi or redPhi itself)
  Op *redStore;          // the store that writes the reduction result to memory
  bool valid = false;
};

ReductionPattern matchReductionPattern(LoopInfo *outer) {
  ReductionPattern pat;
  pat.outer = outer;

  // Outer must have exactly one subloop (the inner reduction loop).
  if (outer->subloops.size() != 1) return pat;
  pat.inner = outer->subloops[0];

  // Find the reduction phi at inner header.
  pat.redPhi = findReductionPhi(pat.inner, outer);
  if (!pat.redPhi) return pat;

  // Find the exit value (LCSSA phi or direct).
  pat.redExitVal = findReductionExitValue(pat.redPhi, pat.inner);

  // Find the store that uses the reduction result.
  pat.redStore = findReductionStore(pat.redPhi, pat.redExitVal, outer, pat.inner);
  if (!pat.redStore) return pat;

  pat.valid = true;
  return pat;
}

} // namespace

std::map<std::string, int> ReductionInterchange::stats() {
  return {
    { "candidates", candidates },
    { "interchanged", interchanged },
    { "rejected-shape", rejectedShape },
  };
}

void ReductionInterchange::run() {
  if (!envEnabled("SISY_ENABLE_REDUCTION_INTERCHANGE", false))
    return;

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();

  LoopAnalysis analysis(module);
  analysis.run();

  for (auto &[func, forest] : analysis.getResult()) {
    // Look for 3-level nests: i (top) → j (outer of pattern) → k (inner)
    for (auto top : forest.getLoops()) {
      if (top->parent) continue;
      if (top->subloops.size() != 1) continue;

      auto j = top->subloops[0];
      auto pat = matchReductionPattern(j);
      if (!pat.valid) {
        rejectedShape++;
        continue;
      }

      candidates++;
      // Phase 1: detection only. Phase 2+ will implement the transformation.
    }
  }
}
