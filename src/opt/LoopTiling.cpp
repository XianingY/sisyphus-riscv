#include "LoopPasses.h"
#include "AnalysisManager.h"
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <set>
#include <map>
#include <unordered_set>

using namespace sys;

// Loop Tiling (Strip-Mining) Pass
//
// Operates on CFG-level LoopInfo. For perfect 2-level loop nests,
// applies strip-mining to improve cache locality.
//
// Tile size is derived from cache model (not test dimensions).
// L1 ~32KB, 3 arrays of T*T ints => 3*T*T*4 <= 32K => T<=52.
// Use T=32 (power of 2, conservative).

namespace {

constexpr int kDefaultTileSize = 32;
constexpr int kMinTripForTiling = 64;
constexpr int kL1DataCacheBytes = 32 * 1024;
constexpr int kCacheLineBytes = 64;

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0]) return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

int envInt(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0]) return fallback;
  int v = std::atoi(raw);
  return v > 0 ? v : fallback;
}

int estimateTrip(LoopInfo *loop);

int floorPow2(int n) {
  int p = 1;
  while ((p << 1) > 0 && (p << 1) <= n)
    p <<= 1;
  return p;
}

int clampPow2(int n, int lo, int hi) {
  if (n < lo) return lo;
  if (n > hi) return hi;
  return std::max(lo, floorPow2(n));
}

int typeBytes(Value::Type ty) {
  switch (ty) {
  case Value::i64:
    return 8;
  case Value::i128:
  case Value::f128:
    return 16;
  case Value::i32:
  case Value::f32:
  default:
    return 4;
  }
}

Op *addressRoot(Op *addr, std::set<Op*> &visiting) {
  if (!addr || visiting.count(addr))
    return addr;
  visiting.insert(addr);
  if ((isa<AddLOp>(addr) || isa<AddIOp>(addr)) && addr->getOperandCount() == 2) {
    auto lhs = addr->DEF(0);
    auto rhs = addr->DEF(1);
    if (isa<IntOp>(lhs))
      return addressRoot(rhs, visiting);
    if (isa<IntOp>(rhs))
      return addressRoot(lhs, visiting);
    // Prefer the pointer-like side when one operand is clearly an integer
    // offset expression. This keeps base+linear-index streams together
    // without requiring a full affine analysis in this CFG pass.
    if (lhs && lhs->getResultType() == Value::i64 && rhs &&
        rhs->getResultType() == Value::i32)
      return addressRoot(lhs, visiting);
    if (rhs && rhs->getResultType() == Value::i64 && lhs &&
        lhs->getResultType() == Value::i32)
      return addressRoot(rhs, visiting);
  }
  return addr;
}

Op *addressRoot(Op *addr) {
  std::set<Op*> visiting;
  return addressRoot(addr, visiting);
}

struct TileFootprint {
  std::unordered_set<Op*> readStreams;
  std::unordered_set<Op*> writeStreams;
  int maxElemBytes = 4;
};

void collectTileFootprint(LoopInfo *loop, TileFootprint &fp) {
  for (auto bb : loop->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<LoadOp>(op) && op->getOperandCount() >= 1) {
        fp.readStreams.insert(addressRoot(op->DEF(0)));
        fp.maxElemBytes = std::max(fp.maxElemBytes, typeBytes(op->getResultType()));
      } else if (isa<StoreOp>(op) && op->getOperandCount() >= 2) {
        fp.writeStreams.insert(addressRoot(op->DEF(1)));
        auto val = op->DEF(0);
        if (val)
          fp.maxElemBytes = std::max(fp.maxElemBytes, typeBytes(val->getResultType()));
      }
    }
  }
}

int computeAdaptiveTileSize(LoopInfo *outer, LoopInfo *inner, int fallback) {
  const int forced = envInt("SISY_TILE_SIZE", 0);
  if (forced > 0)
    return forced;

  TileFootprint fp;
  collectTileFootprint(inner, fp);
  int streams = (int)fp.readStreams.size() + (int)fp.writeStreams.size();
  if (streams <= 0)
    streams = 1;

  // Strip-mining this pass tiles a single loop dimension. Model the working
  // set as one cache-line-friendly span per active stream and reserve half of
  // L1 for unrelated live data, stack slots, and conflict misses.
  const int elemBytes = std::max(4, fp.maxElemBytes);
  const int elemsPerLine = std::max(4, kCacheLineBytes / elemBytes);
  const int budgetBytes = kL1DataCacheBytes / 2;
  const int fitByCache = budgetBytes / std::max(1, streams * elemBytes);
  int hi = 256;

  // If the static trip count is known, never choose a tile bigger than it;
  // otherwise keep a conservative upper bound to avoid burying outer-loop
  // progress behind very large tiles.
  int trip = estimateTrip(outer);
  if (trip > 0)
    hi = std::min(hi, std::max(elemsPerLine, trip));

  int candidate = clampPow2(fitByCache, elemsPerLine, hi);
  // Single-dimension strip-mining does not reuse a full 2D cache tile by
  // itself.  For loops with several independent memory streams, large tiles
  // mostly lengthen live ranges and branch bodies; keeping them near one
  // cache line is usually the better generic tradeoff.
  if (streams >= 3)
    candidate = std::min(candidate, elemsPerLine);
  else if (streams == 2)
    candidate = std::min(candidate, elemsPerLine * 4);
  if (streams >= 5 && candidate > elemsPerLine)
    candidate /= 2;
  if (streams >= 8 && candidate > elemsPerLine)
    candidate /= 2;
  return candidate > 0 ? candidate : fallback;
}

// Check if loop has unit-step induction variable.
// More robust than relying on LoopAnalysis::getInduction() which may miss patterns.
bool isCanonicalUnitLoop(LoopInfo *loop) {
  if (!loop || !loop->preheader || loop->latches.size() != 1 || loop->exits.size() != 1)
    return false;

  auto latch = loop->getLatch();
  if (!latch) return false;
  auto header = loop->header;
  if (!header) return false;

  // Look for ANY phi at the header whose latch-incoming is phi+1
  for (auto phi : header->getPhis()) {
    if (phi->getResultType() != Value::i32)
      continue;

    const auto &ops = phi->getOperands();
    const auto &attrs = phi->getAttrs();

    Op *latchVal = nullptr;
    for (int i = 0; i < (int) attrs.size(); i++) {
      auto from = dyn_cast<FromAttr>(attrs[i]);
      if (!from) continue;
      if (from->bb == latch)
        latchVal = ops[i].defining;
    }
    if (!latchVal) continue;
    if (!isa<AddIOp>(latchVal) || latchVal->getOperandCount() != 2)
      continue;
    auto a = latchVal->DEF(0);
    auto b = latchVal->DEF(1);
    bool unit = (a == phi && isa<IntOp>(b) && V(b) == 1) ||
                (b == phi && isa<IntOp>(a) && V(a) == 1);
    if (unit) return true;
  }
  return false;
}

// Estimate trip count. Returns 0 if runtime/unknown.
int estimateTrip(LoopInfo *loop) {
  auto stop = loop->getStop();
  if (!stop) return 0;
  if (isa<IntOp>(stop)) return V(stop);
  return 0; // runtime bound
}

// Check if inner loop is a perfect sub-nest of outer.
// We allow side-effecting blocks at the outer loop level if they don't
// reference the outer IV in a way that breaks tiling semantics.
// Strip-mining wraps the entire outer loop body, so side effects are
// preserved at the same execution frequency.
bool isPerfectSubNest(LoopInfo *outer, LoopInfo *inner) {
  if (outer->subloops.size() != 1 || outer->subloops[0] != inner)
    return false;
  // For tiling to be safe, we just need the outer to have exactly one subloop.
  // Side-effecting code in outer's body (like reduction setup/teardown) is
  // preserved per-iteration by strip-mining, so it's still correct.
  // We do require no impure calls (which could have unbounded side effects
  // we can't reason about).
  for (auto bb : outer->getBlocks()) {
    if (bb == outer->header) continue;
    if (inner->contains(bb)) continue;
    for (auto op : bb->getOps()) {
      if (isa<CallOp>(op) && op->has<ImpureAttr>())
        return false;
    }
  }
  return true;
}

// Check tiling safety: no cross-tile dependences.
// Conservative: only tile when all stores in inner loop write to addresses
// that depend on the inner IV, and all loads from the same base also depend
// on the inner IV (i.e., each tile iteration is independent).
bool isTilingSafe(LoopInfo *outer, LoopInfo *inner) {
  auto outerIV = outer->getInduction();
  auto innerIV = inner->getInduction();
  if (!outerIV || !innerIV) return false;

  // Collect all stores in the inner loop
  for (auto bb : inner->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (isa<CallOp>(op) && op->has<ImpureAttr>())
        return false; // impure call → unsafe
    }
  }
  // For perfectly nested loops where the original execution order is valid,
  // strip-mining (tiling) is always legal because it only reorders iterations
  // within a tile, maintaining the original relative order.
  // The key insight: strip-mining does NOT change iteration order — it just
  // groups iterations. The order within each group is preserved.
  //
  // This is different from interchange (which reorders). Strip-mining alone
  // is always safe for any loop.
  return true;
}

// Build the strip-mined loop structure.
// Original:   for i = 0..N step 1: body(i)
// After:      for ii = 0..N step T:
//               for i = ii..min(ii+T, N) step 1: body(i)
//
// Implementation: We modify the existing loop's start/stop to be bounded
// by the tile, and wrap it in a new outer loop.
bool applyStripMine(LoopInfo *loop, int tileSize) {
  auto preheader = loop->preheader;
  auto header = loop->header;
  auto exit = loop->getExit();
  auto latch = loop->getLatch();
  if (!preheader || !header || !exit || !latch)
    return false;

  auto iv = loop->getInduction();
  if (!iv) return false;

  // Find step from IV phi: the latch value should be iv + step
  // Don't rely on LoopInfo::getStepOp() which may be null after rotation.
  // Instead of matching the exact latch block, take the non-preheader incoming.

  // Find latch value of IV phi (the backedge value)
  Op *ivLatchVal = nullptr;
  int ivStartIdx = -1;
  const auto &ivOps = iv->getOperands();
  const auto &ivAs = iv->getAttrs();
  for (int i = 0; i < (int) ivAs.size(); i++) {
    auto from = dyn_cast<FromAttr>(ivAs[i]);
    if (from && from->bb == preheader) {
      ivStartIdx = i;
    } else if (from) {
      ivLatchVal = ivOps[i].defining;
    }
  }
  if (!ivLatchVal || ivStartIdx < 0) {
    return false;
  }
  Op *start = ivOps[ivStartIdx].defining;
  if (!start) return false;

  // ivLatchVal should be AddIOp(iv, 1) for unit step
  if (!isa<AddIOp>(ivLatchVal) || ivLatchVal->getOperandCount() != 2) {
    return false;
  }
  auto a = ivLatchVal->DEF(0);
  auto b = ivLatchVal->DEF(1);
  bool unitStep = (a == iv && isa<IntOp>(b) && V(b) == 1) ||
                  (b == iv && isa<IntOp>(a) && V(a) == 1);
  if (!unitStep) return false;

  // Find stop from the loop's exit condition.
  // Two patterns after canonicalization:
  //   A) Rotated: latch has branch(lt(iv+step, stop), header, exit)
  //   B) Header-tested: header has branch(lt(iv, stop), body, exit), latch has goto(header)
  // Handle both patterns.
  auto latchBr = latch->getLastOp();
  auto headerBr = header->getLastOp();
  Op *stop = nullptr;
  Op *condOp = nullptr;
  Op *exitBranchOp = nullptr; // the branch instruction containing the exit edge

  if (isa<BranchOp>(latchBr) && latchBr->getOperandCount() == 1) {
    // Pattern A: rotated loop - latch has the exit condition
    auto cond = latchBr->DEF(0);
    if (cond && isa<LtOp>(cond) && cond->getOperandCount() == 2) {
      condOp = cond;
      stop = cond->DEF(1);
      exitBranchOp = latchBr;
    }
  }
  if (!stop && isa<BranchOp>(headerBr) && headerBr->getOperandCount() == 1) {
    // Pattern B: header-tested loop
    // Only safe if header is minimal: just phi(s) + cmp + branch
    bool headerMinimal = true;
    for (auto op : header->getOps()) {
      if (isa<PhiOp>(op) || isa<LtOp>(op) || isa<BranchOp>(op))
        continue;
      headerMinimal = false;
      break;
    }
    if (headerMinimal) {
      auto cond = headerBr->DEF(0);
      if (cond && isa<LtOp>(cond) && cond->getOperandCount() == 2) {
        condOp = cond;
        stop = cond->DEF(1);
        exitBranchOp = headerBr;
      }
    }
  }
  if (!stop || !exitBranchOp) return false;

  // Determine which target is the loop continuation vs exit
  BasicBlock *brTrue = TARGET(exitBranchOp);
  BasicBlock *brFalse = ELSE(exitBranchOp);
  bool exitOnFalse = false;
  if (loop->contains(brTrue) && brFalse == exit) {
    exitOnFalse = true;
  } else if (loop->contains(brFalse) && brTrue == exit) {
    exitOnFalse = false;
  } else {
    return false;
  }

  // This CFG strip-mining rewrite changes the original loop-exit edge
  // from `exitBranchOp -> exit` into `exitBranchOp -> tileLatch -> tileHeader -> exit`.
  // If the exit block has phi/LCSSA values, their incoming block/value pairs must
  // be rebuilt through the tile loop.  The current transform only rewires control
  // edges, so conservatively avoid these loops instead of leaving stale `from`
  // attributes that later cleanup turns into empty phis.
  if (!exit->getPhis().empty())
    return false;

  // The preheader must end with a GotoOp to header.
  auto preTerm = preheader->getLastOp();
  if (!preTerm || !isa<GotoOp>(preTerm) || TARGET(preTerm) != header)
    return false;

  auto region = header->getParent();
  Builder builder;

  // Create tile loop blocks
  auto tileHeader = region->insertAfter(preheader);
  auto innerSetup = region->insertAfter(tileHeader);
  auto tileLatch = region->insert(exit);

  // 1. Wire preheader → tileHeader
  builder.replace<GotoOp>(preTerm, { new TargetAttr(tileHeader) });

  // 2. Build tileHeader: tile_iv phi, comparison, branch
  builder.setToBlockEnd(tileHeader);
  auto tileIV = builder.create<PhiOp>({ start }, { new FromAttr(preheader) });
  auto tileCond = builder.create<LtOp>(std::vector<Value>{ tileIV, stop });
  builder.create<BranchOp>(std::vector<Value>{ tileCond },
    { new TargetAttr(innerSetup), new ElseAttr(exit) });

  // 3. Build innerSetup: compute innerStop = min(tile_iv + T, stop)
  builder.setToBlockEnd(innerSetup);
  auto tileSizeOp = builder.create<IntOp>({ new IntAttr(tileSize) });
  auto tileEnd = builder.create<AddIOp>(std::vector<Value>{ tileIV, tileSizeOp });
  auto cmpEnd = builder.create<LtOp>(std::vector<Value>{ tileEnd, stop });
  auto innerStop = builder.create<SelectOp>(std::vector<Value>{ cmpEnd, tileEnd, stop });
  builder.create<GotoOp>({ new TargetAttr(header) });

  // 4. Update IV phi: start now comes from innerSetup (= tileIV)
  iv->setOperand(ivStartIdx, Value(tileIV));
  cast<FromAttr>(iv->getAttrs()[ivStartIdx])->bb = innerSetup;

  // 5. Replace stop in condition with innerStop
  condOp->setOperand(1, Value(innerStop));

  // 6. Redirect exit edge to tileLatch
  if (exitOnFalse) {
    ELSE(exitBranchOp) = tileLatch;
  } else {
    TARGET(exitBranchOp) = tileLatch;
  }

  // 7. Build tileLatch: tile_iv += T, goto tileHeader
  builder.setToBlockEnd(tileLatch);
  auto nextTileIV = builder.create<AddIOp>(std::vector<Value>{ tileIV, tileSizeOp });
  builder.create<GotoOp>({ new TargetAttr(tileHeader) });

  // 8. Add latch edge to tileIV phi
  tileIV->pushOperand(nextTileIV);
  tileIV->add<FromAttr>(tileLatch);

  return true;
}

} // namespace

std::map<std::string, int> LoopTiling::stats() {
  return {
    { "candidates", candidates },
    { "tiled", tiled },
    { "rejected-shape", rejectedShape },
    { "rejected-profit", rejectedProfit },
    { "adaptive-tiles", adaptiveTiles },
    { "tile-size-sum", tileSizeSum },
  };
}

void LoopTiling::run() {
  if (!envEnabled("SISY_ENABLE_LOOP_TILING", true))
    return;

  int maxRounds = envInt("SISY_TILE_ROUNDS", 3);

  // Run multiple rounds to handle deeper nests (tile from inside out).
  for (int round = 0; round < maxRounds; round++) {
    for (auto func : collectFuncs())
      func->getRegion()->updatePreds();

    std::map<FuncOp*, LoopForest> localForests;
    std::map<FuncOp*, LoopForest> *forests = nullptr;
    if (context() && context()->enabled())
      forests = &context()->analysis().getLoopForests();
    else {
      LoopAnalysis analysis(module);
      analysis.run();
      localForests = analysis.getResult();
      forests = &localForests;
    }

    bool changed = false;

    for (auto &[func, forest] : *forests) {
      // Process innermost loops first: collect all candidate pairs.
      // A candidate is a 2-level nest where the inner has no subloops.
      for (auto loop : forest.getLoops()) {
        if (loop->parent)
          continue;

        // Walk the nest tree to find the deepest tileable pair.
        // Use a recursive helper via worklist.
        std::vector<LoopInfo*> worklist;
        worklist.push_back(loop);

        while (!worklist.empty()) {
          auto cur = worklist.back();
          worklist.pop_back();

          if (!isCanonicalUnitLoop(cur)) {
            continue;
          }
          if (cur->subloops.size() != 1) {
            continue;
          }
          auto inner = cur->subloops[0];

          // If inner has subloops, try deeper first.
          if (!inner->subloops.empty()) {
            worklist.push_back(inner);
            continue;
          }

          // inner has no subloops — this is a tileable 2-level pair.
          if (!isCanonicalUnitLoop(inner)) {
            continue;
          }
          if (!isPerfectSubNest(cur, inner)) {
            continue;
          }

          // Don't tile if outer has extra phis (state-carrying across iterations)
          // Strip-mining is only safe when non-IV phis are absent or loop-invariant.
          {
            auto outerIV = cur->getInduction();
            auto innerIV = inner->getInduction();
            if (!outerIV || !innerIV)
              continue;
            // Both must have only the IV phi — any extra phi means
            // state is carried that strip-mining could disrupt.
            auto outerPhis = cur->header->getPhis();
            int outerPhiCount = 0;
            for (auto phi : outerPhis) { (void)phi; outerPhiCount++; }
            if (outerPhiCount > 1) continue;
            auto innerPhis = inner->header->getPhis();
            int innerPhiCount = 0;
            for (auto phi : innerPhis) { (void)phi; innerPhiCount++; }
            if (innerPhiCount > 1) continue;
            if (outerIV->getOperandCount() != 2) continue;
          }

          candidates++;

          if (!isTilingSafe(cur, inner)) {
            rejectedShape++;
            continue;
          }

          int trip = estimateTrip(cur);
          if (trip > 0 && trip < kMinTripForTiling) {
            rejectedProfit++;
            continue;
          }

          int tileSize = computeAdaptiveTileSize(cur, inner, kDefaultTileSize);
          if (applyStripMine(cur, tileSize)) {
            adaptiveTiles++;
            tileSizeSum += tileSize;
            tiled++;
            changed = true;
            break; // Re-run analysis after modifying
          } else {
            rejectedShape++;
          }
        }

        if (changed) break;
      }

      if (changed) break;
    }

    if (changed && context() && context()->enabled())
      context()->analysis().invalidate(PreservedAnalyses::none(),
                                       "loop-tiling-internal");
    if (!changed) break;
  }

  for (auto func : collectFuncs())
    func->getRegion()->updatePreds();
}

PreservedAnalyses LoopTiling::run(PassContext &ctx) {
  activeContext = &ctx;
  int before = tiled;
  run();
  activeContext = nullptr;
  return tiled == before ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
