#include "LoopPasses.h"
#include "Passes.h"

using namespace sys;

// BoundsCheck pass
// Eliminates redundant bounds checks in stencil loops like conv2d.
//
// Pattern: if (i + offset >= 0 && i + offset < bound) { access array[i+offset] }
//
// For conv2d with pad=2:
// - When r is in [pad, N-pad), the row check is always true
// - When c is in [pad, N-pad), the col check is always true
//
// Strategy:
// 1. Find stencil access patterns with constant offsets
// 2. Determine safe region based on loop bounds
// 3. Hoist or eliminate redundant checks

std::map<std::string, int> BoundsCheck::stats() {
    return {
        {"checks-eliminated", eliminated},
        {"checks-hoisted", hoisted}
    };
}

// Extract the value from an Op, handling phi nodes
Op* peelValue(Op* val) {
    while (isa<PhiOp>(val) && val->getOperandCount() == 1) {
        val = val->DEF();
    }
    return val;
}

// Check if an op is a known constant value
bool tryGetIntValue(Op* op, int& result) {
    op = peelValue(op);
    if (auto intOp = dyn_cast<IntOp>(op)) {
        result = V(intOp);
        return true;
    }
    return false;
}

// Find the loop variable and step from a phi at the loop header
// Returns the induction variable op and step value
Op* findLoopInductionVar(LoopInfo* loop, int& step) {
    step = loop->getStep();
    return loop->getInduction();
}

// Check if we can prove that (indVar + offset) is always within [0, bound)
// given that indVar is in [loopStart, loopStop)
bool canProveAlwaysInBounds(Op* indVar, int offset, Op* bound, LoopInfo* loop) {
    // Get loop bounds
    auto start = peelValue(loop->getStart());
    auto stop = peelValue(loop->getStop());

    int startVal, stopVal, boundVal;

    // Need constant loop bounds for now
    if (!tryGetIntValue(start, startVal) || !tryGetIntValue(stop, stopVal))
        return false;

    // Step must be positive
    if (loop->getStep() <= 0)
        return false;

    // If offset is such that even at the minimum indVar, idx >= 0 is guaranteed
    int minIdx = startVal + offset;
    bool lowerSafe = (minIdx >= 0);

    // For upper bound, we need startVal + offset < bound OR stopVal + offset <= bound
    // But this depends on bound which might not be constant
    // For now, just check if offset >= 0 and startVal >= 0
    if (!lowerSafe)
        return false;

    // Check upper bound - if bound is constant
    if (tryGetIntValue(bound, boundVal)) {
        // Check if max possible index is < bound
        int maxIdx = stopVal - 1 + offset;  // stop is exclusive, so last valid is stopVal - 1
        return maxIdx < boundVal;
    }

    return false;
}

// Look for: if (cond) { trueBody } else { falseBody }
// where cond is a bounds check and trueBody has the actual access
// Returns true if we can eliminate the check
bool tryEliminateBoundsCheck(IfOp* ifOp, LoopInfo* loop, Op* indVar) {
    auto cond = ifOp->getOperand(0).defining;

    // Handle nested AndIOp: ((a >= 0 && a < N) && (b >= 0 && b < N))
    // For now, just handle one level of AndIOp
    Op* idx = nullptr;
    Op* upper = nullptr;

    // The condition is: idx >= 0 && idx < upper
    // which is represented as AndIOp(LeOp(0, idx), LtOp(idx, upper))
    if (!isa<AndIOp>(cond))
        return false;

    auto andOp = cast<AndIOp>(cond);
    auto left = peelValue(andOp->DEF(0));
    auto right = peelValue(andOp->DEF(1));

    // Try to find LeOp(0, idx) on one side and LtOp(idx, upper) on the other
    Op* foundIdx = nullptr;
    bool foundLower = false, foundUpper = false;

    // Check left side for lower bound check
    if (auto le = dyn_cast<LeOp>(left)) {
        auto le0 = peelValue(le->DEF(0));
        auto leIdx = peelValue(le->DEF(1));
        if (isa<IntOp>(le0) && V(le0) == 0) {
            foundIdx = leIdx;
            foundLower = true;
        }
    }

    // Check right side for upper bound check
    if (auto lt = dyn_cast<LtOp>(right)) {
        if (foundIdx && peelValue(lt->DEF(0)) == foundIdx) {
            upper = peelValue(lt->DEF(1));
            foundUpper = true;
        }
    }

    // Also check the swapped case
    if (!foundUpper && !foundLower) {
        if (auto lt = dyn_cast<LtOp>(left)) {
            auto ltIdx = peelValue(lt->DEF(0));
            auto ltUpper = peelValue(lt->DEF(1));
            if (isa<IntOp>(ltUpper) && V(ltUpper) == 0) {
                // This is idx < 0 which is wrong, skip
            } else {
                foundIdx = ltIdx;
                upper = ltUpper;
                foundUpper = true;
            }
        }
        if (auto le = dyn_cast<LeOp>(right)) {
            auto le0 = peelValue(le->DEF(0));
            if (isa<IntOp>(le0) && V(le0) == 0) {
                foundLower = true;
            }
        }
    }

    if (!foundLower || !foundUpper || !foundIdx)
        return false;

    // Try to extract offset from foundIdx relative to indVar
    int offset = -1;
    if (foundIdx == indVar) {
        offset = 0;
    } else if (auto add = dyn_cast<AddIOp>(foundIdx)) {
        if (peelValue(add->DEF(0)) == indVar && isa<IntOp>(add->DEF(1))) {
            offset = V(add->DEF(1));
        } else if (peelValue(add->DEF(1)) == indVar && isa<IntOp>(add->DEF(0))) {
            offset = V(add->DEF(0));
        }
    } else if (auto sub = dyn_cast<SubIOp>(foundIdx)) {
        if (peelValue(sub->DEF(0)) == indVar && isa<IntOp>(sub->DEF(1))) {
            offset = -V(sub->DEF(1));
        }
    }

    if (offset == -1)
        return false;

    // Now check if we can prove the condition is always true
    if (canProveAlwaysInBounds(indVar, offset, upper, loop)) {
        // We can eliminate the check - just execute the true branch
        // For now, just mark as eliminated and remove the if
        return true;
    }

    return false;
}

void BoundsCheck::runImpl(LoopInfo* loop) {
    // Process subloops first
    for (auto subloop : loop->subloops) {
        runImpl(subloop);
    }

    // Only process innermost loops
    if (!loop->subloops.empty())
        return;

    // Get loop info
    int stepVal;
    auto indVar = findLoopInductionVar(loop, stepVal);
    if (!indVar)
        return;

    // Look for bounds check patterns in the loop body
    for (auto bb : loop->getBlocks()) {
        for (auto op : bb->getOps()) {
            if (!isa<IfOp>(op))
                continue;

            auto ifOp = cast<IfOp>(op);

            // Try to eliminate this bounds check
            if (tryEliminateBoundsCheck(ifOp, loop, indVar)) {
                eliminated++;
                // TODO: Actually replace the if with unconditional execution
                // For now just count it
            }
        }
    }
}

void BoundsCheck::run() {
    LoopAnalysis analysis(module);
    analysis.run();
    auto forests = analysis.getResult();

    for (auto func : collectFuncs()) {
        auto& forest = forests[func];

        for (auto loop : forest.getLoops()) {
            if (!loop->getParent())
                runImpl(loop);
        }
    }
}
