#include "LoopPasses.h"

using namespace sys;

// LoopInterchange pass
// Detects perfect nested loops and checks if they can be interchanged.

std::map<std::string, int> LoopInterchange::stats() {
    return {
        {"detected", detected},
        {"interchanged", interchanged}
    };
}

// Check if a loop is a perfect nest (inner loop is the only content of outer's body)
bool LoopInterchange::isPerfectNest(LoopInfo* outer, LoopInfo* inner) {
    // For a perfect nest, outer should have exactly 2 blocks in its loop body
    // (the header and the inner loop's header/preheader)
    // Actually, we need to check that the inner loop completely fills the outer loop body

    // A perfect nest means:
    // 1. Outer loop has exactly one subloop (the inner)
    // 2. The inner loop's header is dominated by outer's header
    // 3. The inner loop's blocks are the only things in outer's blocks

    if (outer->subloops.size() != 1)
        return false;

    if (outer->subloops[0] != inner)
        return false;

    // Check that inner loop completely dominates outer's body blocks
    // (除了 inner 的 blocks，outer 应该只有 header 自己在用)
    for (auto bb : outer->getBlocks()) {
        if (bb == outer->header)
            continue;
        // If this block belongs to inner loop, it's fine
        if (inner->contains(bb))
            continue;
        // Otherwise, there's something else in outer's body
        return false;
    }

    return true;
}

bool LoopInterchange::canInterchange(LoopInfo* outer, LoopInfo* inner) {
    // For interchange to be safe:
    // 1. Must be perfect nest
    // 2. No loop-carried dependencies with negative direction

    if (!isPerfectNest(outer, inner))
        return false;

    // In a perfect nest, if all loop-carried dependencies have inner direction >= 0,
    // then interchange is safe.

    // For now, we conservatively return false because we can't easily
    // compute the dependence direction without alias analysis.

    // TODO: Implement proper dependence analysis
    return false;
}

void LoopInterchange::runImpl(LoopInfo* info) {
    // Process subloops first
    for (auto subloop : info->subloops) {
        runImpl(subloop);
    }

    // Check if this loop has exactly one subloop (potential perfect nest)
    if (info->subloops.size() != 1)
        return;

    auto inner = info->subloops[0];

    // Check if it's a perfect nest
    if (!isPerfectNest(info, inner))
        return;

    detected++;

    // Try to interchange
    if (canInterchange(info, inner)) {
        // TODO: Actually perform the interchange
        // This requires:
        // 1. Swapping the induction variables
        // 2. Rewiring the CFG
        // 3. Updating all uses
        interchanged++;
    }
}

void LoopInterchange::run() {
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
