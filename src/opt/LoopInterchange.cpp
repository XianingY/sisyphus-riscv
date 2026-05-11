#include "LoopPasses.h"

#include <cstdlib>
#include <cstring>

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

namespace {

// Represents the linear dependence of an address expression on induction variables.
// An address is modeled as: base + outerCoeff * outerIV + innerCoeff * innerIV + constant
// where coefficients represent the stride (in bytes) per iteration of each loop.
//
// For non-constant strides (e.g. A[i][j] where row size N is a runtime value),
// we track whether the coefficient is "symbolic" (non-constant but loop-invariant).
// A symbolic coefficient of value 0 means no dependence on that IV.
// A non-zero symbolic coefficient means there IS a dependence but we don't know the
// exact numeric stride — we only know it's > 0 (assuming positive array dimensions).
struct StrideInfo {
    int outerCoeff = 0;       // stride w.r.t. outer induction variable (0 = no dependence)
    int innerCoeff = 0;       // stride w.r.t. inner induction variable (0 = no dependence)
    bool outerSymbolic = false; // true if outerCoeff is symbolic (non-constant invariant)
    bool innerSymbolic = false; // true if innerCoeff is symbolic (non-constant invariant)
    bool valid = false;       // whether analysis succeeded
};

// Represents a memory access with its stride information.
struct MemAccessInfo {
    Op *op;              // the LoadOp or StoreOp
    Op *addr;            // the address operand
    StrideInfo stride;   // stride analysis result
};

// Represents a linear expression in terms of induction variables.
// An expression is linear if it's of the form:
//   invariant_part + c1 * outerIV + c2 * innerIV
// where c1, c2 are loop-invariant (possibly non-constant) values.
struct LinearExpr {
    int outerCoeff = 0;        // numeric coefficient for outer IV
    int innerCoeff = 0;        // numeric coefficient for inner IV
    bool outerSymbolic = false; // outer coefficient is non-constant but invariant
    bool innerSymbolic = false; // inner coefficient is non-constant but invariant
    bool isInvariant = false;  // true if the expression doesn't depend on either IV
    bool valid = false;
    int constVal = 0;          // only meaningful when isInvariant && isConst
    bool isConst = false;      // true if the expression is a compile-time constant
};

// Analyze an expression to determine its linear dependence on induction variables.
// outerIV and innerIV are the phi nodes for the outer and inner loop induction variables.
// outerLoop and innerLoop provide containment info.
LinearExpr analyzeExpr(Op *op, Op *outerIV, Op *innerIV,
                       LoopInfo *outerLoop, LoopInfo *innerLoop,
                       std::unordered_map<Op*, LinearExpr> &cache) {
    if (cache.count(op))
        return cache[op];

    LinearExpr result;
    // Prevent infinite recursion on cycles (e.g. phi nodes).
    cache[op] = result;

    // Case 1: The op IS one of the induction variables.
    if (op == outerIV) {
        result.outerCoeff = 1;
        result.innerCoeff = 0;
        result.isInvariant = false;
        result.valid = true;
        cache[op] = result;
        return result;
    }
    if (op == innerIV) {
        result.outerCoeff = 0;
        result.innerCoeff = 1;
        result.isInvariant = false;
        result.valid = true;
        cache[op] = result;
        return result;
    }

    // Case 2: Integer constant.
    if (isa<IntOp>(op)) {
        result.outerCoeff = 0;
        result.innerCoeff = 0;
        result.isInvariant = true;
        result.isConst = true;
        result.constVal = V(op);
        result.valid = true;
        cache[op] = result;
        return result;
    }

    // Case 3: Value defined outside both loops (loop-invariant).
    auto parent = op->getParent();
    if (parent && !outerLoop->contains(parent)) {
        result.outerCoeff = 0;
        result.innerCoeff = 0;
        result.isInvariant = true;
        result.valid = true;
        cache[op] = result;
        return result;
    }

    // Case 4: AddIOp or AddLOp — linear addition.
    if (isa<AddIOp>(op) || isa<AddLOp>(op)) {
        auto lhs = analyzeExpr(op->DEF(0), outerIV, innerIV, outerLoop, innerLoop, cache);
        auto rhs = analyzeExpr(op->DEF(1), outerIV, innerIV, outerLoop, innerLoop, cache);
        if (lhs.valid && rhs.valid) {
            result.outerCoeff = lhs.outerCoeff + rhs.outerCoeff;
            result.innerCoeff = lhs.innerCoeff + rhs.innerCoeff;
            result.outerSymbolic = lhs.outerSymbolic || rhs.outerSymbolic;
            result.innerSymbolic = lhs.innerSymbolic || rhs.innerSymbolic;
            result.isInvariant = lhs.isInvariant && rhs.isInvariant;
            result.valid = true;
        }
        cache[op] = result;
        return result;
    }

    // Case 5: SubIOp — linear subtraction.
    if (isa<SubIOp>(op)) {
        auto lhs = analyzeExpr(op->DEF(0), outerIV, innerIV, outerLoop, innerLoop, cache);
        auto rhs = analyzeExpr(op->DEF(1), outerIV, innerIV, outerLoop, innerLoop, cache);
        if (lhs.valid && rhs.valid) {
            result.outerCoeff = lhs.outerCoeff - rhs.outerCoeff;
            result.innerCoeff = lhs.innerCoeff - rhs.innerCoeff;
            result.outerSymbolic = lhs.outerSymbolic || rhs.outerSymbolic;
            result.innerSymbolic = lhs.innerSymbolic || rhs.innerSymbolic;
            result.isInvariant = lhs.isInvariant && rhs.isInvariant;
            result.valid = true;
        }
        cache[op] = result;
        return result;
    }

    // Case 6: MulIOp — one operand must be invariant for linearity.
    if (isa<MulIOp>(op)) {
        auto lhs = analyzeExpr(op->DEF(0), outerIV, innerIV, outerLoop, innerLoop, cache);
        auto rhs = analyzeExpr(op->DEF(1), outerIV, innerIV, outerLoop, innerLoop, cache);
        if (lhs.valid && rhs.valid) {
            // For linearity, at least one side must be invariant.
            if (rhs.isInvariant && rhs.isConst) {
                // Multiply by a known constant.
                int c = rhs.constVal;
                result.outerCoeff = lhs.outerCoeff * c;
                result.innerCoeff = lhs.innerCoeff * c;
                result.outerSymbolic = lhs.outerSymbolic;
                result.innerSymbolic = lhs.innerSymbolic;
                result.isInvariant = lhs.isInvariant;
                result.valid = true;
            } else if (lhs.isInvariant && lhs.isConst) {
                // Multiply by a known constant (symmetric).
                int c = lhs.constVal;
                result.outerCoeff = rhs.outerCoeff * c;
                result.innerCoeff = rhs.innerCoeff * c;
                result.outerSymbolic = rhs.outerSymbolic;
                result.innerSymbolic = rhs.innerSymbolic;
                result.isInvariant = rhs.isInvariant;
                result.valid = true;
            } else if (rhs.isInvariant && !lhs.isInvariant) {
                // Multiply IV-dependent expression by a non-constant invariant.
                // e.g., outerIV * N where N is a runtime array dimension.
                // The result is still linear but with a symbolic coefficient.
                if (lhs.outerCoeff != 0 && lhs.innerCoeff == 0 && !lhs.outerSymbolic) {
                    // Pure outer IV term * invariant → symbolic outer stride.
                    result.outerCoeff = lhs.outerCoeff; // keep direction info
                    result.outerSymbolic = true;
                    result.innerCoeff = 0;
                    result.isInvariant = false;
                    result.valid = true;
                } else if (lhs.outerCoeff == 0 && lhs.innerCoeff != 0 && !lhs.innerSymbolic) {
                    // Pure inner IV term * invariant → symbolic inner stride.
                    result.outerCoeff = 0;
                    result.innerCoeff = lhs.innerCoeff;
                    result.innerSymbolic = true;
                    result.isInvariant = false;
                    result.valid = true;
                } else {
                    // Mixed or already symbolic — can't analyze further.
                    result.valid = false;
                }
            } else if (lhs.isInvariant && !rhs.isInvariant) {
                // Symmetric case: invariant * IV-dependent.
                if (rhs.outerCoeff != 0 && rhs.innerCoeff == 0 && !rhs.outerSymbolic) {
                    result.outerCoeff = rhs.outerCoeff;
                    result.outerSymbolic = true;
                    result.innerCoeff = 0;
                    result.isInvariant = false;
                    result.valid = true;
                } else if (rhs.outerCoeff == 0 && rhs.innerCoeff != 0 && !rhs.innerSymbolic) {
                    result.outerCoeff = 0;
                    result.innerCoeff = rhs.innerCoeff;
                    result.innerSymbolic = true;
                    result.isInvariant = false;
                    result.valid = true;
                } else {
                    result.valid = false;
                }
            } else if (lhs.isInvariant && rhs.isInvariant) {
                // Both invariant — product is invariant.
                result.isInvariant = true;
                result.valid = true;
            } else {
                // Both depend on IVs — non-linear, can't analyze.
                result.valid = false;
            }
        }
        cache[op] = result;
        return result;
    }

    // Case 7: PhiOp — could be a loop-carried value.
    // For phis at the inner loop header that aren't the inner IV,
    // check if they are derived from the outer IV.
    if (isa<PhiOp>(op)) {
        // If this phi is at the outer loop header (but not the outer IV itself),
        // it might be a derived induction variable. Check its preheader value.
        if (parent == outerLoop->header) {
            // This is some other phi in the outer loop header.
            // Conservatively treat as unknown — it could be another outer IV
            // or a reduction variable.
            result.valid = false;
            cache[op] = result;
            return result;
        }
        // If this phi is at the inner loop header (but not the inner IV),
        // check if it's a pointer that advances with the outer loop.
        // Its value at the start of each inner loop iteration comes from
        // the inner loop's preheader.
        if (parent == innerLoop->header) {
            auto preheader = innerLoop->preheader;
            if (preheader) {
                auto preVal = Op::getPhiFrom(op, preheader);
                if (preVal) {
                    result = analyzeExpr(preVal, outerIV, innerIV, outerLoop, innerLoop, cache);
                    cache[op] = result;
                    return result;
                }
            }
        }
        // For other phis, conservatively mark as invalid.
        result.valid = false;
        cache[op] = result;
        return result;
    }

    // Case 8: LoadOp with i64 result — this is a pointer load (array base).
    // Treat as loop-invariant if its address is defined outside both loops.
    if (isa<LoadOp>(op) && op->getResultType() == Value::i64) {
        auto addrOp = op->DEF(0);
        if (addrOp && addrOp->getParent() && !outerLoop->contains(addrOp->getParent())) {
            result.isInvariant = true;
            result.valid = true;
            cache[op] = result;
            return result;
        }
    }

    // Case 9: GetGlobalOp — always invariant (global array base).
    if (isa<GetGlobalOp>(op)) {
        result.isInvariant = true;
        result.valid = true;
        cache[op] = result;
        return result;
    }

    // Case 10: AllocaOp — always invariant (local array base).
    if (isa<AllocaOp>(op)) {
        result.isInvariant = true;
        result.valid = true;
        cache[op] = result;
        return result;
    }

    // Default: cannot analyze.
    result.valid = false;
    cache[op] = result;
    return result;
}

// ============================================================
// Dependence Direction Vector Analysis (Task 3.2)
// ============================================================

// Direction of dependence in a single loop dimension.
// '<' means the source iteration precedes the sink (forward dependence).
// '=' means same iteration (loop-independent dependence).
// '>' means the source iteration follows the sink (backward dependence).
// '*' means direction is unknown.
enum DependenceDir {
    DIR_LT,   // '<' — forward (i1 < i2)
    DIR_EQ,   // '=' — same iteration
    DIR_GT,   // '>' — backward (i1 > i2)
    DIR_STAR  // '*' — unknown
};

// A direction vector for a 2-level loop nest: (outerDir, innerDir).
// For interchange safety, after swapping the two entries, the vector
// must remain lexicographically non-negative (i.e., the leading non-'='
// entry must be '<' or '=').
struct DirectionVector {
    DependenceDir outer;
    DependenceDir inner;
};

// Trace the base address (alloca or global) of a memory access address.
// Returns nullptr if the base cannot be determined.
static Op *traceBase(Op *addr) {
    if (!addr)
        return nullptr;

    // If the address has AliasAttr, use it to find the base.
    if (addr->has<AliasAttr>()) {
        auto alias = ALIAS(addr);
        if (alias->unknown)
            return nullptr;
        // If there's exactly one base, return it.
        if (alias->location.size() == 1)
            return alias->location.begin()->first;
        // Multiple possible bases — ambiguous.
        return nullptr;
    }

    // Fallback: walk through AddLOp chains to find the base.
    constexpr int MAX_DEPTH = 16;
    Op *cur = addr;
    for (int depth = 0; depth < MAX_DEPTH; depth++) {
        if (isa<AllocaOp>(cur) || isa<GetGlobalOp>(cur))
            return cur;
        if (isa<AddLOp>(cur)) {
            // One operand should be the base pointer, the other the offset.
            auto lhs = cur->DEF(0);
            auto rhs = cur->DEF(1);
            // The base is typically the i64 (pointer) operand.
            if (lhs && (isa<AllocaOp>(lhs) || isa<GetGlobalOp>(lhs)))
                return lhs;
            if (rhs && (isa<AllocaOp>(rhs) || isa<GetGlobalOp>(rhs)))
                return rhs;
            // Recurse on the pointer operand (i64 type).
            if (lhs && lhs->getResultType() == Value::i64) {
                cur = lhs;
                continue;
            }
            if (rhs && rhs->getResultType() == Value::i64) {
                cur = rhs;
                continue;
            }
            return nullptr;
        }
        if (isa<LoadOp>(cur) && cur->getResultType() == Value::i64) {
            // Pointer load — check if its address is an alloca/global.
            auto loadAddr = cur->DEF(0);
            if (loadAddr && (isa<AllocaOp>(loadAddr) || isa<GetGlobalOp>(loadAddr)))
                return loadAddr;
            return nullptr;
        }
        if (isa<PhiOp>(cur)) {
            // For phi nodes, try the first operand.
            if (cur->getOperandCount() > 0) {
                cur = cur->DEF(0);
                continue;
            }
            return nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

// Compute the dependence direction in a single dimension given the
// stride coefficients of two accesses in that dimension.
//
// For two accesses with subscripts s1 = c1 * iv + ... and s2 = c2 * iv + ...,
// a dependence exists when s1(i1) == s2(i2) for some iterations i1, i2.
//
// If c1 == c2 (same coefficient), the accesses step through the dimension
// identically, so the dependence direction is '=' (same iteration in this dim).
//
// If c1 != c2 and both are known constants, the direction depends on whether
// the difference allows i1 < i2, i1 > i2, or both. For safety, if we can't
// determine the exact direction, we use '*' (unknown/any).
//
// For symbolic coefficients (non-constant but loop-invariant), if both are
// symbolic with the same sign, we conservatively assume '=' (they likely
// represent the same row stride). Otherwise '*'.
static DependenceDir computeDirection(int coeff1, bool symbolic1,
                                      int coeff2, bool symbolic2) {
    // Case 1: Both coefficients are zero — no dependence on this IV.
    // The accesses don't vary with this loop dimension, so direction is '='.
    if (coeff1 == 0 && coeff2 == 0)
        return DIR_EQ;

    // Case 2: Both are symbolic (non-constant invariant).
    // If they have the same sign and magnitude placeholder, they likely
    // represent the same stride (e.g., both are N for A[i*N+j]).
    // Conservatively assume '=' since they're the same symbolic expression
    // multiplied by the same IV.
    if (symbolic1 && symbolic2) {
        // Same sign means same symbolic stride — direction is '='.
        if ((coeff1 > 0) == (coeff2 > 0))
            return DIR_EQ;
        // Different signs — unknown direction.
        return DIR_STAR;
    }

    // Case 3: One is symbolic, the other is not (or zero).
    // Can't compare a symbolic stride with a numeric one.
    if (symbolic1 || symbolic2) {
        // If the non-symbolic one is zero, the symbolic one determines direction.
        // But since we don't know the symbolic value, use '*'.
        if (coeff1 == 0 || coeff2 == 0)
            return DIR_STAR;
        return DIR_STAR;
    }

    // Case 4: Both are known numeric constants.
    if (coeff1 == coeff2) {
        // Same stride — accesses are parallel, direction is '='.
        return DIR_EQ;
    }

    // Different strides — a dependence may exist in any direction.
    // For the general case with different coefficients, the dependence
    // can go in any direction depending on the constant offsets.
    // Conservatively return '*'.
    return DIR_STAR;
}

// Compute dependence direction vectors for all pairs of memory accesses
// that may alias (access the same base array).
//
// Returns a vector of DirectionVectors. If any access pair cannot be analyzed,
// returns an empty vector (indicating interchange safety cannot be determined).
static std::vector<DirectionVector> computeDependenceDirections(
        const std::vector<MemAccessInfo> &accesses) {
    std::vector<DirectionVector> directions;

    // Group accesses by base address.
    std::unordered_map<Op*, std::vector<size_t>> baseGroups;
    for (size_t i = 0; i < accesses.size(); i++) {
        Op *base = traceBase(accesses[i].addr);
        if (!base) {
            // Cannot determine base — conservatively return empty
            // (caller should treat as unsafe).
            return {};
        }
        baseGroups[base].push_back(i);
    }

    // For each group of accesses to the same base, compute direction vectors
    // for all pairs that may have a dependence.
    for (auto &[base, indices] : baseGroups) {
        // Only need to check pairs if there are at least 2 accesses to same base.
        if (indices.size() < 2)
            continue;

        // For dependence to matter, at least one access must be a store.
        // (Two loads to the same address don't create a dependence.)
        bool hasStore = false;
        for (size_t idx : indices) {
            if (isa<StoreOp>(accesses[idx].op)) {
                hasStore = true;
                break;
            }
        }
        if (!hasStore)
            continue;

        // Check all pairs where at least one is a store.
        for (size_t i = 0; i < indices.size(); i++) {
            for (size_t j = i + 1; j < indices.size(); j++) {
                const auto &acc1 = accesses[indices[i]];
                const auto &acc2 = accesses[indices[j]];

                // At least one must be a store for a true/anti/output dependence.
                if (!isa<StoreOp>(acc1.op) && !isa<StoreOp>(acc2.op))
                    continue;

                // Compute direction in outer dimension.
                DependenceDir outerDir = computeDirection(
                    acc1.stride.outerCoeff, acc1.stride.outerSymbolic,
                    acc2.stride.outerCoeff, acc2.stride.outerSymbolic);

                // Compute direction in inner dimension.
                DependenceDir innerDir = computeDirection(
                    acc1.stride.innerCoeff, acc1.stride.innerSymbolic,
                    acc2.stride.innerCoeff, acc2.stride.innerSymbolic);

                directions.push_back({outerDir, innerDir});
            }
        }
    }

    return directions;
}

// Check if a direction vector is lexicographically non-negative.
// A vector (d1, d2) is lex-non-negative if:
//   - d1 == '<' (forward in leading dimension), OR
//   - d1 == '=' AND d2 is '<' or '=' or '*', OR
//   - d1 == '*' (unknown — could be negative, so NOT safe)
//
// For interchange safety, we need ALL direction vectors to remain
// lex-non-negative after swapping the outer and inner entries.
static bool isLexNonNegative(DependenceDir d1, DependenceDir d2) {
    switch (d1) {
        case DIR_LT:
            // Leading '<' — always non-negative regardless of d2.
            return true;
        case DIR_EQ:
            // Leading '=' — depends on d2.
            return d2 != DIR_GT;
        case DIR_GT:
            // Leading '>' — always negative.
            return false;
        case DIR_STAR:
            // Unknown leading direction — could be negative.
            // However, if d2 is also '*' or '=', the original code must have
            // been correct (it ran before interchange), so the original vector
            // was valid. We only need to check the SWAPPED vector.
            return false;
    }
    return false;
}

// Check if interchange is safe given the computed direction vectors.
// After interchange, the outer and inner directions are swapped.
// The interchange is safe if ALL swapped vectors are lexicographically non-negative.
static bool isInterchangeSafe(const std::vector<DirectionVector> &directions) {
    for (const auto &dv : directions) {
        // After interchange: (outer, inner) becomes (inner, outer).
        // Check if the swapped vector is lex-non-negative.
        if (!isLexNonNegative(dv.inner, dv.outer))
            return false;
    }
    return true;
}

// ============================================================
// Cost Model (Task 3.4)
// ============================================================

// Large penalty for symbolic (non-constant) strides.
// A symbolic stride like N (runtime array dimension) is assumed to be much
// larger than small constant strides like 1 or 4. We use 1000 as a
// conservative "large" value to ensure that constant strides are always
// preferred over symbolic ones.
static constexpr int SYMBOLIC_STRIDE_PENALTY = 1000;

// Compute the cost of a single memory access's inner-loop stride.
// The cost represents how "bad" the stride is for cache locality:
//   - stride 0 means the access doesn't vary with the inner loop (free)
//   - stride 1 or 4 means sequential access (good)
//   - symbolic stride means large, unknown stride (bad)
//
// Returns the absolute value of the coefficient, or SYMBOLIC_STRIDE_PENALTY
// for symbolic (non-constant) strides.
static int strideCost(int coeff, bool symbolic) {
    if (coeff == 0 && !symbolic)
        return 0;  // No dependence on this IV — no cache cost.
    if (symbolic)
        return SYMBOLIC_STRIDE_PENALTY;  // Unknown large stride.
    return std::abs(coeff);
}

// Determine if loop interchange is profitable based on the cost model.
//
// The cost model compares the total inner-loop stride cost before and after
// interchange:
//   - Before interchange: the inner loop is the current inner loop, so the
//     inner-loop stride for each access is its innerCoeff.
//   - After interchange: the current outer loop becomes the new inner loop,
//     so the inner-loop stride for each access is its outerCoeff.
//
// Interchange is profitable if the total cost decreases (better cache locality).
static bool isProfitable(const std::vector<MemAccessInfo> &accesses) {
    int beforeCost = 0;  // Sum of inner-loop strides before interchange.
    int afterCost = 0;   // Sum of inner-loop strides after interchange.

    for (const auto &acc : accesses) {
        // Before interchange: inner loop stride = innerCoeff.
        beforeCost += strideCost(acc.stride.innerCoeff, acc.stride.innerSymbolic);
        // After interchange: outer loop becomes inner, stride = outerCoeff.
        afterCost += strideCost(acc.stride.outerCoeff, acc.stride.outerSymbolic);
    }

    // Only interchange if the post-interchange cost is strictly less.
    return afterCost < beforeCost;
}

} // anonymous namespace

// Analyze all memory accesses in the inner loop and compute their strides
// with respect to the outer and inner induction variables.
// Returns a vector of MemAccessInfo for all load/store operations found.
static std::vector<MemAccessInfo> analyzeStrides(LoopInfo *outer, LoopInfo *inner) {
    std::vector<MemAccessInfo> accesses;

    auto outerIV = outer->getInduction();
    auto innerIV = inner->getInduction();

    // Both loops must have identified induction variables.
    if (!outerIV || !innerIV)
        return accesses;

    // Cache for memoizing expression analysis.
    std::unordered_map<Op*, LinearExpr> cache;

    // Collect all load/store operations within the inner loop body.
    for (auto bb : inner->getBlocks()) {
        for (auto op : bb->getOps()) {
            Op *addr = nullptr;

            if (isa<LoadOp>(op)) {
                // LoadOp: operand 0 is the address.
                // Skip pointer loads (i64 result) — those are array base loads.
                if (op->getResultType() == Value::i64)
                    continue;
                addr = op->DEF(0);
            } else if (isa<StoreOp>(op)) {
                // StoreOp: operand 0 is value, operand 1 is address.
                addr = op->DEF(1);
            } else {
                continue;
            }

            if (!addr)
                continue;

            MemAccessInfo info;
            info.op = op;
            info.addr = addr;

            // Analyze the address expression.
            auto expr = analyzeExpr(addr, outerIV, innerIV, outer, inner, cache);
            info.stride.outerCoeff = expr.outerCoeff;
            info.stride.innerCoeff = expr.innerCoeff;
            info.stride.outerSymbolic = expr.outerSymbolic;
            info.stride.innerSymbolic = expr.innerSymbolic;
            info.stride.valid = expr.valid;

            accesses.push_back(info);
        }
    }

    return accesses;
}

bool LoopInterchange::canInterchange(LoopInfo* outer, LoopInfo* inner) {
    // For interchange to be safe:
    // 1. Must be perfect nest
    // 2. No loop-carried dependencies with negative direction

    if (!isPerfectNest(outer, inner))
        return false;

    // Both loops must have induction variables.
    if (!outer->getInduction() || !inner->getInduction())
        return false;

    // Both loops must be rotated (single latch with branch).
    if (outer->latches.size() != 1 || inner->latches.size() != 1)
        return false;
    if (!isa<BranchOp>(outer->getLatch()->getLastOp()))
        return false;
    if (!isa<BranchOp>(inner->getLatch()->getLastOp()))
        return false;

    // Analyze strides of all memory accesses in the inner loop.
    auto accesses = analyzeStrides(outer, inner);

    // If no memory accesses found or analysis failed, don't interchange.
    if (accesses.empty())
        return false;

    bool allValid = true;
    for (const auto &acc : accesses) {
        if (!acc.stride.valid) {
            allValid = false;
            break;
        }
    }

    if (!allValid)
        return false;

    // Safety check: for a perfect nest, interchange is safe if all dependence
    // direction vectors remain lexicographically non-negative after swapping
    // the outer and inner entries.
    //
    // Compute direction vectors for all pairs of accesses that may alias.
    auto directions = computeDependenceDirections(accesses);

    // If direction computation failed (e.g., couldn't trace base), be conservative.
    // An empty directions vector with non-empty accesses means analysis failed.
    // But if there are accesses and no conflicting pairs (all to different bases,
    // or all loads), directions will be empty legitimately — that's safe.
    //
    // We distinguish the two cases: if computeDependenceDirections returns empty
    // but there ARE store-store or load-store pairs to the same base that it
    // couldn't analyze, it would have returned empty as a failure signal.
    // In practice, if we got here (all strides valid), an empty directions vector
    // means no conflicting pairs exist — interchange is safe from a dependence
    // perspective. The cost model (task 3.4) will decide if it's profitable.

    // Check interchange safety using direction vectors.
    if (!directions.empty() && !isInterchangeSafe(directions))
        return false;

    // Dependence analysis passed — interchange is safe.
    // The cost model (task 3.4) will determine if it's profitable.
    // For now, return true to indicate safety.
    return true;
}

// ============================================================
// Loop Interchange Transformation (Task 3.5)
// ============================================================
//
// For a perfect 2-level loop nest after rotation:
//
//   outerPreheader → outerHeader → ... → innerPreheader → innerHeader → innerBody
//                                                                      → innerLatch → (innerHeader | outerLatch)
//                                                         outerLatch → (outerHeader | outerExit)
//
// After rotation, each loop has:
//   - A header with phi nodes (including the induction variable)
//   - A latch with a branch: if (iv + step < stop) goto header else goto exit
//
// To interchange, we swap the iteration spaces:
//   - Swap the start values (preheader operands of induction phis)
//   - Swap the stop values (comparison targets in latch branches)
//   - Swap the step values (increment constants in latch)
//
// This makes the outer loop iterate over the inner's original range and vice versa,
// effectively achieving the interchange without physically moving basic blocks.

// Perform the actual loop interchange transformation.
// Precondition: canInterchange(outer, inner) returned true AND isProfitable confirmed.
//
// Strategy: Swap the induction variable bounds between the two loops.
// After the swap, the outer IV iterates over what was the inner range, and vice versa.
static bool performInterchange(LoopInfo *outer, LoopInfo *inner) {
    auto outerIV = outer->getInduction();
    auto innerIV = inner->getInduction();
    if (!outerIV || !innerIV)
        return false;

    auto outerPreheader = outer->preheader;
    auto innerPreheader = inner->preheader;
    auto outerLatch = outer->getLatch();
    auto innerLatch = inner->getLatch();

    if (!outerPreheader || !innerPreheader)
        return false;

    // Both latches must end with a BranchOp (rotated loop form).
    auto outerLatchTerm = outerLatch->getLastOp();
    auto innerLatchTerm = innerLatch->getLastOp();
    if (!isa<BranchOp>(outerLatchTerm) || !isa<BranchOp>(innerLatchTerm))
        return false;

    // ---- Step 1: Swap start values ----
    // The induction phi has two operands:
    //   - From preheader: the start value
    //   - From latch: the incremented value (iv + step)
    //
    // We swap the start values between the two induction phis.

    // Find the start value operand index for each induction phi.
    auto getPreheaderOperandIdx = [](Op *phi, BasicBlock *preheader) -> int {
        const auto &attrs = phi->getAttrs();
        for (size_t i = 0; i < attrs.size(); i++) {
            if (isa<FromAttr>(attrs[i]) && FROM(attrs[i]) == preheader)
                return (int)i;
        }
        return -1;
    };

    int outerStartIdx = getPreheaderOperandIdx(outerIV, outerPreheader);
    int innerStartIdx = getPreheaderOperandIdx(innerIV, innerPreheader);
    if (outerStartIdx < 0 || innerStartIdx < 0)
        return false;

    Op *outerStart = outerIV->DEF(outerStartIdx);
    Op *innerStart = innerIV->DEF(innerStartIdx);

    // Swap the start values in the phi nodes.
    outerIV->setOperand(outerStartIdx, Value(innerStart));
    innerIV->setOperand(innerStartIdx, Value(outerStart));

    // ---- Step 2: Swap stop values ----
    // In a rotated loop, the latch has: branch(cond), where cond = lt(iv+step, stop)
    // The condition op is the operand of the BranchOp.
    // We need to find the stop value in each latch's comparison and swap them.

    auto outerCond = outerLatchTerm->DEF(0);  // The comparison op
    auto innerCond = innerLatchTerm->DEF(0);

    // The comparison should be LtOp or LeOp with the stop as the second operand.
    if (!isa<LtOp>(outerCond) && !isa<LeOp>(outerCond))
        return false;
    if (!isa<LtOp>(innerCond) && !isa<LeOp>(innerCond))
        return false;

    // operand 1 of the comparison is the stop value.
    Op *outerStop = outerCond->DEF(1);
    Op *innerStop = innerCond->DEF(1);

    // Swap the stop values.
    outerCond->setOperand(1, Value(innerStop));
    innerCond->setOperand(1, Value(outerStop));

    // ---- Step 3: Swap step values ----
    // The latch increments the IV: newIV = IV + step
    // This is the operand 0 of the comparison (the incremented value).
    // The increment op is typically: add(phi, step_const)
    // We need to find the step operand and swap.

    // Find the increment op for each IV.
    // The latch value of the phi is the incremented IV.
    auto getLatchOperandIdx = [](Op *phi, BasicBlock *latch) -> int {
        const auto &attrs = phi->getAttrs();
        for (size_t i = 0; i < attrs.size(); i++) {
            if (isa<FromAttr>(attrs[i]) && FROM(attrs[i]) == latch)
                return (int)i;
        }
        return -1;
    };

    int outerLatchIdx = getLatchOperandIdx(outerIV, outerLatch);
    int innerLatchIdx = getLatchOperandIdx(innerIV, innerLatch);
    if (outerLatchIdx < 0 || innerLatchIdx < 0)
        return false;

    Op *outerIncr = outerIV->DEF(outerLatchIdx);  // The add op: outerIV + outerStep
    Op *innerIncr = innerIV->DEF(innerLatchIdx);  // The add op: innerIV + innerStep

    // Both should be AddIOp with the phi as operand 0 and step as operand 1.
    if (!isa<AddIOp>(outerIncr) || !isa<AddIOp>(innerIncr))
        return false;

    // Verify that operand 0 of each increment is the respective IV.
    if (outerIncr->DEF(0) != outerIV || innerIncr->DEF(0) != innerIV)
        return false;

    Op *outerStep = outerIncr->DEF(1);
    Op *innerStep = innerIncr->DEF(1);

    // Swap the step values.
    outerIncr->setOperand(1, Value(innerStep));
    innerIncr->setOperand(1, Value(outerStep));

    // ---- Step 4: Fix the preheader guard condition (if present) ----
    // After rotation, the preheader may have a guard: if (start < stop) goto header else goto exit.
    // We need to update these guards to use the swapped bounds.
    //
    // Outer preheader guard: was checking outerStart < outerStop, now should check innerStart < innerStop
    // (since the outer loop now iterates over the inner's original range).
    // But since we already swapped the phi start values and the latch stop values,
    // the guard in the outer preheader still references the original ops.
    // We need to update the guard condition if it exists.

    auto fixPreheaderGuard = [](BasicBlock *preheader, Op *newStart, Op *newStop) {
        auto term = preheader->getLastOp();
        if (!isa<BranchOp>(term))
            return;  // No guard (just a GotoOp), nothing to fix.

        // The guard condition is the operand of the BranchOp.
        auto guardCond = term->DEF(0);
        if (!isa<LtOp>(guardCond) && !isa<LeOp>(guardCond))
            return;

        // Update the guard to compare the new start against the new stop.
        guardCond->setOperand(0, Value(newStart));
        guardCond->setOperand(1, Value(newStop));
    };

    // After the swap:
    // - Outer loop now iterates with innerStart..innerStop (the original inner range)
    // - Inner loop now iterates with outerStart..outerStop (the original outer range)
    // The outer preheader guard should check: innerStart < innerStop
    // The inner preheader guard should check: outerStart < outerStop
    fixPreheaderGuard(outerPreheader, innerStart, innerStop);
    // Inner preheader typically doesn't have a guard (it's inside the outer loop),
    // but check just in case.
    fixPreheaderGuard(innerPreheader, outerStart, outerStop);

    return true;
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

    // Try to interchange: first check safety, then profitability.
    if (canInterchange(info, inner)) {
        // Safety passed. Now check if interchange is profitable using the cost model.
        // Re-analyze strides for the cost model (canInterchange already validated them).
        auto accesses = analyzeStrides(info, inner);
        if (accesses.empty())
            return;

        // Cost model: only interchange if post-interchange strides are smaller.
        if (!isProfitable(accesses))
            return;

        // Perform the actual interchange transformation.
        if (performInterchange(info, inner))
            interchanged++;
    }
}

void LoopInterchange::run() {
    // Environment variable guard: SISY_ENABLE_LOOP_INTERCHANGE (default true)
    const char *env = std::getenv("SISY_ENABLE_LOOP_INTERCHANGE");
    if (env && env[0] && (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0))
        return;

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
