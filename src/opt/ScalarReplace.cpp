#include "LoopPasses.h"
#include "Passes.h"
#include "CleanupPasses.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

using namespace sys;

// ScalarReplace pass
// Promotes loop-invariant load/store patterns to scalar registers.
// For a loop where the same address is repeatedly loaded, computed on, and stored,
// this pass hoists the load to the preheader, replaces in-loop accesses with a phi,
// and sinks the store to the exit block.
//
// Runs after LICM.

std::map<std::string, int> ScalarReplace::stats() {
    return {
        {"detected", detected},
        {"promoted", promoted},
        {"arrays-scalarized", arraysScalarized},
        {"array-elements", arrayElements},
        {"array-accesses", arrayAccesses}
    };
}

namespace {

// A candidate for scalar replacement: an address that is loop-invariant
// and has both loads and stores in the loop body.
struct ScalarCandidate {
    Op *addr;                    // The loop-invariant address SSA value
    std::vector<Op*> loads;      // All LoadOps using this address in the loop
    std::vector<Op*> stores;     // All StoreOps to this address in the loop
};

struct ArrayElementAccess {
    Op *op = nullptr;
    bool store = false;
    int offset = 0;
    size_t size = 4;
};

int getenvInt(const char *name, int fallback, int minValue, int maxValue) {
    const char *raw = std::getenv(name);
    if (!raw || !raw[0])
        return fallback;
    char *end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if ((end && *end) || value < minValue || value > maxValue)
        return fallback;
    return (int) value;
}

size_t memSize(Op *op) {
    auto *size = op->find<SizeAttr>();
    return size ? size->value : 4;
}

bool constantOffsetFromBase(Op *addr, Op *base, int &offset,
                            std::unordered_set<Op*> &visiting) {
    if (!addr || !base)
        return false;
    if (addr == base) {
        offset = 0;
        return true;
    }
    if (visiting.count(addr))
        return false;
    visiting.insert(addr);

    auto tryAdd = [&](Op *lhs, Op *rhs) -> bool {
        if (!isa<IntOp>(lhs))
            return false;
        int rhsOffset = 0;
        if (!constantOffsetFromBase(rhs, base, rhsOffset, visiting))
            return false;
        offset = rhsOffset + V(lhs);
        return true;
    };

    bool ok = false;
    if ((isa<AddLOp>(addr) || isa<AddIOp>(addr)) && addr->getOperandCount() == 2) {
        ok = tryAdd(addr->DEF(0), addr->DEF(1)) ||
             tryAdd(addr->DEF(1), addr->DEF(0));
    }

    visiting.erase(addr);
    return ok;
}

bool constantOffsetFromBase(Op *addr, Op *base, int &offset) {
    std::unordered_set<Op*> visiting;
    return constantOffsetFromBase(addr, base, offset, visiting);
}

bool valueEscapesArray(Op *value, Op *base) {
    int ignored = 0;
    return constantOffsetFromBase(value, base, ignored);
}

bool collectArrayElementAccesses(Op *addr, Op *base,
                                 std::vector<ArrayElementAccess> &accesses,
                                 std::unordered_set<Op*> &seen) {
    if (!seen.insert(addr).second)
        return true;

    int offset = 0;
    if (!constantOffsetFromBase(addr, base, offset))
        return false;

    for (auto *use : addr->getUses()) {
        if (isa<AddLOp>(use) || isa<AddIOp>(use)) {
            int childOffset = 0;
            if (!constantOffsetFromBase(use, base, childOffset))
                return false;
            if (!collectArrayElementAccesses(use, base, accesses, seen))
                return false;
            continue;
        }

        if (auto *load = dyn_cast<LoadOp>(use)) {
            if (load->getOperandCount() != 1 || load->DEF(0) != addr)
                return false;
            // A pointer-typed load means a partial array address escaped.
            if (load->getResultType() == Value::i64)
                return false;
            size_t size = memSize(load);
            if (size == 0 || size > 4)
                return false;
            accesses.push_back({load, false, offset, size});
            continue;
        }

        if (auto *store = dyn_cast<StoreOp>(use)) {
            if (store->getOperandCount() < 2)
                return false;
            if (store->DEF(0) == addr || valueEscapesArray(store->DEF(0), base))
                return false;
            if (store->DEF(1) != addr)
                return false;
            size_t size = memSize(store);
            if (size == 0 || size > 4)
                return false;
            accesses.push_back({store, true, offset, size});
            continue;
        }

        return false;
    }

    return true;
}

// Check if an operation is loop-invariant with respect to the given loop.
// An op is loop-invariant if:
//   - It is defined outside the loop (its parent block is not in the loop), OR
//   - All of its operands are loop-invariant (recursively)
bool isLoopInvariant(Op *op, LoopInfo *loop, std::unordered_map<Op*, bool> &cache) {
    if (!op)
        return false;

    auto it = cache.find(op);
    if (it != cache.end())
        return it->second;

    // Prevent infinite recursion on cycles (phi nodes).
    // Conservatively assume variant until proven otherwise.
    cache[op] = false;

    auto parent = op->getParent();
    // If defined outside the loop, it's invariant.
    if (!parent || !loop->contains(parent)) {
        cache[op] = true;
        return true;
    }

    // Phi nodes at the loop header are loop-variant (they change each iteration).
    if (isa<PhiOp>(op) && parent == loop->header) {
        cache[op] = false;
        return false;
    }

    // Loads and stores are not invariant (they have side effects / depend on memory).
    if (isa<LoadOp>(op) || isa<StoreOp>(op)) {
        cache[op] = false;
        return false;
    }

    // Calls with side effects are not invariant.
    if (isa<CallOp>(op) && op->has<ImpureAttr>()) {
        cache[op] = false;
        return false;
    }

    // For other ops defined inside the loop, check if all operands are loop-invariant.
    for (auto operand : op->getOperands()) {
        if (!isLoopInvariant(operand.defining, loop, cache)) {
            cache[op] = false;
            return false;
        }
    }

    cache[op] = true;
    return true;
}

// Check if a store depends on a load (load→compute→store pattern).
// This verifies that the stored value is derived from the loaded value,
// which is the pattern we want to promote to a scalar accumulator.
bool storeUsesLoad(Op *store, Op *load, LoopInfo *loop,
                   std::unordered_set<Op*> &visited) {
    if (!store || !load)
        return false;

    // StoreOp operand 0 is the value being stored.
    Op *storedVal = store->DEF(0);
    if (!storedVal)
        return false;

    // BFS/DFS from the stored value back to the load.
    std::vector<Op*> worklist;
    worklist.push_back(storedVal);

    while (!worklist.empty()) {
        Op *cur = worklist.back();
        worklist.pop_back();

        if (!cur)
            continue;
        if (visited.count(cur))
            continue;
        visited.insert(cur);

        if (cur == load)
            return true;

        // Only follow operands of ops inside the loop.
        auto parent = cur->getParent();
        if (!parent || !loop->contains(parent))
            continue;

        for (auto operand : cur->getOperands())
            worklist.push_back(operand.defining);
    }

    return false;
}

// Check alias safety for a candidate.
// Returns true if no other store in the loop may alias the candidate address,
// and no impure call exists that could modify memory.
// This is conservative: if we can't prove non-aliasing, we assume aliasing (unsafe).
bool isAliasSafe(const ScalarCandidate &candidate, LoopInfo *loop) {
    Op *candidateAddr = candidate.addr;

    // Collect the candidate's own stores into a set for quick lookup.
    std::unordered_set<Op*> candidateStores(candidate.stores.begin(),
                                            candidate.stores.end());

    // Check all operations in the loop.
    for (auto bb : loop->getBlocks()) {
        for (auto op : bb->getOps()) {
            // Check for impure calls — these could modify any memory location.
            if (isa<CallOp>(op) && op->has<ImpureAttr>())
                return false;

            // Check other stores (not the candidate's own stores).
            if (isa<StoreOp>(op) && !candidateStores.count(op)) {
                Op *otherAddr = op->DEF(1);
                if (!otherAddr)
                    return false;

                // Use the alias analysis infrastructure.
                // If both addresses have AliasAttr, use neverAlias to check.
                // If either lacks AliasAttr, we can't prove non-aliasing → unsafe.
                if (!candidateAddr->has<AliasAttr>() || !otherAddr->has<AliasAttr>())
                    return false;

                // If the other store's address has unknown alias, it could alias anything.
                if (ALIAS(otherAddr)->unknown)
                    return false;

                // If the candidate address has unknown alias, it could be aliased by anything.
                if (ALIAS(candidateAddr)->unknown)
                    return false;

                // Check if the two addresses may alias.
                if (sys::mayAlias(candidateAddr, otherAddr))
                    return false;
            }
        }
    }

    return true;
}

// Collect scalar replacement candidates for a given loop.
// A candidate is an address that:
//   1. Is loop-invariant (doesn't change across iterations)
//   2. Has at least one load and one store in the loop body
//   3. The store depends on the load (load→compute→store pattern)
std::vector<ScalarCandidate> collectCandidates(LoopInfo *loop) {
    // Map from address SSA value to its loads and stores.
    std::unordered_map<Op*, ScalarCandidate> addrMap;
    // Cache for loop-invariance queries.
    std::unordered_map<Op*, bool> invariantCache;

    // Collect all loads and stores in the loop, grouped by address.
    for (auto bb : loop->getBlocks()) {
        for (auto op : bb->getOps()) {
            if (isa<LoadOp>(op)) {
                // Skip pointer loads (i64 result) — those are array base loads,
                // not the data loads we want to promote.
                if (op->getResultType() == Value::i64)
                    continue;

                Op *addr = op->DEF(0);
                if (!addr)
                    continue;

                // Check if the address is loop-invariant.
                if (!isLoopInvariant(addr, loop, invariantCache))
                    continue;

                addrMap[addr].addr = addr;
                addrMap[addr].loads.push_back(op);
            } else if (isa<StoreOp>(op)) {
                // StoreOp: operand 0 is value, operand 1 is address.
                Op *addr = op->DEF(1);
                if (!addr)
                    continue;

                // Check if the address is loop-invariant.
                if (!isLoopInvariant(addr, loop, invariantCache))
                    continue;

                addrMap[addr].addr = addr;
                addrMap[addr].stores.push_back(op);
            }
        }
    }

    // Filter: keep only candidates that have both loads and stores,
    // and where at least one store depends on a load (load→compute→store).
    std::vector<ScalarCandidate> candidates;
    for (auto &[addr, candidate] : addrMap) {
        if (candidate.loads.empty() || candidate.stores.empty())
            continue;

        // Verify the load→compute→store dependency.
        bool hasDep = false;
        for (auto store : candidate.stores) {
            std::unordered_set<Op*> visited;
            for (auto load : candidate.loads) {
                if (storeUsesLoad(store, load, loop, visited)) {
                    hasDep = true;
                    break;
                }
            }
            if (hasDep)
                break;
        }

        if (hasDep)
            candidates.push_back(candidate);
    }

    return candidates;
}

// Promote a single candidate to a scalar register.
// This performs the actual transformation:
//   1. Insert a load in the preheader to get the initial value
//   2. Create a phi in the loop header (init from preheader, updated value from latch)
//   3. Replace all in-loop loads with the phi
//   4. Remove in-loop loads
//   5. Create an LCSSA phi at the exit block and insert a store
//   6. Remove in-loop stores
void promoteCandidate(const ScalarCandidate &candidate, LoopInfo *info) {
    Builder builder;

    auto *preheader = info->preheader;
    auto *header = info->header;
    auto *latch = *info->latches.begin();
    auto *exit = *info->exits.begin();

    Op *addr = candidate.addr;

    // Determine the result type and size from the first load.
    Op *firstLoad = candidate.loads[0];
    auto resultType = firstLoad->getResultType();
    auto *sizeAttr = firstLoad->find<SizeAttr>();
    size_t size = sizeAttr ? sizeAttr->value : 4;

    // Step 1: Insert a load in the preheader to get the initial value.
    // Must be inserted BEFORE the terminator (GotoOp) of the preheader.
    builder.setBeforeOp(preheader->getLastOp());
    auto *initLoad = builder.create<LoadOp>(resultType, { addr }, { new SizeAttr(size) });

    // Step 2: Create a phi in the loop header.
    // The phi has two incoming values:
    //   - From preheader: the initial load value
    //   - From latch: the value being stored (the accumulator after each iteration)
    // Phis must be at the start of the block, before any non-phi ops.
    builder.setToBlockStart(header);
    auto *phi = builder.create<PhiOp>({ initLoad }, { new FromAttr(preheader) });
    // Set the correct result type (PhiOp defaults to i32, but may need f32 etc.)
    phi->setResultType(resultType);

    // Find the value that the stores write (the "new" value computed in the loop body).
    // This is the value that should come from the latch edge of the phi.
    // For the typical load→compute→store pattern, there is one store whose stored
    // value is the result of the computation. We use the last store's stored value.
    // StoreOp: operand 0 is the value being stored, operand 1 is the address.
    Op *latchVal = candidate.stores.back()->DEF(0);

    // Add the latch incoming edge to the phi.
    phi->pushOperand(latchVal);
    phi->add<FromAttr>(latch);

    // Step 3: Replace all uses of in-loop loads with the phi.
    // The phi represents the current value of the scalar accumulator.
    for (auto *load : candidate.loads) {
        load->replaceAllUsesWith(phi);
    }

    // Step 4: Remove the in-loop loads (they are now replaced by the phi).
    for (auto *load : candidate.loads) {
        load->erase();
    }

    // Step 5: Insert a store in the exit block to write back the final value.
    // We need an LCSSA phi at the exit block to capture the phi value from
    // all exiting blocks, then store that value back to the original address.
    //
    // The LCSSA phi collects the accumulator value from each exiting block:
    // - If exiting from the header (loop condition false on first check):
    //   the phi has the initial value (initLoad)
    // - If exiting from the latch (loop condition false after an iteration):
    //   the phi has the value from the last completed iteration
    builder.setToBlockStart(exit);
    auto *lcssaPhi = builder.create<PhiOp>();
    lcssaPhi->setResultType(resultType);

    // Add incoming edges from all exiting blocks.
    for (auto *exiting : info->exitings) {
        lcssaPhi->pushOperand(phi);
        lcssaPhi->add<FromAttr>(exiting);
    }

    // Insert the store after all phis in the exit block (including our new LCSSA phi).
    // Use the first non-phi op as the insertion point.
    // We need to find the first non-phi op AFTER we've inserted our lcssaPhi.
    // Since lcssaPhi was inserted at the start, we scan from the beginning.
    Op *insertBefore = exit->getFirstOp();
    while (!insertBefore->atBack() && isa<PhiOp>(insertBefore))
        insertBefore = insertBefore->nextOp();
    // If the last op is also a phi (degenerate block), insert before it.
    // Otherwise insert before the first non-phi op.
    if (isa<PhiOp>(insertBefore))
        builder.setAfterOp(insertBefore);
    else
        builder.setBeforeOp(insertBefore);

    builder.create<StoreOp>({ lcssaPhi, addr }, { new SizeAttr(size) });

    // Step 6: Remove the in-loop stores (they are now replaced by the phi + exit store).
    for (auto *store : candidate.stores) {
        store->erase();
    }
}

} // anonymous namespace

void ScalarReplace::runArrayScalarization(FuncOp *func) {
    const int maxBytes =
        getenvInt("SISY_SCALAR_REPLACE_MAX_ARRAY_BYTES", 64, 4, 4096);
    auto *region = func->getRegion();
    Builder builder;

    auto allocas = func->findAll<AllocaOp>();
    for (auto *alloca : allocas) {
        if (!alloca->has<DimensionAttr>())
            continue;
        int bytes = (int) SIZE(alloca);
        if (bytes <= 0 || bytes > maxBytes || bytes % 4 != 0)
            continue;

        std::vector<ArrayElementAccess> accesses;
        std::unordered_set<Op*> seen;
        if (!collectArrayElementAccesses(alloca, alloca, accesses, seen))
            continue;
        if (accesses.empty())
            continue;

        bool valid = true;
        for (const auto &access : accesses) {
            if (access.offset < 0 || access.offset + (int) access.size > bytes ||
                access.offset % 4 != 0) {
                valid = false;
                break;
            }
        }
        if (!valid)
            continue;

        const int elems = bytes / 4;
        std::vector<Op*> scalarSlots(elems, nullptr);
        builder.setBeforeOp(alloca);
        for (int i = 0; i < elems; i++) {
            auto *slot = builder.create<AllocaOp>({ new SizeAttr(4) });
            if (alloca->has<FPAttr>())
                slot->add<FPAttr>();
            scalarSlots[i] = slot;
        }

        for (const auto &access : accesses) {
            Op *slot = scalarSlots[access.offset / 4];
            access.op->setOperand(access.store ? 1 : 0, slot);
            arrayAccesses++;
        }

        arraysScalarized++;
        arrayElements += elems;
    }

    region->updatePreds();
}

void ScalarReplace::runImpl(LoopInfo *info) {
    // Process subloops first (innermost-first order).
    for (auto *sub : info->subloops)
        runImpl(sub);

    // Must have a preheader and a single exit for the transformation.
    if (!info->preheader)
        return;
    if (info->exits.size() != 1)
        return;
    if (info->latches.size() != 1)
        return;

    // Task 4.2: Find load/store pairs to loop-invariant addresses.
    auto candidates = collectCandidates(info);
    detected += candidates.size();

    // Task 4.3: Filter candidates by alias safety.
    // Remove candidates where another store in the loop may alias the target address.
    std::vector<ScalarCandidate> safeCandidates;
    for (auto &candidate : candidates) {
        if (isAliasSafe(candidate, info))
            safeCandidates.push_back(candidate);
    }
    candidates = std::move(safeCandidates);

    // Task 4.4: Promote safe candidates to scalar registers.
    for (auto &candidate : candidates) {
        promoteCandidate(candidate, info);
        promoted++;
    }
}

void ScalarReplace::run() {
    // Environment variable guard: SISY_ENABLE_SCALAR_REPLACE (default true)
    const char *env = std::getenv("SISY_ENABLE_SCALAR_REPLACE");
    if (env && env[0] && (std::strcmp(env, "0") == 0 || std::strcmp(env, "false") == 0))
        return;

    int arraysBefore = arraysScalarized;
    for (auto func : collectFuncs())
        runArrayScalarization(func);
    if (arraysScalarized != arraysBefore) {
        Mem2Reg promote(module);
        promote.run();
        DCE cleanup(module);
        cleanup.run();
    }

    LoopAnalysis analysis(module);
    analysis.run();
    auto forests = analysis.getResult();

    for (auto func : collectFuncs()) {
        auto &forest = forests[func];

        for (auto loop : forest.getLoops()) {
            if (!loop->getParent())
                runImpl(loop);
        }
    }
}
