#include "CleanupPasses.h"
#include <cstdlib>
#include <cstring>

using namespace sys;

namespace {

struct Associated {
  bool ref;
  int opid;
  std::vector<Op*> mem;
};

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0]) return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0;
}

bool isReassociableAdd(Op *op, bool fastFloat) {
  return isa<AddIOp>(op) || (fastFloat && isa<AddFOp>(op));
}

Op *createLikeAdd(Builder &builder, Op *root, Op *lhs, Op *rhs) {
  if (isa<AddFOp>(root))
    return builder.create<AddFOp>({ lhs->getResult(), rhs });
  return builder.create<AddIOp>({ lhs->getResult(), rhs });
}

} // namespace

void Reassociate::runImpl(Region *region) {
  const bool fastFloat =
    envEnabled("SISY_ENABLE_FAST_FLOAT_REASSOC", false) ||
    envEnabled("SISY_ENABLE_FAST_MATH", false);
  std::map<Op*, Associated> data;
  auto domtree = getDomTree(region);

  std::vector<BasicBlock*> queue { region->getFirstBlock() };
  while (!queue.empty()) {
    auto bb = queue.back();
    queue.pop_back();

    for (auto op : bb->getOps()) {
      if (!isReassociableAdd(op, fastFloat))
        continue;

      auto x = op->DEF(0), y = op->DEF(1);
      std::vector<Op*> mem;
      
      if (data.count(x) && data[x].opid == op->opid)
        std::copy(data[x].mem.begin(), data[x].mem.end(), std::back_inserter(mem)),
        data[x].ref = true;
      else
        mem.push_back(x);
      
      if (data.count(y) && data[y].opid == op->opid)
        std::copy(data[y].mem.begin(), data[y].mem.end(), std::back_inserter(mem)),
        data[y].ref = true;
      else
        mem.push_back(y);

      data[op] = { false, op->opid, mem };
    }

    for (auto child : domtree[bb])
      queue.push_back(child);
  }

  for (auto [k, v] : data) {
    if (v.ref || v.mem.size() == 2)
      continue;

    // We require every addition is used only once.
    auto mem = v.mem;
    bool good = true;
    for (auto op : mem) {
      if (op->getUses().size() > 1) {
        good = false;
        break;
      }
    }
    if (!good)
      continue;

    // Reassociate.
    std::vector<Op*> copy;
    Builder builder;
    builder.setBeforeOp(k);
    while (mem.size() != 1) {
      for (int i = 0; i + 1 < mem.size(); i += 2) {
        auto add = createLikeAdd(builder, k, mem[i], mem[i + 1]);
        copy.push_back(add);
      }
      if (mem.size() & 1)
        copy.push_back(mem.back());
      mem = copy;
      copy.clear();
    }
    auto fulladd = mem[0];
    k->replaceAllUsesWith(fulladd);
    k->erase();
    if (isa<AddFOp>(fulladd))
      floatReassociated++;
    else
      intReassociated++;
  }
}

void Reassociate::run() {
  auto funcs = collectFuncs();

  for (auto func : funcs)
    runImpl(func->getRegion());
}
