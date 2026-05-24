#ifndef BACKEND_SHARED_REG_ALLOC_HOTNESS_H
#define BACKEND_SHARED_REG_ALLOC_HOTNESS_H

#include "../../codegen/CodeGen.h"
#include "../../codegen/Attrs.h"

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace sys::backend::shared {

template <class IsCallLike>
std::unordered_map<BasicBlock*, int> computeBlockHotness(Region *region,
                                                         IsCallLike isCallLike,
                                                         int backEdgeMultiplier = 8,
                                                         int callLikeMultiplier = 2) {
  std::unordered_map<BasicBlock*, int> bbIndex;
  std::unordered_map<BasicBlock*, int> bbWeight;
  std::vector<BasicBlock*> blocks;
  int idx = 0;
  for (auto bb : region->getBlocks()) {
    bbIndex[bb] = idx++;
    blocks.push_back(bb);
    bbWeight[bb] = 1;
  }

  for (auto bb : region->getBlocks()) {
    if (bb->getOpCount() == 0)
      continue;

    auto term = bb->getLastOp();
    auto markNaturalLoop = [&](BasicBlock *target) {
      if (!target || !bbIndex.count(target))
        return;
      int head = bbIndex[target];
      int latch = bbIndex[bb];
      if (head > latch)
        return;
      for (int i = head; i <= latch && i < (int) blocks.size(); i++)
        bbWeight[blocks[i]] = std::min(bbWeight[blocks[i]] * backEdgeMultiplier, 1000000000);
    };

    if (auto target = term->find<TargetAttr>())
      markNaturalLoop(target->bb);
    if (auto ifnot = term->find<ElseAttr>())
      markNaturalLoop(ifnot->bb);
  }

  for (auto bb : region->getBlocks()) {
    for (auto op : bb->getOps()) {
      if (!isCallLike(op))
        continue;
      bbWeight[bb] = std::min(bbWeight[bb] * callLikeMultiplier, 1000000000);
      break;
    }
  }

  return bbWeight;
}

}  // namespace sys::backend::shared

#endif
