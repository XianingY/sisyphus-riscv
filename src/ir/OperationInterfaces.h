#ifndef SISY_IR_OPERATION_INTERFACES_H
#define SISY_IR_OPERATION_INTERFACES_H

#include "Operation.h"

namespace sys::ir {

struct PureOpInterface {
  static bool classof(const Operation *op) {
    return op && op->isPure();
  }
};

struct MemoryEffectOpInterface {
  static bool classof(const Operation *op) {
    return op && op->hasMemoryEffects();
  }
};

struct TerminatorOpInterface {
  static bool classof(const Operation *op) {
    return op && op->isTerminator();
  }
};

struct RegionBranchOpInterface {
  static bool classof(const Operation *op) {
    return op && op->isRegionBranch();
  }
};

} // namespace sys::ir

#endif
