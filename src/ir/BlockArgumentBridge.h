#ifndef SISY_IR_BLOCK_ARGUMENT_BRIDGE_H
#define SISY_IR_BLOCK_ARGUMENT_BRIDGE_H

#include "../codegen/Ops.h"

#include <iosfwd>

namespace sys::ir {

struct BlockArgumentBridgeStats {
  int phiCandidates = 0;
  int argumentsCreated = 0;
  int rejectedPredMismatch = 0;
  int rejectedMalformedPhi = 0;
  int loweredArguments = 0;
};

class PhiToBlockArgumentBridge {
public:
  static BlockArgumentBridgeStats materialize(ModuleOp *module);
  static BlockArgumentBridgeStats lower(ModuleOp *module);
  static bool roundTrip(ModuleOp *module, std::ostream &os);
};

} // namespace sys::ir

#endif
