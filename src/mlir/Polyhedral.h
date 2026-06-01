#ifndef POLYHEDRAL_H
#define POLYHEDRAL_H

#include "SelfMLIR.h"

namespace sys::mlir {

int runRaiseToAffine(Module &module);
void runAffineLoopTiling(Module &module);
void runAffineLoopFusion(Module &module);
void runAffineLoopInterchange(Module &module);
void runContinueToIfWrap(Module &module);
void runImperfectLoopPromotion(Module &module);

}

#endif
