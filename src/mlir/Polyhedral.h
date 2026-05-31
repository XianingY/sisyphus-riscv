#ifndef POLYHEDRAL_H
#define POLYHEDRAL_H

#include "SelfMLIR.h"

namespace sys::mlir {

void runRaiseToAffine(Module &module);
void runAffineLoopTiling(Module &module);
void runAffineLoopFusion(Module &module);
void runAffineLoopInterchange(Module &module);

}

#endif
