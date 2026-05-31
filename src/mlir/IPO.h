#pragma once

#include "SelfMLIR.h"

namespace sys::mlir {

void runInlining(Module &module);
void runIPCP(Module &module);
void runPureFunctionDeduction(Module &module);

}
