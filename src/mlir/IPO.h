#pragma once

#include "SelfMLIR.h"

namespace sys::mlir {

void runInlining(Module &module, int threshold = 200, SelfOptStats *stats = nullptr);
void runIPCP(Module &module);
void runPureFunctionDeduction(Module &module);

}
