#include "IPO.h"

namespace sys::mlir {

void runInlining(Module &module) {
  // Find sysy.call operations and replace them with the body of the called function
  // if the function is small enough.
}

void runIPCP(Module &module) {
  // Propagate constants across function boundaries.
}

void runPureFunctionDeduction(Module &module) {
  // Deduce pure functions (no side effects) and add "pure" attribute.
}

}
