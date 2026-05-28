#include "Passes.h"
#include "AnalysisManager.h"

#include <cstdlib>
#include <cstring>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

int statValue(const std::map<std::string, int> &stats, const std::string &key) {
  auto it = stats.find(key);
  return it == stats.end() ? 0 : it->second;
}

} // namespace

std::map<std::string, int> LoopReconstruction::stats() {
  return {
    { "candidates", candidates },
    { "reconstructed", reconstructed },
    { "rejected-shape", rejectedShape },
  };
}

void LoopReconstruction::run() {
  if (!envEnabled("SISY_ENABLE_LOOP_RECONSTRUCTION", true))
    return;

  RowScratchMatmul rowScratch(module, /*force=*/ true);
  rowScratch.run();
  auto rowStats = rowScratch.stats();
  candidates += statValue(rowStats, "candidates");
  reconstructed += statValue(rowStats, "replaced");
  rejectedShape += statValue(rowStats, "rejected-shape");
}

PreservedAnalyses LoopReconstruction::run(PassContext &ctx) {
  activeContext = &ctx;
  int before = reconstructed;
  run();
  activeContext = nullptr;
  return reconstructed == before ? PreservedAnalyses::all()
                                 : PreservedAnalyses::none();
}
