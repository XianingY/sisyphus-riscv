#include "Passes.h"
#include "PatternRewriter.h"

using namespace sys;

std::map<std::string, int> PatternCanonicalize::stats() {
  return {
    { "attempts", attempts },
    { "rewrites", rewrites },
    { "iterations", iterations },
    { "rejected", rejected },
    { "convergence-bailouts", convergenceBailouts },
  };
}

void PatternCanonicalize::run() {
  std::vector<std::unique_ptr<Pattern>> patterns;
  populateCoreCanonicalizationPatterns(patterns);
  auto result = RewriteDriver::runGreedy(module->getRegion(), patterns);
  attempts += result.attempts;
  rewrites += result.rewrites;
  iterations += result.iterations;
  rejected += result.rejected;
  convergenceBailouts += result.convergenceBailouts;
}

PreservedAnalyses PatternCanonicalize::run(PassContext &) {
  run();
  return rewrites == 0 ? PreservedAnalyses::all() : PreservedAnalyses::none();
}
