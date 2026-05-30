#include "PassManager.h"
#include "Passes.h"
#include "ScopedPassManager.h"
#include "../utils/Exec.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cstdlib>
#include <cstring>

using namespace sys;

PassManager::PassManager(ModuleOp *module, const Options &opts):
  module(module), opts(opts) {
  if (opts.compareWith.size()) {
    std::ifstream ifs(opts.compareWith);
    std::stringstream ss;
    ss << ifs.rdbuf();
    truth = ss.str();

    // Strip the string.
    while (truth.size() && std::isspace(truth.back()))
      truth.pop_back();
    
    // We need to separate the final line.
    auto pos = truth.rfind('\n');
    if (pos == std::string::npos) {
      // This is the only line of file.
      exitcode = std::stoi(truth);
      truth.clear();
    } else {
      exitcode = std::stoi(truth.substr(pos + 1));
      truth.erase(pos);
    }

    // Strip the output again.
    while (truth.size() && std::isspace(truth.back()))
      truth.pop_back();
  }

  if (opts.simulateInput.size()) {
    std::ifstream ifs(opts.simulateInput);
    std::stringstream ss;
    ss << ifs.rdbuf();
    input = ss.str();
  }
}

namespace {
bool envEnabledPM(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}
}

void PassManager::run() {
  pastFlatten = false;
  pastMem2Reg = false;
  inBackend = false;
  bool compareEveryPass = false;
  if (const char *env = std::getenv("SISY_COMPARE_EACH_PASS"))
    compareEveryPass = env[0] && std::strcmp(env, "0") != 0;
  size_t compareStepLimit = 800000000;
  if (const char *env = std::getenv("SISY_COMPARE_STEP_LIMIT")) {
    long long parsed = std::atoll(env);
    if (parsed > 0)
      compareStepLimit = (size_t) parsed;
  }
  auto totalStart = std::chrono::steady_clock::now();
  double totalMs = 0.0;
  bool analysisManagerEnabled =
      envEnabledPM("SISY_ENABLE_ANALYSIS_MANAGER", true);
  bool dumpAnalysisCache =
      envEnabledPM("SISY_DUMP_ANALYSIS_CACHE", false);
  analysisManager =
      std::make_unique<AnalysisManager>(module, analysisManagerEnabled,
                                        dumpAnalysisCache);
  PassContext passContext(analysisManager.get());
  std::string scopedPassError;
  if (!ScopedPassRegistry::verify(&scopedPassError)) {
    std::cerr << "scoped pass registry verification failed: "
              << scopedPassError << "\n";
    std::exit(1);
  }

  for (auto &passPtr : passes) {
    Pass *pass = passPtr.get();
    const auto &pname = pass->name();
    bool transientIRPass =
      pname == "range" ||
      pname == "range-aware-fold" ||
      pname == "splice";

    if (pname == "flatten-cfg")
      pastFlatten = true;
    if (pname == "mem2reg")
      pastMem2Reg = true;
    if (pname == "rv-lower" || pname == "arm-lower")
      inBackend = true;

    bool verifyThisPass = opts.verify && pastMem2Reg && !transientIRPass;
    // Compare on the stabilized checkpoint before backend to avoid
    // interpreter false positives and excessive N-pass replay cost.
    bool compareThisPass = opts.compareWith.size() && pastFlatten && !inBackend &&
      (compareEveryPass ? !transientIRPass : pname == "inst-schedule");
    if (opts.o2) {
      // O2 includes aggressive transformations with transient SSA/phi states.
      verifyThisPass = opts.verify && pastMem2Reg && pname == "inst-schedule";
    }

    if (pname == opts.printBefore) {
      std::cerr << "===== Before " << pname << " =====\n\n";
      module->dump(std::cerr);
      std::cerr << "\n\n";
    }

    auto passStart = std::chrono::steady_clock::now();
    auto *scopeInfo = ScopedPassRegistry::find(pname);
    if (opts.dumpPassTiming) {
      if (scopeInfo) {
        std::cerr << "[pass-scope-check] " << pname
                  << " scope=" << passScopeName(scopeInfo->scope)
                  << " declared=1\n";
      } else {
        std::cerr << "[pass-scope-check] " << pname
                  << " scope=legacy declared=0\n";
      }
    }
    PreservedAnalyses preserved =
        analysisManagerEnabled ? pass->run(passContext)
                               : (pass->run(), PreservedAnalyses::none());
    pass->cleanup();
    if (analysisManagerEnabled)
      analysisManager->invalidate(preserved, pname);
    auto passEnd = std::chrono::steady_clock::now();
    auto passMs = std::chrono::duration<double, std::milli>(passEnd - passStart).count();
    totalMs += passMs;

    if (opts.verbose || pname == opts.printAfter) {
      std::cerr << "===== After " << pname << " =====\n\n";
      module->dump(std::cerr);
      std::cerr << "\n\n";
    }

    // Before mem2reg, we don't have phis.
    // Verify pass only checks phis; so no point running it before that.
    if (verifyThisPass) {
      std::cerr << "checking " << pname << "...";
      Verify(module).run();
      std::cerr << " passed\n";
    }

    // We can't simulate for backend.
    // Technically we have the capacity, but it's too much work.
    if (compareThisPass) {
      std::cerr << "checking " << pname << "\n";
      exec::Interpreter itp(module, compareStepLimit);
      std::stringstream buffer(input);
      itp.run(buffer);
      if (itp.timedOut()) {
        std::cerr << "compare timed out after step budget in pass: " << pname << "\n";
        std::exit(1);
      }
      std::string str = itp.out();
      // Strip output.
      while (str.size() && std::isspace(str.back()))
        str.pop_back();

      if (str != truth) {
        std::cerr << "output mismatch:\n" << str << "\n";
        std::cerr << "after pass: " << pname << "\n";
        std::exit(1);
      }
      if (exitcode != itp.exitcode()) {
        std::cerr << "exit code mismatch:" << itp.exitcode() << " (expected " << exitcode << ")\n";
        std::cerr << "after pass: " << pname << "\n";
        std::exit(1);
      }
    }
    
    if (opts.stats) {
      std::cerr << pname << ":\n";

      auto stats = pass->stats();
      if (!stats.size())
        std::cerr << "  <no stats>\n";

      for (auto [k, v] : stats)
        std::cerr << "  " << k << " : " << v << "\n";
    }

    if (opts.dumpPassTiming)
      std::cerr << "[pass-timing] " << pname << " : " << passMs << " ms\n";
  }

  if (opts.dumpPassTiming) {
    auto totalEnd = std::chrono::steady_clock::now();
    auto wallMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    std::cerr << "[pass-timing] total-pass-time : " << totalMs << " ms\n";
    std::cerr << "[pass-timing] total-wall-time : " << wallMs << " ms\n";
  }
  if (dumpAnalysisCache && analysisManager) {
    analysisManager->getDataLayout();
    analysisManager->getLegacyAffineNestSummary();
    analysisManager->getMemRefAlias();
    analysisManager->dumpStats(std::cerr);
  }
}

void PassManager::dumpPipelineProfile(std::ostream &os) const {
  os << "===== Pipeline Profile =====\n";
  for (size_t i = 0; i < passes.size(); i++)
    os << i << ": " << passes[i]->name() << "\n";
}
