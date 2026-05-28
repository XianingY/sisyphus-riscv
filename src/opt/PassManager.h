#ifndef PASS_MANAGER_H
#define PASS_MANAGER_H

#include "Pass.h"
#include "AnalysisManager.h"
#include "../main/Options.h"
#include <memory>
#include <ostream>

namespace sys {

class PassManager {
  std::vector<std::unique_ptr<Pass>> passes;
  ModuleOp *module;

  bool pastFlatten;
  bool pastMem2Reg;
  bool inBackend;
  int exitcode;
  
  std::string input;
  std::string truth;

  Options opts;
  std::unique_ptr<AnalysisManager> analysisManager;
public:
  PassManager(ModuleOp *module, const Options &opts);

  void run();
  void dumpPipelineProfile(std::ostream &os) const;
  ModuleOp *getModule() { return module; }

  template<class T, class... Args>
  void addPass(Args&&... args) {
    passes.emplace_back(std::make_unique<T>(module, std::forward<Args>(args)...));
  }
};

}

#endif
