#include "Analysis.h"

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

using namespace sys;

namespace {

int estimateOpCount(FuncOp *func) {
  int total = 0;
  for (auto region : func->getRegions())
    for (auto bb : region->getBlocks())
      total += (int) bb->getOps().size();
  return total;
}

}  // namespace

void ThinSummary::run() {
  const char *out = std::getenv("SISY_THIN_SUMMARY_DUMP");
  if (!out || !out[0])
    return;

  std::ofstream os(out);
  if (!os.is_open())
    return;

  os << "# sisyphus thin summary v1\n";
  os << "# fields: name, argc, impure, pure, readonly, norecurse, argReadMask,"
        " argWriteMask, opCount, callees\n";

  auto funcs = collectFuncs();
  for (auto func : funcs) {
    const auto &name = NAME(func);
    int argc = func->has<ArgCountAttr>() ? func->get<ArgCountAttr>()->count : -1;
    bool impure = func->has<ImpureAttr>();

    int pure = 0, readonly = 0, norecurse = 0;
    uint64_t argRead = 0, argWrite = 0;
    if (auto s = func->find<FunctionSummaryAttr>()) {
      pure = s->pure;
      readonly = s->readonly;
      norecurse = s->norecurse;
      argRead = s->argReadMask;
      argWrite = s->argWriteMask;
    }

    std::set<std::string> callees;
    for (auto call : func->findAll<CallOp>())
      callees.insert(NAME(call));

    os << name << '\t' << argc << '\t' << impure << '\t' << pure << '\t'
       << readonly << '\t' << norecurse << '\t' << std::hex << argRead << '\t'
       << argWrite << std::dec << '\t' << estimateOpCount(func) << '\t';
    bool first = true;
    for (const auto &c : callees) {
      if (!first) os << ',';
      os << c;
      first = false;
    }
    os << '\n';
    emittedFunctions++;
  }
}
