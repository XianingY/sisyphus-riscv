#include "Analysis.h"

#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_set>
#include <vector>

using namespace sys;

namespace {

int estimateOpCount(FuncOp *func) {
  int total = 0;
  for (auto region : func->getRegions())
    for (auto bb : region->getBlocks())
      total += (int) bb->getOps().size();
  return total;
}

std::vector<std::string> splitPathList(const std::string &raw) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : raw) {
    if (c == ',' || c == ':') {
      if (!cur.empty())
        out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  if (!cur.empty())
    out.push_back(cur);
  return out;
}

std::vector<std::string> readSummaryRows(const char *rawPaths) {
  std::vector<std::string> rows;
  if (!rawPaths || !rawPaths[0])
    return rows;
  for (const auto &path : splitPathList(rawPaths)) {
    std::ifstream is(path);
    if (!is.is_open())
      continue;
    std::string line;
    while (std::getline(is, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      rows.push_back(line);
    }
  }
  return rows;
}

}  // namespace

void ThinSummary::run() {
  const char *importPath = std::getenv("SISY_THIN_SUMMARY_IMPORT");
  auto importedRows = readSummaryRows(importPath);
  importedFunctions += (int) importedRows.size();

  const char *out = std::getenv("SISY_THIN_SUMMARY_DUMP");
  const char *linkedOut = std::getenv("SISY_THIN_LINK_OUT");

  std::ostringstream current;

  current << "# sisyphus thin summary v1\n";
  current << "# fields: name, argc, impure, pure, readonly, norecurse, argReadMask,"
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

    current << name << '\t' << argc << '\t' << impure << '\t' << pure << '\t'
            << readonly << '\t' << norecurse << '\t' << std::hex << argRead << '\t'
            << argWrite << std::dec << '\t' << estimateOpCount(func) << '\t';
    bool first = true;
    for (const auto &c : callees) {
      if (!first) current << ',';
      current << c;
      first = false;
    }
    current << '\n';
    emittedFunctions++;
  }

  if (out && out[0]) {
    std::ofstream os(out);
    if (os.is_open())
      os << current.str();
  }

  if (linkedOut && linkedOut[0]) {
    std::ofstream os(linkedOut);
    if (os.is_open()) {
      os << "# sisyphus thin linked summary v1\n";
      os << "# imported=" << importedRows.size()
         << " current=" << emittedFunctions << "\n";
      std::unordered_set<std::string> seen;
      for (const auto &row : importedRows) {
        if (seen.insert(row).second) {
          os << row << '\n';
          linkedFunctions++;
        }
      }
      std::istringstream iss(current.str());
      std::string line;
      while (std::getline(iss, line)) {
        if (line.empty() || line[0] == '#')
          continue;
        if (seen.insert(line).second) {
          os << line << '\n';
          linkedFunctions++;
        }
      }
    }
  }
}
