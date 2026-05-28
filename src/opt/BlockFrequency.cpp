#include "Analysis.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace sys;

namespace {

bool envEnabled(const char *name, bool fallback) {
  const char *raw = std::getenv(name);
  if (!raw || !raw[0])
    return fallback;
  return std::strcmp(raw, "0") != 0 && std::strcmp(raw, "false") != 0 &&
         std::strcmp(raw, "FALSE") != 0;
}

// Singleton side-table.  Stored as relative frequency; entry block ~= 1.0.
std::unordered_map<BasicBlock *, double> &freqMap() {
  static std::unordered_map<BasicBlock *, double> m;
  return m;
}

constexpr double kLoopBoost = 10.0;
constexpr double kColdThreshold = 0.05;
constexpr double kHotThreshold = 100.0;

// Compute loop depth for `bb` by counting strict dominators that are headers
// (have a back-edge: some pred is dominated by them).
int computeLoopDepth(BasicBlock *bb,
                     const std::map<BasicBlock *, bool> &isHeader) {
  int depth = 0;
  for (auto pair : isHeader) {
    if (!pair.second) continue;
    BasicBlock *h = pair.first;
    if (h != bb && bb->dominatedBy(h))
      depth++;
  }
  return depth;
}

// Parse SISY_PROFILE if set: lines of `funcname blockindex freq` (whitespace
// separated, # comments).  Returns nested map[funcname][blockIndex] -> freq.
std::map<std::string, std::map<int, double>> readProfile() {
  std::map<std::string, std::map<int, double>> p;
  const char *path = std::getenv("SISY_PROFILE");
  if (!path || !path[0])
    return p;
  std::ifstream is(path);
  if (!is.is_open())
    return p;
  std::string line;
  while (std::getline(is, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string fn;
    int idx;
    double f;
    if (ss >> fn >> idx >> f)
      p[fn][idx] = f;
  }
  return p;
}

}  // namespace

double BlockFrequency::freqOf(BasicBlock *bb) {
  auto &m = freqMap();
  auto it = m.find(bb);
  return it == m.end() ? 1.0 : it->second;
}

bool BlockFrequency::isCold(BasicBlock *bb) {
  return freqOf(bb) < kColdThreshold;
}

void BlockFrequency::clear() {
  freqMap().clear();
}

void BlockFrequency::run() {
  if (!envEnabled("SISY_ENABLE_BFI", true))
    return;

  freqMap().clear();
  auto profile = readProfile();

  for (auto func : collectFuncs()) {
    auto region = func->getRegion();
    if (!region) continue;

    // Refresh dominator tree so dominatedBy() is accurate at this point.
    (void) getDomTree(region);

    // Identify headers via back-edges.
    std::map<BasicBlock *, bool> isHeader;
    for (auto bb : region->getBlocks()) isHeader[bb] = false;
    for (auto bb : region->getBlocks())
      for (auto pred : bb->preds)
        if (pred->dominatedBy(bb)) {
          isHeader[bb] = true;
          break;
        }

    const auto &fnProfile = profile.count(NAME(func))
        ? profile.at(NAME(func)) : std::map<int, double>{};

    int idx = 0;
    for (auto bb : region->getBlocks()) {
      double f;
      auto pit = fnProfile.find(idx);
      if (pit != fnProfile.end()) {
        f = pit->second;
        profileEntries++;
      } else {
        int depth = computeLoopDepth(bb, isHeader);
        if (depth > maxLoopDepth) maxLoopDepth = depth;
        f = 1.0;
        for (int i = 0; i < depth; i++) f *= kLoopBoost;
      }
      freqMap()[bb] = f;
      if (f >= kHotThreshold) hotBlocks++;
      if (f < kColdThreshold) coldBlocks++;
      idx++;
    }
  }
}
