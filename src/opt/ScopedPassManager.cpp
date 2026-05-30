#include "ScopedPassManager.h"

#include <iostream>
#include <set>

namespace sys {

const char *passScopeName(PassScope scope) {
  switch (scope) {
  case PassScope::Module: return "module";
  case PassScope::Function: return "function";
  case PassScope::Loop: return "loop";
  case PassScope::BasicBlock: return "block";
  }
  return "module";
}

const std::vector<ScopedPassInfo> &ScopedPassRegistry::all() {
  static const std::vector<ScopedPassInfo> registry = {
    { "regular-fold", PassScope::Function, {}, { "domtree", "loop" }, true },
    { "pattern-canonicalize", PassScope::Function, {}, {}, true },
    { "mem2reg", PassScope::Function, { "domtree" }, {}, true },
    { "licm", PassScope::Loop, { "loop", "alias", "memoryssa" }, { "domtree" }, false },
    { "inst-schedule", PassScope::BasicBlock, { "alias" }, {}, true },
    { "rv-regalloc", PassScope::Function, { "block-frequency" }, {}, true },
    { "thin-summary", PassScope::Module, { "function-summary" }, { "all" }, false },
  };
  return registry;
}

const ScopedPassInfo *ScopedPassRegistry::find(const std::string &name) {
  for (const auto &info : all())
    if (info.name == name)
      return &info;
  return nullptr;
}

bool ScopedPassRegistry::verify(std::string *error) {
  std::set<std::string> seen;
  for (const auto &info : all()) {
    if (info.name.empty()) {
      if (error)
        *error = "empty scoped pass name";
      return false;
    }
    if (!seen.insert(info.name).second) {
      if (error)
        *error = "duplicate scoped pass: " + info.name;
      return false;
    }
  }
  return true;
}

void ScopedPassRegistry::dump(std::ostream &os) {
  for (const auto &info : all()) {
    os << "[pass-scope] " << passScopeName(info.scope) << " " << info.name;
    if (!info.requiredAnalyses.empty()) {
      os << " requires=";
      for (size_t i = 0; i < info.requiredAnalyses.size(); ++i) {
        if (i)
          os << ",";
        os << info.requiredAnalyses[i];
      }
    }
    if (!info.preservedAnalyses.empty()) {
      os << " preserves=";
      for (size_t i = 0; i < info.preservedAnalyses.size(); ++i) {
        if (i)
          os << ",";
        os << info.preservedAnalyses[i];
      }
    }
    os << " parallel=" << (info.parallelizable ? 1 : 0) << "\n";
  }
}

} // namespace sys
