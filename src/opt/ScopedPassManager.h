#ifndef SISY_SCOPED_PASS_MANAGER_H
#define SISY_SCOPED_PASS_MANAGER_H

#include <iosfwd>
#include <string>
#include <vector>

namespace sys {

enum class PassScope {
  Module,
  Function,
  Loop,
  BasicBlock,
};

struct ScopedPassInfo {
  std::string name;
  PassScope scope = PassScope::Module;
  std::vector<std::string> requiredAnalyses;
  std::vector<std::string> preservedAnalyses;
  bool parallelizable = false;
};

class ScopedPassRegistry {
public:
  static const std::vector<ScopedPassInfo> &all();
  static const ScopedPassInfo *find(const std::string &name);
  static bool verify(std::string *error = nullptr);
  static void dump(std::ostream &os);
};

const char *passScopeName(PassScope scope);

} // namespace sys

#endif
