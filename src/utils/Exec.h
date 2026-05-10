#ifndef EXEC_H
#define EXEC_H

#include "../codegen/Ops.h"
#include <cstdint>
#include <sstream>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstddef>

namespace sys::exec {

const int CACHE_3_N = 32;
const int CACHE_2_N = 1024;

using cache_3 = int[CACHE_3_N][CACHE_3_N][CACHE_3_N];
using cache_2 = int[CACHE_2_N][CACHE_2_N];

using cache_3_ptr = int(*)[CACHE_3_N][CACHE_3_N];
using cache_2_ptr = int(*)[CACHE_2_N];

const int CACHE_3_TOTAL = sizeof(cache_3) / sizeof(int);
const int CACHE_2_TOTAL = sizeof(cache_2) / sizeof(int);

class Interpreter {
  union Value {
    intptr_t vi;
    float vf;
  };

  using SymbolTable = std::unordered_map<Op*, Value>;

  std::stringstream outbuf, inbuf;
  std::map<std::string, Op*> fnMap;
  std::set<std::string> fpGlobals;
  std::map<std::string, Value> globalMap;
  struct MemoryRange {
    uintptr_t begin;
    uintptr_t end;
  };
  std::vector<MemoryRange> globalRanges;
  std::vector<MemoryRange> stackRanges;
  mutable bool lastValidCached = false;
  mutable uintptr_t lastValidBegin = 0;
  mutable uintptr_t lastValidEnd = 0;

  SymbolTable value;
  // Used for phi functions.
  BasicBlock *prev;
  // Instruction pointer.
  Op *ip;

  intptr_t eval(Op *op);
  float evalf(Op *op);

  void store(Op *op, float v);
  void store(Op *op, intptr_t v);

  void exec(Op *op);
  Value execf(Region *region, const std::vector<Value> &fnArgs);

  Value applyExtern(const std::string &name, const std::vector<Value> &callArgs);
  size_t getAccessSize(Op *op, bool isLoad);
  bool isAddressValid(intptr_t addr, size_t size) const;
  void addRange(std::vector<MemoryRange> &ranges, intptr_t addr, size_t size);

  unsigned retcode;
  size_t stepLimit = 800000000;
  size_t stepCount = 0;
  bool executionTimedOut = false;
  std::unordered_map<Op*, size_t> loadSizeCache;
  std::unordered_map<Op*, size_t> storeSizeCache;
  int *cache = nullptr;
  int cache_type = 0;

  struct SemanticScope {
    Interpreter &parent;
    SymbolTable table;
  public:
    SemanticScope(Interpreter &itp): parent(itp), table(itp.value) {}
    ~SemanticScope() { parent.value = table; }
  };
public:
  Interpreter(ModuleOp *module, size_t explicitStepLimit = 0);
  ~Interpreter();

  void run(std::istream &input);
  void runFunction(const std::string &func, const std::vector<int> &args);
  void useCache(cache_3 cache) { this->cache = (int*) cache; cache_type = 3; }
  void useCache(cache_2 cache) { this->cache = (int*) cache; cache_type = 2; }
  std::string out() { return outbuf.str(); }
  int exitcode() { return retcode & 0xff; }
  int functionResult() const { return static_cast<int32_t>(retcode); }
  bool timedOut() const { return executionTimedOut; }
};

}

#endif
