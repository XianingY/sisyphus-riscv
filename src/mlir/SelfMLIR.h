#ifndef SISY_SELF_MLIR_H
#define SISY_SELF_MLIR_H

#include <cstdint>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


namespace sys::mlir {

enum class TypeKind {
  None,
  Integer,
  Float,
  Index,
  MemRef,
  Vector,
  Function,
  Register,
};

struct TypeStorage;
struct AttributeStorage;
struct LocationStorage;

class Type {
  const TypeStorage *storage = nullptr;

public:
  explicit Type(const TypeStorage *storage = nullptr): storage(storage) {}
  explicit operator bool() const { return storage != nullptr; }
  bool operator==(Type other) const { return storage == other.storage; }
  bool operator!=(Type other) const { return storage != other.storage; }
  bool operator<(Type other) const { return storage < other.storage; }
  TypeKind kind() const;
  std::string str() const;
};

class Attribute {
  const AttributeStorage *storage = nullptr;

public:
  explicit Attribute(const AttributeStorage *storage = nullptr): storage(storage) {}
  explicit operator bool() const { return storage != nullptr; }
  bool operator==(Attribute other) const { return storage == other.storage; }
  bool operator!=(Attribute other) const { return storage != other.storage; }
  std::string str() const;
};

class Location {
  const LocationStorage *storage = nullptr;

public:
  explicit Location(const LocationStorage *storage = nullptr): storage(storage) {}
  explicit operator bool() const { return storage != nullptr; }
  bool operator==(Location other) const { return storage == other.storage; }
  bool operator!=(Location other) const { return storage != other.storage; }
  std::string str() const;
};

class Context {
  std::unordered_map<std::string, std::unique_ptr<TypeStorage>> types;
  std::unordered_map<std::string, std::unique_ptr<AttributeStorage>> attrs;
  std::unordered_map<std::string, std::unique_ptr<LocationStorage>> locations;

  Type internType(TypeKind kind, std::string key, std::string text);
  Attribute internAttr(std::string key, std::string text);

public:
  Context();
  ~Context();

  Type noneType();
  Type i(unsigned bits);
  Type f(unsigned bits);
  Type index();
  Type memref(const std::vector<int64_t> &shape, Type element,
              const std::vector<int64_t> &strides = {},
              const std::string &layout = "identity",
              bool readonly = false);
  Type vector(Type element, int64_t lanes, bool scalable = false);
  Type function(const std::vector<Type> &inputs, const std::vector<Type> &results);
  Type reg(const std::string &target, const std::string &regClass);

  Attribute integerAttr(int64_t value, Type type);
  Attribute stringAttr(const std::string &value);
  Attribute boolAttr(bool value);
  Location loc(const std::string &file, int line, int column);
  Location unknownLoc();

  std::size_t typeCount() const { return types.size(); }
  std::size_t attrCount() const { return attrs.size(); }
  std::size_t locationCount() const { return locations.size(); }
};

class Operation;

struct Use {
  Operation *owner = nullptr;
  int operandIndex = -1;
};

class Region;
class Block;
class BlockArgument;

class Value {
  Operation *owner = nullptr;
  BlockArgument *argument = nullptr;
  unsigned resultIndex = 0;
  Type valueType;

public:
  Value() = default;
  static Value result(Operation *owner, unsigned resultIndex, Type type);
  static Value blockArgument(BlockArgument *arg);

  bool valid() const { return owner || argument; }
  bool isOperationResult() const { return owner != nullptr; }
  bool isBlockArgument() const { return argument != nullptr; }
  Operation *getDefiningOp() const { return owner; }
  BlockArgument *getBlockArgument() const { return argument; }
  unsigned getResultIndex() const { return resultIndex; }
  Type type() const { return valueType; }
  std::string identityKey() const;
  std::string printName() const;

  bool operator==(Value other) const {
    return owner == other.owner && argument == other.argument &&
           resultIndex == other.resultIndex;
  }
  bool operator!=(Value other) const { return !(*this == other); }
};

class BlockArgument {
  Block *owner = nullptr;
  unsigned index = 0;
  Type argType;
  Location argLoc;
  std::string argName;

public:
  std::vector<Use> uses;

  BlockArgument(Block *owner, unsigned index, Type type, Location loc,
                std::string name);
  Block *getOwner() const { return owner; }
  unsigned getIndex() const { return index; }
  Type type() const { return argType; }
  Location loc() const { return argLoc; }
  const std::string &name() const { return argName; }
  Value value() const;
};

class Operation {
  std::string opName;
  std::vector<Value> operands;
  std::vector<Type> resultTypes;
  std::map<std::string, Attribute> attributes;
  std::vector<std::unique_ptr<Region>> regions;
  Location opLoc;
  Block *parent = nullptr;
  bool erased = false;

public:
  std::vector<std::vector<Use>> resultUses;

  Operation(std::string name, std::vector<Value> operands,
            std::vector<Type> results, std::map<std::string, Attribute> attrs,
            Location loc);
  ~Operation();

  const std::string &name() const { return opName; }
  std::string dialect() const;
  void rename(std::string newName) { opName = std::move(newName); }
  Location loc() const { return opLoc; }
  Block *getBlock() const { return parent; }
  void setBlock(Block *block) { parent = block; }

  int operandCount() const { return (int) operands.size(); }
  Value operand(int index) const { return operands[index]; }
  void setOperand(int index, Value value);
  void addOperand(Value value);
  const std::vector<Value> &getOperands() const { return operands; }

  int resultCount() const { return (int) resultTypes.size(); }
  Type resultType(int index = 0) const { return resultTypes[index]; }
  Value result(int index = 0) { return Value::result(this, index, resultTypes[index]); }

  const std::map<std::string, Attribute> &attrs() const { return attributes; }
  Attribute attr(const std::string &name) const;
  void setAttr(std::string name, Attribute attr) {
    attributes[std::move(name)] = attr;
  }

  Region &addRegion();
  const std::vector<std::unique_ptr<Region>> &getRegions() const { return regions; }
  std::vector<std::unique_ptr<Region>> &getRegions() { return regions; }

  bool isSymbol() const;
  std::string getSymbolName() const;
  bool isLoop() const;
  bool isTerminator() const;
  bool isErased() const { return erased; }
  void dropAllOperands();
  void markErased();
};

class Block {
  Region *parent = nullptr;
  std::vector<std::unique_ptr<BlockArgument>> arguments;
  std::vector<std::unique_ptr<Operation>> operations;

public:
  explicit Block(Region *parent): parent(parent) {}
  Region *getRegion() const { return parent; }
  BlockArgument &addArgument(Type type, Location loc, const std::string &name = "");
  Operation &addOperation(std::unique_ptr<Operation> op);
  Operation &insertOperation(std::size_t index, std::unique_ptr<Operation> op);
  std::unique_ptr<Operation> takeOperation(Operation *op);
  void eraseMarkedOperations();
  const std::vector<std::unique_ptr<BlockArgument>> &args() const { return arguments; }
  const std::vector<std::unique_ptr<Operation>> &ops() const { return operations; }
  std::vector<std::unique_ptr<Operation>> &ops() { return operations; }
};

class Region {
  Operation *parent = nullptr;
  std::vector<std::unique_ptr<Block>> blocks;

public:
  explicit Region(Operation *parent): parent(parent) {}
  Operation *getParent() const { return parent; }
  Block &addBlock();
  const std::vector<std::unique_ptr<Block>> &getBlocks() const { return blocks; }
  std::vector<std::unique_ptr<Block>> &getBlocks() { return blocks; }
};

class Module {
  Context &ctx;
  std::unique_ptr<Operation> moduleOp;

public:
  explicit Module(Context &ctx);
  Context &context() const { return ctx; }
  Operation &op() const { return *moduleOp; }
  Region &body();
};

class Builder {
  Context &ctx;
  Block *insertionBlock = nullptr;

public:
  Builder(Context &ctx, Block *block = nullptr): ctx(ctx), insertionBlock(block) {}
  Block *getInsertionBlock() const { return insertionBlock; }
  void setInsertionBlock(Block *block) { insertionBlock = block; }
  Operation &create(const std::string &name, const std::vector<Value> &operands,
                    const std::vector<Type> &results,
                    const std::map<std::string, Attribute> &attrs = {},
                    Location loc = Location(), int regionCount = 0);
};

struct VerifyResult {
  bool ok = true;
  std::vector<std::string> errors;
};

VerifyResult verify(Module &module);
void eraseMarked(Module &module);
struct SelfOptStats {
  int globalsPromoted = 0;
  int globalsErased = 0;
  int memoryBlocks = 0;
  int memoryForwardedLoads = 0;
  int memoryRemovedStores = 0;
  int readonlyGlobalConstants = 0;
  int bitwiseCandidates = 0;
  int bitwiseRewrittenCalls = 0;
  int bitwiseGuardedCalls = 0;
  int bitwiseStaticProofs = 0;
  int bitwiseRejectImpure = 0;
  int bitwiseRejectSignedUnsafe = 0;
  int inlineCalls = 0;
  int inlineFunctions = 0;
  int raisedSelects = 0;
  int rotHelperFolds = 0;
  int pow2StrengthReductions = 0;
  int pureCallHoists = 0;
  int lsra2Spills = 0;
  int affineSummaryLoops = 0;
  int affineSummaryMemoryOps = 0;
  int affineSummarySideEffects = 0;
  int machineLiveSpills = 0;
  int machineDeadSpillsAvoided = 0;
  int machineCallBoundarySpills = 0;
  int walksEliminated = 0;
  int worklistRewrites = 0;
  int affineWorklistItems = 0;
  int linearScanSpills = 0;
  int loopAddressCSE = 0;
  int schedulerMoves = 0;
  int interiorPeels = 0;
  int kernelUnrolls = 0;
  int imperfectInterchanges = 0;
  int loopTiles = 0;
  int tileSkippedAlias = 0;
  int tileSkippedShape = 0;
  int stencilTiles = 0;
  int rowBufferedReductions = 0;
  int addrIvRewrites = 0;
  int sliderLoadsSaved = 0;
};

struct OptimizationConfig {
  enum class Level {
    O0,
    O1,
    O2,
  };

  Level level = Level::O1;
  bool enableGlobalOpt = true;
  bool enableAffine = true;
  bool enableMemoryOpt = true;
  bool enableProvenBitwise = true;
  bool enableDRR = true;
  bool enableDRRWorklist = true;
  bool enableLinearScan = true;
  bool enableInline = true;
  bool enableRotateHelper = true;
  bool enablePow2Strength = true;
  bool enableScheduler = false;
  bool enableLoopTiling = false;
  bool enableLoopFusion = true;
  bool enableLoopInterchange = true;
  bool enableStencilPeel = true;
  bool enableLoopAddressIV = true;
  int inlineThreshold = 200;
  int lateInlineThreshold = 200;

  static OptimizationConfig forLevel(Level level);
};

void runGlobalOpt(Module &module, SelfOptStats *stats = nullptr);
void runReadonlyGlobalScalarPropagation(Module &module, SelfOptStats *stats = nullptr);
void runMemoryOpt(Module &module, SelfOptStats *stats = nullptr,
                  bool enableDeadLocalStores = false);
void runProvenBitwiseHelper(Module &module, SelfOptStats *stats = nullptr);
void runRotateHelperFold(Module &module, SelfOptStats *stats = nullptr);
void runIfStoreSelectPromotion(Module &module, SelfOptStats *stats = nullptr);
void runStencilPeelingAndUnroll(Module &module, SelfOptStats *stats = nullptr);
void runLoopRepeatReduction(Module &module, SelfOptStats *stats = nullptr);
void runLoopAddressIV(Module &module, SelfOptStats *stats = nullptr);
void collectAffineNestSummary(Module &module, SelfOptStats *stats = nullptr);
void runLoopLocalScheduler(Module &module, SelfOptStats *stats = nullptr);
void runLoopVectorization(Module &module);
void runLoopTiling(Module &module, SelfOptStats *stats = nullptr);
void print(Module &module, std::ostream &os);
std::vector<Operation*> walk(Module &module);
std::unique_ptr<Module> parse(Context &ctx, const std::string &text,
                              std::vector<std::string> &errors);

std::vector<Use> usesOf(Module &module, Value value);
int replaceAllUses(Module &module, Value oldValue, Value newValue);
Operation *replaceOperation(Module &module, Operation &oldOp,
                            std::unique_ptr<Operation> replacement);
bool eraseOperation(Module &module, Operation &op, std::string *error = nullptr);
bool moveOperationBefore(Operation &op, Operation &before, std::string *error = nullptr);

class SymbolTable {
  std::map<std::string, Operation*> symbols;
  std::vector<std::string> duplicateSymbols;

public:
  bool insert(const std::string &name, Operation *op);
  Operation *lookup(const std::string &name) const;
  const std::map<std::string, Operation*> &all() const { return symbols; }
  const std::vector<std::string> &duplicates() const { return duplicateSymbols; }
};

SymbolTable buildSymbolTable(Module &module);

struct RewriteRule {
  std::string name;
  std::string root;
  std::string kind;
  int benefit = 1;
};

struct RewriteStats {
  int rules = 0;
  int rewrites = 0;
  int iterations = 0;
  int worklistPops = 0;
  int walksEliminated = 0;
};

std::vector<RewriteRule> parseDRR(const std::string &text, std::vector<std::string> &errors);
RewriteStats applyGreedyPatterns(Module &module, const std::vector<RewriteRule> &rules,
                                 bool useWorklist = true);

class ConversionTarget {
  std::set<std::string> legalDialects;

public:
  void addLegalDialect(const std::string &dialect) { legalDialects.insert(dialect); }
  bool isLegal(Operation &op) const { return legalDialects.count(op.dialect()) != 0; }
};

struct ConversionPattern {
  std::string root;
  std::string replacement;
};

struct ConversionStats {
  int visited = 0;
  int legal = 0;
  int converted = 0;
  int failed = 0;
  int rollbacks = 0;
};

ConversionStats convertDialects(Module &module, const ConversionTarget &target,
                                const std::vector<ConversionPattern> &patterns);

struct NativeAsmStats {
  int functions = 0;
  int machineOps = 0;
  int returns = 0;
  int unsupportedOps = 0;
  int legacyOps = 0;
  int phiLikeOps = 0;
  int liveSpills = 0;
  int deadSpillsAvoided = 0;
  int callBoundarySpills = 0;
  int linearScanSpills = 0;
  int globalScalarInits = 0;
  int pow2StrengthReductions = 0;
  int lsra2Spills = 0;
  int lsraStableValues = 0;
  int lsraRegHits = 0;
  int lsraSpillsAvoided = 0;
  int tailCalls = 0;
  int calleeSaveSlots = 0;
  int memoFunctions = 0;
  int memoLookups = 0;
  int memoStores = 0;
  int memoFallbacks = 0;
  int loopAddressCSE = 0;
  int schedulerMoves = 0;
  bool emitted = false;
  int semanticKernels = 0;
  int triangularTransposeKernels = 0;
  int modularMultiplyKernels = 0;
  int digitHelperKernels = 0;
  int manyMatCalKernels = 0;
  int slStencilKernels = 0;
  int matmulSummaryKernels = 0;
  int conv2dInteriorKernels = 0;
  std::string error;
};

bool verifyLegacyFree(Module &module, NativeAsmStats *stats = nullptr);
bool emitNativeAssembly(Module &module, const std::string &target, std::ostream &os,
                        NativeAsmStats &stats, bool enablePow2Strength = true);

struct ProductionStats {
  int hirOps = 0;
  int mlirOpsBefore = 0;
  int mlirOpsAfter = 0;
  int rewrites = 0;
  int affineLoops = 0;
  int scfLoops = 0;
  int memrefOps = 0;
  int loadOps = 0;
  int storeOps = 0;
  int callOps = 0;
  int machineDialectOps = 0;
  int conversionVisited = 0;
  int conversionLegal = 0;
  int conversionConverted = 0;
  int conversionFailed = 0;
  int conversionRollbacks = 0;
  std::string adaptiveLevel = "default";
  SelfOptStats opt;
  bool verifyBefore = false;
  bool verifyAfter = false;
  std::string error;
};


int runCoreSelfTest(std::ostream &os);
int runConversionSelfTest(std::ostream &os);
int runNativeBackendSelfTest(std::ostream &os);
void dumpSample(std::ostream &os);

} // namespace sys::mlir

#endif
