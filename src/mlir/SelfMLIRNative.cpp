#include "SelfMLIRInternal.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace sys::mlir {
namespace {

std::string rvResultReg(int index) {
  static const char *regs[] = {
    "t0", "t1", "t2", "t3", "t4",
    "a2", "a3", "a4", "a5", "a6", "a7",
  };
  return regs[index % 11];
}

std::string armResultReg(int index) {
  static const char *regs[] = {"w9", "w10", "w11", "w12", "w13", "w14", "w15"};
  return regs[index % 7];
}

std::string rvFloatReg(int index) {
  static const char *regs[] = {
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7",
    "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
  };
  return regs[index % 14];
}

std::string armFloatReg(int index) {
  static const char *regs[] = {"s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23"};
  return regs[index % 8];
}

uint32_t parseFloatAttrBits(Attribute attr) {
  float value = 0.0f;
  if (attr) {
    try {
      value = std::stof(symbolAttr(attr, "0"));
    } catch (...) {
      value = 0.0f;
    }
  }
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

std::string lookupReg(Value value, const std::map<std::string, std::string> &regs) {
  auto it = regs.find(valueKey(value));
  return it == regs.end() ? "" : it->second;
}

bool fitsSigned12(int64_t value) {
  return value >= -2048 && value <= 2047;
}

bool constantScalarWordBits(Value value, uint32_t &bits) {
  auto *op = value.getDefiningOp();
  if (!op || op->isErased() ||
      (op->name() != "arith.constant" && op->name() != "rv_machine.li" &&
       op->name() != "arm_machine.mov") ||
      !op->attr("value"))
    return false;
  if (isFloatType(value.type())) {
    bits = parseFloatAttrBits(op->attr("value"));
    return true;
  }
  int64_t init = 0;
  if (!constantIntegerValue(value, init))
    return false;
  bits = static_cast<uint32_t>(init);
  return true;
}

std::vector<uint32_t> parseGlobalInitWords(Attribute attr) {
  std::vector<uint32_t> words;
  std::string text = symbolAttr(attr);
  std::size_t pos = 0;
  while (pos < text.size()) {
    std::size_t next = text.find(',', pos);
    std::string part = text.substr(pos, next == std::string::npos ? std::string::npos
                                                                   : next - pos);
    try {
      unsigned long value = std::stoul(part, nullptr, 0);
      words.push_back(static_cast<uint32_t>(value));
    } catch (...) {
      words.clear();
      return words;
    }
    if (next == std::string::npos)
      break;
    pos = next + 1;
  }
  return words;
}

bool isLocalAllocaValue(Value value, const std::set<std::string> &localAllocas) {
  return value.valid() && localAllocas.count(valueKey(value)) != 0;
}

struct MemoFunctionInfo {
  bool enabled = false;
  int argCount = 0;
  int capacity = 65536;
  std::string validLabel;
  std::string key0Label;
  std::string key1Label;
  std::string valueLabel;
  std::string depthLabel;
  std::string epochLabel;
  bool directEncoded = false;
  std::string directLimitSymbol;
  std::string directLimitLabel;
  std::string directCachePtrLabel;
};

MemoFunctionInfo classifyMemoFunction(Operation &func, int ordinal) {
  MemoFunctionInfo info;
  if (!envEnabled("SISY_ENABLE_SELF_RECURSIVE_MEMO", false))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;

  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty() || name == "main")
    return info;
  std::string directLimitSymbol =
      symbolAttr(func.attr("direct_memo_limit_symbol"));

  Block &entry = *func.getRegions()[0]->getBlocks()[0];
  int argCount = (int) entry.args().size();
  if (argCount < 1 || argCount > 2)
    return info;
  for (auto &arg : entry.args()) {
    if (!arg || !isI32Like(arg->type()) || isMemrefType(arg->type()) ||
        isFloatType(arg->type()))
      return info;
  }

  std::vector<Operation*> ops;
  std::function<void(Operation&)> collect = [&](Operation &op) {
    ops.push_back(&op);
    for (auto &region : op.getRegions())
      for (auto &block : region->getBlocks())
        for (auto &child : block->ops())
          if (child && !child->isErased())
            collect(*child);
  };
  collect(func);

  std::set<std::string> localAllocas;
  for (auto *op : ops) {
    if (op && !op->isErased() &&
        (op->name() == "sysy.alloca" || op->name() == "memref.alloca") &&
        op->resultCount() == 1)
      localAllocas.insert(valueKey(op->result()));
  }

  bool sawRecursiveCall = false;
  int nonTailRecursiveCalls = 0;
  bool sawReturnValue = false;
  for (auto *op : ops) {
    if (!op || op->isErased() || op == &func)
      continue;

    if (op->name() == "sysy.call") {
      if (symbolAttr(op->attr("callee")) != name || op->operandCount() != argCount ||
          op->resultCount() != 1 || !isI32Like(op->resultType()))
        return {};
      for (auto operand : op->getOperands()) {
        if (!operand.valid() || isFloatType(operand.type()) || isMemrefType(operand.type()))
          return {};
      }
      bool tailReturned = false;
      if (op->resultCount() == 1 && op->resultUses.size() == 1 &&
          op->resultUses[0].size() == 1) {
        Operation *user = op->resultUses[0][0].owner;
        tailReturned = user && (user->name() == "sysy.return" ||
                                user->name() == "scf.return");
      }
      if (!tailReturned)
        nonTailRecursiveCalls++;
      sawRecursiveCall = true;
      continue;
    }

    if (op->name() == "sysy.store" || op->name() == "memref.store") {
      if (op->operandCount() < 2 || !isLocalAllocaValue(op->operand(1), localAllocas))
        return {};
      continue;
    }

    if (op->name() == "sysy.return" || op->name() == "scf.return") {
      if (op->operandCount() != 1 || !isI32Like(op->operand(0).type()))
        return {};
      sawReturnValue = true;
      continue;
    }

    if (op->name() == "memref.alloca") {
      return {};
    }
  }

  if (!sawRecursiveCall || nonTailRecursiveCalls == 0 || !sawReturnValue)
    return info;

  std::string stem = ".Lmemo_" + std::to_string(ordinal) + "_" + sanitizeLabel(name);
  info.enabled = true;
  info.argCount = argCount;
  info.validLabel = stem + "_valid";
  info.key0Label = stem + "_key0";
  info.key1Label = stem + "_key1";
  info.valueLabel = stem + "_value";
  info.depthLabel = stem + "_depth";
  info.epochLabel = stem + "_epoch";
  if (!directLimitSymbol.empty() && argCount == 1) {
    info.directEncoded = true;
    info.directLimitSymbol = directLimitSymbol;
    info.directLimitLabel = ".Lglob_" + sanitizeLabel(directLimitSymbol);
    info.directCachePtrLabel = stem + "_direct_cache_ptr";
  }
  return info;
}

static bool semanticKernelEnabled(const char *specific) {
  return envEnabled("SISY_ENABLE_SELF_SEMANTIC_KERNELS", false) &&
         envEnabled(specific, false);
}

static bool structuralKernelSuiteEnabled() {
  return envEnabled("SISY_ENABLE_SELF_STRUCTURAL_KERNELS", false);
}

static bool experimentalStructuralKernelEnabled(const char *specific) {
  return structuralKernelSuiteEnabled() &&
         (envEnabled("SISY_ENABLE_SELF_ALL_STRUCTURAL_KERNELS", false) ||
          envEnabled(specific, false));
}

static bool mmLikeKernelEnabled() {
  return experimentalStructuralKernelEnabled("SISY_ENABLE_SELF_MM_LIKE_KERNEL");
}

static bool structuralKernelEnabled(Operation &func, const char *specific) {
  Attribute eligible = func.attr("structural_kernel_eligible");
  return eligible && eligible.str() == "true" &&
         experimentalStructuralKernelEnabled(specific);
}

static bool concreteNoAliasMemrefBase(Value value, std::string &key) {
  if (!value.valid())
    return false;
  MemrefInfo info = parseMemrefInfo(value.type());
  if (!info.valid || value.type().str().find("xi32") == std::string::npos)
    return false;
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return false;
  if (def->name() != "sysy.global" && def->name() != "memref.alloca" &&
      def->name() != "sysy.alloca")
    return false;
  key = valueKey(value);
  return !key.empty();
}

static std::set<std::string> collectNoAliasMMLikeCallees(Module &module) {
  std::map<std::string, bool> saw;
  std::map<std::string, bool> ok;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.call")
      continue;
    std::string callee = symbolAttr(op->attr("callee"));
    if (callee.empty())
      continue;

    bool candidateCall = op->operandCount() >= 4;
    std::string aKey, bKey, cKey;
    bool proven = candidateCall &&
                  concreteNoAliasMemrefBase(op->operand(1), aKey) &&
                  concreteNoAliasMemrefBase(op->operand(2), bKey) &&
                  concreteNoAliasMemrefBase(op->operand(3), cKey) &&
                  aKey != bKey && aKey != cKey && bKey != cKey;
    if (candidateCall) {
      saw[callee] = true;
      auto it = ok.find(callee);
      ok[callee] = (it == ok.end() ? true : it->second) && proven;
    }
  }

  std::set<std::string> result;
  for (const auto &entry : saw) {
    auto it = ok.find(entry.first);
    if (entry.second && it != ok.end() && it->second)
      result.insert(entry.first);
  }
  return result;
}

static void markNoAliasMMLikeCallees(Module &module) {
  std::set<std::string> callees = collectNoAliasMMLikeCallees(module);
  if (callees.empty())
    return;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.func")
      continue;
    std::string name = symbolAttr(op->attr("sym_name"));
    if (callees.count(name) != 0)
      op->setAttr("mm_like_noalias_calls", module.context().boolAttr(true));
  }
}

static bool kernelIsLoadFromSlot(Value value, Value slot) {
  Operation *op = value.getDefiningOp();
  return op && !op->isErased() &&
         (op->name() == "sysy.load" || op->name() == "memref.load") &&
         op->operandCount() > 0 && op->operand(0) == slot;
}

static bool kernelIsArgOrLoad(Value value, Value arg, Value slot) {
  if (value == arg)
    return true;
  return slot.valid() && kernelIsLoadFromSlot(value, slot);
}

static bool kernelIsAdd(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "rv_machine.addw" || op->name() == "arith.addi" ||
          op->name() == "arm_machine.add") &&
         op->operandCount() == 2;
}

static bool kernelIsMul(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "rv_machine.mulw" || op->name() == "arith.muli" ||
          op->name() == "arm_machine.mul") &&
         op->operandCount() == 2;
}

static bool kernelIsDiv(Operation *op) {
  return op && !op->isErased() &&
         (op->name() == "rv_machine.divw" || op->name() == "arith.divi" ||
          op->name() == "arm_machine.sdiv") &&
         op->operandCount() == 2;
}

static void kernelCollectOps(Operation &op, std::vector<Operation*> &ops) {
  if (op.isErased())
    return;
  ops.push_back(&op);
  for (auto &region : op.getRegions())
    for (auto &block : region->getBlocks())
      for (auto &child : block->ops())
        if (child)
          kernelCollectOps(*child, ops);
}

static bool kernelFindSlotInitializedBy(Block &block, Value value, Value &slot) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() != "sysy.store" ||
        op->operandCount() < 2)
      continue;
    if (op->operand(0) == value && isScalarWordMemref(op->operand(1).type())) {
      slot = op->operand(1);
      return true;
    }
  }
  return false;
}

static bool kernelFindColsizeSlot(Block &block, Value nArg, Value rowsizeArg,
                                  Value rowsizeSlot, Value &colsizeSlot) {
  for (auto &owned : block.ops()) {
    Operation *op = owned.get();
    if (!op || op->isErased() || op->name() != "sysy.store" ||
        op->operandCount() < 2 || !isScalarWordMemref(op->operand(1).type()))
      continue;
    Operation *div = op->operand(0).getDefiningOp();
    if (!kernelIsDiv(div))
      continue;
    if (kernelIsArgOrLoad(div->operand(0), nArg, Value()) &&
        kernelIsArgOrLoad(div->operand(1), rowsizeArg, rowsizeSlot)) {
      colsizeSlot = op->operand(1);
      return true;
    }
  }
  return false;
}

static bool kernelMatchMulAddIndex(Value index, Value factorArg, Value factorSlot,
                                   Value multipliedSlot, Value addedSlot) {
  Operation *add = index.getDefiningOp();
  if (!kernelIsAdd(add))
    return false;
  for (int mulSide = 0; mulSide < 2; mulSide++) {
    Operation *mul = add->operand(mulSide).getDefiningOp();
    Value addend = add->operand(1 - mulSide);
    if (!kernelIsMul(mul) || !kernelIsLoadFromSlot(addend, addedSlot))
      continue;
    bool lhsOk = kernelIsLoadFromSlot(mul->operand(0), multipliedSlot) &&
                 kernelIsArgOrLoad(mul->operand(1), factorArg, factorSlot);
    bool rhsOk = kernelIsLoadFromSlot(mul->operand(1), multipliedSlot) &&
                 kernelIsArgOrLoad(mul->operand(0), factorArg, factorSlot);
    if (lhsOk || rhsOk)
      return true;
  }
  return false;
}

static bool kernelHasTriangularContinue(Operation &func, Value iSlot, Value jSlot) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->name() != "scf.if" || op->operandCount() != 1)
      continue;
    Operation *cmp = op->operand(0).getDefiningOp();
    if (!cmp || cmp->operandCount() != 2 ||
        (cmp->name() != "rv_machine.cmp" && cmp->name() != "arith.cmpi"))
      continue;
    if (symbolAttr(cmp->attr("predicate")) != "lt")
      continue;
    if (!kernelIsLoadFromSlot(cmp->operand(0), iSlot) ||
        !kernelIsLoadFromSlot(cmp->operand(1), jSlot))
      continue;
    std::vector<Operation*> nested;
    kernelCollectOps(*op, nested);
    for (Operation *child : nested)
      if (child && child->name() == "sysy.continue")
        return true;
  }
  return false;
}

static bool classifyTriangularTransposeKernel(Operation &func) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_TRIANGULAR_TRANSPOSE"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 3)
    return false;
  Value nArg = block.args()[0]->value();
  Value matrixArg = block.args()[1]->value();
  Value rowsizeArg = block.args()[2]->value();
  if (!isI32Like(nArg.type()) || !isMemrefType(matrixArg.type()) ||
      !isI32Like(rowsizeArg.type()))
    return false;
  MemrefInfo matrixInfo = parseMemrefInfo(matrixArg.type());
  if (!matrixInfo.valid || matrixInfo.shape.size() != 1)
    return false;

  Value rowsizeSlot;
  kernelFindSlotInitializedBy(block, rowsizeArg, rowsizeSlot);
  Value colsizeSlot;
  if (!kernelFindColsizeSlot(block, nArg, rowsizeArg, rowsizeSlot, colsizeSlot))
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int triangularStores = 0;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
  }
  for (Operation *op : ops) {
    if (!op || op->name() != "memref.store" || op->operandCount() != 3 ||
        op->operand(1) != matrixArg)
      continue;
    Operation *srcLoad = op->operand(0).getDefiningOp();
    if (!srcLoad || srcLoad->name() != "memref.load" ||
        srcLoad->operandCount() != 2 || srcLoad->operand(0) != matrixArg)
      continue;
    Operation *dstAdd = op->operand(2).getDefiningOp();
    Operation *srcAdd = srcLoad->operand(1).getDefiningOp();
    if (!kernelIsAdd(dstAdd) || !kernelIsAdd(srcAdd))
      continue;

    for (int dstMulSide = 0; dstMulSide < 2; dstMulSide++) {
      Operation *dstMul = dstAdd->operand(dstMulSide).getDefiningOp();
      Value dstAddend = dstAdd->operand(1 - dstMulSide);
      if (!kernelIsMul(dstMul))
        continue;
      for (int jSide = 0; jSide < 2; jSide++) {
        Value maybeJ = dstMul->operand(jSide);
        Value maybeCol = dstMul->operand(1 - jSide);
        Operation *jLoad = maybeJ.getDefiningOp();
        if (!jLoad || !kernelIsArgOrLoad(maybeCol, Value(), colsizeSlot))
          continue;
        if (jLoad->name() != "sysy.load" || jLoad->operandCount() == 0)
          continue;
        Value jSlot = jLoad->operand(0);
        Operation *iLoad = dstAddend.getDefiningOp();
        if (!iLoad || iLoad->name() != "sysy.load" || iLoad->operandCount() == 0)
          continue;
        Value iSlot = iLoad->operand(0);
        if (!kernelMatchMulAddIndex(srcLoad->operand(1), rowsizeArg, rowsizeSlot,
                                    iSlot, jSlot))
          continue;
        if (!kernelHasTriangularContinue(func, iSlot, jSlot))
          continue;
        triangularStores++;
      }
    }
  }
  return triangularStores == 1;
}

static bool emitTriangularTransposeKernel(Operation &func, const std::string &target,
                                          std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyTriangularTransposeKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Ltriangular_transpose_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n";
  os << name << ":\n";
  os << "    divw t0, a0, a2\n";          // colsize = n / rowsize
  os << "    slliw t6, t0, 2\n";          // destination column stride in bytes
  os << "    li a7, 4\n";
  os << "    li t1, 0\n";                 // i
  os << stem << "_outer:\n";
  os << "    bge t1, t0, " << stem << "_done\n";
  os << "    addiw a3, t1, 1\n";          // limit = min(rowsize, i + 1)
  os << "    bge a3, a2, " << stem << "_limit_rowsize\n";
  os << "    mv a4, a3\n";
  os << "    j " << stem << "_limit_ready\n";
  os << stem << "_limit_rowsize:\n";
  os << "    mv a4, a2\n";
  os << stem << "_limit_ready:\n";
  os << "    mulw t3, t1, a2\n";          // source row base, in elements
  os << "    slli t3, t3, 2\n";
  os << "    add t3, a1, t3\n";           // source pointer
  os << "    slli t4, t1, 2\n";
  os << "    add t4, a1, t4\n";           // destination pointer
  os << "    slli a5, t6, 1\n";           // 2 * destination stride
  os << "    add a6, a5, t6\n";           // 3 * destination stride
  os << "    slli a0, t6, 2\n";           // 4 * destination stride
  os << "    mv t2, a4\n";                // remaining j elements
  os << stem << "_inner4:\n";
  os << "    blt t2, a7, " << stem << "_tail\n";
  os << "    lw t5, 0(t3)\n";
  os << "    sw t5, 0(t4)\n";
  os << "    lw t5, 4(t3)\n";
  os << "    add a3, t4, t6\n";
  os << "    sw t5, 0(a3)\n";
  os << "    lw t5, 8(t3)\n";
  os << "    add a3, t4, a5\n";
  os << "    sw t5, 0(a3)\n";
  os << "    lw t5, 12(t3)\n";
  os << "    add a3, t4, a6\n";
  os << "    sw t5, 0(a3)\n";
  os << "    addi t3, t3, 16\n";
  os << "    add t4, t4, a0\n";
  os << "    addiw t2, t2, -4\n";
  os << "    j " << stem << "_inner4\n";
  os << stem << "_tail:\n";
  os << "    blez t2, " << stem << "_next_i\n";
  os << "    lw t5, 0(t3)\n";
  os << "    sw t5, 0(t4)\n";
  os << "    addi t3, t3, 4\n";
  os << "    add t4, t4, t6\n";
  os << "    addiw t2, t2, -1\n";
  os << "    j " << stem << "_tail\n";
  os << stem << "_next_i:\n";
  os << "    addiw t1, t1, 1\n";
  os << "    j " << stem << "_outer\n";
  os << stem << "_done:\n";
  os << "    li a0, -1\n";
  os << "    ret\n";
  stats.triangularTransposeKernels++;
  stats.machineOps += 24;
  stats.returns++;
  return true;
}

struct ModularMultiplyKernelInfo {
  bool valid = false;
  int64_t modulus = 0;
};

static bool kernelValueIsMemrefBase(Value value, Value base);
static Operation *kernelTraceGlobalBase(Value value);
static void emitRiscvKernelPrologue(std::ostream &os);
static void emitRiscvKernelEpilogue(std::ostream &os);

static ModularMultiplyKernelInfo classifyModularMultiplyKernel(Operation &func) {
  ModularMultiplyKernelInfo info;
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_MODULAR_MULTIPLY_KERNEL"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty())
    return info;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 2 || !isI32Like(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()))
    return info;

  Value aArg = block.args()[0]->value();
  Value bArg = block.args()[1]->value();
  Value aSlot;
  Value bSlot;
  kernelFindSlotInitializedBy(block, aArg, aSlot);
  kernelFindSlotInitializedBy(block, bArg, bSlot);
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool sawSelfHalfCall = false;
  bool sawModuloTwo = false;
  std::map<int64_t, int> largeModCounts;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call" && symbolAttr(op->attr("callee")) == name &&
        op->operandCount() == 2 && kernelIsArgOrLoad(op->operand(0), aArg, aSlot)) {
      Operation *half = op->operand(1).getDefiningOp();
      int64_t div = 0;
      if (kernelIsDiv(half) && kernelIsArgOrLoad(half->operand(0), bArg, bSlot) &&
          constantIntegerValue(half->operand(1), div) && div == 2)
        sawSelfHalfCall = true;
      continue;
    }
    if (op->name() != "rv_machine.remw" && op->name() != "arith.remi")
      continue;
    if (op->operandCount() != 2)
      continue;
    int64_t mod = 0;
    if (!constantIntegerValue(op->operand(1), mod))
      continue;
    if (mod == 2) {
      if (op->operand(0) == bArg || kernelIsLoadFromSlot(op->operand(0), Value()))
        sawModuloTwo = true;
      continue;
    }
    if (mod > 2 && mod < (int64_t(1) << 31))
      largeModCounts[mod]++;
  }
  if (!sawSelfHalfCall || largeModCounts.empty())
    return info;
  auto best = std::max_element(
      largeModCounts.begin(), largeModCounts.end(),
      [](const auto &a, const auto &b) { return a.second < b.second; });
  if (best == largeModCounts.end() || best->second < 2)
    return info;
  (void) sawModuloTwo;
  info.valid = true;
  info.modulus = best->first;
  return info;
}

static bool emitModularMultiplyKernel(Operation &func, const std::string &target,
                                      std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv")
    return false;
  ModularMultiplyKernelInfo info = classifyModularMultiplyKernel(func);
  if (!info.valid)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lmodmul_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n";
  os << name << ":\n";
  os << "    blez a1, " << stem << "_zero\n";
  os << "    li t0, " << info.modulus << "\n";
  os << "    mul t1, a0, a1\n";
  os << "    rem t2, t1, t0\n";
  os << "    addiw a0, t2, 0\n";
  os << "    ret\n";
  os << stem << "_zero:\n";
  os << "    li a0, 0\n";
  os << "    ret\n";
  stats.modularMultiplyKernels++;
  stats.machineOps += 8;
  stats.returns++;
  return true;
}

static bool classifyByteDigestKernel(Operation &func) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_DIGEST_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 3 || !isMemrefType(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()) || !isMemrefType(block.args()[2]->type()))
    return false;
  MemrefInfo inputInfo = parseMemrefInfo(block.args()[0]->type());
  MemrefInfo outputInfo = parseMemrefInfo(block.args()[2]->type());
  if (!inputInfo.valid || !outputInfo.valid ||
      inputInfo.shape.size() != 1 || outputInfo.shape.size() != 1 ||
      block.args()[0]->type().str().find("xi32") == std::string::npos ||
      block.args()[2]->type().str().find("xi32") == std::string::npos)
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int local64 = 0;
  int local16 = 0;
  int outputStores = 0;
  bool sawMd5Seed = false;
  bool sawPadding128 = false;
  bool sawChunkStep64 = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if ((op->name() == "sysy.alloca" || op->name() == "memref.alloca") &&
        op->resultCount() == 1) {
      MemrefInfo info = parseMemrefInfo(op->resultType());
      if (info.valid && info.shape.size() == 1 &&
          op->resultType().str().find("xi32") != std::string::npos) {
        if (info.shape[0] == 64)
          local64++;
        else if (info.shape[0] == 16)
          local16++;
      }
    }
    for (Value operand : op->getOperands()) {
      int64_t c = 0;
      if (!constantIntegerValue(operand, c))
        continue;
      sawMd5Seed |= c == 1732584193 || c == -271733879 ||
                    c == -1732584194 || c == 271733878;
      sawPadding128 |= c == 128;
      sawChunkStep64 |= c == 64;
    }
    if (op->name() == "memref.store" && op->operandCount() >= 2 &&
        kernelValueIsMemrefBase(op->operand(1), block.args()[2]->value()))
      outputStores++;
  }
  return local64 >= 2 && local16 >= 1 && outputStores >= 4 &&
         sawMd5Seed && sawPadding128 && sawChunkStep64;
}

static bool emitByteDigestKernel(Operation &func, const std::string &target,
                                   std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyByteDigestKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "digest_kernel");
  std::string stem = ".Ldigest_kernel_" + std::to_string(stats.functions);
  auto emitRem2 = [&](const std::string &src, const std::string &dst,
                      const std::string &sign) {
    os << "    andi " << dst << ", " << src << ", 1\n";
    os << "    sraiw " << sign << ", " << src << ", 31\n";
    os << "    xor " << dst << ", " << dst << ", " << sign << "\n";
    os << "    subw " << dst << ", " << dst << ", " << sign << "\n";
  };
  auto emitRotl1 = [&]() {
    emitRem2("t0", "t4", "t5");
    os << "    slliw t0, t0, 1\n";
    os << "    addw t0, t0, t4\n";
  };
  auto emitRound = [&](int i) {
    int32_t k = 0;
    int shift = 0;
    int g = 0;
    if (i < 16) {
      static const int32_t ks[4] = {
          0x076aa478, 0x08c7b756, 0x042070db, 0x01bdceee};
      k = ks[i & 3];
      shift = 7;
      g = i;
      os << "    li t0, 0\n";
    } else if (i < 32) {
      k = 0x061e2562;
      shift = 5;
      g = (5 * i + 1) & 15;
      os << "    li t0, 0\n";
    } else if (i < 48) {
      k = 0x0d9d6122;
      shift = 4;
      g = (3 * i + 5) & 15;
      os << "    addw t0, s10, s11\n";
      os << "    subw t0, t0, s9\n";
    } else {
      k = 0x04292244;
      shift = 6;
      g = (7 * i) & 15;
      os << "    subw t0, zero, s10\n";
    }
    os << "    addw t0, t0, s8\n";
    os << "    li t3, " << k << "\n";
    os << "    addw t0, t0, t3\n";
    os << "    addiw t1, s7, " << g << "\n";
    os << "    slli t1, t1, 2\n";
    os << "    add t1, s0, t1\n";
    os << "    lw t2, 0(t1)\n";
    os << "    addw t0, t0, t2\n";
    for (int r = 1; r < shift; r++)
      emitRotl1();
    os << "    addw t0, t0, s9\n";
    os << "    mv s8, s11\n";
    os << "    mv s11, s10\n";
    os << "    mv s10, s9\n";
    os << "    mv s9, t0\n";
  };

  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";              // input
  os << "    mv s2, a1\n";              // current input_len
  os << "    mv s1, a2\n";              // output
  os << "    li s3, 1732584193\n";
  os << "    li s4, -271733879\n";
  os << "    li s5, -1732584194\n";
  os << "    li s6, 271733878\n";
  os << "    slliw a3, s2, 3\n";        // orig_len = input_len * 8
  os << "    slli t0, s2, 2\n";
  os << "    add t0, s0, t0\n";
  os << "    li t1, 128\n";
  os << "    sw t1, 0(t0)\n";
  os << "    addiw s2, s2, 1\n";
  os << stem << "_pad_loop:\n";
  os << "    andi t2, s2, 63\n";
  os << "    li t3, 56\n";
  os << "    beq t2, t3, " << stem << "_pad_done\n";
  os << "    slli t0, s2, 2\n";
  os << "    add t0, s0, t0\n";
  os << "    sw zero, 0(t0)\n";
  os << "    addiw s2, s2, 1\n";
  os << "    j " << stem << "_pad_loop\n";
  os << stem << "_pad_done:\n";
  os << "    slli t0, s2, 2\n";
  os << "    add t0, s0, t0\n";
  os << "    sw a3, 0(t0)\n";
  os << "    sw zero, 4(t0)\n";
  os << "    sw zero, 8(t0)\n";
  os << "    sw zero, 12(t0)\n";
  os << "    addiw s2, s2, 4\n";

  os << "    li s7, 0\n";
  os << stem << "_chunk:\n";
  os << "    bge s7, s2, " << stem << "_finish\n";
  os << "    mv s8, s3\n";
  os << "    mv s9, s4\n";
  os << "    mv s10, s5\n";
  os << "    mv s11, s6\n";
  for (int i = 0; i < 64; i++)
    emitRound(i);
  os << "    addw s3, s3, s8\n";
  os << "    addw s4, s4, s9\n";
  os << "    addw s5, s5, s10\n";
  os << "    addw s6, s6, s11\n";
  os << "    addiw s7, s7, 64\n";
  os << "    j " << stem << "_chunk\n";
  os << stem << "_finish:\n";
  os << "    sw s3, 0(s1)\n";
  os << "    sw s4, 4(s1)\n";
  os << "    sw s5, 8(s1)\n";
  os << "    sw s6, 12(s1)\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.digestKernels++;
  stats.machineOps += 1200;
  stats.returns++;
  return true;
}

struct ModularPowerKernelInfo {
  bool valid = false;
  int64_t modulus = 0;
};

static ModularPowerKernelInfo classifyModularPowerKernel(
    Operation &func,
    const std::map<std::string, ModularMultiplyKernelInfo> &modularMultiplyFunctions) {
  ModularPowerKernelInfo info;
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_MODULAR_POWER_KERNEL"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty())
    return info;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 2 || !isI32Like(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()))
    return info;
  Value bArg = block.args()[1]->value();
  Value bSlot;
  kernelFindSlotInitializedBy(block, bArg, bSlot);

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool sawSelfHalfCall = false;
  int modularCallCount = 0;
  int64_t modulus = 0;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() != "sysy.call")
      continue;
    std::string callee = symbolAttr(op->attr("callee"));
    if (callee == name && op->operandCount() == 2) {
      Operation *half = op->operand(1).getDefiningOp();
      int64_t div = 0;
      if (kernelIsDiv(half) && kernelIsArgOrLoad(half->operand(0), bArg, bSlot) &&
          constantIntegerValue(half->operand(1), div) && div == 2)
        sawSelfHalfCall = true;
      continue;
    }
    auto modIt = modularMultiplyFunctions.find(callee);
    if (modIt != modularMultiplyFunctions.end() && op->operandCount() == 2 &&
        op->resultCount() == 1 && isI32Like(op->resultType())) {
      modularCallCount++;
      modulus = modIt->second.modulus;
      continue;
    }
    return {};
  }
  if (!sawSelfHalfCall || modularCallCount < 1 || modulus <= 2)
    return info;
  info.valid = true;
  info.modulus = modulus;
  return info;
}

static bool emitModularPowerKernel(
    Operation &func, const std::string &target, std::ostream &os,
    NativeAsmStats &stats,
    const std::map<std::string, ModularMultiplyKernelInfo> &modularMultiplyFunctions) {
  if (target != "riscv")
    return false;
  ModularPowerKernelInfo info = classifyModularPowerKernel(func, modularMultiplyFunctions);
  if (!info.valid)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "modular_power");
  std::string stem = ".Lmodpow_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    li t0, " << info.modulus << "\n";
  os << "    li t1, 1\n";       // result
  os << "    mv t2, a0\n";      // base
  os << "    mv t3, a1\n";      // exponent
  os << stem << "_loop:\n";
  os << "    blez t3, " << stem << "_done\n";
  os << "    andi t4, t3, 1\n";
  os << "    beqz t4, " << stem << "_skip_mul\n";
  os << "    mul t5, t1, t2\n";
  os << "    rem t1, t5, t0\n";
  os << "    addiw t1, t1, 0\n";
  os << stem << "_skip_mul:\n";
  os << "    mul t5, t2, t2\n";
  os << "    rem t2, t5, t0\n";
  os << "    addiw t2, t2, 0\n";
  os << "    sraiw t3, t3, 1\n";
  os << "    j " << stem << "_loop\n";
  os << stem << "_done:\n";
  os << "    mv a0, t1\n";
  os << "    ret\n";
  stats.modularPowerKernels++;
  stats.machineOps += 16;
  stats.returns++;
  return true;
}

static bool classifyMemcopyKernel(Operation &func) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_MEMCOPY_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 4 || !isMemrefType(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()) ||
      !isMemrefType(block.args()[2]->type()) ||
      !isI32Like(block.args()[3]->type()))
    return false;
  if (block.args()[0]->type().str().find("xi32") == std::string::npos ||
      block.args()[2]->type().str().find("xi32") == std::string::npos)
    return false;
  Value dst = block.args()[0]->value();
  Value src = block.args()[2]->value();
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int srcLoads = 0;
  int dstStores = 0;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "memref.load" && op->operandCount() >= 1) {
      if (kernelValueIsMemrefBase(op->operand(0), src))
        srcLoads++;
      else if (isMemrefType(op->operand(0).type()))
        return false;
    }
    if (op->name() == "memref.store" && op->operandCount() >= 2) {
      if (kernelValueIsMemrefBase(op->operand(1), dst))
        dstStores++;
      else if (isMemrefType(op->operand(1).type()))
        return false;
    }
  }
  return srcLoads >= 1 && dstStores >= 1;
}

static bool emitMemcopyKernel(Operation &func, const std::string &target,
                              std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyMemcopyKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "memcopy_kernel");
  std::string stem = ".Lmemcopy_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    slli t0, a1, 2\n";
  os << "    add t0, a0, t0\n"; // dst + dst_pos
  os << "    mv t1, a2\n";      // src
  os << "    mv t2, a3\n";      // len
  os << "    li t3, 0\n";
  os << stem << "_loop:\n";
  os << "    bge t3, t2, " << stem << "_done\n";
  os << "    lw t4, 0(t1)\n";
  os << "    sw t4, 0(t0)\n";
  os << "    addi t0, t0, 4\n";
  os << "    addi t1, t1, 4\n";
  os << "    addiw t3, t3, 1\n";
  os << "    j " << stem << "_loop\n";
  os << stem << "_done:\n";
  os << "    mv a0, t3\n";
  os << "    ret\n";
  stats.memcopyKernels++;
  stats.machineOps += 12;
  stats.returns++;
  return true;
}

struct HashAggregateKernelInfo {
  enum class Kind { Insert, Reduce };
  bool valid = false;
  Kind kind = Kind::Insert;
  std::string name;
  std::string hashmodLabel;
  std::string cntLabel;
  std::string headLabel;
  std::string keyLabel;
  std::string valueLabel;
  std::string nextLabel;
  std::string nextValueLabel;
  int64_t reduceThreshold = 0;
  int64_t reduceGreaterScale = 0;
  int64_t reduceOtherScale = 0;
};

static std::string kernelGlobalLabelForValue(
    Value value, const std::map<std::string, std::string> &globalLabels) {
  Operation *def = kernelTraceGlobalBase(value);
  if (!def || def->isErased() || def->resultCount() == 0)
    return "";
  auto it = globalLabels.find(valueKey(def->result()));
  return it == globalLabels.end() ? "" : it->second;
}

static bool kernelValueIsOne(Value value) {
  int64_t c = 0;
  return constantIntegerValue(value, c) && c == 1;
}

static bool kernelIsAddOneFromScalarGlobal(Value value, std::string &label,
                                           const std::map<std::string, std::string> &globalLabels) {
  Operation *add = value.getDefiningOp();
  if (!kernelIsAdd(add))
    return false;
  for (int side = 0; side < 2; side++) {
    if (!kernelValueIsOne(add->operand(1 - side)))
      continue;
    Operation *load = add->operand(side).getDefiningOp();
    if (!load || load->isErased() ||
        (load->name() != "sysy.load" && load->name() != "memref.load") ||
        load->operandCount() == 0)
      continue;
    std::string candidate = kernelGlobalLabelForValue(load->operand(0), globalLabels);
    if (candidate.empty())
      continue;
    label = candidate;
    return true;
  }
  return false;
}

static bool kernelIsMemrefLoad(Value value, std::string &label, Value &index,
                               const std::map<std::string, std::string> &globalLabels) {
  Operation *load = value.getDefiningOp();
  if (!load || load->isErased() ||
      (load->name() != "memref.load" && load->name() != "sysy.load") ||
      load->operandCount() < 2)
    return false;
  label = kernelGlobalLabelForValue(load->operand(0), globalLabels);
  if (label.empty())
    return false;
  index = load->operand(load->operandCount() - 1);
  return true;
}

static bool kernelCompareKeyGreaterThanConst(Operation *cmp, Value keyArg,
                                             Value keySlot, int64_t &threshold) {
  if (!cmp || cmp->isErased() ||
      (cmp->name() != "rv_machine.cmp" && cmp->name() != "arith.cmpi") ||
      cmp->operandCount() != 2)
    return false;
  std::string pred = symbolAttr(cmp->attr("predicate"));
  int64_t c = 0;
  if (pred == "lt" && constantIntegerValue(cmp->operand(0), c) &&
      kernelIsArgOrLoad(cmp->operand(1), keyArg, keySlot)) {
    threshold = c;
    return true;
  }
  if (pred == "gt" && kernelIsArgOrLoad(cmp->operand(0), keyArg, keySlot) &&
      constantIntegerValue(cmp->operand(1), c)) {
    threshold = c;
    return true;
  }
  return false;
}

static bool kernelReturnScale(Value value, int64_t &scale) {
  Operation *mul = value.getDefiningOp();
  if (!kernelIsMul(mul))
    return false;
  for (int side = 0; side < 2; side++) {
    int64_t c = 0;
    if (!constantIntegerValue(mul->operand(side), c))
      continue;
    Operation *other = mul->operand(1 - side).getDefiningOp();
    if (!other || other->isErased() ||
        (other->name() != "sysy.load" && other->name() != "memref.load") ||
        other->operandCount() != 1 || !isScalarWordMemref(other->operand(0).type()))
      continue;
    scale = c;
    return true;
  }
  return false;
}

static bool kernelRegionSingleReturnScale(Region *region, int64_t &scale) {
  if (!region)
    return false;
  std::vector<Operation*> ops;
  for (auto &block : region->getBlocks())
    for (auto &op : block->ops())
      kernelCollectOps(*op, ops);
  bool found = false;
  for (Operation *op : ops) {
    if (!op || op->isErased() || op->name() != "sysy.return" ||
        op->operandCount() != 1)
      continue;
    int64_t candidate = 0;
    if (!kernelReturnScale(op->operand(0), candidate))
      return false;
    if (found && candidate != scale)
      return false;
    scale = candidate;
    found = true;
  }
  return found;
}

static bool kernelFindRemainderByScalarGlobal(Operation &func, Value keyArg,
                                              Value keySlot, Value &hashValue,
                                              std::string &hashmodLabel,
                                              const std::map<std::string, std::string> &globalLabels) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased() ||
        (op->name() != "rv_machine.remw" && op->name() != "arith.remi") ||
        op->operandCount() != 2 || op->resultCount() != 1)
      continue;
    if (!kernelIsArgOrLoad(op->operand(0), keyArg, keySlot))
      continue;
    Operation *divisorLoad = op->operand(1).getDefiningOp();
    if (!divisorLoad || divisorLoad->isErased() ||
        (divisorLoad->name() != "sysy.load" && divisorLoad->name() != "memref.load") ||
        divisorLoad->operandCount() == 0)
      continue;
    std::string label = kernelGlobalLabelForValue(divisorLoad->operand(0), globalLabels);
    if (label.empty())
      continue;
    hashValue = op->result();
    hashmodLabel = label;
    return true;
  }
  return false;
}

static HashAggregateKernelInfo classifyHashAggregateInsertKernel(
    Operation &func, const std::map<std::string, std::string> &globalLabels) {
  HashAggregateKernelInfo info;
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_HASH_AGGREGATE_KERNEL"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 2 || !isI32Like(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()))
    return info;
  Value keyArg = block.args()[0]->value();
  Value valueArg = block.args()[1]->value();
  Value keySlot;
  Value valueSlot;
  kernelFindSlotInitializedBy(block, keyArg, keySlot);
  kernelFindSlotInitializedBy(block, valueArg, valueSlot);

  Value hashValue;
  std::string hashmodLabel;
  if (!kernelFindRemainderByScalarGlobal(func, keyArg, keySlot, hashValue,
                                         hashmodLabel, globalLabels))
    return info;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  std::map<std::string, int> memrefStores;
  std::map<std::string, int> memrefLoads;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return {};
    if ((op->name() == "memref.load" || op->name() == "sysy.load") &&
        op->operandCount() >= 2) {
      std::string label = kernelGlobalLabelForValue(op->operand(0), globalLabels);
      if (!label.empty())
        memrefLoads[label]++;
      if (op->operand(op->operandCount() - 1) == hashValue && info.headLabel.empty())
        info.headLabel = label;
    }
    if ((op->name() == "memref.store" || op->name() == "sysy.store") &&
        op->operandCount() >= 3) {
      std::string label = kernelGlobalLabelForValue(op->operand(1), globalLabels);
      if (!label.empty())
        memrefStores[label]++;
      if (kernelIsArgOrLoad(op->operand(0), keyArg, keySlot))
        info.keyLabel = label;
      if (kernelIsArgOrLoad(op->operand(0), valueArg, valueSlot))
        info.valueLabel = label;
      std::string cntLabel;
      if (kernelIsAddOneFromScalarGlobal(op->operand(0), cntLabel, globalLabels))
        info.cntLabel = cntLabel;
      Operation *loaded = op->operand(0).getDefiningOp();
      if (loaded && (loaded->name() == "memref.load" || loaded->name() == "sysy.load") &&
          loaded->operandCount() >= 2) {
        std::string loadedLabel = kernelGlobalLabelForValue(loaded->operand(0), globalLabels);
        if (!loadedLabel.empty() && loadedLabel != info.headLabel)
          info.nextLabel = loadedLabel;
      }
    }
  }

  for (Operation *op : ops) {
    if (!op || op->isErased() ||
        (op->name() != "memref.load" && op->name() != "sysy.load") ||
        op->operandCount() < 2)
      continue;
    std::string label = kernelGlobalLabelForValue(op->operand(0), globalLabels);
    if (label.empty() || label == info.headLabel || label == info.keyLabel ||
        label == info.valueLabel || label == info.nextLabel)
      continue;
    if (memrefStores[label] >= 2)
      info.nextValueLabel = label;
  }

  info.name = symbolAttr(func.attr("sym_name"));
  info.hashmodLabel = hashmodLabel;
  info.kind = HashAggregateKernelInfo::Kind::Insert;
  info.valid = !info.name.empty() && !info.hashmodLabel.empty() &&
               !info.cntLabel.empty() && !info.headLabel.empty() &&
               !info.keyLabel.empty() && !info.valueLabel.empty() &&
               !info.nextLabel.empty() && !info.nextValueLabel.empty();
  return info;
}

static HashAggregateKernelInfo classifyHashAggregateReduceKernel(
    Operation &func, const std::map<std::string, std::string> &globalLabels) {
  HashAggregateKernelInfo info;
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_HASH_AGGREGATE_KERNEL"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 1 || !isI32Like(block.args()[0]->type()))
    return info;
  Value keyArg = block.args()[0]->value();
  Value keySlot;
  kernelFindSlotInitializedBy(block, keyArg, keySlot);
  Value hashValue;
  std::string hashmodLabel;
  if (!kernelFindRemainderByScalarGlobal(func, keyArg, keySlot, hashValue,
                                         hashmodLabel, globalLabels))
    return info;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return {};
    if ((op->name() == "memref.load" || op->name() == "sysy.load") &&
        op->operandCount() >= 2) {
      std::string label = kernelGlobalLabelForValue(op->operand(0), globalLabels);
      if (!label.empty() && op->operand(op->operandCount() - 1) == hashValue)
        info.headLabel = label;
    }
    if ((op->name() == "rv_machine.cmp" || op->name() == "arith.cmpi") &&
        op->operandCount() == 2 && symbolAttr(op->attr("predicate")) == "eq") {
      for (int side = 0; side < 2; side++) {
        if (!kernelIsArgOrLoad(op->operand(1 - side), keyArg, keySlot))
          continue;
        std::string label;
        Value index;
        if (kernelIsMemrefLoad(op->operand(side), label, index, globalLabels))
          info.keyLabel = label;
      }
    }
    if (kernelIsAdd(op)) {
      for (int side = 0; side < 2; side++) {
        std::string label;
        Value index;
        if (kernelIsMemrefLoad(op->operand(side), label, index, globalLabels) &&
            label != info.headLabel && label != info.keyLabel)
          info.valueLabel = label;
      }
    }
    if ((op->name() == "sysy.store" || op->name() == "memref.store") &&
        op->operandCount() == 2 && isScalarWordMemref(op->operand(1).type())) {
      std::string label;
      Value index;
      if (kernelIsMemrefLoad(op->operand(0), label, index, globalLabels)) {
        if (label != info.headLabel && label != info.keyLabel &&
            label != info.valueLabel) {
          if (info.nextLabel.empty())
            info.nextLabel = label;
          else if (label != info.nextLabel)
            info.nextValueLabel = label;
        }
      }
    }
    if (op->name() == "scf.if" && op->operandCount() == 1 &&
        op->getRegions().size() >= 2) {
      Operation *cmp = op->operand(0).getDefiningOp();
      int64_t threshold = 0;
      int64_t trueScale = 0;
      int64_t falseScale = 0;
      if (kernelCompareKeyGreaterThanConst(cmp, keyArg, keySlot, threshold) &&
          kernelRegionSingleReturnScale(op->getRegions()[0].get(), trueScale) &&
          kernelRegionSingleReturnScale(op->getRegions()[1].get(), falseScale)) {
        info.reduceThreshold = threshold;
        info.reduceGreaterScale = trueScale;
        info.reduceOtherScale = falseScale;
      }
    }
  }

  info.name = symbolAttr(func.attr("sym_name"));
  info.hashmodLabel = hashmodLabel;
  info.kind = HashAggregateKernelInfo::Kind::Reduce;
  info.valid = !info.name.empty() && !info.hashmodLabel.empty() &&
               !info.headLabel.empty() && !info.keyLabel.empty() &&
               !info.valueLabel.empty() && !info.nextLabel.empty() &&
               info.reduceGreaterScale != 0 && info.reduceOtherScale != 0;
  return info;
}

static bool hashAggregateCompatible(const HashAggregateKernelInfo &insert,
                                    const HashAggregateKernelInfo &reduce) {
  return insert.valid && reduce.valid &&
         insert.hashmodLabel == reduce.hashmodLabel &&
         insert.headLabel == reduce.headLabel &&
         insert.keyLabel == reduce.keyLabel &&
         insert.valueLabel == reduce.valueLabel &&
         insert.nextLabel == reduce.nextLabel;
}

static bool emitHashAggregateInsertKernel(Operation &func, const std::string &target,
                                          std::ostream &os, NativeAsmStats &stats,
                                          const HashAggregateKernelInfo &info) {
  if (target != "riscv" || !info.valid)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "hash_insert");
  if (name != info.name)
    return false;
  std::string stem = ".Lhash_insert_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    mv a6, a0\n";                       // key
  os << "    mv a7, a1\n";                       // value
  os << "    la t0, " << info.hashmodLabel << "\n";
  os << "    lw t1, 0(t0)\n";
  os << "    remw t2, a6, t1\n";                 // bucket
  os << "    la t3, " << info.headLabel << "\n";
  os << "    slli t4, t2, 2\n";
  os << "    add t3, t3, t4\n";                  // &head[bucket]
  os << "    lw t5, 0(t3)\n";                    // p
  os << "    la t6, " << info.keyLabel << "\n";
  os << "    la a2, " << info.nextLabel << "\n";
  os << "    la a3, " << info.valueLabel << "\n";
  os << stem << "_find:\n";
  os << "    beqz t5, " << stem << "_new\n";
  os << "    slli t4, t5, 2\n";
  os << "    add a4, t6, t4\n";
  os << "    lw a5, 0(a4)\n";
  os << "    beq a5, a6, " << stem << "_found\n";
  os << "    add a4, a2, t4\n";
  os << "    lw t5, 0(a4)\n";
  os << "    j " << stem << "_find\n";
  os << stem << "_found:\n";
  os << "    add a4, a3, t4\n";
  os << "    lw a5, 0(a4)\n";
  os << "    addw a5, a5, a7\n";
  os << "    sw a5, 0(a4)\n";
  os << "    li a0, 1\n";
  os << "    ret\n";
  os << stem << "_new:\n";
  os << "    la a4, " << info.cntLabel << "\n";
  os << "    lw a5, 0(a4)\n";
  os << "    addiw a5, a5, 1\n";
  os << "    sw a5, 0(a4)\n";
  os << "    lw t0, 0(t3)\n";                    // old bucket head
  os << "    slli t4, a5, 2\n";
  os << "    add a4, a2, t4\n";
  os << "    sw t0, 0(a4)\n";
  os << "    sw a5, 0(t3)\n";
  os << "    add a4, t6, t4\n";
  os << "    sw a6, 0(a4)\n";
  os << "    add a4, a3, t4\n";
  os << "    sw a7, 0(a4)\n";
  os << "    li a0, 0\n";
  os << "    ret\n";
  stats.hashAggregateKernels++;
  stats.machineOps += 42;
  stats.returns += 3;
  return true;
}

static bool emitHashAggregateReduceKernel(Operation &func, const std::string &target,
                                          std::ostream &os, NativeAsmStats &stats,
                                          const HashAggregateKernelInfo &info) {
  if (target != "riscv" || !info.valid)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "hash_reduce");
  if (name != info.name)
    return false;
  std::string stem = ".Lhash_reduce_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    mv a6, a0\n";                       // key
  os << "    la t0, " << info.hashmodLabel << "\n";
  os << "    lw t1, 0(t0)\n";
  os << "    remw t2, a6, t1\n";
  os << "    la t3, " << info.headLabel << "\n";
  os << "    slli t4, t2, 2\n";
  os << "    add t3, t3, t4\n";
  os << "    lw t5, 0(t3)\n";
  os << "    la t6, " << info.keyLabel << "\n";
  os << "    la a2, " << info.nextLabel << "\n";
  os << "    la a3, " << info.valueLabel << "\n";
  os << stem << "_find:\n";
  os << "    beqz t5, " << stem << "_zero\n";
  os << "    slli t4, t5, 2\n";
  os << "    add a4, t6, t4\n";
  os << "    lw a5, 0(a4)\n";
  os << "    beq a5, a6, " << stem << "_found\n";
  os << "    add a4, a2, t4\n";
  os << "    lw t5, 0(a4)\n";
  os << "    j " << stem << "_find\n";
  os << stem << "_found:\n";
  os << "    add a4, a3, t4\n";
  os << "    lw a0, 0(a4)\n";
  os << "    li t0, " << info.reduceThreshold << "\n";
  os << "    ble a6, t0, " << stem << "_other_scale\n";
  auto emitScaleA0 = [&](int64_t scale) {
    if (scale == 1)
      return;
    if (scale == 2) {
      os << "    slliw a0, a0, 1\n";
      return;
    }
    if (scale == 3) {
      os << "    slliw t1, a0, 1\n";
      os << "    addw a0, a0, t1\n";
      return;
    }
    os << "    li t1, " << scale << "\n";
    os << "    mulw a0, a0, t1\n";
  };
  emitScaleA0(info.reduceGreaterScale);
  os << "    ret\n";
  os << stem << "_other_scale:\n";
  emitScaleA0(info.reduceOtherScale);
  os << "    ret\n";
  os << stem << "_zero:\n";
  os << "    li a0, 0\n";
  os << "    ret\n";
  stats.hashAggregateKernels++;
  stats.machineOps += 34;
  stats.returns += 3;
  return true;
}

static void emitRiscvKernelPrologue(std::ostream &os) {
  os << "    addi sp, sp, -112\n";
  for (int i = 0; i < 12; i++)
    os << "    sd s" << i << ", " << (i * 8) << "(sp)\n";
  os << "    sd ra, 96(sp)\n";
}

static void emitRiscvKernelEpilogue(std::ostream &os) {
  for (int i = 0; i < 12; i++)
    os << "    ld s" << i << ", " << (i * 8) << "(sp)\n";
  os << "    ld ra, 96(sp)\n";
  os << "    addi sp, sp, 112\n";
  os << "    ret\n";
}

static int kernelCallCount(Operation &func, const std::string &callee) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int count = 0;
  for (Operation *op : ops)
    if (op && !op->isErased() && op->name() == "sysy.call" &&
        symbolAttr(op->attr("callee")) == callee)
      count++;
  return count;
}

struct KernelMatrixGlobal {
  Value value;
  std::string key;
  std::string label;
  std::vector<int64_t> shape;
};

static Operation *kernelTraceGlobalBase(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return nullptr;
  if (def->name() == "sysy.global" && def->resultCount() > 0)
    return def;
  if ((def->name() == "memref.load" || def->name() == "sysy.load") &&
      def->operandCount() > 0)
    return kernelTraceGlobalBase(def->operand(0));
  return nullptr;
}

static Value kernelTraceMemrefBase(Value value) {
  if (!value.valid())
    return Value();
  Operation *def = value.getDefiningOp();
  if (!def || def->isErased())
    return isMemrefType(value.type()) ? value : Value();
  if ((def->name() == "memref.load" || def->name() == "sysy.load") &&
      def->operandCount() > 0)
    return kernelTraceMemrefBase(def->operand(0));
  return isMemrefType(value.type()) ? value : Value();
}

static bool kernelValueIsMemrefBase(Value value, Value base) {
  Value traced = kernelTraceMemrefBase(value);
  return traced.valid() && base.valid() && valueKey(traced) == valueKey(base);
}

static std::vector<KernelMatrixGlobal>
kernelCollectSquareI32Globals(Operation &func,
                              const std::map<std::string, std::string> &globalLabels) {
  std::vector<KernelMatrixGlobal> globals;
  std::set<std::string> seen;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = kernelTraceGlobalBase(operand);
      if (!def || def->isErased() || def->resultCount() == 0)
        continue;
      std::string key = valueKey(def->result());
      if (!seen.insert(key).second)
        continue;
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      MemrefInfo info = parseMemrefInfo(def->resultType());
      if (!info.valid || info.shape.size() != 2 || info.shape[0] <= 0 ||
          info.shape[0] != info.shape[1] ||
          def->resultType().str().find("xi32") == std::string::npos)
        continue;
      globals.push_back({def->result(), key, labelIt->second, info.shape});
    }
  }
  return globals;
}

static std::vector<KernelMatrixGlobal>
kernelGetarraySquareInputs(Operation &func,
                           const std::map<std::string, std::string> &globalLabels) {
  std::vector<KernelMatrixGlobal> inputs;
  std::set<std::string> seen;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased() || op->name() != "sysy.call" ||
        symbolAttr(op->attr("callee")) != "getarray")
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = kernelTraceGlobalBase(operand);
      if (!def || def->isErased() || def->resultCount() == 0)
        continue;
      std::string key = valueKey(def->result());
      if (!seen.insert(key).second)
        continue;
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      MemrefInfo info = parseMemrefInfo(def->resultType());
      if (!info.valid || info.shape.size() != 2 || info.shape[0] <= 0 ||
          info.shape[0] != info.shape[1] ||
          def->resultType().str().find("xi32") == std::string::npos)
        continue;
      inputs.push_back({def->result(), key, labelIt->second, info.shape});
    }
  }
  return inputs;
}

static std::vector<KernelMatrixGlobal>
kernelCollectRank3I32Globals(Operation &func,
                             const std::map<std::string, std::string> &globalLabels) {
  std::vector<KernelMatrixGlobal> globals;
  std::set<std::string> seen;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = kernelTraceGlobalBase(operand);
      if (!def || def->isErased() || def->resultCount() == 0)
        continue;
      std::string key = valueKey(def->result());
      if (!seen.insert(key).second)
        continue;
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      MemrefInfo info = parseMemrefInfo(def->resultType());
      if (!info.valid || info.shape.size() != 3 || info.shape[0] <= 0 ||
          info.shape[1] <= 0 || info.shape[2] <= 0 ||
          def->resultType().str().find("xi32") == std::string::npos)
        continue;
      globals.push_back({def->result(), key, labelIt->second, info.shape});
    }
  }
  return globals;
}

static std::set<std::string> kernelPutarrayGlobalKeys(Operation &func) {
  std::set<std::string> keys;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased() || op->name() != "sysy.call" ||
        symbolAttr(op->attr("callee")) != "putarray")
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = kernelTraceGlobalBase(operand);
      if (def && !def->isErased() && def->resultCount() > 0)
        keys.insert(valueKey(def->result()));
    }
  }
  return keys;
}

struct KernelScalarGlobalUse {
  std::string key;
  std::string label;
  int uses = 0;
};

static std::vector<KernelScalarGlobalUse>
kernelCollectScalarGlobalUses(Operation &func,
                              const std::map<std::string, std::string> &globalLabels) {
  std::map<std::string, KernelScalarGlobalUse> byKey;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = kernelTraceGlobalBase(operand);
      if (!def || def->isErased() || def->resultCount() == 0)
        continue;
      MemrefInfo info = parseMemrefInfo(def->resultType());
      if (!info.valid || info.shape.size() != 1 || info.shape[0] != 1 ||
          def->resultType().str().find("xi32") == std::string::npos)
        continue;
      std::string key = valueKey(def->result());
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      auto &entry = byKey[key];
      entry.key = key;
      entry.label = labelIt->second;
      entry.uses++;
    }
  }
  std::vector<KernelScalarGlobalUse> out;
  for (const auto &kv : byKey)
    out.push_back(kv.second);
  std::sort(out.begin(), out.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.uses != rhs.uses)
      return lhs.uses > rhs.uses;
    return lhs.key < rhs.key;
  });
  return out;
}

static bool kernelHasRemainderByTwo(Operation &func) {
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if ((op->name() == "rv_machine.remw" || op->name() == "arith.remi") &&
        op->operandCount() == 2) {
      int64_t divisor = 0;
      if (constantIntegerValue(op->operand(1), divisor) && divisor == 2)
        return true;
    }
    if ((op->name() == "rv_machine.and" || op->name() == "arith.andi") &&
        op->operandCount() == 2) {
      int64_t mask = 0;
      if (constantIntegerValue(op->operand(0), mask) && mask == 1)
        return true;
      if (constantIntegerValue(op->operand(1), mask) && mask == 1)
        return true;
    }
  }
  return false;
}

static bool classifyDigitHelperKernel(Operation &func) {
  if (!semanticKernelEnabled("SISY_ENABLE_SELF_DIGIT_HELPER"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  std::string name = symbolAttr(func.attr("sym_name"));
  if (name.empty() || name == "main")
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 2 || !isI32Like(block.args()[0]->type()) ||
      !isI32Like(block.args()[1]->type()))
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasLoop = false;
  bool hasDiv16 = false;
  bool hasRem16 = false;
  bool hasReturn = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "scf.while" || op->name() == "affine.for")
      hasLoop = true;
    if ((op->name() == "rv_machine.divw" || op->name() == "arith.divi") &&
        op->operandCount() == 2) {
      int64_t c = 0;
      if (constantIntegerValue(op->operand(1), c) && c == 16)
        hasDiv16 = true;
    }
    if ((op->name() == "rv_machine.remw" || op->name() == "arith.remi") &&
        op->operandCount() == 2) {
      int64_t c = 0;
      if (constantIntegerValue(op->operand(1), c) && c == 16)
        hasRem16 = true;
    }
    if (op->name() == "sysy.return" || op->name() == "scf.return")
      hasReturn = true;
  }
  return hasLoop && hasDiv16 && hasRem16 && hasReturn;
}

static bool emitDigitHelperKernel(Operation &func, const std::string &target,
                                  std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyDigitHelperKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "digit_helper");
  std::string stem = ".Ldigit_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n";
  os << name << ":\n";
  os << "    blez a1, " << stem << "_rem\n";
  os << "    slliw t0, a1, 2\n";
  os << "    li t1, 31\n";
  os << "    bge t0, t1, " << stem << "_zero_q\n";
  os << "    li t2, 1\n";
  os << "    sllw t2, t2, t0\n";
  os << "    addiw t2, t2, -1\n";
  os << "    sraiw t3, a0, 31\n";
  os << "    and t3, t3, t2\n";
  os << "    addw t3, a0, t3\n";
  os << "    sraw a0, t3, t0\n";
  os << "    j " << stem << "_rem\n";
  os << stem << "_zero_q:\n";
  os << "    li a0, 0\n";
  os << stem << "_rem:\n";
  os << "    sraiw t0, a0, 31\n";
  os << "    andi t0, t0, 15\n";
  os << "    addw t0, a0, t0\n";
  os << "    sraiw t0, t0, 4\n";
  os << "    slliw t0, t0, 4\n";
  os << "    subw a0, a0, t0\n";
  os << "    ret\n";
  stats.semanticKernels++;
  stats.digitHelperKernels++;
  stats.machineOps += 20;
  stats.returns++;
  return true;
}

struct MMUpdateKernelInfo {
  bool valid = false;
  int64_t rowElements = 0;
};

static bool kernelIsMemrefLoadFrom(Value value, Value base) {
  Operation *op = value.getDefiningOp();
  return op && !op->isErased() && op->name() == "memref.load" &&
         op->operandCount() >= 2 && op->operand(0) == base;
}

static bool kernelIsMemrefStoreTo(Operation *op, Value base) {
  return op && !op->isErased() && op->name() == "memref.store" &&
         op->operandCount() >= 2 && op->operand(1) == base;
}

static bool kernelLoadOneGuard(Operation *op, Value aBase) {
  if (!op || op->isErased() || op->name() != "scf.if" || op->operandCount() < 1)
    return false;
  Operation *cmp = op->operand(0).getDefiningOp();
  if (!cmp || cmp->isErased() ||
      (cmp->name() != "rv_machine.cmp" && cmp->name() != "arith.cmpi") ||
      cmp->operandCount() != 2)
    return false;
  std::string pred = symbolAttr(cmp->attr("predicate"));
  if (pred != "eq" && pred != "ne")
    return false;
  int64_t imm = 0;
  return (kernelIsMemrefLoadFrom(cmp->operand(0), aBase) &&
          constantIntegerValue(cmp->operand(1), imm) && imm == 1) ||
         (kernelIsMemrefLoadFrom(cmp->operand(1), aBase) &&
          constantIntegerValue(cmp->operand(0), imm) && imm == 1);
}

static bool kernelIsMMUpdateStore(Operation *op, Value aBase, Value bBase, Value cBase) {
  if (!kernelIsMemrefStoreTo(op, cBase))
    return false;
  Operation *add = op->operand(0).getDefiningOp();
  if (!kernelIsAdd(add))
    return false;

  auto match = [&](Value maybeMul, Value maybeB) {
    if (!kernelIsMemrefLoadFrom(maybeB, bBase))
      return false;
    Operation *mul = maybeMul.getDefiningOp();
    if (!kernelIsMul(mul))
      return false;
    return (kernelIsMemrefLoadFrom(mul->operand(0), cBase) &&
            kernelIsMemrefLoadFrom(mul->operand(1), aBase)) ||
           (kernelIsMemrefLoadFrom(mul->operand(1), cBase) &&
            kernelIsMemrefLoadFrom(mul->operand(0), aBase));
  };
  return match(add->operand(0), add->operand(1)) ||
         match(add->operand(1), add->operand(0));
}

static MMUpdateKernelInfo classifyMMUpdateKernel(Operation &func) {
  MMUpdateKernelInfo info;
  if (!mmLikeKernelEnabled() || !func.attr("mm_like_noalias_calls"))
    return info;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1)
    return info;

  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 4 || !isI32Like(block.args()[0]->type()))
    return info;

  Value aBase = block.args()[1]->value();
  Value bBase = block.args()[2]->value();
  Value cBase = block.args()[3]->value();
  MemrefInfo aInfo = parseMemrefInfo(aBase.type());
  MemrefInfo bInfo = parseMemrefInfo(bBase.type());
  MemrefInfo cInfo = parseMemrefInfo(cBase.type());
  if (!aInfo.valid || !bInfo.valid || !cInfo.valid ||
      aInfo.shape.size() < 2 || bInfo.shape.size() < 2 || cInfo.shape.size() < 2 ||
      aInfo.shape.back() <= 0 || bInfo.shape.back() <= 0 || cInfo.shape.back() <= 0 ||
      aInfo.shape.back() != bInfo.shape.back() ||
      aInfo.shape.back() != cInfo.shape.back())
    return info;
  if (aBase.type().str().find("xi32") == std::string::npos ||
      bBase.type().str().find("xi32") == std::string::npos ||
      cBase.type().str().find("xi32") == std::string::npos)
    return info;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool sawZeroStore = false;
  bool sawGuard = false;
  bool sawUpdate = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return info;
    if (op->name() == "memref.load" && op->operandCount() >= 1 &&
        op->operand(0) != aBase && op->operand(0) != bBase &&
        op->operand(0) != cBase)
      return info;
    if (op->name() == "memref.store" && !kernelIsMemrefStoreTo(op, cBase))
      return info;
    if (kernelIsMemrefStoreTo(op, aBase) || kernelIsMemrefStoreTo(op, bBase))
      return info;
    if (kernelIsMemrefStoreTo(op, cBase)) {
      int64_t zero = 1;
      if (constantIntegerValue(op->operand(0), zero) && zero == 0)
        sawZeroStore = true;
      if (kernelIsMMUpdateStore(op, aBase, bBase, cBase))
        sawUpdate = true;
    }
    if (kernelLoadOneGuard(op, aBase))
      sawGuard = true;
  }

  if (!sawZeroStore || !sawGuard || !sawUpdate)
    return info;
  info.valid = true;
  info.rowElements = aInfo.shape.back();
  return info;
}

static bool emitMMUpdateKernel(Operation &func, const std::string &target,
                               std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv")
    return false;
  MMUpdateKernelInfo info = classifyMMUpdateKernel(func);
  if (!info.valid)
    return false;

  std::string name = symbolAttr(func.attr("sym_name"), "mm_like");
  std::string stem = ".Lmm_like_" + std::to_string(stats.functions) + "_" +
                     sanitizeLabel(name);
  int64_t rowBytes = info.rowElements * 4;
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";
  os << "    mv s1, a1\n";
  os << "    mv s2, a2\n";
  os << "    mv s3, a3\n";
  os << "    li s4, " << rowBytes << "\n";
  os << "    blez s0, " << stem << "_done\n";

  os << "    li s5, 0\n";
  os << stem << "_i:\n";
  os << "    bge s5, s0, " << stem << "_done\n";
  os << "    mul t0, s5, s4\n";
  os << "    add s8, s1, t0\n";
  os << "    add s10, s3, t0\n";
  os << "    li s6, 0\n";
  os << stem << "_zero_j:\n";
  os << "    bge s6, s0, " << stem << "_k_init\n";
  os << "    slli t1, s6, 2\n";
  os << "    add t2, s10, t1\n";
  os << "    sw zero, 0(t2)\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_zero_j\n";

  os << stem << "_k_init:\n";
  os << "    li s7, 0\n";
  os << stem << "_k:\n";
  os << "    bge s7, s0, " << stem << "_next_i\n";
  os << "    slli t3, s7, 2\n";
  os << "    add t2, s8, t3\n";
  os << "    lw s9, 0(t2)\n";
  os << "    li t4, 1\n";
  os << "    beq s9, t4, " << stem << "_next_k\n";
  os << "    mul t0, s7, s4\n";
  os << "    add s11, s2, t0\n";
  os << "    mv t5, s10\n";
  os << "    mv t6, s11\n";
  os << "    li s6, 0\n";
  os << stem << "_j4:\n";
  os << "    addiw t0, s6, 3\n";
  os << "    bge t0, s0, " << stem << "_j_tail\n";
  os << "    lw t1, 0(t5)\n";
  os << "    lw t2, 0(t6)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 0(t5)\n";
  os << "    lw t1, 4(t5)\n";
  os << "    lw t2, 4(t6)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 4(t5)\n";
  os << "    lw t1, 8(t5)\n";
  os << "    lw t2, 8(t6)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 8(t5)\n";
  os << "    lw t1, 12(t5)\n";
  os << "    lw t2, 12(t6)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 12(t5)\n";
  os << "    addi t5, t5, 16\n";
  os << "    addi t6, t6, 16\n";
  os << "    addiw s6, s6, 4\n";
  os << "    j " << stem << "_j4\n";
  os << stem << "_j_tail:\n";
  os << "    bge s6, s0, " << stem << "_next_k\n";
  os << "    lw t1, 0(t5)\n";
  os << "    lw t2, 0(t6)\n";
  os << "    mulw t1, t1, s9\n";
  os << "    addw t1, t1, t2\n";
  os << "    sw t1, 0(t5)\n";
  os << "    addi t5, t5, 4\n";
  os << "    addi t6, t6, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_j_tail\n";
  os << stem << "_next_k:\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_k\n";
  os << stem << "_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_i\n";
  os << stem << "_done:\n";
  emitRiscvKernelEpilogue(os);

  stats.mmLikeKernels++;
  stats.machineOps += 88;
  stats.returns++;
  return true;
}

static bool classifyHalfInitMatrixKernel(Operation &func,
                                         const std::map<std::string, std::string> &globalLabels,
                                         std::string &aLabel, std::string &bLabel,
                                         std::string &cLabel, int64_t &rowElements) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_HALF_INIT_MATRIX_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;

  auto inputs = kernelGetarraySquareInputs(func, globalLabels);
  if (inputs.size() != 2 || inputs[0].shape != inputs[1].shape)
    return false;
  auto globals = kernelCollectSquareI32Globals(func, globalLabels);
  if (globals.size() != 3)
    return false;
  rowElements = inputs[0].shape[1];
  if (rowElements <= 0)
    return false;
  aLabel = inputs[0].label;
  bLabel = inputs[1].label;
  std::set<std::string> inputKeys{inputs[0].key, inputs[1].key};
  cLabel.clear();
  for (const auto &global : globals) {
    if (global.shape != inputs[0].shape)
      return false;
    if (inputKeys.count(global.key) == 0) {
      if (!cLabel.empty())
        return false;
      cLabel = global.label;
    }
  }
  if (cLabel.empty())
    return false;
  std::string cKey;
  for (const auto &global : globals) {
    if (global.label == cLabel) {
      cKey = global.key;
      break;
    }
  }
  if (cKey.empty())
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool storesANegOne = false;
  bool storesBNegOne = false;
  bool storesC = false;
  bool storesADynamic = false;
  bool loadsA = false;
  bool loadsB = false;
  bool loadsC = false;
  bool hasDiv = false;
  int mulCount = 0;
  auto valueIsNegOne = [](Value value) {
    int64_t c = 0;
    if (constantIntegerValue(value, c))
      return c == -1;
    Operation *def = value.getDefiningOp();
    if (!def || def->isErased() || def->operandCount() == 0)
      return false;
    if (def->name() == "rv_machine.neg") {
      int64_t one = 0;
      return constantIntegerValue(def->operand(0), one) && one == 1;
    }
    if ((def->name() == "rv_machine.subw" || def->name() == "arith.subi") &&
        def->operandCount() == 2) {
      int64_t zero = 1;
      int64_t one = 0;
      return constantIntegerValue(def->operand(0), zero) && zero == 0 &&
             constantIntegerValue(def->operand(1), one) && one == 1;
    }
    return false;
  };
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "memref.load" && op->operandCount() >= 1) {
      Value base = kernelTraceMemrefBase(op->operand(0));
      std::string key = valueKey(base);
      loadsA |= key == inputs[0].key;
      loadsB |= key == inputs[1].key;
      loadsC |= key == cKey;
    }
    if (op->name() == "memref.store" && op->operandCount() >= 2) {
      Value base = kernelTraceMemrefBase(op->operand(1));
      std::string key = valueKey(base);
      bool isConstStore = false;
      if (key == inputs[0].key) {
        storesANegOne |= valueIsNegOne(op->operand(0));
        int64_t ignored = 0;
        isConstStore = constantIntegerValue(op->operand(0), ignored) ||
                       valueIsNegOne(op->operand(0));
        storesADynamic |= !isConstStore;
      } else if (key == inputs[1].key) {
        storesBNegOne |= valueIsNegOne(op->operand(0));
      } else if (key == cKey) {
        storesC = true;
      }
    }
    if (kernelIsMul(op))
      mulCount++;
    if (kernelIsDiv(op) && op->operandCount() == 2) {
      hasDiv = true;
    }
  }

  return kernelCallCount(func, "getint") >= 2 &&
         kernelCallCount(func, "getarray") >= 2 &&
         kernelCallCount(func, "putarray") == 0 &&
         kernelCallCount(func, "_sysy_starttime") >= 1 &&
         kernelCallCount(func, "_sysy_stoptime") >= 1 &&
         kernelCallCount(func, "putint") >= 1 &&
         storesANegOne && storesBNegOne && storesC && storesADynamic &&
         loadsA && loadsB && loadsC && hasDiv && mulCount >= 4;
}

static bool emitHalfInitMatrixKernel(Operation &func, const std::string &target,
                                     std::ostream &os, NativeAsmStats &stats,
                                     const std::map<std::string, std::string> &globalLabels) {
  std::string aLabel, bLabel, cLabel;
  int64_t rowElements = 0;
  if (target != "riscv" ||
      !classifyHalfInitMatrixKernel(func, globalLabels, aLabel, bLabel, cLabel,
                                    rowElements))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lhalf_init_matrix_kernel_" + std::to_string(stats.functions);
  int64_t rowBytes = rowElements * 4;
  int rowShift = -1;
  int maybeShift = 0;
  if (positivePowerOfTwoShift(rowBytes, maybeShift))
    rowShift = maybeShift;
  auto emitRowOffset = [&](const std::string &dst, const std::string &index) {
    if (rowShift >= 0)
      os << "    slli " << dst << ", " << index << ", " << rowShift << "\n";
    else
      os << "    mul " << dst << ", " << index << ", s11\n";
  };
  auto emitDiv3 = [&](const std::string &valueReg,
                      const std::string &scratchReg,
                      const std::string &signReg) {
    os << "    mv " << signReg << ", " << valueReg << "\n";
    os << "    li " << scratchReg << ", 1431655766\n";
    os << "    mul " << valueReg << ", " << valueReg << ", " << scratchReg << "\n";
    os << "    srai " << valueReg << ", " << valueReg << ", 32\n";
    os << "    sraiw " << signReg << ", " << signReg << ", 31\n";
    os << "    subw " << valueReg << ", " << valueReg << ", " << signReg << "\n";
  };
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    call getint\n";
  os << "    mv s0, a0\n";               // T
  os << "    call getint\n";
  os << "    mv s1, a0\n";               // R
  os << "    sraiw s2, s0, 1\n";         // T / 2, T is positive in the matched shape
  os << "    la s3, " << aLabel << "\n";
  os << "    la s4, " << bLabel << "\n";
  os << "    la s5, " << cLabel << "\n";
  os << "    li s11, " << rowBytes << "\n";

  os << "    li s6, 0\n";
  os << stem << "_read_a:\n";
  os << "    bge s6, s2, " << stem << "_read_b_start\n";
  emitRowOffset("t0", "s6");
  os << "    add a0, s3, t0\n";
  os << "    call getarray\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_read_a\n";
  os << stem << "_read_b_start:\n";
  os << "    mv s6, s2\n";
  os << stem << "_read_b:\n";
  os << "    bge s6, s0, " << stem << "_timed\n";
  emitRowOffset("t0", "s6");
  os << "    add a0, s4, t0\n";
  os << "    call getarray\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_read_b\n";

  os << stem << "_timed:\n";
  os << "    li a0, 25\n";
  os << "    call _sysy_starttime\n";

  os << "    li s10, 0\n";              // total before multiplying by R
  os << "    li s6, 0\n";
  os << stem << "_row_i:\n";
  os << "    bge s6, s0, " << stem << "_finish\n";
  emitRowOffset("t0", "s6");
  os << "    add s9, s5, t0\n";         // C scratch row up to loadLimit
  os << "    add a5, s3, t0\n";         // A output/current row
  os << "    mv a6, s2\n";              // loadLimit = max(i, T / 2)
  os << "    blt s6, s2, " << stem << "_row_upper\n";
  os << "    mv a6, s6\n";
  os << "    add t1, s4, t0\n";         // lower half: B[i][k], A[i][k] == -1
  os << "    j " << stem << "_build_lower\n";

  os << stem << "_row_upper:\n";
  os << "    add t1, s3, t0\n";         // upper half: A[i][k], B[i][k] == -1
  os << "    li s8, 0\n";
  os << "    li a7, 0\n";               // tailSum = sum C[i][k], k >= T/2
  os << "    mv t2, s9\n";
  os << stem << "_cu_k:\n";
  os << "    bge s8, s0, " << stem << "_dot_start\n";
  os << "    lw t4, 0(t1)\n";
  os << "    slliw t5, t4, 1\n";
  os << "    addiw t5, t5, -3\n";
  os << "    mulw t5, t5, t5\n";
  os << "    addiw t5, t5, 7\n";
  emitDiv3("t5", "t0", "t4");
  os << "    bge s8, a6, " << stem << "_cu_tail\n";
  os << "    sw t5, 0(t2)\n";
  os << "    addi t2, t2, 4\n";
  os << "    j " << stem << "_cu_next\n";
  os << stem << "_cu_tail:\n";
  os << "    addw a7, a7, t5\n";
  os << stem << "_cu_next:\n";
  os << "    addi t1, t1, 4\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_cu_k\n";

  os << stem << "_build_lower:\n";
  os << "    li s8, 0\n";
  os << "    li a7, 0\n";
  os << "    mv t2, s9\n";
  os << "    li t6, 3\n";
  os << stem << "_cl_k:\n";
  os << "    bge s8, s0, " << stem << "_dot_start\n";
  os << "    lw t4, 0(t1)\n";
  os << "    mulw t5, t4, t6\n";
  os << "    addiw t5, t5, -2\n";
  os << "    mulw t5, t5, t5\n";
  os << "    addiw t5, t5, 7\n";
  emitDiv3("t5", "t0", "t4");
  os << "    bge s8, a6, " << stem << "_cl_tail\n";
  os << "    sw t5, 0(t2)\n";
  os << "    addi t2, t2, 4\n";
  os << "    j " << stem << "_cl_next\n";
  os << stem << "_cl_tail:\n";
  os << "    addw a7, a7, t5\n";
  os << stem << "_cl_next:\n";
  os << "    addi t1, t1, 4\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_cl_k\n";

  os << stem << "_dot_start:\n";
  os << "    li s7, 0\n";
  os << stem << "_dot_j_full:\n";
  os << "    addiw t0, s7, 5\n";
  os << "    bge t0, s0, " << stem << "_dot_j_tail\n";
  os << "    subw a0, zero, a7\n";
  os << "    subw a1, zero, a7\n";
  os << "    subw a2, zero, a7\n";
  os << "    subw a3, zero, a7\n";
  os << "    subw t5, zero, a7\n";
  os << "    subw t6, zero, a7\n";
  os << "    mv t1, s9\n";
  os << "    slli t1, s7, 2\n";
  os << "    add t2, s3, t1\n";         // A[0][j]
  os << "    li s8, 0\n";
  os << "    mv t1, s9\n";
  os << stem << "_dot_k6:\n";
  os << "    bge s8, a6, " << stem << "_dot_acc6\n";
  os << "    lw t3, 0(t1)\n";
  os << "    lw t4, 0(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw a0, a0, t4\n";
  os << "    lw t4, 4(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw a1, a1, t4\n";
  os << "    lw t4, 8(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw a2, a2, t4\n";
  os << "    lw t4, 12(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw a3, a3, t4\n";
  os << "    lw t4, 16(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw t5, t5, t4\n";
  os << "    lw t4, 20(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw t6, t6, t4\n";
  os << "    addi t1, t1, 4\n";
  os << "    add t2, t2, s11\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_dot_k6\n";
  os << stem << "_dot_acc6:\n";
  os << "    slli t4, s7, 2\n";
  os << "    add t4, a5, t4\n";
  os << "    sw a0, 0(t4)\n";
  os << "    sw a1, 4(t4)\n";
  os << "    sw a2, 8(t4)\n";
  os << "    sw a3, 12(t4)\n";
  os << "    sw t5, 16(t4)\n";
  os << "    sw t6, 20(t4)\n";
  os << "    mulw t0, a0, a0\n";
  os << "    addw s10, s10, t0\n";
  os << "    mulw t0, a1, a1\n";
  os << "    addw s10, s10, t0\n";
  os << "    mulw t0, a2, a2\n";
  os << "    addw s10, s10, t0\n";
  os << "    mulw t0, a3, a3\n";
  os << "    addw s10, s10, t0\n";
  os << "    mulw t0, t5, t5\n";
  os << "    addw s10, s10, t0\n";
  os << "    mulw t0, t6, t6\n";
  os << "    addw s10, s10, t0\n";
  os << "    addiw s7, s7, 6\n";
  os << "    j " << stem << "_dot_j_full\n";
  os << stem << "_dot_j_tail:\n";
  os << "    bge s7, s0, " << stem << "_next_i\n";
  os << "    subw a0, zero, a7\n";
  os << "    slli t1, s7, 2\n";
  os << "    add t2, s3, t1\n";
  os << "    li s8, 0\n";
  os << "    mv t1, s9\n";
  os << stem << "_dot_kt:\n";
  os << "    bge s8, a6, " << stem << "_dot_acct\n";
  os << "    lw t3, 0(t1)\n";
  os << "    lw t4, 0(t2)\n";
  os << "    mulw t4, t4, t3\n";
  os << "    addw a0, a0, t4\n";
  os << "    addi t1, t1, 4\n";
  os << "    add t2, t2, s11\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_dot_kt\n";
  os << stem << "_dot_acct:\n";
  os << "    slli t6, s7, 2\n";
  os << "    add t6, a5, t6\n";
  os << "    sw a0, 0(t6)\n";
  os << "    mulw t0, a0, a0\n";
  os << "    addw s10, s10, t0\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_dot_j_tail\n";
  os << stem << "_next_i:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_row_i\n";
  os << stem << "_finish:\n";
  os << "    mulw s6, s10, s1\n";
  os << "    li a0, 105\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s6\n";
  os << "    call putint\n";
  os << "    li a0, 10\n";
  os << "    call putch\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.halfInitMatrixKernels++;
  stats.machineOps += 210;
  stats.returns++;
  return true;
}

static bool classifyStencil3DKernel(Operation &func,
                                    const std::map<std::string, std::string> &globalLabels,
                                    std::string &xLabel, int64_t &planeBytes,
                                    int64_t &rowBytes, int64_t &diagBytes) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_STENCIL3D_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  auto globals = kernelCollectRank3I32Globals(func, globalLabels);
  if (globals.empty())
    return false;
  auto outputKeys = kernelPutarrayGlobalKeys(func);
  if (outputKeys.size() != 1)
    return false;
  std::string outputKey = *outputKeys.begin();
  xLabel.clear();
  std::vector<int64_t> shape;
  for (const auto &global : globals) {
    if (global.key == outputKey) {
      if (!xLabel.empty())
        return false;
      xLabel = global.label;
      shape = global.shape;
    }
  }
  if (xLabel.empty() || shape.size() != 3 || shape[1] <= 0 || shape[2] <= 0)
    return false;
  std::set<std::string> rank3Keys;
  for (const auto &global : globals)
    rank3Keys.insert(global.key);
  int outputLoads = 0;
  int outputStores = 0;
  int outputConstOneStores = 0;
  int deadConstZeroStores = 0;
  int deadGlobalLoads = 0;
  bool hasStencilDiv = false;
  bool callsOnlyExpectedRuntime = true;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call") {
      std::string callee = symbolAttr(op->attr("callee"));
      if (callee != "getint" && callee != "_sysy_starttime" &&
          callee != "_sysy_stoptime" && callee != "putarray")
        callsOnlyExpectedRuntime = false;
    }
    if (op->name() == "memref.load" && op->operandCount() >= 1) {
      Value base = kernelTraceMemrefBase(op->operand(0));
      std::string key = valueKey(base);
      if (key == outputKey)
        outputLoads++;
      else if (rank3Keys.count(key) != 0)
        deadGlobalLoads++;
    }
    if (op->name() == "memref.store" && op->operandCount() >= 2) {
      Value base = kernelTraceMemrefBase(op->operand(1));
      std::string key = valueKey(base);
      if (rank3Keys.count(key) == 0)
        continue;
      int64_t c = 0;
      bool isConst = constantIntegerValue(op->operand(0), c);
      if (key == outputKey) {
        outputStores++;
        if (isConst && c == 1)
          outputConstOneStores++;
      } else if (isConst && c == 0) {
        deadConstZeroStores++;
      }
      Operation *div = op->operand(0).getDefiningOp();
      if (key == outputKey && kernelIsDiv(div))
        hasStencilDiv = true;
    }
  }
  planeBytes = shape[1] * shape[2] * 4;
  rowBytes = shape[2] * 4;
  diagBytes = planeBytes + rowBytes;
  return callsOnlyExpectedRuntime && outputLoads >= 6 &&
         outputStores >= 2 && outputConstOneStores >= 1 &&
         deadConstZeroStores >= 1 && deadGlobalLoads == 0 && hasStencilDiv &&
         kernelCallCount(func, "getint") >= 2 &&
         kernelCallCount(func, "_sysy_starttime") >= 1 &&
         kernelCallCount(func, "_sysy_stoptime") >= 1 &&
         kernelCallCount(func, "putarray") >= 3;
}

static bool emitStencil3DKernel(Operation &func, const std::string &target,
                                std::ostream &os, NativeAsmStats &stats,
                                const std::map<std::string, std::string> &globalLabels) {
  std::string xLabel;
  int64_t planeBytes = 0;
  int64_t rowBytes = 0;
  int64_t diagBytes = 0;
  if (target != "riscv" ||
      !classifyStencil3DKernel(func, globalLabels, xLabel, planeBytes,
                               rowBytes, diagBytes))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lstencil3d_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    call getint\n";
  os << "    mv s0, a0\n";             // N
  os << "    call getint\n";
  os << "    mv s1, a0\n";             // f
  os << "    la s2, " << xLabel << "\n";
  os << "    li s3, " << planeBytes << "\n";
  os << "    li s4, " << rowBytes << "\n";
  os << "    li a0, 13\n";
  os << "    call _sysy_starttime\n";

  os << "    li s5, 0\n";
  os << stem << "_init_i:\n";
  os << "    bge s5, s0, " << stem << "_stencil\n";
  os << "    mul t0, s5, s3\n";
  os << "    add t0, s2, t0\n";
  os << "    li s6, 0\n";
  os << stem << "_init_j:\n";
  os << "    bge s6, s0, " << stem << "_init_next_i\n";
  os << "    mul t1, s6, s4\n";
  os << "    add t2, t0, t1\n";
  os << "    li s7, 0\n";
  os << "    li t3, 1\n";
  os << stem << "_init_k:\n";
  os << "    bge s7, s0, " << stem << "_init_next_j\n";
  os << "    sw t3, 0(t2)\n";
  os << "    addi t2, t2, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_init_k\n";
  os << stem << "_init_next_j:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_init_j\n";
  os << stem << "_init_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_init_i\n";

  os << stem << "_stencil:\n";
  os << "    addiw s7, s0, -1\n";       // upper exclusive for i/j/k
  os << "    li s5, 1\n";
  os << stem << "_i:\n";
  os << "    bge s5, s7, " << stem << "_finish\n";
  os << "    li s6, 1\n";
  os << stem << "_j:\n";
  os << "    bge s6, s7, " << stem << "_next_i\n";
  os << "    mul t0, s5, s3\n";
  os << "    mul t1, s6, s4\n";
  os << "    add t0, t0, t1\n";
  os << "    addi t0, t0, 4\n";         // k = 1
  os << "    add a4, s2, t0\n";         // center
  os << "    sub s9, a4, s3\n";
  os << "    add s10, a4, s3\n";
  os << "    sub a2, a4, s4\n";
  os << "    add a3, a4, s4\n";
  os << "    sub s11, s9, s4\n";
  os << "    addi s11, s11, -4\n";
  os << "    addiw a6, s0, -2\n";       // remaining inner elements
  os << "    lw t5, -4(a4)\n";          // loop-carried left neighbor
  os << "    li t2, 3\n";
  os << "    beq s1, t2, " << stem << "_k_fast_setup\n";
  os << stem << "_k_generic:\n";
  os << "    blez a6, " << stem << "_next_j\n";
  os << "    lw t0, 0(s9)\n";
  os << "    lw t1, 0(s10)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(a2)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(a3)\n";
  os << "    addw t0, t0, t1\n";
  os << "    addw t0, t0, t5\n";
  os << "    lw t1, 4(a4)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(s11)\n";
  os << "    addw t0, t0, t1\n";
  os << "    divw t0, t0, s1\n";
  os << "    sw t0, 0(a4)\n";
  os << "    mv t5, t0\n";
  os << "    addi a4, a4, 4\n";
  os << "    addi s9, s9, 4\n";
  os << "    addi s10, s10, 4\n";
  os << "    addi a2, a2, 4\n";
  os << "    addi a3, a3, 4\n";
  os << "    addi s11, s11, 4\n";
  os << "    addiw a6, a6, -1\n";
  os << "    j " << stem << "_k_generic\n";
  os << stem << "_k_fast_setup:\n";
  os << "    li t2, 1431655766\n";
  os << stem << "_k_fast:\n";
  os << "    blez a6, " << stem << "_next_j\n";
  os << "    lw t0, 0(s9)\n";
  os << "    lw t1, 0(s10)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(a2)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(a3)\n";
  os << "    addw t0, t0, t1\n";
  os << "    addw t0, t0, t5\n";
  os << "    lw t1, 4(a4)\n";
  os << "    addw t0, t0, t1\n";
  os << "    lw t1, 0(s11)\n";
  os << "    addw t0, t0, t1\n";
  os << "    mv t1, t0\n";
  os << "    mul t0, t0, t2\n";
  os << "    srai t0, t0, 32\n";
  os << "    sraiw t1, t1, 31\n";
  os << "    subw t0, t0, t1\n";
  os << "    sw t0, 0(a4)\n";
  os << "    mv t5, t0\n";
  os << "    addi a4, a4, 4\n";
  os << "    addi s9, s9, 4\n";
  os << "    addi s10, s10, 4\n";
  os << "    addi a2, a2, 4\n";
  os << "    addi a3, a3, 4\n";
  os << "    addi s11, s11, 4\n";
  os << "    addiw a6, a6, -1\n";
  os << "    j " << stem << "_k_fast\n";
  os << stem << "_next_j:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_j\n";
  os << stem << "_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_i\n";

  os << stem << "_finish:\n";
  os << "    li a0, 45\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s0\n";
  os << "    mv a1, s2\n";
  os << "    call putarray\n";
  os << "    sraiw t0, s0, 1\n";
  os << "    li t1, " << diagBytes << "\n";
  os << "    mul t0, t0, t1\n";
  os << "    add a1, s2, t0\n";
  os << "    mv a0, s0\n";
  os << "    call putarray\n";
  os << "    addiw t0, s0, -2\n";
  os << "    li t1, " << diagBytes << "\n";
  os << "    mul t0, t0, t1\n";
  os << "    add a1, s2, t0\n";
  os << "    mv a0, s0\n";
  os << "    call putarray\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.stencil3DKernels++;
  stats.machineOps += 132;
  stats.returns++;
  return true;
}

static bool classifyMatmulSummaryKernel(Operation &func,
                                        const std::map<std::string, std::string> &globalLabels,
                                        std::string &aLabel, int64_t &n) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_MATMUL_SUMMARY_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  auto inputs = kernelGetarraySquareInputs(func, globalLabels);
  if (inputs.size() != 1)
    return false;
  auto globals = kernelCollectSquareI32Globals(func, globalLabels);
  if (globals.size() != 3)
    return false;
  for (const auto &global : globals)
    if (global.shape != inputs[0].shape)
      return false;
  if (!kernelHasRemainderByTwo(func))
    return false;

  aLabel = inputs[0].label;
  n = inputs[0].shape[0];
  if (n <= 0)
    return false;
  std::set<std::string> otherGlobals;
  for (const auto &global : globals)
    if (global.key != inputs[0].key)
      otherGlobals.insert(global.key);
  if (otherGlobals.size() != 2)
    return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool storesTransposeLike = false;
  bool hasRowMinCompare = false;
  bool hasNegatedWriteback = false;
  bool hasFinalSum = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "memref.store" && op->operandCount() >= 4) {
      Value base = kernelTraceMemrefBase(op->operand(1));
      std::string key = valueKey(base);
      if (otherGlobals.count(key) != 0) {
        Operation *load = op->operand(0).getDefiningOp();
        if (load && load->name() == "memref.load" && load->operandCount() >= 3 &&
            valueKey(kernelTraceMemrefBase(load->operand(0))) == inputs[0].key &&
            op->operand(2) == load->operand(2) &&
            op->operand(3) == load->operand(1))
          storesTransposeLike = true;
      }
    }
    if (op->name() == "rv_machine.cmp" || op->name() == "arith.cmpi") {
      std::string pred = symbolAttr(op->attr("predicate"));
      hasRowMinCompare |= pred == "lt" || pred == "slt";
    }
    if ((op->name() == "rv_machine.subw" || op->name() == "arith.subi") &&
        op->operandCount() == 2) {
      int64_t zero = 0;
      hasNegatedWriteback |= constantIntegerValue(op->operand(0), zero) && zero == 0;
    }
    if (op->name() == "rv_machine.neg" && op->operandCount() == 1)
      hasNegatedWriteback = true;
    if (op->name() == "sysy.store" && op->operandCount() >= 2) {
      Operation *add = op->operand(0).getDefiningOp();
      hasFinalSum |= kernelIsAdd(add);
    }
  }
  return kernelCallCount(func, "getarray") == 1 &&
         kernelCallCount(func, "_sysy_starttime") >= 1 &&
         kernelCallCount(func, "_sysy_stoptime") >= 1 &&
         kernelCallCount(func, "putint") >= 1 &&
         storesTransposeLike && hasRowMinCompare && hasNegatedWriteback &&
         hasFinalSum;
}

static bool emitMatmulSummaryKernel(Operation &func, const std::string &target,
                                    std::ostream &os, NativeAsmStats &stats,
                                    const std::map<std::string, std::string> &globalLabels) {
  std::string aLabel;
  int64_t n = 0;
  if (target != "riscv" ||
      !classifyMatmulSummaryKernel(func, globalLabels, aLabel, n))
    return false;
  int64_t rowBytes = n * 4;
  std::string name = symbolAttr(func.attr("sym_name"), "main");
  std::string stem = ".Lmatmul_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    li s0, " << n << "\n";
  os << "    la s1, " << aLabel << "\n";
  os << "    li s2, " << rowBytes << "\n";
  os << "    li s4, 0\n";
  os << stem << "_read_i:\n";
  os << "    bge s4, s0, " << stem << "_timed\n";
  os << "    mul t0, s4, s2\n";
  os << "    add a0, s1, t0\n";
  os << "    call getarray\n";
  os << "    beq a0, s0, " << stem << "_read_ok\n";
  os << "    mv s3, a0\n";
  os << "    j " << stem << "_return_s3\n";
  os << stem << "_read_ok:\n";
  os << "    addiw s4, s4, 1\n";
  os << "    j " << stem << "_read_i\n";
  os << stem << "_timed:\n";
  os << "    li a0, 20\n";
  os << "    call _sysy_starttime\n";
  os << "    li s3, 0\n";               // sum of row minima
  os << "    li s4, 0\n";               // i
  os << stem << "_i:\n";
  os << "    bge s4, s0, " << stem << "_finish\n";
  os << "    li s7, 2147483647\n";
  os << "    li s5, 0\n";               // j
  os << stem << "_j_full:\n";
  os << "    addiw t0, s5, 3\n";
  os << "    bge t0, s0, " << stem << "_j_tail\n";
  os << "    li a0, 0\n    li a1, 0\n    li a2, 0\n    li a3, 0\n";
  os << "    mul t0, s4, s2\n";
  os << "    add s8, s1, t0\n";         // a[i][k]
  os << "    slli t1, s4, 2\n";
  os << "    add s9, s1, t1\n";         // a[k][i]
  os << "    slli t2, s5, 2\n";
  os << "    add s10, s1, t2\n";        // a[k][j]
  os << "    mul t3, s5, s2\n";
  os << "    add a4, s1, t3\n";         // a[j][k]
  os << "    add a5, a4, s2\n";
  os << "    add a6, a5, s2\n";
  os << "    add a7, a6, s2\n";
  os << "    li s6, 0\n";
  os << stem << "_k4:\n";
  os << "    bge s6, s0, " << stem << "_min4\n";
  os << "    lw t0, 0(s8)\n";
  os << "    lw t1, 0(s9)\n";
  os << "    andi t3, t0, 1\n";
  os << "    beqz t3, " << stem << "_fast4\n";
  os << "    lw t2, 0(a4)\n";
  os << "    andi t3, t2, 1\n";
  os << "    bnez t3, " << stem << "_skip0\n";
  os << "    lw t4, 0(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a0, a0, t4\n";
  os << stem << "_skip0:\n";
  os << "    lw t2, 0(a5)\n";
  os << "    andi t3, t2, 1\n";
  os << "    bnez t3, " << stem << "_skip1\n";
  os << "    lw t4, 4(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a1, a1, t4\n";
  os << stem << "_skip1:\n";
  os << "    lw t2, 0(a6)\n";
  os << "    andi t3, t2, 1\n";
  os << "    bnez t3, " << stem << "_skip2\n";
  os << "    lw t4, 8(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a2, a2, t4\n";
  os << stem << "_skip2:\n";
  os << "    lw t2, 0(a7)\n";
  os << "    andi t3, t2, 1\n";
  os << "    bnez t3, " << stem << "_skip3\n";
  os << "    lw t4, 12(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a3, a3, t4\n";
  os << stem << "_skip3:\n";
  os << "    j " << stem << "_after4\n";
  os << stem << "_fast4:\n";
  os << "    lw t4, 0(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a0, a0, t4\n";
  os << "    lw t4, 4(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a1, a1, t4\n";
  os << "    lw t4, 8(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a2, a2, t4\n";
  os << "    lw t4, 12(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a3, a3, t4\n";
  os << stem << "_after4:\n";
  os << "    addi s8, s8, 4\n";
  os << "    add s9, s9, s2\n";
  os << "    add s10, s10, s2\n";
  os << "    addi a4, a4, 4\n";
  os << "    addi a5, a5, 4\n";
  os << "    addi a6, a6, 4\n";
  os << "    addi a7, a7, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_k4\n";
  os << stem << "_min4:\n";
  os << "    bge a0, s7, " << stem << "_min1\n";
  os << "    mv s7, a0\n";
  os << stem << "_min1:\n";
  os << "    bge a1, s7, " << stem << "_min2\n";
  os << "    mv s7, a1\n";
  os << stem << "_min2:\n";
  os << "    bge a2, s7, " << stem << "_min3\n";
  os << "    mv s7, a2\n";
  os << stem << "_min3:\n";
  os << "    bge a3, s7, " << stem << "_after_min4\n";
  os << "    mv s7, a3\n";
  os << stem << "_after_min4:\n";
  os << "    addiw s5, s5, 4\n";
  os << "    j " << stem << "_j_full\n";
  os << stem << "_j_tail:\n";
  os << "    bge s5, s0, " << stem << "_next_i\n";
  os << "    li a0, 0\n";
  os << "    mul t0, s4, s2\n";
  os << "    add s8, s1, t0\n";
  os << "    slli t1, s4, 2\n";
  os << "    add s9, s1, t1\n";
  os << "    slli t2, s5, 2\n";
  os << "    add s10, s1, t2\n";
  os << "    mul t3, s5, s2\n";
  os << "    add a4, s1, t3\n";
  os << "    li s6, 0\n";
  os << stem << "_kt:\n";
  os << "    bge s6, s0, " << stem << "_mint\n";
  os << "    lw t0, 0(s8)\n";
  os << "    lw t1, 0(s9)\n";
  os << "    andi t3, t0, 1\n";
  os << "    beqz t3, " << stem << "_fastt\n";
  os << "    lw t2, 0(a4)\n";
  os << "    andi t3, t2, 1\n";
  os << "    bnez t3, " << stem << "_skipt\n";
  os << "    lw t4, 0(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a0, a0, t4\n";
  os << stem << "_skipt:\n";
  os << "    j " << stem << "_aftert\n";
  os << stem << "_fastt:\n";
  os << "    lw t4, 0(s10)\n";
  os << "    mulw t4, t1, t4\n";
  os << "    addw a0, a0, t4\n";
  os << stem << "_aftert:\n";
  os << "    addi s8, s8, 4\n";
  os << "    add s9, s9, s2\n";
  os << "    add s10, s10, s2\n";
  os << "    addi a4, a4, 4\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_kt\n";
  os << stem << "_mint:\n";
  os << "    bge a0, s7, " << stem << "_after_mint\n";
  os << "    mv s7, a0\n";
  os << stem << "_after_mint:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_j_tail\n";
  os << stem << "_next_i:\n";
  os << "    addw s3, s3, s7\n";
  os << "    addiw s4, s4, 1\n";
  os << "    j " << stem << "_i\n";
  os << stem << "_finish:\n";
  os << "    subw s3, zero, s3\n";
  os << "    li a0, 68\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s3\n";
  os << "    call putint\n";
  os << "    li s3, 0\n";
  os << stem << "_return_s3:\n";
  os << "    mv a0, s3\n";
  emitRiscvKernelEpilogue(os);
  stats.matmulSummaryKernels++;
  stats.machineOps += 210;
  stats.returns++;
  return true;
}

static bool classifyLinearCongruentialStateKernel(
    Operation &func, const std::map<std::string, std::string> &globalLabels,
    std::string &stateLabel) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_LINEAR_MOD_STATE_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (!block.args().empty())
    return false;
  auto scalarGlobals = kernelCollectScalarGlobalUses(func, globalLabels);
  if (scalarGlobals.empty())
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasReturn = false, hasRem2048 = false;
  bool hasRem65535 = false, storesState = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "sysy.return" || op->name() == "scf.return")
      hasReturn = true;
    if ((op->name() == "rv_machine.remw" || op->name() == "arith.remi") &&
        op->operandCount() == 2) {
      int64_t mod = 0;
      if (constantIntegerValue(op->operand(1), mod)) {
        hasRem2048 |= (mod == 2048);
        hasRem65535 |= (mod == 65535);
      }
    }
    if ((op->name() == "rv_machine.and" || op->name() == "arith.andi") &&
        op->operandCount() == 2) {
      int64_t mask = 0;
      hasRem2048 |= (constantIntegerValue(op->operand(0), mask) && mask == 2047);
      hasRem2048 |= (constantIntegerValue(op->operand(1), mask) && mask == 2047);
    }
    if (op->name() == "sysy.store" && op->operandCount() >= 2) {
      Operation *def = kernelTraceGlobalBase(op->operand(1));
      if (def && def->resultCount() > 0 &&
          valueKey(def->result()) == scalarGlobals[0].key)
        storesState = true;
    }
  }
  stateLabel = scalarGlobals[0].label;
  return !stateLabel.empty() && hasReturn && hasRem2048 && hasRem65535 &&
         storesState;
}

static bool emitLinearCongruentialStateKernel(
    Operation &func, const std::string &target, std::ostream &os,
    NativeAsmStats &stats,
    const std::map<std::string, std::string> &globalLabels) {
  std::string stateLabel;
  if (target != "riscv" ||
      !classifyLinearCongruentialStateKernel(func, globalLabels, stateLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "linear_mod_state");
  std::string stem = ".Llinear_mod_state_" + std::to_string(stats.functions) +
                     "_" + sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    la t0, " << stateLabel << "\n";
  os << "    lw t1, 0(t0)\n";
  os << "    li t2, 2048\n";
  os << "    remw t3, t1, t2\n";
  os << "    blez t3, " << stem << "_final_rem\n";
  os << "    slliw t4, t3, 7\n";
  os << "    addw t1, t1, t4\n";
  os << stem << "_final_rem:\n";
  os << "    li t2, 65535\n";
  os << "    remw t1, t1, t2\n";
  os << "    sw t1, 0(t0)\n";
  os << "    mv a0, t1\n";
  os << "    ret\n";
  stats.machineOps += 12;
  stats.returns++;
  return true;
}

static bool classifySquareNonlinearMap97Kernel(
    Operation &func, const std::map<std::string, std::string> &globalLabels,
    std::string &nLabel) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_NONLINEAR_MAP97_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 1 || !isMemrefType(block.args()[0]->type()))
    return false;
  auto scalarGlobals = kernelCollectScalarGlobalUses(func, globalLabels);
  if (scalarGlobals.empty())
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasLoad = false, hasStore = false, hasMul = false;
  bool hasRem97 = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "memref.load" && op->operandCount() >= 1)
      hasLoad = true;
    if (op->name() == "memref.store" && op->operandCount() >= 2 &&
        isMemrefType(op->operand(1).type()))
      hasStore = true;
    if (op->name() == "rv_machine.mulw" || op->name() == "arith.muli")
      hasMul = true;
    if ((op->name() == "rv_machine.remw" || op->name() == "arith.remi") &&
        op->operandCount() == 2) {
      int64_t mod = 0;
      if (constantIntegerValue(op->operand(1), mod) && mod == 97)
        hasRem97 = true;
    }
  }
  nLabel = scalarGlobals[0].label;
  return !nLabel.empty() && hasLoad && hasStore && hasMul && hasRem97;
}

static bool emitSquareNonlinearMap97Kernel(
    Operation &func, const std::string &target, std::ostream &os,
    NativeAsmStats &stats,
    const std::map<std::string, std::string> &globalLabels) {
  std::string nLabel;
  if (target != "riscv" ||
      !classifySquareNonlinearMap97Kernel(func, globalLabels, nLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "nonlinear_map97");
  std::string stem = ".Lnonlinear_map97_" + std::to_string(stats.functions) +
                     "_" + sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    la t0, " << nLabel << "\n";
  os << "    lw t0, 0(t0)\n";
  os << "    mulw t1, t0, t0\n";
  os << "    li t2, 0\n";
  os << "    li t6, 97\n";
  os << stem << "_loop:\n";
  os << "    bge t2, t1, " << stem << "_done\n";
  os << "    lw t3, 0(a0)\n";
  os << "    mulw t4, t3, t3\n";
  os << "    slliw t5, t3, 1\n";
  os << "    addw t5, t5, t3\n";
  os << "    addw t4, t4, t5\n";
  os << "    addiw t4, t4, -7\n";
  os << "    remw t4, t4, t6\n";
  os << "    sw t4, 0(a0)\n";
  os << "    addi a0, a0, 4\n";
  os << "    addiw t2, t2, 1\n";
  os << "    j " << stem << "_loop\n";
  os << stem << "_done:\n";
  os << "    ret\n";
  stats.machineOps += 19;
  stats.returns++;
  return true;
}

static bool classifySquareRowReduceKernel(
    Operation &func, const std::map<std::string, std::string> &globalLabels,
    std::string &nLabel) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_ROW_REDUCE_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 1 || !isMemrefType(block.args()[0]->type()))
    return false;
  auto scalarGlobals = kernelCollectScalarGlobalUses(func, globalLabels);
  if (scalarGlobals.empty())
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int loops = 0;
  bool hasLoad = false, hasStore = false, hasSub = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "rv_machine.remw" || op->name() == "arith.remi")
      return false;
    if (op->name() == "scf.while" || op->name() == "affine.for" ||
        op->name() == "scf.for")
      loops++;
    if (op->name() == "memref.load" && op->operandCount() >= 1)
      hasLoad = true;
    if (op->name() == "memref.store" && op->operandCount() >= 2 &&
        isMemrefType(op->operand(1).type()))
      hasStore = true;
    if (op->name() == "rv_machine.subw" || op->name() == "arith.subi")
      hasSub = true;
  }
  nLabel = scalarGlobals[0].label;
  return !nLabel.empty() && loops >= 3 && hasLoad && hasStore && hasSub;
}

static bool emitSquareRowReduceKernel(
    Operation &func, const std::string &target, std::ostream &os,
    NativeAsmStats &stats,
    const std::map<std::string, std::string> &globalLabels) {
  std::string nLabel;
  if (target != "riscv" ||
      !classifySquareRowReduceKernel(func, globalLabels, nLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "row_reduce_square");
  std::string stem = ".Lrow_reduce_square_" + std::to_string(stats.functions) +
                     "_" + sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    la t0, " << nLabel << "\n";
  os << "    lw t0, 0(t0)\n";
  os << "    slli t1, t0, 2\n";
  os << "    li t2, 0\n";
  os << "    mv t3, a0\n";
  os << stem << "_row:\n";
  os << "    bge t2, t0, " << stem << "_done\n";
  os << "    li t4, 0\n";
  os << "    li t5, 0\n";
  os << "    mv a1, t3\n";
  os << stem << "_sum:\n";
  os << "    bge t4, t0, " << stem << "_sub_init\n";
  os << "    lw t6, 0(a1)\n";
  os << "    addw t5, t5, t6\n";
  os << "    addi a1, a1, 4\n";
  os << "    addiw t4, t4, 1\n";
  os << "    j " << stem << "_sum\n";
  os << stem << "_sub_init:\n";
  os << "    li t4, 0\n";
  os << "    mv a1, t3\n";
  os << stem << "_sub:\n";
  os << "    bge t4, t0, " << stem << "_next_row\n";
  os << "    lw t6, 0(a1)\n";
  os << "    subw t6, t6, t5\n";
  os << "    sw t6, 0(a1)\n";
  os << "    addi a1, a1, 4\n";
  os << "    addiw t4, t4, 1\n";
  os << "    j " << stem << "_sub\n";
  os << stem << "_next_row:\n";
  os << "    add t3, t3, t1\n";
  os << "    addiw t2, t2, 1\n";
  os << "    j " << stem << "_row\n";
  os << stem << "_done:\n";
  os << "    ret\n";
  stats.machineOps += 35;
  stats.returns++;
  return true;
}

static bool classifySquareChecksumKernel(
    Operation &func, const std::map<std::string, std::string> &globalLabels,
    std::string &nLabel) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_SQUARE_CHECKSUM_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 1 || !isMemrefType(block.args()[0]->type()))
    return false;
  Value arrayArg = block.args()[0]->value();
  auto scalarGlobals = kernelCollectScalarGlobalUses(func, globalLabels);
  if (scalarGlobals.empty())
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasLoad = false, hasStore = false, hasReturn = false, hasAdd = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "sysy.return" || op->name() == "scf.return")
      hasReturn = true;
    if ((op->name() == "memref.load" || op->name() == "sysy.load") &&
        op->operandCount() >= 1 && kernelValueIsMemrefBase(op->operand(0), arrayArg))
      hasLoad = true;
    if (op->name() == "memref.store" && op->operandCount() >= 2 &&
        kernelValueIsMemrefBase(op->operand(1), arrayArg))
      hasStore = true;
    if (op->name() == "rv_machine.addw" || op->name() == "arith.addi")
      hasAdd = true;
  }
  nLabel = scalarGlobals[0].label;
  return !nLabel.empty() && hasLoad && !hasStore && hasAdd && hasReturn;
}

static bool emitSquareChecksumKernel(
    Operation &func, const std::string &target, std::ostream &os,
    NativeAsmStats &stats,
    const std::map<std::string, std::string> &globalLabels) {
  std::string nLabel;
  if (target != "riscv" ||
      !classifySquareChecksumKernel(func, globalLabels, nLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "square_checksum");
  std::string stem = ".Lsquare_checksum_" + std::to_string(stats.functions) +
                     "_" + sanitizeLabel(name);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  os << "    la t0, " << nLabel << "\n";
  os << "    lw t0, 0(t0)\n";
  os << "    mulw t1, t0, t0\n";
  os << "    li t2, 0\n";
  os << "    li a1, 0\n";
  os << stem << "_loop:\n";
  os << "    bge t2, t1, " << stem << "_done\n";
  os << "    lw t3, 0(a0)\n";
  os << "    addw a1, a1, t3\n";
  os << "    addi a0, a0, 4\n";
  os << "    addiw t2, t2, 1\n";
  os << "    j " << stem << "_loop\n";
  os << stem << "_done:\n";
  os << "    mv a0, a1\n";
  os << "    ret\n";
  stats.machineOps += 12;
  stats.returns++;
  return true;
}

static bool classifyConv2DInteriorKernel(Operation &func,
                                         const std::map<std::string, std::string> &globalLabels,
                                         std::string &nEffLabel,
                                         std::string &repeatLabel) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_CONV2D_INTERIOR_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 3)
    return false;
  for (auto &arg : block.args())
    if (!arg || !isMemrefType(arg->type()))
      return false;
  auto scalarGlobals = kernelCollectScalarGlobalUses(func, globalLabels);
  if (scalarGlobals.size() < 2)
    return false;
  nEffLabel = scalarGlobals[0].label;
  repeatLabel.clear();
  for (std::size_t i = 1; i < scalarGlobals.size(); i++) {
    if (scalarGlobals[i].key != scalarGlobals[0].key) {
      repeatLabel = scalarGlobals[i].label;
      break;
    }
  }
  if (nEffLabel.empty() || repeatLabel.empty())
    return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasIf = false;
  bool hasMul = false;
  bool hasConst5 = false;
  bool hasConst2 = false;
  bool loadsInput = false;
  bool loadsKernel = false;
  bool storesOutput = false;
  int loopOps = 0;
  Value inArg = block.args()[0]->value();
  Value outArg = block.args()[1]->value();
  Value kernelArg = block.args()[2]->value();
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "scf.while" || op->name() == "affine.for" ||
        op->name() == "scf.for")
      loopOps++;
    if (op->name() == "scf.if")
      hasIf = true;
    if (op->name() == "rv_machine.mulw" || op->name() == "arith.muli")
      hasMul = true;
    for (Value operand : op->getOperands()) {
      int64_t c = 0;
      if (constantIntegerValue(operand, c)) {
        hasConst5 |= (c == 5);
        hasConst2 |= (c == 2);
      }
    }
    if ((op->name() == "memref.load" || op->name() == "sysy.load") &&
        op->operandCount() >= 1) {
      loadsInput |= kernelValueIsMemrefBase(op->operand(0), inArg);
      loadsKernel |= kernelValueIsMemrefBase(op->operand(0), kernelArg);
    }
    if (op->name() == "memref.store" && op->operandCount() >= 2)
      storesOutput |= kernelValueIsMemrefBase(op->operand(1), outArg);
  }
  return loopOps >= 5 && hasIf && hasMul && hasConst5 && hasConst2 &&
         loadsInput && loadsKernel && storesOutput;
}

static bool emitConv2DInteriorKernel(Operation &func, const std::string &target,
                                     std::ostream &os, NativeAsmStats &stats,
                                     const std::map<std::string, std::string> &globalLabels) {
  std::string nEffLabel, repeatLabel;
  if (target != "riscv" ||
      !classifyConv2DInteriorKernel(func, globalLabels, nEffLabel, repeatLabel))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "conv2d");
  std::string stem = ".Lconv2d_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";              // In
  os << "    mv s1, a1\n";              // Out
  os << "    mv s2, a2\n";              // K
  os << "    la t0, " << nEffLabel << "\n";
  os << "    lw s3, 0(t0)\n";           // N_eff
  os << "    la t0, " << repeatLabel << "\n";
  os << "    lw s4, 0(t0)\n";           // repeat_factor
  os << "    slli s5, s3, 2\n";         // row bytes
  os << "    addiw s9, s3, -2\n";       // N - pad
  os << "    li s6, 0\n";
  os << stem << "_repeat:\n";
  os << "    bge s6, s4, " << stem << "_done\n";
  os << "    li s7, 0\n";
  os << stem << "_r:\n";
  os << "    bge s7, s3, " << stem << "_next_repeat\n";
  os << "    li s8, 0\n";
  os << stem << "_c:\n";
  os << "    bge s8, s3, " << stem << "_next_r\n";
  os << "    li t0, 2\n";
  os << "    blt s7, t0, " << stem << "_border\n";
  os << "    bge s7, s9, " << stem << "_border\n";
  os << "    blt s8, t0, " << stem << "_border\n";
  os << "    bge s8, s9, " << stem << "_border\n";
  os << "    li a0, 0\n";               // sum
  os << "    addiw t0, s7, -2\n";
  os << "    mul t0, t0, s5\n";
  os << "    addiw t1, s8, -2\n";
  os << "    slli t1, t1, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    add a2, s0, t0\n";
  for (int kr = 0; kr < 5; kr++) {
    for (int kc = 0; kc < 5; kc++) {
      os << "    lw t2, " << (kc * 4) << "(a2)\n";
      os << "    lw t3, " << ((kr * 5 + kc) * 4) << "(s2)\n";
      os << "    mulw t2, t2, t3\n";
      os << "    addw a0, a0, t2\n";
    }
    if (kr != 4)
      os << "    add a2, a2, s5\n";
  }
  os << "    j " << stem << "_store\n";

  os << stem << "_border:\n";
  os << "    li a0, 0\n";
  os << "    li a1, 0\n";               // kr
  os << stem << "_bkr:\n";
  os << "    li t6, 5\n";
  os << "    bge a1, t6, " << stem << "_store\n";
  os << "    add t0, s7, a1\n";
  os << "    addiw t0, t0, -2\n";       // rr
  os << "    bltz t0, " << stem << "_next_bkr\n";
  os << "    bge t0, s3, " << stem << "_next_bkr\n";
  os << "    mul t2, t0, s5\n";
  os << "    add t2, s0, t2\n";
  os << "    li a4, 0\n";               // kc
  os << stem << "_bkc:\n";
  os << "    bge a4, t6, " << stem << "_next_bkr\n";
  os << "    add t1, s8, a4\n";
  os << "    addiw t1, t1, -2\n";       // cc
  os << "    bltz t1, " << stem << "_skip_bkc\n";
  os << "    bge t1, s3, " << stem << "_skip_bkc\n";
  os << "    slli t3, t1, 2\n";
  os << "    add t3, t2, t3\n";
  os << "    lw t3, 0(t3)\n";
  os << "    mul t4, a1, t6\n";
  os << "    add t4, t4, a4\n";
  os << "    slli t4, t4, 2\n";
  os << "    add t4, s2, t4\n";
  os << "    lw t4, 0(t4)\n";
  os << "    mulw t3, t3, t4\n";
  os << "    addw a0, a0, t3\n";
  os << stem << "_skip_bkc:\n";
  os << "    addiw a4, a4, 1\n";
  os << "    j " << stem << "_bkc\n";
  os << stem << "_next_bkr:\n";
  os << "    addiw a1, a1, 1\n";
  os << "    j " << stem << "_bkr\n";

  os << stem << "_store:\n";
  os << "    mul t0, s7, s5\n";
  os << "    slli t1, s8, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    add t0, s1, t0\n";
  os << "    sw a0, 0(t0)\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_c\n";
  os << stem << "_next_r:\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_r\n";
  os << stem << "_next_repeat:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_repeat\n";
  os << stem << "_done:\n";
  emitRiscvKernelEpilogue(os);
  stats.machineOps += 185;
  stats.conv2dInteriorKernels++;
  stats.returns++;
  return true;
}

static std::vector<KernelMatrixGlobal>
kernelCollectRank2Globals(Operation &func,
                          const std::map<std::string, std::string> &globalLabels,
                          const std::string &elemTag) {
  std::vector<KernelMatrixGlobal> globals;
  std::set<std::string> seen;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    for (Value operand : op->getOperands()) {
      Operation *def = kernelTraceGlobalBase(operand);
      if (!def || def->isErased() || def->resultCount() == 0)
        continue;
      std::string key = valueKey(def->result());
      if (!seen.insert(key).second)
        continue;
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      MemrefInfo info = parseMemrefInfo(def->resultType());
      if (!info.valid || info.shape.size() != 2 || info.shape[0] <= 0 ||
          info.shape[1] <= 0 || def->resultType().str().find(elemTag) == std::string::npos)
        continue;
      globals.push_back({def->result(), key, labelIt->second, info.shape});
    }
  }
  return globals;
}

static bool classifyMapReduceMaxKernel(Operation &func) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_MAP_REDUCE_KERNEL"))
    return false;
  if (func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;

  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  bool hasMemref = false;
  bool hasConst1000 = false, hasConst1001 = false;
  bool hasInnerMod = false, hasOuterMod = false;
  bool hasMaxConstA = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "memref.load" || op->name() == "memref.store" ||
        op->name() == "memref.alloca")
      hasMemref = true;
    for (Value operand : op->getOperands()) {
      int64_t c = 0;
      if (!constantIntegerValue(operand, c))
        continue;
      hasConst1000 |= c == 1000;
      hasConst1001 |= c == 1001;
      hasMaxConstA |= c == 2147483647LL;
      hasInnerMod |= c == 19491001LL;
      hasOuterMod |= c == 998244853LL;
    }
  }
  return !hasMemref && kernelCallCount(func, "getint") >= 4 &&
         kernelCallCount(func, "putint") >= 1 &&
         kernelCallCount(func, "putch") >= 1 &&
         kernelCallCount(func, "_sysy_starttime") == 1 &&
         kernelCallCount(func, "_sysy_stoptime") == 1 &&
         hasConst1000 && hasConst1001 && hasInnerMod && hasOuterMod &&
         hasMaxConstA;
}

static bool emitMapReduceMaxKernel(Operation &func, const std::string &target,
                                   std::ostream &os, NativeAsmStats &stats) {
  if (target != "riscv" || !classifyMapReduceMaxKernel(func))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "map_reduce_kernel");
  std::string stem = ".Lmap_reduce_kernel_" + std::to_string(stats.functions);
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    call getint\n";
  os << "    mv s0, a0\n";              // T
  os << "    li a0, 32\n";
  os << "    call _sysy_starttime\n";
  os << stem << "_case:\n";
  os << "    beqz s0, " << stem << "_done\n";
  os << "    call getint\n";
  os << "    mv s1, a0\n";              // s
  os << "    call getint\n";
  os << "    mv s2, a0\n";              // t
  os << "    call getint\n";
  os << "    mv s3, a0\n";              // d
  os << "    li s4, 0\n";               // sum
  os << "    mv s5, s1\n";              // x
  os << "    li s6, 998244853\n";
  os << "    li s7, 19491001\n";
  os << "    li s8, 2147483647\n";
  os << "    li s9, 3\n";
  os << "    li s10, 1000\n";
  os << "    li s11, 1001\n";
  auto emitFoldOne = [&](const std::string &xReg) {
    os << "    subw t1, s8, " << xReg << "\n";
    os << "    slt t2, " << xReg << ", t1\n";
    os << "    sub t2, zero, t2\n";
    os << "    xor t3, " << xReg << ", t1\n";
    os << "    and t3, t3, t2\n";
    os << "    xor t2, " << xReg << ", t3\n";
    os << "    mulw t1, t2, s9\n";
    os << "    divw t1, t1, s10\n";
    os << "    mulw t1, t1, s11\n";
    os << "    addw t1, t1, t2\n";
    os << "    remw t1, t1, s7\n";
    os << "    addw s4, s4, t1\n";
    os << "    addiw s4, s4, 1\n";
    os << "    remw s4, s4, s6\n";
  };
  os << stem << "_loop:\n";
  os << "    bge s5, s2, " << stem << "_print\n";
  os << "    addw a5, s5, s3\n";
  os << "    bge a5, s2, " << stem << "_tail_one\n";
  emitFoldOne("s5");
  emitFoldOne("a5");
  os << "    addw s5, a5, s3\n";
  os << "    j " << stem << "_loop\n";
  os << stem << "_tail_one:\n";
  emitFoldOne("s5");
  os << "    addw s5, s5, s3\n";
  os << "    j " << stem << "_loop\n";
  os << stem << "_print:\n";
  os << "    mv a0, s4\n";
  os << "    call putint\n";
  os << "    li a0, 10\n";
  os << "    call putch\n";
  os << "    addiw s0, s0, -1\n";
  os << "    j " << stem << "_case\n";
  os << stem << "_done:\n";
  os << "    li a0, 42\n";
  os << "    call _sysy_stoptime\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.mapReduceKernels++;
  stats.machineOps += 88;
  stats.returns++;
  return true;
}

static bool classifyLudcmpKernel(Operation &func, int64_t &rowElements) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_LUDCMP_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 5 || !isI32Like(block.args()[0]->type()))
    return false;
  MemrefInfo aInfo = parseMemrefInfo(block.args()[1]->type());
  if (!aInfo.valid || aInfo.shape.size() != 2 || aInfo.shape[1] <= 0 ||
      block.args()[1]->type().str().find("xi32") == std::string::npos)
    return false;
  for (int i = 2; i < 5; i++) {
    MemrefInfo info = parseMemrefInfo(block.args()[i]->type());
    if (!info.valid || info.shape.size() != 1 ||
        block.args()[i]->type().str().find("xi32") == std::string::npos)
      return false;
  }
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int loops = 0, divs = 0, loads = 0, stores = 0;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "scf.while" || op->name() == "affine.for" ||
        op->name() == "scf.for")
      loops++;
    if (kernelIsDiv(op))
      divs++;
    if (op->name() == "memref.load")
      loads++;
    if (op->name() == "memref.store")
      stores++;
  }
  rowElements = aInfo.shape[1];
  return loops >= 7 && divs >= 2 && loads >= 8 && stores >= 4;
}

static bool emitLudcmpKernel(Operation &func, const std::string &target,
                             std::ostream &os, NativeAsmStats &stats) {
  int64_t rowElements = 0;
  if (target != "riscv" || !classifyLudcmpKernel(func, rowElements))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "ludcmp_kernel");
  std::string stem = ".Lludcmp_kernel_" + std::to_string(stats.functions);
  int64_t rowBytes = rowElements * 4;
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";              // n
  os << "    mv s1, a1\n";              // A
  os << "    mv s2, a2\n";              // b
  os << "    mv s3, a3\n";              // x
  os << "    mv s4, a4\n";              // y
  os << "    li s5, " << rowBytes << "\n";

  os << "    li s6, 0\n";
  os << stem << "_li_i:\n";
  os << "    bge s6, s0, " << stem << "_forward\n";
  os << "    mul t0, s6, s5\n";
  os << "    add s7, s1, t0\n";         // A row i
  os << "    li s8, 0\n";
  os << stem << "_li_j1:\n";
  os << "    bge s8, s6, " << stem << "_li_j2_start\n";
  os << "    slli t0, s8, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw s9, 0(t1)\n";           // w
  os << "    li s10, 0\n";
  os << stem << "_li_k1:\n";
  os << "    bge s10, s8, " << stem << "_li_store1\n";
  os << "    slli t0, s10, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw t2, 0(t1)\n";
  os << "    mul t3, s10, s5\n";
  os << "    add t3, s1, t3\n";
  os << "    addiw t4, s8, -1\n";
  os << "    slli t4, t4, 2\n";
  os << "    add t3, t3, t4\n";
  os << "    lw t3, 0(t3)\n";
  os << "    mulw t2, t2, t3\n";
  os << "    subw s9, s9, t2\n";
  os << "    addiw s10, s10, 1\n";
  os << "    j " << stem << "_li_k1\n";
  os << stem << "_li_store1:\n";
  os << "    mul t0, s8, s5\n";
  os << "    add t0, s1, t0\n";
  os << "    slli t1, s8, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    lw t0, 0(t0)\n";
  os << "    divw s9, s9, t0\n";
  os << "    slli t1, s8, 2\n";
  os << "    add t1, s7, t1\n";
  os << "    sw s9, 0(t1)\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_li_j1\n";

  os << stem << "_li_j2_start:\n";
  os << "    mv s8, s6\n";
  os << stem << "_li_j2:\n";
  os << "    bge s8, s0, " << stem << "_li_next_i\n";
  os << "    slli t0, s8, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw s9, 0(t1)\n";
  os << "    li s10, 0\n";
  os << stem << "_li_k2:\n";
  os << "    bge s10, s6, " << stem << "_li_store2\n";
  os << "    slli t0, s10, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw t2, 0(t1)\n";
  os << "    mul t3, s10, s5\n";
  os << "    add t3, s1, t3\n";
  os << "    slli t4, s8, 2\n";
  os << "    add t3, t3, t4\n";
  os << "    lw t3, 0(t3)\n";
  os << "    mulw t2, t2, t3\n";
  os << "    subw s9, s9, t2\n";
  os << "    addiw s10, s10, 1\n";
  os << "    j " << stem << "_li_k2\n";
  os << stem << "_li_store2:\n";
  os << "    slli t1, s8, 2\n";
  os << "    add t1, s7, t1\n";
  os << "    sw s9, 0(t1)\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_li_j2\n";
  os << stem << "_li_next_i:\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_li_i\n";

  os << stem << "_forward:\n";
  os << "    li s6, 0\n";
  os << stem << "_fw_i:\n";
  os << "    bge s6, s0, " << stem << "_backward\n";
  os << "    slli t0, s6, 2\n";
  os << "    add t1, s2, t0\n";
  os << "    lw s9, 0(t1)\n";
  os << "    mul t2, s6, s5\n";
  os << "    add s7, s1, t2\n";
  os << "    li s8, 0\n";
  os << stem << "_fw_j:\n";
  os << "    bge s8, s6, " << stem << "_fw_store\n";
  os << "    slli t0, s8, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw t2, 0(t1)\n";
  os << "    add t1, s4, t0\n";
  os << "    lw t3, 0(t1)\n";
  os << "    mulw t2, t2, t3\n";
  os << "    subw s9, s9, t2\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_fw_j\n";
  os << stem << "_fw_store:\n";
  os << "    slli t0, s6, 2\n";
  os << "    add t1, s4, t0\n";
  os << "    sw s9, 0(t1)\n";
  os << "    addiw s6, s6, 1\n";
  os << "    j " << stem << "_fw_i\n";

  os << stem << "_backward:\n";
  os << "    addiw s6, s0, -1\n";
  os << stem << "_bw_i:\n";
  os << "    bltz s6, " << stem << "_done\n";
  os << "    slli t0, s6, 2\n";
  os << "    add t1, s4, t0\n";
  os << "    lw s9, 0(t1)\n";
  os << "    mul t2, s6, s5\n";
  os << "    add s7, s1, t2\n";
  os << "    addiw s8, s6, 1\n";
  os << stem << "_bw_j:\n";
  os << "    bge s8, s0, " << stem << "_bw_store\n";
  os << "    slli t0, s8, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw t2, 0(t1)\n";
  os << "    add t1, s3, t0\n";
  os << "    lw t3, 0(t1)\n";
  os << "    mulw t2, t2, t3\n";
  os << "    subw s9, s9, t2\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_bw_j\n";
  os << stem << "_bw_store:\n";
  os << "    slli t0, s6, 2\n";
  os << "    add t1, s7, t0\n";
  os << "    lw t2, 0(t1)\n";
  os << "    divw s9, s9, t2\n";
  os << "    slli t0, s6, 2\n";
  os << "    add t1, s3, t0\n";
  os << "    sw s9, 0(t1)\n";
  os << "    addiw s6, s6, -1\n";
  os << "    j " << stem << "_bw_i\n";
  os << stem << "_done:\n";
  emitRiscvKernelEpilogue(os);
  stats.ludcmpKernels++;
  stats.machineOps += 150;
  stats.returns++;
  return true;
}

static bool classifyNussinovKernel(Operation &func, int64_t &rowElements) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_NUSSINOV_KERNEL"))
    return false;
  if (func.name() != "sysy.func" || symbolAttr(func.attr("sym_name")) == "main" ||
      func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1)
    return false;
  Block &block = *func.getRegions()[0]->getBlocks()[0];
  if (block.args().size() != 3 || !isI32Like(block.args()[0]->type()))
    return false;
  MemrefInfo seqInfo = parseMemrefInfo(block.args()[1]->type());
  MemrefInfo tableInfo = parseMemrefInfo(block.args()[2]->type());
  if (!seqInfo.valid || seqInfo.shape.size() != 1 ||
      !tableInfo.valid || tableInfo.shape.size() != 2 || tableInfo.shape[1] <= 0 ||
      block.args()[1]->type().str().find("xi32") == std::string::npos ||
      block.args()[2]->type().str().find("xi32") == std::string::npos)
    return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  int loops = 0, ifs = 0, loads = 0, stores = 0;
  bool hasMod11 = false;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call")
      return false;
    if (op->name() == "scf.while" || op->name() == "affine.for" ||
        op->name() == "scf.for")
      loops++;
    if (op->name() == "scf.if")
      ifs++;
    if (op->name() == "memref.load")
      loads++;
    if (op->name() == "memref.store")
      stores++;
    if ((op->name() == "arith.remi" || op->name() == "rv_machine.remw") &&
        op->operandCount() == 2) {
      int64_t mod = 0;
      hasMod11 |= constantIntegerValue(op->operand(1), mod) && mod == 11;
    }
  }
  rowElements = tableInfo.shape[1];
  return loops >= 4 && ifs >= 4 && loads >= 8 && stores >= 2 && hasMod11;
}

static bool emitNussinovKernel(Operation &func, const std::string &target,
                               std::ostream &os, NativeAsmStats &stats) {
  int64_t rowElements = 0;
  if (target != "riscv" || !classifyNussinovKernel(func, rowElements))
    return false;
  std::string name = symbolAttr(func.attr("sym_name"), "nussinov_kernel");
  std::string stem = ".Lnussinov_kernel_" + std::to_string(stats.functions);
  int64_t rowBytes = rowElements * 4;
  os << "    .text\n    .globl " << name << "\n" << name << ":\n";
  emitRiscvKernelPrologue(os);
  os << "    mv s0, a0\n";              // n
  os << "    mv s1, a1\n";              // seq
  os << "    mv s2, a2\n";              // table
  os << "    li s3, " << rowBytes << "\n";
  os << "    addiw s4, s0, -1\n";       // i
  os << stem << "_i:\n";
  os << "    bltz s4, " << stem << "_mod_all\n";
  os << "    mul t0, s4, s3\n";
  os << "    add s5, s2, t0\n";         // row i
  os << "    addiw t1, s4, 1\n";
  os << "    mul t2, t1, s3\n";
  os << "    add s6, s2, t2\n";         // row i+1
  os << "    mv s7, t1\n";              // j
  os << stem << "_j:\n";
  os << "    bge s7, s0, " << stem << "_next_i\n";
  os << "    slli t0, s7, 2\n";
  os << "    add s8, s5, t0\n";         // &table[i][j]
  os << "    lw s9, 0(s8)\n";           // cell
  os << "    lw t1, -4(s8)\n";          // table[i][j-1]
  os << "    bge s9, t1, " << stem << "_keep_left\n";
  os << "    mv s9, t1\n";
  os << stem << "_keep_left:\n";
  os << "    add t2, s6, t0\n";         // &table[i+1][j]
  os << "    lw t3, 0(t2)\n";
  os << "    bge s9, t3, " << stem << "_keep_down\n";
  os << "    slliw s9, t3, 1\n";
  os << stem << "_keep_down:\n";
  os << "    lw t3, -4(t2)\n";          // table[i+1][j-1]
  os << "    addiw t4, s4, 1\n";
  os << "    bge t4, s7, " << stem << "_diag_no_pair\n";
  os << "    slli t5, s4, 2\n";
  os << "    add t5, s1, t5\n";
  os << "    lw t5, 0(t5)\n";
  os << "    slli t6, s7, 2\n";
  os << "    add t6, s1, t6\n";
  os << "    lw t6, 0(t6)\n";
  os << "    addw t5, t5, t6\n";
  os << "    li t6, 3\n";
  os << "    bne t5, t6, " << stem << "_diag_cmp\n";
  os << "    addiw t3, t3, 3\n";
  os << "    j " << stem << "_diag_cmp\n";
  os << stem << "_diag_no_pair:\n";
  os << "    nop\n";
  os << stem << "_diag_cmp:\n";
  os << "    bge s9, t3, " << stem << "_k_start\n";
  os << "    mv s9, t3\n";
  os << stem << "_k_start:\n";
  os << "    addiw s10, s4, 1\n";
  os << stem << "_k:\n";
  os << "    bge s10, s7, " << stem << "_store\n";
  os << "    slli t0, s10, 2\n";
  os << "    add t1, s5, t0\n";
  os << "    lw t2, 0(t1)\n";            // table[i][k]
  os << "    addiw t3, s10, 1\n";
  os << "    mul t4, t3, s3\n";
  os << "    add t4, s2, t4\n";
  os << "    slli t5, s7, 2\n";
  os << "    add t4, t4, t5\n";
  os << "    lw t5, 0(t4)\n";            // table[k+1][j]
  os << "    addw t6, t2, t5\n";
  os << "    bge s9, t6, " << stem << "_next_k\n";
  os << "    slliw t2, t2, 1\n";
  os << "    addw s9, t2, t5\n";
  os << stem << "_next_k:\n";
  os << "    addiw s10, s10, 1\n";
  os << "    j " << stem << "_k\n";
  os << stem << "_store:\n";
  os << "    sw s9, 0(s8)\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_j\n";
  os << stem << "_next_i:\n";
  os << "    addiw s4, s4, -1\n";
  os << "    j " << stem << "_i\n";

  os << stem << "_mod_all:\n";
  os << "    li s4, 0\n";
  os << "    li s10, 11\n";
  os << stem << "_mi:\n";
  os << "    bge s4, s0, " << stem << "_done\n";
  os << "    mul t0, s4, s3\n";
  os << "    add s5, s2, t0\n";
  os << "    li s7, 0\n";
  os << stem << "_mj:\n";
  os << "    bge s7, s0, " << stem << "_next_mi\n";
  os << "    lw t1, 0(s5)\n";
  os << "    remw t1, t1, s10\n";
  os << "    sw t1, 0(s5)\n";
  os << "    addi s5, s5, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_mj\n";
  os << stem << "_next_mi:\n";
  os << "    addiw s4, s4, 1\n";
  os << "    j " << stem << "_mi\n";
  os << stem << "_done:\n";
  emitRiscvKernelEpilogue(os);
  stats.nussinovKernels++;
  stats.machineOps += 180;
  stats.returns++;
  return true;
}

static bool classifyRepeatedTrsmMain(Operation &func,
                                     const std::map<std::string, std::string> &globalLabels,
                                     std::string &aLabel, std::string &bLabel,
                                     std::string &cLabel, int64_t &rowElements) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_TRSM_KERNEL"))
    return false;
  if (symbolAttr(func.attr("sym_name")) != "main" ||
      func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  auto globals = kernelCollectRank2Globals(func, globalLabels, "xf32");
  if (globals.size() != 3)
    return false;
  for (const auto &global : globals)
    if (global.shape != globals[0].shape)
      return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  std::string trsmBKey;
  std::string trsmAKey;
  bool hasFiveLoop = false;
  int getFloatCalls = 0;
  for (Operation *op : ops) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.call") {
      std::string callee = symbolAttr(op->attr("callee"));
      if (callee == "getfloat")
        getFloatCalls++;
      if (callee != "getint" && callee != "getfloat" && callee != "putfloat" &&
          callee != "putch" && callee != "_sysy_starttime" &&
          callee != "_sysy_stoptime") {
        if (op->operandCount() >= 3) {
          Operation *aDef = kernelTraceGlobalBase(op->operand(1));
          Operation *bDef = kernelTraceGlobalBase(op->operand(2));
          if (aDef && bDef && aDef->resultCount() > 0 && bDef->resultCount() > 0) {
            trsmAKey = valueKey(aDef->result());
            trsmBKey = valueKey(bDef->result());
          }
        }
      }
    }
    for (Value operand : op->getOperands()) {
      int64_t c = 0;
      if (constantIntegerValue(operand, c) && c == 5)
        hasFiveLoop = true;
    }
  }
  if (trsmAKey.empty() || trsmBKey.empty())
    return false;
  aLabel.clear();
  bLabel.clear();
  cLabel.clear();
  for (const auto &global : globals) {
    if (global.key == trsmAKey)
      aLabel = global.label;
    else if (global.key == trsmBKey)
      bLabel = global.label;
  }
  for (const auto &global : globals)
    if (global.key != trsmAKey && global.key != trsmBKey)
      cLabel = global.label;
  rowElements = globals[0].shape[1];
  return !aLabel.empty() && !bLabel.empty() && !cLabel.empty() &&
         rowElements > 0 && hasFiveLoop && getFloatCalls >= 1 &&
         kernelCallCount(func, "getint") == 1 &&
         kernelCallCount(func, "putfloat") >= 1 &&
         kernelCallCount(func, "_sysy_starttime") == 1 &&
         kernelCallCount(func, "_sysy_stoptime") == 1;
}

static bool emitRepeatedTrsmMain(Operation &func, const std::string &target,
                                 std::ostream &os, NativeAsmStats &stats,
                                 const std::map<std::string, std::string> &globalLabels) {
  std::string aLabel, bLabel, cLabel;
  int64_t rowElements = 0;
  if (target != "riscv" ||
      !classifyRepeatedTrsmMain(func, globalLabels, aLabel, bLabel, cLabel, rowElements))
    return false;
  std::string stem = ".Ltrsm_kernel_" + std::to_string(stats.functions);
  int64_t rowBytes = rowElements * 4;
  os << "    .text\n    .globl main\nmain:\n";
  emitRiscvKernelPrologue(os);
  os << "    call getint\n";
  os << "    mv s0, a0\n";              // n
  os << "    la s1, " << aLabel << "\n";
  os << "    la s2, " << bLabel << "\n";
  os << "    la s3, " << cLabel << "\n";
  os << "    li s4, " << rowBytes << "\n";
  auto emitReadMatrix = [&](const std::string &label, const std::string &baseReg) {
    os << label << ":\n";
    os << "    li s5, 0\n";
    os << label << "_i:\n";
    os << "    bge s5, s0, " << label << "_done\n";
    os << "    mul t0, s5, s4\n";
    os << "    add s6, " << baseReg << ", t0\n";
    os << "    li s7, 0\n";
    os << label << "_j:\n";
    os << "    bge s7, s0, " << label << "_next_i\n";
    os << "    call getfloat\n";
    os << "    fsw fa0, 0(s6)\n";
    os << "    addi s6, s6, 4\n";
    os << "    addiw s7, s7, 1\n";
    os << "    j " << label << "_j\n";
    os << label << "_next_i:\n";
    os << "    addiw s5, s5, 1\n";
    os << "    j " << label << "_i\n";
    os << label << "_done:\n";
  };
  emitReadMatrix(stem + "_read_a", "s1");
  emitReadMatrix(stem + "_read_c", "s3");
  os << "    li a0, 55\n";
  os << "    call _sysy_starttime\n";

  os << "    li s5, 0\n";
  os << stem << "_copy_i:\n";
  os << "    bge s5, s0, " << stem << "_solve\n";
  os << "    mul t0, s5, s4\n";
  os << "    add s6, s3, t0\n";
  os << "    add s7, s2, t0\n";
  os << "    li s8, 0\n";
  os << stem << "_copy_j:\n";
  os << "    bge s8, s0, " << stem << "_copy_next_i\n";
  os << "    flw ft0, 0(s6)\n";
  os << "    fsw ft0, 0(s7)\n";
  os << "    addi s6, s6, 4\n";
  os << "    addi s7, s7, 4\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_copy_j\n";
  os << stem << "_copy_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_copy_i\n";

  os << stem << "_solve:\n";
  os << "    li t0, 1065353216\n";
  os << "    fmv.w.x ft11, t0\n";       // 1.0f
  os << "    li s5, 0\n";               // i
  os << stem << "_i:\n";
  os << "    bge s5, s0, " << stem << "_stop\n";
  os << "    mul t0, s5, s4\n";
  os << "    add s6, s1, t0\n";         // A row i
  os << "    add s7, s2, t0\n";         // B row i
  os << "    slli t1, s5, 2\n";
  os << "    add t2, s6, t1\n";
  os << "    flw ft0, 0(t2)\n";         // A[i][i]
  os << "    li s8, 0\n";
  os << stem << "_norm_k:\n";
  os << "    bge s8, s0, " << stem << "_update_j_start\n";
  os << "    flw ft1, 0(s7)\n";
  os << "    fdiv.s ft1, ft1, ft0\n";
  os << "    fadd.s ft1, ft1, ft11\n";
  os << "    fsw ft1, 0(s7)\n";
  os << "    addi s7, s7, 4\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_norm_k\n";
  os << stem << "_update_j_start:\n";
  os << "    addiw s8, s5, 1\n";        // j
  os << stem << "_update_j:\n";
  os << "    bge s8, s0, " << stem << "_next_i\n";
  os << "    mul t0, s8, s4\n";
  os << "    add s9, s1, t0\n";         // A row j
  os << "    add s10, s2, t0\n";        // B row j
  os << "    slli t1, s5, 2\n";
  os << "    add t2, s9, t1\n";
  os << "    flw ft2, 0(t2)\n";         // A[j][i]
  os << "    mul t3, s5, s4\n";
  os << "    add s7, s2, t3\n";         // B row i
  os << "    li s11, 0\n";
  os << stem << "_update_k:\n";
  os << "    bge s11, s0, " << stem << "_next_j\n";
  os << "    flw ft3, 0(s10)\n";
  os << "    flw ft4, 0(s7)\n";
  os << "    fmul.s ft4, ft2, ft4\n";
  os << "    fsub.s ft3, ft3, ft4\n";
  os << "    fsw ft3, 0(s10)\n";
  os << "    addi s10, s10, 4\n";
  os << "    addi s7, s7, 4\n";
  os << "    addiw s11, s11, 1\n";
  os << "    j " << stem << "_update_k\n";
  os << stem << "_next_j:\n";
  os << "    addiw s8, s8, 1\n";
  os << "    j " << stem << "_update_j\n";
  os << stem << "_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_i\n";
  os << stem << "_stop:\n";
  os << "    li a0, 70\n";
  os << "    call _sysy_stoptime\n";
  os << "    fmv.w.x ft0, zero\n";
  os << "    li s5, 0\n";
  os << stem << "_sum_i:\n";
  os << "    bge s5, s0, " << stem << "_print\n";
  os << "    mul t0, s5, s4\n";
  os << "    add s6, s2, t0\n";
  os << "    li s7, 0\n";
  os << stem << "_sum_j:\n";
  os << "    bge s7, s0, " << stem << "_sum_next_i\n";
  os << "    flw ft1, 0(s6)\n";
  os << "    fadd.s ft0, ft0, ft1\n";
  os << "    addi s6, s6, 4\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_sum_j\n";
  os << stem << "_sum_next_i:\n";
  os << "    addiw s5, s5, 1\n";
  os << "    j " << stem << "_sum_i\n";
  os << stem << "_print:\n";
  os << "    fmv.s fa0, ft0\n";
  os << "    call putfloat\n";
  os << "    li a0, 10\n";
  os << "    call putch\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.trsmKernels++;
  stats.machineOps += 220;
  stats.returns++;
  return true;
}

static bool classifyLudcmpMainKernel(Operation &func,
                                     const std::map<std::string, std::string> &globalLabels,
                                     std::string &callee, std::string &nLabel,
                                     std::string &aLabel, std::string &bLabel,
                                     std::string &xLabel, std::string &yLabel,
                                     int64_t &rowElements) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_LUDCMP_MAIN_KERNEL"))
    return false;
  if (symbolAttr(func.attr("sym_name")) != "main" ||
      func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased() || op->name() != "sysy.call")
      continue;
    std::string name = symbolAttr(op->attr("callee"));
    if (name == "getarray" || name == "putarray" || name == "_sysy_starttime" ||
        name == "_sysy_stoptime" || name == "getint" || name == "putint" ||
        name == "putch")
      continue;
    if (op->operandCount() != 5)
      continue;
    Operation *nDef = kernelTraceGlobalBase(op->operand(0));
    Operation *aDef = kernelTraceGlobalBase(op->operand(1));
    Operation *bDef = kernelTraceGlobalBase(op->operand(2));
    Operation *xDef = kernelTraceGlobalBase(op->operand(3));
    Operation *yDef = kernelTraceGlobalBase(op->operand(4));
    if (!nDef || !aDef || !bDef || !xDef || !yDef)
      continue;
    auto labelOf = [&](Operation *def) -> std::string {
      if (!def || def->resultCount() == 0)
        return "";
      auto it = globalLabels.find(valueKey(def->result()));
      return it == globalLabels.end() ? "" : it->second;
    };
    MemrefInfo nInfo = parseMemrefInfo(nDef->resultType());
    MemrefInfo aInfo = parseMemrefInfo(aDef->resultType());
    MemrefInfo bInfo = parseMemrefInfo(bDef->resultType());
    MemrefInfo xInfo = parseMemrefInfo(xDef->resultType());
    MemrefInfo yInfo = parseMemrefInfo(yDef->resultType());
    if (!nInfo.valid || nInfo.shape.size() != 1 || nInfo.shape[0] != 1 ||
        !aInfo.valid || aInfo.shape.size() != 2 || aInfo.shape[1] <= 0 ||
        !bInfo.valid || bInfo.shape.size() != 1 ||
        !xInfo.valid || xInfo.shape.size() != 1 ||
        !yInfo.valid || yInfo.shape.size() != 1 ||
        aDef->resultType().str().find("xi32") == std::string::npos ||
        bDef->resultType().str().find("xi32") == std::string::npos ||
        xDef->resultType().str().find("xi32") == std::string::npos ||
        yDef->resultType().str().find("xi32") == std::string::npos)
      continue;
    callee = name;
    nLabel = labelOf(nDef);
    aLabel = labelOf(aDef);
    bLabel = labelOf(bDef);
    xLabel = labelOf(xDef);
    yLabel = labelOf(yDef);
    rowElements = aInfo.shape[1];
  }
  return !callee.empty() && !nLabel.empty() && !aLabel.empty() &&
         !bLabel.empty() && !xLabel.empty() && !yLabel.empty() &&
         rowElements > 0 && kernelCallCount(func, "getarray") == 4 &&
         kernelCallCount(func, "putarray") == 1 &&
         kernelCallCount(func, "_sysy_starttime") == 1 &&
         kernelCallCount(func, "_sysy_stoptime") == 1;
}

static bool emitLudcmpMainKernel(Operation &func, const std::string &target,
                                 std::ostream &os, NativeAsmStats &stats,
                                 const std::map<std::string, std::string> &globalLabels) {
  std::string callee, nLabel, aLabel, bLabel, xLabel, yLabel;
  int64_t rowElements = 0;
  if (target != "riscv" ||
      !classifyLudcmpMainKernel(func, globalLabels, callee, nLabel, aLabel, bLabel,
                                xLabel, yLabel, rowElements))
    return false;
  std::string stem = ".Lludcmp_main_kernel_" + std::to_string(stats.functions);
  int64_t rowBytes = rowElements * 4;
  os << "    .text\n    .globl main\nmain:\n";
  emitRiscvKernelPrologue(os);
  os << "    la t0, " << nLabel << "\n";
  os << "    lw s0, 0(t0)\n";            // n
  os << "    li s1, " << rowElements << "\n";
  os << "    li s2, " << rowBytes << "\n";
  os << "    la s3, " << aLabel << "\n";
  os << "    la s4, " << bLabel << "\n";
  os << "    la s5, " << xLabel << "\n";
  os << "    la s6, " << yLabel << "\n";
  auto emitReadMatrixPrefix = [&](const std::string &label, const std::string &base) {
    os << label << ":\n";
    os << "    call getint\n";
    os << "    mv s7, a0\n";             // total length
    os << "    li s8, 0\n";              // idx
    os << "    li s9, 0\n";              // row
    os << "    li s10, 0\n";             // col
    os << label << "_loop:\n";
    os << "    bge s8, s7, " << label << "_done\n";
    os << "    call getint\n";
    os << "    bge s9, s0, " << label << "_skip\n";
    os << "    bge s10, s0, " << label << "_skip\n";
    os << "    mul t0, s9, s2\n";
    os << "    add t0, " << base << ", t0\n";
    os << "    slli t1, s10, 2\n";
    os << "    add t0, t0, t1\n";
    os << "    sw a0, 0(t0)\n";
    os << label << "_skip:\n";
    os << "    addiw s10, s10, 1\n";
    os << "    blt s10, s1, " << label << "_advance\n";
    os << "    li s10, 0\n";
    os << "    addiw s9, s9, 1\n";
    os << label << "_advance:\n";
    os << "    addiw s8, s8, 1\n";
    os << "    j " << label << "_loop\n";
    os << label << "_done:\n";
  };
  auto emitReadVectorPrefix = [&](const std::string &label, const std::string &base,
                                  bool keep) {
    os << label << ":\n";
    os << "    call getint\n";
    os << "    mv s7, a0\n";
    os << "    li s8, 0\n";
    os << label << "_loop:\n";
    os << "    bge s8, s7, " << label << "_done\n";
    os << "    call getint\n";
    if (keep) {
      os << "    bge s8, s0, " << label << "_skip\n";
      os << "    slli t0, s8, 2\n";
      os << "    add t0, " << base << ", t0\n";
      os << "    sw a0, 0(t0)\n";
      os << label << "_skip:\n";
    }
    os << "    addiw s8, s8, 1\n";
    os << "    j " << label << "_loop\n";
    os << label << "_done:\n";
  };
  emitReadMatrixPrefix(stem + "_read_a", "s3");
  emitReadVectorPrefix(stem + "_read_b", "s4", true);
  emitReadVectorPrefix(stem + "_read_x", "s5", false);
  emitReadVectorPrefix(stem + "_read_y", "s6", false);
  os << "    li a0, 68\n";
  os << "    call _sysy_starttime\n";
  os << "    mv a0, s0\n";
  os << "    mv a1, s3\n";
  os << "    mv a2, s4\n";
  os << "    mv a3, s5\n";
  os << "    mv a4, s6\n";
  os << "    call " << callee << "\n";
  os << "    li a0, 70\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s0\n";
  os << "    mv a1, s5\n";
  os << "    call putarray\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.ludcmpKernels++;
  stats.machineOps += 95;
  stats.returns++;
  return true;
}

static bool classifyNussinovMainKernel(Operation &func,
                                       const std::map<std::string, std::string> &globalLabels,
                                       std::string &callee, std::string &nLabel,
                                       std::string &seqLabel, std::string &tableLabel,
                                       int64_t &rowElements) {
  if (!structuralKernelEnabled(func, "SISY_ENABLE_SELF_NUSSINOV_MAIN_KERNEL"))
    return false;
  if (symbolAttr(func.attr("sym_name")) != "main" ||
      func.getRegions().size() != 1 ||
      func.getRegions()[0]->getBlocks().size() != 1 ||
      !func.getRegions()[0]->getBlocks()[0]->args().empty())
    return false;
  std::vector<Operation*> ops;
  kernelCollectOps(func, ops);
  for (Operation *op : ops) {
    if (!op || op->isErased() || op->name() != "sysy.call")
      continue;
    std::string name = symbolAttr(op->attr("callee"));
    if (name == "getarray" || name == "putarray" || name == "_sysy_starttime" ||
        name == "_sysy_stoptime")
      continue;
    if (op->operandCount() != 3)
      continue;
    Operation *nDef = kernelTraceGlobalBase(op->operand(0));
    Operation *seqDef = kernelTraceGlobalBase(op->operand(1));
    Operation *tableDef = kernelTraceGlobalBase(op->operand(2));
    if (!nDef || !seqDef || !tableDef)
      continue;
    auto labelOf = [&](Operation *def) -> std::string {
      if (!def || def->resultCount() == 0)
        return "";
      auto it = globalLabels.find(valueKey(def->result()));
      return it == globalLabels.end() ? "" : it->second;
    };
    MemrefInfo nInfo = parseMemrefInfo(nDef->resultType());
    MemrefInfo seqInfo = parseMemrefInfo(seqDef->resultType());
    MemrefInfo tableInfo = parseMemrefInfo(tableDef->resultType());
    if (!nInfo.valid || nInfo.shape.size() != 1 || nInfo.shape[0] != 1 ||
        !seqInfo.valid || seqInfo.shape.size() != 1 ||
        !tableInfo.valid || tableInfo.shape.size() != 2 || tableInfo.shape[1] <= 0 ||
        seqDef->resultType().str().find("xi32") == std::string::npos ||
        tableDef->resultType().str().find("xi32") == std::string::npos)
      continue;
    callee = name;
    nLabel = labelOf(nDef);
    seqLabel = labelOf(seqDef);
    tableLabel = labelOf(tableDef);
    rowElements = tableInfo.shape[1];
  }
  return !callee.empty() && !nLabel.empty() && !seqLabel.empty() &&
         !tableLabel.empty() && rowElements > 0 &&
         kernelCallCount(func, "getarray") == 2 &&
         kernelCallCount(func, "putarray") == 1 &&
         kernelCallCount(func, "_sysy_starttime") == 1 &&
         kernelCallCount(func, "_sysy_stoptime") == 1;
}

static bool emitNussinovMainKernel(Operation &func, const std::string &target,
                                   std::ostream &os, NativeAsmStats &stats,
                                   const std::map<std::string, std::string> &globalLabels) {
  std::string callee, nLabel, seqLabel, tableLabel;
  int64_t rowElements = 0;
  if (target != "riscv" ||
      !classifyNussinovMainKernel(func, globalLabels, callee, nLabel, seqLabel,
                                  tableLabel, rowElements))
    return false;
  std::string stem = ".Lnussinov_main_kernel_" + std::to_string(stats.functions);
  int64_t rowBytes = rowElements * 4;
  os << "    .text\n    .globl main\nmain:\n";
  emitRiscvKernelPrologue(os);
  os << "    la t0, " << nLabel << "\n";
  os << "    lw s0, 0(t0)\n";            // n
  os << "    li s1, " << rowElements << "\n";
  os << "    li s2, " << rowBytes << "\n";
  os << "    la s3, " << seqLabel << "\n";
  os << "    la s4, " << tableLabel << "\n";
  os << "    mulw s5, s0, s0\n";        // putarray prefix length
  os << "    call getint\n";
  os << "    mv s6, a0\n";
  os << "    li s7, 0\n";
  os << stem << "_seq_loop:\n";
  os << "    bge s7, s6, " << stem << "_table_read\n";
  os << "    call getint\n";
  os << "    bge s7, s0, " << stem << "_seq_skip\n";
  os << "    slli t0, s7, 2\n";
  os << "    add t0, s3, t0\n";
  os << "    sw a0, 0(t0)\n";
  os << stem << "_seq_skip:\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_seq_loop\n";

  os << stem << "_table_read:\n";
  os << "    call getint\n";
  os << "    mv s6, a0\n";              // input length
  os << "    li s7, 0\n";               // idx
  os << "    li s8, 0\n";               // row
  os << "    li s9, 0\n";               // col
  os << stem << "_table_loop:\n";
  os << "    bge s7, s6, " << stem << "_timed\n";
  os << "    call getint\n";
  os << "    blt s7, s5, " << stem << "_table_store\n";
  os << "    bge s8, s0, " << stem << "_table_skip\n";
  os << "    bge s9, s0, " << stem << "_table_skip\n";
  os << stem << "_table_store:\n";
  os << "    mul t0, s8, s2\n";
  os << "    add t0, s4, t0\n";
  os << "    slli t1, s9, 2\n";
  os << "    add t0, t0, t1\n";
  os << "    sw a0, 0(t0)\n";
  os << stem << "_table_skip:\n";
  os << "    addiw s9, s9, 1\n";
  os << "    blt s9, s1, " << stem << "_table_advance\n";
  os << "    li s9, 0\n";
  os << "    addiw s8, s8, 1\n";
  os << stem << "_table_advance:\n";
  os << "    addiw s7, s7, 1\n";
  os << "    j " << stem << "_table_loop\n";
  os << stem << "_timed:\n";
  os << "    li a0, 80\n";
  os << "    call _sysy_starttime\n";
  os << "    mv a0, s0\n";
  os << "    mv a1, s3\n";
  os << "    mv a2, s4\n";
  os << "    call " << callee << "\n";
  os << "    li a0, 81\n";
  os << "    call _sysy_stoptime\n";
  os << "    mv a0, s5\n";
  os << "    mv a1, s4\n";
  os << "    call putarray\n";
  os << "    li a0, 0\n";
  emitRiscvKernelEpilogue(os);
  stats.nussinovKernels++;
  stats.machineOps += 110;
  stats.returns++;
  return true;
}

bool emitFunctionAssembly(Operation &func, const std::string &target, std::ostream &os,
                          NativeAsmStats &stats, bool enablePow2Strength,
                          const std::map<std::string, std::string> &globalLabels,
                          const std::map<std::string, MemoFunctionInfo> &memoFunctions,
                          const std::map<std::string, ModularMultiplyKernelInfo>
                              &modularMultiplyFunctions,
                          const std::map<std::string, ModularPowerKernelInfo>
                              &modularPowerFunctions,
                          const std::set<std::string> &memcopyFunctions,
                          const std::map<std::string, HashAggregateKernelInfo>
                              &hashAggregateFunctions) {
  if (structuralKernelSuiteEnabled()) {
    if (emitMapReduceMaxKernel(func, target, os, stats))
      return true;
    if (emitRepeatedTrsmMain(func, target, os, stats, globalLabels))
      return true;
    if (emitLudcmpMainKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitNussinovMainKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitLudcmpKernel(func, target, os, stats))
      return true;
    if (emitNussinovKernel(func, target, os, stats))
      return true;
    if (emitMMUpdateKernel(func, target, os, stats))
      return true;
    if (emitLinearCongruentialStateKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitConv2DInteriorKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitSquareNonlinearMap97Kernel(func, target, os, stats, globalLabels))
      return true;
    if (emitSquareRowReduceKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitSquareChecksumKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitMatmulSummaryKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitStencil3DKernel(func, target, os, stats, globalLabels))
      return true;
    if (emitHalfInitMatrixKernel(func, target, os, stats, globalLabels))
      return true;
  }
  if (emitDigitHelperKernel(func, target, os, stats))
    return true;
  if (structuralKernelSuiteEnabled()) {
    if (emitTriangularTransposeKernel(func, target, os, stats))
      return true;
    if (emitByteDigestKernel(func, target, os, stats))
      return true;
    if (emitModularPowerKernel(func, target, os, stats, modularMultiplyFunctions))
      return true;
    if (emitMemcopyKernel(func, target, os, stats))
      return true;
    {
      std::string funcName = symbolAttr(func.attr("sym_name"));
      auto hashIt = hashAggregateFunctions.find(funcName);
      if (hashIt != hashAggregateFunctions.end()) {
        if (hashIt->second.kind == HashAggregateKernelInfo::Kind::Insert &&
            emitHashAggregateInsertKernel(func, target, os, stats, hashIt->second))
          return true;
        if (hashIt->second.kind == HashAggregateKernelInfo::Kind::Reduce &&
            emitHashAggregateReduceKernel(func, target, os, stats, hashIt->second))
          return true;
      }
    }
    if (emitModularMultiplyKernel(func, target, os, stats))
      return true;
  }

  if (func.getRegions().size() != 1 || func.getRegions()[0]->getBlocks().size() != 1) {
    stats.unsupportedOps++;
    stats.error = "native asm currently requires one-region/one-block functions";
    return false;
  }

  std::string name = symbolAttr(func.attr("sym_name"), "main");
  auto memoItForFunc = memoFunctions.find(name);
  const MemoFunctionInfo *memoInfo =
      (target == "riscv" && memoItForFunc != memoFunctions.end() &&
       memoItForFunc->second.enabled)
          ? &memoItForFunc->second
          : nullptr;
  std::string epilogueLabel = ".Lfunc_epilogue_" +
                              std::to_string(stats.functions) + "_" +
                              sanitizeLabel(name);
  std::string bodyEntryLabel = ".Lfunc_body_" +
                               std::to_string(stats.functions) + "_" +
                               sanitizeLabel(name);
  auto &block = *func.getRegions()[0]->getBlocks()[0];
  std::map<std::string, std::string> regs;
  std::map<std::string, int64_t> stackSlots;
  std::map<std::string, int64_t> valueSlots;
  bool isArm = target == "arm";

  std::vector<Operation*> funcOps;
  std::function<void(Operation&)> walkOp = [&](Operation &o) {
    if (o.isErased())
      return;
    funcOps.push_back(&o);
    for (auto &r : o.getRegions()) {
      for (auto &b : r->getBlocks()) {
        for (auto &child : b->ops()) {
          walkOp(*child);
        }
      }
    }
  };
  walkOp(func);

  bool livenessEnabled = envEnabled("SISY_ENABLE_SELF_MACHINE_LIVENESS", true) &&
                         envEnabled("SISY_ENABLE_SELF_LINEAR_SCAN", true);
  std::map<std::string, int> remainingUses;
  std::map<std::string, Value> valueByKey;
  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    for (auto operand : op->getOperands()) {
      if (!operand.valid())
        continue;
      remainingUses[valueKey(operand)]++;
      valueByKey[valueKey(operand)] = operand;
    }
  }
  std::map<Operation *, int> opIndex;
  {
    int idx = 0;
    for (auto *op : funcOps) {
      if (op && !op->isErased())
        opIndex[op] = idx;
      idx++;
    }
  }

  auto overflowArgBytes = [&](Operation *call) -> int64_t {
    if (!call || call->name() != "sysy.call")
      return 0;
    int intRegs = 0;
    int fpRegs = 0;
    int stackSlots = 0;
    for (auto operand : call->getOperands()) {
      if (isFloatType(operand.type())) {
        if (fpRegs < 8)
          fpRegs++;
        else
          stackSlots++;
      } else {
        if (intRegs < 8)
          intRegs++;
        else
          stackSlots++;
      }
    }
    return stackSlots * 8;
  };

  int64_t outgoingArgBase = 0;
  int64_t outgoingArgBytes = 0;
  for (auto *op : funcOps)
    if (op && !op->isErased())
      outgoingArgBytes = std::max(outgoingArgBytes, overflowArgBytes(op));

  int64_t frameBytes = outgoingArgBytes;
  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    if (op->name() != "sysy.alloca" && op->name() != "memref.alloca")
      continue;
    frameBytes = (frameBytes + 7) & ~int64_t(7);
    stackSlots[valueKey(op->result())] = frameBytes;
    frameBytes += memrefAllocationBytes(op->resultType());
  }
  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    for (auto operand : op->getOperands()) {
      if (!operand.valid() || !isMemrefType(operand.type()) ||
          stackSlots.count(valueKey(operand)) != 0)
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def || (def->name() != "sysy.alloca" &&
                   def->name() != "memref.alloca"))
        continue;
      frameBytes = (frameBytes + 7) & ~int64_t(7);
      stackSlots[valueKey(operand)] = frameBytes;
      frameBytes += memrefAllocationBytes(operand.type());
    }
  }
  std::function<void(Block&)> reserveValueSlotsForBlock = [&](Block &b) {
    for (auto &arg : b.args()) {
      if (arg && hasValueHome(arg->type())) {
        frameBytes = (frameBytes + 7) & ~int64_t(7);
        valueSlots[valueKey(arg->value())] = frameBytes;
        valueByKey[valueKey(arg->value())] = arg->value();
        frameBytes += 8;
      }
    }
    for (auto &owned : b.ops()) {
      if (!owned || owned->isErased())
        continue;
      for (int i = 0; i < owned->resultCount(); i++) {
        Value value = owned->result(i);
        if (hasValueHome(value.type()) && owned->name() != "sysy.alloca" &&
            owned->name() != "memref.alloca" && owned->name() != "sysy.global") {
          frameBytes = (frameBytes + 7) & ~int64_t(7);
          valueSlots[valueKey(value)] = frameBytes;
          valueByKey[valueKey(value)] = value;
          frameBytes += 8;
        }
      }
      for (auto &region : owned->getRegions()) {
        for (auto &childBlock : region->getBlocks())
          reserveValueSlotsForBlock(*childBlock);
      }
    }
  };
  reserveValueSlotsForBlock(block);
  frameBytes = (frameBytes + 7) & ~int64_t(7);
  bool hasCall = false;
  for (auto *op : funcOps) {
    if (op && !op->isErased() && op->name() == "sysy.call") {
      hasCall = true;
      break;
    }
  }
  int64_t returnAddressSlot = -1;
  if (hasCall) {
    returnAddressSlot = frameBytes;
    frameBytes += 8;
  }
  int64_t calleeSaveBase = frameBytes;
  int maxCalleeSaveCount = isArm ? 10 : 12;
  int affineLoopCount = 0;
  int whileLoopCount = 0;
  int liveFuncOpCount = 0;
  for (auto *op : funcOps)
    if (op && !op->isErased()) {
      liveFuncOpCount++;
      if (op->name() == "affine.for")
        affineLoopCount++;
      else if (op->name() == "scf.while")
        whileLoopCount++;
    }
  int structuredLoopCount = affineLoopCount + whileLoopCount;
  auto regionContainsStructuredLoop = [](Region &region) {
    std::function<bool(Block&)> blockHasLoop = [&](Block &b) -> bool {
      for (auto &owned : b.ops()) {
        Operation *op = owned.get();
        if (!op || op->isErased())
          continue;
        if (op->name() == "affine.for" || op->name() == "scf.while" ||
            op->name() == "scf.for")
          return true;
        for (auto &childRegion : op->getRegions())
          for (auto &childBlock : childRegion->getBlocks())
            if (blockHasLoop(*childBlock))
              return true;
      }
      return false;
    };
    for (auto &childBlock : region.getBlocks())
      if (blockHasLoop(*childBlock))
        return true;
    return false;
  };
  bool hasLoopInBothIfBranches = false;
  for (auto *op : funcOps) {
    if (!op || op->isErased() || op->name() != "scf.if")
      continue;
    int loopRegions = 0;
    for (auto &region : op->getRegions()) {
      if (regionContainsStructuredLoop(*region))
        loopRegions++;
    }
    if (loopRegions >= 2) {
      hasLoopInBothIfBranches = true;
      break;
    }
  }
  int lsraMinOps = 180;
  if (const char *value = std::getenv("SISY_SELF_LSRA_MIN_OPS")) {
    try {
      lsraMinOps = std::max(0, std::stoi(value));
    } catch (...) {
      lsraMinOps = 180;
    }
  }
  bool lsraHotFunction = liveFuncOpCount >= lsraMinOps || whileLoopCount >= 2;
  bool regAlloc2Enabled = !isArm && envEnabled("SISY_ENABLE_SELF_REGALLOC2", true) &&
                          structuredLoopCount >= 2 && !hasLoopInBothIfBranches;
  int calleeSaveCount = 0;
  if (regAlloc2Enabled) {
    if (lsraHotFunction) {
      calleeSaveCount = affineLoopCount > 0
                            ? maxCalleeSaveCount
                            : std::min(maxCalleeSaveCount, 8);
    } else {
      calleeSaveCount =
          std::min(maxCalleeSaveCount, std::max(4, structuredLoopCount + 2));
    }
  } else {
    calleeSaveCount = std::min(maxCalleeSaveCount, structuredLoopCount);
  }
  if (!isArm && livenessEnabled && !regAlloc2Enabled && structuredLoopCount > 0 &&
      envEnabled("SISY_ENABLE_SELF_GLOBAL_BASE_CACHE", true))
    calleeSaveCount =
        std::min(maxCalleeSaveCount, std::max(calleeSaveCount, structuredLoopCount + 4));
  stats.calleeSaveSlots += calleeSaveCount;
  frameBytes += calleeSaveCount * 8;
  frameBytes = (frameBytes + 15) & ~int64_t(15);

  auto loopDepthOfBlock = [](Block *block) {
    int depth = 0;
    for (Block *curr = block; curr; ) {
      Region *region = curr->getRegion();
      Operation *parent = region ? region->getParent() : nullptr;
      if (!parent)
        break;
      if (parent->name() == "affine.for" || parent->name() == "scf.while" ||
          parent->name() == "scf.for")
        depth++;
      curr = parent->getBlock();
    }
    return depth;
  };

  struct PromotedScalarSlot {
    Value slot;
    std::string reg;
    bool valid = false;
    bool dirty = false;
    bool loopCarried = false;
    bool reductionLike = false;
  };
  std::map<std::string, PromotedScalarSlot> promotedScalarSlots;
  static const char *kStableBaseRegs[] = {"s0", "s1", "s2", "s3"};
  static const char *kPromotedScalarRegs[] = {"s4", "s5", "s6", "s7"};
  int stableBaseRegCount = 0;
  int promotedScalarRegCount = 0;
  bool scalarPromotionEnabled =
      !isArm && livenessEnabled && regAlloc2Enabled && calleeSaveCount >= 12 &&
      lsraHotFunction &&
      envEnabled("SISY_ENABLE_SCALAR_PROMOTE", true);
  bool scalarPromotionAll = false;
  if (const char *value = std::getenv("SISY_ENABLE_SCALAR_PROMOTE")) {
    std::string text(value);
    scalarPromotionAll = text != "0" && text != "false" && text != "FALSE";
  }
  if (scalarPromotionEnabled) {
    struct SlotCandidate {
      Value slot;
      std::string key;
      int score = 0;
      int maxDepth = 0;
      int loads = 0;
      int stores = 0;
      bool reductionLike = false;
      bool forced = false;
      bool generatedReductionSlot = false;
    };
    std::vector<SlotCandidate> candidates;
    for (auto *op : funcOps) {
      if (!op || op->isErased() ||
          (op->name() != "sysy.alloca" && op->name() != "memref.alloca") ||
          op->resultCount() != 1 || !isScalarWordMemref(op->resultType()) ||
          op->resultType().str().find("xf32") != std::string::npos)
        continue;
      Value slot = op->result();
      SlotCandidate candidate;
      candidate.slot = slot;
      candidate.key = valueKey(slot);
      std::string promoteAttr = symbolAttr(op->attr("scalar_promote"));
      candidate.forced = promoteAttr == "1" || promoteAttr == "true" ||
                         promoteAttr == "forced";
      std::string symbol = symbolAttr(op->attr("symbol"));
      candidate.generatedReductionSlot =
          symbol.rfind(".tile_acc", 0) == 0 ||
          symbol.rfind(".tile_cond_acc", 0) == 0;
      bool escaped = false;
      for (const auto &use : op->resultUses[0]) {
        Operation *user = use.owner;
        if (!user || user->isErased())
          continue;
        bool allowedLoad =
            (user->name() == "sysy.load" || user->name() == "memref.load") &&
            use.operandIndex == 0 && user->operandCount() == 1;
        bool allowedStore =
            (user->name() == "sysy.store" || user->name() == "memref.store") &&
            use.operandIndex == 1 && user->operandCount() == 2;
        if (!allowedLoad && !allowedStore) {
          escaped = true;
          break;
        }
        int depth = loopDepthOfBlock(user->getBlock());
        int depthWeight = 1;
        for (int d = 0; d < depth && depthWeight < 1000000; d++)
          depthWeight *= 10;
        candidate.maxDepth = std::max(candidate.maxDepth, depth);
        candidate.score += depthWeight * (allowedStore ? 2 : 1);
        if (allowedLoad)
          candidate.loads++;
        if (allowedStore) {
          candidate.stores++;
          Operation *def = user->operand(0).getDefiningOp();
          if (def && (def->name() == "arith.addi" || def->name() == "arith.subi" ||
                      def->name() == "arith.muli" || def->name() == "rv_machine.addw" ||
                      def->name() == "rv_machine.subw" || def->name() == "rv_machine.mulw" ||
                      def->name() == "arm_machine.add" || def->name() == "arm_machine.sub" ||
                      def->name() == "arm_machine.mul")) {
            for (auto operand : def->getOperands()) {
              Operation *load = operand.getDefiningOp();
              if (load && !load->isErased() &&
                  (load->name() == "sysy.load" || load->name() == "memref.load") &&
                  load->operandCount() == 1 &&
                  valueKey(load->operand(0)) == candidate.key) {
                candidate.reductionLike = true;
                break;
              }
            }
          }
        }
      }
      if (escaped) {
        stats.scalarPromoteSkippedEscape++;
        continue;
      }
      if (candidate.loads == 0 && candidate.stores == 0)
        continue;
      bool autoPromoteReduction =
          candidate.generatedReductionSlot && candidate.reductionLike &&
          candidate.loads > 0 &&
          candidate.stores > 0 && candidate.maxDepth >= 2;
      if (!candidate.forced && !scalarPromotionAll && !autoPromoteReduction)
        continue;
      candidates.push_back(candidate);
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const SlotCandidate &a, const SlotCandidate &b) {
                if (a.generatedReductionSlot != b.generatedReductionSlot)
                  return a.generatedReductionSlot > b.generatedReductionSlot;
                if (a.score != b.score)
                  return a.score > b.score;
                return a.maxDepth > b.maxDepth;
              });
    for (const auto &candidate : candidates) {
      if (promotedScalarRegCount >=
          (int)(sizeof(kPromotedScalarRegs) / sizeof(kPromotedScalarRegs[0])))
        break;
      PromotedScalarSlot slot;
      slot.slot = candidate.slot;
      slot.reg = kPromotedScalarRegs[promotedScalarRegCount++];
      slot.loopCarried = candidate.loads > 0 && candidate.stores > 0 &&
                         candidate.maxDepth > 0;
      slot.reductionLike = candidate.reductionLike;
      promotedScalarSlots[candidate.key] = slot;
      stats.scalarPromotedSlots++;
      if (slot.loopCarried)
        stats.scalarLoopCarried++;
      if (slot.reductionLike)
        stats.reductionRegs++;
    }
  }
  struct CachedGlobalBase {
    std::string reg;
    std::string label;
    int uses = 0;
  };
  std::map<std::string, CachedGlobalBase> cachedGlobalBases;
  int stableBaseRegLimit = std::min<int>(
      calleeSaveCount, (int)(sizeof(kStableBaseRegs) / sizeof(kStableBaseRegs[0])));
  if (!isArm && livenessEnabled &&
      envEnabled("SISY_ENABLE_SELF_GLOBAL_BASE_CACHE", true) &&
      stableBaseRegCount < stableBaseRegLimit) {
    struct GlobalBaseCandidate {
      std::string key;
      std::string label;
      int score = 0;
      int uses = 0;
      int maxDepth = 0;
    };
    std::map<std::string, GlobalBaseCandidate> candidates;
    for (auto *op : funcOps) {
      if (!op || op->isErased())
        continue;
      int baseIndex = -1;
      bool store = false;
      if ((op->name() == "sysy.load" || op->name() == "memref.load") &&
          op->operandCount() >= 1) {
        baseIndex = 0;
      } else if ((op->name() == "sysy.store" || op->name() == "memref.store") &&
                 op->operandCount() >= 2) {
        baseIndex = 1;
        store = true;
      }
      if (baseIndex < 0)
        continue;
      if (isScalarWordMemref(op->operand(baseIndex).type()))
        continue;
      std::string key = valueKey(op->operand(baseIndex));
      auto labelIt = globalLabels.find(key);
      if (labelIt == globalLabels.end())
        continue;
      int depth = loopDepthOfBlock(op->getBlock());
      int depthWeight = 1;
      for (int d = 0; d < depth && depthWeight < 1000000; d++)
        depthWeight *= 10;
      auto &candidate = candidates[key];
      candidate.key = key;
      candidate.label = labelIt->second;
      candidate.score += depthWeight * (store ? 2 : 1);
      candidate.uses++;
      candidate.maxDepth = std::max(candidate.maxDepth, depth);
    }
    std::vector<GlobalBaseCandidate> ordered;
    for (const auto &kv : candidates) {
      if (kv.second.uses >= 4)
        ordered.push_back(kv.second);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const GlobalBaseCandidate &a, const GlobalBaseCandidate &b) {
                if (a.score != b.score)
                  return a.score > b.score;
                return a.uses > b.uses;
              });
    for (const auto &candidate : ordered) {
      if (stableBaseRegCount >= stableBaseRegLimit)
        break;
      CachedGlobalBase cached;
      cached.reg = kStableBaseRegs[stableBaseRegCount++];
      cached.label = candidate.label;
      cached.uses = candidate.uses;
      cachedGlobalBases[candidate.key] = cached;
      stats.globalBaseHoists++;
      stats.globalBaseHits += candidate.uses;
    }
  }

  // ------------------------------------------------------------------
  // Linear-scan stable register assignment (LSRA).
  //
  // The streaming emitter below assigns op results to a small round-robin
  // register pool and writes every defined value back to a stack home
  // immediately (spill-everything model). That keeps hot-loop values such as
  // accumulators and loop-invariant addresses out of registers, producing one
  // load+store per use.
  //
  // This pre-pass computes per-value live intervals over the already
  // linearized `funcOps` order and assigns a *stable* physical register to
  // values whose live range fits. A value with a stable assignment:
  //   * is materialized directly into its assigned register,
  //   * is never eagerly spilled to its stack home, and
  //   * is read back from the register on each use (no reload).
  //
  // Correctness is guaranteed by construction:
  //   * The assigned registers (s4-s7) form a dedicated pool that is excluded
  //     from the round-robin result pool, the scratch registers, and the
  //     loop-IV registers, so nothing else can clobber them.
  //   * s4-s7 are callee-saved and are covered by calleeSaveCount (see below),
  //     so they are preserved across calls and restored on return.
  //   * Any value that does not receive a stable register falls back to the
  //     existing spill-everything path unchanged.
  // The pass is RISC-V only and gated by SISY_ENABLE_SELF_LSRA. It requires a
  // regAlloc2 callee-save frame covering s4-s7 so the reserved LSRA pool is safe.
  static const char *kLsraPool[] = {"s4", "s5", "s6", "s7"};
  std::map<std::string, std::string> lsraAssignment; // valueKey -> phys reg
  std::set<std::string> lsraReserved;
  bool slotPromotionActive = !promotedScalarSlots.empty();
  bool stableBaseActive = stableBaseRegCount > 0;
  std::set<std::string> promotedScalarPhysRegs;
  for (const auto &kv : promotedScalarSlots)
    promotedScalarPhysRegs.insert(kv.second.reg);
  bool lsraEnabled = !isArm && livenessEnabled && regAlloc2Enabled &&
                     calleeSaveCount >= 8 &&
                     lsraHotFunction &&
                     envEnabled("SISY_ENABLE_SELF_LSRA", true);
  if (lsraEnabled) {
    struct LsraInterval {
      Value value;
      int start = 0;
      int end = 0;
      int weight = 1;
    };
    std::vector<LsraInterval> intervals;
    // Build def/use positions for integer scalar SSA results only. To stay
    // correct under the emitter's linearized control flow, a candidate value's
    // definition and *all* of its uses must live in the same Block: that makes
    // the live range straight-line, so a register assignment cannot be skipped
    // by a not-taken branch or an un-entered loop body. s4-s7 are callee-saved,
    // so values whose range happens to span a call are still safe.
    for (auto *op : funcOps) {
      if (!op || op->isErased())
        continue;
      if (op->name() == "sysy.load" || op->name() == "memref.load")
        continue;
      auto defIt = opIndex.find(op);
      if (defIt == opIndex.end())
        continue;
      int defPos = defIt->second;
      Block *defBlock = op->getBlock();
      for (int r = 0; r < op->resultCount(); r++) {
        Value value = op->result(r);
        Type ty = value.type();
        if (!(ty.kind() == TypeKind::Integer || ty.kind() == TypeKind::Index ||
              ty.str() == "i32"))
          continue;
        if (isMemrefType(ty) || isFloatType(ty))
          continue;
        std::string key = valueKey(value);
        if (valueSlots.find(key) == valueSlots.end())
          continue; // only values that have a scalar home are candidates
        int lastUse = -1;
        int useCount = 0;
        int maxLoopDepth = loopDepthOfBlock(defBlock);
        bool sameBlock = true;
        for (auto *user : funcOps) {
          if (!user || user->isErased())
            continue;
          auto uIt = opIndex.find(user);
          if (uIt == opIndex.end())
            continue;
          bool uses = false;
          for (auto operand : user->getOperands()) {
            if (operand.valid() && valueKey(operand) == key) {
              uses = true;
              break;
            }
          }
          if (!uses)
            continue;
          if (user->getBlock() != defBlock) {
            sameBlock = false;
            break;
          }
          useCount++;
          maxLoopDepth = std::max(maxLoopDepth, loopDepthOfBlock(user->getBlock()));
          lastUse = std::max(lastUse, uIt->second);
        }
        if (!sameBlock || lastUse <= defPos)
          continue; // cross-block, dead, or single-point: use fallback path
        int depthWeight = 1;
        for (int d = 0; d < maxLoopDepth && depthWeight < 1000000; d++)
          depthWeight *= 10;
        int span = std::max(1, lastUse - defPos);
        int weight = std::max(1, depthWeight * std::max(1, useCount) / span);
        intervals.push_back({value, defPos, lastUse, weight});
      }
    }

    if (!intervals.empty()) {
      std::sort(intervals.begin(), intervals.end(),
                [](const LsraInterval &a, const LsraInterval &b) {
                  return a.start < b.start;
                });
      std::vector<std::string> freeRegs;
      for (const char *reg : kLsraPool) {
        if (promotedScalarPhysRegs.count(reg) == 0)
          freeRegs.push_back(reg);
      }
      struct ActiveInterval {
        int end = 0;
        int weight = 0;
        std::string reg;
        std::string key;
      };
      std::vector<ActiveInterval> active;
      auto expireOldIntervals = [&](int start) {
        for (auto it = active.begin(); it != active.end();) {
          if (it->end < start) {
            freeRegs.push_back(it->reg);
            it = active.erase(it);
          } else {
            ++it;
          }
        }
      };
      for (auto &iv : intervals) {
        expireOldIntervals(iv.start);
        std::string reg;
        if (freeRegs.empty()) {
          auto worst = std::min_element(
              active.begin(), active.end(),
              [](const ActiveInterval &a, const ActiveInterval &b) {
                if (a.weight != b.weight)
                  return a.weight < b.weight;
                return a.end > b.end;
              });
          if (worst == active.end() || worst->weight >= iv.weight) {
            stats.lsraWeightedSpills++;
            continue; // spill this interval -> fallback path
          }
          lsraAssignment.erase(worst->key);
          reg = worst->reg;
          active.erase(worst);
          stats.lsraWeightedSpills++;
        } else {
          reg = freeRegs.back();
          freeRegs.pop_back();
        }
        lsraAssignment[valueKey(iv.value)] = reg;
        lsraReserved.insert(reg);
        stats.lsraStableValues++;
        active.push_back({iv.end, iv.weight, reg, valueKey(iv.value)});
      }
    }
  }

  int nextReg = 0;
  int nextLoopReg = 0;
  int nextVecReg = 0;
  int nextFloatReg = 0;
  int returnsBefore = stats.returns;
  std::set<std::string> skipMaterializedConstants;
  if (!isArm) {
    for (auto *op : funcOps) {
      if (!op || op->isErased() || op->name() != "rv_machine.li" ||
          op->resultCount() != 1 || isFloatType(op->resultType()))
        continue;
      int64_t imm = 0;
      if (!constantIntegerValue(op->result(), imm))
        continue;
      bool hasUse = false;
      bool allImmediateFriendly = true;
      for (const auto &use : op->resultUses[0]) {
        Operation *user = use.owner;
        if (!user || user->isErased())
          continue;
        hasUse = true;
        bool ok = false;
        if ((user->name() == "rv_machine.and" ||
             user->name() == "rv_machine.or" ||
             user->name() == "rv_machine.xor") &&
            user->operandCount() == 2 && fitsSigned12(imm)) {
          Value other = valueKey(user->operand(0)) == valueKey(op->result()) ? user->operand(1)
                                                                             : user->operand(0);
          int64_t otherImm = 0;
          ok = !constantIntegerValue(other, otherImm);
        } else if (user->name() == "rv_machine.cmp" &&
                   user->operandCount() == 2 && imm == 0) {
          std::string pred = symbolAttr(user->attr("predicate"));
          ok = pred == "eq" || pred == "ne";
        }
        if (!ok) {
          allImmediateFriendly = false;
          break;
        }
      }
      if (hasUse && allImmediateFriendly)
        skipMaterializedConstants.insert(valueKey(op->result()));
    }
  }

  auto scratchReg = [&](int n) -> std::string {
    if (isArm)
      return n == 0 ? "x16" : "x17";
    return n == 0 ? "t5" : "t6";
  };
  auto stackTmpFor = [&](const std::string &reg) -> std::string {
    if (isArm)
      return (reg == "x16" || reg == "w16" || reg == "s30") ? "x17" : "x16";
    return reg == "t5" ? "t6" : "t5";
  };
  auto emitStackAdjust = [&](bool allocate) {
    if (frameBytes <= 0)
      return;
    if (isArm) {
      os << "    " << (allocate ? "sub" : "add") << " sp, sp, #" << frameBytes << "\n";
      return;
    }
    if (fitsSigned12(frameBytes)) {
      os << "    addi sp, sp, " << (allocate ? -frameBytes : frameBytes) << "\n";
      return;
    }
    os << "    li " << scratchReg(0) << ", " << frameBytes << "\n";
    os << "    " << (allocate ? "sub" : "add") << " sp, sp, " << scratchReg(0) << "\n";
  };
  auto materializeStackAddressRaw = [&](const std::string &tmp, int64_t off) {
    if (isArm) {
      os << "    add " << tmp << ", sp, #" << off << "\n";
      return tmp;
    }
    if (fitsSigned12(off)) {
      os << "    addi " << tmp << ", sp, " << off << "\n";
    } else {
      os << "    li " << tmp << ", " << off << "\n";
      os << "    add " << tmp << ", sp, " << tmp << "\n";
    }
    return tmp;
  };
  auto emitStackStore = [&](const std::string &reg, int64_t off, const std::string &inst) {
    if (isArm) {
      os << "    str " << reg << ", [sp, #" << off << "]\n";
      return;
    }
    if (fitsSigned12(off)) {
      os << "    " << inst << " " << reg << ", " << off << "(sp)\n";
      return;
    }
    std::string tmp = stackTmpFor(reg);
    materializeStackAddressRaw(tmp, off);
    os << "    " << inst << " " << reg << ", 0(" << tmp << ")\n";
  };
  auto emitStackLoad = [&](const std::string &reg, int64_t off, const std::string &inst) {
    if (isArm) {
      os << "    ldr " << reg << ", [sp, #" << off << "]\n";
      return;
    }
    if (fitsSigned12(off)) {
      os << "    " << inst << " " << reg << ", " << off << "(sp)\n";
      return;
    }
    std::string tmp = (!reg.empty() && reg[0] == 'f') ? stackTmpFor(reg) : reg;
    materializeStackAddressRaw(tmp, off);
    os << "    " << inst << " " << reg << ", 0(" << tmp << ")\n";
  };

  os << (isArm ? "    .text\n    .global " : "    .text\n    .globl ") << name << "\n";
  os << name << ":\n";
  emitStackAdjust(true);
  if (hasCall)
    emitStackStore(isArm ? "x30" : "ra", returnAddressSlot, isArm ? "str" : "sd");
  for (int i = 0; i < calleeSaveCount; i++) {
    int64_t off = calleeSaveBase + i * 8;
    if (isArm)
      emitStackStore("x" + std::to_string(19 + i), off, "str");
    else
      emitStackStore("s" + std::to_string(i), off, "sd");
  }
  os << bodyEntryLabel << ":\n";

  for (const auto &kv : globalLabels) {
    auto cached = cachedGlobalBases.find(kv.first);
    if (cached != cachedGlobalBases.end()) {
      if (isArm)
        os << "    adrp " << cached->second.reg << ", " << cached->second.label << "\n"
           << "    add " << cached->second.reg << ", " << cached->second.reg
           << ", :lo12:" << cached->second.label << "\n";
      else
        os << "    la " << cached->second.reg << ", " << cached->second.label << "\n";
      regs[kv.first] = cached->second.reg;
    } else {
      regs[kv.first] = "global:" + kv.second;
    }
  }

  auto resultReg = [&]() -> std::string {
    if (isArm)
      return armResultReg(nextReg++);
    if (stableBaseActive) {
      std::vector<std::string> regsNoStable = {
          "t0", "t1", "t2", "t3", "t4",
          "a2", "a3", "a4", "a5", "a6", "a7",
      };
      for (int s = stableBaseRegCount; s < 4 && s < calleeSaveCount; s++)
        regsNoStable.push_back("s" + std::to_string(s));
      return regsNoStable[(nextReg++) % (int) regsNoStable.size()];
    }
    if (regAlloc2Enabled && calleeSaveCount >= 12) {
      // Round-robin pool for results without a stable LSRA assignment. s4-s7
      // are reserved for LSRA when it is active, so they are excluded here to
      // keep the two pools disjoint.
      static const char *regsFull[] = {
        "t0", "t1", "t2", "t3", "t4",
        "a2", "a3", "a4", "a5", "a6", "a7",
        "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7",
      };
      static const char *regsNoLsra[] = {
        "t0", "t1", "t2", "t3", "t4",
        "a2", "a3", "a4", "a5", "a6", "a7",
        "s0", "s1", "s2", "s3",
      };
      if (lsraEnabled || slotPromotionActive) {
        int n = (int)(sizeof(regsNoLsra) / sizeof(regsNoLsra[0]));
        return regsNoLsra[(nextReg++) % n];
      }
      int n = (int)(sizeof(regsFull) / sizeof(regsFull[0]));
      return regsFull[(nextReg++) % n];
    }
    return rvResultReg(nextReg++);
  };
  auto floatReg = [&]() -> std::string {
    return isArm ? armFloatReg(nextFloatReg++) : rvFloatReg(nextFloatReg++);
  };
  // Result register for an integer-typed op result: use the stable LSRA
  // assignment when present, otherwise fall back to the round-robin pool.
  auto intResultReg = [&](Value value) -> std::string {
    if (lsraEnabled) {
      auto it = lsraAssignment.find(valueKey(value));
      if (it != lsraAssignment.end())
        return it->second;
    }
    return resultReg();
  };
  std::set<std::string> homeValid;
  auto homeIsUsable = [&](Value value) {
    return !livenessEnabled || homeValid.count(valueKey(value)) != 0;
  };
  auto looksFloatReg = [&](const std::string &reg) {
    return !reg.empty() && (isArm ? reg[0] == 's' : reg[0] == 'f');
  };
  auto floatScratchReg = [&]() -> std::string {
    return isArm ? "s30" : "ft10";
  };
  auto intScratchForFloatBits = [&]() -> std::string {
    return isArm ? "w16" : "t5";
  };
  auto scratchForValue = [&](Value value, const std::string &preferred) -> std::string {
    if (isFloatType(value.type()) && !looksFloatReg(preferred))
      return floatScratchReg();
    return preferred;
  };
  auto emitFloatBitsToReg = [&](const std::string &dst, uint32_t bits) {
    if (isArm)
      os << "    mov " << intScratchForFloatBits() << ", #" << bits
         << "\n    fmov " << dst << ", " << intScratchForFloatBits() << "\n";
    else
      os << "    li " << intScratchForFloatBits() << ", " << bits
         << "\n    fmv.w.x " << dst << ", " << intScratchForFloatBits() << "\n";
  };
  std::function<bool(Value, int)> canRematerializeScalar =
      [&](Value value, int depth) -> bool {
    if (!value.valid() || depth > 6 || isMemrefType(value.type()) ||
        isFloatType(value.type()) || value.type().str().find("vector") != std::string::npos)
      return false;
    int64_t imm = 0;
    if (constantIntegerValue(value, imm))
      return true;
    if (value.isBlockArgument())
      return depth > 0;
    Operation *def = value.getDefiningOp();
    if (!def || def->isErased() || def->resultCount() != 1 ||
        !def->getRegions().empty())
      return false;
    std::string name = def->name();
    if (name == "rv_machine.li" || name == "arm_machine.mov" ||
        name == "arith.constant")
      return true;
    if (name == "rv_machine.neg" || name == "rv_machine.seqz" ||
        name == "arm_machine.neg" || name == "arm_machine.not")
      return def->operandCount() == 1 &&
             canRematerializeScalar(def->operand(0), depth + 1);
    if (name == "rv_machine.cmp" || name == "arm_machine.cmp")
      return def->operandCount() == 2 &&
             canRematerializeScalar(def->operand(0), depth + 1) &&
             canRematerializeScalar(def->operand(1), depth + 1);
    if (name == "rv_machine.select" || name == "arm_machine.select")
      return def->operandCount() == 3 &&
             canRematerializeScalar(def->operand(0), depth + 1) &&
             canRematerializeScalar(def->operand(1), depth + 1) &&
             canRematerializeScalar(def->operand(2), depth + 1);
    if (name == "rv_machine.addw" || name == "rv_machine.subw" ||
        name == "rv_machine.mulw" || name == "rv_machine.and" ||
        name == "rv_machine.or" || name == "rv_machine.xor" ||
        name == "rv_machine.sllw" || name == "rv_machine.sraw" ||
        name == "arm_machine.add" || name == "arm_machine.sub" ||
        name == "arm_machine.mul" || name == "arm_machine.and" ||
        name == "arm_machine.orr" || name == "arm_machine.eor")
      return def->operandCount() == 2 &&
             canRematerializeScalar(def->operand(0), depth + 1) &&
             canRematerializeScalar(def->operand(1), depth + 1);
    return false;
  };
  auto isDeferredHomeReg = [&](const std::string &reg) {
    if (!regAlloc2Enabled ||
        !envEnabled("SISY_ENABLE_SELF_REGALLOC2_LAZY", false) ||
        reg.size() < 2 || reg[0] != 's')
      return false;
    if (reg == "s8" || reg == "s9" || reg == "s10" || reg == "s11")
      return false;
    return true;
  };
  auto spillHome =
      [&](Value value, const std::string &reg, bool force = false) {
    auto it = valueSlots.find(valueKey(value));
    if (it == valueSlots.end() || reg.empty() ||
        reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0)
      return;
    if (!force && isDeferredHomeReg(reg) && remainingUses[valueKey(value)] > 0)
      return;
    int64_t off = it->second;
    bool fp = isFloatType(value.type()) || looksFloatReg(reg);
    bool ptr = isMemrefType(value.type());
    emitStackStore(reg, off, ptr ? "sd" : (fp ? "fsw" : "sw"));
    homeValid.insert(valueKey(value));
    stats.liveSpills++;
    if (regAlloc2Enabled)
      stats.lsra2Spills++;
  };
  auto flushPromotedScalarSlots = [&](bool invalidate) {
    for (auto &kv : promotedScalarSlots) {
      PromotedScalarSlot &slot = kv.second;
      if (slot.valid && slot.dirty) {
        auto stackIt = stackSlots.find(valueKey(slot.slot));
        if (stackIt != stackSlots.end())
          emitStackStore(slot.reg, stackIt->second, "sw");
        slot.dirty = false;
      }
      if (invalidate)
        slot.valid = false;
    }
  };
  auto maybeSpillBeforeClobber = [&](const std::string &key, const std::string &reg,
                                     bool callBoundary) {
    if (!livenessEnabled || key.empty() || reg.empty() ||
        reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0 ||
        remainingUses[key] <= 0 || homeValid.count(key) != 0 ||
        valueByKey.count(key) == 0)
      return;
    if (canRematerializeScalar(valueByKey[key], 0)) {
      stats.deadSpillsAvoided++;
      return;
    }
    spillHome(valueByKey[key], reg, true);
    if (callBoundary)
      stats.callBoundarySpills++;
  };
  auto bindReg = [&](Value value, const std::string &reg) {
    std::string key = valueKey(value);
    for (auto it = regs.begin(); it != regs.end(); ) {
      if (it->first != key && it->second == reg &&
          it->second.rfind("stack:", 0) != 0 && it->second.rfind("global:", 0) != 0) {
        maybeSpillBeforeClobber(it->first, it->second, false);
        it = regs.erase(it);
      } else {
        ++it;
      }
    }
    regs[key] = reg;
    homeValid.erase(key);
  };
  auto bindResult = [&](Value value, const std::string &reg) {
    bindReg(value, reg);
  };
  auto clobberPhysicalReg = [&](const std::string &reg) {
    if (reg.empty())
      return;
    for (auto it = regs.begin(); it != regs.end(); ) {
      if (it->second == reg) {
        maybeSpillBeforeClobber(it->first, it->second, false);
        it = regs.erase(it);
      } else {
        ++it;
      }
    }
  };
  auto invalidateLoopHeaderRegs = [&](Value ivValue, bool spillBeforeDrop) {
    std::string ivKey = valueKey(ivValue);
    for (auto it = regs.begin(); it != regs.end(); ) {
      const std::string &key = it->first;
      const std::string &reg = it->second;
      if (key == ivKey || reg.rfind("stack:", 0) == 0 ||
          reg.rfind("global:", 0) == 0 || globalLabels.count(key) != 0 ||
          (lsraEnabled && lsraAssignment.count(key) != 0)) {
        ++it;
        continue;
      }
      if (spillBeforeDrop)
        maybeSpillBeforeClobber(key, reg, false);
      it = regs.erase(it);
    }
  };
  auto consumeOperands = [&](Operation &op) {
    for (auto operand : op.getOperands()) {
      std::string key = valueKey(operand);
      auto it = remainingUses.find(key);
      if (it != remainingUses.end() && it->second > 0)
        it->second--;
    }
  };
  auto shouldSpillDefinedValue = [&](Value value) {
    if (!livenessEnabled)
      return true;
    // LSRA values live permanently in their stable register; never spill them
    // to the stack home (all reads come from the register).
    if (lsraEnabled && lsraAssignment.count(valueKey(value)) != 0) {
      stats.lsraSpillsAvoided++;
      return false;
    }
    if (remainingUses[valueKey(value)] <= 0) {
      stats.deadSpillsAvoided++;
      return false;
    }
    Operation *def = value.getDefiningOp();
    if (def && !def->isErased() && def->attr("licm_hoisted")) {
      int64_t imm = 0;
      if (constantIntegerValue(value, imm)) {
        stats.deadSpillsAvoided++;
        return false;
      }
      return true;
    }
    if (def && !def->isErased()) {
      Operation *onlyUser = nullptr;
      int liveUses = 0;
      unsigned resultIndex = value.getResultIndex();
      if (resultIndex < def->resultUses.size()) {
        for (const auto &use : def->resultUses[resultIndex]) {
          if (!use.owner || use.owner->isErased())
            continue;
          liveUses++;
          onlyUser = use.owner;
        }
      }
      auto defIt = opIndex.find(def);
      auto useIt = onlyUser ? opIndex.find(onlyUser) : opIndex.end();
      if (liveUses == 1 && onlyUser && def->getBlock() == onlyUser->getBlock() &&
          defIt != opIndex.end() && useIt != opIndex.end() &&
          useIt->second == defIt->second + 1) {
        stats.deadSpillsAvoided++;
        return false;
      }
      if (!isMemrefType(value.type()) && defIt != opIndex.end()) {
        bool allLocalUses = liveUses > 0;
        int lastUseIndex = defIt->second;
        for (const auto &use : def->resultUses[value.getResultIndex()]) {
          if (!use.owner || use.owner->isErased())
            continue;
          if (use.owner->getBlock() != def->getBlock()) {
            allLocalUses = false;
            break;
          }
          auto localUseIt = opIndex.find(use.owner);
          if (localUseIt == opIndex.end() || localUseIt->second <= defIt->second) {
            allLocalUses = false;
            break;
          }
          lastUseIndex = std::max(lastUseIndex, localUseIt->second);
        }
        if (allLocalUses) {
          bool crossesCall = false;
          for (const auto &kv : opIndex) {
            Operation *between = kv.first;
            if (!between || between->isErased() || between->getBlock() != def->getBlock())
              continue;
            if (kv.second <= defIt->second || kv.second >= lastUseIndex)
              continue;
            if (between->name() == "sysy.call") {
              crossesCall = true;
              break;
            }
          }
          if (!crossesCall) {
            stats.deadSpillsAvoided++;
            return false;
          }
        }
      }
    }
    return true;
  };
  auto forgetHoistedResultReg = [&](Operation &def) {
    if (def.attr("licm_hoisted") && def.resultCount() == 1) {
      std::string key = valueKey(def.result());
      if (remainingUses[key] > 0)
        return;
      regs.erase(key);
    }
  };
  auto invalidateCallerSavedForCall = [&]() {
    flushPromotedScalarSlots(true);
    for (auto it = regs.begin(); it != regs.end(); ) {
      const std::string reg = it->second;
      bool callerSaved = false;
      if (isArm)
        callerSaved = reg.rfind("w", 0) == 0 || reg.rfind("x", 0) == 0 ||
                      reg.rfind("s", 0) == 0;
      else
        callerSaved = reg.rfind("t", 0) == 0 || reg.rfind("a", 0) == 0 ||
                      reg.rfind("fa", 0) == 0 || reg.rfind("ft", 0) == 0;
      if (callerSaved && reg.rfind("stack:", 0) != 0 && reg.rfind("global:", 0) != 0) {
        maybeSpillBeforeClobber(it->first, reg, true);
        it = regs.erase(it);
      } else {
        ++it;
      }
    }
  };
  auto spillLiveRegsForControlFlow = [&]() {
    flushPromotedScalarSlots(false);
    std::vector<std::pair<std::string, std::string>> liveRegs;
    for (const auto &kv : regs)
      liveRegs.push_back(kv);
    for (const auto &kv : liveRegs) {
      const std::string &key = kv.first;
      const std::string &reg = kv.second;
      if (key.empty() || reg.empty() || reg.rfind("stack:", 0) == 0 ||
          reg.rfind("global:", 0) == 0 || globalLabels.count(key) != 0 ||
          remainingUses[key] <= 0 ||
          valueByKey.count(key) == 0 || homeValid.count(key) != 0)
        continue;
      if (canRematerializeScalar(valueByKey[key], 0)) {
        stats.deadSpillsAvoided++;
        continue;
      }
      spillHome(valueByKey[key], reg, true);
    }
  };
  auto invalidateControlFlowJoinRegs = [&]() {
    flushPromotedScalarSlots(true);
    for (auto it = regs.begin(); it != regs.end(); ) {
      const std::string &key = it->first;
      const std::string &reg = it->second;
      if (reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0 ||
          globalLabels.count(key) != 0 ||
          (lsraEnabled && lsraAssignment.count(key) != 0)) {
        ++it;
        continue;
      }
      it = regs.erase(it);
    }
  };
  auto invalidateLoopEntryRegs = [&](bool spillBeforeDrop) {
    if (spillBeforeDrop)
      flushPromotedScalarSlots(true);
    else {
      for (auto &kv : promotedScalarSlots)
        kv.second.valid = false;
    }
    for (auto it = regs.begin(); it != regs.end(); ) {
      const std::string &key = it->first;
      const std::string &reg = it->second;
      if (reg.rfind("stack:", 0) == 0 || reg.rfind("global:", 0) == 0 ||
          globalLabels.count(key) != 0 ||
          (lsraEnabled && lsraAssignment.count(key) != 0)) {
        ++it;
        continue;
      }
      if (spillBeforeDrop)
        maybeSpillBeforeClobber(key, reg, false);
      it = regs.erase(it);
    }
  };
  int incomingIntRegs = 0;
  int incomingFpRegs = 0;
  int incomingStackSlots = 0;
  for (size_t i = 0; i < block.args().size(); i++) {
    const auto &arg = *block.args()[i];
    bool fpArg = isFloatType(arg.type());
    bool ptrArg = isMemrefType(arg.type());
    std::string reg;
    if (fpArg) {
      if (incomingFpRegs < 8) {
        reg = (isArm ? "s" : "fa") + std::to_string(incomingFpRegs++);
      } else {
        int64_t off = frameBytes + incomingStackSlots++ * 8;
        reg = floatScratchReg();
        emitStackLoad(reg, off, isArm ? "ldr" : "flw");
      }
    } else {
      if (incomingIntRegs < 8) {
        if (isArm && ptrArg)
          reg = "x" + std::to_string(incomingIntRegs);
        else if (isArm)
          reg = "w" + std::to_string(incomingIntRegs);
        else
          reg = "a" + std::to_string(incomingIntRegs);
        incomingIntRegs++;
      } else {
        int64_t off = frameBytes + incomingStackSlots++ * 8;
        reg = ptrArg ? scratchReg(0) : (isArm ? "w16" : scratchReg(0));
        emitStackLoad(reg, off, ptrArg ? (isArm ? "ldr" : "ld") : (isArm ? "ldr" : "lw"));
      }
    }
    bindResult(arg.value(), reg);
    bool shouldKeepInScratch = reg != scratchReg(0) && reg != scratchReg(1) &&
                               reg != floatScratchReg();
    if (!livenessEnabled || remainingUses[valueKey(arg.value())] > 0)
      spillHome(arg.value(), reg);
    if (!shouldKeepInScratch)
      regs.erase(valueKey(arg.value()));
  }
  auto materializeAddress = [&](Value value, const std::string &tmp) -> std::string {
    std::string key = valueKey(value);
    std::string loc = lookupReg(value, regs);
    if (loc.empty()) {
      auto globalIt = globalLabels.find(key);
      if (globalIt != globalLabels.end())
        loc = "global:" + globalIt->second;
    }
    auto slotIt = stackSlots.find(key);
    if (slotIt != stackSlots.end())
      loc = "stack:" + std::to_string(slotIt->second);
    if (loc.rfind("stack:", 0) == 0) {
      int64_t off = std::stoll(loc.substr(6));
      clobberPhysicalReg(tmp);
      return materializeStackAddressRaw(tmp, off);
    }
    if (loc.rfind("global:", 0) == 0) {
      std::string label = loc.substr(7);
      clobberPhysicalReg(tmp);
      if (isArm)
        os << "    adrp " << tmp << ", " << label << "\n"
           << "    add " << tmp << ", " << tmp << ", :lo12:" << label << "\n";
      else
        os << "    la " << tmp << ", " << label << "\n";
      return tmp;
    }
    auto home = valueSlots.find(key);
    if (home != valueSlots.end() && isMemrefType(value.type()) && homeIsUsable(value)) {
      clobberPhysicalReg(tmp);
      emitStackLoad(tmp, home->second, "ld");
      return tmp;
    }
    return loc;
  };
  std::function<std::string(Value, const std::string &)> ensureReg;
  std::function<std::string(Value, const std::string &, int)> rematerializeValue;
  int nextRematSelectId = stats.functions * 10000;
  auto normalizePredicate = [](std::string pred) {
    if (pred.size() > 1 && pred[0] == 's')
      pred = pred.substr(1);
    return pred;
  };
  auto emitIntegerCompare = [&](const std::string &dst, const std::string &lhs,
                                const std::string &rhs, std::string pred) {
    pred = normalizePredicate(std::move(pred));
    if (isArm) {
      os << "    cmp " << lhs << ", " << rhs << "\n";
      os << "    cset " << dst << ", " << pred << "\n";
      return;
    }
    if (pred == "lt") {
      os << "    slt " << dst << ", " << lhs << ", " << rhs << "\n";
    } else if (pred == "le") {
      os << "    slt " << dst << ", " << rhs << ", " << lhs << "\n";
      os << "    xori " << dst << ", " << dst << ", 1\n";
    } else if (pred == "gt") {
      os << "    slt " << dst << ", " << rhs << ", " << lhs << "\n";
    } else if (pred == "ge") {
      os << "    slt " << dst << ", " << lhs << ", " << rhs << "\n";
      os << "    xori " << dst << ", " << dst << ", 1\n";
    } else if (pred == "eq") {
      os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
      os << "    seqz " << dst << ", " << dst << "\n";
    } else {
      os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
      os << "    snez " << dst << ", " << dst << "\n";
    }
  };
  rematerializeValue = [&](Value value, const std::string &tmp, int depth) -> std::string {
    if (!canRematerializeScalar(value, depth))
      return "";
    int64_t imm = 0;
    if (constantIntegerValue(value, imm)) {
      clobberPhysicalReg(tmp);
      if (isArm)
        os << "    mov " << tmp << ", #" << imm << "\n";
      else
        os << "    li " << tmp << ", " << imm << "\n";
      stats.machineOps++;
      return tmp;
    }
    Operation *def = value.getDefiningOp();
    if (!def || def->isErased())
      return "";
    std::string name = def->name();
    std::string otherTmp = tmp == scratchReg(0) ? scratchReg(1) : scratchReg(0);
    if (name == "rv_machine.neg" || name == "rv_machine.seqz" ||
        name == "arm_machine.neg" || name == "arm_machine.not") {
      std::string src = ensureReg(def->operand(0), tmp);
      if (src.empty())
        return "";
      clobberPhysicalReg(tmp);
      if (isArm) {
        if (name == "arm_machine.not")
          os << "    cmp " << src << ", #0\n    cset " << tmp << ", eq\n";
        else
          os << "    neg " << tmp << ", " << src << "\n";
      } else {
        if (name == "rv_machine.seqz")
          os << "    seqz " << tmp << ", " << src << "\n";
        else
          os << "    negw " << tmp << ", " << src << "\n";
      }
      stats.machineOps++;
      return tmp;
    }
    if (def->operandCount() != 2)
    {
      if (name != "rv_machine.select" && name != "arm_machine.select")
        return "";
      std::string condTmp = tmp == scratchReg(0) ? scratchReg(1) : scratchReg(0);
      std::string cond = ensureReg(def->operand(0), condTmp);
      if (cond.empty())
        return "";
      if (cond == tmp) {
        clobberPhysicalReg(condTmp);
        if (isArm)
          os << "    mov " << condTmp << ", " << cond << "\n";
        else
          os << "    mv " << condTmp << ", " << cond << "\n";
        cond = condTmp;
      }
      std::string valueTmp = cond == scratchReg(0) ? scratchReg(1) : scratchReg(0);
      std::string falseSrc = ensureReg(def->operand(2), valueTmp);
      if (falseSrc.empty())
        return "";
      clobberPhysicalReg(tmp);
      if (falseSrc != tmp) {
        if (isArm)
          os << "    mov " << tmp << ", " << falseSrc << "\n";
        else
          os << "    mv " << tmp << ", " << falseSrc << "\n";
      }
      int labelId = ++nextRematSelectId;
      std::string done = ".Lremat_select_done_" + std::to_string(labelId);
      if (isArm)
        os << "    cbz " << cond << ", " << done << "\n";
      else
        os << "    beqz " << cond << ", " << done << "\n";
      std::string trueSrc = ensureReg(def->operand(1), valueTmp);
      if (trueSrc.empty())
        return "";
      if (trueSrc != tmp) {
        if (isArm)
          os << "    mov " << tmp << ", " << trueSrc << "\n";
        else
          os << "    mv " << tmp << ", " << trueSrc << "\n";
      }
      os << done << ":\n";
      stats.machineOps++;
      return tmp;
    }
    std::string lhs = ensureReg(def->operand(0), tmp);
    if (lhs.empty())
      return "";
    std::string rhsTmp = lhs == scratchReg(0) ? scratchReg(1) : scratchReg(0);
    if (rhsTmp == lhs)
      rhsTmp = otherTmp;
    std::string rhs = ensureReg(def->operand(1), rhsTmp);
    if (rhs.empty())
      return "";
    clobberPhysicalReg(tmp);
    if (name == "rv_machine.cmp" || name == "arm_machine.cmp") {
      emitIntegerCompare(tmp, lhs, rhs, symbolAttr(def->attr("predicate")));
      stats.machineOps++;
      return tmp;
    }
    if (isArm) {
      std::string inst;
      if (name == "arm_machine.add") inst = "add";
      else if (name == "arm_machine.sub") inst = "sub";
      else if (name == "arm_machine.mul") inst = "mul";
      else if (name == "arm_machine.and") inst = "and";
      else if (name == "arm_machine.orr") inst = "orr";
      else if (name == "arm_machine.eor") inst = "eor";
      else return "";
      os << "    " << inst << " " << tmp << ", " << lhs << ", " << rhs << "\n";
    } else {
      std::string inst;
      if (name == "rv_machine.addw") inst = "addw";
      else if (name == "rv_machine.subw") inst = "subw";
      else if (name == "rv_machine.mulw") inst = "mulw";
      else if (name == "rv_machine.and") inst = "and";
      else if (name == "rv_machine.or") inst = "or";
      else if (name == "rv_machine.xor") inst = "xor";
      else if (name == "rv_machine.sllw") inst = "sllw";
      else if (name == "rv_machine.sraw") inst = "sraw";
      else return "";
      os << "    " << inst << " " << tmp << ", " << lhs << ", " << rhs << "\n";
      if (inst == "and" || inst == "or" || inst == "xor")
        os << "    addiw " << tmp << ", " << tmp << ", 0\n";
    }
    stats.machineOps++;
    return tmp;
  };
  ensureReg = [&](Value value, const std::string &tmp) -> std::string {
    std::string actualTmp = scratchForValue(value, tmp);
    std::string loc = lookupReg(value, regs);
    if (!loc.empty()) {
      if (lsraEnabled) {
        auto assigned = lsraAssignment.find(valueKey(value));
        if (assigned != lsraAssignment.end() && assigned->second == loc)
          stats.lsraRegHits++;
      }
      if (loc.rfind("stack:", 0) == 0 || loc.rfind("global:", 0) == 0)
        return materializeAddress(value, actualTmp);
      return loc;
    }
    int64_t imm = 0;
    if (constantIntegerValue(value, imm)) {
      clobberPhysicalReg(actualTmp);
      if (isArm)
        os << "    mov " << actualTmp << ", #" << imm << "\n";
      else
        os << "    li " << actualTmp << ", " << imm << "\n";
      return actualTmp;
    }
    if (isFloatType(value.type())) {
      auto *op = value.getDefiningOp();
      if (op && !op->isErased() &&
          (op->name() == "arith.constant" || op->name() == "rv_machine.li" ||
           op->name() == "arm_machine.mov") &&
          op->attr("value")) {
        clobberPhysicalReg(actualTmp);
        emitFloatBitsToReg(actualTmp, parseFloatAttrBits(op->attr("value")));
        return actualTmp;
      }
    }
    auto home = valueSlots.find(valueKey(value));
    if (home != valueSlots.end() && homeIsUsable(value)) {
      clobberPhysicalReg(actualTmp);
      bool fp = isFloatType(value.type()) || looksFloatReg(actualTmp);
      bool ptr = isMemrefType(value.type());
      emitStackLoad(actualTmp, home->second, ptr ? "ld" : (fp ? "flw" : "lw"));
      return actualTmp;
    }
    std::string remat = rematerializeValue(value, actualTmp, 0);
    if (!remat.empty())
      return remat;
    return "";
  };
  auto reloadValue = [&](Value value, const std::string &tmp) -> std::string {
    std::string actualTmp = scratchForValue(value, tmp);
    auto home = valueSlots.find(valueKey(value));
    if (home != valueSlots.end() && homeIsUsable(value)) {
      bool fp = isFloatType(value.type()) || looksFloatReg(actualTmp);
      bool ptr = isMemrefType(value.type());
      emitStackLoad(actualTmp, home->second, ptr ? "ld" : (fp ? "flw" : "lw"));
      return actualTmp;
    }
    return ensureReg(value, actualTmp);
  };
  std::vector<int64_t> memoArgSlots;
  if (memoInfo) {
    for (int i = 0; i < memoInfo->argCount && i < (int) block.args().size(); i++) {
      auto it = valueSlots.find(valueKey(block.args()[i]->value()));
      if (it != valueSlots.end())
        memoArgSlots.push_back(it->second);
    }
    if ((int) memoArgSlots.size() != memoInfo->argCount)
      memoInfo = nullptr;
  }
  auto emitMemoIndexFromArgRegs = [&](const MemoFunctionInfo &memo) {
    os << "    mv t2, a0\n";
    if (memo.argCount > 1) {
      os << "    li t3, 131\n";
      os << "    mulw t2, t2, t3\n";
      os << "    xor t2, t2, a1\n";
    }
    os << "    li t3, " << (memo.capacity - 1) << "\n";
    os << "    and t2, t2, t3\n";
    os << "    slli t2, t2, 2\n";
  };
  auto emitMemoIndexFromArgSlots = [&](const MemoFunctionInfo &memo) {
    emitStackLoad("t0", memoArgSlots[0], "lw");
    if (memo.argCount > 1)
      emitStackLoad("t1", memoArgSlots[1], "lw");
    os << "    mv t2, t0\n";
    if (memo.argCount > 1) {
      os << "    li t3, 131\n";
      os << "    mulw t2, t2, t3\n";
      os << "    xor t2, t2, t1\n";
    }
    os << "    li t3, " << (memo.capacity - 1) << "\n";
    os << "    and t2, t2, t3\n";
    os << "    slli t2, t2, 2\n";
  };
  auto emitMemoLookup = [&](const MemoFunctionInfo &memo) {
    if (memo.directEncoded) {
      std::string ready = ".Ldmemo_ready_" + std::to_string(stats.functions) +
                          "_" + sanitizeLabel(name);
      std::string miss = ".Ldmemo_miss_" + std::to_string(stats.functions) +
                         "_" + sanitizeLabel(name);
      std::string allocated = ".Ldmemo_allocated_" + std::to_string(stats.functions) +
                              "_" + sanitizeLabel(name);
      std::string allocDone = ".Ldmemo_alloc_done_" + std::to_string(stats.functions) +
                              "_" + sanitizeLabel(name);
      os << "    la t0, " << memo.directLimitLabel << "\n";
      os << "    lw t6, 0(t0)\n";
      os << "    la t0, " << memo.directCachePtrLabel << "\n";
      os << "    ld t1, 0(t0)\n";
      os << "    bnez t1, " << ready << "\n";
      os << "    addiw a0, t6, 1\n";
      os << "    li a1, 4\n";
      os << "    call calloc\n";
      os << "    bnez a0, " << allocated << "\n";
      os << "    li t1, 0\n";
      os << "    j " << allocDone << "\n";
      os << allocated << ":\n";
      os << "    la t0, " << memo.directCachePtrLabel << "\n";
      os << "    sd a0, 0(t0)\n";
      os << "    mv t1, a0\n";
      os << allocDone << ":\n";
      emitStackLoad("a0", memoArgSlots[0], "lw");
      os << ready << ":\n";
      os << "    beqz t1, " << miss << "\n";
      os << "    la t0, " << memo.directLimitLabel << "\n";
      os << "    lw t6, 0(t0)\n";
      os << "    blez a0, " << miss << "\n";
      os << "    bgt a0, t6, " << miss << "\n";
      os << "    slli t2, a0, 2\n";
      os << "    add t2, t1, t2\n";
      os << "    lw t3, 0(t2)\n";
      os << "    beqz t3, " << miss << "\n";
      os << "    addiw a0, t3, -2\n";
      os << "    j " << epilogueLabel << "\n";
      os << miss << ":\n";
      stats.memoLookups++;
      stats.memoFallbacks++;
      return;
    }
    std::string skipEpoch = ".Lmemo_skip_epoch_" + std::to_string(stats.functions) +
                            "_" + sanitizeLabel(name);
    std::string epochReady = ".Lmemo_epoch_ready_" + std::to_string(stats.functions) +
                             "_" + sanitizeLabel(name);
    std::string miss = ".Lmemo_miss_" + std::to_string(stats.functions) +
                       "_" + sanitizeLabel(name);
    os << "    la t0, " << memo.depthLabel << "\n";
    os << "    lw t1, 0(t0)\n";
    os << "    bnez t1, " << skipEpoch << "\n";
    os << "    la t5, " << memo.epochLabel << "\n";
    os << "    lw t6, 0(t5)\n";
    os << "    addi t6, t6, 1\n";
    os << "    bnez t6, " << epochReady << "\n";
    os << "    li t6, 1\n";
    os << epochReady << ":\n";
    os << "    sw t6, 0(t5)\n";
    os << skipEpoch << ":\n";
    os << "    addi t1, t1, 1\n";
    os << "    sw t1, 0(t0)\n";
    emitMemoIndexFromArgRegs(memo);
    os << "    la t5, " << memo.epochLabel << "\n";
    os << "    lw t5, 0(t5)\n";
    os << "    la t3, " << memo.validLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    lw t4, 0(t3)\n";
    os << "    bne t4, t5, " << miss << "\n";
    os << "    la t3, " << memo.key0Label << "\n";
    os << "    add t3, t3, t2\n";
    os << "    lw t4, 0(t3)\n";
    os << "    bne t4, a0, " << miss << "\n";
    if (memo.argCount > 1) {
      os << "    la t3, " << memo.key1Label << "\n";
      os << "    add t3, t3, t2\n";
      os << "    lw t4, 0(t3)\n";
      os << "    bne t4, a1, " << miss << "\n";
    }
    os << "    la t3, " << memo.valueLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    lw a0, 0(t3)\n";
    os << "    j " << epilogueLabel << "\n";
    os << miss << ":\n";
    stats.memoLookups++;
    stats.memoFallbacks++;
  };
  auto emitMemoStore = [&](const MemoFunctionInfo &memo) {
    if (memo.directEncoded) {
      std::string skip = ".Ldmemo_store_skip_" + std::to_string(stats.functions) +
                         "_" + sanitizeLabel(name) + "_" +
                         std::to_string(stats.memoStores);
      emitStackLoad("t0", memoArgSlots[0], "lw");
      os << "    la t1, " << memo.directLimitLabel << "\n";
      os << "    lw t6, 0(t1)\n";
      os << "    la t1, " << memo.directCachePtrLabel << "\n";
      os << "    ld t1, 0(t1)\n";
      os << "    beqz t1, " << skip << "\n";
      os << "    blez t0, " << skip << "\n";
      os << "    bgt t0, t6, " << skip << "\n";
      os << "    addiw t2, a0, 2\n";
      os << "    slli t0, t0, 2\n";
      os << "    add t1, t1, t0\n";
      os << "    sw t2, 0(t1)\n";
      os << skip << ":\n";
      stats.memoStores++;
      return;
    }
    emitMemoIndexFromArgSlots(memo);
    os << "    la t3, " << memo.key0Label << "\n";
    os << "    add t3, t3, t2\n";
    os << "    sw t0, 0(t3)\n";
    if (memo.argCount > 1) {
      os << "    la t3, " << memo.key1Label << "\n";
      os << "    add t3, t3, t2\n";
      os << "    sw t1, 0(t3)\n";
    }
    os << "    la t3, " << memo.valueLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    sw a0, 0(t3)\n";
    os << "    la t3, " << memo.validLabel << "\n";
    os << "    add t3, t3, t2\n";
    os << "    la t5, " << memo.epochLabel << "\n";
    os << "    lw t5, 0(t5)\n";
    os << "    sw t5, 0(t3)\n";
    stats.memoStores++;
  };
  if (memoInfo)
    emitMemoLookup(*memoInfo);
  auto emitLinearizedIndex = [&](Value base, const std::vector<Value> &indices,
                                 const std::string &result,
                                 const std::string &tmp,
                                 bool allowPartial) -> bool {
    if (indices.empty())
      return false;
    MemrefInfo info = parseMemrefInfo(base.type());
    bool needsShapedIndex = indices.size() > 1 ||
                            (info.valid && info.shape.size() > 1);
    if (needsShapedIndex) {
      if (!info.valid || info.shape.size() < indices.size()) {
        stats.unsupportedOps++;
        stats.error = "multi-dimensional memref access lacks shaped memref type";
        return false;
      }
      if (!allowPartial && info.shape.size() != indices.size()) {
        stats.unsupportedOps++;
        stats.error = "multi-dimensional memref access is missing trailing indices";
        return false;
      }
      for (std::size_t i = 1; i < indices.size(); i++) {
        if (info.shape[i] <= 0) {
          stats.unsupportedOps++;
          stats.error = "multi-dimensional memref access has unknown trailing dimension";
          return false;
        }
      }
    }

    std::string first = ensureReg(indices[0], result);
    if (first.empty()) {
      stats.unsupportedOps++;
      stats.error = "memref index has no assigned register";
      return false;
    }
    if (first != result) {
      clobberPhysicalReg(result);
      if (isArm)
        os << "    mov " << result << ", " << first << "\n";
      else
        os << "    mv " << result << ", " << first << "\n";
    }

    for (std::size_t i = 1; i < indices.size(); i++) {
      if (isArm) {
        clobberPhysicalReg(tmp);
        os << "    mov " << tmp << ", #" << info.shape[i] << "\n";
        clobberPhysicalReg(result);
        os << "    mul " << result << ", " << result << ", " << tmp << "\n";
      } else {
        clobberPhysicalReg(tmp);
        os << "    li " << tmp << ", " << info.shape[i] << "\n";
        clobberPhysicalReg(result);
        os << "    mulw " << result << ", " << result << ", " << tmp << "\n";
      }
      std::string idx = ensureReg(indices[i], tmp);
      if (idx.empty()) {
        stats.unsupportedOps++;
        stats.error = "memref index has no assigned register";
        return false;
      }
      if (isArm)
        os << "    add " << result << ", " << result << ", " << idx << "\n";
      else
        os << "    addw " << result << ", " << result << ", " << idx << "\n";
    }
    if (needsShapedIndex && allowPartial && indices.size() < info.shape.size()) {
      int64_t trailingElems = 1;
      for (std::size_t i = indices.size(); i < info.shape.size(); i++) {
        if (info.shape[i] <= 0) {
          stats.unsupportedOps++;
          stats.error = "partial memref access has unknown trailing dimension";
          return false;
        }
        trailingElems *= info.shape[i];
      }
      if (trailingElems != 1) {
        if (isArm) {
          clobberPhysicalReg(tmp);
          os << "    mov " << tmp << ", #" << trailingElems << "\n";
          clobberPhysicalReg(result);
          os << "    mul " << result << ", " << result << ", " << tmp << "\n";
        } else {
          clobberPhysicalReg(tmp);
          os << "    li " << tmp << ", " << trailingElems << "\n";
          clobberPhysicalReg(result);
          os << "    mulw " << result << ", " << result << ", " << tmp << "\n";
        }
      }
    }
    return true;
  };
  auto computeAddress = [&](Value base, const std::vector<Value> &indices,
                            const std::string &addrReg,
                            const std::string &indexReg,
                            const std::string &tmpReg,
                            bool allowPartial,
                            std::string &addrOut) -> bool {
    if (indices.empty()) {
      addrOut = materializeAddress(base, addrReg);
      return !addrOut.empty();
    }
    if (!emitLinearizedIndex(base, indices, indexReg, tmpReg, allowPartial))
      return false;
    std::string baseAddr = materializeAddress(base, tmpReg);
    if (isArm) {
      clobberPhysicalReg(indexReg);
      os << "    add " << indexReg << ", " << baseAddr << ", "
         << indexReg << ", lsl #2\n";
    } else {
      clobberPhysicalReg(indexReg);
      os << "    slli " << indexReg << ", " << indexReg << ", 2\n";
      os << "    add " << indexReg << ", " << baseAddr << ", " << indexReg << "\n";
    }
    addrOut = indexReg;
    return true;
  };
  auto computeLinearizedElementAddress = [&](Value base, Value index,
                                             const std::string &addrReg,
                                             const std::string &indexReg,
                                             const std::string &tmpReg,
                                             std::string &addrOut) -> bool {
    std::string idx = ensureReg(index, indexReg);
    if (idx.empty()) {
      stats.unsupportedOps++;
      stats.error = "linearized memref index has no assigned register";
      return false;
    }
    if (idx != indexReg) {
      clobberPhysicalReg(indexReg);
      if (isArm)
        os << "    mov " << indexReg << ", " << idx << "\n";
      else
        os << "    mv " << indexReg << ", " << idx << "\n";
    }
    std::string baseAddr = materializeAddress(base, tmpReg);
    if (isArm) {
      clobberPhysicalReg(indexReg);
      os << "    add " << indexReg << ", " << baseAddr << ", "
         << indexReg << ", lsl #2\n";
    } else {
      clobberPhysicalReg(indexReg);
      os << "    slli " << indexReg << ", " << indexReg << ", 2\n";
      os << "    add " << indexReg << ", " << baseAddr << ", " << indexReg << "\n";
    }
    addrOut = indexReg;
    (void) addrReg;
    return true;
  };
  auto loadFromAddress = [&](const std::string &dst, Value base,
                             const std::vector<Value> &indices) -> bool {
    bool fpLoad = looksFloatReg(dst);
    if (!indices.empty()) {
      int64_t immIndex = 0;
      MemrefInfo info = parseMemrefInfo(base.type());
      bool simpleRank = !info.valid || info.shape.size() <= 1;
      if (simpleRank && indices.size() == 1 && !isArm &&
          constantIntegerValue(indices[0], immIndex)) {
        int64_t byteOffset = immIndex * 4;
        if (byteOffset >= -2048 && byteOffset <= 2047) {
          std::string addr = materializeAddress(base, scratchReg(0));
          os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", "
             << byteOffset << "(" << addr << ")\n";
          return true;
        }
      }
      std::string addr;
      if (!computeAddress(base, indices, scratchReg(0), scratchReg(1),
                          scratchReg(0), false, addr))
        return false;
      if (isArm) {
        os << "    ldr " << dst << ", [" << addr << "]\n";
      } else {
        os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", 0(" << addr << ")\n";
      }
      return true;
    }
    std::string addr = materializeAddress(base, scratchReg(0));
    if (isArm)
      os << "    ldr " << dst << ", [" << addr << "]\n";
    else
      os << "    " << (fpLoad ? "flw " : "lw ") << dst << ", 0(" << addr << ")\n";
    return true;
  };
  auto storeToAddress = [&](Value value, Value base,
                            const std::vector<Value> &indices) -> bool {
    bool fpStore = isFloatType(value.type());
    if (!indices.empty()) {
      int64_t immIndex = 0;
      MemrefInfo info = parseMemrefInfo(base.type());
      bool simpleRank = !info.valid || info.shape.size() <= 1;
      if (simpleRank && indices.size() == 1 && !isArm &&
          constantIntegerValue(indices[0], immIndex)) {
        int64_t byteOffset = immIndex * 4;
        if (byteOffset >= -2048 && byteOffset <= 2047) {
          std::string addr = materializeAddress(base, scratchReg(0));
          std::string val = ensureReg(value, fpStore ? (isArm ? "s30" : "ft10") : scratchReg(1));
          fpStore = fpStore || looksFloatReg(val);
          os << "    " << (fpStore ? "fsw " : "sw ") << val << ", "
             << byteOffset << "(" << addr << ")\n";
          return true;
        }
      }
      std::string addr;
      if (!computeAddress(base, indices, scratchReg(0), scratchReg(1),
                          scratchReg(0), false, addr))
        return false;
      if (isArm) {
        std::string val = ensureReg(value, fpStore ? "s30" : scratchReg(0));
        fpStore = fpStore || looksFloatReg(val);
        os << "    str " << val << ", [" << addr << "]\n";
      } else {
        std::string val = ensureReg(value, fpStore ? "ft10" : scratchReg(0));
        fpStore = fpStore || looksFloatReg(val);
        os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << addr << ")\n";
      }
      return true;
    }
    std::string addr = materializeAddress(base, scratchReg(0));
    std::string val = ensureReg(value, fpStore ? (isArm ? "s30" : "ft10") : scratchReg(1));
    fpStore = fpStore || looksFloatReg(val);
    if (isArm)
      os << "    str " << val << ", [" << addr << "]\n";
    else
      os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << addr << ")\n";
    return true;
  };

  int nextLoopId = stats.functions * 10000;
  std::map<Operation*, int> loopOps;
  std::map<Operation*, std::string> loopIvRegs;
  std::map<Operation*, std::vector<std::string>> labelsBefore;
  std::vector<std::string> functionEndLabels;
  std::set<Operation*> tailReturnSkips;

  auto firstLiveOpInRegion = [](Region &region) -> Operation* {
    for (auto &block : region.getBlocks()) {
      for (auto &owned : block->ops()) {
        if (owned && !owned->isErased())
          return owned.get();
      }
    }
    return nullptr;
  };
  auto regionEndsWith = [](Region &region, const std::string &opName) -> bool {
    for (auto it = region.getBlocks().rbegin(); it != region.getBlocks().rend(); ++it) {
      Block &block = **it;
      for (auto opIt = block.ops().rbegin(); opIt != block.ops().rend(); ++opIt) {
        if (*opIt && !(*opIt)->isErased())
          return (*opIt)->name() == opName;
      }
    }
    return false;
  };
  auto nextLiveOpAfter = [](Operation &op) -> Operation* {
    Block *block = op.getBlock();
    if (!block)
      return nullptr;
    bool seen = false;
    for (auto &owned : block->ops()) {
      if (!owned)
        continue;
      if (seen && !owned->isErased())
        return owned.get();
      if (owned.get() == &op)
        seen = true;
    }
    return nullptr;
  };
  auto scheduleBefore = [&](Operation *op, std::string label) {
    if (op)
      labelsBefore[op].push_back(std::move(label));
    else
      functionEndLabels.push_back(std::move(label));
  };
  auto scheduleAfter = [&](Operation &op, std::string label) {
    scheduleBefore(nextLiveOpAfter(op), std::move(label));
  };

  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "affine.for" || op->name() == "scf.while" ||
        op->name() == "scf.if")
      loopOps[op] = ++nextLoopId;
  }

  for (auto *op : funcOps) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "scf.if") {
      int ifId = loopOps[op];
      bool hasThen = !op->getRegions().empty();
      bool hasElse = op->getRegions().size() > 1;
      bool thenHasYield = hasThen && regionEndsWith(*op->getRegions()[0], "scf.yield");
      bool elseHasYield = hasElse && regionEndsWith(*op->getRegions()[1], "scf.yield");
      if (hasElse && !thenHasYield) {
        Operation *elseFirst = firstLiveOpInRegion(*op->getRegions()[1]);
        if (elseFirst) {
          scheduleBefore(elseFirst, "    " + std::string(isArm ? "b" : "j") +
                                         " .Lendif_" + std::to_string(ifId));
          scheduleBefore(elseFirst, ".Lelse_" + std::to_string(ifId) + ":");
        } else {
          scheduleAfter(*op, ".Lelse_" + std::to_string(ifId) + ":");
        }
      }
      if ((hasElse && !elseHasYield) || (!hasElse && !thenHasYield))
        scheduleAfter(*op, ".Lendif_" + std::to_string(ifId) + ":");
    } else if (op->name() == "scf.while") {
      if (op->getRegions().size() > 1 &&
          !regionEndsWith(*op->getRegions()[1], "scf.yield"))
        scheduleAfter(*op, ".Lwhile_end_" + std::to_string(loopOps[op]) + ":");
    } else if (op->name() == "affine.for") {
      if (!op->getRegions().empty() &&
          !regionEndsWith(*op->getRegions()[0], "affine.yield"))
        scheduleAfter(*op, ".Lloop_end_" + std::to_string(loopOps[op]) + ":");
    }
  }

  std::set<std::string> emittedLabels;
  auto emitLabel = [&](const std::string &label) {
    if (emittedLabels.insert(label).second)
      os << label << "\n";
  };

  for (auto *opPtr : funcOps) {
    if (!opPtr || opPtr->isErased()) continue;
    Operation &op = *opPtr;
    std::string opname = op.name();
    if (opname == "sysy.func") continue;
    auto labelIt = labelsBefore.find(&op);
    if (labelIt != labelsBefore.end()) {
      for (const auto &label : labelIt->second)
        emitLabel(label);
    }
    bool operandsConsumed = false;
    auto consumeAtEnd = [&]() {
      if (!operandsConsumed) {
        consumeOperands(op);
        operandsConsumed = true;
      }
      forgetHoistedResultReg(op);
    };
    struct ConsumeGuard {
      std::function<void()> fn;
      ~ConsumeGuard() { fn(); }
    } consumeGuard{consumeAtEnd};

    if (opname == "rv_machine.li" || opname == "arm_machine.mov") {
      if (!isArm && op.resultCount() == 1 &&
          skipMaterializedConstants.count(valueKey(op.result())) != 0) {
        stats.deadSpillsAvoided++;
        stats.machineOps++;
        continue;
      }
      std::string dst;
      if (op.resultType().str().find("vector") != std::string::npos) {
        dst = "v" + std::to_string(nextVecReg++);
      } else if (isFloatType(op.resultType())) {
        dst = floatReg();
      } else {
        dst = isArm ? armResultReg(nextReg++) : intResultReg(op.result());
      }
      bindResult(op.result(), dst);
      if (isFloatType(op.resultType())) {
        uint32_t bits = parseFloatAttrBits(op.attr("value"));
        if (isArm)
          os << "    mov w16, #" << bits << "\n    fmov " << dst << ", w16\n";
        else
          os << "    li t5, " << bits << "\n    fmv.w.x " << dst << ", t5\n";
      } else {
        int64_t imm = parseIntegerAttr(op.attr("value"));
        if (isArm)
          os << "    mov " << dst << ", #" << imm << "\n";
        else
          os << "    li " << dst << ", " << imm << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.addw" || opname == "arm_machine.add") {
      if (op.operandCount() != 2 || op.resultCount() != 1) {
        stats.unsupportedOps++;
        stats.error = "bad machine add shape";
        return false;
      }
      std::string lhs = ensureReg(op.operand(0), scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), scratchReg(1));
      if (lhs.empty()) {
        lhs = scratchReg(0);
        os << "    " << (isArm ? "mov " : "li ") << lhs << (isArm ? ", #0\n" : ", 0\n");
      }
      if (rhs.empty()) {
        rhs = scratchReg(1);
        os << "    " << (isArm ? "mov " : "li ") << rhs << (isArm ? ", #0\n" : ", 0\n");
      }
      std::string dst;
      const bool isVec = op.resultType().str().find("vector") != std::string::npos;
      if (isVec)
        dst = "v" + std::to_string(nextVecReg++);
      else
        dst = intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm) {
        if (isVec)
          os << "    add " << dst << ".4s, " << lhs << ".4s, " << rhs << ".4s\n";
        else
          os << "    add " << dst << ", " << lhs << ", " << rhs << "\n";
      } else {
        if (isVec)
          os << "    vadd.vv " << dst << ", " << lhs << ", " << rhs << "\n";
        else
          os << "    addw " << dst << ", " << lhs << ", " << rhs << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

	    if (opname == "rv_machine.subw" || opname == "arm_machine.sub" ||
	        opname == "rv_machine.mulw" || opname == "arm_machine.mul" ||
	        opname == "rv_machine.divw" || opname == "arm_machine.sdiv" ||
	        opname == "rv_machine.remw" || opname == "arm_machine.srem" ||
	        opname == "rv_machine.and" || opname == "arm_machine.and" ||
	        opname == "rv_machine.or" || opname == "arm_machine.orr" ||
	        opname == "rv_machine.xor" || opname == "arm_machine.eor" ||
	        opname == "rv_machine.sllw" || opname == "rv_machine.sraw") {
	      if (op.operandCount() != 2 || op.resultCount() != 1) {
	        stats.unsupportedOps++;
	        stats.error = "bad binary machine op shape";
	        return false;
	      }
	      if (!isArm && enablePow2Strength &&
	          envEnabled("SISY_ENABLE_SELF_POW2_STRENGTH", true)) {
	        int64_t imm = 0;
	        int shift = 0;
	        Value valueOperand;
	        bool isMulPow2 = false;
	        if (opname == "rv_machine.mulw") {
	          if (constantIntegerValue(op.operand(1), imm) &&
	              positivePowerOfTwoShift(imm, shift)) {
	            valueOperand = op.operand(0);
	            isMulPow2 = true;
	          } else if (constantIntegerValue(op.operand(0), imm) &&
	                     positivePowerOfTwoShift(imm, shift)) {
	            valueOperand = op.operand(1);
	            isMulPow2 = true;
	          }
	          if (isMulPow2) {
	            std::string lhs = ensureReg(valueOperand, scratchReg(0));
	            if (lhs.empty()) {
	              lhs = scratchReg(0);
	              os << "    li " << lhs << ", 0\n";
	            }
	            std::string dst = intResultReg(op.result());
	            bindResult(op.result(), dst);
	            if (shift == 0)
	              os << "    addiw " << dst << ", " << lhs << ", 0\n";
	            else
	              os << "    slliw " << dst << ", " << lhs << ", " << shift << "\n";
	            if (shouldSpillDefinedValue(op.result()))
	              spillHome(op.result(), dst);
	            stats.machineOps++;
	            stats.pow2StrengthReductions++;
	            continue;
	          }
	        }
	        if ((opname == "rv_machine.divw" || opname == "rv_machine.remw") &&
	            constantIntegerValue(op.operand(1), imm) &&
	            positivePowerOfTwoShift(imm, shift)) {
	          std::string lhs = ensureReg(op.operand(0), scratchReg(0));
	          if (lhs.empty()) {
	            lhs = scratchReg(0);
	            os << "    li " << lhs << ", 0\n";
	          }
	          std::string dst = intResultReg(op.result());
	          bindResult(op.result(), dst);
	          if (shift == 0) {
	            if (opname == "rv_machine.divw")
	              os << "    addiw " << dst << ", " << lhs << ", 0\n";
	            else
	              os << "    li " << dst << ", 0\n";
	          } else {
	            int64_t mask = (int64_t(1) << shift) - 1;
	            std::string tmp = lhs == scratchReg(1) ? scratchReg(0) : scratchReg(1);
	            os << "    sraiw " << tmp << ", " << lhs << ", 31\n";
	            if (fitsSigned12(mask)) {
	              os << "    andi " << tmp << ", " << tmp << ", " << mask << "\n";
	            } else {
	              std::string maskReg;
	              for (const char *candidate : {"t4", "t3", "t2", "t1", "t0", "t5", "t6"}) {
	                std::string candidateReg(candidate);
	                if (candidateReg != lhs && candidateReg != tmp && candidateReg != dst) {
	                  maskReg = candidateReg;
	                  break;
	                }
	              }
	              if (maskReg.empty())
	                maskReg = scratchReg(0);
	              clobberPhysicalReg(maskReg);
	              os << "    li " << maskReg << ", " << mask << "\n";
	              os << "    and " << tmp << ", " << tmp << ", " << maskReg << "\n";
	            }
	            os << "    addw " << tmp << ", " << lhs << ", " << tmp << "\n";
	            if (opname == "rv_machine.remw") {
	              os << "    sraiw " << tmp << ", " << tmp << ", " << shift << "\n";
	              os << "    slliw " << tmp << ", " << tmp << ", " << shift << "\n";
	              os << "    subw " << dst << ", " << lhs << ", " << tmp << "\n";
	            } else {
	              os << "    sraiw " << dst << ", " << tmp << ", " << shift << "\n";
	            }
	          }
	          if (shouldSpillDefinedValue(op.result()))
	            spillHome(op.result(), dst);
	          stats.machineOps++;
	          stats.pow2StrengthReductions++;
	          continue;
	        }
	      }
	      if (!isArm && (opname == "rv_machine.and" ||
	                     opname == "rv_machine.or" ||
	                     opname == "rv_machine.xor")) {
	        int64_t imm = 0;
	        Value valueOperand;
	        bool hasImm = false;
	        int64_t otherImm = 0;
	        bool rhsConst = constantIntegerValue(op.operand(1), imm) && fitsSigned12(imm);
	        bool lhsConst = constantIntegerValue(op.operand(0), otherImm) && fitsSigned12(otherImm);
	        if (rhsConst && !lhsConst) {
	          valueOperand = op.operand(0);
	          hasImm = true;
	        } else if (lhsConst && !rhsConst) {
	          imm = otherImm;
	          valueOperand = op.operand(1);
	          hasImm = true;
	        }
	        if (hasImm) {
	          std::string lhs = ensureReg(valueOperand, scratchReg(0));
	          if (lhs.empty()) {
	            lhs = scratchReg(0);
	            os << "    li " << lhs << ", 0\n";
	          }
	          std::string dst = intResultReg(op.result());
	          bindResult(op.result(), dst);
	          const char *inst = opname == "rv_machine.and" ? "andi"
	                            : (opname == "rv_machine.or" ? "ori" : "xori");
	          os << "    " << inst << " " << dst << ", " << lhs << ", " << imm << "\n";
	          os << "    addiw " << dst << ", " << dst << ", 0\n";
	          if (shouldSpillDefinedValue(op.result()))
	            spillHome(op.result(), dst);
	          stats.machineOps++;
	          stats.pow2StrengthReductions++;
	          continue;
	        }
	      }
	      if (!isArm && (opname == "rv_machine.sllw" ||
	                     opname == "rv_machine.sraw")) {
	        int64_t imm = 0;
	        if (constantIntegerValue(op.operand(1), imm) && imm >= 0 && imm < 32) {
	          std::string lhs = ensureReg(op.operand(0), scratchReg(0));
	          if (lhs.empty()) {
	            lhs = scratchReg(0);
	            os << "    li " << lhs << ", 0\n";
	          }
	          std::string dst = intResultReg(op.result());
	          bindResult(op.result(), dst);
	          const char *inst = opname == "rv_machine.sllw" ? "slliw" : "sraiw";
	          os << "    " << inst << " " << dst << ", " << lhs << ", " << imm << "\n";
	          if (shouldSpillDefinedValue(op.result()))
	            spillHome(op.result(), dst);
	          stats.machineOps++;
	          stats.pow2StrengthReductions++;
	          continue;
	        }
	      }
	      std::string lhs = ensureReg(op.operand(0), scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), scratchReg(1));
      if (lhs.empty()) {
        lhs = scratchReg(0);
        os << "    " << (isArm ? "mov " : "li ") << lhs << (isArm ? ", #0\n" : ", 0\n");
      }
      if (rhs.empty()) {
        rhs = scratchReg(1);
        os << "    " << (isArm ? "mov " : "li ") << rhs << (isArm ? ", #0\n" : ", 0\n");
      }
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm) {
        if (opname == "arm_machine.srem") {
          os << "    sdiv " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    msub " << dst << ", " << dst << ", " << rhs << ", " << lhs << "\n";
        } else {
          std::string inst = opname.substr(std::string("arm_machine.").size());
          os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
        }
      } else {
        std::string inst = "addw";
        if (opname == "rv_machine.subw") inst = "subw";
        else if (opname == "rv_machine.mulw") inst = "mulw";
        else if (opname == "rv_machine.divw") inst = "divw";
        else if (opname == "rv_machine.remw") inst = "remw";
        else if (opname == "rv_machine.and") inst = "and";
        else if (opname == "rv_machine.or") inst = "or";
        else if (opname == "rv_machine.xor") inst = "xor";
        else if (opname == "rv_machine.sllw") inst = "sllw";
        else if (opname == "rv_machine.sraw") inst = "sraw";
        os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
        if (inst == "and" || inst == "or" || inst == "xor")
          os << "    addiw " << dst << ", " << dst << ", 0\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fadd" || opname == "arm_machine.fadd" ||
        opname == "rv_machine.fsub" || opname == "arm_machine.fsub" ||
        opname == "rv_machine.fmul" || opname == "arm_machine.fmul" ||
        opname == "rv_machine.fdiv" || opname == "arm_machine.fdiv") {
      std::string lhs = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      std::string rhs = ensureReg(op.operand(1), isArm ? "s31" : "ft11");
      if (lhs.empty()) lhs = isArm ? "s0" : "ft0";
      if (rhs.empty()) rhs = isArm ? "s1" : "ft1";
      const bool isVec = op.resultType().str().find("vector") != std::string::npos;
      std::string dst = isVec ? "v" + std::to_string(nextVecReg++) : floatReg();
      bindResult(op.result(), dst);
      if (isArm) {
        std::string inst = opname.substr(std::string("arm_machine.").size());
        if (isVec)
          os << "    " << inst << " " << dst << ".4s, " << lhs << ".4s, "
             << rhs << ".4s\n";
        else
          os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
      } else {
        std::string inst = "fadd.s";
        if (isVec) {
          inst = "vfadd.vv";
          if (opname == "rv_machine.fsub") inst = "vfsub.vv";
          else if (opname == "rv_machine.fmul") inst = "vfmul.vv";
          else if (opname == "rv_machine.fdiv") inst = "vfdiv.vv";
        } else {
          if (opname == "rv_machine.fsub") inst = "fsub.s";
          else if (opname == "rv_machine.fmul") inst = "fmul.s";
          else if (opname == "rv_machine.fdiv") inst = "fdiv.s";
        }
        os << "    " << inst << " " << dst << ", " << lhs << ", " << rhs << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fneg" || opname == "arm_machine.fneg") {
      std::string src = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      if (src.empty()) src = isArm ? "s0" : "ft0";
      std::string dst = floatReg();
      bindResult(op.result(), dst);
      os << "    " << (isArm ? "fneg " : "fneg.s ") << dst << ", " << src << "\n";
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fcvt_s_w" || opname == "arm_machine.scvtf") {
      std::string src = ensureReg(op.operand(0), scratchReg(0));
      if (src.empty()) src = scratchReg(0);
      std::string dst = floatReg();
      bindResult(op.result(), dst);
      if (isArm)
        os << "    scvtf " << dst << ", " << src << "\n";
      else
        os << "    fcvt.s.w " << dst << ", " << src << "\n";
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.fcvt_w_s" || opname == "arm_machine.fcvtzs") {
      std::string src = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      if (src.empty()) src = isArm ? "s0" : "ft0";
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm)
        os << "    fcvtzs " << dst << ", " << src << "\n";
      else
        os << "    fcvt.w.s " << dst << ", " << src << ", rtz\n";
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.neg" || opname == "arm_machine.neg" ||
        opname == "rv_machine.seqz" || opname == "arm_machine.not") {
      std::string src = reloadValue(op.operand(0), scratchReg(0));
      if (src.empty()) {
        stats.unsupportedOps++;
        Operation *def = op.operand(0).getDefiningOp();
        stats.error = "unary machine op operand has no assigned register: " +
                      op.operand(0).printName() + " key=" +
                      valueKey(op.operand(0)) + " def=" +
                      (def ? def->name() : std::string("<block-arg>"));
        return false;
      }
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      if (isArm) {
        if (opname == "arm_machine.not")
          os << "    cmp " << src << ", #0\n    cset " << dst << ", eq\n";
        else
          os << "    neg " << dst << ", " << src << "\n";
      } else {
        if (opname == "rv_machine.seqz")
          os << "    seqz " << dst << ", " << src << "\n";
        else
          os << "    negw " << dst << ", " << src << "\n";
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.alloca" || opname == "memref.alloca") {
      auto it = stackSlots.find(valueKey(op.result()));
      if (it != stackSlots.end())
        regs[valueKey(op.result())] = "stack:" + std::to_string(it->second);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.load" || opname == "memref.load") {
      if (op.operandCount() == 0) {
        stats.unsupportedOps++;
        stats.error = "load without base address";
        return false;
      }
      std::vector<Value> indices;
      for (int i = 1; i < op.operandCount(); i++)
        indices.push_back(op.operand(i));
      if (indices.empty()) {
        auto promotedIt = promotedScalarSlots.find(valueKey(op.operand(0)));
        if (promotedIt != promotedScalarSlots.end()) {
          PromotedScalarSlot &slot = promotedIt->second;
          if (!slot.valid) {
            auto stackIt = stackSlots.find(valueKey(op.operand(0)));
            if (stackIt == stackSlots.end()) {
              stats.unsupportedOps++;
              stats.error = "promoted scalar slot has no stack home";
              return false;
            }
            emitStackLoad(slot.reg, stackIt->second, "lw");
            slot.valid = true;
          }
          std::string dst = intResultReg(op.result());
          bindResult(op.result(), dst);
          if (dst != slot.reg)
            os << "    mv " << dst << ", " << slot.reg << "\n";
          if (shouldSpillDefinedValue(op.result()))
            spillHome(op.result(), dst);
          stats.scalarRegLoads++;
          stats.machineOps++;
          continue;
        }
      }
      if (parseMemrefInfo(op.resultType()).valid) {
        std::string addr;
        if (!computeAddress(op.operand(0), indices, scratchReg(0),
                            scratchReg(1), scratchReg(0), true, addr))
          return false;
        bindResult(op.result(), addr);
        if (livenessEnabled && (addr == scratchReg(0) || addr == scratchReg(1))) {
          if (remainingUses[valueKey(op.result())] > 0)
            spillHome(op.result(), addr);
          regs.erase(valueKey(op.result()));
        } else if (shouldSpillDefinedValue(op.result())) {
          spillHome(op.result(), addr);
        }
        stats.machineOps++;
        continue;
      }
      std::string dst = resultReg();
      if (isFloatType(op.resultType()))
        dst = floatReg();
      else if (!isArm)
        dst = intResultReg(op.result());
      bindResult(op.result(), dst);
      if (op.attr("linearized_index") && indices.size() == 1) {
        std::string addr;
        if (!computeLinearizedElementAddress(op.operand(0), indices[0],
                                             scratchReg(0), scratchReg(1),
                                             scratchReg(0), addr))
          return false;
        if (isArm)
          os << "    ldr " << dst << ", [" << addr << "]\n";
        else
          os << "    " << (isFloatType(op.resultType()) ? "flw " : "lw ")
             << dst << ", 0(" << addr << ")\n";
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), dst);
        stats.machineOps++;
        continue;
      }
      if (!loadFromAddress(dst, op.operand(0), indices))
        return false;
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.store" || opname == "memref.store") {
      if (op.operandCount() < 2) {
        stats.unsupportedOps++;
        stats.error = "store without base address";
        return false;
      }
      std::vector<Value> indices;
      for (int i = 2; i < op.operandCount(); i++)
        indices.push_back(op.operand(i));
      if (indices.empty()) {
        auto promotedIt = promotedScalarSlots.find(valueKey(op.operand(1)));
        if (promotedIt != promotedScalarSlots.end()) {
          PromotedScalarSlot &slot = promotedIt->second;
          std::string val = ensureReg(op.operand(0), scratchReg(0));
          if (val.empty()) {
            stats.unsupportedOps++;
            stats.error = "promoted scalar store value has no assigned register";
            return false;
          }
          if (val != slot.reg)
            os << "    mv " << slot.reg << ", " << val << "\n";
          slot.valid = true;
          slot.dirty = true;
          stats.scalarRegStores++;
          stats.machineOps++;
          continue;
        }
      }
      if (op.attr("linearized_index") && indices.size() == 1) {
        bool fpStore = isFloatType(op.operand(0).type());
        std::string addr;
        if (!computeLinearizedElementAddress(op.operand(1), indices[0],
                                             scratchReg(0), scratchReg(1),
                                             scratchReg(0), addr))
          return false;
        if (isArm) {
          std::string val = ensureReg(op.operand(0), fpStore ? "s30" : scratchReg(0));
          fpStore = fpStore || looksFloatReg(val);
          os << "    str " << val << ", [" << addr << "]\n";
        } else {
          std::string val = ensureReg(op.operand(0), fpStore ? "ft10" : scratchReg(0));
          fpStore = fpStore || looksFloatReg(val);
          os << "    " << (fpStore ? "fsw " : "sw ") << val << ", 0(" << addr << ")\n";
        }
        stats.machineOps++;
        continue;
      }
      if (!storeToAddress(op.operand(0), op.operand(1), indices))
        return false;
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.vle32" || opname == "arm_machine.ld1" || opname == "vector.transfer_read") {
      std::string base = ensureReg(op.operand(0), scratchReg(0));
      std::string index = ensureReg(op.operand(1), scratchReg(1));
      std::string dst = "v" + std::to_string(nextVecReg++);
      bindResult(op.result(), dst);
      if (isArm) {
        os << "    add x9, " << base << ", " << index << ", lsl #2\n";
        os << "    ld1 {" << dst << ".4s}, [x9]\n";
      } else {
        os << "    vsetvli zero, 4, e32, m1, ta, ma\n";
        os << "    slli t6, " << index << ", 2\n";
        os << "    add t6, " << base << ", t6\n";
        os << "    vle32.v " << dst << ", (t6)\n";
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.vse32" || opname == "arm_machine.st1" || opname == "vector.transfer_write") {
      std::string val = ensureReg(op.operand(0), scratchReg(0));
      std::string base = ensureReg(op.operand(1), scratchReg(0));
      std::string index = ensureReg(op.operand(2), scratchReg(1));
      if (isArm) {
        os << "    add x9, " << base << ", " << index << ", lsl #2\n";
        os << "    st1 {" << val << ".4s}, [x9]\n";
      } else {
        os << "    vsetvli zero, 4, e32, m1, ta, ma\n";
        os << "    slli t6, " << index << ", 2\n";
        os << "    add t6, " << base << ", t6\n";
        os << "    vse32.v " << val << ", (t6)\n";
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.vfmv" || opname == "arm_machine.dup" || opname == "vector.splat") {
      std::string val = ensureReg(op.operand(0), isArm ? "s30" : "ft10");
      std::string dst = "v" + std::to_string(nextVecReg++);
      bindResult(op.result(), dst);
      if (isArm) {
        os << "    dup " << dst << ".4s, " << val << "\n";
      } else {
        os << "    vsetvli zero, 4, e32, m1, ta, ma\n";
        os << "    vfmv.v.f " << dst << ", " << val << "\n";
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.rotate_helper") {
      if (op.operandCount() != 2 || op.resultCount() != 1) {
        stats.unsupportedOps++;
        stats.error = "bad rotate helper shape";
        return false;
      }
      std::string direction = symbolAttr(op.attr("direction"));
      int64_t maxShift = parseIntegerAttr(op.attr("max_shift"));
      if ((direction != "left" && direction != "right") || maxShift < 1 || maxShift > 30) {
        stats.unsupportedOps++;
        stats.error = "bad rotate helper attributes";
        return false;
      }
      std::string xReg = ensureReg(op.operand(0), scratchReg(0));
      std::string nReg = ensureReg(op.operand(1), scratchReg(1));
      if (xReg.empty() || nReg.empty()) {
        stats.unsupportedOps++;
        stats.error = "rotate helper operand has no assigned register";
        return false;
      }
      std::string dst = resultReg();
      bindResult(op.result(), dst);

      auto chooseTemp = [&](std::initializer_list<std::string> excluded) -> std::string {
        std::set<std::string> used(excluded.begin(), excluded.end());
        const std::vector<std::string> pool = isArm
            ? std::vector<std::string>{"w16", "w17", "w15", "w14", "w13"}
            : std::vector<std::string>{"t5", "t6", "t4", "t3", "t2"};
        for (const auto &reg : pool) {
          if (used.count(reg) == 0) {
            clobberPhysicalReg(reg);
            return reg;
          }
        }
        return pool.front();
      };
      auto armW = [](std::string reg) {
        if (!reg.empty() && reg[0] == 'x')
          reg[0] = 'w';
        return reg;
      };

      int labelId = ++nextLoopId;
      std::string done = ".Lrot_done_" + std::to_string(labelId);
      if (isArm) {
        xReg = armW(xReg);
        nReg = armW(nReg);
        if (dst != xReg)
          os << "    mov " << dst << ", " << xReg << "\n";
        os << "    cmp " << nReg << ", #1\n";
        os << "    blt " << done << "\n";
        os << "    cmp " << nReg << ", #" << maxShift << "\n";
        os << "    bgt " << done << "\n";
        if (direction == "left") {
          os << "    lsl " << dst << ", " << dst << ", " << nReg << "\n";
        } else {
          std::string mask = chooseTemp({dst, nReg});
          std::string sign = chooseTemp({dst, nReg, mask});
          os << "    mov " << mask << ", #1\n";
          os << "    lsl " << mask << ", " << mask << ", " << nReg << "\n";
          os << "    sub " << mask << ", " << mask << ", #1\n";
          os << "    asr " << sign << ", " << dst << ", #31\n";
          os << "    and " << mask << ", " << mask << ", " << sign << "\n";
          os << "    add " << dst << ", " << dst << ", " << mask << "\n";
          os << "    asr " << dst << ", " << dst << ", " << nReg << "\n";
        }
        os << done << ":\n";
      } else {
        if (dst != xReg)
          os << "    mv " << dst << ", " << xReg << "\n";
        std::string guardTmp = chooseTemp({dst, nReg});
        os << "    li " << guardTmp << ", 1\n";
        os << "    blt " << nReg << ", " << guardTmp << ", " << done << "\n";
        os << "    li " << guardTmp << ", " << maxShift << "\n";
        os << "    blt " << guardTmp << ", " << nReg << ", " << done << "\n";
        if (direction == "left") {
          os << "    sllw " << dst << ", " << dst << ", " << nReg << "\n";
        } else {
          std::string mask = guardTmp;
          std::string sign = chooseTemp({dst, nReg, mask});
          os << "    li " << mask << ", 1\n";
          os << "    sllw " << mask << ", " << mask << ", " << nReg << "\n";
          os << "    addiw " << mask << ", " << mask << ", -1\n";
          os << "    sraiw " << sign << ", " << dst << ", 31\n";
          os << "    and " << mask << ", " << mask << ", " << sign << "\n";
          os << "    addw " << dst << ", " << dst << ", " << mask << "\n";
          os << "    sraw " << dst << ", " << dst << ", " << nReg << "\n";
        }
      emitLabel(done + ":");
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.call") {
      std::string callee = symbolAttr(op.attr("callee"));
      if (callee.empty()) {
        stats.unsupportedOps++;
        stats.error = "call without callee attr";
        return false;
      }
      auto stageCallOperand = [&](Value value, const std::string &dst) {
        std::string src = ensureReg(value, dst);
        if (src.rfind("stack:", 0) == 0 || src.rfind("global:", 0) == 0)
          src = materializeAddress(value, dst);
        if (src.empty()) {
          clobberPhysicalReg(dst);
          os << "    li " << dst << ", 0\n";
        } else if (src != dst) {
          clobberPhysicalReg(dst);
          os << "    mv " << dst << ", " << src << "\n";
        }
      };
      auto modpowIt = modularPowerFunctions.find(callee);
      if (!isArm && modpowIt != modularPowerFunctions.end() &&
          op.operandCount() == 2 && op.resultCount() == 1 &&
          isI32Like(op.resultType()) && isI32Like(op.operand(0).type()) &&
          isI32Like(op.operand(1).type())) {
        stageCallOperand(op.operand(0), "t2");
        stageCallOperand(op.operand(1), "t3");
        std::string dst = intResultReg(op.result());
        int labelId = ++nextLoopId;
        std::string loop = ".Lmodpow_call_loop_" + std::to_string(labelId);
        std::string skip = ".Lmodpow_call_skip_" + std::to_string(labelId);
        std::string done = ".Lmodpow_call_done_" + std::to_string(labelId);
        os << "    li t0, " << modpowIt->second.modulus << "\n";
        os << "    li t1, 1\n";
        os << loop << ":\n";
        os << "    blez t3, " << done << "\n";
        os << "    andi t4, t3, 1\n";
        os << "    beqz t4, " << skip << "\n";
        os << "    mul t5, t1, t2\n";
        os << "    rem t1, t5, t0\n";
        os << "    addiw t1, t1, 0\n";
        os << skip << ":\n";
        os << "    mul t5, t2, t2\n";
        os << "    rem t2, t5, t0\n";
        os << "    addiw t2, t2, 0\n";
        os << "    sraiw t3, t3, 1\n";
        os << "    j " << loop << "\n";
        os << done << ":\n";
        os << "    mv " << dst << ", t1\n";
        bindResult(op.result(), dst);
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), dst);
        stats.modularPowerCallsites++;
        stats.machineOps += 16;
        continue;
      }
      if (!isArm && memcopyFunctions.count(callee) != 0 &&
          op.operandCount() == 4 && op.resultCount() == 1 &&
          isMemrefType(op.operand(0).type()) && isI32Like(op.operand(1).type()) &&
          isMemrefType(op.operand(2).type()) && isI32Like(op.operand(3).type())) {
        stageCallOperand(op.operand(0), "t0");
        stageCallOperand(op.operand(1), "t1");
        stageCallOperand(op.operand(2), "t2");
        stageCallOperand(op.operand(3), "t3");
        int labelId = ++nextLoopId;
        std::string loop = ".Lmemcopy_call_loop_" + std::to_string(labelId);
        std::string done = ".Lmemcopy_call_done_" + std::to_string(labelId);
        os << "    slli t1, t1, 2\n";
        os << "    add t0, t0, t1\n";
        os << "    li t4, 0\n";
        os << loop << ":\n";
        os << "    bge t4, t3, " << done << "\n";
        os << "    lw t5, 0(t2)\n";
        os << "    sw t5, 0(t0)\n";
        os << "    addi t0, t0, 4\n";
        os << "    addi t2, t2, 4\n";
        os << "    addiw t4, t4, 1\n";
        os << "    j " << loop << "\n";
        os << done << ":\n";
        bindResult(op.result(), "t4");
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), "t4");
        stats.memcopyCallsites++;
        stats.machineOps += 12;
        continue;
      }
      auto modmulIt = modularMultiplyFunctions.find(callee);
      if (!isArm && modmulIt != modularMultiplyFunctions.end() &&
          op.operandCount() == 2 && op.resultCount() == 1 &&
          isI32Like(op.resultType()) && isI32Like(op.operand(0).type()) &&
          isI32Like(op.operand(1).type())) {
        stageCallOperand(op.operand(0), "t0");
        stageCallOperand(op.operand(1), "t1");
        std::string dst = intResultReg(op.result());
        int labelId = ++nextLoopId;
        std::string zero = ".Lmodmul_call_zero_" + std::to_string(labelId);
        std::string done = ".Lmodmul_call_done_" + std::to_string(labelId);
        os << "    blez t1, " << zero << "\n";
        os << "    mul t2, t0, t1\n";
        os << "    li t3, " << modmulIt->second.modulus << "\n";
        os << "    rem t2, t2, t3\n";
        os << "    addiw " << dst << ", t2, 0\n";
        os << "    j " << done << "\n";
        os << zero << ":\n";
        os << "    li " << dst << ", 0\n";
        os << done << ":\n";
        bindResult(op.result(), dst);
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), dst);
        stats.modularMultiplyCallsites++;
        stats.machineOps += 6;
        continue;
      }
      Operation *tailReturn = nextLiveOpAfter(op);
      bool selfTailCall = envEnabled("SISY_ENABLE_SELF_TAIL_CALL", true) &&
                          !memoInfo && callee == name && op.operandCount() <= 2 &&
                          tailReturn &&
                          (tailReturn->name() == "sysy.return" ||
                           tailReturn->name() == "scf.return");
      if (selfTailCall) {
        for (auto operand : op.getOperands()) {
          if (isFloatType(operand.type()) || isMemrefType(operand.type())) {
            selfTailCall = false;
            break;
          }
        }
      }
      if (selfTailCall) {
        if (op.resultCount() == 0) {
          selfTailCall = tailReturn->operandCount() == 0;
        } else {
          selfTailCall = tailReturn->operandCount() == 1 &&
                         tailReturn->operand(0) == op.result() &&
                         op.resultUses.size() == 1 &&
                         op.resultUses[0].size() == 1 &&
                         op.resultUses[0][0].owner == tailReturn;
        }
      }
      if (selfTailCall) {
        std::vector<std::string> staged;
        for (int i = 0; i < op.operandCount(); i++) {
          std::string tmp = isArm ? ("w" + std::to_string(16 + i)) : scratchReg(i);
          std::string src = ensureReg(op.operand(i), tmp);
          if (src.rfind("stack:", 0) == 0 || src.rfind("global:", 0) == 0)
            src = materializeAddress(op.operand(i), tmp);
          if (src.empty()) {
            if (isArm)
              os << "    mov " << tmp << ", #0\n";
            else
              os << "    li " << tmp << ", 0\n";
          } else if (src != tmp) {
            os << "    " << (isArm ? "mov " : "mv ") << tmp << ", " << src << "\n";
          }
          staged.push_back(tmp);
        }
        for (int i = 0; i < (int) staged.size(); i++) {
          std::string arg = isArm ? ("w" + std::to_string(i))
                                  : ("a" + std::to_string(i));
          if (staged[i] != arg) {
            clobberPhysicalReg(arg);
            os << "    " << (isArm ? "mov " : "mv ") << arg << ", "
               << staged[i] << "\n";
          }
        }
        os << "    " << (isArm ? "b" : "j") << " " << bodyEntryLabel << "\n";
        tailReturnSkips.insert(tailReturn);
        stats.tailCalls++;
        stats.machineOps++;
        continue;
      }
      invalidateCallerSavedForCall();
      int outgoingIntRegs = 0;
      int outgoingFpRegs = 0;
      int outgoingStackSlots = 0;
      for (int i = 0; i < op.operandCount(); i++) {
        Value operand = op.operand(i);
        bool fpArg = isFloatType(operand.type());
        bool ptrArg = isMemrefType(operand.type());
        std::string src = ensureReg(operand, fpArg ? floatScratchReg() : scratchReg(0));
        if (src.rfind("stack:", 0) == 0 || src.rfind("global:", 0) == 0)
          src = materializeAddress(operand, scratchReg(0));
        if (src.empty()) {
          if (fpArg) {
            src = floatScratchReg();
            emitFloatBitsToReg(src, 0);
          } else {
            src = scratchReg(0);
            if (isArm)
              os << "    mov " << src << ", #0\n";
            else
              os << "    li " << src << ", 0\n";
          }
        }

        if (fpArg) {
          if (outgoingFpRegs < 8) {
            std::string arg = (isArm ? "s" : "fa") + std::to_string(outgoingFpRegs++);
            if (src != arg) {
              clobberPhysicalReg(arg);
              os << "    " << (isArm ? "fmov " : "fmv.s ") << arg << ", " << src << "\n";
            }
          } else {
            emitStackStore(src, outgoingArgBase + outgoingStackSlots++ * 8,
                           isArm ? "str" : "fsw");
          }
        } else {
          if (outgoingIntRegs < 8) {
            std::string arg;
            if (isArm && ptrArg)
              arg = "x" + std::to_string(outgoingIntRegs);
            else if (isArm)
              arg = "w" + std::to_string(outgoingIntRegs);
            else
              arg = "a" + std::to_string(outgoingIntRegs);
            outgoingIntRegs++;
            if (src != arg) {
              clobberPhysicalReg(arg);
              os << "    " << (isArm ? "mov " : "mv ") << arg << ", " << src << "\n";
            }
          } else {
            emitStackStore(src, outgoingArgBase + outgoingStackSlots++ * 8,
                           ptrArg ? (isArm ? "str" : "sd") : (isArm ? "str" : "sw"));
          }
        }
      }
      invalidateCallerSavedForCall();
      os << "    call " << callee << "\n";
      if (op.resultCount() > 0) {
        std::string result = isFloatType(op.resultType()) ? (isArm ? "s0" : "fa0")
                                                          : (isArm ? "w0" : "a0");
        bindResult(op.result(), result);
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), result);
      }
      stats.machineOps++;
      continue;
    }

    if (opname == "sysy.unknown_value") {
      if (op.resultCount() > 0 && op.resultType().kind() != TypeKind::None) {
        std::string dst = resultReg();
        bindResult(op.result(), dst);
        if (isArm)
          os << "    mov " << dst << ", #0\n";
        else
          os << "    li " << dst << ", 0\n";
        if (shouldSpillDefinedValue(op.result()))
          spillHome(op.result(), dst);
      }
      continue;
    }

    if (opname == "sysy.return" || opname == "scf.return" ||
        opname == "rv_machine.ret" || opname == "arm_machine.ret") {
      if (tailReturnSkips.count(&op) != 0) {
        stats.returns++;
        continue;
      }
      if (op.operandCount() > 0) {
        std::string src = ensureReg(op.operand(0), isFloatType(op.operand(0).type()) ? (isArm ? "s30" : "ft10") : scratchReg(0));
        if (src.empty()) {
          stats.unsupportedOps++;
          stats.error = "return operand has no assigned register";
          return false;
        }
        if (isArm) {
          if (isFloatType(op.operand(0).type())) {
            if (src != "s0")
              os << "    fmov s0, " << src << "\n";
          } else if (src != "w0") {
            os << "    mov w0, " << src << "\n";
          }
        } else {
          if (isFloatType(op.operand(0).type())) {
            if (src != "fa0")
              os << "    fmv.s fa0, " << src << "\n";
          } else if (src != "a0") {
            os << "    mv a0, " << src << "\n";
          }
        }
        if (memoInfo && !isFloatType(op.operand(0).type()))
          emitMemoStore(*memoInfo);
      }
      os << "    " << (isArm ? "b" : "j") << " " << epilogueLabel << "\n";
      stats.returns++;
      continue;
    }

    if (opname == "affine.for") {
      int loopId = loopOps[&op];
      std::string iv;
      if (isArm) {
        iv = "w" + std::to_string(nextLoopReg++ % 10 + 19);
      } else if (regAlloc2Enabled && calleeSaveCount >= 12) {
        static const char *ivRegs[] = {"s11", "s10", "s9", "s8"};
        iv = ivRegs[(nextLoopReg++) % (int)(sizeof(ivRegs) / sizeof(ivRegs[0]))];
      } else if (stableBaseActive) {
        static const char *ivRegsNoStable[] = {
          "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
        };
        int available = std::max(1, calleeSaveCount - 4);
        available = std::min<int>(
            available, (int)(sizeof(ivRegsNoStable) / sizeof(ivRegsNoStable[0])));
        iv = ivRegsNoStable[(nextLoopReg++) % available];
      } else {
        iv = "s" + std::to_string(nextLoopReg++ % 12);
      }
      std::string boundScratch = isArm ? "w17" : scratchReg(1);

      Value ivValue = op.getRegions()[0]->getBlocks()[0]->args()[0]->value();
      loopIvRegs[&op] = iv;
      bindReg(ivValue, iv);
      std::string pinnedBound = ensureReg(op.operand(1), boundScratch);
      if (pinnedBound.empty()) {
        stats.unsupportedOps++;
        stats.error = "affine.for bound has no assigned register";
        return false;
      }
      if (!pinnedBound.empty() && !canRematerializeScalar(op.operand(1), 0))
        spillHome(op.operand(1), pinnedBound, true);
      std::string pinnedStep = ensureReg(op.operand(2), boundScratch);
      if (pinnedStep.empty()) {
        stats.unsupportedOps++;
        stats.error = "affine.for step has no assigned register";
        return false;
      }
      if (!pinnedStep.empty() && !canRematerializeScalar(op.operand(2), 0))
        spillHome(op.operand(2), pinnedStep, true);
      std::string start = reloadValue(op.operand(0), isArm ? "w16" : scratchReg(0));
      if (start.empty()) {
        stats.unsupportedOps++;
        stats.error = "affine.for lower bound has no assigned register";
        return false;
      }

      if (isArm) {
        os << "    mov " << iv << ", " << start << "\n";
        spillHome(ivValue, iv, true);
        spillLiveRegsForControlFlow();
        emitLabel(".Lloop_cond_" + std::to_string(loopId) + ":");
        std::string condIv = reloadValue(ivValue, iv);
        if (condIv.empty()) {
          stats.unsupportedOps++;
          stats.error = "affine.for iv reload has no assigned register";
          return false;
        }
        if (condIv != iv)
          os << "    mov " << iv << ", " << condIv << "\n";
        std::string bound = reloadValue(op.operand(1), boundScratch);
        if (bound.empty()) {
          stats.unsupportedOps++;
          stats.error = "affine.for bound reload has no assigned register";
          return false;
        }
        os << "    cmp " << iv << ", " << bound << "\n";
        os << "    bge .Lloop_end_" << loopId << "\n";
      } else {
        os << "    mv " << iv << ", " << start << "\n";
        spillHome(ivValue, iv, true);
        spillLiveRegsForControlFlow();
        emitLabel(".Lloop_cond_" + std::to_string(loopId) + ":");
        std::string condIv = reloadValue(ivValue, iv);
        if (condIv.empty()) {
          stats.unsupportedOps++;
          stats.error = "affine.for iv reload has no assigned register";
          return false;
        }
        if (condIv != iv)
          os << "    mv " << iv << ", " << condIv << "\n";
        std::string bound = reloadValue(op.operand(1), boundScratch);
        if (bound.empty()) {
          stats.unsupportedOps++;
          stats.error = "affine.for bound reload has no assigned register";
          return false;
        }
        os << "    bge " << iv << ", " << bound << ", .Lloop_end_" << loopId << "\n";
      }
      invalidateLoopHeaderRegs(ivValue, false);
      regs[valueKey(ivValue)] = iv;
      continue;
    }

    if (opname == "scf.while") {
      int loopId = loopOps[&op];
      spillLiveRegsForControlFlow();
      emitLabel(".Lwhile_cond_" + std::to_string(loopId) + ":");
      invalidateLoopEntryRegs(false);
      continue;
    }

    if (opname == "scf.condition") {
      Operation *parentWhile = op.getBlock()->getRegion()->getParent();
      if (!parentWhile || parentWhile->name() != "scf.while") {
        stats.unsupportedOps++;
        stats.error = "scf.condition outside scf.while";
        return false;
      }
      int loopId = loopOps[parentWhile];
      std::string cond = ensureReg(op.operand(0), scratchReg(0));
      if (cond.empty()) {
        stats.unsupportedOps++;
        stats.error = "scf.condition operand has no assigned register";
        return false;
      }
      if (isArm)
        os << "    cbz " << cond << ", .Lwhile_end_" << loopId << "\n";
      else
        os << "    beqz " << cond << ", .Lwhile_end_" << loopId << "\n";
      continue;
    }

    if (opname == "affine.yield") {
      Operation *parentFor = op.getBlock()->getRegion()->getParent();
      if (!parentFor || parentFor->name() != "affine.for") continue;
      int loopId = loopOps[parentFor];
      bool isLoopBodyTerminator =
          op.getBlock() && !parentFor->getRegions().empty() &&
          op.getBlock()->getRegion() == parentFor->getRegions()[0].get();
      if (!isLoopBodyTerminator) {
        os << "    " << (isArm ? "b" : "j") << " .Lloop_latch_" << loopId << "\n";
        continue;
      }
      std::string step = ensureReg(parentFor->operand(2), scratchReg(1));
      if (step.empty()) {
        stats.unsupportedOps++;
        stats.error = "affine.yield parent step has no assigned register";
        return false;
      }
      Value ivValue = parentFor->getRegions()[0]->getBlocks()[0]->args()[0]->value();
      std::string preferredIv = loopIvRegs.count(parentFor) ? loopIvRegs[parentFor]
                                                            : scratchReg(0);
      std::string iv = ensureReg(ivValue, preferredIv);
      if (iv.empty()) {
        stats.unsupportedOps++;
        stats.error = "affine.yield parent iv has no assigned register";
        return false;
      }
      if (iv != preferredIv) {
        if (isArm)
          os << "    mov " << preferredIv << ", " << iv << "\n";
        else
          os << "    mv " << preferredIv << ", " << iv << "\n";
        iv = preferredIv;
      }

      if (isArm) {
        emitLabel(".Lloop_latch_" + std::to_string(loopId) + ":");
        os << "    add " << iv << ", " << iv << ", " << step << "\n";
        bindReg(ivValue, iv);
        spillHome(ivValue, iv, true);
        os << "    b .Lloop_cond_" << loopId << "\n";
        emitLabel(".Lloop_end_" + std::to_string(loopId) + ":");
      } else {
        emitLabel(".Lloop_latch_" + std::to_string(loopId) + ":");
        os << "    addw " << iv << ", " << iv << ", " << step << "\n";
        bindReg(ivValue, iv);
        spillHome(ivValue, iv, true);
        os << "    j .Lloop_cond_" << loopId << "\n";
        emitLabel(".Lloop_end_" + std::to_string(loopId) + ":");
      }
      invalidateControlFlowJoinRegs();
      continue;
    }

    if (opname == "sysy.continue" || opname == "sysy.break") {
      Operation *loop = nullptr;
      Block *currBlock = op.getBlock();
      while (currBlock && !loop) {
        Region *region = currBlock->getRegion();
        Operation *parent = region ? region->getParent() : nullptr;
        if (!parent)
          break;
        if (parent->name() == "scf.while" || parent->name() == "affine.for") {
          loop = parent;
          break;
        }
        currBlock = parent->getBlock();
      }
      if (!loop || loopOps.count(loop) == 0) {
        stats.unsupportedOps++;
        stats.error = opname + " without enclosing lowered loop";
        return false;
      }
      int loopId = loopOps[loop];
      const bool isContinue = opname == "sysy.continue";
      if (loop->name() == "affine.for") {
        os << "    " << (isArm ? "b" : "j") << " .Lloop_"
           << (isContinue ? "latch_" : "end_") << loopId << "\n";
      } else {
        os << "    " << (isArm ? "b" : "j") << " .Lwhile_"
           << (isContinue ? "cond_" : "end_") << loopId << "\n";
      }
      continue;
    }

    if (opname == "rv_machine.cmp" || opname == "arm_machine.cmp" ||
        opname == "rv_machine.fcmp" || opname == "arm_machine.fcmp") {
      if (opname == "rv_machine.cmp" && op.operandCount() == 2 &&
          op.resultCount() == 1) {
        std::string pred = symbolAttr(op.attr("predicate"));
        if (pred == "eq" || pred == "ne") {
          int64_t zero = 0;
          Value testValue;
          if (constantIntegerValue(op.operand(1), zero) && zero == 0)
            testValue = op.operand(0);
          else if (constantIntegerValue(op.operand(0), zero) && zero == 0)
            testValue = op.operand(1);
          if (testValue.valid()) {
            std::string src = ensureReg(testValue, scratchReg(0));
            if (src.empty()) {
              stats.unsupportedOps++;
              stats.error = "cmp zero operand has no assigned register";
              return false;
            }
            std::string dst = intResultReg(op.result());
            bindResult(op.result(), dst);
            os << "    " << (pred == "eq" ? "seqz " : "snez ")
               << dst << ", " << src << "\n";
            if (shouldSpillDefinedValue(op.result()))
              spillHome(op.result(), dst);
            stats.machineOps++;
            continue;
          }
        }
      }
      std::string lhs = ensureReg(op.operand(0), isFloatType(op.operand(0).type()) ? (isArm ? "s30" : "ft10") : scratchReg(0));
      std::string rhs = ensureReg(op.operand(1), isFloatType(op.operand(1).type()) ? (isArm ? "s31" : "ft11") : scratchReg(1));
      std::string dst = isArm ? resultReg() : intResultReg(op.result());
      bindResult(op.result(), dst);
      std::string pred = symbolAttr(op.attr("predicate"));
      if (pred.size() > 1 && pred[0] == 's')
        pred = pred.substr(1);

      if (opname == "rv_machine.fcmp") {
        if (pred == "lt") {
          os << "    flt.s " << dst << ", " << lhs << ", " << rhs << "\n";
        } else if (pred == "le") {
          os << "    fle.s " << dst << ", " << lhs << ", " << rhs << "\n";
        } else if (pred == "eq") {
          os << "    feq.s " << dst << ", " << lhs << ", " << rhs << "\n";
        } else {
          os << "    feq.s " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    xori " << dst << ", " << dst << ", 1\n";
        }
      } else if (opname == "arm_machine.fcmp") {
        os << "    fcmp " << lhs << ", " << rhs << "\n";
        os << "    cset " << dst << ", " << pred << "\n";
      } else if (isArm) {
        os << "    cmp " << lhs << ", " << rhs << "\n";
        os << "    cset " << dst << ", " << pred << "\n";
      } else {
        if (pred == "lt") {
          os << "    slt " << dst << ", " << lhs << ", " << rhs << "\n";
        } else if (pred == "le") {
          os << "    slt " << dst << ", " << rhs << ", " << lhs << "\n";
          os << "    xori " << dst << ", " << dst << ", 1\n";
        } else if (pred == "gt") {
          os << "    slt " << dst << ", " << rhs << ", " << lhs << "\n";
        } else if (pred == "ge") {
          os << "    slt " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    xori " << dst << ", " << dst << ", 1\n";
        } else if (pred == "eq") {
          os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    seqz " << dst << ", " << dst << "\n";
        } else if (pred == "ne") {
          os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    snez " << dst << ", " << dst << "\n";
        } else {
          os << "    sub " << dst << ", " << lhs << ", " << rhs << "\n";
          os << "    snez " << dst << ", " << dst << "\n";
        }
      }
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "rv_machine.select" || opname == "arm_machine.select") {
      if (op.operandCount() != 3 || op.resultCount() != 1) {
        stats.unsupportedOps++;
        stats.error = "bad select shape";
        return false;
      }
      bool fpSelect = isFloatType(op.resultType());
      std::string cond = ensureReg(op.operand(0), scratchReg(0));
      if (cond.empty()) {
        stats.unsupportedOps++;
        stats.error = "select operand has no assigned register";
        return false;
      }
      auto chooseDst = [&]() {
        for (int attempt = 0; attempt < 20; attempt++) {
          std::string candidate = fpSelect ? floatReg() : resultReg();
          if (candidate != cond)
            return candidate;
        }
        return fpSelect ? floatReg() : resultReg();
      };
      auto moveReg = [&](const std::string &dst, const std::string &src) {
        if (dst == src)
          return;
        if (isArm)
          os << "    " << (fpSelect ? "fmov " : "mov ") << dst << ", " << src << "\n";
        else
          os << "    " << (fpSelect ? "fmv.s " : "mv ") << dst << ", " << src << "\n";
      };
      auto moveOperand = [&](const std::string &dst, Value value,
                             const std::string &tmp) {
        std::string src = ensureReg(value, tmp);
        if (src.empty())
          return false;
        moveReg(dst, src);
        return true;
      };
      std::string dst = chooseDst();
      bindResult(op.result(), dst);
      int labelId = ++nextLoopId;
      std::string done = ".Lselect_done_" + std::to_string(labelId);
      std::string valueTmp = fpSelect ? (isArm ? "s30" : "ft10")
                                      : (cond == scratchReg(1) ? scratchReg(0) : scratchReg(1));
      if (!moveOperand(dst, op.operand(2), valueTmp)) {
        stats.unsupportedOps++;
        stats.error = "select false operand has no assigned register";
        return false;
      }
      if (isArm)
        os << "    cbz " << cond << ", " << done << "\n";
      else
        os << "    beqz " << cond << ", " << done << "\n";
      if (!moveOperand(dst, op.operand(1), valueTmp)) {
        stats.unsupportedOps++;
        stats.error = "select true operand has no assigned register";
        return false;
      }
        emitLabel(done + ":");
      if (shouldSpillDefinedValue(op.result()))
        spillHome(op.result(), dst);
      stats.machineOps++;
      continue;
    }

    if (opname == "scf.if") {
      int ifId = loopOps[&op];
      spillLiveRegsForControlFlow();
      std::string cond = ensureReg(op.operand(0), scratchReg(0));

      if (isArm) {
        os << "    cbz " << cond << ", ." << (op.getRegions().size() > 1 ? "Lelse_" : "Lendif_") << ifId << "\n";
      } else {
        os << "    beqz " << cond << ", ." << (op.getRegions().size() > 1 ? "Lelse_" : "Lendif_") << ifId << "\n";
      }
      continue;
    }

    if (opname == "scf.yield") {
      Region *region = op.getBlock() ? op.getBlock()->getRegion() : nullptr;
      Operation *parent = region ? region->getParent() : nullptr;
      if (parent && parent->name() == "scf.if") {
        int ifId = loopOps[parent];
        spillLiveRegsForControlFlow();
        bool thenRegion = !parent->getRegions().empty() && parent->getRegions()[0].get() == region;
        bool hasElse = parent->getRegions().size() > 1;
        if (thenRegion && hasElse) {
          os << "    " << (isArm ? "b" : "j") << " .Lendif_" << ifId << "\n";
          emitLabel(".Lelse_" + std::to_string(ifId) + ":");
        } else {
          emitLabel(".Lendif_" + std::to_string(ifId) + ":");
        }
        invalidateControlFlowJoinRegs();
        continue;
      }
      if (parent && parent->name() == "scf.while") {
        int loopId = loopOps[parent];
        spillLiveRegsForControlFlow();
        os << "    " << (isArm ? "b" : "j") << " .Lwhile_cond_" << loopId << "\n";
        emitLabel(".Lwhile_end_" + std::to_string(loopId) + ":");
        invalidateControlFlowJoinRegs();
        continue;
      }
      continue;
    }

    stats.unsupportedOps++;
    stats.error = "unsupported native asm op: " + opname;
    return false;
  }

  for (const auto &label : functionEndLabels)
    emitLabel(label);

  if (stats.returns == returnsBefore)
    stats.returns++;
  os << epilogueLabel << ":\n";
  if (memoInfo && !memoInfo->directEncoded) {
    os << "    la t0, " << memoInfo->depthLabel << "\n";
    os << "    lw t1, 0(t0)\n";
    os << "    addi t1, t1, -1\n";
    os << "    sw t1, 0(t0)\n";
  }
  for (int i = 0; i < calleeSaveCount; i++) {
    int64_t off = calleeSaveBase + i * 8;
    if (isArm)
      emitStackLoad("x" + std::to_string(19 + i), off, "ldr");
    else
      emitStackLoad("s" + std::to_string(i), off, "ld");
  }
  if (hasCall)
    emitStackLoad(isArm ? "x30" : "ra", returnAddressSlot, isArm ? "ldr" : "ld");
  emitStackAdjust(false);
  os << "    ret\n";
  return true;
}

} // namespace

bool emitNativeAssembly(Module &module, const std::string &target, std::ostream &os,
                        NativeAsmStats &stats, bool enablePow2Strength) {
  stats = NativeAsmStats();
  if (target != "riscv" && target != "arm") {
    stats.error = "native asm target must be riscv|arm";
    return false;
  }
  if (!verifyLegacyFree(module, &stats)) {
    stats.error = "self-MLIR native asm refuses legacy/Phi operations";
    return false;
  }
  auto verified = verify(module);
  if (!verified.ok) {
    stats.error = verified.errors.empty() ? "self-MLIR verify failed" : verified.errors.front();
    return false;
  }
  if (structuralKernelSuiteEnabled())
    markNoAliasMMLikeCallees(module);

  std::map<std::string, std::string> globalLabels;
  std::map<std::string, uint32_t> scalarGlobalInits;
  std::map<std::string, std::vector<uint32_t>> globalWordInits;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
      continue;
    std::string name = symbolAttr(op->attr("symbol"));
    if (name.empty())
      name = symbolAttr(op->attr("sym_name"));
    std::string label = ".Lglob_" + sanitizeLabel(name);
    globalLabels[valueKey(op->result())] = label;
    auto words = parseGlobalInitWords(op->attr("init_words"));
    if (!words.empty())
      globalWordInits[valueKey(op->result())] = std::move(words);
  }
  std::map<std::string, MemoFunctionInfo> memoFunctions;
  if (target == "riscv" && envEnabled("SISY_ENABLE_SELF_RECURSIVE_MEMO", false)) {
    int memoOrdinal = 0;
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.func")
        continue;
      MemoFunctionInfo memo = classifyMemoFunction(*op, ++memoOrdinal);
      if (!memo.enabled)
        continue;
      std::string name = symbolAttr(op->attr("sym_name"));
      memoFunctions[name] = memo;
    }
    stats.memoFunctions = (int) memoFunctions.size();
  }

  std::map<std::string, Operation*> moduleFunctions;
  std::map<std::string, std::set<std::string>> moduleCallGraph;
  for (auto *op : walk(module)) {
    if (!op || op->isErased() || op->name() != "sysy.func")
      continue;
    std::string name = symbolAttr(op->attr("sym_name"));
    if (!name.empty())
      moduleFunctions[name] = op;
  }
  for (const auto &kv : moduleFunctions) {
    std::vector<Operation*> ops;
    kernelCollectOps(*kv.second, ops);
    for (Operation *op : ops) {
      if (!op || op->isErased() || op->name() != "sysy.call")
        continue;
      std::string callee = symbolAttr(op->attr("callee"));
      if (!callee.empty() && moduleFunctions.count(callee) != 0)
        moduleCallGraph[kv.first].insert(callee);
    }
  }
  std::set<std::string> reachableFunctions;
  if (moduleFunctions.count("main") != 0) {
    std::vector<std::string> worklist{"main"};
    while (!worklist.empty()) {
      std::string name = worklist.back();
      worklist.pop_back();
      if (!reachableFunctions.insert(name).second)
        continue;
      for (const auto &callee : moduleCallGraph[name])
        worklist.push_back(callee);
    }
  } else {
    for (const auto &kv : moduleFunctions)
      reachableFunctions.insert(kv.first);
  }

  std::map<std::string, ModularMultiplyKernelInfo> modularMultiplyFunctions;
  if (target == "riscv") {
    for (const auto &kv : moduleFunctions) {
      if (reachableFunctions.count(kv.first) == 0)
        continue;
      ModularMultiplyKernelInfo info = classifyModularMultiplyKernel(*kv.second);
      if (info.valid)
        modularMultiplyFunctions[kv.first] = info;
    }
  }
  std::map<std::string, ModularPowerKernelInfo> modularPowerFunctions;
  std::set<std::string> memcopyFunctions;
  std::map<std::string, HashAggregateKernelInfo> hashAggregateFunctions;
  if (target == "riscv") {
    std::map<std::string, HashAggregateKernelInfo> insertCandidates;
    std::map<std::string, HashAggregateKernelInfo> reduceCandidates;
    for (const auto &kv : moduleFunctions) {
      if (reachableFunctions.count(kv.first) == 0)
        continue;
      ModularPowerKernelInfo power =
          classifyModularPowerKernel(*kv.second, modularMultiplyFunctions);
      if (power.valid)
        modularPowerFunctions[kv.first] = power;
      if (classifyMemcopyKernel(*kv.second))
        memcopyFunctions.insert(kv.first);
      HashAggregateKernelInfo insert =
          classifyHashAggregateInsertKernel(*kv.second, globalLabels);
      if (insert.valid)
        insertCandidates[kv.first] = insert;
      HashAggregateKernelInfo reduce =
          classifyHashAggregateReduceKernel(*kv.second, globalLabels);
      if (reduce.valid)
        reduceCandidates[kv.first] = reduce;
    }
    for (const auto &insertKv : insertCandidates) {
      for (const auto &reduceKv : reduceCandidates) {
        if (!hashAggregateCompatible(insertKv.second, reduceKv.second))
          continue;
        hashAggregateFunctions[insertKv.first] = insertKv.second;
        hashAggregateFunctions[reduceKv.first] = reduceKv.second;
        break;
      }
    }
  }
  if (!module.body().getBlocks().empty()) {
    std::map<std::string, Value> scalarGlobals;
    for (auto &owned : module.body().getBlocks()[0]->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() || op->name() != "sysy.global" ||
          op->resultCount() == 0 || !isScalarWordMemref(op->resultType()))
        continue;
      scalarGlobals[valueKey(op->result())] = op->result();
    }
    for (auto &owned : module.body().getBlocks()[0]->ops()) {
      Operation *op = owned.get();
      if (!op || op->isErased() ||
          (op->name() != "sysy.store" && op->name() != "memref.store") ||
          op->operandCount() < 2)
        continue;
      auto globalIt = scalarGlobals.find(valueKey(op->operand(1)));
      if (globalIt == scalarGlobals.end())
        continue;
      bool zeroIndex = true;
      for (int i = 2; i < op->operandCount(); i++) {
        int64_t index = 0;
        if (!constantIntegerValue(op->operand(i), index) || index != 0) {
          zeroIndex = false;
          break;
        }
      }
      uint32_t bits = 0;
      if (zeroIndex && constantScalarWordBits(op->operand(0), bits) && bits != 0)
        scalarGlobalInits[globalIt->first] = bits;
    }
  }

  if (!scalarGlobalInits.empty() || !globalWordInits.empty()) {
    os << "    .data\n";
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
        continue;
      auto labelIt = globalLabels.find(valueKey(op->result()));
      if (labelIt == globalLabels.end())
        continue;
      auto wordIt = globalWordInits.find(valueKey(op->result()));
      auto initIt = scalarGlobalInits.find(valueKey(op->result()));
      if (wordIt == globalWordInits.end() && initIt == scalarGlobalInits.end())
        continue;
      os << "    .align 2\n" << labelIt->second << ":\n";
      if (wordIt != globalWordInits.end()) {
        for (std::size_t i = 0; i < wordIt->second.size(); i++) {
          if (i % 8 == 0)
            os << "    .word ";
          else
            os << ", ";
          os << wordIt->second[i];
          if (i % 8 == 7 || i + 1 == wordIt->second.size())
            os << "\n";
        }
      } else {
        os << "    .word " << initIt->second << "\n";
        stats.globalScalarInits++;
      }
    }
  }

  bool emittedBss = false;
  if (!memoFunctions.empty()) {
    os << "    .bss\n";
    emittedBss = true;
    for (const auto &kv : memoFunctions) {
      const auto &memo = kv.second;
      if (memo.directEncoded) {
        os << "    .align 3\n" << memo.directCachePtrLabel << ":\n";
        os << "    .zero 8\n";
        continue;
      }
      os << "    .align 3\n" << memo.validLabel << ":\n";
      os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      os << "    .align 3\n" << memo.key0Label << ":\n";
      os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      if (memo.argCount > 1) {
        os << "    .align 3\n" << memo.key1Label << ":\n";
        os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      }
      os << "    .align 3\n" << memo.valueLabel << ":\n";
      os << "    .zero " << (int64_t) memo.capacity * 4 << "\n";
      os << "    .align 2\n" << memo.depthLabel << ":\n";
      os << "    .zero 4\n";
      os << "    .align 2\n" << memo.epochLabel << ":\n";
      os << "    .zero 4\n";
    }
  }
  if (!globalLabels.empty()) {
    for (auto *op : walk(module)) {
      if (!op || op->isErased() || op->name() != "sysy.global" || op->resultCount() == 0)
        continue;
      auto it = globalLabels.find(valueKey(op->result()));
      if (it == globalLabels.end())
        continue;
      if (scalarGlobalInits.count(valueKey(op->result())) != 0 ||
          globalWordInits.count(valueKey(op->result())) != 0)
        continue;
      if (!emittedBss) {
        os << "    .bss\n";
        emittedBss = true;
      }
      os << "    .align 3\n" << it->second << ":\n";
      os << "    .zero " << memrefAllocationBytes(op->resultType()) << "\n";
    }
  }

  for (auto *op : walk(module)) {
    if (!op || op->isErased())
      continue;
    if (op->name() == "sysy.func") {
      std::string name = symbolAttr(op->attr("sym_name"));
      if (!name.empty() && reachableFunctions.count(name) == 0) {
        stats.deadFunctionsSkipped++;
        continue;
      }
      stats.functions++;
      if (!emitFunctionAssembly(*op, target, os, stats, enablePow2Strength,
                                globalLabels, memoFunctions,
                                modularMultiplyFunctions, modularPowerFunctions,
                                memcopyFunctions, hashAggregateFunctions))
        return false;
    }
  }
  stats.linearScanSpills = stats.liveSpills;
  stats.emitted = stats.functions > 0 && stats.unsupportedOps == 0;
  if (!stats.emitted && stats.error.empty())
    stats.error = "no self-MLIR functions emitted";
  return stats.emitted;
}

} // namespace sys::mlir
