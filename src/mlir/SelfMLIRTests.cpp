#include "SelfMLIR.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace sys::mlir {

static const char *kCoreRules =
  "rule fold_addi_zero arith.addi addi-zero 10\n"
  "rule fold_muli_one arith.muli muli-one 9\n"
  "rule fold_select_same arith.select select-same 8\n"
  "rule fold_subi_same arith.subi subi-same 12\n"
  "rule fold_subi_zero arith.subi subi-zero 10\n"
  "rule fold_muli_zero arith.muli muli-zero 11\n"
  "rule fold_divi_one arith.divi divi-one 9\n"
  "rule fold_remi_one arith.remi remi-one 9\n"
  "rule fold_andi_same arith.andi andi-same 7\n"
  "rule fold_andi_zero arith.andi andi-zero 8\n"
  "rule fold_ori_same arith.ori ori-same 6\n"
  "rule fold_ori_zero arith.ori ori-zero 7\n"
  "rule fold_double_noti arith.noti double-noti 15\n";


static Module buildSample(Context &ctx) {
  Module module(ctx);
  auto &top = module.body().getBlocks()[0];
  Builder topBuilder(ctx, top.get());
  auto &func = topBuilder.create(
      "sysy.func", {}, {}, {{"sym_name", ctx.stringAttr("main")}},
      ctx.loc("sample.sy", 1, 1), 1);
  auto &entry = func.getRegions()[0]->addBlock();
  auto &arg = entry.addArgument(ctx.i(32), ctx.loc("sample.sy", 1, 12), "n");
  Builder b(ctx, &entry);
  auto &zero = b.create("arith.constant", {}, {ctx.i(32)},
                        {{"value", ctx.integerAttr(0, ctx.i(32))}},
                        ctx.loc("sample.sy", 2, 3));
  auto &add = b.create("arith.addi", {arg.value(), zero.result()}, {ctx.i(32)},
                       {}, ctx.loc("sample.sy", 2, 8));
  b.create("scf.return", {add.result()}, {}, {}, ctx.loc("sample.sy", 3, 1));
  return module;
}

static Module buildNativeAsmSample(Context &ctx) {
  Module module(ctx);
  auto &top = module.body().getBlocks()[0];
  Builder topBuilder(ctx, top.get());
  auto &func = topBuilder.create(
      "sysy.func", {}, {}, {{"sym_name", ctx.stringAttr("main")}},
      ctx.loc("native.sy", 1, 1), 1);
  auto &entry = func.getRegions()[0]->addBlock();
  Builder b(ctx, &entry);
  auto &seven = b.create("arith.constant", {}, {ctx.i(32)},
                         {{"value", ctx.integerAttr(7, ctx.i(32))}},
                         ctx.loc("native.sy", 2, 3));
  auto &one = b.create("arith.constant", {}, {ctx.i(32)},
                       {{"value", ctx.integerAttr(1, ctx.i(32))}},
                       ctx.loc("native.sy", 2, 7));
  auto &sum = b.create("arith.addi", {seven.result(), one.result()}, {ctx.i(32)},
                       {}, ctx.loc("native.sy", 2, 11));
  b.create("sysy.return", {sum.result()}, {}, {}, ctx.loc("native.sy", 3, 1));
  return module;
}

int runCoreSelfTest(std::ostream &os) {
  Context ctx;
  auto i32a = ctx.i(32);
  auto i32b = ctx.i(32);
  auto locA = ctx.loc("sample.sy", 1, 1);
  auto locB = ctx.loc("sample.sy", 1, 1);
  bool uniqued = i32a == i32b && locA == locB;
  Module module = buildSample(ctx);
  auto symtab = buildSymbolTable(module);
  Operation *mainFunc = symtab.lookup("main");
  Operation *addi = nullptr;
  Operation *zero = nullptr;
  for (auto *op : walk(module)) {
    if (!op)
      continue;
    if (op->name() == "arith.addi")
      addi = op;
    if (op->name() == "arith.constant")
      zero = op;
  }
  int addUsesBefore = addi ? (int) usesOf(module, addi->result()).size() : -1;
  int zeroUsesBefore = zero ? (int) usesOf(module, zero->result()).size() : -1;
  auto before = verify(module);
  std::vector<std::string> parseErrors;
  auto rules = parseDRR(kCoreRules, parseErrors);
  auto stats = applyGreedyPatterns(module, rules);
  auto after = verify(module);

  std::string eraseError;
  bool erasedDeadConstant = false;
  if (zero && usesOf(module, zero->result()).empty())
    erasedDeadConstant = eraseOperation(module, *zero, &eraseError);

  std::ostringstream printed;
  print(module, printed);
  std::vector<std::string> roundTripErrors;
  auto roundTrip = parse(ctx, printed.str(), roundTripErrors);
  auto roundTripVerify = roundTrip ? verify(*roundTrip) : VerifyResult{false, {"parse failed"}};
  auto roundTripSymbols = roundTrip ? buildSymbolTable(*roundTrip) : SymbolTable();

  Module mutation = buildNativeAsmSample(ctx);
  Operation *firstConst = nullptr;
  Operation *secondConst = nullptr;
  Operation *mutationAdd = nullptr;
  for (auto *op : walk(mutation)) {
    if (!op)
      continue;
    if (op->name() == "arith.constant") {
      if (!firstConst)
        firstConst = op;
      else if (!secondConst)
        secondConst = op;
    } else if (op->name() == "arith.addi") {
      mutationAdd = op;
    }
  }
  bool moved = firstConst && secondConst && moveOperationBefore(*secondConst, *firstConst);
  Operation *replacement = nullptr;
  if (mutationAdd) {
    auto repl = std::make_unique<Operation>(
        "arith.subi", mutationAdd->getOperands(), std::vector<Type>{mutationAdd->resultType()},
        std::map<std::string, Attribute>{}, mutationAdd->loc());
    replacement = replaceOperation(mutation, *mutationAdd, std::move(repl));
  }
  auto mutationVerify = verify(mutation);

  os << "[self-mlir-core] uniqued=" << (uniqued ? 1 : 0)
     << " types=" << ctx.typeCount()
     << " attrs=" << ctx.attrCount()
     << " locs=" << ctx.locationCount()
     << " block-args=1"
     << " symbols=" << symtab.all().size()
     << " main-symbol=" << (mainFunc ? 1 : 0)
     << " add-uses-before=" << addUsesBefore
     << " zero-uses-before=" << zeroUsesBefore
     << " erased-dead-const=" << (erasedDeadConstant ? 1 : 0)
     << " moved-op=" << (moved ? 1 : 0)
     << " replaced-op=" << (replacement ? 1 : 0)
     << " mutation-verify=" << (mutationVerify.ok ? 1 : 0)
     << " roundtrip-verify=" << (roundTripVerify.ok ? 1 : 0)
     << " roundtrip-errors=" << roundTripErrors.size()
     << " roundtrip-symbols=" << roundTripSymbols.all().size()
     << " rules=" << stats.rules
     << " rewrites=" << stats.rewrites
     << " verify-before=" << (before.ok ? 1 : 0)
     << " verify-after=" << (after.ok ? 1 : 0)
     << " parse-errors=" << parseErrors.size() << "\n";
  print(module, os);
  return uniqued && before.ok && after.ok && parseErrors.empty() &&
         stats.rewrites == 1 && mainFunc && symtab.duplicates().empty() &&
         addUsesBefore == 1 && zeroUsesBefore == 1 && erasedDeadConstant &&
         moved && replacement && mutationVerify.ok && roundTrip &&
         roundTripVerify.ok && roundTripErrors.empty() &&
         roundTripSymbols.lookup("main") ? 0 : 1;
}

int runConversionSelfTest(std::ostream &os) {
  Context ctx;
  Module rvModule = buildSample(ctx);
  ConversionTarget rvTarget;
  rvTarget.addLegalDialect("builtin");
  rvTarget.addLegalDialect("sysy");
  rvTarget.addLegalDialect("scf");
  rvTarget.addLegalDialect("rv_machine");
  auto rvStats = convertDialects(rvModule, rvTarget, {
      {"arith.constant", "rv_machine.li"},
      {"arith.addi", "rv_machine.addw"},
  });
  auto rvVerify = verify(rvModule);

  Module armModule = buildSample(ctx);
  ConversionTarget armTarget;
  armTarget.addLegalDialect("builtin");
  armTarget.addLegalDialect("sysy");
  armTarget.addLegalDialect("scf");
  armTarget.addLegalDialect("arm_machine");
  auto armStats = convertDialects(armModule, armTarget, {
      {"arith.constant", "arm_machine.mov"},
      {"arith.addi", "arm_machine.add"},
  });
  auto armVerify = verify(armModule);

  Module rollbackModule = buildSample(ctx);
  ConversionTarget strictTarget;
  strictTarget.addLegalDialect("builtin");
  auto rollbackStats = convertDialects(rollbackModule, strictTarget, {});
  auto rollbackVerify = verify(rollbackModule);

  os << "[self-mlir-conversion] rv-converted=" << rvStats.converted
     << " rv-failed=" << rvStats.failed
     << " rv-verify=" << (rvVerify.ok ? 1 : 0)
     << " arm-converted=" << armStats.converted
     << " arm-failed=" << armStats.failed
     << " arm-verify=" << (armVerify.ok ? 1 : 0)
     << " rollback-failed=" << rollbackStats.failed
     << " rollback-count=" << rollbackStats.rollbacks
     << " rollback-verify=" << (rollbackVerify.ok ? 1 : 0) << "\n";
  print(rvModule, os);
  print(armModule, os);
  return rvStats.converted == 2 && rvStats.failed == 0 && rvVerify.ok &&
         armStats.converted == 2 && armStats.failed == 0 && armVerify.ok &&
         rollbackStats.failed == 1 && rollbackStats.rollbacks == 1 &&
         rollbackVerify.ok ? 0 : 1;
}

int runNativeBackendSelfTest(std::ostream &os) {
  Context ctx;
  Module rvModule = buildNativeAsmSample(ctx);
  ConversionTarget rvTarget;
  rvTarget.addLegalDialect("builtin");
  rvTarget.addLegalDialect("sysy");
  rvTarget.addLegalDialect("rv_machine");
  auto rvConv = convertDialects(rvModule, rvTarget, {
      {"arith.constant", "rv_machine.li"},
      {"arith.addi", "rv_machine.addw"},
  });
  NativeAsmStats rvStats;
  std::ostringstream rvAsm;
  bool rvOk = emitNativeAssembly(rvModule, "riscv", rvAsm, rvStats);

  Module armModule = buildNativeAsmSample(ctx);
  ConversionTarget armTarget;
  armTarget.addLegalDialect("builtin");
  armTarget.addLegalDialect("sysy");
  armTarget.addLegalDialect("arm_machine");
  auto armConv = convertDialects(armModule, armTarget, {
      {"arith.constant", "arm_machine.mov"},
      {"arith.addi", "arm_machine.add"},
  });
  NativeAsmStats armStats;
  std::ostringstream armAsm;
  bool armOk = emitNativeAssembly(armModule, "arm", armAsm, armStats);

  os << "[self-mlir-native-backend]"
     << " rv-converted=" << rvConv.converted
     << " rv-failed=" << rvConv.failed
     << " rv-emitted=" << (rvOk ? 1 : 0)
     << " rv-machine-ops=" << rvStats.machineOps
     << " rv-unsupported=" << rvStats.unsupportedOps
     << " arm-converted=" << armConv.converted
     << " arm-failed=" << armConv.failed
     << " arm-emitted=" << (armOk ? 1 : 0)
     << " arm-machine-ops=" << armStats.machineOps
     << " arm-unsupported=" << armStats.unsupportedOps
     << " legacy-free=" << ((rvStats.legacyOps == 0 && rvStats.phiLikeOps == 0 &&
                              armStats.legacyOps == 0 && armStats.phiLikeOps == 0) ? 1 : 0)
     << "\n";
  os << "[self-mlir-native-backend-rv-asm]\n" << rvAsm.str();
  os << "[self-mlir-native-backend-arm-asm]\n" << armAsm.str();

  return rvOk && armOk && rvConv.failed == 0 && armConv.failed == 0 &&
         rvStats.legacyOps == 0 && rvStats.phiLikeOps == 0 &&
         armStats.legacyOps == 0 && armStats.phiLikeOps == 0 &&
         rvAsm.str().find("addw") != std::string::npos &&
         armAsm.str().find("add ") != std::string::npos ? 0 : 1;
}

void dumpSample(std::ostream &os) {
  Context ctx;
  Module module = buildSample(ctx);
  print(module, os);
}


} // namespace sys::mlir
