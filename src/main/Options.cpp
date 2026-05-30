#include "Options.h"
#include "DefaultTarget.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using namespace sys;

#define PARSEOPT(str, field) \
  if (strcmp(argv[i], str) == 0) { \
    opts.field = true; \
    continue; \
  }

Options::Options() {
  noLink = false;
  dumpAST = false;
  dumpMidIR = false;
  emitIR = false;
  o1 = false;
  o2 = false;
  arm = false;
  rv = false;
  verbose = false;
  stats = false;
  dumpPassTiming = false;
  verify = false;
  enableExperimental = false;
  disableO2Experimental = false;
  disableLoopRotate = false;
  disableConstUnroll = false;
  enableHIRPipeline = true;
  useLegacyCodegen = false;
  forceDialectCodegen = false;
  dumpHIR = false;
  dumpCFG = false;
  verifyHIR = true;
  verifyCFG = true;
  sat = false;
  bv = false;
  enableRVV = false;
  disableSMTSynth = false;
  dumpAnalysisCache = false;
  dumpOpDescriptors = false;
  dumpIRContext = false;
  dumpPassScopes = false;
  dumpDialectConversion = false;
  dumpBlockArguments = false;
  dumpOperationIR = false;
  verifyOperationBridge = false;
  inlineThreshold = 200;
  lateInlineThreshold = 200;
  inlineThresholdExplicit = false;
  lateInlineThresholdExplicit = false;
  loopRotateExplicit = false;
}

Options sys::parseArgs(int argc, char **argv) {
  Options opts;
  auto parsePositiveInt = [&](const char *raw, const char *optname) -> int {
    char *end = nullptr;
    long value = std::strtol(raw, &end, 10);
    if (!raw[0] || (end && *end) || value <= 0 || value > 1000000) {
      std::cerr << "error: " << optname << " expects a positive integer, got '" << raw << "'\n";
      exit(1);
    }
    return (int) value;
  };
  auto requireValue = [&](int i, const char *optname) -> const char * {
    if (i + 1 >= argc) {
      std::cerr << "error: " << optname << " requires value\n";
      exit(1);
    }
    return argv[i + 1];
  };

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--target") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "error: --target requires value arm|riscv\n";
        exit(1);
      }
      std::string target = argv[++i];
      if (target == "arm")
        opts.arm = true;
      else if (target == "riscv" || target == "rv")
        opts.rv = true;
      else {
        std::cerr << "error: unknown target '" << target << "'\n";
        exit(1);
      }
      continue;
    }

    if (strncmp(argv[i], "--target=", 9) == 0) {
      std::string target = argv[i] + 9;
      if (target == "arm")
        opts.arm = true;
      else if (target == "riscv" || target == "rv")
        opts.rv = true;
      else {
        std::cerr << "error: unknown target '" << target << "'\n";
        exit(1);
      }
      continue;
    }

    if (strcmp(argv[i], "-o") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "error: -o requires output file\n";
        exit(1);
      }
      opts.outputFile = argv[i + 1];
      i++;
      continue;
    }

    if (strcmp(argv[i], "--print-after") == 0) {
      opts.printAfter = requireValue(i, "--print-after");
      i++;
      continue;
    }

    if (strcmp(argv[i], "--print-before") == 0) {
      opts.printBefore = requireValue(i, "--print-before");
      i++;
      continue;
    }
    
    if (strcmp(argv[i], "--compare") == 0) {
      opts.compareWith = requireValue(i, "--compare");
      i++;
      continue;
    }

    if (strcmp(argv[i], "-i") == 0) {
      opts.simulateInput = requireValue(i, "-i");
      i++;
      continue;
    }

    if (strcmp(argv[i], "--inline-threshold") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "error: --inline-threshold requires value\n";
        exit(1);
      }
      opts.inlineThreshold = parsePositiveInt(argv[++i], "--inline-threshold");
      opts.inlineThresholdExplicit = true;
      continue;
    }

    if (strncmp(argv[i], "--inline-threshold=", 19) == 0) {
      opts.inlineThreshold = parsePositiveInt(argv[i] + 19, "--inline-threshold");
      opts.inlineThresholdExplicit = true;
      continue;
    }

    if (strcmp(argv[i], "--late-inline-threshold") == 0) {
      if (i + 1 >= argc) {
        std::cerr << "error: --late-inline-threshold requires value\n";
        exit(1);
      }
      opts.lateInlineThreshold = parsePositiveInt(argv[++i], "--late-inline-threshold");
      opts.lateInlineThresholdExplicit = true;
      continue;
    }

    if (strncmp(argv[i], "--late-inline-threshold=", 24) == 0) {
      opts.lateInlineThreshold = parsePositiveInt(argv[i] + 24, "--late-inline-threshold");
      opts.lateInlineThresholdExplicit = true;
      continue;
    }

    PARSEOPT("--dump-ast", dumpAST);
    PARSEOPT("--dump-mid-ir", dumpMidIR);
    PARSEOPT("--emit-ir", emitIR);
    PARSEOPT("--rv", rv);
    PARSEOPT("--arm", arm);
    if (strcmp(argv[i], "-O1") == 0) {
      opts.o1 = true;
      opts.o2 = false;
      continue;
    }
    if (strcmp(argv[i], "-O2") == 0) {
      opts.o1 = false;
      opts.o2 = true;
      continue;
    }
    if (strcmp(argv[i], "-O0") == 0) {
      opts.o1 = false;
      opts.o2 = false;
      continue;
    }
    PARSEOPT("-S", noLink);
    PARSEOPT("-v", verbose);
    PARSEOPT("--stats", stats);
    PARSEOPT("-s", stats);
    PARSEOPT("--verify", verify);
    PARSEOPT("--verify-ir", verify);
    PARSEOPT("--dump-pass-timing", dumpPassTiming);
    PARSEOPT("--dump-analysis-cache", dumpAnalysisCache);
    PARSEOPT("--dump-op-descriptors", dumpOpDescriptors);
    PARSEOPT("--dump-ir-context", dumpIRContext);
    PARSEOPT("--dump-pass-scopes", dumpPassScopes);
    PARSEOPT("--dump-dialect-conversion", dumpDialectConversion);
    PARSEOPT("--dump-block-arguments", dumpBlockArguments);
    PARSEOPT("--dump-operation-ir", dumpOperationIR);
    PARSEOPT("--verify-operation-bridge", verifyOperationBridge);
    PARSEOPT("--enable-experimental", enableExperimental);
    PARSEOPT("--disable-o2-experimental", disableO2Experimental);
    PARSEOPT("--enable-hir-pipeline", enableHIRPipeline);
    if (strcmp(argv[i], "--disable-hir-pipeline") == 0) {
      opts.enableHIRPipeline = false;
      opts.useLegacyCodegen = true;
      continue;
    }
    PARSEOPT("--use-legacy-codegen", useLegacyCodegen);
    PARSEOPT("--force-dialect-codegen", forceDialectCodegen);
    if (strcmp(argv[i], "--dialect-fallback-report") == 0) {
      opts.dialectFallbackReport = requireValue(i, "--dialect-fallback-report");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--dialect-fallback-report=", 26) == 0) {
      opts.dialectFallbackReport = argv[i] + 26;
      if (opts.dialectFallbackReport.empty()) {
        std::cerr << "error: --dialect-fallback-report requires stderr|<path>\n";
        exit(1);
      }
      continue;
    }
    if (strcmp(argv[i], "--thin-summary-out") == 0) {
      opts.thinSummaryOut = requireValue(i, "--thin-summary-out");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--thin-summary-out=", 19) == 0) {
      opts.thinSummaryOut = argv[i] + 19;
      continue;
    }
    if (strcmp(argv[i], "--thin-summary-in") == 0) {
      opts.thinSummaryIn = requireValue(i, "--thin-summary-in");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--thin-summary-in=", 18) == 0) {
      opts.thinSummaryIn = argv[i] + 18;
      continue;
    }
    if (strcmp(argv[i], "--thin-link-out") == 0) {
      opts.thinLinkOut = requireValue(i, "--thin-link-out");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--thin-link-out=", 16) == 0) {
      opts.thinLinkOut = argv[i] + 16;
      continue;
    }
    if (strcmp(argv[i], "--profile-generate") == 0) {
      opts.profileGenerate = requireValue(i, "--profile-generate");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--profile-generate=", 19) == 0) {
      opts.profileGenerate = argv[i] + 19;
      continue;
    }
    if (strcmp(argv[i], "--profile-use") == 0) {
      opts.profileUse = requireValue(i, "--profile-use");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--profile-use=", 14) == 0) {
      opts.profileUse = argv[i] + 14;
      continue;
    }
    if (strcmp(argv[i], "--fdo-use") == 0) {
      opts.fdoUse = requireValue(i, "--fdo-use");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--fdo-use=", 10) == 0) {
      opts.fdoUse = argv[i] + 10;
      continue;
    }
    if (strcmp(argv[i], "--pass-pipeline") == 0) {
      opts.passPipeline = requireValue(i, "--pass-pipeline");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--pass-pipeline=", 16) == 0) {
      opts.passPipeline = argv[i] + 16;
      if (opts.passPipeline.empty()) {
        std::cerr << "error: --pass-pipeline requires value\n";
        exit(1);
      }
      continue;
    }
    if (strcmp(argv[i], "--run-dialect-conversion") == 0) {
      opts.runDialectConversion = requireValue(i, "--run-dialect-conversion");
      i++;
      continue;
    }
    if (strncmp(argv[i], "--run-dialect-conversion=", 25) == 0) {
      opts.runDialectConversion = argv[i] + 25;
      if (opts.runDialectConversion.empty()) {
        std::cerr << "error: --run-dialect-conversion requires value\n";
        exit(1);
      }
      continue;
    }
    PARSEOPT("--dump-hir", dumpHIR);
    PARSEOPT("--dump-cfg", dumpCFG);
    PARSEOPT("--verify-hir", verifyHIR);
    PARSEOPT("--verify-cfg", verifyCFG);
    PARSEOPT("--enable-rvv", enableRVV);
    PARSEOPT("--disable-smt-synth", disableSMTSynth);
    if (strcmp(argv[i], "--disable-loop-rotate") == 0) {
      opts.disableLoopRotate = true;
      opts.loopRotateExplicit = true;
      continue;
    }
    if (strcmp(argv[i], "--enable-loop-rotate") == 0) {
      opts.disableLoopRotate = false;
      opts.loopRotateExplicit = true;
      continue;
    }
    PARSEOPT("--disable-const-unroll", disableConstUnroll);
    PARSEOPT("--bv", bv);
    PARSEOPT("--sat", sat);

    if (argv[i][0] == '-') {
      std::cerr << "error: unknown option '" << argv[i] << "'\n";
      exit(1);
    }

    if (opts.inputFile != "") {
      std::cerr << "error: multiple inputs\n";
      exit(1);
    }

    opts.inputFile = argv[i];
  }

  if (opts.rv && opts.arm) {
    std::cerr << "error: multiple target\n";
    exit(1);
  }

  if (!opts.rv && !opts.arm) {
    if (kDefaultTarget == DefaultTarget::Arm)
      opts.arm = true;
    else
      opts.rv = true;
  }

  if (opts.emitIR)
    opts.dumpMidIR = true;
  if (opts.useLegacyCodegen)
    opts.enableHIRPipeline = false;
  if (!opts.enableHIRPipeline)
    opts.useLegacyCodegen = true;
  if (opts.forceDialectCodegen && opts.useLegacyCodegen) {
    std::cerr << "error: --force-dialect-codegen conflicts with --use-legacy-codegen/--disable-hir-pipeline\n";
    exit(1);
  }

  if (!opts.inlineThresholdExplicit) {
    if (opts.o2)
      opts.inlineThreshold = 256;
    else
      opts.inlineThreshold = 200;
  }
  if (!opts.lateInlineThresholdExplicit) {
    if (opts.o2)
      opts.lateInlineThreshold = 256;
    else
      opts.lateInlineThreshold = 200;
  }
  if (!opts.loopRotateExplicit) {
    if (opts.o1 && !opts.rv)
      opts.disableLoopRotate = true;
  }

  if (opts.inputFile.empty() && !opts.bv && !opts.sat &&
      opts.thinLinkOut.empty() && !opts.dumpOpDescriptors &&
      !opts.dumpIRContext && !opts.dumpPassScopes &&
      !opts.dumpDialectConversion && !opts.dumpBlockArguments &&
      !opts.dumpOperationIR && !opts.verifyOperationBridge &&
      opts.runDialectConversion.empty()) {
    std::cerr
      << "usage: compiler <input.sy> -S -o <output.s> [-O0|-O1|-O2] [--target=riscv|arm]\n"
      << "       [--inline-threshold=N] [--late-inline-threshold=N]\n"
      << "       [--disable-o2-experimental]\n"
      << "       [--thin-summary-out=<path>] [--thin-summary-in=<path>] [--thin-link-out=<path>]\n"
      << "       [--profile-generate=<path>] [--profile-use=<path>] [--fdo-use=<path>]\n"
      << "       [--pass-pipeline=<comma-separated-passes>] [--dump-analysis-cache]\n"
      << "       [--dump-op-descriptors] [--dump-ir-context] [--dump-pass-scopes]\n"
      << "       [--dump-dialect-conversion] [--dump-block-arguments]\n"
      << "       [--dump-operation-ir] [--verify-operation-bridge]\n"
      << "       [--run-dialect-conversion=<legacy|rollback-test>]\n"
      << "       [--enable-rvv] [--disable-smt-synth]\n"
      << "       [--enable-hir-pipeline|--disable-hir-pipeline|--use-legacy-codegen|--force-dialect-codegen]\n"
      << "       [--dialect-fallback-report=stderr|<path>]\n"
      << "       [--dump-hir] [--dump-cfg] [--verify-hir] [--verify-cfg]\n"
      << "       [--disable-loop-rotate|--enable-loop-rotate] [--disable-const-unroll]\n"
      << "       compiler <input.sy> -S -o <output.s> --emit-ir --verify-ir\n";
    exit(1);
  }

  return opts;
}
