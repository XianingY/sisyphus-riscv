#include "Options.h"

#include <fstream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "../frontend/FrontendFacade.h"

#include "../mlir/ASTToMLIR.h"
#include "../mlir/SelfMLIR.h"
#include "../utils/smt/SMT.h"
#include "../parse/CompileError.h"

using namespace smt;

sys::Options opts;

static void setEnvIfPresent(const char *name, const std::string &value) {
  if (!value.empty())
    setenv(name, value.c_str(), 1);
}

static std::vector<std::string> splitPathList(const std::string &raw) {
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

static int thinLinkOnly(const sys::Options &opts) {
  std::ofstream os(opts.thinLinkOut);
  if (!os) {
    std::cerr << "cannot open thin-link output\n";
    return 1;
  }

  os << "# sisyphus thin linked summary v1\n";
  os << "# input_summaries=" << opts.thinSummaryIn << "\n";
  std::unordered_set<std::string> seen;
  for (const auto &path : splitPathList(opts.thinSummaryIn)) {
    std::ifstream is(path);
    if (!is) {
      std::cerr << "cannot open thin summary input: " << path << "\n";
      return 1;
    }
    std::string line;
    while (std::getline(is, line)) {
      if (line.empty() || line[0] == '#')
        continue;
      if (seen.insert(line).second)
        os << line << "\n";
    }
  }
  return 0;
}



void removeDuplicates(std::vector<Atomic>& clause) {
  std::sort(clause.begin(), clause.end());
  auto last = std::unique(clause.begin(), clause.end());
  clause.erase(last, clause.end());
}

void sat() {
  Solver solver;
  std::string line;
  std::getline(std::cin, line);
  while (line[0] == 'c')
    std::getline(std::cin, line);

  std::istringstream headerStream(line);
  int n, m;
  std::string dummy;
  headerStream >> dummy >> dummy >> m >> n;
  solver.init(m);

  for (int i = 0; i < n; ++i) {
    std::getline(std::cin, line);

    std::vector<Atomic> clause;
    std::istringstream lineStream(line);
    int lit;

    while (lineStream >> lit) {
      if (lit == 0)
        break;
      auto var = std::abs(lit) - 1;
      clause.push_back((Atomic) (lit < 0 ? var * 2 + 1 : var * 2));
    }

    removeDuplicates(clause);
    solver.addClause(clause);
  }
  std::vector<signed char> assignments;
  bool succ = solver.solve(assignments);
  if (!succ) {
    std::cout << "unsat\n";
    return;
  }

  std::cout << "sat\n";
  for (int i = 0; i < m; i++)
    std::cout << (i + 1) << " = " << (assignments[i] ? "true" : "false") << "\n";
}

void bv(const sys::Options &opts) {
  const auto &infer = [&](BvSolver &solver, BvExpr *x) {
    bool succ = solver.infer(x);
    if (succ) {
      std::cout << "sat\n";
      std::cout << "x = " << solver.extract("x") << "\n";
    } else std::cout << "unsat\n";
  };

  BvExprContext ctx;
  assert(ctx.create(BvExpr::Var, "x") == ctx.create(BvExpr::Var, "x"));

  // Test: x = (x == 1) ? 2x : x + 1
  // > unsat
  if (true) {
    BvSolver solver(opts);
    BvExprContext ctx;

    auto _1 = ctx.create(BvExpr::Const, 1);
    auto _2 = ctx.create(BvExpr::Const, 2);
    auto _3 = ctx.create(BvExpr::Var, "x");
    auto _4 = ctx.create(BvExpr::Eq, _3, _1);
    auto _5 = ctx.create(BvExpr::Add, _3, _1);
    auto _6 = ctx.create(BvExpr::Mul, _3, _2);
    auto _7 = ctx.create(BvExpr::Ite, _4, _6, _5);
    auto _8 = ctx.create(BvExpr::Eq, _3, _7);

    infer(solver, _8);
  }

  // Test: 1089 * 2256 = 74448 * (x - 16)
  // > sat, x = 1879048241 (signed wrap)
  // (Note that x = 49 is the obvious solution.)
  if (true) {
    BvSolver solver(opts);
    BvExprContext ctx;

    auto _1 = ctx.create(BvExpr::Var, "x");
    auto _2 = ctx.create(BvExpr::Const, 16);
    auto _3 = ctx.create(BvExpr::Const, 1089);
    auto _4 = ctx.create(BvExpr::Const, 2256);
    auto _5 = ctx.create(BvExpr::Const, 74448);
    auto _6 = ctx.create(BvExpr::Mul, _3, _4);
    auto _7 = ctx.create(BvExpr::Sub, _1, _2);
    auto _8 = ctx.create(BvExpr::Mul, _7, _5);
    auto _9 = ctx.create(BvExpr::Eq, _6, _8);

    infer(solver, _9);
  }

  // Test: 7 / x = -2
  // > sat, x = -3
  // Note: very expensive, ~0.2s
  if (true) {
    BvSolver solver(opts);
    BvExprContext ctx;

    auto _1 = ctx.create(BvExpr::Const, 7);
    auto _2 = ctx.create(BvExpr::Var, "x");
    auto _3 = ctx.create(BvExpr::Div, _1, _2);
    auto _4 = ctx.create(BvExpr::Const, -2);
    auto _5 = ctx.create(BvExpr::Eq, _4, _3);

    infer(solver, _5);
  }

  // Test: -9 % 2 == -1
  if (true) {
    BvSolver solver(opts);
    BvExprContext ctx;

    auto _1 = ctx.create(BvExpr::Const, -9);
    auto _2 = ctx.create(BvExpr::Const, 2);
    auto _3 = ctx.create(BvExpr::Var, "x");
    auto _4 = ctx.create(BvExpr::Mod, _1, _2);
    auto _5 = ctx.create(BvExpr::Eq, _3, _4);
    // _5 = simplify(_5, ctx);

    infer(solver, _5);
  }
}

int main(int argc, char **argv) {
  opts = sys::parseArgs(argc, argv);

  setEnvIfPresent("SISY_THIN_SUMMARY_DUMP", opts.thinSummaryOut);
  setEnvIfPresent("SISY_THIN_SUMMARY_IMPORT", opts.thinSummaryIn);
  setEnvIfPresent("SISY_THIN_LINK_OUT", opts.thinLinkOut);
  setEnvIfPresent("SISY_PROFILE_GENERATE", opts.profileGenerate);
  setEnvIfPresent("SISY_PROFILE", opts.profileUse.empty() ? opts.fdoUse : opts.profileUse);
  if (opts.enableRVV)
    setenv("SISY_ENABLE_RVV", "1", 1);
  if (opts.rv)
    setenv("SISY_TARGET_RISCV", "1", 1);
  if (opts.arm)
    setenv("SISY_TARGET_ARM", "1", 1);
  if (opts.disableSMTSynth)
    setenv("SISY_ENABLE_SMT_SYNTH", "0", 1);
  if (opts.dumpAnalysisCache)
    setenv("SISY_DUMP_ANALYSIS_CACHE", "1", 1);


  if (opts.runSelfMLIRCoreTests)
    return sys::mlir::runCoreSelfTest(std::cout);

  if (opts.runSelfMLIRConversionTests)
    return sys::mlir::runConversionSelfTest(std::cout);

  if (opts.runSelfMLIRNativeBackendTests)
    return sys::mlir::runNativeBackendSelfTest(std::cout);

  if (opts.dumpSelfMLIRSample) {
    sys::mlir::dumpSample(std::cout);
    return 0;
  }



  // Test for submodules: bitvector SMT solver, and CDCL SAT solver.
  if (opts.bv) {
    bv(opts);
    return 0;
  }
  if (opts.sat) {
    sat();
    return 0;
  }

  if (opts.inputFile.empty() && !opts.thinLinkOut.empty())
    return thinLinkOnly(opts);

  // Read input file.
  std::ifstream ifs(opts.inputFile);
  if (!ifs) {
    std::cerr << "cannot open file\n";
    return 1;
  }

  std::stringstream ss;
  ss << ifs.rdbuf() << "\n";

  sys::TypeContext ctx;
  sys::ASTNode *node = nullptr;
  try {
    sys::Parser parser(ss.str(), ctx);
    node = parser.parse();
    sys::Sema sema(node, ctx);
  } catch (const sys::CompileError &e) {
    if (node)
      delete node;
    std::cerr << e.what() << "\n";
    return 1;
  } catch (const std::exception &e) {
    if (node)
      delete node;
    std::cerr << "frontend error: " << e.what() << "\n";
    return 1;
  }

  sys::mlir::ProductionStats selfMLIRStats;
  std::ostringstream dumpBuffer;
  const std::string target = opts.arm ? "arm" : "riscv";

  sys::mlir::Context mlirCtx;
  auto module = sys::mlir::runProductionGateFromAST(mlirCtx, *node, target, selfMLIRStats, &dumpBuffer);

  if (!module) {
    std::cerr << "[stage-fail] file=" << opts.inputFile << " stage=self-mlir-production\n";
    std::cerr << "  - " << (selfMLIRStats.error.empty() ? "self-MLIR production gate failed" : selfMLIRStats.error) << "\n";
    std::cerr << "dump:\n" << dumpBuffer.str() << "\n";
    delete node;
    return 1;
  }

  if (opts.stats) {
    std::cerr << "[self-mlir] target=" << target << " source=ast frontend_path=self-mlir failed=0\n";
  }

  if (getenv("SISY_DUMP_SELF_MLIR")) {
    std::cerr << dumpBuffer.str();
  }

  if (opts.emitIR) {
    sys::mlir::print(*module, std::cerr);
  }

  // (removed opts.noLink early exit)

  sys::mlir::NativeAsmStats asmStats;
  std::ostringstream asmBuffer;
  bool asmOk = sys::mlir::emitNativeAssembly(*module, opts.arm ? "arm" : "riscv", asmBuffer, asmStats);

  if (!asmOk) {
    std::cerr << "[stage-fail] file=" << opts.inputFile << " stage=native-assembly\n";
    std::cerr << "  - " << (asmStats.error.empty() ? "emitNativeAssembly failed" : asmStats.error) << "\n";
    delete node;
    return 1;
  }

  if (opts.outputFile.empty()) {
    std::cout << asmBuffer.str();
  } else {
    std::ofstream ofs(opts.outputFile);
    if (!ofs) {
      std::cerr << "cannot open output file: " << opts.outputFile << "\n";
      delete node;
      return 1;
    }
    ofs << asmBuffer.str();
  }

  delete node;
  return 0;
}
