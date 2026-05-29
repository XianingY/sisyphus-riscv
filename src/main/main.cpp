#include "Options.h"
#include "PipelineProfiles.h"
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <iostream>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "../frontend/FrontendFacade.h"
#include "../cfg/CFGOps.h"
#include "../cfg/HIRToCFG.h"
#include "../cfg/CFGVerifier.h"
#include "../cfg/CFGLegality.h"
#include "../cfg/CFGToLegacy.h"
#include "../hir/HIRBuilder.h"
#include "../hir/HIRVerifier.h"
#include "../hir/HIRCanonicalize.h"
#include "../hir/HIRPolyhedral.h"
#include "../pass/PassRegistry.h"
#include "../codegen/Ops.h"
#include "../codegen/Attrs.h"
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
  // Add a newline at the end.
  // Single-line comments cannot terminate with EOF.
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

  std::unique_ptr<sys::CodeGen> cg;
  std::unique_ptr<sys::ModuleOp> loweredModule;
  std::unique_ptr<sys::hir::Module> hirModule;
  std::unique_ptr<sys::cfg::Module> cfgModule;

  // Captured from the HIR polyhedral stage and forwarded into
  // PipelineMetrics so selectPlan / appendCoreO1 can react to the
  // outcome (e.g. skip a redundant SCEV+GVN cleanup when nothing fired).
  sys::hir::PolyhedralStats hirPolyStats;
  bool hirPolyRan = false;

  auto runStage = [&](const char *stageName, auto &&fn) {
    auto start = std::chrono::steady_clock::now();
    fn();
    if (opts.dumpPassTiming) {
      auto end = std::chrono::steady_clock::now();
      double ms = std::chrono::duration<double, std::milli>(end - start).count();
      std::cerr << "[stage-timing] " << stageName << " : " << ms << " ms\n";
    }
  };
  auto stageDumpPath = [&](const std::string &stage, const std::string &payload) {
    auto path = "/tmp/sisyphus-" + stage + ".dump";
    std::ofstream ofs(path);
    ofs << payload;
    return path;
  };
  auto failStage = [&](const std::string &stage, const std::vector<std::string> &errors, const std::string &payload) {
    auto dumpPath = stageDumpPath(stage, payload);
    std::cerr << "[stage-fail] file=" << opts.inputFile
              << " stage=" << stage
              << " dump=" << dumpPath << "\n";
    for (const auto &e : errors)
      std::cerr << "  - " << e << "\n";
    std::exit(1);
  };

  bool dumpHIR = opts.dumpHIR;
  if (const char *env = std::getenv("SISY_DUMP_HIR"))
    dumpHIR = dumpHIR || (env[0] && std::strcmp(env, "0") != 0);
  bool dumpCFG = opts.dumpCFG;
  if (const char *env = std::getenv("SISY_DUMP_CFG"))
    dumpCFG = dumpCFG || (env[0] && std::strcmp(env, "0") != 0);
  auto emitDialectReport = [&](const std::string &frontendPath, const std::vector<std::string> &reasons) {
    if (opts.dialectFallbackReport.empty())
      return;

    std::ostringstream body;
    body << "frontend_path=" << frontendPath << "\n";
    body << "fallback_reason_codes=";
    if (reasons.empty())
      body << "none";
    else {
      for (size_t i = 0; i < reasons.size(); i++) {
        if (i)
          body << ",";
        body << reasons[i];
      }
    }
    body << "\n";
    body << "force_dialect_codegen=" << (opts.forceDialectCodegen ? 1 : 0) << "\n";
    body << "input_file=" << opts.inputFile << "\n";

    if (opts.dialectFallbackReport == "stderr") {
      std::istringstream iss(body.str());
      std::string line;
      while (std::getline(iss, line))
        std::cerr << "[dialect-report] " << line << "\n";
      return;
    }

    std::ofstream ofs(opts.dialectFallbackReport);
    ofs << body.str();
  };
  auto requiresLegacyFallback = [&](const sys::hir::Module &mod, std::vector<std::string> &reasons) {
    reasons.clear();
    std::unordered_set<std::string> seen;
    std::vector<const sys::hir::Op*> stack;
    if (mod.root)
      stack.push_back(mod.root.get());

    while (!stack.empty()) {
      auto *op = stack.back();
      stack.pop_back();
      if (!op)
        continue;

      for (const auto &child : op->children)
        if (child)
          stack.push_back(child.get());
    }
    return !reasons.empty();
  };

  std::string frontendPath = "dialect";
  std::vector<std::string> fallbackReasons;
  if (opts.useLegacyCodegen) {
    frontendPath = "legacy-forced";
    emitDialectReport(frontendPath, fallbackReasons);
    cg = std::make_unique<sys::CodeGen>(node);
  } else {
    runStage("hir.build", [&]() {
      sys::hir::Builder builder;
      hirModule = std::make_unique<sys::hir::Module>(builder.build(node));
    });
    if (dumpHIR) {
      std::cerr << "===== HIR =====\n";
      sys::hir::dump(*hirModule, std::cerr);
    }

    if (opts.verifyHIR) {
      runStage("hir.verify.pre", [&]() {
        std::vector<std::string> errors;
        if (!sys::hir::verify(*hirModule, errors)) {
          std::ostringstream os;
          sys::hir::dump(*hirModule, os);
          failStage("hir-verify-pre", errors, os.str());
        }
      });
    }

    runStage("hir.canonicalize", [&]() {
      sys::hir::Canonicalizer canonicalizer;
      auto stats = canonicalizer.run(*hirModule);
      if (opts.verbose || opts.stats) {
            std::cerr << "[hir] const_folded=" << stats.constFolded
                  << " dead_branches=" << stats.deadBranchesEliminated
                  << " simple-affine-calls-inlined=" << stats.simpleAffineCallsInlined
                  << "\n";
      }
    });

    if (!cg && opts.rv) {
      bool enableHIRPolyhedral = true;
      if (const char *env = std::getenv("SISY_DISABLE_HIR_POLYHEDRAL"))
        enableHIRPolyhedral = !(env[0] && std::strcmp(env, "0") != 0 &&
                                std::strcmp(env, "false") != 0 &&
                                std::strcmp(env, "FALSE") != 0);
      if (enableHIRPolyhedral) {
        runStage("hir.polyhedral", [&]() {
          sys::hir::PolyhedralOptimizer polyhedral;
          auto stats = polyhedral.run(*hirModule);
          hirPolyStats = stats;
          hirPolyRan = true;
          if (const char *dumpAfterPoly = std::getenv("SISY_DUMP_HIR_AFTER_POLY")) {
            if (dumpAfterPoly[0] && std::strcmp(dumpAfterPoly, "0") != 0) {
              std::cerr << "===== HIR after polyhedral =====\n";
              sys::hir::dump(*hirModule, std::cerr);
            }
          }
          if (opts.verbose || opts.stats) {
            std::cerr << "[hir-poly] reduction-jammed=" << stats.reductionJammed
                      << " reduction-privatized=" << stats.reductionPrivatized
                      << " reduction-rowjam-conditional=" << stats.reductionRowJamConditional
                      << " reduction-microtiled=" << stats.reductionMicroTiled
                      << " reduction-microtile-inplace=" << stats.reductionMicroTileInPlace
                      << " reduction-microtile-conditional=" << stats.reductionMicroTileConditional
                      << " reduction-microtile-3d=" << stats.reductionMicroTile3D
                      << " reduction-microtile-reject-dep=" << stats.reductionMicroTileRejectDependence
                      << " reduction-microtile-reject-pressure=" << stats.reductionMicroTileRejectPressure
                      << " microtile-mr-sum=" << stats.microTileMrSum
                      << " microtile-nr-sum=" << stats.microTileNrSum
                      << " microtile-kc-sum=" << stats.microTileKcSum
                      << " microtile-nc-sum=" << stats.microTileNcSum
                      << " reduction-interchanged=" << stats.reductionInterchanged
                      << " conditional-reduction-interchanged=" << stats.conditionalReductionInterchanged
                      << " repeat-reduced=" << stats.repeatReduced
                      << " repeat-rejected=" << stats.repeatRejected
                      << " repeat-reject-shape=" << stats.repeatRejectShape
                      << " repeat-reject-init=" << stats.repeatRejectInit
                      << " repeat-reject-bound=" << stats.repeatRejectBound
                      << " repeat-reject-legal=" << stats.repeatRejectLegal
                      << " repeat-reject-clone=" << stats.repeatRejectClone
                      << " overwrite-repeat-reduced=" << stats.overwriteRepeatReduced
                      << " overwrite-repeat-rejected=" << stats.overwriteRepeatRejected
                      << " rejected=" << stats.rejected
                      << " tiling-applied=" << stats.tilingApplied
                      << " tiling-rejected=" << stats.tilingRejected
                      << " tiling-reject-shape=" << stats.tilingRejectShape
                      << " tiling-reject-control=" << stats.tilingRejectControl
                      << " tiling-reject-bound-write=" << stats.tilingRejectBoundWrite
                      << " tiling-reject-affine-access=" << stats.tilingRejectAffineAccess
                      << " tiling-reject-no-inner=" << stats.tilingRejectNoInner
                      << " tiling-reject-idempotent=" << stats.tilingRejectIdempotent
                      << " interchange-applied=" << stats.interchangeApplied
                      << " interchange-rejected=" << stats.interchangeRejected
                      << " interchange-reject-shape=" << stats.interchangeRejectShape
                      << " interchange-reject-init=" << stats.interchangeRejectInit
                      << " interchange-reject-bounds=" << stats.interchangeRejectBounds
                      << " interchange-reject-control=" << stats.interchangeRejectControl
                      << " interchange-reject-access=" << stats.interchangeRejectAccess
                      << " interchange-reject-memory=" << stats.interchangeRejectMemory
                      << " interchange-3d-applied=" << stats.interchange3DApplied
                      << " interchange-3d-rejected=" << stats.interchange3DRejected
                      << " interchange-3d-reject-shape=" << stats.interchange3DRejectShape
                      << " interchange-3d-reject-init=" << stats.interchange3DRejectInit
                      << " interchange-3d-reject-bounds=" << stats.interchange3DRejectBounds
                      << " interchange-3d-reject-control=" << stats.interchange3DRejectControl
                      << " interchange-3d-reject-access=" << stats.interchange3DRejectAccess
                      << " interchange-3d-reject-memory=" << stats.interchange3DRejectMemory
                      << " unroll-jammed=" << stats.unrollJammed
                      << " unroll-jam-rejected=" << stats.unrollJamRejected
                      << " unroll-jam-reject-shape=" << stats.unrollJamRejectShape
                      << " unroll-jam-reject-init=" << stats.unrollJamRejectInit
                      << " unroll-jam-reject-bounds=" << stats.unrollJamRejectBounds
                      << " unroll-jam-reject-control=" << stats.unrollJamRejectControl
                      << " unroll-jam-reject-access=" << stats.unrollJamRejectAccess
                      << " unroll-jam-reject-memory=" << stats.unrollJamRejectMemory
                      << " partial-unrolled=" << stats.partialUnrolled
                      << " partial-unroll-rejected=" << stats.partialUnrollRejected
                      << " partial-unroll-reject-shape=" << stats.partialUnrollRejectShape
                      << " partial-unroll-reject-control=" << stats.partialUnrollRejectControl
                      << " partial-unroll-reject-access=" << stats.partialUnrollRejectAccess
                      << " fusion-applied=" << stats.fusionApplied
                      << " fusion-rejected=" << stats.fusionRejected
                      << " fusion-reject-shape=" << stats.fusionRejectShape
                      << " fusion-reject-init=" << stats.fusionRejectInit
                      << " fusion-reject-bounds=" << stats.fusionRejectBounds
                      << " fusion-reject-control=" << stats.fusionRejectControl
                      << " fusion-reject-scalar=" << stats.fusionRejectScalar
                      << " fusion-reject-memory=" << stats.fusionRejectMemory
                      << " forwarded-array-store-load=" << stats.forwardedArrayStoreLoads
                      << " transpose-forwarded-loads=" << stats.transposeForwardedLoads
                      << " presburger-fusion-queries=" << stats.presburgerFusionQueries
                      << " presburger-fusion-no-deps=" << stats.presburgerFusionNoDeps
                      << " presburger-fusion-may-deps=" << stats.presburgerFusionMayDeps
                      << " presburger-fusion-unknown=" << stats.presburgerFusionUnknown
                      << " presburger-interchange-queries=" << stats.presburgerInterchangeQueries
                      << " presburger-interchange-no-deps=" << stats.presburgerInterchangeNoDeps
                      << " presburger-interchange-may-deps=" << stats.presburgerInterchangeMayDeps
                      << " presburger-interchange-unknown=" << stats.presburgerInterchangeUnknown
                      << " presburger-projected-dims=" << stats.presburgerProjectedDims
                      << " presburger-unknown-budget=" << stats.presburgerUnknownBudget
                      << " affine-nest-candidates=" << stats.affineNestCandidates
                      << " affine-nest-reject-shape=" << stats.affineNestRejectedShape
                      << " affine-nest-reject-control=" << stats.affineNestRejectedControl
                      << " affine-nest-reject-access=" << stats.affineNestRejectedAccess
                      << " affine-nest-perfect-2d=" << stats.affineNestPerfect2D
                      << " affine-nest-perfect-3d=" << stats.affineNestPerfect3D
                      << " matmul-like-candidates=" << stats.matmulLikeCandidates
                      << " guarded-scop-candidates=" << stats.guardedScopCandidates
                      << " guarded-scop-applied=" << stats.guardedScopApplied
                      << " guarded-scop-rejected=" << stats.guardedScopRejected
                      << " guarded-scop-reject-pressure=" << stats.guardedScopRejectPressure
                      << " guarded-scop-symbolic=" << stats.guardedScopSymbolic
                      << " symbolic-affine-accesses=" << stats.symbolicAffineAccesses
                      << " symbolic-affine-reject-invariant=" << stats.symbolicAffineRejectInvariant
                      << " symbolic-affine-cancelled=" << stats.symbolicAffineCancelled
                      << " symbolic-presburger-fallback=" << stats.symbolicPresburgerFallback
                      << " stencil-interior-dispatched=" << stats.stencilInteriorDispatched
                      << " stencil-interior-rejected=" << stats.stencilInteriorRejected
                      << " stencil-interior-reject-shape=" << stats.stencilInteriorRejectShape
                      << " stencil-interior-reject-bounds=" << stats.stencilInteriorRejectBounds
                      << " row-stencil-interior-dispatched=" << stats.rowStencilInteriorDispatched
                      << " row-stencil-interior-rejected=" << stats.rowStencilInteriorRejected
                      << " row-stencil-reject-shape=" << stats.rowStencilRejectShape
                      << " row-stencil-reject-bounds=" << stats.rowStencilRejectBounds
                      << " row-stencil-reject-pressure=" << stats.rowStencilRejectPressure
                      << " invariant-guard-hoisted=" << stats.invariantGuardHoisted
                      << " invariant-guard-rejected=" << stats.invariantGuardRejected
                      << " monotone-guard-tightened=" << stats.monotoneGuardTightened
                      << " monotone-guard-rejected=" << stats.monotoneGuardRejected
                      << " monotone-guard-reject-shape=" << stats.monotoneGuardRejectShape
                      << " monotone-guard-reject-use=" << stats.monotoneGuardRejectUse
                      << " loop-distribution-applied=" << stats.loopDistributionApplied
                      << " loop-distribution-rejected=" << stats.loopDistributionRejected
                      << " loop-distribution-reject-shape=" << stats.loopDistributionRejectShape
                      << " loop-distribution-reject-control=" << stats.loopDistributionRejectControl
                      << " loop-distribution-reject-no-split=" << stats.loopDistributionRejectNoSplit
                      << " matmul-tail-collapsed=" << stats.matmulTailCollapsed
                      << " matmul-tail-rejected=" << stats.matmulTailRejected
                      << " matmul-tail-reject-shape=" << stats.matmulTailRejectShape
                      << " matmul-tail-reject-use=" << stats.matmulTailRejectUse
                      << "\n";
          }
        });
      }
    }

    if (opts.verifyHIR) {
      runStage("hir.verify.post", [&]() {
        std::vector<std::string> errors;
        if (!sys::hir::verify(*hirModule, errors)) {
          std::ostringstream os;
          sys::hir::dump(*hirModule, os);
          failStage("hir-verify-post", errors, os.str());
        }
      });
    }

    if (requiresLegacyFallback(*hirModule, fallbackReasons)) {
      if (opts.forceDialectCodegen) {
        frontendPath = "dialect";
        emitDialectReport(frontendPath, fallbackReasons);
        failStage("dialect-forced-fallback", fallbackReasons, "");
      }
      if (opts.verbose || opts.stats) {
        std::cerr << "[dialect-fallback] switch to legacy codegen due to:\n";
        for (const auto &r : fallbackReasons)
          std::cerr << "  - " << r << "\n";
      }
      frontendPath = "legacy-fallback";
      emitDialectReport(frontendPath, fallbackReasons);
      cg = std::make_unique<sys::CodeGen>(node);
    } else {
      frontendPath = "dialect";
      emitDialectReport(frontendPath, fallbackReasons);
    }

    if (!cg) {
      runStage("hir.to-cfg", [&]() {
        std::vector<std::string> errors;
        cfgModule = std::make_unique<sys::cfg::Module>(sys::cfg::lowerFromHIR(*hirModule, errors));
        if (!errors.empty()) {
          std::ostringstream os;
          if (cfgModule)
            sys::cfg::dump(*cfgModule, os);
          failStage("hir-to-cfg", errors, os.str());
        }
      });
      if (dumpCFG) {
        std::cerr << "===== CFG =====\n";
        sys::cfg::dump(*cfgModule, std::cerr);
      }

      if (opts.verifyCFG) {
        runStage("cfg.legality", [&]() {
          std::vector<std::string> errors;
          if (!sys::cfg::verifyHIRToCFGConversion(*hirModule, *cfgModule, errors)) {
            std::ostringstream os;
            sys::cfg::dump(*cfgModule, os);
            failStage("cfg-legality", errors, os.str());
          }
        });
        runStage("cfg.verify", [&]() {
          std::vector<std::string> errors;
          if (!sys::cfg::verify(*cfgModule, errors)) {
            std::ostringstream os;
            sys::cfg::dump(*cfgModule, os);
            failStage("cfg-verify", errors, os.str());
          }
        });
      }

      runStage("cfg.to-legacy", [&]() {
        std::vector<std::string> errors;
        loweredModule = sys::cfg::lowerToLegacyIR(*cfgModule, errors);
        if (!loweredModule || !errors.empty()) {
          std::ostringstream os;
          if (cfgModule)
            sys::cfg::dump(*cfgModule, os);
          failStage("cfg-to-legacy", errors, os.str());
        }
      });
    }
  }
  delete node;

  sys::ModuleOp *module = cg ? cg->getModule() : loweredModule.get();
  sys::pipeline::PipelineMetrics metrics;
  if (module) {
    std::function<void(sys::Region*, int)> walkRegion;
    std::function<void(sys::Op*, int)> walkOp;

    walkOp = [&](sys::Op *op, int loopDepth) {
      if (!op)
        return;
      metrics.moduleOpCount++;
      int nestedDepth = loopDepth +
                        ((sys::isa<sys::ForOp>(op) || sys::isa<sys::WhileOp>(op)) ? 1 : 0);
      if (nestedDepth > metrics.maxLoopDepth)
        metrics.maxLoopDepth = nestedDepth;
      for (auto *region : op->getRegions())
        walkRegion(region, nestedDepth);
    };

    walkRegion = [&](sys::Region *region, int loopDepth) {
      if (!region)
        return;
      for (auto *bb : region->getBlocks()) {
        if (!bb)
          continue;
        metrics.blockCount++;
        for (auto *op : bb->getOps()) {
          if (sys::isa<sys::PhiOp>(op))
            metrics.phiCount++;
          if (sys::isa<sys::GetArgOp>(op)) {
            metrics.getArgCount++;
            auto *idx = op->find<sys::IntAttr>();
            if (idx)
              metrics.maxGetArgArity = std::max(metrics.maxGetArgArity, idx->value + 1);
          }
          if (sys::isa<sys::CallOp>(op) || sys::isa<sys::CloneOp>(op) ||
              sys::isa<sys::JoinOp>(op))
            metrics.callLikeCount++;
          if (bb->getOpCount() > 0 && op == bb->getLastOp()) {
            if (sys::isa<sys::BranchOp>(op))
              metrics.cfgEdgeCount += 2;
            else if (sys::isa<sys::GotoOp>(op))
              metrics.cfgEdgeCount += 1;
          }
          walkOp(op, loopDepth);
        }
      }
    };

    for (auto *func : module->findAll<sys::FuncOp>()) {
      if (func->has<sys::ArgCountAttr>())
        metrics.maxGetArgArity =
            std::max(metrics.maxGetArgArity, func->get<sys::ArgCountAttr>()->count);
      walkRegion(func->getRegion(), 0);
    }
  }

  // Forward HIR polyhedral outcome into pipeline metrics so the planner
  // and the SSA pipeline can adjust cleanup intensity. Only meaningful
  // when the polyhedral stage actually ran (RV path with the env knob on).
  if (hirPolyRan) {
    int applied =
      hirPolyStats.reductionJammed +
      hirPolyStats.reductionInterchanged +
      hirPolyStats.conditionalReductionInterchanged +
      hirPolyStats.repeatReduced +
      hirPolyStats.overwriteRepeatReduced +
      hirPolyStats.tilingApplied +
      hirPolyStats.interchangeApplied +
      hirPolyStats.interchange3DApplied +
      hirPolyStats.unrollJammed +
      hirPolyStats.fusionApplied +
      hirPolyStats.loopDistributionApplied +
      hirPolyStats.stencilInteriorDispatched +
      hirPolyStats.invariantGuardHoisted +
      hirPolyStats.monotoneGuardTightened +
      hirPolyStats.forwardedArrayStoreLoads;
    int rejected =
      hirPolyStats.affineNestRejectedShape +
      hirPolyStats.affineNestRejectedControl +
      hirPolyStats.affineNestRejectedAccess;
    metrics.hirPolyApplied = applied;
    metrics.hirPolyTilingApplied = hirPolyStats.tilingApplied;
    metrics.hirPolyAffineRejected = rejected;
  }

  if (opts.dumpMidIR) {
    if (opts.emitIR)
      std::cerr << "===== Initial IR =====\n";
    std::cerr << module;
  }

  sys::PassManager pm(module, opts);
  auto plan = sys::pipeline::configurePipeline(pm, opts, metrics);
  if (const char *env = std::getenv("SISY_DUMP_PIPELINE_PROFILE")) {
    if (env[0] && std::strcmp(env, "0") != 0) {
      std::cerr << "[pipeline] " << sys::pipeline::formatPlan(plan) << "\n";
      pm.dumpPipelineProfile(std::cerr);
    }
  }
  if (opts.dumpPassTiming) {
    std::cerr << "[pass-timing] budget: large_module_mode=" << (plan.largeModuleMode ? 1 : 0)
              << ", huge_module_mode=" << (plan.hugeModuleMode ? 1 : 0)
              << ", backend_fast_mode=" << (plan.backendFastMode ? 1 : 0)
              << ", arm_timeout_safe=" << (plan.armTimeoutSafeMode ? 1 : 0)
              << ", arm_instcombine_rounds=" << plan.armInstCombineRounds
              << ", arm_peephole_rounds=" << plan.armPeepholeRounds
              << ", arm_ra_call_penalty=" << plan.armRegAllocCallPenalty
              << ", arm_ra_loop_boost=" << plan.armRegAllocLoopBoost
              << ", arm_ra_prefer_budget=" << plan.armRegAllocPreferBudget
              << "\n";
  }
  
  pm.run();
  return 0;
}
