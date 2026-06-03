#include "SelfMLIR.h"
#include "SelfMLIRInternal.h"
#include "Polyhedral.h"


#include "../parse/ASTNode.h"
#include "../parse/Type.h"
#include "../utils/DynamicCast.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

namespace sys::mlir {

OptimizationConfig OptimizationConfig::forLevel(Level level) {
  OptimizationConfig config;
  config.level = level;
  switch (level) {
  case Level::O0:
    config.enableGlobalOpt = false;
    config.enableAffine = false;
    config.enableMemoryOpt = false;
    config.enableProvenBitwise = false;
    config.enableDRR = false;
    config.enableDRRWorklist = false;
    config.enableLinearScan = false;
    config.enableInline = false;
    config.enableRotateHelper = false;
    config.enablePow2Strength = false;
    config.enableScheduler = false;
    config.enableLoopTiling = false;
    config.enableLoopFusion = false;
    config.enableLoopInterchange = false;
    config.enableStencilPeel = false;
    config.enableLoopAddressIV = false;
    break;
  case Level::O1:
    config.enableGlobalOpt = true;
    config.enableAffine = true;
    config.enableMemoryOpt = true;
    config.enableProvenBitwise = true;
    config.enableDRR = true;
    config.enableDRRWorklist = true;
    config.enableLinearScan = true;
    config.enableInline = true;
    config.enableRotateHelper = true;
    config.enablePow2Strength = true;
    config.enableScheduler = true;
    config.enableLoopTiling = true;
    config.enableLoopFusion = true;
    config.enableLoopInterchange = true;
    config.enableStencilPeel = true;
    config.enableLoopAddressIV = true;
    break;
  case Level::O2:
    config.enableGlobalOpt = true;
    config.enableAffine = true;
    config.enableMemoryOpt = true;
    config.enableProvenBitwise = true;
    config.enableDRR = true;
    config.enableDRRWorklist = true;
    config.enableLinearScan = true;
    config.enableInline = true;
    config.enableRotateHelper = true;
    config.enablePow2Strength = true;
    config.enableScheduler = true;
    config.enableLoopTiling = true;
    config.enableLoopFusion = true;
    config.enableLoopInterchange = true;
    config.enableStencilPeel = true;
    config.enableLoopAddressIV = true;
    break;
  }
  return config;
}


std::vector<ConversionPattern> targetPatterns(const std::string &target) {
  const std::string prefix = target == "arm" ? "arm_machine." : "rv_machine.";
  if (target == "arm") {
    return {
      {"arith.constant", prefix + "mov"},
      {"arith.addi", prefix + "add"},
      {"arith.subi", prefix + "sub"},
      {"arith.muli", prefix + "mul"},
      {"arith.divi", prefix + "sdiv"},
      {"arith.remi", prefix + "srem"},
      {"arith.andi", prefix + "and"},
      {"arith.ori", prefix + "orr"},
      {"arith.xori", prefix + "eor"},
      {"arith.noti", prefix + "not"},
      {"arith.cmpi", prefix + "cmp"},
      {"arith.addf", prefix + "fadd"},
      {"arith.subf", prefix + "fsub"},
      {"arith.mulf", prefix + "fmul"},
      {"arith.divf", prefix + "fdiv"},
      {"arith.negf", prefix + "fneg"},
      {"arith.negi", prefix + "neg"},
      {"arith.sitofp", prefix + "scvtf"},
      {"arith.fptosi", prefix + "fcvtzs"},
      {"arith.select", prefix + "select"},
      {"vector.transfer_read", "arm_machine.ld1"},
      {"vector.transfer_write", "arm_machine.st1"},
      {"vector.splat", "arm_machine.dup"},
    };
  }
  return {
    {"arith.constant", prefix + "li"},
    {"arith.addi", prefix + "addw"},
    {"arith.subi", prefix + "subw"},
    {"arith.muli", prefix + "mulw"},
    {"arith.divi", prefix + "divw"},
    {"arith.remi", prefix + "remw"},
    {"arith.andi", prefix + "and"},
    {"arith.ori", prefix + "or"},
    {"arith.xori", prefix + "xor"},
    {"arith.noti", prefix + "seqz"},
    {"arith.cmpi", prefix + "cmp"},
    {"arith.addf", prefix + "fadd"},
    {"arith.subf", prefix + "fsub"},
    {"arith.mulf", prefix + "fmul"},
    {"arith.divf", prefix + "fdiv"},
    {"arith.negf", prefix + "fneg"},
    {"arith.negi", prefix + "neg"},
    {"arith.sitofp", prefix + "fcvt_s_w"},
    {"arith.fptosi", prefix + "fcvt_w_s"},
    {"arith.select", prefix + "select"},
    {"vector.transfer_read", "rv_machine.vle32"},
    {"vector.transfer_write", "rv_machine.vse32"},
    {"vector.splat", "rv_machine.vfmv"},
  };
}

ConversionTarget productionTarget(const std::string &target) {
  ConversionTarget convTarget;
  convTarget.addLegalDialect("builtin");
  convTarget.addLegalDialect("sysy");
  convTarget.addLegalDialect("scf");
  convTarget.addLegalDialect("cf");
  convTarget.addLegalDialect("memref");
  convTarget.addLegalDialect("affine");
  convTarget.addLegalDialect("vector");
  convTarget.addLegalDialect(target == "arm" ? "arm_machine" : "rv_machine");
  return convTarget;
}
} // namespace sys::mlir
