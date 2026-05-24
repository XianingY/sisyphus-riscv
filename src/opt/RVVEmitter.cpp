#include "VectorCodeGen.h"
#include <sstream>

namespace sys {
namespace codegen {

std::string RVVEmitter::emitVectorLoad(int elemWidth, int elemCount, bool strided) {
  std::ostringstream oss;
  std::string elemType;
  if (elemWidth == 32) elemType = "w";
  else if (elemWidth == 64) elemType = "d";
  else elemType = "h";
  
  if (strided) {
    oss << "vlse" << elemType << ".v v0, 0(x0), x1";
  } else {
    oss << "vle" << elemType << ".v v0, 0(x0)";
  }
  return oss.str();
}

std::string RVVEmitter::emitVectorStore(int elemWidth, int elemCount, bool strided) {
  std::ostringstream oss;
  std::string elemType;
  if (elemWidth == 32) elemType = "w";
  else if (elemWidth == 64) elemType = "d";
  else elemType = "h";
  
  if (strided) {
    oss << "vsse" << elemType << ".v v0, 0(x0), x1";
  } else {
    oss << "vse" << elemType << ".v v0, 0(x0)";
  }
  return oss.str();
}

std::string RVVEmitter::emitVectorArith(const std::string& op, int elemWidth, bool isFloat) {
  std::ostringstream oss;
  std::string instr;
  
  if (isFloat) {
    if (op == "+") instr = "vfadd";
    else if (op == "-") instr = "vfsub";
    else if (op == "*") instr = "vfmul";
    else if (op == "/") instr = "vfdiv";
    else instr = "vfadd";
  } else {
    if (op == "+") instr = "vadd";
    else if (op == "-") instr = "vsub";
    else if (op == "*") instr = "vmul";
    else instr = "vadd";
  }
  
  oss << instr << ".vv v0, v0, v1";
  return oss.str();
}

std::string RVVEmitter::emitVectorReduce(const std::string& op, int elemWidth, bool isFloat) {
  std::ostringstream oss;
  if (isFloat) {
    oss << "vfredsum.vs v0, v0, v1";
  } else {
    oss << "vredsum.vs v0, v0, v1";
  }
  return oss.str();
}

std::string RVVEmitter::emitSetVectorLength(int elemWidth, int elemCount) {
  std::ostringstream oss;
  std::string elemType = "e32";
  if (elemWidth == 8) elemType = "e8";
  else if (elemWidth == 16) elemType = "e16";
  else if (elemWidth == 64) elemType = "e64";
  
  oss << "vsetvli x0, " << elemCount << ", " << elemType << ", m1";
  return oss.str();
}

std::string RVVEmitter::emitVectorMask(const std::string& maskOp, int elemCount) {
  std::ostringstream oss;
  if (maskOp == "all_true") {
    oss << "vmset.m v0";
  } else if (maskOp == "all_false") {
    oss << "vmclr.m v0";
  } else {
    oss << "vmand.mm v0, v0, v1";
  }
  return oss.str();
}

}
}
