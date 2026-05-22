#include "VectorCodeGen.h"
#include <sstream>

namespace sys {
namespace codegen {

std::string NEONEmitter::emitNEONLoad(int elemWidth, int laneCount, int interleave) {
  std::ostringstream oss;
  std::string suffix;
  if (elemWidth == 32) suffix = ".32";
  else if (elemWidth == 64) suffix = ".64";
  else suffix = ".32";
  
  if (interleave == 1) {
    oss << "ld1" << suffix << " {v0}, [x0]";
  } else if (interleave == 2) {
    oss << "ld2" << suffix << " {v0, v1}, [x0]";
  } else if (interleave == 3) {
    oss << "ld3" << suffix << " {v0, v1, v2}, [x0]";
  } else {
    oss << "ld4" << suffix << " {v0, v1, v2, v3}, [x0]";
  }
  return oss.str();
}

std::string NEONEmitter::emitNEONStore(int elemWidth, int laneCount, int interleave) {
  std::ostringstream oss;
  std::string suffix;
  if (elemWidth == 32) suffix = ".32";
  else if (elemWidth == 64) suffix = ".64";
  else suffix = ".32";
  
  if (interleave == 1) {
    oss << "st1" << suffix << " {v0}, [x0]";
  } else if (interleave == 2) {
    oss << "st2" << suffix << " {v0, v1}, [x0]";
  } else if (interleave == 3) {
    oss << "st3" << suffix << " {v0, v1, v2}, [x0]";
  } else {
    oss << "st4" << suffix << " {v0, v1, v2, v3}, [x0]";
  }
  return oss.str();
}

std::string NEONEmitter::emitNEONArith(const std::string& op, int elemWidth, bool isFloat) {
  std::ostringstream oss;
  std::string instr;
  
  if (isFloat) {
    if (op == "+") instr = "fadd";
    else if (op == "-") instr = "fsub";
    else if (op == "*") instr = "fmul";
    else if (op == "/") instr = "fdiv";
    else instr = "fadd";
  } else {
    if (op == "+") instr = "add";
    else if (op == "-") instr = "sub";
    else if (op == "*") instr = "mul";
    else instr = "add";
  }
  
  std::string suffix;
  if (elemWidth == 32) suffix = ".4s";
  else if (elemWidth == 64) suffix = ".2d";
  else suffix = ".4s";
  
  oss << instr << suffix << " v0, v0, v1";
  return oss.str();
}

std::string NEONEmitter::emitNEONReduce(const std::string& op, int elemWidth, bool isFloat) {
  std::ostringstream oss;
  if (isFloat) {
    oss << "faddp s0, v0.2s";
  } else {
    oss << "addp v0, v0, v1";
  }
  return oss.str();
}

std::string NEONEmitter::emitNEONHorizontal(const std::string& op, int elemWidth, bool isFloat) {
  std::ostringstream oss;
  std::string suffix;
  if (elemWidth == 32) suffix = ".4s";
  else suffix = ".2d";
  
  if (op == "add") oss << "addp v0" << suffix << ", v0, v1";
  else if (op == "sub") oss << "subp v0" << suffix << ", v0, v1";
  else oss << "addp v0" << suffix << ", v0, v1";
  
  return oss.str();
}

std::string NEONEmitter::emitNEONLaneArith(const std::string& op, int elemWidth, int lane) {
  std::ostringstream oss;
  std::string instr;
  if (op == "mul") instr = "mla";
  else if (op == "sub") instr = "mls";
  else instr = "mla";
  
  std::string suffix;
  if (elemWidth == 32) suffix = ".4s";
  else suffix = ".2d";
  
  oss << instr << suffix << " v0, v1, v2.s[" << lane << "]";
  return oss.str();
}

}
}
