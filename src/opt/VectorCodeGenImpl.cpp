#include "VectorCodeGen.h"
#include <sstream>

namespace sys {
namespace codegen {

std::string VectorCodeGen::genVectorLoad(const std::string& destReg,
                                        const std::string& addrReg,
                                        int elemWidth, int elemCount) {
  if (isa == VectorISA::RVV) {
    return genRVVLoad(destReg, addrReg, elemWidth, elemCount);
  } else {
    return genNEONLoad(destReg, addrReg, elemWidth, elemCount);
  }
}

std::string VectorCodeGen::genVectorStore(const std::string& srcReg,
                                         const std::string& addrReg,
                                         int elemWidth, int elemCount) {
  if (isa == VectorISA::RVV) {
    return RVVEmitter::emitVectorStore(elemWidth, elemCount, false);
  } else {
    return NEONEmitter::emitNEONStore(elemWidth, elemCount, 1);
  }
}

std::string VectorCodeGen::genVectorArith(const std::string& destReg,
                                        const std::string& lhsReg,
                                        const std::string& rhsReg,
                                        const std::string& op,
                                        int elemWidth, bool isFloat) {
  if (isa == VectorISA::RVV) {
    return RVVEmitter::emitVectorArith(op, elemWidth, isFloat);
  } else {
    return NEONEmitter::emitNEONArith(op, elemWidth, isFloat);
  }
}

std::string VectorCodeGen::genVectorReduction(const std::string& destReg,
                                            const std::string& srcReg,
                                            const std::string& op,
                                            int elemWidth, bool isFloat) {
  if (isa == VectorISA::RVV) {
    return RVVEmitter::emitVectorReduce(op, elemWidth, isFloat);
  } else {
    return NEONEmitter::emitNEONReduce(op, elemWidth, isFloat);
  }
}

std::string VectorCodeGen::genMaskedVectorOp(const std::string& destReg,
                                           const std::string& maskReg,
                                           const std::string& op,
                                           int elemWidth) {
  if (isa == VectorISA::RVV) {
    return RVVEmitter::emitVectorMask(op, elemWidth);
  } else {
    return "// NEON masked operation";
  }
}

std::string VectorCodeGen::genRVVLoad(const std::string& destReg,
                                     const std::string& addrReg,
                                     int elemWidth, int elemCount) {
  std::ostringstream oss;
  oss << RVVEmitter::emitSetVectorLength(elemWidth, elemCount) << "\n";
  oss << RVVEmitter::emitVectorLoad(elemWidth, elemCount, false);
  return oss.str();
}

std::string VectorCodeGen::genNEONLoad(const std::string& destReg,
                                      const std::string& addrReg,
                                      int elemWidth, int elemCount) {
  return NEONEmitter::emitNEONLoad(elemWidth, elemCount, 1);
}

}
}
