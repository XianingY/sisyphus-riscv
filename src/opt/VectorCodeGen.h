#pragma once

#include <string>
#include <vector>

namespace sys {
namespace codegen {

class RVVEmitter {
public:
  static std::string emitVectorLoad(int elemWidth, int elemCount, bool strided = false);
  static std::string emitVectorStore(int elemWidth, int elemCount, bool strided = false);
  static std::string emitVectorArith(const std::string& op, int elemWidth, bool isFloat);
  static std::string emitVectorReduce(const std::string& op, int elemWidth, bool isFloat);
  static std::string emitSetVectorLength(int elemWidth, int elemCount);
  static std::string emitVectorMask(const std::string& maskOp, int elemCount);
};

class NEONEmitter {
public:
  static std::string emitNEONLoad(int elemWidth, int laneCount, int interleave = 1);
  static std::string emitNEONStore(int elemWidth, int laneCount, int interleave = 1);
  static std::string emitNEONArith(const std::string& op, int elemWidth, bool isFloat);
  static std::string emitNEONReduce(const std::string& op, int elemWidth, bool isFloat);
  static std::string emitNEONHorizontal(const std::string& op, int elemWidth, bool isFloat);
  static std::string emitNEONLaneArith(const std::string& op, int elemWidth, int lane);
};

class VectorCodeGen {
public:
  enum class VectorISA { RVV, NEON };
  
  VectorCodeGen(VectorISA isa) : isa(isa) {}
  
  std::string genVectorLoad(const std::string& destReg, const std::string& addrReg,
                           int elemWidth, int elemCount);
  std::string genVectorStore(const std::string& srcReg, const std::string& addrReg,
                            int elemWidth, int elemCount);
  std::string genVectorArith(const std::string& destReg, const std::string& lhsReg,
                            const std::string& rhsReg, const std::string& op,
                            int elemWidth, bool isFloat);
  std::string genVectorReduction(const std::string& destReg, const std::string& srcReg,
                               const std::string& op, int elemWidth, bool isFloat);
  std::string genMaskedVectorOp(const std::string& destReg, const std::string& maskReg,
                               const std::string& op, int elemWidth);

private:
  VectorISA isa;
  std::string genRVVLoad(const std::string& destReg, const std::string& addrReg,
                        int elemWidth, int elemCount);
  std::string genNEONLoad(const std::string& destReg, const std::string& addrReg,
                         int elemWidth, int elemCount);
};

}
}
