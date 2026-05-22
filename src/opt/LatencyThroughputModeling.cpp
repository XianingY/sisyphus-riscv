#include "LoopPasses.h"
#include <unordered_map>

namespace sys {

// ===========================================================================
// Latency and Throughput Modeling for Better Scheduling
// ===========================================================================

class LatencyModel {
public:
  enum class InstructionType {
    Integer,
    IntegerMultiply,
    FloatingPoint,
    FloatingPointMultiply,
    Load,
    Store,
    Branch,
    Other
  };

  // Get instruction latency (cycles from operand available to result available)
  static int getLatency(InstructionType type) {
    switch (type) {
      case InstructionType::Integer:
        return 1;  // Most integer ops: 1 cycle
      case InstructionType::IntegerMultiply:
        return 3;  // Multiply: 3-5 cycles typical
      case InstructionType::FloatingPoint:
        return 4;  // FP add/sub: 4 cycles
      case InstructionType::FloatingPointMultiply:
        return 5;  // FP multiply: 5-7 cycles
      case InstructionType::Load:
        return 3;  // Load: 2-3 cycles (L1 hit)
      case InstructionType::Store:
        return 1;  // Store: 1 cycle (can queue)
      case InstructionType::Branch:
        return 0;  // Branch: resolved in fetch/decode
      case InstructionType::Other:
      default:
        return 1;
    }
  }

  // Get instruction throughput (how often it can be issued)
  static double getThroughput(InstructionType type) {
    switch (type) {
      case InstructionType::Integer:
        return 2.0;  // 2 per cycle (dual-issue capable)
      case InstructionType::IntegerMultiply:
        return 1.0;  // 1 per cycle (resource contention)
      case InstructionType::FloatingPoint:
        return 1.0;  // 1 per cycle
      case InstructionType::FloatingPointMultiply:
        return 0.5;  // 1 per 2 cycles
      case InstructionType::Load:
        return 2.0;  // 2 per cycle (pipelining)
      case InstructionType::Store:
        return 1.0;  // 1 per cycle
      case InstructionType::Branch:
        return 1.0;  // 1 per cycle
      case InstructionType::Other:
      default:
        return 1.0;
    }
  }

  // Classify an operation
  static InstructionType classify(Op* op) {
    if (!op) return InstructionType::Other;

    if (isa<AddIOp>(op) || isa<SubIOp>(op))
      return InstructionType::Integer;
    if (isa<MulIOp>(op))
      return InstructionType::IntegerMultiply;
    if (isa<AddFOp>(op) || isa<SubFOp>(op))
      return InstructionType::FloatingPoint;
    if (isa<MulFOp>(op))
      return InstructionType::FloatingPointMultiply;
    if (isa<LoadOp>(op))
      return InstructionType::Load;
    if (isa<StoreOp>(op))
      return InstructionType::Store;
    if (isa<BranchOp>(op))
      return InstructionType::Branch;

    return InstructionType::Other;
  }
};

// Superscalar execution capability model
class SuperscalarModel {
public:
  struct ExecutionPort {
    int portId;
    std::vector<LatencyModel::InstructionType> supportedTypes;
    int cyclesSinceLastUse;
  };

  SuperscalarModel(int numPorts = 4) : ports(numPorts) {
    // Typical 4-issue superscalar setup:
    // Port 0: Integer, Branch
    // Port 1: Integer, Load
    // Port 2: FloatingPoint, Load
    // Port 3: Store

    ports[0].supportedTypes = {LatencyModel::InstructionType::Integer,
                               LatencyModel::InstructionType::Branch};
    ports[1].supportedTypes = {LatencyModel::InstructionType::Integer,
                               LatencyModel::InstructionType::Load};
    ports[2].supportedTypes = {LatencyModel::InstructionType::FloatingPoint,
                               LatencyModel::InstructionType::FloatingPointMultiply};
    ports[3].supportedTypes = {LatencyModel::InstructionType::Store};
  }

  // Find best execution port for operation
  int findBestPort(LatencyModel::InstructionType type) {
    for (int i = 0; i < (int)ports.size(); i++) {
      for (const auto& supported : ports[i].supportedTypes) {
        if (supported == type) {
          return i;
        }
      }
    }
    return 0;  // Default to port 0
  }

private:
  std::vector<ExecutionPort> ports;
};

}  // namespace sys
