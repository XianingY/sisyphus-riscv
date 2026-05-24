#ifndef RISCV_PARAMS_H
#define RISCV_PARAMS_H

namespace sys::backend::riscv {

// RISC-V rv64gc physical registers configuration
constexpr int kPhysGPRs = 32;       // 32 General Purpose Registers (x0-x31)
constexpr int kPhysFPRs = 32;       // 32 Floating-Point Registers (f0-f31)
constexpr int kReservedGPRs = 8;     // Reserved for addressing, SP/GP, control flow, etc.
constexpr int kReservedFPRs = 4;     // Reserved for temporary float evaluation

// Physical registers usable limit
constexpr int kUsableGPRs = kPhysGPRs - kReservedGPRs; // 24
constexpr int kUsableFPRs = kPhysFPRs - kReservedFPRs; // 28

// Cache line and L1 Data Cache parameters
constexpr int kCacheLineSize = 64;   // Bytes (standard for U74 / C906 cores)
constexpr int kL1DataCacheSize = 32768; // 32KB L1 D-Cache

} // namespace sys::backend::riscv

#endif // RISCV_PARAMS_H
