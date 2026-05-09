#ifndef DEFAULT_TARGET_H
#define DEFAULT_TARGET_H

namespace sys {

enum class DefaultTarget {
  Riscv,
  Arm,
};

// Source-level default used by contest evaluators that compile every .cpp
// directly and do not pass CMake definitions. Keep master on RISC-V; set this
// to 1 on the ARM submission branch.
#ifndef SISYPHUS_DEFAULT_TARGET_ARM
#define SISYPHUS_DEFAULT_TARGET_ARM 1
#endif

#if defined(DEFAULT_TARGET_ARM) || SISYPHUS_DEFAULT_TARGET_ARM
constexpr DefaultTarget kDefaultTarget = DefaultTarget::Arm;
#else
constexpr DefaultTarget kDefaultTarget = DefaultTarget::Riscv;
#endif

}

#endif
