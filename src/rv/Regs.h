#ifndef REGS_H
#define REGS_H

#include "RvAttrs.h"

namespace sys::rv {

// We use dedicated registers as the "spill" register, for simplicity.
const Reg spillReg = Reg::s10;
const Reg spillReg2 = Reg::s11;
const Reg fspillReg = Reg::fs10;
const Reg fspillReg2 = Reg::fs11;
const Reg vspillReg = Reg::v30;
const Reg vspillReg2 = Reg::v31;

// Order for leaf functions. Prioritize temporaries.
const Reg leafOrder[] = {
  Reg::a0, Reg::a1, Reg::a2, Reg::a3,
  Reg::a4, Reg::a5, Reg::a6, Reg::a7,

  Reg::t0, Reg::t1, Reg::t2, Reg::t3,
  Reg::t4, Reg::t5, Reg::t6,
  
  Reg::s0, Reg::s1, Reg::s2, Reg::s3, 
  Reg::s4, Reg::s5, Reg::s6, Reg::s7,
  Reg::s8, Reg::s9,
};
// Order for non-leaf functions.
const Reg normalOrder[] = {
  Reg::s0, Reg::s1, Reg::s2, Reg::s3, 
  Reg::s4, Reg::s5, Reg::s6, Reg::s7,
  Reg::s8, Reg::s9,
};
const Reg argRegs[] = {
  Reg::a0, Reg::a1, Reg::a2, Reg::a3,
  Reg::a4, Reg::a5, Reg::a6, Reg::a7,
};
const std::set<Reg> callerSaved = {
  Reg::t0, Reg::t1, Reg::t2, Reg::t3,
  Reg::t4, Reg::t5, Reg::t6,

  Reg::a0, Reg::a1, Reg::a2, Reg::a3,
  Reg::a4, Reg::a5, Reg::a6, Reg::a7,
  Reg::ra,

  Reg::ft0, Reg::ft1, Reg::ft2, Reg::ft3,
  Reg::ft4, Reg::ft5, Reg::ft6, Reg::ft7,
  Reg::ft8, Reg::ft9, Reg::ft10, Reg::ft11,

  Reg::fa0, Reg::fa1, Reg::fa2, Reg::fa3,
  Reg::fa4, Reg::fa5, Reg::fa6, Reg::fa7,

  Reg::v0, Reg::v1, Reg::v2, Reg::v3,
  Reg::v4, Reg::v5, Reg::v6, Reg::v7,
  Reg::v8, Reg::v9, Reg::v10, Reg::v11,
  Reg::v12, Reg::v13, Reg::v14, Reg::v15,
  Reg::v16, Reg::v17, Reg::v18, Reg::v19,
  Reg::v20, Reg::v21, Reg::v22, Reg::v23,
  Reg::v24, Reg::v25, Reg::v26, Reg::v27,
  Reg::v28, Reg::v29, Reg::v30, Reg::v31,
};

const std::set<Reg> calleeSaved = {
  Reg::s0, Reg::s1, Reg::s2, Reg::s3, 
  Reg::s4, Reg::s5, Reg::s6, Reg::s7,
  Reg::s8, Reg::s9, Reg::s10, Reg::s11,

  Reg::fs0, Reg::fs1, Reg::fs2, Reg::fs3, 
  Reg::fs4, Reg::fs5, Reg::fs6, Reg::fs7,
  Reg::fs8, Reg::fs9, Reg::fs10, Reg::fs11,
};
constexpr int leafRegCnt = sizeof(leafOrder) / sizeof(Reg);
constexpr int normalRegCnt = sizeof(normalOrder) / sizeof(Reg);

const Reg leafOrderf[] = {
  Reg::fa0, Reg::fa1, Reg::fa2, Reg::fa3,
  Reg::fa4, Reg::fa5, Reg::fa6, Reg::fa7,

  Reg::ft0, Reg::ft1, Reg::ft2, Reg::ft3,
  Reg::ft4, Reg::ft5, Reg::ft6, Reg::ft7,
  Reg::ft8, Reg::ft9, Reg::ft10, Reg::ft11,
  
  Reg::fs0, Reg::fs1, Reg::fs2, Reg::fs3, 
  Reg::fs4, Reg::fs5, Reg::fs6, Reg::fs7,
  Reg::fs8, Reg::fs9,
};
// Order for non-leaf functions.
const Reg normalOrderf[] = {
  Reg::ft0, Reg::ft1, Reg::ft2, Reg::ft3,
  Reg::ft4, Reg::ft5, Reg::ft6, Reg::ft7,
  Reg::ft8, Reg::ft9, Reg::ft10, Reg::ft11,

  Reg::fa0, Reg::fa1, Reg::fa2, Reg::fa3,
  Reg::fa4, Reg::fa5, Reg::fa6, Reg::fa7,

  Reg::fs0, Reg::fs1, Reg::fs2, Reg::fs3, 
  Reg::fs4, Reg::fs5, Reg::fs6, Reg::fs7,
  Reg::fs8, Reg::fs9,
};
const Reg fargRegs[] = {
  Reg::fa0, Reg::fa1, Reg::fa2, Reg::fa3,
  Reg::fa4, Reg::fa5, Reg::fa6, Reg::fa7,
};
constexpr int leafRegCntf = 30;
constexpr int normalRegCntf = 30;

const Reg leafOrderv[] = {
  Reg::v0, Reg::v1, Reg::v2, Reg::v3,
  Reg::v4, Reg::v5, Reg::v6, Reg::v7,
  Reg::v8, Reg::v9, Reg::v10, Reg::v11,
  Reg::v12, Reg::v13, Reg::v14, Reg::v15,
  Reg::v16, Reg::v17, Reg::v18, Reg::v19,
  Reg::v20, Reg::v21, Reg::v22, Reg::v23,
  Reg::v24, Reg::v25, Reg::v26, Reg::v27,
  Reg::v28, Reg::v29,
};
const Reg normalOrderv[] = {
  Reg::v0, Reg::v1, Reg::v2, Reg::v3,
  Reg::v4, Reg::v5, Reg::v6, Reg::v7,
  Reg::v8, Reg::v9, Reg::v10, Reg::v11,
  Reg::v12, Reg::v13, Reg::v14, Reg::v15,
  Reg::v16, Reg::v17, Reg::v18, Reg::v19,
  Reg::v20, Reg::v21, Reg::v22, Reg::v23,
  Reg::v24, Reg::v25, Reg::v26, Reg::v27,
  Reg::v28, Reg::v29,
};
constexpr int leafRegCntv = 30;
constexpr int normalRegCntv = 30;

inline bool fpreg(Value::Type ty) {
  return ty == Value::f32;
}

inline bool vecreg(Value::Type ty) {
  return ty == Value::i128 || ty == Value::f128;
}

}

#endif
