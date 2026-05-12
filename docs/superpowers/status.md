# 深度优化状态跟踪

最后更新：2026-05-11（状态标记规范化）

## 总体状态

| 模块 | 状态 | 备注 |
|------|------|------|
| StrengthReduce (RV) | ✅ 完成 | 已修复除2^n和模2^n |
| StrengthReduce (ARM) | ✅ 完成 | 已在之前会话修复 |
| Mulw/Addw/Subw 窥孔优化 | ✅ 完成 | `mv a5,a4; mulw a4,a4,a5` → `mulw a4,a4,a4` |
| PartialUnroll | ✅ 完成 | 支持运行时边界的部分展开 |
| StrengthReduce 增强 | ✅ 完成 | 通用乘法强度削减 (x * 105) |
| LoopInterchange | ✅ 框架 | 检测完美嵌套循环（实际交换需依赖分析） |
| BoundsCheck | ⚠️ 架构限制 | 框架完成，CFG降低后无法识别IfOp模式 |
| TailRecElim | ⚠️ 架构限制 | TCO已存在，前端生成call();store;goto模式无法优化 |

## 本次会话完成

### RegPeephole mulw/addw/subw 优化

**问题**：`mv a5, a4; mulw a4, a4, a5` 产生不必要的寄存器复制

**解决方案**：在 `src/rv/RegPeephole.cpp` 添加窥孔优化，检测并消除此模式

**代码变更**：
```cpp
// mv rd, rs -> mulw/addw/subw rd, rd, rd (当两个源操作数相等)
// mv a5, a4; mulw a4, a4, a5  →  mulw a4, a4, a4

if (isa<MulwOp>(next) || isa<AddwOp>(next) || isa<SubwOp>(next)) {
    Reg mvSrc = RS(op);
    Reg mvDst = RD(op);

    // Case 1: 两源相同
    if (RS(next) == RS2(next)) {
        if (RS(next) == mvDst || RS(next) == mvSrc) {
            RS(next) = RD(next);
            RS2(next) = RD(next);
            converted++;
            op->erase();
            return true;
        }
    }

    // Case 2: 目的=源1 且 源2=mv目的
    if (RD(next) == RS(next) && RS2(next) == mvDst) {
        RS(next) = RD(next);
        RS2(next) = RD(next);
        converted++;
        op->erase();
        return true;
    }
}
```

**效果**：
```
优化前: lw a4; mv a5, a4; mulw a4, a4, a5  (3条指令)
优化后: lw a4; mulw a4, a4, a4              (2条指令)
```

### PartialUnroll 实现

**问题**：FFT 等测试的循环边界是运行时参数（如 `n/2`），`ConstLoopUnroll` 无法处理

**解决方案**：扩展 `ConstLoopUnroll` 支持运行时上界估计和部分展开

**代码变更** (`src/opt/LoopUnroll.cpp`)：
```cpp
// 新增：估计运行时上界的辅助函数
auto estimateUpperBound = [](Op* expr) -> int64_t {
    if (!expr) return -1;
    if (isa<IntOp>(expr)) return V(expr);
    if (isa<GetArgOp>(expr)) return 8192;  // 函数参数保守估计
    // 处理 divi 情况...
    return -1;
};

// 部分展开策略
if (loopsize <= 20 && estTimes > 256) unroll = 2;  // 紧凑循环
if (loopsize <= 50 && estTimes > 256) unroll = 2;  // 小循环
```

**效果**：
```
优化前: while (i < n) { ... }  (n 是运行时参数)
优化后: while (i < n) { body; body; i += 2; }  (部分展开 x2)
```

### StrengthReduce 增强（通用乘法）

**问题**：原有实现只处理 2 的幂和少量特殊形式，常数乘法如 `x * 105` 无法优化

**解决方案**：添加通用分解算法，将任意常数分解为移位和加法

**代码变更** (`src/rv/StrengthReduct.cpp`)：
```cpp
// 通用分解：x * c → shift-add 链
if (bits >= 3 && bits <= 16) {
    // 分解 c 为 2^bit[i] 的和
    // 例如 105 = 64 + 32 + 8 + 1 = 2^6 + 2^5 + 2^3 + 2^0
    // 优化为：((x << 6) + (x << 5)) + ((x << 3) + x)
}
```

**效果**：
```
x * 105
优化前: mulw a0, a0, a1  (乘法指令，假设 a1 = 105)
优化后: slliw a2, a0, 6; slliw a3, a0, 5; addw a2, a2, a3; ...
       (纯移位和加法，无乘法指令)
```

### LoopInterchange（循环交换框架）

**问题**：循环交换需要复杂的依赖分析才能安全执行

**解决方案**：实现检测框架，识别完美嵌套循环，为未来优化做准备

**代码变更** (`src/opt/LoopInterchange.cpp` + `LoopPasses.h`)：
```cpp
class LoopInterchange : public Pass {
  int detected = 0;
  int interchanged = 0;

  // 检测完美嵌套循环
  bool isPerfectNest(LoopInfo* outer, LoopInfo* inner);
  // 检查是否可以交换（需依赖分析）
  bool canInterchange(LoopInfo* outer, LoopInfo* inner);
};
```

**当前状态**：
- 框架已完成，可检测完美嵌套
- 实际交换需要依赖分析，暂时返回 false（保守）
- 需依赖分析完善后才能执行实际交换

### BoundsCheck（边界检查消除）

**问题**：conv2d 等 stencil 类循环中存在大量冗余边界检查

**解决方案**：实现边界检查检测框架，识别可消除的检查

**代码变更** (`src/opt/BoundsCheck.cpp` + `LoopPasses.h`)：
```cpp
class BoundsCheck : public Pass {
  int eliminated = 0;
  int hoisted = 0;
  void runImpl(LoopInfo *info);
public:
  BoundsCheck(ModuleOp *module): Pass(module) {}
  std::string name() override { return "bounds-check"; }
  std::map<std::string, int> stats() override;
  void run() override;
};
```

**当前状态**：
- ⚠️ 架构限制：框架代码已完成，但IR已CFG降低，无法直接识别IfOp模式
- conv2d 的边界检查已降低为分支，无法使用简单IfOp分析
- 需要更复杂的CFG分析或循环结构化才能有效消除检查
- 当前仅计数检测到的检查，不执行实际消除

### TailRecElim（尾递归消除）

**问题**：knapsack_naive 存在尾递归调用

**当前状态**：
- ✅ 已存在：TCO pass 已在 `src/pre-opt/TCO.cpp` 实现
- ⚠️ 架构限制：前端生成的IR将调用结果存储到临时变量再返回
  - 实际模式：`call(); store result; goto` 而非 `return call()`
  - TCO 要求直接返回调用结果，无法优化当前模式
- 解决方案需要：前端修改或更复杂的CFG级别尾递归优化

## 性能影响

| 测试 | 瓶颈 | 当前优化 | 预期提升 |
|------|------|---------|---------|
| many_mat_cal-1/2/3 | 内层循环mv | RegPeephole | ~1.1-1.2x |
| fft1 | 循环展开 | PartialUnroll | ~10-15x |
| conv2d-1 | 边界检查 | 待 BoundsCheck | ~5-8x |
| 03_sort1 | 计数循环 | 待 Fusion+PartialUnroll | ~5-8x |

## 下一步行动

### 高优先级

1. **LoopInterchange 增强**
   - 需要依赖分析才能执行实际交换
   - 目标：transpose 系列
   - 当前状态：框架完成，需依赖分析模块

### 架构限制（无法在当前IR上实现）

1. **BoundsCheck** — IR已CFG降低，无法识别IfOp模式
2. **TailRecElim** — 前端生成call();store;goto而非return call()

### 低优先级

1. **LightweightSuperopt** — 模式匹配超优化，无特定目标测试
2. **DPTransforms** — 递归到DP转换，无特定目标测试

## 风险提示

- 窥孔优化可能影响寄存器分配，需在真实硬件上验证
- 部分展开 pass 需要仔细的边界处理测试
