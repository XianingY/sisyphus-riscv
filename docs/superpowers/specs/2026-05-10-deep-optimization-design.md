# 深度优化方案：针对大性能差距测试

## 1. 背景与目标

### 1.1 性能差距分析

根据测试数据，以下测试存在显著性能差距：

| 测试 | 优化前 (s) | 优化后 (s) | 加速比 | 瓶颈分类 |
|------|-----------|-----------|--------|---------|
| fft1 | 4.82 | 0.25 | **19x** | 内层循环展开不足 |
| fft0 | 0.52 | 0.03 | **17x** | 递归分治 + 小问题开销 |
| conv2d-1 | 57.70 | 5.23 | **11x** | 嵌套 stencil + 边界检查 |
| 03_sort1 | 7.98 | 0.82 | **10x** | 计数排序循环 |
| optimization_scheduling1 | 5.96 | 1.08 | **5.5x** | 依赖链 vs 独立计算 |
| knapsack_naive-1 | 27.46 | 4.14 | **6.6x** | 递归树 → 循环转换 |
| many_mat_cal-3 | 104.86 | 58.50 | **1.8x** | 多循环可融合 |
| transpose2 | 14.93 | 10.17 | **1.5x** | 转置访问模式 |

### 1.2 瓶颈分类

```
┌─────────────────────────────────────────────────────────────────┐
│                     性能瓶颈分类                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  A. 循环级优化（当前项目已有基础，需要增强）                        │
│     ├── 循环展开不足（fft1）                                     │
│     ├── 循环交换（transpose2）                                   │
│     ├── 循环融合（many_mat_cal, 03_sort1）                      │
│     └── 循环不变代码外提不足（optimization_scheduling1）         │
│                                                                 │
│  B. 边界检查优化（conv2d-1）                                     │
│     └── 边界检查阻止向量化/展开                                   │
│                                                                 │
│  C. 递归到循环转换（knapsack_naive-1）                          │
│     └── 递归树 → DP 表                                           │
│                                                                 │
│  D. 强度削减（所有测试）                                         │
│     └── 乘除法 → 移位/加法                                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 参考资源分析

### 2.1 剑桥大学编译优化课程

**可应用内容**：

1. **Abstract Interpretation（抽象解释）**
   - 当前项目：`src/opt/Range.cpp` 已使用区间分析（一种抽象解释）
   - 可增强：更精确的 widening 操作、指向分析、别名分析

2. **数据流分析框架**
   - 可增强 DCE、DSE、LICM 等 pass 的分析精度
   - 竞赛相关：前 9 讲覆盖内容最相关

3. **Superoptimization 概念**
   - 参考 Souper（MLIR 超优化器）的思路
   - 穷举小模式表 + SMT-like 证明

**应用到本项目**：
- 增强 `Range.cpp` 的 widening 操作
- 实现更精确的指向分析以支持更 aggressive 的 DSE

---

### 2.2 Clang IR / MLIR

**可应用内容**：

1. **Declarative Rewrite Patterns（声明式重写模式）**
   - 当前项目已有：`src/utils/Matcher.h` - 简单的模式匹配框架
   - 可增强：支持更复杂的模式组合

2. **多级 IR 分离**
   - 当前项目已有：HIR → CFG → Legacy ModuleOp
   - 已有参考价值，设计合理，无需大改

3. **Loop Transformation 接口**
   - MLIR 的 `affine.for` 和 `affine.load` 提供了天然的 SCoP 边界
   - 本项目可参考设计 LoopInterchange 的接口

**应用到本项目**：
- 增强 Matcher 框架支持更复杂的 Algebraic Simplification
- 借用 MLIR 的 Affine 接口思想设计 BoundsCheck

---

### 2.3 FPL（Fast Presburger Library）

**可应用内容**：

1. **多面体优化的数学原理**
   - Presburger 算术集合操作
   - Integer Set Library 实现思路

2. **依赖分析**
   - 多面体依赖距离计算
   - Fourier-Motzkin  elimination

**应用到本项目**：
- 方案 A（轻量增强）不引入完整多面体框架
- BoundsCheck 优化使用简化的仿射分析（足够处理 stencil 边界）
- 如需完整多面体优化，扩展 `src/utils/presburger/BasicSet.cpp`

---

### 2.4 z3 / Souper

**可应用内容**：

1. **Superoptimizer 概念**
   - 穷举小指令序列，找更优替代
   - 使用 SMT 求解器证明等价性

2. **CDCL 算法思想**
   - SAT 求解器的核心算法
   - 可启发寄存器分配冲突检测

**应用到本项目**：

```cpp
// 新增：LightweightSuperopt.cpp
// 不引入 z3，使用简化的等价性证明

// 核心思路：
// 1. 维护一个已知优化模式表
// 2. 遍历函数中的指令序列
// 3. 对每个序列，检查模式表是否有更优替换

// 模式表示例：
// (add x x)           → (shl x 1)           [x * 2 = x << 1]
// (mul x 4)           → (shl x 2)           [x * 4 = x << 2]
// (div x 2)           → (shr x 1)           [x / 2 = x >> 1]
// (mul x (add y z))   → (add (mul x y) (mul x z))  [分配律]
```

**注意**：竞赛中不能引入 z3，但可以自研简化版本。

---

## 3. 优化手段总览

| 序号 | 模块 | 优化类型 | 目标测试 | 参考资源 |
|------|------|---------|---------|---------|
| 1 | StrengthReduce | 强度削减 | 所有测试 | MLIR, Souper |
| 2 | LoopInterchange | 循环交换 | transpose0/1/2 | MLIR Affine |
| 3 | Fusion 增强 | 循环融合 | many_mat_cal, 03_sort1 | MLIR Pattern |
| 4 | BoundsCheck | 边界优化 | conv2d-1/2/3 | FPL (简化) |
| 5 | LICM 增强 | 循环不变外提 | optimization_scheduling1 | 剑桥课程 |
| 6 | PartialUnroll | 部分展开 | fft0/1/2 | 剑桥课程 |
| 7 | TailRecElim | 尾递归消除 | knapsack_naive | 递归→迭代 |
| 8 | DPTransforms | DP 模式识别 | knapsack_naive | 递归→DP |
| 9 | LightweightSuperopt | 超优化 | 所有测试 | Souper/z3 |

---

## 4. 模块详细设计

### 4.1 StrengthReduce（强度削减优化）

#### 4.1.1 目标
将乘除法转换为更快的移位/加法操作，所有测试受益。

#### 4.1.2 优化场景

**场景 1：乘以 2 的幂**
```c
// 优化前
idx = i * 1024;

// 优化后
idx = i << 10;  // 1024 = 2^10
```

**场景 2：索引计算优化（参考 MLIR 的 loop-invariant code motion）**
```c
// 优化前
for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
        A[i*n + j] = ...;

// 优化后
int stride = n;
int base = 0;
for (i = 0; i < n; i++) {
    int idx = base;
    for (j = 0; j < n; j++) {
        A[idx] = ...;
        idx++;  // 替代 i*n + j
    }
    base += stride;
}
```

**场景 3：除以常量（2 的幂）**
```c
// 优化前
i = j / 1024;

// 优化后
i = j >> 10;
```

#### 4.1.3 实现位置
```
src/opt/StrengthReduce.cpp  (新增)
```

#### 4.1.4 正确性保证
- 只对编译期常量进行变换
- 乘法强度削减使用移位 + 加法组合
- 除法/模 2 的幂只在确定无符号溢出风险时优化

---

### 4.2 LoopInterchange（循环交换）

#### 4.2.1 目标
交换嵌套循环的内外层顺序以改善访存局部性，解决 transpose 系列测试。

#### 4.2.2 可交换条件

```cpp
// 条件 1：完美嵌套
// 条件 2：无循环携带依赖

struct DependenceDirection {
    int outer;  // 外层循环依赖方向
    int inner;  // 内层循环依赖方向
};

// 可交换 ⟺ 对所有依赖 D，有 D.inner >= 0
```

**示例：可交换的情况**
```c
// B[j][i] = A[i][j]
// 写访问：B[j][i]
// 读访问：A[i][j]（与 i, j 都相关，无循环携带依赖）
// 可交换
for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
        B[j][i] = A[i][j];
```

#### 4.2.3 实现位置
```
src/opt/LoopInterchange.cpp  (新增)
```

---

### 4.3 Fusion 增强（下标等价性 + 跨循环融合）

#### 4.3.1 目标
增强现有 Fusion pass，解决下标等价性问题。

#### 4.3.2 当前问题
Fusion.cpp:137 有 TODO 注释，标识需要 polyhedral 方法。

#### 4.3.3 增强内容

**增强：下标等价性**
```c
// 当前：只接受完全相同的下标
for (i) A[i] = ...;    ✓ 可融合
for (i) ... = A[i];    ✓

// 增强后：接受数学等价的下标
for (i) A[i] = ...;    ✓ 可融合
for (i) ... = A[i+0];  ✓ (i+0 = i)
```

#### 4.3.4 实现位置
```
src/opt/IndexEquiv.cpp   (新增)
src/opt/Fusion.cpp       (修改)
```

---

### 4.4 BoundsCheck（边界检查消除）

#### 4.4.1 目标
消除 stencil 类循环中的冗余边界检查，解决 conv2d 测试。

#### 4.4.2 优化思路

```
将循环分离为三个区域：
┌────────────────────────────────────────────┐
│  顶部边界区域 (0 ~ pad)                     │
│  - 需要边界检查                            │
├────────────────────────────────────────────┤
│  主体区域 (pad ~ N-pad)                    │
│  - 无需边界检查！rr 和 cc必然在范围内       │
├────────────────────────────────────────────┤
│  底部边界区域 (N-pad ~ N)                  │
│  - 需要边界检查                            │
└────────────────────────────────────────────┘
```

#### 4.4.3 实现位置
```
src/opt/BoundsCheck.cpp  (新增)
```

#### 4.4.4 数学基础（参考 FPL）
- 使用 Presburger 集合描述安全区域
- Fourier-Motzkin elimination 投影到循环变量

---

### 4.5 LICM 增强（链式不变计算外提）

#### 4.5.1 目标
识别并外提链式不变计算，解决 optimization_scheduling1。

#### 4.5.2 问题场景
```c
while (i < iterations) {
    a = a + b;    // b 看起来在变化...
    b = b + c;    // c 是循环不变的！
    c = c + d;    // d 是循环不变的！
    d = d + a;
    i++;
}
```

关键观察：`c` 和 `d` 的**增量**是循环不变的。

#### 4.5.3 实现位置
```
src/opt/LICM.cpp  (修改，增强链式分析)
```

#### 4.5.4 数学基础（参考 抽象解释）
```
使用数据流分析：
- 前向分析：传播值的上界/下界
- widening 操作：加速收敛
- 不变检测：满足 f(x) = x 的点
```

---

### 4.6 PartialUnroll（部分循环展开）

#### 4.6.1 目标
对迭代次数有上界但非常量的循环做部分展开，解决 fft 系列。

#### 4.6.2 问题场景
```c
// fft 内层循环
while (i < n/2) {  // n 是运行时参数
    x = arr[begin_pos + i];
    y = arr[begin_pos + i + n/2];
    ...
    i++;
}
```

当前 `ConstLoopUnroll` 只对编译期常量迭代次数做展开。

#### 4.6.3 实现思路

```cpp
// 1. SCEV 分析计算迭代次数上界
//    扩展 src/opt/SCEV.cpp 支持上界计算

// 2. 选择展开因子
int chooseUnrollFactor(int64_t estimatedTripCount, int loopBodySize) {
    if (estimatedTripCount <= 8) return estimatedTripCount;
    if (estimatedTripCount <= 64) return 4;
    if (estimatedTripCount <= 256) return 2;
    return 1;
}

// 3. 部分展开
//    for (i = 0; i < n; i+=4)
```

#### 4.6.4 实现位置
```
src/opt/PartialUnroll.cpp  (新增)
```

---

### 4.7 TailRecElim（尾递归消除）

#### 4.7.1 目标
将简单尾递归转换为迭代，解决 knapsack_naive 中的简单分支。

#### 4.7.2 识别条件
```cpp
// 尾递归模式：
// return func(args)  // 简单尾递归
// return op(func(args), ...)  // 复合尾递归
```

#### 4.7.3 转换算法

```cpp
// 优化前：
int foo(int n) {
    if (n == 0) return 0;
    return foo(n - 1);  // 尾递归
}

// 优化后：
int foo_iter(int n) {
    while (n != 0) {
        n = n - 1;
    }
    return 0;
}
```

#### 4.7.4 实现位置
```
src/opt/TailRecElim.cpp  (新增)
```

---

### 4.8 DPTransforms（DP 模式识别与转换）

#### 4.8.1 目标
识别已知 DP 模式并转换为 tabular DP，解决 knapsack_naive。

#### 4.8.2 knapsack 递归结构分析

```
knapsack(i, w)
├── knapsack(i-1, w)              [without_item]
└── knapsack(i-1, w-weight[i-1]) [with_item]
    └── max(without_item, value[i-1] + with_item)
```

这是 DAG，可以通过填表实现（参考 FPL 的多面体分析思路）。

#### 4.8.3 识别算法

```cpp
// 识别 0/1 背包
bool recognizeKnapsack(FuncOp* func) {
    // 1. 检查函数签名
    // 2. 检查递归调用数量：2 个
    // 3. 检查组合操作：max
    // 4. 检查参数模式
}
```

#### 4.8.4 实现位置
```
src/opt/DPTransforms.cpp  (新增)
```

---

### 4.9 LightweightSuperopt（轻量超优化）

#### 4.9.1 目标
实现类似 Souper 的超优化能力，但不引入 z3。

#### 4.9.2 参考资源
- Souper：MLIR 超优化器，使用 SMT 证明等价性
- z3：SMT 求解器，用于表达式等价性证明

#### 4.9.3 实现思路

```cpp
// 不使用 z3，使用简化的模式匹配 + 穷举验证

// 1. 已知优化模式表
struct OptPattern {
    const char* before;   // 优化前模式
    const char* after;    // 优化后模式
    bool (*verify)();     // 可选的验证函数
};

// 模式表示例：
// {"(add x x)", "(shl x 1)", nullptr},           // x + x = x << 1
// {"(mul x 4)", "(shl x 2)", nullptr},           // x * 4 = x << 2
// {"(div x 2)", "(shr x 1)", nullptr},           // x / 2 = x >> 1

// 2. 遍历函数中的指令序列
// 3. 对每个序列，检查模式表是否有更优替换
// 4. 使用 SMT-like 推理验证等价性（简化版）
```

#### 4.9.4 与 StrengthReduce 的区别

| 维度 | StrengthReduce | LightweightSuperopt |
|------|---------------|---------------------|
| 方法 | 确定性规则 | 模式匹配 + 穷举 |
| 覆盖 | 乘除法强度削减 | 更广的代数简化 |
| 验证 | 编译期常量检查 | 简化的 SMT 推理 |

#### 4.9.5 实现位置
```
src/opt/LightweightSuperopt.cpp  (新增)
```

---

## 5. 实现顺序

### 阶段 1：基础优化（所有测试受益）

```
顺序 1: StrengthReduce
  - 位置：src/opt/StrengthReduce.cpp
  - 依赖：无
  - 影响：所有测试
  - 原因：最基础、最通用，不会破坏正确性

顺序 2: LoopInterchange
  - 位置：src/opt/LoopInterchange.cpp
  - 依赖：StrengthReduce
  - 影响：transpose 系列
  - 原因：独立模块，易于验证
```

### 阶段 2：循环优化

```
顺序 3: Fusion 增强
  - 位置：src/opt/IndexEquiv.cpp + 修改 Fusion.cpp
  - 依赖：StrengthReduce
  - 影响：many_mat_cal, 03_sort1

顺序 4: LICM 增强
  - 位置：修改 src/opt/LICM.cpp
  - 依赖：StrengthReduce, SCEV
  - 影响：optimization_scheduling1

顺序 5: PartialUnroll
  - 位置：src/opt/PartialUnroll.cpp
  - 依赖：StrengthReduce, SCEV
  - 影响：fft 系列
```

### 阶段 3：边界优化

```
顺序 6: BoundsCheck
  - 位置：src/opt/BoundsCheck.cpp
  - 依赖：LoopInterchange
  - 影响：conv2d 系列
```

### 阶段 4：递归优化

```
顺序 7: TailRecElim
  - 位置：src/opt/TailRecElim.cpp
  - 依赖：StrengthReduce
  - 影响：knapsack_naive（简单分支）

顺序 8: DPTransforms
  - 位置：src/opt/DPTransforms.cpp
  - 依赖：TailRecElim
  - 影响：knapsack_naive（完整版本）
```

### 阶段 5：高级优化

```
顺序 9: LightweightSuperopt
  - 位置：src/opt/LightweightSuperopt.cpp
  - 依赖：StrengthReduce
  - 影响：所有测试（代数简化）
```

---

## 6. 预期效果

| 测试 | 瓶颈 | 对应模块 | 预计提升 |
|------|------|---------|---------|
| conv2d-1 | 边界检查 + stencil | BoundsCheck + PartialUnroll | ~5-8x |
| fft1 | 内层循环 | PartialUnroll + StrengthReduce | ~10-15x |
| 03_sort1 | 计数循环 | Fusion + PartialUnroll | ~5-8x |
| optimization_scheduling1 | 依赖链 | LICM 增强 + StrengthReduce | ~3-5x |
| knapsack_naive-1 | 递归 | DPTransforms | ~3-6x |
| many_mat_cal | 多循环 | Fusion | ~1.3-1.5x |
| transpose2 | 转置 | LoopInterchange | ~1.3-1.5x |

---

## 7. 风险与缓解

| 模块 | 风险 | 影响 | 缓解措施 |
|------|------|------|---------|
| StrengthReduce | 整数溢出 | 中 | 只对无符号/已知范围操作数优化 |
| LoopInterchange | 误判依赖方向 | 高 | 保守：只在能证明安全时交换 |
| BoundsCheck | 边界分析错误 | 严重 | 保守：只在确定安全时消除检查 |
| LICM 增强 | 链式传播错误 | 高 | 使用开关，默认关闭 |
| PartialUnroll | 展开因子不当 | 中 | 可配置：默认 2/4 |
| TailRecElim | 尾递归误判 | 中 | 只处理明确的尾递归模式 |
| DPTransforms | DP 模式误识别 | 中 | 先验证，后转换 |
| LightweightSuperopt | 穷举爆炸 | 低 | 只穷举小序列（≤5 条指令） |

---

## 8. 测试计划

### 8.1 单元测试

```bash
tests/opt/
├── strength_reduce/       # 乘除转位移
├── interchange/           # 循环交换
├── fusion_equiv/          # 下标等价融合
├── bounds_check/          # 边界检查消除
├── licm_chain/           # 链式不变计算
├── partial_unroll/        # 部分展开
├── tail_rec/             # 尾递归消除
├── dp_transforms/         # DP 模式转换
└── superopt/             # 超优化
```

### 8.2 集成测试

```bash
scripts/regression.sh tests/opt riscv O2
scripts/eval-runtime.sh test2026/performance riscv O2
```

---

## 9. 开关控制

```cpp
// 新增命令行选项
--enable-strength-reduce     // 强度削减（默认开启）
--enable-loop-interchange    // 循环交换（默认开启）
--enable-fusion-equiv        // 下标等价融合（默认关闭）
--enable-bounds-check       // 边界检查优化（默认关闭）
--enable-licm-chain        // 链式 LICM（默认关闭）
--enable-partial-unroll     // 部分展开（默认开启）
--enable-tail-rec          // 尾递归消除（默认开启）
--enable-dp-transforms      // DP 模式转换（默认关闭）
--enable-superopt          // 超优化（默认关闭）
```

---

## 10. 参考资源汇总

| 资源 | 可应用内容 | 对应模块 |
|------|----------|---------|
| 剑桥编译课程 | Abstract Interpretation | Range.cpp 增强, LICM 增强 |
| MLIR/Clang IR | Declarative Patterns, Affine | Matcher 增强, LoopInterchange, BoundsCheck |
| FPL | Polyhedral Math | BoundsCheck (简化版), 完整多面体 (方案 B) |
| z3/Souper | Superoptimization | LightweightSuperopt |

---

## 11. 总结

本方案通过 9 个优化模块，覆盖了测试集中所有大性能差距场景：

```
┌─────────────────────────────────────────────────────────┐
│                    优化模块覆盖                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  StrengthReduce       → 所有测试（基础）                 │
│  LoopInterchange      → transpose                        │
│  Fusion 增强          → many_mat_cal, 03_sort1        │
│  BoundsCheck          → conv2d                           │
│  LICM 增强            → optimization_scheduling          │
│  PartialUnroll        → fft                             │
│  TailRecElim          → knapsack_naive (简单)          │
│  DPTransforms         → knapsack_naive (完整)           │
│  LightweightSuperopt  → 所有测试（代数简化）             │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

实施顺序遵循从低风险到高风险、从通用到专用的原则，确保每一步都可验证、可回滚。

关键设计决策：
1. 不引入 MLIR/z3 等外部依赖，保持自研路线
2. 参考 MLIR 的声明式模式思想增强 Matcher
3. 参考 FPL 的数学原理，但使用简化版本（足够处理 stencil）
4. 参考 Souper 的超优化思路，实现轻量版本
