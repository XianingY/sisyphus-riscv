# Sisyphus 编译器性能优化指南

> Compliance note (2026-05-31): this document is an analysis and planning aid for the public RISC-V performance suite. Case names are reporting-only. Default O1 remains scalar rv64gc and may only use self-MLIR/affine/memref/machine facts; RVV, StructuralBitwise, StructuralModMul, FunctionEquivalence, Cached precompute, and helper replacement remain explicit opt-in experiments.


## I. 测试用例分类与特征分析

### 1. 密集矩阵计算（35%权重）
- **用例**: `01_mm*.sy`, `matmul*.sy`, `many_mat_cal*.sy`, `transpose*.sy`, `h-5*.sy`, `h-10*.sy`
- **特征**:
  - 三重嵌套循环（k-j-i 标准矩阵乘法）
  - 大型多维数组（1024×1024、1400×1400）
  - 密集的内存访问模式
  - 整数/浮点混合计算

**关键观察**:
```
matmul1.sy (200×200):   3重嵌套 + 修剪 + 转置 + 最小值传播
matmul2.sy (300×300):   更大规模，缓存压力更大
many_mat_cal (410-412×412×10): 重复矩阵计算，需要向量化
```

**当前瓶颈**:
- ❌ 循环重新排序未被充分利用（`SISY_HIR_ENABLE_INTERCHANGE` 被禁用）
- ❌ 向量化仅在明确启用时工作（`SISY_ENABLE_VECTORIZE=1`）
- ❌ 缓存未被充分优化（无分块策略）
- ❌ 超标量执行机会被浪费

---

### 2. 位运算与密码学（25%权重）
- **用例**: `crypto-*.sy`, `crc*.sy`, `huffman*.sy`, `h-9*.sy`
- **特征**:
  - 密集的位移、异或、按位与非操作
  - 查表（crc32table[256]）和递推（伪 SHA-1）
  - 循环体内部分支较少但数据依赖链长

**关键代码模式**:
```c
// CRC: 逐位实现的位运算（性能杀手）
int _and(int a, int b) {
    while (len) {
        bit_a = a % 2;
        bit_b = b % 2;
        if (bit_a == 1 && bit_b == 1) result += power;
        // 每次迭代：2次除法、1次乘法、多次条件判断
    }
}

// Crypto: 大量位旋转和运算组合
int rotl1(int x) { return x * 2 + x % 2; }  // 可展开为单条指令
int rotl5(int x) { return x * 32 + x % 32; }
```

**当前瓶颈**:
- ❌ 位运算序列未被识别和优化（缺少 `SISY_ENABLE_STRUCTURAL_BITWISE` 在 O2）
- ❌ 模运算和乘法未转换为更便宜的操作
- ❌ 位旋转没有转换为 RISC-V `ror` 伪指令

---

### 3. 卷积与图像处理（10%权重）
- **用例**: `conv2d-*.sy`
- **特征**:
  - 5×5 卷积核，多重循环（6重）
  - 边界检查在内层循环
  - 读写模式复杂（非单调）

**当前瓶颈**:
- ❌ 卷积模式未被识别（缺少 `SISY_ENABLE_ADVANCED_CONV2D`）
- ❌ 核权重未被编译期评估
- ❌ 边界检查与计算缠绕（无数据流图优化）

---

### 4. 排序与数据结构（10%权重）
- **用例**: `03_sort*.sy`, `shuffle*.sy`, `knapsack_naive*.sy`
- **特征**:
  - 基数排序（多轮分组/交换）
  - 哈希表操作（链表遍历、冲突解决）
  - 深层递归（背包 DP）

**当前瓶颈**:
- ❌ 分支预测失败（sort 中的条件交换）
- ❌ 无分支 min/max 未被使用
- ❌ 尾递归未消除（背包问题）

---

### 5. 其他计算密集型（20%权重）
- **用例**: `fft*.sy`, `h-1*.sy`, `h-4*.sy`, `h-8*.sy`, `prime_search*.sy`, `transpose*.sy`
- **特征**:
  - FFT 递归 + 复数运算
  - 素数分解（试除法）
  - 矩阵转置（访存模式坏）
  - LU 分解（数值稳定性）

---

## II. 优化机制与当前状态

### A. 已启用的优化（O1 级别）

| 优化项 | 启用状态 | 代码位置 | 效果 |
|-----|--------|--------|------|
| 循环不变代码移出 | ✅ | `src/opt/LICM.cpp` | 中等 |
| 全局值编号 (GVN) | ✅ | `src/opt/GVN.cpp` | 中等 |
| 别名分析 | ✅ | `src/opt/AliasAnalysis.cpp` | 关键 |
| 死代码删除 | ✅ | `src/opt/DCE.cpp` | 小 |
| 内联 | ✅ | `src/opt/Inlining.cpp` | 中等 |
| 常数折叠 | ✅ | `src/pre-opt/ConstFold.cpp` | 中等 |

### B. 有条件启用的优化（需要环境变量）

| 优化项 | 环境变量 | 适用场景 | 效果评估 |
|-----|--------|--------|---------|
| **循环交换** | `SISY_HIR_ENABLE_INTERCHANGE=1` | 矩阵计算 | ⭐⭐⭐⭐⭐ 高 |
| **循环展开与融合** | `SISY_HIR_ENABLE_UNROLL_JAM=1` | 矩阵、卷积 | ⭐⭐⭐⭐ 高 |
| **向量化** | `SISY_ENABLE_VECTORIZE=1` | 所有循环密集 | ⭐⭐⭐⭐⭐ 极高 |
| **位运算结构化** | `SISY_ENABLE_STRUCTURAL_BITWISE=1` | 密码学 | ⭐⭐⭐ 中 |
| **模运算优化** | `SISY_ENABLE_STRUCTURAL_MODMUL=1` | 密码学、数论 | ⭐⭐⭐⭐ 高 |
| **行抓取矩阵乘** | `SISY_ENABLE_ROW_SCRATCH_MATMUL=1` | 矩阵乘法 | ⭐⭐⭐⭐ 高 |
| **预计算缓存** | `SISY_ENABLE_CACHED_PRECOMPUTE=1` | FFT、多项式 | ⭐⭐⭐ 中 |
| **函数等价性** | `SISY_ENABLE_FUNCTION_EQUIVALENCE=1` | 数据传递重构 | ⭐⭐⭐ 中 |
| **高级卷积** | `SISY_ENABLE_ADVANCED_CONV2D=1` | 卷积运算 | ⭐⭐⭐⭐ 高 |

### C. O2 级别的额外优化

```cpp
// src/main/PipelineProfiles.cpp 中的 O2Profile
- 更激进的循环展开
- 循环向量化（RVV 对 RISC-V）
- 更深的内联深度（>10 层）
- 多轮迭代优化
```

---

## III. 针对性优化方案

### 【优先级 1】矩阵计算优化（收益最大）

#### 1.1 启用多面体循环优化

**当前状态**: `SISY_HIR_ENABLE_INTERCHANGE` 在 O2 中被禁用

**方案**:
```bash
# 测试脚本验证收益
for test in matmul{1,2,3}.sy many_mat_cal-{1,2,3}.sy; do
    echo "=== $test ==="
    SISY_HIR_ENABLE_INTERCHANGE=1 SISY_ENABLE_VECTORIZE=1 \
    ./build/compiler tests/performance_riscv/$test -S -O2 --target=riscv
done

# 预期收益: 20-35% 加速（缓存命中率从 ~45% 提升到 ~70%）
```

**代码修改**:
修改 [src/main/PipelineProfiles.cpp](src/main/PipelineProfiles.cpp)，在 O2 Profile 中启用循环交换：

```cpp
// 在 PipelineProfile O2Profile 中
auto addO2Passes = [&](PassManager &pm) {
    // ... 已有的优化 ...
    if (std::getenv("SISY_HIR_ENABLE_INTERCHANGE")) {
        pm.add(createLoopInterchangePass());  // 当前被禁用
    }
};
```

#### 1.2 启用针对性向量化

**问题**: 当前向量化仅在 `SISY_ENABLE_VECTORIZE=1` 时工作

**方案**: 在实验配置中显式启用 RISC-V 向量化

```cpp
// src/rv/RVVectorizer.cpp 中需要的改进
bool shouldVectorize(Loop *L) {
    // 1. 识别可向量化循环（步长=1、无不规则访问）
    // 2. 检查数据依赖是否允许向量化
    // 3. 评估向量化收益 vs 代码膨胀

    // 当前缺陷: 过度保守的向量化启发式
    // 改进: 对于 matmul 的内层循环，几乎总是有益的
}
```

**预期收益**: RVV 向量化可以将内层循环加速 4-8 倍

#### 1.3 添加分块策略

**新增优化**: 矩阵乘法分块（Tiling）

```cpp
// src/hir/ 中新增 LoopTiling.cpp
// 将 C[i][j] += A[i][k] * B[k][j] 转换为分块形式
// 以提高缓存局部性
//
// 原始:      k < n
// 分块后:    kb < n by block_size
//            k < kb + block_size
//
// 缓存命中率提升: 45% -> 85%
```

---

### 【优先级 2】位运算与密码学优化

#### 2.1 识别和优化位旋转

**问题代码**:
```c
// crypto-1.sy 中的位旋转实现
int rotl1(int x) { return x * 2 + x % 2; }  // 应该是 rol 指令
int rotl5(int x) { return x * 32 + x % 32; } // 应该是 rol r, 5
```

**优化方案**:
在后端中添加窥孔优化识别这些模式

```cpp
// 新增 src/rv/RiscvPeephole.cpp 中的规则
Pattern: (x << k) | (x >> (32-k))  →  rol x, k
Pattern: (x * (1<<k)) | (x / (1<<k))  →  rol x, k
Pattern: (x * 2 + x % 2)  →  rol x, 1
```

**预期收益**: CRC/Crypto 测试加速 15-25%

#### 2.2 模运算优化

**问题代码**:
```c
// h-9*.sy 中
int prime_factors(int n) {
    if (n == 0) return -1;
    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {  // 昂贵的除法和模运算
            // ...
        }
    }
}
```

**优化**: 使用 Barrett Reduction 或预计算

```cpp
// 在 src/opt/ 中新增 ModularArithmetic.cpp
// 识别 (a * b) % m 模式
// 如果 m 是编译期常量，使用预计算的倒数来避免除法
//
// 例如: x % 998244353
// 转换为: (x * INV) >> SHIFT
// 其中 INV 和 SHIFT 在编译期计算
```

#### 2.3 结构化位操作识别

**启用现有优化**:

```bash
SISY_ENABLE_STRUCTURAL_BITWISE=1 SISY_ENABLE_STRUCTURAL_MODMUL=1 \
./build/compiler crypto-1.sy -S -O2 --target=riscv
```

**代码改进**:
在 `src/opt/StructuralAnalysis.cpp` 中添加更多位操作模式识别。

---

### 【优先级 3】卷积优化

#### 3.1 启用高级卷积优化

```bash
SISY_ENABLE_ADVANCED_CONV2D=1 \
./build/compiler conv2d-1.sy -S -O2 --target=riscv
```

#### 3.2 提前评估权重

**改进**: 在编译期识别并内联卷积核权重

```cpp
// src/hir/ConvolutionLowering.cpp 中新增
// 识别: K[idx(kr, kc, KSIZE)] 其中 K 是常量
// 转换为: 直接展开的表达式，消除查表
```

---

### 【优先级 4】分支密集型代码

#### 4.1 消除排序中的分支

**问题**: radixSort 中的条件分支造成分支预测失败

```cpp
// 当前代码中的分支
while (head[i] < tail[i]) {
    int v = a[head[i]];
    while (getNumPos(v, bitround) != i) {  // 难以预测的分支
        // ...
    }
}
```

**优化**: 使用 cmov（条件移动）指令代替分支

```cpp
// RISC-V: 生成无分支代码
// 使用 select 伪指令而非 bne
int new_head = (condition) ? head1 : head2;  // 编译为单条指令
```

#### 4.2 尾递归消除

**目标**: knapsack_naive 中的递归

```c
// 原始: 递归调用
int knapsack_naive(int i, int w) {
    if (weight[i-1] > w) return knapsack_naive(i-1, w);
    else return max(..., knapsack_naive(i-1, w-weight[i-1]));
}

// 优化: 转换为循环
int knapsack_optimized(int N, int W, int weight[], int value[]) {
    int dp[N+1][W+1];
    // 迭代 DP
}
```

---

## IV. 默认 O1 合规 Pipeline

默认评测路径是 `-O1 --target=riscv`，并且必须保持 `rv64gc` 标量输出。
因此本节不建议把 RVV 或高风险语义识别器放入默认 profile。可落地的
默认优化应当写成 self-MLIR 证明型 pass：

```cpp
struct DefaultO1Boundary {
    bool enableSelfAffineOpt = true;          // affine/dependence 证明后才交换、分块
    bool enableSelfMemOpt = true;             // memref/MemorySSA 证明后转发、DSE
    bool enableSynthConstArray = true;        // 全域验证的源常量表综合
    bool enableProvenBitwiseHelper = true;    // 完整 32-bit helper + runtime fallback
    bool enableMachineLiveness = true;        // 本地活跃性减少 home-spill

    bool enableRVV = false;                   // 显式 opt-in；默认禁用
    bool enableStructuralBitwise = false;     // 显式 opt-in；默认禁用
    bool enableStructuralModmul = false;      // 显式 opt-in；默认禁用
    bool enableFunctionEquivalence = false;   // 显式 opt-in；默认禁用
    bool enableCachedPrecompute = false;      // 显式 opt-in；默认禁用
    bool enableRowScratchHelper = false;      // 显式 opt-in；默认禁用
};
```

---

## V. 新增优化通道（建议实现）

### 1. 循环分块（Loop Tiling）- HIGH PRIORITY

**文件**: `src/hir/LoopTiling.cpp`

```cpp
// 伪代码
void tileLoops(Function *F, int tileSize = 64) {
    for (auto &loop : F->getLoops()) {
        if (!isBeneficial(loop)) continue;

        // 将循环分成外层（块）和内层（块内）
        // 原: for (i=0; i<N; i++) for (j=0; j<N; j++) ...
        // 新: for (ii=0; ii<N; ii+=TS)
        //      for (jj=0; jj<N; jj+=TS)
        //        for (i=ii; i<ii+TS; i++)
        //          for (j=jj; j<jj+TS; j++) ...
        //
        // 优势: 提高 L1 缓存利用率
    }
}
```

**预期收益**: 矩阵计算加速 25-40%

### 2. 指令调度优化 - MEDIUM PRIORITY

**文件**: `src/rv/RVScheduler.cpp`

```cpp
// 改进调度启发式以减少加载-使用延迟
void scheduleBasicBlock(BasicBlock *BB) {
    // 当前: 简单的贪心调度
    // 改进: DAG-based 调度，考虑 RISC-V 流水线特性

    // 对于 RV64GC:
    // - Load 延迟: 2 周期
    // - Multiply 延迟: 3 周期
    // - 浮点: 4 周期
}
```

**预期收益**: 各类计算加速 10-15%

### 3. 位运算模式库 - MEDIUM PRIORITY

**文件**: `src/opt/BitwisePatternDB.cpp`

预定义一个模式库，用于识别和优化常见的位运算序列：

```
Pattern "bit_extract":  ((x >> offset) & mask)  →  RISC-V bexti
Pattern "bit_deposit":  ((x & mask1) | (y & mask2))  →  optimize
Pattern "rotate_left":  ((x << k) | (x >> (32-k)))  →  rol
Pattern "popcount":     (count_bits(x))  →  RISC-V popcnt
Pattern "ctz":          (count_trailing_zeros(x))  →  RISC-V ctz
```

---

## VI. 测试和验证策略

### 阶段 1: 基准测试（当前状态）

```bash
#!/bin/bash
cd /Users/byzantium/github/sisyphus

# 编译编译器
scripts/build.sh

# 运行基准测试
for category in matmul crypto sort fft; do
    echo "=== Testing $category ==="
    time scripts/regression.sh tests/performance_riscv $category riscv O2
done
```

### 阶段 2: 逐步启用优化

```bash
# 测试 1: 启用循环交换
SISY_HIR_ENABLE_INTERCHANGE=1 \
time scripts/regression.sh tests/performance_riscv matmul riscv O2

# 测试 2: 显式 opt-in 向量化
SISY_ENABLE_VECTORIZE=1 \
time scripts/regression.sh tests/performance_riscv matmul riscv O2

# 测试 3: 同时启用多个优化
SISY_HIR_ENABLE_INTERCHANGE=1 SISY_ENABLE_VECTORIZE=1 \
SISY_ENABLE_STRUCTURAL_BITWISE=1 SISY_ENABLE_STRUCTURAL_MODMUL=1 \
time scripts/regression.sh tests/performance_riscv all riscv O2
```

### 阶段 3: 性能分析

```bash
# 使用 RISC-V QEMU 进行性能计数
SISY_COMPILER_PATH="$PWD/build/compiler" \
RUNTIME_SOFT_PERF=1 \
scripts/eval-runtime.sh official-riscv-perf riscv O2

# 输出指标:
# - 指令数量
# - 缓存命中率
# - 分支预测失败率
# - 执行周期
```

---

## VII. 按优先级实施计划

### 🔴 **阶段 1 - 立即实施（本周）**

1. ✅ 在实验配置中显式启用 `SISY_HIR_ENABLE_INTERCHANGE`
   - 文件: `src/main/PipelineProfiles.cpp`
   - 预期收益: +15% 矩阵性能

2. ✅ 在 O2 中保持 `SISY_ENABLE_VECTORIZE` 显式 opt-in
   - 文件: `src/main/PipelineProfiles.cpp`
   - 预期收益: +25% 矩阵性能

3. ✅ 在实验配置中显式启用位运算优化标志
   - 预期收益: +10% 密码学性能

### 🟡 **阶段 2 - 近期优化（1-2周）**

4. 实现循环分块（Tiling）
   - 新文件: `src/hir/LoopTiling.cpp`
   - 预期收益: +20% 矩阵性能

5. 改进指令调度
   - 修改: `src/rv/RVScheduler.cpp`
   - 预期收益: +12% 全局性能

6. 添加卷积模式识别
   - 启用: `SISY_ENABLE_ADVANCED_CONV2D` 在 O2
   - 预期收益: +15% 卷积性能

### 🟢 **阶段 3 - 长期优化（2-4周）**

7. 构建位运算模式库
8. 实现无分支排序变体
9. 尾递归消除通道
10. 预计算和缓存策略优化

---

## VIII. 预期性能提升

### 保守估计（启用基础优化）
```
Matmul:     40-50% 加速
Crypto:     15-20% 加速
Sort:       10-15% 加速
Conv2D:     20-30% 加速
FFT:        15-25% 加速
Overall:    ~25-30% 平均加速
```

### 激进估计（全部优化启用）
```
Matmul:     60-80% 加速（默认 scalar tiling；RVV 仅显式实验）
Crypto:     30-40% 加速（位运算优化）
Sort:       20-30% 加速（无分支变体）
Conv2D:     40-50% 加速（高级识别）
FFT:        30-40% 加速（预计算）
Overall:    ~40-50% 平均加速
```

---

## IX. 参考资源

- RISC-V 向量扩展 (RVV): https://github.com/riscv/riscv-v-spec
- 多面体编译器: https://polly.llvm.org/
- 矩阵乘法优化: https://github.com/flame/blis
- 密码学库优化: https://github.com/BearSSL
