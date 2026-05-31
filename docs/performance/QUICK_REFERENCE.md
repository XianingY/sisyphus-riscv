# Sisyphus 编译器性能优化 - 快速参考

> Compliance note (2026-05-31): this document is an analysis and planning aid for the public RISC-V performance suite. Case names are reporting-only. Default O1 remains scalar rv64gc and may only use self-MLIR/affine/memref/machine facts; RVV, StructuralBitwise, StructuralModMul, FunctionEquivalence, Cached precompute, and helper replacement remain explicit opt-in experiments.


## 📊 测试套件概览

```
当前公开 performance_riscv 测试集合（以本仓库 .sy 文件为准）
│
├─ 矩阵计算 (35%) ────── 10 用例 ──── 密集三重循环
│  ├─ matmul*.sy (200×200, 250×250, 300×300)
│  ├─ many_mat_cal*.sy (410-412×412, 重复×8-12)
│  ├─ transpose*.sy (20M 元素转置)
│  └─ h-5*, h-10*.sy (LU/TRSM)
│
├─ 密码学/位运算 (25%) ─ 9 用例 ──── 位运算集约、查表、递推
│  ├─ crypto-*.sy (伪 SHA-1)
│  ├─ crc*.sy (CRC32 - 最坏情况！)
│  ├─ huffman-*.sy (Huffman 编码)
│  └─ h-9*.sy (素数分解)
│
├─ 排序与数据结构 (10%) ─ 4 用例 ──── 分支密集、链表遍历
│  ├─ 03_sort*.sy (基数排序)
│  └─ shuffle*.sy (哈希表)
│
├─ 卷积 (10%) ────────── 3 用例 ──── 6 重嵌套循环
│  └─ conv2d-*.sy (5×5 卷积核)
│
└─ 其他 (20%) ────────── 15 用例 ──── FFT、DP、数值计算
   ├─ fft*.sy
   ├─ h-1*, h-4*, h-8*.sy
   ├─ sl*.sy (多维数组)
   ├─ knapsack*.sy (背包 DP)
   ├─ prime_search*.sy
   └─ optimization_scheduling*.sy
```

---

## ⚡ 最高优先级优化（即时可用）

### 🔴 Issue #1: CRC/Crypto/Huffman 位运算灾难

**问题**: `crc*.sy`, `crypto*.sy`, `huffman*.sy` 中的位运算用循环实现

```c
// ❌ 错误实现 (32 次迭代!)
int _and(int a, int b) {
    int len = 32, result = 0, power = 1;
    while (len) {
        bit_a = a % 2;
        bit_b = b % 2;
        a = a / 2;
        b = b / 2;
        if (bit_a == 1 && bit_b == 1) {
            result = result + power;
        }
        power = power * 2;
        len = len - 1;
    }
    return result;
}
```

**影响**: CRC 测试 **100-200 倍变慢** 😱

**解决方案**: 通过 ProvenBitwiseHelper 在 IR 证明后 lowering 到硬件指令

```c
// ✓ 正确实现 (1 条指令!)
int _and_fixed(int a, int b) {
    return a & b;
}

int _xor_fixed(int a, int b) {
    return a ^ b;
}

int _or_fixed(int a, int b) {
    return a | b;
}
```

**收益**: **+50-100 倍** ⭐⭐⭐⭐⭐

---

### 🔴 Issue #2: 位旋转实现不效

**问题**: `crypto-*.sy` 中的 `rotl1/5/30` 用乘除实现

```c
// ❌ 低效 (5 条指令, ~5 周期)
int rotl1(int x) {
    return x * 2 + x % 2;  // 乘法 + 除法！
}

int rotl5(int x) {
    return x * 32 + x % 32;  // 更糟
}
```

**解决方案**: 编译器识别并优化

```c
// ✓ 高效 (3 条指令, ~2 周期)
int rotl1_opt(int x) {
    return (x << 1) | (x >> 31);  // 编译为 rol x, 1
}

int rotl5_opt(int x) {
    return (x << 5) | (x >> 27);  // 编译为 rol x, 5
}
```

**收益**: **+30-50%**

---

### 🟡 Issue #3: 矩阵计算缓存失配

**问题**: `matmul*.sy` 访存模式不友好

```c
// 循环顺序: k-i-j (列主序访问 B，缓存失配)
for (k = 0; k < n; k++)
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            C[i][j] += A[i][k] * B[k][j];  // ❌ B 跳跃访问
```

**解决方案**: 启用循环交换

```c
// 循环顺序: i-k-j (行主序访问 B，缓存友好)
for (i = 0; i < n; i++)
    for (k = 0; k < n; k++)
        for (j = 0; j < n; j++)
            C[i][j] += A[i][k] * B[k][j];  // ✓ B 顺序访问
```

**收益**: **+15-20%** (缓存命中率 45% → 70%)

---

### 🟡 Issue #4: 矩阵转置灾难

**问题**: `transpose*.sy` 最坏的缓存访问模式

```c
// 写操作: 每次跳跃 (步长 = colsize) → 缓存污染
output[j*colsize + i] = input[i*rowsize + j];
```

**预期缓存命中率**: 仅 **5%** 😱

**解决方案**: 分块转置

```c
// 4×4 块转置 (块完全适应 L1)
for (int ii = 0; ii < rowsize; ii += 4) {
    for (int jj = 0; jj < colsize; jj += 4) {
        for (int i = ii; i < ii+4; i++) {
            for (int j = jj; j < jj+4; j++) {
                out[j*colsize + i] = in[i*rowsize + j];
            }
        }
    }
}
```

**收益**: **+200-300%** (5% → 80% 缓存命中)

---

### 🟡 Issue #5: RVV 只能作为显式实验

**问题**: RVV 对矩阵/卷积很有吸引力，但 `docs/Requirements.md`
约束默认初赛路径为 `rv64gc`，所以默认 O1 不能发 RVV/SIMD。

**合规方案**: 默认做 scalar affine tiling/interchange；RVV 只用于
显式 opt-in A/B 实验。

```bash
# 测试
SISY_ENABLE_VECTORIZE=1 \
./build/compiler test2026/performance_riscv/matmul1.sy \
    -S -O2 --target=riscv

# 显式实验预期: RVV 指令 (v.*, vmul.vv, vadd.vv)
```

**收益**: **+25-35%** (4-8 倍吞吐量)

---

## 📈 按收益排序的优化清单

| 优先级 | 优化项 | 受影响用例 | 收益 | 工作量 |
|-------|--------|---------|------|-------|
| **P0** | 修复位运算 | CRC, Crypto, Huffman (9) | **+50-100%** | ⭐ 极低 |
| **P0** | 位旋转识别 | Crypto (3) | +30-50% | ⭐ 低 |
| **P1** | 启用循环交换 | Matmul (10) | +15-20% | ⭐ 低 |
| **P1** | 显式 opt-in 向量化 | Matmul + Conv (15) | +25-35% | ⭐ 低 |
| **P2** | 实现循环分块 | Matmul, Transpose, Conv (15) | +20-50% | ⭐⭐ 中 |
| **P2** | 卷积识别 | Conv2d (3) | +15-30% | ⭐⭐ 中 |
| **P3** | 指令调度 | All public perf cases | +10-15% | ⭐⭐ 中 |
| **P3** | 模运算优化 | Crypto, Prime (6) | +8-15% | ⭐⭐ 中 |

---

## 🎯 实施策略

### Phase 1 - 即时（1 天）
```bash
# 验证现有优化标志
SISY_HIR_ENABLE_INTERCHANGE=1 \
SISY_ENABLE_VECTORIZE=1 \
SISY_ENABLE_STRUCTURAL_BITWISE=1 \
./build/compiler test.sy -S -O2 --target=riscv

# 如果生效，在实验 profile 中显式启用
# 预期收益: +30-40%
```

### Phase 2 - 短期（2-3 天）
```bash
# 实现循环分块优化
# src/hir/LoopTiling.cpp (新文件)

# 在 O2 中启用
# 预期收益: 另外 +20-30%
```

### Phase 3 - 中期（1 周）
```bash
# 实现卷积识别
# 改进指令调度
# 预期收益: 另外 +15-20%
```

---

## 🔧 三步验证方法

### Step 1: 编译
```bash
./build/compiler test2026/performance_riscv/crc1.sy \
    -S -O2 --target=riscv -o test_crc1.s
```

### Step 2: 分析汇编
```bash
# 查看位运算是否优化
grep -E "and|xor|or" test_crc1.s | head -20

# 应该看到:
# and a1, a1, a2     (✓ 正确)
# 而不是:
# while loop with div/rem  (✗ 错误)
```

### Step 3: 性能测试
```bash
# 编译并运行，测量执行时间
time ./a.out < test2026/performance_riscv/crc1.in
```

---

## 📋 检查清单

### 立即行动
- [ ] 确认 `crc3.sy` 中的 `_and()` 是否用 100+ 条指令实现
  - 如果是，这是第一优先级修复
- [ ] 查找所有位运算函数（`_and`, `_xor`, `_or`）
  - 所有循环实现都应改为直接硬件操作
- [ ] 验证位旋转是否用乘除实现
  - 所有 `rotl*()` 函数应转换为移位或 `rol` 指令

### 编译器改动
- [ ] 在实验 profile 中显式启用 `SISY_HIR_ENABLE_INTERCHANGE`
- [ ] 在 O2 profile 中保持 `SISY_ENABLE_VECTORIZE` 显式 opt-in
- [ ] 在 O2 profile 中保持 `SISY_ENABLE_STRUCTURAL_BITWISE` 显式 opt-in

### 新增优化通道
- [ ] 实现 `LoopTiling.cpp` (循环分块)
- [ ] 实现 `ConvolutionLowering.cpp` (卷积识别)
- [ ] 改进 `RVScheduler.cpp` (指令调度)

---

## 📚 相关文件

```
/Users/byzantium/github/sisyphus/
├─ PERFORMANCE_OPTIMIZATION_GUIDE.md    # 完整优化指南 (45KB)
├─ IMPLEMENTATION_ROADMAP.md            # 实施路线图 (50KB)
├─ TEST_CASE_ANALYSIS.md                # 测试深度分析 (80KB)
└─ QUICK_REFERENCE.md                   # 本文件
```

---

## 🎓 学习资源

- **RISC-V 向量扩展**: https://github.com/riscv/riscv-v-spec
- **矩阵乘法优化**: "The BLIS Framework" (Flame@UT)
- **编译器优化**: "Engineering a Compiler" (Cooper & Torczon)
- **性能分析**: "Software Optimization Secrets" (Zurawski)

---

## 💡 关键洞察

1. **最大收益来自低优先级的修复**: 位运算修复可能带来 50-100 倍改进
2. **缓存是关键**: 大部分矩阵测试的瓶颈都是缓存（交换、分块可解决）
3. **向量化威力巨大但默认受限**: RVV 可以作为实验方向，默认 O1
   仍优先做 scalar affine/memref/machine 优化。
4. **分支预测**: 对于 RISC-V，分支失败代价高（8-12 周期），无分支实现重要
5. **编译器识别模式**: 许多低效代码源于编译器未识别高级模式（位旋转、卷积等）

---

**最后提醒**: CRC 和 Crypto 测试中的位运算实现是编译器可以快速获得巨大收益的地方。集中精力在这些区域可能比优化矩阵计算更经济高效。

Good luck! 🚀
