# 测试用例深度分析与优化方案

> Compliance note (2026-05-31): this document is an analysis and planning aid for the public RISC-V performance suite. Case names are reporting-only. Default O1 remains scalar rv64gc and may only use self-MLIR/affine/memref/machine facts; RVV, StructuralBitwise, StructuralModMul, FunctionEquivalence, Cached precompute, and helper replacement remain explicit opt-in experiments.


## 1. 矩阵乘法系列 (10 个测试用例)

### 1.1 `matmul*.sy` (200×200, 250×250, 300×300)

**代码模式**:
```c
// matmul1.sy: 200×200
for (i = 0; i < 200; i++) {
    for (j = 0; j < 200; j++) {
        for (k = 0; k < 200; k++) {
            if (a[i][k]*b[k][j] % 2 == 0)
                temp += b[i][k]*a[k][j];
        }
        c[i][j] = temp;
    }
}
```

**性能分析**:
- **内存访问模式**: 行-列-行 (非最优)
  - `a[i][k]`: 行序访问 ✓
  - `b[k][j]`: 列序访问 ✗ (缓存失配严重)
  - `b[i][k]`: 行序访问 ✓
  - 每次迭代 3 次访存，2 次条件判断

- **缓存行为**:
  - L1 缓存行大小: 64 字节 (16 个 int)
  - 200×200 矩阵 = 160KB，超出 L1 容量
  - 预期缓存命中率: ~35-45%

- **分支行为**:
  - `if (a[i][k]*b[k][j] % 2 == 0)`: 约 50% 分支率
  - 分支预测失败代价: 8-12 周期延迟

**优化方案**:
```cpp
// 优化 1: 循环交换（j-k 交换或 k-i 交换）
for (i = 0; i < 200; i++) {
    for (k = 0; k < 200; k++) {          // k 外移
        for (j = 0; j < 200; j++) {      // j 变内层
            if (a[i][k]*b[k][j] % 2 == 0)
                temp += b[i][k]*a[k][j];  // 现在 b[k][j] 是顺序访问
        }
    }
}

// 优化 2: 循环分块（64×64）
for (ii = 0; ii < 200; ii += 64) {
    for (kk = 0; kk < 200; kk += 64) {
        for (jj = 0; jj < 200; jj += 64) {
            for (i = ii; i < ii+64; i++) {
                for (k = kk; k < kk+64; k++) {
                    for (j = jj; j < jj+64; j++) {
                        if (a[i][k]*b[k][j] % 2 == 0)
                            temp += b[i][k]*a[k][j];
                    }
                }
            }
        }
    }
}
// 分块后，64×64 矩阵块 = 16KB，完全适应 L1

// 优化 3: 消除分支
int mask = (a[i][k]*b[k][j] % 2) ? 0 : -1;
temp += (b[i][k]*a[k][j]) & mask;  // 无分支条件选择

// 优化 4: 向量化
// 使用 RISC-V RVV 指令
//   vle32.v v1, (a[i][k])   // 向量加载
//   vle32.v v2, (b[k][j])
//   vmul.vv v3, v1, v2      // 向量乘法
//   用 4-8 元素的向量操作替代循环
```

**预期收益**:
- 循环交换: +15-20%（缓存命中 ~70%）
- 循环分块: +20-30%（缓存命中 ~85%）
- 无分支: +5-10%（减少分支预测失败）
- **向量化**: **+40-60%**（4-8 倍吞吐量）
- **总计**: **+60-80%** 加速

**优化前后汇编对比**:

优化前:
```asm
loop_j:
    ld a5, 0(a4)        # 加载 a[i][k]
    ld a6, 0(a5)        # 加载 b[k][j]  (不同地址，缓存失配)
    mul a7, a5, a6      # 乘法
    andi a8, a7, 1      # 取模 2
    bne a8, zero, skip  # 分支（难以预测）
    # ...
    addi a2, a2, 1
    blt a2, 200, loop_j
```

优化后（用 RVV）:
```asm
loop_block_j:
    vle32.v v1, (a4)       # 向量加载 4-8 个元素
    vle32.v v2, (a5)       # 向量加载
    vmul.vv v3, v1, v2     # 向量乘法
    vadd.vv v4, v4, v3     # 向量累加
    addi a2, a2, 8
    blt a2, 200, loop_block_j
```

---

### 1.2 `many_mat_cal*.sy` (410-412 维，重复 8-12 次)

**代码模式**:
```c
// many_mat_cal-1.sy: 多次矩阵乘法和运算
int T = 412, R = 10;
for (r = 0; r < R; r++) {
    // 矩阵乘法: C += A * B
    // 矩阵乘法: B = A * C  (结果反复使用)
    // ... 更多运算
}
```

**性能分析**:
- **复杂度**: O(T³ × R) = O(412³ × 10) ≈ 7 × 10¹¹ 操作
- **主要瓶颈**:
  - 重复矩阵乘法导致严重缓存压力
  - 矩阵大小 (412×412) 略大于最优分块大小
  - 结果重用但访问模式复杂

**优化方案**:
```cpp
// 优化 1: 缓存感知的分块大小调整
// 对于 L1 缓存 (32KB):
//   1 个 412×412 矩阵 = 680 KB
//   最优分块: ~20×20 (32×32 对 int 是 4KB)
//   OR 使用非方形块: 32×24

// 优化 2: 预计算中间结果
// 而不是每次重新计算

// 优化 3: 预热缓存
// 第一次矩阵乘法会产生缓存失配
// 第二、三次会利用缓存

// 优化 4: 在 O2 中启用激进内联
// 内联矩阵乘法函数以减少函数调用开销
```

**预期收益**: **+35-50%** (通过分块和缓存优化)

---

### 1.3 `transpose*.sy` (20000000 元素的矩阵转置)

**代码模式**:
```c
// transpose.sy: 矩阵转置操作
int transpose(int n, int matrix[], int rowsize) {
    int colsize = n / rowsize;
    for (int i = 0; i < colsize; i++) {
        for (int j = 0; j < rowsize; j++) {
            // 从行主序读，列主序写
            output[j*colsize + i] = input[i*rowsize + j];
        }
    }
}
```

**性能分析**:
- **访存模式**: 最坏情况 (完全不友好)
  - 读: 顺序 ✓
  - 写: 每次跳跃 (步长 = colsize) ✗✗

- **缓存行为**:
  - L1 缓存行宽度: 64 字节
  - 每个缓存行只使用 1 个元素，浪费 15 个元素
  - 缓存污染: 来自写操作的无用数据占满 L1

- **预期缓存命中率**: ~5% (灾难级别)

**优化方案**:
```cpp
// 优化 1: 分块转置（4×4 或 8×8 块）
// 将大型转置分解为小块转置
// 每个块完全适应缓存

for (int ii = 0; ii < rowsize; ii += BLOCK_SIZE) {
    for (int jj = 0; jj < colsize; jj += BLOCK_SIZE) {
        // 转置 BLOCK_SIZE × BLOCK_SIZE 块
        // 块内转置更缓存友好
    }
}

// 优化 2: 使用临时缓冲区
// 避免交织读写

// 优化 3: 向量化读写
// 使用 RISC-V 向量指令加载/存储多个元素
```

**预期收益**: **+200-300%** (从灾难级缓存性能恢复)

---

### 1.4 `h-5*.sy` - LU 分解 (1400×1400)

**代码模式**:
```c
void kernel_ludcmp(int n, int A[][1400], int b[], int x[], int y[]) {
    // LU 分解
    for (i = 0; i < n; i++) {
        for (j = 0; j < i; j++) {
            w = A[i][j];
            for (k = 0; k < j; k++) {
                w -= A[i][k] * A[k][j-1];  // 数据依赖链
            }
            A[i][j] = w / A[j][j];
        }
        // ...
    }
    // 前向/后向代入
}
```

**性能分析**:
- **数据依赖**: 极强 (A[i][k] 依赖前面的计算)
  - 每条依赖链长度 ~20 指令
  - 无法通过循环展开增加指令级并行度

- **内存访问**: 三角形模式
  - 第 i 行只访问前 i 个元素
  - 前几行访问少，后面行访问多
  - 不规则访问模式

**优化方案**:
```cpp
// 优化 1: 减少依赖链长度通过分块
// LAPACK 中的标准技巧

// 优化 2: 指令调度优化
// 在加载和使用之间插入独立指令
for (i = ...) {
    w1 = A[i][k1] * A[k1][j];
    // 插入独立计算（如果有）
    w2 = A[i][k2] * A[k2][j];
    sum = w1 + w2;
}

// 优化 3: 避免除法
// A[i][j] = w / A[j][j] 很昂贵
// 预计算 1/A[j][j]，然后乘法

// 优化 4: 内存对齐
// 确保对齐访问以减少缓存失配
```

**预期收益**: **+15-25%** (指令调度和算术优化)

---

## 2. 密码学与位运算 (9 个测试用例)

### 2.1 `crypto-*.sy` - 伪 SHA-1 哈希

**代码模式**:
```c
int rotl1(int x) { return x * 2 + x % 2; }        // 应该用 rol 指令
int rotl5(int x) { return x * 32 + x % 32; }      // 应该用 rol 指令
int rotl30(int x) { return x * 1073741824 + x % 1073741824; }

int _and(int a, int b) { return a + b; }          // ❌ 错误！应该是 a & b
int _xor(int a, int b) { return a - _and(a,b) + b - _and(a,b); }
int _or(int a, int b) { return _xor(_xor(a,b), _and(a,b)); }

void pseudo_md5(int input[], int input_len, int output[]) {
    for (i = 0; i < 64; i++) {
        // 循环展开的哈希计算
        // 大量的位运算操作
    }
}
```

**性能分析**:
- **位运算实现不优**:
  - `rotl1(x)` = `x*2 + x%2` 需要 1 次乘法 + 1 次除法 (RISC-V: 3+2 周期)
  - 应该是 `(x << 1) | (x >> 31)` 或 `rol x, 1` (RISC-V: 1-2 周期)
  - **损耗**: 2-3 倍性能

- **位逻辑混乱**:
  - `_and(a,b) = a + b` 是错误实现
  - 应该是硬件按位与
  - 这会导致整个哈希计算出错

- **循环特征**:
  - 64 次迭代的内循环，无数据依赖
  - 完全可以向量化或完全展开

**优化方案**:
```cpp
// 优化 1: 位旋转识别和替换
Pattern: (x << k) | (x >> (32-k)) → rol x, k
Pattern: x * (1<<k) + x % (1<<k) → rol x, k
Pattern: x * 2 + x % 2 → rol x, 1

int rotl1_optimized(int x) {
    return (x << 1) | (x >> 31);  // 编译为单条 rol 指令
}

// 优化 2: 位逻辑正确实现
int _and_correct(int a, int b) { return a & b; }
int _xor_correct(int a, int b) { return a ^ b; }
int _or_correct(int a, int b) { return a | b; }

// 优化 3: 循环完全展开（小循环）
// 64 次迭代可以部分展开（展开因子 4-8）
for (i = 0; i < 64; i += 4) {
    // 4 次迭代的循环体
    // ...
    // ...
    // ...
    // ...
}

// 优化 4: 指令级并行
// 在位运算之间交错其他操作
a = x ^ y;
b = p ^ q;     // 独立，可并行执行
c = a & b;     // 等待 a, b
d = a | c;
```

**优化前后对比**:

```
优化前 rotl1(x):
  li a1, 1
  mul a1, x, a1         # a1 = x * 2
  rem a2, x, 2          # a2 = x % 2
  add a1, a1, a2        # return a1 + a2
  总: 3 条指令, ~5 周期

优化后 rotl1(x):
  sll a1, x, 1          # a1 = x << 1
  srl a2, x, 31         # a2 = x >> 31
  or a1, a1, a2         # a1 = a1 | a2 (rol x, 1)
  总: 3 条指令, ~2 周期 (改进 2.5 倍)
```

**预期收益**: **+30-50%** (位运算优化)

---

### 2.2 `crc*.sy` - CRC32 计算

**代码模式**:
```c
int _and(int a, int b) {
    int len = 32, result = 0, power = 1;
    while (len) {
        bit_a = a % 2;
        bit_b = b % 2;
        a = a / 2;
        b = b / 2;
        if (bit_a == 1 && bit_b == 1) {
            result += power;
        }
        power *= 2;
        len--;
    }
    return result;  // 32 次迭代实现一次按位与！
}

int crc32(int crc, int p[], int len) {
    for (i = 0; i < len; i++) {
        index = _xor(_and(crc, 255), p[i]);  // 实现三个位运算函数
        crc = _xor(rotr8(crc), crc32table[index]);
    }
}
```

**性能分析**:
- **最严重的低效**:
  - `_and()` 函数: 32 次迭代（循环）执行一次按位与操作
  - 应该是 1 个硬件指令，现在是 **100+ 条指令**
  - **性能损耗**: 100-200 倍

- **整体性能**:
  - 对于长度为 N 的输入，CRC 计算从 O(N) 变成 O(32N)
  - 测试输入长度: 100000，损耗更严重

**优化方案**:
```cpp
// 优化 1: 通过 ProvenBitwiseHelper 在 IR 证明后使用硬件操作
int _and_fixed(int a, int b) {
    return a & b;  // 1 条指令！
}

int _xor_fixed(int a, int b) {
    return a ^ b;  // 1 条指令！
}

// 优化 2: 查表优化（已有，但被位运算拖累）
int crc32_optimized(int crc, int p[], int len) {
    for (int i = 0; i < len; i++) {
        int index = ((crc ^ p[i]) & 0xFF);  // 直接按位操作
        crc = (crc >> 8) ^ crc32table[index];
    }
    return crc;
}

// 优化 3: 循环展开
// 每 4 次迭代合并
for (i = 0; i < len; i += 4) {
    crc = process_4_bytes(crc, p[i..i+3], table);
}

// 优化 4: 向量化（如果编译器支持）
// 8-16 个字节并行处理
```

**优化效果**:
- 从 100+ 指令恢复到 1 指令: **100 倍改进**
- 整体 CRC 计算: **50-100 倍改进**

**预期收益**: **⭐⭐⭐⭐⭐ 极高 (50-100 倍)**

---

### 2.3 `huffman-*.sy` - Huffman 编码

**代码模式**:
```c
int _and(int a, int b) { /* 同 CRC */ }
int _xor(int a, int b) { /* 同 CRC */ }
int _or(int a, int b) { /* 同 CRC */ }
int rotrN(int x, int n) { /* N 轮位旋转 */ }

// Huffman 编码/解码，大量位操作
```

**性能分析**: 同 CRC，有相同的位运算实现问题

**预期收益**: **50-100 倍改进** (修复位运算)

---

### 2.4 `h-9*.sy` - 素数因数分解

**代码模式**:
```c
int prime_factors(int n) {
    if (n == 0) return -1;
    for (int i = 2; i * i <= n; i++) {
        while (n % i == 0) {        // 频繁的模运算
            // 处理因子
        }
    }
    return result;
}
```

**性能分析**:
- **关键瓶颈**: `n % i` 除法/模运算
  - RISC-V 中除法: 8-20 周期
  - 内循环中多次执行

- **数据流分析**:
  - `n` 在每次 while 迭代中递减
  - 依赖链: `n = n / i`
  - 难以展开或并行

**优化方案**:
```cpp
// 优化 1: 减少模运算次数
int prime_factors_optimized(int n) {
    if (n == 0) return -1;
    for (int i = 2; i * i <= n; ) {
        if (n % i == 0) {
            do {
                n /= i;
                count++;
            } while (n % i == 0);  // 循环展开，只计算一次 %
        }
        i++;  // 或者 i += (i==2) ? 1 : 2  (跳过偶数)
    }
}

// 优化 2: 预计算质数
// 对于固定范围的输入，预计算所有质数
// 然后只试除质数

// 优化 3: 试除优化
// 只尝试 2 和奇数
int factors = 0;
while (n % 2 == 0) { n /= 2; factors++; }  // 特殊处理 2
for (int i = 3; i*i <= n; i += 2) {
    while (n % i == 0) { n /= i; factors++; }
}

// 优化 4: 使用位运算检查
// (n & 1) 替代 n % 2
// (n & (i-1)) 替代 n % i （仅当 i 是 2 的幂）
```

**预期收益**: **+20-35%** (减少除法次数)

---

## 3. 排序算法 (3 个测试用例)

### 3.1 `03_sort*.sy` - 基数排序

**代码模式**:
```c
void radixSort(int bitround, int a[], int l, int r) {
    if (bitround == -1 || l+1 >= r) return;

    // 计算桶的边界
    for (i = l; i < r; i++) {
        cnt[getNumPos(a[i], bitround)]++;
    }

    // 分配元素
    for (i = l; i < r; i++) {
        while (getNumPos(a[i], bitround) != i) {
            // 交换
        }
    }
}
```

**性能分析**:
- **分支预测问题**:
  - 内层 while 循环的条件难以预测
  - 根据数据分布变化
  - 分支失败率: 50-70%

- **缓存友好性**: 相对较好 (顺序读，随机写)

**优化方案**:
```cpp
// 优化 1: 无分支实现
// 使用 RISC-V cmov（条件移动）或 select 指令
int pos = getNumPos(a[i], bitround);
int keep = (pos == target) ? 1 : 0;
// 然后用 keep 进行条件移动（无分支）

// 优化 2: 循环展开
// 处理 4-8 个元素
for (i = l; i < r; i += 4) {
    // 同时处理 4 个元素
}

// 优化 3: 预取
// prefetch(&a[i+16])  预加载将要访问的数据

// 优化 4: 两遍排序
// 一次读通道 + 一次写通道
// 分离读写依赖
```

**预期收益**: **+15-25%** (减少分支失败)

---

## 4. 卷积运算 (3 个测试用例)

### 4.1 `conv2d-*.sy` - 2D 卷积

**已在前面详细分析**

**预期收益**: **+35-50%**

---

## 5. FFT (3 个测试用例)

### 5.1 `fft*.sy` - 快速傅里叶变换

**代码模式**:
```c
int fft(int arr[], int begin_pos, int n, int w) {
    if (n == 1) return 1;

    // 递归分解
    fft(arr, begin_pos, n/2, w*w);
    fft(arr, begin_pos + n/2, n/2, w*w);

    // 合并（蝴蝶操作）
    int wn = 1;
    for (int i = 0; i < n/2; i++) {
        int x = arr[begin_pos + i];
        int y = arr[begin_pos + i + n/2];
        arr[begin_pos + i] = (x + wn*y) % mod;
        arr[begin_pos + i + n/2] = (x - wn*y + mod) % mod;
        wn = (wn * w) % mod;
    }
}
```

**性能分析**:
- **递归调用**: 深度 log(N)
  - 函数调用开销
  - 栈操作

- **蝴蝶操作**:
  - 大量模运算 (expensive)
  - 内存访问模式: 步长变化

- **数据依赖**:
  - `wn = (wn * w) % mod` 产生依赖链

**优化方案**:
```cpp
// 优化 1: 内联递归（模板中间体展开）
// 避免函数调用开销
// 对于小 N (8, 16)，直接展开

// 优化 2: 预计算根
// `wn = wn * w` 中的 w 可以预计算表

// 优化 3: 优化模运算
// 使用 Barrett reduction 加速 `% mod`

// 优化 4: 迭代 FFT （iterative）
// 递归 → 迭代以减少栈操作

// 优化 5: 缓存预热
// 第一次 FFT 会缓存失配
// 结构化安排以最小化缓存冲突

// 优化 6: 向量化蝴蝶操作
// 多个蝴蝶并行处理
```

**预期收益**: **+20-35%** (模运算优化 + 递归优化)

---

## 6. 其他计算密集型

### 6.1 `h-1*.sy` - 递归数论计算

**优化**:
- 尾递归消除
- 缓存 GCD 结果
- 预计算

**预期收益**: **+15-20%**

### 6.2 `h-4*.sy` - 循环优化测试

**优化**:
- 循环交换
- 常量展开
- 循环融合

**预期收益**: **+25-30%**

### 6.3 `h-8*.sy` - Nussinov 算法 (RNA 配对)

**代码特征**: 三重嵌套循环，动态规划

**优化**:
- 分块 DP
- 缓存优化
- 向量化

**预期收益**: **+20-30%**

### 6.4 `h-10*.sy` - 三角矩阵求解 (TRSM)

**代码特征**: 浮点运算

**优化**:
- 循环展开
- 指令调度
- 浮点指令并行度

**预期收益**: **+15-25%**

### 6.5 `shuffle*.sy` - 哈希表操作

**代码特征**: 链表遍历，随机访问

**优化**:
- 减少指针追踪（缓存友好的哈希表结构）
- 批量操作

**预期收益**: **+10-15%**

### 6.6 `sl*.sy` - 多维数组操作

**代码特征**: 三维数组，三重循环

**优化**:
- 循环交换
- 分块

**预期收益**: **+20-30%**

### 6.7 `knapsack_naive*.sy` - 背包问题

**代码特征**: 指数时间递归

**优化**:
- 尾递归消除
- 备忘录（缓存）
- 迭代 DP

**预期收益**: **+30-50%**（通过消除递归开销）

### 6.8 `optimization_scheduling*.sy` - 指令调度测试

**代码特征**: 小循环，数据依赖

**优化**:
- 指令调度
- 循环展开

**预期收益**: **+20-40%**

### 6.9 `prime_search*.sy` - 质数搜索

**代码特征**: 大量试除法

**优化**:
- Sieve 优化
- 减少除法

**预期收益**: **+20-30%**

---

## 总结表

| 类别 | 用例 | 主要瓶颈 | 推荐优化 | 预期收益 |
|-----|------|---------|---------|----------|
| **矩阵** | matmul* | 缓存失配 | 循环交换 + scalar 分块；RVV 仅显式实验 | **+60-80%** |
| | many_mat_cal* | 重复计算 | 缓存感知分块 | +35-50% |
| | transpose* | 最坏缓存 | 分块转置 | **+200-300%** |
| | h-5* | 数据依赖 | 指令调度 | +15-25% |
| **密码学** | crypto-* | 低效位运算 | 位旋转识别 | +30-50% |
| | crc* | 位运算循环 | ProvenBitwiseHelper lowering | **+50-100%** |
| | huffman-* | 位运算循环 | ProvenBitwiseHelper lowering | **+50-100%** |
| | h-9* | 除法频繁 | 试除优化 | +20-35% |
| **排序** | 03_sort* | 分支失败 | 无分支实现 | +15-25% |
| **卷积** | conv2d-* | 通用循环 | 卷积识别 | +35-50% |
| **FFT** | fft* | 递归开销 | 内联 + 模优化 | +20-35% |
| **其他** | h-1* | 递归 | 尾递归消除 | +15-20% |
| | h-4* | 循环 | 循环优化 | +25-30% |
| | h-8* | 三重循环 | scalar 分块；向量化仅显式实验 | +20-30% |
| | h-10* | 浮点 | 调度 + 并行度 | +15-25% |
| | shuffle* | 缓存 | 结构优化 | +10-15% |
| | sl* | 多维数组 | 循环优化 | +20-30% |
| | knapsack* | 递归 | 尾递归消除 | +30-50% |
| | opt_scheduling* | 依赖 | 指令调度 | +20-40% |
| | prime_search* | 除法 | 试除优化 | +20-30% |

---

## 快速优化清单

按优化复杂度排序：

### 🟢 极简单（仅需识别和替换）
- [ ] 实现 `_and()`, `_xor()` 完整 helper 的 IR 证明 lowering，保留 runtime fallback ➜ **+50-100%**
- [ ] 识别位旋转模式并使用 `rol`/`ror` 指令 ➜ **+30-50%**

### 🟡 简单（需要编译器改动）
- [ ] 在 O2 中启用循环交换 ➜ **+15-20%**
- [ ] 在实验配置中显式 opt-in 向量化；默认 O1 保持 scalar ➜ **+25-35%**
- [ ] 优化模运算 ➜ **+8-15%**

### 🔴 复杂（需要新增通道）
- [ ] 实现循环分块通道 ➜ **+20-30%**
- [ ] 实现卷积识别 ➜ **+15-30%**
- [ ] 改进指令调度 ➜ **+10-15%**
