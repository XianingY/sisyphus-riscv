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

### 1.3 优化手段总览

| 序号 | 模块 | 优化类型 | 目标测试 |
|------|------|---------|---------|
| 1 | StrengthReduce | 强度削减 | 所有测试 |
| 2 | LoopInterchange | 循环交换 | transpose0/1/2 |
| 3 | Fusion 增强 | 循环融合 | many_mat_cal, 03_sort1 |
| 4 | BoundsCheck | 边界优化 | conv2d-1/2/3 |
| 5 | LICM 增强 | 循环不变外提 | optimization_scheduling1 |
| 6 | PartialUnroll | 部分展开 | fft0/1/2 |
| 7 | TailRecElim | 尾递归消除 | knapsack_naive |
| 8 | DPTransforms | DP 模式识别 | knapsack_naive |

---

## 2. 模块详细设计

### 2.1 StrengthReduce（强度削减优化）

#### 2.1.1 目标
将乘除法转换为更快的移位/加法操作，所有测试受益。

#### 2.1.2 优化场景

**场景 1：乘以 2 的幂**
```c
// 优化前
idx = i * 1024;

// 优化后
idx = i << 10;  // 1024 = 2^10
```

**场景 2：乘以常量**
```c
// 优化前
sum = sum + v * 5;

// 优化后
sum = sum + (v << 2) + v;  // 5 = 4 + 1
```

**场景 3：除以常量（2 的幂）**
```c
// 优化前
i = j / 1024;

// 优化后
i = j >> 10;
```

**场景 4：模 2 的幂**
```c
// 优化前
i = j % 1024;

// 优化后
i = j & 1023;  // 1024 - 1
```

**场景 5：索引计算优化**
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

#### 2.1.3 实现位置
```
src/opt/StrengthReduce.cpp  (新增)
```

#### 2.1.4 依赖关系
- 依赖 SCEV 分析（已有 `src/opt/SCEV.cpp`）
- 被其他优化 pass 调用

#### 2.1.5 正确性保证
- 只对编译期常量进行变换
- 乘法强度削减使用移位 + 加法组合，需要验证语义等价
- 除法/模 2 的幂只在**确定无符号溢出风险**时优化

---

### 2.2 LoopInterchange（循环交换）

#### 2.2.1 目标
交换嵌套循环的内外层顺序以改善访存局部性，解决 transpose 系列测试。

#### 2.2.2 目标场景

**转置场景**
```c
// 优化前：按列访问（cache 不友好）
for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
        B[j][i] = A[i][j];

// 优化后：按行访问（cache 友好）
for (j = 0; j < n; j++)
    for (i = 0; i < n; i++)
        B[j][i] = A[i][j];
```

#### 2.2.3 可交换条件

```cpp
// 条件 1：完美嵌套
//   外层循环只包含内层循环，不包含其他语句

// 条件 2：无循环携带依赖
//   所有依赖的方向必须是 [>=0, >=0]
//   即：内层循环的依赖不能指向"外层更早的迭代"

struct DependenceDirection {
    int outer;  // 外层循环依赖方向
    int inner;  // 内层循环依赖方向
};

// 可交换 ⟺ 对所有依赖 D，有 D.inner >= 0
```

**示例：不可交换的情况**
```c
// A[i][j] 依赖 A[i-1][j]
// 依赖方向：[-1, 0]
// D.outer = -1 < 0 → 不可交换
for (i = 1; i < n; i++)
    for (j = 0; j < n; j++)
        A[i][j] = A[i-1][j];
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

#### 2.2.4 实现位置
```
src/opt/LoopInterchange.cpp  (新增)
src/opt/LoopInterchange.h   (新增)
```

#### 2.2.5 实现步骤

```cpp
// 1. 检测完美嵌套
bool isPerfectNest(ForOp* outer) {
    // outer 的 body 必须只有一个 Block
    // 该 Block 的最后一个 Op 是 inner ForOp
    // inner ForOp 之前没有其他 Op
}

// 2. 提取访问模式
struct ArrayAccess {
    Op* array;
    std::vector<Expr*> subscripts;
    bool isWrite;
};

// 3. 依赖方向分析
//    对每个语句对 (S1, S2)，计算依赖距离向量
//    只处理 A[i+dx][j+dy] 形式，其中 dx, dy 是常量

// 4. 执行交换
void interchange(ForOp* outer, ForOp* inner) {
    // 交换 ForOp 的 induction variable
    // 重写内层循环体的访问
    // 调整循环边界
}
```

#### 2.2.6 限制条件
```
只处理以下情况：
  ✓ 完美嵌套循环
  ✓ 单个数组访问（或可分析的多重访问）
  ✓ 依赖是 A[i±c1][j±c2] = A[i±d1][j±d2] 形式
  ✓ c1, c2, d1, d2 都是整型常量

不处理：
  ✗ 非完美嵌套
  ✗ 间接访问 A[B[i]]
  ✗ 运行时依赖分析
```

---

### 2.3 Fusion 增强（下标等价性 + 跨循环融合）

#### 2.3.1 目标
增强现有 Fusion pass，解决 two_D_offset 和下标等价性问题。

#### 2.3.2 当前问题
Fusion.cpp:137 有 TODO 注释：
```cpp
// A very strict way of checking dependency:
// 1) all subscripts must be the same (except for those that are only read);
// 2) non-array variables must be disjoint.
// Polyhedral approach here? Might extend when I have time.
```

#### 2.3.3 增强内容

**增强 1：下标等价性**
```c
// 当前：只接受完全相同的下标
for (i) A[i] = ...;    ✓ 可融合
for (i) ... = A[i];    ✓

// 增强后：接受数学等价的下标
for (i) A[i] = ...;    ✓ 可融合
for (i) ... = A[i+0];  ✓ (i+0 = i)
```

**增强 2：跨循环依赖松弛**
```c
// 多个相邻独立循环可融合
for (i) A[i] = -1;    →  for (i) {
for (i) B[i] = -1;    →      A[i] = -1;
for (i) C[i] = A[i];  →      B[i] = -1;
                           }
```

#### 2.3.4 实现位置
```
src/opt/Fusion.cpp  (修改)
src/opt/IndexEquiv.cpp   (新增)
src/opt/IndexEquiv.h     (新增)
```

#### 2.3.5 IndexEquiv 核心算法

```cpp
// 下标表达式归一化
struct AffineExpr {
    int64_t constant;           // 常数偏移
    std::vector<std::pair<std::string, int64_t>> coeffs;  // 变量系数
};

// 归一化规则
AffineExpr normalize(Expr* e) {
    // i     → {constant=0, coeffs=[{i,1}]}
    // i+1   → {constant=1, coeffs=[{i,1}]}
    // i+j   → {constant=0, coeffs=[{i,1},{j,1}]}
    // i+0   → {constant=0, coeffs=[{i,1}]}  ← 与 i 等价
    // 0+i   → {constant=0, coeffs=[{i,1}]}  ← 与 i 等价
}

// 等价判断
bool isEquivalent(AffineExpr a, AffineExpr b) {
    // 去掉零系数
    // 比较常数和所有系数
}
```

#### 2.3.6 限制条件
```
只处理：
  ✓ A[i] 和 A[i+0]（加零）
  ✓ A[i][j] 和 A[i][j+0]
  ✓ A[i] 和 A[0+i]

不处理：
  ✗ A[i+j] 除非 j 是循环不变量
  ✗ A[i+1] 和 A[i]（不是等价，是平移）
```

---

### 2.4 BoundsCheck（边界检查消除）

#### 2.4.1 目标
消除或外提 stencil 类循环中的冗余边界检查，解决 conv2d 测试。

#### 2.4.2 问题场景
```c
// conv2d 边界检查（每次 stencil 访问都要执行）
if (rr >= 0 && rr < N_eff && cc >= 0 && cc < N_eff) {
    sum = sum + In[idx(rr,cc)] * K[idx(kr,kc)];
}
```

#### 2.4.3 优化思路

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

#### 2.4.4 实现位置
```
src/opt/BoundsCheck.cpp  (新增)
src/opt/BoundsCheck.h   (新增)
```

#### 2.4.5 实现步骤

```cpp
// 1. 识别 stencil 边界条件
//    if (rr >= 0 && rr < N && cc >= 0 && cc < M)
//        access A[rr][cc]

// 2. 分析循环边界
//    for (r = 0; r < N; r++)
//        rr = r + kr - pad;
//        if (rr >= 0 && rr < N) ...

// 3. 计算"安全区域"
struct SafeRegion {
    int64_t lower;   // 安全下界
    int64_t upper;   // 安全上界
};

// 对于 r in [pad, N-pad)：
//   rr = r + kr - pad，当 kr in [0, KSIZE) 时
//   rr_min = pad + 0 - pad = 0
//   rr_max = (N-pad) + (KSIZE-1) - pad = N + KSIZE - 1 - pad
//   安全条件：0 <= rr < N

// 4. 分割循环
void splitLoop(ForOp* loop, SafeRegion safe) {
    // for (r = 0; r < N; r++) →
    //   for (r = 0; r < pad; r++) { /* 有检查 */ }
    //   for (r = pad; r < N-pad; r++) { /* 无检查 */ }
    //   for (r = N-pad; r < N; r++) { /* 有检查 */ }
}
```

#### 2.4.6 限制条件
```
只处理：
  ✓ 边界是常量（如 pad = KSIZE/2）
  ✓ 循环边界是仿射形式
  ✓ 访问是 A[i+const] 或 A[i+j+const]

不处理：
  ✗ 边界是运行时变量
  ✗ 非完美嵌套循环
  ✗ 复杂条件分支
```

---

### 2.5 LICM 增强（链式不变计算外提）

#### 2.5.1 目标
识别并外提链式不变计算，解决 optimization_scheduling1。

#### 2.5.2 问题场景
```c
// optimization_scheduling1: dependent_computation
while (i < iterations) {
    a = a + b;    // b 看起来在变化...
    b = b + c;    // c 是循环不变的！
    c = c + d;    // d 是循环不变的！
    d = d + a;
    i++;
}
```

关键观察：`c` 和 `d` 的**增量**（`d` 和 `a`）在循环中不变。

#### 2.5.3 实现位置
```
src/opt/LICM.cpp  (修改，增强链式分析)
```

#### 2.5.4 增强算法

```cpp
// 1. 构建赋值链
//    a = a + b
//    b = b + c
//    c = c + d
//
//    链：a -= b, b -= c, c -= d, d -= a

// 2. 识别增量
//    对于 x = x + inc：
//    - inc 是 x 的增量
//    - inc 是循环不变 ⟺ inc 定义点的所有操作数都是循环不变

// 3. 计算增量的定义链
//    inc = d  // d 在循环中变化
//    d = d + a  // d 的增量是 a
//    a = a + b  // a 的增量是 b
//    链式传播：d 的增量 = a, a 的增量 = b, b 的增量 = c, c 的增量 = d
//    如果 c, d 是初始常量，则链式增量的增量...最终收敛到常量

// 4. 外提后变换
//    // 循环前
//    int inc_c = c;  // c 的初始值
//    int inc_d = d;  // d 的初始值
//    int inc_a = a;  // a 的增量
//
//    while (i < iterations) {
//        a = a + b;
//        b = b + inc_c;
//        c = c + inc_d;
//        d = d + inc_a;
//        i++;
//    }
```

#### 2.5.5 正确性条件
```
✓ 增量是循环不变的（通过 SSA 分析确定）
✓ 链式传播最终收敛到常量
✓ 外提后不影响依赖关系

不处理：
✗ 链式形成环（a = a + b, b = b + a）→ 无法优化
```

---

### 2.6 PartialUnroll（部分循环展开）

#### 2.6.1 目标
对迭代次数有上界但非常量的循环做部分展开，解决 fft 系列。

#### 2.6.2 问题场景
```c
// fft 内层循环
while (i < n/2) {  // n 是运行时参数
    x = arr[begin_pos + i];
    y = arr[begin_pos + i + n/2];
    arr[begin_pos + i] = (x + wn*y) % mod;
    arr[begin_pos + i + n/2] = (x - wn*y + mod) % mod;
    wn = multiply(wn, w);
    i++;
}
```

当前 `ConstLoopUnroll` 只对**编译期常量迭代次数**的循环做展开，但 `n/2` 是运行时值。

#### 2.6.3 实现位置
```
src/opt/PartialUnroll.cpp  (新增)
src/opt/PartialUnroll.h   (新增)
```

#### 2.6.4 实现思路

```cpp
// 1. 估计迭代次数上界
LoopTripBound analyzeTripBound(ForOp* loop) {
    // 使用 SCEV 分析
    // 尝试计算 trip count 的上界

    // 如果循环边界是 0 <= i < n/2
    // trip count = n/2
    // 如果 n 有已知范围（如 fft 中 n 是 2 的幂，最大 2^21）
    // 则上界 = 2^21 / 2 = 2^20 ≈ 1M
}

// 2. 选择展开因子
int chooseUnrollFactor(int64_t estimatedTripCount, int loopBodySize) {
    if (estimatedTripCount <= 8) return estimatedTripCount;  // 完全展开
    if (estimatedTripCount <= 64) return 4;
    if (estimatedTripCount <= 256) return 2;
    return 1;  // 不展开
}

// 3. 部分展开
//    for (i = 0; i < n; i+=4)
//        stmt;
//        stmt;
//        stmt;
//        stmt;
```

#### 2.6.5 限制条件
```
只处理：
  ✓ trip count 上界 ≤ 4096
  ✓ 循环体较小（< 50 ops）
  ✓ 无复杂控制流

不处理：
  ✗ trip count 上界 > 4096
  ✗ 循环体过大（展开后代码膨胀严重）
```

---

### 2.7 TailRecElim（尾递归消除）

#### 2.7.1 目标
将简单尾递归转换为迭代，解决 knapsack_naive 中的简单分支。

#### 2.7.2 问题场景
```c
// knapsack_naive 的简单分支
if (weight[i-1] > w)
    return knapsack_naive(i-1, w);  // 直接尾递归
```

#### 2.7.3 实现位置
```
src/opt/TailRecElim.cpp  (新增)
src/opt/TailRecElim.h    (新增)
```

#### 2.7.4 识别条件
```cpp
// 尾递归模式：
// 1. return func(args)
// 2. return op(func(args), ...)  （如 return func(args) + 1）
// 3. return op1(op2(func(args), ...), ...) （如 return func(a) + func(b)）← 复杂

// 简单尾递归：只满足条件 1
// 复合尾递归：满足条件 2
// 双重尾递归：两个递归调用 ← 不处理
```

#### 2.7.5 转换算法

```cpp
// 简单尾递归转换
// 优化前：
int foo(int n) {
    if (n == 0) return 0;
    return foo(n - 1);  // 尾递归
}

// 优化后：
int foo_iter(int n) {
    while (n != 0) {
        n = n - 1;  // 替代递归调用
    }
    return 0;
}
```

#### 2.7.6 限制条件
```
只处理：
  ✓ 单个尾递归调用
  ✓ return 直接返回递归结果（无额外操作）
  ✓ 递归参数是线性变换

不处理：
  ✗ 双重递归（如 knapsack）
  ✗ 尾递归后还有操作（如 return foo(n-1) + 1）
```

---

### 2.8 DPTransforms（DP 模式识别与转换）

#### 2.8.1 目标
识别已知 DP 模式并转换为 tabular DP，解决 knapsack_naive。

#### 2.8.2 递归结构分析

knapsack_naive 的递归调用图：
```
knapsack(i, w)
├── knapsack(i-1, w)              [without_item]
└── knapsack(i-1, w-weight[i-1]) [with_item]
    └── max(without_item, value[i-1] + with_item)
```

这是 **DAG（有向无环图）**，可以通过填表实现。

#### 2.8.3 实现位置
```
src/opt/DPTransforms.cpp  (新增)
src/opt/DPTransforms.h    (新增)
```

#### 2.8.4 识别算法

```cpp
struct DPPattern {
    const char* name;
    int numRecursiveCalls;   // 递归调用数量
    CombineOp combine;       // 组合操作（max, min, sum）
};

// 识别 0/1 背包
bool recognizeKnapsack(FuncOp* func) {
    // 1. 检查函数签名：int func(int i, int w)
    // 2. 检查递归调用数量：2 个
    // 3. 检查组合操作：max
    // 4. 检查参数模式：
    //    - 调用1: (i-1, w)
    //    - 调用2: (i-1, w-weight[i-1])
    // 5. 如果都匹配 → 认为是 knapsack
}
```

#### 2.8.5 转换算法

```cpp
// knapsack 转换后的代码结构：
int knapsack_dp(int init_i, int init_w, int weight[], int value[], int N, int W) {
    // 1. 创建 dp 表
    int dp[N+1][W+1];

    // 2. 填表
    for (int i = 0; i <= N; i++) {
        for (int w = 0; w <= W; w++) {
            if (i == 0 || w == 0) {
                dp[i][w] = 0;
            } else if (weight[i-1] > w) {
                dp[i][w] = dp[i-1][w];
            } else {
                int without = dp[i-1][w];
                int with = value[i-1] + dp[i-1][w - weight[i-1]];
                dp[i][w] = max(without, with);
            }
        }
    }

    // 3. 返回初始参数对应的结果
    return dp[init_i][init_w];
}
```

#### 2.8.6 支持的 DP 模式

| 模式 | 递归类型 | 组合操作 | 转换方法 |
|------|---------|---------|---------|
| 0/1 背包 | 双重递归 | max | tabular DP |
| 斐波那契 | 单递归 | sum | 展开 + 滑动变量 |
| LCS | 双重递归 | max | tabular DP |

#### 2.8.7 限制条件
```
只处理：
  ✓ 递归调用数 ≤ 2
  ✓ 组合操作是 max/min/sum
  ✓ 参数传递是线性变换

不处理：
  ✗ 三重及以上递归
  ✗ 非 DAG 递归
  ✗ 复杂组合操作
```

---

## 3. 实现顺序

### 阶段 1：基础优化（所有测试受益）

```
顺序 1: StrengthReduce
  - 位置：src/opt/StrengthReduce.cpp
  - 依赖：无
  - 影响：所有测试
  - 原因：这是最基础、最通用的优化，不会破坏正确性

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
  - 原因：在现有代码基础上增强，风险可控

顺序 4: LICM 增强
  - 位置：修改 src/opt/LICM.cpp
  - 依赖：StrengthReduce, SCEV
  - 影响：optimization_scheduling1
  - 原因：增强现有 pass，需要理解现有逻辑

顺序 5: PartialUnroll
  - 位置：src/opt/PartialUnroll.cpp
  - 依赖：StrengthReduce, SCEV
  - 影响：fft 系列
  - 原因：需要 SCEV 分析支持
```

### 阶段 3：边界优化

```
顺序 6: BoundsCheck
  - 位置：src/opt/BoundsCheck.cpp
  - 依赖：LoopInterchange（需要完美嵌套检测）
  - 影响：conv2d 系列
  - 原因：需要对循环结构有较好理解
```

### 阶段 4：递归优化

```
顺序 7: TailRecElim
  - 位置：src/opt/TailRecElim.cpp
  - 依赖：StrengthReduce
  - 影响：knapsack_naive（简单分支）
  - 原因：先做简单版本

顺序 8: DPTransforms
  - 位置：src/opt/DPTransforms.cpp
  - 依赖：TailRecElim
  - 影响：knapsack_naive（完整版本）
  - 原因：在简单尾递归基础上扩展
```

---

## 4. 预期效果

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

## 5. 风险与缓解

| 模块 | 风险 | 影响 | 缓解措施 |
|------|------|------|---------|
| StrengthReduce | 整数溢出 | 中 | 只对无符号/已知范围操作数优化 |
| LoopInterchange | 误判依赖方向 | 高 | 保守：只在能证明安全时交换 |
| BoundsCheck | 边界分析错误 | 严重 | 保守：只在确定安全时消除检查 |
| LICM 增强 | 链式传播错误 | 高 | 使用开关，默认关闭 |
| PartialUnroll | 展开因子不当 | 中 | 可配置：默认 2/4，保守 2 |
| TailRecElim | 尾递归误判 | 中 | 只处理明确的尾递归模式 |
| DPTransforms | DP 模式误识别 | 中 | 先验证，后转换；有测试框架 |

---

## 6. 测试计划

### 6.1 单元测试

```bash
# 每个模块应有独立的测试文件
tests/opt/
├── strength_reduce/
│   ├── multiply_pow2.sy       # 乘以 2 的幂
│   ├── multiply_const.sy      # 乘以常量
│   ├── divide_pow2.sy         # 除以 2 的幂
│   └── index_calc.sy          # 索引计算优化
├── interchange/
│   ├── perfect_nest.sy        # 完美嵌套
│   ├── with_dep.sy            # 有依赖（应跳过）
│   └── transpose.sy            # 转置场景
├── fusion_equiv/
│   ├── simple_offset.sy        # A[i] 和 A[i+0]
│   └── cross_loop.sy          # 跨循环融合
├── bounds_check/
│   ├── stencil.sy             # stencil 边界检查
│   └── simple_bounds.sy       # 简单边界
├── licm_chain/
│   └── chain_invariant.sy     # 链式不变计算
├── partial_unroll/
│   └── bounded_loop.sy         # 有界循环
├── tail_rec/
│   └── simple_tail.sy         # 简单尾递归
└── dp_transforms/
    ├── knapsack.sy            # 0/1 背包
    └── fibonacci.sy           # 斐波那契
```

### 6.2 集成测试

```bash
# 使用现有回归测试
scripts/regression.sh tests/opt riscv O2
scripts/regression.sh tests/opt arm O2

# 性能测试
scripts/eval-runtime.sh test2026/performance riscv O2
```

### 6.3 正确性保证

```
1. 每个模块独立测试
2. 与现有测试套件完全兼容（无回归）
3. 新增测试覆盖边界情况
4. 性能测试验证加速效果
```

---

## 7. 开关控制

```cpp
// 新增命令行选项
--enable-strength-reduce     // 强度削减（默认开启）
--enable-loop-interchange    // 循环交换（默认开启）
--enable-fusion-equiv        // 下标等价融合（默认关闭，保守）
--enable-bounds-check        // 边界检查优化（默认关闭，保守）
--enable-licm-chain         // 链式 LICM（默认关闭，保守）
--enable-partial-unroll      // 部分展开（默认开启）
--enable-tail-rec           // 尾递归消除（默认开启）
--enable-dp-transforms       // DP 模式转换（默认关闭，保守）
```

---

## 8. 总结

本方案通过 8 个优化模块，覆盖了测试集中所有大性能差距场景：

```
┌─────────────────────────────────────────────────────────┐
│                    优化模块覆盖                           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  强度削减        → 所有测试                              │
│  循环交换        → transpose                            │
│  循环融合增强    → many_mat_cal, 03_sort1              │
│  边界检查消除    → conv2d                               │
│  LICM 增强       → optimization_scheduling              │
│  部分展开        → fft                                  │
│  尾递归消除      → knapsack_naive (简单)               │
│  DP 模式转换     → knapsack_naive (完整)                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

实施顺序遵循从低风险到高风险、从通用到专用的原则，确保每一步都可验证、可回滚。
