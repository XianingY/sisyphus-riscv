# Sisyphus 深度优化实施方案

## 执行摘要

本文档是 `docs/superpowers/specs/2026-05-10-deep-optimization-design.md` 的实施细化方案，将设计文档中的9个优化模块转化为具体的代码实现计划。

**目标**：解决以下性能瓶颈测试：
- fft1 (19x 加速)
- conv2d-1 (11x 加速)
- 03_sort1 (10x 加速)
- many_mat_cal-3 (1.8x 加速)

---

## 1. 当前项目状态分析

### 1.1 已有优化基础设施

```
src/opt/
├── StrengthReduct.cpp      # 乘除法强度削减（已有基础）
├── LICM.cpp                # 循环不变代码外提（需增强）
├── Fusion.cpp              # 循环融合（需增强）
├── SCEV.cpp                # 标量演化分析（需扩展）
├── Range.cpp               # 区间分析（已有）
└── Passes.h                # Pass 注册表

src/rv/
├── StrengthReduct.cpp      # RISC-V 特定强度削减
└── RegPeephole.cpp         # 寄存器窥孔优化（已有 mulw 优化）

src/arm/
└── StrengthReduct.cpp      # ARM 特定强度削减
```

### 1.2 待新增模块

| 模块 | 优先级 | 状态 |
|------|--------|------|
| StrengthReduce (通用) | P0 | ✅ 已增强（通用乘法分解） |
| LoopInterchange | P1 | ✅ 框架完成（检测待完善） |
| PartialUnroll | P1 | ✅ 已完成（运行时边界支持） |
| BoundsCheck | P2 | 🔲 框架完成（CFG级别分析待实现） |
| TailRecElim | P2 | ✅ TCO已存在（受前端IR限制） |
| LightweightSuperopt | P3 | 🔲 待办 |

---

## 2. 模块详细实施

### 2.1 StrengthReduce 增强 (P0)

**文件**: `src/opt/StrengthReduce.cpp` (修改现有)

**当前能力**:
- 乘法：2的幂、特殊形式 (x*7, x*15, x*9)
- 除法：2的幂（使用 multiplicative inverse）
- 取模：2的幂

**缺失能力**:
1. 乘以常量泛化（非2的幂）
2. 除以常量泛化
3. 复合表达式强度削减

**实现计划**:

```cpp
// 新增函数：通用乘法强度削减
// 输入：MulOp *mul, int constant
// 输出：是否成功优化
bool strengthReduceMul(MulOp *op) {
    auto rhs = op->getOperand(1);
    if (!isa<LiOp>(rhs.defining)) return false;

    int c = V(rhs.defining);
    if (c < 0) return false;  // 暂时只处理正数

    // 分解 c = 2^a + 2^b + ... (贪心)
    std::vector<int> bits;
    int remaining = c;
    while (remaining > 0) {
        int lowestBit = __builtin_ctz(remaining);
        bits.push_back(lowestBit);
        remaining &= ~(1 << lowestBit);
    }

    if (bits.size() == 1) {
        // 2^a 形式：直接移位
        return replaceWithShift(op, bits[0]);
    }

    if (bits.size() == 2) {
        // 2^a + 2^b 形式
        return replaceWithShiftAdd(op, bits[0], bits[1]);
    }

    // 复杂情况：递归分解
    // c = 2^a + d, 其中 d < c
    int a = bits[0];
    int d = c - (1 << a);
    return replaceWithShiftAddRecursive(op, a, d);
}
```

**测试用例**:
```
tests/opt/strength_reduce/
├── multiply_pow2.sy       # x * 8 → x << 3
├── multiply_sum.sy        # x * 10 → (x << 3) + (x << 1)
├── multiply_complex.sy    # x * 105 → ...
├── divide_pow2.sy         # x / 8 → x >> 3
└── modulo_pow2.sy         # x % 8 → x & 7
```

---

### 2.2 LoopInterchange (P1)

**文件**: `src/opt/LoopInterchange.cpp` (新建)

**目标**: 解决 transpose 系列测试

**核心算法**:

```cpp
// Step 1: 检测完美嵌套循环
struct PerfectNest {
    ForOp* outer;
    ForOp* inner;
};

// Step 2: 分析依赖方向
struct DependenceDirection {
    Reg loopVar;        // 依赖涉及的循环变量
    int outerDir;       // 外层循环方向 (-1, 0, 1)
    int innerDir;       // 内层循环方向 (-1, 0, 1)
};

// Step 3: 判断可交换性
// 可交换 ⟺ 对所有依赖 D，有 D.innerDir >= 0
bool canInterchange(DependenceAnalysis& dep, PerfectNest& nest) {
    auto deps = dep.getDependences(nest.inner);
    for (auto& d : deps) {
        if (d.innerDir < 0) return false;
    }
    return true;
}

// Step 4: 执行交换
void doInterchange(PerfectNest& nest) {
    // 交换循环变量的下标
    // 交换外层和内层循环体
}
```

**入口点**:
```cpp
// 在 src/opt/Passes.h 注册
Pass* createLoopInterchange();

// 在 PipelineProfiles.cpp 添加 O2 pipeline 调用
```

---

### 2.3 PartialUnroll (P1)

**文件**: `src/opt/PartialUnroll.cpp` (新建)

**目标**: 解决 fft 系列（迭代次数为运行时参数）

**核心算法**:

```cpp
// Step 1: SCEV 分析获取迭代次数上界
SCEVHandle getTripCountUpperBound(ForOp* loop, SCEVAnalysis& scev) {
    auto tripCount = scev.getTripCount(loop);
    // 如果 tripCount 是常量，直接返回
    // 如果 tripCount 是表达式如 n/2，分析 n 的上界
    return scev.getUpperBound(tripCount);
}

// Step 2: 选择展开因子
int chooseUnrollFactor(int64_t upperBound, int loopBodySize) {
    if (upperBound <= 4) return upperBound;  // 完全展开
    if (upperBound <= 16) return 4;
    if (upperBound <= 64) return 2;
    return 1;  // 不展开
}

// Step 3: 部分展开
// for (i = 0; i < n; i++) → for (i = 0; i < n; i+=4)
void partialUnroll(ForOp* loop, int factor) {
    // 复制循环体 factor 次
    // 调整每份的索引偏移
    // 处理余数
}
```

**SCEV 扩展**:
```cpp
// src/opt/SCEV.cpp 新增方法
SCEVHandle upperBound(SCEVHandle expr);

// 使用示例
for (i = 0; i < n/2; i++) {
    // SCEV 分析：n/2 的上界是 n/2（如果 n 已知上界可收紧）
}
```

---

### 2.4 BoundsCheck (P2)

**文件**: `src/opt/BoundsCheck.cpp` (新建)

**目标**: 解决 conv2d-1（消除 stencil 边界检查）

**核心算法**:

```cpp
// 思想：将循环分为安全区域和边界区域
//
// ┌────────────────────────────────────┐
// │  顶部边界区域 (0 ~ pad)             │
// │  - 需要边界检查                     │
// ├────────────────────────────────────┤
// │  主体区域 (pad ~ N-pad)             │
// │  - 无需边界检查！                   │
// ├────────────────────────────────────┤
// │  底部边界区域 (N-pad ~ N)           │
// │  - 需要边界检查                     │
// └────────────────────────────────────┘

struct LoopBounds {
    Value start;   // 循环起始值
    Value end;     // 循环结束值
    Value step;    // 步长
};

struct StencilBounds {
    int pad;           // padding 大小
    int N;             // 数组维度
    LoopBounds i, j;   // 循环边界
};

// 分析 stencil 访问模式
StencilBounds analyzeStencil(ForOp* loop, GetGlobalOp* access) {
    // 提取循环变量的系数和偏移
    // 计算安全区域
}

// 分离循环
std::pair<ForOp*, ForOp*> splitLoopForStencil(ForOp* loop, StencilBounds& bounds) {
    // 创建三个版本：
    // 1. 边界版本（有检查）
    // 2. 主体版本（无检查）
    // 3. 尾部边界版本（有检查）
}
```

---

### 2.5 TailRecElim (P2)

**文件**: `src/opt/TailRecElim.cpp` (新建)

**目标**: 解决 knapsack_naive 简单分支

**核心算法**:

```cpp
// 识别尾递归模式
struct TailCallInfo {
    CallOp* call;           // 尾调用
    std::vector<Value> args; // 传递的参数
    bool isSimple;          // 简单 vs 复合
};

// 简单尾递归：return func(args)
// 复合尾递归：return op(func(args), ...)

bool recognizeTailCall(ReturnOp* ret) {
    if (ret->getOperandCount() == 0) return false;
    auto call = dyn_cast<CallOp>(ret->getOperand(0));
    if (!call) return false;

    // 检查是否是函数的最后一个操作
    auto func = call->getCallee();
    return isLastUse(call) && call->getNumResults() == ret->getNumOperands();
}

// 转换算法
void eliminateTailRecursion(FuncOp* func, TailCallInfo& info) {
    // 1. 创建新参数保存返回值
    // 2. 将函数转换为迭代形式
    // 3. 用 while 循环替换递归调用
}
```

---

## 3. 实现顺序与依赖

```
Phase 1: 基础设施 (第1-2周)
├── StrengthReduce 增强
│   ├── 依赖：无
│   └── 产出：所有测试受益
└── SCEV 扩展
    ├── 依赖：StrengthReduce
    └── 产出：PartialUnroll 所需

Phase 2: 循环优化 (第3-4周)
├── LoopInterchange
│   ├── 依赖：SCEV
│   └── 目标：transpose 系列
├── PartialUnroll
│   ├── 依赖：SCEV
│   └── 目标：fft 系列
└── LICM 增强
    ├── 依赖：无
    └── 目标：optimization_scheduling1

Phase 3: 边界与递归 (第5-6周)
├── BoundsCheck
│   ├── 依赖：LoopInterchange
│   └── 目标：conv2d 系列
└── TailRecElim
    ├── 依赖：无
    └── 目标：knapsack_naive

Phase 4: 高级优化 (第7-8周)
├── LightweightSuperopt
│   ├── 依赖：StrengthReduce
│   └── 目标：代数简化
└── DPTransforms
    ├── 依赖：TailRecElim
    └── 目标：完整 knapsack
```

---

## 4. 验证策略

### 4.1 单元测试

```bash
tests/opt/
├── test_strength_reduce.sh
│   ├── multiply_pow2     # x * 8 = x << 3
│   ├── multiply_sum      # x * 10 = (x << 3) + (x << 1)
│   ├── divide_pow2       # x / 8 = x >> 3
│   └── modulo_pow2       # x % 8 = x & 7
├── test_loop_interchange.sh
│   ├── perfect_nest      # 完美嵌套可交换
│   └── imperfect_nest    # 不可交换
├── test_partial_unroll.sh
│   ├── runtime_bound      # 运行时确定上界
│   └── compile_time       # 编译期常量
└── test_tail_rec.sh
    ├── simple            # 简单尾递归
    └── compound          # 复合尾递归
```

### 4.2 集成测试

```bash
# 回归测试
scripts/regression.sh tests/smoke riscv O2

# 性能测试
scripts/eval-runtime.sh test2026/performance riscv O2

# 重点测试
echo "Testing fft1..."
time ./build/compiler tests/perf/fft1.sy -O2 -o /tmp/fft1.s
spike pk /tmp/fft1.s < tests/perf/fft1.in > /tmp/fft1.out
diff /tmp/fft1.out tests/perf/fft1.out
```

---

## 5. 风险控制

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| StrengthReduce 引入 bug | 中 | 高 | 充分单元测试，覆盖边界情况 |
| LoopInterchange 误判依赖 | 低 | 高 | 保守策略：默认关闭 |
| PartialUnroll 代码膨胀 | 中 | 低 | 限制最大展开因子 |
| BoundsCheck 边界错误 | 低 | 严重 | 验证后默认关闭 |

**默认开关策略**：
```cpp
// --enable-xxx 格式
--enable-strength-reduce     // 默认开启（稳定）
--enable-loop-interchange    // 默认关闭（需验证）
--enable-partial-unroll      // 默认开启（安全）
--enable-bounds-check        // 默认关闭（实验性）
--enable-tail-rec           // 默认开启（安全）
```

---

## 6. 下一步行动

### 立即执行 (本周)

1. **StrengthReduce 增强**
   - 扩展 src/opt/StrengthReduct.cpp 支持通用乘法
   - 目标：覆盖更多 2^a + 2^b 形式

2. **创建测试框架**
   - 创建 tests/opt/ 目录结构
   - 添加基础强度削减测试

### 短期 (2周内)

1. **PartialUnroll 实现**
   - 扩展 SCEV 支持上界分析
   - 实现部分展开 pass

2. **集成测试**
   - 在 riscv O2 pipeline 中启用 PartialUnroll
   - 验证 fft1 性能提升

---

## 7. 文档索引

- 设计文档：`docs/superpowers/specs/2026-05-10-deep-optimization-design.md`
- 本文档：`docs/superpowers/specs/implementation-plan.md`
- 状态跟踪：`docs/superpowers/status.md` (待创建)
