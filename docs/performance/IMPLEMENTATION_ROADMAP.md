# Sisyphus 编译器实施优化计划

> Compliance note (2026-05-31): this document is an analysis and planning aid for the public RISC-V performance suite. Case names are reporting-only. Default O1 remains scalar rv64gc and may only use self-MLIR/affine/memref/machine facts; RVV, StructuralBitwise, StructuralModMul, FunctionEquivalence, Cached precompute, and helper replacement remain explicit opt-in experiments.


## 一、快速优化（无代码改动，仅需调整编译脚本）

### 测试当前最优配置

```bash
#!/bin/bash
# benchmark_current.sh

COMPILER="./build/compiler"
TEST_DIR="test2026/performance_riscv"

echo "=== 测试当前 O2 性能 ==="
time $COMPILER $TEST_DIR/matmul1.sy -S -O2 --target=riscv

echo "=== 启用实验性特性 ==="
# 根据 CLAUDE.md 的记录，这些标志可能已实现
SISY_HIR_ENABLE_INTERCHANGE=1 \
SISY_ENABLE_VECTORIZE=1 \
SISY_ENABLE_STRUCTURAL_BITWISE=1 \
SISY_ENABLE_STRUCTURAL_MODMUL=1 \
time $COMPILER $TEST_DIR/matmul1.sy -S -O2 --target=riscv

echo "=== 对比矩阵乘法性能 ==="
for test in matmul1.sy matmul2.sy matmul3.sy many_mat_cal-1.sy; do
    echo "Test: $test"

    echo "  基础 O2:"
    time $COMPILER $TEST_DIR/$test -S -O2 --target=riscv -o /tmp/basic.s

    echo "  启用所有优化:"
    SISY_HIR_ENABLE_INTERCHANGE=1 \
    SISY_ENABLE_VECTORIZE=1 \
    SISY_ENABLE_STRUCTURAL_BITWISE=1 \
    time $COMPILER $TEST_DIR/$test -S -O2 --target=riscv -o /tmp/optimized.s
done
```

---

## 二、代码级优化（需要源代码改动）

### 2.1 显式 opt-in 向量化（RISC-V RVV）

**现状**: RISC-V 向量化当前在 O2 中被禁用

**改动**: 在 O2 profile 中显式 opt-in 向量化

**改动位置**: `src/main/Options.cpp` 或相关的 pass 管理器

```cpp
// 建议添加到 Options 类中
struct Options {
    // ...已有字段...
    bool enableVectorizeO2 = false;  // 添加这一行
    // ...
};

// 在 parseArgs 中添加
if (strcmp(argv[i], "--enable-vectorize-o2") == 0) {
    opts.enableVectorizeO2 = true;
    continue;
}

// 或者直接在 O2 pipeline 中始终启用
```

**验证方法**:
```bash
./build/compiler test2026/performance_riscv/matmul1.sy -S -O2 --target=riscv --dump-pass-timing
# 查看是否有 VectorizePass 或类似的向量化通道
```

---

### 2.2 启用循环交换（Loop Interchange）

**现状**: 多面体循环优化可能默认禁用

**改动**: 在 O2 中启用循环交换以改善缓存局部性

**改动位置**: `src/hir/` 中的循环优化通道（可能是 `LoopTransform.cpp` 或类似）

```cpp
// 伪代码：在 O2 优化通道中添加
if (opts.o2 && !shouldDisableLoopInterchange()) {
    pm.addPass(createLoopInterchangePass());  // 启用循环交换
}
```

**预期生成代码变化**:

```c
// 优化前: 行主序遍历（缓存不友好）
for (int k = 0; k < n; k++)        // 外层
    for (int i = 0; i < n; i++)    // 中层
        for (int j = 0; j < n; j++)
            C[i][j] += A[i][k] * B[k][j];  // 跳跃式访问 B

// 优化后: 列主序遍历（缓存友好）
for (int i = 0; i < n; i++)
    for (int k = 0; k < n; k++)
        for (int j = 0; j < n; j++)
            C[i][j] += A[i][k] * B[k][j];  // 顺序访问 B
```

---

### 2.3 添加循环分块（Loop Tiling）

**现状**: 无循环分块优化

**改动**: 新增循环分块通道以提高缓存效率

**创建文件**: `src/hir/LoopTiling.cpp`

```cpp
#pragma once
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace hir {

// 循环分块优化
// 将矩阵乘法 C[i][j] += A[i][k] * B[k][j] 转换为分块形式
// 以适应 CPU 缓存层级
//
// 示例（块大小=64）：
// for (ii = 0; ii < n; ii += 64)
//   for (kk = 0; kk < n; kk += 64)
//     for (jj = 0; jj < n; jj += 64)
//       for (i = ii; i < min(ii+64, n); i++)
//         for (k = kk; k < min(kk+64, n); k++)
//           for (j = jj; j < min(jj+64, n); j++)
//             C[i][j] += A[i][k] * B[k][j]
//
// 优势：
// - 三个嵌套的 64×64 块完全适应 L1 缓存
// - 减少 L2 缓存失配
// - 提高内存带宽利用率

std::unique_ptr<Pass> createLoopTilingPass(int tileSize = 64);

}  // namespace hir
```

**实现框架**:
```cpp
// src/hir/LoopTiling.cpp

#include "LoopTiling.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/LoopUtils.h"

namespace hir {

struct LoopTilingPass : public PassWrapper<LoopTilingPass, FunctionPass> {
    StringRef getArgument() const final { return "loop-tiling"; }
    StringRef getDescription() const final {
        return "Tile loops to improve cache locality for matrix operations";
    }

    Option<int> tileSize{*this, "tile-size",
        llvm::cl::desc("Tile size for loop tiling"),
        llvm::cl::init(64)};

    void runOnFunction() override {
        auto func = getFunction();

        // 1. 识别可分块的循环（矩阵乘法模式）
        // 2. 应用分块变换
        // 3. 验证正确性
    }
};

std::unique_ptr<Pass> createLoopTilingPass(int tileSize) {
    auto pass = std::make_unique<LoopTilingPass>();
    pass->tileSize = tileSize;
    return pass;
}

}  // namespace hir
```

---

### 2.4 增强位运算优化

**现状**: 位运算优化有限

**改动**: 扩展位运算模式识别库

**创建文件**: `src/opt/BitwisePatternLibrary.cpp`

```cpp
#pragma once
#include <unordered_map>
#include "mlir/IR/Operation.h"

namespace opt {

// 位运算模式库
// 用于识别和优化常见的位运算序列

struct BitwisePattern {
    std::string name;
    std::string description;
    // 模式匹配逻辑
    std::function<bool(Operation*)> matches;
    // 转换逻辑
    std::function<void(Operation*, PatternRewriter&)> rewrite;
};

class BitwisePatternLibrary {
public:
    static BitwisePatternLibrary& getInstance() {
        static BitwisePatternLibrary instance;
        return instance;
    }

    void registerPattern(const BitwisePattern& pattern);
    std::vector<BitwisePattern> getPatternsFor(Operation* op);

private:
    BitwisePatternLibrary() {
        initializePatterns();
    }

    void initializePatterns();
    std::vector<BitwisePattern> patterns_;
};

// 预定义模式示例
namespace patterns {

// 位旋转识别: (x << k) | (x >> (32-k)) → rotate_left(x, k)
BitwisePattern createRotatePattern();

// 按位与优化: (a & b) & c → a & (b & c)（结合律）
BitwisePattern createBitwiseAssociativityPattern();

// 位提取: (x >> offset) & mask → bexti(x, offset, width)
BitwisePattern createBitExtractPattern();

}  // namespace patterns

}  // namespace opt
```

**具体优化示例** (针对 `crypto-*.sy`):

```c
// 原始: CRC3.sy 中的位运算
int _and(int a, int b) {
    int bit_a, bit_b;
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

// 优化: ProvenBitwiseHelper 在完整 IR 形状证明后使用硬件 AND 指令，
// 无法证明 signed safety 时保留 runtime fallback。
int _and(int a, int b) {
    return a & b;  // 编译器应该识别这是按位与的等价形式
}

// 或者通过模式识别转换为:
int _and_optimized(int a, int b) {
    // 编译器在识别逐位模式后转换
}
```

---

### 2.5 改进指令调度

**现状**: 基础指令调度，未考虑 RISC-V 流水线特性

**改动**: 增强 RISC-V 后端的调度器

**改动位置**: `src/rv/RVScheduler.cpp`（如果存在）或后端相关文件

```cpp
// 改进的调度启发式（伪代码）

struct RiscvMachineModel {
    // RISC-V RV64GC 流水线特性
    static constexpr int LoadLatency = 2;           // 加载延迟 2 周期
    static constexpr int IntMultLatency = 3;        // 乘法延迟 3 周期
    static constexpr int BranchLatency = 1;         // 分支延迟 1 周期
    static constexpr int IssueWidth = 4;            // 每周期最多 4 条指令
};

class EnhancedRVScheduler {
    void scheduleBasicBlock(BasicBlock* BB) {
        // 1. 构建指令依赖 DAG
        // 2. 计算关键路径
        // 3. 按照流水线特性调度指令
        // 4. 最小化加载-使用延迟

        // 例如: 在加载指令后立即放置
        // 不依赖加载结果的独立指令
    }
};
```

---

### 2.6 卷积优化

**现状**: 卷积作为通用循环处理，无特殊优化

**改动**: 识别卷积模式并应用专用优化

**创建文件**: `src/opt/ConvolutionLowering.cpp`

```cpp
#pragma once
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace opt {

// 卷积操作识别和优化通道

class ConvolutionLoweringPass
    : public PassWrapper<ConvolutionLoweringPass, FunctionPass> {
public:
    StringRef getArgument() const final { return "conv-lowering"; }
    StringRef getDescription() const final {
        return "Identify and optimize 2D convolution patterns";
    }

    void runOnFunction() override {
        auto func = getFunction();

        // 1. 识别 2D 卷积循环模式（6 重嵌套循环）
        // 2. 提取卷积核和输入矩阵
        // 3. 应用优化：
        //    - 预计算核权重
        //    - 分块以改善缓存局部性
        //    - 向量化内层循环
        // 4. 生成高效代码
    }
};

std::unique_ptr<Pass> createConvolutionLoweringPass();

// 卷积模式识别辅助函数
bool isConvolutionPattern(Operation* op);
void extractConvolutionParameters(Operation* op,
    int& outputH, int& outputW, int& kernelSize);

}  // namespace opt
```

**卷积优化前后对比**:

```c
// 优化前: 通用 6 重循环
void conv2d(int In[], int Out[], int K[]) {
    for (int repeat = 0; repeat < repeat_factor; repeat++) {
        for (int r = 0; r < N_eff; r++) {
            for (int c = 0; c < N_eff; c++) {
                int sum = 0;
                for (int kr = 0; kr < KSIZE; kr++) {
                    for (int kc = 0; kc < KSIZE; kc++) {
                        int rr = r + kr - pad;
                        int cc = c + kc - pad;
                        if (rr >= 0 && rr < N_eff && cc >= 0 && cc < N_eff) {
                            sum += In[idx(rr,cc,N_eff)] * K[idx(kr,kc,KSIZE)];
                        }
                    }
                }
                Out[idx(r,c,N_eff)] = sum;
            }
        }
    }
}

// 优化后: 预计算核权重、边界处理分离、向量化
void conv2d_optimized(int In[], int Out[], int K[]) {
    // 步骤 1: 预计算核权重（常量传播）
    // K[i] 的值在编译期已知

    // 步骤 2: 分离边界处理（减少内层循环的条件判断）
    // - 处理中心区域：no boundary check
    // - 分别处理四条边

    // 步骤 3: 向量化内层循环
    // for (r=pad; r<N_eff-pad; r+=VF) {
    //     for (c=pad; c<N_eff-pad; c+=VF) {
    //         // 向量化卷积计算
    //     }
    // }
}
```

---

## 三、性能测试框架

### 3.1 构建测试脚本

```bash
#!/bin/bash
# comprehensive_benchmark.sh

COMPILER="./build/compiler"
TEST_BASE="test2026/performance_riscv"
RESULTS_DIR="benchmark_results"

mkdir -p "$RESULTS_DIR"

# 测试用例分类
declare -A test_categories
test_categories[matmul]="matmul1.sy matmul2.sy matmul3.sy"
test_categories[matrix_cal]="many_mat_cal-1.sy many_mat_cal-2.sy many_mat_cal-3.sy"
test_categories[crypto]="crypto-1.sy crypto-2.sy crypto-3.sy"
test_categories[crc]="crc1.sy crc2.sy crc3.sy"
test_categories[sort]="03_sort1.sy 03_sort2.sy 03_sort3.sy"
test_categories[conv]="conv2d-1.sy conv2d-2.sy conv2d-3.sy"
test_categories[fft]="fft0.sy fft1.sy fft2.sy"

benchmark_test() {
    local test_file=$1
    local opt_flag=$2
    local opt_name=$3

    local output_file="$RESULTS_DIR/${test_file%.sy}_${opt_name}.s"

    time $COMPILER "$TEST_BASE/$test_file" \
        -S -O2 --target=riscv \
        $opt_flag \
        -o "$output_file" 2>&1
}

# 测试配置
declare -A config_flags
config_flags[baseline]=""
config_flags[vectorize]="SISY_ENABLE_VECTORIZE=1"
config_flags[interchange]="SISY_HIR_ENABLE_INTERCHANGE=1"
config_flags[bitwise]="SISY_ENABLE_STRUCTURAL_BITWISE=1"
config_flags[modmul]="SISY_ENABLE_STRUCTURAL_MODMUL=1"
config_flags[all]="SISY_ENABLE_VECTORIZE=1 SISY_HIR_ENABLE_INTERCHANGE=1 SISY_ENABLE_STRUCTURAL_BITWISE=1 SISY_ENABLE_STRUCTURAL_MODMUL=1"

# 执行测试
for category in "${!test_categories[@]}"; do
    echo "=== Testing $category ==="

    for test in ${test_categories[$category]}; do
        echo "  Test: $test"

        for config in "${!config_flags[@]}"; do
            echo "    Config: $config"
            env ${config_flags[$config]} benchmark_test "$test" "" "$config"
        done
    done
done

# 生成报告
echo "Benchmark results saved to $RESULTS_DIR/"
```

---

## 四、预期性能收益量化

### 矩阵乘法 (matmul*.sy)

| 优化项 | 预期收益 | 实施难度 |
|-----|--------|--------|
| 显式 opt-in 向量化 | +25-35% | 低 |
| 启用循环交换 | +15-20% | 低 |
| 添加循环分块 | +20-30% | 中 |
| **总计** | **+50-70%** | **中** |

### 密码学 (crypto-*.sy, crc*.sy)

| 优化项 | 预期收益 | 实施难度 |
|-----|--------|--------|
| 位旋转识别 | +10-15% | 中 |
| 位运算序列优化 | +5-10% | 中 |
| 模运算优化 | +8-12% | 中 |
| **总计** | **+20-35%** | **中** |

### 卷积 (conv2d-*.sy)

| 优化项 | 预期收益 | 实施难度 |
|-----|--------|--------|
| 高级卷积识别 | +15-20% | 中 |
| 向量化 | +20-30% | 低 |
| **总计** | **+35-50%** | **中** |

### 整体预期收益

- **保守估计**（只启用合规 self-MLIR/affine/memref/machine 默认优化）: **+25-35%**
- **积极估计**（实现所有新优化）: **+40-60%**

---

## 五、分阶段实施建议

### Phase 1（第1-2天）: 快速收益
```bash
# 1. 验证现有优化标志是否可用
./build/compiler test2026/performance_riscv/matmul1.sy \
    -S -O2 --target=riscv --dump-pass-timing

# 2. A/B 测试 opt-in 标志；不得把高风险语义识别器放入默认 O1
env SISY_ENABLE_VECTORIZE=1 ./build/compiler ...
env SISY_HIR_ENABLE_INTERCHANGE=1 ./build/compiler ...

# 3. 在实验配置中显式启用这些标志
# 编辑 src/main/Options.cpp 或相关文件
```

### Phase 2（第3-5天）: 实现新的优化通道
```
1. 实现 LoopTiling.cpp
2. 增强 BitwisePatternLibrary
3. 改进 RVScheduler
```

### Phase 3（第6-7天）: 卷积和其他特殊优化
```
1. 实现 ConvolutionLowering
2. 实现尾递归消除
3. 实现无分支排序变体
```

---

## 六、验证与测试

### 正确性验证

```bash
# 生成 IR 并验证
./build/compiler test.sy -S -O2 --emit-ir --verify-ir -o test.s

# 对比优化前后的输出
./build/compiler test.sy -S -O0 --emit-ir > baseline.ir
./build/compiler test.sy -S -O2 --emit-ir > optimized.ir

# 用 diff 对比（应该只有指令顺序不同，语义相同）
diff baseline.ir optimized.ir
```

### 性能验证

```bash
# 编译到汇编后，使用 objdump 分析
objdump -d output.o | grep -A 20 "matmul>"

# 查找优化迹象：
# - 向量化: 是否有 v.* 指令（RISC-V 向量扩展）
# - 循环分块: 是否有 outer/inner 循环标签
# - 指令调度: 是否减少了 load-to-use 延迟
```

---

## 七、参考资源

- RISC-V Vector 扩展: https://github.com/riscv/riscv-v-spec/releases/download/v1.0/riscv-v-spec-1.0.pdf
- MLIR Loop Utilities: https://mlir.llvm.org/docs/Dialects/LLVM/
- 矩阵乘法优化手册: https://github.com/flame/blis/blob/master/docs/DesignAndUsage/BLIS_design.pdf
- 高效密码实现: https://bearssl.org/ctmul.html
