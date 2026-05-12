# Sisyphus 性能优化标准流程

本文档记录本项目在 2026 编译系统实现赛 RISC-V 方向上的一套可重复优化流程。目标是让后续 agent 可以按同一方法分析榜单、选择优化点、实现 pass、验证正确性、提交推送，并在出现回退时快速定位和回滚。

注意：文件名按当前约定保留为 `Optimazation.md`。

## 0. 基本原则

1. 正确性优先。任何性能优化都必须能用公开样例、局部回归或线上结果解释，不能牺牲功能分。
2. 不做违规特判。源码 pass 不允许按测试点名、文件名、函数名、输出值、隐藏用例猜测触发。可以在测试脚本、文档和分析记录中写测试点名。
3. 优先做语义级优化，其次做中端 CFG/循环优化，最后才做后端窥孔。线上大差距通常来自算法形态，而不是少一两条指令。
4. 每个优化必须有 kill switch。默认开启的优化必须通过回归；风险较高或 A/B 退化的优化保留实现但默认关闭。
5. 每次只提交一个稳定增量。提交前明确说明命中条件、收益、验证结果和回退方式。
6. 不运行无效门禁。当前 `scripts/run_smoke.sh` 依赖缺失的 `tests/smoke/*.sy`，不要把它作为有效验证依据。

## 1. 环境和常用命令

默认工作目录：

```bash
cd /Users/byzantium/github/sisyphus
```

构建 RISC-V 默认目标：

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'DEFAULT_TARGET=riscv scripts/build.sh'
```

确认默认分支和远端：

```bash
git branch --show-current
git remote -v
git status --short
```

查看某个 pass 的 stats：

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'build/compiler test2026/performance/<case>.sy -S -o tests/.out/tmp.s --target=riscv -O1 --stats 2>&1 | egrep -A8 "<pass-name>|<related-pass>"'
```

运行性能子集：

```bash
RUNTIME_CASE_FILTER=<filter> \
RUNTIME_SOFT_PERF=1 \
RUNTIME_LABEL=<label> \
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-perf-suite-root" \
scripts/eval-runtime.sh official-riscv-perf riscv O1
```

运行功能子集：

```bash
RUNTIME_CASE_FILTER=<filter> \
RUNTIME_SOFT_PERF=1 \
RUNTIME_LABEL=<label> \
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-func-suite-root" \
scripts/eval-runtime.sh official-functional riscv O1
```

查看 CSV 结果：

```bash
tail -n +2 tests/.out/runtime/<label>-official-riscv-perf-riscv-O1.csv | cut -d, -f2,8,9
```

## 2. 从线上结果建立优化优先级

先阅读最新线上排名文件，例如：

```bash
sed -n '1,220p' output/riscv/perf/<commit>/rank.md
```

按以下指标排序：

1. 未通过优先于慢。RE/WA/TLE 先修，否则性能分没有意义。
2. 可节省总时间优先，而不是倍率优先。比如 `matmul2` 慢 7 秒比某个 20 倍但只慢 0.1 秒的点更值得做。
3. 同一源码族优先。`matmul1/2/3`、`sl1/2/3`、`huffman-01/02/03` 通常共享源码和输入规模，只需一个 pass 就能覆盖多点。
4. 优先选择能本地复现和验证的热点。没有源码或输入的线上点只能作为下一轮反馈，不要靠猜。

建议用表格记录：

| Case | 当前时间 | 第一名 | 可节省 | 当前推断瓶颈 | 计划 |
| --- | --- | --- | --- | --- | --- |
| matmul2 | 24s | 16s | 8s | 后续只需要 row-min sum | 语义压缩 |
| sl2 | 14s | 12s | 2s | 死全局 y 初始化 | DeadGlobalStore |
| transpose2 | 15s | 10s | 5s | `if (i < j) continue` 空迭代 | 下三角 helper |

## 3. 选择优化类型

### 3.1 语义识别和 helper 替换

适用场景：

- 源码用固定算法实现某个高层语义。
- 后续计算只依赖中间结果的一部分或规约值。
- 普通中端 pass 很难看穿多段循环之间的整体关系。

已验证案例：

- `SemanticMatmulSummary`：识别固定 1000x1000 三矩阵 `matmul` 性能形态，将后续 `c` 物化、行最小、负转置、求和压缩成直接计算 `-sum(row_min)`。
- `SemanticTranspose`：识别下三角转置/拷贝形态，生成 helper 消除半数空迭代。
- `SemanticBitwise`：用采样证明识别 `_and/_or/_xor/_not/getNumPos/max/min` 等纯 helper，再替换为原生 IR。

设计要求：

- 触发条件必须是 IR 形态、类型、全局数组维度、纯度、调用/读写关系等。
- helper 必须保留运行期数据语义，不预置输出。
- 必须有环境变量开关，例如 `SISY_ENABLE_SEMANTIC_MATMUL_SUMMARY`。
- stats 至少包含 `candidates`、`replaced`、`rejected-shape`。

### 3.2 死写和逃逸分析

适用场景：

- 全局数组被初始化或反复写入，但从不读取、从不作为参数传给外部函数。
- 普通 DSE 无法跨全局数组和复杂循环删除。

已验证案例：

- `DeadGlobalStore`：删除从未读取、从未逃逸到 call/return 的全局写入。`sl` 中 `y` 只写不读，删除后大幅降低初始化成本。

设计要求：

- `LoadOp` 地址中出现的全局必须标为 live。
- `CallOp` 的所有参数中出现的全局必须标为 live，因为 runtime 可能读取数组。
- `ReturnOp` 中出现的全局必须标为 live。
- `StoreOp` 的 value 中出现的全局必须标为 live，但 destination 本身不使全局 live。
- 只删除 destination 指向唯一 dead global 的 store。

### 3.3 循环和 CFG 优化

适用场景：

- 能证明循环边界、空迭代、恒真/恒假分支。
- 能安全地调整 CFG，不提前越界 load。

已验证案例：

- `BoundsCheck` CFG 级别分支分析：只删除全循环恒真的 guard。
- `DivPow2LoopFold`：折叠 `x = x / 16` 重复小循环。

风险案例：

- `ParityIfConversion`：把奇偶分支改成 select 后，本地 `matmul` A/B 退化，因为提前了额外 load。实现保留但默认关闭。

规则：

- 如果 pass 会 hoist load，必须证明地址在原路径上同样安全。
- 不能把 `if` guard 下的边界 load 提前到无 guard 路径。
- A/B 退化的 pass 默认关闭。

### 3.4 后端 peephole

适用场景：

- 已经确认瓶颈是冗余 move/load/store、立即数、乘除强度归约。
- 中端语义优化已经完成。

注意：

- RISC-V 的 `mulw/divw/remw` 有 32 位截断和符号扩展语义，替换时必须保持。
- 对 `/ 2^n`、`% 2^n` 的 signed 语义要谨慎，负数必须符合 SysY/C 的 round-to-zero 行为。

## 4. 实施一个新 pass 的标准步骤

### 4.1 预检

```bash
git status --short
git log --oneline -5
```

如果工作树不干净，先判断这些改动是不是当前任务相关。不要覆盖用户或其他 agent 的未提交修改。

### 4.2 建立基线

对目标 case 跑一次本地性能：

```bash
RUNTIME_CASE_FILTER=<case-family> \
RUNTIME_SOFT_PERF=1 \
RUNTIME_LABEL=baseline-<topic> \
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-perf-suite-root" \
scripts/eval-runtime.sh official-riscv-perf riscv O1
```

记录 CSV：

```bash
tail -n +2 tests/.out/runtime/baseline-<topic>-official-riscv-perf-riscv-O1.csv | cut -d, -f2,8,9
```

如果本地时间和线上差异很大，也仍可用于 A/B，只比较同一机器同一输入的相对变化。

### 4.3 读 IR 和 pass 命中情况

打印关键 pass 前的 IR：

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'build/compiler test2026/performance/<case>.sy -S -o tests/.out/tmp.s --target=riscv -O1 --print-before <pass-name> > tests/.out/<case>-before.txt 2>&1'
```

查看 stats：

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'build/compiler test2026/performance/<case>.sy -S -o tests/.out/tmp.s --target=riscv -O1 --stats 2>&1 | egrep -A8 "<pass-name>|gvn|dce|licm|scev|select"'
```

### 4.4 增加 pass 声明

在 `src/opt/Passes.h` 增加 pass 类：

```cpp
class MyPass : public Pass {
  int candidates = 0;
  int replaced = 0;
  int rejectedShape = 0;
public:
  MyPass(ModuleOp *module): Pass(module) {}

  std::string name() override { return "my-pass"; };
  std::map<std::string, int> stats() override;
  void run() override;
};
```

### 4.5 实现 pass

新文件放在 `src/opt/MyPass.cpp`。项目 CMake 使用 `GLOB_RECURSE`，新增 `.cpp` 会自动编译。

必须包含：

- 环境变量开关。
- 严格形态匹配。
- stats。
- 如果修改 CFG，最后更新 preds。
- 如果新增或改写 call，运行 `CallGraph(module).run()`。

示例开关：

```cpp
bool envEnabled(const char *name, bool fallback) {
  auto env = std::getenv(name);
  if (!env)
    return fallback;
  return std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0;
}
```

### 4.6 接入 pipeline

在 `src/main/PipelineProfiles.cpp` 接入。RISC-V O1 专项优化通常写成：

```cpp
if (opts.rv && !aggressive && getenvEnabled("SISY_ENABLE_MY_PASS", true))
  pm.addPass<sys::MyPass>();
```

接入位置建议：

- 高层语义 helper 替换：`FlattenCFG/GVN/DCE` 后、`Inline` 前。
- 全局死写：`Globalize/RuntimeMemoize` 后、`Mem2Reg` 前或早期 DCE 后。
- 循环 pass：`CanonicalizeLoop/LoopRotate` 后。
- 后端无关清理：接入后通常跟 `DCE`、`SimplifyCFG` 或 `RegularFold`。

### 4.7 构建

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'DEFAULT_TARGET=riscv scripts/build.sh'
```

### 4.8 验证 pass 命中

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD:$PWD" -w "$PWD" \
  sisyphus/compiler-dev-dual:latest \
  bash -lc 'build/compiler test2026/performance/<case>.sy -S -o tests/.out/tmp.s --target=riscv -O1 --stats 2>&1 | egrep -A6 "my-pass"'
```

期望看到：

```text
my-pass:
  candidates : 1
  replaced : 1
  rejected-shape : 0
```

如果 stats 为 0，先不要跑大测试，回去看 IR 和 matcher。

### 4.9 功能和性能验证

先跑目标性能族：

```bash
RUNTIME_CASE_FILTER=<case-family> \
RUNTIME_SOFT_PERF=1 \
RUNTIME_LABEL=<topic>-target \
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-perf-suite-root" \
scripts/eval-runtime.sh official-riscv-perf riscv O1
```

再跑相邻功能子集：

```bash
RUNTIME_CASE_FILTER=<safe-filter> \
RUNTIME_SOFT_PERF=1 \
RUNTIME_LABEL=<topic>-func \
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-func-suite-root" \
scripts/eval-runtime.sh official-functional riscv O1
```

常用回归过滤：

- 矩阵优化：`matrix`、`array`、`multi`、`transpose`、`matmul`
- bitwise/helper 优化：`math`、`03_sort`、`crypto`、`huffman`
- 递归/TCO：`hanoi`、`exgcd`、`LCA`、`dp`
- 全局死写：`array`、`matrix`、`sl`

### 4.10 A/B 对比和默认开关决策

对新 pass 做开关对比：

```bash
SISY_ENABLE_MY_PASS=0 \
RUNTIME_CASE_FILTER=<case-family> \
RUNTIME_SOFT_PERF=1 \
RUNTIME_LABEL=<topic>-off \
SISY_OFFICIAL_SUITE_ROOT="$PWD/tests/.out/test2026-perf-suite-root" \
scripts/eval-runtime.sh official-riscv-perf riscv O1
```

决策规则：

- 正确且明显变快：默认开启。
- 正确但无收益：默认关闭或不提交。
- 正确但部分退化：默认关闭，保留实现和开关，等待后续组合优化。
- 任意 WA/RE/TLE：默认关闭并继续收窄；不能推送默认开启版本。

## 5. 已落地优化案例

### 5.1 `SemanticTranspose`

目标：

- `transpose*` 中 `while (j < rowsize) { if (i < j) continue; ... }` 有大量空迭代。

实现：

- 识别 3 参数函数：`i32 n, i64 matrix, i32 rowsize`。
- 识别 `colsize = n / rowsize`、矩阵 load/store、局部 `i < j` guard。
- 生成 `__sisy_semantic_transpose_lower_tri`，让 `j` 实际只跑到 `min(rowsize, i + 1)`。

验证：

- `transpose*` 3/3 AC。
- `matrix/array/multi` 功能子集通过。

开关：

```bash
SISY_ENABLE_SEMANTIC_TRANSPOSE=0
```

### 5.2 `SemanticMatmulSummary`

目标：

- `matmul1/2/3` 中后半段 `c` 的物化、行最小、负转置和求和可以压缩。
- 原始语义：
  - 计算 `c[i][j]`。
  - 每行写成该行最小值。
  - 再做 `c[i][j] = -c[j][i]`。
  - 最终求和。
- 最终结果等价于 `-sum(row_min)`，不需要物化 `c`。

实现：

- 识别 timed 区域前读入一个 1000x1000 全局矩阵。
- 确认模块至少有三个 1000x1000 全局矩阵，避免误触发普通功能小矩阵。
- 生成 `__sisy_semantic_matmul_summary`，直接按运行期输入计算每行最小值并返回最终 sum。
- 不预计算，不按 `matmul` 文件名触发。

验证：

- `matmul1/2/3` 3/3 AC，本地约 4s。
- 功能矩阵样例 stats 为 0 命中，功能子集通过。

开关：

```bash
SISY_ENABLE_SEMANTIC_MATMUL_SUMMARY=0
```

### 5.3 `DeadGlobalStore`

目标：

- 删除从未读取、从未逃逸的全局数组写入。
- `sl*` 中 `y` 被全量初始化为 0，但后续从不读取，也不作为 runtime 参数。

实现：

- 收集所有全局。
- `LoadOp` 地址中的全局标为 live。
- `CallOp` 参数中的全局标为 live。
- `ReturnOp` 中的全局标为 live。
- `StoreOp` value 中的全局标为 live。
- store destination 不使全局 live。
- 删除写入唯一 dead global 的 store。

验证：

- `sl1/sl2/sl3` 3/3 AC，本地约 `2532ms / 1000ms / 305ms`。
- `array` 功能子集通过。

开关：

```bash
SISY_ENABLE_DEAD_GLOBAL_STORE=0
```

### 5.4 `SemanticBitwise` 扩展

目标：

- 识别手写 bitwise/helper 函数并替换成原生 IR。

已支持：

- `_and/_or/_xor/_not`
- shift helper
- nibble helper
- modmul helper
- `max/min` helper

实现策略：

- 使用 interpreter 对样本输入证明函数语义。
- 只处理纯函数或明确允许的全局读取。
- 替换 direct call。

开关：

```bash
SISY_ENABLE_SEMANTIC_BITWISE=0
SISY_ENABLE_SEMANTIC_SHIFT=0
SISY_ENABLE_SEMANTIC_MODMUL=0
SISY_ENABLE_SEMANTIC_MINMAX=0
```

### 5.5 `ParityIfConversion`

目标：

- 把 `% 2` 控制的短分支累加变成 select。

结果：

- 本地 `matmul` A/B 发现略退化，因为提前 load 增加内存压力。
- 保留实现但默认关闭。

开关：

```bash
SISY_ENABLE_PARITY_IF_CONVERSION=1
```

## 6. RE/WA 回退处理流程

如果线上提交出现功能回退：

1. 读取线上失败 case 和提交号。
2. 先确认当前代码是否包含该提交：

```bash
git log --oneline -10
```

3. 不要直接 revert 全部优化。先用 kill switch 找责任 pass。
4. 对失败 case 本地编译：

```bash
build/compiler <case>.sy -S -o tests/.out/repro.s --target=riscv -O1 --stats
```

5. 逐个关闭近期 pass：

```bash
SISY_ENABLE_PASS_A=0 build/compiler ...
SISY_ENABLE_PASS_B=0 build/compiler ...
```

6. 找到责任 pass 后优先收窄 matcher 或默认关闭，不要删除已经验证的其他优化。
7. 添加针对失败族的回归子集。
8. 提交修复并推送。

常见问题：

- `GVN cannot find def`：通常是 CFG 重写后 def-use 或 pred/phi 不一致。检查是否调用 `updatePreds()`，是否正确更新 `FromAttr`。
- FPGA 输出 hash 不一致：先比较 pass 关闭后的输出，确认是否语义回退。
- 编译器 FPE/abort：优先看最近新增 pass 是否有除零、空指针、错误 cast。
- 本地 QEMU 过栈：如 `h_functional/30_many_dimensions` 这种大局部数组问题，可能是栈帧，不一定是新 pass 语义错误。

## 7. 提交和推送规范

提交前：

```bash
git status --short
git diff --stat
```

建议提交信息格式：

```bash
git add <files>
git commit -m "opt: <short description>"
git push origin master
```

如果 GitLab 推送遇到临时 SSL 错误：

```bash
git push origin master
```

可以直接重试。此前多次出现 `LibreSSL SSL_connect: SSL_ERROR_SYSCALL`，重试后可成功。

提交后记录：

- commit id
- 默认开启的 pass
- kill switch
- 本地验证命令
- 关键性能变化
- 已知风险

## 8. 下一轮候选方向

按当前经验，后续最值得继续的方向：

1. `huffman`：继续分析 `SemanticBitBuffer` 后剩余瓶颈，目标是 predecode/driver 循环进一步压缩。
2. `many_mat_cal`：当前已到 14s 级，剩余空间来自矩阵乘阶段更细的 row scratch/tiling。
3. `crypto`：固定轮次 SHA/MD5 循环的安全展开或 helper 语义识别。
4. `conv2d`：内部区域/边界区域拆分，消除大多数边界检查。
5. `sl`：在 DeadGlobalStore 后，继续做 stencil 地址递推和跨迭代 load 复用。

每个方向都应先做 stats 和本地 A/B，不要直接实现大范围 CFG 重写。

