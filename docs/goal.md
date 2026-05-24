**一、循环与多面体优化**
已实现部分：

- `src/hir/HIRAffine.cpp/.h` 已经有统一 affine 表达、array access 收集、canonical loop 匹配。
- 已接入 `src/utils/presburger/BasicSet.cpp`，用于 fusion/interchange 的安全性查询。
- `HIRPolyhedral.cpp` 已经在 fusion、loop interchange、unroll-and-jam 中调用 Presburger-based legality。
- `scripts/run_polyhedral_tests.sh` 覆盖：
  - Presburger fusion 证明错位 domain 安全；
  - commuted induction step；
  - reduction unroll-and-jam。

局限：

- 仍主要支持 canonical `while iv < bound; iv += 1`。
- Presburger 查询是定制化 dependence check，不是完整 relation/projection/schedule optimizer。
- 还没有完整 SCoP 提取、参数化调度、skewing、skewed tiling。
- `HIRPolyhedral` 仍混有较多 shape matcher，复杂控制流/非仿射边界会拒绝。

后续建议：

1. 把 `HIRAffine` 扩展成统一 `AffineNest` 数据结构：loop domain、access relation、side-effect summary 分开。
2. 用 Presburger 统一回答 `interchange/fusion/tiling/unroll-jam` legality，减少各 pass 自己写 shape 规则。
3. 下一步优先支持 3D perfect nest 的 dependence direction，这对 `many_mat_cal/matmul/transpose` 最直接。
4. 暂不做完整 Pluto/Polly 调度，先做固定变换的 Presburger legality。

**二、向量化**
已实现部分：

- `src/opt/Vectorize.cpp` 已有 loop vectorization 和 SLP 相关逻辑。
- RVV/NEON emitter 文件存在：`RVVEmitter.cpp`、`NEONEmitter.cpp`、`VectorCodeGenImpl.cpp`。
- `scripts/run_vectorize_tests.sh` 覆盖 RVV int/float loop vectorization、SLP add/sub/copy/splat，以及 ARM NEON 输出。
- 有同 base loop-carried dependence 的负例测试，说明合法性不是完全裸奔。

局限：

- 依赖 `--enable-experimental`，不是完全默认高置信启用。
- 成本模型较粗，未结合寄存器压力、trip count、memory alignment、目标微架构。
- RVV/NEON 代码生成更多是 pattern-level emission，不是后端统一 vector IR lowering。
- 对 reduction、gather/scatter、masked tail、复杂 stride 支持有限。

后续建议：

1. 先完善成本模型：小 trip count、过高 live vector 数、跨 call/branch 的块拒绝。
2. 增加 vector legality 与 `HIRAffine`/Alias 共享，减少重复判断。
3. 对 `conv2d/matmul` 先做固定 stride contiguous vectorization，而不是全功能 RVV。
4. 默认启用前，增加 performance/functional 子集回归，尤其 float ABI 和 spill。

**三、Unroll-and-Jam**
已实现部分：

- `HIRPolyhedral.cpp` 中有 `tryLoopUnrollJam`。
- `scripts/run_polyhedral_tests.sh` 里 `reduction_unroll_jam.sy` 要求 `reduction-jammed=1`。
- legality 已调用 Presburger interchange safety。

局限：

- 仍是 HIR 上的特定 2 层 canonical nest。
- jam factor 当前是固定启发式，不够结合 register pressure。
- 对非 reduction 通用 unroll-and-jam 支持有限。
- 还没有和后端 spill 反馈闭环，可能出现展开收益被 spill 吃掉。

后续建议：

- 把 unroll-jam factor 接入 `RegisterPressure` 估算。
- 对 2D/3D affine nest 做“外层小因子 jam + 内层保持循环”的低风险版本。
- 默认只对 `Presburger safe + body op count 小 + estimated pressure 低` 的 nest 开启。

**四、MemorySSA / Alias / Scalar Promotion**
已实现部分：

- `DLE.cpp` / `DSE.cpp` 已经具备 MemorySSA 风格 reaching-store/load forwarding。
- `scripts/run_memoryssa_tests.sh` 覆盖：
  - join 后相同 store 的 load forwarding；
  - runtime store forwarding；
  - 不同 store 通过 value phi forwarding；
  - branch common store sinking；
  - readonly call 保留 reaching store。
- 这部分已经比较接近“轻量 MemorySSA”。

局限：

- 不是完整 MemorySSA 图，没有显式 MemoryPhi/MemoryUse/MemoryDef 作为一等 IR。
- alias 查询仍依赖现有 `Alias` 和局部规则。
- 跨循环、跨复杂 CFG 的内存版本关系还保守。
- Escape-analysis-guided scalar promotion 尚未系统化；`ScalarReplace` 有，但不是完整逃逸分析驱动。

后续建议：

1. 建立显式 `MemoryAccess` side table：每个 load/store/call 对应 memory def/use。
2. 在 join block 构建虚拟 MemoryPhi，即使不改 IR，也在分析表里表达。
3. 给 `ScalarReplace` 增加 escape summary：局部数组是否传参、是否存入全局、是否被未知 call 触达。
4. 优先推广小固定局部数组标量化，这个收益稳且合规。

**五、SCEV / LSR**
已实现部分：

- `src/opt/SCEV.cpp` 已有 induction/increase 分析、replaceAfter、地址 stride 改写。
- `scripts/run_scev_tests.sh` 覆盖 reverse int-array traversal 生成 `-4` byte stride。
- pipeline 中多轮运行 `SCEV`，说明它已经是核心优化链的一部分。

局限：

- SCEV 表达能力仍偏工程化，非完整 scalar evolution lattice。
- 对多维数组寻址、复杂 affine recurrence 的统一表达还不够。
- LSR 与后端 addressing mode 结合有限。
- 没有系统做 recurrence cost model。

后续建议：

1. 把 SCEV 的 affine 表达和 `HIRAffine` 的表达统一，避免两套分析。
2. 对 `base + i * stride + j * stride2` 做 CFG 层地址 recurrence 识别。
3. 在 lowering 前把乘法寻址改成 induction pointer increment。
4. 增加 matmul/transpose/conv2d 的 `--print-after scev` 回归检查。

**六、CVP / Jump Threading**
已实现部分：

- `RangeAwareFold.cpp` 已有 correlated range fold、path-sensitive equality replacement、jump threading。
- `scripts/run_cvp_tests.sh` 覆盖 `<=`、`==`、reversed operand、nonzero、value substitution、threaded edges。
- 最近修复了 jump threading 穿过带 phi block 破坏 SSA 的问题。

局限：

- 不是完整 SCCP；没有 lattice worklist 驱动全函数常量传播。
- Jump threading 很保守，遇到 phi block 直接拒绝。
- 分支复制/路径复制能力有限。

后续建议：

1. 做 Sparse Conditional Constant Propagation：reachable block + SSA value lattice。
2. 在 phi block threading 上实现安全版：复制 decision block 或正确重写 phi incoming，而不是直接跳过。
3. CVP 和 EqClass、Range 分析做固定点迭代，避免单轮错过机会。

**七、后端 RegAlloc / Schedule**
已实现部分：

- RV/ARM `RegAlloc` 都接入 `RegAllocHotness`，根据 back-edge hotness 加权 spill weight。
- RV regalloc 有 spill weight、call span penalty、copy preference、callee-saved preference。
- `scripts/run_regalloc_hotness_tests.sh` 验证 nested loop hotness >= 64。
- `rv/Schedule.cpp` 有 pre-RA list scheduling，带 load/mul/div latency 模型、memory dependency DAG、AliasAttr 辅助。
- `src/opt/InstSchedule.cpp` 刚改成 side-map DAG，避免调度失败污染 IR，并消除了线上 `index == allowed.size()` 断言风险。

局限：

- 尚无真正 live interval splitting。现在只是 spill 权重排序，不会把长 live range 拆成热/冷段。
- spill/reload 插入仍是统一 stack slot 模式，没有 loop-boundary splitting。
- 调度模型是粗延迟模型，不是目标 pipeline model。
- RV scheduler 对 FP-heavy block 直接保守跳过。
- ARM 后端调度深度不如 RV。

后续建议：

1. 先做“伪 live range splitting”：对跨 call/跨循环但局部热使用的值，在热块入口插 reload-like copy，缩短干扰范围。
2. RegAlloc 加 spill candidate 分类：rematerializable `li/la`、loop-carried phi、call-spanning value 分开处理。
3. RV Schedule 加 critical path height，当前只看 local latency，不能优先排长链 producer。
4. ARM 复用 RV side-map DAG scheduler 思路，先保证安全再加 latency。
5. 增加 asm-level 统计脚本：spill load/store 数、callee-saved 使用、schedule reordered 数，用于对比优化是否真的减少栈流量。