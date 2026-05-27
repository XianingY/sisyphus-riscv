# 深度优化状态跟踪

最后更新：2026-05-27

> 本文件是当前优化状态摘要。更早的 `docs/superpowers/specs/` 内容属于历史设计草案，不能替代源码、`docs/Compliance.md` 和 `docs/Optimization.md`。

## 当前总体结论

- 默认优化路线已经从“语义识别替换”收敛到“通用编译器优化”。
- 高风险 pass 默认关闭：`FunctionEquivalence`、`StructuralBitwise`、`StructuralModMul`、`RowScratchMatmul`、`Cached`、`SynthConstArray`、`AdvancedConv2DTransform`。
- 当前最稳的提分方向是 HIR affine/Presburger、MemorySSA-style 内存优化、SCEV/range、后端寄存器分配和调度。
- RISC-V 目前是主力优化路径；ARM 分支仍需要同步更多后端和 HIR 侧收益。

## 已落地模块

| 模块 | 状态 | 说明 |
| --- | --- | --- |
| HIR affine framework | 已落地 | `HIRAffine` 收集 affine loop/access，并为 fusion/interchange/jam 提供基础。 |
| HIRPolyhedral | 已落地并持续迭代 | fusion、interchange、unroll-and-jam、reduction privatization、guard invariant hoist 等通用 loop transform。 |
| Presburger legality | 部分落地 | 用于轻量 dependence/feasibility 查询；不是完整 Polly/Pluto 调度器。 |
| MemorySSA-style DLE/DSE | 部分落地 | 有 reaching-store/load forwarding 思路，但还不是显式 MemorySSA IR。 |
| SCCP / RangeAwareFold | 已落地 | 路径敏感常量与范围折叠能力持续增强。 |
| SCEV / LSR | 已落地 | 支持归纳变量和部分地址 stride 改写，仍需与 HIRAffine 统一表达。 |
| Vectorize / RVV / NEON | 实验到半稳定 | 具备基础 loop/SLP/vector emitter；默认策略需继续受成本模型约束。 |
| RV RegAlloc/Schedule | 已落地 | hotness/spill 估算、pre-RA scheduling、peepholes；live-range splitting 仍可加强。 |
| ARM backend optimization | 可用但偏弱 | 正确性可用，性能仍落后 RISC-V 优化路径。 |

## 默认关闭的高风险模块

| 模块 | 原因 |
| --- | --- |
| `FunctionEquivalence` | 编译期采样/解释纯函数并替换为已知操作，合规风险高。 |
| `StructuralBitwise` | 直接识别软件位运算并替换，容易被视为算法语义替换。 |
| `StructuralModMul` | 直接识别递归模乘并替换，风险高。 |
| `RowScratchMatmul` | 替换矩阵循环为 helper，默认不作为合规路径。 |
| `Cached` | 编译期预计算递归结果，有硬编码答案风险。 |
| `SynthConstArray` | SMT 合成数组值，默认不启用。 |
| `AdvancedConv2DTransform` | Winograd/im2col dispatcher 需要完整 affine/padding 合法性证明，否则不默认开启。 |

## 当前主要瓶颈与合规路线

| 目标 | 当前建议路线 |
| --- | --- |
| `many_mat_cal` / `matmul` | 继续做 affine dependence、cache/register tiling、unroll-and-jam 成本模型、backend spill 控制。 |
| `transpose` | 通用 2D cache tiling、stride-aware interchange、address recurrence、vector handoff。 |
| `conv2d` | guarded/imperfect nest 提取、border peeling、interior 快路径、kernel 小常量循环展开，避免 conv-specific dispatcher。 |
| `crc` / `huffman` | range 证明、parity select、spill-aware unroll、readonly load forwarding；不启用 bitwise 语义识别。 |
| `fft` | TCO、单路递归转循环、普通代数折叠、range 暴露 `/2` `%2`；不启用 modmul 识别。 |
| `optimization_scheduling` | RegAlloc hotness、live-range splitting、rematerialization、pre-RA schedule，避免 recurrence 特判。 |

## 下一步优先级

1. 放宽 HIR affine nest 提取，支持 guarded/imperfect loop 的安全子区域。
2. 对 transpose/conv2d 共用的 2D tiling 做统一 legality 与成本模型。
3. 将 SCEV 地址 recurrence 与 HIRAffine 表达合并，减少多维数组寻址乘法。
4. 强化 RegAlloc 的 hot live-range splitting 和 rematerialization。
5. 为 Vectorize 接入 affine legality 与寄存器压力成本模型。

## 验证要求

每个阶段性优化至少需要：

- `scripts/build.sh`
- 受影响测试点的单点编译/运行验证
- 一个无关热点的回归探针
- `git diff --check`
- pass stats 对比，确认优化命中和 bailout 都符合预期
