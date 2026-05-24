# Compliance Notes

- This project is implemented as a standalone codebase in `sisyphus`.
- `base` and `biframe` are used as design references for architecture and optimization ideas.
- Any generated or AI-assisted edits should be documented in commit messages or code review notes with impacted file paths.
- Do not perform testcase-specific optimizations. Passes must remain semantics-preserving and input-agnostic.
- The default RISC-V profile keeps general optimizations enabled, but high-risk
  semantic or structural recognizers are strict-mode opt-in only. Use
  `SISY_ENABLE_*` switches for explicit comparison runs instead of editing pass
  code during validation.
- Compile-time recursive precomputation, structural bitwise/modmul recognition,
  row-scratch matrix helper replacement, and SMT synthesized constant arrays
  must remain disabled by default. Prefer runtime memoization, affine loop
  transforms, scalar replacement, DSE/DLE, SCEV, and ordinary algebraic folds.
- Keep generated outputs under ignored directories such as `tests/.out/`,
  `build/`, `build-linux/`, or `output/`. Do not commit local assembly dumps,
  logs, `.DS_Store`, or temporary files.

See `docs/Optimization.md` for the optimization legality checklist and
`docs/Commands.md` for the current verification commands.
