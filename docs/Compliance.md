# Compliance Notes

- This project is implemented as a standalone codebase in `sisyphus`.
- `base` and `biframe` are used as design references for architecture and optimization ideas.
- Any generated or AI-assisted edits should be documented in commit messages or code review notes with impacted file paths.
- Do not perform testcase-specific optimizations. Passes must remain semantics-preserving and input-agnostic.
- The current `pure-rv` branch defaults to all-optimization mode. Riskier
  recognizers are still controlled by `SISY_ENABLE_*` environment switches; use
  those switches for bisection instead of editing pass code during validation.
- Keep generated outputs under ignored directories such as `tests/.out/`,
  `build/`, `build-linux/`, or `output/`. Do not commit local assembly dumps,
  logs, `.DS_Store`, or temporary files.

See `docs/Optimization.md` for the optimization legality checklist and
`docs/Commands.md` for the current verification commands.
