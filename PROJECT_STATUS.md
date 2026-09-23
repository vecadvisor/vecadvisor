# VecAdvisor Project Status

VecAdvisor is at a public alpha closure point: MVP1 and the MVP2 foundation are
complete enough to present as a serious systems project, while hardware-specific
native optimizations remain explicit post-alpha roadmap work.

## Complete

- MVP1 Python advisor:
  - PostgreSQL catalog/statistics introspection.
  - Safe predicate parsing and selectivity estimation.
  - Local selectivity probes with p10 aggregation and graceful fallback.
  - Cost-based recommendations for exact, post-filter ANN, iterative ANN,
    partial index, and partition-pruned strategies.
  - Reproducible synthetic, real pgvector, and SIFT1M benchmark artifacts.
  - Publishable CLI, README, examples, CI, package build, and PyPI releases.
- MVP2 native foundation:
  - C++17 float32 scalar kernels and AVX2/FMA runtime dispatch.
  - Bounded native top-k C ABI for exact ground-truth search without
    materializing an `N x Q` distance matrix.
  - Python `ctypes` wrapper and CLI evidence command for native-backed exact
    ground truth.
  - Scalar int8 C ABI helpers for scalar-quantized vectors:
    `compute_i8`, `compute_many_i8`, and `topk_i8`.
  - Linux and Windows native CI coverage.
- MVP2 pgrx foundation:
  - Rust/pgrx extension scaffold.
  - Read-only metadata functions.
  - Pure SQL-callable `vecadvisor_postfilter_risk(...)` estimator.
  - No planner hooks, catalog writes, probe execution, or GUC mutation.

## Current Public Surface

- Python CLI package: `vecadvisor`.
- Latest alpha release: see `docs/release.md`.
- Native C++ and pgrx code are foundation layers. They are tested and
  documented, but the Python CLI remains the supported user-facing advisor.

## Post-Alpha Roadmap

The remaining work is intentionally scoped as post-alpha engineering, not a
blocker for the current project phase:

- ARM NEON distance kernels for Apple Silicon and ARM servers.
- AVX-512 distance kernels if benchmark evidence justifies the extra path.
- fp16/halfvec native kernels.
- pgrx parity fixtures against the Python CLI before any deeper SQL advisor
  surface.
- Planner hooks only after parity, regression tests, opt-in GUCs, and safety
  controls exist.

## Closure Criteria

This phase is considered complete when:

- CI and Native workflows pass on `main`.
- Latest alpha is tagged and published.
- Remaining roadmap items are tracked as post-alpha issues.
- The repository clearly states what is complete and what is intentionally
  deferred.
