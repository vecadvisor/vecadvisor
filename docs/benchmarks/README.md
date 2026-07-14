# Benchmark Artifacts

This directory contains deterministic synthetic evidence for the current
VecAdvisor alpha.

- `synthetic-calibration.json`: fitted constants from a small synthetic run.
- `synthetic-sweep.json`: selectivity/correlation sweep with measured and
  predicted strategy outcomes.
- `synthetic-proof.json`: proof summary derived from the sweep.
- `real-pgvector-benchmark.json`: actual PostgreSQL/pgvector benchmark with
  exact, post-filter HNSW, iterative HNSW, partial HNSW, and partition-pruned
  HNSW strategies.
- `real-pgvector-benchmark.md`: summary and reproduction commands for the real
  pgvector artifact.
- `real-pgvector-calibration.json`: small actual-Postgres calibration profile.
- `real-pgvector-sweep.json`: actual-Postgres local-selectivity validation
  sweep.
- `real-pgvector-plan.md`: plan for the first Docker-reproducible benchmark
  against real PostgreSQL and pgvector behavior.

These files are intentionally small enough to regenerate during local
development. Synthetic files validate cost-model behavior and safety checks.
Real pgvector files measure actual PostgreSQL/pgvector SQL on small
deterministic datasets and should not be generalized without workload-specific
calibration.
