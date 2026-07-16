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
- `scale-benchmark.md`: reproducible SIFT1M real-embedding benchmark recipe.
- `sift1m-pgvector-benchmark.json`: actual PostgreSQL/pgvector benchmark on
  one million real SIFT vectors.
- `sift1m-pgvector-benchmark.md`: summary and reproduction commands for the
  projection-tail SIFT1M artifact.
- `sift1m-anticorrelated-pgvector-benchmark.json`: actual PostgreSQL/pgvector
  benchmark on one million real SIFT vectors with a query anti-correlated
  scalar filter.
- `sift1m-anticorrelated-pgvector-benchmark.md`: summary and reproduction
  commands for the SIFT1M recall-collapse artifact.

These files are intentionally small enough to regenerate during local
development. Synthetic files validate cost-model behavior and safety checks.
Real pgvector files measure actual PostgreSQL/pgvector SQL and should not be
generalized without workload-specific calibration. The SIFT1M artifacts commit
only JSON/SVG output; source vectors are downloaded into ignored `data/` paths
and should be removed after local runs.
