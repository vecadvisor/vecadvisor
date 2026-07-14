# Real pgvector Benchmark Plan

The synthetic proof validates VecAdvisor's cost-model behavior. Before MVP2
native kernels, the next credibility step is one Docker-reproducible benchmark
against actual PostgreSQL and pgvector behavior.

## Goal

Measure whether VecAdvisor's predicted strategy matches real pgvector query
behavior across selective and correlated filters, without running every
strategy at recommendation time.

## Benchmark Shape

- Database: bundled `pgvector/pgvector:pg17` container.
- Dataset: deterministic clustered synthetic embeddings loaded into
  PostgreSQL.
- Sizes: start with `100k` rows for CI/manual repeatability; allow larger local
  runs through parameters.
- Dimensions: `64` and `128`.
- Query vectors: representative vectors drawn from multiple clusters.
- Filters:
  - rare tenant-like equality predicate
  - medium-selectivity tenant/date-like predicate
  - anti-correlated filter where post-filter ANN is expected to fail
  - locally dense filter where post-filter ANN may remain viable

## Strategies

Compare only strategies that map to real PostgreSQL actions:

- exact filter-first scan
- HNSW post-filter ANN
- HNSW iterative scan when supported
- partial HNSW index for stable predicates
- partition-pruned HNSW for tenant-like predicates

## Metrics

- p50 and p95 latency
- rows returned versus `k`
- recall@k against exact filtered ground truth
- index build time
- index size from PostgreSQL relation size
- planning time from `EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)`
- buffer hits/reads for IO visibility

## Implementation Guardrails

- Use deterministic seeds and record them in the output artifact.
- Separate calibration runs from validation runs.
- Keep the default run small enough for a laptop; document larger commands
  separately.
- Store summarized artifacts and charts, not raw row-level vectors.
- Make all claims explicit about hardware, PostgreSQL version, pgvector
  version, and dataset size.

## Acceptance Criteria

The first real benchmark artifact is good enough when it produces:

- one JSON summary under `docs/benchmarks/`
- one chart under `docs/assets/`
- a README paragraph explaining what was measured
- commands that regenerate the artifact from a clean checkout
- at least one case where local selectivity changes the recommendation versus
  a global-selectivity-only model
