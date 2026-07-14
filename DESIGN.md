# VecAdvisor Design

VecAdvisor is a cost-based CLI advisor for filtered vector search in
PostgreSQL with pgvector. It does not execute every possible strategy to pick
a winner. It predicts a safe strategy from catalog statistics, PostgreSQL plan
evidence, bounded local-selectivity probes, and calibrated cost constants.

## Problem

The common filtered vector query shape is:

```sql
SELECT id
FROM documents
WHERE tenant_id = 42
ORDER BY embedding <-> $query_vector
LIMIT 10;
```

If pgvector searches an approximate vector index first and PostgreSQL applies
the scalar filter after candidate generation, selective filters can silently
lose recall or return fewer than `LIMIT` rows. Global filter selectivity is not
enough to model that risk because embeddings and predicates are often
correlated. The useful signal is local selectivity: the fraction of nearest
neighbors that satisfy the filter near representative query vectors.

## Goals

- Predict recall-safe strategies without brute-forcing every candidate per
  query.
- Keep production safety defaults: read-only SQL, statement timeouts,
  identifier quoting, bounded probes, and no planner modification.
- Prefer low IO, CPU, and memory cost by using catalog statistics first and
  running bounded probes only when representative query vectors are supplied.
- Make uncertainty explicit through confidence, notes, and exact fallbacks.
- Preserve a clean path to a later PostgreSQL extension and SIMD exact kernels.

## MVP1 Architecture

The Python CLI has five layers:

- Catalog introspection reads table rows, vector dimension, column statistics,
  pgvector indexes, extended statistics, and capability metadata.
- Predicate parsing accepts a safe subset of SQL filters and rejects unsupported
  shapes instead of estimating them silently.
- Selectivity modeling estimates global filter selectivity and compares it with
  PostgreSQL EXPLAIN plan rows.
- Local probing runs an unfiltered top-m vector neighborhood query and counts
  filter-passing rows with `count(*) FILTER`. Multi-vector recommendations use
  p10 local selectivity for conservative costing.
- The cost model ranks exact, fixed post-filter ANN, iterative ANN, partial
  index, and partition-pruned strategies using calibrated constants and
  strategy-specific recall/returns-k estimates.

## Local-Selectivity Probe

The probe query is intentionally bounded:

```sql
WITH nearest AS MATERIALIZED (
  SELECT filter_columns
  FROM table
  ORDER BY embedding <-> $query_vector
  LIMIT $probe_rows
)
SELECT count(*) AS sample_size,
       count(*) FILTER (WHERE filter_predicate) AS passing_rows
FROM nearest;
```

Before the probe runs, VecAdvisor uses a read-only transaction-local setup:

- `statement_timeout` limits accidental long scans.
- `enable_seqscan=off` biases the probe toward the ANN index.
- `hnsw.ef_search` is raised to at least `probe_rows` for HNSW indexes.
- `ivfflat.probes` is raised from index metadata and table cardinality for
  IVFFlat indexes.
- `EXPLAIN (FORMAT JSON)` confirms that the probe plan uses an HNSW or IVFFlat
  index when index-backed probing is required.

This keeps the measured neighborhood aligned with the requested probe
resolution. If no representative query vector is available, VecAdvisor clearly
falls back to global selectivity or an explicitly requested low-confidence
table-sampled vector path.

## Costing Contract

Benchmarking is for calibration and validation, not for production-time
selection. The advisor computes candidate estimates from:

- row count, pages, vector dimension, and filter selectivity,
- local selectivity p10 and median,
- observed PostgreSQL planner shape,
- pgvector capabilities such as iterative scan support,
- calibration profile constants for distance cost, scan cost, HNSW overhead,
  strict-ordering penalty, and recall curve.

Every candidate emits estimated latency, recall, returns-k, confidence, and
notes. If the faster candidates cannot satisfy quality constraints with enough
confidence, exact search remains the recall-safe fallback.

## Performance Invariants

- Ground-truth exact search is blocked per query and never materializes an
  `N x Q` distance matrix.
- Local-selectivity cache keys include query/filter shape, query-vector
  fingerprint, table stats fingerprint, and index fingerprint.
- Cache entries store aggregate selectivity only, not raw vectors or row IDs.
- CLI SQL is read-only for advisory paths and uses parameterized values plus
  quoted identifiers.
- Tests cover the model math, local probing, PostgreSQL EXPLAIN selectivity
  anchors, benchmark proof artifacts, and packaging checks.

## Extension Path

MVP2 keeps native code where it earns its keep:

- C++ SIMD distance kernels for high-throughput exact ground truth and
  benchmark calibration.
- A Rust/pgrx PostgreSQL extension path for planner hooks, cost-estimate
  integration, GUCs, and future index access-method experiments.

MVP3 adds workload collection and deeper production ergonomics, including
`pg_stat_statements` mining, richer predicate support, and more benchmark
artifacts.

## Non-Goals

- VecAdvisor does not replace pgvector.
- The MVP1 CLI does not modify PostgreSQL's planner.
- It does not promise universal SQL predicate support.
- It does not bundle large embedding datasets in the source repository.
