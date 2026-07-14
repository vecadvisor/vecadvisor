# VecAdvisor

[![CI](https://github.com/vecadvisor/vecadvisor/actions/workflows/ci.yml/badge.svg)](https://github.com/vecadvisor/vecadvisor/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/vecadvisor/vecadvisor)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

Cost-based CLI advisor for filtered vector search in PostgreSQL and pgvector.

VecAdvisor helps answer a question PostgreSQL does not currently cost
well:

```sql
SELECT id
FROM documents
WHERE tenant_id = 42
ORDER BY embedding <-> $query_vector
LIMIT 10;
```

When a vector index is searched first and SQL filters are applied after the
ANN candidate set, selective filters can silently lose recall or return fewer
than `k` rows. Sometimes a filter-first exact scan is faster and exact.
Sometimes pgvector iterative scans are the right answer. Sometimes a partial
HNSW index or partitioning is the durable fix.

This project models that choice with:

- PostgreSQL catalog and statistics introspection.
- Global selectivity estimation and PostgreSQL plan-row cross-checks.
- Local selectivity probes over the query vector neighborhood.
- A calibrated cost model for exact, post-filter ANN, iterative ANN, partial
  index, and partition-pruned strategies.
- Reproducible synthetic benchmark, calibration, validation, crossover, and
  proof reports.

Status: alpha. The Python CLI is useful today for experimentation and design
validation. It does not modify PostgreSQL's planner.

VecAdvisor is an independent third-party tool. It works with PostgreSQL and
pgvector, but it is not affiliated with the official pgvector project.

## Why Local Selectivity Matters

Global selectivity is the fraction of the whole table matching a filter.
Local selectivity is the fraction of near-neighbor candidates matching the
filter for a query vector.

For filtered ANN, local selectivity is the important signal. A tenant may be
globally rare but locally dense near its own documents. Or it may be locally
sparse for a query outside that tenant's embedding cluster. A naive `k / s`
model misses this. VecAdvisor probes a bounded unfiltered top-m
neighborhood and costs durable recommendations against p10 local selectivity
across representative query vectors.

## Benchmark Evidence

The repository includes a deterministic synthetic crossover sweep under
`docs/benchmarks/` and a rendered chart:

![Synthetic crossover chart](docs/assets/synthetic-crossover.svg)

This sweep varies filter selectivity (`0.01`, `0.05`, `0.1`, `0.3`) and
filter/vector correlation (`-0.6`, `0`, `0.6`) across 12 points. In this
strategy-semantics simulation:

- Prediction match rate: `100%`.
- Default post-filter ANN missed recall or returns-k targets in `12/12` bins.
- VecAdvisor avoided post-filter ANN in all post-filter failure bins.
- Mean speedup versus post-filter ANN: `1.98x`.

Reproduce the artifact:

```bash
vecadvisor calibrate \
  --dataset synthetic \
  --rows 2048 \
  --dim 32 \
  --queries 24 \
  --clusters 8 \
  --filter-selectivity 0.1 \
  --correlation 0.5 \
  --limit 10 \
  --ef-sweep 10,20,40,80,160 \
  --seed 410 \
  --dataset-id synthetic-readme \
  --hardware-id github-readme-synthetic \
  --out docs/benchmarks/synthetic-calibration.json

vecadvisor benchmark-sweep \
  --dataset synthetic \
  --rows 2048 \
  --dim 32 \
  --queries 24 \
  --clusters 8 \
  --filter-selectivities 0.01,0.05,0.1,0.3 \
  --correlations -0.6,0,0.6 \
  --limit 10 \
  --ef-search 40 \
  --max-scan-tuples 1000 \
  --probe-rows 200 \
  --calibration docs/benchmarks/synthetic-calibration.json \
  --seed 411 \
  --out docs/benchmarks/synthetic-sweep.json

vecadvisor proof docs/benchmarks/synthetic-sweep.json \
  --out docs/benchmarks/synthetic-proof.json

vecadvisor plot-crossover docs/benchmarks/synthetic-sweep.json \
  --out docs/assets/synthetic-crossover.svg \
  --title "VecAdvisor synthetic crossover"
```

The current proof is synthetic and intentionally small enough to run quickly
in a developer checkout. It validates the advisor's cost-model behavior and
quality safeguards; it is not a replacement for workload-specific calibration
against a production pgvector index.

## Install

For local development:

```bash
python -m pip install -e ".[dev]"
```

For package users once released on PyPI:

```bash
python -m pip install vecadvisor
```

The CLI entry point is:

```bash
vecadvisor --help
```

## Local PostgreSQL

Start the bundled pgvector database:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Load the example table:

```bash
psql "postgresql://postgres:postgres@localhost:5432/vecadvisor" -f examples/demo.sql
```

If you do not have `psql` locally, use Docker:

```bash
docker compose -f docker/docker-compose.yml exec -T postgres \
  psql -U postgres -d vecadvisor < examples/demo.sql
```

The default test DSN is:

```text
postgresql://postgres:postgres@localhost:5432/vecadvisor
```

## Analyze A Table

```bash
vecadvisor analyze \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding
```

This reports table rows, vector dimension, indexes, extended statistics,
catalog fingerprints, stats freshness, and pgvector capabilities.

## Explain One Query Vector

Use `explain` when you have one concrete query vector and want a one-query
diagnostic:

```bash
vecadvisor explain \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 1" \
  --q-vector examples/query-vector.json \
  --probe-rows 16 \
  --format text
```

`explain` reads `--q-vector`, runs one local selectivity probe, observes the
planner's vector query shape, and renders an `EXPLAIN VECTOR` style report.

## Recommend A Durable Strategy

Use `recommend` when you have representative query vectors for a workload:

```bash
vecadvisor recommend \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 1" \
  --q-vectors examples/query-vectors.json \
  --probe-rows 16 \
  --max-query-vectors 3 \
  --local-cache-dir .vecadvisor-cache
```

`recommend` uses p10 local selectivity across the vector sample. It emits:

- Ranked candidate strategies.
- Estimated latency, recall, returns-k, and confidence.
- A concise decision summary.
- PostgreSQL observed plan, when a vector sample is available.
- Statistics suggestions such as `CREATE STATISTICS`.
- Optional partial index or partition strategy suggestions.

If you do not have production query vectors yet, you may explicitly opt in to
a low-confidence table-sample fallback:

```bash
vecadvisor recommend \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --table public.documents \
  --vector embedding \
  --query "tenant_id = 1" \
  --allow-table-sample-vectors
```

Table-sampled vectors are marked in the output and reduce confidence.

## Local Selectivity Cache

Repeated multi-vector probes can be cached:

```bash
vecadvisor recommend ... --local-cache-dir .vecadvisor-cache
```

Cache keys include:

- table name
- filter shape
- probe rows
- query vector source
- query vector fingerprint
- table statistics fingerprint
- index fingerprint

Changing table statistics or index definitions naturally produces a different
cache key. Cache entries contain only aggregate local selectivity, not raw
vectors or row ids.

Use `--refresh-local-cache` to ignore and replace an existing cache entry.

## Calibrate

Synthetic calibration fits cost constants and an unfiltered recall curve:

```bash
vecadvisor calibrate \
  --dataset synthetic \
  --rows 5000 \
  --dim 64 \
  --queries 50 \
  --clusters 16 \
  --filter-selectivity 0.1 \
  --correlation 0.5 \
  --out profiles/local.json
```

Load a profile during `explain` or `recommend`:

```bash
vecadvisor recommend ... --calibration profiles/local.json
```

Without a profile, the advisor uses conservative default constants and
marks that in candidate notes.

## Benchmark, Sweep, Validate, Prove

Run a small synthetic benchmark:

```bash
vecadvisor benchmark \
  --dataset synthetic \
  --strategies exact,postfilter,iterative \
  --rows 512 \
  --dim 16 \
  --queries 8 \
  --clusters 8 \
  --filter-selectivity 0.1 \
  --correlation 0.5 \
  --limit 10 \
  --ef-search 40 \
  --out benchmarks/synthetic.json
```

Run a selectivity and correlation sweep:

```bash
vecadvisor benchmark-sweep \
  --dataset synthetic \
  --rows 512 \
  --dim 16 \
  --queries 8 \
  --clusters 8 \
  --filter-selectivities 0.01,0.05,0.1 \
  --correlations 0,0.5 \
  --limit 10 \
  --ef-search 40 \
  --out benchmarks/sweep.json
```

Analyze crossover behavior:

```bash
vecadvisor crossover benchmarks/sweep.json --out benchmarks/crossover.json
```

Build a publishability proof report:

```bash
vecadvisor proof benchmarks/sweep.json --out benchmarks/proof.json
```

Render a crossover chart:

```bash
vecadvisor plot-crossover benchmarks/sweep.json --out benchmarks/crossover.svg
```

## Output Contract Highlights

Important JSON fields:

- `catalog_snapshot`: stats and index fingerprints for reproducibility.
- `selectivity`: advisor-vs-PostgreSQL selectivity cross-check with severity.
- `local_selectivity`: p10 and median local selectivity, confidence, and notes.
- `local_selectivity_cache`: cache hit/store metadata when enabled.
- `recommendation.decision`: winner, runner-up, viability, confidence, and why.
- `recommendation.ranked`: full candidate plan and estimate details.

## Current Limitations

- Alpha quality: APIs and output shape may change before v0.1.
- The current release is an external CLI. It does not change PostgreSQL planner behavior.
- Realistic public benchmark datasets are not bundled yet.
- Calibration constants are workload and hardware dependent.
- Predicate parsing intentionally supports a restricted safe subset of filters.
- 10M-scale exact ground truth is deferred to the planned C++ SIMD kernel work.
- Postgres integration tests require a reachable PostgreSQL database with
  pgvector installed. Tests skip those cases if the database is unavailable.

## Development

Install dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run checks:

```bash
python -m pytest
python -m ruff check .
python -m mypy src/vecadvisor
python -m build
```

## Prior Art And Clean-Room Notes

This project builds from public PostgreSQL, pgvector, and filtered ANN
literature. It does not copy source from AGPL or source-available vector
extensions. Ideas such as local selectivity, filtered ANN recall risk,
iterative scans, partial indexes, and calibration are implemented from public
documentation, papers, and first-principles cost modeling.

## License

Apache License 2.0. See `LICENSE`.

Built and maintained by Varun Sharma.
