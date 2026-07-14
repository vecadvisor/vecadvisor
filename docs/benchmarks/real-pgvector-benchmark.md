# Real pgvector Benchmark Artifact

This artifact measures actual PostgreSQL/pgvector SQL on the bundled Docker
database. It is intentionally small enough to regenerate from a developer
checkout.

## Environment

- Container image: `pgvector/pgvector:pg17`
- PostgreSQL: `17.10`
- pgvector: `0.8.4`
- Dataset: deterministic clustered synthetic embeddings
- Rows: `4096`
- Dimensions: `32`
- Query vectors: `8`
- Target global selectivity: `0.05`
- Observed global selectivity: `0.05126953125`
- Filter/vector correlation: `-0.6`
- `k`: `10`
- `ef_search`: `40`

## Result Summary

| Strategy | Recall@k | Returns-k rate | p95 latency |
| --- | ---: | ---: | ---: |
| exact | `1.0000` | `1.0000` | `1.78 ms` |
| postfilter | `0.2125` | `0.0000` | `66.31 ms` |
| iterative | `0.9625` | `1.0000` | `21.29 ms` |
| partial | `1.0000` | `1.0000` | `20.00 ms` |
| partition | `1.0000` | `1.0000` | `35.88 ms` |

The important signal is not that exact wins on this tiny dataset. The useful
system behavior is that fixed-size HNSW post-filtering returns too few rows
and loses recall under a selective anti-correlated filter, while iterative
scan, partial HNSW, and partition-pruned HNSW recover quality.

![Real pgvector Pareto chart](../assets/real-pgvector-pareto.svg)

## Reproduce

Start PostgreSQL:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Run the benchmark:

```bash
vecadvisor benchmark-db \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --dataset synthetic \
  --strategies exact,postfilter,iterative,partial,partition \
  --rows 4096 \
  --dim 32 \
  --queries 8 \
  --clusters 8 \
  --filter-selectivity 0.05 \
  --correlation -0.6 \
  --limit 10 \
  --ef-search 40 \
  --max-scan-tuples 1000 \
  --iterative-order relaxed_order \
  --hnsw-m 8 \
  --hnsw-ef-construction 32 \
  --block-rows 512 \
  --seed 610 \
  --out docs/benchmarks/real-pgvector-benchmark.json

vecadvisor plot-benchmark \
  docs/benchmarks/real-pgvector-benchmark.json \
  --out docs/assets/real-pgvector-pareto.svg \
  --title "VecAdvisor real pgvector Pareto"
```

## Validation Sweep

The companion files `real-pgvector-calibration.json`,
`real-pgvector-sweep.json`, and `real-pgvector-crossover.svg` are a smaller
actual-Postgres sweep for local-selectivity validation. In that sweep,
post-filter HNSW misses recall or returns-k targets in all four bins, while
the current calibrated predictor matches the measured winner in two of four
bins. This is useful evidence and also a calibration-improvement target, not a
claim that the model is finished.
