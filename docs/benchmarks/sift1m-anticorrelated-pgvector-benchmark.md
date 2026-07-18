# SIFT1M Anti-Correlated pgvector Benchmark Artifact

This artifact measures actual PostgreSQL/pgvector SQL on one million real
SIFT vectors prepared from the ANN-Benchmarks `sift-128-euclidean` dataset.
The scalar filter is intentionally anti-correlated with each query's immediate
vector neighborhood: every query has zero passing rows in its exact top-40
neighbors, while the global selectivity remains `5%`.

The purpose is to make the filtered-ANN failure mode visible at SIFT1M scale.
Global selectivity alone says `5%`; local selectivity at the default frontier
says `0%`.

## Environment

- Machine: Intel Core i7-8665U, 4 cores / 8 logical processors
- Docker Desktop engine: 8 CPUs, 7.66 GiB memory limit
- Container image: `pgvector/pgvector:pg17`
- PostgreSQL: `17.10`
- pgvector: `0.8.4`
- Dataset: SIFT1M real image-descriptor vectors
- Rows: `1,000,000`
- Dimensions: `128`
- Query vectors: `16`
- Filter: query anti-correlated band
- Observed global selectivity: `0.05`
- Exact top-40 local passing rows: `0` for all `16` queries
- Exact top-5,000 local passing rows: at least `74` for every query
- `k`: `10`
- `ef_search`: `40`
- `hnsw.m`: `16`
- `hnsw.ef_construction`: `64`
- `hnsw.max_scan_tuples`: `5000`
- `maintenance_work_mem`: `2GB`
- Wall-clock benchmark time: `15.81 min`

## Result Summary

| Strategy | Recall@k | Returns-k rate | Mean latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| exact | `1.0000` | `1.0000` | `301.30 ms` | `340.59 ms` |
| postfilter | `0.3438` | `0.2500` | `109.18 ms` | `171.22 ms` |
| iterative | `1.0000` | `1.0000` | `269.95 ms` | `324.56 ms` |
| partial | `0.8563` | `1.0000` | `29.55 ms` | `32.64 ms` |

![SIFT1M anti-correlated quality chart](../assets/sift1m-anticorrelated-quality-bars.svg)

![SIFT1M anti-correlated pgvector Pareto chart](../assets/sift1m-anticorrelated-pgvector-pareto.svg)

Fixed-frontier postfiltering is fast but unsafe here: it misses `65.6`
percentage points of recall versus iterative HNSW and returns fewer than `k`
rows for `75%` of queries. Iterative HNSW recovers exact recall and full
`k` while staying slightly faster than filtered exact scan on this machine.

The partial HNSW index is much faster, but with `ef_search=40` it reaches only
`0.8563` recall@k. That is useful evidence too: a durable filtered index is
not automatically enough unless its own recall settings are calibrated.

## Reproduce

Prepare the SIFT1M files:

```bash
python -m pip install h5py

python tools/prepare_ann_benchmark_dataset.py \
  --dataset-url https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  --out-dir data/sift1m-anticorrelated \
  --rows 1000000 \
  --queries 16 \
  --filter-selectivity 0.05 \
  --filter-mode query_anticorrelated_band \
  --anti-start-rank 40 \
  --seed 20260714 \
  --force
```

Run the benchmark:

```bash
vecadvisor benchmark-db \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --dataset file \
  --vectors data/sift1m-anticorrelated/vectors.npy \
  --filter-mask data/sift1m-anticorrelated/filter_mask.npy \
  --query-vectors data/sift1m-anticorrelated/query_vectors.npy \
  --strategies exact,postfilter,iterative,partial \
  --limit 10 \
  --metric l2 \
  --ef-search 40 \
  --max-scan-tuples 5000 \
  --iterative-order relaxed_order \
  --hnsw-m 16 \
  --hnsw-ef-construction 64 \
  --block-rows 8192 \
  --maintenance-work-mem 2GB \
  --statement-timeout-ms 28800000 \
  --out docs/benchmarks/sift1m-anticorrelated-pgvector-benchmark.json
```

Render the chart:

```bash
vecadvisor plot-quality-bars \
  docs/benchmarks/sift1m-anticorrelated-pgvector-benchmark.json \
  --out docs/assets/sift1m-anticorrelated-quality-bars.svg \
  --title "SIFT1M filtered pgvector quality" \
  --subtitle "global selectivity 5%; exact top-40 local selectivity 0% across 16 queries"

vecadvisor plot-benchmark \
  docs/benchmarks/sift1m-anticorrelated-pgvector-benchmark.json \
  --out docs/assets/sift1m-anticorrelated-pgvector-pareto.svg \
  --title "VecAdvisor SIFT1M anti-correlated pgvector Pareto" \
  --subtitle "global selectivity 5%; exact top-40 local selectivity 0% across 16 queries"
```

After reproducing locally, remove the ignored data files to reclaim space:

```powershell
Remove-Item -Recurse -Force data
```
