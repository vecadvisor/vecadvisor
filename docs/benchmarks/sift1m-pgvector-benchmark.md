# SIFT1M pgvector Benchmark Artifact

This artifact measures actual PostgreSQL/pgvector SQL on one million real
SIFT vectors prepared from the ANN-Benchmarks `sift-128-euclidean` dataset.
The source HDF5 and converted `.npy` files are not committed.

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
- Filter: seeded random-projection top tail
- Observed global selectivity: `0.05`
- `k`: `10`
- `ef_search`: `40`
- `hnsw.m`: `16`
- `hnsw.ef_construction`: `64`
- `hnsw.max_scan_tuples`: `5000`
- `maintenance_work_mem`: `2GB`

## Result Summary

| Strategy | Recall@k | Returns-k rate | Mean latency | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| exact | `1.0000` | `1.0000` | `301.37 ms` | `353.81 ms` |
| postfilter | `0.9875` | `1.0000` | `311.69 ms` | `386.78 ms` |
| iterative | `1.0000` | `1.0000` | `285.55 ms` | `332.19 ms` |

![SIFT1M pgvector Pareto chart](../assets/sift1m-pgvector-pareto.svg)

This run is scale evidence, not the recall-collapse headline. On this
projection-filtered SIFT workload, fixed post-filter HNSW did not collapse:
it reached `0.9875` recall@k and returned full `k` for every query.
Iterative HNSW was still the best measured strategy, with exact recall and
the lowest p95 latency among the three measured strategies.

The anti-correlated small pgvector benchmark remains the clearest recall-risk
artifact because it intentionally places the scalar filter against the local
vector neighborhoods.

## Reproduce

Prepare the SIFT1M files:

```bash
python -m pip install h5py

python tools/prepare_ann_benchmark_dataset.py \
  --dataset-url https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  --out-dir data/sift1m \
  --rows 1000000 \
  --queries 16 \
  --filter-selectivity 0.05 \
  --seed 20260714 \
  --force
```

Run the benchmark:

```bash
vecadvisor benchmark-db \
  --dsn postgresql://postgres:postgres@localhost:5432/vecadvisor \
  --dataset file \
  --vectors data/sift1m/vectors.npy \
  --filter-mask data/sift1m/filter_mask.npy \
  --query-vectors data/sift1m/query_vectors.npy \
  --strategies exact,postfilter,iterative \
  --limit 10 \
  --metric l2 \
  --ef-search 40 \
  --max-scan-tuples 5000 \
  --iterative-order relaxed_order \
  --hnsw-m 16 \
  --hnsw-ef-construction 64 \
  --block-rows 8192 \
  --maintenance-work-mem 2GB \
  --statement-timeout-ms 14400000 \
  --out docs/benchmarks/sift1m-pgvector-benchmark.json
```

Render the chart:

```bash
vecadvisor plot-benchmark \
  docs/benchmarks/sift1m-pgvector-benchmark.json \
  --out docs/assets/sift1m-pgvector-pareto.svg \
  --title "VecAdvisor SIFT1M pgvector Pareto"
```

After reproducing locally, remove the ignored data files to reclaim space:

```powershell
Remove-Item -Recurse -Force data
```
