# Real-Embedding Scale Benchmark

This page is the reproducible path for the first large public benchmark
artifact. It uses the ANN-Benchmarks SIFT dataset (`sift-128-euclidean.hdf5`),
which contains one million 128-dimensional image-descriptor vectors. The data
file is not committed to this repository.

The benchmark keeps the vectors real and makes the scalar filter deterministic
from the vector space: VecAdvisor projects each vector onto a seeded random
direction and marks the top selectivity tail as `passes_filter`. That gives a
real-embedding workload with a known correlated scalar predicate.

## Prepare Dataset Files

Install the optional HDF5 reader:

```bash
python -m pip install h5py
```

Download and convert SIFT1M into `.npy` files:

```bash
python tools/prepare_ann_benchmark_dataset.py \
  --dataset-url https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  --out-dir data/sift1m \
  --rows 1000000 \
  --queries 16 \
  --filter-selectivity 0.05 \
  --seed 20260714
```

The tool writes:

- `data/sift1m/vectors.npy`
- `data/sift1m/filter_mask.npy`
- `data/sift1m/query_vectors.npy`
- `data/sift1m/manifest.json`

## Run PostgreSQL/pgvector

Start the bundled database:

```bash
docker compose -f docker/docker-compose.yml up -d
```

Run the measured benchmark:

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
  --statement-timeout-ms 600000 \
  --out docs/benchmarks/sift1m-pgvector-benchmark.json
```

Render a chart:

```bash
vecadvisor plot-benchmark \
  docs/benchmarks/sift1m-pgvector-benchmark.json \
  --out docs/assets/sift1m-pgvector-pareto.svg \
  --title "VecAdvisor SIFT1M pgvector Pareto"
```

## Publication Criteria

Commit the generated JSON and SVG only after recording:

- machine CPU and memory,
- Docker/PostgreSQL/pgvector versions,
- row count, dimensions, queries, selectivity, and seed,
- exact command lines,
- whether every strategy completed under the timeout,
- recall@k, returns-k rate, and p95 latency.

Until that file exists in this directory, the repository has a reproducible
large-scale path but not a committed large-scale benchmark result.
