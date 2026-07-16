# Real-Embedding Scale Benchmark

This page is the reproducible path for the first large public benchmark
artifact. It uses the ANN-Benchmarks SIFT dataset (`sift-128-euclidean.hdf5`),
which contains one million 128-dimensional image-descriptor vectors. The data
file is not committed to this repository.

The benchmark keeps the vectors real and makes the scalar filter deterministic
from the vector space. Two filter modes are useful:

- `random_projection_top_tail`: projects each vector onto a seeded random
  direction and marks the top selectivity tail as `passes_filter`. This gives
  a deterministic vector-derived scalar predicate.
- `query_anticorrelated_band`: excludes each benchmark query's immediate
  top-N neighborhood, then selects the nearest remaining rows. This keeps
  global selectivity fixed while making local selectivity low at the fixed
  HNSW frontier.

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

For the recall-collapse artifact, use the anti-correlated mode:

```bash
python tools/prepare_ann_benchmark_dataset.py \
  --dataset-url https://ann-benchmarks.com/sift-128-euclidean.hdf5 \
  --out-dir data/sift1m-anticorrelated \
  --rows 1000000 \
  --queries 16 \
  --filter-selectivity 0.05 \
  --filter-mode query_anticorrelated_band \
  --anti-start-rank 40 \
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
  --maintenance-work-mem 2GB \
  --statement-timeout-ms 14400000 \
  --out docs/benchmarks/sift1m-pgvector-benchmark.json
```

Render a chart:

```bash
vecadvisor plot-benchmark \
  docs/benchmarks/sift1m-pgvector-benchmark.json \
  --out docs/assets/sift1m-pgvector-pareto.svg \
  --title "VecAdvisor SIFT1M pgvector Pareto"
```

## Published Artifact

The committed run is summarized in
[`sift1m-pgvector-benchmark.md`](sift1m-pgvector-benchmark.md), with machine
and PostgreSQL details recorded alongside:

- machine CPU and memory,
- Docker/PostgreSQL/pgvector versions,
- row count, dimensions, queries, selectivity, and seed,
- exact command lines,
- whether every strategy completed under the timeout,
- recall@k, returns-k rate, and p95 latency.

The committed anti-correlated run is summarized in
[`sift1m-anticorrelated-pgvector-benchmark.md`](sift1m-anticorrelated-pgvector-benchmark.md).
That artifact demonstrates the local-selectivity failure mode at SIFT1M scale:
fixed-frontier postfilter reaches `0.3438` recall@k and returns full `k` for
only `25%` of queries, while iterative HNSW recovers `1.0000` recall@k and
full `k`.

The projection-tail run also shows the risk: fixed-frontier postfilter reaches
`0.1750` recall@k and returns full `k` for only `12.5%` of queries, while
iterative HNSW recovers `0.9875` recall@k and full `k`.

After reproducing the benchmark, remove the ignored `data/` directory so the
downloaded HDF5 file, converted `.npy` files, and transient logs do not keep
consuming local disk space.
