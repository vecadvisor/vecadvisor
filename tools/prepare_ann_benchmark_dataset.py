from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

DEFAULT_DATASET_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
DEFAULT_DATASET_NAME = "sift-128-euclidean"
CHUNK_BYTES = 16 * 1024 * 1024

h5py: Any
np: Any


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_optional_dependencies()
    hdf5_path = _resolve_hdf5_path(args)
    if args.dataset_url and not hdf5_path.exists():
        _download(args.dataset_url, hdf5_path)
    if not hdf5_path.exists():
        raise SystemExit(f"HDF5 dataset not found: {hdf5_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = args.out_dir / "vectors.npy"
    filter_mask_path = args.out_dir / "filter_mask.npy"
    query_vectors_path = args.out_dir / "query_vectors.npy"
    manifest_path = args.out_dir / "manifest.json"
    for path in (vectors_path, filter_mask_path, query_vectors_path, manifest_path):
        if path.exists() and not args.force:
            raise SystemExit(f"{path} already exists; pass --force to replace it")

    with h5py.File(hdf5_path, "r") as hdf5_file:
        train = hdf5_file[args.train_key]
        test = hdf5_file[args.test_key]
        rows = min(args.rows, int(train.shape[0]))
        queries = min(args.queries, int(test.shape[0]))
        dim = int(train.shape[1])
        _copy_matrix_to_npy(train, vectors_path, rows=rows, chunk_rows=args.chunk_rows)
        filter_mask = _write_projection_filter(
            train,
            filter_mask_path,
            rows=rows,
            selectivity=args.filter_selectivity,
            seed=args.seed,
            chunk_rows=args.chunk_rows,
        )
        _copy_matrix_to_npy(test, query_vectors_path, rows=queries, chunk_rows=args.chunk_rows)

    manifest = {
        "dataset": args.dataset_name,
        "dataset_url": args.dataset_url,
        "source_hdf5": str(hdf5_path),
        "vectors": str(vectors_path),
        "filter_mask": str(filter_mask_path),
        "query_vectors": str(query_vectors_path),
        "rows": rows,
        "dimensions": dim,
        "queries": queries,
        "filter_mode": "random_projection_top_tail",
        "filter_selectivity_target": args.filter_selectivity,
        "filter_selectivity_observed": float(filter_mask.mean()),
        "seed": args.seed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


def _load_optional_dependencies() -> None:
    global h5py, np
    try:
        import h5py as h5py_module
        import numpy as np_module
    except ImportError as exc:  # pragma: no cover - exercised by users without h5py
        raise SystemExit(
            "This tool requires numpy and h5py. Install with: python -m pip install numpy h5py"
        ) from exc
    h5py = h5py_module
    np = np_module


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an ANN-Benchmarks HDF5 dataset for VecAdvisor file benchmarks.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--hdf5", type=Path, default=None, help="Existing HDF5 file.")
    parser.add_argument("--download-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/sift1m"))
    parser.add_argument("--train-key", default="train")
    parser.add_argument("--test-key", default="test")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--filter-selectivity", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--chunk-rows", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.rows <= 0:
        parser.error("--rows must be positive")
    if args.queries <= 0:
        parser.error("--queries must be positive")
    if not 0.0 < args.filter_selectivity < 1.0:
        parser.error("--filter-selectivity must be in (0, 1)")
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    return args


def _resolve_hdf5_path(args: argparse.Namespace) -> Path:
    if args.hdf5 is not None:
        return args.hdf5
    args.download_dir.mkdir(parents=True, exist_ok=True)
    return args.download_dir / f"{args.dataset_name}.hdf5"


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": "VecAdvisor benchmark prep"})
    with urlopen(request) as response, destination.open("wb") as output:
        while True:
            chunk = response.read(CHUNK_BYTES)
            if not chunk:
                break
            output.write(chunk)


def _copy_matrix_to_npy(
    matrix: Any,
    path: Path,
    *,
    rows: int,
    chunk_rows: int,
) -> None:
    dim = int(matrix.shape[1])
    output = np.lib.format.open_memmap(path, mode="w+", dtype="float32", shape=(rows, dim))
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        output[start:end] = np.asarray(matrix[start:end], dtype="float32")
    output.flush()


def _write_projection_filter(
    matrix: Any,
    path: Path,
    *,
    rows: int,
    selectivity: float,
    seed: int,
    chunk_rows: int,
) -> Any:
    dim = int(matrix.shape[1])
    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0, size=dim).astype("float32")
    projection /= max(float(np.linalg.norm(projection)), 1e-12)
    scores = np.empty(rows, dtype="float32")
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        block = np.asarray(matrix[start:end], dtype="float32")
        scores[start:end] = block @ projection

    target_count = max(1, min(rows - 1, round(rows * selectivity)))
    top_indices = np.argpartition(scores, rows - target_count)[rows - target_count :]
    mask = np.zeros(rows, dtype="bool")
    mask[top_indices] = True
    np.save(path, mask, allow_pickle=False)
    return mask


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
