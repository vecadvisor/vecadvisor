from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SyntheticDataset:
    vectors: Any
    filter_mask: Any
    cluster_ids: Any
    centers: Any
    filter_probabilities: tuple[float, ...]
    filter_selectivity: float
    correlation: float
    seed: int
    dataset_id: str = "synthetic"

    @property
    def n_rows(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    @property
    def observed_selectivity(self) -> float:
        return float(self.filter_mask.mean())


@dataclass(frozen=True)
class SyntheticQueries:
    vectors: Any
    cluster_ids: Any
    seed: int
    cluster_policy: str

    @property
    def n_queries(self) -> int:
        return int(self.vectors.shape[0])


def generate_synthetic_dataset(
    *,
    n_rows: int = 10_000,
    dim: int = 64,
    n_clusters: int = 16,
    filter_selectivity: float = 0.1,
    correlation: float = 0.0,
    seed: int = 0,
    cluster_std: float = 0.08,
) -> SyntheticDataset:
    """Generate clustered vectors with a tunable correlated boolean filter."""

    if n_rows <= 0:
        raise ValueError("n_rows must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if not 0.0 < filter_selectivity < 1.0:
        raise ValueError("filter_selectivity must be in (0, 1)")
    if not -1.0 <= correlation <= 1.0:
        raise ValueError("correlation must be in [-1, 1]")
    if cluster_std <= 0.0:
        raise ValueError("cluster_std must be positive")

    np = _numpy()
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 1.0, size=(n_clusters, dim)).astype("float32")
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)

    cluster_ids = rng.integers(0, n_clusters, size=n_rows, dtype="int32")
    vectors = centers[cluster_ids] + rng.normal(0.0, cluster_std, size=(n_rows, dim)).astype(
        "float32"
    )
    probabilities = _cluster_filter_probabilities(
        filter_selectivity=filter_selectivity,
        correlation=correlation,
        n_clusters=n_clusters,
    )
    probability_array = np.asarray(probabilities, dtype="float64")
    filter_mask = rng.random(n_rows) < probability_array[cluster_ids]
    if not bool(filter_mask.any()):
        filter_mask[int(rng.integers(0, n_rows))] = True
    if bool(filter_mask.all()):
        filter_mask[int(rng.integers(0, n_rows))] = False

    return SyntheticDataset(
        vectors=vectors.astype("float32", copy=False),
        filter_mask=filter_mask,
        cluster_ids=cluster_ids,
        centers=centers,
        filter_probabilities=probabilities,
        filter_selectivity=filter_selectivity,
        correlation=correlation,
        seed=seed,
    )


def generate_synthetic_queries(
    dataset: SyntheticDataset,
    *,
    n_queries: int = 100,
    seed: int = 1,
    cluster_policy: str = "uniform",
    query_std: float = 0.04,
) -> SyntheticQueries:
    """Generate query vectors around dataset clusters."""

    if n_queries <= 0:
        raise ValueError("n_queries must be positive")
    if query_std <= 0.0:
        raise ValueError("query_std must be positive")

    np = _numpy()
    rng = np.random.default_rng(seed)
    n_clusters = int(dataset.centers.shape[0])
    if cluster_policy == "uniform":
        probabilities = None
    elif cluster_policy == "filter_hot":
        weights = np.asarray(dataset.filter_probabilities, dtype="float64")
        probabilities = weights / weights.sum()
    elif cluster_policy == "filter_cold":
        weights = 1.0 - np.asarray(dataset.filter_probabilities, dtype="float64")
        probabilities = weights / weights.sum()
    else:
        raise ValueError("cluster_policy must be one of: uniform, filter_hot, filter_cold")

    cluster_ids = rng.choice(n_clusters, size=n_queries, replace=True, p=probabilities).astype(
        "int32"
    )
    vectors = dataset.centers[cluster_ids] + rng.normal(
        0.0,
        query_std,
        size=(n_queries, dataset.dim),
    ).astype("float32")
    return SyntheticQueries(
        vectors=vectors.astype("float32", copy=False),
        cluster_ids=cluster_ids,
        seed=seed,
        cluster_policy=cluster_policy,
    )


def load_file_dataset(
    *,
    vectors_path: Path,
    filter_mask_path: Path,
    dataset_id: str = "file",
) -> SyntheticDataset:
    """Load vectors and a boolean filter mask from .npy files."""

    np = _numpy()
    vectors = np.load(vectors_path, mmap_mode="r")
    if int(vectors.ndim) != 2:
        raise ValueError("vectors .npy must be a 2D array")
    if int(vectors.shape[0]) <= 0 or int(vectors.shape[1]) <= 0:
        raise ValueError("vectors .npy must have at least one row and one dimension")
    if vectors.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("vectors .npy must contain numeric values")

    raw_mask = np.load(filter_mask_path, mmap_mode="r")
    if int(raw_mask.ndim) != 1:
        raise ValueError("filter mask .npy must be a 1D array")
    if int(raw_mask.shape[0]) != int(vectors.shape[0]):
        raise ValueError("filter mask length must match vector row count")
    if raw_mask.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("filter mask .npy must contain boolean or numeric values")

    filter_mask = np.asarray(raw_mask, dtype="bool")
    observed_selectivity = float(filter_mask.mean())
    cluster_ids = np.zeros(int(vectors.shape[0]), dtype="int32")
    centers = np.zeros((1, int(vectors.shape[1])), dtype="float32")
    return SyntheticDataset(
        vectors=vectors,
        filter_mask=filter_mask,
        cluster_ids=cluster_ids,
        centers=centers,
        filter_probabilities=(observed_selectivity,),
        filter_selectivity=observed_selectivity,
        correlation=0.0,
        seed=0,
        dataset_id=dataset_id,
    )


def load_file_queries(
    *,
    query_vectors_path: Path,
    expected_dim: int,
) -> SyntheticQueries:
    """Load benchmark query vectors from a .npy file."""

    np = _numpy()
    vectors = np.load(query_vectors_path, mmap_mode="r")
    if int(vectors.ndim) != 2:
        raise ValueError("query vectors .npy must be a 2D array")
    if int(vectors.shape[0]) <= 0:
        raise ValueError("query vectors .npy must contain at least one query")
    if int(vectors.shape[1]) != expected_dim:
        raise ValueError("query vector dimension must match dataset vector dimension")
    if vectors.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("query vectors .npy must contain numeric values")
    return SyntheticQueries(
        vectors=vectors,
        cluster_ids=np.zeros(int(vectors.shape[0]), dtype="int32"),
        seed=0,
        cluster_policy="file",
    )


def _cluster_filter_probabilities(
    *,
    filter_selectivity: float,
    correlation: float,
    n_clusters: int,
) -> tuple[float, ...]:
    np = _numpy()
    if abs(correlation) < 1e-12:
        return tuple(float(filter_selectivity) for _ in range(n_clusters))

    scores = np.linspace(1.0, -1.0, n_clusters, dtype="float64")
    if correlation < 0.0:
        scores = -scores
    weights = np.exp(abs(correlation) * 4.0 * scores)
    low = 0.0
    high = max(1.0, filter_selectivity / float(weights.mean()))
    while float(np.minimum(1.0, high * weights).mean()) < filter_selectivity:
        high *= 2.0
    for _ in range(64):
        mid = (low + high) / 2.0
        mean = float(np.minimum(1.0, mid * weights).mean())
        if mean < filter_selectivity:
            low = mid
        else:
            high = mid
    probabilities = np.minimum(1.0, high * weights)
    return tuple(float(value) for value in probabilities)


def _numpy() -> Any:
    return importlib.import_module("numpy")
