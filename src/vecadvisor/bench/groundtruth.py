from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

MiB = 1024 * 1024
DEFAULT_DISTANCE_MATRIX_BUDGET_BYTES = 256 * MiB
SUPPORTED_METRICS = {"l2", "ip", "cosine"}


@dataclass(frozen=True)
class ExactTopKResult:
    indices: Any
    distances: Any
    metric: str
    k: int
    candidate_count: int
    block_rows: int
    blocks_scanned: int


@dataclass(frozen=True)
class RecallAtK:
    mean: float
    per_query: tuple[float, ...]
    k: int


def exact_topk(
    base_vectors: Any,
    query_vectors: Any,
    *,
    k: int,
    metric: str = "l2",
    filter_mask: Any | None = None,
    block_rows: int | None = None,
    max_distance_matrix_bytes: int = DEFAULT_DISTANCE_MATRIX_BUDGET_BYTES,
) -> ExactTopKResult:
    """Compute filtered exact kNN without materializing an N x Q distance matrix."""

    if k <= 0:
        raise ValueError("k must be positive")
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of: {', '.join(sorted(SUPPORTED_METRICS))}")

    np = _numpy()
    base = _as_2d_float32(np, base_vectors, name="base_vectors")
    queries = _as_2d_float32(np, query_vectors, name="query_vectors")
    if int(base.shape[1]) != int(queries.shape[1]):
        raise ValueError("base and query vectors must have the same dimension")

    mask = _coerce_filter_mask(np, filter_mask, int(base.shape[0]))
    candidate_count = int(mask.sum()) if mask is not None else int(base.shape[0])
    effective_block_rows = (
        block_rows
        if block_rows is not None
        else max_block_rows_for_memory(
            dim=int(base.shape[1]),
            bytes_budget=max_distance_matrix_bytes,
        )
    )
    if effective_block_rows <= 0:
        raise ValueError("block_rows must be positive")

    out_indices = np.full((int(queries.shape[0]), k), -1, dtype="int64")
    out_distances = np.full((int(queries.shape[0]), k), np.inf, dtype="float64")
    blocks_scanned = math.ceil(int(base.shape[0]) / effective_block_rows)

    for query_index in range(int(queries.shape[0])):
        best_scores = np.full(k, np.inf, dtype="float64")
        best_indices = np.full(k, -1, dtype="int64")
        query = queries[query_index]
        for start in range(0, int(base.shape[0]), effective_block_rows):
            end = min(start + effective_block_rows, int(base.shape[0]))
            block = base[start:end]
            block_indices = np.arange(start, end, dtype="int64")
            if mask is not None:
                local_mask = mask[start:end]
                if not bool(local_mask.any()):
                    continue
                block = block[local_mask]
                block_indices = block_indices[local_mask]
            scores = _distance_scores(np, block, query, metric=metric)
            if int(scores.shape[0]) == 0:
                continue
            take = min(k, int(scores.shape[0]))
            local_top = _topk_unsorted(np, scores, take)
            merged_scores = np.concatenate((best_scores, scores[local_top]))
            merged_indices = np.concatenate((best_indices, block_indices[local_top]))
            take_merged = min(k, int(merged_scores.shape[0]))
            merged_top = _topk_unsorted(np, merged_scores, take_merged)
            order = np.argsort(merged_scores[merged_top], kind="stable")
            chosen = merged_top[order]
            best_scores = merged_scores[chosen]
            best_indices = merged_indices[chosen]

        valid = best_indices >= 0
        out_indices[query_index, : int(valid.sum())] = best_indices[valid]
        distances = _scores_to_distances(np, best_scores[valid], metric=metric)
        out_distances[query_index, : int(valid.sum())] = distances

    return ExactTopKResult(
        indices=out_indices,
        distances=out_distances,
        metric=metric,
        k=k,
        candidate_count=candidate_count,
        block_rows=effective_block_rows,
        blocks_scanned=blocks_scanned,
    )


def recall_at_k(
    truth_indices: Any,
    candidate_indices: Any,
    *,
    k: int | None = None,
) -> RecallAtK:
    """Compute mean recall@k against exact index ids, ignoring padded -1 entries."""

    np = _numpy()
    truth = np.asarray(truth_indices)
    candidate = np.asarray(candidate_indices)
    if truth.ndim != 2 or candidate.ndim != 2:
        raise ValueError("truth_indices and candidate_indices must be two-dimensional")
    if int(truth.shape[0]) != int(candidate.shape[0]):
        raise ValueError("truth and candidate query counts must match")
    effective_k = int(k if k is not None else min(int(truth.shape[1]), int(candidate.shape[1])))
    if effective_k <= 0:
        raise ValueError("k must be positive")

    per_query: list[float] = []
    for row_index in range(int(truth.shape[0])):
        truth_set = {
            int(value)
            for value in truth[row_index, :effective_k].tolist()
            if int(value) >= 0
        }
        if not truth_set:
            per_query.append(1.0)
            continue
        candidate_set = {
            int(value)
            for value in candidate[row_index, :effective_k].tolist()
            if int(value) >= 0
        }
        per_query.append(len(truth_set & candidate_set) / len(truth_set))
    mean = float(sum(per_query) / len(per_query)) if per_query else 0.0
    return RecallAtK(mean=mean, per_query=tuple(per_query), k=effective_k)


def max_block_rows_for_memory(
    *,
    dim: int,
    bytes_budget: int,
    dtype_bytes: int = 4,
) -> int:
    """Choose base-vector rows per block from a memory budget."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    if bytes_budget <= 0:
        raise ValueError("bytes_budget must be positive")
    if dtype_bytes <= 0:
        raise ValueError("dtype_bytes must be positive")
    return max(1, bytes_budget // max(dtype_bytes * dim, 1))


def _topk_unsorted(np: Any, scores: Any, k: int) -> Any:
    if k <= 0:
        raise ValueError("k must be positive")
    if k >= int(scores.shape[0]):
        return np.arange(int(scores.shape[0]))
    return np.argpartition(scores, k - 1)[:k]


def _distance_scores(np: Any, block: Any, query: Any, *, metric: str) -> Any:
    if metric == "l2":
        delta = block - query
        return np.einsum("ij,ij->i", delta, delta, optimize=True).astype("float64", copy=False)
    if metric == "ip":
        return -(block @ query).astype("float64", copy=False)
    if metric == "cosine":
        numerator = block @ query
        block_norm = np.maximum(np.linalg.norm(block, axis=1), 1e-12)
        query_norm = max(float(np.linalg.norm(query)), 1e-12)
        return (1.0 - numerator / (block_norm * query_norm)).astype("float64", copy=False)
    raise ValueError(f"unsupported metric: {metric}")


def _scores_to_distances(np: Any, scores: Any, *, metric: str) -> Any:
    if metric == "l2":
        return np.sqrt(np.maximum(scores, 0.0))
    return scores


def _as_2d_float32(np: Any, value: Any, *, name: str) -> Any:
    array = np.asarray(value, dtype="float32")
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a one- or two-dimensional vector array")
    if int(array.shape[0]) == 0 or int(array.shape[1]) == 0:
        raise ValueError(f"{name} must be non-empty")
    return array


def _coerce_filter_mask(np: Any, filter_mask: Any | None, n_rows: int) -> Any | None:
    if filter_mask is None:
        return None
    mask = np.asarray(filter_mask, dtype=bool)
    if mask.ndim != 1 or int(mask.shape[0]) != n_rows:
        raise ValueError("filter_mask must be a one-dimensional boolean mask matching rows")
    return mask


def _numpy() -> Any:
    return importlib.import_module("numpy")
