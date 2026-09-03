from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vecadvisor.native_distance import NATIVE_DISTANCE_LIB_ENV

from .datasets import SyntheticDataset, SyntheticQueries
from .groundtruth import ExactTopKResult, exact_topk, recall_at_k
from .runner import ground_truth_to_json


@dataclass(frozen=True)
class TimedExactTopKRun:
    backend: str
    native_requested: bool
    result: ExactTopKResult
    iterations: int
    warmup_iterations: int
    elapsed_ms_total: float
    elapsed_ms_mean: float
    elapsed_ms_min: float


@dataclass(frozen=True)
class GroundTruthComparisonReport:
    dataset: dict[str, object]
    k: int
    metric: str
    block_rows: int | None
    max_distance_matrix_bytes: int
    warmup_iterations: int
    numpy: TimedExactTopKRun
    native: TimedExactTopKRun
    recall_at_k: float
    exact_index_match: bool
    max_abs_distance_delta: float | None
    speedup: float | None
    notes: tuple[str, ...]


def run_groundtruth_comparison(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str = "l2",
    block_rows: int | None = None,
    max_distance_matrix_bytes: int = 256 * 1024 * 1024,
    iterations: int = 3,
    warmup_iterations: int = 1,
    native_library: Path | None = None,
    require_native: bool = False,
) -> GroundTruthComparisonReport:
    """Compare NumPy exact top-k with the optional native-backed exact top-k path."""

    if k <= 0:
        raise ValueError("k must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if max_distance_matrix_bytes <= 0:
        raise ValueError("max_distance_matrix_bytes must be positive")

    numpy_run = _time_exact_topk(
        backend="numpy",
        dataset=dataset,
        queries=queries,
        k=k,
        metric=metric,
        block_rows=block_rows,
        max_distance_matrix_bytes=max_distance_matrix_bytes,
        use_native=False,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
    )
    with _configured_native_library(native_library):
        native_run = _time_exact_topk(
            backend="native",
            dataset=dataset,
            queries=queries,
            k=k,
            metric=metric,
            block_rows=block_rows,
            max_distance_matrix_bytes=max_distance_matrix_bytes,
            use_native=True,
            iterations=iterations,
            warmup_iterations=warmup_iterations,
        )

    if require_native and not native_run.result.native_used:
        raise ValueError(
            "native distance library was required but was not used; run "
            "`vecadvisor native-info` or set VECADVISOR_NATIVE_DISTANCE_LIB"
        )

    recall = recall_at_k(numpy_run.result.indices, native_run.result.indices, k=k)
    speedup = (
        numpy_run.elapsed_ms_mean / native_run.elapsed_ms_mean
        if native_run.result.native_used and native_run.elapsed_ms_mean > 0.0
        else None
    )
    notes = _comparison_notes(native_run)
    return GroundTruthComparisonReport(
        dataset=_dataset_to_json(dataset, queries),
        k=k,
        metric=metric,
        block_rows=block_rows,
        max_distance_matrix_bytes=max_distance_matrix_bytes,
        warmup_iterations=warmup_iterations,
        numpy=numpy_run,
        native=native_run,
        recall_at_k=recall.mean,
        exact_index_match=_indices_match(numpy_run.result.indices, native_run.result.indices),
        max_abs_distance_delta=_max_abs_distance_delta(
            numpy_run.result.distances,
            native_run.result.distances,
        ),
        speedup=speedup,
        notes=notes,
    )


def groundtruth_comparison_to_json(report: GroundTruthComparisonReport) -> dict[str, object]:
    return {
        "dataset": report.dataset,
        "k": report.k,
        "metric": report.metric,
        "block_rows": report.block_rows,
        "max_distance_matrix_bytes": report.max_distance_matrix_bytes,
        "warmup_iterations": report.warmup_iterations,
        "numpy": _timed_run_to_json(report.numpy),
        "native": _timed_run_to_json(report.native),
        "comparison": {
            "recall_at_k": report.recall_at_k,
            "exact_index_match": report.exact_index_match,
            "max_abs_distance_delta": report.max_abs_distance_delta,
            "speedup": report.speedup,
        },
        "notes": list(report.notes),
    }


def write_groundtruth_comparison_report(
    report: GroundTruthComparisonReport,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(groundtruth_comparison_to_json(report), indent=2) + "\n",
        encoding="utf-8",
    )


def _time_exact_topk(
    *,
    backend: str,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str,
    block_rows: int | None,
    max_distance_matrix_bytes: int,
    use_native: bool,
    iterations: int,
    warmup_iterations: int,
) -> TimedExactTopKRun:
    elapsed_ms: list[float] = []
    result: ExactTopKResult | None = None
    for _ in range(warmup_iterations):
        result = exact_topk(
            dataset.vectors,
            queries.vectors,
            k=k,
            metric=metric,
            filter_mask=dataset.filter_mask,
            block_rows=block_rows,
            max_distance_matrix_bytes=max_distance_matrix_bytes,
            use_native=use_native,
        )

    for _ in range(iterations):
        started = time.perf_counter()
        result = exact_topk(
            dataset.vectors,
            queries.vectors,
            k=k,
            metric=metric,
            filter_mask=dataset.filter_mask,
            block_rows=block_rows,
            max_distance_matrix_bytes=max_distance_matrix_bytes,
            use_native=use_native,
        )
        elapsed_ms.append((time.perf_counter() - started) * 1000.0)

    assert result is not None
    total = float(sum(elapsed_ms))
    return TimedExactTopKRun(
        backend=backend,
        native_requested=use_native,
        result=result,
        iterations=iterations,
        warmup_iterations=warmup_iterations,
        elapsed_ms_total=total,
        elapsed_ms_mean=total / iterations,
        elapsed_ms_min=float(min(elapsed_ms)),
    )


@contextmanager
def _configured_native_library(native_library: Path | None) -> Iterator[None]:
    from vecadvisor.native_distance import load_default_native_distance_library

    original = os.environ.get(NATIVE_DISTANCE_LIB_ENV)
    if native_library is not None:
        os.environ[NATIVE_DISTANCE_LIB_ENV] = str(native_library)
    load_default_native_distance_library.cache_clear()
    try:
        yield
    finally:
        if native_library is not None:
            if original is None:
                os.environ.pop(NATIVE_DISTANCE_LIB_ENV, None)
            else:
                os.environ[NATIVE_DISTANCE_LIB_ENV] = original
        load_default_native_distance_library.cache_clear()


def _dataset_to_json(dataset: SyntheticDataset, queries: SyntheticQueries) -> dict[str, object]:
    return {
        "id": dataset.dataset_id,
        "rows": dataset.n_rows,
        "dim": dataset.dim,
        "queries": queries.n_queries,
        "clusters": len(dataset.filter_probabilities),
        "query_policy": queries.cluster_policy,
        "target_filter_selectivity": dataset.filter_selectivity,
        "observed_filter_selectivity": dataset.observed_selectivity,
        "correlation": dataset.correlation,
        "dataset_seed": dataset.seed,
        "query_seed": queries.seed,
    }


def _timed_run_to_json(run: TimedExactTopKRun) -> dict[str, object]:
    return {
        "backend": run.backend,
        "native_requested": run.native_requested,
        "iterations": run.iterations,
        "warmup_iterations": run.warmup_iterations,
        "elapsed_ms_total": run.elapsed_ms_total,
        "elapsed_ms_mean": run.elapsed_ms_mean,
        "elapsed_ms_min": run.elapsed_ms_min,
        "ground_truth": ground_truth_to_json(run.result),
    }


def _comparison_notes(native_run: TimedExactTopKRun) -> tuple[str, ...]:
    if native_run.result.native_used:
        return (
            "native run used vecadvisor_distance_topk for at least one block",
            "both paths use blocked exact search and avoid materializing an N x Q matrix",
        )
    return (
        "native distance library unavailable or rejected the workload; native-requested run "
        "fell back to NumPy",
        "use --require-native to fail when the native path is not active",
    )


def _indices_match(left: Any, right: Any) -> bool:
    np = _numpy()
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _max_abs_distance_delta(left: Any, right: Any) -> float | None:
    np = _numpy()
    left_distances = np.asarray(left, dtype="float64")
    right_distances = np.asarray(right, dtype="float64")
    if left_distances.shape != right_distances.shape:
        return None

    left_finite = np.isfinite(left_distances)
    right_finite = np.isfinite(right_distances)
    if not bool(np.array_equal(left_finite, right_finite)):
        return None
    if not bool(left_finite.any()):
        return 0.0
    delta = np.abs(left_distances[left_finite] - right_distances[right_finite])
    return float(delta.max()) if int(delta.shape[0]) > 0 else 0.0


def _numpy() -> Any:
    import importlib

    return importlib.import_module("numpy")
