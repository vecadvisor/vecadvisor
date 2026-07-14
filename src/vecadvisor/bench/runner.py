from __future__ import annotations

import csv
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .datasets import SyntheticDataset, SyntheticQueries
from .groundtruth import ExactTopKResult, exact_topk, recall_at_k

STRATEGY_EXACT = "exact"
STRATEGY_POSTFILTER = "postfilter"
STRATEGY_ITERATIVE = "iterative"
STRATEGY_PARTIAL = "partial"
STRATEGY_PARTITION = "partition"
DEFAULT_STRATEGIES = (STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE)
OUTPUT_FORMATS = {"json", "csv"}


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    params: dict[str, object]
    query_count: int
    recall_at_k: float
    returns_k_rate: float
    result_count_mean: float
    latency_ms_total: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class BenchmarkReport:
    dataset: dict[str, object]
    ground_truth: dict[str, object]
    strategies: tuple[StrategyMetrics, ...]
    elapsed_ms: float
    notes: tuple[str, ...] = ()


def parse_strategy_list(strategies: str | None) -> tuple[str, ...]:
    if strategies is None or not strategies.strip() or strategies.strip().lower() == "all":
        return DEFAULT_STRATEGIES
    parsed = tuple(item.strip().lower() for item in strategies.split(",") if item.strip())
    if not parsed:
        raise ValueError("at least one benchmark strategy is required")
    unknown = sorted(set(parsed) - set(DEFAULT_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown benchmark strategy: {', '.join(unknown)}")
    return parsed


def run_synthetic_benchmark(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str = "l2",
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
    ef_search: int = 40,
    max_scan_tuples: int = 1_000,
    block_rows: int | None = None,
) -> BenchmarkReport:
    """Run a synthetic filtered-vector benchmark with exact ground truth."""

    if k <= 0:
        raise ValueError("k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if max_scan_tuples <= 0:
        raise ValueError("max_scan_tuples must be positive")

    started = time.perf_counter()
    truth, exact_latencies = _compute_ground_truth(
        dataset=dataset,
        queries=queries,
        k=k,
        metric=metric,
        block_rows=block_rows,
    )
    strategy_metrics: list[StrategyMetrics] = []
    for strategy in strategies:
        if strategy == STRATEGY_EXACT:
            strategy_metrics.append(
                _metrics_from_indices(
                    strategy=STRATEGY_EXACT,
                    params={"mode": "filtered_exact"},
                    truth_indices=truth.indices,
                    candidate_indices=truth.indices,
                    latencies_ms=exact_latencies,
                    k=k,
                    notes=("filtered exact search is the recall baseline",),
                )
            )
        elif strategy == STRATEGY_POSTFILTER:
            candidate_indices, latencies = _run_postfilter(
                dataset=dataset,
                queries=queries,
                k=k,
                metric=metric,
                ef_search=ef_search,
                block_rows=block_rows,
            )
            strategy_metrics.append(
                _metrics_from_indices(
                    strategy=STRATEGY_POSTFILTER,
                    params={"ef_search": ef_search, "mode": "exact_candidate_pool"},
                    truth_indices=truth.indices,
                    candidate_indices=candidate_indices,
                    latencies_ms=latencies,
                    k=k,
                    notes=(
                        "simulates post-filter semantics with exact unfiltered candidates",
                        "latency is CPU exact-search time, not pgvector HNSW latency",
                    ),
                )
            )
        elif strategy == STRATEGY_ITERATIVE:
            scan_limit = min(max_scan_tuples, dataset.n_rows)
            candidate_indices, latencies = _run_iterative(
                dataset=dataset,
                queries=queries,
                k=k,
                metric=metric,
                max_scan_tuples=scan_limit,
                block_rows=block_rows,
            )
            strategy_metrics.append(
                _metrics_from_indices(
                    strategy=STRATEGY_ITERATIVE,
                    params={"max_scan_tuples": scan_limit, "mode": "exact_candidate_expansion"},
                    truth_indices=truth.indices,
                    candidate_indices=candidate_indices,
                    latencies_ms=latencies,
                    k=k,
                    notes=(
                        "simulates iterative expansion with exact unfiltered candidates",
                        "latency is CPU exact-search time, not pgvector HNSW latency",
                    ),
                )
            )
        else:
            raise ValueError(f"unknown benchmark strategy: {strategy}")

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return BenchmarkReport(
        dataset={
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
        },
        ground_truth={
            "metric": truth.metric,
            "k": truth.k,
            "candidate_count": truth.candidate_count,
            "block_rows": truth.block_rows,
            "blocks_scanned": truth.blocks_scanned,
            "first_query_indices": [
                int(index) for index in truth.indices[0].tolist() if int(index) >= 0
            ],
        },
        strategies=tuple(strategy_metrics),
        elapsed_ms=elapsed_ms,
        notes=(
            "ground truth computed with blocked exact filtered search",
            "postfilter and iterative rows are strategy-semantics simulations",
        ),
    )


def benchmark_report_to_json(report: BenchmarkReport) -> dict[str, object]:
    return {
        "dataset": report.dataset,
        "ground_truth": report.ground_truth,
        "strategies": [_strategy_metrics_to_json(metrics) for metrics in report.strategies],
        "elapsed_ms": report.elapsed_ms,
        "notes": list(report.notes),
    }


def strategy_metrics_from_indices(
    *,
    strategy: str,
    params: dict[str, object],
    truth_indices: Any,
    candidate_indices: Any,
    latencies_ms: tuple[float, ...],
    k: int,
    notes: tuple[str, ...],
) -> StrategyMetrics:
    return _metrics_from_indices(
        strategy=strategy,
        params=params,
        truth_indices=truth_indices,
        candidate_indices=candidate_indices,
        latencies_ms=latencies_ms,
        k=k,
        notes=notes,
    )


def write_benchmark_report(
    report: BenchmarkReport,
    path: Path,
    *,
    output_format: str,
) -> None:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(sorted(OUTPUT_FORMATS))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        path.write_text(
            json.dumps(benchmark_report_to_json(report), indent=2) + "\n",
            encoding="utf-8",
        )
        return
    rows = _benchmark_report_to_csv_rows(report)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def infer_output_format(path: Path | None, requested: str | None) -> str:
    if requested is not None and requested != "auto":
        if requested not in OUTPUT_FORMATS:
            valid = ", ".join(("auto", *sorted(OUTPUT_FORMATS)))
            raise ValueError(f"output format must be one of: {valid}")
        return requested
    if path is not None and path.suffix.lower() == ".csv":
        return "csv"
    return "json"


def _compute_ground_truth(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str,
    block_rows: int | None,
) -> tuple[ExactTopKResult, tuple[float, ...]]:
    np = _numpy()
    indices = np.full((queries.n_queries, k), -1, dtype="int64")
    distances = np.full((queries.n_queries, k), np.inf, dtype="float64")
    latencies_ms: list[float] = []
    first: ExactTopKResult | None = None
    for query_index in range(queries.n_queries):
        started = time.perf_counter()
        result = exact_topk(
            dataset.vectors,
            queries.vectors[query_index],
            k=k,
            metric=metric,
            filter_mask=dataset.filter_mask,
            block_rows=block_rows,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        indices[query_index] = result.indices[0]
        distances[query_index] = result.distances[0]
        if first is None:
            first = result
    assert first is not None
    return (
        ExactTopKResult(
            indices=indices,
            distances=distances,
            metric=first.metric,
            k=first.k,
            candidate_count=first.candidate_count,
            block_rows=first.block_rows,
            blocks_scanned=first.blocks_scanned,
        ),
        tuple(latencies_ms),
    )


def _run_postfilter(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str,
    ef_search: int,
    block_rows: int | None,
) -> tuple[Any, tuple[float, ...]]:
    np = _numpy()
    indices = np.full((queries.n_queries, k), -1, dtype="int64")
    latencies_ms: list[float] = []
    candidate_pool = min(ef_search, dataset.n_rows)
    for query_index in range(queries.n_queries):
        started = time.perf_counter()
        unfiltered = exact_topk(
            dataset.vectors,
            queries.vectors[query_index],
            k=candidate_pool,
            metric=metric,
            block_rows=block_rows,
        )
        indices[query_index] = _filter_ranked_indices(
            np,
            unfiltered.indices[0],
            dataset.filter_mask,
            k=k,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    return indices, tuple(latencies_ms)


def _run_iterative(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str,
    max_scan_tuples: int,
    block_rows: int | None,
) -> tuple[Any, tuple[float, ...]]:
    np = _numpy()
    indices = np.full((queries.n_queries, k), -1, dtype="int64")
    for_query_k = min(max_scan_tuples, dataset.n_rows)
    latencies_ms: list[float] = []
    for query_index in range(queries.n_queries):
        started = time.perf_counter()
        unfiltered = exact_topk(
            dataset.vectors,
            queries.vectors[query_index],
            k=for_query_k,
            metric=metric,
            block_rows=block_rows,
        )
        indices[query_index] = _filter_ranked_indices(
            np,
            unfiltered.indices[0],
            dataset.filter_mask,
            k=k,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    return indices, tuple(latencies_ms)


def _filter_ranked_indices(np: Any, ranked_indices: Any, filter_mask: Any, *, k: int) -> Any:
    output = np.full(k, -1, dtype="int64")
    write_index = 0
    for raw_index in ranked_indices.tolist():
        index = int(raw_index)
        if index < 0 or not bool(filter_mask[index]):
            continue
        output[write_index] = index
        write_index += 1
        if write_index == k:
            break
    return output


def _metrics_from_indices(
    *,
    strategy: str,
    params: dict[str, object],
    truth_indices: Any,
    candidate_indices: Any,
    latencies_ms: tuple[float, ...],
    k: int,
    notes: tuple[str, ...],
) -> StrategyMetrics:
    np = _numpy()
    recall = recall_at_k(truth_indices, candidate_indices, k=k)
    truth_counts = _valid_counts(np, truth_indices, k=k)
    candidate_counts = _valid_counts(np, candidate_indices, k=k)
    required_counts = np.minimum(truth_counts, k)
    returns_k_rate = float(np.mean(candidate_counts >= required_counts))
    return StrategyMetrics(
        strategy=strategy,
        params=params,
        query_count=int(candidate_indices.shape[0]),
        recall_at_k=recall.mean,
        returns_k_rate=returns_k_rate,
        result_count_mean=float(np.mean(candidate_counts)),
        latency_ms_total=float(sum(latencies_ms)),
        latency_ms_mean=float(np.mean(latencies_ms)),
        latency_ms_p50=_percentile(np, latencies_ms, 50.0),
        latency_ms_p95=_percentile(np, latencies_ms, 95.0),
        latency_ms_p99=_percentile(np, latencies_ms, 99.0),
        notes=notes,
    )


def _valid_counts(np: Any, indices: Any, *, k: int) -> Any:
    return np.sum(np.asarray(indices[:, :k]) >= 0, axis=1)


def _percentile(np: Any, values: tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype="float64"), percentile))


def _strategy_metrics_to_json(metrics: StrategyMetrics) -> dict[str, object]:
    return {
        "strategy": metrics.strategy,
        "params": metrics.params,
        "query_count": metrics.query_count,
        "recall_at_k": metrics.recall_at_k,
        "returns_k_rate": metrics.returns_k_rate,
        "result_count_mean": metrics.result_count_mean,
        "latency_ms": {
            "total": metrics.latency_ms_total,
            "mean": metrics.latency_ms_mean,
            "p50": metrics.latency_ms_p50,
            "p95": metrics.latency_ms_p95,
            "p99": metrics.latency_ms_p99,
        },
        "notes": list(metrics.notes),
    }


def _benchmark_report_to_csv_rows(report: BenchmarkReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for metrics in report.strategies:
        rows.append(
            {
                "dataset": report.dataset["id"],
                "rows": report.dataset["rows"],
                "dim": report.dataset["dim"],
                "queries": report.dataset["queries"],
                "observed_filter_selectivity": report.dataset["observed_filter_selectivity"],
                "correlation": report.dataset["correlation"],
                "strategy": metrics.strategy,
                "params": json.dumps(metrics.params, sort_keys=True),
                "recall_at_k": metrics.recall_at_k,
                "returns_k_rate": metrics.returns_k_rate,
                "result_count_mean": metrics.result_count_mean,
                "latency_ms_mean": metrics.latency_ms_mean,
                "latency_ms_p50": metrics.latency_ms_p50,
                "latency_ms_p95": metrics.latency_ms_p95,
                "latency_ms_p99": metrics.latency_ms_p99,
            }
        )
    return rows


def _numpy() -> Any:
    return importlib.import_module("numpy")
