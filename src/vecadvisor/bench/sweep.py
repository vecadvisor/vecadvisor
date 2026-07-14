from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from psycopg import Connection

from ..calibration import calibration_profile_to_json
from ..models import CalibrationProfile
from .datasets import generate_synthetic_dataset, generate_synthetic_queries
from .db_runner import ITERATIVE_ORDERS, run_postgres_synthetic_benchmark
from .runner import (
    DEFAULT_STRATEGIES,
    OUTPUT_FORMATS,
    BenchmarkReport,
    StrategyMetrics,
    benchmark_report_to_json,
    run_synthetic_benchmark,
)
from .validate import (
    LocalSelectivitySummary,
    PredictedStrategy,
    estimate_local_selectivity,
    measured_best_strategy,
    run_postgres_validation,
    run_synthetic_validation,
)

DEFAULT_FILTER_SELECTIVITY_SWEEP = (0.001, 0.01, 0.1, 0.3)
DEFAULT_CORRELATION_SWEEP = (-0.8, 0.0, 0.8)


@dataclass(frozen=True)
class SweepPoint:
    target_filter_selectivity: float
    target_correlation: float
    benchmark: BenchmarkReport
    local_selectivity: LocalSelectivitySummary
    measured_best: str
    predicted_best: str | None
    prediction_match: bool | None
    predictions: tuple[PredictedStrategy, ...]


@dataclass(frozen=True)
class SweepReport:
    backend: str
    rows: int
    dim: int
    queries: int
    clusters: int
    limit: int
    metric: str
    ef_search: int
    max_scan_tuples: int
    probe_rows: int
    recall_target: float
    returns_k_target: float
    block_rows: int | None
    query_policy: str
    iterative_order: str | None
    hnsw_m: int | None
    hnsw_ef_construction: int | None
    statement_timeout_ms: int | None
    seed: int
    filter_selectivities: tuple[float, ...]
    correlations: tuple[float, ...]
    calibration: CalibrationProfile | None
    points: tuple[SweepPoint, ...]
    elapsed_ms: float
    notes: tuple[str, ...] = ()


def parse_float_sweep(
    value: str | None,
    *,
    default: tuple[float, ...],
    name: str,
    min_value: float,
    max_value: float,
    include_min: bool = True,
    include_max: bool = True,
) -> tuple[float, ...]:
    """Parse a comma-separated float sweep into sorted unique points."""

    if value is None or not value.strip():
        return default

    points: list[float] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            point = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{name} sweep values must be numeric") from exc
        if not math.isfinite(point):
            raise ValueError(f"{name} sweep values must be finite")
        if not _within_bounds(
            point,
            min_value=min_value,
            max_value=max_value,
            include_min=include_min,
            include_max=include_max,
        ):
            bounds = _bounds_message(
                min_value=min_value,
                max_value=max_value,
                include_min=include_min,
                include_max=include_max,
            )
            raise ValueError(f"{name} sweep values must be in {bounds}")
        points.append(point)

    if not points:
        raise ValueError(f"{name} sweep must contain at least one point")
    return tuple(sorted(set(points)))


def run_synthetic_sweep(
    *,
    rows: int = 5_000,
    dim: int = 64,
    queries: int = 50,
    clusters: int = 16,
    filter_selectivities: tuple[float, ...] = DEFAULT_FILTER_SELECTIVITY_SWEEP,
    correlations: tuple[float, ...] = DEFAULT_CORRELATION_SWEEP,
    limit: int = 10,
    metric: str = "l2",
    ef_search: int = 40,
    max_scan_tuples: int = 1_000,
    probe_rows: int = 200,
    recall_target: float = 0.9,
    returns_k_target: float = 1.0,
    block_rows: int | None = None,
    query_policy: str = "uniform",
    seed: int = 0,
    calibration: CalibrationProfile | None = None,
) -> SweepReport:
    """Run a synthetic selectivity/correlation benchmark sweep."""

    _validate_common_sweep_args(
        rows=rows,
        dim=dim,
        queries=queries,
        clusters=clusters,
        filter_selectivities=filter_selectivities,
        correlations=correlations,
        limit=limit,
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        probe_rows=probe_rows,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
    )

    started = time.perf_counter()
    points: list[SweepPoint] = []
    for point_index, (filter_selectivity, correlation) in enumerate(
        product(filter_selectivities, correlations)
    ):
        point_seed = seed + point_index * 1009
        dataset = generate_synthetic_dataset(
            n_rows=rows,
            dim=dim,
            n_clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            seed=point_seed,
        )
        query_set = generate_synthetic_queries(
            dataset,
            n_queries=queries,
            seed=point_seed + 1,
            cluster_policy=query_policy,
        )
        if calibration is None:
            benchmark = run_synthetic_benchmark(
                dataset=dataset,
                queries=query_set,
                k=limit,
                metric=metric,
                strategies=DEFAULT_STRATEGIES,
                ef_search=ef_search,
                max_scan_tuples=max_scan_tuples,
                block_rows=block_rows,
            )
            local_selectivity = estimate_local_selectivity(
                dataset=dataset,
                queries=query_set,
                metric=metric,
                probe_rows=probe_rows,
                block_rows=block_rows,
            )
            measured = measured_best_strategy(
                benchmark,
                recall_target=recall_target,
                returns_k_target=returns_k_target,
            )
            points.append(
                SweepPoint(
                    target_filter_selectivity=filter_selectivity,
                    target_correlation=correlation,
                    benchmark=benchmark,
                    local_selectivity=local_selectivity,
                    measured_best=measured.strategy,
                    predicted_best=None,
                    prediction_match=None,
                    predictions=(),
                )
            )
            continue

        validation = run_synthetic_validation(
            dataset=dataset,
            queries=query_set,
            calibration=calibration,
            k=limit,
            metric=metric,
            ef_search=ef_search,
            max_scan_tuples=max_scan_tuples,
            probe_rows=probe_rows,
            recall_target=recall_target,
            returns_k_target=returns_k_target,
            block_rows=block_rows,
        )
        points.append(
            SweepPoint(
                target_filter_selectivity=filter_selectivity,
                target_correlation=correlation,
                benchmark=validation.benchmark,
                local_selectivity=validation.local_selectivity,
                measured_best=validation.measured_best,
                predicted_best=validation.predicted_best,
                prediction_match=validation.match,
                predictions=validation.predictions,
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SweepReport(
        backend="synthetic",
        rows=rows,
        dim=dim,
        queries=queries,
        clusters=clusters,
        limit=limit,
        metric=metric,
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        probe_rows=probe_rows,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
        block_rows=block_rows,
        query_policy=query_policy,
        iterative_order=None,
        hnsw_m=None,
        hnsw_ef_construction=None,
        statement_timeout_ms=None,
        seed=seed,
        filter_selectivities=filter_selectivities,
        correlations=correlations,
        calibration=calibration,
        points=tuple(points),
        elapsed_ms=elapsed_ms,
        notes=(
            "sweep varies global selectivity and filter/vector correlation",
            "measured_best is lowest p95 latency among strategies meeting quality targets",
            "local selectivity is estimated from exact unfiltered probes for every point",
        ),
    )


def run_postgres_sweep(
    conn: Connection[Any],
    *,
    rows: int = 1_000,
    dim: int = 32,
    queries: int = 10,
    clusters: int = 8,
    filter_selectivities: tuple[float, ...] = DEFAULT_FILTER_SELECTIVITY_SWEEP,
    correlations: tuple[float, ...] = DEFAULT_CORRELATION_SWEEP,
    limit: int = 10,
    metric: str = "l2",
    ef_search: int = 40,
    max_scan_tuples: int = 1_000,
    iterative_order: str = "relaxed_order",
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
    probe_rows: int = 200,
    recall_target: float = 0.9,
    returns_k_target: float = 1.0,
    block_rows: int | None = None,
    query_policy: str = "uniform",
    seed: int = 0,
    calibration: CalibrationProfile | None = None,
    statement_timeout_ms: int = 30_000,
) -> SweepReport:
    """Run a selectivity/correlation sweep against actual PostgreSQL/pgvector SQL."""

    _validate_common_sweep_args(
        rows=rows,
        dim=dim,
        queries=queries,
        clusters=clusters,
        filter_selectivities=filter_selectivities,
        correlations=correlations,
        limit=limit,
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        probe_rows=probe_rows,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
    )
    if iterative_order not in ITERATIVE_ORDERS:
        raise ValueError(
            f"iterative_order must be one of: {', '.join(sorted(ITERATIVE_ORDERS))}"
        )
    if hnsw_m <= 0:
        raise ValueError("hnsw_m must be positive")
    if hnsw_ef_construction <= 0:
        raise ValueError("hnsw_ef_construction must be positive")
    if statement_timeout_ms <= 0:
        raise ValueError("statement_timeout_ms must be positive")

    started = time.perf_counter()
    points: list[SweepPoint] = []
    for point_index, (filter_selectivity, correlation) in enumerate(
        product(filter_selectivities, correlations)
    ):
        point_seed = seed + point_index * 1009
        dataset = generate_synthetic_dataset(
            n_rows=rows,
            dim=dim,
            n_clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            seed=point_seed,
        )
        query_set = generate_synthetic_queries(
            dataset,
            n_queries=queries,
            seed=point_seed + 1,
            cluster_policy=query_policy,
        )
        if calibration is None:
            benchmark = run_postgres_synthetic_benchmark(
                conn,
                dataset=dataset,
                queries=query_set,
                k=limit,
                metric=metric,
                strategies=DEFAULT_STRATEGIES,
                ef_search=ef_search,
                max_scan_tuples=max_scan_tuples,
                iterative_order=iterative_order,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                block_rows=block_rows,
                statement_timeout_ms=statement_timeout_ms,
            )
            local_selectivity = estimate_local_selectivity(
                dataset=dataset,
                queries=query_set,
                metric=metric,
                probe_rows=probe_rows,
                block_rows=block_rows,
            )
            measured = measured_best_strategy(
                benchmark,
                recall_target=recall_target,
                returns_k_target=returns_k_target,
            )
            points.append(
                SweepPoint(
                    target_filter_selectivity=filter_selectivity,
                    target_correlation=correlation,
                    benchmark=benchmark,
                    local_selectivity=local_selectivity,
                    measured_best=measured.strategy,
                    predicted_best=None,
                    prediction_match=None,
                    predictions=(),
                )
            )
            continue

        validation = run_postgres_validation(
            conn,
            dataset=dataset,
            queries=query_set,
            calibration=calibration,
            k=limit,
            metric=metric,
            ef_search=ef_search,
            max_scan_tuples=max_scan_tuples,
            iterative_order=iterative_order,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            probe_rows=probe_rows,
            recall_target=recall_target,
            returns_k_target=returns_k_target,
            block_rows=block_rows,
            statement_timeout_ms=statement_timeout_ms,
        )
        points.append(
            SweepPoint(
                target_filter_selectivity=filter_selectivity,
                target_correlation=correlation,
                benchmark=validation.benchmark,
                local_selectivity=validation.local_selectivity,
                measured_best=validation.measured_best,
                predicted_best=validation.predicted_best,
                prediction_match=validation.match,
                predictions=validation.predictions,
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SweepReport(
        backend="postgres",
        rows=rows,
        dim=dim,
        queries=queries,
        clusters=clusters,
        limit=limit,
        metric=metric,
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        probe_rows=probe_rows,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
        block_rows=block_rows,
        query_policy=query_policy,
        iterative_order=iterative_order,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        statement_timeout_ms=statement_timeout_ms,
        seed=seed,
        filter_selectivities=filter_selectivities,
        correlations=correlations,
        calibration=calibration,
        points=tuple(points),
        elapsed_ms=elapsed_ms,
        notes=(
            "sweep varies global selectivity and filter/vector correlation",
            "measured_best is lowest p95 latency among actual PostgreSQL strategies meeting "
            "quality targets",
            "local selectivity is estimated from exact unfiltered probes for every point",
            "each point rebuilds a session-local temp table and HNSW index",
        ),
    )


def sweep_report_to_json(
    report: SweepReport,
    *,
    calibration_source: str,
) -> dict[str, object]:
    return {
        "sweep": {
            "backend": report.backend,
            "points": len(report.points),
            "rows": report.rows,
            "dim": report.dim,
            "queries": report.queries,
            "clusters": report.clusters,
            "limit": report.limit,
            "metric": report.metric,
            "ef_search": report.ef_search,
            "max_scan_tuples": report.max_scan_tuples,
            "probe_rows": report.probe_rows,
            "recall_target": report.recall_target,
            "returns_k_target": report.returns_k_target,
            "block_rows": report.block_rows,
            "query_policy": report.query_policy,
            "iterative_order": report.iterative_order,
            "hnsw_m": report.hnsw_m,
            "hnsw_ef_construction": report.hnsw_ef_construction,
            "statement_timeout_ms": report.statement_timeout_ms,
            "seed": report.seed,
            "filter_selectivities": list(report.filter_selectivities),
            "correlations": list(report.correlations),
        },
        "points": [_sweep_point_to_json(point) for point in report.points],
        "calibration": {
            "source": calibration_source,
            "profile": (
                calibration_profile_to_json(report.calibration)
                if report.calibration is not None
                else None
            ),
        },
        "elapsed_ms": report.elapsed_ms,
        "notes": list(report.notes),
    }


def write_sweep_report(
    report: SweepReport,
    path: Path,
    *,
    output_format: str,
    calibration_source: str,
) -> None:
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of: {', '.join(sorted(OUTPUT_FORMATS))}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "json":
        path.write_text(
            json.dumps(
                sweep_report_to_json(report, calibration_source=calibration_source),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    rows = _sweep_report_to_csv_rows(report)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _validate_common_sweep_args(
    *,
    rows: int,
    dim: int,
    queries: int,
    clusters: int,
    filter_selectivities: tuple[float, ...],
    correlations: tuple[float, ...],
    limit: int,
    ef_search: int,
    max_scan_tuples: int,
    probe_rows: int,
    recall_target: float,
    returns_k_target: float,
) -> None:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if dim <= 0:
        raise ValueError("dim must be positive")
    if queries <= 0:
        raise ValueError("queries must be positive")
    if clusters <= 0:
        raise ValueError("clusters must be positive")
    if limit <= 0:
        raise ValueError("limit must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if max_scan_tuples <= 0:
        raise ValueError("max_scan_tuples must be positive")
    if probe_rows <= 0:
        raise ValueError("probe_rows must be positive")
    if not 0.0 < recall_target <= 1.0:
        raise ValueError("recall_target must be in (0, 1]")
    if not 0.0 <= returns_k_target <= 1.0:
        raise ValueError("returns_k_target must be in [0, 1]")
    _validate_sweep_points(filter_selectivities, name="filter_selectivity")
    _validate_sweep_points(correlations, name="correlation", min_value=-1.0, max_value=1.0)


def _validate_sweep_points(
    values: tuple[float, ...],
    *,
    name: str,
    min_value: float = 0.0,
    max_value: float = 1.0,
) -> None:
    if not values:
        raise ValueError(f"{name} sweep must contain at least one point")
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} sweep values must be finite")
        if name == "filter_selectivity" and not 0.0 < value < 1.0:
            raise ValueError("filter_selectivity sweep values must be in (0, 1)")
        if name != "filter_selectivity" and not min_value <= value <= max_value:
            raise ValueError(f"{name} sweep values must be in [{min_value}, {max_value}]")


def _within_bounds(
    point: float,
    *,
    min_value: float,
    max_value: float,
    include_min: bool,
    include_max: bool,
) -> bool:
    above_min = point >= min_value if include_min else point > min_value
    below_max = point <= max_value if include_max else point < max_value
    return above_min and below_max


def _bounds_message(
    *,
    min_value: float,
    max_value: float,
    include_min: bool,
    include_max: bool,
) -> str:
    left = "[" if include_min else "("
    right = "]" if include_max else ")"
    return f"{left}{min_value}, {max_value}{right}"


def _sweep_point_to_json(point: SweepPoint) -> dict[str, object]:
    benchmark = benchmark_report_to_json(point.benchmark)
    return {
        "target_filter_selectivity": point.target_filter_selectivity,
        "target_correlation": point.target_correlation,
        "dataset": benchmark["dataset"],
        "ground_truth": benchmark["ground_truth"],
        "local_selectivity": _local_selectivity_to_json(point.local_selectivity),
        "measured_best": point.measured_best,
        "predicted_best": point.predicted_best,
        "prediction_match": point.prediction_match,
        "predictions": [_prediction_to_json(prediction) for prediction in point.predictions],
        "strategies": benchmark["strategies"],
        "elapsed_ms": point.benchmark.elapsed_ms,
        "notes": list(point.benchmark.notes),
    }


def _sweep_report_to_csv_rows(report: SweepReport) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in report.points:
        prediction_by_strategy = {
            prediction.strategy: prediction for prediction in point.predictions
        }
        for metrics in point.benchmark.strategies:
            prediction = prediction_by_strategy.get(metrics.strategy)
            rows.append(
                _sweep_strategy_csv_row(
                    report=report,
                    point=point,
                    metrics=metrics,
                    prediction=prediction,
                )
            )
    return rows


def _sweep_strategy_csv_row(
    *,
    report: SweepReport,
    point: SweepPoint,
    metrics: StrategyMetrics,
    prediction: PredictedStrategy | None,
) -> dict[str, object]:
    return {
        "backend": report.backend,
        "target_filter_selectivity": point.target_filter_selectivity,
        "target_correlation": point.target_correlation,
        "observed_filter_selectivity": point.benchmark.dataset[
            "observed_filter_selectivity"
        ],
        "query_policy": point.benchmark.dataset["query_policy"],
        "rows": point.benchmark.dataset["rows"],
        "dim": point.benchmark.dataset["dim"],
        "queries": point.benchmark.dataset["queries"],
        "clusters": point.benchmark.dataset["clusters"],
        "limit": point.benchmark.ground_truth["k"],
        "ef_search": report.ef_search,
        "max_scan_tuples": report.max_scan_tuples,
        "probe_rows": point.local_selectivity.probe_rows,
        "s_global": point.local_selectivity.s_global,
        "s_local_p10": point.local_selectivity.s_local_p10,
        "s_local_median": point.local_selectivity.s_local_median,
        "strategy": metrics.strategy,
        "params": json.dumps(metrics.params, sort_keys=True),
        "recall_at_k": metrics.recall_at_k,
        "returns_k_rate": metrics.returns_k_rate,
        "result_count_mean": metrics.result_count_mean,
        "latency_ms_mean": metrics.latency_ms_mean,
        "latency_ms_p50": metrics.latency_ms_p50,
        "latency_ms_p95": metrics.latency_ms_p95,
        "latency_ms_p99": metrics.latency_ms_p99,
        "iterative_order": report.iterative_order or "",
        "hnsw_m": "" if report.hnsw_m is None else report.hnsw_m,
        "hnsw_ef_construction": (
            "" if report.hnsw_ef_construction is None else report.hnsw_ef_construction
        ),
        "statement_timeout_ms": (
            "" if report.statement_timeout_ms is None else report.statement_timeout_ms
        ),
        "measured_best": point.measured_best,
        "predicted_best": point.predicted_best or "",
        "prediction_match": "" if point.prediction_match is None else point.prediction_match,
        "est_latency_ms": "" if prediction is None else prediction.est_latency_ms,
        "est_recall": "" if prediction is None else prediction.est_recall,
        "est_returns_k": "" if prediction is None else prediction.est_returns_k,
        "confidence": "" if prediction is None else prediction.confidence,
    }


def _local_selectivity_to_json(summary: LocalSelectivitySummary) -> dict[str, object]:
    return {
        "s_global": summary.s_global,
        "s_local_p10": summary.s_local_p10,
        "s_local_median": summary.s_local_median,
        "probe_rows": summary.probe_rows,
        "query_count": summary.query_count,
    }


def _prediction_to_json(prediction: PredictedStrategy) -> dict[str, object]:
    return {
        "strategy": prediction.strategy,
        "est_latency_ms": prediction.est_latency_ms,
        "est_recall": prediction.est_recall,
        "est_returns_k": prediction.est_returns_k,
        "confidence": prediction.confidence,
        "notes": list(prediction.notes),
    }
