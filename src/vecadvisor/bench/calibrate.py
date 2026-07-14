from __future__ import annotations

import math
from dataclasses import dataclass

from psycopg import Connection

from ..models import CalibrationProfile
from .datasets import generate_synthetic_dataset, generate_synthetic_queries
from .db_runner import run_postgres_synthetic_benchmark
from .runner import (
    STRATEGY_EXACT,
    STRATEGY_POSTFILTER,
    BenchmarkReport,
    StrategyMetrics,
    run_synthetic_benchmark,
)

DEFAULT_EF_SWEEP = (10, 20, 40, 80, 160)
MIN_RECALL = 1e-6


@dataclass(frozen=True)
class CalibrationFit:
    profile: CalibrationProfile
    reports: tuple[BenchmarkReport, ...]
    notes: tuple[str, ...]


def parse_ef_sweep(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return DEFAULT_EF_SWEEP
    points: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        try:
            point = int(stripped)
        except ValueError as exc:
            raise ValueError("ef sweep values must be positive integers") from exc
        if point <= 0:
            raise ValueError("ef sweep values must be positive integers")
        points.append(point)
    if not points:
        raise ValueError("ef sweep must contain at least one point")
    return tuple(sorted(set(points)))


def run_synthetic_calibration(
    *,
    rows: int = 5_000,
    dim: int = 64,
    queries: int = 50,
    clusters: int = 16,
    filter_selectivity: float = 0.1,
    correlation: float = 0.0,
    limit: int = 10,
    metric: str = "l2",
    block_rows: int | None = None,
    seed: int = 0,
    ef_sweep: tuple[int, ...] = DEFAULT_EF_SWEEP,
    dataset_id: str = "synthetic-simulated",
    hardware_id: str = "local-synthetic-cpu",
    index_method: str = "hnsw",
) -> CalibrationFit:
    """Calibrate a profile from the synthetic benchmark semantics runner."""

    if not ef_sweep:
        raise ValueError("ef_sweep must not be empty")
    dataset = generate_synthetic_dataset(
        n_rows=rows,
        dim=dim,
        n_clusters=clusters,
        filter_selectivity=filter_selectivity,
        correlation=correlation,
        seed=seed,
    )
    query_set = generate_synthetic_queries(
        dataset,
        n_queries=queries,
        seed=seed + 1,
        cluster_policy="uniform",
    )
    reports = tuple(
        run_synthetic_benchmark(
            dataset=dataset,
            queries=query_set,
            k=limit,
            metric=metric,
            strategies=(STRATEGY_EXACT, STRATEGY_POSTFILTER),
            ef_search=ef,
            block_rows=block_rows,
        )
        for ef in ef_sweep
    )
    profile = fit_profile_from_reports(
        reports,
        dataset_id=dataset_id,
        hardware_id=hardware_id,
        index_method=index_method,
    )
    return CalibrationFit(
        profile=profile,
        reports=reports,
        notes=(
            "profile fitted from synthetic strategy-semantics benchmark measurements",
            "c_h reflects exact-candidate simulation, not real pgvector HNSW latency",
            "run DB-backed calibration before using this profile for production planning",
        ),
    )


def run_postgres_calibration(
    conn: Connection[object],
    *,
    rows: int = 1_000,
    dim: int = 32,
    queries: int = 10,
    clusters: int = 8,
    filter_selectivity: float = 0.1,
    correlation: float = 0.0,
    limit: int = 10,
    metric: str = "l2",
    block_rows: int | None = None,
    seed: int = 0,
    ef_sweep: tuple[int, ...] = DEFAULT_EF_SWEEP,
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
    statement_timeout_ms: int = 30_000,
    dataset_id: str = "postgres-synthetic",
    hardware_id: str = "local-postgres-pgvector",
    index_method: str = "hnsw",
) -> CalibrationFit:
    """Calibrate a profile from actual PostgreSQL/pgvector benchmark measurements."""

    if not ef_sweep:
        raise ValueError("ef_sweep must not be empty")
    dataset = generate_synthetic_dataset(
        n_rows=rows,
        dim=dim,
        n_clusters=clusters,
        filter_selectivity=filter_selectivity,
        correlation=correlation,
        seed=seed,
    )
    query_set = generate_synthetic_queries(
        dataset,
        n_queries=queries,
        seed=seed + 1,
        cluster_policy="uniform",
    )
    reports = tuple(
        run_postgres_synthetic_benchmark(
            conn,
            dataset=dataset,
            queries=query_set,
            k=limit,
            metric=metric,
            strategies=(STRATEGY_EXACT, STRATEGY_POSTFILTER),
            ef_search=ef,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            block_rows=block_rows,
            statement_timeout_ms=statement_timeout_ms,
        )
        for ef in ef_sweep
    )
    profile = fit_profile_from_reports(
        reports,
        dataset_id=dataset_id,
        hardware_id=hardware_id,
        index_method=index_method,
    )
    return CalibrationFit(
        profile=profile,
        reports=reports,
        notes=(
            "profile fitted from actual PostgreSQL/pgvector benchmark measurements",
            "postfilter latency comes from real HNSW SQL with scalar post-filter",
            "ground truth for recall still comes from bounded exact in-process computation",
        ),
    )


def fit_profile_from_reports(
    reports: tuple[BenchmarkReport, ...],
    *,
    dataset_id: str,
    hardware_id: str,
    index_method: str = "hnsw",
) -> CalibrationProfile:
    if not reports:
        raise ValueError("at least one benchmark report is required")
    first = reports[0]
    exact = _strategy(first, STRATEGY_EXACT)
    rows = _positive_float(first.dataset["rows"], "rows")
    selectivity = max(
        _positive_float(first.dataset["observed_filter_selectivity"], "selectivity"),
        1 / rows,
    )
    k = _positive_int(first.ground_truth["k"], "k")
    qualifying_rows = max(1.0, rows * selectivity)
    exact_latency_us = max(exact.latency_ms_mean * 1000.0, MIN_RECALL)
    exact_denominator = qualifying_rows * (2.0 + math.log2(max(k, 2)))
    c_scan = max(MIN_RECALL, exact_latency_us / max(exact_denominator, MIN_RECALL))
    c_d = c_scan
    c_h_values: list[float] = []
    recall_points: list[tuple[int, float]] = []
    running_recall = 0.0
    for report in reports:
        postfilter = _strategy(report, STRATEGY_POSTFILTER)
        ef = _positive_int(postfilter.params["ef_search"], "ef_search")
        latency_us = max(postfilter.latency_ms_mean * 1000.0, MIN_RECALL)
        denominator = c_d * ef * math.log(max(rows, 2.0))
        c_h_values.append(max(MIN_RECALL, latency_us / max(denominator, MIN_RECALL)))
        running_recall = max(running_recall, postfilter.recall_at_k)
        recall_points.append((ef, min(1.0, max(MIN_RECALL, running_recall))))

    return CalibrationProfile(
        dataset_id=dataset_id,
        hardware_id=hardware_id,
        index_method=index_method,
        c_d=c_d,
        c_scan=c_scan,
        c_h=_median(tuple(c_h_values)),
        delta_strict=0.0,
        recall_curve=tuple(sorted(recall_points)),
    )


def calibration_fit_to_json(fit: CalibrationFit) -> dict[str, object]:
    from ..calibration import calibration_profile_to_json

    return {
        "profile": calibration_profile_to_json(fit.profile),
        "ef_sweep": [ef for ef, _ in fit.profile.recall_curve],
        "fit_reports": [
            {
                "ef_search": _strategy(report, STRATEGY_POSTFILTER).params["ef_search"],
                "postfilter_recall_at_k": _strategy(report, STRATEGY_POSTFILTER).recall_at_k,
                "postfilter_latency_ms_mean": _strategy(
                    report,
                    STRATEGY_POSTFILTER,
                ).latency_ms_mean,
                "exact_latency_ms_mean": _strategy(report, STRATEGY_EXACT).latency_ms_mean,
            }
            for report in fit.reports
        ],
        "notes": list(fit.notes),
    }


def _strategy(report: BenchmarkReport, strategy: str) -> StrategyMetrics:
    for metrics in report.strategies:
        if metrics.strategy == strategy:
            return metrics
    raise ValueError(f"benchmark report does not contain strategy: {strategy}")


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _median(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("cannot compute median of empty values")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0
