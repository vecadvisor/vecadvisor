from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from psycopg import Connection

from ..calibration import calibration_profile_to_json
from ..costmodel import choose_best, cost_exact, cost_iterative, cost_postfilter
from ..models import CalibrationProfile, CostEstimate, Strategy
from .datasets import SyntheticDataset, SyntheticQueries
from .db_runner import run_postgres_synthetic_benchmark
from .groundtruth import exact_topk
from .runner import (
    STRATEGY_EXACT,
    STRATEGY_ITERATIVE,
    STRATEGY_POSTFILTER,
    BenchmarkReport,
    StrategyMetrics,
    benchmark_report_to_json,
    run_synthetic_benchmark,
)


@dataclass(frozen=True)
class LocalSelectivitySummary:
    s_global: float
    s_local_p10: float
    s_local_median: float
    probe_rows: int
    query_count: int


@dataclass(frozen=True)
class PredictedStrategy:
    strategy: str
    est_latency_ms: float
    est_recall: float
    est_returns_k: bool
    confidence: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    benchmark: BenchmarkReport
    calibration: CalibrationProfile
    local_selectivity: LocalSelectivitySummary
    predictions: tuple[PredictedStrategy, ...]
    predicted_best: str
    measured_best: str
    match: bool
    recall_target: float
    returns_k_target: float
    notes: tuple[str, ...] = ()


def run_synthetic_validation(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    calibration: CalibrationProfile,
    k: int,
    metric: str = "l2",
    ef_search: int = 40,
    max_scan_tuples: int = 1_000,
    probe_rows: int = 200,
    recall_target: float = 0.9,
    returns_k_target: float = 1.0,
    block_rows: int | None = None,
) -> ValidationReport:
    """Compare cost-model prediction with measured synthetic benchmark winner."""

    if k <= 0:
        raise ValueError("k must be positive")
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

    benchmark = run_synthetic_benchmark(
        dataset=dataset,
        queries=queries,
        k=k,
        metric=metric,
        strategies=(STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE),
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        block_rows=block_rows,
    )
    local = estimate_local_selectivity(
        dataset=dataset,
        queries=queries,
        metric=metric,
        probe_rows=probe_rows,
        block_rows=block_rows,
    )
    estimates = (
        cost_exact(
            n_rows=dataset.n_rows,
            s_global=local.s_global,
            limit=k,
            cal=calibration,
            confidence=0.85,
        ),
        cost_postfilter(
            n_rows=dataset.n_rows,
            s_local=local.s_local_p10,
            limit=k,
            ef_search=ef_search,
            cal=calibration,
            confidence=0.8,
        ),
        cost_iterative(
            n_rows=dataset.n_rows,
            s_local=local.s_local_p10,
            limit=k,
            ef_search=ef_search,
            cal=calibration,
            strict_order=False,
            max_scan_tuples=max_scan_tuples,
            confidence=0.8,
        ),
    )
    predicted_best = _runner_strategy_name(choose_best(estimates, recall_target).strategy)
    measured_best = measured_best_strategy(
        benchmark,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
    )
    return ValidationReport(
        benchmark=benchmark,
        calibration=calibration,
        local_selectivity=local,
        predictions=tuple(_prediction_from_estimate(estimate) for estimate in estimates),
        predicted_best=predicted_best,
        measured_best=measured_best.strategy,
        match=predicted_best == measured_best.strategy,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
        notes=(
            "prediction uses p10 local selectivity estimated from exact unfiltered probes",
            "measured winner is lowest p95 latency among strategies meeting recall/returns targets",
            "synthetic validation still uses strategy-semantics simulations for ANN strategies",
        ),
    )


def run_postgres_validation(
    conn: Connection[object],
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    calibration: CalibrationProfile,
    k: int,
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
    statement_timeout_ms: int = 30_000,
) -> ValidationReport:
    """Compare cost-model prediction with actual PostgreSQL/pgvector measurements."""

    if k <= 0:
        raise ValueError("k must be positive")
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

    benchmark = run_postgres_synthetic_benchmark(
        conn,
        dataset=dataset,
        queries=queries,
        k=k,
        metric=metric,
        strategies=(STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE),
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        iterative_order=iterative_order,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        block_rows=block_rows,
        statement_timeout_ms=statement_timeout_ms,
    )
    local = estimate_local_selectivity(
        dataset=dataset,
        queries=queries,
        metric=metric,
        probe_rows=probe_rows,
        block_rows=block_rows,
    )
    estimates = (
        cost_exact(
            n_rows=dataset.n_rows,
            s_global=local.s_global,
            limit=k,
            cal=calibration,
            confidence=0.85,
        ),
        cost_postfilter(
            n_rows=dataset.n_rows,
            s_local=local.s_local_p10,
            limit=k,
            ef_search=ef_search,
            cal=calibration,
            confidence=0.8,
        ),
        cost_iterative(
            n_rows=dataset.n_rows,
            s_local=local.s_local_p10,
            limit=k,
            ef_search=ef_search,
            cal=calibration,
            strict_order=iterative_order == "strict_order",
            max_scan_tuples=max_scan_tuples,
            confidence=0.8,
        ),
    )
    predicted_best = _runner_strategy_name(choose_best(estimates, recall_target).strategy)
    measured_best = measured_best_strategy(
        benchmark,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
    )
    return ValidationReport(
        benchmark=benchmark,
        calibration=calibration,
        local_selectivity=local,
        predictions=tuple(_prediction_from_estimate(estimate) for estimate in estimates),
        predicted_best=predicted_best,
        measured_best=measured_best.strategy,
        match=predicted_best == measured_best.strategy,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
        notes=(
            "prediction uses p10 local selectivity estimated from exact unfiltered probes",
            "measured winner is lowest p95 latency among actual PostgreSQL strategies "
            "meeting recall/returns targets",
            "ground truth for recall comes from bounded exact in-process computation",
        ),
    )


def estimate_local_selectivity(
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    metric: str,
    probe_rows: int,
    block_rows: int | None = None,
) -> LocalSelectivitySummary:
    """Estimate p10/median local selectivity with exact unfiltered top-m probes."""

    if probe_rows <= 0:
        raise ValueError("probe_rows must be positive")
    effective_probe_rows = min(probe_rows, dataset.n_rows)
    values: list[float] = []
    for query_index in range(queries.n_queries):
        nearest = exact_topk(
            dataset.vectors,
            queries.vectors[query_index],
            k=effective_probe_rows,
            metric=metric,
            block_rows=block_rows,
        )
        passing = 0
        sample_size = 0
        for raw_index in nearest.indices[0].tolist():
            index = int(raw_index)
            if index < 0:
                continue
            sample_size += 1
            if bool(dataset.filter_mask[index]):
                passing += 1
        values.append(passing / sample_size if sample_size else 0.0)
    return LocalSelectivitySummary(
        s_global=dataset.observed_selectivity,
        s_local_p10=_percentile(tuple(values), 0.10),
        s_local_median=_percentile(tuple(values), 0.50),
        probe_rows=effective_probe_rows,
        query_count=queries.n_queries,
    )


def validation_report_to_json(
    report: ValidationReport,
    *,
    calibration_source: str,
) -> dict[str, object]:
    return {
        "validation": {
            "predicted_best": report.predicted_best,
            "measured_best": report.measured_best,
            "match": report.match,
            "recall_target": report.recall_target,
            "returns_k_target": report.returns_k_target,
        },
        "local_selectivity": {
            "s_global": report.local_selectivity.s_global,
            "s_local_p10": report.local_selectivity.s_local_p10,
            "s_local_median": report.local_selectivity.s_local_median,
            "probe_rows": report.local_selectivity.probe_rows,
            "query_count": report.local_selectivity.query_count,
        },
        "predictions": [
            {
                "strategy": prediction.strategy,
                "est_latency_ms": prediction.est_latency_ms,
                "est_recall": prediction.est_recall,
                "est_returns_k": prediction.est_returns_k,
                "confidence": prediction.confidence,
                "notes": list(prediction.notes),
            }
            for prediction in report.predictions
        ],
        "benchmark": benchmark_report_to_json(report.benchmark),
        "calibration": {
            "source": calibration_source,
            "profile": calibration_profile_to_json(report.calibration),
        },
        "notes": list(report.notes),
    }


def write_validation_report(
    report: ValidationReport,
    path: Path,
    *,
    calibration_source: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            validation_report_to_json(report, calibration_source=calibration_source),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def measured_best_strategy(
    report: BenchmarkReport,
    *,
    recall_target: float,
    returns_k_target: float,
) -> StrategyMetrics:
    """Return the lowest-p95 strategy that satisfies measured quality targets."""

    viable = [
        metrics
        for metrics in report.strategies
        if metrics.recall_at_k >= recall_target and metrics.returns_k_rate >= returns_k_target
    ]
    if not viable:
        viable = [metrics for metrics in report.strategies if metrics.strategy == STRATEGY_EXACT]
    if not viable:
        raise ValueError("benchmark report has no viable measured strategy")
    return min(viable, key=lambda metrics: (metrics.latency_ms_p95, metrics.latency_ms_mean))


def _prediction_from_estimate(estimate: CostEstimate) -> PredictedStrategy:
    return PredictedStrategy(
        strategy=_runner_strategy_name(estimate.strategy),
        est_latency_ms=estimate.est_latency_us / 1000.0,
        est_recall=estimate.est_recall,
        est_returns_k=estimate.est_returns_k,
        confidence=estimate.confidence,
        notes=estimate.notes,
    )


def _runner_strategy_name(strategy: Strategy) -> str:
    if strategy is Strategy.EXACT:
        return STRATEGY_EXACT
    if strategy is Strategy.POSTFILTER:
        return STRATEGY_POSTFILTER
    if strategy in {Strategy.ITERATIVE_RELAXED, Strategy.ITERATIVE_STRICT}:
        return STRATEGY_ITERATIVE
    return strategy.value


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
