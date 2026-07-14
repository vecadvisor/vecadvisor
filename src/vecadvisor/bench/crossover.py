from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .sweep import SweepReport, sweep_report_to_json


@dataclass(frozen=True)
class StrategyPointSummary:
    strategy: str
    recall_at_k: float
    returns_k_rate: float
    latency_ms_mean: float
    latency_ms_p95: float


@dataclass(frozen=True)
class SweepAnalysisPoint:
    target_filter_selectivity: float
    target_correlation: float
    observed_filter_selectivity: float
    s_local_p10: float
    s_local_median: float
    measured_best: str
    predicted_best: str | None
    prediction_match: bool | None
    postfilter_recall_at_k: float | None
    postfilter_returns_k_rate: float | None
    postfilter_viable: bool | None
    strategies: tuple[StrategyPointSummary, ...]


@dataclass(frozen=True)
class WinnerRegion:
    kind: str
    correlation: float
    strategy: str
    selectivity_min: float
    selectivity_max: float
    point_count: int


@dataclass(frozen=True)
class WinnerCrossover:
    kind: str
    correlation: float
    lower_selectivity: float
    upper_selectivity: float
    estimated_selectivity: float
    from_strategy: str
    to_strategy: str


@dataclass(frozen=True)
class CrossoverAnalysis:
    backend: str
    point_count: int
    correlations: tuple[float, ...]
    recall_target: float
    returns_k_target: float
    prediction_match_rate: float | None
    measured_win_counts: Mapping[str, int]
    predicted_win_counts: Mapping[str, int]
    regions: tuple[WinnerRegion, ...]
    measured_crossovers: tuple[WinnerCrossover, ...]
    predicted_crossovers: tuple[WinnerCrossover, ...]
    postfilter_failure_count: int
    points: tuple[SweepAnalysisPoint, ...]
    notes: tuple[str, ...] = ()


def load_sweep_payload(path: Path) -> Mapping[str, Any]:
    """Load a JSON sweep report produced by benchmark-sweep or benchmark-sweep-db."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read sweep report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"sweep report JSON is invalid: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("sweep report must be a JSON object")
    return raw


def analyze_sweep_report(report: SweepReport) -> CrossoverAnalysis:
    """Analyze an in-memory sweep report."""

    return analyze_sweep_payload(sweep_report_to_json(report, calibration_source="in-memory"))


def analyze_sweep_payload(payload: Mapping[str, Any]) -> CrossoverAnalysis:
    """Compute winner regions, crossovers, and prediction quality from sweep JSON."""

    sweep = _mapping_value(payload, "sweep")
    backend = _str_value(sweep, "backend")
    recall_target = _float_value(sweep, "recall_target")
    returns_k_target = _float_value(sweep, "returns_k_target")
    points = tuple(
        _analysis_point(point, recall_target=recall_target, returns_k_target=returns_k_target)
        for point in _sequence_value(payload, "points")
    )
    if not points:
        raise ValueError("sweep report must contain at least one point")

    correlations = tuple(sorted({point.target_correlation for point in points}))
    predicted_points = [point for point in points if point.predicted_best is not None]
    prediction_match_rate = (
        sum(1 for point in predicted_points if point.prediction_match) / len(predicted_points)
        if predicted_points
        else None
    )
    measured_win_counts = dict(Counter(point.measured_best for point in points))
    predicted_winners: list[str] = []
    for point in predicted_points:
        assert point.predicted_best is not None
        predicted_winners.append(point.predicted_best)
    predicted_win_counts = dict(Counter(predicted_winners))
    postfilter_failure_count = sum(1 for point in points if point.postfilter_viable is False)

    measured_regions = _winner_regions(points, kind="measured")
    predicted_regions = _winner_regions(points, kind="predicted")
    measured_crossovers = _winner_crossovers(points, kind="measured")
    predicted_crossovers = _winner_crossovers(points, kind="predicted")

    return CrossoverAnalysis(
        backend=backend,
        point_count=len(points),
        correlations=correlations,
        recall_target=recall_target,
        returns_k_target=returns_k_target,
        prediction_match_rate=prediction_match_rate,
        measured_win_counts=measured_win_counts,
        predicted_win_counts=predicted_win_counts,
        regions=(*measured_regions, *predicted_regions),
        measured_crossovers=measured_crossovers,
        predicted_crossovers=predicted_crossovers,
        postfilter_failure_count=postfilter_failure_count,
        points=points,
        notes=(
            "crossovers are estimated between adjacent sampled selectivity points",
            "estimated_selectivity uses the geometric midpoint for positive selectivities",
            "postfilter_failure_count uses the sweep recall and returns-k targets",
        ),
    )


def crossover_analysis_to_json(analysis: CrossoverAnalysis) -> dict[str, object]:
    return {
        "analysis": {
            "backend": analysis.backend,
            "point_count": analysis.point_count,
            "correlations": list(analysis.correlations),
            "recall_target": analysis.recall_target,
            "returns_k_target": analysis.returns_k_target,
            "prediction_match_rate": analysis.prediction_match_rate,
            "postfilter_failure_count": analysis.postfilter_failure_count,
        },
        "measured_win_counts": dict(analysis.measured_win_counts),
        "predicted_win_counts": dict(analysis.predicted_win_counts),
        "regions": [_region_to_json(region) for region in analysis.regions],
        "measured_crossovers": [
            _crossover_to_json(crossover) for crossover in analysis.measured_crossovers
        ],
        "predicted_crossovers": [
            _crossover_to_json(crossover) for crossover in analysis.predicted_crossovers
        ],
        "points": [_point_to_json(point) for point in analysis.points],
        "notes": list(analysis.notes),
    }


def write_crossover_analysis(analysis: CrossoverAnalysis, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(crossover_analysis_to_json(analysis), indent=2) + "\n",
        encoding="utf-8",
    )


def _analysis_point(
    raw: object,
    *,
    recall_target: float,
    returns_k_target: float,
) -> SweepAnalysisPoint:
    point = _as_mapping(raw, "sweep point")
    dataset = _mapping_value(point, "dataset")
    local = _mapping_value(point, "local_selectivity")
    strategies = tuple(
        _strategy_summary(strategy) for strategy in _sequence_value(point, "strategies")
    )
    postfilter = next(
        (strategy for strategy in strategies if strategy.strategy == "postfilter"),
        None,
    )
    postfilter_viable = (
        postfilter.recall_at_k >= recall_target
        and postfilter.returns_k_rate >= returns_k_target
        if postfilter is not None
        else None
    )
    return SweepAnalysisPoint(
        target_filter_selectivity=_float_value(point, "target_filter_selectivity"),
        target_correlation=_float_value(point, "target_correlation"),
        observed_filter_selectivity=_float_value(dataset, "observed_filter_selectivity"),
        s_local_p10=_float_value(local, "s_local_p10"),
        s_local_median=_float_value(local, "s_local_median"),
        measured_best=_str_value(point, "measured_best"),
        predicted_best=_optional_str_value(point, "predicted_best"),
        prediction_match=_optional_bool_value(point, "prediction_match"),
        postfilter_recall_at_k=None if postfilter is None else postfilter.recall_at_k,
        postfilter_returns_k_rate=None if postfilter is None else postfilter.returns_k_rate,
        postfilter_viable=postfilter_viable,
        strategies=strategies,
    )


def _strategy_summary(raw: object) -> StrategyPointSummary:
    strategy = _as_mapping(raw, "strategy")
    latency = _mapping_value(strategy, "latency_ms")
    return StrategyPointSummary(
        strategy=_str_value(strategy, "strategy"),
        recall_at_k=_float_value(strategy, "recall_at_k"),
        returns_k_rate=_float_value(strategy, "returns_k_rate"),
        latency_ms_mean=_float_value(latency, "mean"),
        latency_ms_p95=_float_value(latency, "p95"),
    )


def _winner_regions(
    points: Sequence[SweepAnalysisPoint],
    *,
    kind: str,
) -> tuple[WinnerRegion, ...]:
    regions: list[WinnerRegion] = []
    for correlation in sorted({point.target_correlation for point in points}):
        group = _points_for_correlation(points, correlation=correlation, kind=kind)
        if not group:
            continue
        start = group[0]
        current_strategy = _winner(start, kind=kind)
        assert current_strategy is not None
        current_points = [start]
        for point in group[1:]:
            strategy = _winner(point, kind=kind)
            if strategy == current_strategy:
                current_points.append(point)
                continue
            regions.append(_region_from_points(kind, correlation, current_strategy, current_points))
            assert strategy is not None
            current_strategy = strategy
            current_points = [point]
        regions.append(_region_from_points(kind, correlation, current_strategy, current_points))
    return tuple(regions)


def _winner_crossovers(
    points: Sequence[SweepAnalysisPoint],
    *,
    kind: str,
) -> tuple[WinnerCrossover, ...]:
    crossovers: list[WinnerCrossover] = []
    for correlation in sorted({point.target_correlation for point in points}):
        group = _points_for_correlation(points, correlation=correlation, kind=kind)
        for lower, upper in zip(group, group[1:], strict=False):
            lower_winner = _winner(lower, kind=kind)
            upper_winner = _winner(upper, kind=kind)
            if lower_winner is None or upper_winner is None or lower_winner == upper_winner:
                continue
            crossovers.append(
                WinnerCrossover(
                    kind=kind,
                    correlation=correlation,
                    lower_selectivity=lower.target_filter_selectivity,
                    upper_selectivity=upper.target_filter_selectivity,
                    estimated_selectivity=_midpoint_selectivity(
                        lower.target_filter_selectivity,
                        upper.target_filter_selectivity,
                    ),
                    from_strategy=lower_winner,
                    to_strategy=upper_winner,
                )
            )
    return tuple(crossovers)


def _points_for_correlation(
    points: Sequence[SweepAnalysisPoint],
    *,
    correlation: float,
    kind: str,
) -> list[SweepAnalysisPoint]:
    return sorted(
        (
            point
            for point in points
            if point.target_correlation == correlation and _winner(point, kind=kind) is not None
        ),
        key=lambda point: point.target_filter_selectivity,
    )


def _winner(point: SweepAnalysisPoint, *, kind: str) -> str | None:
    if kind == "measured":
        return point.measured_best
    if kind == "predicted":
        return point.predicted_best
    raise ValueError("kind must be measured or predicted")


def _region_from_points(
    kind: str,
    correlation: float,
    strategy: str,
    points: Sequence[SweepAnalysisPoint],
) -> WinnerRegion:
    selectivities = [point.target_filter_selectivity for point in points]
    return WinnerRegion(
        kind=kind,
        correlation=correlation,
        strategy=strategy,
        selectivity_min=min(selectivities),
        selectivity_max=max(selectivities),
        point_count=len(points),
    )


def _midpoint_selectivity(lower: float, upper: float) -> float:
    if lower > 0.0 and upper > 0.0:
        return math.sqrt(lower * upper)
    return (lower + upper) / 2.0


def _point_to_json(point: SweepAnalysisPoint) -> dict[str, object]:
    return {
        "target_filter_selectivity": point.target_filter_selectivity,
        "target_correlation": point.target_correlation,
        "observed_filter_selectivity": point.observed_filter_selectivity,
        "s_local_p10": point.s_local_p10,
        "s_local_median": point.s_local_median,
        "measured_best": point.measured_best,
        "predicted_best": point.predicted_best,
        "prediction_match": point.prediction_match,
        "postfilter_recall_at_k": point.postfilter_recall_at_k,
        "postfilter_returns_k_rate": point.postfilter_returns_k_rate,
        "postfilter_viable": point.postfilter_viable,
        "strategies": [_strategy_to_json(strategy) for strategy in point.strategies],
    }


def _strategy_to_json(strategy: StrategyPointSummary) -> dict[str, object]:
    return {
        "strategy": strategy.strategy,
        "recall_at_k": strategy.recall_at_k,
        "returns_k_rate": strategy.returns_k_rate,
        "latency_ms_mean": strategy.latency_ms_mean,
        "latency_ms_p95": strategy.latency_ms_p95,
    }


def _region_to_json(region: WinnerRegion) -> dict[str, object]:
    return {
        "kind": region.kind,
        "correlation": region.correlation,
        "strategy": region.strategy,
        "selectivity_min": region.selectivity_min,
        "selectivity_max": region.selectivity_max,
        "point_count": region.point_count,
    }


def _crossover_to_json(crossover: WinnerCrossover) -> dict[str, object]:
    return {
        "kind": crossover.kind,
        "correlation": crossover.correlation,
        "lower_selectivity": crossover.lower_selectivity,
        "upper_selectivity": crossover.upper_selectivity,
        "estimated_selectivity": crossover.estimated_selectivity,
        "from_strategy": crossover.from_strategy,
        "to_strategy": crossover.to_strategy,
    }


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(payload.get(key), key)


def _sequence_value(payload: Mapping[str, Any], key: str) -> Sequence[object]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a list")
    return value


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _str_value(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str_value(payload: Mapping[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _optional_bool_value(payload: Mapping[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be null or boolean")
    return value


def _float_value(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number
