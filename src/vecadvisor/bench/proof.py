from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .crossover import CrossoverAnalysis, StrategyPointSummary, SweepAnalysisPoint


@dataclass(frozen=True)
class ProofCheck:
    name: str
    passed: bool
    actual: object
    required: object
    detail: str


@dataclass(frozen=True)
class ProofPoint:
    target_filter_selectivity: float
    target_correlation: float
    measured_best: str
    predicted_best: str | None
    prediction_match: bool | None
    predicted_strategy_viable: bool | None
    postfilter_viable: bool | None
    measured_best_latency_ms_p95: float | None
    predicted_latency_ms_p95: float | None
    postfilter_latency_ms_p95: float | None
    speedup_vs_postfilter: float | None


@dataclass(frozen=True)
class ProofReport:
    passed: bool
    backend: str
    point_count: int
    selectivity_bin_count: int
    prediction_match_rate: float | None
    postfilter_failure_count: int
    safe_advisor_on_postfilter_failures: int
    mean_speedup_vs_postfilter: float | None
    checks: tuple[ProofCheck, ...]
    points: tuple[ProofPoint, ...]
    notes: tuple[str, ...] = ()


def build_proof_report(
    analysis: CrossoverAnalysis,
    *,
    min_points: int = 3,
    min_selectivity_bins: int = 3,
    min_match_rate: float = 0.8,
    min_postfilter_failures: int = 1,
) -> ProofReport:
    """Build a publishability proof report from a calibrated sweep analysis."""

    if min_points <= 0:
        raise ValueError("min_points must be positive")
    if min_selectivity_bins <= 0:
        raise ValueError("min_selectivity_bins must be positive")
    if not 0.0 <= min_match_rate <= 1.0:
        raise ValueError("min_match_rate must be in [0, 1]")
    if min_postfilter_failures < 0:
        raise ValueError("min_postfilter_failures must be non-negative")

    points = tuple(_proof_point(point, analysis=analysis) for point in analysis.points)
    predicted_points = [point for point in points if point.predicted_best is not None]
    matched_points = [point for point in predicted_points if point.prediction_match]
    match_rate = len(matched_points) / len(predicted_points) if predicted_points else None
    selectivity_bin_count = len({point.target_filter_selectivity for point in points})
    postfilter_failures = [point for point in points if point.postfilter_viable is False]
    safe_postfilter_failures = [
        point
        for point in postfilter_failures
        if point.predicted_best != "postfilter" and point.predicted_strategy_viable is True
    ]
    speedups = [
        point.speedup_vs_postfilter
        for point in points
        if point.speedup_vs_postfilter is not None
    ]
    mean_speedup = sum(speedups) / len(speedups) if speedups else None

    checks = (
        ProofCheck(
            name="minimum_points",
            passed=len(points) >= min_points,
            actual=len(points),
            required=min_points,
            detail="sweep should cover enough points to avoid one-off conclusions",
        ),
        ProofCheck(
            name="minimum_selectivity_bins",
            passed=selectivity_bin_count >= min_selectivity_bins,
            actual=selectivity_bin_count,
            required=min_selectivity_bins,
            detail="proof should span multiple selectivity bins",
        ),
        ProofCheck(
            name="predictions_present",
            passed=len(predicted_points) == len(points),
            actual=len(predicted_points),
            required=len(points),
            detail="run benchmark-sweep with --calibration so predicted winners are present",
        ),
        ProofCheck(
            name="prediction_match_rate",
            passed=match_rate is not None and match_rate >= min_match_rate,
            actual=match_rate,
            required=min_match_rate,
            detail="advisor predicted winner should track measured lowest-p95 winner",
        ),
        ProofCheck(
            name="postfilter_failure_coverage",
            passed=len(postfilter_failures) >= min_postfilter_failures,
            actual=len(postfilter_failures),
            required=min_postfilter_failures,
            detail="proof should include bins where default postfilter loses recall/returns-k",
        ),
        ProofCheck(
            name="advisor_safe_when_postfilter_fails",
            passed=len(safe_postfilter_failures) == len(postfilter_failures),
            actual=len(safe_postfilter_failures),
            required=len(postfilter_failures),
            detail="when postfilter fails quality targets, advisor must not pick postfilter",
        ),
    )
    return ProofReport(
        passed=all(check.passed for check in checks),
        backend=analysis.backend,
        point_count=len(points),
        selectivity_bin_count=selectivity_bin_count,
        prediction_match_rate=match_rate,
        postfilter_failure_count=len(postfilter_failures),
        safe_advisor_on_postfilter_failures=len(safe_postfilter_failures),
        mean_speedup_vs_postfilter=mean_speedup,
        checks=checks,
        points=points,
        notes=(
            "proof consumes measured strategy metrics and predicted winners from sweep JSON",
            "prediction quality is evaluated against measured lowest-p95 recall-safe winner",
            "postfilter failure means recall or returns-k fell below sweep targets",
        ),
    )


def proof_report_to_json(report: ProofReport) -> dict[str, object]:
    return {
        "proof": {
            "passed": report.passed,
            "backend": report.backend,
            "point_count": report.point_count,
            "selectivity_bin_count": report.selectivity_bin_count,
            "prediction_match_rate": report.prediction_match_rate,
            "postfilter_failure_count": report.postfilter_failure_count,
            "safe_advisor_on_postfilter_failures": report.safe_advisor_on_postfilter_failures,
            "mean_speedup_vs_postfilter": report.mean_speedup_vs_postfilter,
        },
        "checks": [_check_to_json(check) for check in report.checks],
        "points": [_point_to_json(point) for point in report.points],
        "notes": list(report.notes),
    }


def write_proof_report(report: ProofReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(proof_report_to_json(report), indent=2) + "\n",
        encoding="utf-8",
    )


def _proof_point(point: SweepAnalysisPoint, *, analysis: CrossoverAnalysis) -> ProofPoint:
    predicted = _strategy(point, point.predicted_best)
    measured = _strategy(point, point.measured_best)
    postfilter = _strategy(point, "postfilter")
    predicted_viable = (
        predicted.recall_at_k >= analysis.recall_target
        and predicted.returns_k_rate >= analysis.returns_k_target
        if predicted is not None
        else None
    )
    speedup = None
    if (
        predicted is not None
        and postfilter is not None
        and predicted.latency_ms_p95 > 0.0
    ):
        speedup = postfilter.latency_ms_p95 / predicted.latency_ms_p95
    return ProofPoint(
        target_filter_selectivity=point.target_filter_selectivity,
        target_correlation=point.target_correlation,
        measured_best=point.measured_best,
        predicted_best=point.predicted_best,
        prediction_match=point.prediction_match,
        predicted_strategy_viable=predicted_viable,
        postfilter_viable=point.postfilter_viable,
        measured_best_latency_ms_p95=(
            None if measured is None else measured.latency_ms_p95
        ),
        predicted_latency_ms_p95=(
            None if predicted is None else predicted.latency_ms_p95
        ),
        postfilter_latency_ms_p95=(
            None if postfilter is None else postfilter.latency_ms_p95
        ),
        speedup_vs_postfilter=speedup,
    )


def _strategy(
    point: SweepAnalysisPoint,
    strategy: str | None,
) -> StrategyPointSummary | None:
    if strategy is None:
        return None
    for candidate in point.strategies:
        if candidate.strategy == strategy:
            return candidate
    return None


def _check_to_json(check: ProofCheck) -> dict[str, object]:
    return {
        "name": check.name,
        "passed": check.passed,
        "actual": check.actual,
        "required": check.required,
        "detail": check.detail,
    }


def _point_to_json(point: ProofPoint) -> dict[str, object]:
    return {
        "target_filter_selectivity": point.target_filter_selectivity,
        "target_correlation": point.target_correlation,
        "measured_best": point.measured_best,
        "predicted_best": point.predicted_best,
        "prediction_match": point.prediction_match,
        "predicted_strategy_viable": point.predicted_strategy_viable,
        "postfilter_viable": point.postfilter_viable,
        "measured_best_latency_ms_p95": point.measured_best_latency_ms_p95,
        "predicted_latency_ms_p95": point.predicted_latency_ms_p95,
        "postfilter_latency_ms_p95": point.postfilter_latency_ms_p95,
        "speedup_vs_postfilter": point.speedup_vs_postfilter,
    }
