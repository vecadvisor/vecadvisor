from __future__ import annotations

import json

import pytest

from vecadvisor.bench.crossover import analyze_sweep_payload
from vecadvisor.bench.proof import (
    build_proof_report,
    proof_report_to_json,
    write_proof_report,
)


def test_build_proof_report_passes_for_calibrated_safe_sweep(tmp_path) -> None:
    analysis = analyze_sweep_payload(_proof_sweep_payload())

    report = build_proof_report(analysis)

    assert report.passed is True
    assert report.point_count == 3
    assert report.selectivity_bin_count == 3
    assert report.prediction_match_rate == pytest.approx(1.0)
    assert report.postfilter_failure_count == 1
    assert report.safe_advisor_on_postfilter_failures == 1
    assert all(check.passed for check in report.checks)
    assert report.points[0].predicted_best == "exact"
    assert report.points[0].postfilter_viable is False
    assert report.points[0].predicted_strategy_viable is True

    payload = proof_report_to_json(report)
    assert payload["proof"]["passed"] is True
    assert payload["proof"]["mean_speedup_vs_postfilter"] == pytest.approx(
        (2.0 / 5.0 + 4.0 / 3.0 + 1.0) / 3
    )
    assert payload["checks"][0]["name"] == "minimum_points"
    assert payload["points"][0]["speedup_vs_postfilter"] == pytest.approx(2.0 / 5.0)

    out_path = tmp_path / "proof.json"
    write_proof_report(report, out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["proof"]["postfilter_failure_count"] == 1


def test_build_proof_report_fails_when_predictions_are_missing() -> None:
    payload = _proof_sweep_payload()
    for point in payload["points"]:
        assert isinstance(point, dict)
        point["predicted_best"] = None
        point["prediction_match"] = None
    analysis = analyze_sweep_payload(payload)

    report = build_proof_report(analysis)

    assert report.passed is False
    checks = {check.name: check for check in report.checks}
    assert checks["predictions_present"].passed is False
    assert checks["prediction_match_rate"].passed is False
    assert report.prediction_match_rate is None


def test_build_proof_report_fails_when_advisor_keeps_failed_postfilter() -> None:
    payload = _proof_sweep_payload()
    first_point = payload["points"][0]
    assert isinstance(first_point, dict)
    first_point["predicted_best"] = "postfilter"
    first_point["prediction_match"] = False
    analysis = analyze_sweep_payload(payload)

    report = build_proof_report(analysis, min_match_rate=0.5)

    assert report.passed is False
    checks = {check.name: check for check in report.checks}
    assert checks["advisor_safe_when_postfilter_fails"].passed is False
    assert report.safe_advisor_on_postfilter_failures == 0


def _proof_sweep_payload() -> dict[str, object]:
    return {
        "sweep": {
            "backend": "synthetic",
            "recall_target": 0.9,
            "returns_k_target": 1.0,
        },
        "points": [
            _point(
                selectivity=0.01,
                measured_best="exact",
                predicted_best="exact",
                prediction_match=True,
                postfilter_recall=0.3,
                postfilter_returns_k=0.5,
                exact_p95=5.0,
                postfilter_p95=2.0,
                iterative_p95=3.5,
            ),
            _point(
                selectivity=0.05,
                measured_best="iterative",
                predicted_best="iterative",
                prediction_match=True,
                postfilter_recall=0.95,
                postfilter_returns_k=1.0,
                exact_p95=8.0,
                postfilter_p95=4.0,
                iterative_p95=3.0,
            ),
            _point(
                selectivity=0.2,
                measured_best="postfilter",
                predicted_best="postfilter",
                prediction_match=True,
                postfilter_recall=0.95,
                postfilter_returns_k=1.0,
                exact_p95=8.0,
                postfilter_p95=1.0,
                iterative_p95=2.0,
            ),
        ],
    }


def _point(
    *,
    selectivity: float,
    measured_best: str,
    predicted_best: str,
    prediction_match: bool,
    postfilter_recall: float,
    postfilter_returns_k: float,
    exact_p95: float,
    postfilter_p95: float,
    iterative_p95: float,
) -> dict[str, object]:
    return {
        "target_filter_selectivity": selectivity,
        "target_correlation": 0.0,
        "dataset": {
            "observed_filter_selectivity": selectivity,
        },
        "local_selectivity": {
            "s_local_p10": selectivity,
            "s_local_median": selectivity,
        },
        "measured_best": measured_best,
        "predicted_best": predicted_best,
        "prediction_match": prediction_match,
        "strategies": [
            _strategy("exact", recall=1.0, returns_k=1.0, p95_ms=exact_p95),
            _strategy(
                "postfilter",
                recall=postfilter_recall,
                returns_k=postfilter_returns_k,
                p95_ms=postfilter_p95,
            ),
            _strategy("iterative", recall=0.95, returns_k=1.0, p95_ms=iterative_p95),
        ],
    }


def _strategy(
    strategy: str,
    *,
    recall: float,
    returns_k: float,
    p95_ms: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "recall_at_k": recall,
        "returns_k_rate": returns_k,
        "latency_ms": {
            "mean": p95_ms / 2,
            "p95": p95_ms,
        },
    }
