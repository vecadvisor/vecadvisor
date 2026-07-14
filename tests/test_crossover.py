from __future__ import annotations

import json
import math

import pytest

from vecadvisor.bench.crossover import (
    analyze_sweep_payload,
    crossover_analysis_to_json,
    load_sweep_payload,
    write_crossover_analysis,
)


def test_analyze_sweep_payload_detects_crossovers_and_prediction_rate(tmp_path) -> None:
    payload = _two_point_sweep_payload()

    analysis = analyze_sweep_payload(payload)

    assert analysis.backend == "synthetic"
    assert analysis.point_count == 2
    assert analysis.correlations == (0.0,)
    assert analysis.prediction_match_rate == pytest.approx(0.5)
    assert analysis.measured_win_counts == {"postfilter": 1, "iterative": 1}
    assert analysis.predicted_win_counts == {"postfilter": 1, "exact": 1}
    assert analysis.postfilter_failure_count == 1

    measured = analysis.measured_crossovers[0]
    assert measured.kind == "measured"
    assert measured.from_strategy == "postfilter"
    assert measured.to_strategy == "iterative"
    assert measured.estimated_selectivity == pytest.approx(math.sqrt(0.01 * 0.1))

    predicted = analysis.predicted_crossovers[0]
    assert predicted.kind == "predicted"
    assert predicted.from_strategy == "postfilter"
    assert predicted.to_strategy == "exact"

    json_path = tmp_path / "crossover.json"
    write_crossover_analysis(analysis, json_path)
    written = json.loads(json_path.read_text(encoding="utf-8"))
    assert written["analysis"]["postfilter_failure_count"] == 1
    assert written["measured_crossovers"][0]["to_strategy"] == "iterative"

    loaded = load_sweep_payload(json_path)
    with pytest.raises(ValueError, match="sweep"):
        analyze_sweep_payload(loaded)


def test_crossover_analysis_json_shape() -> None:
    analysis = analyze_sweep_payload(_two_point_sweep_payload())

    payload = crossover_analysis_to_json(analysis)

    assert payload["analysis"]["prediction_match_rate"] == pytest.approx(0.5)
    assert payload["points"][1]["postfilter_viable"] is False
    assert {row["kind"] for row in payload["regions"]} == {"measured", "predicted"}


def _two_point_sweep_payload() -> dict[str, object]:
    return {
        "sweep": {
            "backend": "synthetic",
            "recall_target": 0.9,
            "returns_k_target": 1.0,
        },
        "points": [
            _point(
                selectivity=0.01,
                measured_best="postfilter",
                predicted_best="postfilter",
                prediction_match=True,
                postfilter_recall=0.95,
                postfilter_returns_k=1.0,
            ),
            _point(
                selectivity=0.1,
                measured_best="iterative",
                predicted_best="exact",
                prediction_match=False,
                postfilter_recall=0.4,
                postfilter_returns_k=0.5,
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
            "probe_rows": 20,
            "query_count": 2,
        },
        "measured_best": measured_best,
        "predicted_best": predicted_best,
        "prediction_match": prediction_match,
        "strategies": [
            _strategy("exact", recall=1.0, returns_k=1.0, mean_ms=4.0, p95_ms=5.0),
            _strategy(
                "postfilter",
                recall=postfilter_recall,
                returns_k=postfilter_returns_k,
                mean_ms=1.0,
                p95_ms=2.0,
            ),
            _strategy("iterative", recall=0.92, returns_k=1.0, mean_ms=2.0, p95_ms=3.0),
        ],
    }


def _strategy(
    strategy: str,
    *,
    recall: float,
    returns_k: float,
    mean_ms: float,
    p95_ms: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "recall_at_k": recall,
        "returns_k_rate": returns_k,
        "latency_ms": {
            "mean": mean_ms,
            "p95": p95_ms,
        },
    }
