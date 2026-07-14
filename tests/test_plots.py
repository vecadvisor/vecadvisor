from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from vecadvisor.bench.crossover import analyze_sweep_payload
from vecadvisor.bench.plots import (
    load_benchmark_payload,
    render_benchmark_pareto_svg,
    render_crossover_svg,
    write_benchmark_pareto_svg,
    write_crossover_svg,
)


def test_render_crossover_svg_outputs_valid_chart(tmp_path) -> None:
    analysis = analyze_sweep_payload(_chart_sweep_payload())

    svg = render_crossover_svg(analysis, title="Test Money Chart", width=900)

    ET.fromstring(svg)
    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "Test Money Chart" in svg
    assert "measured crossover" in svg
    assert "predicted crossover" in svg
    assert "p95 latency ms" in svg
    assert "recall@k" in svg
    assert "postfilter recall below target" in svg

    path = tmp_path / "chart.svg"
    write_crossover_svg(analysis, path, title="Test Money Chart", width=900)
    written = path.read_text(encoding="utf-8")
    assert written == svg


def test_render_crossover_svg_validates_width() -> None:
    analysis = analyze_sweep_payload(_chart_sweep_payload())

    with pytest.raises(ValueError, match="width"):
        render_crossover_svg(analysis, width=700)


def test_render_benchmark_pareto_svg_outputs_valid_chart(tmp_path) -> None:
    payload = _benchmark_payload()

    svg = render_benchmark_pareto_svg(payload, title="Test Pareto", width=820)

    ET.fromstring(svg)
    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "Test Pareto" in svg
    assert "Recall versus QPS Pareto" in svg
    assert "QPS, higher is better" in svg
    assert "Pareto frontier" in svg
    assert "returns-k below target" in svg
    assert "postfilter" in svg

    path = tmp_path / "pareto.svg"
    write_benchmark_pareto_svg(payload, path, title="Test Pareto", width=820)
    assert path.read_text(encoding="utf-8") == svg

    report_path = tmp_path / "benchmark.json"
    report_path.write_text('{"dataset": {}, "strategies": []}', encoding="utf-8")
    loaded = load_benchmark_payload(report_path)
    assert loaded["strategies"] == []


def test_render_benchmark_pareto_svg_validates_input() -> None:
    with pytest.raises(ValueError, match="width"):
        render_benchmark_pareto_svg(_benchmark_payload(), width=640)

    with pytest.raises(ValueError, match="strategy"):
        render_benchmark_pareto_svg({"dataset": {}, "strategies": []})


def _chart_sweep_payload() -> dict[str, object]:
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
            ),
            _point(
                selectivity=0.1,
                measured_best="iterative",
                predicted_best="exact",
                prediction_match=False,
                postfilter_recall=0.4,
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
) -> dict[str, object]:
    return {
        "target_filter_selectivity": selectivity,
        "target_correlation": 0.0,
        "dataset": {"observed_filter_selectivity": selectivity},
        "local_selectivity": {
            "s_local_p10": selectivity,
            "s_local_median": selectivity,
        },
        "measured_best": measured_best,
        "predicted_best": predicted_best,
        "prediction_match": prediction_match,
        "strategies": [
            _strategy("exact", recall=1.0, returns_k=1.0, mean_ms=4.0, p95_ms=5.0),
            _strategy(
                "postfilter",
                recall=postfilter_recall,
                returns_k=1.0,
                mean_ms=1.0,
                p95_ms=2.0,
            ),
            _strategy("iterative", recall=0.95, returns_k=1.0, mean_ms=2.0, p95_ms=3.0),
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


def _benchmark_payload() -> dict[str, object]:
    return {
        "dataset": {
            "id": "synthetic",
            "rows": 1000,
            "queries": 10,
        },
        "ground_truth": {
            "metric": "l2",
            "k": 10,
        },
        "strategies": [
            _benchmark_strategy(
                "exact",
                recall=1.0,
                returns_k=1.0,
                query_count=10,
                total_ms=100.0,
                p95_ms=12.0,
            ),
            _benchmark_strategy(
                "postfilter",
                recall=0.4,
                returns_k=0.5,
                query_count=10,
                total_ms=20.0,
                p95_ms=3.0,
            ),
            _benchmark_strategy(
                "iterative",
                recall=0.95,
                returns_k=1.0,
                query_count=10,
                total_ms=40.0,
                p95_ms=5.0,
            ),
        ],
    }


def _benchmark_strategy(
    strategy: str,
    *,
    recall: float,
    returns_k: float,
    query_count: int,
    total_ms: float,
    p95_ms: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "params": {},
        "query_count": query_count,
        "recall_at_k": recall,
        "returns_k_rate": returns_k,
        "result_count_mean": 10.0 * returns_k,
        "latency_ms": {
            "total": total_ms,
            "mean": total_ms / query_count,
            "p50": total_ms / query_count,
            "p95": p95_ms,
            "p99": p95_ms,
        },
    }
