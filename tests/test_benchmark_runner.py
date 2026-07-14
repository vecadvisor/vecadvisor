from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from vecadvisor.bench.datasets import generate_synthetic_dataset, generate_synthetic_queries
from vecadvisor.bench.runner import (
    STRATEGY_EXACT,
    STRATEGY_ITERATIVE,
    STRATEGY_POSTFILTER,
    benchmark_report_to_json,
    infer_output_format,
    parse_strategy_list,
    run_synthetic_benchmark,
    write_benchmark_report,
)


def test_run_synthetic_benchmark_reports_strategy_metrics() -> None:
    dataset = generate_synthetic_dataset(
        n_rows=256,
        dim=6,
        n_clusters=4,
        filter_selectivity=0.1,
        correlation=0.9,
        seed=21,
    )
    queries = generate_synthetic_queries(
        dataset,
        n_queries=5,
        seed=22,
        cluster_policy="filter_cold",
    )

    report = run_synthetic_benchmark(
        dataset=dataset,
        queries=queries,
        k=5,
        strategies=(STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE),
        ef_search=8,
        max_scan_tuples=128,
        block_rows=32,
    )

    metrics_by_strategy = {metrics.strategy: metrics for metrics in report.strategies}
    assert set(metrics_by_strategy) == {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert report.dataset["query_policy"] == "filter_cold"
    assert report.ground_truth["block_rows"] == 32
    assert metrics_by_strategy[STRATEGY_EXACT].recall_at_k == pytest.approx(1.0)
    assert 0.0 <= metrics_by_strategy[STRATEGY_POSTFILTER].recall_at_k <= 1.0
    assert (
        metrics_by_strategy[STRATEGY_ITERATIVE].recall_at_k
        >= metrics_by_strategy[STRATEGY_POSTFILTER].recall_at_k
    )
    assert metrics_by_strategy[STRATEGY_POSTFILTER].params["ef_search"] == 8
    assert metrics_by_strategy[STRATEGY_ITERATIVE].params["max_scan_tuples"] == 128


def test_benchmark_report_serializes_json_and_csv(tmp_path) -> None:
    dataset = generate_synthetic_dataset(n_rows=64, dim=4, n_clusters=4, seed=31)
    queries = generate_synthetic_queries(dataset, n_queries=2, seed=32)
    report = run_synthetic_benchmark(
        dataset=dataset,
        queries=queries,
        k=3,
        strategies=(STRATEGY_EXACT, STRATEGY_POSTFILTER),
        ef_search=5,
        block_rows=16,
    )

    json_path = tmp_path / "bench.json"
    csv_path = tmp_path / "bench.csv"
    write_benchmark_report(report, json_path, output_format="json")
    write_benchmark_report(report, csv_path, output_format="csv")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dataset"]["rows"] == 64
    assert [row["strategy"] for row in payload["strategies"]] == [
        STRATEGY_EXACT,
        STRATEGY_POSTFILTER,
    ]

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["strategy"] for row in rows] == [STRATEGY_EXACT, STRATEGY_POSTFILTER]
    assert json.loads(rows[1]["params"])["ef_search"] == 5


def test_parse_strategy_list_and_output_format_validation() -> None:
    assert parse_strategy_list(None) == (STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE)
    assert parse_strategy_list("exact, iterative") == (STRATEGY_EXACT, STRATEGY_ITERATIVE)
    assert infer_output_format(None, "auto") == "json"
    assert infer_output_format(None, "csv") == "csv"
    assert infer_output_format(Path("x.csv"), "auto") == "csv"

    with pytest.raises(ValueError, match="unknown benchmark strategy"):
        parse_strategy_list("exact,unknown")
    with pytest.raises(ValueError, match="output format"):
        infer_output_format(None, "xml")


def test_benchmark_report_to_json_shape() -> None:
    dataset = generate_synthetic_dataset(n_rows=32, dim=3, n_clusters=2, seed=41)
    queries = generate_synthetic_queries(dataset, n_queries=1, seed=42)
    report = run_synthetic_benchmark(
        dataset=dataset,
        queries=queries,
        k=2,
        strategies=(STRATEGY_EXACT,),
        block_rows=8,
    )

    payload = benchmark_report_to_json(report)
    assert payload["strategies"][0]["latency_ms"]["p95"] >= 0.0
    assert "strategy-semantics simulations" in payload["notes"][1]
