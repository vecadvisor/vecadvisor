from __future__ import annotations

import json

import pytest

from vecadvisor.bench.calibrate import run_synthetic_calibration
from vecadvisor.bench.datasets import generate_synthetic_dataset, generate_synthetic_queries
from vecadvisor.bench.runner import STRATEGY_EXACT, STRATEGY_ITERATIVE, STRATEGY_POSTFILTER
from vecadvisor.bench.validate import (
    estimate_local_selectivity,
    run_synthetic_validation,
    validation_report_to_json,
    write_validation_report,
)


def test_estimate_local_selectivity_reports_p10_and_median() -> None:
    dataset = generate_synthetic_dataset(
        n_rows=128,
        dim=4,
        n_clusters=4,
        filter_selectivity=0.25,
        correlation=0.7,
        seed=201,
    )
    queries = generate_synthetic_queries(dataset, n_queries=4, seed=202)

    local = estimate_local_selectivity(
        dataset=dataset,
        queries=queries,
        metric="l2",
        probe_rows=16,
        block_rows=16,
    )

    assert local.s_global == pytest.approx(dataset.observed_selectivity)
    assert 0.0 <= local.s_local_p10 <= local.s_local_median <= 1.0
    assert local.probe_rows == 16
    assert local.query_count == 4


def test_run_synthetic_validation_compares_prediction_to_measurement() -> None:
    fit = run_synthetic_calibration(
        rows=192,
        dim=4,
        queries=3,
        clusters=4,
        filter_selectivity=0.2,
        correlation=0.5,
        limit=4,
        block_rows=32,
        ef_sweep=(8, 16),
        seed=211,
    )
    dataset = generate_synthetic_dataset(
        n_rows=192,
        dim=4,
        n_clusters=4,
        filter_selectivity=0.2,
        correlation=0.5,
        seed=211,
    )
    queries = generate_synthetic_queries(dataset, n_queries=3, seed=212)

    report = run_synthetic_validation(
        dataset=dataset,
        queries=queries,
        calibration=fit.profile,
        k=4,
        ef_search=8,
        max_scan_tuples=64,
        probe_rows=16,
        block_rows=32,
        recall_target=0.8,
    )

    assert report.predicted_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert report.measured_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert isinstance(report.match, bool)
    assert [prediction.strategy for prediction in report.predictions] == [
        STRATEGY_EXACT,
        STRATEGY_POSTFILTER,
        STRATEGY_ITERATIVE,
    ]
    assert report.local_selectivity.s_local_p10 <= report.local_selectivity.s_local_median


def test_validation_report_serializes_and_writes_json(tmp_path) -> None:
    fit = run_synthetic_calibration(
        rows=96,
        dim=4,
        queries=2,
        clusters=4,
        filter_selectivity=0.25,
        correlation=0.4,
        limit=3,
        block_rows=16,
        ef_sweep=(6, 12),
        seed=221,
    )
    dataset = generate_synthetic_dataset(
        n_rows=96,
        dim=4,
        n_clusters=4,
        filter_selectivity=0.25,
        correlation=0.4,
        seed=221,
    )
    queries = generate_synthetic_queries(dataset, n_queries=2, seed=222)
    report = run_synthetic_validation(
        dataset=dataset,
        queries=queries,
        calibration=fit.profile,
        k=3,
        ef_search=6,
        max_scan_tuples=48,
        probe_rows=12,
        block_rows=16,
        recall_target=0.8,
    )
    path = tmp_path / "validation.json"

    write_validation_report(report, path, calibration_source="test-profile")

    payload = validation_report_to_json(report, calibration_source="test-profile")
    assert payload["validation"]["predicted_best"] == report.predicted_best
    assert payload["validation"]["measured_best"] == report.measured_best
    assert payload["calibration"]["source"] == "test-profile"
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["validation"] == payload["validation"]
