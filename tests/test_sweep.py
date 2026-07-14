from __future__ import annotations

import csv
import json

import pytest

from vecadvisor.bench.calibrate import run_synthetic_calibration
from vecadvisor.bench.runner import STRATEGY_EXACT, STRATEGY_ITERATIVE, STRATEGY_POSTFILTER
from vecadvisor.bench.sweep import (
    parse_float_sweep,
    run_synthetic_sweep,
    sweep_report_to_json,
    write_sweep_report,
)


def test_parse_float_sweep_sorts_deduplicates_and_validates() -> None:
    assert parse_float_sweep(
        "0.3,0.1,0.3",
        default=(0.2,),
        name="filter_selectivity",
        min_value=0.0,
        max_value=1.0,
        include_min=False,
        include_max=False,
    ) == (0.1, 0.3)
    assert parse_float_sweep(
        None,
        default=(-0.8, 0.0, 0.8),
        name="correlation",
        min_value=-1.0,
        max_value=1.0,
    ) == (-0.8, 0.0, 0.8)

    with pytest.raises(ValueError, match="filter_selectivity sweep values must be in"):
        parse_float_sweep(
            "0,0.2",
            default=(0.2,),
            name="filter_selectivity",
            min_value=0.0,
            max_value=1.0,
            include_min=False,
            include_max=False,
        )
    with pytest.raises(ValueError, match="correlation sweep values must be numeric"):
        parse_float_sweep(
            "cold",
            default=(0.0,),
            name="correlation",
            min_value=-1.0,
            max_value=1.0,
        )


def test_run_synthetic_sweep_reports_measured_winners_and_local_selectivity() -> None:
    report = run_synthetic_sweep(
        rows=96,
        dim=4,
        queries=2,
        clusters=4,
        filter_selectivities=(0.2, 0.3),
        correlations=(0.0,),
        limit=3,
        ef_search=6,
        max_scan_tuples=48,
        probe_rows=12,
        block_rows=16,
        seed=301,
    )

    assert report.backend == "synthetic"
    assert len(report.points) == 2
    assert report.calibration is None
    for point in report.points:
        assert point.measured_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
        assert point.predicted_best is None
        assert point.prediction_match is None
        assert point.predictions == ()
        assert point.local_selectivity.query_count == 2
        assert 0.0 <= point.local_selectivity.s_local_p10 <= 1.0

    payload = sweep_report_to_json(report, calibration_source="none")
    assert payload["sweep"]["points"] == 2
    assert payload["calibration"]["source"] == "none"
    assert payload["points"][0]["predicted_best"] is None


def test_synthetic_sweep_with_calibration_adds_predictions_and_writes_reports(
    tmp_path,
) -> None:
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
        seed=311,
    )
    report = run_synthetic_sweep(
        rows=96,
        dim=4,
        queries=2,
        clusters=4,
        filter_selectivities=(0.25,),
        correlations=(0.4,),
        limit=3,
        ef_search=6,
        max_scan_tuples=48,
        probe_rows=12,
        block_rows=16,
        seed=311,
        calibration=fit.profile,
    )

    assert report.calibration == fit.profile
    assert len(report.points) == 1
    point = report.points[0]
    assert point.predicted_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert isinstance(point.prediction_match, bool)
    assert [prediction.strategy for prediction in point.predictions] == [
        STRATEGY_EXACT,
        STRATEGY_POSTFILTER,
        STRATEGY_ITERATIVE,
    ]

    json_path = tmp_path / "sweep.json"
    csv_path = tmp_path / "sweep.csv"
    write_sweep_report(
        report,
        json_path,
        output_format="json",
        calibration_source="test-profile",
    )
    write_sweep_report(
        report,
        csv_path,
        output_format="csv",
        calibration_source="test-profile",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["calibration"]["source"] == "test-profile"
    assert payload["points"][0]["predicted_best"] == point.predicted_best

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["strategy"] for row in rows] == [
        STRATEGY_EXACT,
        STRATEGY_POSTFILTER,
        STRATEGY_ITERATIVE,
    ]
    assert rows[0]["predicted_best"] == point.predicted_best
