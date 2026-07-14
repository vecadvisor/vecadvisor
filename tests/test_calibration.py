from __future__ import annotations

import pytest

from vecadvisor.bench.calibrate import (
    calibration_fit_to_json,
    parse_ef_sweep,
    run_synthetic_calibration,
)
from vecadvisor.calibration import (
    CalibrationProfileError,
    calibration_profile_from_json,
    calibration_profile_to_json,
    load_calibration_profile,
    save_calibration_profile,
)
from vecadvisor.models import CalibrationProfile


def test_calibration_profile_round_trips_json_file(tmp_path) -> None:
    profile = CalibrationProfile(
        dataset_id="synthetic",
        hardware_id="local-ci",
        index_method="hnsw",
        c_d=0.02,
        c_scan=0.01,
        c_h=3.0,
        delta_strict=0.15,
        recall_curve=((40, 0.9), (80, 0.95), (160, 0.98)),
    )
    path = tmp_path / "calibration.json"

    save_calibration_profile(profile, path)

    assert load_calibration_profile(path) == profile
    assert calibration_profile_to_json(profile)["version"] == 1


def test_save_calibration_profile_creates_parent_directory(tmp_path) -> None:
    profile = CalibrationProfile(
        dataset_id="synthetic",
        hardware_id="local-ci",
        index_method="hnsw",
        c_d=0.02,
        c_scan=0.01,
        c_h=3.0,
        recall_curve=((40, 0.9),),
    )
    path = tmp_path / "profiles" / "local.json"

    save_calibration_profile(profile, path)

    assert load_calibration_profile(path) == profile


def test_calibration_profile_loader_sorts_valid_recall_curve() -> None:
    profile = calibration_profile_from_json(
        {
            "dataset_id": "synthetic",
            "hardware_id": "local-ci",
            "index_method": "hnsw",
            "c_d": 0.02,
            "c_scan": 0.01,
            "c_h": 3.0,
            "delta_strict": 0.15,
            "recall_curve": [[80, 0.95], [40, 0.9]],
        }
    )

    assert profile.recall_curve == ((40, 0.9), (80, 0.95))


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"index_method": "diskann"}, "index_method"),
        ({"c_d": 0.0}, "c_d"),
        ({"delta_strict": -0.1}, "delta_strict"),
        ({"recall_curve": [[40, 0.95], [80, 0.94]]}, "non-decreasing"),
        ({"extra": True}, "unknown"),
    ],
)
def test_calibration_profile_rejects_invalid_payloads(
    patch: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "dataset_id": "synthetic",
        "hardware_id": "local-ci",
        "index_method": "hnsw",
        "c_d": 0.02,
        "c_scan": 0.01,
        "c_h": 3.0,
        "delta_strict": 0.15,
        "recall_curve": [[40, 0.9], [80, 0.95]],
    }
    payload.update(patch)

    with pytest.raises(CalibrationProfileError, match=message):
        calibration_profile_from_json(payload)


def test_parse_ef_sweep_sorts_deduplicates_and_validates() -> None:
    assert parse_ef_sweep("80, 20, 20, 40") == (20, 40, 80)

    with pytest.raises(ValueError, match="positive integers"):
        parse_ef_sweep("20, nope")


def test_run_synthetic_calibration_fits_profile() -> None:
    fit = run_synthetic_calibration(
        rows=256,
        dim=6,
        queries=4,
        clusters=4,
        filter_selectivity=0.2,
        correlation=0.5,
        limit=5,
        block_rows=32,
        ef_sweep=(8, 16, 32),
        seed=101,
        dataset_id="synthetic-test",
        hardware_id="ci-cpu",
    )

    assert fit.profile.dataset_id == "synthetic-test"
    assert fit.profile.hardware_id == "ci-cpu"
    assert fit.profile.index_method == "hnsw"
    assert fit.profile.c_d > 0
    assert fit.profile.c_scan > 0
    assert fit.profile.c_h > 0
    assert fit.profile.delta_strict == pytest.approx(0.0)
    assert [ef for ef, _ in fit.profile.recall_curve] == [8, 16, 32]
    assert len(fit.reports) == 3

    payload = calibration_fit_to_json(fit)
    assert payload["profile"]["dataset_id"] == "synthetic-test"
    assert payload["ef_sweep"] == [8, 16, 32]
    assert len(payload["fit_reports"]) == 3
