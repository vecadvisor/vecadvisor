from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .models import CalibrationProfile

PROFILE_VERSION = 1
SUPPORTED_INDEX_METHODS = {"hnsw", "ivfflat"}


class CalibrationProfileError(ValueError):
    """Raised when a calibration profile is missing or invalid."""


def load_calibration_profile(path: Path) -> CalibrationProfile:
    """Load and validate a calibration profile JSON file."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CalibrationProfileError(f"could not read calibration profile: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CalibrationProfileError(f"calibration profile JSON is invalid: {exc}") from exc

    if not isinstance(raw, dict):
        raise CalibrationProfileError("calibration profile must be a JSON object")
    return calibration_profile_from_json(raw)


def save_calibration_profile(profile: CalibrationProfile, path: Path) -> None:
    """Write a calibration profile JSON file using the stable profile format."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration_profile_to_json(profile), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def calibration_profile_to_json(profile: CalibrationProfile) -> dict[str, object]:
    """Convert a CalibrationProfile to a JSON-serializable dict."""

    _validate_profile(profile)
    return {
        "version": PROFILE_VERSION,
        "dataset_id": profile.dataset_id,
        "hardware_id": profile.hardware_id,
        "index_method": profile.index_method,
        "c_d": profile.c_d,
        "c_scan": profile.c_scan,
        "c_h": profile.c_h,
        "delta_strict": profile.delta_strict,
        "recall_curve": [[ef, recall] for ef, recall in profile.recall_curve],
    }


def calibration_profile_from_json(payload: Mapping[str, Any]) -> CalibrationProfile:
    """Build and validate a CalibrationProfile from a JSON object."""

    allowed = {
        "version",
        "dataset_id",
        "hardware_id",
        "index_method",
        "c_d",
        "c_scan",
        "c_h",
        "delta_strict",
        "recall_curve",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CalibrationProfileError(f"unknown calibration profile fields: {', '.join(unknown)}")

    version = payload.get("version", PROFILE_VERSION)
    if version != PROFILE_VERSION:
        raise CalibrationProfileError(f"unsupported calibration profile version: {version}")

    profile = CalibrationProfile(
        dataset_id=_required_str(payload, "dataset_id"),
        hardware_id=_required_str(payload, "hardware_id"),
        index_method=_required_str(payload, "index_method"),
        c_d=_positive_float(payload, "c_d"),
        c_scan=_positive_float(payload, "c_scan"),
        c_h=_positive_float(payload, "c_h"),
        delta_strict=_nonnegative_float(payload, "delta_strict", default=0.0),
        recall_curve=_recall_curve(payload.get("recall_curve")),
    )
    _validate_profile(profile)
    return profile


def _validate_profile(profile: CalibrationProfile) -> None:
    if not profile.dataset_id:
        raise CalibrationProfileError("dataset_id must not be empty")
    if not profile.hardware_id:
        raise CalibrationProfileError("hardware_id must not be empty")
    if profile.index_method not in SUPPORTED_INDEX_METHODS:
        raise CalibrationProfileError(
            f"index_method must be one of: {', '.join(sorted(SUPPORTED_INDEX_METHODS))}"
        )
    _validate_positive("c_d", profile.c_d)
    _validate_positive("c_scan", profile.c_scan)
    _validate_positive("c_h", profile.c_h)
    _validate_nonnegative("delta_strict", profile.delta_strict)
    _validate_recall_curve(profile.recall_curve)


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CalibrationProfileError(f"{key} must be a non-empty string")
    return value.strip()


def _positive_float(payload: Mapping[str, Any], key: str) -> float:
    if key not in payload:
        raise CalibrationProfileError(f"{key} is required")
    value = _float_value(payload[key], key)
    _validate_positive(key, value)
    return value


def _nonnegative_float(payload: Mapping[str, Any], key: str, *, default: float) -> float:
    value = _float_value(payload.get(key, default), key)
    _validate_nonnegative(key, value)
    return value


def _float_value(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise CalibrationProfileError(f"{key} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationProfileError(f"{key} must be numeric") from exc
    if not math.isfinite(number):
        raise CalibrationProfileError(f"{key} must be finite")
    return number


def _recall_curve(value: Any) -> tuple[tuple[int, float], ...]:
    if not isinstance(value, list) or not value:
        raise CalibrationProfileError("recall_curve must be a non-empty list")

    points: list[tuple[int, float]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise CalibrationProfileError("recall_curve entries must be [ef, recall] pairs")
        ef_raw, recall_raw = item
        if isinstance(ef_raw, bool):
            raise CalibrationProfileError("recall_curve ef values must be positive integers")
        try:
            ef = int(ef_raw)
        except (TypeError, ValueError) as exc:
            raise CalibrationProfileError(
                "recall_curve ef values must be positive integers"
            ) from exc
        if ef != ef_raw and not (isinstance(ef_raw, float) and ef_raw.is_integer()):
            raise CalibrationProfileError("recall_curve ef values must be integers")
        recall = _float_value(recall_raw, "recall_curve recall")
        points.append((ef, recall))

    curve = tuple(sorted(points))
    _validate_recall_curve(curve)
    return curve


def _validate_positive(key: str, value: float) -> None:
    if value <= 0:
        raise CalibrationProfileError(f"{key} must be > 0")


def _validate_nonnegative(key: str, value: float) -> None:
    if value < 0:
        raise CalibrationProfileError(f"{key} must be >= 0")


def _validate_recall_curve(curve: tuple[tuple[int, float], ...]) -> None:
    if not curve:
        raise CalibrationProfileError("recall_curve must not be empty")
    previous_ef = 0
    previous_recall = 0.0
    for ef, recall in curve:
        if ef <= previous_ef:
            raise CalibrationProfileError("recall_curve ef values must be strictly increasing")
        if not 0.0 < recall <= 1.0:
            raise CalibrationProfileError("recall_curve recall values must be in (0, 1]")
        if recall < previous_recall:
            raise CalibrationProfileError("recall_curve recall values must be non-decreasing")
        previous_ef = ef
        previous_recall = recall
