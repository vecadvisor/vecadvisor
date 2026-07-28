from __future__ import annotations

import importlib

import pytest

bench = importlib.import_module("tools.native_distance_benchmark")
compare = importlib.import_module("tools.native_distance_compare")


def test_generated_matrix_is_deterministic_float32() -> None:
    np = pytest.importorskip("numpy")

    first = bench.generated_matrix(3, 4, bench.CORPUS_SALT)
    second = bench.generated_matrix(3, 4, bench.CORPUS_SALT)

    assert first.dtype == np.float32
    assert first.shape == (3, 4)
    assert np.array_equal(first, second)
    assert float(first.min()) >= -1.0
    assert float(first.max()) < 1.0


def test_numpy_metric_checksum_rejects_unknown_metric() -> None:
    with pytest.raises(ValueError, match="unsupported metric"):
        bench.numpy_metric_checksum("unknown", rows=2, queries=1, dim=3)


def test_validate_numpy_checksums_accepts_close_native_payload() -> None:
    rows = 8
    queries = 2
    dim = 5
    reference = bench.numpy_metric_checksum("l2", rows, queries, dim)
    payload = {
        "data_generator": bench.GENERATOR_ID,
        "rows": rows,
        "queries": queries,
        "dim": dim,
        "results": [
            {
                "metric": "l2",
                "scalar": {"checksum": reference},
                "dispatch": {"checksum": reference * 1.000001},
            }
        ],
    }

    validations = bench.validate_numpy_checksums(payload)

    assert len(validations) == 1
    assert validations[0].passed is True
    assert validations[0].dispatch_abs_error <= validations[0].tolerance


def test_validate_numpy_checksums_rejects_bad_payload() -> None:
    payload = {
        "data_generator": bench.GENERATOR_ID,
        "rows": 4,
        "queries": 1,
        "dim": 3,
        "results": [
            {
                "metric": "ip",
                "scalar": {"checksum": 1.0e9},
                "dispatch": {"checksum": 1.0e9},
            }
        ],
    }

    validations = bench.validate_numpy_checksums(payload)

    assert validations[0].passed is False


def test_render_report_and_chart_include_key_fields() -> None:
    payload = {
        "rows": 128,
        "queries": 4,
        "dim": 31,
        "iterations": 1,
        "capabilities": {"avx2_compiled": True, "avx2_runtime": False},
        "results": [
            {
                "metric": "l2",
                "selected_kernel": "scalar",
                "scalar": {"ns_per_distance": 12.0},
                "dispatch": {"ns_per_distance": 10.0},
                "speedup_vs_scalar": 1.2,
                "max_abs_error_vs_scalar": 0.0,
            }
        ],
    }
    validation = bench.NumpyValidation(
        metric="l2",
        reference_checksum=10.0,
        scalar_checksum=10.0,
        dispatch_checksum=10.0,
        scalar_abs_error=0.0,
        dispatch_abs_error=0.0,
        tolerance=0.01,
        passed=True,
    )

    markdown = bench.render_markdown_report(payload, [validation])
    svg = bench.render_svg_chart(payload, [validation])

    assert "Native Distance Kernel Benchmark" in markdown
    assert "| l2 | scalar | 12.000 | 10.000 | 1.200x" in markdown
    assert "NumPy Correctness Check" in markdown
    assert "<svg" in svg
    assert "nanoseconds per distance" in svg


def test_compare_combines_avx2_and_scalar_payloads() -> None:
    avx2_payload = _native_payload(
        avx2_compiled=True,
        avx2_runtime=True,
        selected_kernel="avx2",
        dispatch_ns=4.0,
    )
    scalar_payload = _native_payload(
        avx2_compiled=False,
        avx2_runtime=False,
        selected_kernel="scalar",
        dispatch_ns=10.0,
    )

    combined = compare.build_combined_payload(
        avx2_payload,
        scalar_payload,
        avx2_source="avx2.json",
        scalar_source="scalar.json",
    )
    markdown = compare.render_markdown_report(combined)
    svg = compare.render_svg_chart(combined)

    assert combined["summary"][0]["speedup_vs_scalar_fallback"] == pytest.approx(2.5)
    assert combined["summary"][0]["numpy_checks_passed"] is True
    assert "scalar-only fallback build" in markdown
    assert "| l2 | avx2 | 4.000 | 10.000 | 2.500x" in markdown
    assert "<svg" in svg
    assert "AVX2 dispatch" in svg


def test_compare_rejects_mismatched_payloads() -> None:
    avx2_payload = _native_payload(
        avx2_compiled=True,
        avx2_runtime=True,
        selected_kernel="avx2",
        dispatch_ns=4.0,
    )
    scalar_payload = _native_payload(
        avx2_compiled=False,
        avx2_runtime=False,
        selected_kernel="scalar",
        dispatch_ns=10.0,
    )
    scalar_payload["dim"] = 256

    with pytest.raises(ValueError, match="dim"):
        compare.build_combined_payload(
            avx2_payload,
            scalar_payload,
            avx2_source="avx2.json",
            scalar_source="scalar.json",
        )


def _native_payload(
    *,
    avx2_compiled: bool,
    avx2_runtime: bool,
    selected_kernel: str,
    dispatch_ns: float,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "rows": 128,
        "queries": 4,
        "dim": 64,
        "iterations": 2,
        "capabilities": {
            "avx2_compiled": avx2_compiled,
            "avx2_runtime": avx2_runtime,
        },
        "results": [
            {
                "metric": "l2",
                "selected_kernel": selected_kernel,
                "scalar": {"ns_per_distance": 12.0},
                "dispatch": {"ns_per_distance": dispatch_ns},
                "speedup_vs_scalar": 12.0 / dispatch_ns,
                "max_abs_error_vs_scalar": 0.0,
            }
        ],
        "python_wrapper": {
            "platform": "test-platform",
            "numpy_validation": [{"metric": "l2", "passed": True}],
        },
    }
