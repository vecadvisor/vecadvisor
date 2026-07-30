from __future__ import annotations

import argparse
import html
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CORPUS_SALT = 0xA511E9B3
QUERY_SALT = 0x63D83595
GENERATOR_ID = "vecadvisor_native_v1"


@dataclass(frozen=True)
class NumpyValidation:
    metric: str
    reference_checksum: float
    scalar_checksum: float
    dispatch_checksum: float
    scalar_abs_error: float
    dispatch_abs_error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class ExternalBaselineResult:
    name: str
    metric: str
    available: bool
    reason: str | None
    seconds: float | None
    checksum: float | None
    distances: int
    ns_per_distance: float | None
    distances_per_second: float | None
    checksum_abs_error: float | None
    tolerance: float | None
    passed: bool | None


def generated_matrix(count: int, dim: int, salt: int) -> Any:
    np = _numpy()
    rows = np.arange(count, dtype=np.uint64)[:, None] + np.uint64(1)
    cols = np.arange(dim, dtype=np.uint64)[None, :] + np.uint64(1)
    mixed = (
        (rows * np.uint64(2654435761))
        ^ (cols * np.uint64(2246822519))
        ^ (np.uint64(salt) * np.uint64(3266489917))
    )
    mixed ^= mixed >> np.uint64(16)
    mixed *= np.uint64(2246822519)
    mixed ^= mixed >> np.uint64(13)
    scaled = (mixed & np.uint64(0xFFFF)).astype(np.float32) / np.float32(32768.0)
    return scaled - np.float32(1.0)


def numpy_metric_checksum(metric: str, rows: int, queries: int, dim: int) -> float:
    np = _numpy()
    corpus = generated_matrix(rows, dim, CORPUS_SALT)
    query_vectors = generated_matrix(queries, dim, QUERY_SALT)
    checksum = 0.0
    for query in query_vectors:
        if metric == "l2":
            distances = np.square(corpus - query, dtype=np.float32).sum(axis=1, dtype=np.float32)
        elif metric == "ip":
            distances = np.multiply(corpus, query, dtype=np.float32).sum(axis=1, dtype=np.float32)
        elif metric == "cosine":
            dot = np.multiply(corpus, query, dtype=np.float32).sum(axis=1, dtype=np.float32)
            corpus_norm = np.sqrt(np.square(corpus, dtype=np.float32).sum(axis=1, dtype=np.float32))
            query_norm = np.sqrt(np.square(query, dtype=np.float32).sum(dtype=np.float32))
            denominator = np.maximum(corpus_norm * query_norm, np.float32(1.0e-12))
            distances = np.float32(1.0) - dot / denominator
        else:
            raise ValueError(f"unsupported metric: {metric}")
        checksum += float(distances.astype(np.float64).sum())
    return checksum


def run_external_baselines(
    native_payload: dict[str, Any],
    baseline_names: Sequence[str],
) -> list[ExternalBaselineResult]:
    rows = int(native_payload["rows"])
    queries = int(native_payload["queries"])
    dim = int(native_payload["dim"])
    iterations = int(native_payload.get("iterations", 1))
    metrics = [str(result["metric"]) for result in native_payload["results"]]
    results: list[ExternalBaselineResult] = []
    for baseline_name in baseline_names:
        if baseline_name == "simsimd":
            results.extend(
                _run_simsimd_baseline(
                    rows=rows,
                    queries=queries,
                    dim=dim,
                    iterations=iterations,
                    metrics=metrics,
                )
            )
        else:
            raise ValueError(f"unsupported external baseline: {baseline_name}")
    return results


def _run_simsimd_baseline(
    *,
    rows: int,
    queries: int,
    dim: int,
    iterations: int,
    metrics: Sequence[str],
) -> list[ExternalBaselineResult]:
    try:
        import simsimd
    except ImportError as exc:
        return [
            ExternalBaselineResult(
                name="simsimd",
                metric=metric,
                available=False,
                reason=str(exc),
                seconds=None,
                checksum=None,
                distances=iterations * queries * rows,
                ns_per_distance=None,
                distances_per_second=None,
                checksum_abs_error=None,
                tolerance=None,
                passed=None,
            )
            for metric in metrics
        ]

    np = _numpy()
    corpus = generated_matrix(rows, dim, CORPUS_SALT)
    query_vectors = generated_matrix(queries, dim, QUERY_SALT)
    simsimd_metrics = {
        "l2": "sqeuclidean",
        "ip": "inner",
        "cosine": "cosine",
    }
    results: list[ExternalBaselineResult] = []
    for metric in metrics:
        simsimd_metric = simsimd_metrics.get(metric)
        if simsimd_metric is None:
            results.append(
                ExternalBaselineResult(
                    name="simsimd",
                    metric=metric,
                    available=False,
                    reason=f"metric {metric} is not supported by SimSIMD baseline",
                    seconds=None,
                    checksum=None,
                    distances=iterations * queries * rows,
                    ns_per_distance=None,
                    distances_per_second=None,
                    checksum_abs_error=None,
                    tolerance=None,
                    passed=None,
                )
            )
            continue

        checksum = 0.0
        out = np.empty((queries, rows), dtype=np.float32)
        start = time.perf_counter()
        for _ in range(iterations):
            simsimd.cdist(query_vectors, corpus, simsimd_metric, out=out)
            checksum += float(out.sum(dtype=np.float64))
        seconds = time.perf_counter() - start
        distance_count = iterations * queries * rows
        reference = numpy_metric_checksum(metric, rows, queries, dim) * iterations
        tolerance = max(1.0e-2, abs(reference) * 5.0e-4)
        checksum_error = abs(checksum - reference)
        results.append(
            ExternalBaselineResult(
                name="simsimd",
                metric=metric,
                available=True,
                reason=None,
                seconds=seconds,
                checksum=checksum,
                distances=distance_count,
                ns_per_distance=seconds * 1.0e9 / distance_count,
                distances_per_second=distance_count / seconds if seconds > 0.0 else None,
                checksum_abs_error=checksum_error,
                tolerance=tolerance,
                passed=checksum_error <= tolerance,
            )
        )
    return results


def validate_numpy_checksums(native_payload: dict[str, Any]) -> list[NumpyValidation]:
    if native_payload.get("data_generator") != GENERATOR_ID:
        raise ValueError("native payload uses an unknown data generator")

    rows = int(native_payload["rows"])
    queries = int(native_payload["queries"])
    dim = int(native_payload["dim"])
    iterations = int(native_payload.get("iterations", 1))
    validations: list[NumpyValidation] = []
    for result in native_payload["results"]:
        metric = str(result["metric"])
        reference = numpy_metric_checksum(metric, rows, queries, dim) * iterations
        scalar_checksum = float(result["scalar"]["checksum"])
        dispatch_checksum = float(result["dispatch"]["checksum"])
        scalar_error = abs(scalar_checksum - reference)
        dispatch_error = abs(dispatch_checksum - reference)
        tolerance = max(1.0e-2, abs(reference) * 5.0e-4)
        validations.append(
            NumpyValidation(
                metric=metric,
                reference_checksum=reference,
                scalar_checksum=scalar_checksum,
                dispatch_checksum=dispatch_checksum,
                scalar_abs_error=scalar_error,
                dispatch_abs_error=dispatch_error,
                tolerance=tolerance,
                passed=scalar_error <= tolerance and dispatch_error <= tolerance,
            )
        )
    return validations


def render_markdown_report(payload: dict[str, Any], validations: Sequence[NumpyValidation]) -> str:
    external_baselines = _external_baselines_from_payload(payload)
    has_simsimd = any(
        baseline.name == "simsimd" and baseline.available for baseline in external_baselines
    )
    lines = [
        "# Native Distance Kernel Benchmark",
        "",
        "This report is generated by `tools/native_distance_benchmark.py`.",
        "",
        (
            f"- rows: `{payload['rows']}`\n"
            f"- queries: `{payload['queries']}`\n"
            f"- dimensions: `{payload['dim']}`\n"
            f"- iterations: `{payload['iterations']}`\n"
            f"- AVX2 compiled: `{payload['capabilities']['avx2_compiled']}`\n"
            f"- AVX2 available at runtime: `{payload['capabilities']['avx2_runtime']}`"
        ),
        "",
        (
            "| metric | selected kernel | scalar ns/dist | dispatch ns/dist | "
            "speedup | max abs error | NumPy check |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    validation_by_metric = {validation.metric: validation for validation in validations}
    for result in payload["results"]:
        validation = validation_by_metric.get(str(result["metric"]))
        check = "not run"
        if validation is not None:
            check = "pass" if validation.passed else "fail"
        lines.append(
            "| {metric} | {kernel} | {scalar_ns:.3f} | {dispatch_ns:.3f} | "
            "{speedup:.3f}x | {error:.6g} | {check} |".format(
                metric=result["metric"],
                kernel=result["selected_kernel"],
                scalar_ns=float(result["scalar"]["ns_per_distance"]),
                dispatch_ns=float(result["dispatch"]["ns_per_distance"]),
                speedup=float(result["speedup_vs_scalar"]),
                error=float(result["max_abs_error_vs_scalar"]),
                check=check,
            )
        )

    if validations:
        lines.extend(
            [
                "",
                "## NumPy Correctness Check",
                "",
                (
                    "The Python wrapper regenerates the same deterministic `float32` matrices "
                    "and compares native scalar and dispatch checksums against NumPy."
                ),
                "",
                (
                    "| metric | reference checksum | scalar abs error | "
                    "dispatch abs error | tolerance |"
                ),
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for validation in validations:
            lines.append(
                f"| {validation.metric} | {validation.reference_checksum:.9g} | "
                f"{validation.scalar_abs_error:.6g} | "
                f"{validation.dispatch_abs_error:.6g} | {validation.tolerance:.6g} |"
            )

    if external_baselines:
        lines.extend(
            [
                "",
                "## External Baselines",
                "",
                (
                    "| baseline | metric | available | ns/dist | checksum abs error | "
                    "tolerance | check |"
                ),
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for baseline in external_baselines:
            lines.append(
                "| {name} | {metric} | {available} | {ns} | {error} | {tolerance} | "
                "{check} |".format(
                    name=baseline.name,
                    metric=baseline.metric,
                    available="yes" if baseline.available else "no",
                    ns=(
                        "n/a"
                        if baseline.ns_per_distance is None
                        else f"{baseline.ns_per_distance:.3f}"
                    ),
                    error=(
                        "n/a"
                        if baseline.checksum_abs_error is None
                        else f"{baseline.checksum_abs_error:.6g}"
                    ),
                    tolerance=(
                        "n/a"
                        if baseline.tolerance is None
                        else f"{baseline.tolerance:.6g}"
                    ),
                    check=_baseline_check_text(baseline),
                )
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `dispatch` is the runtime-selected kernel path used by future bindings.",
            "- `scalar` remains the correctness baseline for every SIMD implementation.",
            *(
                [
                    "- `simsimd` is an optional external baseline and is not a VecAdvisor "
                    "runtime dependency.",
                ]
                if has_simsimd
                else []
            ),
            "- Speedups are hardware-specific; regenerate this report on the target host.",
            "",
        ]
    )
    return "\n".join(lines)


def render_svg_chart(payload: dict[str, Any], validations: Sequence[NumpyValidation]) -> str:
    del validations
    width = 960
    height = 520
    margin_left = 82
    margin_right = 32
    margin_top = 76
    margin_bottom = 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    results = list(payload["results"])
    max_value = max(
        max(
            float(result["scalar"]["ns_per_distance"]),
            float(result["dispatch"]["ns_per_distance"]),
        )
        for result in results
    )
    y_max = max(max_value * 1.15, 1.0)
    group_width = plot_width / max(len(results), 1)
    bar_width = min(54.0, group_width * 0.24)

    def y(value: float) -> float:
        return margin_top + plot_height - (value / y_max) * plot_height

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img">'
        ),
        "<title>VecAdvisor native distance kernel benchmark</title>",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        (
            f'<text x="{margin_left}" y="34" font-family="Arial, sans-serif" '
            'font-size="24" font-weight="700">VecAdvisor native distance kernels</text>'
        ),
        (
            f'<text x="{margin_left}" y="58" font-family="Arial, sans-serif" '
            'font-size="12" fill="#334155">'
            f"rows: {payload['rows']} | queries: {payload['queries']} | dim: {payload['dim']} | "
            f"iterations: {payload['iterations']}</text>"
        ),
    ]

    for tick in range(5):
        value = y_max * tick / 4.0
        tick_y = y(value)
        lines.append(
            f'<line x1="{margin_left}" y1="{tick_y:.2f}" x2="{width - margin_right}" '
            'y2="{tick_y:.2f}" stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{tick_y + 4:.2f}" text-anchor="end" '
            'font-family="Arial, sans-serif" font-size="11" fill="#475569">'
            f"{value:.1f}</text>"
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" '
        f'y2="{margin_top + plot_height}" stroke="#94a3b8" stroke-width="1"/>'
    )
    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" '
        'font-family="Arial, sans-serif" font-size="12" fill="#334155" '
        'transform="rotate(-90 24 '
        f'{margin_top + plot_height / 2:.2f})">nanoseconds per distance</text>'
    )

    for index, result in enumerate(results):
        center = margin_left + group_width * (index + 0.5)
        scalar_value = float(result["scalar"]["ns_per_distance"])
        dispatch_value = float(result["dispatch"]["ns_per_distance"])
        scalar_height = margin_top + plot_height - y(scalar_value)
        dispatch_height = margin_top + plot_height - y(dispatch_value)
        scalar_x = center - bar_width - 4
        dispatch_x = center + 4
        baseline = margin_top + plot_height
        lines.extend(
            [
                (
                    f'<rect x="{scalar_x:.2f}" y="{y(scalar_value):.2f}" width="{bar_width:.2f}" '
                    f'height="{scalar_height:.2f}" fill="#2563eb"/>'
                ),
                (
                    f'<rect x="{dispatch_x:.2f}" y="{y(dispatch_value):.2f}" '
                    f'width="{bar_width:.2f}" '
                    f'height="{dispatch_height:.2f}" fill="#059669"/>'
                ),
                (
                    f'<text x="{center:.2f}" y="{baseline + 24:.2f}" text-anchor="middle" '
                    'font-family="Arial, sans-serif" font-size="13" fill="#0f172a">'
                    f"{html.escape(str(result['metric']))}</text>"
                ),
                (
                    f'<text x="{center:.2f}" y="{baseline + 42:.2f}" text-anchor="middle" '
                    'font-family="Arial, sans-serif" font-size="11" fill="#475569">'
                    f"{html.escape(str(result['selected_kernel']))}</text>"
                ),
            ]
        )

    lines.extend(
        [
            '<rect x="690" y="24" width="12" height="12" fill="#2563eb"/>',
            (
                '<text x="708" y="34" font-family="Arial, sans-serif" '
                'font-size="12" fill="#334155">scalar</text>'
            ),
            '<rect x="760" y="24" width="12" height="12" fill="#059669"/>',
            (
                '<text x="778" y="34" font-family="Arial, sans-serif" '
                'font-size="12" fill="#334155">dispatch</text>'
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines)


def merge_payload(
    payload: dict[str, Any],
    validations: Sequence[NumpyValidation],
    external_baselines: Sequence[ExternalBaselineResult] = (),
) -> dict[str, Any]:
    merged = dict(payload)
    merged["python_wrapper"] = {
        "platform": platform.platform(),
        "numpy_validation": [
            {
                "metric": validation.metric,
                "reference_checksum": validation.reference_checksum,
                "scalar_checksum": validation.scalar_checksum,
                "dispatch_checksum": validation.dispatch_checksum,
                "scalar_abs_error": validation.scalar_abs_error,
                "dispatch_abs_error": validation.dispatch_abs_error,
                "tolerance": validation.tolerance,
                "passed": validation.passed,
            }
            for validation in validations
        ],
        "external_baselines": [
            _external_baseline_to_json(baseline) for baseline in external_baselines
        ],
    }
    return merged


def configure_and_build(
    build_dir: Path,
    build_type: str,
    *,
    source_dir: Path,
    enable_avx2: bool,
) -> None:
    command = [
        "cmake",
        "-S",
        str(source_dir / "native"),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    if not enable_avx2:
        command.append("-DVECADVISOR_NATIVE_ENABLE_AVX2=OFF")
    _run(command)
    _run(["cmake", "--build", str(build_dir), "--config", build_type, "--parallel"])


def find_benchmark_binary(build_dir: Path, build_type: str) -> Path:
    executable = (
        "vecadvisor_distance_bench.exe"
        if sys.platform == "win32"
        else "vecadvisor_distance_bench"
    )
    candidates = [
        build_dir / executable,
        build_dir / build_type / executable,
        build_dir / "bin" / executable,
        build_dir / "bin" / build_type / executable,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidate_text = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"native benchmark binary was not found; checked: {candidate_text}")


def run_native_benchmark(
    binary: Path,
    *,
    rows: int,
    queries: int,
    dim: int,
    iterations: int,
    metrics: str,
) -> dict[str, Any]:
    command = [
        str(binary),
        "--rows",
        str(rows),
        "--queries",
        str(queries),
        "--dim",
        str(dim),
        "--iterations",
        str(iterations),
        "--metrics",
        metrics,
        "--json",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def write_outputs(
    payload: dict[str, Any],
    validations: Sequence[NumpyValidation],
    external_baselines: Sequence[ExternalBaselineResult],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
    svg_out: Path | None,
) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(
            json.dumps(merge_payload(payload, validations, external_baselines), indent=2)
            + "\n",
            encoding="utf-8",
        )
    render_payload = merge_payload(payload, validations, external_baselines)
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(
            render_markdown_report(render_payload, validations),
            encoding="utf-8",
        )
    if svg_out is not None:
        svg_out.parent.mkdir(parents=True, exist_ok=True)
        svg_out.write_text(render_svg_chart(render_payload, validations), encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and run the VecAdvisor native kernel benchmark."
    )
    parser.add_argument("--build-dir", type=Path, default=Path("native/build"))
    parser.add_argument("--build-type", default="Release")
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Use an existing native build directory.",
    )
    parser.add_argument(
        "--disable-avx2",
        action="store_true",
        help="Configure a scalar-only native build.",
    )
    parser.add_argument("--rows", type=_positive_int, default=4096)
    parser.add_argument("--queries", type=_positive_int, default=16)
    parser.add_argument("--dim", type=_positive_int, default=128)
    parser.add_argument("--iterations", type=_positive_int, default=5)
    parser.add_argument("--metrics", default="l2,ip,cosine")
    parser.add_argument("--skip-numpy-validation", action="store_true")
    parser.add_argument(
        "--external-baselines",
        default="",
        help="Comma-separated optional external baselines to run, currently: simsimd.",
    )
    parser.add_argument(
        "--require-external-baselines",
        action="store_true",
        help="Fail when a requested external baseline is unavailable or fails validation.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("docs/benchmarks/native-distance-kernels.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("docs/benchmarks/native-distance-kernels.md"),
    )
    parser.add_argument(
        "--svg-out",
        type=Path,
        default=Path("docs/assets/native-distance-kernels.svg"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    source_dir = Path(__file__).resolve().parents[1]
    if not args.no_build:
        configure_and_build(
            args.build_dir,
            args.build_type,
            source_dir=source_dir,
            enable_avx2=not args.disable_avx2,
        )
    binary = find_benchmark_binary(args.build_dir, args.build_type)
    payload = run_native_benchmark(
        binary,
        rows=args.rows,
        queries=args.queries,
        dim=args.dim,
        iterations=args.iterations,
        metrics=args.metrics,
    )
    validations = [] if args.skip_numpy_validation else validate_numpy_checksums(payload)
    external_baselines = run_external_baselines(
        payload,
        _parse_external_baselines(args.external_baselines),
    )
    external_baselines_failed = any(
        baseline.available is False or baseline.passed is False
        for baseline in external_baselines
    )
    if any(not validation.passed for validation in validations):
        write_outputs(
            payload,
            validations,
            external_baselines,
            json_out=args.json_out,
            markdown_out=args.markdown_out,
            svg_out=args.svg_out,
        )
        print(
            render_markdown_report(
                merge_payload(payload, validations, external_baselines),
                validations,
            )
        )
        return 1
    if args.require_external_baselines and external_baselines_failed:
        write_outputs(
            payload,
            validations,
            external_baselines,
            json_out=args.json_out,
            markdown_out=args.markdown_out,
            svg_out=args.svg_out,
        )
        print(
            render_markdown_report(
                merge_payload(payload, validations, external_baselines),
                validations,
            )
        )
        return 1
    write_outputs(
        payload,
        validations,
        external_baselines,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
        svg_out=args.svg_out,
    )
    print(
        render_markdown_report(
            merge_payload(payload, validations, external_baselines),
            validations,
        )
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required unless --skip-numpy-validation is used") from exc
    return np


def _parse_external_baselines(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    baselines = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    supported = {"simsimd"}
    unsupported = sorted(set(baselines) - supported)
    if unsupported:
        raise argparse.ArgumentTypeError(
            "unsupported external baseline(s): " + ", ".join(unsupported)
        )
    return baselines


def _external_baseline_to_json(baseline: ExternalBaselineResult) -> dict[str, Any]:
    return {
        "name": baseline.name,
        "metric": baseline.metric,
        "available": baseline.available,
        "reason": baseline.reason,
        "seconds": baseline.seconds,
        "checksum": baseline.checksum,
        "distances": baseline.distances,
        "ns_per_distance": baseline.ns_per_distance,
        "distances_per_second": baseline.distances_per_second,
        "checksum_abs_error": baseline.checksum_abs_error,
        "tolerance": baseline.tolerance,
        "passed": baseline.passed,
    }


def _external_baselines_from_payload(payload: dict[str, Any]) -> tuple[ExternalBaselineResult, ...]:
    baselines = payload.get("python_wrapper", {}).get("external_baselines", [])
    return tuple(
        ExternalBaselineResult(
            name=str(baseline["name"]),
            metric=str(baseline["metric"]),
            available=bool(baseline["available"]),
            reason=(
                str(baseline["reason"]) if baseline.get("reason") is not None else None
            ),
            seconds=(
                float(baseline["seconds"]) if baseline.get("seconds") is not None else None
            ),
            checksum=(
                float(baseline["checksum"]) if baseline.get("checksum") is not None else None
            ),
            distances=int(baseline["distances"]),
            ns_per_distance=(
                float(baseline["ns_per_distance"])
                if baseline.get("ns_per_distance") is not None
                else None
            ),
            distances_per_second=(
                float(baseline["distances_per_second"])
                if baseline.get("distances_per_second") is not None
                else None
            ),
            checksum_abs_error=(
                float(baseline["checksum_abs_error"])
                if baseline.get("checksum_abs_error") is not None
                else None
            ),
            tolerance=(
                float(baseline["tolerance"]) if baseline.get("tolerance") is not None else None
            ),
            passed=(
                bool(baseline["passed"]) if baseline.get("passed") is not None else None
            ),
        )
        for baseline in baselines
    )


def _baseline_check_text(baseline: ExternalBaselineResult) -> str:
    if not baseline.available:
        return baseline.reason or "not available"
    if baseline.passed is None:
        return "not checked"
    return "pass" if baseline.passed else "fail"


def _run(command: Sequence[str]) -> None:
    print("+ " + " ".join(command), file=sys.stderr)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
