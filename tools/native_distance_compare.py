from __future__ import annotations

import argparse
import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_combined_payload(
    avx2_payload: dict[str, Any],
    scalar_payload: dict[str, Any],
    *,
    avx2_source: str,
    scalar_source: str,
) -> dict[str, Any]:
    _validate_compatible(avx2_payload, scalar_payload)
    avx2_results = _results_by_metric(avx2_payload)
    scalar_results = _results_by_metric(scalar_payload)

    summary = []
    for metric in avx2_results:
        avx2_result = avx2_results[metric]
        scalar_result = scalar_results[metric]
        avx2_dispatch_ns = float(avx2_result["dispatch"]["ns_per_distance"])
        scalar_dispatch_ns = float(scalar_result["dispatch"]["ns_per_distance"])
        speedup = scalar_dispatch_ns / avx2_dispatch_ns if avx2_dispatch_ns > 0.0 else None
        simsimd_baseline = _external_baseline(avx2_payload, name="simsimd", metric=metric)
        simsimd_ns = (
            float(simsimd_baseline["ns_per_distance"])
            if simsimd_baseline is not None
            and simsimd_baseline.get("ns_per_distance") is not None
            else None
        )
        summary.append(
            {
                "metric": metric,
                "avx2_selected_kernel": avx2_result["selected_kernel"],
                "scalar_fallback_selected_kernel": scalar_result["selected_kernel"],
                "avx2_dispatch_ns_per_distance": avx2_dispatch_ns,
                "avx2_scalar_ns_per_distance": float(
                    avx2_result["scalar"]["ns_per_distance"]
                ),
                "scalar_fallback_dispatch_ns_per_distance": scalar_dispatch_ns,
                "speedup_vs_scalar_fallback": speedup,
                "simsimd_ns_per_distance": simsimd_ns,
                "avx2_vs_simsimd_ratio": (
                    avx2_dispatch_ns / simsimd_ns
                    if simsimd_ns is not None and simsimd_ns > 0.0
                    else None
                ),
                "simsimd_check_passed": (
                    bool(simsimd_baseline["passed"])
                    if simsimd_baseline is not None
                    and simsimd_baseline.get("passed") is not None
                    else None
                ),
                "avx2_max_abs_error_vs_scalar": float(
                    avx2_result["max_abs_error_vs_scalar"]
                ),
                "scalar_fallback_max_abs_error_vs_scalar": float(
                    scalar_result["max_abs_error_vs_scalar"]
                ),
                "numpy_checks_passed": _numpy_checks_passed(avx2_payload, metric)
                and _numpy_checks_passed(scalar_payload, metric),
            }
        )

    return {
        "schema_version": 1,
        "artifact": "native-distance-kernels",
        "rows": avx2_payload["rows"],
        "queries": avx2_payload["queries"],
        "dim": avx2_payload["dim"],
        "iterations": avx2_payload["iterations"],
        "runs": [
            _run_payload("avx2-dispatch", avx2_payload, avx2_source),
            _run_payload("scalar-fallback", scalar_payload, scalar_source),
        ],
        "summary": summary,
    }


def render_markdown_report(combined: dict[str, Any]) -> str:
    avx2_run = combined["runs"][0]
    scalar_run = combined["runs"][1]
    has_simsimd = any(row.get("simsimd_ns_per_distance") is not None for row in combined["summary"])
    lines = [
        "# Native Distance Kernel Benchmark",
        "",
        (
            "This committed MVP2 artifact compares the normal runtime-dispatch build "
            "against an explicit scalar-only fallback build."
        ),
        "",
        f"- rows: `{combined['rows']}`",
        f"- queries: `{combined['queries']}`",
        f"- dimensions: `{combined['dim']}`",
        f"- iterations: `{combined['iterations']}`",
        f"- AVX2 run platform: `{avx2_run['platform']}`",
        f"- scalar fallback platform: `{scalar_run['platform']}`",
        f"- AVX2 runtime available: `{avx2_run['capabilities']['avx2_runtime']}`",
        f"- scalar build compiled AVX2: `{scalar_run['capabilities']['avx2_compiled']}`",
        "",
    ]
    if has_simsimd:
        lines.extend(
            [
                (
                    "| metric | AVX2 kernel | AVX2 ns/dist | SimSIMD ns/dist | "
                    "AVX2/SimSIMD | scalar fallback ns/dist | speedup | "
                    "max AVX2 error | checks |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "| metric | AVX2 kernel | AVX2 ns/dist | scalar fallback ns/dist | "
                    "speedup | max AVX2 error | NumPy checks |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
    for row in combined["summary"]:
        speedup = row["speedup_vs_scalar_fallback"]
        speedup_text = "n/a" if speedup is None else f"{float(speedup):.3f}x"
        check_text = "pass" if row["numpy_checks_passed"] else "fail"
        if has_simsimd:
            simsimd_ns = row.get("simsimd_ns_per_distance")
            simsimd_ratio = row.get("avx2_vs_simsimd_ratio")
            simsimd_check = row.get("simsimd_check_passed")
            checks = check_text
            if simsimd_check is not None:
                checks += f" / simsimd:{'pass' if simsimd_check else 'fail'}"
            lines.append(
                f"| {row['metric']} | {row['avx2_selected_kernel']} | "
                f"{float(row['avx2_dispatch_ns_per_distance']):.3f} | "
                f"{_optional_float_text(simsimd_ns)} | "
                f"{_optional_ratio_text(simsimd_ratio)} | "
                f"{float(row['scalar_fallback_dispatch_ns_per_distance']):.3f} | "
                f"{speedup_text} | {float(row['avx2_max_abs_error_vs_scalar']):.6g} | "
                f"{checks} |"
            )
        else:
            lines.append(
                f"| {row['metric']} | {row['avx2_selected_kernel']} | "
                f"{float(row['avx2_dispatch_ns_per_distance']):.3f} | "
                f"{float(row['scalar_fallback_dispatch_ns_per_distance']):.3f} | "
                f"{speedup_text} | {float(row['avx2_max_abs_error_vs_scalar']):.6g} | "
                f"{check_text} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The `AVX2 ns/dist` column is the runtime-dispatch path future Python "
                "bindings would call on an AVX2-capable host. The scalar fallback column "
                "comes from a separate build configured with "
                "`-DVECADVISOR_NATIVE_ENABLE_AVX2=OFF`, so it verifies the portable path "
                "used on machines without AVX2 support."
            ),
            *(
                [
                    "",
                    (
                        "`SimSIMD ns/dist` is an optional external baseline measured by "
                        "the Python benchmark wrapper on the same deterministic matrices. "
                        "`AVX2/SimSIMD` is a latency ratio; lower is better for VecAdvisor."
                    ),
                ]
                if has_simsimd
                else []
            ),
            "",
            "## Reproduce",
            "",
            "```bash",
            "python tools/native_distance_benchmark.py \\",
            "  --build-dir native/build-avx2 \\",
            f"  --rows {combined['rows']} \\",
            f"  --queries {combined['queries']} \\",
            f"  --dim {combined['dim']} \\",
            f"  --iterations {combined['iterations']} \\",
            "  --external-baselines simsimd \\",
            "  --json-out native/build-avx2/native-distance-kernels.json \\",
            "  --markdown-out native/build-avx2/native-distance-kernels.md \\",
            "  --svg-out native/build-avx2/native-distance-kernels.svg",
            "",
            "python tools/native_distance_benchmark.py \\",
            "  --build-dir native/build-scalar \\",
            "  --disable-avx2 \\",
            f"  --rows {combined['rows']} \\",
            f"  --queries {combined['queries']} \\",
            f"  --dim {combined['dim']} \\",
            f"  --iterations {combined['iterations']} \\",
            "  --json-out native/build-scalar/native-distance-kernels.json \\",
            "  --markdown-out native/build-scalar/native-distance-kernels.md \\",
            "  --svg-out native/build-scalar/native-distance-kernels.svg",
            "",
            "python tools/native_distance_compare.py \\",
            "  --avx2-json native/build-avx2/native-distance-kernels.json \\",
            "  --scalar-json native/build-scalar/native-distance-kernels.json",
            "```",
            "",
            "## Inputs",
            "",
            f"- AVX2 source: `{avx2_run['source']}`",
            f"- scalar fallback source: `{scalar_run['source']}`",
            "",
        ]
    )
    return "\n".join(lines)


def render_svg_chart(combined: dict[str, Any]) -> str:
    width = 980
    height = 540
    margin_left = 82
    margin_right = 36
    margin_top = 84
    margin_bottom = 92
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    rows = list(combined["summary"])
    has_simsimd = any(row.get("simsimd_ns_per_distance") is not None for row in rows)
    max_value = max(
        max(
            float(row["avx2_dispatch_ns_per_distance"]),
            float(row["scalar_fallback_dispatch_ns_per_distance"]),
            float(row["simsimd_ns_per_distance"])
            if row.get("simsimd_ns_per_distance") is not None
            else 0.0,
        )
        for row in rows
    )
    y_max = max(max_value * 1.15, 1.0)
    group_width = plot_width / max(len(rows), 1)
    bar_width = min(52.0, group_width * (0.18 if has_simsimd else 0.24))
    baseline = margin_top + plot_height

    def y(value: float) -> float:
        return baseline - (value / y_max) * plot_height

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        "<title>VecAdvisor native distance kernel benchmark</title>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{margin_left}" y="36" font-family="Arial, sans-serif" '
            'font-size="24" font-weight="700">VecAdvisor native kernels</text>'
        ),
        (
            f'<text x="{margin_left}" y="60" font-family="Arial, sans-serif" '
            'font-size="12" fill="#334155">'
            f"rows: {combined['rows']} | queries: {combined['queries']} | "
            f"dim: {combined['dim']} | iterations: {combined['iterations']}</text>"
        ),
    ]

    for tick in range(5):
        value = y_max * tick / 4.0
        tick_y = y(value)
        lines.append(
            f'<line x1="{margin_left}" y1="{tick_y:.2f}" '
            f'x2="{width - margin_right}" y2="{tick_y:.2f}" '
            'stroke="#e2e8f0" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{margin_left - 10}" y="{tick_y + 4:.2f}" '
            'text-anchor="end" font-family="Arial, sans-serif" '
            'font-size="11" fill="#475569">'
            f"{value:.1f}</text>"
        )

    lines.append(
        f'<line x1="{margin_left}" y1="{baseline}" x2="{width - margin_right}" '
        f'y2="{baseline}" stroke="#94a3b8" stroke-width="1"/>'
    )
    lines.append(
        f'<text x="24" y="{margin_top + plot_height / 2:.2f}" '
        'font-family="Arial, sans-serif" font-size="12" fill="#334155" '
        'transform="rotate(-90 24 '
        f'{margin_top + plot_height / 2:.2f})">nanoseconds per distance</text>'
    )

    for index, row in enumerate(rows):
        center = margin_left + group_width * (index + 0.5)
        avx2_value = float(row["avx2_dispatch_ns_per_distance"])
        scalar_value = float(row["scalar_fallback_dispatch_ns_per_distance"])
        simsimd_value = row.get("simsimd_ns_per_distance")
        avx2_x = center - bar_width - 4 if not has_simsimd else center - 1.5 * bar_width - 6
        simsimd_x = center - bar_width / 2.0 if has_simsimd else center
        scalar_x = center + 4 if not has_simsimd else center + 0.5 * bar_width + 6
        lines.extend(
            [
                (
                    f'<rect x="{avx2_x:.2f}" y="{y(avx2_value):.2f}" '
                    f'width="{bar_width:.2f}" height="{baseline - y(avx2_value):.2f}" '
                    'fill="#2563eb"/>'
                ),
                (
                    f'<rect x="{scalar_x:.2f}" y="{y(scalar_value):.2f}" '
                    f'width="{bar_width:.2f}" height="{baseline - y(scalar_value):.2f}" '
                    'fill="#dc2626"/>'
                ),
                *(
                    [
                        (
                            f'<rect x="{simsimd_x:.2f}" y="{y(float(simsimd_value)):.2f}" '
                            f'width="{bar_width:.2f}" '
                            f'height="{baseline - y(float(simsimd_value)):.2f}" '
                            'fill="#7c3aed"/>'
                        )
                    ]
                    if has_simsimd and simsimd_value is not None
                    else []
                ),
                (
                    f'<text x="{center:.2f}" y="{baseline + 26:.2f}" '
                    'text-anchor="middle" font-family="Arial, sans-serif" '
                    'font-size="13" fill="#0f172a">'
                    f"{html.escape(str(row['metric']))}</text>"
                ),
                (
                    f'<text x="{center:.2f}" y="{baseline + 46:.2f}" '
                    'text-anchor="middle" font-family="Arial, sans-serif" '
                    'font-size="11" fill="#475569">'
                    f"{float(row['speedup_vs_scalar_fallback']):.2f}x faster</text>"
                ),
            ]
        )

    lines.extend(
        [
            '<rect x="620" y="28" width="12" height="12" fill="#2563eb"/>',
            (
                '<text x="638" y="38" font-family="Arial, sans-serif" '
                'font-size="12" fill="#334155">AVX2 dispatch</text>'
            ),
            *(
                [
                    '<rect x="744" y="28" width="12" height="12" fill="#7c3aed"/>',
                    (
                        '<text x="762" y="38" font-family="Arial, sans-serif" '
                        'font-size="12" fill="#334155">SimSIMD</text>'
                    ),
                ]
                if has_simsimd
                else []
            ),
            '<rect x="838" y="28" width="12" height="12" fill="#dc2626"/>',
            (
                '<text x="856" y="38" font-family="Arial, sans-serif" '
                'font-size="12" fill="#334155">scalar fallback</text>'
            ),
            "</svg>",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    combined: dict[str, Any],
    *,
    json_out: Path,
    markdown_out: Path,
    svg_out: Path,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    svg_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    markdown_out.write_text(render_markdown_report(combined), encoding="utf-8")
    svg_out.write_text(render_svg_chart(combined), encoding="utf-8")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine AVX2 and scalar-only native benchmark reports."
    )
    parser.add_argument("--avx2-json", type=Path, required=True)
    parser.add_argument("--scalar-json", type=Path, required=True)
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
    args = parse_args(argv if argv is not None else [])
    avx2_payload = load_payload(args.avx2_json)
    scalar_payload = load_payload(args.scalar_json)
    combined = build_combined_payload(
        avx2_payload,
        scalar_payload,
        avx2_source=str(args.avx2_json),
        scalar_source=str(args.scalar_json),
    )
    write_outputs(
        combined,
        json_out=args.json_out,
        markdown_out=args.markdown_out,
        svg_out=args.svg_out,
    )
    print(render_markdown_report(combined))
    return 0


def _validate_compatible(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("rows", "queries", "dim", "iterations"):
        if left.get(key) != right.get(key):
            raise ValueError(f"benchmark payload mismatch for {key}")
    if set(_results_by_metric(left)) != set(_results_by_metric(right)):
        raise ValueError("benchmark payloads contain different metrics")


def _results_by_metric(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(result["metric"]): result for result in payload["results"]}


def _run_payload(name: str, payload: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "name": name,
        "source": source,
        "platform": payload.get("python_wrapper", {}).get("platform", "unknown"),
        "capabilities": payload["capabilities"],
        "results": payload["results"],
        "numpy_validation": payload.get("python_wrapper", {}).get("numpy_validation", []),
        "external_baselines": payload.get("python_wrapper", {}).get(
            "external_baselines",
            [],
        ),
    }


def _numpy_checks_passed(payload: dict[str, Any], metric: str) -> bool:
    validations = payload.get("python_wrapper", {}).get("numpy_validation", [])
    for validation in validations:
        if validation.get("metric") == metric:
            return bool(validation.get("passed"))
    return False


def _external_baseline(
    payload: dict[str, Any],
    *,
    name: str,
    metric: str,
) -> dict[str, Any] | None:
    baselines = payload.get("python_wrapper", {}).get("external_baselines", [])
    for baseline in baselines:
        if (
            baseline.get("name") == name
            and baseline.get("metric") == metric
            and baseline.get("available") is True
        ):
            return baseline
    return None


def _optional_float_text(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _optional_ratio_text(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.3f}x"


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
