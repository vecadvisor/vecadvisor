from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .crossover import CrossoverAnalysis, SweepAnalysisPoint, WinnerCrossover

DEFAULT_CHART_TITLE = "VecAdvisor crossover"
DEFAULT_PARETO_TITLE = "VecAdvisor recall/QPS Pareto"
DEFAULT_CHART_WIDTH = 1120
DEFAULT_PARETO_WIDTH = 920
PANEL_HEIGHT = 340
HEADER_HEIGHT = 130
FOOTER_HEIGHT = 48
PLOT_LEFT = 86
PLOT_RIGHT = 40
LATENCY_HEIGHT = 138
RECALL_HEIGHT = 74
STRATEGY_COLORS = {
    "exact": "#2563eb",
    "postfilter": "#dc2626",
    "iterative": "#059669",
    "partial": "#7c3aed",
    "partition": "#d97706",
}
FALLBACK_COLORS = ("#0891b2", "#be123c", "#4d7c0f", "#9333ea", "#0f766e")
PARETO_HEIGHT = 560
PARETO_TOP = 124
PARETO_BOTTOM = 78


@dataclass(frozen=True)
class BenchmarkParetoPoint:
    strategy: str
    recall_at_k: float
    returns_k_rate: float
    qps: float
    latency_ms_p95: float
    query_count: int


def render_crossover_svg(
    analysis: CrossoverAnalysis,
    *,
    title: str = DEFAULT_CHART_TITLE,
    width: int = DEFAULT_CHART_WIDTH,
) -> str:
    """Render a dependency-free SVG money chart from crossover analysis."""

    if width < 760:
        raise ValueError("width must be at least 760")
    if not analysis.points:
        raise ValueError("analysis must contain at least one point")

    correlations = analysis.correlations or tuple(
        sorted({point.target_correlation for point in analysis.points})
    )
    height = HEADER_HEIGHT + PANEL_HEIGHT * len(correlations) + FOOTER_HEIGHT
    plot_width = width - PLOT_LEFT - PLOT_RIGHT
    strategy_names = _strategy_names(analysis.points)
    latency_min, latency_max = _latency_bounds(analysis.points)

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{_escape(title)}</title>",
        (
            f'<desc id="desc">Filtered vector search crossover chart for '
            f'{_escape(analysis.backend)} sweep results.</desc>'
        ),
        _style_block(),
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        _header(title=title, analysis=analysis, strategy_names=strategy_names, width=width),
    ]

    for panel_index, correlation in enumerate(correlations):
        panel_y = HEADER_HEIGHT + panel_index * PANEL_HEIGHT
        group = sorted(
            (
                point
                for point in analysis.points
                if point.target_correlation == correlation
            ),
            key=lambda point: point.target_filter_selectivity,
        )
        if not group:
            continue
        parts.append(
            _panel(
                analysis=analysis,
                points=group,
                correlation=correlation,
                panel_y=panel_y,
                plot_width=plot_width,
                latency_min=latency_min,
                latency_max=latency_max,
                strategy_names=strategy_names,
            )
        )

    parts.append(
        f'<text class="footnote" x="{PLOT_LEFT}" y="{height - 18}">'
        "Solid lines show p95 latency. Lower thin lines show recall@k. "
        "Squares are measured winners; circles are predicted winners.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_crossover_svg(
    analysis: CrossoverAnalysis,
    path: Path,
    *,
    title: str = DEFAULT_CHART_TITLE,
    width: int = DEFAULT_CHART_WIDTH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_crossover_svg(analysis, title=title, width=width),
        encoding="utf-8",
    )


def load_benchmark_payload(path: Path) -> Mapping[str, Any]:
    """Load a JSON benchmark report produced by benchmark or benchmark-db."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"could not read benchmark report: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"benchmark report JSON is invalid: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("benchmark report must be a JSON object")
    return raw


def render_benchmark_pareto_svg(
    payload: Mapping[str, Any],
    *,
    title: str = DEFAULT_PARETO_TITLE,
    width: int = DEFAULT_PARETO_WIDTH,
) -> str:
    """Render a dependency-free recall-vs-QPS Pareto chart from benchmark JSON."""

    if width < 700:
        raise ValueError("width must be at least 700")
    dataset = _mapping_value(payload, "dataset")
    ground_truth = _optional_mapping_value(payload, "ground_truth")
    points = _benchmark_pareto_points(payload)
    if not points:
        raise ValueError("benchmark report must contain at least one strategy")

    plot_width = width - PLOT_LEFT - PLOT_RIGHT
    plot_height = PARETO_HEIGHT - PARETO_TOP - PARETO_BOTTOM
    x_min, x_max = _qps_bounds(points)
    dataset_id = str(dataset.get("id", "unknown"))
    rows = dataset.get("rows", "unknown")
    queries = dataset.get("queries", "unknown")
    k = ground_truth.get("k", "unknown") if ground_truth is not None else "unknown"
    metric = ground_truth.get("metric", "unknown") if ground_truth is not None else "unknown"

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{PARETO_HEIGHT}" '
            f'viewBox="0 0 {width} {PARETO_HEIGHT}" role="img" aria-labelledby="title desc">'
        ),
        f"<title id=\"title\">{_escape(title)}</title>",
        (
            f'<desc id="desc">Recall versus QPS Pareto chart for '
            f'{_escape(dataset_id)} benchmark results.</desc>'
        ),
        _pareto_style_block(),
        f'<rect width="{width}" height="{PARETO_HEIGHT}" fill="#ffffff"/>',
        f'<text class="chart-title" x="{PLOT_LEFT}" y="38">{_escape(title)}</text>',
        (
            f'<text class="summary" x="{PLOT_LEFT}" y="64">dataset: {_escape(dataset_id)}'
            f"  |  rows: {_escape(rows)}  |  queries: {_escape(queries)}"
            f"  |  k: {_escape(k)}  |  metric: {_escape(metric)}</text>"
        ),
        _pareto_legend(points, width=width),
        _plot_frame(
            x=PLOT_LEFT,
            y=PARETO_TOP,
            width=plot_width,
            height=plot_height,
        ),
        _qps_axis(
            points=points,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
            plot_height=plot_height,
        ),
        _recall_pareto_axis(plot_width=plot_width, plot_height=plot_height),
        f'<text class="axis-label" x="24" y="{PARETO_TOP + 32}" '
        f'transform="rotate(-90 24 {PARETO_TOP + 32})">recall@k</text>',
        f'<text class="axis-label" text-anchor="end" x="{PLOT_LEFT + plot_width}" '
        f'y="{PARETO_TOP + plot_height + 48}">QPS, higher is better</text>',
        _pareto_frontier_svg(
            points=points,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
            plot_height=plot_height,
        ),
    ]

    for point in points:
        parts.append(
            _pareto_point_svg(
                point,
                x_min=x_min,
                x_max=x_max,
                plot_width=plot_width,
                plot_height=plot_height,
            )
        )

    parts.append(
        f'<text class="footnote" x="{PLOT_LEFT}" y="{PARETO_HEIGHT - 22}">'
        "Each point is one measured strategy. Hollow markers under-return for at least "
        "one query. Frontier ignores dominated points.</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_benchmark_pareto_svg(
    payload: Mapping[str, Any],
    path: Path,
    *,
    title: str = DEFAULT_PARETO_TITLE,
    width: int = DEFAULT_PARETO_WIDTH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_benchmark_pareto_svg(payload, title=title, width=width),
        encoding="utf-8",
    )


def _benchmark_pareto_points(payload: Mapping[str, Any]) -> tuple[BenchmarkParetoPoint, ...]:
    points: list[BenchmarkParetoPoint] = []
    for raw_strategy in _sequence_value(payload, "strategies"):
        strategy = _as_mapping(raw_strategy, "strategy")
        latency = _mapping_value(strategy, "latency_ms")
        strategy_name = _str_value(strategy, "strategy")
        query_count = _int_value(strategy, "query_count")
        latency_ms_total = _positive_float_value(latency, "total")
        points.append(
            BenchmarkParetoPoint(
                strategy=strategy_name,
                recall_at_k=_bounded_float_value(strategy, "recall_at_k"),
                returns_k_rate=_bounded_float_value(strategy, "returns_k_rate"),
                qps=(query_count * 1000.0) / latency_ms_total,
                latency_ms_p95=_positive_float_value(latency, "p95"),
                query_count=query_count,
            )
        )
    return tuple(points)


def _pareto_style_block() -> str:
    return """<style>
text { font-family: "Segoe UI", Arial, sans-serif; fill: #111827; }
.chart-title { font-size: 24px; font-weight: 700; }
.summary, .legend, .footnote { font-size: 12px; fill: #4b5563; }
.axis-label, .tick { font-size: 11px; fill: #6b7280; }
.frame { fill: #ffffff; stroke: #d1d5db; stroke-width: 1; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.frontier { fill: none; stroke: #111827; stroke-width: 1.8; stroke-dasharray: 6 4; }
.pareto-point { stroke-width: 2; }
.point-label { font-size: 12px; font-weight: 600; }
.under-return { stroke: #991b1b; stroke-width: 2; fill: none; }
</style>"""


def _pareto_legend(points: tuple[BenchmarkParetoPoint, ...], *, width: int) -> str:
    cursor = PLOT_LEFT
    y = 92
    parts: list[str] = []
    for point in points:
        color = _strategy_color(point.strategy)
        parts.append(f'<circle cx="{cursor}" cy="{y}" r="5" fill="{color}"/>')
        parts.append(
            f'<text class="legend" x="{cursor + 10}" y="{y + 4}">'
            f"{_escape(point.strategy)}</text>"
        )
        cursor += max(80, len(point.strategy) * 9 + 40)
        if cursor > width - 230:
            break
    parts.append(
        f'<line class="frontier" x1="{width - 190}" y1="{y}" '
        f'x2="{width - 150}" y2="{y}"/>'
    )
    parts.append(f'<text class="legend" x="{width - 142}" y="{y + 4}">Pareto frontier</text>')
    return "\n".join(parts)


def _qps_axis(
    *,
    points: tuple[BenchmarkParetoPoint, ...],
    x_min: float,
    x_max: float,
    plot_width: int,
    plot_height: int,
) -> str:
    ticks = _qps_ticks(points, x_min=x_min, x_max=x_max)
    parts: list[str] = []
    for tick in ticks:
        x = _qps_x(tick, x_min=x_min, x_max=x_max, plot_width=plot_width)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" y1="{PARETO_TOP}" x2="{x:.2f}" '
            f'y2="{PARETO_TOP + plot_height}"/>'
        )
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{x:.2f}" '
            f'y="{PARETO_TOP + plot_height + 20}">{_escape(_format_number(tick))}</text>'
        )
    return "\n".join(parts)


def _recall_pareto_axis(*, plot_width: int, plot_height: int) -> str:
    parts: list[str] = []
    for tick in (0.0, 0.5, 0.9, 1.0):
        y = _pareto_recall_y(tick, plot_height=plot_height)
        parts.append(
            f'<line class="grid" x1="{PLOT_LEFT}" y1="{y:.2f}" '
            f'x2="{PLOT_LEFT + plot_width}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="tick" text-anchor="end" x="{PLOT_LEFT - 8}" y="{y + 4:.2f}">'
            f"{tick:.1f}</text>"
        )
    return "\n".join(parts)


def _pareto_frontier_svg(
    *,
    points: tuple[BenchmarkParetoPoint, ...],
    x_min: float,
    x_max: float,
    plot_width: int,
    plot_height: int,
) -> str:
    frontier = _pareto_frontier(points)
    if len(frontier) < 2:
        return ""
    positions = [
        (
            _qps_x(point.qps, x_min=x_min, x_max=x_max, plot_width=plot_width),
            _pareto_recall_y(point.recall_at_k, plot_height=plot_height),
        )
        for point in frontier
    ]
    return (
        '<polyline class="frontier" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in positions)
        + '"><title>Pareto frontier</title></polyline>'
    )


def _pareto_point_svg(
    point: BenchmarkParetoPoint,
    *,
    x_min: float,
    x_max: float,
    plot_width: int,
    plot_height: int,
) -> str:
    x = _qps_x(point.qps, x_min=x_min, x_max=x_max, plot_width=plot_width)
    y = _pareto_recall_y(point.recall_at_k, plot_height=plot_height)
    color = _strategy_color(point.strategy)
    fill = color if point.returns_k_rate >= 1.0 else "#ffffff"
    label_x = min(x + 10, PLOT_LEFT + plot_width - 80)
    label_y = max(y - 8, PARETO_TOP + 14)
    title = (
        f"{point.strategy}: recall {point.recall_at_k:.3g}, "
        f"QPS {point.qps:.3g}, p95 {point.latency_ms_p95:.3g} ms, "
        f"returns-k {point.returns_k_rate:.3g}"
    )
    parts = [
        f'<circle class="pareto-point" cx="{x:.2f}" cy="{y:.2f}" r="7" '
        f'fill="{fill}" stroke="{color}"><title>{_escape(title)}</title></circle>',
        f'<text class="point-label" x="{label_x:.2f}" y="{label_y:.2f}">'
        f"{_escape(point.strategy)}</text>",
    ]
    if point.returns_k_rate < 1.0:
        parts.append(
            f'<line class="under-return" x1="{x - 6:.2f}" y1="{y - 6:.2f}" '
            f'x2="{x + 6:.2f}" y2="{y + 6:.2f}"><title>returns-k below target'
            "</title></line>"
        )
        parts.append(
            f'<line class="under-return" x1="{x + 6:.2f}" y1="{y - 6:.2f}" '
            f'x2="{x - 6:.2f}" y2="{y + 6:.2f}"><title>returns-k below target'
            "</title></line>"
        )
    return "\n".join(parts)


def _pareto_frontier(
    points: tuple[BenchmarkParetoPoint, ...],
) -> tuple[BenchmarkParetoPoint, ...]:
    frontier = []
    for point in sorted(points, key=lambda candidate: candidate.qps):
        dominated = any(
            other.qps >= point.qps
            and other.recall_at_k >= point.recall_at_k
            and (other.qps > point.qps or other.recall_at_k > point.recall_at_k)
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return tuple(frontier)


def _qps_bounds(points: tuple[BenchmarkParetoPoint, ...]) -> tuple[float, float]:
    values = [max(point.qps, 1e-9) for point in points]
    lower = min(values)
    upper = max(values)
    if lower == upper:
        return max(lower / 2.0, 1e-9), max(upper * 2.0, 1e-8)
    return lower, upper


def _qps_ticks(
    points: tuple[BenchmarkParetoPoint, ...],
    *,
    x_min: float,
    x_max: float,
) -> tuple[float, ...]:
    unique = tuple(sorted({point.qps for point in points}))
    if len(unique) <= 4:
        return unique
    return (x_min, _midpoint_positive(x_min, x_max), x_max)


def _qps_x(value: float, *, x_min: float, x_max: float, plot_width: int) -> float:
    if x_min == x_max:
        return PLOT_LEFT + plot_width / 2.0
    lo = math.log10(max(x_min, 1e-9))
    hi = math.log10(max(x_max, x_min * 1.01))
    ratio = (math.log10(max(value, 1e-9)) - lo) / (hi - lo)
    return PLOT_LEFT + max(0.0, min(1.0, ratio)) * plot_width


def _pareto_recall_y(value: float, *, plot_height: int) -> float:
    return PARETO_TOP + plot_height - max(0.0, min(1.0, value)) * plot_height


def _header(
    *,
    title: str,
    analysis: CrossoverAnalysis,
    strategy_names: tuple[str, ...],
    width: int,
) -> str:
    match_text = (
        "prediction match: n/a"
        if analysis.prediction_match_rate is None
        else f"prediction match: {analysis.prediction_match_rate * 100:.1f}%"
    )
    summary = (
        f"backend: {analysis.backend}  |  points: {analysis.point_count}  |  "
        f"recall target: {analysis.recall_target:.3g}  |  "
        f"returns-k target: {analysis.returns_k_target:.3g}  |  {match_text}  |  "
        f"postfilter failures: {analysis.postfilter_failure_count}"
    )
    legend_x = PLOT_LEFT
    legend_y = 92
    parts = [
        f'<text class="chart-title" x="{PLOT_LEFT}" y="38">{_escape(title)}</text>',
        f'<text class="summary" x="{PLOT_LEFT}" y="64">{_escape(summary)}</text>',
    ]
    cursor = legend_x
    for strategy in strategy_names:
        color = _strategy_color(strategy)
        parts.append(f'<circle cx="{cursor}" cy="{legend_y}" r="5" fill="{color}"/>')
        parts.append(
            f'<text class="legend" x="{cursor + 10}" y="{legend_y + 4}">'
            f"{_escape(strategy)}</text>"
        )
        cursor += max(78, len(strategy) * 9 + 38)
    parts.append(
        f'<line x1="{width - 300}" y1="{legend_y}" x2="{width - 260}" '
        f'y2="{legend_y}" class="measured-crossover"/>'
    )
    parts.append(
        f'<text class="legend" x="{width - 252}" y="{legend_y + 4}">measured crossover</text>'
    )
    parts.append(
        f'<line x1="{width - 300}" y1="{legend_y + 22}" x2="{width - 260}" '
        f'y2="{legend_y + 22}" class="predicted-crossover"/>'
    )
    parts.append(
        f'<text class="legend" x="{width - 252}" y="{legend_y + 26}">predicted crossover</text>'
    )
    return "\n".join(parts)


def _panel(
    *,
    analysis: CrossoverAnalysis,
    points: list[SweepAnalysisPoint],
    correlation: float,
    panel_y: int,
    plot_width: int,
    latency_min: float,
    latency_max: float,
    strategy_names: tuple[str, ...],
) -> str:
    latency_y = panel_y + 42
    recall_y = latency_y + LATENCY_HEIGHT + 54
    strip_y = recall_y + RECALL_HEIGHT + 28
    x_min = min(point.target_filter_selectivity for point in points)
    x_max = max(point.target_filter_selectivity for point in points)
    parts = [
        f'<g class="panel" id="corr-{_safe_id(correlation)}">',
        f'<text class="panel-title" x="{PLOT_LEFT}" y="{panel_y + 22}">'
        f"correlation = {_format_number(correlation)}</text>",
        _plot_frame(
            x=PLOT_LEFT,
            y=latency_y,
            width=plot_width,
            height=LATENCY_HEIGHT,
        ),
        _plot_frame(
            x=PLOT_LEFT,
            y=recall_y,
            width=plot_width,
            height=RECALL_HEIGHT,
        ),
        _x_grid(
            points=points,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
            latency_y=latency_y,
            recall_y=recall_y,
        ),
        _latency_axis(
            latency_min=latency_min,
            latency_max=latency_max,
            latency_y=latency_y,
            plot_width=plot_width,
        ),
        _recall_axis(recall_y=recall_y, plot_width=plot_width),
        f'<text class="axis-label" x="18" y="{latency_y + 20}" '
        'transform="rotate(-90 18 '
        f'{latency_y + 20})">p95 latency ms</text>',
        f'<text class="axis-label" x="28" y="{recall_y + 22}" '
        'transform="rotate(-90 28 '
        f'{recall_y + 22})">recall@k</text>',
    ]

    for strategy in strategy_names:
        parts.append(
            _strategy_lines(
                strategy=strategy,
                points=points,
                x_min=x_min,
                x_max=x_max,
                plot_width=plot_width,
                latency_min=latency_min,
                latency_max=latency_max,
                latency_y=latency_y,
                recall_y=recall_y,
            )
        )

    parts.append(
        _crossover_lines(
            crossovers=analysis.measured_crossovers,
            correlation=correlation,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
            y1=latency_y,
            y2=recall_y + RECALL_HEIGHT,
            css_class="measured-crossover",
            label="measured",
            label_y=latency_y + 13,
        )
    )
    parts.append(
        _crossover_lines(
            crossovers=analysis.predicted_crossovers,
            correlation=correlation,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
            y1=latency_y,
            y2=recall_y + RECALL_HEIGHT,
            css_class="predicted-crossover",
            label="predicted",
            label_y=latency_y + 29,
        )
    )
    parts.append(
        _winner_strip(
            points=points,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
            strip_y=strip_y,
        )
    )
    parts.append("</g>")
    return "\n".join(parts)


def _style_block() -> str:
    return """<style>
text { font-family: "Segoe UI", Arial, sans-serif; fill: #111827; }
.chart-title { font-size: 24px; font-weight: 700; }
.summary, .legend, .footnote { font-size: 12px; fill: #4b5563; }
.panel-title { font-size: 15px; font-weight: 700; }
.axis-label, .tick { font-size: 11px; fill: #6b7280; }
.frame { fill: #ffffff; stroke: #d1d5db; stroke-width: 1; }
.grid { stroke: #e5e7eb; stroke-width: 1; }
.axis { stroke: #9ca3af; stroke-width: 1; }
.latency-line { fill: none; stroke-width: 2.4; stroke-linejoin: round; stroke-linecap: round; }
.recall-line { fill: none; stroke-width: 1.4; stroke-linejoin: round;
  stroke-linecap: round; opacity: 0.8; }
.point { stroke: #ffffff; stroke-width: 1.4; }
.recall-fail { stroke: #991b1b; stroke-width: 2; }
.measured-crossover { stroke: #111827; stroke-width: 1.6; stroke-dasharray: 5 4; }
.predicted-crossover { stroke: #b45309; stroke-width: 1.6; stroke-dasharray: 2 4; }
.winner-label { font-size: 10px; fill: #6b7280; }
</style>"""


def _plot_frame(*, x: int, y: int, width: int, height: int) -> str:
    return f'<rect class="frame" x="{x}" y="{y}" width="{width}" height="{height}" rx="4"/>'


def _x_grid(
    *,
    points: list[SweepAnalysisPoint],
    x_min: float,
    x_max: float,
    plot_width: int,
    latency_y: int,
    recall_y: int,
) -> str:
    ticks = _selectivity_ticks(points, x_min=x_min, x_max=x_max)
    parts: list[str] = []
    for tick in ticks:
        x = _x_pos(tick, x_min=x_min, x_max=x_max, plot_width=plot_width)
        parts.append(
            f'<line class="grid" x1="{x:.2f}" y1="{latency_y}" x2="{x:.2f}" '
            f'y2="{recall_y + RECALL_HEIGHT}"/>'
        )
        parts.append(
            f'<text class="tick" text-anchor="middle" x="{x:.2f}" '
            f'y="{recall_y + RECALL_HEIGHT + 18}">{_escape(_format_number(tick))}</text>'
        )
    parts.append(
        f'<text class="axis-label" text-anchor="end" x="{PLOT_LEFT + plot_width}" '
        f'y="{recall_y + RECALL_HEIGHT + 37}">target selectivity</text>'
    )
    return "\n".join(parts)


def _latency_axis(
    *,
    latency_min: float,
    latency_max: float,
    latency_y: int,
    plot_width: int,
) -> str:
    ticks = (
        latency_min,
        _midpoint_positive(latency_min, latency_max),
        latency_max,
    )
    parts: list[str] = []
    for tick in ticks:
        y = _latency_y(tick, latency_min=latency_min, latency_max=latency_max, latency_y=latency_y)
        parts.append(
            f'<line class="grid" x1="{PLOT_LEFT}" y1="{y:.2f}" '
            f'x2="{PLOT_LEFT + plot_width}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="tick" text-anchor="end" x="{PLOT_LEFT - 8}" y="{y + 4:.2f}">'
            f"{_escape(_format_number(tick))}</text>"
        )
    return "\n".join(parts)


def _recall_axis(*, recall_y: int, plot_width: int) -> str:
    parts: list[str] = []
    for tick in (0.0, 0.5, 1.0):
        y = _recall_y(tick, recall_y=recall_y)
        parts.append(
            f'<line class="grid" x1="{PLOT_LEFT}" y1="{y:.2f}" '
            f'x2="{PLOT_LEFT + plot_width}" y2="{y:.2f}"/>'
        )
        parts.append(
            f'<text class="tick" text-anchor="end" x="{PLOT_LEFT - 8}" y="{y + 4:.2f}">'
            f"{tick:.1f}</text>"
        )
    return "\n".join(parts)


def _strategy_lines(
    *,
    strategy: str,
    points: list[SweepAnalysisPoint],
    x_min: float,
    x_max: float,
    plot_width: int,
    latency_min: float,
    latency_max: float,
    latency_y: int,
    recall_y: int,
) -> str:
    color = _strategy_color(strategy)
    strategy_points = []
    for point in points:
        summary = next(
            (candidate for candidate in point.strategies if candidate.strategy == strategy),
            None,
        )
        if summary is not None:
            strategy_points.append((point, summary))
    if not strategy_points:
        return ""

    latency_points = []
    recall_points = []
    markers: list[str] = []
    for point, summary in strategy_points:
        x = _x_pos(
            point.target_filter_selectivity,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
        )
        latency = max(summary.latency_ms_p95, 1e-9)
        y_latency = _latency_y(
            latency,
            latency_min=latency_min,
            latency_max=latency_max,
            latency_y=latency_y,
        )
        y_recall = _recall_y(summary.recall_at_k, recall_y=recall_y)
        latency_points.append(f"{x:.2f},{y_latency:.2f}")
        recall_points.append(f"{x:.2f},{y_recall:.2f}")
        markers.append(
            f'<circle class="point" cx="{x:.2f}" cy="{y_latency:.2f}" r="4" '
            f'fill="{color}"><title>{_escape(strategy)} p95 '
            f'{summary.latency_ms_p95:.3g} ms at selectivity '
            f'{point.target_filter_selectivity:.3g}</title></circle>'
        )
        markers.append(
            f'<circle cx="{x:.2f}" cy="{y_recall:.2f}" r="2.8" fill="{color}" '
            f'opacity="0.82"><title>{_escape(strategy)} recall '
            f'{summary.recall_at_k:.3g}</title></circle>'
        )
    return "\n".join(
        (
            (
                f'<polyline class="latency-line" stroke="{color}" '
                f'points="{" ".join(latency_points)}"/>'
            ),
            (
                f'<polyline class="recall-line" stroke="{color}" '
                f'points="{" ".join(recall_points)}"/>'
            ),
            *markers,
        )
    )


def _crossover_lines(
    *,
    crossovers: tuple[WinnerCrossover, ...],
    correlation: float,
    x_min: float,
    x_max: float,
    plot_width: int,
    y1: int,
    y2: int,
    css_class: str,
    label: str,
    label_y: int,
) -> str:
    parts: list[str] = []
    for crossover in crossovers:
        if crossover.correlation != correlation:
            continue
        x = _x_pos(
            crossover.estimated_selectivity,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
        )
        title = (
            f"{label} crossover: {crossover.from_strategy} to {crossover.to_strategy} "
            f"near selectivity {crossover.estimated_selectivity:.3g}"
        )
        parts.append(
            f'<line class="{css_class}" x1="{x:.2f}" y1="{y1}" '
            f'x2="{x:.2f}" y2="{y2}"><title>{_escape(title)}</title></line>'
        )
        parts.append(
            f'<text class="winner-label" text-anchor="middle" x="{x:.2f}" y="{label_y}">'
            f"{_escape(label)}</text>"
        )
    return "\n".join(parts)


def _winner_strip(
    *,
    points: list[SweepAnalysisPoint],
    x_min: float,
    x_max: float,
    plot_width: int,
    strip_y: int,
) -> str:
    parts = [
        f'<text class="winner-label" x="{PLOT_LEFT}" y="{strip_y - 8}">'
        "winner by selectivity</text>"
    ]
    for point in points:
        x = _x_pos(
            point.target_filter_selectivity,
            x_min=x_min,
            x_max=x_max,
            plot_width=plot_width,
        )
        measured_color = _strategy_color(point.measured_best)
        parts.append(
            f'<rect x="{x - 6:.2f}" y="{strip_y}" width="12" height="12" rx="2" '
            f'fill="{measured_color}"><title>measured winner: '
            f'{_escape(point.measured_best)}</title></rect>'
        )
        if point.predicted_best is not None:
            predicted_color = _strategy_color(point.predicted_best)
            parts.append(
                f'<circle cx="{x:.2f}" cy="{strip_y + 27}" r="5" fill="{predicted_color}" '
                'stroke="#111827" stroke-width="1"><title>predicted winner: '
                f'{_escape(point.predicted_best)}</title></circle>'
            )
        if point.postfilter_viable is False:
            fail_y = strip_y + 44
            parts.append(
                f'<line class="recall-fail" x1="{x - 5:.2f}" y1="{fail_y - 5}" '
                f'x2="{x + 5:.2f}" y2="{fail_y + 5}"><title>postfilter recall below '
                'target</title></line>'
            )
            parts.append(
                f'<line class="recall-fail" x1="{x + 5:.2f}" y1="{fail_y - 5}" '
                f'x2="{x - 5:.2f}" y2="{fail_y + 5}"><title>postfilter recall below '
                'target</title></line>'
            )
    return "\n".join(parts)


def _strategy_names(points: tuple[SweepAnalysisPoint, ...]) -> tuple[str, ...]:
    seen = {summary.strategy for point in points for summary in point.strategies}
    preferred: tuple[str, ...] = ("exact", "postfilter", "iterative")
    ordered: list[str] = [strategy for strategy in preferred if strategy in seen]
    ordered_set = set(ordered)
    ordered.extend(sorted(strategy for strategy in seen if strategy not in ordered_set))
    return tuple(ordered)


def _latency_bounds(points: tuple[SweepAnalysisPoint, ...]) -> tuple[float, float]:
    values = [
        max(summary.latency_ms_p95, 1e-9)
        for point in points
        for summary in point.strategies
    ]
    if not values:
        return 1e-3, 1.0
    lower = min(values)
    upper = max(values)
    if lower == upper:
        lower = max(lower / 2.0, 1e-9)
        upper *= 2.0
    return lower, upper


def _selectivity_ticks(
    points: list[SweepAnalysisPoint],
    *,
    x_min: float,
    x_max: float,
) -> tuple[float, ...]:
    unique = tuple(sorted({point.target_filter_selectivity for point in points}))
    if len(unique) <= 6:
        return unique
    return (x_min, _midpoint_positive(x_min, x_max), x_max)


def _x_pos(value: float, *, x_min: float, x_max: float, plot_width: int) -> float:
    if x_min == x_max:
        return PLOT_LEFT + plot_width / 2.0
    if x_min > 0.0 and x_max > 0.0 and value > 0.0:
        lo = math.log10(x_min)
        hi = math.log10(x_max)
        ratio = (math.log10(value) - lo) / (hi - lo)
    else:
        ratio = (value - x_min) / (x_max - x_min)
    return PLOT_LEFT + max(0.0, min(1.0, ratio)) * plot_width


def _latency_y(
    value: float,
    *,
    latency_min: float,
    latency_max: float,
    latency_y: int,
) -> float:
    if latency_min == latency_max:
        ratio = 0.5
    else:
        lo = math.log10(max(latency_min, 1e-9))
        hi = math.log10(max(latency_max, latency_min * 1.01))
        ratio = (math.log10(max(value, 1e-9)) - lo) / (hi - lo)
    return latency_y + LATENCY_HEIGHT - max(0.0, min(1.0, ratio)) * LATENCY_HEIGHT


def _recall_y(value: float, *, recall_y: int) -> float:
    return recall_y + RECALL_HEIGHT - max(0.0, min(1.0, value)) * RECALL_HEIGHT


def _midpoint_positive(lower: float, upper: float) -> float:
    if lower > 0.0 and upper > 0.0:
        return math.sqrt(lower * upper)
    return (lower + upper) / 2.0


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    return _as_mapping(payload.get(key), key)


def _optional_mapping_value(
    payload: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    return _as_mapping(value, key)


def _sequence_value(payload: Mapping[str, Any], key: str) -> Sequence[object]:
    value = payload.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{key} must be a list")
    return value


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _str_value(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _int_value(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value <= 0:
        raise ValueError(f"{key} must be positive")
    return value


def _positive_float_value(payload: Mapping[str, Any], key: str) -> float:
    number = _float_value(payload, key)
    if number <= 0.0:
        raise ValueError(f"{key} must be positive")
    return number


def _bounded_float_value(payload: Mapping[str, Any], key: str) -> float:
    number = _float_value(payload, key)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{key} must be in [0, 1]")
    return number


def _float_value(payload: Mapping[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _strategy_color(strategy: str) -> str:
    if strategy in STRATEGY_COLORS:
        return STRATEGY_COLORS[strategy]
    index = abs(hash(strategy)) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[index]


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 10_000:
        return f"{value:.2e}"
    return f"{value:.4g}"


def _safe_id(value: float) -> str:
    return str(value).replace("-", "neg").replace(".", "_")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)
