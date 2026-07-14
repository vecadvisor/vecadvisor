from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

DIAGNOSTIC_FORMATS = {"json", "text"}


def parse_diagnostic_format(value: str) -> str:
    output_format = value.strip().lower()
    if output_format not in DIAGNOSTIC_FORMATS:
        raise ValueError(
            f"diagnostic format must be one of: {', '.join(sorted(DIAGNOSTIC_FORMATS))}"
        )
    return output_format


def render_explain_vector(payload: Mapping[str, Any]) -> str:
    """Render an EXPLAIN VECTOR style text report from CLI JSON payload."""

    recommendation = _mapping(payload.get("recommendation"))
    selectivity = _mapping(payload.get("selectivity"))
    local = payload.get("local_selectivity")
    plan = _mapping(payload.get("plan"))
    ranked = _sequence(recommendation.get("ranked"))
    best = _mapping(ranked[0]) if ranked else {}
    best_strategy = _string(best.get("strategy"), default="unknown")
    planner = _mapping(plan.get("observed_vector_query"))
    planner_strategy = _string(
        planner.get("strategy"),
        default=_string(recommendation.get("planner_would_pick"), default="not observed"),
    )

    lines = [
        _header(payload),
        f"  filter: {_string(payload.get('filter'), default='<none>')}",
        (
            f"  limit: {_integer(payload.get('limit'))}   "
            f"ef_search: {_integer(payload.get('ef_search'))}   "
            f"recall_target: {_format_float(payload.get('recall_target'))}"
        ),
        _selectivity_line(recommendation, selectivity, local),
        _planner_line(planner_strategy, plan),
        "",
        "  PLAN                         est.latency   est.recall   returns k   conf   note",
    ]

    for index, candidate_raw in enumerate(ranked):
        candidate = _mapping(candidate_raw)
        lines.append(_candidate_line(candidate, index=index, planner_strategy=planner_strategy))

    lines.extend(
        [
            "",
            f"  VERDICT: {_string(recommendation.get('verdict'), default='n/a')}",
        ]
    )
    lines.extend(_catalog_snapshot_lines(payload))
    lines.extend(_postgres_capability_lines(payload))
    lines.extend(_stats_health_lines(payload))
    lines.extend(_selectivity_cross_check_lines(selectivity))
    lines.extend(_statistics_lines(payload))
    lines.extend(_local_probe_lines(local))
    lines.extend(_local_probe_cache_lines(payload))
    lines.extend(_note_lines(payload))
    lines.extend(_candidate_note_lines(ranked))
    lines.extend(
        _why_lines(
            decision=recommendation.get("decision"),
            best_strategy=best_strategy,
            planner_strategy=planner_strategy,
            local=local,
        )
    )
    return "\n".join(lines) + "\n"


def _header(payload: Mapping[str, Any]) -> str:
    table = _string(payload.get("table"), default="<unknown>")
    rows = _format_float(_mapping(payload.get("selectivity")).get("advisor_rows"))
    dim = _integer(payload.get("vector_dim"))
    calibration = _mapping(payload.get("calibration"))
    source = _string(calibration.get("source"), default="unknown")
    return f"EXPLAIN VECTOR  {table}  (advisor_rows={rows}, d={dim}, calibration={source})"


def _selectivity_line(
    recommendation: Mapping[str, Any],
    selectivity: Mapping[str, Any],
    local: object,
) -> str:
    rho = _format_float(recommendation.get("rho"))
    local_payload = _mapping(local)
    sample = _integer(local_payload.get("sample_size")) if local_payload else "n/a"
    return (
        f"  s(global): {_format_float(recommendation.get('s_global'))}   "
        f"s(local): {_format_float(recommendation.get('s_local'))}   "
        f"rho: {rho}   "
        f"postgres_s: {_format_float(selectivity.get('postgres_selectivity'))}   "
        f"probe_rows: {sample}"
    )


def _planner_line(planner_strategy: str, plan: Mapping[str, Any]) -> str:
    root = _string(plan.get("root_node_type"), default="unknown")
    planning_ms = _format_float(plan.get("planning_time_ms"))
    return f"  planner: {planner_strategy}   root: {root}   planning_ms: {planning_ms}"


def _candidate_line(
    candidate: Mapping[str, Any],
    *,
    index: int,
    planner_strategy: str,
) -> str:
    strategy = _string(candidate.get("strategy"), default="unknown")
    plan = _mapping(candidate.get("plan"))
    estimate = _mapping(candidate.get("estimate"))
    marker = "->" if index == 0 else "  "
    planner_marker = " planner" if strategy == planner_strategy else ""
    object_marker = _object_marker(plan)
    return (
        f"  {marker} {strategy:<25} "
        f"{_latency_ms(estimate.get('est_latency_us')):>10}   "
        f"{_format_float(estimate.get('est_recall')):>9}   "
        f"{_yes_no(estimate.get('est_returns_k')):<9}   "
        f"{_format_float(estimate.get('confidence')):>4}   "
        f"{object_marker}{planner_marker}"
    ).rstrip()


def _statistics_lines(payload: Mapping[str, Any]) -> list[str]:
    suggestions = _sequence(payload.get("statistics_suggestions"))
    if not suggestions:
        return []
    lines = ["", "  STATISTICS:"]
    for suggestion_raw in suggestions:
        suggestion = _mapping(suggestion_raw)
        ddl = _string(suggestion.get("ddl"), default="")
        reason = _string(suggestion.get("reason"), default="")
        confidence = _format_float(suggestion.get("confidence"))
        lines.append(f"    - {ddl}")
        lines.append(f"      reason: {reason} (confidence={confidence})")
    return lines


def _catalog_snapshot_lines(payload: Mapping[str, Any]) -> list[str]:
    snapshot = _mapping(payload.get("catalog_snapshot"))
    if not snapshot:
        return []
    stats_fingerprint = _string(snapshot.get("stats_fingerprint"), default="")
    index_fingerprint = _string(snapshot.get("index_fingerprint"), default="")
    if not stats_fingerprint and not index_fingerprint:
        return []
    last_analyze = _string(snapshot.get("last_analyze_at"), default="n/a")
    changed_rows = _integer(snapshot.get("n_mod_since_analyze"))
    return [
        "",
        "  CATALOG SNAPSHOT:",
        (
            f"    - stats_fingerprint: {stats_fingerprint}   "
            f"index_fingerprint: {index_fingerprint}"
        ),
        f"    - last_analyze: {last_analyze}   changed_rows: {changed_rows}",
    ]


def _postgres_capability_lines(payload: Mapping[str, Any]) -> list[str]:
    postgres = _mapping(payload.get("postgres"))
    if not postgres:
        return []
    installed = postgres.get("pgvector_installed")
    iterative = postgres.get("supports_hnsw_iterative_scan")
    if installed is True and iterative is True:
        return []
    version = _string(postgres.get("pgvector_version"), default="not installed")
    lines = ["", "  PGVECTOR CAPABILITIES:"]
    lines.append(
        "    - "
        f"pgvector_version={version}   "
        f"iterative_scan={_yes_no(iterative)}"
    )
    for note in _sequence(postgres.get("notes")):
        lines.append(f"      {_string(note, default='')}")
    return lines


def _stats_health_lines(payload: Mapping[str, Any]) -> list[str]:
    health = _mapping(payload.get("stats_health"))
    if not health or health.get("stale") is not True:
        return []
    lines = ["", "  STATS HEALTH:"]
    status = _string(health.get("status"), default="unknown")
    analyze_sql = _string(health.get("analyze_sql"), default="")
    ratio = _format_float(health.get("modification_ratio"))
    changed = _integer(health.get("n_mod_since_analyze"))
    lines.append(f"    - status: {status}   changed_rows: {changed}   changed_ratio: {ratio}")
    if analyze_sql:
        lines.append(f"    - {analyze_sql}")
    for note in _sequence(health.get("notes")):
        lines.append(f"      {_string(note, default='')}")
    return lines


def _selectivity_cross_check_lines(selectivity: Mapping[str, Any]) -> list[str]:
    severity = _string(selectivity.get("severity"), default="unknown")
    status = _string(selectivity.get("status"), default="unknown")
    notes = [_string(note, default="") for note in _sequence(selectivity.get("notes"))]
    notes = [note for note in notes if note]
    if severity == "unknown" and status == "unknown" and not notes:
        return []
    if severity == "ok":
        return []

    ratio = _format_float(selectivity.get("ratio"))
    delta = _format_float(selectivity.get("absolute_delta"))
    advisor_rows = _format_float(selectivity.get("advisor_rows"))
    postgres_rows = _format_float(selectivity.get("postgres_plan_rows"))
    lines = ["", "  SELECTIVITY CROSS-CHECK:"]
    lines.append(
        f"    - status: {status}   severity: {severity}   "
        f"ratio: {ratio}   abs_delta: {delta}"
    )
    lines.append(
        f"    - advisor_rows: {advisor_rows}   postgres_plan_rows: {postgres_rows}"
    )
    for note in notes:
        lines.append(f"      {note}")
    return lines


def _local_probe_lines(local: object) -> list[str]:
    local_payload = _mapping(local)
    if not local_payload:
        return []
    passing = _integer(local_payload.get("passing_rows"))
    sample = _integer(local_payload.get("sample_size"))
    floor = _format_float(local_payload.get("resolution_floor"))
    lines = [
        "",
        (
            f"  LOCAL PROBE: passing_rows={passing}/{sample}   "
            f"resolution_floor={floor}   "
            f"confidence={_format_float(local_payload.get('confidence'))}"
        ),
    ]
    for note in _sequence(local_payload.get("notes")):
        lines.append(f"    - {_string(note, default='')}")
    return lines


def _local_probe_cache_lines(payload: Mapping[str, Any]) -> list[str]:
    cache = _mapping(payload.get("local_selectivity_cache"))
    if not cache or cache.get("enabled") is not True:
        return []
    key = _string(cache.get("key"), default="n/a")
    hit = _yes_no(cache.get("hit"))
    stored = _yes_no(cache.get("stored"))
    refresh = _yes_no(cache.get("refresh"))
    path = _string(cache.get("path"), default="n/a")
    return [
        "",
        "  LOCAL PROBE CACHE:",
        f"    - hit: {hit}   stored: {stored}   refresh: {refresh}",
        f"    - key: {key}",
        f"    - path: {path}",
    ]


def _note_lines(payload: Mapping[str, Any]) -> list[str]:
    notes = [_string(note, default="") for note in _sequence(payload.get("notes"))]
    notes = [note for note in notes if note]
    if not notes:
        return []
    return ["", "  NOTES:", *(f"    - {note}" for note in notes)]


def _candidate_note_lines(ranked: Sequence[object]) -> list[str]:
    lines: list[str] = []
    for candidate_raw in ranked:
        candidate = _mapping(candidate_raw)
        strategy = _string(candidate.get("strategy"), default="unknown")
        estimate = _mapping(candidate.get("estimate"))
        notes = [_string(note, default="") for note in _sequence(estimate.get("notes"))]
        notes = [note for note in notes if note]
        if not notes:
            continue
        if not lines:
            lines.extend(["", "  COST NOTES:"])
        for note in notes:
            lines.append(f"    - {strategy}: {note}")
    return lines


def _why_lines(
    *,
    decision: object,
    best_strategy: str,
    planner_strategy: str,
    local: object,
) -> list[str]:
    lines = ["", "  WHY:"]
    decision_payload = _mapping(decision)
    decision_reasons = [
        _string(reason, default="")
        for reason in _sequence(decision_payload.get("why"))
    ]
    decision_reasons = [reason for reason in decision_reasons if reason]
    if decision_reasons:
        return [*lines, *(f"    - {reason}" for reason in decision_reasons)]

    local_payload = _mapping(local)
    if local_payload:
        lines.append(
            "    - Costing used measured local selectivity from the query neighborhood."
        )
    else:
        lines.append(
            "    - Local selectivity was not measured; global selectivity was used as fallback."
        )
    if planner_strategy not in {"not observed", best_strategy}:
        lines.append(
            f"    - Advisor recommends {best_strategy}, while PostgreSQL observed plan is "
            f"{planner_strategy}."
        )
    else:
        lines.append(f"    - Advisor's top-ranked strategy is {best_strategy}.")
    if local_payload and _number(local_payload.get("passing_rows")) == 0:
        lines.append("    - Zero passing probe rows make ANN post-filter recall unsafe.")
    return lines


def _object_marker(plan: Mapping[str, Any]) -> str:
    requires = _string(plan.get("requires_new_object"), default="")
    uses = _string(plan.get("uses_index"), default="")
    ef_search = plan.get("ef_search")
    parts = []
    if ef_search is not None:
        parts.append(f"ef={_integer(ef_search)}")
    if uses:
        parts.append(f"uses={uses}")
    if requires:
        parts.append(f"needs={requires}")
    return ", ".join(parts)


def _latency_ms(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return f"{number / 1000.0:.3g} ms"


def _yes_no(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "NO"
    return "n/a"


def _format_float(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    if number == 0:
        return "0"
    if abs(number) < 0.001 or abs(number) >= 10_000:
        return f"{number:.3e}"
    return f"{number:.4g}"


def _integer(value: object) -> str:
    number = _number(value)
    if number is None:
        return "n/a"
    return str(int(number))


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return ()


def _string(value: object, *, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)
