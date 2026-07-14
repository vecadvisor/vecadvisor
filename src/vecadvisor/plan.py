from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from psycopg import Connection

from .models import PlanNode, PlanSummary, SelectivityCrossCheck, TableStats
from .selectivity import clamp_selectivity

SELECTIVITY_WARNING_RATIO_LOW = 0.5
SELECTIVITY_WARNING_RATIO_HIGH = 2.0
SELECTIVITY_CRITICAL_RATIO_LOW = 0.1
SELECTIVITY_CRITICAL_RATIO_HIGH = 10.0
SELECTIVITY_WARNING_ABSOLUTE_DELTA = 0.05
SELECTIVITY_CRITICAL_ABSOLUTE_DELTA = 0.25


class PlanCollectionError(RuntimeError):
    """Raised when a query cannot be safely planned or parsed."""


def explain_query(
    conn: Connection[Any],
    query_sql: str,
    params: Sequence[Any] = (),
    *,
    analyze: bool = False,
    buffers: bool = False,
    statement_timeout_ms: int = 30_000,
) -> PlanSummary:
    """Run EXPLAIN JSON and normalize the root plan.

    By default this uses plain EXPLAIN, so PostgreSQL plans the query without
    executing it. `analyze=True` is reserved for benchmark/validation paths.
    """

    readonly_sql = normalize_readonly_query(query_sql)
    options = ["FORMAT JSON"]
    if analyze:
        options.append("ANALYZE")
        if buffers:
            options.append("BUFFERS")
    elif buffers:
        raise PlanCollectionError("BUFFERS requires analyze=True")

    explain_sql = f"EXPLAIN ({', '.join(options)}) {readonly_sql}"
    with conn.transaction():
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{statement_timeout_ms}ms",),
        )
        row = conn.execute(explain_sql, params).fetchone()
    if row is None:
        raise PlanCollectionError("EXPLAIN returned no rows")

    raw_plan = _first_column(row)
    if not isinstance(raw_plan, list) or not raw_plan:
        raise PlanCollectionError("unexpected EXPLAIN JSON shape")
    top = raw_plan[0]
    if not isinstance(top, dict) or "Plan" not in top:
        raise PlanCollectionError("EXPLAIN JSON did not contain a Plan object")

    return PlanSummary(
        root=_normalize_plan_node(top["Plan"]),
        planning_time_ms=_optional_float(top.get("Planning Time")),
        execution_time_ms=_optional_float(top.get("Execution Time")),
        raw=tuple(raw_plan),
    )


def find_scan_node(root: PlanNode, relation_name: str) -> PlanNode | None:
    """Find the first scan node for a relation, accepting schema-qualified names."""

    short_name = relation_name.rsplit(".", 1)[-1]
    for node in root.walk():
        if node.relation_name in {relation_name, short_name}:
            return node
    return None


def compare_selectivity(
    table: TableStats,
    plan: PlanSummary,
    advisor_selectivity: float,
) -> SelectivityCrossCheck:
    """Compare advisor global selectivity to PostgreSQL's plan-row estimate."""

    scan = find_scan_node(plan.root, table.relname)
    advisor_rows = table.n_rows * advisor_selectivity
    if scan is None or scan.plan_rows is None or table.n_rows <= 0:
        status, severity, notes = classify_selectivity_cross_check(
            advisor_selectivity=advisor_selectivity,
            postgres_selectivity=None,
            absolute_delta=None,
            ratio=None,
        )
        return SelectivityCrossCheck(
            relation_name=table.relname,
            advisor_selectivity=clamp_selectivity(advisor_selectivity),
            postgres_selectivity=None,
            advisor_rows=advisor_rows,
            postgres_plan_rows=None,
            absolute_delta=None,
            ratio=None,
            plan_node_type=scan.node_type if scan else None,
            status=status,
            severity=severity,
            notes=notes,
        )

    postgres_selectivity = clamp_selectivity(scan.plan_rows / table.n_rows)
    absolute_delta = abs(postgres_selectivity - advisor_selectivity)
    ratio = postgres_selectivity / advisor_selectivity if advisor_selectivity > 0 else None
    status, severity, notes = classify_selectivity_cross_check(
        advisor_selectivity=advisor_selectivity,
        postgres_selectivity=postgres_selectivity,
        absolute_delta=absolute_delta,
        ratio=ratio,
    )
    return SelectivityCrossCheck(
        relation_name=table.relname,
        advisor_selectivity=clamp_selectivity(advisor_selectivity),
        postgres_selectivity=postgres_selectivity,
        advisor_rows=advisor_rows,
        postgres_plan_rows=scan.plan_rows,
        absolute_delta=absolute_delta,
        ratio=ratio,
        plan_node_type=scan.node_type,
        status=status,
        severity=severity,
        notes=notes,
    )


def classify_selectivity_cross_check(
    *,
    advisor_selectivity: float,
    postgres_selectivity: float | None,
    absolute_delta: float | None,
    ratio: float | None,
) -> tuple[str, str, tuple[str, ...]]:
    """Classify trust in the global selectivity input."""

    if postgres_selectivity is None or absolute_delta is None or ratio is None:
        return (
            "unavailable",
            "unknown",
            ("PostgreSQL plan rows were unavailable; selectivity cross-check could not run",),
        )

    notes: list[str] = []
    if (
        ratio <= SELECTIVITY_CRITICAL_RATIO_LOW
        or ratio >= SELECTIVITY_CRITICAL_RATIO_HIGH
        or absolute_delta >= SELECTIVITY_CRITICAL_ABSOLUTE_DELTA
    ):
        severity = "critical"
    elif (
        ratio <= SELECTIVITY_WARNING_RATIO_LOW
        or ratio >= SELECTIVITY_WARNING_RATIO_HIGH
        or absolute_delta >= SELECTIVITY_WARNING_ABSOLUTE_DELTA
    ):
        severity = "warning"
    else:
        severity = "ok"

    if severity == "ok":
        notes.append("advisor global selectivity is aligned with PostgreSQL plan rows")
        return "aligned", severity, tuple(notes)

    notes.append(
        "advisor global selectivity diverges from PostgreSQL plan rows; inspect stats "
        "freshness and extended statistics before trusting cost estimates"
    )
    if ratio <= SELECTIVITY_WARNING_RATIO_LOW:
        notes.append(
            "PostgreSQL estimates fewer filter rows than the advisor; advisor may "
            "over-cost exact search"
        )
    elif ratio >= SELECTIVITY_WARNING_RATIO_HIGH:
        notes.append(
            "PostgreSQL estimates more filter rows than the advisor; advisor may "
            "under-cost exact search"
        )
    return "diverged", severity, tuple(notes)


def normalize_readonly_query(query_sql: str) -> str:
    stripped = query_sql.strip()
    if not stripped:
        raise PlanCollectionError("query SQL is empty")
    if stripped.count(";") > 1 or (";" in stripped and not stripped.endswith(";")):
        raise PlanCollectionError("only one SELECT/WITH statement is allowed")
    stripped = stripped[:-1].rstrip() if stripped.endswith(";") else stripped
    lowered = stripped.lstrip(" \t\r\n(").lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise PlanCollectionError("only SELECT/WITH queries can be planned")
    return stripped


def _normalize_plan_node(raw: dict[str, Any]) -> PlanNode:
    children = tuple(_normalize_plan_node(child) for child in raw.get("Plans", ()))
    return PlanNode(
        node_type=str(raw.get("Node Type", "")),
        relation_name=_optional_str(raw.get("Relation Name")),
        index_name=_optional_str(raw.get("Index Name")),
        startup_cost=_optional_float(raw.get("Startup Cost")),
        total_cost=_optional_float(raw.get("Total Cost")),
        plan_rows=_optional_float(raw.get("Plan Rows")),
        plan_width=_optional_int(raw.get("Plan Width")),
        actual_rows=_optional_float(raw.get("Actual Rows")),
        actual_loops=_optional_int(raw.get("Actual Loops")),
        actual_total_time_ms=_optional_float(raw.get("Actual Total Time")),
        rows_removed_by_filter=_optional_int(raw.get("Rows Removed by Filter")),
        filter_text=_optional_str(raw.get("Filter")),
        index_cond=_optional_str(raw.get("Index Cond")),
        order_by=_optional_str(raw.get("Order By")),
        shared_hit_blocks=_optional_int(raw.get("Shared Hit Blocks")),
        shared_read_blocks=_optional_int(raw.get("Shared Read Blocks")),
        children=children,
    )


def _first_column(row: Any) -> Any:
    if isinstance(row, dict):
        return next(iter(row.values()))
    return row[0]


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)
