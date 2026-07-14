from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from psycopg import Connection

from .local_probe import SUPPORTED_DISTANCE_OPS, vector_literal
from .models import PlanNode, PlanSummary, QuerySpec, Strategy, TableStats
from .plan import explain_query
from .query_spec import quote_identifier, quote_qualified_identifier


@dataclass(frozen=True)
class ObservedPlannerChoice:
    strategy: Strategy
    reason: str
    root_node_type: str
    scan_node_type: str | None = None
    index_name: str | None = None
    full_query_sql: str = ""


class PlannerObservationError(RuntimeError):
    """Raised when full-query planner observation cannot be performed safely."""


def observe_planner_choice(
    conn: Connection[Any],
    *,
    table: TableStats,
    query: QuerySpec,
    filter_sql: str,
    query_vector: Sequence[float],
    statement_timeout_ms: int = 30_000,
) -> ObservedPlannerChoice:
    """Run EXPLAIN for the full filtered vector query and classify the chosen strategy."""

    sql = build_vector_order_sql(query=query, filter_sql=filter_sql)
    plan = explain_query(
        conn,
        sql,
        (vector_literal(query_vector), query.limit),
        statement_timeout_ms=statement_timeout_ms,
    )
    observed = classify_observed_plan(table=table, query=query, plan=plan)
    return ObservedPlannerChoice(
        strategy=observed.strategy,
        reason=observed.reason,
        root_node_type=observed.root_node_type,
        scan_node_type=observed.scan_node_type,
        index_name=observed.index_name,
        full_query_sql=sql,
    )


def build_vector_order_sql(*, query: QuerySpec, filter_sql: str) -> str:
    """Build the read-only full vector query used only for EXPLAIN observation."""

    if query.distance_op not in SUPPORTED_DISTANCE_OPS:
        raise PlannerObservationError(
            f"unsupported vector distance operator: {query.distance_op}"
        )
    relation = quote_qualified_identifier(query.relname)
    vector_column = quote_identifier(query.vector_column)
    return (
        f"SELECT * FROM {relation} "
        f"WHERE {filter_sql.strip()} "
        f"ORDER BY {vector_column} {query.distance_op} %s::vector "
        "LIMIT %s"
    )


def classify_observed_plan(
    *,
    table: TableStats,
    query: QuerySpec,
    plan: PlanSummary,
) -> ObservedPlannerChoice:
    """Classify PostgreSQL's observed plan into the advisor's strategy vocabulary."""

    vector_index_names = {
        index.name: index
        for index in table.indexes
        if index.method in {"hnsw", "ivfflat"} and query.vector_column in index.columns
    }
    for node in plan.root.walk():
        if node.index_name is None or node.index_name not in vector_index_names:
            continue
        index = vector_index_names[node.index_name]
        if index.is_partial:
            return ObservedPlannerChoice(
                strategy=Strategy.PARTIAL,
                reason="PostgreSQL chose a partial vector index.",
                root_node_type=plan.root.node_type,
                scan_node_type=node.node_type,
                index_name=node.index_name,
            )
        return ObservedPlannerChoice(
            strategy=Strategy.POSTFILTER,
            reason=(
                "PostgreSQL chose a non-partial vector index; filters are applied after "
                "ANN candidate generation in pgvector's plan shape."
            ),
            root_node_type=plan.root.node_type,
            scan_node_type=node.node_type,
            index_name=node.index_name,
        )

    scan = _first_relation_scan(plan.root)
    return ObservedPlannerChoice(
        strategy=Strategy.EXACT,
        reason="PostgreSQL did not choose a vector index; it will filter then sort by distance.",
        root_node_type=plan.root.node_type,
        scan_node_type=scan.node_type if scan else None,
        index_name=scan.index_name if scan else None,
    )


def _first_relation_scan(root: PlanNode) -> PlanNode | None:
    for node in root.walk():
        if "Scan" in node.node_type:
            return node
    return None
