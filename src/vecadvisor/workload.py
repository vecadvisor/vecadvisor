from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from pglast import ast, enums, parse_sql
from pglast.parser import ParseError
from psycopg import Connection

from .query_spec import quote_identifier, quote_qualified_identifier

VECTOR_DISTANCE_OPERATORS = frozenset({"<->", "<#>", "<=>", "<+>", "<~>", "<%"})
DEFAULT_WORKLOAD_LIMIT = 50
DEFAULT_WORKLOAD_MIN_CALLS = 1


class WorkloadCollectionError(RuntimeError):
    """Raised when pg_stat_statements cannot be read."""


@dataclass(frozen=True)
class WorkloadPredicateShape:
    column: str
    kind: str
    operator: str
    value_kind: str


@dataclass(frozen=True)
class WorkloadLimitShape:
    kind: str
    value: int | None


@dataclass(frozen=True)
class WorkloadShape:
    fingerprint: str
    queryid: str | None
    table: str | None
    vector_column: str
    distance_operator: str
    limit: WorkloadLimitShape
    predicates: tuple[WorkloadPredicateShape, ...]
    unsupported_predicates: tuple[str, ...]
    calls: int
    rows: int
    total_exec_time_ms: float
    mean_exec_time_ms: float
    recommend_filter_template: str | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class WorkloadCollectionReport:
    source_view: str
    scanned_statements: int
    vector_query_shapes: tuple[WorkloadShape, ...]
    rejected_statements: int
    min_calls: int
    limit: int


@dataclass(frozen=True)
class _SourceView:
    name: str
    sql_identifier: str


def collect_workload_shapes(
    conn: Connection[Any],
    *,
    limit: int = DEFAULT_WORKLOAD_LIMIT,
    min_calls: int = DEFAULT_WORKLOAD_MIN_CALLS,
    statement_timeout_ms: int = 30_000,
) -> WorkloadCollectionReport:
    """Collect sanitized filtered-vector query shapes from pg_stat_statements."""

    if limit <= 0:
        raise ValueError("limit must be positive")
    if min_calls <= 0:
        raise ValueError("min_calls must be positive")
    if statement_timeout_ms <= 0:
        raise ValueError("statement_timeout_ms must be positive")

    source_view = _resolve_pg_stat_statements_view(conn)
    columns = _pg_stat_statements_columns(conn, source_view.name)
    queryid_expr = (
        f"{quote_identifier('queryid')}::text"
        if "queryid" in columns
        else "NULL::text"
    )
    total_time_column = _first_present(columns, ("total_exec_time", "total_time"))
    mean_time_column = _first_present(columns, ("mean_exec_time", "mean_time"))
    if total_time_column is None:
        raise WorkloadCollectionError(
            "pg_stat_statements is missing total execution time columns"
        )
    total_time_sql = quote_identifier(total_time_column)
    mean_time_sql = (
        quote_identifier(mean_time_column)
        if mean_time_column is not None
        else f"({total_time_sql} / NULLIF({quote_identifier('calls')}, 0))"
    )

    with conn.transaction():
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{statement_timeout_ms}ms",),
        )
        rows = conn.execute(
            f"""
            SELECT
                {queryid_expr} AS queryid,
                {quote_identifier('query')} AS query,
                {quote_identifier('calls')} AS calls,
                {quote_identifier('rows')} AS result_rows,
                {total_time_sql} AS total_exec_time_ms,
                {mean_time_sql} AS mean_exec_time_ms
            FROM {source_view.sql_identifier}
            WHERE {quote_identifier('calls')} >= %s
            ORDER BY {total_time_sql} DESC, {quote_identifier('calls')} DESC
            LIMIT %s
            """,
            (min_calls, limit),
        ).fetchall()

    shapes: list[WorkloadShape] = []
    rejected = 0
    for row in rows:
        try:
            shape = shape_from_pg_stat_statement(
                query=str(row["query"]),
                queryid=str(row["queryid"]) if row["queryid"] is not None else None,
                calls=int(row["calls"] or 0),
                rows=int(row["result_rows"] or 0),
                total_exec_time_ms=float(row["total_exec_time_ms"] or 0.0),
                mean_exec_time_ms=float(row["mean_exec_time_ms"] or 0.0),
            )
        except ValueError:
            rejected += 1
        else:
            if shape is None:
                rejected += 1
            else:
                shapes.append(shape)

    return WorkloadCollectionReport(
        source_view=source_view.name,
        scanned_statements=len(rows),
        vector_query_shapes=tuple(shapes),
        rejected_statements=rejected,
        min_calls=min_calls,
        limit=limit,
    )


def shape_from_pg_stat_statement(
    *,
    query: str,
    queryid: str | None,
    calls: int,
    rows: int,
    total_exec_time_ms: float,
    mean_exec_time_ms: float,
) -> WorkloadShape | None:
    """Parse one pg_stat_statements query into a privacy-preserving vector shape."""

    try:
        statements = parse_sql(query)
    except ParseError as exc:
        raise ValueError(f"could not parse statement: {exc}") from exc

    if len(statements) != 1:
        return None
    select = _unwrap_select(statements[0].stmt)
    if select is None:
        return None

    vector_order = _find_vector_order(select)
    if vector_order is None:
        return None
    vector_column, distance_operator = vector_order
    table = _single_table_name(select)
    predicates, unsupported = _predicate_shapes(select.whereClause)
    limit_shape = _limit_shape(select.limitCount)
    filter_template = _recommend_filter_template(predicates)
    notes = _shape_notes(table=table, predicates=predicates, unsupported=unsupported)
    fingerprint = _shape_fingerprint(
        {
            "table": table,
            "vector_column": vector_column,
            "distance_operator": distance_operator,
            "limit": {"kind": limit_shape.kind, "value": limit_shape.value},
            "predicates": [predicate_to_json(predicate) for predicate in predicates],
            "unsupported_predicates": list(unsupported),
        }
    )

    return WorkloadShape(
        fingerprint=fingerprint,
        queryid=queryid,
        table=table,
        vector_column=vector_column,
        distance_operator=distance_operator,
        limit=limit_shape,
        predicates=predicates,
        unsupported_predicates=unsupported,
        calls=calls,
        rows=rows,
        total_exec_time_ms=total_exec_time_ms,
        mean_exec_time_ms=mean_exec_time_ms,
        recommend_filter_template=filter_template,
        notes=notes,
    )


def workload_report_to_json(report: WorkloadCollectionReport) -> dict[str, object]:
    return {
        "source": {
            "view": report.source_view,
            "scanned_statements": report.scanned_statements,
            "rejected_statements": report.rejected_statements,
            "min_calls": report.min_calls,
            "limit": report.limit,
        },
        "privacy": {
            "raw_query_text": False,
            "sql_literals": "redacted",
            "query_vectors": "not collected",
            "fingerprint": "sha256 over sanitized query shape",
        },
        "vector_query_shapes": [
            workload_shape_to_json(shape) for shape in report.vector_query_shapes
        ],
    }


def workload_shape_to_json(shape: WorkloadShape) -> dict[str, object]:
    return {
        "fingerprint": shape.fingerprint,
        "pg_stat_statements_queryid": shape.queryid,
        "table": shape.table,
        "vector_column": shape.vector_column,
        "distance_operator": shape.distance_operator,
        "limit": {
            "kind": shape.limit.kind,
            "value": shape.limit.value,
        },
        "predicates": [predicate_to_json(predicate) for predicate in shape.predicates],
        "unsupported_predicates": list(shape.unsupported_predicates),
        "stats": {
            "calls": shape.calls,
            "rows": shape.rows,
            "total_exec_time_ms": shape.total_exec_time_ms,
            "mean_exec_time_ms": shape.mean_exec_time_ms,
        },
        "recommend_template": _recommend_template_to_json(shape),
        "notes": list(shape.notes),
    }


def predicate_to_json(predicate: WorkloadPredicateShape) -> dict[str, object]:
    return {
        "column": predicate.column,
        "kind": predicate.kind,
        "operator": predicate.operator,
        "value_kind": predicate.value_kind,
    }


def _recommend_template_to_json(shape: WorkloadShape) -> dict[str, object] | None:
    if shape.table is None or shape.recommend_filter_template is None:
        return None
    return {
        "table": shape.table,
        "vector": shape.vector_column,
        "query": shape.recommend_filter_template,
        "notes": [
            "replace <value> placeholders with representative literal values",
            "supply representative vectors with --q-vectors or --q-vector-sql",
        ],
    }


def _resolve_pg_stat_statements_view(conn: Connection[Any]) -> _SourceView:
    row = conn.execute(
        """
        SELECT n.nspname AS schema_name
        FROM pg_extension e
        JOIN pg_namespace n ON n.oid = e.extnamespace
        WHERE e.extname = 'pg_stat_statements'
        """
    ).fetchone()
    if row is None:
        raise WorkloadCollectionError(
            "pg_stat_statements extension is not installed in this database"
        )
    name = f"{row['schema_name']}.pg_stat_statements"
    return _SourceView(
        name=name,
        sql_identifier=quote_qualified_identifier(name),
    )


def _pg_stat_statements_columns(conn: Connection[Any], source_view: str) -> frozenset[str]:
    row = conn.execute("SELECT to_regclass(%s)::oid AS oid", (source_view,)).fetchone()
    if row is None or row["oid"] is None:
        raise WorkloadCollectionError(f"could not resolve {source_view}")
    rows = conn.execute(
        """
        SELECT attname
        FROM pg_attribute
        WHERE attrelid = %s
          AND attnum > 0
          AND NOT attisdropped
        """,
        (row["oid"],),
    ).fetchall()
    return frozenset(str(item["attname"]) for item in rows)


def _first_present(columns: frozenset[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _unwrap_select(statement: object) -> ast.SelectStmt | None:
    if isinstance(statement, ast.SelectStmt):
        return statement
    return None


def _find_vector_order(select: ast.SelectStmt) -> tuple[str, str] | None:
    for sort in select.sortClause or ():
        if not isinstance(sort, ast.SortBy):
            continue
        node = sort.node
        if not isinstance(node, ast.A_Expr) or node.kind is not enums.A_Expr_Kind.AEXPR_OP:
            continue
        operator = _operator_name(node.name)
        if operator not in VECTOR_DISTANCE_OPERATORS:
            continue
        left_column = _column_name_if_simple(node.lexpr)
        right_column = _column_name_if_simple(node.rexpr)
        if left_column is not None and right_column is None:
            return left_column, operator
        if right_column is not None and left_column is None:
            return right_column, operator
        if left_column is not None:
            return left_column, operator
    return None


def _single_table_name(select: ast.SelectStmt) -> str | None:
    from_clause = select.fromClause or ()
    range_vars = [item for item in from_clause if isinstance(item, ast.RangeVar)]
    if len(range_vars) != 1:
        return None
    table = range_vars[0]
    if table.relname is None:
        return None
    if table.schemaname is None:
        return str(table.relname)
    return f"{table.schemaname}.{table.relname}"


def _predicate_shapes(
    node: object | None,
) -> tuple[tuple[WorkloadPredicateShape, ...], tuple[str, ...]]:
    if node is None:
        return (), ()
    predicates: list[WorkloadPredicateShape] = []
    unsupported: list[str] = []
    _collect_predicate_shapes(node, predicates, unsupported)
    return tuple(predicates), tuple(unsupported)


def _collect_predicate_shapes(
    node: object,
    predicates: list[WorkloadPredicateShape],
    unsupported: list[str],
) -> None:
    if isinstance(node, ast.BoolExpr):
        if node.boolop is enums.BoolExprType.AND_EXPR:
            for child in node.args or ():
                _collect_predicate_shapes(child, predicates, unsupported)
        else:
            unsupported.append("OR predicates are not summarized")
        return

    if isinstance(node, ast.ColumnRef):
        column = _column_name_if_simple(node)
        if column is None:
            unsupported.append("qualified boolean predicate")
            return
        predicates.append(
            WorkloadPredicateShape(
                column=column,
                kind="bool",
                operator="IS TRUE",
                value_kind="implicit",
            )
        )
        return

    if isinstance(node, ast.A_Expr):
        parsed = _parse_predicate_expr(node)
        if parsed is None:
            unsupported.append(_unsupported_expression_label(node))
        else:
            predicates.append(parsed)
        return

    unsupported.append(type(node).__name__)


def _parse_predicate_expr(node: ast.A_Expr) -> WorkloadPredicateShape | None:
    if node.kind is enums.A_Expr_Kind.AEXPR_OP:
        operator = _operator_name(node.name)
        kind = _predicate_kind_for_operator(operator)
        if kind is None:
            return None
        left_column = _column_name_if_simple(node.lexpr)
        right_column = _column_name_if_simple(node.rexpr)
        if left_column is not None and right_column is None:
            return WorkloadPredicateShape(
                column=left_column,
                kind=kind,
                operator=operator,
                value_kind=_value_kind(node.rexpr),
            )
        if right_column is not None and left_column is None:
            return WorkloadPredicateShape(
                column=right_column,
                kind=kind,
                operator=_invert_operator(operator),
                value_kind=_value_kind(node.lexpr),
            )
        return None

    if node.kind is enums.A_Expr_Kind.AEXPR_IN:
        column = _column_name_if_simple(node.lexpr)
        if column is None:
            return None
        return WorkloadPredicateShape(
            column=column,
            kind="in",
            operator="IN",
            value_kind=_value_kind(node.rexpr),
        )

    if node.kind is enums.A_Expr_Kind.AEXPR_BETWEEN:
        column = _column_name_if_simple(node.lexpr)
        if column is None:
            return None
        return WorkloadPredicateShape(
            column=column,
            kind="between",
            operator="BETWEEN",
            value_kind=_value_kind(node.rexpr),
        )

    return None


def _predicate_kind_for_operator(operator: str) -> str | None:
    if operator == "=":
        return "eq"
    if operator in {"<", "<="}:
        return "range_lt"
    if operator in {">", ">="}:
        return "range_gt"
    return None


def _limit_shape(node: object | None) -> WorkloadLimitShape:
    if node is None:
        return WorkloadLimitShape(kind="absent", value=None)
    if isinstance(node, ast.A_Const) and isinstance(node.val, ast.Integer):
        if node.val.ival is None:
            return WorkloadLimitShape(kind="constant", value=None)
        return WorkloadLimitShape(kind="constant", value=int(node.val.ival))
    if isinstance(node, ast.ParamRef):
        return WorkloadLimitShape(kind="parameter", value=None)
    return WorkloadLimitShape(kind=type(node).__name__, value=None)


def _recommend_filter_template(predicates: tuple[WorkloadPredicateShape, ...]) -> str | None:
    if not predicates:
        return None
    parts = []
    for predicate in predicates:
        if predicate.kind == "bool":
            parts.append(predicate.column)
        elif predicate.kind == "in":
            parts.append(f"{predicate.column} IN (<value>, ...)")
        elif predicate.kind == "between":
            parts.append(f"{predicate.column} BETWEEN <value> AND <value>")
        else:
            parts.append(f"{predicate.column} {predicate.operator} <value>")
    return " AND ".join(parts)


def _shape_notes(
    *,
    table: str | None,
    predicates: tuple[WorkloadPredicateShape, ...],
    unsupported: tuple[str, ...],
) -> tuple[str, ...]:
    notes: list[str] = []
    if table is None:
        notes.append("could not summarize a single base table")
    if not predicates:
        notes.append("no supported filter predicates found")
    if unsupported:
        notes.append("some predicates were unsupported and omitted from recommend template")
    notes.append("raw SQL text, SQL literals, and query vectors are not emitted")
    return tuple(notes)


def _shape_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _operator_name(name: object) -> str:
    if not isinstance(name, tuple) or len(name) != 1:
        return ""
    item = name[0]
    if isinstance(item, ast.String) and item.sval is not None:
        return str(item.sval)
    return ""


def _column_name_if_simple(node: object) -> str | None:
    if not isinstance(node, ast.ColumnRef) or node.fields is None:
        return None
    string_fields = [
        str(field.sval)
        for field in node.fields
        if isinstance(field, ast.String) and field.sval is not None
    ]
    if len(string_fields) != len(node.fields) or not string_fields:
        return None
    return string_fields[-1]


def _value_kind(node: object) -> str:
    if isinstance(node, ast.ParamRef):
        return "parameter"
    if isinstance(node, ast.A_Const):
        return "constant"
    if isinstance(node, ast.TypeCast):
        return _value_kind(node.arg)
    if isinstance(node, tuple):
        kinds = {_value_kind(item) for item in node}
        if len(kinds) == 1:
            return f"{next(iter(kinds))}_list"
        return "mixed_list"
    return "expression"


def _invert_operator(operator: str) -> str:
    return {
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
        "=": "=",
    }.get(operator, operator)


def _unsupported_expression_label(node: ast.A_Expr) -> str:
    operator = _operator_name(node.name)
    if operator:
        return f"unsupported operator {operator}"
    kind = node.kind.name if node.kind is not None else "unknown"
    return f"unsupported expression {kind}"
