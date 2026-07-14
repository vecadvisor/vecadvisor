from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pglast import ast, enums, parse_sql
from pglast.parser import ParseError

from .models import Predicate, PredicateKind, QuerySpec


class QueryParseError(ValueError):
    """Raised when a filter cannot be represented by the supported predicate model."""


def parse_filter(
    filter_sql: str,
    *,
    allowed_columns: Iterable[str] | None = None,
) -> tuple[Predicate, ...]:
    """Parse a restricted SQL WHERE predicate into advisor predicates.

    The current parser intentionally supports only conjunctions of simple column predicates:
    equality, inequality ranges, BETWEEN, IN, and bare boolean columns. This is
    enough for cost-model work while keeping unsupported SQL from silently
    producing bad estimates.
    """

    stripped = filter_sql.strip()
    if not stripped:
        raise QueryParseError("filter predicate is empty")

    allowed = set(allowed_columns) if allowed_columns is not None else None
    try:
        statements = parse_sql(f"SELECT * FROM __vecadvisor_filter_probe WHERE {stripped}")
    except ParseError as exc:
        raise QueryParseError(f"could not parse filter predicate: {exc}") from exc

    if len(statements) != 1:
        raise QueryParseError("filter predicate must contain exactly one expression")

    statement = statements[0].stmt
    if not isinstance(statement, ast.SelectStmt) or statement.whereClause is None:
        raise QueryParseError("filter predicate must parse as a WHERE clause")

    predicates = _parse_predicate_node(statement.whereClause, allowed)
    if not predicates:
        raise QueryParseError("filter predicate did not contain supported predicates")
    return tuple(predicates)


def query_spec_from_filter(
    *,
    relname: str,
    vector_column: str,
    filter_sql: str,
    limit: int,
    distance_op: str = "<->",
    allowed_columns: Iterable[str] | None = None,
) -> QuerySpec:
    """Build a QuerySpec from a SQL filter string."""

    return QuerySpec(
        relname=relname,
        vector_column=vector_column,
        distance_op=distance_op,
        predicates=parse_filter(filter_sql, allowed_columns=allowed_columns),
        limit=limit,
    )


def build_filter_select_sql(relname: str, filter_sql: str) -> str:
    """Build the read-only query used for PostgreSQL selectivity cross-checks."""

    return f"SELECT * FROM {quote_qualified_identifier(relname)} WHERE {filter_sql.strip()}"


def quote_qualified_identifier(qualified_name: str) -> str:
    """Quote a schema-qualified PostgreSQL identifier returned by catalog introspection."""

    parts = qualified_name.split(".")
    if not parts or any(not part for part in parts):
        raise QueryParseError(f"invalid relation name: {qualified_name!r}")
    return ".".join(quote_identifier(part) for part in parts)


def quote_identifier(identifier: str) -> str:
    """Quote one PostgreSQL identifier."""

    if not identifier:
        raise QueryParseError("identifier must not be empty")
    return '"' + identifier.replace('"', '""') + '"'


def _parse_predicate_node(node: Any, allowed_columns: set[str] | None) -> list[Predicate]:
    if isinstance(node, ast.BoolExpr):
        if node.boolop is not enums.BoolExprType.AND_EXPR:
            raise QueryParseError("only AND predicates are supported")
        if node.args is None:
            raise QueryParseError("AND predicate was empty")
        predicates: list[Predicate] = []
        for arg in node.args:
            predicates.extend(_parse_predicate_node(arg, allowed_columns))
        return predicates

    if isinstance(node, ast.A_Expr):
        return [_parse_a_expr(node, allowed_columns)]

    if isinstance(node, ast.ColumnRef):
        column = _column_name(node)
        _validate_column(column, allowed_columns)
        return [Predicate(column=column, kind=PredicateKind.BOOL, values=(True,))]

    raise QueryParseError(f"unsupported predicate form: {type(node).__name__}")


def _parse_a_expr(node: ast.A_Expr, allowed_columns: set[str] | None) -> Predicate:
    if node.kind is enums.A_Expr_Kind.AEXPR_OP:
        return _parse_operator_expr(node, allowed_columns)
    if node.kind is enums.A_Expr_Kind.AEXPR_IN:
        column = _column_name(node.lexpr)
        _validate_column(column, allowed_columns)
        values = _literal_tuple(node.rexpr)
        if not values:
            raise QueryParseError("IN predicates must contain at least one literal")
        return Predicate(column=column, kind=PredicateKind.IN, values=values)
    if node.kind is enums.A_Expr_Kind.AEXPR_BETWEEN:
        column = _column_name(node.lexpr)
        _validate_column(column, allowed_columns)
        values = _literal_tuple(node.rexpr)
        if len(values) != 2:
            raise QueryParseError("BETWEEN predicates must contain exactly two literals")
        return Predicate(column=column, kind=PredicateKind.RANGE_BETWEEN, values=values)
    kind_name = node.kind.name if node.kind is not None else "unknown"
    raise QueryParseError(f"unsupported expression kind: {kind_name}")


def _parse_operator_expr(node: ast.A_Expr, allowed_columns: set[str] | None) -> Predicate:
    op = _operator_name(node.name)
    if isinstance(node.lexpr, ast.ColumnRef):
        column = _column_name(node.lexpr)
        value = _literal_value(node.rexpr)
        return _predicate_for_operator(column, op, value, allowed_columns)

    if isinstance(node.rexpr, ast.ColumnRef):
        column = _column_name(node.rexpr)
        value = _literal_value(node.lexpr)
        return _predicate_for_operator(column, _invert_operator(op), value, allowed_columns)

    raise QueryParseError("comparison predicates must compare one column to one literal")


def _predicate_for_operator(
    column: str,
    op: str,
    value: Any,
    allowed_columns: set[str] | None,
) -> Predicate:
    _validate_column(column, allowed_columns)
    if op == "=":
        return Predicate(column=column, kind=PredicateKind.EQ, values=(value,))
    if op in {"<", "<="}:
        return Predicate(column=column, kind=PredicateKind.RANGE_LT, values=(value,))
    if op in {">", ">="}:
        return Predicate(column=column, kind=PredicateKind.RANGE_GT, values=(value,))
    raise QueryParseError(f"unsupported comparison operator: {op}")


def _operator_name(name: Any) -> str:
    if not isinstance(name, tuple) or len(name) != 1:
        raise QueryParseError("only simple comparison operators are supported")
    item = name[0]
    if not isinstance(item, ast.String):
        raise QueryParseError("operator name was not a simple token")
    if item.sval is None:
        raise QueryParseError("operator name was empty")
    return str(item.sval)


def _invert_operator(op: str) -> str:
    inverted = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "=": "="}
    if op not in inverted:
        raise QueryParseError(f"unsupported comparison operator: {op}")
    return inverted[op]


def _column_name(node: Any) -> str:
    if not isinstance(node, ast.ColumnRef):
        raise QueryParseError("predicate left side must be a simple column reference")
    if node.fields is None or len(node.fields) != 1:
        raise QueryParseError("qualified columns and expressions are not supported in filters")
    field = node.fields[0]
    if not isinstance(field, ast.String):
        raise QueryParseError("column reference must name a concrete column")
    if field.sval is None:
        raise QueryParseError("column reference was empty")
    return str(field.sval)


def _validate_column(column: str, allowed_columns: set[str] | None) -> None:
    if allowed_columns is not None and column not in allowed_columns:
        raise QueryParseError(f"unknown filter column: {column}")


def _literal_tuple(node: Any) -> tuple[Any, ...]:
    if isinstance(node, tuple):
        return tuple(_literal_value(item) for item in node)
    return (_literal_value(node),)


def _literal_value(node: Any) -> Any:
    if isinstance(node, ast.TypeCast):
        return _literal_value(node.arg)
    if not isinstance(node, ast.A_Const):
        raise QueryParseError("predicate value must be a literal")
    if node.isnull:
        raise QueryParseError("NULL predicates are not supported")

    value = node.val
    if isinstance(value, ast.Integer):
        return value.ival
    if isinstance(value, ast.Float):
        if value.fval is None:
            raise QueryParseError("float literal was empty")
        return float(value.fval)
    if isinstance(value, ast.String):
        return value.sval
    if isinstance(value, ast.Boolean):
        return value.boolval

    raise QueryParseError(f"unsupported literal type: {type(value).__name__}")
