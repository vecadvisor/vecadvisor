from __future__ import annotations

import hashlib
import re

from .models import Predicate, QuerySpec, SelectivityCrossCheck, StatisticsSuggestion, TableStats
from .query_spec import quote_identifier, quote_qualified_identifier

DEFAULT_STATS_KINDS = ("dependencies", "mcv")
DIVERGENCE_RATIO_HIGH = 2.0
DIVERGENCE_RATIO_LOW = 0.5
MIN_ABSOLUTE_DELTA = 0.05


def suggest_statistics(
    *,
    table: TableStats,
    query: QuerySpec,
    cross_check: SelectivityCrossCheck,
) -> tuple[StatisticsSuggestion, ...]:
    """Suggest extended statistics for multi-column filters when useful."""

    columns = _predicate_columns(query.predicates)
    if len(columns) < 2 or _has_covering_extended_stats(table, columns):
        return ()

    reason, confidence = _suggestion_reason(cross_check)
    ddl = _create_statistics_ddl(table=table, columns=columns)
    return (
        StatisticsSuggestion(
            columns=columns,
            kinds=DEFAULT_STATS_KINDS,
            ddl=ddl,
            reason=reason,
            confidence=confidence,
            advisor_selectivity=cross_check.advisor_selectivity,
            postgres_selectivity=cross_check.postgres_selectivity,
            ratio=cross_check.ratio,
        ),
    )


def _predicate_columns(predicates: tuple[Predicate, ...]) -> tuple[str, ...]:
    columns: list[str] = []
    seen: set[str] = set()
    for predicate in predicates:
        if predicate.column in seen:
            continue
        seen.add(predicate.column)
        columns.append(predicate.column)
    return tuple(columns)


def _has_covering_extended_stats(table: TableStats, columns: tuple[str, ...]) -> bool:
    wanted = set(columns)
    for stats in table.extended_stats:
        if not wanted.issubset(set(stats.columns)):
            continue
        if {"dependencies", "mcv"}.intersection(stats.kinds):
            return True
    return False


def _suggestion_reason(cross_check: SelectivityCrossCheck) -> tuple[str, float]:
    if _has_selectivity_divergence(cross_check):
        return (
            "multi-column selectivity estimate diverges from PostgreSQL plan estimate; "
            "extended statistics may improve correlated-filter estimates",
            0.75,
        )
    return (
        "multi-column filter relies on independence assumptions and no covering extended "
        "statistics were found",
        0.35,
    )


def _has_selectivity_divergence(cross_check: SelectivityCrossCheck) -> bool:
    if cross_check.postgres_selectivity is None:
        return False
    if cross_check.absolute_delta is not None and cross_check.absolute_delta >= MIN_ABSOLUTE_DELTA:
        return True
    if cross_check.ratio is None:
        return False
    return cross_check.ratio >= DIVERGENCE_RATIO_HIGH or cross_check.ratio <= DIVERGENCE_RATIO_LOW


def _create_statistics_ddl(*, table: TableStats, columns: tuple[str, ...]) -> str:
    name = _statistics_name(table.relname, columns)
    qualified_name = _qualified_statistics_name(table.relname, name)
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    return (
        f"CREATE STATISTICS IF NOT EXISTS {qualified_name} "
        f"({', '.join(DEFAULT_STATS_KINDS)}) "
        f"ON {column_sql} "
        f"FROM {quote_qualified_identifier(table.relname)};"
    )


def _statistics_name(relname: str, columns: tuple[str, ...]) -> str:
    table_name = relname.rsplit(".", 1)[-1]
    column_part = "_".join(columns)
    digest = hashlib.sha1(f"{relname}:{','.join(columns)}".encode()).hexdigest()[:10]
    base = re.sub(r"[^A-Za-z0-9_]+", "_", f"vecadvisor_{table_name}_{column_part}_{digest}")
    suffix = "_stats"
    return f"{base[: 63 - len(suffix)]}{suffix}"


def _qualified_statistics_name(relname: str, stats_name: str) -> str:
    parts = relname.split(".")
    if len(parts) <= 1:
        return quote_identifier(stats_name)
    return f"{quote_identifier(parts[0])}.{quote_identifier(stats_name)}"
