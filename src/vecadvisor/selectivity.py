from __future__ import annotations

from bisect import bisect_left
from datetime import date, datetime, time
from math import prod
from typing import Any

from .models import ColumnStats, Predicate, PredicateKind, TableStats

EPS = 1e-12
DEFAULT_INEQ_SEL = 1.0 / 3.0


def clamp_selectivity(value: float) -> float:
    return min(1.0, max(EPS, value))


def eq_selectivity(column: ColumnStats, value: Any, n_rows: int) -> float:
    for mcv_value, freq in zip(column.mcv, column.mcf, strict=False):
        if mcv_value == value:
            return clamp_selectivity(freq)

    ndistinct = column.resolved_ndistinct(n_rows)
    mcv_count = len(column.mcv)
    mcf_sum = sum(column.mcf)
    remaining_freq = max(0.0, 1.0 - column.null_frac - mcf_sum)
    remaining_distinct = max(1.0, ndistinct - mcv_count)
    return clamp_selectivity(remaining_freq / remaining_distinct)


def unknown_eq_selectivity(column: ColumnStats, n_rows: int) -> float:
    ndistinct = column.resolved_ndistinct(n_rows)
    return clamp_selectivity((1.0 - column.null_frac) / ndistinct)


def range_selectivity(column: ColumnStats, op: PredicateKind, value: Any) -> float:
    if op is PredicateKind.RANGE_BETWEEN:
        if not isinstance(value, tuple) or len(value) != 2:
            return DEFAULT_INEQ_SEL
        lower = _range_selectivity_lt(column, value[0])
        upper = _range_selectivity_lt(column, value[1])
        return clamp_selectivity(max(upper - lower, EPS))
    if op is PredicateKind.RANGE_LT:
        return _range_selectivity_lt(column, value)
    if op is PredicateKind.RANGE_GT:
        return clamp_selectivity(1.0 - _range_selectivity_lt(column, value))
    return DEFAULT_INEQ_SEL


def _range_selectivity_lt(column: ColumnStats, value: Any) -> float:
    if not column.histogram:
        return DEFAULT_INEQ_SEL

    hist_fraction = _histogram_fraction_lt(column.histogram, value)
    if hist_fraction is None:
        return DEFAULT_INEQ_SEL

    mcv_less = 0.0
    mcf_sum = sum(column.mcf)
    for mcv_value, freq in zip(column.mcv, column.mcf, strict=False):
        comparison = _compare_values(mcv_value, value)
        if comparison is not None and comparison < 0:
            mcv_less += freq

    histogram_mass = max(0.0, 1.0 - column.null_frac - mcf_sum)
    return clamp_selectivity(mcv_less + histogram_mass * hist_fraction)


def _histogram_fraction_lt(histogram: tuple[Any, ...], value: Any) -> float | None:
    normalized = [_ordered_value(item) for item in histogram]
    target = _ordered_value(value)
    if target is None or any(item is None for item in normalized) or len(normalized) < 2:
        return None

    values = [item for item in normalized if item is not None]
    if _all_numeric(values) and isinstance(target, int | float):
        numeric_values = tuple(float(item) for item in values)
        if target <= numeric_values[0]:
            return 0.0
        if target >= numeric_values[-1]:
            return 1.0
        return _numeric_histogram_fraction_lt(numeric_values, float(target))

    if not _all_strings(values) or not isinstance(target, str):
        return None
    string_values = tuple(str(item) for item in values)
    if target <= string_values[0]:
        return 0.0
    if target >= string_values[-1]:
        return 1.0

    bucket = bisect_left(string_values, target)
    return bucket / (len(string_values) - 1)


def _numeric_histogram_fraction_lt(histogram: tuple[float, ...], value: float) -> float:
    bucket = bisect_left(histogram, value)
    lower = histogram[bucket - 1]
    upper = histogram[bucket]
    bucket_width = upper - lower
    bucket_fraction = 0.0 if bucket_width <= 0 else (value - lower) / bucket_width
    return ((bucket - 1) + max(0.0, min(1.0, bucket_fraction))) / (len(histogram) - 1)


def _compare_values(left: Any, right: Any) -> int | None:
    left_value = _ordered_value(left)
    right_value = _ordered_value(right)
    if left_value is None or right_value is None:
        return None
    if isinstance(left_value, int | float) and isinstance(right_value, int | float):
        if float(left_value) < float(right_value):
            return -1
        if float(left_value) > float(right_value):
            return 1
        return 0
    if isinstance(left_value, str) and isinstance(right_value, str):
        if left_value < right_value:
            return -1
        if left_value > right_value:
            return 1
        return 0
    return None


def _ordered_value(value: Any) -> float | str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, date):
        return datetime.combine(value, time()).timestamp()
    if isinstance(value, str):
        stripped = value.strip()
        numeric = _numeric_string(stripped)
        if numeric is not None:
            return numeric
        temporal = _temporal_string(stripped)
        if temporal is not None:
            return temporal
        return stripped
    return None


def _numeric_string(value: str) -> float | None:
    try:
        number = float(value)
    except ValueError:
        return None
    return number


def _temporal_string(value: str) -> float | None:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(normalized)
        except ValueError:
            return None
        parsed = datetime.combine(parsed_date, time())
    return parsed.timestamp()


def _all_numeric(values: list[float | str]) -> bool:
    return all(isinstance(value, int | float) for value in values)


def _all_strings(values: list[float | str]) -> bool:
    return all(isinstance(value, str) for value in values)


def predicate_selectivity(table: TableStats, predicate: Predicate) -> float:
    column = table.column(predicate.column)
    if predicate.kind is PredicateKind.EQ:
        if predicate.is_literal and predicate.values:
            return eq_selectivity(column, predicate.values[0], table.n_rows)
        return unknown_eq_selectivity(column, table.n_rows)
    range_kinds = {
        PredicateKind.RANGE_LT,
        PredicateKind.RANGE_GT,
        PredicateKind.RANGE_BETWEEN,
    }
    if predicate.kind in range_kinds:
        value = predicate.values if predicate.kind is PredicateKind.RANGE_BETWEEN else (
            predicate.values[0] if predicate.values else None
        )
        return range_selectivity(column, predicate.kind, value)
    if predicate.kind is PredicateKind.IN:
        return clamp_selectivity(
            sum(eq_selectivity(column, value, table.n_rows) for value in predicate.values)
        )
    return 0.5


def conjunction_selectivity(table: TableStats, predicates: tuple[Predicate, ...]) -> float:
    independent_estimate = prod(predicate_selectivity(table, predicate) for predicate in predicates)
    return clamp_selectivity(independent_estimate)


def rho_from_selectivities(s_global: float, s_local: float) -> float:
    ratio = s_local / max(s_global, EPS)
    return max(-1.0, min(1.0, (ratio - 1.0) / (ratio + 1.0)))
