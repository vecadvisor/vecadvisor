from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class PredicateKind(StrEnum):
    EQ = "eq"
    RANGE_LT = "range_lt"
    RANGE_GT = "range_gt"
    RANGE_BETWEEN = "range_between"
    IN = "in"
    BOOL = "bool"


class Strategy(StrEnum):
    EXACT = "exact"
    POSTFILTER = "postfilter"
    ITERATIVE_RELAXED = "iterative_relaxed"
    ITERATIVE_STRICT = "iterative_strict"
    PARTIAL = "partial"
    PARTITION = "partition"


@dataclass(frozen=True)
class ColumnStats:
    name: str
    n_distinct: float
    null_frac: float
    type_name: str = ""
    atttypmod: int = -1
    mcv: tuple[Any, ...] = ()
    mcf: tuple[float, ...] = ()
    histogram: tuple[Any, ...] | None = None
    correlation: float = 0.0
    avg_width: int = 0
    has_stats: bool = False

    def resolved_ndistinct(self, n_rows: int) -> float:
        if self.n_distinct >= 0:
            return max(self.n_distinct, 1.0)
        return max(-self.n_distinct * n_rows, 1.0)


@dataclass(frozen=True)
class IndexMeta:
    name: str
    method: str
    columns: tuple[str, ...]
    opclass: str | None = None
    is_partial: bool = False
    predicate: str | None = None
    m: int | None = None
    ef_construction: int | None = None
    lists: int | None = None
    pages: int = 0
    tuples: float = 0.0


@dataclass(frozen=True)
class ExtendedStatsMeta:
    name: str
    schema: str
    columns: tuple[str, ...]
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class TableStats:
    relname: str
    n_rows: int
    n_pages: int
    columns: tuple[ColumnStats, ...]
    indexes: tuple[IndexMeta, ...] = ()
    extended_stats: tuple[ExtendedStatsMeta, ...] = ()
    vector_dim: int = 0
    partitioned_by: tuple[str, ...] | None = None
    last_analyze: datetime | None = None
    last_autoanalyze: datetime | None = None
    n_live_tup: int | None = None
    n_mod_since_analyze: int | None = None
    stats_fingerprint: str = ""
    index_fingerprint: str = ""

    def column(self, name: str) -> ColumnStats:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"unknown column: {name}")


@dataclass(frozen=True)
class Predicate:
    column: str
    kind: PredicateKind
    values: tuple[Any, ...]
    is_literal: bool = True


@dataclass(frozen=True)
class QuerySpec:
    relname: str
    vector_column: str
    distance_op: str
    predicates: tuple[Predicate, ...]
    limit: int

    @property
    def has_only_literals(self) -> bool:
        return all(predicate.is_literal for predicate in self.predicates)


@dataclass(frozen=True)
class CalibrationProfile:
    dataset_id: str
    hardware_id: str
    index_method: str
    c_d: float
    c_scan: float
    c_h: float
    delta_strict: float = 0.0
    recall_curve: tuple[tuple[int, float], ...] = ((40, 0.9),)

    def recall_at(self, ef: int) -> float:
        points = sorted(self.recall_curve)
        if not points:
            return 0.0
        if ef <= points[0][0]:
            return points[0][1]
        if ef >= points[-1][0]:
            return points[-1][1]
        for (ef0, r0), (ef1, r1) in zip(points, points[1:], strict=False):
            if ef0 <= ef <= ef1:
                width = ef1 - ef0
                if width <= 0:
                    return r1
                alpha = (ef - ef0) / width
                return r0 + alpha * (r1 - r0)
        return points[-1][1]


@dataclass(frozen=True)
class LocalSelectivity:
    s_global: float
    s_local_p10: float
    s_local_median: float
    rho: float
    confidence: float
    sample_size: int
    passing_rows: int
    resolution_floor: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CostEstimate:
    strategy: Strategy
    est_latency_us: float
    est_recall: float
    est_returns_k: bool
    confidence: float
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategyPlan:
    strategy: Strategy
    ef_search: int | None = None
    uses_index: str | None = None
    requires_new_object: str | None = None
    sql_hint: str = ""


@dataclass(frozen=True)
class Recommendation:
    query: QuerySpec
    s_global: float
    s_local: float
    rho: float
    planner_would_pick: Strategy | None
    ranked: tuple[tuple[StrategyPlan, CostEstimate], ...]
    verdict: str


@dataclass(frozen=True)
class PlanNode:
    node_type: str
    relation_name: str | None = None
    index_name: str | None = None
    startup_cost: float | None = None
    total_cost: float | None = None
    plan_rows: float | None = None
    plan_width: int | None = None
    actual_rows: float | None = None
    actual_loops: int | None = None
    actual_total_time_ms: float | None = None
    rows_removed_by_filter: int | None = None
    filter_text: str | None = None
    index_cond: str | None = None
    order_by: str | None = None
    shared_hit_blocks: int | None = None
    shared_read_blocks: int | None = None
    children: tuple[PlanNode, ...] = ()

    def walk(self) -> tuple[PlanNode, ...]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return tuple(nodes)


@dataclass(frozen=True)
class PlanSummary:
    root: PlanNode
    planning_time_ms: float | None = None
    execution_time_ms: float | None = None
    raw: tuple[Any, ...] = ()


@dataclass(frozen=True)
class SelectivityCrossCheck:
    relation_name: str
    advisor_selectivity: float
    postgres_selectivity: float | None
    advisor_rows: float
    postgres_plan_rows: float | None
    absolute_delta: float | None
    ratio: float | None
    plan_node_type: str | None
    status: str = "unknown"
    severity: str = "unknown"
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StatisticsSuggestion:
    columns: tuple[str, ...]
    kinds: tuple[str, ...]
    ddl: str
    reason: str
    confidence: float
    advisor_selectivity: float
    postgres_selectivity: float | None
    ratio: float | None
