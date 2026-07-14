from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

from .costmodel import (
    DEFAULT_HNSW_MAX_SCAN_TUPLES,
    choose_best,
    cost_exact,
    cost_iterative,
    cost_partial,
    cost_partition,
    cost_postfilter,
)
from .models import (
    CalibrationProfile,
    CostEstimate,
    IndexMeta,
    LocalSelectivity,
    Predicate,
    PredicateKind,
    QuerySpec,
    Recommendation,
    Strategy,
    StrategyPlan,
    TableStats,
)
from .query_spec import QueryParseError, parse_filter, quote_identifier, quote_qualified_identifier
from .selectivity import conjunction_selectivity, rho_from_selectivities

DEFAULT_EF_SEARCH = 40
DEFAULT_RECALL_TARGET = 0.9
DEFAULT_CALIBRATION = CalibrationProfile(
    dataset_id="mvp1-default",
    hardware_id="uncalibrated",
    index_method="hnsw",
    c_d=0.01,
    c_scan=0.004,
    c_h=2.0,
    delta_strict=0.2,
    recall_curve=((40, 0.9), (80, 0.95), (160, 0.98)),
)
LOW_CONFIDENCE_WITHOUT_LOCAL_PROBE = 0.35
DEFAULT_GLOBAL_CONFIDENCE = 0.65
PARTIAL_SUGGESTION_CONFIDENCE = 0.45
PARTITION_CONFIDENCE = 0.5
MAX_PARTIAL_INDEX_SELECTIVITY = 0.10
MAX_PARTIAL_INDEX_NDISTINCT = 1024.0


def build_recommendation(
    *,
    table: TableStats,
    query: QuerySpec,
    s_global: float,
    local_selectivity: LocalSelectivity | None = None,
    calibration: CalibrationProfile = DEFAULT_CALIBRATION,
    ef_search: int = DEFAULT_EF_SEARCH,
    recall_target: float = DEFAULT_RECALL_TARGET,
    max_scan_tuples: int = DEFAULT_HNSW_MAX_SCAN_TUPLES,
    planner_would_pick: Strategy | None = None,
    supports_iterative_scan: bool = True,
) -> Recommendation:
    """Rank viable filtered vector-search strategies for one query shape."""

    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if not 0.0 < recall_target <= 1.0:
        raise ValueError("recall_target must be in (0, 1]")

    local_signal = _local_signal(s_global, local_selectivity)
    vector_index = _find_vector_index(table, query.vector_column)
    global_confidence = _global_confidence(table, query)
    ann_confidence = (
        local_selectivity.confidence
        if local_selectivity is not None
        else LOW_CONFIDENCE_WITHOUT_LOCAL_PROBE
    )

    ranked = _rank_candidates(
        candidates=_candidate_plans(
            table=table,
            query=query,
            s_global=s_global,
            s_local=local_signal.s_local,
            calibration=calibration,
            ef_search=ef_search,
            recall_target=recall_target,
            max_scan_tuples=max_scan_tuples,
            supports_iterative_scan=supports_iterative_scan,
            vector_index=vector_index,
            global_confidence=global_confidence,
            ann_confidence=ann_confidence,
            local_probe_notes=_local_probe_notes(local_selectivity),
        ),
        recall_target=recall_target,
    )
    best_plan, best_estimate = ranked[0]

    return Recommendation(
        query=query,
        s_global=s_global,
        s_local=local_signal.s_local,
        rho=local_signal.rho,
        planner_would_pick=planner_would_pick,
        ranked=ranked,
        verdict=_verdict(
            best_plan=best_plan,
            best=best_estimate,
            vector_index=vector_index,
            has_local_probe=local_selectivity is not None,
        ),
    )


def _candidate_plans(
    *,
    table: TableStats,
    query: QuerySpec,
    s_global: float,
    s_local: float,
    calibration: CalibrationProfile,
    ef_search: int,
    recall_target: float,
    max_scan_tuples: int,
    supports_iterative_scan: bool,
    vector_index: IndexMeta | None,
    global_confidence: float,
    ann_confidence: float,
    local_probe_notes: tuple[str, ...],
) -> tuple[tuple[StrategyPlan, CostEstimate], ...]:
    recall_ef = _choose_recall_ef(
        calibration=calibration,
        recall_target=recall_target,
        minimum_ef=max(ef_search, _min_calibrated_ef(calibration)),
    )
    postfilter_ef = _choose_postfilter_ef(
        calibration=calibration,
        s_local=s_local,
        limit=query.limit,
        recall_target=recall_target,
        minimum_ef=max(ef_search, _min_calibrated_ef(calibration)),
    )
    candidates = [
        (
            StrategyPlan(
                strategy=Strategy.EXACT,
                sql_hint=(
                    "Filter first with scalar predicates, then compute exact vector distance "
                    f"ORDER BY {query.vector_column} {query.distance_op} $query "
                    f"LIMIT {query.limit}."
                ),
            ),
            _append_notes(
                cost_exact(
                    n_rows=table.n_rows,
                    s_global=s_global,
                    limit=query.limit,
                    cal=calibration,
                    confidence=global_confidence,
                ),
                ("exact is the recall-safe fallback",),
                calibration=calibration,
            ),
        )
    ]
    if vector_index is not None:
        candidates.append(
            (
                StrategyPlan(
                    strategy=Strategy.POSTFILTER,
                    ef_search=postfilter_ef,
                    uses_index=vector_index.name,
                    sql_hint=(
                        f"SET LOCAL hnsw.ef_search = {postfilter_ef}; "
                        "SET LOCAL hnsw.iterative_scan = off; "
                        f"use {vector_index.method} index {vector_index.name}; "
                        "scalar filter is applied after ANN candidates."
                    ),
                ),
                _append_notes(
                    cost_postfilter(
                        n_rows=table.n_rows,
                        s_local=s_local,
                        limit=query.limit,
                        ef_search=postfilter_ef,
                        cal=calibration,
                        confidence=ann_confidence,
                    ),
                    _postfilter_ef_notes(
                        chosen_ef=postfilter_ef,
                        requested_ef=ef_search,
                        s_local=s_local,
                        limit=query.limit,
                        recall_target=recall_target,
                    )
                    + _iterative_capability_notes(supports_iterative_scan)
                    + local_probe_notes,
                    calibration=calibration,
                ),
            )
        )
        if supports_iterative_scan:
            candidates.extend(
                [
                    (
                        StrategyPlan(
                            strategy=Strategy.ITERATIVE_RELAXED,
                            ef_search=recall_ef,
                            uses_index=vector_index.name,
                            sql_hint=(
                                f"SET LOCAL hnsw.ef_search = {recall_ef}; "
                                "SET LOCAL hnsw.iterative_scan = relaxed_order; "
                                f"SET LOCAL hnsw.max_scan_tuples = {max_scan_tuples}; "
                                f"use {vector_index.method} index {vector_index.name} with "
                                "relaxed iterative scan so ANN candidates expand until enough "
                                "rows pass."
                            ),
                        ),
                        _append_notes(
                            cost_iterative(
                                n_rows=table.n_rows,
                                s_local=s_local,
                                limit=query.limit,
                                ef_search=recall_ef,
                                cal=calibration,
                                strict_order=False,
                                max_scan_tuples=max_scan_tuples,
                                confidence=ann_confidence,
                            ),
                            _recall_ef_notes(
                                chosen_ef=recall_ef,
                                requested_ef=ef_search,
                                recall_target=recall_target,
                            )
                            + local_probe_notes,
                            calibration=calibration,
                        ),
                    ),
                    (
                        StrategyPlan(
                            strategy=Strategy.ITERATIVE_STRICT,
                            ef_search=recall_ef,
                            uses_index=vector_index.name,
                            sql_hint=(
                                f"SET LOCAL hnsw.ef_search = {recall_ef}; "
                                "SET LOCAL hnsw.iterative_scan = strict_order; "
                                f"SET LOCAL hnsw.max_scan_tuples = {max_scan_tuples}; "
                                f"use {vector_index.method} index {vector_index.name} with "
                                "strict iterative scan when exact distance ordering must be "
                                "preserved."
                            ),
                        ),
                        _append_notes(
                            cost_iterative(
                                n_rows=table.n_rows,
                                s_local=s_local,
                                limit=query.limit,
                                ef_search=recall_ef,
                                cal=calibration,
                                strict_order=True,
                                max_scan_tuples=max_scan_tuples,
                                confidence=ann_confidence,
                            ),
                            _recall_ef_notes(
                                chosen_ef=recall_ef,
                                requested_ef=ef_search,
                                recall_target=recall_target,
                            )
                            + local_probe_notes,
                            calibration=calibration,
                        ),
                    ),
                ]
            )

    candidates.extend(
        _partial_candidates(
            table=table,
            query=query,
            vector_index=vector_index,
            s_global=s_global,
            calibration=calibration,
            ef_search=recall_ef,
            requested_ef=ef_search,
            recall_target=recall_target,
            ann_confidence=ann_confidence,
        )
    )
    if vector_index is not None:
        candidates.extend(
            _partition_candidates(
                table=table,
                query=query,
                vector_index=vector_index,
                s_global=s_global,
                calibration=calibration,
                ef_search=recall_ef,
                requested_ef=ef_search,
                recall_target=recall_target,
            )
        )
    return tuple(candidates)


def _rank_candidates(
    *,
    candidates: tuple[tuple[StrategyPlan, CostEstimate], ...],
    recall_target: float,
) -> tuple[tuple[StrategyPlan, CostEstimate], ...]:
    estimates = tuple(estimate for _, estimate in candidates)
    winner = choose_best(estimates, recall_target=recall_target)

    def sort_key(candidate: tuple[StrategyPlan, CostEstimate]) -> tuple[int, int, float, float]:
        estimate = candidate[1]
        is_winner = 0 if estimate.strategy is winner.strategy else 1
        is_viable = 0 if _is_viable(estimate, recall_target) else 1
        return (is_winner, is_viable, estimate.est_latency_us, -estimate.confidence)

    return tuple(sorted(candidates, key=sort_key))


def _is_viable(estimate: CostEstimate, recall_target: float) -> bool:
    return estimate.est_returns_k and estimate.est_recall >= recall_target


def _choose_postfilter_ef(
    *,
    calibration: CalibrationProfile,
    s_local: float,
    limit: int,
    recall_target: float,
    minimum_ef: int,
) -> int:
    survivor_ef = _required_survivor_ef(s_local=s_local, limit=limit)
    return _choose_recall_ef(
        calibration=calibration,
        recall_target=recall_target,
        minimum_ef=max(minimum_ef, survivor_ef),
    )


def _choose_recall_ef(
    *,
    calibration: CalibrationProfile,
    recall_target: float,
    minimum_ef: int,
) -> int:
    if minimum_ef <= 0:
        raise ValueError("minimum_ef must be positive")
    if calibration.recall_at(minimum_ef) >= recall_target:
        return minimum_ef
    for ef, recall in sorted(calibration.recall_curve):
        if ef >= minimum_ef and recall >= recall_target:
            return ef
    if calibration.recall_curve:
        return max(minimum_ef, max(ef for ef, _ in calibration.recall_curve))
    return minimum_ef


def _required_survivor_ef(*, s_local: float, limit: int) -> int:
    return max(1, math.ceil(limit / max(s_local, 1e-12)))


def _min_calibrated_ef(calibration: CalibrationProfile) -> int:
    positive_efs = [ef for ef, _ in calibration.recall_curve if ef > 0]
    if not positive_efs:
        return 1
    return min(positive_efs)


def _recall_ef_notes(
    *,
    chosen_ef: int,
    requested_ef: int,
    recall_target: float,
) -> tuple[str, ...]:
    if chosen_ef == requested_ef:
        return (f"ef_search={chosen_ef} satisfies recall target {recall_target:.3g}",)
    return (
        f"ef_search raised from {requested_ef} to {chosen_ef} to satisfy "
        f"recall target {recall_target:.3g}",
    )


def _postfilter_ef_notes(
    *,
    chosen_ef: int,
    requested_ef: int,
    s_local: float,
    limit: int,
    recall_target: float,
) -> tuple[str, ...]:
    survivor_ef = _required_survivor_ef(s_local=s_local, limit=limit)
    if chosen_ef == requested_ef:
        return (
            f"ef_search={chosen_ef} satisfies recall target {recall_target:.3g} "
            f"and expected survivors >= {limit}",
        )
    return (
        f"ef_search raised from {requested_ef} to {chosen_ef} to satisfy recall target "
        f"{recall_target:.3g} and survivor ef >= {survivor_ef}",
    )


def _iterative_capability_notes(supports_iterative_scan: bool) -> tuple[str, ...]:
    if supports_iterative_scan:
        return ()
    return ("iterative HNSW scan is unavailable on this pgvector version",)


@dataclass(frozen=True)
class _PartialMatch:
    index: IndexMeta
    predicates: tuple[Predicate, ...]


def _partial_candidates(
    *,
    table: TableStats,
    query: QuerySpec,
    vector_index: IndexMeta | None,
    s_global: float,
    calibration: CalibrationProfile,
    ef_search: int,
    requested_ef: int,
    recall_target: float,
    ann_confidence: float,
) -> tuple[tuple[StrategyPlan, CostEstimate], ...]:
    match = _matching_partial_index(table, query)
    if match is not None:
        return (
            (
                StrategyPlan(
                    strategy=Strategy.PARTIAL,
                    ef_search=ef_search,
                    uses_index=match.index.name,
                    sql_hint=(
                        f"SET LOCAL hnsw.ef_search = {ef_search}; "
                        f"use existing partial {match.index.method} index {match.index.name}; "
                        "its predicate exactly matches the supported filter predicates."
                    ),
                ),
                _append_notes(
                    cost_partial(
                        n_rows=table.n_rows,
                        s_index=conjunction_selectivity(table, match.predicates),
                        ef_search=ef_search,
                        cal=calibration,
                        confidence=ann_confidence,
                    ),
                    _recall_ef_notes(
                        chosen_ef=ef_search,
                        requested_ef=requested_ef,
                        recall_target=recall_target,
                    )
                    + ("existing partial index exactly matches the filter predicate",),
                    calibration=calibration,
                ),
            ),
        )

    if not _can_suggest_partial_index(table, query, s_global):
        return ()

    predicate_sql = _predicate_conjunction_sql(query.predicates)
    ddl = _partial_index_ddl(
        table=table,
        query=query,
        predicate_sql=predicate_sql,
        vector_index=vector_index,
    )
    return (
        (
            StrategyPlan(
                strategy=Strategy.PARTIAL,
                ef_search=ef_search,
                requires_new_object=ddl,
                sql_hint=(
                    f"SET LOCAL hnsw.ef_search = {ef_search}; "
                    "Create a partial HNSW index for this stable literal filter before "
                    "using the partial-index strategy."
                ),
            ),
            _append_notes(
                cost_partial(
                    n_rows=table.n_rows,
                    s_index=s_global,
                    ef_search=ef_search,
                    cal=calibration,
                    confidence=PARTIAL_SUGGESTION_CONFIDENCE,
                ),
                _recall_ef_notes(
                    chosen_ef=ef_search,
                    requested_ef=requested_ef,
                    recall_target=recall_target,
                )
                + (
                    "partial index suggestion is limited to low-cardinality literal filters",
                    "validate workload frequency before creating new indexes",
                ),
                calibration=calibration,
            ),
        ),
    )


def _partition_candidates(
    *,
    table: TableStats,
    query: QuerySpec,
    vector_index: IndexMeta,
    s_global: float,
    calibration: CalibrationProfile,
    ef_search: int,
    requested_ef: int,
    recall_target: float,
) -> tuple[tuple[StrategyPlan, CostEstimate], ...]:
    if not table.partitioned_by:
        return ()
    matching_columns = tuple(
        predicate.column
        for predicate in query.predicates
        if _partition_def_mentions_column(table.partitioned_by, predicate.column)
    )
    if not matching_columns:
        return ()

    partition_def = ", ".join(table.partitioned_by)
    return (
        (
            StrategyPlan(
                strategy=Strategy.PARTITION,
                ef_search=ef_search,
                uses_index=vector_index.name,
                sql_hint=(
                    f"SET LOCAL hnsw.ef_search = {ef_search}; "
                    f"Use partition pruning on {partition_def}; keep per-partition "
                    f"{vector_index.method} indexes on {query.vector_column}."
                ),
            ),
            _append_notes(
                cost_partition(
                    n_rows=table.n_rows,
                    s_partition=s_global,
                    ef_search=ef_search,
                    cal=calibration,
                    confidence=PARTITION_CONFIDENCE,
                ),
                _recall_ef_notes(
                    chosen_ef=ef_search,
                    requested_ef=requested_ef,
                    recall_target=recall_target,
                )
                + (
                    f"partition key overlaps filter columns: {', '.join(matching_columns)}",
                    "assumes partition pruning and per-partition vector indexes are available",
                ),
                calibration=calibration,
            ),
        ),
    )


def _matching_partial_index(table: TableStats, query: QuerySpec) -> _PartialMatch | None:
    allowed_columns = tuple(column.name for column in table.columns)
    query_keys = {_predicate_key(predicate) for predicate in query.predicates}
    for index in table.indexes:
        if (
            not index.is_partial
            or index.method not in {"hnsw", "ivfflat"}
            or query.vector_column not in index.columns
            or index.predicate is None
        ):
            continue
        try:
            predicates = parse_filter(index.predicate, allowed_columns=allowed_columns)
        except QueryParseError:
            continue
        if {_predicate_key(predicate) for predicate in predicates} == query_keys:
            return _PartialMatch(index=index, predicates=predicates)
    return None


def _can_suggest_partial_index(table: TableStats, query: QuerySpec, s_global: float) -> bool:
    if not query.predicates or s_global > MAX_PARTIAL_INDEX_SELECTIVITY:
        return False
    for predicate in query.predicates:
        if not _is_partial_index_safe_predicate(predicate):
            return False
        try:
            column = table.column(predicate.column)
        except KeyError:
            return False
        if (
            not column.has_stats
            or column.resolved_ndistinct(table.n_rows) > MAX_PARTIAL_INDEX_NDISTINCT
        ):
            return False
    return True


def _is_partial_index_safe_predicate(predicate: Predicate) -> bool:
    if not predicate.is_literal:
        return False
    if predicate.kind in {PredicateKind.EQ, PredicateKind.IN}:
        return bool(predicate.values)
    if predicate.kind is PredicateKind.BOOL:
        return predicate.values in {(True,), (False,)}
    return False


def _predicate_key(predicate: Predicate) -> tuple[str, PredicateKind, tuple[str, ...], bool]:
    return (
        predicate.column,
        predicate.kind,
        tuple(_literal_key(value) for value in predicate.values),
        predicate.is_literal,
    )


def _literal_key(value: Any) -> str:
    return f"{type(value).__name__}:{value!r}"


def _predicate_conjunction_sql(predicates: tuple[Predicate, ...]) -> str:
    return " AND ".join(_predicate_sql(predicate) for predicate in predicates)


def _predicate_sql(predicate: Predicate) -> str:
    column_sql = quote_identifier(predicate.column)
    if predicate.kind is PredicateKind.EQ:
        return f"{column_sql} = {_sql_literal(predicate.values[0])}"
    if predicate.kind is PredicateKind.IN:
        values = ", ".join(_sql_literal(value) for value in predicate.values)
        return f"{column_sql} IN ({values})"
    if predicate.kind is PredicateKind.BOOL:
        value = bool(predicate.values[0]) if predicate.values else True
        return column_sql if value else f"NOT {column_sql}"
    raise ValueError(f"predicate is not safe for a partial-index predicate: {predicate.kind}")


def _partial_index_ddl(
    *,
    table: TableStats,
    query: QuerySpec,
    predicate_sql: str,
    vector_index: IndexMeta | None,
) -> str:
    index_name = _partial_index_name(query.relname, query.vector_column, predicate_sql)
    opclass = _vector_opclass(table, query, vector_index)
    return (
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
        f"{quote_identifier(index_name)} "
        f"ON {quote_qualified_identifier(query.relname)} "
        f"USING hnsw ({quote_identifier(query.vector_column)} {opclass}) "
        f"WHERE {predicate_sql};"
    )


def _partial_index_name(relname: str, vector_column: str, predicate_sql: str) -> str:
    table_name = relname.rsplit(".", 1)[-1]
    digest = hashlib.sha1(f"{relname}:{vector_column}:{predicate_sql}".encode()).hexdigest()[:10]
    base = re.sub(r"[^A-Za-z0-9_]+", "_", f"vecadvisor_{table_name}_{vector_column}_{digest}")
    suffix = "_hnsw_idx"
    return f"{base[: 63 - len(suffix)]}{suffix}"


def _vector_opclass(
    table: TableStats,
    query: QuerySpec,
    vector_index: IndexMeta | None,
) -> str:
    if vector_index is not None and vector_index.opclass is not None:
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", vector_index.opclass):
            return vector_index.opclass

    try:
        type_name = table.column(query.vector_column).type_name
    except KeyError:
        type_name = "vector"

    opclasses = {
        ("vector", "<->"): "vector_l2_ops",
        ("vector", "<#>"): "vector_ip_ops",
        ("vector", "<=>"): "vector_cosine_ops",
        ("vector", "<+>"): "vector_l1_ops",
        ("halfvec", "<->"): "halfvec_l2_ops",
        ("halfvec", "<#>"): "halfvec_ip_ops",
        ("halfvec", "<=>"): "halfvec_cosine_ops",
        ("halfvec", "<+>"): "halfvec_l1_ops",
        ("sparsevec", "<->"): "sparsevec_l2_ops",
        ("sparsevec", "<#>"): "sparsevec_ip_ops",
        ("sparsevec", "<=>"): "sparsevec_cosine_ops",
        ("bit", "<~>"): "bit_hamming_ops",
        ("bit", "<%>"): "bit_jaccard_ops",
    }
    return opclasses.get((type_name, query.distance_op), "vector_l2_ops")


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("partial-index predicate literals must be finite")
        return format(value, ".17g")
    return "'" + str(value).replace("'", "''") + "'"


def _partition_def_mentions_column(partitioned_by: tuple[str, ...], column: str) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(column)}(?![A-Za-z0-9_])", re.I)
    return any(pattern.search(partition_def) for partition_def in partitioned_by)


def _find_vector_index(table: TableStats, vector_column: str) -> IndexMeta | None:
    for method in ("hnsw", "ivfflat"):
        for index in table.indexes:
            if not index.is_partial and index.method == method and vector_column in index.columns:
                return index
    return None


def _global_confidence(table: TableStats, query: QuerySpec) -> float:
    if not query.predicates:
        return DEFAULT_GLOBAL_CONFIDENCE
    known_stats = 0
    for predicate in query.predicates:
        try:
            column = table.column(predicate.column)
        except KeyError:
            continue
        if column.has_stats:
            known_stats += 1
    return max(0.25, DEFAULT_GLOBAL_CONFIDENCE * known_stats / len(query.predicates))


class _LocalSignal:
    def __init__(self, *, s_local: float, rho: float) -> None:
        self.s_local = s_local
        self.rho = rho


def _local_signal(
    s_global: float,
    local_selectivity: LocalSelectivity | None,
) -> _LocalSignal:
    if local_selectivity is None:
        return _LocalSignal(s_local=s_global, rho=rho_from_selectivities(s_global, s_global))
    return _LocalSignal(s_local=local_selectivity.s_local_p10, rho=local_selectivity.rho)


def _local_probe_notes(local_selectivity: LocalSelectivity | None) -> tuple[str, ...]:
    if local_selectivity is None:
        return ("local selectivity not measured; using global selectivity as fallback",)
    return (
        "local selectivity measured from query neighborhood; costing uses p10",
        *local_selectivity.notes,
    )


def _append_notes(
    estimate: CostEstimate,
    notes: tuple[str, ...],
    *,
    calibration: CalibrationProfile,
) -> CostEstimate:
    return CostEstimate(
        strategy=estimate.strategy,
        est_latency_us=estimate.est_latency_us,
        est_recall=estimate.est_recall,
        est_returns_k=estimate.est_returns_k,
        confidence=estimate.confidence,
        notes=estimate.notes + notes + (_calibration_note(calibration),),
    )


def _calibration_note(calibration: CalibrationProfile) -> str:
    if calibration == DEFAULT_CALIBRATION:
        return "uses default calibration constants"
    return (
        "uses calibration profile "
        f"dataset={calibration.dataset_id}, "
        f"hardware={calibration.hardware_id}, "
        f"index_method={calibration.index_method}"
    )


def _verdict(
    *,
    best_plan: StrategyPlan,
    best: CostEstimate,
    vector_index: IndexMeta | None,
    has_local_probe: bool,
) -> str:
    probe_clause = "local probe" if has_local_probe else "global selectivity fallback"
    if best.strategy is Strategy.PARTIAL:
        if best_plan.requires_new_object is not None:
            return "Create the suggested partial HNSW index for this recurring literal filter."
        return "Use the existing partial vector index; its predicate matches this filter."
    if best.strategy is Strategy.PARTITION:
        return "Use partition pruning with per-partition vector indexes for this filter."
    if vector_index is None:
        return "Use exact filtered scan; no HNSW or IVFFlat index exists on the vector column."
    if best.strategy is Strategy.EXACT:
        return f"Use exact filtered scan under current estimates from {probe_clause}."
    if best.strategy is Strategy.POSTFILTER:
        return f"Use post-filter ANN; expected survivors satisfy k under {probe_clause}."
    if best.strategy is Strategy.ITERATIVE_RELAXED:
        return "Prefer relaxed iterative ANN scan; plain post-filter may under-return."
    if best.strategy is Strategy.ITERATIVE_STRICT:
        return "Prefer strict iterative ANN scan when exact ordering is required."
    return "Use the top-ranked strategy under current estimates."
