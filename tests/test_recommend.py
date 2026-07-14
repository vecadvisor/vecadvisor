from __future__ import annotations

import pytest

from vecadvisor.models import (
    CalibrationProfile,
    ColumnStats,
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
from vecadvisor.recommend import build_recommendation
from vecadvisor.selectivity import rho_from_selectivities


def test_recommendation_uses_local_selectivity_for_ann_costs() -> None:
    s_global = 0.25
    s_local = 0.01
    local = LocalSelectivity(
        s_global=s_global,
        s_local_p10=s_local,
        s_local_median=s_local,
        rho=rho_from_selectivities(s_global, s_local),
        confidence=0.9,
        sample_size=100,
        passing_rows=1,
    )

    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=s_global,
        local_selectivity=local,
        ef_search=40,
        recall_target=0.9,
    )

    assert recommendation.s_local == pytest.approx(s_local)
    assert recommendation.rho == pytest.approx(local.rho)
    assert recommendation.ranked[0][0].strategy is Strategy.ITERATIVE_RELAXED

    postfilter = _estimate_by_strategy(recommendation, Strategy.POSTFILTER)
    postfilter_plan = _candidate_by_strategy(recommendation, Strategy.POSTFILTER)[0]
    assert postfilter_plan.ef_search == 1000
    assert postfilter.est_returns_k
    assert postfilter.est_recall == pytest.approx(0.98)


def test_recommendation_costs_ann_with_p10_local_selectivity() -> None:
    local = LocalSelectivity(
        s_global=0.25,
        s_local_p10=0.01,
        s_local_median=0.25,
        rho=rho_from_selectivities(0.25, 0.01),
        confidence=0.8,
        sample_size=1_600,
        passing_rows=400,
    )

    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=0.25,
        local_selectivity=local,
        ef_search=40,
        recall_target=0.9,
    )

    assert recommendation.s_local == pytest.approx(0.01)
    postfilter = _estimate_by_strategy(recommendation, Strategy.POSTFILTER)
    postfilter_plan = _candidate_by_strategy(recommendation, Strategy.POSTFILTER)[0]
    assert postfilter_plan.ef_search == 1000
    assert postfilter.est_returns_k
    assert any("costing uses p10" in note for note in postfilter.notes)
    assert recommendation.ranked[0][0].strategy is Strategy.ITERATIVE_RELAXED


def test_recommendation_falls_back_to_exact_without_vector_index() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=False),
        query=_query(),
        s_global=0.2,
    )

    assert [plan.strategy for plan, _ in recommendation.ranked] == [Strategy.EXACT]
    assert "no HNSW or IVFFlat index" in recommendation.verdict


def test_recommendation_marks_global_selectivity_fallback_as_low_confidence() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=0.5,
    )

    assert recommendation.s_local == pytest.approx(0.5)
    assert recommendation.rho == pytest.approx(0.0)

    postfilter = _estimate_by_strategy(recommendation, Strategy.POSTFILTER)
    assert postfilter.confidence == pytest.approx(0.35)
    assert any("local selectivity not measured" in note for note in postfilter.notes)


def test_recommendation_marks_zero_pass_local_probe_as_low_confidence() -> None:
    local = LocalSelectivity(
        s_global=0.25,
        s_local_p10=0.0,
        s_local_median=0.0,
        rho=-1.0,
        confidence=0.0,
        sample_size=100,
        passing_rows=0,
        resolution_floor=0.01,
        notes=("no filter-passing rows in local sample; s_local is below probe resolution floor",),
    )

    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=0.25,
        local_selectivity=local,
    )

    assert recommendation.ranked[0][0].strategy is Strategy.EXACT
    postfilter = _estimate_by_strategy(recommendation, Strategy.POSTFILTER)
    assert postfilter.confidence == 0.0
    assert any("below probe resolution floor" in note for note in postfilter.notes)


def test_recommendation_omits_iterative_when_pgvector_does_not_support_it() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=0.25,
        supports_iterative_scan=False,
    )

    strategies = {plan.strategy for plan, _ in recommendation.ranked}
    assert Strategy.ITERATIVE_RELAXED not in strategies
    assert Strategy.ITERATIVE_STRICT not in strategies
    postfilter = _estimate_by_strategy(recommendation, Strategy.POSTFILTER)
    assert any("iterative HNSW scan is unavailable" in note for note in postfilter.notes)


def test_recommendation_raises_ef_search_to_meet_recall_target() -> None:
    local = LocalSelectivity(
        s_global=0.5,
        s_local_p10=0.5,
        s_local_median=0.5,
        rho=0.0,
        confidence=0.9,
        sample_size=200,
        passing_rows=100,
    )

    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=0.5,
        local_selectivity=local,
        calibration=_calibration(),
        ef_search=40,
        recall_target=0.95,
    )

    plan, estimate = _candidate_by_strategy(recommendation, Strategy.POSTFILTER)
    assert plan.ef_search == 80
    assert "SET LOCAL hnsw.ef_search = 80" in plan.sql_hint
    assert estimate.est_recall == pytest.approx(0.95)
    assert any("raised from 40 to 80" in note for note in estimate.notes)


def test_recommendation_raises_postfilter_ef_for_low_local_selectivity() -> None:
    local = LocalSelectivity(
        s_global=0.25,
        s_local_p10=0.02,
        s_local_median=0.02,
        rho=rho_from_selectivities(0.25, 0.02),
        confidence=0.9,
        sample_size=500,
        passing_rows=10,
    )

    recommendation = build_recommendation(
        table=_table(has_vector_index=True),
        query=_query(),
        s_global=0.25,
        local_selectivity=local,
        calibration=_calibration(),
        ef_search=40,
        recall_target=0.9,
    )

    plan, estimate = _candidate_by_strategy(recommendation, Strategy.POSTFILTER)
    assert plan.ef_search == 500
    assert estimate.est_returns_k is True
    assert estimate.est_recall == pytest.approx(0.98)
    assert any("survivor ef >= 500" in note for note in estimate.notes)


def test_recommendation_suggests_partial_index_for_safe_low_cardinality_filter() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True, n_distinct=100),
        query=_query(),
        s_global=0.01,
    )

    plan, estimate = _candidate_by_strategy(recommendation, Strategy.PARTIAL)
    assert plan.requires_new_object is not None
    assert plan.requires_new_object.startswith("CREATE INDEX CONCURRENTLY IF NOT EXISTS")
    assert 'WHERE "tenant_id" = 1;' in plan.requires_new_object
    assert estimate.confidence == pytest.approx(0.45)
    assert recommendation.ranked[0][0].strategy is Strategy.PARTIAL
    assert "Create the suggested partial HNSW index" in recommendation.verdict


def test_recommendation_raises_partial_index_ef_to_meet_recall_target() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True, n_distinct=100),
        query=_query(),
        s_global=0.01,
        calibration=_calibration(),
        ef_search=40,
        recall_target=0.95,
    )

    plan, estimate = _candidate_by_strategy(recommendation, Strategy.PARTIAL)
    assert plan.ef_search == 80
    assert "SET LOCAL hnsw.ef_search = 80" in plan.sql_hint
    assert estimate.est_recall == pytest.approx(0.95)


def test_recommendation_uses_existing_matching_partial_index() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True, has_partial_index=True, n_distinct=100),
        query=_query(),
        s_global=0.01,
    )

    plan, estimate = _candidate_by_strategy(recommendation, Strategy.PARTIAL)
    assert plan.uses_index == "docs_embedding_tenant_1_hnsw_idx"
    assert plan.requires_new_object is None
    assert estimate.confidence == pytest.approx(0.35)
    assert any("existing partial index exactly matches" in note for note in estimate.notes)


def test_recommendation_does_not_suggest_partial_index_for_high_cardinality_filter() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True, n_distinct=100_000),
        query=_query(),
        s_global=0.001,
    )

    assert Strategy.PARTIAL not in {plan.strategy for plan, _ in recommendation.ranked}


def test_recommendation_adds_partition_candidate_when_filter_matches_partition_key() -> None:
    recommendation = build_recommendation(
        table=_table(has_vector_index=True, partitioned_by=("LIST (tenant_id)",)),
        query=_query(),
        s_global=0.25,
    )

    plan, estimate = _candidate_by_strategy(recommendation, Strategy.PARTITION)
    assert plan.uses_index == "docs_embedding_hnsw_idx"
    assert "partition pruning" in plan.sql_hint
    assert estimate.confidence == pytest.approx(0.5)
    assert any("partition key overlaps filter columns" in note for note in estimate.notes)


def _table(
    *,
    has_vector_index: bool,
    has_partial_index: bool = False,
    n_distinct: float = 4,
    partitioned_by: tuple[str, ...] | None = None,
) -> TableStats:
    indexes: list[IndexMeta] = []
    if has_vector_index:
        indexes.append(
            IndexMeta(
                name="docs_embedding_hnsw_idx",
                method="hnsw",
                columns=("embedding",),
                opclass="vector_l2_ops",
            )
        )
    if has_partial_index:
        indexes.append(
            IndexMeta(
                name="docs_embedding_tenant_1_hnsw_idx",
                method="hnsw",
                columns=("embedding",),
                opclass="vector_l2_ops",
                is_partial=True,
                predicate="(tenant_id = 1)",
            )
        )
    return TableStats(
        relname="public.docs",
        n_rows=1_000_000,
        n_pages=10_000,
        vector_dim=3,
        columns=(
            ColumnStats(
                name="tenant_id",
                n_distinct=n_distinct,
                null_frac=0.0,
                type_name="integer",
                has_stats=True,
            ),
        ),
        indexes=tuple(indexes),
        partitioned_by=partitioned_by,
    )


def _query() -> QuerySpec:
    return QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="<->",
        predicates=(Predicate("tenant_id", PredicateKind.EQ, (1,)),),
        limit=10,
    )


def _calibration() -> CalibrationProfile:
    return CalibrationProfile(
        dataset_id="unit",
        hardware_id="unit",
        index_method="hnsw",
        c_d=0.01,
        c_scan=0.004,
        c_h=2.0,
        delta_strict=0.2,
        recall_curve=((40, 0.9), (80, 0.95), (160, 0.98)),
    )


def _estimate_by_strategy(
    recommendation: Recommendation,
    strategy: Strategy,
) -> CostEstimate:
    return _candidate_by_strategy(recommendation, strategy)[1]


def _candidate_by_strategy(
    recommendation: Recommendation,
    strategy: Strategy,
) -> tuple[StrategyPlan, CostEstimate]:
    for plan, estimate in recommendation.ranked:
        if plan.strategy is strategy:
            return plan, estimate
    raise AssertionError(f"strategy not found: {strategy}")
