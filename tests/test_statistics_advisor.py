from __future__ import annotations

from vecadvisor.models import (
    ExtendedStatsMeta,
    Predicate,
    PredicateKind,
    QuerySpec,
    SelectivityCrossCheck,
    TableStats,
)
from vecadvisor.statistics_advisor import suggest_statistics


def test_suggest_statistics_for_uncovered_multi_column_filter() -> None:
    suggestions = suggest_statistics(
        table=_table(),
        query=_query(),
        cross_check=_cross_check(ratio=1.0, absolute_delta=0.0),
    )

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.columns == ("tenant_id", "region")
    assert suggestion.kinds == ("dependencies", "mcv")
    assert suggestion.confidence == 0.35
    assert "independence assumptions" in suggestion.reason
    assert suggestion.ddl.startswith('CREATE STATISTICS IF NOT EXISTS "public".')
    assert 'ON "tenant_id", "region" FROM "public"."docs";' in suggestion.ddl


def test_suggest_statistics_raises_confidence_when_estimates_diverge() -> None:
    suggestions = suggest_statistics(
        table=_table(),
        query=_query(),
        cross_check=_cross_check(ratio=3.0, absolute_delta=0.2),
    )

    assert len(suggestions) == 1
    assert suggestions[0].confidence == 0.75
    assert "diverges" in suggestions[0].reason


def test_suggest_statistics_skips_when_covering_stats_exist() -> None:
    table = _table(
        extended_stats=(
            ExtendedStatsMeta(
                name="docs_tenant_region_stats",
                schema="public",
                columns=("tenant_id", "region"),
                kinds=("dependencies", "mcv"),
            ),
        )
    )

    assert suggest_statistics(table=table, query=_query(), cross_check=_cross_check()) == ()


def test_suggest_statistics_skips_single_column_filters() -> None:
    query = QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="<->",
        predicates=(Predicate("tenant_id", PredicateKind.EQ, (1,)),),
        limit=10,
    )

    assert suggest_statistics(table=_table(), query=query, cross_check=_cross_check()) == ()


def _table(
    *,
    extended_stats: tuple[ExtendedStatsMeta, ...] = (),
) -> TableStats:
    return TableStats(
        relname="public.docs",
        n_rows=1000,
        n_pages=100,
        columns=(),
        extended_stats=extended_stats,
        vector_dim=3,
    )


def _query() -> QuerySpec:
    return QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="<->",
        predicates=(
            Predicate("tenant_id", PredicateKind.EQ, (1,)),
            Predicate("region", PredicateKind.EQ, ("us",)),
        ),
        limit=10,
    )


def _cross_check(
    *,
    ratio: float | None = 1.0,
    absolute_delta: float | None = 0.0,
) -> SelectivityCrossCheck:
    return SelectivityCrossCheck(
        relation_name="public.docs",
        advisor_selectivity=0.1,
        postgres_selectivity=0.1 if ratio is not None else None,
        advisor_rows=100,
        postgres_plan_rows=100,
        absolute_delta=absolute_delta,
        ratio=ratio,
        plan_node_type="Seq Scan",
    )
