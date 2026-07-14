from __future__ import annotations

import pytest

from vecadvisor.models import PredicateKind
from vecadvisor.query_spec import (
    QueryParseError,
    build_filter_select_sql,
    parse_filter,
    query_spec_from_filter,
    quote_qualified_identifier,
)


def test_parse_filter_supports_and_comparison_in_between_and_bool() -> None:
    predicates = parse_filter(
        "tenant_id = 1 AND created_at >= '2026-01-01' "
        "AND region IN ('us', 'eu') AND score BETWEEN 10 AND 20 AND active",
        allowed_columns={"tenant_id", "created_at", "region", "score", "active"},
    )

    assert [(predicate.column, predicate.kind, predicate.values) for predicate in predicates] == [
        ("tenant_id", PredicateKind.EQ, (1,)),
        ("created_at", PredicateKind.RANGE_GT, ("2026-01-01",)),
        ("region", PredicateKind.IN, ("us", "eu")),
        ("score", PredicateKind.RANGE_BETWEEN, (10, 20)),
        ("active", PredicateKind.BOOL, (True,)),
    ]


def test_parse_filter_inverts_literal_first_range_predicates() -> None:
    predicates = parse_filter("10 < score", allowed_columns={"score"})

    assert len(predicates) == 1
    assert predicates[0].column == "score"
    assert predicates[0].kind is PredicateKind.RANGE_GT
    assert predicates[0].values == (10,)


def test_parse_filter_unwraps_literal_type_casts_from_catalog_predicates() -> None:
    predicates = parse_filter("region = 'us'::text", allowed_columns={"region"})

    assert len(predicates) == 1
    assert predicates[0].column == "region"
    assert predicates[0].kind is PredicateKind.EQ
    assert predicates[0].values == ("us",)


@pytest.mark.parametrize(
    "filter_sql",
    [
        "tenant_id = 1 OR tenant_id = 2",
        "lower(region) = 'us'",
        "tenant_id IS NULL",
        "tenant_id = 1; SELECT 1",
    ],
)
def test_parse_filter_rejects_unsupported_or_unsafe_sql(filter_sql: str) -> None:
    with pytest.raises(QueryParseError):
        parse_filter(filter_sql, allowed_columns={"tenant_id", "region"})


def test_parse_filter_rejects_unknown_columns() -> None:
    with pytest.raises(QueryParseError, match="unknown filter column"):
        parse_filter("missing_column = 1", allowed_columns={"tenant_id"})


def test_query_spec_from_filter_builds_model_object() -> None:
    query_spec = query_spec_from_filter(
        relname="public.docs",
        vector_column="embedding",
        filter_sql="tenant_id = 1",
        limit=10,
        allowed_columns={"tenant_id", "embedding"},
    )

    assert query_spec.relname == "public.docs"
    assert query_spec.vector_column == "embedding"
    assert query_spec.distance_op == "<->"
    assert query_spec.limit == 10
    assert query_spec.has_only_literals
    assert query_spec.predicates[0].kind is PredicateKind.EQ


def test_filter_select_sql_quotes_relation_name() -> None:
    assert quote_qualified_identifier('public.weird"name') == '"public"."weird""name"'
    assert (
        build_filter_select_sql("public.docs", "tenant_id = 1")
        == 'SELECT * FROM "public"."docs" WHERE tenant_id = 1'
    )
