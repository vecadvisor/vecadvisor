from __future__ import annotations

from vecadvisor.local_cache import (
    build_local_selectivity_cache_key,
    load_local_selectivity_cache,
    local_selectivity_cache_path,
    store_local_selectivity_cache,
)
from vecadvisor.models import (
    LocalSelectivity,
    Predicate,
    PredicateKind,
    QuerySpec,
    TableStats,
)


def test_local_selectivity_cache_key_is_stable_and_catalog_sensitive() -> None:
    query = _query()
    key = build_local_selectivity_cache_key(
        table=_table(stats_fingerprint="stats-a"),
        query=query,
        filter_sql="tenant_id = 1",
        vector_source="file:queries.json",
        query_vectors=((1.0, 2.0, 3.0),),
        probe_rows=16,
        s_global=0.25,
    )
    same_key = build_local_selectivity_cache_key(
        table=_table(stats_fingerprint="stats-a"),
        query=query,
        filter_sql="tenant_id = 1",
        vector_source="file:queries.json",
        query_vectors=((1.0, 2.0, 3.0),),
        probe_rows=16,
        s_global=0.25,
    )
    changed_stats_key = build_local_selectivity_cache_key(
        table=_table(stats_fingerprint="stats-b"),
        query=query,
        filter_sql="tenant_id = 1",
        vector_source="file:queries.json",
        query_vectors=((1.0, 2.0, 3.0),),
        probe_rows=16,
        s_global=0.25,
    )

    assert len(key) == 24
    assert key == same_key
    assert key != changed_stats_key


def test_local_selectivity_cache_round_trips_and_malformed_entries_miss(tmp_path) -> None:
    local = LocalSelectivity(
        s_global=0.25,
        s_local_p10=0.05,
        s_local_median=0.10,
        rho=-0.5,
        confidence=0.7,
        sample_size=48,
        passing_rows=5,
        resolution_floor=1 / 16,
        notes=("local selectivity aggregated from 3 representative query vectors",),
    )

    path = store_local_selectivity_cache(tmp_path, "abc123", local)

    assert path == local_selectivity_cache_path(tmp_path, "abc123")
    assert load_local_selectivity_cache(tmp_path, "abc123") == local

    path.write_text("{not-json", encoding="utf-8")
    assert load_local_selectivity_cache(tmp_path, "abc123") is None


def _table(*, stats_fingerprint: str) -> TableStats:
    return TableStats(
        relname="public.docs",
        n_rows=1_000,
        n_pages=100,
        columns=(),
        vector_dim=3,
        stats_fingerprint=stats_fingerprint,
        index_fingerprint="index-a",
    )


def _query() -> QuerySpec:
    return QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="<->",
        predicates=(Predicate("tenant_id", PredicateKind.EQ, (1,)),),
        limit=10,
    )
