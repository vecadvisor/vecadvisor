from __future__ import annotations

import pytest

from vecadvisor.local_probe import (
    LocalProbeError,
    QueryVectorError,
    _ivfflat_probe_target,
    _plan_payload_uses_index,
    _probe_confidence,
    _probe_notes,
    _set_int_guc_at_least,
    _table_sample_percent,
    aggregate_local_selectivity,
    build_local_probe_sql,
    load_query_vectors,
    parse_query_vector_text,
    vector_literal,
)
from vecadvisor.models import (
    IndexMeta,
    LocalSelectivity,
    Predicate,
    PredicateKind,
    QuerySpec,
    TableStats,
)


def test_parse_query_vector_text_accepts_json_and_plain_text() -> None:
    assert parse_query_vector_text("[1, 2.5, -3]", expected_dim=3) == (1.0, 2.5, -3.0)
    assert parse_query_vector_text("1, 2.5 -3", expected_dim=3) == (1.0, 2.5, -3.0)


def test_load_query_vectors_accepts_json_and_line_samples(tmp_path) -> None:
    json_file = tmp_path / "queries.json"
    json_file.write_text('{"vectors": [[1, 2, 3], [4, 5, 6]]}', encoding="utf-8")
    assert load_query_vectors(json_file, expected_dim=3) == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )

    text_file = tmp_path / "queries.txt"
    text_file.write_text("# comment\n1 2 3\n[4, 5, 6]\n", encoding="utf-8")
    assert load_query_vectors(text_file, expected_dim=3, max_vectors=1) == (
        (1.0, 2.0, 3.0),
    )


def test_parse_query_vector_text_rejects_bad_dimensions_and_non_finite_values() -> None:
    with pytest.raises(QueryVectorError, match="dimension mismatch"):
        parse_query_vector_text("[1, 2]", expected_dim=3)

    with pytest.raises(QueryVectorError, match="finite"):
        parse_query_vector_text("[1, 2, Infinity]")


def test_vector_literal_formats_pgvector_text_literal() -> None:
    assert vector_literal((1.0, 2.5, -3.0)) == "[1,2.5,-3]"


def test_build_local_probe_sql_projects_only_filter_columns() -> None:
    query = QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="<->",
        predicates=(
            Predicate("tenant_id", PredicateKind.EQ, (1,)),
            Predicate("region", PredicateKind.IN, ("us", "eu")),
            Predicate("tenant_id", PredicateKind.RANGE_GT, (0,)),
        ),
        limit=10,
    )

    sql = build_local_probe_sql(query=query, filter_sql="tenant_id = 1 AND region IN ('us','eu')")

    assert 'SELECT "tenant_id", "region" FROM "public"."docs"' in sql
    assert '"embedding" <-> %s::vector' in sql
    assert "LIMIT %s" in sql
    assert "count(*) FILTER (WHERE tenant_id = 1 AND region IN ('us','eu'))" in sql


def test_build_local_probe_sql_rejects_unsupported_distance_operator() -> None:
    query = QuerySpec(
        relname="public.docs",
        vector_column="embedding",
        distance_op="; DROP TABLE docs; --",
        predicates=(Predicate("tenant_id", PredicateKind.EQ, (1,)),),
        limit=10,
    )

    with pytest.raises(LocalProbeError, match="unsupported vector distance operator"):
        build_local_probe_sql(query=query, filter_sql="tenant_id = 1")


def test_probe_confidence_depends_on_passing_rows() -> None:
    assert _probe_confidence(sample_size=100, passing_rows=0, probe_rows=100) == 0.0
    assert _probe_confidence(sample_size=100, passing_rows=1, probe_rows=100) == 0.2
    assert _probe_confidence(sample_size=100, passing_rows=5, probe_rows=100) == 1.0
    assert _probe_confidence(sample_size=50, passing_rows=5, probe_rows=100) == 0.5


def test_probe_notes_flag_resolution_floor_and_tiny_samples() -> None:
    zero_notes = _probe_notes(
        sample_size=100,
        passing_rows=0,
        probe_rows=100,
        resolution_floor=0.01,
    )
    assert any("below probe resolution floor" in note for note in zero_notes)

    sparse_notes = _probe_notes(
        sample_size=100,
        passing_rows=2,
        probe_rows=100,
        resolution_floor=0.01,
    )
    assert any("confidence is reduced" in note for note in sparse_notes)


def test_table_sample_percent_bounds_large_tables_and_full_samples_small_tables() -> None:
    assert _table_sample_percent(n_rows=100, max_vectors=64) == pytest.approx(100.0)
    assert _table_sample_percent(n_rows=10_000_000, max_vectors=64) == pytest.approx(
        0.0128
    )
    assert _table_sample_percent(n_rows=1_000_000_000, max_vectors=1) == pytest.approx(
        0.01
    )


def test_ivfflat_probe_target_scales_with_lists_and_probe_rows() -> None:
    table = TableStats(
        relname="public.docs",
        n_rows=10_000,
        n_pages=100,
        columns=(),
        indexes=(
            IndexMeta(
                name="docs_embedding_ivf_idx",
                method="ivfflat",
                columns=("embedding",),
                lists=100,
            ),
        ),
    )

    assert _ivfflat_probe_target(
        table=table,
        vector_column="embedding",
        probe_rows=200,
    ) == 2

    small_table = TableStats(
        relname="public.docs",
        n_rows=100,
        n_pages=10,
        columns=(),
        indexes=table.indexes,
    )
    assert _ivfflat_probe_target(
        table=small_table,
        vector_column="embedding",
        probe_rows=200,
    ) == 100


def test_set_int_guc_keeps_existing_higher_value() -> None:
    conn = _FakeConn({"hnsw.ef_search": "320"})

    _set_int_guc_at_least(conn, "hnsw.ef_search", 200)

    assert conn.set_calls == [("hnsw.ef_search", "320")]


def test_set_int_guc_raises_low_default_to_probe_rows() -> None:
    conn = _FakeConn({"hnsw.ef_search": "40"})

    _set_int_guc_at_least(conn, "hnsw.ef_search", 200)

    assert conn.set_calls == [("hnsw.ef_search", "200")]


def test_probe_plan_detection_requires_ann_index_name() -> None:
    plan = [
        {
            "Plan": {
                "Node Type": "Limit",
                "Plans": [
                    {
                        "Node Type": "Index Scan",
                        "Index Name": "docs_embedding_hnsw_idx",
                    }
                ],
            }
        }
    ]

    assert _plan_payload_uses_index(plan, {"docs_embedding_hnsw_idx"}) is True
    assert _plan_payload_uses_index(plan, {"docs_other_idx"}) is False


def test_aggregate_local_selectivity_uses_p10_and_reduces_small_sample_confidence() -> None:
    probes = (
        _local(s_local=0.01, confidence=1.0, passing_rows=1),
        _local(s_local=0.20, confidence=1.0, passing_rows=20),
        _local(s_local=0.30, confidence=1.0, passing_rows=30),
    )

    aggregate = aggregate_local_selectivity(
        probes,
        s_global=0.25,
        high_confidence_vectors=6,
    )

    assert aggregate.s_local_p10 == pytest.approx(0.048)
    assert aggregate.s_local_median == pytest.approx(0.20)
    assert aggregate.rho < 0
    assert aggregate.confidence == pytest.approx(0.5)
    assert aggregate.sample_size == 300
    assert aggregate.passing_rows == 51
    assert any("costing uses p10" in note for note in aggregate.notes)
    assert any("fewer than 6" in note for note in aggregate.notes)


def _local(
    *,
    s_local: float,
    confidence: float,
    passing_rows: int,
) -> LocalSelectivity:
    return LocalSelectivity(
        s_global=0.25,
        s_local_p10=s_local,
        s_local_median=s_local,
        rho=0.0,
        confidence=confidence,
        sample_size=100,
        passing_rows=passing_rows,
        resolution_floor=0.01,
    )


class _FakeCursor:
    def __init__(self, row: object) -> None:
        self._row = row

    def fetchone(self) -> object:
        return self._row


class _FakeConn:
    def __init__(self, settings: dict[str, str]) -> None:
        self.settings = settings
        self.set_calls: list[tuple[str, str]] = []

    def execute(self, sql: str, params: tuple[object, ...]) -> _FakeCursor:
        if "current_setting" in sql:
            return _FakeCursor((self.settings.get(str(params[0])),))
        if "set_config" in sql:
            self.set_calls.append((str(params[0]), str(params[1])))
            return _FakeCursor((None,))
        raise AssertionError(f"unexpected SQL: {sql}")
