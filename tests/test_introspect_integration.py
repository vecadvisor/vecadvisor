from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from vecadvisor.introspect import IntrospectionError, connect, introspect_table

TEST_DSN = os.getenv(
    "VECADVISOR_TEST_DSN",
    "postgresql://postgres:postgres@localhost:5432/vecadvisor",
)


@pytest.fixture(scope="module")
def pg_table() -> Iterator[str]:
    try:
        conn = connect(TEST_DSN)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is not available: {exc}")

    conn.autocommit = True
    table_name = "vecadvisor_introspect_fixture"
    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
                id bigserial PRIMARY KEY,
                tenant_id int NOT NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                body text,
                embedding vector(3)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO {table_name} (tenant_id, created_at, body, embedding)
            SELECT (g % 4),
                   now() - (g || ' hours')::interval,
                   'doc ' || g,
                   ARRAY[
                       (g % 5)::float4,
                       (g % 7)::float4,
                       (g % 11)::float4
                   ]::vector
            FROM generate_series(1, 64) AS g
            """
        )
        conn.execute(f"CREATE INDEX {table_name}_tenant_idx ON {table_name} (tenant_id)")
        conn.execute(
            f"""
            CREATE INDEX {table_name}_embedding_hnsw_idx
            ON {table_name}
            USING hnsw (embedding vector_l2_ops)
            WITH (m = 8, ef_construction = 32)
            """
        )
        conn.execute(
            f"""
            CREATE INDEX {table_name}_embedding_partial_idx
            ON {table_name}
            USING hnsw (embedding vector_l2_ops)
            WHERE tenant_id = 1
            """
        )
        conn.execute(
            f"""
            CREATE STATISTICS {table_name}_tenant_created_stats
            (dependencies, mcv)
            ON tenant_id, created_at
            FROM {table_name}
            """
        )
        conn.execute(f"ANALYZE {table_name}")
        yield f"public.{table_name}"
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")


def test_introspect_table_reads_columns_stats_and_vector_dim(pg_table: str) -> None:
    with connect(TEST_DSN) as conn:
        stats = introspect_table(conn, pg_table, vector_column="embedding")
        stats_again = introspect_table(conn, pg_table, vector_column="embedding")

    assert stats.relname == pg_table
    assert stats.n_rows == 64
    assert stats.vector_dim == 3
    assert stats.last_analyze is not None
    assert stats.n_mod_since_analyze == 0
    assert len(stats.stats_fingerprint) == 24
    assert len(stats.index_fingerprint) == 24
    assert stats.stats_fingerprint == stats_again.stats_fingerprint
    assert stats.index_fingerprint == stats_again.index_fingerprint

    tenant = stats.column("tenant_id")
    assert tenant.type_name == "integer"
    assert tenant.has_stats
    assert tenant.n_distinct == 4
    assert tenant.mcv == (0, 1, 2, 3)
    assert tenant.mcf == (0.25, 0.25, 0.25, 0.25)


def test_introspect_table_reads_scalar_hnsw_and_partial_indexes(pg_table: str) -> None:
    with connect(TEST_DSN) as conn:
        stats = introspect_table(conn, pg_table, vector_column="embedding")

    by_name = {index.name: index for index in stats.indexes}
    hnsw = by_name["vecadvisor_introspect_fixture_embedding_hnsw_idx"]
    partial = by_name["vecadvisor_introspect_fixture_embedding_partial_idx"]
    tenant = by_name["vecadvisor_introspect_fixture_tenant_idx"]

    assert hnsw.method == "hnsw"
    assert hnsw.columns == ("embedding",)
    assert hnsw.opclass == "vector_l2_ops"
    assert hnsw.m == 8
    assert hnsw.ef_construction == 32
    assert not hnsw.is_partial

    assert partial.method == "hnsw"
    assert partial.is_partial
    assert partial.predicate == "(tenant_id = 1)"

    assert tenant.method == "btree"
    assert tenant.columns == ("tenant_id",)


def test_introspect_table_reads_extended_statistics(pg_table: str) -> None:
    with connect(TEST_DSN) as conn:
        stats = introspect_table(conn, pg_table, vector_column="embedding")

    by_name = {stat.name: stat for stat in stats.extended_stats}
    stat = by_name["vecadvisor_introspect_fixture_tenant_created_stats"]
    assert stat.schema == "public"
    assert stat.columns == ("tenant_id", "created_at")
    assert stat.kinds == ("dependencies", "mcv")


def test_introspect_table_rejects_unknown_vector_column(pg_table: str) -> None:
    with connect(TEST_DSN) as conn:
        with pytest.raises(IntrospectionError):
            introspect_table(conn, pg_table, vector_column="missing_embedding")
