from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from vecadvisor.introspect import connect, introspect_table
from vecadvisor.models import Predicate, PredicateKind
from vecadvisor.plan import (
    PlanCollectionError,
    classify_selectivity_cross_check,
    compare_selectivity,
    explain_query,
    find_scan_node,
)
from vecadvisor.selectivity import conjunction_selectivity

TEST_DSN = os.getenv(
    "VECADVISOR_TEST_DSN",
    "postgresql://postgres:postgres@localhost:5432/vecadvisor",
)


@pytest.fixture(scope="module")
def pg_plan_table() -> Iterator[str]:
    try:
        conn = connect(TEST_DSN)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is not available: {exc}")

    conn.autocommit = True
    table_name = "vecadvisor_plan_fixture"
    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
                id bigserial PRIMARY KEY,
                tenant_id int NOT NULL,
                score int NOT NULL,
                body text,
                embedding vector(3)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO {table_name} (tenant_id, score, body, embedding)
            SELECT (g % 4),
                   g,
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
        conn.execute(f"ANALYZE {table_name}")
        yield f"public.{table_name}"
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")


def test_explain_query_normalizes_json_plan_tree(pg_plan_table: str) -> None:
    with connect(TEST_DSN) as conn:
        plan = explain_query(
            conn,
            f"SELECT * FROM {pg_plan_table} WHERE tenant_id = %s",
            (1,),
        )

    scan = find_scan_node(plan.root, pg_plan_table)
    assert scan is not None
    assert scan.relation_name == pg_plan_table.rsplit(".", 1)[-1]
    assert scan.plan_rows == 16
    assert scan.node_type in {"Seq Scan", "Index Scan", "Bitmap Heap Scan"}


def test_compare_selectivity_matches_postgres_plan_estimate(pg_plan_table: str) -> None:
    with connect(TEST_DSN) as conn:
        table = introspect_table(conn, pg_plan_table, vector_column="embedding")
        advisor_selectivity = conjunction_selectivity(
            table,
            (Predicate("tenant_id", PredicateKind.EQ, (1,)),),
        )
        plan = explain_query(
            conn,
            f"SELECT * FROM {pg_plan_table} WHERE tenant_id = %s",
            (1,),
        )

    cross_check = compare_selectivity(table, plan, advisor_selectivity)
    assert cross_check.postgres_plan_rows == 16
    assert cross_check.postgres_selectivity == pytest.approx(0.25)
    assert cross_check.advisor_selectivity == pytest.approx(0.25)
    assert cross_check.absolute_delta == pytest.approx(0.0)
    assert cross_check.ratio == pytest.approx(1.0)
    assert cross_check.status == "aligned"
    assert cross_check.severity == "ok"
    assert any("aligned" in note for note in cross_check.notes)


def test_range_selectivity_tracks_postgres_histogram_estimate(pg_plan_table: str) -> None:
    with connect(TEST_DSN) as conn:
        table = introspect_table(conn, pg_plan_table, vector_column="embedding")
        advisor_selectivity = conjunction_selectivity(
            table,
            (Predicate("score", PredicateKind.RANGE_LT, (33,)),),
        )
        plan = explain_query(
            conn,
            f"SELECT * FROM {pg_plan_table} WHERE score < %s",
            (33,),
        )

    cross_check = compare_selectivity(table, plan, advisor_selectivity)
    assert cross_check.postgres_selectivity is not None
    assert cross_check.advisor_selectivity == pytest.approx(
        cross_check.postgres_selectivity,
        abs=0.08,
    )


def test_selectivity_cross_check_classifier_flags_divergence() -> None:
    status, severity, notes = classify_selectivity_cross_check(
        advisor_selectivity=0.01,
        postgres_selectivity=0.1,
        absolute_delta=0.09,
        ratio=10.0,
    )
    assert status == "diverged"
    assert severity == "critical"
    assert any("under-cost exact search" in note for note in notes)

    status, severity, notes = classify_selectivity_cross_check(
        advisor_selectivity=0.1,
        postgres_selectivity=None,
        absolute_delta=None,
        ratio=None,
    )
    assert status == "unavailable"
    assert severity == "unknown"
    assert any("could not run" in note for note in notes)


def test_explain_query_rejects_non_readonly_sql(pg_plan_table: str) -> None:
    with connect(TEST_DSN) as conn:
        with pytest.raises(PlanCollectionError):
            explain_query(conn, f"DELETE FROM {pg_plan_table}")
