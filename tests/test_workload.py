from __future__ import annotations

import json
from typing import Any

import pytest
from typer.testing import CliRunner

import vecadvisor.cli as cli_module
from vecadvisor.cli import app
from vecadvisor.workload import (
    WorkloadCollectionReport,
    shape_from_pg_stat_statement,
    workload_report_to_json,
    workload_shape_to_json,
)


def test_shape_from_pg_stat_statement_redacts_literals_and_vectors() -> None:
    shape = shape_from_pg_stat_statement(
        query=(
            "SELECT id FROM public.documents d "
            "WHERE tenant_id = $1 AND created_at >= $2 "
            "ORDER BY d.embedding <-> $3 LIMIT $4"
        ),
        queryid="12345",
        calls=17,
        rows=153,
        total_exec_time_ms=42.5,
        mean_exec_time_ms=2.5,
    )

    assert shape is not None
    assert shape.queryid == "12345"
    assert shape.table == "public.documents"
    assert shape.vector_column == "embedding"
    assert shape.distance_operator == "<->"
    assert shape.limit.kind == "parameter"
    assert shape.limit.value is None
    assert shape.recommend_filter_template == "tenant_id = <value> AND created_at >= <value>"
    predicate_shapes = [
        (item.column, item.kind, item.operator, item.value_kind)
        for item in shape.predicates
    ]
    assert predicate_shapes == [
        ("tenant_id", "eq", "=", "parameter"),
        ("created_at", "range_gt", ">=", "parameter"),
    ]
    assert len(shape.fingerprint) == 24

    payload = workload_shape_to_json(shape)
    serialized = json.dumps(payload, sort_keys=True)
    assert "$1" not in serialized
    assert "$2" not in serialized
    assert "$3" not in serialized
    assert "42.5" in serialized
    assert "SELECT id" not in serialized
    assert "recommend_template" in payload
    assert payload["recommend_template"] == {
        "table": "public.documents",
        "vector": "embedding",
        "query": "tenant_id = <value> AND created_at >= <value>",
        "notes": [
            "replace <value> placeholders with representative literal values",
            "supply representative vectors with --q-vectors or --q-vector-sql",
        ],
    }


def test_workload_report_to_json_documents_privacy_contract() -> None:
    shape = shape_from_pg_stat_statement(
        query=(
            "SELECT id FROM documents "
            "WHERE tenant_id = 42 "
            "ORDER BY embedding <-> '[1,2,3]' LIMIT 10"
        ),
        queryid=None,
        calls=3,
        rows=12,
        total_exec_time_ms=9.0,
        mean_exec_time_ms=3.0,
    )
    assert shape is not None

    report = WorkloadCollectionReport(
        source_view="public.pg_stat_statements",
        scanned_statements=1,
        vector_query_shapes=(shape,),
        rejected_statements=0,
        min_calls=1,
        limit=50,
    )

    payload = workload_report_to_json(report)
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["privacy"]["raw_query_text"] is False
    assert payload["privacy"]["sql_literals"] == "redacted"
    assert payload["privacy"]["query_vectors"] == "not collected"
    assert "tenant_id = 42" not in serialized
    assert "[1,2,3]" not in serialized
    assert "tenant_id = <value>" in serialized


def test_shape_from_pg_stat_statement_supports_in_and_between() -> None:
    shape = shape_from_pg_stat_statement(
        query=(
            "SELECT id FROM public.documents "
            "WHERE tenant_id IN ($1, $2) AND created_at BETWEEN $3 AND $4 "
            "ORDER BY embedding <=> $5 LIMIT 10"
        ),
        queryid="999",
        calls=11,
        rows=80,
        total_exec_time_ms=33.0,
        mean_exec_time_ms=3.0,
    )

    assert shape is not None
    assert shape.distance_operator == "<=>"
    assert shape.limit.kind == "constant"
    assert shape.limit.value == 10
    predicate_shapes = [
        (item.column, item.kind, item.operator, item.value_kind)
        for item in shape.predicates
    ]
    assert predicate_shapes == [
        ("tenant_id", "in", "IN", "parameter_list"),
        ("created_at", "between", "BETWEEN", "parameter_list"),
    ]
    assert shape.recommend_filter_template == (
        "tenant_id IN (<value>, ...) AND created_at BETWEEN <value> AND <value>"
    )


def test_shape_from_pg_stat_statement_records_unsupported_predicates() -> None:
    shape = shape_from_pg_stat_statement(
        query=(
            "SELECT id FROM documents "
            "WHERE tenant_id = $1 OR region = $2 "
            "ORDER BY embedding <-> $3 LIMIT 10"
        ),
        queryid="777",
        calls=2,
        rows=5,
        total_exec_time_ms=4.0,
        mean_exec_time_ms=2.0,
    )

    assert shape is not None
    assert shape.predicates == ()
    assert shape.recommend_filter_template is None
    assert shape.unsupported_predicates == ("OR predicates are not summarized",)
    assert "no supported filter predicates found" in shape.notes


def test_shape_from_pg_stat_statement_ignores_non_vector_order_by() -> None:
    shape = shape_from_pg_stat_statement(
        query="SELECT id FROM documents WHERE tenant_id = $1 ORDER BY created_at DESC LIMIT 10",
        queryid="333",
        calls=1,
        rows=10,
        total_exec_time_ms=1.0,
        mean_exec_time_ms=1.0,
    )

    assert shape is None


def test_shape_from_pg_stat_statement_raises_for_unparseable_sql() -> None:
    with pytest.raises(ValueError, match="could not parse statement"):
        shape_from_pg_stat_statement(
            query="SELECT FROM WHERE",
            queryid=None,
            calls=1,
            rows=0,
            total_exec_time_ms=0.0,
            mean_exec_time_ms=0.0,
        )


def test_workload_cli_outputs_redacted_collection_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = shape_from_pg_stat_statement(
        query=(
            "SELECT id FROM public.documents "
            "WHERE tenant_id = $1 "
            "ORDER BY embedding <-> $2 LIMIT 10"
        ),
        queryid="abc",
        calls=5,
        rows=50,
        total_exec_time_ms=20.0,
        mean_exec_time_ms=4.0,
    )
    assert shape is not None

    class FakeConnection:
        def __enter__(self) -> FakeConnection:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_connect(dsn: str) -> FakeConnection:
        assert dsn == "postgresql://user:secret@localhost/db"
        return FakeConnection()

    def fake_collect_workload_shapes(
        conn: Any,
        *,
        limit: int,
        min_calls: int,
        statement_timeout_ms: int,
    ) -> WorkloadCollectionReport:
        assert isinstance(conn, FakeConnection)
        assert limit == 3
        assert min_calls == 2
        assert statement_timeout_ms == 99
        return WorkloadCollectionReport(
            source_view="public.pg_stat_statements",
            scanned_statements=1,
            vector_query_shapes=(shape,),
            rejected_statements=0,
            min_calls=min_calls,
            limit=limit,
        )

    monkeypatch.setattr(cli_module, "connect", fake_connect)
    monkeypatch.setattr(cli_module, "collect_workload_shapes", fake_collect_workload_shapes)

    result = CliRunner().invoke(
        app,
        [
            "workload",
            "--dsn",
            "postgresql://user:secret@localhost/db",
            "--limit",
            "3",
            "--min-calls",
            "2",
            "--statement-timeout-ms",
            "99",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dsn"] == "postgresql://***:***@localhost/db"
    assert payload["source"]["scanned_statements"] == 1
    assert payload["vector_query_shapes"][0]["recommend_template"]["query"] == (
        "tenant_id = <value>"
    )
    assert "$1" not in result.output
    assert "secret" not in result.output
