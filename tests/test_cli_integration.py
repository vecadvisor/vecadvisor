from __future__ import annotations

import csv
import json
import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from typer.testing import CliRunner

from vecadvisor.calibration import load_calibration_profile
from vecadvisor.cli import app
from vecadvisor.introspect import connect

TEST_DSN = os.getenv(
    "VECADVISOR_TEST_DSN",
    "postgresql://postgres:postgres@localhost:5432/vecadvisor",
)


def test_version_option_outputs_package_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("VecAdvisor ")


@pytest.fixture(scope="module")
def pg_cli_table() -> Iterator[str]:
    try:
        conn = connect(TEST_DSN)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is not available: {exc}")

    conn.autocommit = True
    table_name = "vecadvisor_cli_fixture"
    with conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.execute(
            f"""
            CREATE TABLE {table_name} (
                id bigserial PRIMARY KEY,
                tenant_id int NOT NULL,
                region text NOT NULL,
                embedding vector(3)
            )
            """
        )
        conn.execute(
            f"""
            INSERT INTO {table_name} (tenant_id, region, embedding)
            SELECT (g % 4),
                   CASE WHEN g % 2 = 0 THEN 'us' ELSE 'eu' END,
                   ARRAY[
                       (g % 5)::float4,
                       (g % 7)::float4,
                       (g % 11)::float4
                   ]::vector
            FROM generate_series(1, 64) AS g
            """
        )
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


def test_explain_cli_outputs_selectivity_cross_check(pg_cli_table: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "explain",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["table"] == pg_cli_table
    assert payload["vector"] == "embedding"
    assert payload["vector_dim"] == 3
    assert len(payload["catalog_snapshot"]["stats_fingerprint"]) == 24
    assert len(payload["catalog_snapshot"]["index_fingerprint"]) == 24
    assert payload["catalog_snapshot"]["n_mod_since_analyze"] == 0
    assert payload["predicates"] == [
        {
            "column": "tenant_id",
            "kind": "eq",
            "values": [1],
            "is_literal": True,
        }
    ]
    assert payload["selectivity"]["advisor_selectivity"] == pytest.approx(0.25)
    assert payload["selectivity"]["postgres_selectivity"] == pytest.approx(0.25)
    assert payload["selectivity"]["postgres_plan_rows"] == 16
    assert payload["selectivity"]["status"] == "aligned"
    assert payload["selectivity"]["severity"] == "ok"
    assert any("aligned" in note for note in payload["selectivity"]["notes"])
    assert payload["stats_health"]["status"] == "fresh"
    assert payload["stats_health"]["stale"] is False
    assert payload["stats_health"]["analyze_sql"].startswith("ANALYZE ")
    assert payload["postgres"]["pgvector_installed"] is True
    assert isinstance(payload["postgres"]["supports_hnsw_iterative_scan"], bool)
    assert payload["plan"]["filter_select_sql"] == (
        f'SELECT * FROM "{pg_cli_table.split(".")[0]}"."{pg_cli_table.split(".")[1]}" '
        "WHERE tenant_id = 1"
    )
    assert payload["local_selectivity"] is None
    assert payload["q_vector"] is None
    assert payload["calibration"]["source"] == "default"
    assert payload["calibration"]["profile"]["dataset_id"] == "mvp1-default"
    assert payload["plan"]["observed_vector_query"] is None
    assert payload["recommendation"]["s_global"] == pytest.approx(0.25)
    assert payload["recommendation"]["s_local"] == pytest.approx(0.25)
    if payload["postgres"]["supports_hnsw_iterative_scan"]:
        assert len(payload["recommendation"]["ranked"]) == 4
    else:
        assert len(payload["recommendation"]["ranked"]) == 2
        assert {
            row["strategy"] for row in payload["recommendation"]["ranked"]
        } == {"exact", "postfilter"}
    assert payload["recommendation"]["ranked"][0]["strategy"] in {
        "exact",
        "postfilter",
        "iterative_relaxed",
        "iterative_strict",
    }


def test_explain_cli_outputs_text_diagnostics(pg_cli_table: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "explain",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
            "--limit",
            "10",
            "--format",
            "text",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.startswith(f"EXPLAIN VECTOR  {pg_cli_table}")
    assert "filter: tenant_id = 1" in result.output
    assert "s(global): 0.25" in result.output
    assert "PLAN" in result.output
    assert "VERDICT:" in result.output
    assert "CATALOG SNAPSHOT:" in result.output
    assert "WHY:" in result.output
    assert "Local selectivity was not measured" in result.output


def test_explain_cli_outputs_statistics_suggestion_for_multicolumn_filter(
    pg_cli_table: str,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "explain",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1 AND region = 'eu'",
            "--limit",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    suggestions = payload["statistics_suggestions"]
    assert len(suggestions) == 1
    assert suggestions[0]["columns"] == ["tenant_id", "region"]
    assert suggestions[0]["kinds"] == ["dependencies", "mcv"]
    assert "CREATE STATISTICS IF NOT EXISTS" in suggestions[0]["ddl"]
    assert 'ON "tenant_id", "region"' in suggestions[0]["ddl"]


def test_explain_cli_outputs_local_selectivity_probe(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    vector_file = tmp_path / "query-vector.json"
    vector_file.write_text("[1, 1, 1]", encoding="utf-8")
    calibration_file = tmp_path / "calibration.json"
    calibration_file.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset_id": "cli-fixture",
                "hardware_id": "local-test",
                "index_method": "hnsw",
                "c_d": 0.02,
                "c_scan": 0.01,
                "c_h": 3.0,
                "delta_strict": 0.15,
                "recall_curve": [[40, 0.9], [80, 0.95], [160, 0.98]],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "explain",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
            "--q-vector",
            str(vector_file),
            "--probe-rows",
            "16",
            "--calibration",
            str(calibration_file),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["q_vector"] == {"path": str(vector_file), "dimensions": 3}
    assert payload["calibration"]["source"] == str(calibration_file)
    assert payload["calibration"]["profile"]["dataset_id"] == "cli-fixture"
    assert payload["probe_rows"] == 16
    assert payload["local_selectivity"]["s_global"] == pytest.approx(0.25)
    assert payload["local_selectivity"]["sample_size"] == 16
    assert payload["local_selectivity"]["resolution_floor"] == pytest.approx(1 / 16)
    assert isinstance(payload["local_selectivity"]["notes"], list)
    assert 0 <= payload["local_selectivity"]["passing_rows"] <= 16
    assert 0.0 <= payload["local_selectivity"]["s_local_median"] <= 1.0
    assert -1.0 <= payload["local_selectivity"]["rho"] <= 1.0
    assert payload["recommendation"]["s_local"] == pytest.approx(
        payload["local_selectivity"]["s_local_median"]
    )
    observed = payload["plan"]["observed_vector_query"]
    assert observed["strategy"] in {"exact", "postfilter", "partial"}
    assert observed["full_query_sql"].endswith('ORDER BY "embedding" <-> %s::vector LIMIT %s')
    assert payload["recommendation"]["planner_would_pick"] == observed["strategy"]
    notes = [
        note
        for candidate in payload["recommendation"]["ranked"]
        for note in candidate["estimate"]["notes"]
    ]
    assert any("dataset=cli-fixture" in note for note in notes)


def test_recommend_cli_aggregates_query_vector_file(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    vector_file = tmp_path / "query-vectors.json"
    vector_file.write_text(
        json.dumps({"vectors": [[1, 1, 1], [2, 2, 2], [3, 3, 3]]}),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "recommend",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
            "--q-vectors",
            str(vector_file),
            "--probe-rows",
            "16",
            "--max-query-vectors",
            "3",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query_vectors"] == {
        "source": f"file:{vector_file}",
        "count": 3,
        "max_query_vectors": 3,
    }
    assert payload["local_selectivity"]["s_global"] == pytest.approx(0.25)
    assert payload["local_selectivity"]["sample_size"] == 48
    assert payload["local_selectivity"]["resolution_floor"] == pytest.approx(1 / 16)
    assert payload["recommendation"]["s_local"] == pytest.approx(
        payload["local_selectivity"]["s_local_p10"]
    )
    decision = payload["recommendation"]["decision"]
    assert decision["recommended_strategy"] == payload["recommendation"]["ranked"][0]["strategy"]
    assert decision["estimated_latency_ms"] == pytest.approx(
        payload["recommendation"]["ranked"][0]["estimate"]["est_latency_us"] / 1000.0
    )
    assert decision["selectivity_source"] == "local_probe"
    assert decision["viable"] is True
    assert decision["viable_candidates"] >= 1
    assert decision["confidence_level"] in {"low", "medium", "high"}
    assert isinstance(decision["why"], list)
    assert any(
        "lowest estimated-latency candidate" in reason
        or "recall-safe fallback" in reason
        for reason in decision["why"]
    )
    assert any("p10 local selectivity" in note for note in payload["notes"])
    assert payload["plan"]["observed_vector_query"] is not None


def test_recommend_cli_loads_query_vectors_from_sql(pg_cli_table: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "recommend",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
            "--q-vector-sql",
            f"SELECT embedding::text FROM {pg_cli_table} ORDER BY id LIMIT 5",
            "--probe-rows",
            "8",
            "--max-query-vectors",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query_vectors"] == {
        "source": "sql",
        "count": 2,
        "max_query_vectors": 2,
    }
    assert payload["local_selectivity"]["sample_size"] == 16
    assert payload["recommendation"]["s_local"] == pytest.approx(
        payload["local_selectivity"]["s_local_p10"]
    )


def test_recommend_cli_reuses_local_selectivity_cache(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    vector_file = tmp_path / "query-vectors.json"
    vector_file.write_text(
        json.dumps({"vectors": [[1, 1, 1], [2, 2, 2]]}),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "cache"
    args = [
        "recommend",
        "--dsn",
        TEST_DSN,
        "--table",
        pg_cli_table,
        "--vector",
        "embedding",
        "--query",
        "tenant_id = 1",
        "--q-vectors",
        str(vector_file),
        "--probe-rows",
        "8",
        "--max-query-vectors",
        "2",
        "--local-cache-dir",
        str(cache_dir),
    ]

    runner = CliRunner()
    first = runner.invoke(app, args)
    second = runner.invoke(app, args)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    first_payload = json.loads(first.output)
    second_payload = json.loads(second.output)
    first_cache = first_payload["local_selectivity_cache"]
    second_cache = second_payload["local_selectivity_cache"]

    assert first_cache["enabled"] is True
    assert first_cache["hit"] is False
    assert first_cache["stored"] is True
    assert Path(first_cache["path"]).exists()
    assert second_cache["enabled"] is True
    assert second_cache["hit"] is True
    assert second_cache["stored"] is False
    assert second_cache["key"] == first_cache["key"]
    assert second_payload["local_selectivity"] == first_payload["local_selectivity"]


def test_recommend_cli_without_query_vectors_marks_low_confidence_fallback(
    pg_cli_table: str,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "recommend",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query_vectors"]["source"] == "global_selectivity_fallback"
    assert payload["query_vectors"]["count"] == 0
    assert payload["local_selectivity"] is None
    assert payload["recommendation"]["s_local"] == pytest.approx(0.25)
    decision = payload["recommendation"]["decision"]
    assert decision["recommended_strategy"] == payload["recommendation"]["ranked"][0]["strategy"]
    assert decision["confidence_level"] in {"low", "medium", "high"}
    assert decision["selectivity_source"] == "global_selectivity_fallback"
    assert decision["planner_strategy"] is None
    assert decision["planner_mismatch"] is None
    assert any("low-confidence fallback" in note for note in payload["notes"])


def test_recommend_cli_uses_table_sample_vector_fallback(pg_cli_table: str) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "recommend",
            "--dsn",
            TEST_DSN,
            "--table",
            pg_cli_table,
            "--vector",
            "embedding",
            "--query",
            "tenant_id = 1",
            "--allow-table-sample-vectors",
            "--max-query-vectors",
            "2",
            "--probe-rows",
            "8",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["query_vectors"] == {
        "source": f"table_sample:{pg_cli_table}.embedding",
        "count": 2,
        "max_query_vectors": 2,
    }
    assert payload["local_selectivity"] is not None
    assert payload["local_selectivity"]["sample_size"] == 16
    assert payload["local_selectivity"]["confidence"] <= 0.5
    assert any("table-sampled vectors" in note for note in payload["notes"])
    assert any(
        "table sample" in note for note in payload["local_selectivity"]["notes"]
    )
    assert payload["plan"]["observed_vector_query"] is not None


def test_benchmark_cli_outputs_synthetic_strategy_metrics(tmp_path: Path) -> None:
    out_path = tmp_path / "synthetic-benchmark.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--dataset",
            "synthetic",
            "--strategies",
            "exact,postfilter,iterative",
            "--rows",
            "128",
            "--dim",
            "4",
            "--queries",
            "3",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.25",
            "--correlation",
            "0.5",
            "--limit",
            "5",
            "--ef-search",
            "8",
            "--max-scan-tuples",
            "64",
            "--block-rows",
            "16",
            "--seed",
            "11",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"]["id"] == "synthetic"
    assert payload["dataset"]["rows"] == 128
    assert payload["dataset"]["queries"] == 3
    assert payload["dataset"]["query_policy"] == "uniform"
    assert payload["ground_truth"]["k"] == 5
    assert payload["ground_truth"]["block_rows"] == 16
    assert payload["ground_truth"]["blocks_scanned"] == 8
    assert len(payload["ground_truth"]["first_query_indices"]) <= 5
    assert [row["strategy"] for row in payload["strategies"]] == [
        "exact",
        "postfilter",
        "iterative",
    ]
    assert payload["strategies"][0]["recall_at_k"] == pytest.approx(1.0)
    assert payload["strategies"][1]["params"]["ef_search"] == 8
    assert payload["strategies"][2]["params"]["max_scan_tuples"] == 64
    assert payload["output"] == {"path": str(out_path), "format": "csv"}
    assert any("blocked exact filtered search" in note for note in payload["notes"])

    with out_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["strategy"] for row in rows] == ["exact", "postfilter", "iterative"]


def test_benchmark_cli_outputs_file_dataset_metrics(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    vectors_path = tmp_path / "vectors.npy"
    filter_path = tmp_path / "filter.npy"
    queries_path = tmp_path / "queries.npy"
    out_path = tmp_path / "file-benchmark.json"
    np.save(vectors_path, np.arange(96, dtype="float32").reshape(32, 3))
    np.save(filter_path, np.asarray([index % 4 == 0 for index in range(32)]))
    np.save(queries_path, np.asarray([[0.0, 1.0, 2.0], [9.0, 10.0, 11.0]], dtype="float32"))

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark",
            "--dataset",
            "file",
            "--vectors",
            str(vectors_path),
            "--filter-mask",
            str(filter_path),
            "--query-vectors",
            str(queries_path),
            "--strategies",
            "exact,postfilter",
            "--limit",
            "3",
            "--ef-search",
            "6",
            "--block-rows",
            "8",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"]["id"] == "file"
    assert payload["dataset"]["rows"] == 32
    assert payload["dataset"]["dim"] == 3
    assert payload["dataset"]["queries"] == 2
    assert payload["dataset"]["query_policy"] == "file"
    assert payload["dataset"]["observed_filter_selectivity"] == pytest.approx(0.25)
    assert [row["strategy"] for row in payload["strategies"]] == ["exact", "postfilter"]
    assert payload["output"] == {"path": str(out_path), "format": "json"}

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["dataset"]["id"] == "file"


def test_plot_benchmark_cli_writes_pareto_svg(tmp_path: Path) -> None:
    benchmark_path = tmp_path / "benchmark.json"
    out_path = tmp_path / "pareto.svg"
    benchmark_path.write_text(json.dumps(_benchmark_plot_payload()), encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "plot-benchmark",
            str(benchmark_path),
            "--out",
            str(out_path),
            "--title",
            "CLI Pareto",
            "--width",
            "820",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output"] == {"path": str(out_path), "format": "svg"}
    assert payload["chart"] == {
        "title": "CLI Pareto",
        "width": 820,
        "kind": "benchmark-pareto",
    }

    svg = out_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "CLI Pareto" in svg
    assert "Pareto frontier" in svg


def test_benchmark_sweep_cli_outputs_grid_csv(tmp_path: Path) -> None:
    out_path = tmp_path / "synthetic-sweep.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark-sweep",
            "--dataset",
            "synthetic",
            "--rows",
            "64",
            "--dim",
            "4",
            "--queries",
            "2",
            "--clusters",
            "4",
            "--filter-selectivities",
            "0.2,0.3",
            "--correlations",
            "0,0.5",
            "--limit",
            "3",
            "--ef-search",
            "6",
            "--max-scan-tuples",
            "32",
            "--probe-rows",
            "8",
            "--block-rows",
            "16",
            "--seed",
            "17",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sweep"]["backend"] == "synthetic"
    assert payload["sweep"]["points"] == 4
    assert payload["sweep"]["filter_selectivities"] == [0.2, 0.3]
    assert payload["sweep"]["correlations"] == [0.0, 0.5]
    assert payload["calibration"]["source"] == "none"
    assert payload["output"] == {"path": str(out_path), "format": "csv"}
    assert payload["points"][0]["local_selectivity"]["probe_rows"] == 8
    assert payload["points"][0]["predicted_best"] is None

    with out_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert {row["strategy"] for row in rows} == {"exact", "postfilter", "iterative"}
    assert rows[0]["s_local_p10"] != ""


def test_crossover_cli_analyzes_sweep_json(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.json"
    out_path = tmp_path / "crossover.json"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep": {
                    "backend": "synthetic",
                    "recall_target": 0.9,
                    "returns_k_target": 1.0,
                },
                "points": [
                    _crossover_cli_point(
                        selectivity=0.01,
                        measured_best="postfilter",
                        predicted_best="postfilter",
                        prediction_match=True,
                        postfilter_recall=0.95,
                    ),
                    _crossover_cli_point(
                        selectivity=0.1,
                        measured_best="iterative",
                        predicted_best="exact",
                        prediction_match=False,
                        postfilter_recall=0.4,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["crossover", str(sweep_path), "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["analysis"]["backend"] == "synthetic"
    assert payload["analysis"]["point_count"] == 2
    assert payload["analysis"]["prediction_match_rate"] == pytest.approx(0.5)
    assert payload["analysis"]["postfilter_failure_count"] == 1
    assert payload["measured_crossovers"][0]["from_strategy"] == "postfilter"
    assert payload["measured_crossovers"][0]["to_strategy"] == "iterative"
    assert payload["predicted_crossovers"][0]["to_strategy"] == "exact"
    assert payload["output"] == {"path": str(out_path), "format": "json"}

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["analysis"]["point_count"] == 2


def test_proof_cli_builds_publishability_report(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.json"
    out_path = tmp_path / "proof.json"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep": {
                    "backend": "synthetic",
                    "recall_target": 0.9,
                    "returns_k_target": 1.0,
                },
                "points": [
                    _crossover_cli_point(
                        selectivity=0.01,
                        measured_best="exact",
                        predicted_best="exact",
                        prediction_match=True,
                        postfilter_recall=0.4,
                    ),
                    _crossover_cli_point(
                        selectivity=0.05,
                        measured_best="iterative",
                        predicted_best="iterative",
                        prediction_match=True,
                        postfilter_recall=0.95,
                    ),
                    _crossover_cli_point(
                        selectivity=0.2,
                        measured_best="postfilter",
                        predicted_best="postfilter",
                        prediction_match=True,
                        postfilter_recall=0.95,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(app, ["proof", str(sweep_path), "--out", str(out_path)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["proof"]["passed"] is True
    assert payload["proof"]["point_count"] == 3
    assert payload["proof"]["prediction_match_rate"] == pytest.approx(1.0)
    assert payload["proof"]["postfilter_failure_count"] == 1
    assert payload["proof"]["safe_advisor_on_postfilter_failures"] == 1
    assert payload["output"] == {"path": str(out_path), "format": "json"}
    assert all(check["passed"] for check in payload["checks"])

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["proof"]["passed"] is True


def test_plot_crossover_cli_writes_svg(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.json"
    out_path = tmp_path / "crossover.svg"
    sweep_path.write_text(
        json.dumps(
            {
                "sweep": {
                    "backend": "synthetic",
                    "recall_target": 0.9,
                    "returns_k_target": 1.0,
                },
                "points": [
                    _crossover_cli_point(
                        selectivity=0.01,
                        measured_best="postfilter",
                        predicted_best="postfilter",
                        prediction_match=True,
                        postfilter_recall=0.95,
                    ),
                    _crossover_cli_point(
                        selectivity=0.1,
                        measured_best="iterative",
                        predicted_best="exact",
                        prediction_match=False,
                        postfilter_recall=0.4,
                    ),
                ],
            }
        ),
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "plot-crossover",
            str(sweep_path),
            "--out",
            str(out_path),
            "--title",
            "CLI Chart",
            "--width",
            "900",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output"] == {"path": str(out_path), "format": "svg"}
    assert payload["chart"]["title"] == "CLI Chart"
    assert payload["chart"]["points"] == 2

    svg = out_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "CLI Chart" in svg
    assert "measured crossover" in svg


def test_benchmark_db_cli_outputs_postgres_strategy_metrics(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    _ = pg_cli_table
    out_path = tmp_path / "postgres-benchmark.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark-db",
            "--dsn",
            TEST_DSN,
            "--dataset",
            "synthetic",
            "--strategies",
            "exact,postfilter,iterative",
            "--rows",
            "64",
            "--dim",
            "3",
            "--queries",
            "2",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.25",
            "--correlation",
            "0.5",
            "--limit",
            "3",
            "--ef-search",
            "8",
            "--max-scan-tuples",
            "48",
            "--iterative-order",
            "relaxed_order",
            "--hnsw-m",
            "8",
            "--hnsw-ef-construction",
            "32",
            "--block-rows",
            "16",
            "--seed",
            "31",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"]["id"] == "postgres-synthetic"
    assert payload["dataset"]["rows"] == 64
    assert [row["strategy"] for row in payload["strategies"]] == [
        "exact",
        "postfilter",
        "iterative",
    ]
    assert payload["strategies"][0]["recall_at_k"] == pytest.approx(1.0)
    assert payload["strategies"][1]["params"]["mode"] == "postgres_hnsw_postfilter"
    assert payload["strategies"][2]["params"]["mode"] == "postgres_hnsw_iterative"
    assert payload["strategies"][2]["params"]["max_scan_tuples"] == 48
    assert payload["output"] == {"path": str(out_path), "format": "json"}
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["dataset"]["id"] == "postgres-synthetic"


def test_benchmark_sweep_db_cli_outputs_postgres_grid_csv(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    _ = pg_cli_table
    out_path = tmp_path / "postgres-sweep.csv"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "benchmark-sweep-db",
            "--dsn",
            TEST_DSN,
            "--dataset",
            "synthetic",
            "--rows",
            "48",
            "--dim",
            "3",
            "--queries",
            "1",
            "--clusters",
            "4",
            "--filter-selectivities",
            "0.25",
            "--correlations",
            "0.5",
            "--limit",
            "2",
            "--ef-search",
            "6",
            "--max-scan-tuples",
            "32",
            "--iterative-order",
            "relaxed_order",
            "--hnsw-m",
            "8",
            "--hnsw-ef-construction",
            "32",
            "--probe-rows",
            "8",
            "--block-rows",
            "16",
            "--seed",
            "37",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sweep"]["backend"] == "postgres"
    assert payload["sweep"]["points"] == 1
    assert payload["sweep"]["iterative_order"] == "relaxed_order"
    assert payload["sweep"]["hnsw_m"] == 8
    assert payload["points"][0]["dataset"]["id"] == "postgres-synthetic"
    assert payload["points"][0]["predicted_best"] is None
    assert payload["output"] == {"path": str(out_path), "format": "csv"}

    with out_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["strategy"] for row in rows] == ["exact", "postfilter", "iterative"]
    assert {row["backend"] for row in rows} == {"postgres"}
    assert {row["hnsw_m"] for row in rows} == {"8"}


def test_calibrate_cli_writes_loadable_profile(tmp_path: Path) -> None:
    out_path = tmp_path / "local-calibration.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--dataset",
            "synthetic",
            "--rows",
            "128",
            "--dim",
            "4",
            "--queries",
            "3",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.25",
            "--correlation",
            "0.5",
            "--limit",
            "5",
            "--block-rows",
            "16",
            "--ef-sweep",
            "8,16",
            "--dataset-id",
            "synthetic-cli-test",
            "--hardware-id",
            "ci-local",
            "--seed",
            "13",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["profile"]["dataset_id"] == "synthetic-cli-test"
    assert payload["profile"]["hardware_id"] == "ci-local"
    assert payload["ef_sweep"] == [8, 16]
    assert payload["output"] == {"path": str(out_path)}

    profile = load_calibration_profile(out_path)
    assert profile.dataset_id == "synthetic-cli-test"
    assert profile.recall_curve[0][0] == 8


def test_validate_cli_outputs_prediction_comparison(tmp_path: Path) -> None:
    out_path = tmp_path / "validation.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate",
            "--dataset",
            "synthetic",
            "--rows",
            "96",
            "--dim",
            "4",
            "--queries",
            "2",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.25",
            "--correlation",
            "0.5",
            "--limit",
            "3",
            "--ef-search",
            "6",
            "--max-scan-tuples",
            "48",
            "--probe-rows",
            "12",
            "--block-rows",
            "16",
            "--ef-sweep",
            "6,12",
            "--seed",
            "23",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["validation"]["predicted_best"] in {"exact", "postfilter", "iterative"}
    assert payload["validation"]["measured_best"] in {"exact", "postfilter", "iterative"}
    assert isinstance(payload["validation"]["match"], bool)
    assert payload["local_selectivity"]["probe_rows"] == 12
    assert [row["strategy"] for row in payload["predictions"]] == [
        "exact",
        "postfilter",
        "iterative",
    ]
    assert payload["calibration"]["source"] == "fitted_synthetic"
    assert payload["output"] == {"path": str(out_path), "format": "json"}
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["validation"] == payload["validation"]


def test_validate_cli_outputs_file_dataset_prediction_comparison(tmp_path: Path) -> None:
    vectors_path, filter_path, queries_path = _write_file_benchmark_fixture(tmp_path)
    calibration_path = _write_validation_profile(tmp_path / "file-calibration.json")
    out_path = tmp_path / "file-validation.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate",
            "--dataset",
            "file",
            "--vectors",
            str(vectors_path),
            "--filter-mask",
            str(filter_path),
            "--query-vectors",
            str(queries_path),
            "--limit",
            "3",
            "--ef-search",
            "6",
            "--max-scan-tuples",
            "24",
            "--probe-rows",
            "8",
            "--block-rows",
            "8",
            "--calibration",
            str(calibration_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["calibration"]["source"] == str(calibration_path)
    assert payload["benchmark"]["dataset"]["id"] == "file"
    assert payload["benchmark"]["dataset"]["rows"] == 32
    assert payload["benchmark"]["dataset"]["query_policy"] == "file"
    assert payload["local_selectivity"]["query_count"] == 2
    assert payload["validation"]["predicted_best"] in {"exact", "postfilter", "iterative"}
    assert payload["validation"]["measured_best"] in {"exact", "postfilter", "iterative"}
    assert payload["output"] == {"path": str(out_path), "format": "json"}


def test_db_calibrate_and_validate_cli_flow(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    _ = pg_cli_table
    calibration_path = tmp_path / "db-calibration.json"
    validation_path = tmp_path / "db-validation.json"
    runner = CliRunner()
    calibrate_result = runner.invoke(
        app,
        [
            "calibrate-db",
            "--dsn",
            TEST_DSN,
            "--dataset",
            "synthetic",
            "--rows",
            "48",
            "--dim",
            "3",
            "--queries",
            "1",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.25",
            "--correlation",
            "0.5",
            "--limit",
            "2",
            "--block-rows",
            "16",
            "--ef-sweep",
            "6,12",
            "--dataset-id",
            "postgres-cli-test",
            "--hardware-id",
            "ci-pgvector",
            "--seed",
            "41",
            "--out",
            str(calibration_path),
        ],
    )

    assert calibrate_result.exit_code == 0, calibrate_result.output
    calibration_payload = json.loads(calibrate_result.output)
    assert calibration_payload["profile"]["dataset_id"] == "postgres-cli-test"
    assert calibration_payload["ef_sweep"] == [6, 12]
    assert calibration_payload["output"] == {"path": str(calibration_path)}
    assert load_calibration_profile(calibration_path).hardware_id == "ci-pgvector"

    validate_result = runner.invoke(
        app,
        [
            "validate-db",
            "--dsn",
            TEST_DSN,
            "--dataset",
            "synthetic",
            "--rows",
            "48",
            "--dim",
            "3",
            "--queries",
            "1",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.25",
            "--correlation",
            "0.5",
            "--limit",
            "2",
            "--ef-search",
            "6",
            "--max-scan-tuples",
            "32",
            "--probe-rows",
            "12",
            "--block-rows",
            "16",
            "--calibration",
            str(calibration_path),
            "--seed",
            "41",
            "--out",
            str(validation_path),
        ],
    )

    assert validate_result.exit_code == 0, validate_result.output
    validation_payload = json.loads(validate_result.output)
    assert validation_payload["calibration"]["source"] == str(calibration_path)
    assert validation_payload["benchmark"]["dataset"]["id"] == "postgres-synthetic"
    assert validation_payload["validation"]["predicted_best"] in {
        "exact",
        "postfilter",
        "iterative",
    }
    assert validation_payload["validation"]["measured_best"] in {
        "exact",
        "postfilter",
        "iterative",
    }
    assert validation_payload["output"] == {"path": str(validation_path), "format": "json"}


def test_validate_db_cli_outputs_file_dataset_prediction_comparison(
    pg_cli_table: str,
    tmp_path: Path,
) -> None:
    _ = pg_cli_table
    vectors_path, filter_path, queries_path = _write_file_benchmark_fixture(tmp_path)
    calibration_path = _write_validation_profile(tmp_path / "file-db-calibration.json")
    out_path = tmp_path / "file-db-validation.json"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "validate-db",
            "--dsn",
            TEST_DSN,
            "--dataset",
            "file",
            "--vectors",
            str(vectors_path),
            "--filter-mask",
            str(filter_path),
            "--query-vectors",
            str(queries_path),
            "--limit",
            "2",
            "--ef-search",
            "6",
            "--max-scan-tuples",
            "24",
            "--probe-rows",
            "8",
            "--block-rows",
            "8",
            "--calibration",
            str(calibration_path),
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["calibration"]["source"] == str(calibration_path)
    assert payload["benchmark"]["dataset"]["id"] == "postgres-file"
    assert payload["benchmark"]["dataset"]["rows"] == 32
    assert payload["benchmark"]["dataset"]["query_policy"] == "file"
    assert payload["validation"]["predicted_best"] in {"exact", "postfilter", "iterative"}
    assert payload["validation"]["measured_best"] in {"exact", "postfilter", "iterative"}
    assert payload["output"] == {"path": str(out_path), "format": "json"}


def _write_file_benchmark_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    np = pytest.importorskip("numpy")
    vectors_path = tmp_path / "vectors.npy"
    filter_path = tmp_path / "filter.npy"
    queries_path = tmp_path / "queries.npy"
    np.save(vectors_path, np.arange(96, dtype="float32").reshape(32, 3))
    np.save(filter_path, np.asarray([index % 4 == 0 for index in range(32)]))
    np.save(queries_path, np.asarray([[0.0, 1.0, 2.0], [9.0, 10.0, 11.0]], dtype="float32"))
    return vectors_path, filter_path, queries_path


def _write_validation_profile(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "dataset_id": "file-test",
                "hardware_id": "local-test",
                "index_method": "hnsw",
                "c_d": 0.02,
                "c_scan": 0.01,
                "c_h": 3.0,
                "delta_strict": 0.15,
                "recall_curve": [[6, 0.8], [12, 0.9], [40, 0.95]],
            }
        ),
        encoding="utf-8",
    )
    return path


def _benchmark_plot_payload() -> dict[str, object]:
    return {
        "dataset": {
            "id": "synthetic",
            "rows": 1000,
            "queries": 10,
        },
        "ground_truth": {
            "metric": "l2",
            "k": 10,
        },
        "strategies": [
            _benchmark_plot_strategy("exact", recall=1.0, returns_k=1.0, total_ms=100.0),
            _benchmark_plot_strategy("postfilter", recall=0.5, returns_k=0.4, total_ms=20.0),
            _benchmark_plot_strategy("iterative", recall=0.95, returns_k=1.0, total_ms=40.0),
        ],
    }


def _benchmark_plot_strategy(
    strategy: str,
    *,
    recall: float,
    returns_k: float,
    total_ms: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "params": {},
        "query_count": 10,
        "recall_at_k": recall,
        "returns_k_rate": returns_k,
        "result_count_mean": 10.0 * returns_k,
        "latency_ms": {
            "total": total_ms,
            "mean": total_ms / 10,
            "p50": total_ms / 10,
            "p95": total_ms / 8,
            "p99": total_ms / 7,
        },
    }


def _crossover_cli_point(
    *,
    selectivity: float,
    measured_best: str,
    predicted_best: str,
    prediction_match: bool,
    postfilter_recall: float,
) -> dict[str, object]:
    return {
        "target_filter_selectivity": selectivity,
        "target_correlation": 0.0,
        "dataset": {"observed_filter_selectivity": selectivity},
        "local_selectivity": {
            "s_local_p10": selectivity,
            "s_local_median": selectivity,
        },
        "measured_best": measured_best,
        "predicted_best": predicted_best,
        "prediction_match": prediction_match,
        "strategies": [
            _crossover_cli_strategy("exact", recall=1.0, returns_k=1.0),
            _crossover_cli_strategy("postfilter", recall=postfilter_recall, returns_k=1.0),
            _crossover_cli_strategy("iterative", recall=0.95, returns_k=1.0),
        ],
    }


def _crossover_cli_strategy(
    strategy: str,
    *,
    recall: float,
    returns_k: float,
) -> dict[str, object]:
    return {
        "strategy": strategy,
        "recall_at_k": recall,
        "returns_k_rate": returns_k,
        "latency_ms": {
            "mean": 1.0,
            "p95": 2.0,
        },
    }
