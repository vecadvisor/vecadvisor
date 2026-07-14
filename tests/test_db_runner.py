from __future__ import annotations

import os
from collections.abc import Iterator

import psycopg
import pytest

from vecadvisor.bench.calibrate import run_postgres_calibration
from vecadvisor.bench.datasets import generate_synthetic_dataset, generate_synthetic_queries
from vecadvisor.bench.db_runner import (
    parse_db_strategy_list,
    run_postgres_synthetic_benchmark,
)
from vecadvisor.bench.runner import (
    STRATEGY_EXACT,
    STRATEGY_ITERATIVE,
    STRATEGY_PARTIAL,
    STRATEGY_PARTITION,
    STRATEGY_POSTFILTER,
)
from vecadvisor.bench.sweep import run_postgres_sweep, sweep_report_to_json
from vecadvisor.bench.validate import run_postgres_validation
from vecadvisor.introspect import connect

TEST_DSN = os.getenv(
    "VECADVISOR_TEST_DSN",
    "postgresql://postgres:postgres@localhost:5432/vecadvisor",
)


@pytest.fixture(scope="module")
def pg_conn() -> Iterator[object]:
    try:
        conn = connect(TEST_DSN)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL test database is not available: {exc}")
    with conn:
        yield conn


def test_parse_db_strategy_list_supports_iterative() -> None:
    assert parse_db_strategy_list(None) == (
        STRATEGY_EXACT,
        STRATEGY_POSTFILTER,
        STRATEGY_ITERATIVE,
        STRATEGY_PARTIAL,
        STRATEGY_PARTITION,
    )
    assert parse_db_strategy_list("exact, partial, partition") == (
        STRATEGY_EXACT,
        STRATEGY_PARTIAL,
        STRATEGY_PARTITION,
    )

    with pytest.raises(ValueError, match="unknown DB benchmark strategy"):
        parse_db_strategy_list("exact,unknown")


def test_run_postgres_synthetic_benchmark_measures_actual_sql(pg_conn: object) -> None:
    dataset = generate_synthetic_dataset(
        n_rows=96,
        dim=3,
        n_clusters=4,
        filter_selectivity=0.25,
        correlation=0.5,
        seed=301,
    )
    queries = generate_synthetic_queries(dataset, n_queries=2, seed=302)

    report = run_postgres_synthetic_benchmark(
        pg_conn,  # type: ignore[arg-type]
        dataset=dataset,
        queries=queries,
        k=3,
        strategies=(
            STRATEGY_EXACT,
            STRATEGY_POSTFILTER,
            STRATEGY_ITERATIVE,
            STRATEGY_PARTIAL,
            STRATEGY_PARTITION,
        ),
        ef_search=8,
        max_scan_tuples=48,
        iterative_order="relaxed_order",
        hnsw_m=8,
        hnsw_ef_construction=32,
        block_rows=16,
        maintenance_work_mem="64MB",
    )

    metrics_by_strategy = {metrics.strategy: metrics for metrics in report.strategies}
    assert report.dataset["id"] == "postgres-synthetic"
    assert report.dataset["rows"] == 96
    assert report.dataset["maintenance_work_mem"] == "64MB"
    assert report.ground_truth["k"] == 3
    assert set(metrics_by_strategy) == {
        STRATEGY_EXACT,
        STRATEGY_POSTFILTER,
        STRATEGY_ITERATIVE,
        STRATEGY_PARTIAL,
        STRATEGY_PARTITION,
    }
    assert metrics_by_strategy[STRATEGY_EXACT].recall_at_k == pytest.approx(1.0)
    assert metrics_by_strategy[STRATEGY_EXACT].latency_ms_mean >= 0.0
    assert metrics_by_strategy[STRATEGY_POSTFILTER].params["ef_search"] == 8
    assert metrics_by_strategy[STRATEGY_POSTFILTER].latency_ms_mean >= 0.0
    assert metrics_by_strategy[STRATEGY_ITERATIVE].params["max_scan_tuples"] == 48
    assert metrics_by_strategy[STRATEGY_ITERATIVE].params["iterative_order"] == "relaxed_order"
    assert metrics_by_strategy[STRATEGY_ITERATIVE].latency_ms_mean >= 0.0
    assert metrics_by_strategy[STRATEGY_PARTIAL].params["mode"] == (
        "postgres_hnsw_partial_index"
    )
    assert metrics_by_strategy[STRATEGY_PARTIAL].latency_ms_mean >= 0.0
    assert metrics_by_strategy[STRATEGY_PARTITION].params["mode"] == (
        "postgres_hnsw_partition_pruned"
    )
    assert metrics_by_strategy[STRATEGY_PARTITION].latency_ms_mean >= 0.0
    assert any("actual PostgreSQL" in note for note in report.notes)


def test_run_postgres_calibration_fits_profile_from_actual_sql(pg_conn: object) -> None:
    fit = run_postgres_calibration(
        pg_conn,  # type: ignore[arg-type]
        rows=64,
        dim=3,
        queries=2,
        clusters=4,
        filter_selectivity=0.25,
        correlation=0.5,
        limit=3,
        block_rows=16,
        ef_sweep=(6, 12),
        seed=311,
        dataset_id="postgres-test",
        hardware_id="ci-pgvector",
    )

    assert fit.profile.dataset_id == "postgres-test"
    assert fit.profile.hardware_id == "ci-pgvector"
    assert fit.profile.c_d > 0
    assert fit.profile.c_scan > 0
    assert fit.profile.c_h > 0
    assert [ef for ef, _ in fit.profile.recall_curve] == [6, 12]
    assert len(fit.reports) == 2
    assert any("actual PostgreSQL" in note for note in fit.notes)


def test_run_postgres_validation_compares_cost_model_to_actual_sql(pg_conn: object) -> None:
    fit = run_postgres_calibration(
        pg_conn,  # type: ignore[arg-type]
        rows=64,
        dim=3,
        queries=2,
        clusters=4,
        filter_selectivity=0.25,
        correlation=0.5,
        limit=3,
        block_rows=16,
        ef_sweep=(6, 12),
        seed=321,
    )
    dataset = generate_synthetic_dataset(
        n_rows=64,
        dim=3,
        n_clusters=4,
        filter_selectivity=0.25,
        correlation=0.5,
        seed=321,
    )
    queries = generate_synthetic_queries(dataset, n_queries=2, seed=322)

    report = run_postgres_validation(
        pg_conn,  # type: ignore[arg-type]
        dataset=dataset,
        queries=queries,
        calibration=fit.profile,
        k=3,
        ef_search=6,
        max_scan_tuples=48,
        probe_rows=12,
        block_rows=16,
        recall_target=0.8,
    )

    assert report.predicted_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert report.measured_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert isinstance(report.match, bool)
    assert report.benchmark.dataset["id"] == "postgres-synthetic"
    assert any("actual PostgreSQL" in note for note in report.notes[1:])


def test_run_postgres_sweep_measures_actual_sql_grid(pg_conn: object) -> None:
    report = run_postgres_sweep(
        pg_conn,  # type: ignore[arg-type]
        rows=64,
        dim=3,
        queries=2,
        clusters=4,
        filter_selectivities=(0.25,),
        correlations=(0.5,),
        limit=3,
        ef_search=6,
        max_scan_tuples=48,
        iterative_order="relaxed_order",
        hnsw_m=8,
        hnsw_ef_construction=32,
        probe_rows=12,
        block_rows=16,
        seed=331,
    )

    assert report.backend == "postgres"
    assert report.iterative_order == "relaxed_order"
    assert report.hnsw_m == 8
    assert len(report.points) == 1
    point = report.points[0]
    assert point.benchmark.dataset["id"] == "postgres-synthetic"
    assert point.measured_best in {STRATEGY_EXACT, STRATEGY_POSTFILTER, STRATEGY_ITERATIVE}
    assert point.predicted_best is None
    assert point.local_selectivity.probe_rows == 12

    payload = sweep_report_to_json(report, calibration_source="none")
    assert payload["sweep"]["backend"] == "postgres"
    assert payload["sweep"]["hnsw_m"] == 8
    assert payload["points"][0]["dataset"]["id"] == "postgres-synthetic"
