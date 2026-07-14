from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from . import __version__
from .bench.calibrate import (
    calibration_fit_to_json,
    parse_ef_sweep,
    run_postgres_calibration,
    run_synthetic_calibration,
)
from .bench.crossover import (
    analyze_sweep_payload,
    crossover_analysis_to_json,
    load_sweep_payload,
    write_crossover_analysis,
)
from .bench.datasets import (
    SyntheticDataset,
    SyntheticQueries,
    generate_synthetic_dataset,
    generate_synthetic_queries,
    load_file_dataset,
    load_file_queries,
)
from .bench.db_runner import parse_db_strategy_list, run_postgres_synthetic_benchmark
from .bench.plots import (
    DEFAULT_CHART_TITLE,
    DEFAULT_PARETO_TITLE,
    load_benchmark_payload,
    write_benchmark_pareto_svg,
    write_crossover_svg,
)
from .bench.proof import build_proof_report, proof_report_to_json, write_proof_report
from .bench.runner import (
    benchmark_report_to_json,
    infer_output_format,
    parse_strategy_list,
    run_synthetic_benchmark,
    write_benchmark_report,
)
from .bench.sweep import (
    DEFAULT_CORRELATION_SWEEP,
    DEFAULT_FILTER_SELECTIVITY_SWEEP,
    parse_float_sweep,
    run_postgres_sweep,
    run_synthetic_sweep,
    sweep_report_to_json,
    write_sweep_report,
)
from .bench.validate import (
    run_postgres_validation,
    run_synthetic_validation,
    validation_report_to_json,
    write_validation_report,
)
from .calibration import (
    calibration_profile_to_json,
    load_calibration_profile,
    save_calibration_profile,
)
from .diagnostics import parse_diagnostic_format, render_explain_vector
from .introspect import connect, introspect_table
from .local_cache import (
    build_local_selectivity_cache_key,
    load_local_selectivity_cache,
    local_selectivity_cache_path,
    store_local_selectivity_cache,
)
from .local_probe import (
    DEFAULT_MAX_QUERY_VECTORS,
    DEFAULT_PROBE_ROWS,
    LocalProbeError,
    load_query_vector,
    load_query_vectors,
    load_query_vectors_from_sql,
    load_query_vectors_from_table_sample,
    run_local_selectivity_probe,
    run_local_selectivity_probes,
)
from .models import (
    CostEstimate,
    ExtendedStatsMeta,
    LocalSelectivity,
    Predicate,
    Recommendation,
    SelectivityCrossCheck,
    StatisticsSuggestion,
    StrategyPlan,
    TableStats,
)
from .pgversion import load_pg_capabilities, pg_capabilities_to_json
from .plan import compare_selectivity, explain_query
from .planner_observer import ObservedPlannerChoice, observe_planner_choice
from .query_spec import build_filter_select_sql, query_spec_from_filter
from .recommend import (
    DEFAULT_CALIBRATION,
    DEFAULT_EF_SEARCH,
    DEFAULT_RECALL_TARGET,
    build_recommendation,
)
from .selectivity import conjunction_selectivity
from .statistics_advisor import suggest_statistics
from .stats_health import assess_stats_health, stats_health_to_json

app = typer.Typer(
    help="Cost-based advisor for filtered pgvector search.",
    invoke_without_command=True,
)
console = Console()
DEFAULT_FILTER_SELECTIVITY_SWEEP_OPTION = ",".join(
    str(value) for value in DEFAULT_FILTER_SELECTIVITY_SWEEP
)
DEFAULT_CORRELATION_SWEEP_OPTION = ",".join(str(value) for value in DEFAULT_CORRELATION_SWEEP)
TABLE_SAMPLE_VECTOR_CONFIDENCE_FACTOR = 0.5


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", help="Show version and exit."),
    ] = False,
) -> None:
    if version:
        console.print(f"VecAdvisor {__version__}")
        raise typer.Exit


@app.command()
def analyze(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    table: Annotated[str, typer.Option(help="Target table, optionally schema-qualified.")],
    vector: Annotated[str | None, typer.Option(help="Vector column name.")] = None,
) -> None:
    """Print table/index inventory. DB introspection is the next implementation step."""
    with connect(dsn) as conn:
        stats = introspect_table(conn, table, vector_column=vector)
        capabilities = load_pg_capabilities(conn)
    stats_health = assess_stats_health(stats)
    console.print_json(
        json.dumps(
            {
                "dsn": _redact_dsn(dsn),
                "table": stats.relname,
                "rows": stats.n_rows,
                "pages": stats.n_pages,
                "vector_dim": stats.vector_dim,
                "catalog_snapshot": _catalog_snapshot_to_json(stats),
                "columns": [
                    {
                        "name": column.name,
                        "type": column.type_name,
                        "has_stats": column.has_stats,
                        "n_distinct": column.n_distinct,
                    }
                    for column in stats.columns
                ],
                "indexes": [
                    {
                        "name": index.name,
                        "method": index.method,
                        "columns": index.columns,
                        "opclass": index.opclass,
                        "partial": index.is_partial,
                        "predicate": index.predicate,
                        "m": index.m,
                        "ef_construction": index.ef_construction,
                        "lists": index.lists,
                    }
                    for index in stats.indexes
                ],
                "extended_stats": [
                    _extended_stats_to_json(stat) for stat in stats.extended_stats
                ],
                "stats_health": stats_health_to_json(stats_health),
                "postgres": pg_capabilities_to_json(capabilities),
            }
        )
    )


@app.command()
def explain(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    table: Annotated[str, typer.Option(help="Target table.")],
    vector: Annotated[str, typer.Option(help="Vector column.")],
    query: Annotated[str, typer.Option(help="Restricted filter predicate.")],
    q_vector: Annotated[Path | None, typer.Option("--q-vector", help="Query vector file.")] = None,
    limit: Annotated[int, typer.Option(min=1, help="LIMIT k.")] = 10,
    probe_rows: Annotated[
        int,
        typer.Option(
            "--probe-rows",
            min=1,
            help="Top-m unfiltered neighbors used for the local selectivity probe.",
        ),
    ] = DEFAULT_PROBE_ROWS,
    ef_search: Annotated[
        int,
        typer.Option(
            "--ef-search",
            min=1,
            help="Candidate ef_search used by the recommendation cost model.",
        ),
    ] = DEFAULT_EF_SEARCH,
    recall_target: Annotated[
        float,
        typer.Option(
            "--recall-target",
            min=0.01,
            max=1.0,
            help="Minimum estimated recall required for ANN candidates.",
        ),
    ] = DEFAULT_RECALL_TARGET,
    calibration: Annotated[
        Path | None,
        typer.Option(
            "--calibration",
            help="CalibrationProfile JSON file produced by benchmark/calibrate.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: json or text."),
    ] = "json",
) -> None:
    """Estimate filter selectivity and compare it with PostgreSQL's plan estimate."""
    try:
        diagnostic_format = parse_diagnostic_format(output_format)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    calibration_profile = (
        load_calibration_profile(calibration) if calibration is not None else DEFAULT_CALIBRATION
    )
    with connect(dsn) as conn:
        stats = introspect_table(conn, table, vector_column=vector)
        capabilities = load_pg_capabilities(conn)
        stats_health = assess_stats_health(stats)
        query_spec = query_spec_from_filter(
            relname=stats.relname,
            vector_column=vector,
            filter_sql=query,
            limit=limit,
            allowed_columns=(column.name for column in stats.columns),
        )
        advisor_selectivity = conjunction_selectivity(stats, query_spec.predicates)
        filter_select_sql = build_filter_select_sql(stats.relname, query)
        plan = explain_query(conn, filter_select_sql)
        cross_check = compare_selectivity(stats, plan, advisor_selectivity)
        statistics_suggestions = suggest_statistics(
            table=stats,
            query=query_spec,
            cross_check=cross_check,
        )
        query_vector_dims: int | None = None
        local_selectivity: LocalSelectivity | None = None
        local_probe_warning: str | None = None
        observed_planner: ObservedPlannerChoice | None = None
        if q_vector is not None:
            query_vector = load_query_vector(q_vector, expected_dim=stats.vector_dim)
            query_vector_dims = len(query_vector)
            try:
                local_selectivity = run_local_selectivity_probe(
                    conn,
                    table=stats,
                    query=query_spec,
                    filter_sql=query,
                    query_vector=query_vector,
                    s_global=advisor_selectivity,
                    probe_rows=probe_rows,
                )
            except LocalProbeError as exc:
                local_probe_warning = _local_probe_fallback_warning(exc)
            observed_planner = observe_planner_choice(
                conn,
                table=stats,
                query=query_spec,
                filter_sql=query,
                query_vector=query_vector,
            )
        recommendation = build_recommendation(
            table=stats,
            query=query_spec,
            s_global=advisor_selectivity,
            local_selectivity=local_selectivity,
            calibration=calibration_profile,
            ef_search=ef_search,
            recall_target=recall_target,
            planner_would_pick=(
                observed_planner.strategy if observed_planner is not None else None
            ),
            supports_iterative_scan=capabilities.supports_hnsw_iterative_scan,
        )

    payload = {
        "dsn": _redact_dsn(dsn),
        "table": stats.relname,
        "vector": query_spec.vector_column,
        "vector_dim": stats.vector_dim,
        "catalog_snapshot": _catalog_snapshot_to_json(stats),
        "filter": query,
        "predicates": [_predicate_to_json(predicate) for predicate in query_spec.predicates],
        "selectivity": _cross_check_to_json(cross_check),
        "statistics_suggestions": [
            _statistics_suggestion_to_json(suggestion) for suggestion in statistics_suggestions
        ],
        "stats_health": stats_health_to_json(stats_health),
        "postgres": pg_capabilities_to_json(capabilities),
        "local_selectivity": _local_selectivity_to_json(local_selectivity),
        "plan": {
            "root_node_type": plan.root.node_type,
            "planning_time_ms": plan.planning_time_ms,
            "filter_select_sql": filter_select_sql,
            "observed_vector_query": _observed_planner_to_json(observed_planner),
        },
        "q_vector": (
            {
                "path": str(q_vector),
                "dimensions": query_vector_dims,
            }
            if q_vector
            else None
        ),
        "limit": limit,
        "probe_rows": probe_rows,
        "ef_search": ef_search,
        "recall_target": recall_target,
        "calibration": {
            "source": str(calibration) if calibration is not None else "default",
            "profile": calibration_profile_to_json(calibration_profile),
        },
        "recommendation": _recommendation_to_json(
            recommendation,
            local_selectivity=local_selectivity,
            recall_target=recall_target,
        ),
        "notes": _explain_notes(q_vector, local_selectivity, local_probe_warning),
    }
    _print_diagnostic_payload(payload, diagnostic_format)


@app.command(name="recommend")
def recommend_command(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    table: Annotated[str, typer.Option(help="Target table.")],
    vector: Annotated[str, typer.Option(help="Vector column.")],
    query: Annotated[str, typer.Option(help="Restricted filter predicate.")],
    q_vectors: Annotated[
        Path | None,
        typer.Option(
            "--q-vectors",
            help="Representative query-vector sample file: JSON, JSONL/text, or .npy.",
        ),
    ] = None,
    q_vector_sql: Annotated[
        str | None,
        typer.Option(
            "--q-vector-sql",
            help="One-column SELECT/WITH SQL returning representative query vectors.",
        ),
    ] = None,
    allow_table_sample_vectors: Annotated[
        bool,
        typer.Option(
            "--allow-table-sample-vectors",
            help=(
                "Opt in to a low-confidence fallback that samples query vectors from the "
                "target table when no representative query-vector source is supplied."
            ),
        ),
    ] = False,
    limit: Annotated[int, typer.Option(min=1, help="LIMIT k.")] = 10,
    probe_rows: Annotated[
        int,
        typer.Option(
            "--probe-rows",
            min=1,
            help="Top-m unfiltered neighbors used for each local selectivity probe.",
        ),
    ] = DEFAULT_PROBE_ROWS,
    max_query_vectors: Annotated[
        int,
        typer.Option(
            "--max-query-vectors",
            min=1,
            help="Maximum representative query vectors to load/probe.",
        ),
    ] = DEFAULT_MAX_QUERY_VECTORS,
    local_cache_dir: Annotated[
        Path | None,
        typer.Option(
            "--local-cache-dir",
            help=(
                "Optional directory for cached aggregate local-selectivity probes. "
                "Cache keys include catalog fingerprints and query-vector fingerprints."
            ),
        ),
    ] = None,
    refresh_local_cache: Annotated[
        bool,
        typer.Option(
            "--refresh-local-cache",
            help="Ignore an existing local-selectivity cache entry and overwrite it.",
        ),
    ] = False,
    ef_search: Annotated[
        int,
        typer.Option(
            "--ef-search",
            min=1,
            help="Candidate ef_search used by the recommendation cost model.",
        ),
    ] = DEFAULT_EF_SEARCH,
    recall_target: Annotated[
        float,
        typer.Option(
            "--recall-target",
            min=0.01,
            max=1.0,
            help="Minimum estimated recall required for ANN candidates.",
        ),
    ] = DEFAULT_RECALL_TARGET,
    calibration: Annotated[
        Path | None,
        typer.Option(
            "--calibration",
            help="CalibrationProfile JSON file produced by benchmark/calibrate.",
        ),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format: json or text."),
    ] = "json",
) -> None:
    """Recommend a durable filtered-vector strategy from representative query vectors."""

    if q_vectors is not None and q_vector_sql is not None:
        raise typer.BadParameter("use only one of --q-vectors or --q-vector-sql")
    try:
        diagnostic_format = parse_diagnostic_format(output_format)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    calibration_profile = (
        load_calibration_profile(calibration) if calibration is not None else DEFAULT_CALIBRATION
    )
    with connect(dsn) as conn:
        stats = introspect_table(conn, table, vector_column=vector)
        capabilities = load_pg_capabilities(conn)
        stats_health = assess_stats_health(stats)
        query_spec = query_spec_from_filter(
            relname=stats.relname,
            vector_column=vector,
            filter_sql=query,
            limit=limit,
            allowed_columns=(column.name for column in stats.columns),
        )
        advisor_selectivity = conjunction_selectivity(stats, query_spec.predicates)
        filter_select_sql = build_filter_select_sql(stats.relname, query)
        plan = explain_query(conn, filter_select_sql)
        cross_check = compare_selectivity(stats, plan, advisor_selectivity)
        statistics_suggestions = suggest_statistics(
            table=stats,
            query=query_spec,
            cross_check=cross_check,
        )
        query_vectors, vector_source = _representative_query_vectors(
            conn,
            table=stats,
            vector_column=query_spec.vector_column,
            q_vectors=q_vectors,
            q_vector_sql=q_vector_sql,
            expected_dim=stats.vector_dim,
            max_query_vectors=max_query_vectors,
            allow_table_sample_vectors=allow_table_sample_vectors,
        )
        local_selectivity: LocalSelectivity | None = None
        local_cache_key: str | None = None
        local_cache_file: Path | None = None
        local_cache_hit = False
        local_cache_stored = False
        local_probe_warning: str | None = None
        observed_planner: ObservedPlannerChoice | None = None
        if query_vectors:
            if local_cache_dir is not None:
                local_cache_key = build_local_selectivity_cache_key(
                    table=stats,
                    query=query_spec,
                    filter_sql=query,
                    vector_source=vector_source,
                    query_vectors=query_vectors,
                    probe_rows=probe_rows,
                    s_global=advisor_selectivity,
                )
                local_cache_file = local_selectivity_cache_path(
                    local_cache_dir,
                    local_cache_key,
                )
                if not refresh_local_cache:
                    local_selectivity = load_local_selectivity_cache(
                        local_cache_dir,
                        local_cache_key,
                    )
                    local_cache_hit = local_selectivity is not None
            if local_selectivity is None:
                try:
                    local_selectivity = run_local_selectivity_probes(
                        conn,
                        table=stats,
                        query=query_spec,
                        filter_sql=query,
                        query_vectors=query_vectors,
                        s_global=advisor_selectivity,
                        probe_rows=probe_rows,
                    )
                except LocalProbeError as exc:
                    local_probe_warning = _local_probe_fallback_warning(exc)
                else:
                    if _is_table_sample_vector_source(vector_source):
                        local_selectivity = _downgrade_table_sample_local_selectivity(
                            local_selectivity
                        )
                    if local_cache_dir is not None and local_cache_key is not None:
                        local_cache_file = store_local_selectivity_cache(
                            local_cache_dir,
                            local_cache_key,
                            local_selectivity,
                        )
                        local_cache_stored = True
            observed_planner = observe_planner_choice(
                conn,
                table=stats,
                query=query_spec,
                filter_sql=query,
                query_vector=query_vectors[0],
            )
        recommendation = build_recommendation(
            table=stats,
            query=query_spec,
            s_global=advisor_selectivity,
            local_selectivity=local_selectivity,
            calibration=calibration_profile,
            ef_search=ef_search,
            recall_target=recall_target,
            planner_would_pick=(
                observed_planner.strategy if observed_planner is not None else None
            ),
            supports_iterative_scan=capabilities.supports_hnsw_iterative_scan,
        )

    payload = {
        "dsn": _redact_dsn(dsn),
        "table": stats.relname,
        "vector": query_spec.vector_column,
        "vector_dim": stats.vector_dim,
        "catalog_snapshot": _catalog_snapshot_to_json(stats),
        "filter": query,
        "predicates": [_predicate_to_json(predicate) for predicate in query_spec.predicates],
        "selectivity": _cross_check_to_json(cross_check),
        "statistics_suggestions": [
            _statistics_suggestion_to_json(suggestion) for suggestion in statistics_suggestions
        ],
        "stats_health": stats_health_to_json(stats_health),
        "postgres": pg_capabilities_to_json(capabilities),
        "local_selectivity": _local_selectivity_to_json(local_selectivity),
        "local_selectivity_cache": _local_selectivity_cache_to_json(
            cache_dir=local_cache_dir,
            cache_key=local_cache_key,
            cache_file=local_cache_file,
            hit=local_cache_hit,
            stored=local_cache_stored,
            refresh=refresh_local_cache,
        ),
        "plan": {
            "root_node_type": plan.root.node_type,
            "planning_time_ms": plan.planning_time_ms,
            "filter_select_sql": filter_select_sql,
            "observed_vector_query": _observed_planner_to_json(observed_planner),
        },
        "query_vectors": {
            "source": vector_source,
            "count": len(query_vectors),
            "max_query_vectors": max_query_vectors,
        },
        "limit": limit,
        "probe_rows": probe_rows,
        "ef_search": ef_search,
        "recall_target": recall_target,
        "calibration": {
            "source": str(calibration) if calibration is not None else "default",
            "profile": calibration_profile_to_json(calibration_profile),
        },
        "recommendation": _recommendation_to_json(
            recommendation,
            local_selectivity=local_selectivity,
            recall_target=recall_target,
        ),
        "notes": _recommend_notes(vector_source, local_selectivity, local_probe_warning),
    }
    _print_diagnostic_payload(payload, diagnostic_format)


@app.command()
def calibrate(
    out: Annotated[
        Path,
        typer.Option("--out", help="CalibrationProfile JSON output path."),
    ],
    dataset: Annotated[str, typer.Option(help="Calibration dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 5_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 64,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 50,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 16,
    filter_selectivity: Annotated[
        float,
        typer.Option(
            "--filter-selectivity",
            min=0.000001,
            max=0.999999,
            help="Target global selectivity for the synthetic filter.",
        ),
    ] = 0.1,
    correlation: Annotated[
        float,
        typer.Option(
            "--correlation",
            min=-1.0,
            max=1.0,
            help="Synthetic filter/cluster correlation in [-1, 1].",
        ),
    ] = 0.0,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    block_rows: Annotated[
        int | None,
        typer.Option(
            "--block-rows",
            min=1,
            help="Rows per exact-search block. Defaults to memory-budget derived value.",
        ),
    ] = None,
    ef_sweep: Annotated[
        str,
        typer.Option("--ef-sweep", help="Comma-separated ef_search sweep."),
    ] = "10,20,40,80,160",
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    dataset_id: Annotated[
        str,
        typer.Option("--dataset-id", help="dataset_id stored in the calibration profile."),
    ] = "synthetic-simulated",
    hardware_id: Annotated[
        str,
        typer.Option("--hardware-id", help="hardware_id stored in the calibration profile."),
    ] = "local-synthetic-cpu",
) -> None:
    """Fit and write a CalibrationProfile JSON file."""

    if dataset != "synthetic":
        raise typer.BadParameter("only --dataset synthetic is implemented for calibration")
    try:
        ef_points = parse_ef_sweep(ef_sweep)
        fit = run_synthetic_calibration(
            rows=rows,
            dim=dim,
            queries=queries,
            clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            limit=limit,
            metric=metric,
            block_rows=block_rows,
            seed=seed,
            ef_sweep=ef_points,
            dataset_id=dataset_id,
            hardware_id=hardware_id,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    save_calibration_profile(fit.profile, out)
    payload = calibration_fit_to_json(fit)
    payload["output"] = {"path": str(out)}
    console.print_json(json.dumps(payload))


@app.command(name="calibrate-db")
def calibrate_db(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    out: Annotated[
        Path,
        typer.Option("--out", help="CalibrationProfile JSON output path."),
    ],
    dataset: Annotated[str, typer.Option(help="Calibration dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 1_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 32,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 10,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 8,
    filter_selectivity: Annotated[
        float,
        typer.Option(
            "--filter-selectivity",
            min=0.000001,
            max=0.999999,
            help="Target global selectivity for the synthetic filter.",
        ),
    ] = 0.1,
    correlation: Annotated[
        float,
        typer.Option(
            "--correlation",
            min=-1.0,
            max=1.0,
            help="Synthetic filter/cluster correlation in [-1, 1].",
        ),
    ] = 0.0,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    block_rows: Annotated[
        int | None,
        typer.Option("--block-rows", min=1, help="Rows per exact ground-truth block."),
    ] = None,
    ef_sweep: Annotated[
        str,
        typer.Option("--ef-sweep", help="Comma-separated ef_search sweep."),
    ] = "10,20,40,80,160",
    hnsw_m: Annotated[
        int,
        typer.Option("--hnsw-m", min=1, help="HNSW m used for temporary indexes."),
    ] = 8,
    hnsw_ef_construction: Annotated[
        int,
        typer.Option(
            "--hnsw-ef-construction",
            min=1,
            help="HNSW ef_construction used for temporary indexes.",
        ),
    ] = 32,
    statement_timeout_ms: Annotated[
        int,
        typer.Option(
            "--statement-timeout-ms",
            min=1,
            help="Statement timeout for PostgreSQL calibration queries.",
        ),
    ] = 30_000,
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    dataset_id: Annotated[
        str,
        typer.Option("--dataset-id", help="dataset_id stored in the calibration profile."),
    ] = "postgres-synthetic",
    hardware_id: Annotated[
        str,
        typer.Option("--hardware-id", help="hardware_id stored in the calibration profile."),
    ] = "local-postgres-pgvector",
) -> None:
    """Fit and write a CalibrationProfile from actual PostgreSQL/pgvector timings."""

    if dataset != "synthetic":
        raise typer.BadParameter("only --dataset synthetic is implemented for DB calibration")
    try:
        with connect(dsn) as conn:
            fit = run_postgres_calibration(
                conn,
                rows=rows,
                dim=dim,
                queries=queries,
                clusters=clusters,
                filter_selectivity=filter_selectivity,
                correlation=correlation,
                limit=limit,
                metric=metric,
                block_rows=block_rows,
                seed=seed,
                ef_sweep=parse_ef_sweep(ef_sweep),
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                statement_timeout_ms=statement_timeout_ms,
                dataset_id=dataset_id,
                hardware_id=hardware_id,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    save_calibration_profile(fit.profile, out)
    payload = calibration_fit_to_json(fit)
    payload["output"] = {"path": str(out)}
    console.print_json(json.dumps(payload))


@app.command()
def benchmark(
    dataset: Annotated[str, typer.Option(help="Benchmark dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 5_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 64,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 50,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 16,
    vectors_path: Annotated[
        Path | None,
        typer.Option("--vectors", help="2D .npy vector matrix for --dataset file."),
    ] = None,
    filter_mask_path: Annotated[
        Path | None,
        typer.Option(
            "--filter-mask",
            help="1D .npy boolean/numeric filter mask for --dataset file.",
        ),
    ] = None,
    query_vectors_path: Annotated[
        Path | None,
        typer.Option("--query-vectors", help="2D .npy query vector matrix for --dataset file."),
    ] = None,
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help="Comma-separated strategies: exact,postfilter,iterative, or all.",
        ),
    ] = "all",
    filter_selectivity: Annotated[
        float,
        typer.Option(
            "--filter-selectivity",
            min=0.000001,
            max=0.999999,
            help="Target global selectivity for the synthetic filter.",
        ),
    ] = 0.1,
    correlation: Annotated[
        float,
        typer.Option(
            "--correlation",
            min=-1.0,
            max=1.0,
            help="Synthetic filter/cluster correlation in [-1, 1].",
        ),
    ] = 0.0,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    ef_search: Annotated[
        int,
        typer.Option(
            "--ef-search",
            min=1,
            help="Candidate pool size for the postfilter semantics simulation.",
        ),
    ] = 40,
    max_scan_tuples: Annotated[
        int,
        typer.Option(
            "--max-scan-tuples",
            min=1,
            help="Candidate expansion cap for the iterative semantics simulation.",
        ),
    ] = 1_000,
    block_rows: Annotated[
        int | None,
        typer.Option(
            "--block-rows",
            min=1,
            help="Rows per exact-search block. Defaults to memory-budget derived value.",
        ),
    ] = None,
    query_policy: Annotated[
        str,
        typer.Option(
            "--query-policy",
            help="Synthetic query clusters: uniform, filter_hot, or filter_cold.",
        ),
    ] = "uniform",
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON/CSV benchmark output path."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format for --out: auto, json, or csv.",
        ),
    ] = "auto",
) -> None:
    """Run calibration/validation benchmark building blocks."""

    try:
        strategy_ids = parse_strategy_list(strategies)
        report_format = infer_output_format(out, output_format)
        benchmark_dataset, query_set = _load_cli_benchmark_inputs(
            dataset=dataset,
            rows=rows,
            dim=dim,
            queries=queries,
            clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            query_policy=query_policy,
            seed=seed,
            vectors_path=vectors_path,
            filter_mask_path=filter_mask_path,
            query_vectors_path=query_vectors_path,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    report = run_synthetic_benchmark(
        dataset=benchmark_dataset,
        queries=query_set,
        k=limit,
        metric=metric,
        strategies=strategy_ids,
        ef_search=ef_search,
        max_scan_tuples=max_scan_tuples,
        block_rows=block_rows,
    )
    if out is not None:
        write_benchmark_report(report, out, output_format=report_format)
    payload = benchmark_report_to_json(report)
    if out is not None:
        payload["output"] = {"path": str(out), "format": report_format}

    console.print_json(
        json.dumps(payload)
    )


@app.command(name="benchmark-sweep")
def benchmark_sweep(
    dataset: Annotated[str, typer.Option(help="Sweep dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 5_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 64,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 50,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 16,
    filter_selectivities: Annotated[
        str,
        typer.Option(
            "--filter-selectivities",
            help="Comma-separated global selectivity sweep points in (0, 1).",
        ),
    ] = DEFAULT_FILTER_SELECTIVITY_SWEEP_OPTION,
    correlations: Annotated[
        str,
        typer.Option(
            "--correlations",
            help="Comma-separated filter/vector correlation sweep points in [-1, 1].",
        ),
    ] = DEFAULT_CORRELATION_SWEEP_OPTION,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    ef_search: Annotated[
        int,
        typer.Option(
            "--ef-search",
            min=1,
            help="Candidate pool size for the postfilter semantics simulation.",
        ),
    ] = 40,
    max_scan_tuples: Annotated[
        int,
        typer.Option(
            "--max-scan-tuples",
            min=1,
            help="Candidate expansion cap for the iterative semantics simulation.",
        ),
    ] = 1_000,
    probe_rows: Annotated[
        int,
        typer.Option(
            "--probe-rows",
            min=1,
            help="Unfiltered top-m rows for local selectivity probes.",
        ),
    ] = 200,
    recall_target: Annotated[
        float,
        typer.Option(
            "--recall-target",
            min=0.01,
            max=1.0,
            help="Minimum measured/estimated recall for viable strategies.",
        ),
    ] = DEFAULT_RECALL_TARGET,
    returns_k_target: Annotated[
        float,
        typer.Option(
            "--returns-k-target",
            min=0.0,
            max=1.0,
            help="Minimum fraction of queries returning all available k rows.",
        ),
    ] = 1.0,
    block_rows: Annotated[
        int | None,
        typer.Option(
            "--block-rows",
            min=1,
            help="Rows per exact-search block. Defaults to memory-budget derived value.",
        ),
    ] = None,
    query_policy: Annotated[
        str,
        typer.Option(
            "--query-policy",
            help="Synthetic query clusters: uniform, filter_hot, or filter_cold.",
        ),
    ] = "uniform",
    calibration: Annotated[
        Path | None,
        typer.Option(
            "--calibration",
            help="Optional CalibrationProfile JSON for prediction-vs-measured columns.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON/CSV sweep output path."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format for --out: auto, json, or csv."),
    ] = "auto",
) -> None:
    """Run a selectivity/correlation benchmark sweep for chart-ready analysis."""

    if dataset != "synthetic":
        raise typer.BadParameter("only --dataset synthetic is implemented for sweep")
    try:
        selectivity_points = parse_float_sweep(
            filter_selectivities,
            default=DEFAULT_FILTER_SELECTIVITY_SWEEP,
            name="filter_selectivity",
            min_value=0.0,
            max_value=1.0,
            include_min=False,
            include_max=False,
        )
        correlation_points = parse_float_sweep(
            correlations,
            default=DEFAULT_CORRELATION_SWEEP,
            name="correlation",
            min_value=-1.0,
            max_value=1.0,
        )
        report_format = infer_output_format(out, output_format)
        calibration_profile = load_calibration_profile(calibration) if calibration else None
        calibration_source = str(calibration) if calibration else "none"
        report = run_synthetic_sweep(
            rows=rows,
            dim=dim,
            queries=queries,
            clusters=clusters,
            filter_selectivities=selectivity_points,
            correlations=correlation_points,
            limit=limit,
            metric=metric,
            ef_search=ef_search,
            max_scan_tuples=max_scan_tuples,
            probe_rows=probe_rows,
            recall_target=recall_target,
            returns_k_target=returns_k_target,
            block_rows=block_rows,
            query_policy=query_policy,
            seed=seed,
            calibration=calibration_profile,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_sweep_report(
            report,
            out,
            output_format=report_format,
            calibration_source=calibration_source,
        )
    payload = sweep_report_to_json(report, calibration_source=calibration_source)
    if out is not None:
        payload["output"] = {"path": str(out), "format": report_format}
    console.print_json(json.dumps(payload))


@app.command(name="benchmark-sweep-db")
def benchmark_sweep_db(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    dataset: Annotated[str, typer.Option(help="Sweep dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 1_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 32,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 10,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 8,
    filter_selectivities: Annotated[
        str,
        typer.Option(
            "--filter-selectivities",
            help="Comma-separated global selectivity sweep points in (0, 1).",
        ),
    ] = DEFAULT_FILTER_SELECTIVITY_SWEEP_OPTION,
    correlations: Annotated[
        str,
        typer.Option(
            "--correlations",
            help="Comma-separated filter/vector correlation sweep points in [-1, 1].",
        ),
    ] = DEFAULT_CORRELATION_SWEEP_OPTION,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    ef_search: Annotated[
        int,
        typer.Option("--ef-search", min=1, help="hnsw.ef_search for sweep queries."),
    ] = 40,
    max_scan_tuples: Annotated[
        int,
        typer.Option(
            "--max-scan-tuples",
            min=1,
            help="hnsw.max_scan_tuples for iterative sweep queries.",
        ),
    ] = 1_000,
    iterative_order: Annotated[
        str,
        typer.Option(
            "--iterative-order",
            help="hnsw.iterative_scan mode: relaxed_order or strict_order.",
        ),
    ] = "relaxed_order",
    hnsw_m: Annotated[
        int,
        typer.Option("--hnsw-m", min=1, help="HNSW m used for temporary indexes."),
    ] = 8,
    hnsw_ef_construction: Annotated[
        int,
        typer.Option(
            "--hnsw-ef-construction",
            min=1,
            help="HNSW ef_construction used for temporary indexes.",
        ),
    ] = 32,
    probe_rows: Annotated[
        int,
        typer.Option(
            "--probe-rows",
            min=1,
            help="Unfiltered top-m rows for local selectivity probes.",
        ),
    ] = 200,
    recall_target: Annotated[
        float,
        typer.Option(
            "--recall-target",
            min=0.01,
            max=1.0,
            help="Minimum measured/estimated recall for viable strategies.",
        ),
    ] = DEFAULT_RECALL_TARGET,
    returns_k_target: Annotated[
        float,
        typer.Option(
            "--returns-k-target",
            min=0.0,
            max=1.0,
            help="Minimum fraction of queries returning all available k rows.",
        ),
    ] = 1.0,
    block_rows: Annotated[
        int | None,
        typer.Option("--block-rows", min=1, help="Rows per exact ground-truth block."),
    ] = None,
    query_policy: Annotated[
        str,
        typer.Option(
            "--query-policy",
            help="Synthetic query clusters: uniform, filter_hot, or filter_cold.",
        ),
    ] = "uniform",
    calibration: Annotated[
        Path | None,
        typer.Option(
            "--calibration",
            help="Optional CalibrationProfile JSON for prediction-vs-measured columns.",
        ),
    ] = None,
    statement_timeout_ms: Annotated[
        int,
        typer.Option(
            "--statement-timeout-ms",
            min=1,
            help="Statement timeout for PostgreSQL sweep queries.",
        ),
    ] = 30_000,
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON/CSV sweep output path."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format for --out: auto, json, or csv."),
    ] = "auto",
) -> None:
    """Run an actual PostgreSQL/pgvector selectivity/correlation sweep."""

    if dataset != "synthetic":
        raise typer.BadParameter("only --dataset synthetic is implemented for DB sweep")
    try:
        selectivity_points = parse_float_sweep(
            filter_selectivities,
            default=DEFAULT_FILTER_SELECTIVITY_SWEEP,
            name="filter_selectivity",
            min_value=0.0,
            max_value=1.0,
            include_min=False,
            include_max=False,
        )
        correlation_points = parse_float_sweep(
            correlations,
            default=DEFAULT_CORRELATION_SWEEP,
            name="correlation",
            min_value=-1.0,
            max_value=1.0,
        )
        report_format = infer_output_format(out, output_format)
        calibration_profile = load_calibration_profile(calibration) if calibration else None
        calibration_source = str(calibration) if calibration else "none"
        with connect(dsn) as conn:
            report = run_postgres_sweep(
                conn,
                rows=rows,
                dim=dim,
                queries=queries,
                clusters=clusters,
                filter_selectivities=selectivity_points,
                correlations=correlation_points,
                limit=limit,
                metric=metric,
                ef_search=ef_search,
                max_scan_tuples=max_scan_tuples,
                iterative_order=iterative_order,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                probe_rows=probe_rows,
                recall_target=recall_target,
                returns_k_target=returns_k_target,
                block_rows=block_rows,
                query_policy=query_policy,
                seed=seed,
                calibration=calibration_profile,
                statement_timeout_ms=statement_timeout_ms,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_sweep_report(
            report,
            out,
            output_format=report_format,
            calibration_source=calibration_source,
        )
    payload = sweep_report_to_json(report, calibration_source=calibration_source)
    if out is not None:
        payload["output"] = {"path": str(out), "format": report_format}
    console.print_json(json.dumps(payload))


@app.command(name="crossover")
def crossover_command(
    sweep: Annotated[
        Path,
        typer.Argument(help="Sweep JSON produced by benchmark-sweep or benchmark-sweep-db."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON crossover analysis output path."),
    ] = None,
) -> None:
    """Analyze winner regions and selectivity crossovers from a sweep JSON file."""

    try:
        payload = load_sweep_payload(sweep)
        analysis = analyze_sweep_payload(payload)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_crossover_analysis(analysis, out)
    output = crossover_analysis_to_json(analysis)
    if out is not None:
        output["output"] = {"path": str(out), "format": "json"}
    console.print_json(json.dumps(output))


@app.command(name="proof")
def proof_command(
    sweep: Annotated[
        Path,
        typer.Argument(help="Sweep JSON produced by benchmark-sweep or benchmark-sweep-db."),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON proof report output path."),
    ] = None,
    min_points: Annotated[
        int,
        typer.Option("--min-points", min=1, help="Minimum sweep point count."),
    ] = 3,
    min_selectivity_bins: Annotated[
        int,
        typer.Option(
            "--min-selectivity-bins",
            min=1,
            help="Minimum unique target selectivity bins.",
        ),
    ] = 3,
    min_match_rate: Annotated[
        float,
        typer.Option(
            "--min-match-rate",
            min=0.0,
            max=1.0,
            help="Minimum predicted-vs-measured winner match rate.",
        ),
    ] = 0.8,
    min_postfilter_failures: Annotated[
        int,
        typer.Option(
            "--min-postfilter-failures",
            min=0,
            help="Minimum bins where postfilter fails recall or returns-k targets.",
        ),
    ] = 1,
    fail_on_miss: Annotated[
        bool,
        typer.Option("--fail-on-miss", help="Exit non-zero when proof checks fail."),
    ] = False,
) -> None:
    """Build a publishability proof report from a calibrated sweep JSON file."""

    try:
        payload = load_sweep_payload(sweep)
        analysis = analyze_sweep_payload(payload)
        report = build_proof_report(
            analysis,
            min_points=min_points,
            min_selectivity_bins=min_selectivity_bins,
            min_match_rate=min_match_rate,
            min_postfilter_failures=min_postfilter_failures,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_proof_report(report, out)
    output = proof_report_to_json(report)
    if out is not None:
        output["output"] = {"path": str(out), "format": "json"}
    console.print_json(json.dumps(output))
    if fail_on_miss and not report.passed:
        raise typer.Exit(1)


@app.command(name="plot-crossover")
def plot_crossover_command(
    sweep: Annotated[
        Path,
        typer.Argument(help="Sweep JSON produced by benchmark-sweep or benchmark-sweep-db."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="SVG chart output path."),
    ],
    title: Annotated[
        str,
        typer.Option("--title", help="Chart title."),
    ] = DEFAULT_CHART_TITLE,
    width: Annotated[
        int,
        typer.Option("--width", min=760, help="SVG width in pixels."),
    ] = 1120,
) -> None:
    """Render the crossover money chart as SVG."""

    try:
        payload = load_sweep_payload(sweep)
        analysis = analyze_sweep_payload(payload)
        write_crossover_svg(analysis, out, title=title, width=width)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print_json(
        json.dumps(
            {
                "input": {"path": str(sweep)},
                "output": {"path": str(out), "format": "svg"},
                "chart": {
                    "title": title,
                    "width": width,
                    "backend": analysis.backend,
                    "points": analysis.point_count,
                    "correlations": list(analysis.correlations),
                    "prediction_match_rate": analysis.prediction_match_rate,
                    "postfilter_failure_count": analysis.postfilter_failure_count,
                },
            }
        )
    )


@app.command(name="plot-benchmark")
def plot_benchmark_command(
    benchmark: Annotated[
        Path,
        typer.Argument(help="Benchmark JSON produced by benchmark or benchmark-db."),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="SVG chart output path."),
    ],
    title: Annotated[
        str,
        typer.Option("--title", help="Chart title."),
    ] = DEFAULT_PARETO_TITLE,
    width: Annotated[
        int,
        typer.Option("--width", min=700, help="SVG width in pixels."),
    ] = 920,
) -> None:
    """Render a benchmark recall-vs-QPS Pareto chart as SVG."""

    try:
        payload = load_benchmark_payload(benchmark)
        write_benchmark_pareto_svg(payload, out, title=title, width=width)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print_json(
        json.dumps(
            {
                "input": {"path": str(benchmark)},
                "output": {"path": str(out), "format": "svg"},
                "chart": {
                    "title": title,
                    "width": width,
                    "kind": "benchmark-pareto",
                },
            }
        )
    )


@app.command(name="benchmark-db")
def benchmark_db(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    dataset: Annotated[str, typer.Option(help="Benchmark dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 1_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 32,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 10,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 8,
    vectors_path: Annotated[
        Path | None,
        typer.Option("--vectors", help="2D .npy vector matrix for --dataset file."),
    ] = None,
    filter_mask_path: Annotated[
        Path | None,
        typer.Option("--filter-mask", help="1D .npy filter mask for --dataset file."),
    ] = None,
    query_vectors_path: Annotated[
        Path | None,
        typer.Option("--query-vectors", help="2D .npy query vector matrix for --dataset file."),
    ] = None,
    strategies: Annotated[
        str,
        typer.Option(
            "--strategies",
            help=(
                "Comma-separated DB strategies: exact,postfilter,iterative,partial,"
                "partition, or all."
            ),
        ),
    ] = "all",
    filter_selectivity: Annotated[
        float,
        typer.Option(
            "--filter-selectivity",
            min=0.000001,
            max=0.999999,
            help="Target global selectivity for the synthetic filter.",
        ),
    ] = 0.1,
    correlation: Annotated[
        float,
        typer.Option(
            "--correlation",
            min=-1.0,
            max=1.0,
            help="Synthetic filter/cluster correlation in [-1, 1].",
        ),
    ] = 0.0,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    ef_search: Annotated[
        int,
        typer.Option(
            "--ef-search",
            min=1,
            help="hnsw.ef_search for the postfilter benchmark.",
        ),
    ] = 40,
    max_scan_tuples: Annotated[
        int,
        typer.Option(
            "--max-scan-tuples",
            min=1,
            help="hnsw.max_scan_tuples for the iterative benchmark.",
        ),
    ] = 1_000,
    iterative_order: Annotated[
        str,
        typer.Option(
            "--iterative-order",
            help="hnsw.iterative_scan mode: relaxed_order or strict_order.",
        ),
    ] = "relaxed_order",
    hnsw_m: Annotated[
        int,
        typer.Option("--hnsw-m", min=1, help="HNSW m used for the temporary index."),
    ] = 8,
    hnsw_ef_construction: Annotated[
        int,
        typer.Option(
            "--hnsw-ef-construction",
            min=1,
            help="HNSW ef_construction used for the temporary index.",
        ),
    ] = 32,
    block_rows: Annotated[
        int | None,
        typer.Option(
            "--block-rows",
            min=1,
            help="Rows per exact ground-truth block.",
        ),
    ] = None,
    query_policy: Annotated[
        str,
        typer.Option(
            "--query-policy",
            help="Synthetic query clusters: uniform, filter_hot, or filter_cold.",
        ),
    ] = "uniform",
    statement_timeout_ms: Annotated[
        int,
        typer.Option(
            "--statement-timeout-ms",
            min=1,
            help="Statement timeout for PostgreSQL benchmark queries.",
        ),
    ] = 30_000,
    maintenance_work_mem: Annotated[
        str | None,
        typer.Option(
            "--maintenance-work-mem",
            help="Optional transaction-local maintenance_work_mem for DB index builds.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON/CSV benchmark output path."),
    ] = None,
    output_format: Annotated[
        str,
        typer.Option("--format", help="Output format for --out: auto, json, or csv."),
    ] = "auto",
) -> None:
    """Run actual PostgreSQL/pgvector benchmark queries on a temp table."""

    try:
        strategy_ids = parse_db_strategy_list(strategies)
        report_format = infer_output_format(out, output_format)
        benchmark_dataset, query_set = _load_cli_benchmark_inputs(
            dataset=dataset,
            rows=rows,
            dim=dim,
            queries=queries,
            clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            query_policy=query_policy,
            seed=seed,
            vectors_path=vectors_path,
            filter_mask_path=filter_mask_path,
            query_vectors_path=query_vectors_path,
        )
        with connect(dsn) as conn:
            report = run_postgres_synthetic_benchmark(
                conn,
                dataset=benchmark_dataset,
                queries=query_set,
                k=limit,
                metric=metric,
                strategies=strategy_ids,
                ef_search=ef_search,
                max_scan_tuples=max_scan_tuples,
                iterative_order=iterative_order,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                block_rows=block_rows,
                statement_timeout_ms=statement_timeout_ms,
                maintenance_work_mem=maintenance_work_mem,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_benchmark_report(report, out, output_format=report_format)
    payload = benchmark_report_to_json(report)
    if out is not None:
        payload["output"] = {"path": str(out), "format": report_format}
    console.print_json(json.dumps(payload))


@app.command()
def validate(
    dataset: Annotated[str, typer.Option(help="Validation dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 5_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 64,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 50,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 16,
    vectors_path: Annotated[
        Path | None,
        typer.Option("--vectors", help="2D .npy vector matrix for --dataset file."),
    ] = None,
    filter_mask_path: Annotated[
        Path | None,
        typer.Option(
            "--filter-mask",
            help="1D .npy boolean/numeric filter mask for --dataset file.",
        ),
    ] = None,
    query_vectors_path: Annotated[
        Path | None,
        typer.Option("--query-vectors", help="2D .npy query vector matrix for --dataset file."),
    ] = None,
    filter_selectivity: Annotated[
        float,
        typer.Option(
            "--filter-selectivity",
            min=0.000001,
            max=0.999999,
            help="Target global selectivity for the synthetic filter.",
        ),
    ] = 0.1,
    correlation: Annotated[
        float,
        typer.Option(
            "--correlation",
            min=-1.0,
            max=1.0,
            help="Synthetic filter/cluster correlation in [-1, 1].",
        ),
    ] = 0.0,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    ef_search: Annotated[
        int,
        typer.Option(
            "--ef-search",
            min=1,
            help="Candidate pool size for the postfilter semantics simulation.",
        ),
    ] = 40,
    max_scan_tuples: Annotated[
        int,
        typer.Option(
            "--max-scan-tuples",
            min=1,
            help="Candidate expansion cap for the iterative semantics simulation.",
        ),
    ] = 1_000,
    probe_rows: Annotated[
        int,
        typer.Option(
            "--probe-rows",
            min=1,
            help="Unfiltered top-m rows for local selectivity validation probes.",
        ),
    ] = 200,
    recall_target: Annotated[
        float,
        typer.Option(
            "--recall-target",
            min=0.01,
            max=1.0,
            help="Minimum measured/estimated recall for viable strategies.",
        ),
    ] = DEFAULT_RECALL_TARGET,
    returns_k_target: Annotated[
        float,
        typer.Option(
            "--returns-k-target",
            min=0.0,
            max=1.0,
            help="Minimum fraction of queries returning all available k rows.",
        ),
    ] = 1.0,
    block_rows: Annotated[
        int | None,
        typer.Option(
            "--block-rows",
            min=1,
            help="Rows per exact-search block. Defaults to memory-budget derived value.",
        ),
    ] = None,
    query_policy: Annotated[
        str,
        typer.Option(
            "--query-policy",
            help="Synthetic query clusters: uniform, filter_hot, or filter_cold.",
        ),
    ] = "uniform",
    calibration: Annotated[
        Path | None,
        typer.Option("--calibration", help="Existing CalibrationProfile JSON path."),
    ] = None,
    ef_sweep: Annotated[
        str,
        typer.Option("--ef-sweep", help="Comma-separated ef_search sweep when fitting."),
    ] = "10,20,40,80,160",
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON validation report output path."),
    ] = None,
) -> None:
    """Validate cost-model prediction against measured strategy metrics."""

    try:
        if calibration is not None:
            calibration_profile = load_calibration_profile(calibration)
            calibration_source = str(calibration)
        elif dataset == "file":
            raise ValueError("--dataset file requires --calibration")
        else:
            fit = run_synthetic_calibration(
                rows=rows,
                dim=dim,
                queries=queries,
                clusters=clusters,
                filter_selectivity=filter_selectivity,
                correlation=correlation,
                limit=limit,
                metric=metric,
                block_rows=block_rows,
                seed=seed,
                ef_sweep=parse_ef_sweep(ef_sweep),
            )
            calibration_profile = fit.profile
            calibration_source = "fitted_synthetic"
        validation_dataset, query_set = _load_cli_benchmark_inputs(
            dataset=dataset,
            rows=rows,
            dim=dim,
            queries=queries,
            clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            query_policy=query_policy,
            seed=seed,
            vectors_path=vectors_path,
            filter_mask_path=filter_mask_path,
            query_vectors_path=query_vectors_path,
        )
        report = run_synthetic_validation(
            dataset=validation_dataset,
            queries=query_set,
            calibration=calibration_profile,
            k=limit,
            metric=metric,
            ef_search=ef_search,
            max_scan_tuples=max_scan_tuples,
            probe_rows=probe_rows,
            recall_target=recall_target,
            returns_k_target=returns_k_target,
            block_rows=block_rows,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_validation_report(report, out, calibration_source=calibration_source)
    payload = validation_report_to_json(report, calibration_source=calibration_source)
    if out is not None:
        payload["output"] = {"path": str(out), "format": "json"}
    console.print_json(json.dumps(payload))


@app.command(name="validate-db")
def validate_db(
    dsn: Annotated[str, typer.Option(help="PostgreSQL connection string.")],
    dataset: Annotated[str, typer.Option(help="Validation dataset id.")] = "synthetic",
    rows: Annotated[int, typer.Option("--rows", min=1, help="Synthetic base row count.")] = 1_000,
    dim: Annotated[int, typer.Option("--dim", min=1, help="Synthetic vector dimension.")] = 32,
    queries: Annotated[
        int,
        typer.Option("--queries", min=1, help="Synthetic query-vector count."),
    ] = 10,
    clusters: Annotated[
        int,
        typer.Option("--clusters", min=1, help="Synthetic cluster count."),
    ] = 8,
    vectors_path: Annotated[
        Path | None,
        typer.Option("--vectors", help="2D .npy vector matrix for --dataset file."),
    ] = None,
    filter_mask_path: Annotated[
        Path | None,
        typer.Option("--filter-mask", help="1D .npy filter mask for --dataset file."),
    ] = None,
    query_vectors_path: Annotated[
        Path | None,
        typer.Option("--query-vectors", help="2D .npy query vector matrix for --dataset file."),
    ] = None,
    filter_selectivity: Annotated[
        float,
        typer.Option(
            "--filter-selectivity",
            min=0.000001,
            max=0.999999,
            help="Target global selectivity for the synthetic filter.",
        ),
    ] = 0.1,
    correlation: Annotated[
        float,
        typer.Option(
            "--correlation",
            min=-1.0,
            max=1.0,
            help="Synthetic filter/cluster correlation in [-1, 1].",
        ),
    ] = 0.0,
    limit: Annotated[int, typer.Option("--limit", min=1, help="Ground-truth k.")] = 10,
    metric: Annotated[
        str,
        typer.Option("--metric", help="Distance metric: l2, ip, cosine."),
    ] = "l2",
    ef_search: Annotated[
        int,
        typer.Option("--ef-search", min=1, help="hnsw.ef_search for validation."),
    ] = 40,
    max_scan_tuples: Annotated[
        int,
        typer.Option(
            "--max-scan-tuples",
            min=1,
            help="hnsw.max_scan_tuples for iterative validation.",
        ),
    ] = 1_000,
    iterative_order: Annotated[
        str,
        typer.Option(
            "--iterative-order",
            help="hnsw.iterative_scan mode: relaxed_order or strict_order.",
        ),
    ] = "relaxed_order",
    probe_rows: Annotated[
        int,
        typer.Option(
            "--probe-rows",
            min=1,
            help="Unfiltered top-m rows for local selectivity validation probes.",
        ),
    ] = 200,
    recall_target: Annotated[
        float,
        typer.Option(
            "--recall-target",
            min=0.01,
            max=1.0,
            help="Minimum measured/estimated recall for viable strategies.",
        ),
    ] = DEFAULT_RECALL_TARGET,
    returns_k_target: Annotated[
        float,
        typer.Option(
            "--returns-k-target",
            min=0.0,
            max=1.0,
            help="Minimum fraction of queries returning all available k rows.",
        ),
    ] = 1.0,
    hnsw_m: Annotated[
        int,
        typer.Option("--hnsw-m", min=1, help="HNSW m used for temporary indexes."),
    ] = 8,
    hnsw_ef_construction: Annotated[
        int,
        typer.Option(
            "--hnsw-ef-construction",
            min=1,
            help="HNSW ef_construction used for temporary indexes.",
        ),
    ] = 32,
    block_rows: Annotated[
        int | None,
        typer.Option("--block-rows", min=1, help="Rows per exact ground-truth block."),
    ] = None,
    query_policy: Annotated[
        str,
        typer.Option(
            "--query-policy",
            help="Synthetic query clusters: uniform, filter_hot, or filter_cold.",
        ),
    ] = "uniform",
    calibration: Annotated[
        Path | None,
        typer.Option("--calibration", help="Existing CalibrationProfile JSON path."),
    ] = None,
    ef_sweep: Annotated[
        str,
        typer.Option("--ef-sweep", help="Comma-separated ef_search sweep when fitting."),
    ] = "10,20,40,80,160",
    statement_timeout_ms: Annotated[
        int,
        typer.Option(
            "--statement-timeout-ms",
            min=1,
            help="Statement timeout for PostgreSQL validation queries.",
        ),
    ] = 30_000,
    seed: Annotated[int, typer.Option("--seed", help="Synthetic dataset seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Optional JSON validation report output path."),
    ] = None,
) -> None:
    """Validate cost-model prediction against actual PostgreSQL/pgvector measurements."""

    try:
        with connect(dsn) as conn:
            if calibration is not None:
                calibration_profile = load_calibration_profile(calibration)
                calibration_source = str(calibration)
            elif dataset == "file":
                raise ValueError("--dataset file requires --calibration")
            else:
                fit = run_postgres_calibration(
                    conn,
                    rows=rows,
                    dim=dim,
                    queries=queries,
                    clusters=clusters,
                    filter_selectivity=filter_selectivity,
                    correlation=correlation,
                    limit=limit,
                    metric=metric,
                    block_rows=block_rows,
                    seed=seed,
                    ef_sweep=parse_ef_sweep(ef_sweep),
                    hnsw_m=hnsw_m,
                    hnsw_ef_construction=hnsw_ef_construction,
                    statement_timeout_ms=statement_timeout_ms,
                )
                calibration_profile = fit.profile
                calibration_source = "fitted_postgres"
            validation_dataset, query_set = _load_cli_benchmark_inputs(
                dataset=dataset,
                rows=rows,
                dim=dim,
                queries=queries,
                clusters=clusters,
                filter_selectivity=filter_selectivity,
                correlation=correlation,
                query_policy=query_policy,
                seed=seed,
                vectors_path=vectors_path,
                filter_mask_path=filter_mask_path,
                query_vectors_path=query_vectors_path,
            )
            report = run_postgres_validation(
                conn,
                dataset=validation_dataset,
                queries=query_set,
                calibration=calibration_profile,
                k=limit,
                metric=metric,
                ef_search=ef_search,
                max_scan_tuples=max_scan_tuples,
                iterative_order=iterative_order,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                probe_rows=probe_rows,
                recall_target=recall_target,
                returns_k_target=returns_k_target,
                block_rows=block_rows,
                statement_timeout_ms=statement_timeout_ms,
            )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if out is not None:
        write_validation_report(report, out, calibration_source=calibration_source)
    payload = validation_report_to_json(report, calibration_source=calibration_source)
    if out is not None:
        payload["output"] = {"path": str(out), "format": "json"}
    console.print_json(json.dumps(payload))


def _representative_query_vectors(
    conn: Any,
    *,
    table: TableStats,
    vector_column: str,
    q_vectors: Path | None,
    q_vector_sql: str | None,
    expected_dim: int | None,
    max_query_vectors: int,
    allow_table_sample_vectors: bool,
) -> tuple[tuple[tuple[float, ...], ...], str]:
    if q_vectors is not None:
        return (
            load_query_vectors(
                q_vectors,
                expected_dim=expected_dim,
                max_vectors=max_query_vectors,
            ),
            f"file:{q_vectors}",
        )
    if q_vector_sql is not None:
        return (
            load_query_vectors_from_sql(
                conn,
                q_vector_sql,
                expected_dim=expected_dim,
                max_vectors=max_query_vectors,
            ),
            "sql",
        )
    if allow_table_sample_vectors:
        return (
            load_query_vectors_from_table_sample(
                conn,
                table=table,
                vector_column=vector_column,
                expected_dim=expected_dim,
                max_vectors=max_query_vectors,
            ),
            f"table_sample:{table.relname}.{vector_column}",
        )
    return (), "global_selectivity_fallback"


def _load_cli_benchmark_inputs(
    *,
    dataset: str,
    rows: int,
    dim: int,
    queries: int,
    clusters: int,
    filter_selectivity: float,
    correlation: float,
    query_policy: str,
    seed: int,
    vectors_path: Path | None,
    filter_mask_path: Path | None,
    query_vectors_path: Path | None,
) -> tuple[SyntheticDataset, SyntheticQueries]:
    if dataset == "synthetic":
        synthetic = generate_synthetic_dataset(
            n_rows=rows,
            dim=dim,
            n_clusters=clusters,
            filter_selectivity=filter_selectivity,
            correlation=correlation,
            seed=seed,
        )
        query_set = generate_synthetic_queries(
            synthetic,
            n_queries=queries,
            seed=seed + 1,
            cluster_policy=query_policy,
        )
        return synthetic, query_set

    if dataset != "file":
        raise ValueError("dataset must be one of: synthetic, file")
    if vectors_path is None or filter_mask_path is None or query_vectors_path is None:
        raise ValueError("--dataset file requires --vectors, --filter-mask, and --query-vectors")

    file_dataset = load_file_dataset(
        vectors_path=vectors_path,
        filter_mask_path=filter_mask_path,
    )
    query_set = load_file_queries(
        query_vectors_path=query_vectors_path,
        expected_dim=file_dataset.dim,
    )
    return file_dataset, query_set


def _print_diagnostic_payload(payload: dict[str, object], output_format: str) -> None:
    if output_format == "json":
        console.print_json(json.dumps(payload))
        return
    console.out(render_explain_vector(payload), end="")


def _redact_dsn(dsn: str) -> str:
    if "@" not in dsn:
        return dsn
    scheme, rest = dsn.split("://", 1) if "://" in dsn else ("", dsn)
    after_at = rest.split("@", 1)[1]
    prefix = f"{scheme}://" if scheme else ""
    return f"{prefix}***:***@{after_at}"


def _predicate_to_json(predicate: Predicate) -> dict[str, object]:
    return {
        "column": predicate.column,
        "kind": predicate.kind.value,
        "values": list(predicate.values),
        "is_literal": predicate.is_literal,
    }


def _catalog_snapshot_to_json(table: TableStats) -> dict[str, object]:
    return {
        "stats_fingerprint": table.stats_fingerprint,
        "index_fingerprint": table.index_fingerprint,
        "last_analyze_at": _datetime_to_json(table.last_analyze),
        "last_autoanalyze_at": _datetime_to_json(table.last_autoanalyze),
        "n_live_tup": table.n_live_tup,
        "n_mod_since_analyze": table.n_mod_since_analyze,
        "partitioned_by": list(table.partitioned_by or ()),
    }


def _datetime_to_json(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _extended_stats_to_json(stat: ExtendedStatsMeta) -> dict[str, object]:
    return {
        "name": stat.name,
        "schema": stat.schema,
        "columns": list(stat.columns),
        "kinds": list(stat.kinds),
    }


def _cross_check_to_json(cross_check: SelectivityCrossCheck) -> dict[str, object]:
    return {
        "relation_name": cross_check.relation_name,
        "advisor_selectivity": cross_check.advisor_selectivity,
        "postgres_selectivity": cross_check.postgres_selectivity,
        "advisor_rows": cross_check.advisor_rows,
        "postgres_plan_rows": cross_check.postgres_plan_rows,
        "absolute_delta": cross_check.absolute_delta,
        "ratio": cross_check.ratio,
        "plan_node_type": cross_check.plan_node_type,
        "status": cross_check.status,
        "severity": cross_check.severity,
        "notes": list(cross_check.notes),
    }


def _statistics_suggestion_to_json(suggestion: StatisticsSuggestion) -> dict[str, object]:
    return {
        "columns": list(suggestion.columns),
        "kinds": list(suggestion.kinds),
        "ddl": suggestion.ddl,
        "reason": suggestion.reason,
        "confidence": suggestion.confidence,
        "advisor_selectivity": suggestion.advisor_selectivity,
        "postgres_selectivity": suggestion.postgres_selectivity,
        "ratio": suggestion.ratio,
    }


def _local_selectivity_to_json(
    local_selectivity: LocalSelectivity | None,
) -> dict[str, object] | None:
    if local_selectivity is None:
        return None
    return {
        "s_global": local_selectivity.s_global,
        "s_local_p10": local_selectivity.s_local_p10,
        "s_local_median": local_selectivity.s_local_median,
        "rho": local_selectivity.rho,
        "confidence": local_selectivity.confidence,
        "sample_size": local_selectivity.sample_size,
        "passing_rows": local_selectivity.passing_rows,
        "resolution_floor": local_selectivity.resolution_floor,
        "notes": list(local_selectivity.notes),
    }


def _local_selectivity_cache_to_json(
    *,
    cache_dir: Path | None,
    cache_key: str | None,
    cache_file: Path | None,
    hit: bool,
    stored: bool,
    refresh: bool,
) -> dict[str, object]:
    return {
        "enabled": cache_dir is not None,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "key": cache_key,
        "path": str(cache_file) if cache_file is not None else None,
        "hit": hit,
        "stored": stored,
        "refresh": refresh,
    }


def _recommendation_to_json(
    recommendation: Recommendation,
    *,
    local_selectivity: LocalSelectivity | None,
    recall_target: float,
) -> dict[str, object]:
    ranked = [
        _ranked_candidate_to_json(plan, estimate)
        for plan, estimate in recommendation.ranked
    ]
    return {
        "verdict": recommendation.verdict,
        "s_global": recommendation.s_global,
        "s_local": recommendation.s_local,
        "rho": recommendation.rho,
        "planner_would_pick": (
            recommendation.planner_would_pick.value
            if recommendation.planner_would_pick is not None
            else None
        ),
        "decision": _recommendation_decision_to_json(
            recommendation,
            has_local_selectivity=local_selectivity is not None,
            recall_target=recall_target,
        ),
        "ranked": ranked,
    }


def _recommendation_decision_to_json(
    recommendation: Recommendation,
    *,
    has_local_selectivity: bool,
    recall_target: float,
) -> dict[str, object]:
    if not recommendation.ranked:
        return {
            "recommended_strategy": None,
            "planner_strategy": _planner_strategy_value(recommendation),
            "planner_mismatch": None,
            "selectivity_source": (
                "local_probe" if has_local_selectivity else "global_selectivity_fallback"
            ),
            "viable": False,
            "viable_candidates": 0,
            "rejected_candidates": 0,
            "why": ["no candidate plans were produced"],
        }

    best_plan, best_estimate = recommendation.ranked[0]
    best_viable = _candidate_is_viable(best_estimate, recall_target)
    viable_candidates = [
        (plan, estimate)
        for plan, estimate in recommendation.ranked
        if _candidate_is_viable(estimate, recall_target)
    ]
    runner_up = _runner_up_candidate(
        recommendation.ranked,
        recall_target=recall_target,
    )
    planner_strategy = _planner_strategy_value(recommendation)
    return {
        "recommended_strategy": best_plan.strategy.value,
        "planner_strategy": planner_strategy,
        "planner_mismatch": (
            None
            if planner_strategy is None
            else planner_strategy != best_plan.strategy.value
        ),
        "selectivity_source": (
            "local_probe" if has_local_selectivity else "global_selectivity_fallback"
        ),
        "viable": best_viable,
        "meets_recall_target": best_estimate.est_recall >= recall_target,
        "returns_k": best_estimate.est_returns_k,
        "viable_candidates": len(viable_candidates),
        "rejected_candidates": len(recommendation.ranked) - len(viable_candidates),
        "confidence": best_estimate.confidence,
        "confidence_level": _confidence_level(best_estimate.confidence),
        "estimated_latency_ms": _micros_to_ms(best_estimate.est_latency_us),
        "estimated_recall": best_estimate.est_recall,
        "uses_index": best_plan.uses_index,
        "requires_new_object": best_plan.requires_new_object,
        "runner_up": _runner_up_to_json(
            runner_up,
            best_estimate=best_estimate,
            recall_target=recall_target,
        ),
        "why": _decision_why(
            recommendation=recommendation,
            best_plan=best_plan,
            best_estimate=best_estimate,
            runner_up=runner_up,
            best_viable=best_viable,
            has_local_selectivity=has_local_selectivity,
            recall_target=recall_target,
        ),
    }


def _candidate_is_viable(estimate: CostEstimate, recall_target: float) -> bool:
    return estimate.est_returns_k and estimate.est_recall >= recall_target


def _runner_up_candidate(
    ranked: tuple[tuple[StrategyPlan, CostEstimate], ...],
    *,
    recall_target: float,
) -> tuple[StrategyPlan, CostEstimate, bool] | None:
    if len(ranked) <= 1:
        return None
    for plan, estimate in ranked[1:]:
        if _candidate_is_viable(estimate, recall_target):
            return plan, estimate, True
    plan, estimate = ranked[1]
    return plan, estimate, False


def _runner_up_to_json(
    runner_up: tuple[StrategyPlan, CostEstimate, bool] | None,
    *,
    best_estimate: CostEstimate,
    recall_target: float,
) -> dict[str, object] | None:
    if runner_up is None:
        return None
    plan, estimate, viable = runner_up
    return {
        "strategy": plan.strategy.value,
        "viable": viable,
        "meets_recall_target": estimate.est_recall >= recall_target,
        "returns_k": estimate.est_returns_k,
        "confidence": estimate.confidence,
        "estimated_latency_ms": _micros_to_ms(estimate.est_latency_us),
        "estimated_recall": estimate.est_recall,
        "latency_delta_ms": _micros_to_ms(
            estimate.est_latency_us - best_estimate.est_latency_us
        ),
        "latency_ratio": _latency_ratio(
            numerator_us=estimate.est_latency_us,
            denominator_us=best_estimate.est_latency_us,
        ),
    }


def _decision_why(
    *,
    recommendation: Recommendation,
    best_plan: StrategyPlan,
    best_estimate: CostEstimate,
    runner_up: tuple[StrategyPlan, CostEstimate, bool] | None,
    best_viable: bool,
    has_local_selectivity: bool,
    recall_target: float,
) -> list[str]:
    reasons: list[str] = []
    if best_viable:
        reasons.append(
            f"{best_plan.strategy.value} is the lowest estimated-latency candidate that "
            f"meets recall target {recall_target:.3g} and returns k"
        )
    else:
        reasons.append(
            f"{best_plan.strategy.value} is selected as the recall-safe fallback because "
            f"no candidate meets recall target {recall_target:.3g} and returns k"
        )

    if not has_local_selectivity:
        reasons.append(
            "Local selectivity was not measured; global selectivity was used as fallback"
        )

    if runner_up is None:
        reasons.append("no runner-up candidate is available")
    else:
        runner_plan, runner_estimate, runner_viable = runner_up
        ratio = _latency_ratio(
            numerator_us=runner_estimate.est_latency_us,
            denominator_us=best_estimate.est_latency_us,
        )
        if runner_viable and ratio is not None:
            reasons.append(
                f"estimated {ratio:.3g}x latency versus next viable strategy "
                f"{runner_plan.strategy.value}"
            )
        elif not runner_viable:
            reasons.append(
                f"next ranked alternative {runner_plan.strategy.value} fails the "
                "recall/returns-k viability gate"
            )

    if best_plan.requires_new_object is not None:
        reasons.append("recommended strategy requires creating the suggested database object")
    elif best_plan.uses_index is not None:
        reasons.append(f"recommended strategy uses index {best_plan.uses_index}")

    planner_strategy = _planner_strategy_value(recommendation)
    if planner_strategy is not None and planner_strategy != best_plan.strategy.value:
        reasons.append(
            f"PostgreSQL observed plan is {planner_strategy}, which differs from the "
            f"advisor recommendation {best_plan.strategy.value}"
        )

    if recommendation.rho <= -0.25:
        reasons.append(
            "local selectivity is lower than global selectivity; fixed-size ANN "
            "post-filtering has elevated recall risk"
        )
    elif recommendation.rho >= 0.25:
        reasons.append(
            "local selectivity is higher than global selectivity; ANN post-filtering is "
            "less risky for this query neighborhood"
        )

    if best_estimate.confidence < 0.45:
        reasons.append("winner confidence is low; validate with benchmark or production vectors")
    return reasons


def _planner_strategy_value(recommendation: Recommendation) -> str | None:
    if recommendation.planner_would_pick is None:
        return None
    return recommendation.planner_would_pick.value


def _confidence_level(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.45:
        return "medium"
    return "low"


def _micros_to_ms(value: float) -> float:
    return value / 1000.0


def _latency_ratio(*, numerator_us: float, denominator_us: float) -> float | None:
    if denominator_us <= 0:
        return None
    return numerator_us / denominator_us


def _observed_planner_to_json(
    observed_planner: ObservedPlannerChoice | None,
) -> dict[str, object] | None:
    if observed_planner is None:
        return None
    return {
        "strategy": observed_planner.strategy.value,
        "reason": observed_planner.reason,
        "root_node_type": observed_planner.root_node_type,
        "scan_node_type": observed_planner.scan_node_type,
        "index_name": observed_planner.index_name,
        "full_query_sql": observed_planner.full_query_sql,
    }


def _ranked_candidate_to_json(
    plan: StrategyPlan,
    estimate: CostEstimate,
) -> dict[str, object]:
    return {
        "strategy": plan.strategy.value,
        "plan": {
            "ef_search": plan.ef_search,
            "uses_index": plan.uses_index,
            "requires_new_object": plan.requires_new_object,
            "sql_hint": plan.sql_hint,
        },
        "estimate": {
            "est_latency_us": estimate.est_latency_us,
            "est_recall": estimate.est_recall,
            "est_returns_k": estimate.est_returns_k,
            "confidence": estimate.confidence,
            "notes": list(estimate.notes),
        },
    }


def _explain_notes(
    q_vector: Path | None,
    local_selectivity: LocalSelectivity | None,
    local_probe_warning: str | None = None,
) -> list[str]:
    if q_vector is None:
        return ["local selectivity probe skipped because --q-vector was not supplied"]
    if local_probe_warning is not None:
        return [
            local_probe_warning,
            "recommendation uses global selectivity as a low-confidence fallback",
        ]
    if local_selectivity is None:
        return ["local selectivity probe did not run"]
    return [
        "local selectivity estimated from the unfiltered top-m vector neighborhood",
        *local_selectivity.notes,
    ]


def _is_table_sample_vector_source(vector_source: str) -> bool:
    return vector_source.startswith("table_sample:")


def _downgrade_table_sample_local_selectivity(
    local_selectivity: LocalSelectivity,
) -> LocalSelectivity:
    return LocalSelectivity(
        s_global=local_selectivity.s_global,
        s_local_p10=local_selectivity.s_local_p10,
        s_local_median=local_selectivity.s_local_median,
        rho=local_selectivity.rho,
        confidence=(
            local_selectivity.confidence * TABLE_SAMPLE_VECTOR_CONFIDENCE_FACTOR
        ),
        sample_size=local_selectivity.sample_size,
        passing_rows=local_selectivity.passing_rows,
        resolution_floor=local_selectivity.resolution_floor,
        notes=local_selectivity.notes
        + (
            "query vectors came from a table sample, not production query logs; durable "
            "recommendation confidence is reduced",
        ),
    )


def _recommend_notes(
    vector_source: str,
    local_selectivity: LocalSelectivity | None,
    local_probe_warning: str | None = None,
) -> list[str]:
    if local_selectivity is None:
        if local_probe_warning is not None:
            return [
                f"representative query vector source: {vector_source}",
                local_probe_warning,
                "recommendation uses global selectivity as a low-confidence fallback",
                "verify that PostgreSQL can use an hnsw or ivfflat index for the probe",
            ]
        return [
            "representative query vectors were not supplied; recommendation uses global "
            "selectivity as a low-confidence fallback",
            "provide --q-vectors or --q-vector-sql for durable local-selectivity advice",
        ]
    notes = [
        f"representative query vector source: {vector_source}",
        "durable recommendation uses p10 local selectivity across the vector sample",
    ]
    if _is_table_sample_vector_source(vector_source):
        notes.append(
            "table-sampled vectors are an explicit low-confidence fallback; prefer "
            "--q-vectors or --q-vector-sql from production query logs"
        )
    notes.extend(local_selectivity.notes)
    return notes


def _local_probe_fallback_warning(exc: LocalProbeError) -> str:
    return f"local selectivity probe skipped: {exc}"
