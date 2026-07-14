from __future__ import annotations

import importlib
import time
from collections.abc import Sequence
from typing import Any

from psycopg import Connection

from ..local_probe import vector_literal
from ..pgversion import load_pg_capabilities, pg_capabilities_to_json
from ..query_spec import quote_identifier
from .datasets import SyntheticDataset, SyntheticQueries
from .groundtruth import exact_topk
from .runner import (
    STRATEGY_EXACT,
    STRATEGY_ITERATIVE,
    STRATEGY_PARTIAL,
    STRATEGY_PARTITION,
    STRATEGY_POSTFILTER,
    BenchmarkReport,
    strategy_metrics_from_indices,
)

DB_STRATEGIES = (
    STRATEGY_EXACT,
    STRATEGY_POSTFILTER,
    STRATEGY_ITERATIVE,
    STRATEGY_PARTIAL,
    STRATEGY_PARTITION,
)
ITERATIVE_ORDERS = {"relaxed_order", "strict_order"}
METRIC_SQL = {
    "l2": ("<->", "vector_l2_ops"),
    "ip": ("<#>", "vector_ip_ops"),
    "cosine": ("<=>", "vector_cosine_ops"),
}


def parse_db_strategy_list(strategies: str | None) -> tuple[str, ...]:
    if strategies is None or not strategies.strip() or strategies.strip().lower() == "all":
        return DB_STRATEGIES
    parsed = tuple(item.strip().lower() for item in strategies.split(",") if item.strip())
    if not parsed:
        raise ValueError("at least one DB benchmark strategy is required")
    unknown = sorted(set(parsed) - set(DB_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown DB benchmark strategy: {', '.join(unknown)}")
    return parsed


def run_postgres_synthetic_benchmark(
    conn: Connection[Any],
    *,
    dataset: SyntheticDataset,
    queries: SyntheticQueries,
    k: int,
    metric: str = "l2",
    strategies: tuple[str, ...] = DB_STRATEGIES,
    ef_search: int = 40,
    max_scan_tuples: int = 1_000,
    iterative_order: str = "relaxed_order",
    hnsw_m: int = 8,
    hnsw_ef_construction: int = 32,
    block_rows: int | None = None,
    statement_timeout_ms: int = 30_000,
) -> BenchmarkReport:
    """Run exact and HNSW post-filter strategies against PostgreSQL/pgvector."""

    if k <= 0:
        raise ValueError("k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if max_scan_tuples <= 0:
        raise ValueError("max_scan_tuples must be positive")
    if iterative_order not in ITERATIVE_ORDERS:
        raise ValueError(
            f"iterative_order must be one of: {', '.join(sorted(ITERATIVE_ORDERS))}"
        )
    if hnsw_m <= 0:
        raise ValueError("hnsw_m must be positive")
    if hnsw_ef_construction <= 0:
        raise ValueError("hnsw_ef_construction must be positive")
    if metric not in METRIC_SQL:
        raise ValueError(f"metric must be one of: {', '.join(sorted(METRIC_SQL))}")
    unknown = sorted(set(strategies) - set(DB_STRATEGIES))
    if unknown:
        raise ValueError(f"unknown DB benchmark strategy: {', '.join(unknown)}")

    started = time.perf_counter()
    truth = exact_topk(
        dataset.vectors,
        queries.vectors,
        k=k,
        metric=metric,
        filter_mask=dataset.filter_mask,
        block_rows=block_rows,
    )
    table_name = _temp_table_name(conn)
    distance_op, opclass = METRIC_SQL[metric]
    _load_synthetic_table(
        conn,
        table_name=table_name,
        dataset=dataset,
        opclass=opclass,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        statement_timeout_ms=statement_timeout_ms,
    )
    capabilities = load_pg_capabilities(conn)
    if STRATEGY_ITERATIVE in strategies and not capabilities.supports_hnsw_iterative_scan:
        raise ValueError("DB iterative benchmark requires pgvector >= 0.8.0")

    metrics = []
    if STRATEGY_EXACT in strategies:
        exact_indices, exact_latencies = _run_exact_sql(
            conn,
            table_name=table_name,
            queries=queries.vectors,
            k=k,
            distance_op=distance_op,
            statement_timeout_ms=statement_timeout_ms,
        )
        metrics.append(
            strategy_metrics_from_indices(
                strategy=STRATEGY_EXACT,
                params={"mode": "postgres_filtered_exact"},
                truth_indices=truth.indices,
                candidate_indices=exact_indices,
                latencies_ms=exact_latencies,
                k=k,
                notes=("actual PostgreSQL filtered exact SQL; vector index disabled",),
            )
        )
    if STRATEGY_POSTFILTER in strategies:
        postfilter_indices, postfilter_latencies = _run_postfilter_sql(
            conn,
            table_name=table_name,
            queries=queries.vectors,
            k=k,
            distance_op=distance_op,
            ef_search=ef_search,
            statement_timeout_ms=statement_timeout_ms,
        )
        metrics.append(
            strategy_metrics_from_indices(
                strategy=STRATEGY_POSTFILTER,
                params={
                    "mode": "postgres_hnsw_postfilter",
                    "ef_search": ef_search,
                    "hnsw_m": hnsw_m,
                    "hnsw_ef_construction": hnsw_ef_construction,
                },
                truth_indices=truth.indices,
                candidate_indices=postfilter_indices,
                latencies_ms=postfilter_latencies,
                k=k,
                notes=("actual PostgreSQL/pgvector HNSW SQL with scalar post-filter",),
            )
        )
    if STRATEGY_ITERATIVE in strategies:
        iterative_indices, iterative_latencies = _run_iterative_sql(
            conn,
            table_name=table_name,
            queries=queries.vectors,
            k=k,
            distance_op=distance_op,
            ef_search=ef_search,
            max_scan_tuples=max_scan_tuples,
            iterative_order=iterative_order,
            statement_timeout_ms=statement_timeout_ms,
        )
        metrics.append(
            strategy_metrics_from_indices(
                strategy=STRATEGY_ITERATIVE,
                params={
                    "mode": "postgres_hnsw_iterative",
                    "ef_search": ef_search,
                    "max_scan_tuples": max_scan_tuples,
                    "iterative_order": iterative_order,
                    "hnsw_m": hnsw_m,
                    "hnsw_ef_construction": hnsw_ef_construction,
                },
                truth_indices=truth.indices,
                candidate_indices=iterative_indices,
                latencies_ms=iterative_latencies,
                k=k,
                notes=("actual PostgreSQL/pgvector HNSW iterative scan SQL",),
            )
        )
    if STRATEGY_PARTIAL in strategies:
        _create_partial_hnsw_index(
            conn,
            table_name=table_name,
            opclass=opclass,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            statement_timeout_ms=statement_timeout_ms,
        )
        partial_indices, partial_latencies = _run_partial_sql(
            conn,
            table_name=table_name,
            queries=queries.vectors,
            k=k,
            distance_op=distance_op,
            ef_search=ef_search,
            statement_timeout_ms=statement_timeout_ms,
        )
        metrics.append(
            strategy_metrics_from_indices(
                strategy=STRATEGY_PARTIAL,
                params={
                    "mode": "postgres_hnsw_partial_index",
                    "ef_search": ef_search,
                    "hnsw_m": hnsw_m,
                    "hnsw_ef_construction": hnsw_ef_construction,
                    "predicate": "passes_filter",
                },
                truth_indices=truth.indices,
                candidate_indices=partial_indices,
                latencies_ms=partial_latencies,
                k=k,
                notes=("actual PostgreSQL/pgvector partial HNSW index SQL",),
            )
        )
    if STRATEGY_PARTITION in strategies:
        partition_table_name = f"{table_name}_part"
        _load_partitioned_synthetic_table(
            conn,
            table_name=partition_table_name,
            dataset=dataset,
            opclass=opclass,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            statement_timeout_ms=statement_timeout_ms,
        )
        partition_indices, partition_latencies = _run_partition_sql(
            conn,
            table_name=partition_table_name,
            queries=queries.vectors,
            k=k,
            distance_op=distance_op,
            ef_search=ef_search,
            statement_timeout_ms=statement_timeout_ms,
        )
        metrics.append(
            strategy_metrics_from_indices(
                strategy=STRATEGY_PARTITION,
                params={
                    "mode": "postgres_hnsw_partition_pruned",
                    "ef_search": ef_search,
                    "hnsw_m": hnsw_m,
                    "hnsw_ef_construction": hnsw_ef_construction,
                    "partition_key": "passes_filter",
                },
                truth_indices=truth.indices,
                candidate_indices=partition_indices,
                latencies_ms=partition_latencies,
                k=k,
                notes=("actual PostgreSQL partition-pruned pgvector HNSW SQL",),
            )
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return BenchmarkReport(
        dataset={
            "id": f"postgres-{dataset.dataset_id}",
            "rows": dataset.n_rows,
            "dim": dataset.dim,
            "queries": queries.n_queries,
            "clusters": len(dataset.filter_probabilities),
            "query_policy": queries.cluster_policy,
            "target_filter_selectivity": dataset.filter_selectivity,
            "observed_filter_selectivity": dataset.observed_selectivity,
            "correlation": dataset.correlation,
            "dataset_seed": dataset.seed,
            "query_seed": queries.seed,
            "temp_table": table_name,
            "postgres": pg_capabilities_to_json(capabilities),
        },
        ground_truth={
            "metric": truth.metric,
            "k": truth.k,
            "candidate_count": truth.candidate_count,
            "block_rows": truth.block_rows,
            "blocks_scanned": truth.blocks_scanned,
            "first_query_indices": [
                int(index) for index in truth.indices[0].tolist() if int(index) >= 0
            ],
        },
        strategies=tuple(metrics),
        elapsed_ms=elapsed_ms,
        notes=(
            "benchmark measured actual PostgreSQL/pgvector SQL execution",
            "ground truth still comes from blocked exact in-process computation",
            "iterative requires pgvector >= 0.8.0 when selected",
            "partial strategy builds a session-local partial HNSW index",
            "partition strategy builds a session-local list-partitioned table",
            "temporary benchmark table is session-local and discarded on disconnect",
        ),
    )


def _load_synthetic_table(
    conn: Connection[Any],
    *,
    table_name: str,
    dataset: SyntheticDataset,
    opclass: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    statement_timeout_ms: int,
) -> None:
    table_sql = quote_identifier(table_name)
    index_sql = quote_identifier(f"{table_name}_embedding_hnsw_idx")
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{statement_timeout_ms}ms",),
        )
        conn.execute(f"DROP TABLE IF EXISTS {table_sql}")
        conn.execute(
            f"CREATE TEMP TABLE {table_sql} ("
            "id integer PRIMARY KEY, "
            "passes_filter boolean NOT NULL, "
            f"embedding vector({dataset.dim}) NOT NULL"
            ") ON COMMIT PRESERVE ROWS"
        )
        _copy_dataset_rows(conn, table_sql=table_sql, dataset=dataset)
        conn.execute(
            f"CREATE INDEX {index_sql} ON {table_sql} "
            f"USING hnsw (embedding {opclass}) "
            f"WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})"
        )
        conn.execute(f"ANALYZE {table_sql}")


def _create_partial_hnsw_index(
    conn: Connection[Any],
    *,
    table_name: str,
    opclass: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    statement_timeout_ms: int,
) -> None:
    table_sql = quote_identifier(table_name)
    index_sql = quote_identifier(f"{table_name}_embedding_partial_hnsw_idx")
    with conn.transaction():
        _set_common_timeout(conn, statement_timeout_ms)
        conn.execute(f"DROP INDEX IF EXISTS {index_sql}")
        conn.execute(
            f"CREATE INDEX {index_sql} ON {table_sql} "
            f"USING hnsw (embedding {opclass}) "
            f"WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction}) "
            "WHERE passes_filter"
        )
        conn.execute(f"ANALYZE {table_sql}")


def _load_partitioned_synthetic_table(
    conn: Connection[Any],
    *,
    table_name: str,
    dataset: SyntheticDataset,
    opclass: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    statement_timeout_ms: int,
) -> None:
    table_sql = quote_identifier(table_name)
    true_table_sql = quote_identifier(f"{table_name}_true")
    false_table_sql = quote_identifier(f"{table_name}_false")
    true_index_sql = quote_identifier(f"{table_name}_true_embedding_hnsw_idx")
    with conn.transaction():
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        _set_common_timeout(conn, statement_timeout_ms)
        conn.execute(f"DROP TABLE IF EXISTS {table_sql}")
        conn.execute(
            f"CREATE TEMP TABLE {table_sql} ("
            "id integer NOT NULL, "
            "passes_filter boolean NOT NULL, "
            f"embedding vector({dataset.dim}) NOT NULL"
            ") PARTITION BY LIST (passes_filter)"
        )
        conn.execute(
            f"CREATE TEMP TABLE {true_table_sql} "
            f"PARTITION OF {table_sql} FOR VALUES IN (true)"
        )
        conn.execute(
            f"CREATE TEMP TABLE {false_table_sql} "
            f"PARTITION OF {table_sql} FOR VALUES IN (false)"
        )
        _copy_dataset_rows(conn, table_sql=table_sql, dataset=dataset)
        conn.execute(
            f"CREATE INDEX {true_index_sql} ON {true_table_sql} "
            f"USING hnsw (embedding {opclass}) "
            f"WITH (m = {hnsw_m}, ef_construction = {hnsw_ef_construction})"
        )
        conn.execute(f"ANALYZE {table_sql}")


def _run_exact_sql(
    conn: Connection[Any],
    *,
    table_name: str,
    queries: Any,
    k: int,
    distance_op: str,
    statement_timeout_ms: int,
) -> tuple[Any, tuple[float, ...]]:
    np = _numpy()
    table_sql = quote_identifier(table_name)
    result_ids = np.full((int(queries.shape[0]), k), -1, dtype="int64")
    latencies_ms: list[float] = []
    sql = (
        f"SELECT id FROM {table_sql} "
        "WHERE passes_filter "
        f"ORDER BY embedding {distance_op} %s::vector "
        "LIMIT %s"
    )
    with conn.transaction():
        _set_common_timeout(conn, statement_timeout_ms)
        conn.execute("SELECT set_config('enable_indexscan', 'off', true)")
        conn.execute("SELECT set_config('enable_bitmapscan', 'off', true)")
        for query_index in range(int(queries.shape[0])):
            started = time.perf_counter()
            rows = conn.execute(sql, (_pgvector_literal(queries[query_index]), k)).fetchall()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            _store_ids(result_ids, query_index, _row_ids(rows))
    return result_ids, tuple(latencies_ms)


def _copy_dataset_rows(
    conn: Connection[Any],
    *,
    table_sql: str,
    dataset: SyntheticDataset,
) -> None:
    copy_sql = f"COPY {table_sql} (id, passes_filter, embedding) FROM STDIN"
    with conn.cursor() as cursor:
        with cursor.copy(copy_sql) as copy:
            for row_id in range(dataset.n_rows):
                copy.write_row(
                    (
                        row_id,
                        bool(dataset.filter_mask[row_id]),
                        _pgvector_literal(dataset.vectors[row_id]),
                    )
                )


def _run_postfilter_sql(
    conn: Connection[Any],
    *,
    table_name: str,
    queries: Any,
    k: int,
    distance_op: str,
    ef_search: int,
    statement_timeout_ms: int,
) -> tuple[Any, tuple[float, ...]]:
    np = _numpy()
    table_sql = quote_identifier(table_name)
    result_ids = np.full((int(queries.shape[0]), k), -1, dtype="int64")
    latencies_ms: list[float] = []
    sql = (
        f"SELECT id FROM {table_sql} "
        "WHERE passes_filter "
        f"ORDER BY embedding {distance_op} %s::vector "
        "LIMIT %s"
    )
    with conn.transaction():
        _set_common_timeout(conn, statement_timeout_ms)
        conn.execute("SELECT set_config('enable_seqscan', 'off', true)")
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        conn.execute("SELECT set_config('hnsw.iterative_scan', 'off', true)")
        for query_index in range(int(queries.shape[0])):
            started = time.perf_counter()
            rows = conn.execute(sql, (_pgvector_literal(queries[query_index]), k)).fetchall()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            _store_ids(result_ids, query_index, _row_ids(rows))
    return result_ids, tuple(latencies_ms)


def _run_partial_sql(
    conn: Connection[Any],
    *,
    table_name: str,
    queries: Any,
    k: int,
    distance_op: str,
    ef_search: int,
    statement_timeout_ms: int,
) -> tuple[Any, tuple[float, ...]]:
    return _run_filtered_hnsw_sql(
        conn,
        table_name=table_name,
        queries=queries,
        k=k,
        distance_op=distance_op,
        ef_search=ef_search,
        statement_timeout_ms=statement_timeout_ms,
    )


def _run_partition_sql(
    conn: Connection[Any],
    *,
    table_name: str,
    queries: Any,
    k: int,
    distance_op: str,
    ef_search: int,
    statement_timeout_ms: int,
) -> tuple[Any, tuple[float, ...]]:
    return _run_filtered_hnsw_sql(
        conn,
        table_name=table_name,
        queries=queries,
        k=k,
        distance_op=distance_op,
        ef_search=ef_search,
        statement_timeout_ms=statement_timeout_ms,
    )


def _run_filtered_hnsw_sql(
    conn: Connection[Any],
    *,
    table_name: str,
    queries: Any,
    k: int,
    distance_op: str,
    ef_search: int,
    statement_timeout_ms: int,
) -> tuple[Any, tuple[float, ...]]:
    np = _numpy()
    table_sql = quote_identifier(table_name)
    result_ids = np.full((int(queries.shape[0]), k), -1, dtype="int64")
    latencies_ms: list[float] = []
    sql = (
        f"SELECT id FROM {table_sql} "
        "WHERE passes_filter "
        f"ORDER BY embedding {distance_op} %s::vector "
        "LIMIT %s"
    )
    with conn.transaction():
        _set_common_timeout(conn, statement_timeout_ms)
        conn.execute("SELECT set_config('enable_seqscan', 'off', true)")
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        conn.execute("SELECT set_config('hnsw.iterative_scan', 'off', true)")
        for query_index in range(int(queries.shape[0])):
            started = time.perf_counter()
            rows = conn.execute(sql, (_pgvector_literal(queries[query_index]), k)).fetchall()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            _store_ids(result_ids, query_index, _row_ids(rows))
    return result_ids, tuple(latencies_ms)


def _run_iterative_sql(
    conn: Connection[Any],
    *,
    table_name: str,
    queries: Any,
    k: int,
    distance_op: str,
    ef_search: int,
    max_scan_tuples: int,
    iterative_order: str,
    statement_timeout_ms: int,
) -> tuple[Any, tuple[float, ...]]:
    np = _numpy()
    table_sql = quote_identifier(table_name)
    result_ids = np.full((int(queries.shape[0]), k), -1, dtype="int64")
    latencies_ms: list[float] = []
    sql = (
        f"SELECT id FROM {table_sql} "
        "WHERE passes_filter "
        f"ORDER BY embedding {distance_op} %s::vector "
        "LIMIT %s"
    )
    with conn.transaction():
        _set_common_timeout(conn, statement_timeout_ms)
        conn.execute("SELECT set_config('enable_seqscan', 'off', true)")
        conn.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(ef_search),))
        conn.execute("SELECT set_config('hnsw.iterative_scan', %s, true)", (iterative_order,))
        conn.execute(
            "SELECT set_config('hnsw.max_scan_tuples', %s, true)",
            (str(max_scan_tuples),),
        )
        for query_index in range(int(queries.shape[0])):
            started = time.perf_counter()
            rows = conn.execute(sql, (_pgvector_literal(queries[query_index]), k)).fetchall()
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            _store_ids(result_ids, query_index, _row_ids(rows))
    return result_ids, tuple(latencies_ms)


def _set_common_timeout(conn: Connection[Any], statement_timeout_ms: int) -> None:
    conn.execute(
        "SELECT set_config('statement_timeout', %s, true)",
        (f"{statement_timeout_ms}ms",),
    )


def _pgvector_literal(vector: Any) -> str:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return vector_literal(vector)


def _store_ids(result_ids: Any, query_index: int, ids: Sequence[int]) -> None:
    for offset, row_id in enumerate(ids[: int(result_ids.shape[1])]):
        result_ids[query_index, offset] = row_id


def _row_ids(rows: Sequence[Any]) -> tuple[int, ...]:
    ids: list[int] = []
    for row in rows:
        if isinstance(row, dict):
            ids.append(int(row["id"]))
        else:
            ids.append(int(row[0]))
    return tuple(ids)


def _temp_table_name(conn: Connection[Any]) -> str:
    row = conn.execute("SELECT pg_backend_pid() AS pid").fetchone()
    if row is None:
        raise RuntimeError("could not read PostgreSQL backend pid")
    pid = int(row["pid"] if isinstance(row, dict) else row[0])
    return f"vecadvisor_db_bench_{pid}"


def _numpy() -> Any:
    return importlib.import_module("numpy")
