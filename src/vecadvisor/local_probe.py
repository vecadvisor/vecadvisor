from __future__ import annotations

import importlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from psycopg import Connection

from .models import IndexMeta, LocalSelectivity, QuerySpec, TableStats
from .plan import normalize_readonly_query
from .query_spec import quote_identifier, quote_qualified_identifier
from .selectivity import rho_from_selectivities

DEFAULT_PROBE_ROWS = 200
DEFAULT_MAX_QUERY_VECTORS = 64
MIN_QUERY_VECTORS_FOR_HIGH_CONFIDENCE = 16
MIN_PASSING_ROWS_FOR_HIGH_CONFIDENCE = 5
SUPPORTED_DISTANCE_OPS = {"<->", "<#>", "<=>", "<+>", "<~>", "<%>"}


class QueryVectorError(ValueError):
    """Raised when a query-vector file cannot be parsed safely."""


class LocalProbeError(RuntimeError):
    """Raised when the local selectivity probe cannot run safely."""


def load_query_vector(path: Path, *, expected_dim: int | None = None) -> tuple[float, ...]:
    """Load a query vector from a JSON array or simple comma/space separated text file."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QueryVectorError(f"could not read query vector file: {path}") from exc
    return parse_query_vector_text(text, expected_dim=expected_dim)


def load_query_vectors(
    path: Path,
    *,
    expected_dim: int | None = None,
    max_vectors: int = DEFAULT_MAX_QUERY_VECTORS,
) -> tuple[tuple[float, ...], ...]:
    """Load a bounded representative query-vector sample from text/JSON/JSONL/NumPy."""

    _validate_max_vectors(max_vectors)
    if path.suffix.lower() == ".npy":
        return _load_query_vectors_npy(
            path,
            expected_dim=expected_dim,
            max_vectors=max_vectors,
        )

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QueryVectorError(f"could not read query vector file: {path}") from exc
    vectors = _parse_query_vectors_text(text, expected_dim=expected_dim)
    return _limit_vectors(vectors, max_vectors=max_vectors)


def load_query_vectors_from_sql(
    conn: Connection[Any],
    query_sql: str,
    *,
    expected_dim: int | None = None,
    max_vectors: int = DEFAULT_MAX_QUERY_VECTORS,
    statement_timeout_ms: int = 30_000,
) -> tuple[tuple[float, ...], ...]:
    """Read a bounded query-vector sample from one-column SELECT/WITH SQL."""

    _validate_max_vectors(max_vectors)
    readonly_sql = normalize_readonly_query(query_sql)
    sample_sql = f"SELECT vecadvisor_q.* FROM ({readonly_sql}) AS vecadvisor_q LIMIT %s"
    with conn.transaction():
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{statement_timeout_ms}ms",),
        )
        rows = conn.execute(sample_sql, (max_vectors,)).fetchall()

    vectors: list[tuple[float, ...]] = []
    for row in rows:
        values = tuple(row.values()) if isinstance(row, dict) else tuple(row)
        if len(values) != 1:
            raise QueryVectorError("query-vector SQL must return exactly one column")
        vectors.append(coerce_query_vector(values[0], expected_dim=expected_dim))
    if not vectors:
        raise QueryVectorError("query-vector SQL returned no rows")
    return tuple(vectors)


def load_query_vectors_from_table_sample(
    conn: Connection[Any],
    *,
    table: TableStats,
    vector_column: str,
    expected_dim: int | None = None,
    max_vectors: int = DEFAULT_MAX_QUERY_VECTORS,
    statement_timeout_ms: int = 30_000,
) -> tuple[tuple[float, ...], ...]:
    """Read a bounded low-confidence query-vector fallback from the target table."""

    _validate_max_vectors(max_vectors)
    try:
        table.column(vector_column)
    except KeyError as exc:
        raise QueryVectorError(
            f"unknown vector column for table-sample fallback: {vector_column}"
        ) from exc
    relation = quote_qualified_identifier(table.relname)
    vector_sql = quote_identifier(vector_column)
    sample_percent = _table_sample_percent(n_rows=table.n_rows, max_vectors=max_vectors)
    sample_sql = (
        f"SELECT {vector_sql}::text AS query_vector "
        f"FROM {relation} TABLESAMPLE SYSTEM ({sample_percent:.12g}) REPEATABLE (1) "
        f"WHERE {vector_sql} IS NOT NULL "
        "LIMIT %s"
    )
    fallback_sql = (
        f"SELECT {vector_sql}::text AS query_vector "
        f"FROM {relation} "
        f"WHERE {vector_sql} IS NOT NULL "
        "LIMIT %s"
    )
    with conn.transaction():
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{statement_timeout_ms}ms",),
        )
        rows = []
        if table.partitioned_by is None:
            rows = conn.execute(sample_sql, (max_vectors,)).fetchall()
        if not rows:
            rows = conn.execute(fallback_sql, (max_vectors,)).fetchall()

    vectors = tuple(
        coerce_query_vector(
            row["query_vector"] if isinstance(row, dict) else row[0],
            expected_dim=expected_dim,
        )
        for row in rows
    )
    if not vectors:
        raise QueryVectorError("table-sample query-vector fallback returned no rows")
    return vectors


def parse_query_vector_text(
    text: str,
    *,
    expected_dim: int | None = None,
) -> tuple[float, ...]:
    """Parse query-vector text into finite floats."""

    stripped = text.strip()
    if not stripped:
        raise QueryVectorError("query vector is empty")

    raw_values: Any
    if stripped.startswith("["):
        try:
            raw_values = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise QueryVectorError(f"query vector JSON is invalid: {exc}") from exc
        if not isinstance(raw_values, list):
            raise QueryVectorError("query vector JSON must be an array")
    else:
        raw_values = stripped.replace(",", " ").split()

    vector = tuple(_coerce_float(value) for value in raw_values)
    if not vector:
        raise QueryVectorError("query vector must contain at least one dimension")
    if expected_dim is not None and expected_dim > 0 and len(vector) != expected_dim:
        raise QueryVectorError(
            f"query vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
        )
    return vector


def coerce_query_vector(
    value: Any,
    *,
    expected_dim: int | None = None,
) -> tuple[float, ...]:
    """Coerce a SQL/file value into one finite query vector."""

    if isinstance(value, bytes):
        return parse_query_vector_text(value.decode("utf-8"), expected_dim=expected_dim)
    if isinstance(value, str):
        return parse_query_vector_text(value, expected_dim=expected_dim)
    if hasattr(value, "tolist"):
        return coerce_query_vector(value.tolist(), expected_dim=expected_dim)
    if isinstance(value, Sequence):
        vector = tuple(_coerce_float(item) for item in value)
        if not vector:
            raise QueryVectorError("query vector must contain at least one dimension")
        if expected_dim is not None and expected_dim > 0 and len(vector) != expected_dim:
            raise QueryVectorError(
                f"query vector dimension mismatch: expected {expected_dim}, got {len(vector)}"
            )
        return vector
    raise QueryVectorError(f"unsupported query vector value: {value!r}")


def vector_literal(vector: Sequence[float]) -> str:
    """Format a vector as a pgvector text literal."""

    if not vector:
        raise QueryVectorError("query vector must contain at least one dimension")
    return "[" + ",".join(format(_coerce_float(value), ".17g") for value in vector) + "]"


def run_local_selectivity_probes(
    conn: Connection[Any],
    *,
    table: TableStats,
    query: QuerySpec,
    filter_sql: str,
    query_vectors: Sequence[Sequence[float]],
    s_global: float,
    probe_rows: int = DEFAULT_PROBE_ROWS,
    statement_timeout_ms: int = 30_000,
    require_vector_index: bool = True,
) -> LocalSelectivity:
    """Probe multiple query neighborhoods and aggregate them for durable advice."""

    if not query_vectors:
        raise LocalProbeError("at least one representative query vector is required")
    probes = tuple(
        run_local_selectivity_probe(
            conn,
            table=table,
            query=query,
            filter_sql=filter_sql,
            query_vector=query_vector,
            s_global=s_global,
            probe_rows=probe_rows,
            statement_timeout_ms=statement_timeout_ms,
            require_vector_index=require_vector_index,
            validate_index_plan=query_index == 0,
        )
        for query_index, query_vector in enumerate(query_vectors)
    )
    return aggregate_local_selectivity(probes, s_global=s_global)


def run_local_selectivity_probe(
    conn: Connection[Any],
    *,
    table: TableStats,
    query: QuerySpec,
    filter_sql: str,
    query_vector: Sequence[float],
    s_global: float,
    probe_rows: int = DEFAULT_PROBE_ROWS,
    statement_timeout_ms: int = 30_000,
    require_vector_index: bool = True,
    validate_index_plan: bool = True,
) -> LocalSelectivity:
    """Estimate local selectivity from the unfiltered vector top-m neighborhood."""

    if probe_rows <= 0:
        raise LocalProbeError("probe_rows must be positive")
    ann_indexes = _ann_indexes(table, query.vector_column)
    if require_vector_index and not ann_indexes:
        raise LocalProbeError(
            f"local probe requires an hnsw or ivfflat index on {query.vector_column!r}"
        )

    sql = build_local_probe_sql(query=query, filter_sql=filter_sql)
    params = (vector_literal(query_vector), probe_rows)
    with conn.transaction():
        conn.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (f"{statement_timeout_ms}ms",),
        )
        conn.execute("SELECT set_config('enable_seqscan', 'off', true)")
        _apply_probe_search_gucs(
            conn,
            table=table,
            vector_column=query.vector_column,
            probe_rows=probe_rows,
            indexes=ann_indexes,
        )
        if (
            require_vector_index
            and validate_index_plan
            and not _probe_plan_uses_ann_index(conn, sql, params, indexes=ann_indexes)
        ):
            raise LocalProbeError(
                "local probe did not use an hnsw or ivfflat index; cannot measure "
                "pgvector's ANN frontier"
            )
        row = conn.execute(sql, params).fetchone()

    if row is None:
        raise LocalProbeError("local probe returned no rows")

    sample_size = int(row["sample_size"])
    passing_rows = int(row["passing_rows"])
    s_local = passing_rows / sample_size if sample_size else 0.0
    confidence = _probe_confidence(
        sample_size=sample_size,
        passing_rows=passing_rows,
        probe_rows=probe_rows,
    )
    resolution_floor = 1.0 / sample_size if sample_size else 1.0

    return LocalSelectivity(
        s_global=s_global,
        s_local_p10=s_local,
        s_local_median=s_local,
        rho=rho_from_selectivities(s_global, s_local),
        confidence=confidence,
        sample_size=sample_size,
        passing_rows=passing_rows,
        resolution_floor=resolution_floor,
        notes=_probe_notes(
            sample_size=sample_size,
            passing_rows=passing_rows,
            probe_rows=probe_rows,
            resolution_floor=resolution_floor,
        ),
    )


def aggregate_local_selectivity(
    probes: Sequence[LocalSelectivity],
    *,
    s_global: float,
    high_confidence_vectors: int = MIN_QUERY_VECTORS_FOR_HIGH_CONFIDENCE,
) -> LocalSelectivity:
    """Aggregate per-vector local selectivity using p10 for conservative costing."""

    if not probes:
        raise LocalProbeError("cannot aggregate an empty local-selectivity sample")
    if high_confidence_vectors <= 0:
        raise LocalProbeError("high_confidence_vectors must be positive")

    local_values = tuple(probe.s_local_median for probe in probes)
    s_local_p10 = _percentile(local_values, 0.10)
    s_local_median = _percentile(local_values, 0.50)
    confidence_values = tuple(probe.confidence for probe in probes)
    sample_coverage = min(1.0, len(probes) / high_confidence_vectors)
    confidence = sample_coverage * _percentile(confidence_values, 0.50)
    sample_size = sum(probe.sample_size for probe in probes)
    passing_rows = sum(probe.passing_rows for probe in probes)
    resolution_floor = max((probe.resolution_floor for probe in probes), default=1.0)

    return LocalSelectivity(
        s_global=s_global,
        s_local_p10=s_local_p10,
        s_local_median=s_local_median,
        rho=rho_from_selectivities(s_global, s_local_p10),
        confidence=confidence,
        sample_size=sample_size,
        passing_rows=passing_rows,
        resolution_floor=resolution_floor,
        notes=_aggregate_notes(
            probes=probes,
            s_local_p10=s_local_p10,
            resolution_floor=resolution_floor,
            high_confidence_vectors=high_confidence_vectors,
        ),
    )


def build_local_probe_sql(*, query: QuerySpec, filter_sql: str) -> str:
    """Build the bounded top-m local-neighborhood probe query."""

    if query.distance_op not in SUPPORTED_DISTANCE_OPS:
        raise LocalProbeError(f"unsupported vector distance operator: {query.distance_op}")

    projection = _probe_projection(query)
    relation = quote_qualified_identifier(query.relname)
    vector_column = quote_identifier(query.vector_column)
    return (
        "WITH nearest AS MATERIALIZED ("
        f"SELECT {projection} "
        f"FROM {relation} "
        f"ORDER BY {vector_column} {query.distance_op} %s::vector "
        "LIMIT %s"
        ") "
        "SELECT count(*)::int AS sample_size, "
        f"count(*) FILTER (WHERE {filter_sql.strip()})::int AS passing_rows "
        "FROM nearest"
    )


def _probe_projection(query: QuerySpec) -> str:
    columns: list[str] = []
    seen: set[str] = set()
    for predicate in query.predicates:
        if predicate.column in seen:
            continue
        seen.add(predicate.column)
        columns.append(quote_identifier(predicate.column))
    if not columns:
        raise LocalProbeError("local probe requires at least one filter column")
    return ", ".join(columns)


def _has_vector_index(table: TableStats, vector_column: str) -> bool:
    return bool(_ann_indexes(table, vector_column))


def _ann_indexes(table: TableStats, vector_column: str) -> tuple[IndexMeta, ...]:
    return tuple(
        index
        for index in table.indexes
        if index.method in {"hnsw", "ivfflat"} and vector_column in index.columns
    )


def _apply_probe_search_gucs(
    conn: Connection[Any],
    *,
    table: TableStats,
    vector_column: str,
    probe_rows: int,
    indexes: Sequence[IndexMeta],
) -> None:
    methods = {index.method for index in indexes}
    if "hnsw" in methods:
        _set_int_guc_at_least(conn, "hnsw.ef_search", probe_rows)
    if "ivfflat" in methods:
        _set_int_guc_at_least(
            conn,
            "ivfflat.probes",
            _ivfflat_probe_target(
                table=table,
                vector_column=vector_column,
                probe_rows=probe_rows,
                indexes=indexes,
            ),
        )


def _set_int_guc_at_least(
    conn: Connection[Any],
    setting_name: str,
    minimum_value: int,
) -> None:
    current = _current_int_setting(conn, setting_name)
    target = max(int(minimum_value), current or 0, 1)
    conn.execute("SELECT set_config(%s, %s, true)", (setting_name, str(target)))


def _current_int_setting(conn: Connection[Any], setting_name: str) -> int | None:
    row = conn.execute("SELECT current_setting(%s, true)", (setting_name,)).fetchone()
    if row is None:
        return None
    raw_value = _first_column(row)
    if raw_value in (None, ""):
        return None
    try:
        return int(str(raw_value))
    except ValueError:
        return None


def _ivfflat_probe_target(
    *,
    table: TableStats,
    vector_column: str,
    probe_rows: int,
    indexes: Sequence[IndexMeta] | None = None,
) -> int:
    ivfflat_indexes = tuple(
        index
        for index in (indexes if indexes is not None else _ann_indexes(table, vector_column))
        if index.method == "ivfflat"
    )
    list_counts = tuple(
        int(index.lists)
        for index in ivfflat_indexes
        if index.lists is not None and index.lists > 0
    )
    if not list_counts:
        return max(1, math.ceil(math.sqrt(probe_rows)))

    lists = max(list_counts)
    rows_per_list = max(table.n_rows / lists, 1.0)
    proportional = math.ceil(probe_rows / rows_per_list)
    floor = 2 if lists > 1 and probe_rows > 1 else 1
    return min(lists, max(floor, proportional, 1))


def _probe_plan_uses_ann_index(
    conn: Connection[Any],
    sql: str,
    params: tuple[Any, ...],
    *,
    indexes: Sequence[IndexMeta],
) -> bool:
    index_names = {index.name for index in indexes}
    if not index_names:
        return False
    row = conn.execute(f"EXPLAIN (FORMAT JSON) {sql}", params).fetchone()
    if row is None:
        return False
    return _plan_payload_uses_index(_first_column(row), index_names)


def _plan_payload_uses_index(payload: Any, index_names: set[str]) -> bool:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return False
    if isinstance(payload, list):
        return any(_plan_payload_uses_index(item, index_names) for item in payload)
    if not isinstance(payload, dict):
        return False
    if payload.get("Index Name") in index_names:
        return True
    plan = payload.get("Plan")
    if isinstance(plan, dict) and _plan_payload_uses_index(plan, index_names):
        return True
    plans = payload.get("Plans")
    if isinstance(plans, list):
        return any(_plan_payload_uses_index(child, index_names) for child in plans)
    return False


def _first_column(row: Any) -> Any:
    if isinstance(row, dict):
        return next(iter(row.values()), None)
    return row[0]


def _probe_confidence(*, sample_size: int, passing_rows: int, probe_rows: int) -> float:
    if sample_size <= 0 or probe_rows <= 0:
        return 0.0
    coverage = min(1.0, sample_size / min(probe_rows, DEFAULT_PROBE_ROWS))
    passing_evidence = min(1.0, passing_rows / MIN_PASSING_ROWS_FOR_HIGH_CONFIDENCE)
    return coverage * passing_evidence


def _probe_notes(
    *,
    sample_size: int,
    passing_rows: int,
    probe_rows: int,
    resolution_floor: float,
) -> tuple[str, ...]:
    notes: list[str] = []
    if sample_size <= 0:
        return ("local probe returned no candidate rows",)
    if sample_size < probe_rows:
        notes.append("local probe returned fewer rows than requested")
    if passing_rows == 0:
        notes.append(
            "no filter-passing rows in local sample; s_local is below probe resolution floor "
            f"~{resolution_floor:.6g}"
        )
    elif passing_rows < MIN_PASSING_ROWS_FOR_HIGH_CONFIDENCE:
        notes.append(
            f"only {passing_rows} filter-passing rows in local sample; local selectivity "
            "confidence is reduced"
        )
    return tuple(notes)


def _parse_query_vectors_text(
    text: str,
    *,
    expected_dim: int | None,
) -> tuple[tuple[float, ...], ...]:
    stripped = text.strip()
    if not stripped:
        raise QueryVectorError("query vector file is empty")
    if stripped.startswith(("[", "{")):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return _parse_query_vector_lines(stripped, expected_dim=expected_dim)
        return _vectors_from_json_payload(payload, expected_dim=expected_dim)
    return _parse_query_vector_lines(stripped, expected_dim=expected_dim)


def _parse_query_vector_lines(
    text: str,
    *,
    expected_dim: int | None,
) -> tuple[tuple[float, ...], ...]:
    lines = tuple(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not lines:
        raise QueryVectorError("query vector file is empty")
    return tuple(parse_query_vector_text(line, expected_dim=expected_dim) for line in lines)


def _vectors_from_json_payload(
    payload: Any,
    *,
    expected_dim: int | None,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(payload, dict):
        if "vectors" not in payload:
            raise QueryVectorError("query vector JSON object must contain a 'vectors' array")
        payload = payload["vectors"]
    if _looks_like_one_vector(payload):
        return (coerce_query_vector(payload, expected_dim=expected_dim),)
    if not isinstance(payload, list):
        raise QueryVectorError("query vector JSON must be an array or {'vectors': [...]}")
    vectors = tuple(coerce_query_vector(row, expected_dim=expected_dim) for row in payload)
    if not vectors:
        raise QueryVectorError("query vector JSON contains no vectors")
    return vectors


def _looks_like_one_vector(payload: Any) -> bool:
    return isinstance(payload, list) and all(
        not isinstance(item, (dict, list, tuple)) and not hasattr(item, "tolist")
        for item in payload
    )


def _load_query_vectors_npy(
    path: Path,
    *,
    expected_dim: int | None,
    max_vectors: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        np = importlib.import_module("numpy")
    except ImportError as exc:
        raise QueryVectorError("loading .npy query-vector files requires numpy") from exc
    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r")
    except OSError as exc:
        raise QueryVectorError(f"could not read query vector file: {path}") from exc
    ndim = getattr(array, "ndim", None)
    if ndim == 1:
        return (coerce_query_vector(array, expected_dim=expected_dim),)
    if ndim != 2:
        raise QueryVectorError(".npy query-vector array must be one- or two-dimensional")
    count = min(int(array.shape[0]), max_vectors)
    vectors = tuple(
        coerce_query_vector(array[row_index], expected_dim=expected_dim)
        for row_index in range(count)
    )
    if not vectors:
        raise QueryVectorError(".npy query-vector array contains no vectors")
    return vectors


def _limit_vectors(
    vectors: tuple[tuple[float, ...], ...],
    *,
    max_vectors: int,
) -> tuple[tuple[float, ...], ...]:
    _validate_max_vectors(max_vectors)
    limited = vectors[:max_vectors]
    if not limited:
        raise QueryVectorError("query vector sample contains no vectors")
    return limited


def _validate_max_vectors(max_vectors: int) -> None:
    if max_vectors <= 0:
        raise QueryVectorError("max_vectors must be positive")


def _table_sample_percent(*, n_rows: int, max_vectors: int) -> float:
    if n_rows <= 0 or n_rows <= max_vectors * 100:
        return 100.0
    target_rows = max_vectors * 20
    return min(100.0, max(0.01, target_rows * 100.0 / n_rows))


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise LocalProbeError("cannot compute percentile of empty values")
    if not 0.0 <= percentile <= 1.0:
        raise LocalProbeError("percentile must be in [0, 1]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def _aggregate_notes(
    *,
    probes: Sequence[LocalSelectivity],
    s_local_p10: float,
    resolution_floor: float,
    high_confidence_vectors: int,
) -> tuple[str, ...]:
    notes = [
        (
            f"local selectivity aggregated from {len(probes)} representative query vectors; "
            "costing uses p10 local selectivity"
        )
    ]
    if len(probes) < high_confidence_vectors:
        notes.append(
            f"fewer than {high_confidence_vectors} representative query vectors supplied; "
            "durable recommendation confidence is reduced"
        )
    zero_pass_probes = sum(1 for probe in probes if probe.passing_rows == 0)
    if zero_pass_probes:
        notes.append(
            f"{zero_pass_probes} query-vector probes had no filter-passing rows in the "
            "local neighborhood"
        )
    if s_local_p10 <= resolution_floor:
        notes.append(
            "p10 local selectivity is at or below the probe resolution floor; increase "
            "probe rows for sharper advice"
        )
    return tuple(notes)


def _coerce_float(value: Any) -> float:
    if isinstance(value, bool):
        raise QueryVectorError("query vector dimensions must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise QueryVectorError(f"invalid query vector dimension: {value!r}") from exc
    if not math.isfinite(number):
        raise QueryVectorError("query vector dimensions must be finite")
    return number
