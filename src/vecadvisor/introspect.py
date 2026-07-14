from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

from .models import ColumnStats, ExtendedStatsMeta, IndexMeta, TableStats


class IntrospectionError(RuntimeError):
    """Raised when catalog introspection cannot resolve the requested object."""


def introspect_table(
    conn: Connection[Any],
    relation: str,
    *,
    vector_column: str | None = None,
) -> TableStats:
    """Read PostgreSQL catalog/statistics metadata for a table.

    The function performs catalog reads only. It does not run ANALYZE, create
    indexes, or modify the target relation.
    """

    rel = _resolve_relation(conn, relation)
    columns = _load_columns(conn, rel["schema_name"], rel["relname"], rel["oid"])
    indexes = _load_indexes(conn, rel["oid"])
    extended_stats = _load_extended_stats(conn, rel["oid"])
    partitioned_by = _load_partition_keys(conn, rel["oid"])
    vector_dim = _select_vector_dim(columns, vector_column)
    stats_fingerprint = _stable_hash(
        _stats_fingerprint_payload(
            rel=rel,
            columns=columns,
            extended_stats=extended_stats,
            vector_dim=vector_dim,
            partitioned_by=partitioned_by,
        )
    )
    index_fingerprint = _stable_hash(_index_fingerprint_payload(indexes))

    return TableStats(
        relname=f"{rel['schema_name']}.{rel['relname']}",
        n_rows=max(0, int(rel["reltuples"])),
        n_pages=max(0, int(rel["relpages"])),
        columns=columns,
        indexes=indexes,
        extended_stats=extended_stats,
        vector_dim=vector_dim,
        partitioned_by=partitioned_by,
        last_analyze=rel["last_analyze"],
        last_autoanalyze=rel["last_autoanalyze"],
        n_live_tup=_optional_int(rel["n_live_tup"]),
        n_mod_since_analyze=_optional_int(rel["n_mod_since_analyze"]),
        stats_fingerprint=stats_fingerprint,
        index_fingerprint=index_fingerprint,
    )


def connect(dsn: str) -> Connection[Any]:
    """Open a psycopg connection configured for dict-like rows."""

    return psycopg.connect(dsn, row_factory=dict_row)


def _resolve_relation(conn: Connection[Any], relation: str) -> dict[str, Any]:
    query = """
        SELECT
            c.oid,
            n.nspname AS schema_name,
            c.relname,
            c.reltuples,
            c.relpages,
            st.last_analyze,
            st.last_autoanalyze,
            st.n_live_tup,
            st.n_mod_since_analyze
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_all_tables st ON st.relid = c.oid
        WHERE c.oid = to_regclass(%s)
          AND c.relkind IN ('r', 'p')
    """
    row = conn.execute(query, (relation,)).fetchone()
    if row is None:
        raise IntrospectionError(f"could not resolve table: {relation}")
    return dict(row)


def _load_columns(
    conn: Connection[Any],
    schema_name: str,
    table_name: str,
    rel_oid: int,
) -> tuple[ColumnStats, ...]:
    query = """
        SELECT
            a.attname,
            a.atttypid::regtype::text AS type_name,
            a.atttypmod,
            s.null_frac,
            s.n_distinct,
            to_json(s.most_common_vals) AS most_common_vals,
            to_json(s.most_common_freqs) AS most_common_freqs,
            to_json(s.histogram_bounds) AS histogram_bounds,
            s.correlation,
            s.avg_width
        FROM pg_attribute a
        LEFT JOIN pg_stats s
          ON s.schemaname = %s
         AND s.tablename = %s
         AND s.attname = a.attname
        WHERE a.attrelid = %s
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
    """
    rows = conn.execute(query, (schema_name, table_name, rel_oid)).fetchall()
    columns: list[ColumnStats] = []
    for row in rows:
        columns.append(
            ColumnStats(
                name=str(row["attname"]),
                type_name=str(row["type_name"]),
                atttypmod=int(row["atttypmod"]),
                n_distinct=float(row["n_distinct"] or 0.0),
                null_frac=float(row["null_frac"] or 0.0),
                mcv=_json_tuple(row["most_common_vals"]),
                mcf=tuple(float(value) for value in _json_tuple(row["most_common_freqs"])),
                histogram=_json_tuple(row["histogram_bounds"]) or None,
                correlation=float(row["correlation"] or 0.0),
                avg_width=int(row["avg_width"] or 0),
                has_stats=row["n_distinct"] is not None,
            )
        )
    return tuple(columns)


def _load_indexes(conn: Connection[Any], rel_oid: int) -> tuple[IndexMeta, ...]:
    query = """
        SELECT
            c.relname AS index_name,
            am.amname AS method,
            c.relpages,
            c.reltuples,
            pg_get_expr(i.indpred, i.indrelid) AS predicate,
            c.reloptions,
            array_agg(
                COALESCE(a.attname, pg_get_indexdef(i.indexrelid, key.ord::int, true))
                ORDER BY key.ord
            ) AS columns,
            array_agg(opc.opcname ORDER BY key.ord) AS opclasses
        FROM pg_index i
        JOIN pg_class c ON c.oid = i.indexrelid
        JOIN pg_am am ON am.oid = c.relam
        CROSS JOIN LATERAL unnest(i.indkey, i.indclass)
             WITH ORDINALITY AS key(attnum, opcoid, ord)
        LEFT JOIN pg_attribute a
          ON a.attrelid = i.indrelid
         AND a.attnum = key.attnum
        LEFT JOIN pg_opclass opc ON opc.oid = key.opcoid
        WHERE i.indrelid = %s
        GROUP BY c.relname, am.amname, c.relpages, c.reltuples,
                 i.indpred, i.indrelid, c.reloptions
        ORDER BY c.relname
    """
    rows = conn.execute(query, (rel_oid,)).fetchall()
    indexes: list[IndexMeta] = []
    for row in rows:
        reloptions = _parse_reloptions(row["reloptions"])
        opclasses = tuple(str(value) for value in row["opclasses"] if value is not None)
        indexes.append(
            IndexMeta(
                name=str(row["index_name"]),
                method=str(row["method"]),
                columns=tuple(str(value) for value in row["columns"] if value is not None),
                opclass=opclasses[0] if opclasses else None,
                is_partial=row["predicate"] is not None,
                predicate=str(row["predicate"]) if row["predicate"] is not None else None,
                m=_int_option(reloptions, "m"),
                ef_construction=_int_option(reloptions, "ef_construction"),
                lists=_int_option(reloptions, "lists"),
                pages=int(row["relpages"] or 0),
                tuples=float(row["reltuples"] or 0.0),
            )
        )
    return tuple(indexes)


def _load_extended_stats(conn: Connection[Any], rel_oid: int) -> tuple[ExtendedStatsMeta, ...]:
    query = """
        SELECT
            n.nspname AS schema_name,
            e.stxname AS stats_name,
            array_agg(a.attname ORDER BY key.ord) AS columns,
            string_to_array(trim(both '{}' from e.stxkind::text), ',') AS kinds
        FROM pg_statistic_ext e
        JOIN pg_namespace n ON n.oid = e.stxnamespace
        CROSS JOIN LATERAL unnest(string_to_array(e.stxkeys::text, ' ')::int2[])
             WITH ORDINALITY AS key(attnum, ord)
        JOIN pg_attribute a
          ON a.attrelid = e.stxrelid
         AND a.attnum = key.attnum
        WHERE e.stxrelid = %s
        GROUP BY n.nspname, e.stxname, e.stxkind
        ORDER BY e.stxname
    """
    rows = conn.execute(query, (rel_oid,)).fetchall()
    return tuple(
        ExtendedStatsMeta(
            name=str(row["stats_name"]),
            schema=str(row["schema_name"]),
            columns=tuple(str(value) for value in row["columns"] if value is not None),
            kinds=tuple(_stats_kind_name(str(value)) for value in row["kinds"] if value),
        )
        for row in rows
    )


def _load_partition_keys(conn: Connection[Any], rel_oid: int) -> tuple[str, ...] | None:
    query = """
        SELECT pg_get_partkeydef(%s) AS partkey
        WHERE EXISTS (
            SELECT 1 FROM pg_partitioned_table WHERE partrelid = %s
        )
    """
    row = conn.execute(query, (rel_oid, rel_oid)).fetchone()
    if row is None or row["partkey"] is None:
        return None
    return (str(row["partkey"]),)


def _select_vector_dim(columns: tuple[ColumnStats, ...], vector_column: str | None) -> int:
    vector_columns = [
        column
        for column in columns
        if column.type_name in {"vector", "halfvec", "bit", "sparsevec"}
    ]
    if vector_column is not None:
        for column in vector_columns:
            if column.name == vector_column:
                return max(0, column.atttypmod)
        raise IntrospectionError(f"could not resolve vector column: {vector_column}")
    if not vector_columns:
        return 0
    return max(0, vector_columns[0].atttypmod)


def _json_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        decoded = json.loads(value)
    else:
        decoded = value
    if decoded is None:
        return ()
    if isinstance(decoded, list):
        return tuple(decoded)
    return (decoded,)


def _parse_reloptions(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    options: Iterable[Any]
    if isinstance(value, str):
        options = value.strip("{}").split(",") if value else ()
    else:
        options = value
    parsed: dict[str, str] = {}
    for option in options:
        key, sep, option_value = str(option).partition("=")
        if sep:
            parsed[key] = option_value
    return parsed


def _int_option(options: dict[str, str], key: str) -> int | None:
    value = options.get(key)
    if value is None:
        return None
    return int(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        default=str,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _stats_fingerprint_payload(
    *,
    rel: dict[str, Any],
    columns: tuple[ColumnStats, ...],
    extended_stats: tuple[ExtendedStatsMeta, ...],
    vector_dim: int,
    partitioned_by: tuple[str, ...] | None,
) -> dict[str, object]:
    return {
        "schema": str(rel["schema_name"]),
        "table": str(rel["relname"]),
        "reltuples": max(0, int(rel["reltuples"])),
        "relpages": max(0, int(rel["relpages"])),
        "vector_dim": vector_dim,
        "last_analyze": _json_scalar(rel["last_analyze"]),
        "last_autoanalyze": _json_scalar(rel["last_autoanalyze"]),
        "n_live_tup": _optional_int(rel["n_live_tup"]),
        "n_mod_since_analyze": _optional_int(rel["n_mod_since_analyze"]),
        "partitioned_by": list(partitioned_by or ()),
        "columns": [_column_fingerprint_payload(column) for column in columns],
        "extended_stats": [
            {
                "name": stat.name,
                "schema": stat.schema,
                "columns": list(stat.columns),
                "kinds": list(stat.kinds),
            }
            for stat in extended_stats
        ],
    }


def _column_fingerprint_payload(column: ColumnStats) -> dict[str, object]:
    return {
        "name": column.name,
        "type_name": column.type_name,
        "atttypmod": column.atttypmod,
        "n_distinct": column.n_distinct,
        "null_frac": column.null_frac,
        "mcv": [_json_scalar(value) for value in column.mcv],
        "mcf": list(column.mcf),
        "histogram": (
            [_json_scalar(value) for value in column.histogram]
            if column.histogram is not None
            else None
        ),
        "correlation": column.correlation,
        "avg_width": column.avg_width,
        "has_stats": column.has_stats,
    }


def _index_fingerprint_payload(indexes: tuple[IndexMeta, ...]) -> list[dict[str, object]]:
    return [
        {
            "name": index.name,
            "method": index.method,
            "columns": list(index.columns),
            "opclass": index.opclass,
            "is_partial": index.is_partial,
            "predicate": index.predicate,
            "m": index.m,
            "ef_construction": index.ef_construction,
            "lists": index.lists,
        }
        for index in sorted(indexes, key=lambda item: item.name)
    ]


def _json_scalar(value: Any) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _stats_kind_name(kind: str) -> str:
    return {
        "d": "ndistinct",
        "f": "dependencies",
        "m": "mcv",
        "e": "expressions",
    }.get(kind, kind)
