from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from psycopg import Connection

PGVECTOR_ITERATIVE_MIN_VERSION = (0, 8, 0)


@dataclass(frozen=True)
class PgCapabilities:
    server_version: str | None
    server_version_num: int | None
    pgvector_installed: bool
    pgvector_version: str | None
    supports_hnsw_iterative_scan: bool
    supports_hnsw_max_scan_tuples: bool
    notes: tuple[str, ...]


def load_pg_capabilities(conn: Connection[Any]) -> PgCapabilities:
    """Read PostgreSQL and pgvector capability metadata."""

    server_version = _show_value(conn, "server_version")
    server_version_num_raw = _show_value(conn, "server_version_num")
    server_version_num = (
        int(server_version_num_raw)
        if server_version_num_raw is not None and server_version_num_raw.isdigit()
        else None
    )
    pgvector_version = _pgvector_version(conn)
    supports_iterative = (
        pgvector_version is not None
        and version_at_least(pgvector_version, PGVECTOR_ITERATIVE_MIN_VERSION)
    )
    notes = []
    if pgvector_version is None:
        notes.append("pgvector extension is not installed in the current database")
    elif not supports_iterative:
        notes.append("pgvector iterative scans require extension version >= 0.8.0")
    else:
        notes.append("pgvector iterative scan GUCs are available")

    return PgCapabilities(
        server_version=server_version,
        server_version_num=server_version_num,
        pgvector_installed=pgvector_version is not None,
        pgvector_version=pgvector_version,
        supports_hnsw_iterative_scan=supports_iterative,
        supports_hnsw_max_scan_tuples=supports_iterative,
        notes=tuple(notes),
    )


def pg_capabilities_to_json(capabilities: PgCapabilities) -> dict[str, object]:
    return {
        "server_version": capabilities.server_version,
        "server_version_num": capabilities.server_version_num,
        "pgvector_installed": capabilities.pgvector_installed,
        "pgvector_version": capabilities.pgvector_version,
        "supports_hnsw_iterative_scan": capabilities.supports_hnsw_iterative_scan,
        "supports_hnsw_max_scan_tuples": capabilities.supports_hnsw_max_scan_tuples,
        "notes": list(capabilities.notes),
    }


def version_at_least(version: str, minimum: tuple[int, int, int]) -> bool:
    return parse_version(version) >= minimum


def parse_version(version: str) -> tuple[int, int, int]:
    parts = []
    for raw_part in version.split("."):
        match = re.match(r"\d+", raw_part)
        if match is None:
            break
        parts.append(int(match.group(0)))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return (parts[0], parts[1], parts[2])


def _show_value(conn: Connection[Any], name: str) -> str | None:
    row = conn.execute(f"SHOW {name}").fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return str(next(iter(row.values())))
    return str(row[0])


def _pgvector_version(conn: Connection[Any]) -> str | None:
    row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
    if row is None:
        return None
    if isinstance(row, dict):
        return str(row["extversion"])
    return str(row[0])
