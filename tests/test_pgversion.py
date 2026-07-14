from __future__ import annotations

from vecadvisor.pgversion import (
    PgCapabilities,
    parse_version,
    pg_capabilities_to_json,
    version_at_least,
)


def test_parse_version_handles_release_suffixes() -> None:
    assert parse_version("0.8.0") == (0, 8, 0)
    assert parse_version("0.8.0-beta1") == (0, 8, 0)
    assert parse_version("0.7") == (0, 7, 0)
    assert parse_version("bad") == (0, 0, 0)


def test_version_at_least_compares_pgvector_versions() -> None:
    assert version_at_least("0.8.0", (0, 8, 0))
    assert version_at_least("0.8.1", (0, 8, 0))
    assert not version_at_least("0.7.4", (0, 8, 0))


def test_pg_capabilities_to_json_shape() -> None:
    capabilities = PgCapabilities(
        server_version="16.4",
        server_version_num=160004,
        pgvector_installed=True,
        pgvector_version="0.8.0",
        supports_hnsw_iterative_scan=True,
        supports_hnsw_max_scan_tuples=True,
        notes=("pgvector iterative scan GUCs are available",),
    )

    payload = pg_capabilities_to_json(capabilities)

    assert payload["server_version_num"] == 160004
    assert payload["pgvector_version"] == "0.8.0"
    assert payload["supports_hnsw_iterative_scan"] is True
    assert payload["notes"] == ["pgvector iterative scan GUCs are available"]
