from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import LocalSelectivity, Predicate, QuerySpec, TableStats

LOCAL_SELECTIVITY_CACHE_VERSION = 1


def build_local_selectivity_cache_key(
    *,
    table: TableStats,
    query: QuerySpec,
    filter_sql: str,
    vector_source: str,
    query_vectors: Sequence[Sequence[float]],
    probe_rows: int,
    s_global: float,
) -> str:
    """Build a stable key for cached aggregate local-selectivity probes."""

    payload = {
        "version": LOCAL_SELECTIVITY_CACHE_VERSION,
        "table": table.relname,
        "vector_column": query.vector_column,
        "distance_op": query.distance_op,
        "limit": query.limit,
        "filter_sql": filter_sql.strip(),
        "predicates": [_predicate_payload(predicate) for predicate in query.predicates],
        "probe_rows": probe_rows,
        "s_global": _float_key(s_global),
        "stats_fingerprint": table.stats_fingerprint,
        "index_fingerprint": table.index_fingerprint,
        "vector_source": vector_source,
        "query_vector_count": len(query_vectors),
        "query_vector_fingerprint": _query_vector_fingerprint(query_vectors),
    }
    return _stable_hash(payload)


def local_selectivity_cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"local-selectivity-{key}.json"


def load_local_selectivity_cache(cache_dir: Path, key: str) -> LocalSelectivity | None:
    """Load a cached local-selectivity result; malformed entries fail open as misses."""

    path = local_selectivity_cache_path(cache_dir, key)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("version") != LOCAL_SELECTIVITY_CACHE_VERSION:
        return None
    if payload.get("key") != key:
        return None
    local_payload = payload.get("local_selectivity")
    if not isinstance(local_payload, dict):
        return None
    return _local_selectivity_from_json(local_payload)


def store_local_selectivity_cache(
    cache_dir: Path,
    key: str,
    local_selectivity: LocalSelectivity,
) -> Path:
    """Persist a cached local-selectivity result with atomic file replacement."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    path = local_selectivity_cache_path(cache_dir, key)
    payload = {
        "version": LOCAL_SELECTIVITY_CACHE_VERSION,
        "key": key,
        "local_selectivity": _local_selectivity_to_json(local_selectivity),
    }
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def _predicate_payload(predicate: Predicate) -> dict[str, object]:
    return {
        "column": predicate.column,
        "kind": predicate.kind.value,
        "values": [_json_value(value) for value in predicate.values],
        "is_literal": predicate.is_literal,
    }


def _query_vector_fingerprint(query_vectors: Sequence[Sequence[float]]) -> str:
    return _stable_hash(
        [
            [_float_key(float(value)) for value in vector]
            for vector in query_vectors
        ]
    )


def _local_selectivity_to_json(local_selectivity: LocalSelectivity) -> dict[str, object]:
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


def _local_selectivity_from_json(payload: dict[str, object]) -> LocalSelectivity | None:
    try:
        notes_raw = payload.get("notes", ())
        if not isinstance(notes_raw, list):
            return None
        return LocalSelectivity(
            s_global=_required_float(payload, "s_global"),
            s_local_p10=_required_float(payload, "s_local_p10"),
            s_local_median=_required_float(payload, "s_local_median"),
            rho=_required_float(payload, "rho"),
            confidence=_required_float(payload, "confidence"),
            sample_size=_required_int(payload, "sample_size"),
            passing_rows=_required_int(payload, "passing_rows"),
            resolution_floor=_optional_float(payload.get("resolution_floor"), default=0.0),
            notes=tuple(str(note) for note in notes_raw),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _float_key(value: float) -> str:
    return format(value, ".17g")


def _required_float(payload: dict[str, object], key: str) -> float:
    return _coerce_float(payload[key])


def _optional_float(value: object, *, default: float) -> float:
    return default if value is None else _coerce_float(value)


def _coerce_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError("expected numeric JSON value")
    return float(value)


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise TypeError("expected integer JSON value")
    return int(value)


def _json_value(value: Any) -> object:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)
