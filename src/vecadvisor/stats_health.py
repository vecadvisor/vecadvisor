from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import TableStats
from .query_spec import quote_qualified_identifier

STALE_MODIFICATION_RATIO = 0.20
STALE_MIN_MODIFICATIONS = 1_000


@dataclass(frozen=True)
class StatsHealth:
    status: str
    stale: bool
    last_analyze_at: datetime | None
    last_autoanalyze_at: datetime | None
    n_live_tup: int | None
    n_mod_since_analyze: int | None
    modification_ratio: float | None
    analyze_sql: str
    notes: tuple[str, ...]


def assess_stats_health(
    table: TableStats,
    *,
    stale_modification_ratio: float = STALE_MODIFICATION_RATIO,
    stale_min_modifications: int = STALE_MIN_MODIFICATIONS,
) -> StatsHealth:
    """Assess whether planner statistics are fresh enough for advisor estimates."""

    if not 0.0 < stale_modification_ratio <= 1.0:
        raise ValueError("stale_modification_ratio must be in (0, 1]")
    if stale_min_modifications < 0:
        raise ValueError("stale_min_modifications must be non-negative")

    modification_ratio = _modification_ratio(table)
    analyze_sql = f"ANALYZE {quote_qualified_identifier(table.relname)};"

    if table.last_analyze is None and table.last_autoanalyze is None:
        return _health(
            table,
            status="never_analyzed",
            stale=True,
            modification_ratio=modification_ratio,
            analyze_sql=analyze_sql,
            notes=(
                "table has no recorded ANALYZE timestamp; selectivity estimates may be weak",
                "run ANALYZE before trusting advisor cost estimates",
            ),
        )

    if (
        table.n_mod_since_analyze is not None
        and modification_ratio is not None
        and table.n_mod_since_analyze >= stale_min_modifications
        and modification_ratio >= stale_modification_ratio
    ):
        return _health(
            table,
            status="stale",
            stale=True,
            modification_ratio=modification_ratio,
            analyze_sql=analyze_sql,
            notes=(
                "many rows changed since the last ANALYZE; reltuples and column stats may be stale",
                "refresh statistics before treating selectivity/cost estimates as durable",
            ),
        )

    return _health(
        table,
        status="fresh",
        stale=False,
        modification_ratio=modification_ratio,
        analyze_sql=analyze_sql,
        notes=("planner statistics look fresh enough for advisor estimates",),
    )


def stats_health_to_json(health: StatsHealth) -> dict[str, object]:
    return {
        "status": health.status,
        "stale": health.stale,
        "last_analyze_at": _datetime_to_json(health.last_analyze_at),
        "last_autoanalyze_at": _datetime_to_json(health.last_autoanalyze_at),
        "n_live_tup": health.n_live_tup,
        "n_mod_since_analyze": health.n_mod_since_analyze,
        "modification_ratio": health.modification_ratio,
        "analyze_sql": health.analyze_sql,
        "notes": list(health.notes),
    }


def _health(
    table: TableStats,
    *,
    status: str,
    stale: bool,
    modification_ratio: float | None,
    analyze_sql: str,
    notes: tuple[str, ...],
) -> StatsHealth:
    return StatsHealth(
        status=status,
        stale=stale,
        last_analyze_at=table.last_analyze,
        last_autoanalyze_at=table.last_autoanalyze,
        n_live_tup=table.n_live_tup,
        n_mod_since_analyze=table.n_mod_since_analyze,
        modification_ratio=modification_ratio,
        analyze_sql=analyze_sql,
        notes=notes,
    )


def _modification_ratio(table: TableStats) -> float | None:
    if table.n_mod_since_analyze is None:
        return None
    denominator = table.n_live_tup if table.n_live_tup and table.n_live_tup > 0 else table.n_rows
    denominator = max(denominator, 1)
    return table.n_mod_since_analyze / denominator


def _datetime_to_json(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()
