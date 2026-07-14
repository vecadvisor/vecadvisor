from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vecadvisor.models import TableStats
from vecadvisor.stats_health import assess_stats_health, stats_health_to_json


def test_assess_stats_health_marks_never_analyzed_table_stale() -> None:
    health = assess_stats_health(_table(last_analyze=None, n_mod_since_analyze=10))

    assert health.status == "never_analyzed"
    assert health.stale is True
    assert health.analyze_sql == 'ANALYZE "public"."docs";'
    assert any("no recorded ANALYZE" in note for note in health.notes)


def test_assess_stats_health_marks_large_change_ratio_stale() -> None:
    health = assess_stats_health(
        _table(last_analyze=datetime(2026, 1, 1, tzinfo=UTC), n_mod_since_analyze=3_000)
    )

    assert health.status == "stale"
    assert health.stale is True
    assert health.modification_ratio == pytest.approx(0.3)
    payload = stats_health_to_json(health)
    assert payload["status"] == "stale"
    assert payload["last_analyze_at"] == "2026-01-01T00:00:00+00:00"


def test_assess_stats_health_ignores_small_tables_with_few_modifications() -> None:
    health = assess_stats_health(
        _table(last_analyze=datetime(2026, 1, 1, tzinfo=UTC), n_mod_since_analyze=20)
    )

    assert health.status == "fresh"
    assert health.stale is False
    assert health.modification_ratio == pytest.approx(0.002)


def test_assess_stats_health_validates_thresholds() -> None:
    with pytest.raises(ValueError, match="stale_modification_ratio"):
        assess_stats_health(_table(), stale_modification_ratio=0.0)

    with pytest.raises(ValueError, match="stale_min_modifications"):
        assess_stats_health(_table(), stale_min_modifications=-1)


def _table(
    *,
    last_analyze: datetime | None = datetime(2026, 1, 1, tzinfo=UTC),
    n_mod_since_analyze: int | None = 0,
) -> TableStats:
    return TableStats(
        relname="public.docs",
        n_rows=10_000,
        n_pages=100,
        columns=(),
        last_analyze=last_analyze,
        n_live_tup=10_000,
        n_mod_since_analyze=n_mod_since_analyze,
    )
