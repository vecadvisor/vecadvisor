from __future__ import annotations

import pytest

from vecadvisor.diagnostics import parse_diagnostic_format, render_explain_vector


def test_render_explain_vector_includes_optimizer_evidence() -> None:
    text = render_explain_vector(_diagnostic_payload())

    assert text.startswith("EXPLAIN VECTOR  public.docs")
    assert "filter: tenant_id = 42" in text
    assert "s(global): 0.01" in text
    assert "s(local): 0.002" in text
    assert "PLAN" in text
    assert "-> exact" in text
    assert "postfilter" in text
    assert "planner" in text
    assert "STATISTICS" in text
    assert "CREATE STATISTICS" in text
    assert "STATS HEALTH" in text
    assert "ANALYZE" in text
    assert "SELECTIVITY CROSS-CHECK" in text
    assert "severity: warning" in text
    assert "CATALOG SNAPSHOT" in text
    assert "stats_fingerprint: abc123" in text
    assert "PGVECTOR CAPABILITIES" in text
    assert "iterative_scan=NO" in text
    assert "LOCAL PROBE" in text
    assert "LOCAL PROBE CACHE" in text
    assert "hit: yes" in text
    assert "WHY:" in text
    assert "decision reason from summary" in text


def test_parse_diagnostic_format_validates_values() -> None:
    assert parse_diagnostic_format("TEXT") == "text"
    assert parse_diagnostic_format(" json ") == "json"

    with pytest.raises(ValueError, match="diagnostic format"):
        parse_diagnostic_format("xml")


def _diagnostic_payload() -> dict[str, object]:
    return {
        "table": "public.docs",
        "vector_dim": 768,
        "filter": "tenant_id = 42",
        "limit": 10,
        "ef_search": 40,
        "recall_target": 0.95,
        "catalog_snapshot": {
            "stats_fingerprint": "abc123",
            "index_fingerprint": "def456",
            "last_analyze_at": "2026-01-01T00:00:00+00:00",
            "n_mod_since_analyze": 3_000_000,
        },
        "selectivity": {
            "advisor_selectivity": 0.01,
            "postgres_selectivity": 0.1,
            "advisor_rows": 100_000,
            "postgres_plan_rows": 1_000_000,
            "absolute_delta": 0.09,
            "ratio": 10.0,
            "status": "diverged",
            "severity": "warning",
            "notes": ["advisor global selectivity diverges from PostgreSQL plan rows"],
        },
        "statistics_suggestions": [
            {
                "ddl": "CREATE STATISTICS docs_tenant_region ON tenant_id, region FROM docs",
                "reason": "PostgreSQL selectivity differs from advisor estimate",
                "confidence": 0.8,
            }
        ],
        "stats_health": {
            "status": "stale",
            "stale": True,
            "last_analyze_at": "2026-01-01T00:00:00+00:00",
            "last_autoanalyze_at": None,
            "n_live_tup": 10_000_000,
            "n_mod_since_analyze": 3_000_000,
            "modification_ratio": 0.3,
            "analyze_sql": 'ANALYZE "public"."docs";',
            "notes": ["many rows changed since the last ANALYZE"],
        },
        "local_selectivity": {
            "s_global": 0.01,
            "s_local_p10": 0.002,
            "s_local_median": 0.002,
            "rho": -0.8,
            "confidence": 0.7,
            "sample_size": 200,
            "passing_rows": 1,
            "resolution_floor": 0.005,
            "notes": ["local selectivity is lower than global selectivity"],
        },
        "local_selectivity_cache": {
            "enabled": True,
            "cache_dir": ".vecadvisor-cache",
            "key": "cache-key",
            "path": ".vecadvisor-cache/local-selectivity-cache-key.json",
            "hit": True,
            "stored": False,
            "refresh": False,
        },
        "plan": {
            "root_node_type": "Seq Scan",
            "planning_time_ms": 0.2,
            "observed_vector_query": {
                "strategy": "postfilter",
            },
        },
        "calibration": {
            "source": "test-profile",
        },
        "postgres": {
            "server_version": "16.4",
            "server_version_num": 160004,
            "pgvector_installed": True,
            "pgvector_version": "0.7.4",
            "supports_hnsw_iterative_scan": False,
            "supports_hnsw_max_scan_tuples": False,
            "notes": ["pgvector iterative scans require extension version >= 0.8.0"],
        },
        "recommendation": {
            "verdict": "Use exact search for this selective filter",
            "s_global": 0.01,
            "s_local": 0.002,
            "rho": -0.8,
            "planner_would_pick": "postfilter",
            "decision": {
                "recommended_strategy": "exact",
                "planner_strategy": "postfilter",
                "planner_mismatch": True,
                "viable": True,
                "why": ["decision reason from summary"],
            },
            "ranked": [
                {
                    "strategy": "exact",
                    "plan": {
                        "ef_search": None,
                        "uses_index": None,
                        "requires_new_object": None,
                    },
                    "estimate": {
                        "est_latency_us": 14_200.0,
                        "est_recall": 1.0,
                        "est_returns_k": True,
                        "confidence": 0.9,
                        "notes": ["exact over ~100000 filtered rows"],
                    },
                },
                {
                    "strategy": "postfilter",
                    "plan": {
                        "ef_search": 40,
                        "uses_index": "docs_embedding_idx",
                        "requires_new_object": None,
                    },
                    "estimate": {
                        "est_latency_us": 2_100.0,
                        "est_recall": 0.1,
                        "est_returns_k": False,
                        "confidence": 0.6,
                        "notes": ["expected survivors ~= 0.08"],
                    },
                },
            ],
        },
        "notes": ["local selectivity estimated from the unfiltered top-m vector neighborhood"],
    }
