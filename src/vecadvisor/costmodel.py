from __future__ import annotations

import math

from .models import CalibrationProfile, CostEstimate, Strategy

EPS = 1e-12
DEFAULT_HNSW_MAX_SCAN_TUPLES = 20_000
STRATEGY_TIE_BREAK = {
    Strategy.PARTIAL: 0,
    Strategy.PARTITION: 1,
    Strategy.ITERATIVE_RELAXED: 2,
    Strategy.ITERATIVE_STRICT: 3,
    Strategy.POSTFILTER: 4,
    Strategy.EXACT: 5,
}


def cost_exact(
    *,
    n_rows: int,
    s_global: float,
    limit: int,
    cal: CalibrationProfile,
    confidence: float,
) -> CostEstimate:
    qualifying_rows = max(1.0, n_rows * max(s_global, EPS))
    latency = (cal.c_scan + cal.c_d) * qualifying_rows
    latency += cal.c_scan * qualifying_rows * math.log2(max(limit, 2))
    return CostEstimate(
        strategy=Strategy.EXACT,
        est_latency_us=latency,
        est_recall=1.0,
        est_returns_k=True,
        confidence=confidence,
        notes=(f"exact over ~{qualifying_rows:.0f} filtered rows",),
    )


def cost_postfilter(
    *,
    n_rows: int,
    s_local: float,
    limit: int,
    ef_search: int,
    cal: CalibrationProfile,
    confidence: float,
) -> CostEstimate:
    latency = cal.c_d * cal.c_h * ef_search * math.log(max(n_rows, 2))
    survivors = s_local * ef_search
    recall = cal.recall_at(ef_search) * min(1.0, survivors / max(limit, 1))
    returns_k = survivors >= limit
    return CostEstimate(
        strategy=Strategy.POSTFILTER,
        est_latency_us=latency,
        est_recall=recall,
        est_returns_k=returns_k,
        confidence=confidence,
        notes=(f"expected survivors ~= {survivors:.2f}",),
    )


def cost_iterative(
    *,
    n_rows: int,
    s_local: float,
    limit: int,
    ef_search: int,
    cal: CalibrationProfile,
    strict_order: bool,
    max_scan_tuples: int = DEFAULT_HNSW_MAX_SCAN_TUPLES,
    confidence: float,
) -> CostEstimate:
    candidates = limit / max(s_local, EPS)
    latency = cal.c_d * cal.c_h * math.log(max(n_rows, 2)) * candidates
    if strict_order:
        latency *= 1.0 + cal.delta_strict

    cap_hit = candidates > max_scan_tuples
    if cap_hit:
        recall = cal.recall_at(ef_search) * min(1.0, s_local * max_scan_tuples / max(limit, 1))
    else:
        recall = cal.recall_at(ef_search)

    return CostEstimate(
        strategy=Strategy.ITERATIVE_STRICT if strict_order else Strategy.ITERATIVE_RELAXED,
        est_latency_us=latency,
        est_recall=recall,
        est_returns_k=not cap_hit,
        confidence=confidence,
        notes=(f"expected candidates ~= {candidates:.0f}",),
    )


def cost_partial(
    *,
    n_rows: int,
    s_index: float,
    ef_search: int,
    cal: CalibrationProfile,
    confidence: float,
) -> CostEstimate:
    indexed_rows = max(1.0, n_rows * max(s_index, EPS))
    latency = cal.c_d * cal.c_h * ef_search * math.log(max(indexed_rows, 2))
    return CostEstimate(
        strategy=Strategy.PARTIAL,
        est_latency_us=latency,
        est_recall=cal.recall_at(ef_search),
        est_returns_k=True,
        confidence=confidence,
        notes=(f"partial HNSW over ~{indexed_rows:.0f} indexed rows",),
    )


def cost_partition(
    *,
    n_rows: int,
    s_partition: float,
    ef_search: int,
    cal: CalibrationProfile,
    confidence: float,
) -> CostEstimate:
    touched_rows = max(1.0, n_rows * max(s_partition, EPS))
    latency = cal.c_d * cal.c_h * ef_search * math.log(max(touched_rows, 2))
    return CostEstimate(
        strategy=Strategy.PARTITION,
        est_latency_us=latency,
        est_recall=cal.recall_at(ef_search),
        est_returns_k=True,
        confidence=confidence,
        notes=(f"partition-pruned HNSW over ~{touched_rows:.0f} rows",),
    )


def choose_best(candidates: tuple[CostEstimate, ...], recall_target: float) -> CostEstimate:
    viable = [
        candidate
        for candidate in candidates
        if candidate.est_returns_k and candidate.est_recall >= recall_target
    ]
    if not viable:
        exact = [candidate for candidate in candidates if candidate.strategy is Strategy.EXACT]
        if exact:
            return exact[0]
        raise ValueError("no viable candidates and no EXACT fallback")
    return min(
        viable,
        key=lambda candidate: (
            round(candidate.est_latency_us, 9),
            STRATEGY_TIE_BREAK.get(candidate.strategy, 99),
            -candidate.confidence,
        ),
    )
