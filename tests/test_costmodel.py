from vecadvisor.costmodel import (
    choose_best,
    cost_exact,
    cost_iterative,
    cost_partial,
    cost_postfilter,
)
from vecadvisor.models import CalibrationProfile, CostEstimate, Strategy


def _cal() -> CalibrationProfile:
    return CalibrationProfile(
        dataset_id="unit",
        hardware_id="unit",
        index_method="hnsw",
        c_d=0.01,
        c_scan=0.005,
        c_h=2.0,
        delta_strict=0.25,
        recall_curve=((40, 0.9), (80, 0.95), (160, 0.98)),
    )


def test_postfilter_recall_collapses_when_expected_survivors_below_k() -> None:
    estimate = cost_postfilter(
        n_rows=1_000_000,
        s_local=0.01,
        limit=10,
        ef_search=40,
        cal=_cal(),
        confidence=1.0,
    )

    assert not estimate.est_returns_k
    assert estimate.est_recall < 0.1


def test_iterative_latency_increases_as_local_selectivity_drops() -> None:
    fast = cost_iterative(
        n_rows=1_000_000,
        s_local=0.1,
        limit=10,
        ef_search=40,
        cal=_cal(),
        strict_order=False,
        confidence=1.0,
    )
    slow = cost_iterative(
        n_rows=1_000_000,
        s_local=0.01,
        limit=10,
        ef_search=40,
        cal=_cal(),
        strict_order=False,
        confidence=1.0,
    )

    assert slow.est_latency_us > fast.est_latency_us


def test_exact_cost_increases_with_global_selectivity() -> None:
    low = cost_exact(n_rows=100_000, s_global=0.001, limit=10, cal=_cal(), confidence=1.0)
    high = cost_exact(n_rows=100_000, s_global=0.1, limit=10, cal=_cal(), confidence=1.0)

    assert high.est_latency_us > low.est_latency_us


def test_partial_index_cost_increases_with_index_size() -> None:
    small = cost_partial(
        n_rows=1_000_000,
        s_index=0.001,
        ef_search=40,
        cal=_cal(),
        confidence=1.0,
    )
    large = cost_partial(
        n_rows=1_000_000,
        s_index=0.1,
        ef_search=40,
        cal=_cal(),
        confidence=1.0,
    )

    assert large.est_latency_us > small.est_latency_us
    assert small.est_recall == _cal().recall_at(40)


def test_choose_best_prefers_iterative_over_postfilter_on_latency_tie() -> None:
    postfilter = CostEstimate(
        strategy=Strategy.POSTFILTER,
        est_latency_us=100.0,
        est_recall=0.95,
        est_returns_k=True,
        confidence=0.9,
    )
    iterative = CostEstimate(
        strategy=Strategy.ITERATIVE_RELAXED,
        est_latency_us=100.0,
        est_recall=0.95,
        est_returns_k=True,
        confidence=0.9,
    )

    assert choose_best((postfilter, iterative), recall_target=0.95).strategy is (
        Strategy.ITERATIVE_RELAXED
    )
