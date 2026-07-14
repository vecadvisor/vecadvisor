from vecadvisor.models import ColumnStats, Predicate, PredicateKind, TableStats
from vecadvisor.selectivity import (
    conjunction_selectivity,
    eq_selectivity,
    range_selectivity,
    rho_from_selectivities,
)


def test_eq_selectivity_uses_mcv_frequency() -> None:
    column = ColumnStats(
        name="tenant_id",
        n_distinct=100,
        null_frac=0.0,
        mcv=(42,),
        mcf=(0.2,),
    )

    assert eq_selectivity(column, 42, 10_000) == 0.2


def test_conjunction_multiplies_initial_independent_estimates() -> None:
    table = TableStats(
        relname="docs",
        n_rows=10_000,
        n_pages=100,
        columns=(
            ColumnStats(name="tenant_id", n_distinct=100, null_frac=0.0),
            ColumnStats(name="region", n_distinct=10, null_frac=0.0),
        ),
    )
    predicates = (
        Predicate("tenant_id", PredicateKind.EQ, (42,)),
        Predicate("region", PredicateKind.EQ, ("us",)),
    )

    assert conjunction_selectivity(table, predicates) == 0.001


def test_range_selectivity_interpolates_numeric_histogram() -> None:
    column = ColumnStats(
        name="score",
        n_distinct=101,
        null_frac=0.0,
        histogram=(0, 25, 50, 75, 100),
    )

    assert range_selectivity(column, PredicateKind.RANGE_LT, 37.5) == 0.375
    assert range_selectivity(column, PredicateKind.RANGE_GT, 75) == 0.25


def test_range_selectivity_accounts_for_mcv_mass() -> None:
    column = ColumnStats(
        name="score",
        n_distinct=101,
        null_frac=0.0,
        mcv=(10,),
        mcf=(0.2,),
        histogram=(0, 50, 100),
    )

    assert range_selectivity(column, PredicateKind.RANGE_LT, 25) == 0.4


def test_between_range_selectivity_uses_both_bounds() -> None:
    column = ColumnStats(
        name="score",
        n_distinct=101,
        null_frac=0.0,
        histogram=(0, 25, 50, 75, 100),
    )

    assert range_selectivity(column, PredicateKind.RANGE_BETWEEN, (25, 75)) == 0.5


def test_rho_is_negative_for_sparse_local_neighborhood() -> None:
    assert rho_from_selectivities(0.1, 0.01) < 0
