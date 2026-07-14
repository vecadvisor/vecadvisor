from __future__ import annotations

import pytest

from vecadvisor.bench.datasets import (
    generate_synthetic_dataset,
    generate_synthetic_queries,
    load_file_dataset,
    load_file_queries,
)


def test_generate_synthetic_dataset_is_deterministic_and_selective() -> None:
    left = generate_synthetic_dataset(
        n_rows=5_000,
        dim=8,
        n_clusters=6,
        filter_selectivity=0.2,
        correlation=0.7,
        seed=42,
    )
    right = generate_synthetic_dataset(
        n_rows=5_000,
        dim=8,
        n_clusters=6,
        filter_selectivity=0.2,
        correlation=0.7,
        seed=42,
    )

    assert left.vectors.shape == (5_000, 8)
    assert left.filter_mask.shape == (5_000,)
    assert left.observed_selectivity == pytest.approx(0.2, abs=0.04)
    assert left.filter_probabilities[0] > left.filter_probabilities[-1]
    assert left.vectors.tolist() == right.vectors.tolist()
    assert left.filter_mask.tolist() == right.filter_mask.tolist()


def test_generate_synthetic_dataset_negative_correlation_reverses_hot_clusters() -> None:
    dataset = generate_synthetic_dataset(
        n_rows=2_000,
        dim=4,
        n_clusters=5,
        filter_selectivity=0.15,
        correlation=-0.8,
        seed=7,
    )

    assert dataset.filter_probabilities[0] < dataset.filter_probabilities[-1]


def test_generate_synthetic_queries_follow_cluster_policy() -> None:
    dataset = generate_synthetic_dataset(
        n_rows=1_000,
        dim=4,
        n_clusters=4,
        filter_selectivity=0.2,
        correlation=0.9,
        seed=9,
    )

    hot = generate_synthetic_queries(
        dataset,
        n_queries=32,
        seed=10,
        cluster_policy="filter_hot",
    )
    cold = generate_synthetic_queries(
        dataset,
        n_queries=32,
        seed=10,
        cluster_policy="filter_cold",
    )

    assert hot.vectors.shape == (32, 4)
    assert hot.cluster_ids.mean() < cold.cluster_ids.mean()


def test_load_file_dataset_and_queries_from_npy(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    vectors_path = tmp_path / "vectors.npy"
    filter_path = tmp_path / "filter.npy"
    queries_path = tmp_path / "queries.npy"
    np.save(vectors_path, np.arange(24, dtype="float32").reshape(8, 3))
    np.save(filter_path, np.asarray([True, False, True, False, False, True, False, False]))
    np.save(queries_path, np.ones((2, 3), dtype="float32"))

    dataset = load_file_dataset(
        vectors_path=vectors_path,
        filter_mask_path=filter_path,
    )
    queries = load_file_queries(query_vectors_path=queries_path, expected_dim=dataset.dim)

    assert dataset.dataset_id == "file"
    assert dataset.n_rows == 8
    assert dataset.dim == 3
    assert dataset.observed_selectivity == pytest.approx(3 / 8)
    assert dataset.filter_probabilities == (pytest.approx(3 / 8),)
    assert queries.n_queries == 2
    assert queries.cluster_policy == "file"


def test_load_file_queries_rejects_dimension_mismatch(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    queries_path = tmp_path / "queries.npy"
    np.save(queries_path, np.ones((2, 4), dtype="float32"))

    with pytest.raises(ValueError, match="dimension"):
        load_file_queries(query_vectors_path=queries_path, expected_dim=3)
