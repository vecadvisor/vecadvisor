from __future__ import annotations

import importlib

import pytest


def test_query_anticorrelated_band_skips_immediate_query_neighborhood(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    prep = importlib.import_module("tools.prepare_ann_benchmark_dataset")
    prep.np = np
    vectors = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype="float32")
    queries = np.asarray([[0.0]], dtype="float32")
    out = tmp_path / "mask.npy"

    mask = prep._write_query_anticorrelated_band_filter(
        vectors,
        out,
        rows=6,
        selectivity=2 / 6,
        query_vectors=queries,
        chunk_rows=2,
        anti_start_rank=1,
    )

    assert mask.tolist() == [False, True, True, False, False, False]
    assert np.load(out, allow_pickle=False).tolist() == mask.tolist()


def test_query_anticorrelated_tail_selects_farthest_query_rows(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    prep = importlib.import_module("tools.prepare_ann_benchmark_dataset")
    prep.np = np
    vectors = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype="float32")
    queries = np.asarray([[0.0]], dtype="float32")
    out = tmp_path / "mask.npy"

    mask = prep._write_query_anticorrelated_tail_filter(
        vectors,
        out,
        rows=6,
        selectivity=2 / 6,
        query_vectors=queries,
        chunk_rows=2,
    )

    assert mask.tolist() == [False, False, False, False, True, True]


def test_query_anticorrelated_band_excludes_each_query_neighborhood(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    prep = importlib.import_module("tools.prepare_ann_benchmark_dataset")
    prep.np = np
    vectors = np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]], dtype="float32")
    queries = np.asarray([[0.0], [5.0]], dtype="float32")
    out = tmp_path / "mask.npy"

    mask = prep._write_query_anticorrelated_band_filter(
        vectors,
        out,
        rows=6,
        selectivity=2 / 6,
        query_vectors=queries,
        chunk_rows=2,
        anti_start_rank=1,
    )

    assert mask[0] == np.False_
    assert mask[5] == np.False_
    assert int(mask.sum()) == 2
