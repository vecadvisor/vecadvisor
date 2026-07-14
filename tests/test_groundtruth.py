from __future__ import annotations

import pytest

from vecadvisor.bench import groundtruth
from vecadvisor.bench.groundtruth import exact_topk, max_block_rows_for_memory, recall_at_k


def test_exact_topk_respects_filter_mask_and_blocks() -> None:
    result = exact_topk(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
        [[0.2, 0.0], [9.0, 0.0]],
        k=2,
        filter_mask=[True, True, False, True],
        block_rows=2,
    )

    assert result.candidate_count == 3
    assert result.block_rows == 2
    assert result.blocks_scanned == 2
    assert result.indices.tolist() == [[0, 1], [3, 1]]
    assert result.distances.tolist()[0] == pytest.approx([0.2, 0.8])
    assert result.distances.tolist()[1] == pytest.approx([1.0, 8.0])


def test_exact_topk_pads_when_filter_has_fewer_than_k_candidates() -> None:
    result = exact_topk(
        [[0.0, 0.0], [1.0, 0.0]],
        [[0.2, 0.0]],
        k=3,
        filter_mask=[False, True],
        block_rows=1,
    )

    assert result.indices.tolist() == [[1, -1, -1]]
    assert result.distances[0, 0] == pytest.approx(0.8)


def test_recall_at_k_ignores_padded_truth_entries() -> None:
    recall = recall_at_k(
        [[10, 20, -1], [30, 40, 50]],
        [[10, 99, -1], [30, 50, 60]],
        k=3,
    )

    assert recall.per_query == pytest.approx((0.5, 2 / 3))
    assert recall.mean == pytest.approx((0.5 + 2 / 3) / 2)


def test_max_block_rows_for_memory_uses_dimension_and_budget() -> None:
    assert max_block_rows_for_memory(dim=128, bytes_budget=1024, dtype_bytes=4) == 2


def test_exact_topk_streams_query_blocks_without_materializing_full_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    np = pytest.importorskip("numpy")
    base = np.arange(1000 * 4, dtype="float32").reshape(1000, 4)
    queries = np.zeros((3, 4), dtype="float32")
    seen_blocks: list[tuple[int, int]] = []
    original_distance_scores = groundtruth._distance_scores

    def wrapped_distance_scores(
        np_module: object,
        block: object,
        query: object,
        *,
        metric: str,
    ) -> object:
        shape = tuple(int(value) for value in block.shape)
        seen_blocks.append(shape)
        assert shape[0] <= 128
        return original_distance_scores(np_module, block, query, metric=metric)

    monkeypatch.setattr(groundtruth, "_distance_scores", wrapped_distance_scores)

    result = groundtruth.exact_topk(base, queries, k=5, block_rows=128)

    assert result.blocks_scanned == 8
    assert len(seen_blocks) == 3 * 8
