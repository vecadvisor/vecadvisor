from __future__ import annotations

import pytest

from vecadvisor.bench import groundtruth
from vecadvisor.bench.groundtruth import exact_topk, max_block_rows_for_memory, recall_at_k
from vecadvisor.native_distance import NativeDistanceError, NativeTopKResult


def test_exact_topk_respects_filter_mask_and_blocks() -> None:
    result = exact_topk(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
        [[0.2, 0.0], [9.0, 0.0]],
        k=2,
        filter_mask=[True, True, False, True],
        block_rows=2,
        use_native=False,
    )

    assert result.candidate_count == 3
    assert result.block_rows == 2
    assert result.blocks_scanned == 2
    assert result.native_used is False
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
        use_native=False,
    )

    assert result.indices.tolist() == [[1, -1, -1]]
    assert result.distances[0, 0] == pytest.approx(0.8)


def test_exact_topk_uses_native_loader_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_native = _FakeNativeLibrary()
    monkeypatch.setattr(groundtruth, "_load_native_distance_library", lambda: fake_native)

    result = exact_topk(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
        [[0.2, 0.0], [9.0, 0.0]],
        k=2,
        filter_mask=[True, True, False, True],
        block_rows=2,
    )

    assert result.native_used is True
    assert fake_native.calls == 4
    assert result.indices.tolist() == [[0, 1], [3, 1]]
    assert result.distances.tolist()[0] == pytest.approx([0.2, 0.8])
    assert result.distances.tolist()[1] == pytest.approx([1.0, 8.0])


def test_exact_topk_converts_native_inner_product_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_native = _FakeNativeLibrary()
    monkeypatch.setattr(groundtruth, "_load_native_distance_library", lambda: fake_native)

    result = exact_topk(
        [[1.0, 0.0], [3.0, 0.0], [-1.0, 0.0]],
        [[1.0, 0.0]],
        k=2,
        metric="ip",
        block_rows=3,
    )

    assert result.native_used is True
    assert result.indices.tolist() == [[1, 0]]
    assert result.distances.tolist()[0] == pytest.approx([-3.0, -1.0])


def test_exact_topk_falls_back_when_native_topk_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_native = _FailingNativeLibrary()
    monkeypatch.setattr(groundtruth, "_load_native_distance_library", lambda: failing_native)

    result = exact_topk(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
        [[0.2, 0.0]],
        k=2,
        block_rows=2,
    )

    assert failing_native.calls > 0
    assert result.native_used is False
    assert result.indices.tolist() == [[0, 1]]
    assert result.distances.tolist()[0] == pytest.approx([0.2, 0.8])


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

    result = groundtruth.exact_topk(base, queries, k=5, block_rows=128, use_native=False)

    assert result.blocks_scanned == 8
    assert len(seen_blocks) == 3 * 8


class _FakeNativeLibrary:
    def __init__(self) -> None:
        self.calls = 0

    def topk(self, query: object, corpus: object, *, k: int, metric: str) -> NativeTopKResult:
        self.calls += 1
        np = pytest.importorskip("numpy")
        query_array = np.asarray(query, dtype="float32")
        corpus_array = np.asarray(corpus, dtype="float32")
        if metric == "l2":
            delta = corpus_array - query_array
            scores = np.einsum("ij,ij->i", delta, delta, optimize=True)
            order = np.argsort(scores, kind="stable")[:k]
            return NativeTopKResult(
                indices=order,
                distances=scores[order],
                count=int(order.shape[0]),
            )
        if metric == "ip":
            scores = corpus_array @ query_array
            order = np.argsort(-scores, kind="stable")[:k]
            return NativeTopKResult(
                indices=order,
                distances=scores[order],
                count=int(order.shape[0]),
            )
        if metric == "cosine":
            numerator = corpus_array @ query_array
            corpus_norm = np.maximum(np.linalg.norm(corpus_array, axis=1), 1e-12)
            query_norm = max(float(np.linalg.norm(query_array)), 1e-12)
            distances = 1.0 - numerator / (corpus_norm * query_norm)
            order = np.argsort(distances, kind="stable")[:k]
            return NativeTopKResult(
                indices=order,
                distances=distances[order],
                count=int(order.shape[0]),
            )
        raise ValueError(f"unsupported metric: {metric}")


class _FailingNativeLibrary:
    def __init__(self) -> None:
        self.calls = 0

    def topk(self, query: object, corpus: object, *, k: int, metric: str) -> NativeTopKResult:
        del query, corpus, k, metric
        self.calls += 1
        raise NativeDistanceError("synthetic native failure")
