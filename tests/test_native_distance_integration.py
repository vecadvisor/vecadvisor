from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from vecadvisor import native_distance
from vecadvisor.bench.groundtruth import exact_topk
from vecadvisor.native_distance import NativeDistanceLibrary


@pytest.fixture()
def native_library_path() -> Iterator[Path]:
    raw_path = os.environ.get(native_distance.NATIVE_DISTANCE_LIB_ENV)
    if raw_path is None:
        pytest.skip(
            f"{native_distance.NATIVE_DISTANCE_LIB_ENV} is not set; "
            "skipping native shared-library integration test"
        )
    path = Path(raw_path)
    if not path.exists():
        pytest.fail(f"{native_distance.NATIVE_DISTANCE_LIB_ENV} does not exist: {path}")
    native_distance.load_default_native_distance_library.cache_clear()
    yield path
    native_distance.load_default_native_distance_library.cache_clear()


def test_native_topk_c_abi_matches_expected_distances(native_library_path: Path) -> None:
    np = pytest.importorskip("numpy")
    library = NativeDistanceLibrary.load(native_library_path)
    capabilities = library.capabilities()

    assert capabilities.source == str(native_library_path)
    assert capabilities.l2_kernel in {"avx2", "scalar"}
    assert capabilities.inner_product_kernel in {"avx2", "scalar"}
    assert capabilities.cosine_kernel in {"avx2", "scalar"}
    assert capabilities.avx2_runtime <= capabilities.avx2_compiled

    l2 = library.topk(
        np.asarray([0.2, 0.0], dtype="float32"),
        np.asarray([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype="float32"),
        k=2,
        metric="l2",
    )
    assert l2.count == 2
    assert l2.indices.tolist() == [0, 1]
    assert l2.distances.tolist() == pytest.approx([0.04, 0.64])

    inner_product = library.topk(
        np.asarray([1.0, 0.0], dtype="float32"),
        np.asarray([[1.0, 0.0], [3.0, 0.0], [-1.0, 0.0]], dtype="float32"),
        k=2,
        metric="ip",
    )
    assert inner_product.count == 2
    assert inner_product.indices.tolist() == [1, 0]
    assert inner_product.distances.tolist() == pytest.approx([3.0, 1.0])


def test_exact_topk_uses_real_native_shared_library(
    monkeypatch: pytest.MonkeyPatch,
    native_library_path: Path,
) -> None:
    monkeypatch.setenv(native_distance.NATIVE_DISTANCE_LIB_ENV, str(native_library_path))
    native_distance.load_default_native_distance_library.cache_clear()

    result = exact_topk(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
        [[0.2, 0.0], [9.0, 0.0]],
        k=2,
        filter_mask=[True, True, False, True],
        block_rows=2,
    )

    assert result.native_used is True
    assert result.native_library_source == str(native_library_path)
    assert result.native_capabilities is not None
    assert result.indices.tolist() == [[0, 1], [3, 1]]
    assert result.distances.tolist()[0] == pytest.approx([0.2, 0.8])
    assert result.distances.tolist()[1] == pytest.approx([1.0, 8.0])
