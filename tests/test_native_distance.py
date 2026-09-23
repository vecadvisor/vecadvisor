from __future__ import annotations

import ctypes
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vecadvisor import native_distance
from vecadvisor.cli import app
from vecadvisor.native_distance import NativeDistanceLibrary


def test_default_native_loader_returns_none_when_library_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_distance.load_default_native_distance_library.cache_clear()
    monkeypatch.delenv(native_distance.NATIVE_DISTANCE_LIB_ENV, raising=False)
    monkeypatch.setattr(native_distance.ctypes.util, "find_library", lambda name: None)

    assert native_distance.load_default_native_distance_library() is None

    native_distance.load_default_native_distance_library.cache_clear()


def test_native_info_cli_reports_unavailable_when_library_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_distance.load_default_native_distance_library.cache_clear()
    monkeypatch.delenv(native_distance.NATIVE_DISTANCE_LIB_ENV, raising=False)
    monkeypatch.setattr(native_distance.ctypes.util, "find_library", lambda name: None)

    result = CliRunner().invoke(app, ["native-info"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["available"] is False
    assert payload["env_var"] == native_distance.NATIVE_DISTANCE_LIB_ENV
    assert payload["capabilities"] is None

    native_distance.load_default_native_distance_library.cache_clear()


def test_native_info_cli_reports_explicit_library_load_error(tmp_path: Path) -> None:
    missing_library = tmp_path / "missing-native-library.dll"

    result = CliRunner().invoke(app, ["native-info", "--library", str(missing_library)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["available"] is False
    assert payload["source_kind"] == "explicit"
    assert payload["source"] == str(missing_library)


def test_default_native_loader_ignores_unloadable_configured_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_distance.load_default_native_distance_library.cache_clear()
    monkeypatch.setenv(native_distance.NATIVE_DISTANCE_LIB_ENV, "missing-vecadvisor-native-lib")

    assert native_distance.load_default_native_distance_library() is None

    native_distance.load_default_native_distance_library.cache_clear()


def test_default_native_loader_ignores_incompatible_configured_library(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_distance.load_default_native_distance_library.cache_clear()
    monkeypatch.setenv(native_distance.NATIVE_DISTANCE_LIB_ENV, "old-native-library")
    monkeypatch.setattr(native_distance.NativeDistanceLibrary, "load", _raise_attribute_error)

    assert native_distance.load_default_native_distance_library() is None

    native_distance.load_default_native_distance_library.cache_clear()


def _raise_attribute_error(path: object) -> object:
    del path
    raise AttributeError("missing symbol")


def test_native_capabilities_decode_from_c_abi() -> None:
    library = NativeDistanceLibrary(_FakeNativeCAbi(), source="fake-library")  # type: ignore[arg-type]

    capabilities = library.capabilities()

    assert capabilities.source == "fake-library"
    assert capabilities.avx2_compiled is True
    assert capabilities.avx2_runtime is False
    assert capabilities.l2_kernel == "avx2"
    assert capabilities.inner_product_kernel == "scalar"
    assert capabilities.cosine_kernel == "scalar"
    assert capabilities.to_json() == {
        "source": "fake-library",
        "avx2_compiled": True,
        "avx2_runtime": False,
        "l2_kernel": "avx2",
        "inner_product_kernel": "scalar",
        "cosine_kernel": "scalar",
    }


def test_native_int8_compute_many_and_topk_decode_from_c_abi() -> None:
    np = pytest.importorskip("numpy")
    library = NativeDistanceLibrary(_FakeNativeCAbi(), source="fake-library")  # type: ignore[arg-type]

    query = np.asarray([0, 0], dtype=np.int8)
    corpus = np.asarray([[2, 0], [1, 0], [1, 0], [0, 3], [0, 0]], dtype=np.int8)

    distances = library.compute_many_int8(query, corpus, metric="l2")
    topk = library.topk_int8(query, corpus, k=3, metric="l2")

    assert distances.tolist() == pytest.approx([4.0, 1.0, 1.0, 9.0, 0.0])
    assert topk.count == 3
    assert topk.indices.tolist() == [4, 1, 2]
    assert topk.distances.tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_native_int8_methods_report_missing_optional_abi() -> None:
    library = NativeDistanceLibrary(_FakeNativeCAbi(include_int8=False), source="fake-library")  # type: ignore[arg-type]

    with pytest.raises(native_distance.NativeDistanceError, match="int8 compute_many ABI"):
        library.compute_many_int8([0, 0], [[0, 0]], metric="l2")

    with pytest.raises(native_distance.NativeDistanceError, match="int8 top-k ABI"):
        library.topk_int8([0, 0], [[0, 0]], k=1, metric="l2")


class _FakeCFunction:
    def __init__(self, callback: object) -> None:
        self._callback = callback
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        return self._callback(*args)


class _FakeNativeCAbi:
    def __init__(self, *, include_int8: bool = True) -> None:
        self.vecadvisor_distance_get_capabilities = _FakeCFunction(self._capabilities)
        self.vecadvisor_distance_topk = _FakeCFunction(lambda *args: 0)
        if include_int8:
            self.vecadvisor_distance_compute_many_i8 = _FakeCFunction(self._compute_many_i8)
            self.vecadvisor_distance_topk_i8 = _FakeCFunction(self._topk_i8)

    def _capabilities(self, out_pointer: object) -> int:
        out = ctypes.cast(
            out_pointer,
            ctypes.POINTER(native_distance._NativeKernelCapabilitiesStruct),
        ).contents
        out.avx2_compiled = 1
        out.avx2_runtime = 0
        out.l2_kernel = b"avx2"
        out.inner_product_kernel = b"scalar"
        out.cosine_kernel = b"scalar"
        return 0

    def _compute_many_i8(
        self,
        metric: object,
        query_pointer: object,
        corpus_pointer: object,
        rows_value: object,
        dim_value: object,
        out_pointer: object,
    ) -> int:
        np = pytest.importorskip("numpy")
        rows = int(rows_value.value)
        dim = int(dim_value.value)
        query = np.ctypeslib.as_array(query_pointer, shape=(dim,))
        corpus = np.ctypeslib.as_array(corpus_pointer, shape=(rows * dim,)).reshape(rows, dim)
        out = np.ctypeslib.as_array(out_pointer, shape=(rows,))
        distances = _int8_distances(np, int(metric), query, corpus)
        out[:] = distances.astype(np.float32)
        return 0

    def _topk_i8(
        self,
        metric: object,
        query_pointer: object,
        corpus_pointer: object,
        rows_value: object,
        dim_value: object,
        k_value: object,
        out_indices_pointer: object,
        out_distances_pointer: object,
        out_count_pointer: object,
    ) -> int:
        np = pytest.importorskip("numpy")
        rows = int(rows_value.value)
        dim = int(dim_value.value)
        k = int(k_value.value)
        query = np.ctypeslib.as_array(query_pointer, shape=(dim,))
        corpus = np.ctypeslib.as_array(corpus_pointer, shape=(rows * dim,)).reshape(rows, dim)
        distances = _int8_distances(np, int(metric), query, corpus)
        order = sorted(
            range(rows),
            key=lambda index: (
                -float(distances[index]) if int(metric) == 2 else float(distances[index]),
                index,
            ),
        )[: min(k, rows)]
        out_indices = np.ctypeslib.as_array(out_indices_pointer, shape=(k,))
        out_distances = np.ctypeslib.as_array(out_distances_pointer, shape=(k,))
        for output_index, row_index in enumerate(order):
            out_indices[output_index] = row_index
            out_distances[output_index] = distances[row_index]
        ctypes.cast(out_count_pointer, ctypes.POINTER(ctypes.c_size_t)).contents.value = len(order)
        return 0


def _int8_distances(np: object, metric: int, query: object, corpus: object) -> object:
    query_array = np.asarray(query, dtype=np.float32)
    corpus_array = np.asarray(corpus, dtype=np.float32)
    if metric == 1:
        delta = corpus_array - query_array
        return np.einsum("ij,ij->i", delta, delta, optimize=True)
    if metric == 2:
        return corpus_array @ query_array
    if metric == 3:
        numerator = corpus_array @ query_array
        corpus_norm = np.maximum(np.linalg.norm(corpus_array, axis=1), 1e-12)
        query_norm = max(float(np.linalg.norm(query_array)), 1e-12)
        return 1.0 - numerator / (corpus_norm * query_norm)
    raise AssertionError(f"unexpected metric: {metric}")
