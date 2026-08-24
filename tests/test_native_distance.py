from __future__ import annotations

import ctypes

import pytest

from vecadvisor import native_distance
from vecadvisor.native_distance import NativeDistanceLibrary


def test_default_native_loader_returns_none_when_library_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_distance.load_default_native_distance_library.cache_clear()
    monkeypatch.delenv(native_distance.NATIVE_DISTANCE_LIB_ENV, raising=False)
    monkeypatch.setattr(native_distance.ctypes.util, "find_library", lambda name: None)

    assert native_distance.load_default_native_distance_library() is None

    native_distance.load_default_native_distance_library.cache_clear()


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


class _FakeCFunction:
    def __init__(self, callback: object) -> None:
        self._callback = callback
        self.argtypes: object = None
        self.restype: object = None

    def __call__(self, *args: object) -> object:
        return self._callback(*args)


class _FakeNativeCAbi:
    def __init__(self) -> None:
        self.vecadvisor_distance_get_capabilities = _FakeCFunction(self._capabilities)
        self.vecadvisor_distance_topk = _FakeCFunction(lambda *args: 0)

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
