from __future__ import annotations

import pytest

from vecadvisor import native_distance


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
