from __future__ import annotations

import ctypes
import ctypes.util
import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

NATIVE_DISTANCE_LIB_ENV = "VECADVISOR_NATIVE_DISTANCE_LIB"

_STATUS_OK = 0
_METRIC_CODES = {
    "l2": 1,
    "ip": 2,
    "cosine": 3,
}


class NativeDistanceError(RuntimeError):
    """Raised when the optional native distance library rejects a call."""


@dataclass(frozen=True)
class NativeTopKResult:
    indices: Any
    distances: Any
    count: int


class NativeDistanceLibrary:
    """ctypes wrapper around the VecAdvisor native distance C ABI."""

    def __init__(self, library: ctypes.CDLL, *, source: str) -> None:
        self._library = library
        self.source = source
        self._configure_abi()

    @classmethod
    def load(cls, path: str | Path) -> NativeDistanceLibrary:
        raw_path = str(path)
        return cls(ctypes.CDLL(raw_path), source=raw_path)

    def topk(self, query: Any, corpus: Any, *, k: int, metric: str) -> NativeTopKResult:
        if k <= 0:
            raise ValueError("k must be positive")
        metric_code = _metric_code(metric)
        np = _numpy()
        query_array = np.ascontiguousarray(query, dtype="float32")
        corpus_array = np.ascontiguousarray(corpus, dtype="float32")
        if query_array.ndim != 1:
            raise ValueError("query must be one-dimensional")
        if corpus_array.ndim != 2:
            raise ValueError("corpus must be two-dimensional")
        rows = int(corpus_array.shape[0])
        dim = int(corpus_array.shape[1])
        if rows <= 0 or dim <= 0:
            raise ValueError("corpus must be non-empty")
        if int(query_array.shape[0]) != dim:
            raise ValueError("query and corpus dimensions must match")

        out_indices = np.empty(k, dtype=np.uintp)
        out_distances = np.empty(k, dtype=np.float32)
        out_count = ctypes.c_size_t(0)
        status = self._library.vecadvisor_distance_topk(
            metric_code,
            query_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            corpus_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(rows),
            ctypes.c_size_t(dim),
            ctypes.c_size_t(k),
            out_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t)),
            out_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(out_count),
        )
        if int(status) != _STATUS_OK:
            raise NativeDistanceError(
                f"vecadvisor_distance_topk failed with status {int(status)}"
            )
        count = int(out_count.value)
        return NativeTopKResult(
            indices=out_indices[:count].astype("int64", copy=True),
            distances=out_distances[:count].astype("float64", copy=True),
            count=count,
        )

    def _configure_abi(self) -> None:
        topk = self._library.vecadvisor_distance_topk
        topk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        topk.restype = ctypes.c_int


@lru_cache(maxsize=1)
def load_default_native_distance_library() -> NativeDistanceLibrary | None:
    """Load the optional native distance library, returning None when unavailable."""

    configured_path = os.environ.get(NATIVE_DISTANCE_LIB_ENV)
    candidates: list[str] = []
    if configured_path:
        candidates.append(configured_path)
    else:
        found = ctypes.util.find_library("vecadvisor_distance")
        if found is not None:
            candidates.append(found)

    for candidate in candidates:
        try:
            return NativeDistanceLibrary.load(candidate)
        except OSError:
            continue
    return None


def _metric_code(metric: str) -> int:
    try:
        return _METRIC_CODES[metric]
    except KeyError as exc:
        valid = ", ".join(sorted(_METRIC_CODES))
        raise ValueError(f"metric must be one of: {valid}") from exc


def _numpy() -> Any:
    return importlib.import_module("numpy")
