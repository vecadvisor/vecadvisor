from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from vecadvisor.bench import groundtruth
from vecadvisor.bench.datasets import (
    SyntheticDataset,
    SyntheticQueries,
    generate_synthetic_dataset,
    generate_synthetic_queries,
)
from vecadvisor.bench.groundtruth_compare import (
    groundtruth_comparison_to_json,
    run_groundtruth_comparison,
    write_groundtruth_comparison_report,
)
from vecadvisor.cli import app
from vecadvisor.native_distance import NativeKernelCapabilities, NativeTopKResult


def test_groundtruth_comparison_reports_native_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_native = _FakeNativeLibrary()
    monkeypatch.setattr(groundtruth, "_load_native_distance_library", lambda: fake_native)
    dataset, queries = _small_dataset()

    report = run_groundtruth_comparison(
        dataset=dataset,
        queries=queries,
        k=3,
        metric="l2",
        block_rows=8,
        iterations=1,
    )

    assert fake_native.calls > 0
    assert report.native.result.native_used is True
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.exact_index_match is True
    assert report.max_abs_distance_delta == pytest.approx(0.0, abs=1e-5)
    assert report.speedup is not None

    payload = groundtruth_comparison_to_json(report)
    assert payload["comparison"]["recall_at_k"] == pytest.approx(1.0)
    native_ground_truth = payload["native"]["ground_truth"]
    assert native_ground_truth["native_capabilities"] == _fake_capabilities().to_json()
    assert any("native run used" in note for note in payload["notes"])


def test_groundtruth_comparison_degrades_when_native_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(groundtruth, "_load_native_distance_library", lambda: None)
    dataset, queries = _small_dataset()

    report = run_groundtruth_comparison(
        dataset=dataset,
        queries=queries,
        k=3,
        block_rows=8,
        iterations=1,
    )

    assert report.native.result.native_used is False
    assert report.speedup is None
    assert report.recall_at_k == pytest.approx(1.0)
    assert any("fell back to NumPy" in note for note in report.notes)

    with pytest.raises(ValueError, match="native distance library was required"):
        run_groundtruth_comparison(
            dataset=dataset,
            queries=queries,
            k=3,
            block_rows=8,
            iterations=1,
            require_native=True,
        )


def test_write_groundtruth_comparison_report(tmp_path: Path) -> None:
    dataset, queries = _small_dataset()
    out_path = tmp_path / "groundtruth.json"
    report = run_groundtruth_comparison(
        dataset=dataset,
        queries=queries,
        k=3,
        block_rows=8,
        iterations=1,
    )

    write_groundtruth_comparison_report(report, out_path)

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["dataset"]["id"] == "synthetic"
    assert payload["comparison"]["recall_at_k"] == pytest.approx(1.0)


def test_benchmark_groundtruth_cli_outputs_json_and_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(groundtruth, "_load_native_distance_library", lambda: None)
    out_path = tmp_path / "groundtruth.json"

    result = CliRunner().invoke(
        app,
        [
            "benchmark-groundtruth",
            "--rows",
            "48",
            "--dim",
            "4",
            "--queries",
            "2",
            "--clusters",
            "4",
            "--filter-selectivity",
            "0.5",
            "--limit",
            "3",
            "--block-rows",
            "12",
            "--iterations",
            "1",
            "--out",
            str(out_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["dataset"]["rows"] == 48
    assert payload["numpy"]["ground_truth"]["native_used"] is False
    assert payload["native"]["native_requested"] is True
    assert payload["native"]["ground_truth"]["native_used"] is False
    assert payload["comparison"]["recall_at_k"] == pytest.approx(1.0)
    assert payload["output"] == {"path": str(out_path), "format": "json"}
    assert json.loads(out_path.read_text(encoding="utf-8"))["dataset"]["rows"] == 48


def _small_dataset() -> tuple[SyntheticDataset, SyntheticQueries]:
    dataset = generate_synthetic_dataset(
        n_rows=64,
        dim=4,
        n_clusters=4,
        filter_selectivity=0.5,
        correlation=0.0,
        seed=10,
    )
    queries = generate_synthetic_queries(dataset, n_queries=3, seed=11)
    return dataset, queries


class _FakeNativeLibrary:
    def __init__(self) -> None:
        self.calls = 0
        self.source = "fake-native"

    def capabilities(self) -> NativeKernelCapabilities:
        return _fake_capabilities()

    def topk(self, query: object, corpus: object, *, k: int, metric: str) -> NativeTopKResult:
        self.calls += 1
        np = pytest.importorskip("numpy")
        query_array = np.asarray(query, dtype="float32")
        corpus_array = np.asarray(corpus, dtype="float32")
        if metric == "l2":
            delta = corpus_array - query_array
            distances = np.einsum("ij,ij->i", delta, delta, optimize=True)
            order = np.argsort(distances, kind="stable")[:k]
            return NativeTopKResult(
                indices=order,
                distances=distances[order],
                count=int(order.shape[0]),
            )
        raise ValueError(f"unsupported metric: {metric}")


def _fake_capabilities() -> NativeKernelCapabilities:
    return NativeKernelCapabilities(
        source="fake-native",
        avx2_compiled=True,
        avx2_runtime=True,
        l2_kernel="fake-avx2",
        inner_product_kernel="fake-avx2",
        cosine_kernel="fake-avx2",
    )
