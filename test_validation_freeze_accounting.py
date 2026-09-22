"""Regression tests for immutable validation freeze/execution accounting."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from run_real_complex_pilot import _load_frozen_targets, _load_frozen_target_ids, _reconcile_frozen_targets


def test_load_frozen_target_ids_accepts_selected_targets_schema(tmp_path: Path) -> None:
    path = tmp_path / "selected_targets.json"
    path.write_text(json.dumps([
        {"target": "4ABC", "source_id": "src-a", "graph_path": "graphs/test/a.pt", "graph_sha256": "a"*64},
        {"pdb_id": "5Def", "source_id": "src-b", "graph_path": "graphs/test/b.pt", "graph_sha256": "b"*64},
    ]), encoding="utf-8")
    assert _load_frozen_target_ids(path) == ["4abc", "5def"]


def test_load_frozen_target_ids_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "selected_targets.json"
    path.write_text(json.dumps(["4abc", "4ABC"]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _load_frozen_target_ids(path)


def test_reconcile_frozen_targets_keeps_preexecution_failures_in_denominator() -> None:
    completed, failed, closed = _reconcile_frozen_targets(
        ["4abc", "5def", "6ghi"],
        {"4abc", "5def"},
        {"5def"},
    )
    assert completed == ["4abc"]
    assert failed == ["5def", "6ghi"]
    assert closed is True
    assert set(completed) | set(failed) == {"4abc", "5def", "6ghi"}
    assert not (set(completed) & set(failed))


def test_reconcile_frozen_targets_rejects_target_outside_freeze() -> None:
    with pytest.raises(ValueError, match="outside frozen set"):
        _reconcile_frozen_targets(["4abc"], {"4abc", "9xyz"}, set())


def test_reconcile_frozen_targets_rejects_failure_not_selected() -> None:
    with pytest.raises(ValueError, match="outside execution-selected set"):
        _reconcile_frozen_targets(["4abc", "5def"], {"4abc"}, {"5def"})


def test_frozen_target_records_preserve_exact_graph_identity(tmp_path: Path) -> None:
    path = tmp_path / "selected_targets.json"
    path.write_text(json.dumps([{
        "target": "4ABC",
        "source_id": "source-1",
        "graph_path": "graphs\\test_snac_hard\\sample.pt",
        "graph_sha256": "ABCDEF",
    }]), encoding="utf-8")
    records = _load_frozen_targets(path)
    assert records == [{
        "target": "4abc",
        "source_id": "source-1",
        "graph_path": "graphs/test_snac_hard/sample.pt",
        "graph_sha256": "abcdef",
        "legacy_pdb_only": False,
    }]


def test_structured_frozen_target_requires_graph_identity(tmp_path: Path) -> None:
    path = tmp_path / "selected_targets.json"
    path.write_text(json.dumps([{"target": "4ABC", "source_id": "source-1"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="graph identity"):
        _load_frozen_targets(path)
