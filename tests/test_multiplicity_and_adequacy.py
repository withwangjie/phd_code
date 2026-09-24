"""Serial gatekeeping (A4) and outcome-free cluster adequacy (A5)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoqc.inference.paired_statistics import serial_gatekeeping
from nanoqc.pipeline.run_full_experiment import cluster_adequacy


def test_serial_gatekeeping_holm_within_primary_and_gated_secondary():
    adjusted = serial_gatekeeping({"qc": 0.01, "scaling": 0.04}, {"a": 0.001, "b": 0.2})
    assert adjusted["qc"] == pytest.approx(0.02)
    assert adjusted["scaling"] == pytest.approx(0.04)
    # Secondary Holm (0.002, 0.2) floored at the largest primary adjusted p.
    assert adjusted["a"] == pytest.approx(0.04)
    assert adjusted["b"] == pytest.approx(0.2)
    # A non-rejected primary hypothesis closes the gate for everything behind it.
    closed = serial_gatekeeping({"qc": 0.01, "scaling": 0.3}, {"a": 0.001})
    assert closed["a"] >= 0.3
    # A missing primary p leaves the secondary family untestable.
    missing = serial_gatekeeping({"qc": 0.01, "scaling": None}, {"a": 0.001})
    assert missing["scaling"] is None and missing["a"] is None


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cluster_adequacy_counts_independent_clusters_before_any_outcome(tmp_path: Path):
    manifest = [dict(split="test_snac_hard", pdb_id=f"H{i}") for i in range(6)] + [dict(split="train", pdb_id="T0")]
    _write(tmp_path / "dataset" / "graph_manifest.json", manifest)
    selected = _write(tmp_path / "selected.json", [dict(target=f"v{i}") for i in range(4)])
    cluster_map = {f"h{i}": f"c{i // 2}" for i in range(6)}          # 3 hard clusters
    cluster_map.update({f"v{i}": f"vc{i}" for i in range(4)})         # 4 validation clusters
    clusters = _write(tmp_path / "clusters.json", cluster_map)
    config = {"statistics": {"min_qc_clusters": 3, "min_scaling_clusters": 3,
                             "min_primary_clusters": 4, "min_rq5_clusters": 4}}
    report = cluster_adequacy(config, tmp_path / "dataset", selected, clusters)
    assert report["adequate"] and report["outcome_free"]
    assert report["test_snac_hard_clusters"] == 3 and report["validation_queue_clusters"] == 4
    config["statistics"]["min_primary_clusters"] = 10
    short = cluster_adequacy(config, tmp_path / "dataset", selected, clusters)
    assert not short["adequate"]
    assert any("structural primary endpoint" in s for s in short["shortfalls"])
