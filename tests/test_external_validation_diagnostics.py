"""The independence audit always leaves a manifest that names its failure."""
import json
import sys
from pathlib import Path

import pytest

from nanoqc.data import audit_external_vhh_independence as audit
from nanoqc.pipeline.stages_structure import _baseline_failure_detail, _independence_failure_detail


def _run(tmp_path: Path, monkeypatch, manifest_rows: list) -> Path:
    dataset=tmp_path/"dataset"; dataset.mkdir()
    (dataset/"graph_manifest.json").write_text(json.dumps(manifest_rows))
    graphs=tmp_path/"graphs"; graphs.mkdir()
    sources=tmp_path/"sources"; sources.mkdir()
    cluster=tmp_path/"clusters.json"; cluster.write_text(json.dumps({"1abc":"c1"}))
    out=tmp_path/"out"/"manifest.json"
    monkeypatch.setattr(sys,"argv",["audit","--training-dataset",str(dataset),
        "--external-graph-dir",str(graphs),"--external-source-dir",str(sources),
        "--cluster-map",str(cluster),"--out",str(out)])
    return out


def test_fatal_error_still_writes_manifest(tmp_path, monkeypatch):
    out=_run(tmp_path,monkeypatch,[])
    with pytest.raises(ValueError,match="No training rows"):
        audit.main()
    manifest=json.loads(out.read_text())
    assert "No training rows" in manifest["fatal_error"]
    assert manifest["training_family_overlap_zero"] is False
    assert "No training rows" in _independence_failure_detail(out)


def test_unverifiable_target_is_a_failed_row(tmp_path):
    row,version=audit._external_target_audit((tmp_path/"9xyz.pt",tmp_path,[],{},set(),(0.8,0.5,0.3,0.7)))
    assert version=="" and row["passes"] is False and row["family_cluster_overlap"] is True
    assert row["pdb_id"]=="9xyz" and row["error"]


def test_failure_details(tmp_path):
    manifest=tmp_path/"m.json"
    manifest.write_text(json.dumps(dict(target_count=4,failed_targets=["a","b"],
                                        errored_targets={"b":"ValueError: CDR-H3 absent"})))
    detail=_independence_failure_detail(manifest)
    assert "2/4" in detail and "CDR-H3 absent" in detail
    summary=tmp_path/"s.json"
    summary.write_text(json.dumps(dict(failures=[dict(target="7wki",seed="42",
        error="RuntimeError: Missing final relaxed structure for qaoa\nmore")])))
    assert "7wki/seed_42: RuntimeError: Missing final relaxed structure for qaoa" in _baseline_failure_detail(summary)
    assert _baseline_failure_detail(tmp_path/"absent.json")==""
