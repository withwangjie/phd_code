import copy
import json
from pathlib import Path

import pytest
import yaml

from scripts import amend_failed_egnn_fp32 as recovery


def _fixture(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "old"
    run.mkdir(parents=True)
    old = {
        "paths": {"repo_root": str(tmp_path), "run_root": str(tmp_path / "runs"),
                  "checkpoint_dir": "checkpoints"},
        "egnn_train": {"amp": True},
    }
    new = copy.deepcopy(old)
    new["egnn_train"]["amp"] = False
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(new), encoding="utf-8")
    (run / "run_manifest.json").write_text(json.dumps({"config": old, "code_sha256": {"old": "hash"}}))
    (run / "frozen_config.yaml").write_text(yaml.safe_dump(old), encoding="utf-8")
    (run / "seed_streams.json").write_text("{}", encoding="utf-8")
    for stage, status in (("env_check", "completed"), ("data_audit", "completed"),
                          ("queue_freeze", "completed"), ("egnn_train", "failed")):
        path = run / "stage_status" / f"{stage}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"status": status}))
        if status == "completed":
            result = run / "results_manifests" / f"{stage}.json"
            result.parent.mkdir(exist_ok=True)
            result.write_text("{}")
    checkpoint = run / "checkpoints" / "last_egnn_pruning.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"partial fp16 checkpoint")
    monkeypatch.setattr(recovery, "verify_stream_map", lambda _: True)
    monkeypatch.setattr(recovery.Orchestrator, "_validate_completed_stage_artifacts",
                        lambda *_: (True, "verified"))
    monkeypatch.setattr(recovery.Orchestrator, "_validate_upstream_chain_fresh",
                        lambda *_: (True, "verified"))
    monkeypatch.setattr(recovery, "build_run_manifest",
                        lambda cfg, _: {"config": cfg, "code_sha256": {"new": "hash"}})
    return run, config_path, checkpoint


def test_fp32_amendment_preserves_queue_and_quarantines_fp16_checkpoint(tmp_path, monkeypatch):
    run, config_path, checkpoint = _fixture(tmp_path, monkeypatch)
    amendment_path = recovery.amend(run, config_path)
    amendment = json.loads(amendment_path.read_text())
    assert amendment["changed_config_field"] == "egnn_train.amp"
    assert not checkpoint.exists()
    assert (amendment_path.parent / "failed_fp16_checkpoints" / checkpoint.name).read_bytes() == b"partial fp16 checkpoint"
    assert json.loads((run / "run_manifest.json").read_text())["config"]["egnn_train"]["amp"] is False
    assert (run / "stage_status" / "queue_freeze.json").is_file()
    with pytest.raises(ValueError, match="inapplicable"):
        recovery.amend(run, config_path)


def test_fp32_amendment_rejects_other_config_changes_before_mutation(tmp_path, monkeypatch):
    run, config_path, checkpoint = _fixture(tmp_path, monkeypatch)
    changed = yaml.safe_load(config_path.read_text())
    changed["egnn_train"]["batch_size"] = 8
    config_path.write_text(yaml.safe_dump(changed))
    with pytest.raises(ValueError, match="Only egnn_train.amp"):
        recovery.amend(run, config_path)
    assert checkpoint.is_file()
    assert json.loads((run / "run_manifest.json").read_text())["config"]["egnn_train"]["amp"] is True


def test_fp32_amendment_rejects_unverified_queue(tmp_path, monkeypatch):
    run, config_path, checkpoint = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery.Orchestrator, "_validate_completed_stage_artifacts",
                        lambda _, stage: (False, "changed") if stage == "queue_freeze" else (True, "ok"))
    with pytest.raises(ValueError, match="queue_freeze artifacts failed"):
        recovery.amend(run, config_path)
    assert checkpoint.is_file()
