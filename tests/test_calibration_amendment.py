import copy
import json

import pytest
import yaml

from scripts import amend_failed_calibration as recovery


def _run(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "old"
    run.mkdir(parents=True)
    old = {
        "paths": {"repo_root": str(tmp_path), "run_root": str(tmp_path / "runs"),
                  "checkpoint_dir": "checkpoints"},
        "qc_benchmark": {"energy_calibration": {"acceptance": {"max_generation_failure_fraction": 0.1}}},
    }
    new = copy.deepcopy(old)
    new["qc_benchmark"]["energy_calibration"]["acceptance"]["max_input_quality_exclusion_fraction"] = None
    config_path = tmp_path / "resolved.yaml"
    config_path.write_text(yaml.safe_dump(new))
    old_hashes = {name: "old" for name in recovery.ALLOWED_CHANGED_SOURCES}
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROTOCOL_AMENDMENTS.md").write_text("A21")
    for name in old_hashes:
        (tmp_path / name).write_text(name)
    monkeypatch.setattr(recovery, "repo_path", lambda name, root: root / name)
    (run / "run_manifest.json").write_text(json.dumps({"config": old, "code_sha256": old_hashes}))
    (run / "frozen_config.yaml").write_text(yaml.safe_dump(old))
    (run / "seed_streams.json").write_text("{}")
    for stage in recovery.STAGE_ORDER:
        status = "completed" if stage in ("env_check", "smoke_check", "data_audit", "queue_freeze", "egnn_train") else "failed"
        path = run / "stage_status" / f"{stage}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps({"status": status}))
        if status == "completed":
            result = run / "results_manifests" / f"{stage}.json"
            result.parent.mkdir(exist_ok=True)
            result.write_text("{}")
    (run / "calibration").mkdir()
    (run / "calibration" / "old.csv").write_text("old")
    (run / "validation_queue" / "freeze").mkdir(parents=True)
    (run / "validation_queue" / "freeze" / "queue.json").write_text("frozen")
    (run / "validation_queue" / "results").mkdir()
    (run / "validation_queue" / "results" / "old.csv").write_text("old")
    monkeypatch.setattr(recovery, "verify_stream_map", lambda _: True)
    monkeypatch.setattr(recovery.Orchestrator, "_validate_completed_stage_artifacts",
                        lambda *_: (True, "verified"))
    monkeypatch.setattr(recovery.Orchestrator, "_validate_upstream_chain_fresh",
                        lambda *_: (True, "verified"))
    monkeypatch.setattr(recovery, "build_run_manifest", lambda cfg, _: {
        "config": cfg, "code_sha256": {name: "new" for name in old_hashes},
    })
    return run, config_path


def test_calibration_amendment_keeps_upstream_and_archives_downstream(tmp_path, monkeypatch):
    run, config_path = _run(tmp_path, monkeypatch)
    path = recovery.amend(run, config_path)
    archive = path.parent / "superseded_downstream"
    assert (run / "validation_queue" / "freeze" / "queue.json").read_text() == "frozen"
    assert (run / "stage_status" / "egnn_train.json").is_file()
    assert not (run / "stage_status" / "energy_calibration.json").exists()
    assert (archive / "calibration" / "old.csv").read_text() == "old"
    assert (archive / "validation_queue" / "results" / "old.csv").read_text() == "old"
    assert json.loads((run / "run_manifest.json").read_text())["config"]["qc_benchmark"]["energy_calibration"]["acceptance"]["max_input_quality_exclusion_fraction"] is None


def test_calibration_amendment_rejects_unrelated_code_change(tmp_path, monkeypatch):
    run, config_path = _run(tmp_path, monkeypatch)
    monkeypatch.setattr(recovery, "build_run_manifest", lambda cfg, _: {
        "config": cfg, "code_sha256": {"train_egnn_pruning.py": "new"},
    })
    with pytest.raises(ValueError, match="Unexpected scientific source changes"):
        recovery.amend(run, config_path)
    assert (run / "calibration" / "old.csv").is_file()
