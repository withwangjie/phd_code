"""Audited, one-time continuation of a failed FP16 EGNN stage with FP32.

This is deliberately separate from the ordinary resume path. Completed audit
and queue artifacts are verified before their frozen run manifest is amended.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import yaml

from nanoqc.common.seed_streams import verify_stream_map
from nanoqc.pipeline.run_full_experiment import (
    Orchestrator,
    STAGE_ORDER,
    atomic_write_json,
    build_run_manifest,
    sha256_of,
    utc_timestamp,
)


def amend(run_dir: Path, config_path: Path) -> Path:
    run_dir = run_dir.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = Path(config["paths"]["repo_root"]).resolve()
    run_root = Path(config["paths"]["run_root"]).resolve()
    if not run_dir.is_relative_to(run_root):
        raise ValueError(f"Run directory is outside the configured run root: {run_dir}")
    manifest_path = run_dir / "run_manifest.json"
    frozen_path = run_dir / "frozen_config.yaml"
    previous_bytes = manifest_path.read_bytes()
    previous = json.loads(previous_bytes)
    old_config = previous["config"]
    expected_config = copy.deepcopy(old_config)
    if expected_config["egnn_train"].get("amp") is not True:
        raise ValueError("Original run did not use FP16 AMP; FP32 amendment is inapplicable")
    expected_config["egnn_train"]["amp"] = False
    if config != expected_config:
        raise ValueError("Only egnn_train.amp: true -> false may change in this run")
    if yaml.safe_load(frozen_path.read_text(encoding="utf-8")) != old_config:
        raise ValueError("Frozen config does not match the original run manifest")
    seed_path = run_dir / "seed_streams.json"
    if not seed_path.is_file() or not verify_stream_map(seed_path):
        raise ValueError("Frozen seed streams are missing or invalid")

    statuses = {}
    for stage in STAGE_ORDER:
        path = run_dir / "stage_status" / f"{stage}.json"
        statuses[stage] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    for stage in ("env_check", "data_audit", "queue_freeze"):
        if not statuses[stage] or statuses[stage]["status"] not in ("completed", "completed_with_failures"):
            raise ValueError(f"Required frozen stage {stage} is not completed")
    if not statuses["egnn_train"] or statuses["egnn_train"]["status"] != "failed":
        raise ValueError("EGNN stage must have failed before this amendment")
    for stage in STAGE_ORDER[STAGE_ORDER.index("egnn_train") + 1:]:
        if statuses[stage] and statuses[stage]["status"] in ("completed", "completed_with_failures"):
            raise ValueError(f"Downstream stage {stage} is already completed")

    verifier = Orchestrator(old_config, run_dir)
    for stage in ("env_check", "smoke_check", "data_audit", "queue_freeze"):
        if not statuses[stage] or statuses[stage]["status"] not in ("completed", "completed_with_failures"):
            continue
        valid, detail = verifier._validate_completed_stage_artifacts(stage)
        if not valid:
            raise ValueError(f"Frozen {stage} artifacts failed verification: {detail}")
        valid, detail = verifier._validate_upstream_chain_fresh(stage)
        if not valid:
            raise ValueError(f"Frozen {stage} dependency chain is stale: {detail}")

    current = build_run_manifest(config, repo_root)
    amendment_dir = run_dir / "provenance" / "fp32_egnn_amendment"
    if amendment_dir.exists():
        raise ValueError(f"FP32 amendment already exists: {amendment_dir}")
    checkpoint_dir = verifier.checkpoint_dir().resolve()
    if not checkpoint_dir.is_relative_to(run_dir):
        raise ValueError("Checkpoint directory escapes this run")
    checkpoint_files = []
    if checkpoint_dir.exists():
        checkpoint_files = [
            {"path": p.relative_to(checkpoint_dir).as_posix(), "sha256": sha256_of(p)}
            for p in sorted(checkpoint_dir.rglob("*")) if p.is_file()
        ]

    amendment_dir.mkdir(parents=True)
    (amendment_dir / "original_run_manifest.json").write_bytes(previous_bytes)
    shutil.copy2(frozen_path, amendment_dir / "original_frozen_config.yaml")
    shutil.copy2(config_path, amendment_dir / "fp32_resolved_config.yaml")
    if checkpoint_dir.exists():
        shutil.move(str(checkpoint_dir), str(amendment_dir / "failed_fp16_checkpoints"))
    amendment = {
        "amended_utc": utc_timestamp(),
        "reason": "EGNN FP16 AMP failed; continue this frozen queue with CUDA FP32",
        "changed_config_field": "egnn_train.amp",
        "old_value": True,
        "new_value": False,
        "original_run_manifest_sha256": sha256_of(amendment_dir / "original_run_manifest.json"),
        "original_frozen_config_sha256": sha256_of(amendment_dir / "original_frozen_config.yaml"),
        "fp32_resolved_config_sha256": sha256_of(amendment_dir / "fp32_resolved_config.yaml"),
        "old_code_sha256": previous.get("code_sha256"),
        "new_code_sha256": current.get("code_sha256"),
        "frozen_stage_results_sha256": {
            stage: sha256_of(run_dir / "results_manifests" / f"{stage}.json")
            for stage in ("env_check", "data_audit", "queue_freeze")
        },
        "quarantined_checkpoint_files": checkpoint_files,
        "amendment_tool_sha256": sha256_of(Path(__file__)),
    }
    atomic_write_json(amendment_dir / "amendment.json", amendment)
    current["fp32_egnn_amendment"] = str((amendment_dir / "amendment.json").relative_to(run_dir))
    atomic_write_json(manifest_path, current)
    frozen_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return amendment_dir / "amendment.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"FP32 EGNN amendment recorded: {amend(args.run_dir, args.config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
