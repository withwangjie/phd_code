#!/usr/bin/env python3
"""Reuse verified upstream stages after a documented calibration protocol repair."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

import yaml

from nanoqc.common.seed_streams import verify_stream_map
from nanoqc.pipeline.run_full_experiment import (
    Orchestrator, STAGE_ORDER, atomic_write_json, build_run_manifest,
    repo_path, sha256_of, utc_timestamp,
)


ALLOWED_CHANGED_SOURCES = {
    "subgraph_to_qubo.py",
    "generate_energy_calibration_dataset.py",
    "run_full_experiment.py",
}
DOWNSTREAM = STAGE_ORDER[STAGE_ORDER.index("energy_calibration"):]


def amend(run_dir: Path, config_path: Path) -> Path:
    run_dir = run_dir.resolve(strict=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_root = Path(config["paths"]["run_root"]).resolve()
    if not run_dir.is_relative_to(run_root):
        raise ValueError("Run directory is outside the configured run root")
    repo_root = Path(config["paths"]["repo_root"]).resolve()
    manifest_path = run_dir / "run_manifest.json"
    frozen_path = run_dir / "frozen_config.yaml"
    previous_bytes = manifest_path.read_bytes()
    previous = json.loads(previous_bytes)
    old_config = previous["config"]
    if yaml.safe_load(frozen_path.read_text(encoding="utf-8")) != old_config:
        raise ValueError("Frozen config does not match the original run manifest")
    expected_config = copy.deepcopy(old_config)
    old_acceptance = expected_config["qc_benchmark"]["energy_calibration"]["acceptance"]
    if "max_input_quality_exclusion_fraction" in old_acceptance:
        if old_acceptance["max_input_quality_exclusion_fraction"] != 0.40:
            raise ValueError("Unexpected existing input-quality exclusion limit")
        old_acceptance["max_input_quality_exclusion_fraction"] = None
    else:
        old_acceptance["max_input_quality_exclusion_fraction"] = None
    # A prior A21 amendment may already contain the old 0.40 field; A22
    # additionally introduces the predeclared usable-complexity floor.
    old_acceptance["min_eligible_complexes"] = 100
    if config != expected_config:
        raise ValueError("Only the calibrated input-quality exclusion limit may change")
    seed_path = run_dir / "seed_streams.json"
    if not seed_path.is_file() or not verify_stream_map(seed_path):
        raise ValueError("Frozen seed streams are missing or invalid")

    current = build_run_manifest(config, repo_root)
    old_code, new_code = previous.get("code_sha256"), current.get("code_sha256")
    if not isinstance(old_code, dict) or not isinstance(new_code, dict):
        raise ValueError("Missing original/current code fingerprints")
    changed = {name for name in set(old_code) | set(new_code)
               if old_code.get(name) != new_code.get(name)}
    if not changed or not changed <= ALLOWED_CHANGED_SOURCES:
        raise ValueError(f"Unexpected scientific source changes: {sorted(changed)}")
    for field in ("methods_evidence_sha256", "results_contract_sha256"):
        if previous.get(field) != current.get(field):
            raise ValueError(f"Unrelated evidence/contract change: {field}")

    verifier = Orchestrator(old_config, run_dir)
    for stage in STAGE_ORDER[:STAGE_ORDER.index("energy_calibration")]:
        record = verifier._load_stage_status(stage)
        if stage == "smoke_check" and record and record.get("status") == "skipped":
            continue
        if not record or record.get("status") not in ("completed", "completed_with_failures"):
            raise ValueError(f"Upstream stage {stage} is not complete")
        valid, detail = verifier._validate_completed_stage_artifacts(stage)
        if not valid:
            raise ValueError(f"Upstream {stage} artifacts failed verification: {detail}")
        valid, detail = verifier._validate_upstream_chain_fresh(stage)
        if not valid:
            raise ValueError(f"Upstream {stage} chain is stale: {detail}")
    calibration = verifier._load_stage_status("energy_calibration")
    if not calibration or calibration.get("status") != "failed":
        raise ValueError("This amendment requires a failed energy_calibration stage")

    amendment_dir = run_dir / "provenance" / "calibration_coverage_amendment"
    if amendment_dir.exists():
        raise ValueError(f"Calibration coverage amendment already exists: {amendment_dir}")
    archived = amendment_dir / "superseded_downstream"
    output_paths = [
        run_dir / name for name in (
            "calibration", "method_sensitivity", "qc_benchmark", "quantum_exploration",
            "dev_queue", "external_validation", "statistics",
            (config.get("final_report", {}) or {}).get("filename", "FINAL_RESEARCH_REPORT.md"),
            "EXPERIMENT_RESULTS_AUDIT.json", "EXPERIMENT_RESULTS_AUDIT.md",
            "RUN_SUMMARY.json", "artifact_inventory.json",
        )
    ]
    output_paths.extend(sorted(run_dir.glob("dev_queue_solvent_*")))
    validation = run_dir / "validation_queue"
    if validation.is_dir():
        output_paths.extend(path for path in validation.iterdir() if path.name != "freeze")
    for stage in DOWNSTREAM:
        output_paths.extend((run_dir / "stage_status" / f"{stage}.json",
                             run_dir / "results_manifests" / f"{stage}.json"))
    # Check every target before any mutation; do not follow a symlink outside
    # the run or archive the queue-freeze input subtree.
    for path in output_paths:
        if path.exists() and not path.resolve().is_relative_to(run_dir):
            raise ValueError(f"Downstream artifact escapes run directory: {path}")
    amendment_dir.mkdir(parents=True)
    (amendment_dir / "original_run_manifest.json").write_bytes(previous_bytes)
    shutil.copy2(frozen_path, amendment_dir / "original_frozen_config.yaml")
    shutil.copy2(config_path, amendment_dir / "amended_resolved_config.yaml")
    shutil.copy2(repo_root / "docs" / "PROTOCOL_AMENDMENTS.md",
                 amendment_dir / "amended_protocol_amendments.md")
    source_archive=amendment_dir / "amended_sources"
    source_archive.mkdir()
    for name in sorted(changed):
        shutil.copy2(repo_path(name,repo_root), source_archive / name)
    moved = []
    for path in output_paths:
        if not path.exists():
            continue
        target = archived / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append(path.relative_to(run_dir).as_posix())
    amendment = {
        "amended_utc": utc_timestamp(),
        "reason": "Preserve three real chi1 wells and separate missing-atom eligibility from generation failure",
        "original_run_manifest_sha256": sha256_of(amendment_dir / "original_run_manifest.json"),
        "amended_resolved_config_sha256": sha256_of(amendment_dir / "amended_resolved_config.yaml"),
        "unchanged_upstream_stage_results_sha256": {
            stage: sha256_of(run_dir / "results_manifests" / f"{stage}.json")
            for stage in ("env_check", "data_audit", "queue_freeze", "egnn_train")
        },
        "changed_scientific_sources": {name: {"before": old_code[name], "after": new_code[name]}
                                       for name in sorted(changed)},
        "archived_downstream_paths": moved,
        "amendment_tool_sha256": sha256_of(Path(__file__)),
    }
    atomic_write_json(amendment_dir / "amendment.json", amendment)
    current["calibration_eligibility_amendment"] = str((amendment_dir / "amendment.json").relative_to(run_dir))
    atomic_write_json(manifest_path, current)
    frozen_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return amendment_dir / "amendment.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"Calibration protocol amendment recorded: {amend(args.run_dir, args.config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
