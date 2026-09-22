"""Regression tests for method-upgrade-v2 High/Medium formal-run repairs."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import batch_benchmark_hard_set as bbh
import run_real_complex_pilot as pilot
import run_full_experiment as full
from run_full_experiment import Orchestrator, StageResult, apply_runtime_mode_overrides


def test_no_hidden_4s10_target_fallback() -> None:
    source=inspect.getsource(pilot.main)
    assert "requested_pdb=args.pdb_id" in source
    assert "targets==1" not in source
    assert "'4s10' if args.eval_shots" not in source


def test_qaoa_output_curve_reuses_optimization_and_explicit_time_donor() -> None:
    source=inspect.getsource(bbh._ablation_run_case)
    assert "qaoa_cache = {}" in source
    assert "if cache_key not in qaoa_cache:" in source
    assert "elapsed=optimization_seconds+sampling_seconds" in source
    assert "args.time_donor_objective" in source
    assert "args.time_donor_restarts" in source
    assert 'termination_reason")!="all_restarts_failed"' in source


def _metric(solver: str, *, reason=None, objective=None, restarts=None) -> dict:
    return dict(
        solver=solver,outputs=1000,gap=1.0,hit=0,
        ground_probability=0.0,low_energy_mass=0.1,
        low_energy_coverage=0.2,entropy=1.0,solver_seconds=1.0,
        qaoa_objective=objective,qaoa_restarts=restarts,
        termination_reason=reason,
    )


def test_all_restarts_failed_is_excluded_from_paired_inference(tmp_path: Path) -> None:
    root=tmp_path/"results"; cases=root/"cases"; cases.mkdir(parents=True)
    rows=[
        _metric("qaoa",reason="all_restarts_failed",objective="cvar",restarts=4),
        _metric("sa"),_metric("uniform"),_metric("greedy"),
    ]
    (cases/"case.json").write_text(
        json.dumps({"config":{"pdb_id":"1abc"},"metrics":rows}),
        encoding="utf-8",
    )
    rc=bbh._paired_statistics_main([
        "--results-dir",str(root),
        "--budget-mode","outputs",
        "--primary-outputs","1000",
        "--primary-objective","cvar",
        "--primary-restarts","4",
        "--resamples","100",
    ])
    assert rc==0
    payload=json.loads((root/"statistics_outputs.json").read_text(encoding="utf-8"))
    assert payload["exclusions"]["qaoa:all_restarts_failed"]==1
    assert payload["effects"]==[]


def test_dev_subruns_are_single_target_and_emit_root_summary() -> None:
    source=inspect.getsource(Orchestrator.stage_structure_experiment)
    assert '"--targets", "1"' in source
    assert 'out_dir/"run_summary.json"' in source
    assert "planned_target_ids=sorted(planned)" in source


def test_smoke_only_override_enables_smoke_without_mutating_source() -> None:
    config={"stages":{"env_check":True,"smoke_check":False}}
    resolved=apply_runtime_mode_overrides(config,smoke_only=True)
    assert resolved["stages"]["smoke_check"] is True
    assert config["stages"]["smoke_check"] is False


class _StageHarness:
    only=None
    force_restage=set()
    config={"stages":{"target":True}}

    def __init__(self,record):
        self.record=record
        self.saved=[]

    def _load_stage_status(self,stage):
        return self.record if stage in ("smoke_check","energy_calibration") else None

    def _save_stage_status(self,result):
        self.saved.append(result)

    def _validate_completed_stage_artifacts(self,stage):
        return True,"ok"


def test_only_smoke_skip_is_optional_prerequisite() -> None:
    harness=_StageHarness({"status":"skipped"})
    result=Orchestrator.run_stage(
        harness,"target",["smoke_check"],
        lambda: StageResult("target","completed","a","b",0,"ok"),
    )
    assert result.status=="completed"


def test_disabled_scientific_stage_skip_does_not_satisfy_prerequisite() -> None:
    harness=_StageHarness({"status":"skipped"})
    result=Orchestrator.run_stage(
        harness,"target",["energy_calibration"],
        lambda: StageResult("target","completed","a","b",0,"ok"),
    )
    assert result.status=="failed"


def test_stage_timeout_terminates_subprocess(tmp_path: Path) -> None:
    class Dummy:
        config={"hardware":{},"control":{"stage_timeout_seconds":0.1}}
        repo_root=tmp_path
        log_dir=tmp_path
    started=time.monotonic()
    rc,log_path=Orchestrator._run_subprocess(
        Dummy(),"timeout_probe",
        [sys.executable,"-c","import time; time.sleep(10)"],
    )
    assert rc==124
    assert time.monotonic()-started<6
    assert "stage timeout" in log_path.read_text(encoding="utf-8")


def test_effective_runtime_config_is_frozen() -> None:
    source=inspect.getsource(full.main)
    assert "yaml.safe_dump(config" in source
    assert "shutil.copy2(args.config, frozen_config_path)" not in source
