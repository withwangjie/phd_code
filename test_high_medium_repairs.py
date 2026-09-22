"""Regression tests for method-upgrade-v2 High/Medium formal-run repairs."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import batch_benchmark_hard_set as bbh
import build_final_pyg_dataset as dataset_builder
import run_real_complex_pilot as pilot
import run_full_experiment as full
import resolve_server_config as server_resolver
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



def test_singleton_hard_set_pair_similarity_is_zero() -> None:
    assert dataset_builder.max_pair_similarity([])==0.0
    assert dataset_builder.max_pair_similarity(["CARDRST"])==0.0



def test_method_sensitivity_requires_zero_failures(tmp_path: Path) -> None:
    source=inspect.getsource(Orchestrator.stage_method_sensitivity)
    assert "if rc!=0 or not summary_path.is_file()" in source
    assert "failures_total" in source
    assert "sensitivity sub-run not closed" in source


def test_statistics_resume_checks_statistics_subdirectory(tmp_path: Path) -> None:
    class Dummy:
        run_dir=tmp_path
        config={"statistics":{"budget_modes":["outputs","time"]},"final_report":{}}

        def dataset_dir(self):
            return tmp_path/"dataset"

        def checkpoint_dir(self):
            return tmp_path/"checkpoints"

        _artifacts_present=staticmethod(Orchestrator._artifacts_present)

    qc=tmp_path/"qc_benchmark"
    qc.mkdir()
    for mode in ("outputs","time"):
        (qc/f"statistics_{mode}.json").write_text("{}",encoding="utf-8")
        (qc/f"statistics_{mode}.md").write_text("ok",encoding="utf-8")
    stats=tmp_path/"statistics"
    stats.mkdir()
    (stats/"structure_statistics.json").write_text("{}",encoding="utf-8")
    (stats/"structure_statistics.md").write_text("ok",encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(Dummy(),"statistics")
    assert ok is True, detail



def test_active_site_scaling_axis_and_primary_size_are_frozen() -> None:
    import yaml
    cfg=yaml.safe_load(Path("full_experiment_config.yaml").read_text(encoding="utf-8"))
    assert cfg["qc_benchmark"]["active_sites"] == [4,6,8,10]
    assert cfg["statistics"]["primary_active_sites"] == 6
    assert cfg["queue_freeze"]["validation_queue"]["sites"] == 6
    assert cfg["qc_benchmark"]["sensitivity"]["active_sites"] == 6


def test_ablation_active_sites_are_a_case_dimension() -> None:
    source=inspect.getsource(bbh._ablation_main)
    case_source=inspect.getsource(bbh._ablation_run_case)
    assert "args.active_sites,args.seeds" in source
    assert "active_sites=setting[4]" in source
    assert 'active_sites=int(config["active_sites"])' in case_source
    assert "min_variables=3 * active_sites" in case_source
    assert "max_sites=active_sites" in case_source


def test_primary_statistics_filter_active_site_scale() -> None:
    source=inspect.getsource(bbh._paired_statistics_main)
    assert "--primary-active-sites" in source
    assert 'get("active_sites",args.primary_active_sites)' in source
    assert "primary_active_sites=args.primary_active_sites" in source



def test_methods_evidence_register_covers_formal_design() -> None:
    source=Path("METHODS_EVIDENCE.md").read_text(encoding="utf-8")
    for token in (
        "E(n)-equivariant",
        "Dunbrack 2010",
        "heavy-atom distance <=5",
        "QAOA",
        "CVaR",
        "FASPR",
        "ff14SB",
        "OpenMM",
        "Foldseek",
        "Simulated annealing",
        "Study-specific preregistration",
        "GLINTER",
        "CDR-H3 isolation",
        "identity <0.30 with >=0.70 coverage",
        "alpha=0.1",
    ):
        assert token in source


def test_final_report_embeds_methods_references() -> None:
    source=inspect.getsource(full)
    assert "METHODS_EVIDENCE.md" in source
    assert "section_literature_basis" in source
    assert "alpha=0.1 is literature-supported as an empirical CVaR setting" in source
    manifest_source=inspect.getsource(full.build_run_manifest)
    assert "methods_evidence_sha256" in manifest_source



def test_resume_rejects_methods_evidence_changes() -> None:
    source=inspect.getsource(full.main)
    assert 'previous.get("methods_evidence_sha256") != current["methods_evidence_sha256"]' in source
    assert "literature-evidence hash" in source



def test_server_resolver_preserves_target_global_batch() -> None:
    assert server_resolver._choose_ddp_ranks(1,4,4) == 1
    assert server_resolver._choose_ddp_ranks(2,4,4) == 2
    assert server_resolver._choose_ddp_ranks(3,4,4) == 2
    assert server_resolver._choose_ddp_ranks(4,4,4) == 4


def test_server_config_is_infrastructure_only() -> None:
    import yaml
    server=yaml.safe_load(Path("server_config.yaml").read_text(encoding="utf-8"))
    assert "paths" in server and "resources" in server
    forbidden=("homology_isolation","active_sites","cvar_alpha","qaoa_objective","interface_label_cutoff_angstrom")
    rendered=json.dumps(server,sort_keys=True)
    assert all(key not in rendered for key in forbidden)
