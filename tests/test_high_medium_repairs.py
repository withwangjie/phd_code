"""Regression tests for method-upgrade-v2 High/Medium formal-run repairs."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import nanoqc.experiments.batch_benchmark_hard_set as bbh
import nanoqc.data.build_final_pyg_dataset as dataset_builder
import nanoqc.reporting.generate_final_research_report as report
import nanoqc.experiments.run_real_complex_pilot as pilot
import nanoqc.pipeline.run_full_experiment as full
import nanoqc.pipeline.resolve_server_config as server_resolver
from nanoqc.pipeline.run_full_experiment import Orchestrator, StageResult, apply_runtime_mode_overrides


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
        # Frozen primary dimensions (the defaults of --paired-statistics), so the
        # case reaches the all-restarts-failed check instead of being filtered.
        json.dumps({"config":{"pdb_id":"1abc","pruning":"egnn","radius":6.0,"depth":2,
                              "max_evals":90,"active_sites":6},"metrics":rows}),
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


def test_paired_denominator_gate_covers_all_exclusion_reasons() -> None:
    from nanoqc.inference.paired_statistics import paired_denominator_failures

    output_reasons = {
        "sa:unequal_outputs": 1,
        "uniform:gap:nonfinite": 1,
        "greedy:entropy:nonfinite": 1,
        "sa_time:overrun": 1,
    }
    assert paired_denominator_failures(output_reasons,"outputs") == {
        "sa:unequal_outputs": 1,
        "uniform:gap:nonfinite": 1,
        "greedy:entropy:nonfinite": 1,
    }
    time_reasons = {
        "sa_time:overrun": 2,
        "uniform_time:invalid_budget": 1,
        "greedy_time:hit:nonfinite": 1,
        # Time analysis intentionally does not test same-output diversity metrics.
        "sa_time:entropy:nonfinite": 1,
    }
    assert paired_denominator_failures(time_reasons,"time") == {
        "sa_time:overrun": 2,
        "uniform_time:invalid_budget": 1,
        "greedy_time:hit:nonfinite": 1,
    }


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

    def _validate_completed_stage_artifacts(self,stage,*,require_results_manifest=True):
        return True,"ok"

    def _write_stage_results_manifest(self,stage,result,validation_detail,upstream=None):
        self.saved.append(("results_manifest",stage,result.status))

    def _upstream_fingerprints(self,prerequisites):
        return {name:None for name in prerequisites}


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
    assert dataset_builder.max_pair_cdr_h3_loop_identity([])==0.0
    assert dataset_builder.max_pair_cdr_h3_loop_identity(["CARDRST"])==0.0



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
    # batch_benchmark_hard_set names matched-time effects "<baseline>_time".
    for mode,baseline in (("outputs","sa"),("time","sa_time")):
        paired={"effects":[{"baseline":baseline,"metric":"log10_qts99","n_clusters":10}],"exclusions":{}}
        (qc/f"statistics_{mode}.json").write_text(json.dumps(paired),encoding="utf-8")
        (qc/f"statistics_{mode}.md").write_text("ok",encoding="utf-8")
    stats=tmp_path/"statistics"
    stats.mkdir()
    (stats/"quantum_scaling_statistics.json").write_text(
        json.dumps({"primary":{"n_clusters":10}}),encoding="utf-8")
    (stats/"quantum_scaling_statistics.md").write_text("ok",encoding="utf-8")
    valid_rq5={"n_clusters":10,"spearman_rho":0.4,"p_value":0.03,
               "ci_low":0.1,"ci_high":0.7,"p_holm_confirmatory_family":0.06}
    (stats/"structure_statistics.json").write_text(
        json.dumps({"primary":{"n_clusters":10},"rq5":valid_rq5}),encoding="utf-8")
    (stats/"structure_statistics.md").write_text("ok",encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"statistics",require_results_manifest=False)
    assert ok is True, detail

    (stats/"structure_statistics.json").write_text(
        json.dumps({"primary":{"n_clusters":10},"rq5":{**valid_rq5,"spearman_rho":None}}),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"statistics",require_results_manifest=False)
    assert ok is False
    assert "spearman_rho" in detail

    (stats/"structure_statistics.json").write_text(
        json.dumps({"primary":{"n_clusters":10},"rq5":valid_rq5}),encoding="utf-8")
    (stats/"quantum_scaling_statistics.json").write_text(
        json.dumps({"primary":{"n_clusters":0}}),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"statistics",require_results_manifest=False)
    assert ok is False
    assert "Scaling inference" in detail

    (stats/"quantum_scaling_statistics.json").write_text(
        json.dumps({"primary":{"n_clusters":10}}),encoding="utf-8")
    (qc/"statistics_outputs.json").write_text(json.dumps({
        **paired,"exclusions":{"qaoa:all_restarts_failed":1}}),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"statistics",require_results_manifest=False)
    assert ok is False
    assert "denominator is incomplete" in detail

    for mode,reason in (
        ("outputs","sa:unequal_outputs"),
        ("outputs","sa:gap:nonfinite"),
        ("time","sa_time:overrun"),
        ("time","sa_time:invalid_budget"),
        ("time","sa_time:hit:nonfinite"),
    ):
        for valid_mode,valid_baseline in (("outputs","sa"),("time","sa_time")):
            (qc/f"statistics_{valid_mode}.json").write_text(json.dumps({
                "effects":[{"baseline":valid_baseline,"metric":"log10_qts99","n_clusters":10}],
                "exclusions":{},
            }),encoding="utf-8")
        (qc/f"statistics_{mode}.json").write_text(json.dumps({
            "effects":[{"baseline":"sa" if mode=="outputs" else "sa_time",
                        "metric":"log10_qts99","n_clusters":10}],
            "exclusions":{reason:1},
        }),encoding="utf-8")
        ok,detail=Orchestrator._validate_completed_stage_artifacts(
            Dummy(),"statistics",require_results_manifest=False)
        assert ok is False, (mode,reason,detail)
        assert reason in detail



def test_active_site_scaling_axis_and_primary_size_are_frozen() -> None:
    import yaml
    cfg=yaml.safe_load((REPO/"configs"/"full_experiment_config.yaml").read_text(encoding="utf-8"))
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
    assert "min_variables=args.states_per_site * active_sites" in case_source
    assert "max_sites=active_sites" in case_source


def test_primary_statistics_filter_active_site_scale() -> None:
    source=inspect.getsource(bbh._paired_statistics_main)
    assert "--primary-active-sites" in source
    assert 'get("active_sites",args.primary_active_sites)' in source
    assert "primary_active_sites=args.primary_active_sites" in source


def test_smoke_passes_dunbrack_library_to_both_entrypoints() -> None:
    source=inspect.getsource(Orchestrator.stage_smoke_check)
    assert '"--rotamer-library", str(smoke_rotamer_library)' in source
    assert '"--rotamer-mode", "dunbrack2010"' in source


def test_calibration_is_fixed_three_well_and_buffers_complex_rows() -> None:
    source=inspect.getsource(__import__(
        "nanoqc.experiments.generate_energy_calibration_dataset",
        fromlist=["main"],
    ).main)
    assert "complex_rows=[]" in source
    assert "fixed_chi1_wells=True,fixed_states_per_site=3" in source
    assert "for output_row in complex_rows" in source



def test_methods_evidence_register_covers_formal_design() -> None:
    source=(REPO/"docs"/"METHODS_EVIDENCE.md").read_text(encoding="utf-8")
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
    report_source=inspect.getsource(report)
    assert "METHODS_EVIDENCE.md" in report_source
    assert "section_literature_basis" in report_source
    assert "alpha=0.1 is literature-supported as an empirical CVaR setting" in report_source
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
    server=yaml.safe_load((REPO/"configs"/"server_config.yaml").read_text(encoding="utf-8"))
    assert "paths" in server and "resources" in server
    forbidden=("homology_isolation","active_sites","cvar_alpha","qaoa_objective","interface_label_cutoff_angstrom")
    rendered=json.dumps(server,sort_keys=True)
    assert all(key not in rendered for key in forbidden)



def test_unified_run_directory_inventory_and_summary(tmp_path: Path) -> None:
    run=tmp_path/"run"; run.mkdir()
    (run/"logs").mkdir()
    (run/"logs"/"x.log").write_text("ok",encoding="utf-8")
    results={"env_check": full.StageResult("env_check","completed","a","b",0,"ok")}
    # RUN_SUMMARY is "completed" only when the global results audit passed.
    (run/"EXPERIMENT_RESULTS_AUDIT.json").write_text(
        json.dumps({"all_required_results_present":True}),encoding="utf-8")
    full.write_run_inventory(run,results)
    inventory=json.loads((run/"artifact_inventory.json").read_text(encoding="utf-8"))
    summary=json.loads((run/"RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert any(x["path"]=="logs/x.log" for x in inventory["artifacts"])
    assert summary["status"]=="completed"
    assert summary["artifact_inventory"]=="artifact_inventory.json"


def test_one_click_launcher_uses_precreated_run_directory() -> None:
    source=(REPO/"scripts"/"run_full_experiment.sh").read_text(encoding="utf-8")
    assert '--run-dir "$RUN_DIR"' in source
    assert 'PREFLIGHT_LOG="$RUN_DIR/logs/formal_preflight_' in source
    assert 'LAUNCH_LOG="$RUN_DIR/logs/launch_' in source
    assert 'provenance/resolved_runtime_config.yaml' in source



def test_formal_result_contract_requires_core_experimental_outputs() -> None:
    source=inspect.getsource(full.Orchestrator._validate_completed_stage_artifacts)
    for token in (
        'root/"metrics.csv"',
        'root/"summary.md"',
        'sensitivity_summary.csv',
        'sensitivity_summary.json',
        'sensitivity_summary.md',
        'real_complex_metrics.csv',
        'real_complex_report.md',
        'external_baseline_metrics.csv',
        'external_baseline_report.md',
        'statistics_outputs.json',
        'structure_statistics.json',
    ):
        assert token in source


def test_stage_results_manifest_is_written_and_required() -> None:
    source=inspect.getsource(full.Orchestrator)
    assert "results_manifest_dir" in source
    assert "results_manifests" in source
    assert "_write_stage_results_manifest" in source
    assert "Missing stage results manifest" in source


def test_run_summary_requires_global_results_audit(tmp_path: Path) -> None:
    run=tmp_path/"run"; run.mkdir()
    (run/"EXPERIMENT_RESULTS_AUDIT.json").write_text(
        json.dumps({"all_required_results_present":False}),encoding="utf-8")
    results={"env_check":full.StageResult("env_check","completed","a","b",0,"ok")}
    full.write_run_inventory(run,results)
    summary=json.loads((run/"RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["status"]=="failed"
    assert summary["results_audit_ok"] is False



def test_results_contract_is_frozen_and_archived() -> None:
    manifest_source=inspect.getsource(full.build_run_manifest)
    assert "results_contract_sha256" in manifest_source
    launcher=(REPO/"scripts"/"run_full_experiment.sh").read_text(encoding="utf-8")
    assert "provenance/RESULTS_CONTRACT.md" in launcher
    contract=(REPO/"docs"/"RESULTS_CONTRACT.md").read_text(encoding="utf-8")
    for token in (
        "qc_benchmark/metrics.csv",
        "sensitivity_summary.csv",
        "real_complex_metrics.csv",
        "external_baseline_metrics.csv",
        "structure_statistics.json",
        "EXPERIMENT_RESULTS_AUDIT.json",
    ):
        assert token in contract


def test_structure_contract_requires_per_target_recovery_outputs() -> None:
    source=inspect.getsource(full.Orchestrator._validate_completed_stage_artifacts)
    assert 'target/"run_manifest.json"' in source
    assert 'target/"recovery_metrics.csv"' in source
    assert 'target/"recovery_report.md"' in source



def test_primary_statistics_freeze_every_case_dimension() -> None:
    source=inspect.getsource(bbh._paired_statistics_main)
    for token in (
        "--primary-pruning",
        "--primary-radius",
        "--primary-depth",
        "--primary-max-evals",
        "--primary-active-sites",
        "--primary-outputs",
        "--primary-objective",
        "--primary-restarts",
        "nonprimary_radius",
        "nonprimary_depth",
        "nonprimary_max_evals",
    ):
        assert token in source


def test_scaling_cases_require_exact_requested_rotamer_site_count() -> None:
    source=inspect.getsource(bbh._ablation_run_case)
    assert "len(qubo.site_to_variables) != active_sites" in source
    selector=inspect.getsource(__import__("nanoqc.model.model_egnn_pruning", fromlist=["_"]).select_ablation_active)
    assert "backbone_phi" in selector and "backbone_psi" in selector
    assert 'aa not in {"A", "G", "P", "C"} and backbone_ok' in selector
    assert "scores[candidates] = (distances < float(contact_ca_cutoff)).sum(1).float()" in selector
    assert "scores[cdr_allowed] += float(scores[candidates].max()) + 1.0" in selector


def test_structural_statistics_use_shared_sign_flip_protocol() -> None:
    import nanoqc.inference.analyze_structure_recovery as structure
    from nanoqc.inference import paired_statistics
    source=inspect.getsource(structure.sign_flip_p)
    assert "sign_flip_pvalue" in source
    assert "200000" not in source
    assert paired_statistics.SIGN_FLIP_EXACT_MAX_CLUSTERS == 16


def test_formal_statistics_require_scaling_outputs() -> None:
    source=inspect.getsource(full.Orchestrator._validate_completed_stage_artifacts)
    assert "quantum_scaling_statistics.json" in source
    assert "quantum_scaling_statistics.md" in source
    assert "min_scaling_clusters" in inspect.getsource(full._validate_scientific_config)


def test_statistics_reuse_run_local_frozen_cluster_map() -> None:
    source=inspect.getsource(full.Orchestrator.stage_statistics)
    assert "statistics_cluster_path = self.frozen_cluster_map_path()" in source
    assert 'cfg_probe["cluster_map"]' not in source
    assert "resolve_path(self.config" not in source


def test_statistics_cluster_map_override_is_rejected() -> None:
    source=inspect.getsource(full._validate_scientific_config)
    assert "statistics.cluster_map is obsolete" in source


def test_partial_only_run_cannot_pass_global_audit_without_prior_results(tmp_path: Path) -> None:
    class Dummy:
        run_dir=tmp_path
        config={"stages":{stage:True for stage in full.STAGE_ORDER}}

        def _load_stage_status(self,stage):
            return None

        def _validate_completed_stage_artifacts(self,stage):
            return False,"missing"

    results={
        stage:full.StageResult(stage,"skipped","a","b",None,"Skipped (--only targets a different stage; not yet run).")
        for stage in full.STAGE_ORDER
    }
    ok,payload=full.audit_experiment_results(Dummy(),results)
    assert ok is False
    assert any(
        row["status"]=="skipped" and row["results_contract_ok"] is False
        for row in payload["stages"]
    )


def test_pair_table_coverage_is_required_before_frozen_clustering() -> None:
    source=inspect.getsource(full.Orchestrator.stage_queue_freeze)
    assert "missing_pair_coverage" in source
    assert "not demonstrated as searched" in source
    # The failure has to name the run whose universe the table must match.
    assert "foldseek --run-dir" in source
