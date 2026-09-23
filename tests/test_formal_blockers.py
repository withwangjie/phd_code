from __future__ import annotations

import inspect
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

import nanoqc.reporting.generate_final_research_report as report
import nanoqc.experiments.run_real_complex_pilot as pilot
from nanoqc.pipeline.run_full_experiment import Orchestrator, rq5_inference_failures
from nanoqc.inference.analyze_structure_recovery import rq5_energy_structure


def test_rq5_constant_energy_fails_despite_sufficient_clusters() -> None:
    rows=[]
    clusters={}
    for index in range(12):
        target=f"target_{index}"
        clusters[target]=f"cluster_{index}"
        rows.extend([
            {"target":target,"seed":42,"method":"qaoa",
             "discrete_energy_kcal":-10,"final_rmsd":float(index+1)},
            {"target":target,"seed":42,"method":"sa",
             "discrete_energy_kcal":-10,"final_rmsd":1.0},
        ])
    rq5=rq5_energy_structure(rows,clusters,"qaoa_vs_sa",1000,42)
    assert rq5["n_clusters"]==12
    assert rq5["spearman_rho"] is None
    failures=rq5_inference_failures(rq5,10)
    assert any("spearman_rho" in failure for failure in failures)


def test_rq5_finite_inference_passes_gate() -> None:
    rq5={"n_clusters":12,"spearman_rho":0.4,"p_value":0.03,
         "ci_low":0.1,"ci_high":0.7,"p_holm_confirmatory_family":0.06}
    assert rq5_inference_failures(rq5,10)==[]


def test_final_report_filters_budget_modes() -> None:
    rows=[
        {"solver":"qaoa","budget_mode":"matched_outputs"},
        {"solver":"sa_time","budget_mode":"matched_time_soft_deadline"},
        {"solver":"legacy"},
    ]
    selected=report._filter_qc_rows(rows,"matched_outputs")
    assert [r["solver"] for r in selected]==["qaoa","legacy"]


def test_final_report_reads_formal_statistics_payload(tmp_path: Path) -> None:
    class Ctx:
        run_dir=tmp_path
    root=tmp_path/"qc_benchmark"
    root.mkdir()
    payload={"primary_outputs":1000,"primary_objective":"cvar","primary_restarts":4}
    (root/"statistics_outputs.json").write_text(json.dumps(payload),encoding="utf-8")
    loaded=report._formal_statistics_payload(Ctx(),"outputs")
    assert loaded==payload


def test_statistics_fails_closed_when_qc_dir_missing(tmp_path: Path) -> None:
    class Harness:
        run_dir=tmp_path
        venv_python="python"
        config={
            "master_seed":20260917,
            "qc_benchmark":{
                "outputs":[10,30,100,300,1000],
                "qaoa_objective":["mean","cvar"],
                "qaoa_restarts":[1,4],
            },
            "statistics":{
                "resamples":100,
                "budget_modes":["outputs","time"],
                "primary_outputs":1000,
                "primary_objective":"cvar",
                "primary_restarts":4,
                "max_time_overrun_fraction":0.1,
                "cluster_map":None,
                "min_primary_clusters":2,
                "min_rq5_clusters":2,
            },
            "queue_freeze":{"independence_clustering":{}},
        }

        def _run_subprocess(self,stage,argv):
            raise AssertionError("subprocess must not run when qc_benchmark is missing")

        def frozen_cluster_map_path(self):
            return tmp_path/"independence"/"pdb_family_clusters.json"

        @staticmethod
        def _artifacts_present(paths):
            return Orchestrator._artifacts_present(paths)

    result=Orchestrator.stage_statistics(Harness())
    assert result.status=="failed"
    assert result.returncode==1
    assert "input directory is missing" in result.detail


def test_qc_resume_rechecks_failure_fraction(tmp_path: Path) -> None:
    class Dummy:
        run_dir=tmp_path
        config={"qc_benchmark":{"max_failure_fraction":0.0}}
        _artifacts_present=staticmethod(Orchestrator._artifacts_present)

    root=tmp_path/"qc_benchmark"
    (root/"cases").mkdir(parents=True)
    for name in ("run_manifest.json","metrics.csv","summary.md",
                 "failed_case_keys.json","seed_streams.json"):
        (root/name).write_text("x",encoding="utf-8")
    (root/"cases"/"case.json").write_text("{}",encoding="utf-8")
    (root/"run_summary.json").write_text(json.dumps({
        "closed":True,"cases_completed_total":1,"total_cases_planned":2,
        "failures_total":1}),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"qc_benchmark",require_results_manifest=False)
    assert ok is False
    assert "failure fraction" in detail


def test_structure_resume_rejects_failed_validation_target(tmp_path: Path) -> None:
    class Dummy:
        run_dir=tmp_path
        config={}
        _artifacts_present=staticmethod(Orchestrator._artifacts_present)

    for label in ("dev_queue","validation_queue"):
        root=tmp_path/label
        root.mkdir()
        for name in ("run_manifest.json","seed_streams.json","eligibility.json",
                     "selected_targets.json","real_complex_metrics.csv","real_complex_report.md"):
            (root/name).write_text("x",encoding="utf-8")
    (tmp_path/"dev_queue"/"run_summary.json").write_text(
        json.dumps({"closed":True}),encoding="utf-8")
    (tmp_path/"validation_queue"/"run_summary.json").write_text(json.dumps({
        "closed":True,"frozen_set_accounting_ok":True,
        "structure_experiment_failed_targets":["1abc"]}),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"structure_experiment",require_results_manifest=False)
    assert ok is False
    assert "failed targets" in detail


def test_results_manifest_detects_same_size_artifact_change(tmp_path: Path) -> None:
    import hashlib

    class Dummy:
        run_dir=tmp_path
        results_manifest_dir=tmp_path/"results_manifests"
        _artifacts_present=staticmethod(Orchestrator._artifacts_present)

    artifact=tmp_path/"env_check.json"
    artifact.write_text('{"ok":1}',encoding="utf-8")
    Dummy.results_manifest_dir.mkdir()
    (Dummy.results_manifest_dir/"env_check.json").write_text(json.dumps({
        "stage":"env_check","status":"completed","artifact_count":1,
        "artifacts":[{"path":"env_check.json","size_bytes":artifact.stat().st_size,
                      "sha256":hashlib.sha256(artifact.read_bytes()).hexdigest()}],
    }),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(Dummy(),"env_check")
    assert ok is True,detail
    artifact.write_text('{"ok":2}',encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(Dummy(),"env_check")
    assert ok is False
    assert "sha256 mismatch" in detail


def test_validation_freeze_requires_exact_graph_identity_and_frozen_accounting() -> None:
    source=inspect.getsource(pilot.main)
    for token in ("source_id","graph_path","graph_sha256","frozen_set_accounting_ok"):
        assert token in source
    assert "completed_ids|failed_ids==set(frozen_target_ids)" in source


def test_preflight_contains_gpu_ddp_openmm_and_resource_gates() -> None:
    source=(REPO/"scripts"/"formal_preflight.sh").read_text(encoding="utf-8")
    for token in (
        "nvidia-smi",
        "torch.cuda.device_count()",
        'backend="nccl"',
        "all_reduce",
        "OpenMM",
        "Precision",
        "FORMAL_MIN_FREE_DISK_GB",
        "FORMAL_MIN_AVAILABLE_RAM_GB",
    ):
        assert token in source


def test_launcher_runs_preflight_before_nohup() -> None:
    source=(REPO/"scripts"/"run_full_experiment.sh").read_text(encoding="utf-8")
    assert source.index('bash "$PREFLIGHT_SCRIPT"') < source.index("nohup python")


def test_launcher_rejects_config_and_run_dir_override_before_preflight() -> None:
    source=(REPO/"scripts"/"run_full_experiment.sh").read_text(encoding="utf-8")
    guard='--config|--config=*|--run-dir|--run-dir=*)'
    assert guard in source
    assert source.index(guard)<source.index('bash "$PREFLIGHT_SCRIPT"')



def test_execution_provenance_binds_frozen_allowlist_sha() -> None:
    source=inspect.getsource(pilot.main)
    assert "pdb_allowlist_sha256" in source
    assert "_ablation_digest(args.pdb_allowlist_file)" in source


def test_queue_freeze_resume_rejects_tampered_selected_targets(tmp_path: Path) -> None:
    class Dummy:
        run_dir=tmp_path
        config={
            "queue_freeze":{"independence_clustering":{"cluster_map":"source.json"}},
            "statistics":{},
            "final_report":{},
        }

        def dataset_dir(self):
            return tmp_path/"dataset"

        def checkpoint_dir(self):
            return tmp_path/"checkpoints"

        def frozen_cluster_map_path(self):
            return tmp_path/"independence"/"pdb_family_clusters.json"

        _artifacts_present=staticmethod(Orchestrator._artifacts_present)

    dataset=tmp_path/"dataset"
    dataset.mkdir()
    for name,content in {
        "graph_manifest.csv":"header\n",
        "graph_manifest.json":"{}",
        "graph_dataset_delivery_report.md":"ok",
        "cdr3_clusters.json":"{}",
        "excluded_samples.csv":"header\n",
        "processing_failures.csv":"header\n",
    }.items():
        (dataset/name).write_text(content,encoding="utf-8")
    (dataset/"run_summary.json").write_text(
        json.dumps({"complete":True}),encoding="utf-8")

    audit=tmp_path/"audit"; audit.mkdir()
    universe=audit/"cluster_universe.txt"
    universe.write_text("1abc\n",encoding="utf-8")

    independence=tmp_path/"independence"; independence.mkdir()
    cluster=independence/"pdb_family_clusters.json"
    cluster.write_text(json.dumps({"1abc":"cluster_1"}),encoding="utf-8")
    cluster_prov=independence/"pdb_family_clusters.provenance.json"
    cluster_prov.write_text(json.dumps({"status":"frozen"}),encoding="utf-8")

    freeze=tmp_path/"validation_queue"/"freeze"
    freeze.mkdir(parents=True)
    selected=freeze/"selected_targets.json"
    eligibility=freeze/"eligibility.json"
    selected.write_text(json.dumps([{
        "target":"1abc","source_id":"s","graph_path":"g.pt","graph_sha256":"a"*64,
        "eligibility_compatible_residues":["A:1"],
    }]),encoding="utf-8")
    eligibility.write_text("[]",encoding="utf-8")
    (freeze/"run_manifest.json").write_text("{}",encoding="utf-8")
    prepared=freeze/"prepared"/"1abc";prepared.mkdir(parents=True)
    (prepared/"recovery_manifest.json").write_text("{}",encoding="utf-8")

    import hashlib
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (freeze/"freeze_manifest.json").write_text(json.dumps({
        "schema_version":2,
        "selected_targets_sha256":digest(selected),
        "eligibility_sha256":digest(eligibility),
        "graph_manifest_sha256":digest(dataset/"graph_manifest.csv"),
        "cluster_map_sha256":digest(cluster),
        "cluster_map_provenance_sha256":digest(cluster_prov),
        "cluster_universe_sha256":digest(universe),
    }),encoding="utf-8")

    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"queue_freeze",require_results_manifest=False)
    assert ok is True,detail

    selected.write_text(json.dumps([{
        "target":"9xyz","source_id":"changed","graph_path":"changed.pt","graph_sha256":"b"*64,
        "eligibility_compatible_residues":["A:2"],
    }]),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(
        Dummy(),"queue_freeze",require_results_manifest=False)
    assert ok is False
    assert "provenance mismatch" in detail
