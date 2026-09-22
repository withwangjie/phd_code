from __future__ import annotations

import inspect
import json
from pathlib import Path

import generate_final_research_report as report
import run_real_complex_pilot as pilot
from run_full_experiment import Orchestrator


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

        @staticmethod
        def _artifacts_present(paths):
            return Orchestrator._artifacts_present(paths)

    result=Orchestrator.stage_statistics(Harness())
    assert result.status=="failed"
    assert result.returncode==1
    assert "input directory is missing" in result.detail


def test_validation_freeze_requires_exact_graph_identity_and_frozen_accounting() -> None:
    source=inspect.getsource(pilot.main)
    for token in ("source_id","graph_path","graph_sha256","frozen_set_accounting_ok"):
        assert token in source
    assert "completed_ids|failed_ids==set(frozen_target_ids)" in source


def test_preflight_contains_gpu_ddp_openmm_and_resource_gates() -> None:
    source=Path("formal_preflight.sh").read_text(encoding="utf-8")
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
    source=Path("run_full_experiment.sh").read_text(encoding="utf-8")
    assert source.index('bash "$PREFLIGHT_SCRIPT"') < source.index("nohup python")



def test_execution_provenance_binds_frozen_allowlist_sha() -> None:
    source=inspect.getsource(pilot.main)
    assert "pdb_allowlist_sha256" in source
    assert "_ablation_digest(args.pdb_allowlist_file)" in source


def test_queue_freeze_resume_rejects_tampered_selected_targets(tmp_path: Path) -> None:
    class Dummy:
        run_dir=tmp_path
        config={
            "queue_freeze":{"independence_clustering":{"cluster_map":None}},
            "statistics":{},
            "final_report":{},
        }

        def dataset_dir(self):
            return tmp_path/"dataset"

        def checkpoint_dir(self):
            return tmp_path/"checkpoints"

        _artifacts_present=staticmethod(Orchestrator._artifacts_present)

    dataset=tmp_path/"dataset"
    dataset.mkdir()
    for name,content in {
        "graph_manifest.csv":"header\n",
        "graph_manifest.json":"{}",
        "graph_dataset_delivery_report.md":"ok",
    }.items():
        (dataset/name).write_text(content,encoding="utf-8")
    (dataset/"run_summary.json").write_text(
        json.dumps({"complete":True}),encoding="utf-8")

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

    import hashlib
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (freeze/"freeze_manifest.json").write_text(json.dumps({
        "schema_version":1,
        "selected_targets_sha256":digest(selected),
        "eligibility_sha256":digest(eligibility),
        "graph_manifest_sha256":digest(dataset/"graph_manifest.csv"),
        "cluster_map_sha256":None,
    }),encoding="utf-8")

    ok,detail=Orchestrator._validate_completed_stage_artifacts(Dummy(),"queue_freeze")
    assert ok is True,detail

    selected.write_text(json.dumps([{
        "target":"9xyz","source_id":"changed","graph_path":"changed.pt","graph_sha256":"b"*64,
        "eligibility_compatible_residues":["A:2"],
    }]),encoding="utf-8")
    ok,detail=Orchestrator._validate_completed_stage_artifacts(Dummy(),"queue_freeze")
    assert ok is False
    assert "provenance mismatch" in detail
