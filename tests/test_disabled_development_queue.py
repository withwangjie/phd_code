"""Removing developmental targets must preserve the formal validation gates."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from nanoqc.common.repo_io import sha256_file
from nanoqc.pipeline.stages_structure import StructureStagesMixin
from nanoqc.pipeline.run_full_experiment import Orchestrator
from nanoqc.reporting import report_sections_structure as report


def test_disabled_development_launches_only_frozen_validation(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((repo/"configs/full_experiment_config.yaml").read_text(encoding="utf-8"))
    config["paths"].update(repo_root=str(repo),data_root=str(tmp_path/"data"))
    config["structure_experiment"]["solvent_sensitivity"]=["gbn2"]  # old field cannot re-enable dev work
    freeze=tmp_path/"validation_queue/freeze";freeze.mkdir(parents=True)
    targets=freeze/"selected_targets.json";targets.write_text('["8ybl"]')
    (freeze/"freeze_manifest.json").write_text(json.dumps({"selected_targets_sha256":sha256_file(targets)}))
    cluster=tmp_path/"clusters.json";cluster.write_text("{}")
    calls=[]
    class Harness:
        run_dir=tmp_path
        venv_python="python"
        def dataset_dir(self):return tmp_path/"dataset_not_read_for_dev"
        def checkpoint_dir(self):return tmp_path/"checkpoints"
        def frozen_cluster_map_path(self):return cluster
        def _run_subprocess(self,name,argv,**kwargs):
            calls.append((name,argv))
            log=tmp_path/"validation.log";log.write_text("synthetic validation failure")
            return 1,log
    harness=Harness();harness.config=config
    result=StructureStagesMixin.stage_structure_experiment(harness)
    assert result.status=="failed"  # a real validation failure still blocks completion
    assert [name for name,_ in calls]==["structure_experiment_validation_queue"]
    assert "--candidate-split" not in calls[0][1]
    assert not (tmp_path/"dev_queue").exists()
    plan=json.loads((tmp_path/"structure_execution_plan.json").read_text())
    assert plan["execution_queues"]==["validation_queue"]
    assert plan["development_targets_enabled"] is False
    assert plan["development_solvent_sensitivity_enabled"] is False


def test_completion_requires_validation_denominator_without_development_files(tmp_path):
    config={"queue_freeze":{"dev_queue":{"enabled":False}},
            "structure_experiment":{"seeds":[42],"solvent_sensitivity":["gbn2"]}}
    root=tmp_path/"validation_queue";root.mkdir()
    for name in ("run_manifest.json","seed_streams.json","eligibility.json","selected_targets.json",
                 "real_complex_metrics.csv","real_complex_report.md"):
        (root/name).write_text("fixture")
    (root/"run_summary.json").write_text(json.dumps(dict(closed=True,frozen_set_accounting_ok=True,
        structure_experiment_failed_targets=[],structure_experiment_completed_target_ids=["8ybl"])))
    target=root/"results/8ybl";target.mkdir(parents=True)
    for name in ("run_manifest.json","recovery_report.md"):(target/name).write_text("fixture")
    metrics=target/"recovery_metrics.csv"
    def write(methods):
        with metrics.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=["seed","method"]);writer.writeheader()
            writer.writerows(dict(seed=42,method=method) for method in methods)
    write(["qaoa","sa","uniform","greedy"])
    class Harness:
        run_dir=tmp_path
        _artifacts_present=staticmethod(Orchestrator._artifacts_present)
    harness=Harness();harness.config=config
    ok,detail=Orchestrator._validate_completed_stage_artifacts(harness,"structure_experiment",require_results_manifest=False)
    assert ok,detail
    write(["qaoa","sa","uniform"])
    ok,detail=Orchestrator._validate_completed_stage_artifacts(harness,"structure_experiment",require_results_manifest=False)
    assert not ok and "denominator mismatch" in detail


def test_report_ignores_stale_development_results(tmp_path,monkeypatch):
    ctx=SimpleNamespace(run_dir=tmp_path,frozen_config={"queue_freeze":{"dev_queue":{"enabled":False}}})
    stale=tmp_path/"dev_queue_solvent_gbn2";stale.mkdir()
    seen=[]
    monkeypatch.setattr(report,"stage_ok",lambda *a:True)
    monkeypatch.setattr(report,"_load_recovery_rows",lambda path:seen.append(path) or [])
    structural="\n".join(report.section_structural_benefit(ctx))
    robustness="\n".join(report.section_external_and_robustness(ctx))
    assert seen==[tmp_path/"validation_queue"]
    assert "Historical development queue" not in structural
    assert "disabled by protocol" in robustness
