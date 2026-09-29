"""Historical exclusions cannot dictate nonexistent independent-test targets."""
import csv
import json
from pathlib import Path

import pytest
import yaml

from nanoqc.experiments import run_real_complex_pilot as pilot
from nanoqc.pipeline.stages_structure import StructureStagesMixin


def _rows():
    return [dict(split="train",pdb_id="4S10",nodes=300,path="graphs/train/large.pt"),
            dict(split="train",pdb_id="4S10",nodes=200,path="graphs/train/small.pt"),
            dict(split="train",pdb_id="9GCN",nodes=250,path="graphs/train/9gcn.pt"),
            dict(split="test_snac_hard",pdb_id="8YBL",nodes=350,path="graphs/test_snac_hard/8ybl.pt")]


def test_training_recovery_has_no_independent_training_comparison_pool():
    training,candidates=pilot._pilot_populations(_rows(),"train","dev")
    assert training==[]
    assert [row["path"] for row in candidates]==[
        "graphs/train/small.pt","graphs/train/9gcn.pt","graphs/train/large.pt"]


def test_validation_population_and_training_isolation_are_preserved():
    training,candidates=pilot._pilot_populations(_rows(),"test_snac_hard","validation")
    assert len(training)==3
    assert [row["pdb_id"] for row in candidates]==["8YBL"]
    with pytest.raises(ValueError,match="only for the development"):
        pilot._pilot_populations(_rows(),"train","validation")


@pytest.mark.parametrize("flags",[["--queue-role","validation","--pdb-id","4s10"],[]])
def test_cli_rejects_training_as_validation_or_unrestricted_development(flags):
    with pytest.raises(SystemExit):
        pilot.main(["--candidate-split","train",*flags])


def test_orchestrator_uses_available_training_targets_and_audits_missing_ids(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    config=yaml.safe_load((repo/"configs/full_experiment_config.yaml").read_text(encoding="utf-8"))
    # Exercise old frozen configs as well: only the historical exclusion field existed.
    config["queue_freeze"]["dev_queue"].pop("target_pdb_ids",None)
    config["queue_freeze"]["dev_queue"].pop("candidate_split",None)
    config["queue_freeze"]["dev_queue"].pop("enabled",None)
    config["paths"]["data_root"]=str(tmp_path/"data")
    config["paths"]["repo_root"]=str(repo)
    dataset=tmp_path/"dataset";dataset.mkdir()
    with (dataset/"graph_manifest.csv").open("w",newline="") as handle:
        rows=_rows();writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    checkpoints=tmp_path/"checkpoints";checkpoints.mkdir()
    (checkpoints/"best_egnn_pruning.pt").write_bytes(b"fixture checkpoint")
    cluster=tmp_path/"clusters.json";cluster.write_text("{}")
    calls=[]
    class Harness:
        run_dir=tmp_path
        venv_python="python"
        def dataset_dir(self):return dataset
        def checkpoint_dir(self):return checkpoints
        def frozen_cluster_map_path(self):return cluster
        def _run_subprocess(self,name,argv,**kwargs):
            calls.append((name,argv))
            log=tmp_path/(name+".log");log.write_text("injected failure")
            return 1,log
    harness=Harness();harness.config=config
    result=StructureStagesMixin.stage_structure_experiment(harness)
    assert result.status=="failed"  # synthetic execution failures must remain failures
    assert sorted(name for name,_ in calls)==["structure_experiment_dev_4s10","structure_experiment_dev_9gcn"]
    assert all(argv[argv.index("--candidate-split")+1]=="train" for _,argv in calls)
    availability=json.loads((tmp_path/"dev_queue/target_availability.json").read_text())
    assert availability["unavailable_target_ids"]==["8yvo"]
    assert availability["historical_validation_exclusions"]==["4s10","8yvo","9gcn"]
    summary=json.loads((tmp_path/"dev_queue/run_summary.json").read_text())
    assert summary["planned_target_ids"]==["4s10","9gcn"]
    assert summary["unavailable_target_ids"]==["8yvo"]
