"""Infrastructure faults must abort, never shrink a scientific denominator."""
import argparse
import concurrent.futures
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from nanoqc.common.device_errors import DeviceResourceError, is_resource_error, raise_if_resource_error
from nanoqc.common.repo_io import sha256_file
from nanoqc.experiments import run_real_complex_pilot as pilot
from nanoqc.experiments import generate_energy_calibration_dataset as calibration
from nanoqc.experiments import training_energy_diagnostic as diagnostic
from nanoqc.experiments import structure_benchmarks as structures


@pytest.mark.parametrize("error", [MemoryError(), RuntimeError("CUDA error: out of memory"),
                                    RuntimeError("CUDA_ERROR_ILLEGAL_ADDRESS")])
def test_site_resource_fault_cannot_reduce_compatible_pool(monkeypatch, error):
    calls=[]
    def builder(native, ids, **kwargs):
        calls.extend(ids)
        if ids==["H:1"]:
            raise error
        return object()
    monkeypatch.setattr(pilot,"AllAtomInterfaceQUBOBuilder",builder)
    config=dict(eligible_residues=["H:1","H:2","H:3"])
    args=argparse.Namespace(eligibility_only=True,sites=2,rotamer_mode="legacy",
        rotamer_library=None,rotamer_probability_floor=1e-4,
        rotamer_sigma_offsets=[-1,0,1],solvent_model="vacuum")
    with pytest.raises(DeviceResourceError):
        pilot._check_prepared_sites(config,args,Path("unused.cif"))
    assert calls==["H:1"]
    assert "eligibility_compatible_residues" not in config


def test_wrapped_memory_fault_and_spawn_serialization_keep_identity(tmp_path):
    original=DeviceResourceError("allocation stopped",device="1",stage_hint="preparation")
    restored=pickle.loads(pickle.dumps(original))
    assert restored.device=="1" and restored.stage_hint=="preparation"
    wrapper=RuntimeError("worker failed")
    wrapper.__cause__=MemoryError()
    assert is_resource_error(wrapper)
    with pytest.raises(DeviceResourceError):
        raise_if_resource_error(restored,record_path=tmp_path/"fault.json")
    report=json.loads((tmp_path/"fault.json").read_text())
    assert report["device"]=="1" and report["policy"]=="stage_abort_not_sample_exclusion"
    assert not is_resource_error(ValueError("Missing heavy atoms H:1: ['CG']"))


def test_unadmitted_prefetched_candidate_resource_fault_aborts(tmp_path):
    future=concurrent.futures.Future()
    error=DeviceResourceError("CUDA out of memory",device="1")
    future.set_result(dict(prepare_error=None,physical_error=pilot._preparation_error(error)))
    with pytest.raises(DeviceResourceError):
        pilot._verify_preparation_resources({"not_selected":future},tmp_path)
    assert json.loads((tmp_path/"device_resource_failure.json").read_text())["device"]=="1"


def test_parent_eligibility_does_not_exclude_resource_failed_target(tmp_path,monkeypatch):
    dataset=tmp_path/"dataset";dataset.mkdir()
    graph=dataset/"graph.pt";graph.write_bytes(b"fixture")
    row=dict(split="test_snac_hard",pdb_id="1abc",nodes=2,path="graph.pt",sha256=sha256_file(graph))
    with (dataset/"graph_manifest.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    monkeypatch.setattr(pilot,"load_graph",lambda _:object())
    error=pilot._preparation_error(DeviceResourceError("CUDA out of memory",device="1"))
    monkeypatch.setattr(pilot,"_prepare_candidate_on_device",lambda *a:
                        dict(config=None,raw=None,prepare_error=error,physical_error=None))
    out=tmp_path/"out"
    with pytest.raises(DeviceResourceError):
        pilot.main(["--dataset",str(dataset),"--data-root",str(tmp_path),"--out-dir",str(out),
                    "--pruning","contact","--rotamer-mode","legacy","--prepare-only"])
    assert not (out/"selected_targets.json").exists()
    assert not (out/"eligibility.json").exists()
    assert (out/"device_resource_failure.json").is_file()


def test_diagnostic_resource_fault_retains_measurements_but_aborts(tmp_path,monkeypatch):
    class Builder:
        topology=object()
        def positions_for_chi_assignment(self,angles):return np.ones((2,3))
        def write_structure(self,positions,path):path.write_text("raw coordinates")
        def energy(self,positions):return 123.
        def energy_components(self):return {"NonbondedForce":120.}
        def relax_positions(self,*a,**k):raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(diagnostic,"topology_geometry_audit",lambda *a:
                        dict(geometry_passed=True,closest_nonbonded_pair=None))
    with pytest.raises(DeviceResourceError):
        diagnostic.diagnostic_assignment(Builder(),{},tmp_path/"state.cif",200)
    saved=json.loads((tmp_path/"state.json").read_text())
    assert saved["result"]["raw_amber_kcal"]==123.
    assert (tmp_path/"state_discrete.cif").exists()
    assert (tmp_path/"state_resource_failure.json").exists()


def test_calibration_shard_resource_exit_cannot_publish_merged_denominator(tmp_path,monkeypatch):
    def fail(argv,device,log):
        log.write_text("RuntimeError: CUDA out of memory")
        return 1
    monkeypatch.setattr(calibration,"_calibration_shard",fail)
    args=argparse.Namespace(out_csv=tmp_path/"train.csv",out_provenance=None,
                            workers=2,gpu_devices=["0","1"],workers_per_gpu=1)
    with pytest.raises(DeviceResourceError):
        calibration._run_parallel_shards(args)
    assert not args.out_csv.exists()
    assert not args.out_csv.with_suffix(".provenance.json").exists()
    assert (tmp_path/"device_resource_failure.json").exists()


def test_recovery_seed_fault_aborts_without_failed_seed_summary(tmp_path,monkeypatch):
    native=tmp_path/"native.cif";native.write_text("fixture")
    case=dict(native_structure="native.cif",active_residues=["H:1"],alignment_residues=["H:2"],
              partner_residues=["A:1"],selection_origin="fixture")
    manifest=tmp_path/"case.json";manifest.write_text(json.dumps(case))
    class Builder:
        input_quality={"geometry_passed":True}
        preparation_quality={}
        def __init__(self,*a,**k):pass
        def perturb_sidechain_chis(self,*a):raise RuntimeError("CUDA out of memory")
    from nanoqc.qubo import subgraph_to_qubo
    monkeypatch.setattr(subgraph_to_qubo,"AllAtomInterfaceQUBOBuilder",Builder)
    out=tmp_path/"recovery"
    with pytest.raises(DeviceResourceError):
        structures._recovery_benchmark_main(["--manifest",str(manifest),"--out-dir",str(out),"--seeds","42"])
    assert (out/"device_resource_failure.json").exists()
    assert not (out/"recovery_quality_summary.json").exists()
    assert not (out/"failed_cases.log").exists()


def test_cpu_audit_cannot_label_memory_fault_as_invalid_structure(monkeypatch):
    from nanoqc.data import audit_all_datasets as audit
    def fail(*a,**k):raise MemoryError()
    monkeypatch.setattr(audit,"read_structure",fail)
    with pytest.raises(DeviceResourceError):
        audit.audit(dict(id="fixture",subset="train_rcsb",path="unused.pdb"))


def test_graph_build_cannot_label_memory_fault_as_failed_graph(tmp_path,monkeypatch):
    from nanoqc.data import build_final_pyg_dataset as graphs
    def fail(*a,**k):raise MemoryError()
    monkeypatch.setattr(graphs,"verify_audited_source",fail)
    with pytest.raises(DeviceResourceError):
        graphs.save_graph(dict(id="fixture",subset="train_rcsb",pdb_id="1abc"),"train",tmp_path)
    assert (tmp_path/"device_resource_failure.json").exists()


def test_qc_worker_cannot_return_resource_fault_as_failed_target(monkeypatch):
    from nanoqc.experiments import hard_set_evaluation as qc
    monkeypatch.setattr(qc,"_WORKER_SCORER",object())
    monkeypatch.setattr(qc,"_WORKER_ARGS",object())
    def fail(*a,**k):raise MemoryError()
    monkeypatch.setattr(qc,"evaluate_single_target",fail)
    with pytest.raises(DeviceResourceError):
        qc._evaluate_worker("unused.pt")


def test_qaoa_resource_fault_is_not_an_algorithmic_restart_failure(monkeypatch):
    from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler
    from nanoqc.solvers import qaoa_optimization
    sampler=XYMixerQAOASampler([0.,2.,0.,5.],np.zeros((4,4)),{0:[0,1],1:[2,3]},
                               p=2,simulation_mode="subspace",seed=42)
    def fail(*a,**k):raise MemoryError()
    monkeypatch.setattr(qaoa_optimization,"minimize",fail)
    with pytest.raises(DeviceResourceError):
        sampler.optimize_robust(max_evals=25,restarts=4,eval_shots=50)
