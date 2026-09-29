"""Parallel preparation must not race admission or lose physical failure audits."""
import argparse
import concurrent.futures
import contextlib
import csv
import json
from pathlib import Path

from nanoqc.experiments import run_real_complex_pilot as pilot
from nanoqc.common.gpu_runtime import GPUStageMonitor
from nanoqc.structure.physical_quality import StructureQualityError


def test_preparation_prefetch_deduplicates_and_retains_input_order(tmp_path,monkeypatch):
    seen=[]
    class Pool:
        def __init__(self,**kwargs):self.device=kwargs["initargs"][0]
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,row,args,work,homology,frozen_pool,device):
            assert device==self.device
            seen.append((row["pdb_id"],row["path"],device,frozen_pool))
            future=concurrent.futures.Future();future.set_result(row)
            return future
    monkeypatch.setattr(pilot.concurrent.futures,"ProcessPoolExecutor",Pool)
    rows=[dict(pdb_id=pdb,path=path) for pdb,path in
          [("a","first"),("b","b"),("A","duplicate"),("c","c"),("d","d"),("e","e")]]
    args=argparse.Namespace(pdb_allowlist_file=tmp_path/"frozen.json",prepare_only=False)
    with contextlib.ExitStack() as stack:
        futures=pilot._prefetch_preparations(rows,args,tmp_path,{},
            {"a":{"eligibility_compatible_residues":["H:31"]}},["0","1"],stack,
            {"c"},{"d"},{"a":"1","b":"2","c":"3","d":"4","e":"5"},{"5"})
        assert list(futures)==["a","b"]
        assert [f.result()["path"] for f in futures.values()]==["first","b"]
    assert seen==[("a","first","0",{"H:31"}),("b","b","1",set())]


def test_compatibility_keeps_each_site_failure_and_requires_same_count(monkeypatch):
    calls=[]
    def check(native,ids,**kwargs):
        calls.append(ids)
        if ids==["H:2"]:raise ValueError("missing chi")
    monkeypatch.setattr(pilot,"AllAtomInterfaceQUBOBuilder",check)
    config=dict(eligible_residues=["H:1","H:2","H:3"])
    args=argparse.Namespace(eligibility_only=True,sites=2,rotamer_mode="pyrosetta_dun10",
        rotamer_library=None,rotamer_probability_floor=1e-4,rotamer_sigma_offsets=[-1,0,1],solvent_model="vacuum")
    pilot._check_prepared_sites(config,args,Path("native.cif"))
    assert config["eligibility_compatible_residues"]==["H:1","H:3"]
    assert config["eligibility_site_failures"]==[dict(residue_id="H:2",error="ValueError: missing chi")]
    assert calls==[["H:1"],["H:2"],["H:3"]]
    args.sites=3
    import pytest
    with pytest.raises(ValueError,match="Only 2"):pilot._check_prepared_sites(config,args,Path("native.cif"))


def test_preparation_physical_failure_roundtrip_keeps_category_and_audit():
    import pytest
    exc=StructureQualityError("chain break",category="input_topology",audit={"topology_passed":False})
    with pytest.raises(StructureQualityError) as caught:
        pilot._raise_preparation_error(json.loads(json.dumps(pilot._preparation_error(exc))))
    assert caught.value.category=="input_topology"
    assert caught.value.audit=={"topology_passed":False}


def test_candidate_worker_keeps_preparation_when_physical_checks_fail(tmp_path,monkeypatch):
    from nanoqc.common.repo_io import sha256_file
    from nanoqc.data import audit_all_datasets as audit
    graph_path=tmp_path/"graph.pt";graph_path.write_bytes(b"trusted fixture")
    row=dict(path="graph.pt",sha256=sha256_file(graph_path))
    args=argparse.Namespace(dataset=tmp_path,data_root=tmp_path,sites=2,pruning="contact",seeds=[42],
        checkpoint=None,antigen_guidance_weight=.25,antigen_proximity_scale=6.,contact_ca_cutoff=8.,
        eligibility_only=False,rotamer_mode="legacy")
    monkeypatch.setattr(pilot,"load_graph",lambda path:argparse.Namespace(source_id="fixture"))
    def source(source_id,root,dest):dest.write_text("raw");return dest
    monkeypatch.setattr(pilot,"extract_source",source)
    monkeypatch.setattr(audit,"materialize_graph_complex",lambda raw,graph,dest:raw)
    def prepare(graph,raw,dest,sites,**kwargs):
        dest.write_text("native")
        return dict(active_residues=["H:1","H:2"],preparation_changes=[])
    monkeypatch.setattr(pilot,"prepare",prepare)
    monkeypatch.setattr(pilot,"complete_terminal_oxygen",lambda path:[])
    def fail(*args):raise StructureQualityError("chain break",category="input_topology",audit={"bad":True})
    monkeypatch.setattr(pilot,"_check_prepared_sites",fail)
    result=pilot._prepare_candidate_on_device(row,args,tmp_path/"work",{},None,"1")
    assert result["prepare_error"] is None
    assert result["config"]["active_residues"]==["H:1","H:2"]
    assert result["physical_error"]["category"]=="input_topology"
    record=json.loads((tmp_path/"work"/"preparation_execution.json").read_text())
    assert record["device"]=="1" and record["physical_error"]["audit"]=={"bad":True}


def test_gpu_monitor_preserves_both_devices_and_never_masks_query_failure(tmp_path,monkeypatch):
    from nanoqc.common import gpu_runtime
    monkeypatch.setattr(gpu_runtime.shutil,"which",lambda _:"nvidia-smi")
    monitor=GPUStageMonitor(tmp_path/"stage.gpu.csv",enabled=True)
    def query(command,**kwargs):
        assert kwargs["timeout"]==3
        return argparse.Namespace(returncode=0,stderr="",stdout="0, GPU-0, 18, 337, 15360, 43\n1, GPU-1, 0, 3, 15360, 11\n")
    monkeypatch.setattr(gpu_runtime.subprocess,"run",query)
    rows=monitor.sample()
    assert [row[2] for row in rows]==["0","1"]
    monitor.stop.set();monitor._record()
    with monitor.path.open(newline="") as handle:recorded=list(csv.DictReader(handle))
    assert len(recorded)==2 and recorded[0]["utilization_percent"]=="18"
    monkeypatch.setattr(gpu_runtime.subprocess,"run",lambda *a,**k:argparse.Namespace(returncode=1,stderr="driver failed",stdout=""))
    assert monitor.sample()[0][-1]=="driver failed"
    with GPUStageMonitor(tmp_path/"disabled.csv",enabled=False):pass
    assert not (tmp_path/"disabled.csv").exists()
