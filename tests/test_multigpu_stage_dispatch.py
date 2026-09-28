"""Multi-GPU dispatch must preserve the frozen target and calibration order."""

import argparse
import concurrent.futures
import contextlib
import csv
import json
import multiprocessing as mp
import os
import pytest
from pathlib import Path

from nanoqc.experiments import generate_energy_calibration_dataset as calibration
from nanoqc.experiments import run_real_complex_pilot as pilot
from nanoqc.data import audit_all_datasets as audit
from nanoqc.data import build_foldseek_pairs as foldseek_pairs


@pytest.mark.parametrize("workers,per_gpu",[(2,1),(8,4)])
def test_calibration_shards_merge_in_training_manifest_order(tmp_path, monkeypatch,workers,per_gpu):
    def fake_shard(argv, device, log_path):
        shard=int(argv[argv.index("--shard-index")+1])
        csv_path=Path(argv[argv.index("--out-csv")+1])
        provenance_path=Path(argv[argv.index("--out-provenance")+1])
        with csv_path.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=["training_row_index","assignment_index","device"])
            writer.writeheader()
            writer.writerow({"training_row_index":shard,"assignment_index":0,"device":device})
        provenance_path.write_text(json.dumps({
            "complexes_discovered":1,"complexes_attempted":1,"complexes_succeeded":1,
            "rows_written":1,"failures":[],"input_quality_exclusions":[],
        }),encoding="utf-8")
        return 0

    monkeypatch.setattr(calibration,"_calibration_shard",fake_shard)
    args=argparse.Namespace(out_csv=tmp_path/"train.csv",out_provenance=None,
        workers=workers,gpu_devices=["0","1"],workers_per_gpu=per_gpu)
    assert calibration._run_parallel_shards(args)==0
    with args.out_csv.open(newline="",encoding="utf-8") as handle:
        rows=list(csv.DictReader(handle))
    expected=[(str(index),str(index%2)) for index in range(workers)]
    assert [(row["training_row_index"],row["device"]) for row in rows]==expected
    provenance=json.loads((tmp_path/"train.provenance.json").read_text(encoding="utf-8"))
    assert provenance["complexes_succeeded"]==workers
    assert provenance["parallel_workers"]==workers
    assert provenance["worker_gpu_assignment"]==[device for _,device in expected]
    assert provenance["workers_per_gpu_limit"]==per_gpu
    assert provenance["csv_sha256"]==calibration.sha256(args.out_csv)


def test_calibration_worker_bounds_cpu_threads_and_pins_device(tmp_path,monkeypatch):
    seen={}
    def capture(command,**kwargs):
        seen.update(kwargs["env"])
        return argparse.Namespace(returncode=0)
    monkeypatch.setattr(calibration.subprocess,"run",capture)
    monkeypatch.setenv("OMP_NUM_THREADS","80")
    assert calibration._calibration_shard([],"1",tmp_path/"worker.log")==0
    assert seen["QP_OPENMM_DEVICE"]=="1"
    for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
        assert seen[variable]=="1"


def test_structural_worker_sets_openmm_device_before_solver(monkeypatch):
    from nanoqc.experiments import batch_benchmark_hard_set as benchmark
    seen=[]
    monkeypatch.setattr(benchmark,"_recovery_benchmark_main",
        lambda argv: seen.append((list(argv),os.environ["QP_OPENMM_DEVICE"])) or 0)
    monkeypatch.setenv("QP_OPENMM_DEVICE","0")
    assert pilot._run_recovery_on_device(["--manifest","frozen.json"],"1")==0
    assert seen==[(["--manifest","frozen.json"],"1")]


def test_shared_gpu_structural_dispatch_keeps_fixed_device_and_target_order(monkeypatch):
    assignments=[]
    pools=[]
    class Pool:
        def __init__(self,**kwargs):
            for variable in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
                assert os.environ[variable]=="1"
            self.device=kwargs["initargs"][0]
            assert kwargs["initializer"] is pilot._initialize_recovery_worker
            pools.append(self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def submit(self,fn,argv,device):
            assert device==self.device
            assignments.append((argv[0],device))
            future=concurrent.futures.Future()
            future.set_result(int(argv[0]))
            return future
    monkeypatch.setattr(pilot.concurrent.futures,"ProcessPoolExecutor",Pool)
    devices=pilot._structural_worker_devices(8,["0","1"],4)
    jobs=[(str(i),None,[str(i)]) for i in range(11)]
    with contextlib.ExitStack() as stack:
        futures=pilot._submit_structural_jobs(jobs,devices,stack)
        assert [future.result() for *_,future in futures]==list(range(11))
    assert len(pools)==8
    assert assignments==[(str(i),str(i%2)) for i in range(11)]


@pytest.mark.parametrize("workers,devices,cap",[(8,["0","1"],1),(9,["0","1"],4),
    (2,["0","0"],4),(1,[],4),(1,["bad"],4),(1,["0"],0)])
def test_structural_sharing_requires_valid_explicit_limit(workers,devices,cap):
    with pytest.raises(ValueError):pilot._structural_worker_devices(workers,devices,cap)


def test_audit_spawned_processes_preserve_discovery_order(tmp_path):
    thresholds=(4.5,3.0,1.0,.9,False,True,True)
    tasks=[({"id":str(i)},{"id":str(i),"valid":True}) for i in range(4)]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=2,mp_context=mp.get_context("spawn"),
            initializer=audit._initialize_audit_process,
            initargs=(str(tmp_path),thresholds)) as pool:
        rows=list(pool.map(audit._audit_or_cached,tasks))
    assert [row["id"] for row in rows]==["0","1","2","3"]


def test_foldseek_preparation_spawned_processes_preserve_universe_order(tmp_path):
    tasks=[(pdb,[],[],[],15,tmp_path) for pdb in ("1abc","2def","3ghi")]
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=2,mp_context=mp.get_context("spawn"),
            initializer=audit.load_annotations,initargs=(tmp_path,)) as pool:
        rows=list(pool.map(foldseek_pairs._prepare_antigen_worker,tasks))
    assert [row[0] for row in rows]==["1abc","2def","3ghi"]
    assert all(row[2] and row[3]==0 for row in rows)
