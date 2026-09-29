#!/usr/bin/env python3
"""Resolve server settings into a concrete runtime config.

Scientific parameters are loaded from full_experiment_config.yaml and preserved.
Paths, worker counts, DDP ranks, batch split, OpenMM platform/device and tool
paths may be overridden here. DDP ranks affect rank seeds and sampler partitions,
so the resolved value is a result-affecting runtime parameter.
"""
from __future__ import annotations
import argparse, json, math, os, shutil
from pathlib import Path
from typing import Any
import yaml

REPO_ROOT=Path(__file__).resolve().parents[3]

def _load(path: Path) -> dict[str,Any]:
    data=yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data,dict):
        raise SystemExit(f"Expected YAML mapping: {path}")
    return data

def _cpu_count() -> int:
    try:
        import psutil
        return int(psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    except Exception:
        return int(os.cpu_count() or 1)

def _logical_cpu_count() -> int:
    """Logical CPUs this process may run on: CPU affinity, then any cgroup v2 quota.

    Utilization reported by top/htop/psutil is a fraction of these logical CPUs,
    so a utilization target is expressed against this count.
    """
    try:
        count=len(os.sched_getaffinity(0))
    except (AttributeError,OSError):
        count=int(os.cpu_count() or 1)
    try:
        quota,period=Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()[:2]
        if quota!="max":
            count=min(count,max(1,math.ceil(int(quota)/int(period))))
    except (OSError,ValueError):
        pass
    return max(1,count)

def _ram_gb() -> float:
    try:
        import psutil
        return float(psutil.virtual_memory().total)/(1024**3)
    except Exception:
        return 0.0

def _gpu_count() -> int:
    try:
        import torch
        return int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        return 0

def _find_existing(candidates: list[str|Path], *, executable: bool=False) -> str|None:
    for raw in candidates:
        if not raw:
            continue
        s=str(raw)
        if not Path(s).is_absolute() and "/" not in s and "\\" not in s:
            found=shutil.which(s)
            if found:
                return found
            continue
        p=Path(s).expanduser()
        if p.exists() and (not executable or os.access(p,os.X_OK)):
            return str(p.resolve())
    return None

def _resolve_venv(server: dict[str,Any]) -> str:
    def valid(path: Path) -> bool:
        posix_python=path/"bin/python"
        windows_python=path/"Scripts/python.exe"
        return (
            (path/"bin/activate").is_file()
            and posix_python.is_file()
            and os.access(posix_python,os.X_OK)
        ) or (
            (path/"Scripts/activate").is_file()
            and windows_python.is_file()
        )

    env=os.environ.get("QP_VENV")
    if env:
        p=Path(str(env)).expanduser().resolve()
        if not valid(p):
            raise SystemExit(f"QP_VENV is invalid: {p}")
        return str(p)
    raw=server.get("venv","auto")
    if raw!="auto":
        p=Path(str(raw)).expanduser().resolve()
        if not valid(p):
            raise SystemExit(f"Configured venv is invalid: {p}")
        return str(p)
    candidates=[
        REPO_ROOT/".venv",
        "/data/quantum-protein/.venv",
    ]
    for raw_candidate in candidates:
        if not raw_candidate:
            continue
        candidate=Path(str(raw_candidate)).expanduser().resolve()
        if valid(candidate):
            return str(candidate)
    raise SystemExit(
        "Unable to resolve a usable virtual environment. "
        "Set QP_VENV or server_config.yaml:venv."
    )

def _resolve_data_root(server: dict[str,Any]) -> str:
    raw=(server.get("paths") or {}).get("data_root","auto")
    if raw!="auto":
        return str(Path(str(raw)).expanduser().resolve())
    env=os.environ.get("QP_DATA_ROOT")
    candidates=[env, "/data/quantum-protein/data", REPO_ROOT/"data"]
    found=_find_existing([p for p in candidates if p])
    if not found:
        raise SystemExit("Unable to resolve data_root. Set QP_DATA_ROOT or server_config.yaml paths.data_root.")
    return found

def _resolve_run_root(server: dict[str,Any], data_root: str) -> str:
    raw=(server.get("paths") or {}).get("run_root","auto")
    if raw!="auto":
        return str(Path(str(raw)).expanduser().resolve())
    env=os.environ.get("QP_RUN_ROOT")
    if env:
        return str(Path(env).expanduser().resolve())
    data_parent=Path(data_root).resolve().parent
    return str((data_parent/"runs").resolve())

def _resolve_tool(server: dict[str,Any], key: str, env_name: str, common: list[str]) -> str|None:
    raw=(server.get("tools") or {}).get(key,"auto")
    if raw!="auto":
        found=_find_existing([str(raw)],executable=True)
        if not found:
            raise SystemExit(f"Configured executable not found/executable for {key}: {raw}")
        return found
    candidates=[os.environ.get(env_name), key.replace("_executable",""), *common]
    return _find_existing([p for p in candidates if p],executable=True)

def _choose_ddp_ranks(gpus: int, target_global_batch: int, max_gpus: int) -> int:
    if gpus <= 0:
        return 0
    for ranks in range(min(gpus,max_gpus,target_global_batch),0,-1):
        if target_global_batch % ranks == 0:
            return ranks
    return 1

def resolve(scientific: dict[str,Any], server: dict[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    out=json.loads(json.dumps(scientific))
    res=server.get("resources") or {}
    physical=_cpu_count(); ram=_ram_gb(); gpus=_gpu_count()
    basis=str(res.get("cpu_count_basis","physical"))
    if basis not in ("physical","logical"):
        raise SystemExit("cpu_count_basis must be 'physical' or 'logical'")
    cpu=_logical_cpu_count() if basis=="logical" else physical
    reserve=max(0,int(res.get("cpu_reserve_cores",2)))
    usable=max(1,cpu-reserve)
    threads=max(1,int(res.get("cpu_threads_per_process",2)))
    target_utilization=res.get("target_cpu_utilization")
    if target_utilization is not None:
        target_utilization=float(target_utilization)
        if not 0<target_utilization<=1:
            raise SystemExit("target_cpu_utilization must be in (0, 1]")
    min_workers=int(res.get("min_workers",1))
    if str(res.get("max_workers",16))=="auto":
        # Size independent CPU pools to the utilization target of the counted CPUs.
        if target_utilization is None:
            raise SystemExit("max_workers: auto requires target_cpu_utilization")
        max_workers=max(1,int(target_utilization*cpu)//threads)
        if min_workers<1:
            raise SystemExit("CPU worker range requires 1 <= min_workers")
    else:
        max_workers=max(1,int(res.get("max_workers",16)))
        if not 1<=min_workers<=max_workers:
            raise SystemExit("CPU worker range requires 1 <= min_workers <= max_workers")
    workers=max(1,min(max_workers,max(1,usable//threads)))
    worker_ram_gb=res.get("worker_ram_gb")
    ram_limited=False
    if worker_ram_gb is not None and ram>0:
        # Never size a pool past what RAM holds above the preflight free-RAM floor.
        headroom=ram-float(res.get("min_available_ram_gb",16))
        ram_workers=max(1,int(headroom//float(worker_ram_gb)))
        ram_limited=ram_workers<workers
        workers=min(workers,ram_workers)
    # Graph building runs in spawned processes (A35), each with its own
    # GRAPH_MEMORY_BUDGET_BYTES edge-allocation budget, so RAM bounds it.
    graph_ram_gb=res.get("graph_worker_ram_gb")
    graph_cap=res.get("max_graph_workers",workers)
    if str(graph_cap)=="auto":
        if graph_ram_gb is None:
            raise SystemExit("max_graph_workers: auto requires graph_worker_ram_gb")
        graph_cap=workers
    graph_workers=max(1,min(workers,int(graph_cap)))
    graph_ram_limited=False
    if graph_ram_gb is not None and ram>0:
        headroom=ram-float(res.get("min_available_ram_gb",16))
        graph_ram_workers=max(1,int(headroom//float(graph_ram_gb)))
        graph_ram_limited=graph_ram_workers<graph_workers
        graph_workers=min(graph_workers,graph_ram_workers)
    loader_workers=max(1,min(int(res.get("max_egnn_loader_workers",8)),usable))

    target_global=max(1,int(res.get("target_global_batch_size",4)))
    max_ddp=max(1,int(res.get("max_ddp_gpus",4)))
    ranks=_choose_ddp_ranks(gpus,target_global,max_ddp)
    require_cuda=bool(res.get("require_cuda",True))
    if require_cuda and ranks < 1:
        raise SystemExit("CUDA is required by server_config.yaml but no CUDA GPU was detected.")
    if ranks < 1:
        ranks=1
    per_gpu_batch=max(1,target_global//ranks)

    data_root=_resolve_data_root(server)
    run_root=_resolve_run_root(server,data_root)
    Path(run_root).mkdir(parents=True,exist_ok=True)

    out.setdefault("paths",{})["repo_root"]=str(REPO_ROOT)
    out["paths"]["data_root"]=data_root
    out["paths"]["run_root"]=run_root

    external_source=os.environ.get("QP_EXTERNAL_VHH_SOURCE_DIR") or (
        (server.get("paths") or {}).get("external_vhh_source_dir")
    )
    if external_source and external_source!="auto":
        out.setdefault("external_validation",{}).setdefault("external_vhh",{})[
            "source_structure_dir"]=str(Path(str(external_source)).expanduser().resolve())

    out.setdefault("data_audit",{})["workers"]=workers
    out.setdefault("queue_freeze",{}).setdefault("graph_build",{})["workers"]=graph_workers
    out.setdefault("egnn_train",{})["nproc_per_node"]=ranks
    out["egnn_train"]["batch_size"]=per_gpu_batch
    out["egnn_train"]["threads"]=max(1,min(threads,usable))
    out["egnn_train"]["num_workers"]=loader_workers
    out.setdefault("qc_benchmark",{})["workers"]=workers

    hw=out.setdefault("hardware",{})
    hw["cpu_threads_per_process"]=max(1,min(threads,usable))
    hw["openmm_cpu_threads"]=max(1,min(int(res.get("openmm_cpu_threads",8)),usable))
    platform=str(res.get("openmm_platform","auto"))
    hw["openmm_platform"]="CUDA" if platform=="auto" and gpus>0 else ("CPU" if platform=="auto" else platform)
    hw["openmm_device"]=str(res.get("openmm_device","0"))
    hw["openmm_precision"]=str(res.get("openmm_precision","double"))
    max_openmm_gpus=max(1,int(res.get("max_openmm_gpus",1)))
    hw["structural_gpu_devices"]=(
        [str(index) for index in range(min(gpus,max_openmm_gpus))]
        if hw["openmm_platform"]=="CUDA" else [hw["openmm_device"]]
    )
    per_device=int(res.get("structural_workers_per_gpu",1))
    if per_device<1:
        raise SystemExit("structural_workers_per_gpu must be positive")
    hw["structural_workers_per_gpu"]=per_device if hw["openmm_platform"]=="CUDA" else 1
    hw["structural_target_workers"]=len(hw["structural_gpu_devices"])*hw["structural_workers_per_gpu"]
    for kind in ("structural_prepare","calibration"):
        per_device=int(res.get(kind+"_workers_per_gpu",1))
        if per_device<1:raise SystemExit(kind+"_workers_per_gpu must be positive")
        hw[kind+"_workers_per_gpu"]=per_device if hw["openmm_platform"]=="CUDA" else 1
        hw[kind+"_workers"]=len(hw["structural_gpu_devices"])*hw[kind+"_workers_per_gpu"]
    hw["gpu_monitor_enabled"]=bool(res.get("gpu_monitor_enabled",False))
    hw["gpu_monitor_interval_seconds"]=float(res.get("gpu_monitor_interval_seconds",10))
    if hw["gpu_monitor_interval_seconds"]<1:raise SystemExit("GPU monitor interval must be >=1 second")
    hw["cpu_monitor_enabled"]=bool(res.get("cpu_monitor_enabled",False))
    hw["cpu_monitor_interval_seconds"]=float(res.get("cpu_monitor_interval_seconds",10))
    if hw["cpu_monitor_interval_seconds"]<1:raise SystemExit("CPU monitor interval must be >=1 second")
    # "auto" follows the independent CPU pool; a number keeps its own cap (A36).
    def _cpu_pool_limit(key: str, default: int) -> int:
        value=res.get(key,default)
        return workers if str(value)=="auto" else max(1,min(workers,int(value)))
    hw["external_audit_workers"]=_cpu_pool_limit("max_external_audit_workers",4)
    hw["foldseek_prepare_workers"]=_cpu_pool_limit("max_foldseek_prepare_workers",4)
    hw["exploration_depth_workers"]=max(1,min(
        int(res.get("max_exploration_depth_workers",1)),workers))
    hw["sensitivity_case_workers"]=max(1,min(
        int(res.get("max_sensitivity_case_workers",1)),workers))
    hw["statistics_mode_workers"]=max(1,min(
        int(res.get("max_statistics_mode_workers",1)),workers))
    # Development-only seed replicates share the GPUs; each is a full DDP job,
    # so the bound is per-GPU memory, not core count (A35).
    hw["egnn_replicate_workers"]=max(1,int(res.get("max_egnn_replicate_workers",1)))

    structural=out.setdefault("external_validation",{}).setdefault("structural_baselines",{})
    faspr=_resolve_tool(
        server,"faspr_executable","QP_FASPR",
        ["FASPR","/opt/FASPR/FASPR"],
    )
    phenix=_resolve_tool(
        server,"phenix_clashscore_executable","QP_PHENIX_CLASHSCORE",
        ["phenix.clashscore","/opt/phenix/phenix.clashscore"],
    )
    if faspr:
        structural["faspr_executable"]=faspr
    if phenix:
        structural["phenix_clashscore_executable"]=phenix
    if bool(structural.get("required",False)):
        if not faspr:
            raise SystemExit("Required FASPR executable could not be auto-resolved. Set QP_FASPR or server_config.yaml.")
        if not phenix:
            raise SystemExit("Required phenix.clashscore executable could not be auto-resolved. Set QP_PHENIX_CLASHSCORE or server_config.yaml.")

    clustering=((out.get("queue_freeze",{}) or {}).get("independence_clustering",{}) or {})
    foldseek=None
    if clustering.get("required",False) and clustering.get("build_per_run",False):
        foldseek=_resolve_tool(server,"foldseek_executable","QP_FOLDSEEK",["foldseek"])
        if not foldseek:
            raise SystemExit("Required Foldseek executable could not be resolved. Set QP_FOLDSEEK or server_config.yaml:tools.foldseek_executable.")

    report={
        "repo_root":str(REPO_ROOT),
        "venv":_resolve_venv(server),
        "data_root":data_root,
        "run_root":run_root,
        "external_vhh_source_dir":out.get("external_validation",{}).get(
            "external_vhh",{}).get("source_structure_dir"),
        "cpu_physical_cores":physical,
        "cpu_count_basis":basis,
        "cpu_counted":cpu,
        "target_cpu_utilization":target_utilization,
        # Share of the counted CPUs kept busy when an independent CPU pool is full.
        "cpu_pool_utilization":round(workers*threads/cpu,4),
        "ram_gb":ram,
        "cuda_gpus":gpus,
        "ddp_ranks":ranks,
        "result_affecting_runtime_parameters":["ddp_ranks","openmm_platform",
            "openmm_precision","structural_gpu_devices"],
        "target_global_batch_size":target_global,
        "per_gpu_batch_size":per_gpu_batch,
        "data_audit_workers":workers,
        "cpu_min_workers_target":min_workers,
        "cpu_max_workers":max_workers,
        "cpu_min_workers_target_met":workers>=min_workers,
        "cpu_worker_limit_reason":(
            ("RAM per worker" if ram_limited else "available cores / threads per process / utilization target")
            if workers<min_workers else None),
        "worker_ram_gb":None if worker_ram_gb is None else float(worker_ram_gb),
        "graph_build_workers":graph_workers,
        "graph_worker_ram_gb":None if graph_ram_gb is None else float(graph_ram_gb),
        "graph_build_workers_ram_limited":graph_ram_limited,
        "qc_workers":workers,
        "egnn_loader_workers":loader_workers,
        "cpu_threads_per_process":hw["cpu_threads_per_process"],
        "openmm_platform":hw["openmm_platform"],
        "openmm_device":hw["openmm_device"],
        "openmm_precision":hw["openmm_precision"],
        "structural_gpu_devices":hw["structural_gpu_devices"],
        "structural_target_workers":hw["structural_target_workers"],
        "structural_workers_per_gpu":hw["structural_workers_per_gpu"],
        "structural_prepare_workers":hw["structural_prepare_workers"],
        "calibration_workers":hw["calibration_workers"],
        "external_audit_workers":hw["external_audit_workers"],
        "foldseek_prepare_workers":hw["foldseek_prepare_workers"],
        "exploration_depth_workers":hw["exploration_depth_workers"],
        "sensitivity_case_workers":hw["sensitivity_case_workers"],
        "statistics_mode_workers":hw["statistics_mode_workers"],
        "egnn_replicate_workers":hw["egnn_replicate_workers"],
        "faspr_executable":structural.get("faspr_executable"),
        "foldseek_executable":foldseek,
        "phenix_clashscore_executable":structural.get("phenix_clashscore_executable"),
        "min_free_disk_gb":float(res.get("min_free_disk_gb",50)),
        "min_available_ram_gb":float(res.get("min_available_ram_gb",16)),
    }
    out.setdefault("runtime_resolution",{}).update(report)
    return out,report

def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--scientific-config",type=Path,default=REPO_ROOT/"configs"/"full_experiment_config.yaml")
    p.add_argument("--server-config",type=Path,default=REPO_ROOT/"configs"/"server_config.yaml")
    p.add_argument("--out-config",type=Path,required=True)
    p.add_argument("--out-report",type=Path,required=True)
    args=p.parse_args()
    scientific=_load(args.scientific_config)
    server=_load(args.server_config)
    resolved,report=resolve(scientific,server)
    args.out_config.write_text(yaml.safe_dump(resolved,sort_keys=False,allow_unicode=True),encoding="utf-8")
    args.out_report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
