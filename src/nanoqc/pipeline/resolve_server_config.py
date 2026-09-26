#!/usr/bin/env python3
"""Resolve server settings into a concrete runtime config.

Scientific parameters are loaded from full_experiment_config.yaml and preserved.
Paths, worker counts, DDP ranks, batch split, OpenMM platform/device and tool
paths may be overridden here. DDP ranks affect rank seeds and sampler partitions,
so the resolved value is a result-affecting runtime parameter.
"""
from __future__ import annotations
import argparse, json, os, shutil
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
    cpu=_cpu_count(); ram=_ram_gb(); gpus=_gpu_count()
    reserve=max(0,int(res.get("cpu_reserve_cores",2)))
    usable=max(1,cpu-reserve)
    max_workers=max(1,int(res.get("max_workers",16)))
    workers=max(1,min(max_workers,usable))
    threads=max(1,int(res.get("cpu_threads_per_process",2)))
    graph_workers=max(1,min(workers,max(1,usable//2)))
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
    hydrogen_platform=str(res.get("openmm_hydrogen_platform","Reference"))
    if hydrogen_platform not in ("Reference","CPU","CUDA"):
        raise ValueError("resources.openmm_hydrogen_platform must be Reference, CPU or CUDA")
    if hydrogen_platform=="CUDA" and gpus<1:
        raise ValueError("resources.openmm_hydrogen_platform=CUDA requires a detected CUDA GPU")
    hw["openmm_hydrogen_platform"]=hydrogen_platform
    hw["openmm_device"]=str(res.get("openmm_device","0"))
    hw["openmm_precision"]=str(res.get("openmm_precision","double"))

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

    report={
        "repo_root":str(REPO_ROOT),
        "venv":_resolve_venv(server),
        "data_root":data_root,
        "run_root":run_root,
        "external_vhh_source_dir":out.get("external_validation",{}).get(
            "external_vhh",{}).get("source_structure_dir"),
        "cpu_physical_cores":cpu,
        "ram_gb":ram,
        "cuda_gpus":gpus,
        "ddp_ranks":ranks,
        "result_affecting_runtime_parameters":["ddp_ranks","openmm_platform",
                                               "openmm_hydrogen_platform","openmm_precision"],
        "target_global_batch_size":target_global,
        "per_gpu_batch_size":per_gpu_batch,
        "data_audit_workers":workers,
        "graph_build_workers":graph_workers,
        "qc_workers":workers,
        "egnn_loader_workers":loader_workers,
        "cpu_threads_per_process":hw["cpu_threads_per_process"],
        "openmm_platform":hw["openmm_platform"],
        "openmm_hydrogen_platform":hw["openmm_hydrogen_platform"],
        "openmm_device":hw["openmm_device"],
        "openmm_precision":hw["openmm_precision"],
        "faspr_executable":structural.get("faspr_executable"),
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
