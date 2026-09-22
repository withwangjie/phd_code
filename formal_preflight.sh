#!/usr/bin/env bash
# Formal preflight gate for method-upgrade-v2.
# Any failure aborts before a formal background process is created.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[formal_preflight] $*"; }
fail() { echo "[formal_preflight] ERROR: $*" >&2; exit 1; }

if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.venv/Scripts/activate"
else
    fail "No project .venv is available."
fi

log "Python: $(command -v python)"
python --version
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    log "Git HEAD: $(git rev-parse HEAD)"
fi

log "Running formal regression suite..."
python -m pytest -q \
  test_formal_blockers.py \
  test_high_medium_repairs.py \
  test_audit_remediation.py \
  test_robust_qaoa.py \
  test_evaluate_complex_metrics.py

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable."
log "GPU inventory:"
nvidia-smi -L

DDP_RANKS="$(python - <<'PY'
import yaml
from pathlib import Path
cfg=yaml.safe_load(Path("full_experiment_config.yaml").read_text(encoding="utf-8"))
print(int(cfg.get("egnn_train",{}).get("nproc_per_node",1)))
PY
)"
if [ "$DDP_RANKS" -lt 2 ]; then
    fail "Formal protocol requires at least 2 DDP ranks; config has $DDP_RANKS."
fi

log "Running ${DDP_RANKS}-rank DDP/NCCL all-reduce probe..."
DDP_PROBE="$(mktemp "${TMPDIR:-/tmp}/formal_ddp_probe.XXXXXX.py")"
trap 'rm -f "$DDP_PROBE"' EXIT
cat >"$DDP_PROBE" <<'PY'
import os
import torch
import torch.distributed as dist

rank=int(os.environ["RANK"])
local_rank=int(os.environ["LOCAL_RANK"])
world_size=int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
x=torch.tensor([float(rank+1)],device=f"cuda:{local_rank}")
dist.all_reduce(x,op=dist.ReduceOp.SUM)
expected=world_size*(world_size+1)/2
if abs(x.item()-expected)>1e-6:
    raise SystemExit(f"NCCL all-reduce mismatch on rank {rank}: {x.item()} != {expected}")
print(f"DDP/NCCL rank {rank}/{world_size} cuda:{local_rank}: OK",flush=True)
dist.destroy_process_group()
PY
python -m torch.distributed.run --standalone --nproc-per-node="$DDP_RANKS" "$DDP_PROBE"
rm -f "$DDP_PROBE"
trap - EXIT

export FORMAL_MIN_FREE_DISK_GB="${FORMAL_MIN_FREE_DISK_GB:-50}"
export FORMAL_MIN_AVAILABLE_RAM_GB="${FORMAL_MIN_AVAILABLE_RAM_GB:-16}"

log "Running CUDA/OpenMM/resource probes..."
python - <<'PY'
from __future__ import annotations
import os, shutil, tempfile
from pathlib import Path
import psutil
import torch
import yaml
import openmm as mm
from openmm import unit

cfg=yaml.safe_load(Path("full_experiment_config.yaml").read_text(encoding="utf-8"))
required_gpus=int(cfg.get("egnn_train",{}).get("nproc_per_node",1))
if not torch.cuda.is_available():
    raise SystemExit("PyTorch reports CUDA unavailable")
count=torch.cuda.device_count()
if count<required_gpus:
    raise SystemExit(f"Need >= {required_gpus} CUDA devices, found {count}")
for index in range(required_gpus):
    x=torch.arange(1024,dtype=torch.float32,device=f"cuda:{index}")
    y=(x*x).sum()
    torch.cuda.synchronize(index)
    if not torch.isfinite(y):
        raise SystemExit(f"Non-finite CUDA probe on cuda:{index}")
    print(f"CUDA cuda:{index}: {torch.cuda.get_device_name(index)} OK")

hardware=cfg.get("hardware",{})
platform_name=str(hardware.get("openmm_platform","CUDA"))
platform=mm.Platform.getPlatformByName(platform_name)
properties={}
if platform_name=="CUDA":
    properties["DeviceIndex"]=str(hardware.get("openmm_device","0"))
    properties["Precision"]=str(hardware.get("openmm_precision","double"))
system=mm.System(); system.addParticle(12.0)
force=mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
force.addGlobalParameter("k",1.0); force.addParticle(0,[]); system.addForce(force)
integrator=mm.VerletIntegrator(0.001)
context=mm.Context(system,integrator,platform,properties)
context.setPositions([mm.Vec3(0.1,0.0,0.0)]*unit.nanometer)
energy=context.getState(getEnergy=True).getPotentialEnergy()
if context.getPlatform().getName()!=platform_name:
    raise SystemExit("OpenMM platform mismatch")
print(f"OpenMM {platform_name} properties={properties} energy={energy}: OK")
del context,integrator

paths=cfg["paths"]
data_root=Path(paths["data_root"])
run_root=Path(paths["run_root"])
if not data_root.is_dir():
    raise SystemExit(f"Configured data_root missing: {data_root}")
run_root.mkdir(parents=True,exist_ok=True)
with tempfile.NamedTemporaryFile(prefix=".preflight_",dir=run_root,delete=True) as handle:
    handle.write(b"preflight"); handle.flush()

disk=shutil.disk_usage(run_root)
free_disk_gb=disk.free/(1024**3)
min_disk=float(os.environ["FORMAL_MIN_FREE_DISK_GB"])
if free_disk_gb<min_disk:
    raise SystemExit(f"Free disk {free_disk_gb:.1f} GiB < {min_disk:.1f} GiB")
available_ram_gb=psutil.virtual_memory().available/(1024**3)
min_ram=float(os.environ["FORMAL_MIN_AVAILABLE_RAM_GB"])
if available_ram_gb<min_ram:
    raise SystemExit(f"Available RAM {available_ram_gb:.1f} GiB < {min_ram:.1f} GiB")
print(f"data_root: {data_root} OK")
print(f"run_root writable: {run_root} OK")
print(f"free disk: {free_disk_gb:.1f} GiB")
print(f"available RAM: {available_ram_gb:.1f} GiB")
PY

log "FORMAL PREFLIGHT PASSED."
