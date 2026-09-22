#!/usr/bin/env bash
# Formal preflight gate for the full experiment.
# Any failure aborts before a new/resumed formal run is launched.
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
  test_validation_freeze_accounting.py \
  test_p1_protocol_repairs.py \
  test_p2_pipeline_repairs.py \
  test_formal_preflight_repairs.py \
  test_audit_remediation.py \
  test_robust_qaoa.py \
  test_evaluate_complex_metrics.py

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is unavailable."
log "GPU inventory:"
nvidia-smi -L

# Operational thresholds can be raised by the deployment environment without
# changing the frozen scientific protocol.
export FORMAL_MIN_FREE_DISK_GB="${FORMAL_MIN_FREE_DISK_GB:-50}"
export FORMAL_MIN_AVAILABLE_RAM_GB="${FORMAL_MIN_AVAILABLE_RAM_GB:-16}"

log "Running 2-rank DDP/NCCL all-reduce probe..."
DDP_PROBE="$(mktemp "${TMPDIR:-/tmp}/formal_ddp_probe.XXXXXX.py")"
cat >"$DDP_PROBE" <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
if world_size < 2:
    raise SystemExit(f"DDP preflight requires at least 2 ranks, got {world_size}")
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
x = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}")
dist.all_reduce(x, op=dist.ReduceOp.SUM)
expected = world_size * (world_size + 1) / 2
if abs(x.item() - expected) > 1e-6:
    raise SystemExit(f"NCCL all-reduce mismatch on rank {rank}: got {x.item()}, expected {expected}")
print(f"DDP/NCCL rank {rank}/{world_size} on cuda:{local_rank}: all_reduce={x.item()} OK", flush=True)
dist.destroy_process_group()
PY
python -m torch.distributed.run --standalone --nproc-per-node=2 "$DDP_PROBE"
rm -f "$DDP_PROBE"

log "Running CUDA/OpenMM/resource probes..."
python - <<'PY'
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import psutil
import torch
import yaml
import openmm as mm
from openmm import unit

from subgraph_to_qubo import _openmm_context

root = Path(".").resolve()
config = yaml.safe_load((root / "full_experiment_config.yaml").read_text(encoding="utf-8"))

required_gpus = int(config.get("egnn_train", {}).get("nproc_per_node", 1))
if not torch.cuda.is_available():
    raise SystemExit("PyTorch reports CUDA unavailable")
count = torch.cuda.device_count()
if count < required_gpus:
    raise SystemExit(f"Need at least {required_gpus} CUDA devices, found {count}")

for index in range(required_gpus):
    device = torch.device(f"cuda:{index}")
    x = torch.arange(1024, dtype=torch.float32, device=device)
    y = (x * x).sum()
    torch.cuda.synchronize(index)
    if not torch.isfinite(y):
        raise SystemExit(f"Non-finite CUDA probe result on cuda:{index}")
    print(f"CUDA cuda:{index}: {torch.cuda.get_device_name(index)} OK")

hardware = config.get("hardware", {})
os.environ["QP_OPENMM_PLATFORM"] = str(hardware.get("openmm_platform", "CUDA"))
os.environ["QP_OPENMM_DEVICE"] = str(hardware.get("openmm_device", "0"))
os.environ["QP_OPENMM_PRECISION"] = str(hardware.get("openmm_precision", "double"))
os.environ["OPENMM_CPU_THREADS"] = str(hardware.get("openmm_cpu_threads", 8))

system = mm.System()
system.addParticle(12.0)
force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
force.addGlobalParameter("k", 1.0)
force.addParticle(0, [])
system.addForce(force)
integrator = mm.VerletIntegrator(0.001)
context = _openmm_context(mm, system, integrator)
context.setPositions([mm.Vec3(0.1, 0.0, 0.0)] * unit.nanometer)
state = context.getState(getEnergy=True)
energy = state.getPotentialEnergy()
platform_name = context.getPlatform().getName()
if platform_name != str(hardware.get("openmm_platform", "CUDA")):
    raise SystemExit(f"OpenMM platform mismatch: expected {hardware.get('openmm_platform')}, got {platform_name}")
print(
    "OpenMM:",
    platform_name,
    "device", os.environ["QP_OPENMM_DEVICE"],
    "precision", os.environ["QP_OPENMM_PRECISION"],
    "energy", energy,
    "OK",
)
del context, integrator

run_root = Path(config["paths"]["run_root"])
run_root.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile(prefix=".preflight_", dir=run_root, delete=True) as handle:
    handle.write(b"preflight")
    handle.flush()

disk = shutil.disk_usage(run_root)
free_disk_gb = disk.free / (1024 ** 3)
min_disk_gb = float(os.environ["FORMAL_MIN_FREE_DISK_GB"])
if free_disk_gb < min_disk_gb:
    raise SystemExit(
        f"Insufficient free disk at {run_root}: {free_disk_gb:.1f} GiB < required {min_disk_gb:.1f} GiB")

available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
min_ram_gb = float(os.environ["FORMAL_MIN_AVAILABLE_RAM_GB"])
if available_ram_gb < min_ram_gb:
    raise SystemExit(
        f"Insufficient available RAM: {available_ram_gb:.1f} GiB < required {min_ram_gb:.1f} GiB")

data_root = Path(config["paths"]["data_root"])
if not data_root.is_dir():
    raise SystemExit(f"Configured data_root does not exist: {data_root}")

print(f"Disk free at {run_root}: {free_disk_gb:.1f} GiB (minimum {min_disk_gb:.1f})")
print(f"Available RAM: {available_ram_gb:.1f} GiB (minimum {min_ram_gb:.1f})")
print(f"Data root: {data_root} OK")
PY

log "FORMAL PREFLIGHT PASSED."
