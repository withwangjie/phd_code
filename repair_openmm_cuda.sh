#!/usr/bin/env bash
# Install CUDA extra in the EXISTING server venv. No training or trial runs.
set -euo pipefail
VENV="${1:-/data/quantum-protein/.venv}"
PYTHON="$VENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Missing venv interpreter: $PYTHON" >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo "NVIDIA driver utility missing; repair the host driver first." >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version --format=csv
VERSION="$($PYTHON -c 'from importlib.metadata import version; print(version("openmm"))')"
LOGDIR="${PWD}/openmm_cuda_repair_$(date -u +%Y%m%dT%H%M%SZ)_$$"
mkdir -p "$LOGDIR"
"$PYTHON" -m pip freeze > "$LOGDIR/packages_before.txt"
# Freeze existing numerical/runtime distributions as constraints. If the extra
# conflicts, pip stops instead of silently changing the existing CUDA stack.
"$PYTHON" -c 'from importlib.metadata import distributions; import re; print("\n".join(sorted({d.metadata["Name"]+"=="+d.version for d in distributions() if d.metadata.get("Name") and re.fullmatch(r"[A-Za-z0-9_.+-]+",d.metadata["Name"])})))' > "$LOGDIR/constraints.txt"
echo "Installing CUDA 12 extra for existing OpenMM $VERSION using configured pip index."
"$PYTHON" -m pip install --constraint "$LOGDIR/constraints.txt" "openmm[cuda12]==${VERSION}" 2>&1 | tee "$LOGDIR/install.log"
"$PYTHON" -m pip freeze > "$LOGDIR/packages_after.txt"
# Registration/import diagnostics only: does not create a simulation Context.
"$PYTHON" - <<'PY' 2>&1 | tee "$LOGDIR/platforms.log"
import json
import openmm as mm
names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
print(json.dumps({"platforms": names, "plugin_directory": mm.Platform.getDefaultPluginsDirectory(),
                  "plugin_load_failures": list(mm.Platform.getPluginLoadFailures())}, indent=2))
if "CUDA" not in names:
    raise SystemExit("CUDA is still unregistered. Inspect plugin_load_failures and install.log; do not resume eligibility.")
print("CUDA platform registered. Actual device/context execution is checked by the formal run.")
PY
echo "Repair logs: $LOGDIR"
