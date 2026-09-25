#!/usr/bin/env bash
#
# prepare_external_vhh.sh -- two-pass preparation of the external VHH set
# (docs/PROTOCOL_AMENDMENTS.md A8). Run from anywhere; steps in order:
#
# With the antigen-fold holdout (the default, PROTOCOL_AMENDMENTS.md A10):
#   ./scripts/prepare_external_vhh.sh audit
#   ./scripts/prepare_external_vhh.sh foldseek [--foldseek /path/to/foldseek] [--threads N]
#   ./scripts/deploy_launch.sh
#
# For a GENUINELY EXTERNAL VHH set (external_vhh.graph_dir configured), build
# it first so its PDBs enter the clustering universe:
#   ./scripts/prepare_external_vhh.sh pass1 --sabdab-summary sabdab_summary_all.tsv --download
#   ./scripts/prepare_external_vhh.sh foldseek [--foldseek /path/to/foldseek]
#   ./scripts/deploy_launch.sh --stop-after queue_freeze
#   ./scripts/prepare_external_vhh.sh pass2 --sabdab-summary sabdab_summary_all.tsv \
#         --run-dir <that run directory>
#   ./scripts/deploy_launch.sh            # fresh formal run, pair table unchanged
#
# Honours QP_DATA_ROOT and QP_EXTERNAL_VHH_SOURCE_DIR. Installs nothing; reads
# no experimental outcome.
#
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [ -f "${REPO_ROOT}/.venv/bin/activate" ] && [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
elif [ -f "${REPO_ROOT}/.venv/Scripts/activate" ] && [ -f "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/Scripts/activate"
else
    SHARED_VENV="${QP_VENV:-/data/quantum-protein/.venv}"
    if [ -f "${SHARED_VENV}/bin/activate" ] && [ -x "${SHARED_VENV}/bin/python" ]; then
        # shellcheck disable=SC1091
        source "${SHARED_VENV}/bin/activate"
    elif [ -f "${SHARED_VENV}/Scripts/activate" ] && [ -f "${SHARED_VENV}/Scripts/python.exe" ]; then
        # shellcheck disable=SC1091
        source "${SHARED_VENV}/Scripts/activate"
    else
        echo "[prepare_external_vhh] ERROR: no usable virtual environment found. Set QP_VENV or create ${REPO_ROOT}/.venv." >&2
        exit 1
    fi
fi
exec python -m nanoqc.data.prepare_external_vhh "$@"
