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
SERVER_CONFIG_FILE="${QP_SERVER_CONFIG:-${REPO_ROOT}/configs/server_config.yaml}"
if [[ "$SERVER_CONFIG_FILE" != /* ]]; then
    SERVER_CONFIG_FILE="${REPO_ROOT}/${SERVER_CONFIG_FILE}"
fi
server_venv_hint() {
    [ -f "$SERVER_CONFIG_FILE" ] || return 0
    awk -F: '$1 ~ /^[[:space:]]*venv[[:space:]]*$/ {
        v=substr($0,index($0,":")+1); sub(/#.*/,"",v);
        gsub(/^[[:space:]]+|[[:space:]]+$/,"",v);
        gsub(/^"|"$/,"",v); gsub(/^\047|\047$/,"",v);
        if (v!="" && v!="auto") print v; exit
    }' "$SERVER_CONFIG_FILE"
}
SERVER_VENV_HINT="$(server_venv_hint)"
if [ -n "${QP_VENV:-}" ]; then
    SERVER_VENV_HINT=""
fi
if [ -n "$SERVER_VENV_HINT" ]; then
    case "$SERVER_VENV_HINT" in
        "~/"*) SERVER_VENV_HINT="${HOME}/${SERVER_VENV_HINT#~/}" ;;
        /*) ;;
        *) SERVER_VENV_HINT="${REPO_ROOT}/${SERVER_VENV_HINT}" ;;
    esac
fi
if [ -n "$SERVER_VENV_HINT" ]; then
    if [ -f "${SERVER_VENV_HINT}/bin/activate" ] && [ -x "${SERVER_VENV_HINT}/bin/python" ]; then
        # shellcheck disable=SC1091
        source "${SERVER_VENV_HINT}/bin/activate"
    elif [ -f "${SERVER_VENV_HINT}/Scripts/activate" ] && [ -f "${SERVER_VENV_HINT}/Scripts/python.exe" ]; then
        # shellcheck disable=SC1091
        source "${SERVER_VENV_HINT}/Scripts/activate"
    else
        echo "[prepare_external_vhh] ERROR: server_config.yaml declares an unusable venv: ${SERVER_VENV_HINT}" >&2
        exit 1
    fi
elif [ -f "${REPO_ROOT}/.venv/bin/activate" ] && [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
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
        echo "[prepare_external_vhh] ERROR: no usable virtual environment found. Set QP_VENV, configure server_config.yaml:venv, or create ${REPO_ROOT}/.venv." >&2
        exit 1
    fi
fi
exec python -m nanoqc.data.prepare_external_vhh "$@"
