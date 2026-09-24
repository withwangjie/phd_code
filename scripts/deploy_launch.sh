#!/usr/bin/env bash
#
# deploy_launch.sh -- the single command this deployment's README asks the
# user to run after extracting the bundle: it wires this versioned code
# directory to the server's already-existing shared Python environment
# (never installing anything itself), then delegates straight to
# run_full_experiment.sh, which launches the full formal pipeline in the
# background (survives SSH disconnect), writes its PID to a lock file, and
# refuses a second concurrent launch. See run_full_experiment.sh --help /
# its own header comment for --resume, --only, --force-restage.
#
# This script installs NOTHING and connects to no network resource on its
# own; it only creates one symlink (idempotent) and execs an already-present
# script. If the shared environment below does not already exist on this
# server, it stops with a clear error instead of guessing.
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root: the .venv link lives there (see resolve_server_config.py).
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

log()  { echo "[deploy_launch] $*"; }
fail() { echo "[deploy_launch] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# The project's dependencies (torch, torch-geometric, pennylane, openmm,
# ...) are assumed to already be installed in a shared venv at this fixed
# server path from this project's earlier setup (the same one the formal
# validation instructions this session referenced with
# `source /data/quantum-protein/.venv/bin/activate`). This deployment never
# runs `pip install` itself -- if that environment is missing or was moved,
# fix SHARED_VENV below (or pass --venv <path>) and re-run, rather than
# letting run_full_experiment.sh silently fall back to a bare `python3`
# that is very likely missing this project's dependencies.
# ---------------------------------------------------------------------------
SHARED_VENV="${QP_VENV:-/data/quantum-protein/.venv}"
if [ "${1:-}" = "--venv" ]; then
    [ "$#" -ge 2 ] || fail "--venv requires a path"
    SHARED_VENV="$2"
    shift 2
fi

venv_usable() {
    [ -f "$1/bin/activate" ] || [ -f "$1/Scripts/activate" ]
}

if [ -L "${REPO_ROOT}/.venv" ] && ! venv_usable "${REPO_ROOT}/.venv"; then
    log "Removing stale/broken environment symlink: ${REPO_ROOT}/.venv"
    rm -f "${REPO_ROOT}/.venv"
fi

if venv_usable "${REPO_ROOT}/.venv"; then
    log "Environment available: ${REPO_ROOT}/.venv -> $(readlink -f "${REPO_ROOT}/.venv" 2>/dev/null || printf '%s' "${REPO_ROOT}/.venv")"
elif [ -e "${REPO_ROOT}/.venv" ]; then
    fail "${REPO_ROOT}/.venv exists but is not a usable virtual environment; move/remove it or set QP_VENV."
elif venv_usable "${SHARED_VENV}"; then
    ln -s "${SHARED_VENV}" "${REPO_ROOT}/.venv"
    log "Linked this deployment to the existing shared environment: ${SHARED_VENV}"
else
    fail "No virtual environment found at ${SHARED_VENV}. Set QP_VENV, use --venv /path/to/existing/.venv, or create/restore that environment first."
fi

log "Handing off to run_full_experiment.sh (formal pipeline uses the stage toggles frozen in configs/full_experiment_config.yaml; current source config includes smoke_check before data_audit)."
exec bash "${SCRIPT_DIR}/run_full_experiment.sh" "$@"
