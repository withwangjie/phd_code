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
PREFER_SHARED=0
[ -n "${QP_VENV:-}" ] && PREFER_SHARED=1
VENV_FROM_CLI=0
if [ "${1:-}" = "--venv" ]; then
    [ "$#" -ge 2 ] || fail "--venv requires a path"
    SHARED_VENV="$2"
    PREFER_SHARED=1
    VENV_FROM_CLI=1
    shift 2
fi

if [ "$VENV_FROM_CLI" -eq 0 ] && [ -z "${QP_VENV:-}" ]; then
    SERVER_CONFIG_FILE="${QP_SERVER_CONFIG:-${REPO_ROOT}/configs/server_config.yaml}"
    if [[ "$SERVER_CONFIG_FILE" != /* ]]; then
        SERVER_CONFIG_FILE="${REPO_ROOT}/${SERVER_CONFIG_FILE}"
    fi
    if [ -f "$SERVER_CONFIG_FILE" ]; then
        SERVER_VENV_HINT="$(awk -F: '$1 ~ /^[[:space:]]*venv[[:space:]]*$/ {
            v=substr($0,index($0,":")+1); sub(/#.*/,"",v);
            gsub(/^[[:space:]]+|[[:space:]]+$/,"",v);
            gsub(/^"|"$/,"",v); gsub(/^\047|\047$/,"",v);
            if (v!="" && v!="auto") print v; exit
        }' "$SERVER_CONFIG_FILE")"
        if [ -n "$SERVER_VENV_HINT" ]; then
            case "$SERVER_VENV_HINT" in
                "~/"*) SHARED_VENV="${HOME}/${SERVER_VENV_HINT#~/}" ;;
                /*) SHARED_VENV="$SERVER_VENV_HINT" ;;
                *) SHARED_VENV="${REPO_ROOT}/${SERVER_VENV_HINT}" ;;
            esac
            PREFER_SHARED=1
        fi
    fi
fi

venv_usable() {
    { [ -f "$1/bin/activate" ] && [ -x "$1/bin/python" ]; } ||
    { [ -f "$1/Scripts/activate" ] && [ -f "$1/Scripts/python.exe" ]; }
}

if [ -L "${REPO_ROOT}/.venv" ] && ! venv_usable "${REPO_ROOT}/.venv"; then
    log "Removing stale/broken environment symlink: ${REPO_ROOT}/.venv"
    rm -f "${REPO_ROOT}/.venv"
fi

if [ "$PREFER_SHARED" -eq 1 ]; then
    venv_usable "${SHARED_VENV}" || fail "Selected virtual environment is unusable: ${SHARED_VENV}"
    ACTIVE_VENV="$(readlink -f "${SHARED_VENV}" 2>/dev/null || printf '%s' "${SHARED_VENV}")"
    if [ -L "${REPO_ROOT}/.venv" ]; then
        CURRENT_LINK="$(readlink -f "${REPO_ROOT}/.venv" 2>/dev/null || true)"
        if [ "$CURRENT_LINK" != "$ACTIVE_VENV" ]; then
            rm -f "${REPO_ROOT}/.venv"
            ln -s "$ACTIVE_VENV" "${REPO_ROOT}/.venv"
            log "Updated repository environment link -> ${ACTIVE_VENV}"
        fi
    elif [ ! -e "${REPO_ROOT}/.venv" ]; then
        ln -s "$ACTIVE_VENV" "${REPO_ROOT}/.venv"
        log "Linked repository environment -> ${ACTIVE_VENV}"
    else
        log "Explicit environment selected: ${ACTIVE_VENV}; existing real ${REPO_ROOT}/.venv left untouched."
    fi
elif venv_usable "${REPO_ROOT}/.venv"; then
    ACTIVE_VENV="$(readlink -f "${REPO_ROOT}/.venv" 2>/dev/null || printf '%s' "${REPO_ROOT}/.venv")"
    log "Environment available: ${REPO_ROOT}/.venv -> ${ACTIVE_VENV}"
elif [ -e "${REPO_ROOT}/.venv" ]; then
    fail "${REPO_ROOT}/.venv exists but is not a usable virtual environment; move/remove it or set QP_VENV."
elif venv_usable "${SHARED_VENV}"; then
    ln -s "${SHARED_VENV}" "${REPO_ROOT}/.venv"
    ACTIVE_VENV="$(readlink -f "${REPO_ROOT}/.venv" 2>/dev/null || printf '%s' "${SHARED_VENV}")"
    log "Linked this deployment to the default shared environment: ${ACTIVE_VENV}"
else
    fail "No virtual environment found at ${SHARED_VENV}. Set QP_VENV, use --venv /path/to/existing/.venv, or create/restore that environment first."
fi

# Export the environment that was actually selected. QP_VENV has highest
# precedence in the resolver and downstream shell entrypoints.
export QP_VENV="${ACTIVE_VENV}"
log "Handing off to run_full_experiment.sh (formal pipeline uses the stage toggles frozen in configs/full_experiment_config.yaml; current source config includes smoke_check before data_audit)."
exec bash "${SCRIPT_DIR}/run_full_experiment.sh" "$@"
