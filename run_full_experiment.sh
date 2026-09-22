#!/usr/bin/env bash
#
# run_full_experiment.sh -- single unattended entry point for the full
# nanobody-interface quantum/classical benchmark research pipeline:
#
#   env check -> smoke check -> data audit -> queue freeze (uncapped split +
#   frozen validation-target queue) -> EGNN training -> quantum/classical
#   ablation benchmark -> real-atom structural experiment (dev + validation
#   queues) -> paired statistics -> FINAL_RESEARCH_REPORT.md
#
# All stage sequencing, resumability, provenance and per-stage artifact
# acceptance live in run_full_experiment.py; this script only activates the
# environment, launches it in the background so this command returns
# immediately, and reports where to watch progress -- the same pattern
# launch_server.sh already uses for the narrower benchmark-only pipeline.
#
# Usage:
#   ./run_full_experiment.sh
#   ./run_full_experiment.sh --resume experiments_full_run_20260920_010203
#   ./run_full_experiment.sh --only qc_benchmark
#   ./run_full_experiment.sh --smoke-only
#
# Every argument after the script name is forwarded verbatim to
# run_full_experiment.py (see its --help for the full flag list).
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[run_full_experiment] $*"; }
fail() { echo "[run_full_experiment] ERROR: $*" >&2; exit 1; }

SCIENTIFIC_CONFIG="full_experiment_config.yaml"
SERVER_CONFIG="${QP_SERVER_CONFIG:-server_config.yaml}"
[ -f "$SCIENTIFIC_CONFIG" ] || fail "Scientific config file not found: ${SCRIPT_DIR}/${SCIENTIFIC_CONFIG}"
[ -f "$SERVER_CONFIG" ] || fail "Server config file not found: ${SCRIPT_DIR}/${SERVER_CONFIG}"

# ---------------------------------------------------------------------------
# 1. Activate the local virtual environment (POSIX or Windows layout).
# ---------------------------------------------------------------------------
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/Scripts/activate"
else
    fail "No virtual environment found at .venv/bin/activate or .venv/Scripts/activate. Create one first (python -m venv .venv) and install this project's dependencies (requirements-quantum.txt, requirements-allatom.txt, plus pyyaml) before launching."
fi
log "Activated virtual environment: $(command -v python)"
python --version

# ---------------------------------------------------------------------------
# 1b. Resolve infrastructure/runtime settings for THIS server.
# Scientific protocol values remain unchanged; only paths, worker counts,
# DDP ranks/batch split, OpenMM platform/device and external executable paths
# are resolved here.
# ---------------------------------------------------------------------------
RUNTIME_DIR="${SCRIPT_DIR}/.runtime"
mkdir -p "$RUNTIME_DIR"
RESOLVED_CONFIG="${RUNTIME_DIR}/resolved_runtime_config.yaml"
SERVER_REPORT="${RUNTIME_DIR}/server_resolution.json"
log "Resolving server paths/resources from ${SERVER_CONFIG}..."
python "${SCRIPT_DIR}/resolve_server_config.py" \
    --scientific-config "${SCRIPT_DIR}/${SCIENTIFIC_CONFIG}" \
    --server-config "${SCRIPT_DIR}/${SERVER_CONFIG}" \
    --out-config "$RESOLVED_CONFIG" \
    --out-report "$SERVER_REPORT"
log "Resolved runtime config: $RESOLVED_CONFIG"

# ---------------------------------------------------------------------------
# 1c. Resolve ONE run directory for all artifacts, including preflight and
# launch logs. Fresh runs are created here before preflight; resumed runs
# reuse their original directory.
# ---------------------------------------------------------------------------
RUN_ROOT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_root"])' "$SERVER_REPORT")"
RUN_PREFIX="$(python -c 'import yaml,sys; print((yaml.safe_load(open(sys.argv[1]))["paths"]).get("run_prefix","experiments_full_run_"))' "$RESOLVED_CONFIG")"
mkdir -p "$RUN_ROOT"

RESUME_TARGET=""
PREV_ARG=""
for ARG in "$@"; do
    if [ "$PREV_ARG" = "--resume" ]; then
        RESUME_TARGET="$ARG"
        break
    fi
    case "$ARG" in
        --resume=*) RESUME_TARGET="${ARG#--resume=}"; break ;;
    esac
    PREV_ARG="$ARG"
done

if [ -n "$RESUME_TARGET" ]; then
    if [[ "$RESUME_TARGET" = /* ]]; then
        RUN_DIR="$RESUME_TARGET"
    else
        RUN_DIR="${RUN_ROOT}/${RESUME_TARGET}"
    fi
    [ -d "$RUN_DIR" ] || fail "Resume run directory not found: $RUN_DIR"
    FRESH_RUN=0
else
    RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
    RUN_DIR="${RUN_ROOT}/${RUN_PREFIX}${RUN_STAMP}"
    SUFFIX=0
    while [ -e "$RUN_DIR" ]; do
        SUFFIX=$((SUFFIX+1))
        RUN_DIR="${RUN_ROOT}/${RUN_PREFIX}${RUN_STAMP}_${SUFFIX}"
    done
    mkdir -p "$RUN_DIR"
    FRESH_RUN=1
fi

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/provenance"
if [ "$FRESH_RUN" -eq 1 ]; then
    cp "$SCIENTIFIC_CONFIG" "$RUN_DIR/provenance/scientific_config.source.yaml"
    cp "$SERVER_CONFIG" "$RUN_DIR/provenance/server_config.source.yaml"
    cp "$RESOLVED_CONFIG" "$RUN_DIR/provenance/resolved_runtime_config.yaml"
    cp "$SERVER_REPORT" "$RUN_DIR/provenance/server_resolution.json"
    [ -f "${SCRIPT_DIR}/METHODS_EVIDENCE.md" ] && cp "${SCRIPT_DIR}/METHODS_EVIDENCE.md" "$RUN_DIR/provenance/METHODS_EVIDENCE.md"
    git rev-parse HEAD > "$RUN_DIR/provenance/git_head.txt" 2>/dev/null || true
    python -m pip freeze > "$RUN_DIR/provenance/pip_freeze.txt" 2>/dev/null || true
    env | sort > "$RUN_DIR/provenance/environment.txt"
fi

# ---------------------------------------------------------------------------
# 2. Fast fail on a second concurrent launch, BEFORE even importing Python --
#    run_full_experiment.py holds its own filelock for the run_root for the
#    full duration too; this is a cheap early check so a duplicate launch
#    does not even pay the interpreter-startup cost.
# ---------------------------------------------------------------------------
LOCK_FILE="${SCRIPT_DIR}/.run_full_experiment.lock"
command -v flock >/dev/null || fail "Linux flock is required for atomic launch locking."
exec 9>"${SCRIPT_DIR}/.launch.guard"
flock -n 9 || fail "A pipeline launch or process already owns this deployment."
if [ -f "$LOCK_FILE" ]; then
    HELD_PID="$(cat "$LOCK_FILE" 2>/dev/null || true)"
    if [ -n "$HELD_PID" ] && kill -0 "$HELD_PID" 2>/dev/null; then
        fail "run_full_experiment.sh already appears to be running (PID ${HELD_PID}, lock: ${LOCK_FILE}). Refusing to start a second instance. If that process is gone, remove the lock file manually."
    else
        log "Stale lock file found (PID ${HELD_PID:-unknown} not running); removing it."
        rm -f "$LOCK_FILE"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Formal preflight gate.
# ---------------------------------------------------------------------------
# Runs synchronously before any background process is created. Any regression,
# CUDA/DDP/NCCL/OpenMM/resource failure aborts the formal launch.
PREFLIGHT_SCRIPT="${SCRIPT_DIR}/formal_preflight.sh"
[ -f "$PREFLIGHT_SCRIPT" ] || fail "Formal preflight script missing: $PREFLIGHT_SCRIPT"
SESSION_STAMP="$(date -u +%Y%m%d_%H%M%S)"
PREFLIGHT_LOG="$RUN_DIR/logs/formal_preflight_${SESSION_STAMP}.log"
log "Running formal preflight gate..."
if ! QP_RESOLVED_CONFIG="$RESOLVED_CONFIG" QP_SERVER_REPORT="$SERVER_REPORT" \
    bash "$PREFLIGHT_SCRIPT" 2>&1 | tee "$PREFLIGHT_LOG"; then
    echo "failed" > "$RUN_DIR/PREFLIGHT_STATUS"
    fail "Formal preflight failed. Full log: $PREFLIGHT_LOG"
fi
echo "passed" > "$RUN_DIR/PREFLIGHT_STATUS"
log "Formal preflight gate passed."

# ---------------------------------------------------------------------------
# 4. Formal execution only; no trial/smoke stage is injected here.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 5. Launch run_full_experiment.py in the background under nohup, so this
#    command returns immediately even though the full pipeline can run for
#    many hours to days. Resolve one shared timestamped log file up front so
#    it can be reported below regardless of which/how many run directories
#    the Python orchestrator itself creates or resumes.
# ---------------------------------------------------------------------------
LAUNCH_LOG="$RUN_DIR/logs/launch_${SESSION_STAMP}.log"

echo $ > "$LOCK_FILE"
if [ "$FRESH_RUN" -eq 1 ]; then
    nohup python -u "${SCRIPT_DIR}/run_full_experiment.py" --config "$RESOLVED_CONFIG" \
        --run-dir "$RUN_DIR" "$@" > "$LAUNCH_LOG" 2>&1 &
else
    nohup python -u "${SCRIPT_DIR}/run_full_experiment.py" --config "$RESOLVED_CONFIG" \
        "$@" > "$LAUNCH_LOG" 2>&1 &
fi
PIPELINE_PID=$!
disown "$PIPELINE_PID" 2>/dev/null || true
# Reassign the lock to the actual background process (not this shell), and
# clean it up if that process later exits normally.
echo "$PIPELINE_PID" > "$LOCK_FILE"
(
    while kill -0 "$PIPELINE_PID" 2>/dev/null; do sleep 5; done
    rm -f "$LOCK_FILE"
) >/dev/null 2>&1 &
disown $! 2>/dev/null || true

log ""
log "=== Full experiment pipeline launched ==="
log "PID:              ${PIPELINE_PID}"
log "Run directory:     ${RUN_DIR}"
log "Lock file:         ${LOCK_FILE}"
log "Preflight log:     ${PREFLIGHT_LOG}"
log "Launch log:        ${LAUNCH_LOG}"
log "Scientific config: ${SCRIPT_DIR}/${SCIENTIFIC_CONFIG}"
log "Server config:     ${SCRIPT_DIR}/${SERVER_CONFIG}"
log "Resolved config:   ${RESOLVED_CONFIG}"
log "Server report:     ${SERVER_REPORT}"
log ""
log "This run directory is the complete experiment archive: provenance,"
log "preflight, launch/stage logs, dataset, checkpoints, benchmark outputs,"
log "structural results, statistics and final report all live under it."
log ""
log "Tail overall progress with:   tail -f '${LAUNCH_LOG}'"
log "Check process status with:    kill -0 ${PIPELINE_PID} 2>/dev/null && echo running || echo finished"
log "Resume an interrupted run with:"
log "  ./run_full_experiment.sh --resume <experiments_full_run_directory_name>"
