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
# acceptance live in src/nanoqc/pipeline/run_full_experiment.py; this script only activates the
# environment, launches it in the background so this command returns
# immediately, and reports where to watch progress -- the same pattern
# the earlier benchmark-only launcher used.
#
# Usage:
#   ./scripts/run_full_experiment.sh
#   ./scripts/run_full_experiment.sh --resume experiments_full_run_20260920_010203
#   ./scripts/run_full_experiment.sh --only qc_benchmark
#   ./scripts/run_full_experiment.sh --smoke-only
#
# Supported run controls are forwarded to run_full_experiment.py. The launcher
# owns --config and --run-dir because they must match the preflight/provenance.
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root: holds configs/, docs/, src/, .venv and .runtime/.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

log() { echo "[run_full_experiment] $*"; }
fail() { echo "[run_full_experiment] ERROR: $*" >&2; exit 1; }

SCIENTIFIC_CONFIG="configs/full_experiment_config.yaml"
SERVER_CONFIG="${QP_SERVER_CONFIG:-configs/server_config.yaml}"
if [[ "$SERVER_CONFIG" != /* ]]; then
    SERVER_CONFIG="${REPO_ROOT}/${SERVER_CONFIG}"
fi
[ -f "$SCIENTIFIC_CONFIG" ] || fail "Scientific config file not found: ${REPO_ROOT}/${SCIENTIFIC_CONFIG}"
[ -f "$SERVER_CONFIG" ] || fail "Server config file not found: ${SERVER_CONFIG}"
for ARG in "$@"; do
    case "$ARG" in
        --config|--config=*|--run-dir|--run-dir=*)
            fail "$ARG is launcher-owned; use the resolved config and selected run directory"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# 1. Activate the local virtual environment (POSIX or Windows layout).
# ---------------------------------------------------------------------------
if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
elif [ -f "${REPO_ROOT}/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/Scripts/activate"
else
    fail "No virtual environment found at .venv/bin/activate or .venv/Scripts/activate. Create one first (python -m venv .venv) and install this project's dependencies (requirements.txt) before launching."
fi
log "Activated virtual environment: $(command -v python)"
python --version

# ---------------------------------------------------------------------------
# 1b. Resolve infrastructure/runtime settings for THIS server.
# Scientific protocol values remain unchanged. DDP ranks/batch split are
# resolved here and rank count affects the trained EGNN checkpoint; the value
# is recorded in the resolved config and provenance.
# ---------------------------------------------------------------------------
RUNTIME_DIR="${REPO_ROOT}/.runtime"
mkdir -p "$RUNTIME_DIR"
RESOLVED_CONFIG="${RUNTIME_DIR}/resolved_runtime_config.yaml"
SERVER_REPORT="${RUNTIME_DIR}/server_resolution.json"
log "Resolving server paths/resources from ${SERVER_CONFIG}..."
python -m nanoqc.pipeline.resolve_server_config \
    --scientific-config "${REPO_ROOT}/${SCIENTIFIC_CONFIG}" \
    --server-config "$SERVER_CONFIG" \
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
    [ -f "${REPO_ROOT}/docs/METHODS_EVIDENCE.md" ] && cp "${REPO_ROOT}/docs/METHODS_EVIDENCE.md" "$RUN_DIR/provenance/METHODS_EVIDENCE.md"
    [ -f "${REPO_ROOT}/docs/RESULTS_CONTRACT.md" ] && cp "${REPO_ROOT}/docs/RESULTS_CONTRACT.md" "$RUN_DIR/provenance/RESULTS_CONTRACT.md"
    [ -f "${REPO_ROOT}/docs/PROTOCOL_AMENDMENTS.md" ] && cp "${REPO_ROOT}/docs/PROTOCOL_AMENDMENTS.md" "$RUN_DIR/provenance/PROTOCOL_AMENDMENTS.md"
    git rev-parse HEAD > "$RUN_DIR/provenance/git_head.txt" 2>/dev/null || true
    python -m pip freeze > "$RUN_DIR/provenance/pip_freeze.txt" 2>/dev/null || true
    {
        echo "PATH=${PATH:-}"
        echo "PYTHONPATH=${PYTHONPATH:-}"
        echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
        echo "OMP_NUM_THREADS=${OMP_NUM_THREADS:-}"
        echo "MKL_NUM_THREADS=${MKL_NUM_THREADS:-}"
        echo "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-}"
        echo "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-}"
        echo "QP_VENV=${QP_VENV:-}"
        echo "QP_DATA_ROOT=${QP_DATA_ROOT:-}"
        echo "QP_RUN_ROOT=${QP_RUN_ROOT:-}"
        echo "QP_FASPR=${QP_FASPR:-}"
        echo "QP_PHENIX_CLASHSCORE=${QP_PHENIX_CLASHSCORE:-}"
    } > "$RUN_DIR/provenance/environment.txt"
fi

# ---------------------------------------------------------------------------
# 2. Fast fail on a second concurrent launch, BEFORE even importing Python --
#    run_full_experiment.py holds its own filelock for the run_root for the
#    full duration too; this is a cheap early check so a duplicate launch
#    does not even pay the interpreter-startup cost.
# ---------------------------------------------------------------------------
LOCK_FILE="${REPO_ROOT}/.run_full_experiment.lock"
command -v flock >/dev/null || fail "Linux flock is required for atomic launch locking."
exec 9>"${REPO_ROOT}/.launch.guard"
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

printf '%s\n' "$BASHPID" > "$LOCK_FILE"
if [ "$FRESH_RUN" -eq 1 ]; then
    nohup python -u -m nanoqc.pipeline.run_full_experiment --config "$RESOLVED_CONFIG" \
        --run-dir "$RUN_DIR" "$@" > "$LAUNCH_LOG" 2>&1 &
else
    nohup python -u -m nanoqc.pipeline.run_full_experiment --config "$RESOLVED_CONFIG" \
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
log "Scientific config: ${REPO_ROOT}/${SCIENTIFIC_CONFIG}"
log "Server config:     ${SERVER_CONFIG}"
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
log "  ./scripts/run_full_experiment.sh --resume <experiments_full_run_directory_name>"
