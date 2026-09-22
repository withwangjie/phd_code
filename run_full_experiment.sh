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

CONFIG_FILE="full_experiment_config.yaml"
[ -f "$CONFIG_FILE" ] || fail "Config file not found: ${SCRIPT_DIR}/${CONFIG_FILE}"

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
# This runs synchronously and MUST pass before any formal background process
# is created. It covers regression tests, two-GPU CUDA execution, OpenMM CUDA
# double-precision context creation, writable run_root, disk and RAM checks.
PREFLIGHT_SCRIPT="${SCRIPT_DIR}/formal_preflight.sh"
[ -f "$PREFLIGHT_SCRIPT" ] || fail "Formal preflight script missing: $PREFLIGHT_SCRIPT"
log "Running formal preflight gate..."
bash "$PREFLIGHT_SCRIPT"
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
LAUNCH_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LAUNCH_LOG="${SCRIPT_DIR}/run_full_experiment_launch_${LAUNCH_TIMESTAMP}.log"

echo $$ > "$LOCK_FILE"
nohup python -u "${SCRIPT_DIR}/run_full_experiment.py" --config "$CONFIG_FILE" "$@" \
    > "$LAUNCH_LOG" 2>&1 &
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
log "Lock file:         ${LOCK_FILE}"
log "Launch log:        ${LAUNCH_LOG}"
log "Config used:       ${SCRIPT_DIR}/${CONFIG_FILE}"
log ""
log "run_full_experiment.py creates its own uniquely timestamped run"
log "directory (experiments_full_run_<UTC timestamp>/) and prints it near"
log "the top of the launch log; each stage's own stdout/stderr additionally"
log "goes to <run_dir>/logs/<stage>.log, and <run_dir>/progress.json is"
log "updated after every stage."
log ""
log "Tail overall progress with:   tail -f '${LAUNCH_LOG}'"
log "Check process status with:    kill -0 ${PIPELINE_PID} 2>/dev/null && echo running || echo finished"
log "Resume an interrupted run with:"
log "  ./run_full_experiment.sh --resume <experiments_full_run_directory_name>"
