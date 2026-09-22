#!/usr/bin/env bash
#
# check_status.sh -- read-only status query for the formal pipeline launched
# by deploy_launch.sh / run_full_experiment.sh. Never starts, stops, or
# modifies anything.
#
# Usage:
#   ./check_status.sh                    # most recently created run under run_root
#   ./check_status.sh <run_dir_name>      # a specific run, by directory name
#   ./check_status.sh /abs/path/to/run    # a specific run, by absolute path
#
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/full_experiment_config.yaml"

RUN_ROOT="$(sed -n '/^paths:/,/^[a-zA-Z_][a-zA-Z_]*:/{/run_root:/p}' "$CONFIG" | head -1 \
    | sed -E 's/.*run_root:[[:space:]]*"?([^"#]*)"?.*/\1/')"
[ -n "$RUN_ROOT" ] || RUN_ROOT="/data/quantum-protein/runs"

if [ -n "${1:-}" ]; then
    if [[ "$1" = /* ]]; then RUN_DIR="$1"; else RUN_DIR="${RUN_ROOT}/$1"; fi
else
    RUN_DIR="$(ls -dt "${RUN_ROOT}"/experiments_full_run_* 2>/dev/null | head -1 || true)"
fi

if [ -z "${RUN_DIR:-}" ] || [ ! -d "$RUN_DIR" ]; then
    echo "No run directory found under ${RUN_ROOT} (pass a run directory name or path explicitly)." >&2
    exit 1
fi

echo "=== Run directory: ${RUN_DIR} ==="
if [ -f "${RUN_DIR}/progress.json" ]; then
    echo "--- progress.json (per-stage status) ---"
    cat "${RUN_DIR}/progress.json"
    echo ""
else
    echo "(no progress.json yet -- the orchestrator has not written status for this run)"
fi

LOCK="${SCRIPT_DIR}/.run_full_experiment.lock"
if [ -f "$LOCK" ]; then
    PID="$(cat "$LOCK" 2>/dev/null || true)"
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "Orchestrator process: RUNNING (PID ${PID})"
    else
        echo "Orchestrator process: NOT RUNNING (lock file is stale -- process ${PID:-unknown} is gone)"
    fi
else
    echo "Orchestrator process: no lock file present (not currently running via deploy_launch.sh/run_full_experiment.sh)"
fi

echo ""
echo "Resume this run with:"
echo "  cd '${SCRIPT_DIR}' && ./run_full_experiment.sh --resume '$(basename "$RUN_DIR")'"
echo "Re-run only one stage (its prerequisites must already show completed/skipped above):"
echo "  cd '${SCRIPT_DIR}' && ./run_full_experiment.sh --resume '$(basename "$RUN_DIR")' --only <stage_name>"
