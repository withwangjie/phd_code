#!/usr/bin/env bash
#
# launch_server.sh -- production entry point for the unattended,
# server-side quantum-classical rotamer-recovery benchmark:
#
#   "Unraveling Variational Collapse in Constrained Biomolecular
#    Optimization: A Physics-Preserving Quantum-Classical Benchmark and
#    Boundary Analysis."
#
# What this script does, in order:
#   1. Activates this project's local virtual environment (.venv).
#   2. Runs the preflight diagnostic suite (test_audit_remediation.py,
#      test_robust_qaoa.py, test_evaluate_complex_metrics.py). A preflight
#      failure aborts BEFORE any batch execution starts -- an unattended run
#      must never begin against code that fails its own regression suite.
#   3. Launches run_server_pipeline.py in the background via nohup, so this
#      script returns immediately rather than blocking for the run's full
#      (potentially many-hour) duration, with stdout/stderr captured to
#      pipeline_execution.log.
#   4. Chains generate_final_report.py to run automatically once the
#      pipeline process exits successfully, against that same run's output
#      directory.
#   5. Prints the background process PID, the log path, and every artifact
#      location, so the operator (or a monitoring script) knows exactly
#      what to watch.
#
# Usage:
#   ./launch_server.sh [run_server_pipeline.py arguments...]
#
# Examples:
#   ./launch_server.sh
#   ./launch_server.sh --targets 4s10 8yvo 9gcn --seeds 42 43 44
#   ./launch_server.sh --dry-run
#
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[launch_server] $*"; }
fail() { echo "[launch_server] ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 1. Activate the local virtual environment.
#
# Both POSIX (.venv/bin/activate) and Windows (.venv/Scripts/activate) venv
# layouts are supported without further configuration, since this project's
# OpenMM/gemmi toolchain (see run_manifest.json's recorded `openmm` version
# in prior runs) has historically been exercised from a Windows checkout.
# ---------------------------------------------------------------------------
if [ -f "${SCRIPT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/bin/activate"
elif [ -f "${SCRIPT_DIR}/.venv/Scripts/activate" ]; then
    # shellcheck disable=SC1091
    source "${SCRIPT_DIR}/.venv/Scripts/activate"
else
    fail "No virtual environment found at .venv/bin/activate or .venv/Scripts/activate. Create one first (python -m venv .venv) and install this project's dependencies before launching."
fi
log "Activated virtual environment: $(command -v python)"
python --version

# ---------------------------------------------------------------------------
# 2. Preflight diagnostic suite. Any failure here aborts the launch; an
#    unattended, multi-hour batch run must never start against code that
#    fails its own existing regression tests.
# ---------------------------------------------------------------------------
PREFLIGHT_TESTS=(
    "test_audit_remediation.py"
    "test_robust_qaoa.py"
    "test_evaluate_complex_metrics.py"
)

log "Byte-compiling the two new batch-pipeline entry points as a fast first check..."
python -m py_compile "${SCRIPT_DIR}/run_server_pipeline.py" "${SCRIPT_DIR}/generate_final_report.py" \
    || fail "run_server_pipeline.py / generate_final_report.py failed to byte-compile; aborting before any execution."

log "Running preflight diagnostics: ${PREFLIGHT_TESTS[*]}"
for test_file in "${PREFLIGHT_TESTS[@]}"; do
    test_path="${SCRIPT_DIR}/${test_file}"
    if [ ! -f "$test_path" ]; then
        fail "Preflight test file not found: ${test_path}"
    fi
    log "--- ${test_file} ---"
    # Probe that "python -m pytest" actually works in THIS interpreter (a
    # "pytest" executable can be present on PATH -- e.g. a stray entry
    # point from an unrelated environment -- while the pytest module itself
    # is not importable here); fall back to running the test file directly
    # only when the probe genuinely fails, rather than trusting mere
    # presence of the name on PATH.
    if python -m pytest --version >/dev/null 2>&1; then
        python -m pytest -q "$test_path" || fail "Preflight test failed: ${test_file}"
    else
        python "$test_path" || fail "Preflight test failed: ${test_file}"
    fi
done
log "All preflight diagnostics passed."

# ---------------------------------------------------------------------------
# 3. Resolve this run's isolated, UTC-timestamped output directory up front,
#    so it can be passed explicitly to both run_server_pipeline.py and the
#    chained generate_final_report.py call -- this guarantees the report is
#    generated against exactly the directory this launch produced, with no
#    dependency on parsing the pipeline's stdout to discover it.
# ---------------------------------------------------------------------------
RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${SCRIPT_DIR}/experiments_run_${RUN_TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

PIPELINE_LOG="${OUTPUT_DIR}/pipeline_execution.log"
PID_FILE="${OUTPUT_DIR}/pipeline.pid"
RUNNER_SCRIPT="${OUTPUT_DIR}/_run_and_report.sh"

# ---------------------------------------------------------------------------
# 4. Generate a small runner script that chains run_server_pipeline.py and
#    generate_final_report.py, then launch IT under nohup. Generating a real
#    script file (rather than an inline `nohup bash -c "..."`) avoids fragile
#    nested-quoting of $?, $@, and the output-directory path, and leaves a
#    durable, inspectable record of exactly what this launch executed.
# ---------------------------------------------------------------------------
cat > "$RUNNER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -eo pipefail
echo "[run_and_report] Starting run_server_pipeline.py at \$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python "${SCRIPT_DIR}/run_server_pipeline.py" --output-dir "${OUTPUT_DIR}" "\$@"
status=\$?
if [ \$status -eq 0 ]; then
    echo "[run_and_report] Pipeline finished (exit 0) at \$(date -u +%Y-%m-%dT%H:%M:%SZ); generating final report."
    python "${SCRIPT_DIR}/generate_final_report.py" --run-dir "${OUTPUT_DIR}"
    echo "[run_and_report] Final report written: ${OUTPUT_DIR}/FINAL_BENCHMARK_REPORT.md"
else
    echo "[run_and_report] Pipeline exited with status \$status; skipping report generation." >&2
fi
exit \$status
EOF
chmod +x "$RUNNER_SCRIPT"

nohup "$RUNNER_SCRIPT" "$@" > "$PIPELINE_LOG" 2>&1 &
PIPELINE_PID=$!
echo "$PIPELINE_PID" > "$PID_FILE"
disown "$PIPELINE_PID" 2>/dev/null || true

# ---------------------------------------------------------------------------
# 5. Report the launch: PID, tracking logs, artifact locations.
# ---------------------------------------------------------------------------
log ""
log "=== Pipeline launched ==="
log "PID:                          ${PIPELINE_PID}"
log "PID file:                     ${PID_FILE}"
log "Runner script:                ${RUNNER_SCRIPT}"
log "Execution log:                ${PIPELINE_LOG}"
log "Output directory:             ${OUTPUT_DIR}"
log "Run manifest (provenance):    ${OUTPUT_DIR}/run_manifest.json"
log "Failure ledger:               ${OUTPUT_DIR}/failed_cases.log"
log "Benchmark summary:            ${OUTPUT_DIR}/benchmark_summary.csv"
log "Trajectory summary:           ${OUTPUT_DIR}/trajectory_summary.csv"
log "Final report (on completion): ${OUTPUT_DIR}/FINAL_BENCHMARK_REPORT.md"
log ""
log "Tail progress with:   tail -f '${PIPELINE_LOG}'"
log "Check process status: kill -0 ${PIPELINE_PID} 2>/dev/null && echo running || echo finished"
