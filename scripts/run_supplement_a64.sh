#!/usr/bin/env bash
# A64 supplementary experiments on an existing formal run (exploratory).
#
#   ./scripts/run_supplement_a64.sh experiments_full_run_20261011_075258 [--workers N]
#
# Order: (1) force-field reproduction control on one target -- the frozen
# model run through the A64 driver must reproduce the formal rows, otherwise
# nothing else is run; (2) noise sensitivity and (3) budget-optimised classical
# baselines on CPU; (4) the CHARMM36 and unshielded-ff14SB arms on the GPUs.
# Writes only under <run>/analysis/a64/; never changes stage status.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
[ -f .venv/bin/activate ] && source .venv/bin/activate

[ $# -ge 1 ] || { echo "usage: $0 <run directory or name> [--workers N]" >&2; exit 2; }
RUN="$1"; shift; [ -d "$RUN" ] || RUN="runs/$RUN"
[ -f "$RUN/run_manifest.json" ] || { echo "not a run directory: $RUN" >&2; exit 2; }
WORKERS="$(python -c 'import os; print(max(1, (os.cpu_count() or 2) - 2))')"
while [ $# -gt 0 ]; do case "$1" in --workers) WORKERS="$2"; shift 2;; *) echo "unknown option $1" >&2; exit 2;; esac; done
OUT="$RUN/analysis/a64"; mkdir -p "$OUT"
log() { echo "[a64 $(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
log "code $(git rev-parse --short HEAD); run $RUN; CPU workers $WORKERS"

CONTROL_TARGET="$(ls "$RUN/validation_queue/prepared" | head -1)"
log "1/4 reproduction control: frozen amber14_a50 on $CONTROL_TARGET seed 42"
python -m nanoqc.supplement.forcefield_sensitivity --run-dir "$RUN" --arms amber14_a50 \
    --targets "$CONTROL_TARGET" --seeds 42 2>&1 | tee -a "$OUT/run.log"
python - "$OUT/forcefield/forcefield_paired.csv" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["arm"] == "amber14_a50"]
diffs = [abs(float(r["arm_final_rmsd"]) - float(r["formal_final_rmsd"])) for r in rows
         if r["arm_final_rmsd"] and r["formal_final_rmsd"]]
if len(diffs) != len(rows) or not rows:
    sys.exit(f"control: {len(diffs)}/{len(rows)} rows comparable; refusing to run the arms. "
             "Inspect analysis/a64/forcefield/amber14_a50/results/<target>/ and the "
             "'[forcefield] ... FAILED' / Traceback lines above in run.log")
worst = max(diffs)
print(f"control: {len(rows)} rows, max |arm - formal| final RMSD = {worst:.2e} A")
if worst > 1e-3:
    sys.exit("control does not reproduce the formal rows; refusing to run the arms")
PY
mv "$OUT/forcefield/forcefield_summary.md" "$OUT/forcefield/control_summary.md"

log "2/4 noise sensitivity"
python -m nanoqc.supplement.noise_simulation --run-dir "$RUN" --workers "$WORKERS" 2>&1 | tee -a "$OUT/run.log"
log "3/4 budget-optimised classical baselines"
python -m nanoqc.supplement.classical_baselines --run-dir "$RUN" --workers "$WORKERS" 2>&1 | tee -a "$OUT/run.log"
log "4/4 force-field arms (GPU)"
python -m nanoqc.supplement.forcefield_sensitivity --run-dir "$RUN" --arms charmm36 amber14_unshielded \
    2>&1 | tee -a "$OUT/run.log"
log "done: $OUT/{noise,classical,forcefield}/*_summary.md"
