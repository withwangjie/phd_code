#!/usr/bin/env bash
# Re-check external validation (independence audit + FASPR/Phenix baselines)
# against an existing run's frozen outputs with the CURRENT code.
#
#   ./scripts/check_external_validation.sh experiments_full_run_20261009_073701
#
# Diagnostic only: writes to <run>/diagnostics/external_validation_check/ and
# never changes stage status. A formal result needs a fresh run, because
# resume refuses a run whose recorded code fingerprint differs.
set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
[ -f .venv/bin/activate ] && source .venv/bin/activate

[ $# -ge 1 ] || { echo "usage: $0 <run directory or name>" >&2; exit 2; }
RUN="$1"; [ -d "$RUN" ] || RUN="runs/$1"
[ -f "$RUN/run_manifest.json" ] || { echo "not a run directory: $1" >&2; exit 2; }
CFG="$RUN/provenance/resolved_runtime_config.yaml"
[ -f "$CFG" ] || CFG="configs/full_experiment_config.yaml"
OUT="$RUN/diagnostics/external_validation_check"
mkdir -p "$OUT"

python - "$CFG" "$RUN" "$OUT" <<'PY'
import json, subprocess, sys
from pathlib import Path
import yaml

cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
run, out = Path(sys.argv[2]), Path(sys.argv[3])
iso = cfg["queue_freeze"]["homology_isolation"]
ext = cfg.get("external_validation", {}) or {}
vhh = ext.get("external_vhh", {}) or {}
base = ext.get("structural_baselines", {}) or {}
py = sys.executable
status = 0

if vhh.get("graph_dir"):
    root = Path(cfg["paths"]["repo_root"])
    graphs, sources, ledger = root/vhh["graph_dir"], root/vhh["source_structure_dir"], []
else:
    graphs, sources = run/"dataset/graphs/holdout", run/"dataset/holdout_source_structures"
    ledger = ["--audit-details", str(run/"audit/data_audit_details.jsonl")]
manifest = out/"external_vhh_independence_manifest.json"
argv = [py, "-m", "nanoqc.data.audit_external_vhh_independence",
        "--training-dataset", str(run/"dataset"), "--external-graph-dir", str(graphs),
        "--external-source-dir", str(sources),
        "--cluster-map", str(run/"independence/pdb_family_clusters.json"),
        "--out", str(manifest),
        "--vhh-threshold", str(iso.get("vhh_full_chain_identity", 0.80)),
        "--cdr-h3-threshold", str(iso.get("cdr_h3_identity", 0.50)),
        "--antigen-threshold", str(iso.get("antigen_identity", 0.30)),
        "--antigen-min-length-coverage", str(iso.get("antigen_min_length_coverage", 0.70)),
        "--workers", str((cfg.get("hardware", {}) or {}).get("external_audit_workers", 1))] + ledger
with open(out/"audit.log", "w") as log:
    rc = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT)
print(f"[independence] exit={rc}")
if manifest.is_file():
    m = json.loads(manifest.read_text())
    print(f"  fatal_error={m.get('fatal_error')}  targets={m.get('target_count')}  "
          f"failed={m.get('failed_targets')}  binding={m.get('cdr_h3_binding')}")
    for pdb, err in sorted((m.get("errored_targets") or {}).items())[:10]:
        print(f"  {pdb}: {err}")
    for row in m.get("targets", []):
        if not row.get("passes") and not row.get("error"):
            print(f"  {row['pdb_id']}: vhh={row['max_vhh_full_chain_identity']:.3f} "
                  f"cdr={row['max_cdr_h3_loop_identity']:.3f} ag={row['max_antigen_full_chain_identity']:.3f} "
                  f"cov={row['antigen_length_coverage']:.2f} cluster_overlap={row['family_cluster_overlap']}")
else:
    print(f"  no manifest; see {out/'audit.log'}")
status |= rc != 0

if base.get("required", False):
    seeds = [str(s) for s in cfg["structure_experiment"].get("seeds", [42, 43, 44, 45, 46])]
    argv = [py, "-m", "nanoqc.experiments.run_external_structure_baselines",
            "--validation-dir", str(run/"validation_queue"),
            "--faspr", str(base["faspr_executable"]),
            "--phenix-clashscore", str(base["phenix_clashscore_executable"]),
            "--out-dir", str(out/"structural_baselines"),
            "--timeout-seconds", str(base.get("timeout_seconds", 1800)),
            "--expected-seeds", *seeds]
    with open(out/"baselines.log", "w") as log:
        rc = subprocess.call(argv, stdout=log, stderr=subprocess.STDOUT)
    summary = out/"structural_baselines/run_summary.json"
    print(f"[structural_baselines] exit={rc}")
    if summary.is_file():
        s = json.loads(summary.read_text())
        print(f"  rows={s['rows']} targets={s['targets']} failures={len(s['failures'])}")
        for f in s["failures"][:10]:
            print(f"  {f['target']}/seed_{f['seed']}: {str(f['error']).splitlines()[0][:200]}")
    else:
        print(f"  no summary; see {out/'baselines.log'}")
    status |= rc != 0
sys.exit(status)
PY
