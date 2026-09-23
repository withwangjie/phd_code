# Formal experiment results contract

This document defines the minimum auditable output set for a completed formal run.
A stage is not accepted as completed merely because its subprocess returned zero.
The orchestrator verifies the required result set, writes a per-stage manifest under
`results_manifests/`, and performs a run-level audit at shutdown.

## Stage contracts

### env_check
Required:
- `env_check.json`

### data_audit
Required:
- `audit/data_audit_report.md`
- `audit/data_audit_details.csv`
- `audit/data_audit_details.jsonl`
- `audit/data_audit_inventory.json`
- `audit/data_audit_db55_pairs.json`

### queue_freeze / graph construction
Required dataset outputs:
- `dataset/graph_manifest.csv`
- `dataset/graph_manifest.json`
- `dataset/run_summary.json`
- `dataset/graph_dataset_delivery_report.md`
- `dataset/cdr3_clusters.json`
- `dataset/excluded_samples.csv`
- `dataset/processing_failures.csv`

Required run-local independence outputs:
- `independence/pdb_family_clusters.json`
- `independence/pdb_family_clusters.provenance.json`
- `audit/cluster_universe.txt`

The clustering universe contains both internal study PDBs and required external-VHH PDBs. The frozen pair table/source map must cover this complete universe. When a pair TSV is the formal structure-similarity source, every universe PDB must appear in its query or target columns; an absent PDB is treated as "not demonstrated as searched", not as an independent singleton.

Required frozen-validation outputs:
- `validation_queue/freeze/selected_targets.json`
- `validation_queue/freeze/eligibility.json`
- `validation_queue/freeze/run_manifest.json`
- `validation_queue/freeze/freeze_manifest.json`
- one `prepared/<pdb>/recovery_manifest.json` for every frozen target

The freeze manifest (schema v2) hashes selected targets, eligibility, graph manifest,
the run-local family/structure cluster map, cluster-map provenance, and the complete
internal+external clustering universe.

### egnn_train
Required:
- `checkpoints/best_egnn_pruning.pt`
- `checkpoints/training_summary.json`
- `checkpoints/egnn_training_summary.md`
- `checkpoints/egnn_training_history.csv`
- `checkpoints/geometry_baseline.json`

The best checkpoint SHA256 must match the training summary and strict reload must
be recorded as verified.

### energy_calibration
Required:
- `calibration/coarse_to_amber_train.csv`
- `calibration/coarse_to_amber_train.provenance.json`
- `calibration/coarse_to_amber.json`
- `calibration/calibration_report.md`

The frozen calibration must pass all configured acceptance gates.

### method_sensitivity
Every configured shots × CVaR-alpha sub-run requires:
- `run_manifest.json`
- `run_summary.json`
- `metrics.csv`
- `summary.md`
- `failed_case_keys.json`
- `seed_streams.json`
- raw `cases/*.json` count equal to `cases_completed_total`

Aggregate sensitivity outputs:
- `method_sensitivity/sensitivity_summary.csv`
- `method_sensitivity/sensitivity_summary.json`
- `method_sensitivity/sensitivity_summary.md`

### qc_benchmark
Required:
- `qc_benchmark/run_manifest.json`
- `qc_benchmark/run_summary.json`
- `qc_benchmark/metrics.csv`
- `qc_benchmark/summary.md`
- `qc_benchmark/failed_case_keys.json`
- `qc_benchmark/seed_streams.json`
- raw `qc_benchmark/cases/*.json`

The raw case count must equal `cases_completed_total`. Every scaling case must contain exactly the requested number of Dunbrack-compatible rotamer sites; a mislabeled requested-site condition is a failed case, never silently accepted.

### structure_experiment
Both `dev_queue/` and `validation_queue/` require:
- `run_manifest.json`
- `seed_streams.json`
- `eligibility.json`
- `selected_targets.json`
- `run_summary.json`
- `real_complex_metrics.csv`
- `real_complex_report.md`

Every completed validation target additionally requires:
- `validation_queue/results/<pdb>/run_manifest.json`
- `validation_queue/results/<pdb>/recovery_metrics.csv`
- `validation_queue/results/<pdb>/recovery_report.md`

Every completed historical development target additionally requires:
- `dev_queue/<pdb>/results/<pdb>/run_manifest.json`
- `dev_queue/<pdb>/results/<pdb>/recovery_metrics.csv`
- `dev_queue/<pdb>/results/<pdb>/recovery_report.md`

Every configured non-primary solvent sensitivity requires a `dev_queue_solvent_<model>/`
root summary, aggregate recovery metrics, and report. Formal validation must close
exactly against frozen target × pre-registered seed × four-method denominators.

### external_validation
External VHH coarse benchmark requires:
- run-local `external_vhh_independence_manifest.json`
- `vhh_coarse/run_manifest.json`
- `vhh_coarse/run_summary.json`
- `vhh_coarse/metrics.csv`
- `vhh_coarse/summary.md`
- `vhh_coarse/seed_streams.json`
- `vhh_coarse/statistics_outputs.json`
- `vhh_coarse/statistics_outputs.md`
- raw `vhh_coarse/cases/*.json` consistent with its summary

External structural baseline requires:
- `structural_baselines/run_summary.json`
- `structural_baselines/external_baseline_metrics.csv`
- `structural_baselines/external_baseline_report.md`

### statistics
Required:
- `qc_benchmark/statistics_outputs.json`
- `qc_benchmark/statistics_outputs.md`
- `qc_benchmark/statistics_time.json`
- `qc_benchmark/statistics_time.md`
- `statistics/quantum_scaling_statistics.json`
- `statistics/quantum_scaling_statistics.md`
- `statistics/structure_statistics.json`
- `statistics/structure_statistics.md`

The primary coarse solver inference is restricted to the frozen primary pruning path,
radius, QAOA depth, optimization-evaluation budget, active-site size, output budget,
QAOA objective/restarts, and requires the configured minimum independent clusters.
Scaling inference uses the pre-declared active-site levels under the same frozen primary
radius/depth/evaluation budget and a cluster-aware within-PDB slope analysis with
log10(feasible configuration count) as the primary complexity axis.

### final_report
Required:
- `FINAL_RESEARCH_REPORT.md`

## Run-level audit outputs

Every formal run terminates with:
- `results_manifests/<stage>.json` for executed stages
- `EXPERIMENT_RESULTS_AUDIT.json`
- `EXPERIMENT_RESULTS_AUDIT.md`
- `artifact_inventory.json`
- `RUN_SUMMARY.json`

`RUN_SUMMARY.json.status` may be `completed` only when no stage failed and
`EXPERIMENT_RESULTS_AUDIT.json.all_required_results_present` is true.


## Partial reruns

`--only` is an operational repair tool, not a way to declare a partial experiment complete.
A required stage skipped in the current invocation is acceptable only if an earlier completed
stage marker exists in the same run and its artifacts revalidate successfully. Otherwise the
run-level results audit fails and `RUN_SUMMARY.json.status` remains `failed`.
