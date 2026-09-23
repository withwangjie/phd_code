# Formal experiment results contract

This document defines the minimum auditable output set for a completed formal run.
A stage is not accepted as completed merely because its subprocess returned zero.
The orchestrator verifies the required result set, writes a per-stage manifest under
`results_manifests/`, and performs a run-level audit at shutdown.
On resume, every listed artifact must still exist with its recorded size;
recorded SHA-256 values are recomputed and compared. The manifest currently
records SHA-256 for artifacts up to 64 MiB and size only for larger files.

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

`test_db55` is a reserved graph set. The formal training and evaluation stages
consume `train` and `test_snac_hard`; they do not score `test_db55`. DB5.5
graphs receive the documented backbone and bound-interface checks, not the
full structure-quality gate. Every DB5.5 pair PDB ID in the audited pair table
is excluded from the training candidate pool, including pairs whose graphs
fail construction. This PDB-level reservation is not a homology claim.

Required run-local independence outputs:
- `independence/pdb_family_clusters.json`
- `independence/pdb_family_clusters.provenance.json`

Similarity edges use connected components (single linkage). A chain of edges
can merge endpoints with no direct high-similarity edge. This is conservative
for leakage prevention and may reduce the number of independent clusters;
`largest_component` is reported for auditing, with no fixed rejection cutoff.
Every non-comment pair row must contain valid identifiers and a numeric score;
malformed rows abort map construction. Formal queue freezing also rejects older
maps whose provenance reports skipped rows.
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
Scaling cases also require exactly three retained states per site, one in each
chi1 well. The statistics stage rejects older adaptive-allocation cases. A
fresh benchmark output directory is required after this protocol change.
The training-only rotamer-resolution sensitivity runs eight paired conditions:
4 and 5 sites crossed with 3, 4, 5 and 6 states per site, using the same
target subset and repeat seeds. Its `method_sensitivity/rotamer_resolution/`
directory contains separate case runs plus `summary.csv`, `summary.json`, and
`summary.md`.
These solver metrics do not establish native chi1/chi2 or all-atom recovery.

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

The independence audit reconstructs each chain sequence from the graph's
one-hot residue nodes and chain indices. It rejects mismatched chain, partner,
or CDR-H3 metadata. The required `source_structure_dir` supplies one trusted
raw `<pdb>.pdb`, `.cif`, or `.mmcif` file per external graph. Every graph
residue identity, amino acid, and CA position must match that raw structure;
the source file SHA-256 is recorded and rechecked before benchmarking. The
directory itself must come from a trusted structure source.
The server may set its path with `QP_EXTERNAL_VHH_SOURCE_DIR` (or
`server_config.yaml`); the resolved path is frozen in run provenance.

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

RQ5 completion requires the configured minimum independent clusters and finite
Spearman rho, permutation p value, bootstrap confidence limits, and Holm-adjusted
p value. A constant cluster-level energy or RMSD difference makes rho undefined;
the statistics stage fails and reports the missing fields in that case. Resume
validation applies the same gate to previously written results.
Resume also rechecks the original coarse and scaling cluster minima, primary
structural cluster minimum, failed QAOA/pair exclusions, benchmark failure
fraction, and zero failed targets in the formal validation queue.

The primary coarse solver inference is restricted to the frozen primary pruning path,
radius, QAOA depth, optimization-evaluation budget, active-site size, output budget,
QAOA objective/restarts, and requires the configured minimum independent clusters.
The paired-statistics output records all exclusion counts and a separate
`denominator_failures` summary. Formal stage completion and resume validation
reject any excluded case in the paired QAOA-vs-classical contrasts: failed
QAOA restarts, missing/ambiguous contrasts or pairs, unequal output counts,
invalid time budgets, excessive time overruns, or nonfinite analyzed metrics.
Time-mode diversity metrics are intentionally not analyzed, so their
nonfinite values are not denominator failures. Scaling inference also rejects
an incomplete primary denominator.
Scaling inference uses the pre-declared active-site levels under the same frozen primary
radius/depth/evaluation budget and a cluster-aware within-PDB slope analysis with
log10(feasible configuration count) as the primary complexity axis.

The matched-output QC effects and the primary scaling slope form one Holm-adjusted
inferential family; raw and adjusted p values are both retained. Structural primary
and RQ5 tests retain their separate two-test Holm family.

The external FASPR comparison uses an Active-only packing scope: backbone and
non-Active side-chain coordinates are restored from the perturbed input before
scoring. Energy-calibration rows are buffered per training complex and committed
only after that complex completes; calibration uses the same fixed three-chi1-well
state policy as the formal scaling benchmark.

The resolved DDP rank count is a result-affecting runtime parameter: rank seeds
and sampler partitions can change the EGNN checkpoint even at fixed global
batch size. `runtime_resolution.ddp_ranks` and the resolved-config hash bind
each run to the chosen rank count.

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
