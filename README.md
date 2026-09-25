# phd_code

## Research scope

This repository benchmarks constrained quantum-classical optimization for
nanobody (VHH)-antigen interface side-chain reconstruction under a known
complex pose/backbone. It is a retrospective side-chain recovery benchmark,
not blind docking, de novo complex prediction, or quantitative binding-affinity
prediction.

## Formal research pipeline

1. **Leakage-controlled data construction**
   - Formal source roles are explicit: `snac_db` is the primary VHH-antigen source,
     `sabdab_vhh` is auxiliary training data, and generic `train_rcsb` is audit-only
     because a strongest-contact chain pair does not establish VHH identity.
     Cross-source duplicate PDBs are resolved before quality/outcome inspection with
     fixed priority SNAC-DB > SAbDab > RCSB.
   - Complex definition: `sabdab_vhh` is read as the first author-determined biological
     assembly (entries without assembly annotation are excluded); SAbDab H/L/antigen
     metadata defines the VHH entry and only annotated antigen chains within 7.5 A of the
     VHH paratope are retained. `snac_db` uses SNAC-DB's assembly-curated complexes [R34].
   - Interface labels: primary cross-partner heavy-atom contact <= 4.5 A [R49]. Node-aligned 3.5 A and 5.0 A labels are frozen in graph v1.11 for strict/permissive sensitivity analyses; graph edges remain threshold-independent KNN.
   - Graph edges: intra-chain CA radius < 8 A plus fixed cross-partner KNN. An 8 A C-alpha residue-graph cutoff has direct protein-GNN precedent [R22]; the cross-partner KNN degree k=3 remains a study-specific leakage-control choice rather than a literature-optimal constant.
   - EGNN train/validation split: layered connected components. Complexes are
     joined if VHH full-chain identity >=80% [R24], CDR-H3 loop-only identity >=50% [R23], or
     antigen full-chain identity >=30% with >=70% minimum length coverage [R25]; no random 90/10 split.
     All formal training graphs have annotation-anchored VHH/antigen roles.
   - Antigen-fold holdout: layered components are built jointly from SNAC and SAbDab so auxiliary homologues cannot remain in training. Scored targets are SNAC-only (one deterministic graph per PDB); SAbDab and duplicate same-PDB members of selected components are quarantined outside both training and evaluation. Holdout raw structures are bound by exact audit source_id + SHA-256.

2. **Antigen-conditioned Active-site selection**
   - E(n)-equivariant EGNN [R1] provides residue-level interface probabilities.
   - Formal EGNN ranking uses
     `(1-w) * EGNN + w * exp(-d_Ag/6A)`, with `w=0.25` by default.
   - Contact, nearest-distance, CDR and random strategies are explicit ablation baselines.

3. **Adaptive side-chain state construction**
   - Formal coarse modeling uses a chi1-oriented pseudo-atom approximation,
     while formal all-atom validation uses complete backbone-dependent Dunbrack
     2010 rotamer states [R2] (chi1..chiN) queried from residue phi/psi context.
     Legacy hand-written chi1 priors are debug/compatibility only.
   - Candidate pre-screening uses local environment / antigen-conditioned
     interaction scoring (coarse model; the antigen and fixed-VHH terms see
     every residue of the full complex, limited only by the 8 A atom-pair
     cutoff [R31,R32]; phi/psi for Dunbrack lookup are defined only across
     real peptide bonds, so residues at chain breaks are not Active sites) or Amber14 single-candidate energy
     (all-atom validation). Coarse prior/VHH/antigen/pair terms may be linearly
     calibrated to Amber delta-E using training complexes only, with frozen
     coefficients for validation/test.
   - 3--6 states per Active residue are retained under a global <=30-variable
     budget. Adaptive residue-dependent coarse rotamer counts including 1/3/6-state schemes have direct precedent [R26]; this study's minimum of 3 states and <=30-bit global cap remain preregistered resource constraints.
   - Formal quantum-classical scaling axis: 4, 6, 8, and 10 Active residues. Six sites remains the preregistered primary confirmatory/all-atom size; the other sizes are scaling conditions, not literature-defined standards.

4. **Constrained discrete optimization**
   - Fixed-backbone rotamer selection is treated as a combinatorial side-chain positioning problem [R10-R12] and encoded with one-hot QUBO/Ising variables.
   - QAOA follows the hybrid variational framework of Farhi et al. [R7]; protein/peptide quantum-optimization precedent is provided by [R16-R18].
   - XY-mixer QAOA preserves local Hamming weight and therefore feasibility.
   - Classical baselines include exact feasible-state enumeration and simulated annealing [R13].
   - Mean-energy and finite-shot CVaR objectives are compared. CVaR is supported by [R8], which explicitly evaluates alpha=0.10 and recommends approximately 0.1-0.25 as a useful empirical range; alpha=0.1 is therefore literature-supported but still preregistered and sensitivity-tested here.
   - Primary (confirmatory) endpoint: log10 exact ground-state amplification of
     QAOA over uniform feasible sampling at 6 sites, and its scaling slope
     [R29,R37]. QAOA-vs-classical time-to-solution (`log10_qts99`, with and
     without training shots) is secondary [R37,R45]. See
     `docs/PROTOCOL_AMENDMENTS.md` A1, A7.
   - Exploratory `quantum_exploration` stage: depth p in {1,2,3,4,6} at 20
     optimizer evaluations per parameter, and QAOA angles fitted on training
     complexes transferred untrained to the hard set [R45-R47].

5. **Structure-level validation**
   - Solver assignments are reconstructed as side-chain conformations.
   - OpenMM [R19] constrained relaxation with the ff14SB protein force field [R15] evaluates whether discrete energy gains persist after continuous structural refinement.
   - Structural metrics include Active side-chain RMSD, Fnat, interface RMSD,
     ligand RMSD, clash measures, and related trajectory metrics.

## Central research questions

1. How faithfully can the constrained rotamer-assignment problem be encoded as
   an auditable one-hot QUBO/Ising instance for gate-based quantum optimization?
2. Does feasibility-preserving XY-QAOA concentrate probability on the ground
   state beyond uniform feasible sampling (primary: log10 exact ground-state
   amplification at the preregistered size), and how does it compare with
   classical baselines in resource-normalized time-to-solution with and
   without training shots [R37,R45] (secondary)?
3. How does QAOA's ground-state amplification change as feasible
   configuration count, logical qubit count, and logical two-qubit-gate count
   grow across the preregistered problem-size axis (primary scaling slope)?
   Exploratory: how do circuit depth and optimizer budget trade off, and do
   QAOA angles fitted on training complexes transfer to new complexes without
   re-optimization [R46,R47]?
4. Does finite-shot CVaR improve the low-energy sampling behavior of the
   variational quantum solver relative to mean-energy optimization?
5. Do solver-level discrete energy gains propagate to all-atom structural
   improvements after reconstruction/relaxation?
6. Supporting problem-reduction questions ask whether EGNN interface selection
   remains useful after homology/topology-leakage control and whether adaptive
   rotamer discretization preserves adequate state coverage. These are enabling
   components, not the primary research object.

## Interpretation limits

- No hardware quantum advantage or quantum speedup claim is made from
  simulator data. QAOA results are noiseless exact-subspace simulations;
  matched-output/matched-time and scaling results are interpreted only as
  algorithmic relative-performance evidence [R20]. Matched-time comparisons
  measure the classical cost of emulating QAOA and are descriptive.
- Coarse QC multiplicity uses serial gatekeeping: only QAOA's primary-size
  ground-state amplification and its scaling slope are confirmatory at first;
  QAOA-vs-classical effects are confirmatory only after both are rejected
  [R41,R42]. Depth/budget and parameter-transfer analyses are exploratory and
  descriptive.
- Independent-cluster adequacy is checked at queue freeze, before any outcome
  (`--stop-after queue_freeze`), and EGNN training-seed variance is reported
  from development-only replicates.
- Coarse antigen interaction scores are not binding free energies; contact
  number is a geometry baseline, not an affinity estimator.
- All-atom primary experiments are strict fixed-backbone: N/CA/C/O coordinates
  never move (`loop_relax_iterations: 0`). Formal recovery perturbs and
  reconstructs every defined Active side-chain chi. The coarse
  energy surrogate remains chi1-oriented and is calibrated against Amber14 on
  training complexes only.
- Smoke checks and legacy explicit `chi1_angles` overrides are engineering or
  ablation paths and are not the formal main protocol.
- Every protocol change after the original freeze is listed, with its reason
  and inspection status, in `docs/PROTOCOL_AMENDMENTS.md`.

## Scientific configuration

The formal pipeline separates **configurable experimental parameters** from
**fixed model-definition constants**.

Configurable in `configs/full_experiment_config.yaml`:

- Frozen quantum protocol: XY-QAOA encoding/mixer semantics, primary depth,
  finite-shot objective/budgets, benchmark objective/restart ablations, and
  development-only QAOA sensitivity are centralized under `quantum_protocol`.
  Downstream benchmark, structural, external-validation and statistical stages
  read this single source of truth; duplicated QAOA settings in stage-specific
  blocks are rejected.
- Dataset/graph protocol: sequence identity threshold, heavy-atom interface
  label cutoff, intra-chain CA radius, cross-partner KNN degree, and minimum
  interface-residue count.
- Site-selection protocol: Active-site count, antigen-guidance weight,
  antigen-proximity decay length, contact-baseline CA cutoff, and environment
  radius.
- Coarse interaction model: non-bonded cutoff, soft-core delta, hard-core
  fraction/penalty, LJ caps, Coulomb cap, dielectric model parameters, and kT.
- Benchmark resources: classical baseline budgets, matched-output curves and
  low-energy windows remain under `qc_benchmark`; QAOA-specific budgets live
  only under `quantum_protocol`.
- Structural experiment budgets: perturbation range, local/final relaxation
  iterations, output shots, and optimization budget.

Every graph records its graph protocol; every EGNN checkpoint records the
identity split threshold and graph protocol; every QUBO records its force-field
parameters; run manifests record CLI arguments/config hashes. Resume is
fail-closed when these protocol-defining values differ.

Deliberately fixed model-definition constants include the Coulomb conversion
constant, one-hot register semantics, amino-acid chemistry tables, and the
current 3--6-state / <=30-variable formal representation. Those should only be
changed as a new method/version, not casually swept as run-time hyperparameters.

## Energy calibration input

The coarse-to-Amber calibration is **training-only** and frozen before any
validation/test benchmark. The input CSV must contain:

- `pdb_id`
- `split` (every row must be exactly `train`)
- `prior_energy`
- `vhh_environment_energy`
- `antigen_energy`
- `pair_energy`
- `amber_delta_kcal`

Rows from one PDB are kept together. The fitter uses deterministic
PDB-grouped SHA256 5-fold cross-validation and a ridge-regularized linear
model with an unconstrained intercept and **nonnegative component weights**.
The frozen JSON records train/CV RMSE and MAE, train R2, PDB/sample counts,
coefficient constraints, ridge alpha, and source SHA256. Validation/test rows
are rejected at fit time, and the JSON loader rejects files that do not state
training-only provenance.

If the frozen JSON is absent but the configured training CSV exists,
`run_full_experiment.py` fits the JSON before the coarse benchmark. If both
are absent while `require_calibrated: true`, the run fails closed.

The formal rotamer model reads the official Dunbrack 2010
`ALL.bbdep.rotamers.lib` φ/ψ bins, rotamer probabilities, χ1..χN means and
σ values. χ1 is expanded by configured σ offsets; distal χ values retain the
corresponding statistical rotamer means. The all-atom model applies complete
rotamer χ1..χN states and retains 3--6 states/site under the <=30-variable
budget. The coarse pseudo-atom model intentionally remains χ1-oriented. The
official library file is an external scientific input and is never silently
replaced by the legacy table in formal mode.

## Independence, external validation, and structural baselines

Formal confirmation requires both layered sequence isolation and a frozen
PDB-to-family/structure cluster map. The same cluster map is used for
train/test exclusion, validation-queue eligibility, and cluster-level
statistics. Missing required cluster metadata fails closed.

The formal pipeline also has an external-validation stage. By default it
scores an **antigen-fold holdout** (`docs/PROTOCOL_AMENDMENTS.md` A10): whole
layered-isolation components are carved out of the training split at
`queue_freeze`, before any training, and the frozen EGNN, calibrated coarse
model and primary solver protocol are applied to them without refitting. The
claim it supports is that the frozen pipeline still holds on antigen folds
training never saw; it is a cluster-level holdout of the same audited
snapshot, not a separate database. Setting
`external_validation.external_vhh.graph_dir` and `source_structure_dir`
scores an independently certified graph-v1.11 VHH dataset instead, with the
identical independence audit.
FASPR is supported as a mature biological side-chain packing baseline and
Phenix clashscore as a standard steric-quality diagnostic. These are
scientific external dependencies: they are never substituted or fabricated
when executables/data are unavailable.

The pre-registered primary structural endpoint is post-relaxation,
symmetry-corrected Active-side-chain heavy-atom RMSD, with QAOA-vs-SA as the
primary structural contrast. RQ5 additionally reports a within-target/method
centered Spearman association between discrete energy and final RMSD with
family-cluster bootstrap confidence intervals.

## Formal external resources and fail-closed preflight

A formal run intentionally fails before expensive computation when required
scientific inputs are unavailable. The preflight verifies the Dunbrack 2010
library, a frozen family/structure similarity input (cluster map or pair TSV),
FASPR, Phenix clashscore, and OpenMM GBN2 parameters when the declared solvent
sensitivity is enabled. When a genuinely external VHH set is configured, its
graph/raw-structure inputs are also required; under the default antigen-fold
holdout protocol those run-local inputs are created later by queue_freeze and
are therefore not required at preflight time.

Family/structure clusters can be reproducibly built from a frozen Foldseek (or
equivalent) pair table with `build_independence_cluster_map.py`. The script
keeps connected components, supports singleton universe IDs, and records the
pair-table SHA256 and threshold. Dataset resume is invalidated if the frozen
cluster map changes.

External VHH independence is not accepted from a hand-written Boolean alone.
`audit_external_vhh_independence.py` compares every external graph against
the frozen training graphs using the same VHH/CDR-H3/antigen thresholds and
the same family/structure cluster map, then writes a per-target auditable
manifest. The external-validation stage verifies that manifest before running
the frozen model.

### Preparing the external VHH set

External complexes are built with the training-graph definition (graph v1.11,
biological assembly, 7.5 A SAbDab antigen rule, the same labels and edges):

Structure files that carry no resolution record need the entry table the audit
reads by PDB ID. Ask a finished audit which entries those are, so one table
covers every subset, and fetch them once:

```bash
python -m nanoqc.data.fetch_entry_resolution      # --resume continues an interrupted fetch
```

With no arguments it takes the most recent run's audit and writes the data
root's `entry_resolution.tsv`; `--missing-from-audit` names a different audit.

Any `*entry_resolution.tsv` under the data root (outside pipeline staging) is
read, listed in the audit report with its SHA-256, and needs no network again.

With the antigen-fold holdout (the default), the only other input a formal run
needs beyond the raw data is the frozen Foldseek pair table:

```bash
./scripts/prepare_external_vhh.sh audit                     # study PDB IDs from a standalone audit
./scripts/prepare_external_vhh.sh foldseek --foldseek /path/to/foldseek --threads 32
./scripts/deploy_launch.sh                                  # queue_freeze carves the holdout
```

The `foldseek` step searches antigen chains only. Antibody chains are
recognised from SNAC/SAbDab annotations, SAbDab chain IDs, or an Ig V-domain
detector (ANARCI when installed); annotated antigen chains are always kept.
It runs an exhaustive all-versus-all search (E <= 10) and writes the pair
table, a manifest of every chain's role, and an explicit self row for a PDB
without an antigen chain of at least 20 residues. Scores are symmetric
(`mintmscore`, the chain-pair TM-score normalized by the longer chain; A11), so
a short chain cannot join whole complexes. `--reuse-raw` rescores the previous
search (`prep/foldseek/foldseek_raw.m8`) without running Foldseek again. A
layered component larger than one fold's share of the pool always trains (A11).
The holdout takes `queue_freeze.antigen_fold_holdout.fold`, one index or several
(`"1,2"`); several are taken, lowest first, when one fold does not reach
`min_components` (A13).

To score a **genuinely external** VHH set instead, set
`external_validation.external_vhh.graph_dir` and `source_structure_dir`, and
build that set with the two passes below before the Foldseek step, so its
PDBs enter the clustering universe:

```bash
./scripts/prepare_external_vhh.sh pass1 --sabdab-summary sabdab_summary_all.tsv --download
./scripts/prepare_external_vhh.sh foldseek --foldseek /path/to/foldseek
./scripts/deploy_launch.sh --stop-after queue_freeze
./scripts/prepare_external_vhh.sh pass2 --sabdab-summary sabdab_summary_all.tsv --run-dir <that run>
./scripts/deploy_launch.sh
```

`--released-after YYYY-MM-DD` adds a temporal holdout when PDB releases
postdate the training snapshot. Pass 2 refuses to run if the pair table
changed after pass 1, because
external PDBs can link internal clusters. With the table unchanged, the fresh
run rebuilds the identical training/test split.

Each graph is one VHH-antigen complex, as in the SNAC-DB per-VHH complexes
behind the training and hard test sets. Entries with several nanobodies give
one complex per PDB: the VHH with the largest passing interface. Other
antibody chains are never antigen, and entries with a VH/VL chain are
excluded. The final set (pass 2) keeps one representative per layered
homology group. Selection also requires a protein or peptide antigen, resolution
<= 3.0 A, and the audit's interface quality gates. It then groups the complexes by the layered
homology rule and reports whether `min_clusters` independent groups exist.
CDR-H3 follows IMGT (105-117), like the training SNAC `Region_Split_VH.cdr3`;
with `--training-dataset` the report states how often the external CDR-H3
rule reproduces the training annotation. Formal SAbDab/external candidates
require ANARCI IMGT numbering and fail closed when ANARCI is unavailable.
The motif locator is retained only for non-formal debug/legacy paths and
cannot define a formal external graph. Every formal run re-certifies
independence against its own frozen training set and cluster map.

The primary all-atom protocol uses vacuum/NoCutoff Amber14 packing energy.
GBN2 energies are not exactly pair-decomposable (Born radii depend on every atom), so
the GBN2 QUBO is a recorded pairwise approximation (`pair_decomposition`,
`all_atom_equivalence_max_error/rms_error`, per-structure `qubo_energy_discrepancy_kcal`);
the vacuum primary protocol still requires exact decomposition (1e-4 kcal/mol).
A pre-declared GBN2 sensitivity run is performed on development targets only;
the validation queue never chooses its solvent model after inspecting results.

## Repository layout

```text
configs/   full_experiment_config.yaml (frozen scientific protocol), server_config.yaml (infrastructure only)
docs/      METHODS_EVIDENCE.md (design-to-literature register), RESULTS_CONTRACT.md (required outputs),
           PROTOCOL_AMENDMENTS.md (dated post-freeze protocol changes),
           PIPELINE_WALKTHROUGH.md (stage-by-stage walkthrough of what each stage does and why)
scripts/   deploy_launch.sh, run_full_experiment.sh, formal_preflight.sh, check_status.sh, repair_openmm_cuda.sh,
           prepare_external_vhh.sh
src/nanoqc/
  pipeline/     run_full_experiment.py (end-to-end orchestrator), resolve_server_config.py
  data/         audit_all_datasets.py, build_final_pyg_dataset.py (audited graphs + split),
                build_independence_cluster_map.py, audit_external_vhh_independence.py, sequence_identity.py,
                select_external_vhh_candidates.py, build_external_vhh_graphs.py,
                prepare_external_vhh.py (external VHH set)
  model/        model_egnn_pruning.py (interface scoring, Active-site selection, checkpoint loading),
                train_egnn_pruning.py (leakage-controlled training),
                egnn_seed_sensitivity.py (development-only seed variance)
  qubo/         subgraph_to_qubo.py (coarse / all-atom adaptive side-chain QUBO builders)
  quantum/      instance.py (quantum instance contract), resource_estimation.py (logical resources)
  solvers/      qaoa_interface_sampler.py (XY-mixer QAOA)
  experiments/  batch_benchmark_hard_set.py (matched quantum/classical benchmark),
                fit_qaoa_transfer_parameters.py (train-split QAOA angle transfer),
                run_real_complex_pilot.py (all-atom retrospective recovery),
                generate_energy_calibration_dataset.py, run_external_structure_baselines.py (FASPR / Phenix)
  structure/    evaluate_complex_metrics.py, structural_quality.py, residue_tables.py
  inference/    paired_statistics.py (incl. serial gatekeeping), analyze_quantum_scaling.py,
                analyze_quantum_exploration.py, analyze_structure_recovery.py (RQ5)
  reporting/    generate_final_research_report.py, generate_figure1_pymol_script.py
  common/       repo_io.py (hashing, layout), seed_streams.py, prediction_contract.py
tests/     regression suite run by formal_preflight.sh (`python -m pytest tests`)
```

Every stage runs as `python -m nanoqc.<package>.<module>` from the repository
root with `src/` on `PYTHONPATH`; the scripts in `scripts/` set this up. Run
manifests keep fingerprinting modules under their bare file names
(`code_sha256["subgraph_to_qubo.py"]`), resolved through
`nanoqc.common.repo_io.MODULE_LAYOUT`. `tests/test_refactor_equivalence.py`
checks the shared helpers against the implementations they replaced.

## Literature basis

The authoritative design-to-literature mapping is maintained in `docs/METHODS_EVIDENCE.md`. Reference labels [R1]–[R47] in this README refer to that file. The register explicitly separates direct literature support from literature-informed preregistration and study-specific preregistration so that exact numerical choices are never misrepresented as published standards.

## Portable server runtime configuration

Scientific protocol and server infrastructure are intentionally separated.

- `configs/full_experiment_config.yaml` contains the frozen scientific protocol.
- `configs/server_config.yaml` contains server paths, resource policy, external-tool locations, and operational resource gates.
- `nanoqc.pipeline.resolve_server_config` detects CPU/GPU/RAM and resolves paths/tools before formal preflight, then writes `.runtime/resolved_runtime_config.yaml` and `.runtime/server_resolution.json`.
- `scripts/run_full_experiment.sh` uses the same resolved runtime config for both preflight and the formal run.
- `requirements.txt` lists the runtime dependencies (unpinned, Python >= 3.10). The exact installed versions of each run are archived in `provenance/pip_freeze.txt`.

Portable overrides can be supplied without editing the scientific protocol:

```bash
export QP_VENV=/path/to/.venv
export QP_DATA_ROOT=/path/to/data
export QP_RUN_ROOT=/path/to/runs
export QP_FASPR=/path/to/FASPR
export QP_PHENIX_CLASHSCORE=/path/to/phenix.clashscore
./scripts/deploy_launch.sh
```

In auto mode the resolver chooses DDP ranks from available GPUs while preserving the configured target global batch size, derives CPU worker counts from available physical cores, selects the OpenMM platform from available hardware, and records every resolved value in the run configuration/provenance. Scientific thresholds, QAOA protocol values, data-split rules, endpoints, and statistical choices are never hardware-auto-tuned.

## One-click formal experiment

A formal experiment is launched with one command:

```bash
./scripts/deploy_launch.sh
```

The launcher resolves the current server, creates exactly one timestamped run directory before preflight, and stores the complete experiment archive under that directory. A fresh run contains:

```text
<run_root>/experiments_full_run_<UTC>/
├── provenance/
│   ├── scientific_config.source.yaml
│   ├── server_config.source.yaml
│   ├── resolved_runtime_config.yaml
│   ├── server_resolution.json
│   ├── METHODS_EVIDENCE.md
│   ├── RESULTS_CONTRACT.md
│   ├── PROTOCOL_AMENDMENTS.md
│   ├── git_head.txt
│   ├── pip_freeze.txt
│   └── environment.txt
├── logs/
│   ├── formal_preflight_<UTC>.log
│   ├── launch_<UTC>.log
│   └── <stage>.log
├── frozen_config.yaml
├── run_manifest.json
├── seed_streams.json
├── progress.json
├── audit/
├── dataset/
├── independence/
├── checkpoints/
├── calibration/
├── method_sensitivity/
├── qc_benchmark/
├── quantum_exploration/
├── dev_queue/
├── validation_queue/
├── external_validation/
├── statistics/
├── results_manifests/
├── FINAL_RESEARCH_REPORT.md
├── EXPERIMENT_RESULTS_AUDIT.json
├── EXPERIMENT_RESULTS_AUDIT.md
├── artifact_inventory.json
└── RUN_SUMMARY.json
```

If preflight fails, the same run directory is retained with its provenance and preflight log. If a run is resumed, new timestamped preflight/launch logs are appended as new files inside the same original run directory rather than creating a second result tree.

The formal manuscript/analysis should treat one run directory as the atomic reproducibility unit.

## Mandatory experiment-result audit

The formal output requirements are defined in `docs/RESULTS_CONTRACT.md`. A zero subprocess return code is never sufficient to mark an experimental stage complete.

For every executed stage, the orchestrator validates its mandatory raw/aggregate result files and writes:

```text
results_manifests/<stage>.json
```

At the end of the run it re-validates all completed stages and writes:

```text
EXPERIMENT_RESULTS_AUDIT.json
EXPERIMENT_RESULTS_AUDIT.md
artifact_inventory.json
RUN_SUMMARY.json
```

`RUN_SUMMARY.json.status == "completed"` is permitted only when all formal stages that ran satisfy their result contracts and the global results audit passes. Missing QC metrics, missing sensitivity aggregates, missing per-target recovery outputs, missing external-baseline results, missing statistical artifacts, or an inconsistent raw-case count therefore converts the formal run to failed rather than silently producing a partial result set.
