# phd_code

## Research scope

This repository benchmarks constrained quantum-classical optimization for
nanobody (VHH)-antigen interface side-chain reconstruction under a known
complex pose/backbone. It is a retrospective side-chain recovery benchmark,
not blind docking, de novo complex prediction, or quantitative binding-affinity
prediction.

The study's object is the **quantum algorithm itself**, not a speed claim
against classical solvers. Earlier inspected benchmark cases showed simulated
annealing reaching the ground state consistently; that observation does not
predetermine outcomes in a new formal run. The confirmatory endpoints are
QAOA's own ground-state amplification and its finite-range scaling slope
(`docs/PROTOCOL_AMENDMENTS.md` A7, A25).

## Formal research pipeline

1. **Leakage-controlled data construction**
   - Formal source roles are explicit: `snac_db` is the primary VHH-antigen source,
     `sabdab_vhh` is auxiliary training data, and generic `train_rcsb` is audit-only
     because a strongest-contact chain pair does not establish VHH identity.
     Cross-source duplicate PDBs are resolved before quality/outcome inspection with
     fixed priority SNAC-DB > SAbDab > RCSB (A16).
   - Complex definition: `sabdab_vhh` is read as the first author-determined biological
     assembly (entries without assembly annotation are excluded); SAbDab H/L/antigen
     metadata defines the VHH entry and only annotated antigen chains within 7.5 A of the
     VHH paratope are retained. `snac_db` uses SNAC-DB's assembly-curated complexes [R34].
     A file already served as an assembly (`<entry>_assembly<N>`) is read as that assembly
     and never transformed twice (A14).
   - Entry resolution comes from the structure file, else by PDB ID from curation
     metadata (SNAC summaries, SAbDab summaries, a fetched RCSB table), worst value
     of a multi-valued field; every used metadata file is listed with its SHA-256 in
     the audit report (A12, A15).
   - Interface labels: primary cross-partner heavy-atom contact <= 4.5 A [R49]. Node-aligned 3.5 A and 5.0 A labels are frozen in graph v1.11 for strict/permissive sensitivity analyses; graph edges remain threshold-independent KNN.
   - Graph edges: intra-chain CA radius < 8 A plus fixed cross-partner KNN. An 8 A C-alpha residue-graph cutoff has direct protein-GNN precedent [R22]; the cross-partner KNN degree k=3 remains a study-specific leakage-control choice rather than a literature-optimal constant.
   - EGNN train/validation split: layered connected components. Complexes are
     joined if VHH full-chain identity >=80% [R24], CDR-H3 loop-only identity >=50% [R23], or
     antigen full-chain identity >=30% with >=70% minimum length coverage [R25]; no random 90/10 split.
     All formal training graphs have annotation-anchored VHH/antigen roles.
   - Antigen-fold holdout: layered components are built jointly from SNAC and SAbDab so auxiliary homologues cannot remain in training. Scored targets are SNAC-only (one deterministic graph per PDB); SAbDab and duplicate same-PDB members of selected components are quarantined outside both training and evaluation. Holdout raw structures are bound by exact audit source_id + SHA-256 (A17).

2. **Antigen-conditioned Active-site selection**
   - E(n)-equivariant EGNN [R1] provides residue-level interface probabilities.
   - Formal EGNN ranking uses
     `(1-w) * EGNN + w * exp(-d_Ag/6A)`, with `w=0.25` by default.
   - Contact, nearest-distance, CDR and random strategies are explicit ablation baselines.
   - Formal training runs in CUDA FP32. FP16 automatic mixed precision is disabled
     because unnormalized squared distances and the unbounded coordinate update
     path can overflow the FP16 range (A20).

3. **Adaptive side-chain state construction**
   - Formal coarse modeling uses a chi1-oriented pseudo-atom approximation,
     while formal all-atom validation uses complete backbone-dependent Dunbrack
     2010 rotamer states [R2] (chi1..chiN) queried from the installed PyRosetta
     dun10 database at the residue's nearest 10-degree phi/psi bin (A19).
     Legacy text-library and hand-written chi1 priors are debug/compatibility only.
   - Candidate pre-screening uses local environment / antigen-conditioned
     interaction scoring (coarse model; the antigen and fixed-VHH terms see
     every residue of the full complex, limited only by the 8 A atom-pair
     cutoff [R31,R32]; phi/psi for Dunbrack lookup are defined only across
     real peptide bonds, so residues at chain breaks are not Active sites) or Amber14 single-candidate energy
     (all-atom validation). Coarse prior/VHH/antigen/pair terms may be linearly
     compared with Amber delta-E on training complexes as a diagnostic. The
     current formal branch uses the uncalibrated coarse QUBO for matched solver
     comparisons and evaluates atomistic structural outcomes separately.
   - Fixed-resolution three-well cases keep the highest positive-probability actual
     library sample in each chi1 well even below the global probability floor; the
     rescued sample keeps its own probability and prior-energy penalty, and a well
     with no positive-probability sample still fails (A21).
   - 3--6 states per Active residue are retained under a global <=30-variable
     budget. Adaptive residue-dependent coarse rotamer counts including 1/3/6-state schemes have direct precedent [R26]; this study's minimum of 3 states and <=30-bit global cap remain preregistered resource constraints.
   - Formal quantum-classical scaling axis: 4, 6, 8, and 10 Active residues. Six sites remains the preregistered primary confirmatory/all-atom size; the other sizes are scaling conditions, not literature-defined standards.

4. **Constrained discrete optimization**
   - Fixed-backbone rotamer selection is treated as a combinatorial side-chain positioning problem [R10-R12] and encoded with one-hot QUBO/Ising variables.
   - QAOA follows the hybrid variational framework of Farhi et al. [R7]; protein/peptide quantum-optimization precedent is provided by [R16-R18].
   - XY-mixer QAOA preserves local Hamming weight and therefore feasibility [R28,R30];
     the initial state is a product of local one-hot W states.
   - Classical baselines include exact feasible-state enumeration and simulated annealing [R13].
   - Mean-energy and finite-shot CVaR objectives are compared. CVaR is supported by [R8], which explicitly evaluates alpha=0.10 and recommends approximately 0.1-0.25 as a useful empirical range; alpha=0.1 is therefore literature-supported but still preregistered and sensitivity-tested here.
   - Primary (confirmatory) endpoint: log10 exact ground-state amplification of
     QAOA over uniform feasible sampling at 6 sites, and its scaling slope
     [R29,R37]. QAOA-vs-classical `log10_qts99` with and without training
     shots is descriptive abstract accounting [R37,R45]. Other matched-output
     solver effects are secondary under the gatekeeping rule. See
     `docs/PROTOCOL_AMENDMENTS.md` A1, A7, A25.
   - Reported two-qubit resource counts are variational-layer counts (cost ZZ plus
     local XY mixer). Full-circuit counts stay null until a concrete W-state
     StatePrep decomposition is frozen; the variational count is never presented
     as a total-gate count (A18).
   - Exploratory `quantum_exploration` stage: depth p in {1,2,3,4,6} at 20
     optimizer evaluations per parameter, and QAOA angles fitted on training
     complexes transferred untrained to the hard set [R45-R47].

5. **Structure-level validation**
   - Solver assignments are reconstructed as side-chain conformations.
   - OpenMM [R19] constrained relaxation with the ff14SB protein force field [R15] evaluates whether discrete energy gains persist after continuous structural refinement.
   - Formal all-atom runs are strict fixed backbone (`loop_relax_iterations: 0`),
     validated before execution: N/CA/C/O coordinates cannot move (A18).
   - Structural metrics include Active side-chain RMSD, Fnat, interface RMSD,
     ligand RMSD, clash measures, and related trajectory metrics.

## Central research questions

1. How faithfully can the constrained rotamer-assignment problem be encoded as
   an auditable one-hot QUBO/Ising instance for gate-based quantum optimization?
2. Does feasibility-preserving XY-QAOA concentrate probability on the ground
   state beyond uniform feasible sampling (primary: log10 exact ground-state
   amplification at the preregistered size), and how does it compare with
   classical baselines under a declared abstract shot/query accounting
   convention with and without training shots [R37,R45] (descriptive)?
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

Each question is answered in `FINAL_RESEARCH_REPORT.md` section R, which gives
its evidence artifact, estimate, multiplicity family and verdict under the
pre-declared rule in `docs/RESULTS_CONTRACT.md` (A52).

## Interpretation limits

- No hardware quantum advantage or quantum speedup claim is made from
  simulator data. QAOA results are noiseless exact-subspace simulations;
  matched-output/matched-time and scaling results are interpreted only as
  algorithmic relative-performance evidence [R20]. Matched-time comparisons
  measure the classical cost of emulating QAOA and are descriptive.
- The 4/6/8/10-site slope is a finite-range, fixed-depth and fixed-optimizer-budget
  trend. Added sites also change residue identities and the QUBO landscape;
  exact enumeration remains feasible even at 10 sites (59,049 assignments).
  The slope is not an asymptotic complexity exponent. Quantum measurement
  shots and classical single-state energy queries are reported separately;
  the historical QTS99 conversion is descriptive abstract accounting, not
  matched hardware or total computational cost.
- Coarse QC multiplicity uses serial gatekeeping: only QAOA's primary-size
  ground-state amplification and its scaling slope are confirmatory at first;
  QAOA-vs-classical effects are confirmatory only after both are rejected
  [R41,R42]. Depth/budget and parameter-transfer analyses are exploratory and
  descriptive.
- Independent-cluster adequacy is checked at queue freeze, before any outcome
  (`--stop-after queue_freeze`), and EGNN training-seed variance is reported
  from development-only replicates.
- Cluster-level confirmatory tests are two-sided sign-flip tests, whose
  smallest attainable p value with G independent clusters is 2/2^G; G >= 6 is
  the first that can reach p < 0.05 [R43]. This arithmetic, not power, sets the
  preregistered minimum cluster counts.
- The default external-validation population is an antigen-fold holdout carved
  from the same audited snapshot, not an independent database. The manuscript
  must call it an antigen-fold holdout and report its component and graph
  counts with the result (A10, A13).
- Coarse antigen interaction scores are not binding free energies; contact
  number is a geometry baseline, not an affinity estimator.
- All-atom primary experiments are strict fixed-backbone: N/CA/C/O coordinates
  never move (`loop_relax_iterations: 0`). Formal recovery perturbs and
  reconstructs every defined Active side-chain chi. The coarse
  energy surrogate remains chi1-oriented. Amber14 calibration is a failed or
  successful training-only diagnostic under the current protocol; fitted
  coefficients are never applied to solver benchmarks.
- Smoke checks and legacy explicit `chi1_angles` overrides are engineering or
  ablation paths and are not the formal main protocol.
- Every protocol change after the original freeze is listed, with its reason
  and inspection status, in `docs/PROTOCOL_AMENDMENTS.md` (currently A1-A59).

## Scientific configuration

The formal pipeline separates **configurable experimental parameters** from
**fixed model-definition constants**.

`configs/full_experiment_config.yaml` is the single source of truth for the
frozen scientific protocol: stage toggles, the master seed, homology-isolation
thresholds, the graph protocol, the antigen-fold holdout, the QAOA protocol
(`quantum_protocol.primary`: depth 2, 90 evaluations, CVaR alpha 0.10, 4
restarts, 500 evaluation shots, 1000 output shots), calibration acceptance
gates, and the statistical plan. Validation and test results are never used to
retune it.

Fixed model-definition constants (residue tables, side-chain atom sets,
symmetry conventions) live in code and change only through a dated protocol
amendment.

## Training-only Amber calibration diagnostic

The coarse-to-Amber fit is **training-only** and is reported before any
validation/test benchmark. In the current `diagnostic` protocol its fitted
coefficients are never applied to a solver; failing the historical fit-quality
thresholds is retained as a negative diagnostic result. The input CSV contains:

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

A source complex missing observed protein heavy atoms is excluded before any
calibration energy row is accepted; provenance separately records discovered
complexes, input-quality exclusions, eligible attempts and generation
failures, and the fractions must close arithmetically (A21). The diagnostic
retains the historical minimum of 100 eligible complexes, at least 20
independent family groups and five grouped CV folds. The exclusion fraction
is recorded as a coverage descriptor rather than a pass/fail gate (A22).

The pipeline attempts the CSV and fit before the coarse benchmark. An absent
fit or one with inadequate predictive quality receives a `completed_with_failures`
diagnostic status and its coefficients remain unused. Provenance and result
integrity checks still stop the run if recorded artifacts are inconsistent.
The solver benchmark optimizes a coarse surrogate; the independent structural
experiment supplies atomistic outcomes. This protocol change and its inspected
training diagnostics are recorded in `docs/PROTOCOL_AMENDMENTS.md` A23.
The final report also computes within-complex rank correlations between the
same assignment's coarse and raw Amber scores in the training CSV. This is a
diagnostic of the unrelaxed training cases, not held-out evidence that lower
coarse energy produces better structures.

Input complex quality also includes the A24 conservative 1.0 Å minimum
distinct-residue protein heavy-atom distance screen. The audit counts rejected
complexes and records the closest offending pair before formal graph selection;
generated rotamer clashes are reported separately.

The formal rotamer model reads Dunbrack 2010 samples from the installed
PyRosetta/Rosetta database (`pyrosetta_dun10`, pinned to build 2026.29 in the
frozen config). It uses backbone phi/psi, rotamer probabilities, chi1..chiN means and
standard deviations. chi1 is expanded by configured standard-deviation offsets;
distal chi values retain their corresponding means. The all-atom model applies
complete chi1..chiN states and retains 3--6 states/site under the <=30-variable
budget. The coarse pseudo-atom model remains chi1-oriented. PyRosetta is a
separately installed, licensed dependency. Formal preflight checks its build,
dun10 option and a real rotamer sample. The prior `dunbrack2010` text-file
mode remains available for reproduction of earlier runs but is not selected
by the frozen formal config. See `docs/PROTOCOL_AMENDMENTS.md` A19; calibrations
and all downstream results from the text-file protocol cannot be reused.

## Independence, external validation, and structural baselines

Formal confirmation requires both layered sequence isolation and a frozen
PDB-to-family/structure cluster map. The same cluster map is used for
train/test exclusion, validation-queue eligibility, and cluster-level
statistics. Missing required cluster metadata fails closed.

Structure clusters come from an antigen-chain-only Foldseek search. Antibody
chains share the Ig fold and are removed first, or single linkage would join
almost every complex (A9). A chain pair scores
`mintmscore = min(qtmscore q->t, qtmscore t->q)`, the TM-score normalized by
the longer chain, so a short chain cannot bridge whole complexes through a
query-normalized hit; a PDB pair takes the maximum over its chain pairs and
the threshold stays 0.50 [R5,R27] (A11). The clustering universe contains only
source-verified, row-level QC-eligible formal VHH candidates from the same
preferred source used by graph admission (A16).

A layered component larger than one fold's share of the training pool is
pinned to training and never becomes the internal validation fold or the
antigen-fold holdout (A11).

The formal pipeline also has an external-validation stage. By default it
scores an **antigen-fold holdout** (`docs/PROTOCOL_AMENDMENTS.md` A10, A13): whole
layered-isolation components are carved out of the training split at
`queue_freeze`, before any training, and the frozen EGNN, uncalibrated coarse
model and primary solver protocol are applied to them without refitting. The
holdout takes `queue_freeze.antigen_fold_holdout.fold` — one index or several
(`"1,2"`); several are taken, lowest index first, when one fold does not reach
`min_components`, which is a rule fixed in advance rather than a search for the
fold holding the most components (A13). The claim it supports is that the
frozen pipeline still holds on antigen folds training never saw; it is a
cluster-level holdout of the same audited snapshot, not a separate database.
Setting `external_validation.external_vhh.graph_dir` and `source_structure_dir`
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
scientific inputs are unavailable. The preflight verifies the configured
rotamer source (the installed PyRosetta build, its active dun10 option and a
real rotamer sample; or the legacy text library when that mode is selected),
ANARCI with HMMER when formal IMGT numbering is required (A18), a Foldseek
executable when the run builds its own pair table, FASPR and its rotamer
binary, Phenix clashscore, and OpenMM GBN2 parameters when the declared
solvent sensitivity is enabled. When a genuinely external VHH set is
configured, its graph/raw-structure inputs are also required; under the
default antigen-fold holdout protocol those run-local inputs are created later
by queue_freeze and are therefore not required at preflight time.

With `independence_clustering.build_per_run: true` the Foldseek pair table is
built inside each formal run after its own data audit, and a resumed run
verifies and reuses its own frozen table. A prebuilt table can still be
supplied at the configured path.

External VHH independence is not accepted from a hand-written Boolean alone.
`audit_external_vhh_independence.py` compares every external graph against
the frozen training graphs using the same VHH/CDR-H3/antigen thresholds and
the same family/structure cluster map, then writes a per-target auditable
manifest. The external-validation stage verifies that manifest before running
the frozen model.

### Preparing external inputs

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

To score a **genuinely external** VHH set instead of the run-local holdout,
set `external_validation.external_vhh.graph_dir` and `source_structure_dir`,
and build that set with the two passes below, so its PDBs enter the clustering
universe:

```bash
./scripts/prepare_external_vhh.sh pass1 --sabdab-summary sabdab_summary_all.tsv --download
./scripts/prepare_external_vhh.sh foldseek --foldseek /path/to/foldseek
./scripts/deploy_launch.sh --stop-after queue_freeze
./scripts/prepare_external_vhh.sh pass2 --sabdab-summary sabdab_summary_all.tsv --run-dir <that run>
./scripts/deploy_launch.sh
```

`--released-after YYYY-MM-DD` adds a temporal holdout when PDB releases
postdate the training snapshot. Pass 2 refuses to run if the pair table changed
after pass 1, because external PDBs can link internal clusters. Formal SAbDab
and external-VHH candidate selection require ANARCI with IMGT numbering and
fail closed without it (A18).

The `foldseek` step searches antigen chains only, exactly as the in-run build
does. `--reuse-raw` rescores the previous search without running Foldseek
again; `--run-dir <run>` binds the table to that run's own frozen universe.

Each external graph is one VHH-antigen complex, as in the SNAC-DB per-VHH
complexes behind the training and hard test sets. Entries with several
nanobodies give one complex per PDB: the VHH with the largest passing
interface. Other antibody chains are never antigen, and entries with a VH/VL
chain are excluded. The final set (pass 2) keeps one representative per layered
homology group. Selection also requires a protein or peptide antigen,
resolution <= 3.0 A, and the audit's interface quality gates. It then groups
the complexes by the layered homology rule and reports whether `min_clusters`
independent groups exist.

## Repository layout

```text
configs/   full_experiment_config.yaml (frozen scientific protocol), server_config.yaml (infrastructure only)
docs/      METHODS_EVIDENCE.md (design-to-literature register), RESULTS_CONTRACT.md (required outputs),
           PROTOCOL_AMENDMENTS.md (dated post-freeze protocol changes),
           PIPELINE_WALKTHROUGH.md (stage-by-stage walkthrough of what each stage does and why)
scripts/   deploy_launch.sh, run_full_experiment.sh, formal_preflight.sh, check_status.sh,
           repair_openmm_cuda.sh, prepare_external_vhh.sh,
           diagnose_calibration_eligibility.py, diagnose_calibration_relaxation.py (read-only),
           discover_external_vhh.py (candidate discovery for manual curation)
src/nanoqc/
  pipeline/     run_full_experiment.py (entry point: Orchestrator core, resume, main),
                orchestrator_common.py (stage order, fingerprint list, StageResult),
                config_validation.py (frozen-protocol validator), run_records.py (manifest, results audit),
                stages_data.py (env_check .. queue_freeze), stages_training.py (egnn_train,
                energy_calibration, method_sensitivity), stages_quantum.py (qc_benchmark,
                quantum_exploration), stages_structure.py (structure_experiment, external_validation),
                stages_reporting.py (statistics, final_report), stage_contracts.py (result contracts),
                resolve_server_config.py
  data/         audit_all_datasets.py (structure audit CLI), audit_structures.py (assembly/structure reading),
                build_final_pyg_dataset.py (audited graphs + split),
                complex_extraction.py (atom extraction, antigen partner-chain selection),
                graph_build_parallel.py (spawned graph/homology workers),
                build_foldseek_pairs.py (antigen-chain-only pair table),
                build_independence_cluster_map.py, carve_holdout_clusters.py (antigen-fold holdout),
                audit_external_vhh_independence.py, sequence_identity.py, safe_graph_load.py,
                fetch_entry_resolution.py, convert_sabdab2_summary.py,
                select_external_vhh_candidates.py, build_external_vhh_graphs.py,
                prepare_external_vhh.py (external VHH set)
  model/        model_egnn_pruning.py (interface scoring, Active-site selection, checkpoint loading),
                train_egnn_pruning.py (leakage-controlled split, checkpoints, training CLI),
                egnn_graph_data.py (dataset, labels, loaders), egnn_split_records.py (split records,
                digests), egnn_metrics.py (AUC/F1 metrics, geometry baseline), egnn_training_loop.py
                (epoch, validation, DDP, checkpoint files),
                egnn_seed_sensitivity.py (development-only seed variance)
  qubo/         subgraph_to_qubo.py (entry point, re-exports, self-test CLI),
                coarse_qubo.py (InterfaceQUBOBuilder), allatom_qubo.py (AllAtomInterfaceQUBOBuilder),
                rotamer_library.py (PyRosetta dun10 / legacy library), qubo_types.py (config, states,
                QUBOResult), atomistic_structure.py (parsing, chi angles, evaluation), ising.py
  quantum/      instance.py (quantum instance contract), resource_estimation.py (logical resources)
  solvers/      qaoa_interface_sampler.py (XY-mixer QAOA: circuit, sampler class, CLI),
                qaoa_optimization.py (optimizer mixin), qaoa_sampling.py (enumeration, sampling,
                simulated annealing mixin), qaoa_results.py (result records, CVaR)
  experiments/  batch_benchmark_hard_set.py (benchmark CLI: dispatches to one module per mode),
                hard_set_evaluation.py (default mode), research_ablation.py (--research-ablation),
                calibration_fit.py (ridge calibration fit), benchmark_statistics.py (--paired-statistics),
                structure_benchmarks.py (all-atom modes), benchmark_common.py (fingerprint list),
                fit_qaoa_transfer_parameters.py (train-split QAOA angle transfer),
                run_real_complex_pilot.py (all-atom retrospective recovery, queue-freeze eligibility),
                real_complex_preparation.py (target structure preparation, shared with calibration),
                generate_energy_calibration_dataset.py, run_external_structure_baselines.py (FASPR / Phenix)
  structure/    evaluate_complex_metrics.py (complex-metrics CLI), complex_atoms.py (reading, Kabsch,
                contacts, Fnat), side_chain_metrics.py (Active side-chain RMSD, clashes, DockQ),
                structural_quality.py, residue_tables.py
  inference/    paired_statistics.py (incl. serial gatekeeping), analyze_quantum_scaling.py,
                analyze_quantum_exploration.py, analyze_structure_recovery.py (RQ5)
  reporting/    generate_final_research_report.py (report CLI, data/cost/limits sections),
                report_common.py (ReportContext, readers), report_sections_quantum.py,
                report_sections_structure.py, generate_figure1_pymol_script.py
  common/       repo_io.py (hashing, layout), seed_streams.py, prediction_contract.py,
                gpu_runtime.py / cpu_runtime.py (stage telemetry), stage_utilization.py (utilization summary),
                device_errors.py (device limits are never scientific exclusions)
tests/     regression suite run by formal_preflight.sh (`python -m pytest tests`)
```

Every stage runs as `python -m nanoqc.<package>.<module>` from the repository
root with `src/` on `PYTHONPATH`; the scripts in `scripts/` set this up. Run
manifests keep fingerprinting modules under their bare file names
(`code_sha256["subgraph_to_qubo.py"]`), resolved through
`nanoqc.common.repo_io.MODULE_LAYOUT`; every module under `src/nanoqc/` must be
registered there, which `tests/test_refactor_equivalence.py` enforces.

No source file exceeds 1000 lines. The former large files
(`run_full_experiment.py`, `subgraph_to_qubo.py`, `batch_benchmark_hard_set.py`,
`qaoa_interface_sampler.py`, `train_egnn_pruning.py`, `evaluate_complex_metrics.py`,
`generate_final_research_report.py`, `build_final_pyg_dataset.py`,
`audit_all_datasets.py`) are now entry points over focused modules; import a
function from the module that defines it. Every split-out module is in the code
fingerprints (`ORCHESTRATED_SCRIPTS`, the benchmark's `SHARED_HELPER_MODULES`),
which tests check against the actual imports. Functions that read a threshold
the command line rebinds at start-up (`global` in `main`) stay in their entry
module, so a moved function never reads a stale copy.
Read-only diagnostics live in `scripts/` precisely so they stay outside that
formal module layout.

## Literature basis

The authoritative design-to-literature mapping is maintained in
`docs/METHODS_EVIDENCE.md`. Reference labels [R1]-[R49] in this README refer to
that file. The register explicitly separates direct literature support from
literature-informed preregistration and study-specific preregistration so that
exact numerical choices are never misrepresented as published standards.

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
export QP_FOLDSEEK=/path/to/foldseek
export QP_PHENIX_CLASHSCORE=/path/to/phenix.clashscore
./scripts/deploy_launch.sh
```

In auto mode the resolver chooses DDP ranks from available GPUs while preserving the configured target global batch size, derives CPU worker counts from the logical CPUs the process may use (80 % of them for independent CPU pools, A34), selects the OpenMM platform from available hardware, and records every resolved value in the run configuration/provenance. Scientific thresholds, QAOA protocol values, data-split rules, endpoints, and statistical choices are never hardware-auto-tuned.

On the two-T4 server, EGNN training uses two DDP ranks. After sequential queue selection, independent structural targets run in separate spawned processes on CUDA devices 0 and 1. Training-only Amber diagnostic complexes are sharded across those devices and merged in frozen manifest order; their per-complex random streams are independent of worker scheduling. Data audit, Foldseek, grouped regression/statistics, and the formal legal-subspace QAOA simulator remain CPU tasks. This execution and calibration sampling change requires a **new formal run**, not resume of an older frozen directory (A26).

The CPU stages also use bounded parallelism: data audit, Foldseek antigen-input preparation and external VHH sequence checks use spawned processes; the Foldseek search uses its own threads; the solver benchmark uses a process pool. Sensitivity cases, exploratory QAOA depths and paired-statistics budget modes run in parallel while each depth's training fit, parameter transfer and hard-set evaluation stay ordered. Development structural targets can occupy both T4s; frozen validation target selection remains sequential (A27).

Every orchestrated stage records CPU and GPU samples in `logs/<stage>.cpu.csv` and `logs/<stage>.gpu.csv`. `python -m nanoqc.common.stage_utilization <run_dir>` (also printed by `scripts/check_status.sh`) lists each stage's mean CPU and GPU utilization against the 80 % target; see `docs/PIPELINE_GPU_UTILIZATION.md` for which stages are expected to reach it (A34).

Graph construction, its homology screens and the development-only EGNN seed replicates run as independent processes rather than one after another, and candidate preparation uses eight workers per GPU. A CUDA out-of-memory or unavailable-device error aborts its stage instead of excluding a complex, so raising concurrency cannot move a scheduling artifact into the study's denominator (A35).

## One-click formal experiment

A formal experiment is launched with one command:

```bash
./scripts/deploy_launch.sh
```

Useful variants, all passed straight through to `run_full_experiment.sh`:

```bash
./scripts/deploy_launch.sh --stop-after queue_freeze        # check cluster adequacy before training
./scripts/deploy_launch.sh --resume=<run directory name>    # same code/evidence/config only; otherwise refused
./scripts/deploy_launch.sh --resume=<run> --force-restage egnn_train
./scripts/deploy_launch.sh --only <stage>                   # prerequisites must already be complete
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

## Amended-lineage recovery

The one-time recovery commands used for the FP16 EGNN (A20) and calibration
(A21, A22) amended lineages applied only to runs launched before the module
splits and have been removed. They remain in the git history (last version at
commit `cbec29a`). Any new protocol change starts a new formal run
(`./scripts/deploy_launch.sh`) rather than amending an old directory.

Two read-only diagnostics change nothing in a run:
`scripts/diagnose_calibration_eligibility.py` inspects calibration exclusions,
and `scripts/diagnose_calibration_relaxation.py` compares raw against
fixed-backbone-relaxed Amber energies from rows an earlier calibration already
generated.

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
