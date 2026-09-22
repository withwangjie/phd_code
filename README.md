# phd_code

## Research scope

This repository benchmarks constrained quantum-classical optimization for
nanobody (VHH)-antigen interface side-chain reconstruction under a known
complex pose/backbone. It is a retrospective side-chain recovery benchmark,
not blind docking, de novo complex prediction, or quantitative binding-affinity
prediction.

## Formal research pipeline

1. **Leakage-controlled data construction**
   - Interface labels: cross-partner heavy-atom contact < 5 A, following antibody-antigen/CAPRI-style contact definitions [R3,R4].
   - Graph edges: intra-chain CA radius < 8 A plus fixed cross-partner KNN. An 8 A C-alpha residue-graph cutoff has direct protein-GNN precedent [R22]; the cross-partner KNN degree k=3 remains a study-specific leakage-control choice rather than a literature-optimal constant.
   - EGNN train/validation split: layered connected components. Complexes are
     joined if VHH full-chain identity >=80% [R24], CDR-H3 identity >=50% [R23], or
     antigen identity >=30% with >=70% minimum length coverage [R25]; no random 90/10 split.

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
     interaction scoring (coarse model) or Amber14 single-candidate energy
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

5. **Structure-level validation**
   - Solver assignments are reconstructed as side-chain conformations.
   - OpenMM [R19] constrained relaxation with the ff14SB protein force field [R15] evaluates whether discrete energy gains persist after continuous structural refinement.
   - Structural metrics include Active side-chain RMSD, Fnat, interface RMSD,
     ligand RMSD, clash measures, and related trajectory metrics.

## Central research questions

- Does EGNN retain useful interface-selection ability after homology isolation
  and topology-leakage control?
- Does adaptive 3--6-state modeling improve coverage over fixed low-state
  discretization?
- How do XY-QAOA and classical solvers differ in energy gap, low-energy
  coverage, sampling diversity, and budget efficiency?
- Does CVaR emphasize useful low-energy conformations better than mean-energy
  optimization?
- Most importantly, do solver-level discrete energy gains propagate to
  all-atom structural improvements?

## Interpretation limits

- Coarse antigen interaction scores are not binding free energies.
- Contact number is a geometry baseline, not an affinity estimator.
- Current all-atom experiments remain native-backbone-conditioned, but formal
  recovery now perturbs and reconstructs every defined Active side-chain chi.
  The coarse energy surrogate remains chi1-oriented and is explicitly
  calibrated against Amber14 on training complexes only.
- No hardware quantum advantage or quantum speedup claim is made from simulator data. Matched-output/matched-time and scaling results are interpreted only as quantum-classical algorithmic relative-performance evidence, consistent with modern quantum-optimization benchmarking guidance [R20].
- Smoke checks and legacy explicit `chi1_angles` overrides are engineering or
  ablation paths and are not the formal main protocol.


## Scientific configuration

The formal pipeline separates **configurable experimental parameters** from
**fixed model-definition constants**.

Configurable in `full_experiment_config.yaml`:

- Dataset/graph protocol: sequence identity threshold, heavy-atom interface
  label cutoff, intra-chain CA radius, cross-partner KNN degree, and minimum
  interface-residue count.
- Site-selection protocol: Active-site count, antigen-guidance weight,
  antigen-proximity decay length, contact-baseline CA cutoff, and environment
  radius.
- Coarse interaction model: non-bonded cutoff, soft-core delta, hard-core
  fraction/penalty, LJ caps, Coulomb cap, dielectric model parameters, and kT.
- Solver budgets: QAOA depth, objectives, restarts, finite shots, evaluation
  budgets, SA/greedy passes, output budgets, and low-energy window.
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

The formal pipeline also has an external-validation stage. It requires an
independently certified graph-v1.6 VHH dataset and runs the frozen EGNN,
calibrated coarse model, and primary solver protocol without refitting.
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
the external VHH graph set, FASPR, Phenix clashscore, and OpenMM GBN2
parameters when the declared solvent sensitivity is enabled.

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

The primary all-atom protocol uses vacuum/NoCutoff Amber14 packing energy.
A pre-declared GBN2 sensitivity run is performed on development targets only;
the validation queue never chooses its solvent model after inspecting results.

## Main entry points

- `run_full_experiment.py`: end-to-end orchestrator.
- `build_final_pyg_dataset.py`: audited graph construction.
- `build_independence_cluster_map.py`: frozen family/structure cluster-map construction.
- `audit_external_vhh_independence.py`: external VHH train-overlap audit.
- `generate_energy_calibration_dataset.py`: training-only multi-chi coarse/Amber calibration rows.
- `train_egnn_pruning.py`: leakage-controlled EGNN training.
- `model_egnn_pruning.py`: interface scoring and Active-site selection.
- `subgraph_to_qubo.py`: coarse/all-atom adaptive side-chain QUBO builders.
- `qaoa_interface_sampler.py`: XY-mixer QAOA.
- `batch_benchmark_hard_set.py`: matched quantum/classical benchmark.
- `run_real_complex_pilot.py`: all-atom retrospective recovery experiment.
- `evaluate_complex_metrics.py`: structure-level evaluation.
- `analyze_structure_recovery.py`: pre-registered structural endpoint and RQ5 inference.
- `run_external_structure_baselines.py`: FASPR and Phenix clashscore baselines.


## Literature basis

The authoritative design-to-literature mapping is maintained in `METHODS_EVIDENCE.md`. Reference labels [R1]–[R21] in this README refer to that file. The register explicitly separates direct literature support from literature-informed preregistration and study-specific preregistration so that exact numerical choices are never misrepresented as published standards.


## Portable server runtime configuration

Scientific protocol and server infrastructure are intentionally separated.

- `full_experiment_config.yaml` contains the frozen scientific protocol.
- `server_config.yaml` contains server paths, resource policy, external-tool locations, and operational resource gates.
- `resolve_server_config.py` detects CPU/GPU/RAM and resolves paths/tools before formal preflight, then writes `.runtime/resolved_runtime_config.yaml` and `.runtime/server_resolution.json`.
- `run_full_experiment.sh` uses the same resolved runtime config for both preflight and the formal run.

Portable overrides can be supplied without editing the scientific protocol:

```bash
export QP_VENV=/path/to/.venv
export QP_DATA_ROOT=/path/to/data
export QP_RUN_ROOT=/path/to/runs
export QP_FASPR=/path/to/FASPR
export QP_PHENIX_CLASHSCORE=/path/to/phenix.clashscore
./deploy_launch.sh
```

In auto mode the resolver chooses DDP ranks from available GPUs while preserving the configured target global batch size, derives CPU worker counts from available physical cores, selects the OpenMM platform from available hardware, and records every resolved value in the run configuration/provenance. Scientific thresholds, QAOA protocol values, data-split rules, endpoints, and statistical choices are never hardware-auto-tuned.
