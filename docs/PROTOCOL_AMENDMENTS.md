# Protocol amendments

Preregistration only protects inference if every later change to the frozen
protocol is disclosed with its timing and reason, so readers can tell tested
predictions from post-hoc choices [R40]. This file is that record. Each entry
states **what** changed, **why**, the **literature basis**, the **affected
results**, and whether validation/test **results had been inspected** before
the change.

> **Inspection status is not yet confirmed.** The field "Results inspected
> before this amendment" must be completed by the investigator. If any formal
> run had produced validation/test (SNAC hard, validation queue, external VHH)
> results that were looked at before an amendment below, that amendment must
> be reported as a post-hoc deviation in the manuscript, and the pre-amendment
> analysis must also be reported.

All amendments below were made on 2026-09-24 on branch
`claude/blissful-heisenberg-3n2jf3` (base `reorg` @ `e8515ed`). Every one of
them changes code hashes, so no earlier run directory can be resumed; a fresh
formal run is required.

## Summary

| ID | Change | Affected stages | Inspection status |
|---|---|---|---|
| A1 | Resource-normalized time-to-solution (`log10_qts99`) replaces best-of-N gap (secondary since A7) | qc_benchmark, statistics, external_validation, final_report | to be completed |
| A2 | Pre-specified non-estimable RQ5 outcomes | statistics (structural), final_report | to be completed |
| A3 | Code-level fixes preceding A1 (seeds, symmetry, full-complex environment, biological assembly) | dataset through final_report | to be completed |
| A4 | Serial gatekeeping replaces one flat Holm family | statistics (QC), final_report | to be completed |
| A5 | Outcome-free cluster adequacy gate at queue freeze | queue_freeze (gate only) | not applicable |
| A6 | Development-only EGNN training-seed sensitivity | egnn_train (descriptive) | not applicable |
| A7 | Quantum-intrinsic primary endpoint; `quantum_exploration` stage | qc_benchmark, statistics, quantum_exploration, final_report | to be completed |
| A8 | Construction rules for the external VHH set | external_validation (input set) | not applicable (no external set existed) |
| A9 | Foldseek clustering input: antigen chains only | queue_freeze (cluster map, split), all cluster-level statistics | not applicable (no pair table existed) |

## A1. Primary coarse solver endpoint: resource-normalized time-to-solution

> **Partly superseded by A7:** `log10_qts99` is now a *secondary* QAOA-vs-classical
> effect, and the scaling response is QAOA exact ground-state amplification.

- **Before:** primary QAOA-vs-SA contrast on best-of-N energy gap at 1000
  matched outputs; scaling response = QAOA-minus-SA gap.
- **After:** primary metric `log10_qts99`, the log10 number of measurement
  shots / single-state energy queries needed to observe a ground state at
  least once with 99% probability, R99 = F + c ln(0.01)/ln(1-p) [R37]. QAOA is
  charged its optimization shots (F) plus one shot per output (c = 1); SA,
  greedy and uniform are charged their energy queries per read (c >= 1). The
  per-sample success probability p uses the Jeffreys estimate (k+1/2)/(n+1)
  [R38], finite for k = 0 or k = n. The scaling slope uses the same response.
  Best-of-N gap/hit remain secondary effects.
- **Why:** a simulation with the frozen solver settings showed SA reaching the
  ground state in 100% of instances at both 6 and 10 sites (gap identically 0),
  and uniform sampling beating QAOA on best-of-1000 hit at 6 sites (75% vs
  62%). The primary contrast was therefore at a ceiling and its sign fixed in
  advance. "Matched outputs" also gave SA about 13x more energy queries
  (1000 x 601) than QAOA's total shots (90 x 500 + 1000). Time-to-solution with
  explicit resource accounting is the standard remedy [R37,R20,R29].
- **Limitation:** with zero observed hits p is an estimate (Jeffreys); at the
  primary 1000 outputs its bias is small, at small output budgets it is
  optimistic. The time-matched analysis measures classical *emulation* cost of
  QAOA and is descriptive only.
- **Affected results:** qc_benchmark, statistics (QC family and scaling),
  external VHH statistics, final report.
- **Results inspected before this amendment:** _to be completed by the investigator_.

## A2. Pre-specified handling of a non-estimable RQ5 association

- **Before:** a constant cluster-level energy or RMSD difference made Spearman
  rho undefined and failed the statistics stage.
- **After:** such cases are named outcomes
  (`not_estimable_identical_discrete_energies`,
  `not_estimable_constant_energy_difference`,
  `not_estimable_constant_rmsd_difference`), reported descriptively with the
  mean differences, and removed from the confirmatory Holm family. The cluster
  minimum still applies. An undefined rho without such a declared status still
  fails. The primary structural contrast needs no rule: all-zero paired
  differences give a sign-flip p of 1.
- **Why:** both solvers finding the same all-atom optimum (<= 46,656 states)
  is a plausible, scientifically meaningful outcome; statistical analysis
  plans should pre-specify how undefined estimates are handled [R39].
- **Affected results:** statistics (structural), final report.
- **Results inspected before this amendment:** _to be completed by the investigator_.

## A3. Code-level amendments preceding A1 (2026-09-24)

These commits were made first, before A1, and are numbered A3 only because
they were logged later.

| Commit | Change | Scientific effect |
|---|---|---|
| `a851f34` | Quantum modules in code hashes; child seeds for external statistics and calibration; unified ground-energy tolerance (1e-9) | Different random draws in calibration/external statistics; `hit` stricter |
| `d982002` | EGNN train/validation split crash fixed | Formal EGNN training could not run before |
| `34334ee` | VAL/LEU methyls no longer treated as symmetric; dataset checks survive `python -O` | Side-chain RMSD and chi recovery at VAL/LEU sites (primary structural endpoint) |
| `5373dbe` | Identity fields named by region (full chain vs CDR-H3 loop) | None (naming only) |
| `0e824aa` | Coarse antigen term uses the full complex antigen [R31,R32] | Coarse energies, candidate screening, calibration |
| `769d80e` | Coarse fixed-VHH term uses the full complex; phi/psi undefined at chain breaks; graph v1.7 | Coarse energies, Active-site eligibility, rotamer candidates |
| `e5e2e88` | Raw-PDB subsets built from the biological assembly with SAbDab antigen-chain rule [R33-R36]; graph v1.8 | Training set composition, interface labels, EGNN, calibration |

**Results inspected before these amendments:** _to be completed by the investigator_.

## A4. Serial gatekeeping replaces one flat Holm family

- **Before:** the primary QC contrast, the primary scaling slope and all 18+
  exploratory matched-output effects were Holm-adjusted as one family, which
  diluted the two preregistered confirmatory tests.
- **After:** serial gatekeeping with Holm inside each family [R41,R42]. The
  primary family {primary QC contrast (`primary_qc_baseline`/
  `primary_qc_metric`, matched outputs), primary scaling slope} is
  Holm-adjusted alone at full alpha. Every other matched-output effect is
  secondary: its adjusted p is max(largest primary adjusted p, Holm-adjusted p
  within the secondary family), so it can be claimed only after both primary
  hypotheses are rejected; strong FWER control holds across both families.
  The per-report all-effects Holm column is kept as descriptive. The
  structural two-test Holm family (primary structural endpoint, RQ5) is
  unchanged.
- **Affected results:** statistics (QC family), final report.
- **Results inspected before this amendment:** _to be completed by the investigator_.

## A5. Outcome-free cluster adequacy gate at queue freeze

- **Before:** too few independent clusters surfaced only in the statistics
  stage, after all training, benchmark and structure computation.
- **After:** `queue_freeze` counts independent family/structure clusters among
  `test_snac_hard` graphs and the frozen validation queue, writes
  `independence/cluster_adequacy.json`, and fails if any preregistered minimum
  (`min_qc_clusters`, `min_scaling_clusters`, `min_primary_clusters`,
  `min_rq5_clusters`) cannot be met. Cluster-level inference is unreliable
  with few clusters [R43]. The check uses only data composition, never an
  outcome, so it is compatible with preregistration.
  `run_full_experiment.py --stop-after queue_freeze` runs just this far.
- **Affected results:** none (gate only); earlier failure when inadequate.
- **Results inspected before this amendment:** not applicable (no outcome involved).

## A6. Development-only EGNN training-seed sensitivity

- **Before:** Active-site selection depended on a single EGNN training run;
  seed variance (initialization, data order, DDP partitions, AMP/TF32
  non-determinism) was not assessed.
- **After:** `egnn_train.seed_replicates` (default 2) extra models are trained
  on the same split with derived seeds. `egnn_seed_sensitivity.py` reports
  the spread of best internal-validation ROC-AUC and the Jaccard stability of
  the formal Active-site selection on the internal validation fold, following
  the recommendation to account for seed variance in learned benchmarks
  [R44]. Only training-split data are used; the primary checkpoint stays the
  only formal model and replicates are never used for model selection.
- **Affected results:** egnn_train (additional descriptive outputs; extra compute).
- **Results inspected before this amendment:** not applicable (development data only).

## A7. Research aim is exploratory quantum computing: quantum-intrinsic primary endpoint

- **Decision (investigator, 2026-09-24):** the study's main purpose is to
  explore the quantum algorithm itself, not to show it beats classical
  solvers. At <= 10 sites a head-to-head against SA is decided in advance
  (SA reaches the ground state in 100% of simulated instances), and the A1
  endpoint was dominated by QAOA training shots (log10 QTS99 ~ log10 45,000
  at every depth in a simulation).
- **Primary (confirmatory) family, gatekept per A4:**
  1. mean log10 *exact* ground-state amplification of per-instance-trained
     QAOA over uniform feasible sampling at the preregistered primary size
     (6 sites), two-sided cluster sign-flip test against 0;
  2. the within-PDB slope of that amplification versus log10 feasible
     configuration count (scaling).
  Exact subspace probabilities are a noiseless-simulator property of the
  algorithm; the shot-based Jeffreys amplification is reported alongside.
- **Secondary:** every QAOA-vs-classical matched-output effect, including
  the A1 `log10_qts99` (now secondary) and a new execution-only
  `log10_qts99_execution` that excludes training shots, separating training
  from execution cost as in Shaydulin et al. [R45].
- **Exploratory, descriptive (new `quantum_exploration` stage):**
  (a) depth p in {1,2,3,4,6} with optimizer budget 20 evaluations per
  parameter, separating depth from optimizer budget [R7,R45];
  (b) parameter transfer: component-wise median of optimized,
  scale-normalised angles fitted on training graphs only, applied without
  per-instance training to the hard set [R46,R47]. Cluster-bootstrap
  intervals, no hypothesis tests. The supported depth range is now 1..12.
- **Affected results:** qc_benchmark (new row fields and transfer rows),
  statistics (primary family), new quantum_exploration stage, final report.
- **Results inspected before this amendment:** _to be completed by the investigator_.

## A8. Construction rules for the external VHH set

- **Before:** the protocol required an external VHH graph set but defined
  neither how its complexes are selected nor how graphs are built. No
  external set existed.
- **After:** `select_external_vhh_candidates.py` and
  `build_external_vhh_graphs.py` define it.
  - **Source:** SAbDab entries released after a declared cutoff and absent
    from the study's audited PDB universe.
  - **Complex:** one VHH-antigen complex per entry, matching the SNAC-DB
    per-VHH complexes that supply training and the hard test set [R34]. When
    an entry has several nanobodies, the VHH with the largest passing
    interface is chosen; this depends only on structure. Other antibody chains
    are never antigen, and entries with a VH/VL chain are excluded (as in the
    training audit).
  - **Graphs:** the graph-v1.8 definition (biological assembly, 7.5 A antigen
    rule [R33], 5 A labels) and the audit's quality gates.
  - **CDR-H3:** IMGT 105-117, like the training SNAC annotation. ANARCI is
    used when installed; otherwise CDR-H3 comes from the IMGT 104/118 anchor
    motifs, and its agreement with the training annotation is reported.
  - **Adequacy:** the number of independent groups is checked against
    `min_clusters` before any run.
  - The formal external audit now verifies assembly-built graphs against the
    rebuilt assembly.
- **Why:** the external stage tests the frozen pipeline on independent
  complexes, so each external graph must be the same kind of object as the
  training and hard test graphs. The `sabdab_vhh` rule (one VHH in the entry)
  concerned an unambiguous anchor from annotations. SAbDab annotates each VHH
  chain, so that rule is not needed here, and it would remove most recent
  complexes.
- **Affected results:** external_validation.
- **Results inspected before this amendment:** not applicable (no external set or result existed).

## A9. Foldseek clustering input: antigen chains only

- **Before:** the protocol required a frozen Foldseek pair table
  (`qtmscore`, TM-score >= 0.50, single linkage), but did not say which
  chains are searched. No pair table had been built.
- **After:** `build_foldseek_pairs.py` searches only non-antibody protein
  chains of at least 20 residues from every PDB in the clustering universe,
  all-versus-all and exhaustively.
  - **Antibody chains** are recognised from SNAC/SAbDab annotation (the
    audit's sequence-match rule), SAbDab chain IDs for external entries, or
    an Ig V-domain detector: ANARCI when installed, otherwise the conserved
    C23/W41/C104 anchors plus the J-region [WF]GxG motif. The method is
    recorded for every chain.
  - **DB5.5** uses bound files only.
  - **A PDB with no antigen chain** gets an explicit self row and is listed
    as `self_only`.
  - **A searched PDB with no Foldseek hit** stops the build unless the
    investigator explicitly allows it.
- **Why:** structure clusters are meant to separate antigen families. VHH,
  VH/VL and TCR variable domains all share the immunoglobulin fold with
  TM-score well above 0.5, so including them would put almost every complex
  into one component. Leakage on the antibody side is controlled separately
  by the VHH full-chain (0.80) and CDR-H3 (0.50) identity thresholds [R23,R24].
- **Limitations:**
  - The motif detector can miss unusual V domains, which would then be kept
    as antigen. This is conservative: extra edges, fewer clusters.
  - Peptide antigens have no structure family; they remain covered by the
    antigen sequence threshold.
- **Affected results:** cluster map, train/test isolation, cluster adequacy,
  every cluster-level test.
- **Results inspected before this amendment:** not applicable (no pair table or result existed).

