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

A1-A15 were made on 2026-09-24; A16 was added on 2026-09-25 on branch
`claude/blissful-heisenberg-3n2jf3`. Every amendment changes code hashes, so
no earlier run directory can be resumed; a fresh formal run is required.

## Summary

| ID | Change | Affected stages | Inspection status |
|---|---|---|---|
| A1 | Abstract time-to-solution (`log10_qts99`) replaces best-of-N gap (secondary after A7, descriptive after A25) | qc_benchmark, statistics, external_validation, final_report | to be completed |
| A2 | Pre-specified non-estimable RQ5 outcomes | statistics (structural), final_report | to be completed |
| A3 | Code-level fixes preceding A1 (seeds, symmetry, full-complex environment, biological assembly) | dataset through final_report | to be completed |
| A4 | Serial gatekeeping replaces one flat Holm family | statistics (QC), final_report | to be completed |
| A5 | Outcome-free cluster adequacy gate at queue freeze | queue_freeze (gate only) | not applicable |
| A6 | Development-only EGNN training-seed sensitivity | egnn_train (descriptive) | not applicable |
| A7 | Quantum-intrinsic primary endpoint; `quantum_exploration` stage | qc_benchmark, statistics, quantum_exploration, final_report | to be completed |
| A8 | Construction rules for the external VHH set (superseded as the default by A10) | external_validation (input set) | not applicable (no external set existed) |
| A9 | Foldseek clustering input: antigen chains only | queue_freeze (cluster map, split), all cluster-level statistics | not applicable (no pair table existed) |
| A10 | External validation becomes an antigen-fold holdout | queue_freeze (split), egnn_train, energy_calibration, external_validation | not applicable (no external set or result existed) |
| A11 | Symmetric Foldseek score; oversized components always train | queue_freeze (cluster map, split), egnn_train, external_validation | not applicable (data composition only; no split or result existed) |
| A12 | Entry resolution from curation metadata; pipeline staging not audited | audit, dataset (admission) | not applicable (audit fields only; no graph or result existed) |
| A13 | The antigen-fold holdout takes several folds when one is too small | queue_freeze (split), egnn_train, external_validation | not applicable (component counts only; no result existed) |
| A14 | A downloaded RCSB assembly file is the assembly, not an unannotated ASU | audit, dataset (admission), everything downstream | not applicable (admission counts only; no result existed) |
| A15 | Entry resolution fetched from RCSB for files that carry none | audit, dataset (admission), everything downstream | not applicable (admission counts only; no result existed) |
| A16 | Formal VHH sources and cross-source PDB precedence | audit, Foldseek universe, dataset admission, EGNN/calibration and everything downstream | to be completed |
| A17 | SNAC-only primary antigen-fold targets; SAbDab quarantine; exact source_id binding | queue_freeze, external_validation, final_report | to be completed |
| A25 | Training-only same-assignment rank diagnostic and explicit finite-range/resource interpretation | final_report, scaling analysis prose | to be completed |
| A26 | Independent calibration complexes and frozen structural targets run on separate CUDA workers | energy_calibration, structure_experiment | to be completed |
| A27 | Independent audit entries, sensitivity cases, external targets, exploratory depths, development structures and paired-statistics modes run concurrently | data_audit, method_sensitivity, quantum_exploration, structure_experiment, external_validation, statistics | to be completed |

## A1. Primary coarse solver endpoint: resource-normalized time-to-solution

> **Superseded by A7 and A25:** the scaling response is QAOA exact ground-state
> amplification; `log10_qts99` remains an abstract descriptive diagnostic.

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
- **Secondary under A7:** QAOA-vs-classical matched-output effects. The A1
  `log10_qts99` and execution-only `log10_qts99_execution` separated training
  from execution cost as in Shaydulin et al. [R45]; A25 later made these
  abstract shot/query comparisons descriptive only.
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

> **Superseded as the default by A10:** SAbDab cannot supply an external set
> with enough independent clusters, so the stage scores an antigen-fold
> holdout instead. Everything below still governs a genuinely external set
> whenever one is configured.

- **Before:** the protocol required an external VHH graph set but defined
  neither how its complexes are selected nor how graphs are built. No
  external set existed.
- **After:** `select_external_vhh_candidates.py` and
  `build_external_vhh_graphs.py` define it.
  - **Source:** SAbDab entries absent from the study's audited PDB universe.
    A release-date cutoff is optional and applies on top: it is available
    when PDB releases postdate the training snapshot, and omitted otherwise.
    Independence never rests on the date. It rests on excluding the study's
    own PDB IDs and on the layered sequence and structure-cluster checks,
    which the formal audit re-certifies each run. The manifest records
    `holdout` as `temporal_and_homology` or `homology`, and the manuscript
    must describe the external set accordingly rather than as a time split.
  - **Complex:** one VHH-antigen complex per entry, matching the SNAC-DB
    per-VHH complexes that supply training and the hard test set [R34]. When
    an entry has several nanobodies, the VHH with the largest passing
    interface is chosen; this depends only on structure. Other antibody chains
    are never antigen, and entries with a VH/VL chain are excluded (as in the
    training audit).
  - **Antigen eligibility:** an entry qualifies when at least one antigen
    chain is of SAbDab type protein or peptide. `antigen_type` is the
    deduplicated set of types over the antigen chains, so `protein | sugar`,
    `ion | protein` and the like denote a polypeptide antigen accompanied by
    another entity and are kept; `ion`, `hapten | ion` and `carbohydrate`
    have no polypeptide antigen and are rejected. Ions, glycans and ligands
    are never encoded in a graph, and SNAC-DB [R34] curates the protein
    complex the same way, so requiring their absence would make the external
    set stricter than the training set it validates.
  - **Graphs:** the graph-v1.8 definition (biological assembly, 7.5 A antigen
    rule [R33], 5 A labels) and the audit's quality gates.
  - **CDR-H3:** IMGT 105-117, like the training SNAC annotation. ANARCI is
    used when installed; otherwise CDR-H3 comes from the IMGT 104/118 anchor
    motifs, and its agreement with the training annotation is reported.
  - **Redundancy:** the final set keeps one representative per layered
    homology group (the largest interface, then PDB ID), like the hard-set
    de-redundancy. Complexes sharing a VHH are therefore not counted as
    independent when Foldseek separates their antigens (e.g. peptides).
  - **Adequacy:** the number of independent groups is checked against
    `min_clusters` before any run; with the A10 holdout the same minimum is
    counted at `queue_freeze` by `cluster_adequacy`.
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
  - **Annotated antigens:** chains annotated as antigen (SNAC `Chain_Ag`,
    SAbDab antigen chains) are always kept, so the detector cannot remove an
    Ig-superfamily antigen.
  - **Search options:** exhaustive search with E <= 10, which bounds the
    output instead of writing all N^2 pairs.
  - **A PDB with no antigen chain** gets an explicit self row and is listed
    as `self_only`. This also covers files the audit could not parse, which
    never become graphs.
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

## A10. External validation becomes an antigen-fold holdout

- **Before:** the protocol required an independently sourced external VHH
  graph set (A8), with at least 10 independent clusters.
- **Measured:** SAbDab cannot supply one. Of its 2429 single-domain entries,
  2424 are already in this study's audited PDB universe; 5 remain, 1 fails the
  structure gates, and the surviving 4 form **2** independent groups. The
  study's own training data is drawn from the same SAbDab single-domain
  snapshot, so the overlap is structural, not incidental. AVIDbase covers 92%
  of the SAbDab VHH-antigen complexes [R35] and cannot close the gap, and no
  PDB release postdates a training snapshot downloaded now, so a temporal
  holdout is empty as well (A8).
- **Why 2 clusters cannot be used:** the confirmatory tests are two-sided
  cluster sign-flip tests. With G clusters the smallest attainable p value is
  2/2^G, so G=2 gives p >= 0.5 and G=6 is the first that can reach p < 0.05.
  This is arithmetic, not low power: at 2 clusters no effect of any size can
  be claimed. Cluster-robust inference with few clusters is unreliable in
  general [R43], and sign-flip/wild-bootstrap procedures are the accepted
  remedy from roughly 6 clusters upward.
- **After:** the stage scores an **antigen-fold holdout** carved from the
  training split at `queue_freeze`, before any training or outcome.
  - Complexes are grouped by the study's own layered isolation relation (VHH
    full-chain identity, CDR-H3 loop identity, antigen full-chain identity
    with coverage, shared frozen family/structure cluster) and **whole
    components** are held out, so no holdout complex is homologous to any
    training complex under any criterion.
  - Which components go is decided by the deterministic name-hash fold that
    already assigns the internal validation fold, on a different fold index.
    It depends only on file names, never on an outcome. Removing whole
    components leaves the remaining components and their folds unchanged, so
    the internal validation split is exactly what it would have been.
  - Graphs move to `graphs/holdout` and the manifest is rewritten, so training
    and calibration read `graphs/train` and cannot reach the holdout. One
    audited raw structure per holdout PDB is copied beside them, so the
    unchanged independence audit binds every holdout graph to a raw file.
  - `min_components` (10) is preregistered and fails `queue_freeze` when unmet.
  - A genuinely external directory pair can still be configured; the audit and
    every downstream check are identical.
- **Scientific claim, and its limit:** the result supports "the frozen
  pipeline still holds on antigen folds it never saw in training". It is a
  cluster-level holdout of the kind current antibody benchmarks use, taken
  from the same audited PDB snapshot, and the manuscript must call it an
  antigen-fold holdout, never an external dataset. It does not demonstrate
  robustness to another curation pipeline, another structure-determination
  era, or another database's assembly conventions.
- **Affected results:** queue_freeze (split), egnn_train and
  energy_calibration (smaller training pool), external_validation.
- **Results inspected before this amendment:** not applicable (no external set
  or result existed; the counts above are data composition, not outcomes).

## A11. Symmetric Foldseek score; oversized components always train

- **Before:** A9 joined two PDBs when any antigen-chain pair had Foldseek
  `qtmscore >= 0.50`, a TM-score normalized by the query chain alone, under
  single linkage. A10 held out whole layered components by name hash.
- **Measured (before any split or outcome):** on the 2852-PDB study universe
  the query-normalized rule gave 345 clusters, the largest holding **2291
  (80.3%)**. Its hub PDBs (8vhf, 7wcm, 8wst, 9n29, ...) each linked to ~1800
  others and shared antigen chains of only 55-58 residues (e.g. G-protein
  gamma subunits of the Nb35-stabilised GPCR complexes): a short helix
  normalized by its own length scores high against any protein holding a
  similar helix. Antibody filtering was not the cause (7902 chains removed by
  annotation, 2054 by the V-domain motif).
- **Change 1, symmetric score:** a chain pair now counts through
  `mintmscore = min(qtmscore(q->t), qtmscore(t->q))`, the TM-score normalized
  by the longer chain; a pair found in one direction only scores 0. A PDB
  pair keeps the maximum over its chain pairs; the threshold stays 0.50 [R27].
  This is the structural counterpart of the antigen sequence rule's length
  coverage (identity >= 0.30 **and** coverage >= 0.70): a fragment must not
  make two proteins "the same". Measured on the same Foldseek search: 486
  clusters, largest **1953 (68.5%)**, 348 singletons.
- **Why the remaining component is kept:** it is chained through genuine fold
  families shared by many nanobody targets (7TM receptors, G-protein
  beta-propellers, ...). Restricting the search to SNAC/SAbDab-annotated
  antigen chains was previewed and did not break it (annotations exist for
  1435 of 2852 PDBs). Raising the TM-score threshold would rest on no
  published fold criterion, so it was not done.
- **Change 2, oversized components always train:** a layered component with
  at least 50 complexes and more than 1/5 of the pool being split (one fold's
  share) is assigned to training, never to the internal validation fold or the
  antigen-fold holdout. Under the plain hash such a component would land in
  either with probability 1/5 each and would then be most of that fold. The
  carve verifies that the pinned set is identical in the full pool and in the
  smaller pool training later splits, and fails otherwise, so the internal
  validation split is still exactly the one the carve assumed.
- **Scientific effect:** the internal validation fold and the antigen-fold
  holdout are both drawn from complexes outside the dominant fold family; the
  holdout claim of A10 ("antigen folds never seen in training") is unchanged
  and, if anything, stricter. The training set contains the dominant family.
  `antigen_fold_holdout.json` records the pinned component sizes.
- **Implementation:** `build_foldseek_pairs.py` writes
  `query<TAB>target<TAB>mintmscore`; `--reuse-raw` rescores an existing search
  of the same chains. `independence_clustering.score_semantics` is
  `mintmscore`; tables with the old `qtmscore` header fail provenance checks
  and must be rebuilt. The pin rule is `train_egnn_pruning.assigned_fold`.
- **Affected results:** queue_freeze (cluster map, split), egnn_train,
  energy_calibration, external_validation.
- **Results inspected before this amendment:** not applicable (cluster sizes
  are data composition; no split, graph set or outcome existed).

## A12. Entry resolution from curation metadata; pipeline staging not audited

- **Before:** the structure-quality gate (resolution <= 3.0 A, required)
  read the resolution only from the structure file.
- **Measured:** SNAC-DB's curated complex files carry no resolution record, so
  all 3634 `snac_db` rows failed with `unknown_resolution` and the hard test
  pool was empty; 993 of 2424 `sabdab_vhh` files also reported none. This is a
  missing field, not a quality outcome.
- **After:** when the file reports none, the audit takes the entry's
  resolution from the SNAC curation summaries (`Resolution`) and, failing
  that, from SAbDab summary tables (`resolution`) under the data root, keyed
  by PDB ID; the value's origin is recorded as `resolution_source`
  (`structure_file`, `snac_curation_summary`, `sabdab_summary`). Both carry
  the deposition's own value [R33,R34], so this reads a published field, not
  an estimate. A field listing one value per entry ("2.5, 2.7") is read at
  its **worst** (largest) value, because the gate is an upper bound and a
  field's order must not decide admission. An entry with no resolution
  anywhere (e.g. NMR, "Resolution is Missing") still fails. The threshold and
  every other quality gate are unchanged.
- **Provenance:** the gate's outcome now depends on metadata files, so the
  audit report lists every one it used with its SHA-256, and files under
  pipeline working directories are excluded from the search. Otherwise a
  table dropped anywhere under the data root could silently change which
  training structures are admitted, with nothing in the run naming it [R48].
- **Also:** `data/external_vhh/` (external-VHH staging: downloaded candidates,
  the Foldseek antigen-only input structures and the preparation's own copies
  of summary tables) is no longer read as study input, neither as audited
  structures nor as resolution metadata; it had entered the audit as 2672
  `extra_external_vhh` rows.
- **Affected results:** audit, dataset admission (training pool, hard test
  pool), everything downstream.
- **Results inspected before this amendment:** not applicable (the dataset
  stage had produced no graph).

## A13. The antigen-fold holdout takes several folds when one is too small

- **Before:** A10 held out one fold (fold 1 of 5) and required at least 10
  independent components.
- **Measured (before any training or outcome):** the training split holds 233
  graphs in 35 layered components, of which one component of 112 graphs (48%
  of the pool) is pinned to training (A11). The remaining 121 graphs in 34
  components spread over 5 folds as 6, 5, 6, 6 and 11 components; the internal
  validation fold (0) takes the first. No single holdout fold reaches 10, so
  the carve failed the preregistered gate, as designed.
- **After:** the holdout may take several folds, and the rule is **the lowest
  non-validation fold indices** (`1`, then `1,2`, ...), never the fold that
  happens to hold the most components. Holding out every non-validation fold
  is refused. `fold: "1,2"` gives 11 components and 45 graphs.
- **Why this and not a lower threshold:** the confirmatory tests are two-sided
  cluster sign-flip tests, whose smallest attainable p value is 2/2^G. G=6 is
  the first that can reach p < 0.05 [R43], so a 5-component holdout cannot
  support a confirmatory claim at any effect size (A10). Lowering
  `min_components` below the fold's own count would only move the gate past
  the arithmetic it exists to enforce.
- **Why the rule, not a choice:** fold 4 holds 11 components and would pass on
  its own. Selecting it would be choosing the split by its component count.
  Taking the lowest indices in order is a rule fixed before the counts were
  read, and the run records which folds were taken (`folds_held_out`).
- **Limit to disclose:** with 233 training graphs, 112 of them one homologous
  component, the holdout is 45 graphs in 11 components and the internal
  validation fold is 17 graphs in 6 components. The manuscript must report
  these sizes with the holdout result; they bound what any external claim can
  assert, and the 11 components are the G of the sign-flip test.
- **Affected results:** queue_freeze (split), egnn_train and
  energy_calibration (smaller training pool), external_validation.
- **Results inspected before this amendment:** not applicable (component
  counts are data composition; no graph outcome, training run or metric
  existed).

## A14. A downloaded RCSB assembly file is the assembly, not an unannotated ASU

- **Before:** A3 made `sabdab_vhh` and `train_rcsb` complexes read from the
  biological assembly, failing closed when an entry carries no assembly
  annotation, so a crystal-packing neighbour is never mistaken for antigen
  [R33,R34,R35,R36].
- **Measured:** all 5000 `train_rcsb` entries failed with `no biological
  assembly annotation (REMARK 350/pdbx_struct_assembly)`; the subset
  contributed 0 eligible complexes to every run so far. The files are
  `rcsb_non_redundant_dataset/cif_assemblies/<entry>_assembly1.cif`: RCSB
  serves a biological assembly as its own file, whose coordinates are already
  assembled. Such a file carries no `pdbx_struct_assembly`, because that
  record says how to *build* an assembly from the asymmetric unit. The rule
  was therefore rejecting files that are already what it asks for.
- **After:** a file named `<entry>_assembly<N>`, `<entry>-assembly<N>` or
  `<entry>.pdb<N>` (RCSB's assembly naming, also inside an archive member) is
  read as the assembly itself in the assembly subsets: no transform is
  applied, and `structure_source` records
  `prebuilt_assembly_file:assembly<N>` instead of
  `biological_assembly:<basis>:<name>`. Every other file keeps failing closed;
  an asymmetric unit without assembly annotation is still excluded.
- **Why this is not a loosening:** the requirement is that a complex be the
  biological assembly, not that a particular record be present. Applying a
  transform to an already-assembled file would build the assembly twice. The
  distinction is recorded per graph, so any analysis can separate complexes
  assembled by this pipeline from complexes served pre-assembled.
- **Affected results:** audit validity for `train_rcsb`, dataset admission
  (the training pool), and everything trained or calibrated on it. Cluster
  counts, the internal validation split and the antigen-fold holdout (A13) are
  all recomputed from the larger pool, so A13's measured counts describe the
  pool before this amendment and must be re-read after it.
- **Results inspected before this amendment:** not applicable (admission
  counts are data composition; no training run or metric existed).

## A15. Entry resolution fetched from RCSB for files that carry none

- **Before:** A12 let the audit take a missing resolution from curation
  metadata (SNAC, SAbDab). Both cover antibody entries only.
- **Measured:** with A14 admitting them, all 5000 `train_rcsb` files parse,
  but 4505 have no resolution from any source and 4580 fail the
  structure-quality gate. The files hold `_exptl.method` ("X-RAY DIFFRACTION")
  and no `_refine` block at all, and their `_exptl.entry_id` is the
  placeholder `XXXX`; the dataset ships only `non_redundant_pdb_ids.txt`, a
  bare ID list. The 495 that did resolve were entries that happen to appear in
  SNAC's summaries. So the subset would again be rejected for a missing field
  rather than for quality.
- **After:** `fetch_entry_resolution.py` asks a finished audit which valid
  structures it could not date (`--missing-from-audit`), queries RCSB once for
  those entries' own `rcsb_entry_info.resolution_combined` and writes
  `pdb<TAB>resolution<TAB>method`. Driving it from the audit rather than from
  one subset's ID list means the same table also covers the 993 `sabdab_vhh`
  files whose own records carry no resolution. A file named `*entry_resolution.tsv` under
  the data root (outside pipeline staging) is read by PDB ID exactly as the
  SNAC and SAbDab tables are, with `resolution_source =
  rcsb_entry_resolution`, and is listed with its SHA-256 in the audit report
  [R48]. An entry RCSB reports no resolution for (NMR, some cryo-EM) is
  written empty and stays excluded. A multi-method entry is read at its worst
  (largest) value, as in A12.
- **Why the gate is not waived for this subset:** the study reconstructs side
  chains, and side-chain positions are the least reliable part of a
  low-resolution model. `train_rcsb` supplies the interface geometry the EGNN
  learns from, so admitting structures whose side chains are unreliable would
  teach the model that geometry. The 3.0 A limit therefore applies to it
  exactly as to the antibody subsets; the amendment supplies the missing
  field, it does not lower the bar.
- **Reproducibility:** the fetch runs once and its table is kept beside the
  structures, so the audit itself never needs the network, and the value used
  is fixed by the recorded hash rather than by whatever RCSB serves later.
- **Affected results:** audit resolution fields, dataset admission, and
  everything trained or calibrated on the resulting pool. As with A14, the
  component and fold counts in A13 predate this amendment and must be re-read
  after it.
- **Results inspected before this amendment:** not applicable (admission
  counts are data composition; no training run or metric existed).


## A16. Formal VHH sources and cross-source PDB precedence

- **Before:** `build_final_pyg_dataset.py` concatenated `train_rcsb`,
  `sabdab_vhh` and `snac_db` as equal training candidates. Duplicate PDBs
  across sources could be represented more than once. The formal SAbDab path
  could depend on same-PDB SNAC annotations, while generic RCSB complexes had
  no independent proof that the strongest-contact partner was a VHH.
- **After:** formal source roles are fixed before any quality or outcome is
  inspected: SNAC-DB is primary, SAbDab is auxiliary, and generic RCSB is
  audit-only. Cross-source duplicates use deterministic precedence
  `SNAC > SAbDab > RCSB`; all rows inside the winning source remain eligible
  for their source-specific gates. SAbDab formal ingestion now reads its own
  H/L/antigen metadata, rejects L-chain/scFv/non-polypeptide-antigen entries,
  verifies an unambiguous VHH in the biological assembly, records the CDR
  annotation method, rejects unannotated extra Ig variable domains, and binds
  formal antigen chains to SAbDab metadata plus the 7.5 A paratope-contact
  rule [R33]. SNAC keeps its curated per-complex annotations [R34].
  The Foldseek universe now contains only source-verified, row-level
  QC-eligible formal VHH candidates from the same preferred source used by
  graph admission; rejected structures cannot bridge single-linkage antigen
  components. Graph protocol v1.10 additionally binds each graph-consumed raw
  structure to the SHA-256 recorded by data audit and rechecks the bytes before
  construction, so audit/build cannot silently span different structure
  snapshots. Earlier v1.8/v1.9 graphs cannot resume under these semantics.
- **Why RCSB is audit-only:** biological-assembly and resolution checks prove
  coordinate quality, not nanobody identity. Treating the first chain of the
  strongest generic protein interface as VHH would change the learning task.
  RCSB can re-enter formal training only after an independently auditable
  strict VHH and antigen-role annotation is added in a future amendment.
- **Why precedence is outcome-free:** source precedence is decided solely from
  source identity and PDB ID. A lower-priority representation is not used as a
  fallback when the preferred source later fails a quality gate, so admission
  cannot be rescued by inspecting downstream quality or model outcomes.
- **Affected results:** audit metadata, Foldseek clustering universe, formal
  graph composition, EGNN training/validation, calibration, holdout,
  QAOA/structure analyses and all downstream reports.
- **Results inspected before this amendment:** _to be completed by the investigator_.


## A17. Primary-source antigen-fold holdout and exact source binding

- **Before:** antigen-fold components were carved jointly from SNAC and SAbDab
  and every graph in a selected component became a scored holdout target.
  Holdout raw structures were selected by PDB ID only, so a PDB represented by
  more than one source (or more than one curated complex) could bind a graph to
  the wrong raw file.
- **After:** layered components are still built jointly from SNAC primary and
  SAbDab auxiliary graphs, preserving the same leakage barriers. A component is
  eligible for the antigen-fold holdout only when it contains a SNAC primary
  graph. The whole selected component is removed from training; exactly one
  deterministic SNAC graph per PDB is scored, while SAbDab component members
  and duplicate same-PDB SNAC graphs are moved to
  `graphs/holdout_quarantine` and never scored. Every scored target resolves
  its raw structure by exact audit `source_id` and audit-time SHA-256, not by
  PDB ID alone. The holdout manifest schema is v2.
- **Why:** SAbDab is an auxiliary source in the frozen data design and should
  prevent leakage without becoming a co-primary evaluation population.
  One-target-per-PDB avoids pseudo-replication and matches the downstream raw
  source contract. Exact source binding closes the ambiguity created by
  cross-source or per-PDB duplicate representations.
- **Affected results:** antigen-fold holdout composition, external_validation
  (when it scores the run-local holdout), cluster adequacy counts and final
  reporting of that holdout.
- **Results inspected before this amendment:** _to be completed by the investigator_.


## A18. Literature-audited interface definition, formal IMGT numbering and strict fixed backbone

- **Primary interface:** the formal VHH-antigen interface is now cross-partner heavy-atom distance **<=4.5 A** [METHODS_EVIDENCE R49]. Graph v1.11 also stores node-aligned 3.5 A and 5.0 A labels as preregistered strict/permissive sensitivity definitions. They do not alter cross-partner KNN edges or the primary EGNN target.
- **Audit/build consistency:** audit receives the same 4.5 A cutoff, records it in `data_audit_inventory.json`, and graph construction rejects an audit made with a different primary cutoff. The standalone external-preparation audit now forwards this setting as well.
- **Formal CDR definition:** formal SAbDab and genuine external-VHH candidate selection/building require ANARCI with IMGT numbering and fail closed if the dependency is absent. The motif CDR-H3 locator remains only for non-formal debug/legacy paths. SNAC retains its curated source IMGT region annotation.
- **Fixed backbone:** formal structural runs require `loop_relax_iterations: 0`; orchestration validates this before execution and now also defaults every formal stage invocation to zero. Primary all-atom relaxation therefore cannot move N/CA/C/O coordinates.
- **DB5.5:** remains optional (`include_db55_auxiliary: false` by default); when explicitly enabled, the original exact-248 fail-closed contract is preserved.
- **Quantum resource wording:** current two-qubit resource counts are explicitly variational-layer counts (cost ZZ + local XY mixer). Full-circuit counts remain null until a concrete W-state StatePrep decomposition is frozen; no total-gate claim is made by substituting the variational count.
- **Affected results:** graph labels/protocol hashes, EGNN training, Active-site selection, formal external VHH preparation, structure endpoints and all downstream reports. Old graph v1.10 artifacts are intentionally incompatible.
- **Results inspected before this amendment:** no formal validation/test result was used to select these protocol changes; they implement the literature and protocol audit performed before formal rerun.


## A19. PyRosetta dun10 rotamer source

- **Before:** the frozen formal protocol parsed the external `ALL.bbdep.rotamers.lib` text file directly and selected its nearest 10-degree backbone bin.
- **After:** both coarse and all-atom candidate builders query the installed PyRosetta dun10 library at the residue's nearest 10-degree φ/ψ bin. The returned probabilities, χ means and standard deviations enter the existing probability filter and χ1 expansion. The installed PyRosetta build is pinned in the frozen configuration; preflight checks the build, active dun10 option and a real LYS sample. Candidate and calibration provenance record the PyRosetta version and source. The old text parser remains available only under its explicit legacy mode.
- **Why:** a single Rosetta-maintained library API provides the candidate statistics used by the study. This is a scientific source change, not an implementation-only refactor: Rosetta's sample set and interpolation/semibackbone conventions may differ from the earlier text parser.
- **Affected results:** candidate pools and probabilities, energy calibration, QUBO coefficients, solver results, structure analyses and reports. Complete formal results must be generated in a new run directory; they cannot be resumed from a text-library run.
- **Results inspected before this amendment:** the investigator supplied progress from the earlier run through queue freeze and eligibility screening, but no final performance or structural endpoint was supplied for this amendment. The change follows the explicit source choice and a direct PyRosetta API check on the server.


## A20. One-time FP32 continuation after failed EGNN AMP training

- **Before:** the affected formal run completed data audit and queue freeze, then EGNN training failed under FP16 automatic mixed precision. Its downstream stages were blocked.
- **After:** a dedicated recovery command verifies the completed stage artifacts and their upstream bindings, requires that EGNN failed and no downstream stage completed, and allows only `egnn_train.amp: true` to `false` in the resolved configuration. It archives the original run manifest, frozen configuration and any partial FP16 checkpoints, then records old and new source hashes and the completed-stage result hashes. EGNN trains from scratch in CUDA FP32 on the same frozen graph split; downstream stages use its FP32 checkpoint.
- **Why:** FP16 can overflow on coordinate and squared-distance computations. Reusing the failed FP16 checkpoint would mix training precision regimes. The completed upstream outputs are retained only after their recorded content hashes and dependency chain pass verification.
- **Affected results:** EGNN checkpoint and all stages after `queue_freeze`. Data audit and queue freeze remain the original frozen outputs. The resulting run is explicitly marked as an amended protocol lineage, not an unmodified resume.
- **Results inspected before this amendment:** the reported training failure and missing EGNN artifacts; no downstream performance or structural result was available.
- **Enforcement (2026-10-01):** configuration validation now rejects any formal configuration whose `egnn_train.amp` is not explicitly `false`, before any stage runs; a missing key is rejected too. Previously FP32 depended only on the config value: the stage fell back to FP16 when the key was absent and the trainer's own default was FP16. Both fallbacks are now FP32. The protocol itself is unchanged.


## A21. Calibration eligibility and rare real χ1-well samples

- **Trigger:** the first FP32 formal continuation attempted calibration on 235 frozen training graphs. It generated rows for 68; 93 failures reported a missing χ1 well after the global rotamer probability floor, and many others reported absent source heavy atoms. No solver comparison or structural endpoint was used to choose this repair.
- **Rotamer rule:** fixed-resolution three-well cases retain the highest positive-probability *actual PyRosetta/Dunbrack sample* from each χ1 well even if that sample is below the global `1e-4` floor. All other samples remain subject to the floor. The rescued sample keeps its original probability and corresponding prior-energy penalty. If the source library truly lacks a positive-probability sample in a well, the case still fails. This preserves exactly one state per well without inventing coordinates or flattening priors.
- **Calibration input eligibility:** a source complex missing observed protein heavy atoms is excluded before calibration energy rows are accepted. The provenance separately records all discovered training complexes, quality exclusions, eligible attempts and generation failures, and both fractions must close arithmetically. At most 40% of discovered complexes may be excluded for this input-quality reason; at most 10% of eligible attempts may fail row generation. The existing minimum independent training complexes/groups, grouped cross-validation and fit-quality gates remain in force. This exclusion is based only on raw structure completeness, not energy or solver outcomes.
- **Run lineage:** the one-time amendment verifies and preserves data audit, queue freeze, seed streams and the FP32 EGNN checkpoint. It archives all downstream statuses and outputs, including any prior structural run, then reruns calibration and every later stage under the new source/config hashes. The original manifest/config and superseded outputs remain in run-local provenance. The run must be idle before the amendment is applied.
- **Affected results:** rotamer candidate pools, coarse and all-atom energy modeling, calibration fit, all solver comparisons, structural analyses and reports. Upstream graph split and EGNN model remain the original frozen inputs.
- **Results inspected before this amendment:** calibration failure counts and five raw-versus-filtered PyRosetta χ1-well diagnostics. The diagnostics showed all three raw wells in those examples; the missing well was removed by the probability floor. No later-stage performance result was used.


## A22. Calibration completeness gate based on usable training count

- **Trigger:** the formal run found 235 training structures, of which 106 lacked observed heavy atoms required by Amber preparation. The remaining 129 structures produced one generation failure. A 40% exclusion-fraction gate therefore blocked a structurally usable calibration subset even though the denominator and family-group checks were independently auditable.
- **After:** missing-atom exclusions remain recorded with exact counts and hashes, but are no longer accepted or rejected by a post hoc percentage threshold. Formal calibration requires at least 100 eligible complexes, at least 20 family groups, grouped cross-validation with at least five folds, and the existing fit-quality thresholds. Eligible-attempt generation failures remain capped at 10%.
- **Why:** the scientific estimand is the calibration fit on structures that satisfy its predeclared Amber input contract. The exclusion fraction is a data-coverage descriptor; the minimum eligible count and family-group gates protect statistical identifiability without allowing a small but artificially clean subset.
- **Run lineage:** the coverage amendment preserves the already verified audit, queue freeze and FP32 EGNN checkpoint, archives the prior calibration/downstream outputs, updates the manifest/config lineage, and reruns calibration and all downstream stages.
- **Results inspected before this amendment:** provenance counts supplied from the formal run: 235 discovered, 106 input-quality exclusions, 129 eligible attempts and one generation failure. No solver or endpoint result was used to set the gate.


## A23. Surrogate solver benchmark with an independent atomistic endpoint

- **Trigger:** the training-only Amber calibration failed its frozen acceptance gates (grouped CV RMSE about 2.64e19 kcal/mol, CV Spearman about 0.007). The median absolute raw Amber delta was about 3.27e7 kcal/mol across 8,192 rows. A read-only diagnostic found a 0.046 Å nonbonded overlap; 1,000 iterations of fixed-backbone relaxation still left two affected states roughly 6,850 kcal/mol above the anchor. A PyRosetta candidate scan showed that mandatory coverage of all three chi1 wells can retain highly clashing states on this structure.
- **Revised question:** compare QAOA and classical solvers on the same frozen coarse-grained QUBO, and test structural outcomes independently with the all-atom experiment. Solver objective energy is a surrogate score, not a calibrated Amber energy or binding free energy. The coarse and atomistic stages have different candidate constructions, so no cross-stage equality of rotamer states is claimed.
- **Diagnostic:** attempt the training-only Amber fit and preserve its CSV, provenance, coefficients when available, grouped CV metrics and acceptance failures. In `diagnostic` mode, an unavailable fit or a fit that misses scientific acceptance is recorded as `completed_with_failures`. A machine-readable assessment binds the hashes of any generated artifacts and states that the coefficients were not applied. Inconsistent provenance or tampered recorded results still stop the run.
- **Solver rule:** diagnostic coefficients are never passed to method sensitivity, QC benchmarking, quantum exploration or external VHH benchmarking, even if the fit JSON exists. These stages use the original unit-weight coarse components. All solver methods within an experiment receive the same QUBO and budget rules. The independent atomistic structural stage retains its separate Amber protocol.
- **Reporting:** the final report states the failed calibration and the consequent limit on coarse-energy interpretation alongside the solver and structural results. It cannot claim that coarse QUBO gains predict Amber energy gains merely from the solver benchmark.
- **Lineage:** this is a new frozen protocol branch and must run in a new directory before inspecting the untouched validation/test endpoints. The prior failed calibration run remains an immutable development record. Reuse of any earlier audit, queue or checkpoint requires the run's normal hash and upstream-lineage checks; a simple `--resume` of the previous manifest is not valid.

## A24. Absolute minimum distance screen for input complex quality

- **Trigger:** a calibration diagnostic found a near-zero atom separation in a generated rotamer state. That observation motivates checking source complex geometry before graph construction, but does **not** itself show that the deposited structure was defective. Generated-state clashes remain solver/candidate diagnostics and cannot be relabelled as input-data exclusions.
- **Frozen rule:** on the first model of each prepared biological assembly, select the strongest allowed VHH–antigen protein-chain pair using the existing interface rule. For each chain, retain one positive-occupancy heavy atom per name (highest occupancy). Exclude atom pairs within the same residue, then reject a formal complex if any distinct-residue heavy-atom centers are **strictly closer than 1.0 Å**. Ordinary peptide and disulfide bond lengths exceed this hard floor. The threshold is a deliberately conservative absolute coordinate-overlap rule, **not** the MolProbity clashscore or its 0.4 Å van der Waals overlap definition ([MolProbity 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC1933162/), [MolProbity 2010](https://pmc.ncbi.nlm.nih.gov/articles/PMC2803126/)).
- **Accounting:** the audit writes the offending pair, minimum observed distance below the floor, and pair count per structure; the inventory gives evaluated, excluded, and atom-pair totals. A structure with this reason receives `structure_quality_status=fail`, cannot enter formal graph construction or the Foldseek universe, and remains visible in the audit denominator. The external VHH quality check uses the same rule on its chosen VHH and antigen chains.
- **Scope and limitation:** the screen uses experimentally deposited protein heavy atoms after assembly selection. It does not inspect added hydrogens, ligand atoms, alternate conformer combinations, or every generated rotamer. It must not be used to erase difficult solver cases after outcomes are seen. Previously frozen audits and queues do not satisfy the revised contract; rerun affected stages under a new manifest before evaluating held-out results.

## A25. Surrogate rank and finite-range resource interpretation

- **Trigger:** independent workflow review found that the coarse χ1-oriented score and atomistic energy had no displayed same-assignment rank diagnostic, that the four-size slope could be overread as asymptotic scaling, and that shots and single-state energy queries were presented in a shared QTS99 unit.
- **Change:** the final report computes within-PDB Spearman ranks from the existing training-only calibration CSV, including constant groups and extreme raw Amber deltas. No fitted coefficient or target/test outcome enters solver selection. The scaling output states that 4/6/8/10 sites at three states each is a finite-range trend with changing residues and landscapes; exact enumeration remains feasible. The resource section displays measurement shots, energy queries and elapsed time in separate columns. Historical QTS99 becomes descriptive abstract accounting: its formula and raw values are unchanged, but it is removed from the secondary gatekeeping inference family because the two operation types cannot be cost-equated.
- **Claim boundary:** a training-only unrelaxed rank correlation cannot establish held-out physical fidelity or all-atom RMSD improvement. No asymptotic or hardware advantage claim follows from the slope or QTS99. Independent structural outcomes remain the only evidence for recovery.
- **Affected results:** final report, scaling metadata/prose and secondary multiplicity interpretation. Frozen solver instances, optimizer settings and raw p-values are unchanged; adjusted secondary-family p-values can change because QTS99 tests are removed. The source/config hash contract requires a new run for these changes.
- **Results inspected before this amendment:** the earlier training-only failed Amber calibration and generated-clash diagnostics described in A23 were inspected; whether any validation/test outcomes had been inspected requires investigator confirmation.

## A26. Multi-process, multi-GPU execution for independent complexes

- **Trigger:** on the two-T4 server, the diagnostic Amber row generator and independent frozen structural targets were sequential and used only CUDA device 0. The investigator requested multi-process, multi-GPU execution for a **new** formal run.
- **Change:** after sequential eligibility and independence selection, structural targets run in distinct spawned processes, one per assigned CUDA device. Calibration training complexes are sharded across distinct CUDA processes; each complex has an RNG stream derived from the frozen seed and its sorted training-manifest row index, independent of worker count. The parent merges CSV rows in manifest and assignment order and aggregates all exclusions and failures. EGNN DDP and the existing CPU benchmark workers remain as configured.
- **Reproducibility:** device list, worker count and code hashes are recorded in the resolved config or output provenance. CUDA device/precision can affect floating-point results; this is a new protocol, not a valid resume of an earlier frozen run. GPU assignment never changes the queue membership, target denominator, solver budgets or per-target structural seed streams.
- **Affected results:** calibration samples and fit can change because the earlier global RNG stream is replaced by per-complex streams. Structural numerical outputs may differ across CUDA devices. The Amber fit remains diagnostic and its coefficients remain unused by solvers. The formal run must start in a new directory.
- **Results inspected before this amendment:** the training-only failed Amber calibration diagnostics were inspected. Validation/test inspection status requires investigator confirmation.

## A27. Additional independent-task concurrency

- **Trigger:** a stage-by-stage concurrency audit found that source-structure audit entries and external VHH target comparisons were still evaluated in a thread pool or serially, while method sensitivity cases, exploratory QAOA depths and historical development structures ran one after another despite independent inputs and output directories.
- **Change:** the source-structure audit now uses spawned CPU processes with each worker loading the same annotation and threshold contract; the parent writes results in discovery order. Foldseek antigen input structures and external VHH targets use spawned processes and are returned in universe or sorted graph order before downstream checks. Method sensitivity cases, exploratory depths and independent paired-statistics budget modes run concurrently with bounded separate subprocesses; each depth retains fit-to-transfer-to-hard-set ordering. Historical development structures run concurrently in separate subprocesses with one active target per CUDA device. Validation queue membership and the formal stage order remain sequential.
- **Resource bound:** the 80-core/256-GB server profile allows 48 independent CPU workers with one compute thread each; external audit and Foldseek preparation are capped at 12 workers. Sensitivity cases split that total across three concurrent jobs, and exploratory depths across two. The resolver limits worker count by usable cores divided by threads per worker. Graph construction has a separate 16-thread cap and keeps its existing thread pool: it already performs parallel C-library work while preserving one-process memory accounting, and its hard-set candidate choice is ordered and outcome-dependent. GPU structure workers remain limited to one per device. These are configured limits, not measured throughput guarantees.
- **Affected results:** no scientific threshold, target order, seed, solver budget or endpoint changes. Runtime and floating-point device assignment may change; this amendment requires a new frozen run and server verification.
- **Results inspected before this amendment:** training-only failed Amber diagnostic results were inspected. Validation/test inspection status requires investigator confirmation.

## A28. Topology-aware preparation and physical acceptance of atomistic outputs

- **Trigger:** the investigator requested strengthening structure preparation and evaluation before reconsidering Amber calibration. Earlier development diagnostics found unobserved CDR-H3 residues and a generated hydrogen/nonbonded overlap near 0.046 A; energy decrease after capped relaxation did not establish convergence or accuracy.
- **Preparation:** retain the canonical-heavy-atom completeness rule and exact CDR mapping. Record annotation, heavy-atom and topology reasons separately. Check actual C--N connectivity and the existing 1.9 A distance bound rather than author-number continuity. Add an explicitly chosen 0.4 A absolute near-coincidence screen with bonded/1--3 exclusions. This is not MolProbity's van der Waals overlap definition and does not replace the source heavy-atom 1.0 A screen in A24.
- **Evaluation:** audit hydrogens and retained candidate geometry, record force-group energies and measure final movable-atom RMS force against 10 kJ/mol/nm. Final topology, extreme geometry and convergence failures remain method-output failures, with structures and metrics retained. A skipped relaxation cannot pass convergence acceptance. Optional restrained loop relaxation uses the restrained objective for the force test. No candidate state is removed by these diagnostics, and no iteration budget or energy is silently changed.
- **Accounting:** preserve output/seed failure summaries, propagate a failed child experiment to the frozen target closure and withhold the completion marker. Deliberately perturbed inputs are retained; generated failures cannot become exclusions of the original experimental data. Formal statistical input rejects invalid output rows rather than shrinking the denominator.
- **Protocol scope:** prospective atomistic eligibility and output acceptance change; coarse solver settings and failed diagnostic Amber acceptance are unchanged. This requires a new frozen manifest and re-frozen eligibility before untouched endpoint evaluation. Existing result directories remain historical records and will fail the changed code-hash resume contract. No prior checkpoint/queue is certified reusable by this amendment alone.
- **Evidence and limits:** see `STRUCTURE_PREPARATION_AND_EVALUATION.md` for verified primary sources and operational definitions. Passing the hard floor is not full stereochemical validation or binding-affinity evidence. Server PyRosetta/CUDA convergence and exclusion rates remain to be measured. Previously shared development/training diagnostics were inspected; validation/test inspection status must be recorded by the investigator before claiming prospective confirmation.

## A29. Full training-manifest raw and relaxed Amber diagnosis

- **Request:** diagnose all training complexes rather than a small pilot before reconsidering the failed Amber fit.
- **Scope:** the standalone diagnostic reads the frozen source run, overrides the training-complex cap to zero, retains its assignment count/sampling rule and evaluates both raw and uniformly constrained-relaxed Amber energies. It uses no validation/test graphs and writes outside the frozen run. Full training coverage does not mean exhaustive enumeration of every legal state.
- **Measurements:** preserve preparation checks, complete chi assignments, coordinates/hashes, energy force groups, geometry and residual forces. Individual state failures remain recorded while later states are evaluated. Missing anchor energies remain unavailable rather than causing a replacement anchor or disappearing states.
- **Accounting:** every training-manifest entry must close as an assessed complex, input-quality exclusion or complex execution failure. Every assessed complex must contain its planned assignment indices, including failed assignments. The report separates all-finite-state ranks from geometry/convergence-screened subset ranks and keeps constant correlations explicit.
- **Execution and interpretation:** two independent workers can use the two declared CUDA devices with scheduling-independent per-training-row seed streams. This is a training diagnostic, not a new calibrated solver run, acceptance-gate relaxation or confirmatory structure result. CSVs marked as this diagnostic are rejected by the calibration fitter. See `FULL_TRAINING_DUAL_ENERGY_DIAGNOSTIC.md` for the server command and artifacts.
- **Results inspected:** previously shared training/development calibration and clash diagnostics motivated the assessment. No server full-training dual-energy result has been inspected or produced during implementation. Subsequent model choices based on this assessment remain training/development decisions and require separate prospective validation.

## A30. Explicit GPU sharing for training diagnostics and higher CPU concurrency

- **Request:** improve utilization of the 80-core/256-GB/two-T4 server.
- **Execution:** the standalone full-training diagnostic defaults to eight workers, four per declared GPU. The generator retains its one-worker-per-GPU default unless `--workers-per-gpu` explicitly permits sharing. Round-robin device assignments and the sharing limit are recorded, child BLAS/OpenMP threads are fixed at one, and per-training-row seeds and ordered merge are retained. The server profile permits 64 independent CPU workers while retaining core reservation and separate memory-heavy task caps.
- **Scientific settings:** no change to state sampling, force field, precision, iteration budget or acceptance rules. All failures remain in the denominator. No server throughput or numerical equivalence measurement has yet been performed; identical seeds alone do not guarantee bitwise CUDA equality.
- **Scope:** the diagnostic writes a new separate directory. Changes to the server profile apply to newly resolved formal runs, not an in-progress frozen run. Formal structural targets retain the existing one-process-per-GPU setting. Optional NVIDIA MPS is administrator-managed, not started by this code; more processes are not a guarantee of higher throughput.

## A31. CPU worker target range of 48 to 64

- **Request:** set the independent CPU worker lower target to 48.
- **Resolution:** `resources.min_workers: 48` and `max_workers: 64` define the target range for data-audit and QC worker pools. The existing automatic resolver selects 64 on the 80-core host with eight reserved cores and one thread per process. It rejects an inverted range and records the requested range, whether the lower target is met and a hardware/thread-budget limitation when it is not.
- **Limits:** available cores and actual task count can reduce concurrency. The target is not an instruction to launch idle processes or exceed memory-heavy stage limits. Graph building, external audit, Foldseek preparation, EGNN loaders and GPU stages retain their separately configured caps. Changes apply to newly resolved runs; scientific settings are unchanged.

## A32. Four structural target workers per CUDA device

- **Request:** run `structure_experiment` with eight independent target workers across the two T4 GPUs, four per device.
- **Execution:** the server profile sets `structural_workers_per_gpu: 4`; the resolver freezes eight target workers for two devices. The structural driver accepts an explicit `--workers-per-gpu` limit and assigns each spawned worker a fixed device. Independent targets retain their frozen arguments, seeds, output directories and ordered result collection. Spawned workers use one Torch/BLAS/OpenMP compute thread. Fewer available targets create fewer workers. The historical development queue uses the same configured concurrency bound for separate subprocesses; eligibility and within-target method/seed evaluation remain ordered.
- **Audit:** `structure_execution.json` records requested workers, active worker-to-device mapping, target count and OpenMM runtime settings. All existing target-failure and denominator checks remain in force. Sampling, precision, energy model, iteration budget and acceptance rules are unchanged.
- **Scope and limits:** applies to newly resolved formal runs. Existing frozen runs are not silently amended; their code/config resume checks remain active. CUDA/OpenMM throughput, memory use and numerical reproducibility must be measured on the server; low memory use alone does not establish an acceleration. The code does not start NVIDIA MPS.

## A33. Pipeline-wide concurrent atomistic preparation and GPU runtime evidence

- **Request:** improve GPU utilization throughout the pipeline after a snapshot showed only one active CUDA process.
- **Changes:** independent candidate preparation for queue freezing and structural execution uses four spawned workers per declared CUDA GPU. Admission remains in original candidate order with all sequence/cluster exclusions and per-site compatibility failures retained. Physical errors are consumed after sequence checks as before. Preparation ends before target recovery starts. Formal calibration now honors its explicit four-workers-per-GPU setting instead of limiting the process count to the number of devices. No failed diagnostic coefficients are reinstated.
- **Evidence:** every orchestrated subprocess can write ten-second GPU utilization/memory/power samples to a stage-local CSV; candidate preparation records PID, device, time and failures. The monitor only reads device information. Missing NVIDIA utilities or query errors do not change scientific stage outcomes.
- **Scientific scope:** retain numerical backends for hydrogen placement and site ranking, training FP32/global batch, solver/subspace implementation, seeds, state count, precision, iteration budgets and acceptance rules. Faster preparation completion cannot decide which target is admitted. Data-dependent future work still belongs to training/development unless separately validated. Applies to new resolved runs, with code/config hashes protecting old runs.
- **Limits:** eight processes are a configuration, not measured speedup or guaranteed full GPU utilization. Data parsing, CPU rotamer generation, small feasible-subspace simulation, external structural tools, statistics and reporting remain CPU work. See `PIPELINE_GPU_UTILIZATION.md` for the complete mapping, runtime artifacts and verified NVIDIA/OpenMM sources. No server throughput experiment has yet been performed.

## A34. Logical-CPU pool sizing to an 80 % utilization target and stage CPU telemetry

- **Request:** keep the 80-CPU/256-GB/two-T4 server at about 80 % utilization.
- **Finding:** the resolver counted physical cores first. On a host with 40 physical cores and 80 hardware threads it resolved 32 workers (40 % of the logical CPUs that top/htop report) and silently missed the A31 lower target of 48. A fixed `max_workers: 64` is likewise only 40 % on an 80-core/160-thread host.
- **Change:** `cpu_count_basis: logical` counts the logical CPUs the process may run on (CPU affinity, then any cgroup v2 quota). `max_workers: auto` sizes the independent data-audit/QC pools to `target_cpu_utilization: 0.80` of those CPUs divided by threads per process (80 logical CPUs give 64 workers), still bounded by `cpu_reserve_cores` and by `worker_ram_gb: 1.5` per worker above the preflight free-RAM floor. `runtime_resolution` records the basis, counted and physical CPUs, the full-pool utilization and any limiting reason. Every orchestrated subprocess can also write ten-second host and stage-process-tree CPU samples to `logs/<stage>.cpu.csv`; `python -m nanoqc.common.stage_utilization <run_dir>` summarizes CPU/GPU utilization per stage against the target.
- **Scientific scope:** worker counts only change scheduling. Threads per process stay at one, and seeds, ordered merges, solver budgets, precision, batch size and acceptance rules are unchanged. GPU per-device worker counts are unchanged because additional processes could exhaust T4 memory and record preparation failures in the denominator. The monitor only reads process and host counters and cannot change a stage outcome.
- **Limits:** stages with fewer independent tasks than workers, the thread-pool graph build, EGNN training with its frozen global batch of 4, sequential admission and reporting stay below the target by design. Mean utilization is not throughput. Applies to newly resolved runs; code/config hashes protect earlier runs. No server measurement has yet been performed.

## A35. Process-parallel graph building, shared-GPU seed replicates and device-limit separation

- **Request:** raise the whole pipeline toward the 80 % utilization target after A34 showed graph building, EGNN training and the OpenMM stages staying far below it.
- **Graph building:** the train and (uncapped) hard-set graph loops, the hard-set de-redundancy screen and the train/test isolation screen ran in one interpreter, so a thread pool left them serialized by the GIL. They now run in spawned worker processes (`graph_build_parallel.py`), each initialized with the parent's command-line settings and the same audited annotations. Results are consumed in submission order, so the manifest, exclusions, failures and the written graphs are what the sequential code produced: hard-set candidate *k* of a cluster is attempted only after candidates 0..*k*-1 failed; the de-redundancy scan replays the original greedy order over a precomputed pairwise table; isolation details stay in hard-graph order. Homology workers receive only the four sequence attributes those functions read. The graph-worker count is now bounded by RAM per worker (`graph_worker_ram_gb`) rather than a fixed 16, since each worker holds its own edge-allocation budget.
- **EGNN seed replicates:** the development-only replicates (A6) ran one after another, leaving both T4s idle between them. They now run together, each an independent DDP job with its own derived seed, checkpoint directory and torchrun rendezvous, bounded by `max_egnn_replicate_workers`. Every replicate failure is reported instead of only the first. The single preregistered formal checkpoint, its seed and the primary training job are unchanged, and no replicate can be selected as the formal model. Global batch size, AMP and the training loop are untouched: EGNN GPU occupancy is bounded by the frozen global batch of 4, which is a protocol parameter and is not changed here.
- **Device limits versus exclusions:** a CUDA out-of-memory or unavailable-device error was caught by the same handler as a scientific preparation failure, so raising per-GPU concurrency could have recorded scheduling artifacts in the study denominator. `device_errors.py` recognizes such failures through the exception chain; candidate preparation and target recovery now raise `DeviceResourceError` and abort the stage instead of excluding a complex or failing a target. Only with that separation in place is candidate preparation raised to 8 workers per device, because it is mixed CPU/GPU work (structure parsing, Rosetta rotamers, CPU EGNN ranking, Reference-platform hydrogens). Frozen target recovery and calibration keep four workers per device (A32, A33).
- **Scientific scope:** no change to seeds, admission order, exclusion reasons, force field, precision, state sampling, solver budgets, iteration budgets, batch size, AMP or acceptance rules. Failures that are properties of a complex remain in the denominator; failures that are properties of the machine now stop the run instead. Statistics and reporting stay sequential by design.
- **Limits:** these are configurations, not measured speedups, and no server throughput experiment has been performed. Sequential admission, the frozen global batch, external structural tools, statistics and reporting remain below the target by design. Applies to newly resolved runs; code and config hashes protect earlier runs.

## A36. Data-audit startup, reuse and DB5.5 pair concurrency

- **Request:** the data audit stayed slow after A34 raised it to 64 workers.
- **Findings:** (1) every worker ran `load_annotations` itself, so the stage performed one recursive walk of the whole data root and one parse of every SNAC curation summary, SAbDab summary and RCSB resolution table *per process* before auditing anything; (2) `--reuse-non-nano` rows were pickled into a worker and returned unchanged, spending inter-process traffic to compute nothing; (3) the 248 DB5.5 receptor/ligand contact counts ran serially in the parent after the pool had closed, parsing up to ~500 structures single-threaded.
- **Changes:** the parent's loaded annotation tables are handed to workers through the pool initializer (`install_annotations`), so each worker installs the same snapshot instead of re-reading files. Reused rows are merged in the parent and only uncached tasks reach the pool. DB5.5 pair contacts run on the same pool; each pair returns its own failure text instead of raising, so one unreadable pair cannot discard the others' results.
- **Same reload in queue_freeze:** the Foldseek antigen-preparation pool and the graph-build workers (A35) also used `load_annotations` as their initializer, repeating the same walk and parse per process; both now install the parent's snapshot. Foldseek preparation and external-VHH auditing follow the independent CPU pool instead of a fixed 12 workers, since each holds one structure at a time. The orchestrator's pair-coverage check streams the frozen pair table rather than reading its millions of rows into one string plus a list.
- **Equivalence:** rows are written in discovery order exactly as before, whether a row was reused or audited. Worker snapshots are now identical by construction rather than by re-reading files that a concurrent process could have touched. DB5.5 pair rows, their weak/pass status, contact counts and error strings are unchanged, as is the 248-pair contract. No cutoff, resolution rule, occupancy rule, overlap screen, exclusion reason or report field changes.
- **Limits:** no server timing was measured here; annotation parsing in the parent, structure geometry, the Foldseek search itself and the per-residue Dunbrack/Amber eligibility construction remain the real cost of these stages. The eligibility scan keeps its existing numerical backends (A33), so it is not moved off CUDA to use idle cores. Applies to newly resolved runs.

## A37. Per-epoch constant work removed from EGNN training

- **Request:** `egnn_train` stayed slow after A35 let the development-only seed replicates share the GPUs.
- **Finding:** each epoch's checkpoint payload recomputed `family_structure_assignment_sha256`, which `torch.load`s every training and validation graph from disk, plus one more load for the recorded graph protocol. With 233 graphs over 50 epochs that is roughly 11,700 full graph reads per training job, repeated in every seed replicate, although the file set and its frozen family/structure assignment cannot change during a run. The per-step loop also synchronized with the device twice, once for the A20 non-finite guard and once to read the same scalar for loss accounting.
- **Changes:** `run_split_identity` computes the two name digests, the family/structure assignment digest and the graph protocol once before the epoch loop, and the payload uses them; the recomputing path is retained for any other caller. The training step reads the loss scalar once and applies `math.isfinite` to it.
- **Equivalence:** the stored digests, graph protocol and every other payload field are unchanged, and a test asserts the payload is identical with and without the precomputed values. Converting a float32 nan or inf to a Python float preserves it, so the A20 guard still raises on the first non-finite batch (also tested), and the accumulated node-weighted loss is bit-for-bit the same because the same float64 values are summed in the same order. Missing PDB/family metadata now fails before the first epoch instead of after it. Seeds, split, batch size, AMP, optimizer, gradient clipping, early stopping and the selected checkpoint are untouched.
- **Limits:** EGNN GPU occupancy is still bounded by the frozen global batch of 4, which is a protocol parameter and is not changed here. No server timing was measured. Applies to newly resolved runs.

## A38. Streamed large tables and ledgers

- **Request:** continue removing fixed costs from the remaining stages after A36 and A37.
- **Finding:** `symmetric_pdb_scores` read the raw Foldseek output with `read_text().splitlines()`, holding the whole multi-million-row chain-level table plus a list of every one of its lines in memory before scoring any pair. Five readers of the data-audit ledger (one JSON object per audited structure) did the same, three of them with their own copy of the blank-line and truncated-line handling.
- **Changes:** the Foldseek score reader streams its file. A shared `repo_io.iter_jsonl` streams the ledger, skipping blank lines and, when the caller asks, a truncated line exactly as those readers did themselves; the orchestrator, holdout carving, external-VHH preparation, Foldseek pair building and the audit's own `--reuse-non-nano` scan all use it.
- **Equivalence:** a test compares `iter_jsonl` against the former `read_text().splitlines()` expression on the same file, including a blank line and a truncated final line, and another asserts these tables are no longer read whole. Pair scores, cluster maps, holdout carving, universe membership and reuse decisions are unchanged; no threshold, exclusion reason or output field changes.
- **Reviewed and left alone:** the QC benchmark already loads one EGNN scorer per worker rather than per target, and the per-run provenance hashes are computed once per entry point, not per case. The remaining cost of the solver, calibration and structural stages is their own science (subspace simulation, OpenMM energies and relaxation), not repeated setup.
- **Limits:** no server timing was measured. Applies to newly resolved runs.

## A39. Resource failures propagate before scientific failure accounting

- **Finding:** A35's resource classifier did not reach every enclosing exception handler. A CUDA allocation failure during single-site compatibility could become a missing compatible site; an outer eligibility handler could turn the restored resource error into an exclusion. Recovery seeds and training-energy assignments could likewise record a compute failure as a failed scientific result. CPU memory failures had analogous paths in audit, graph building, checkpoint loading and solver restarts.
- **Correction:** recognized CPU/GPU allocation and CUDA device errors, including chained errors and broken process pools, propagate as `DeviceResourceError` before exclusion, target, state or restart accounting. Completed speculative preparations are checked even when the parent does not admit that target. Known scientific failures retain their existing categories and counting. Stage/assignment output directories record resource-failure JSON where an output destination is available; raw diagnostic structures and measured energies are preserved before propagation.
- **Completion:** an interrupted calibration dataset generation is a failed stage even in diagnostic mode. A completed fit failing its scientific acceptance remains an unused diagnostic as before. A failed shard cannot publish merged calibration rows or a closed denominator. Preparation cannot publish the selected-target file until prefetched resource checks finish.
- **Scientific scope:** this corrects execution semantics and enforces A35; it changes no worker count, seed, precision, candidate strategy, energy model, solver budget or physical-quality threshold. Code fingerprints include the shared resource helper. Existing frozen runs retain their code hashes; do not label historical resource-based exclusions as repaired without rerunning the affected stages.
- **Verification:** fault injection covers site compatibility, parent eligibility, unused prefetch results, recovery seeds, diagnostic assignments, calibration shards, CPU audit/graph building, QC workers and QAOA restarts. It also checks that scientific nonconvergence retains its diagnostic record. Local tests do not measure T4 memory capacity or certify any concurrency setting as safe.

## A40. Separate developmental training recovery from validation exclusions

- **Evidence:** the supplied graph manifest contains 236 training graphs, 94 hard-test graphs, 34 holdout graphs and 13 quarantined graphs. Historical targets 4S10 and 9GCN occur only in training (two and four complexes respectively); 8YVO is absent. The orchestrator incorrectly reused `excluded_pdb` as a required test-set development target list, while the pilot searched only `test_snac_hard` and correctly rejected training overlap.
- **Correction:** development configuration now declares `target_pdb_ids: [4s10, 9gcn]` and `candidate_split: train`. All three historical IDs remain permanent validation exclusions. Explicit training recovery is restricted to queue role `dev` and explicit target IDs; validation continues to use hard-test candidates and full training isolation. Training recovery is labelled `development_training_only` and development-exposed, with no independent validation claim. One complex per PDB is chosen in the existing deterministic node-count/PDB/path order, without inspecting solver or energy outcomes.
- **Missing inputs:** before launching development subprocesses, the orchestrator writes a manifest-bound availability record. Requested IDs absent from the declared split are recorded separately from executed targets and remain visible in the developmental summary. Legacy configurations fall back to the historical requested IDs in the training split, thereby recording absent 8YVO without launching it. An empty available development set remains a stage failure; scientific failures of available targets and required GBN2 sensitivity remain failures.
- **Scope:** this repairs the developmental data source. It does not change the frozen independent validation set, training graph membership, checkpoint, solver budgets or acceptance rules. It cannot resolve a separate validation/QC failure or authorize replacing the hashes of an existing run. A current frozen run needs a documented, verified execution repair before rerunning the affected structural outputs; preserve upstream artifacts and prior failure evidence.

## A41. Development target experiments removed at the user's request

- **Request:** remove the developmental structural targets. The current configuration sets `queue_freeze.dev_queue.enabled: false`, removes its execution target list and removes development-only solvent sensitivity. Structure execution now schedules only the frozen validation queue. The historical exposure exclusion IDs remain exclusion metadata, preventing prior engineering targets from becoming independent validation cases.
- **Execution and completion:** disabled development does not read or select training recovery candidates, spawn developmental targets, create development output directories or require development/GBN2 results for stage completion. An execution-plan JSON records the active queues. Validation closure, seed/method denominators, physical-quality acceptance and downstream inferential gates remain required. Legacy configurations with development enabled retain their explicit execution path.
- **Reporting:** disabled developmental results are omitted even if historical files remain on disk. The report explicitly identifies the missing developmental solvent analysis as disabled, rather than an executed or successful sensitivity experiment. This reduces the study's robustness evidence and creates no new performance or structure-benefit conclusion.
- **Existing runs:** the change applies when the disabled-development configuration is frozen. It does not overwrite an earlier run's frozen protocol or authorize reuse of stale structural outputs. Preserve upstream artifacts and failure records when recording a verified continuation amendment.

## A42. CDR-H3 baseline mapping with unresolved internal residues

- **Trigger:** the 2026-09-29 formal QC run recorded 40 failed cases for the same 4GRW complex. Its annotated CDR-H3 is `ATDPECYRVRGYYNGEYDY` (19 residues); the observed VHH chain contains `ATDPECYRVRGYYNGDY` (17 residues). Exact full-string matching cannot locate a loop with two unresolved internal residues, although its observed residues can be mapped.
- **Correction:** the CDR-priority baseline first requires a unique exact match as before. If none exists, it may map a unique observed loop after removal of at most two **internal** annotated residues, with at least 80% observed coverage and the conserved C and W flanks present in the same VHH chain. Only observed residues enter the site ranking. Ambiguous matches, missing anchors, larger gaps and absent CDR annotation remain failures. No other site-selection method, solver, energy, budget or quality gate changes.
- **Scope:** this changes the CDR baseline and thus scientific outputs for affected cases. It is for a newly frozen run or a separately documented continuation that invalidates and reruns affected QC outputs and all dependent analyses. The 40 historical failures remain part of the original run record; they are not relabelled as successful.

## A43. Exact movable-coordinate minimization and rotamer geometry screening

- **Trigger:** the 2026-09-29 structural run failed all 28 frozen validation targets. Every inspected seed had a nonconverged fixed-backbone relaxation under the 200-iteration cap; some reconstructed candidates also had sub-0.4-A nonbonded overlaps. In 4AQ1, a candidate placed Tyr H:59 CE1 and HH about 0.001 A apart. These are recorded physical failures, not missing metrics.
- **Numerical correction:** final and single-candidate Amber14 relaxations now minimize only the declared movable Cartesian coordinates with SciPy L-BFGS-B, using the same OpenMM potential and exact OpenMM forces. All frozen coordinates remain unchanged. The prior OpenMM mass-zero minimizer could stop with a large residual force: on a separately saved 4AQ1 stage-1 structure, another 1000 OpenMM minimizer iterations left the movable-force RMS near 93 kJ/mol/nm, while the explicit movable-coordinate optimizer reduced it to about 4.7 in 122 iterations. This is a numerical diagnostic on an earlier saved structure, not a claim that the current validation run passes.
- **Candidate quality:** before retaining rotamers, topology-excluded intramolecular pairs closer than the existing absolute 0.4-A floor are excluded and recorded with residue, raw variable, chi angles and atom pair. If a site lacks its required number of valid states, construction fails with an explicit candidate-geometry audit. Cross-residue interactions remain in the Amber energy/QUBO; this screen addresses internally impossible conformers only.
- **Budgets and acceptance:** candidate preparation retains its 100-iteration cap; final relaxation raises its cap from 200 to 1000, with early optimizer termination allowed. The force threshold remains 10 kJ/mol/nm, the geometry floor remains 0.4 A, and every failed method/control output stays in the denominator. The optimizer's iteration count, evaluation count and stop reason are saved. The code and scientific-state changes require a newly frozen run; historical failed artifacts must not be relabelled or silently reused. Server-side validation on the full frozen target set remains necessary.

## A44. Physically valid perturbed recovery inputs

- **Status — superseded by A45 (2026-10-01):** A44 and A45 were written in parallel for the same failure and the same policy, then merged. **A45's implementation is the one in the code:** up to `structure_experiment.perturbation_max_attempts` (32) draws per seed, attempt 0 using the seed itself and later attempts using derived child seeds, recorded in a hash-bound attempt ledger. A44's shared-generator rejection sampler, its 1,000-attempt default and `perturb_valid_input` were removed so that one implementation remains. Two parts of A44 are kept: the orchestrator passes the configured `data_audit.min_interresidue_heavy_distance_angstrom` to the generated-input check, and that value is now the floor A45's audit actually applies. Before this reconciliation the audit used a fixed 1.0 Å while the ledger recorded the configured value; the two agree under the current config, so no completed result is affected. The diagnosis below and its open items remain valid under A45.
- **Trigger:** in run `experiments_full_run_20260930_010659` (with A43), 13 of 28 frozen validation targets failed. Every solver output passed physical acceptance in every seed with a quality summary (QAOA, SA, uniform and greedy each 129/129); only the relax-only control failed, in 14 of 129 seeds (`extreme_nonbonded_overlap` 9, `relaxation_not_converged` 14), with residual movable-force RMS between 3.7×10⁸ and 6.2×10⁸ kJ/mol/nm against a 10 kJ/mol/nm limit. The 56 failed control rows are those 14 seeds attached to four method rows each.
- **Cause:** the shared recovery input was generated by rotating every chi of every Active residue by a signed 40–120° draw with no validity check of any kind (its recorded protocol said "no clash/energy/reference-based rejection"). Such a draw can drive side chains into one another. The relax-only control relaxes that input directly; solver methods rebuild Active side chains from the A43-screened rotamer library and never inherit it. The failures therefore came from constructing an impossible input, not from any solver or from the relaxation shared by all methods.
- **Change:** perturbations are rejection-sampled until the input meets two validity criteria the protocol already applies: the A24 inter-residue heavy-atom floor that deposited inputs pass (`data_audit.min_interresidue_heavy_distance_angstrom`, 1.0 Å, same definition: protein heavy atoms on distinct residues, absolute distance) and the all-atom near-coincidence floor that outputs and candidates pass (0.4 Å, A28/A43). One generator seeded with the seed serves every attempt, and the per-draw code is AST-identical to the former perturbation body, so a seed whose first draw is valid keeps its former input bit for bit. Each `perturbation.json` records the accepted attempt and every rejected draw with its reasons and closest atom pair. A seed with no valid draw in 1,000 attempts is a recorded `generated_input` failure that stays in the denominator. Chi1-only mode is treated the same way.
- **Not changed:** no reference-, energy- or outcome-based criterion; the perturbation angle range; the relaxation shared by all methods, including the control; physical acceptance; the fail-closed statistics gate; failure accounting. The control is not given a different minimizer, and failed control rows are not excluded from inference.
- **Claim boundary:** the recovery task now starts from the original perturbation distribution conditioned on physical validity, and is reported that way. Inputs that a draw would have placed inside other atoms are no longer part of the task.
- **Not yet verified:** the per-seed closest-atom pairs of the 14 failed inputs were not inspected, so it is not yet shown that every one violates these floors. If a control still fails on an input that passes them, that is a separate finding to investigate, not a reason to tighten the floor. This amendment does not explain the other 4 failed frozen targets, which have no physical-acceptance failure, nor the `energy_calibration` diagnostic's cv_rmse of about 4×10¹⁹, which remains diagnostic-only under A23.
- **Results inspected before this amendment:** per-method acceptance counts, failure reasons and residual forces from the run above. The criteria were taken unchanged from A24 and A28/A43 rather than chosen on those results. Whether method-comparison recovery metrics in that run's `real_complex_report.md` were inspected requires investigator confirmation.
- **Affected results:** the perturbed inputs of seeds whose first draw was invalid, and with them those seeds' initial RMSD, relax-only control and improvement metrics. This requires a newly frozen run; the failed run's artifacts are kept and not relabelled.

## A45. Prospective geometry-qualified generated recovery inputs

- **Trigger and policy change:** deliberately perturbed recovery inputs could carry catastrophic atom overlap into the relax-only control. For future frozen runs, sample up to 32 deterministic perturbations per target and seed, taking the first that passes the already declared topology-aware 0.4-A all-atom near-coincidence floor and 1.0-A inter-residue heavy-atom input floor. Applying the source heavy-atom floor to generated inputs is a new sampling policy, even though the numerical floors are unchanged.
- **Blinding and accounting:** every attempt, seed and geometry audit is written to a hash-bound ledger. Selection never reads the native reference match, Amber energy, QUBO value or solver outcome. The selected input is shared by QAOA, SA, uniform, greedy and relax-only. Exhausting the cap fails that seed and retains it in the frozen denominator; downstream stage acceptance is not weakened. Candidate and method-output geometry failures remain distinct from input admission failures.
- **Scope:** generated-input selection changes the task distribution and requires a newly frozen protocol and independent prospective validation. Runs and targets already inspected during September debugging remain historical repair assessments. The full pipeline emits a diagnostic incomplete report and a nonzero exit if physical controls or any other gate still fail; it does not fabricate a completed formal result.

## A46. Movable-coordinate L-BFGS with a per-atom step cap

- **Trigger:** in run `experiments_full_run_20260930_010659`, 2 of the 14 failed relax-only seeds (7NXX seed 43, 7ZRA seed 44) have inputs that pass the A45 generated-input floors. Rerunning 7NXX seed 43 with the A43 minimizer still failed: Tyr H:55 HH collapsed onto its own ring carbon CE2 (0.00 A), movable-force RMS 2.67×10⁸ kJ/mol/nm, max 2.72×10⁹, stop reason RELATIVE REDUCTION after 67 iterations. The input was valid; the relaxation created the overlap.
- **Cause:** in the Amber14 force field used here, polar hydrogens of type `protein-HO` (Tyr HH, Ser HG, Thr HG1) have Lennard-Jones ε = 0, so nothing repels them at short range, and their positive charge (Tyr HH +0.3992) is attracted without bound to negatively charged atoms such as Tyr CE2 (−0.2341). From a physical geometry that singularity lies behind valence (bond and angle) barriers, so a local descent does not reach it. The A43 minimizer, SciPy L-BFGS-B, starts with a step of unit length (1 nm) along the steepest descent, and its line search accepts any step that lowers the energy, so it can jump over the barrier. In a one-atom reproduction (a tethered charge near an unscreened opposite charge), unbounded L-BFGS-B ended 0.0000 nm from the singularity in every geometry tried.
- **Change:** the movable coordinates are minimized with a limited-memory BFGS (10 correction pairs) and a backtracking Armijo line search, written in the repository. Every accepted iteration moves each movable atom by at most `MINIMIZER_MAX_ATOM_STEP_NM` = 0.03 nm (0.3 A). The quasi-Newton history is kept across iterations, so the total distance an atom may travel is limited only by the iteration cap. The minimizer stops when the exact movable-force RMS is at most 10 kJ/mol/nm (`force_tolerance`), at the iteration cap (`iteration_cap`), or when no energy-decreasing step exists even along the steepest descent (`line_search_failed`). It has no relative-energy stop. Provenance records the minimizer name `lbfgs_exact_movable_capped_atom_step`, the step cap, the history length, iterations, energy/force evaluations, the stop reason and the number of history resets.
- **GPU force overflow:** a second, independent problem was found on 7NXX seed 43. Its input already carries Tyr H:55 CZ 0.505 A from the frozen hydrogen H:58 Gly HA2. That passes the A45 floors: the all-atom floor is 0.4 A, and the 1.0 A floor applies only to heavy–heavy pairs. Sub-angstrom hydrogen contacts are common in accepted perturbed inputs. In the local check below (seeds 42–81), 35 seeds had a qualified input, and its closest moved-atom contact was 0.41–0.95 A in every one. A46 relaxed all 15 of those it was run on (seeds 42–57) on the CPU platform. A hydrogen-inclusive 1.0 A floor would have rejected all 40 seeds tried within 32 attempts, so it is not an admissible repair. On the server, the 7NXX relaxation instead stopped after one iteration, with three different atoms reporting the identical force magnitude 3.72×10⁹ kJ/mol/nm, which is 2³¹·√3. OpenMM's GPU platforms accumulate forces as 64-bit fixed point with 32 fractional bits, so a force component cannot exceed 2³¹ ≈ 2.1×10⁹ kJ/mol/nm: larger sums saturate or wrap around, and a wrapped value can even look small. The local input above has a largest movable force component of 3.6×10¹¹ kJ/mol/nm. On a GPU, the minimizer therefore received a gradient that did not match the energy, and the line search failed. The post-relaxation force audit read the same GPU forces, so an unrelaxed overlap could in principle also have been mis-audited.
- **Overflow repair:** whether a force can exceed the limit is decided from geometry, never from the GPU's own output. At construction, the strongest r⁻¹² wall of the System is found: ε·σ¹² maximized over Lorentz–Berthelot combinations of the particle types and over every nonzero NonbondedForce exception. From it follows the distance at which a single pair's Lennard-Jones force reaches 2³¹/64 kJ/mol/nm: 1.375 A for the Amber14 vacuum System. If any interacting pair involving a movable atom is closer than that, energy and forces are computed on the double-precision Reference platform from the same System. Otherwise the GPU result is used unchanged. Fully excluded pairs (1-2, 1-3) are ignored, and 1-4 pairs count. The final force audit uses the same rule. Provenance records the Reference evaluation count, the distance and the audit platform. The CPU and Reference platforms have no fixed-point accumulator and are never redirected. In the local check, run with a GPU-labelled CPU context, 24–69 of about 300–450 evaluations per relaxation went to Reference, at about 0.2 s each for a 3,873-atom complex. Structural-stage time therefore rises. Unit tests reproduce the failure with an emulated wrapping accumulator: without the rule, the minimizer stops unconverged; with it, it converges.
- **Revision before use:** the first version of A46 (commit 59975ea) kept SciPy L-BFGS-B and bounded each 50-iteration stage to ±0.03 nm per coordinate. It never produced a formal result. It removed the collapse but limited each coordinate to 0.6 nm over 1000 iterations. On the server, 7ZRA seed 44 then reached force RMS 35.9 (from 3.9×10⁸) but no convergence within the cap. Locally, on one prepared 7ZRA VHH–antigen complex with eight interface sites, its seed-42 input needed 1644 iterations. The per-iteration cap above replaces it.
- **Development check (not a validation result):** that same local complex was prepared from PDB 7ZRA (chain H with antigen chain D; an approximately placed OXT replaced PDBFixer, which was unavailable) with Active sites H:37, H:59, H:98, H:56, H:57, H:58, H:64, H:96, 1000-iteration cap and the CPU platform. A45 inputs were drawn for seeds 42–57; seed 51 had no qualified input in 32 attempts. In one run over the 15 remaining seeds, the A46 minimizer converged in all 15, in 249–528 iterations. The A43 minimizer converged in 12 and failed on seeds 42, 47 and 48, ending at positive total energy (+33,958, +221 and +14,068 kcal/mol). Of the 12 seeds where both converged, 6 reached the same energy within 1 kcal/mol; A46 ended lower in 2 (by 1,257 and 3,748 kcal/mol, where A43 stayed in a high-energy state), and A43 ended lower in 4 (by 5–208 kcal/mol, different local minima). A46 stops once the force criterion is met, whereas A43 continued past it, so A46's final force RMS is typically 8–10 rather than 1–8 kJ/mol/nm. Multithreaded CPU force summation is not bitwise reproducible, and a repeated run of seeds 46–57 changed iteration counts and turned one A43 outcome (seed 47) from passed to failed; A46 converged on every seed in both runs. These targets, sites and seeds are not the frozen validation set, and this check is not evidence about the study's results.
- **Not changed:** the Amber energy function, the movable/frozen atom sets, iteration caps (100 for candidate preparation, 1000 for final relaxation), the 10 kJ/mol/nm force threshold, the 0.4-A geometry floor, the A45 generated-input policy, and failure accounting. The same relaxation code serves QAOA, SA, uniform, greedy and relax-only, so the control still gets no different minimizer. The step cap limits step length only; it is not a restraint and adds no energy term.
- **Claim boundary:** the minimizer and the platform used for close-contact evaluations change, so relaxed structures, energies, candidate-preparation outcomes and every downstream metric can change for every method. A newly frozen run is required; artifacts from earlier runs stay as recorded and are not relabelled. On the server, the capped-step minimizer without the overflow repair (commit 80d1394) relaxed 7ZRA seed 44 (input closest 0.73 A, final 1.52 A, force RMS 9.31, 302 iterations). 7NXX seed 43 with the overflow repair is not yet verified there. The 4 failed frozen targets with no physical-acceptance failure remain unexplained.

## A47. Numerically sound QUBO construction for impossible discrete states

- **Trigger:** in run `experiments_full_run_20260930_010659`, 11 of 140 validation seeds stopped with an execution error before any physical evaluation. These are the 4 failed frozen targets that had no physical-acceptance failure. Six seeds (5FOJ 44, 6I8G 45 and 46, 6RVC 45, 7WKI 43, 9GV3 42) failed the QUBO→Ising equivalence check in `to_quantum_instance` ("Analytical Ising coefficients do not reconstruct Q", "Ising constant offset is inconsistent with Q", "QUBO/Ising mismatch 1.863e-09 exceeds 1.000e-09"). Five (7R1Z 42–45, 9FVC 42) failed the all-atom precision budget in `build` (float64 roundoff bound 0.0093–4.27 kcal/mol against 0.001). That bound requires a total coefficient magnitude above 1.4×10¹¹ kcal/mol.
- **Cause 1, a fixed absolute tolerance:** `build` validated the conversion against a float64 roundoff bound scaled to the coefficients. `to_quantum_instance` repeated the same check with a fixed absolute tolerance of 1e-9 kcal/mol. For coefficients of order 10⁷ or larger, rounding alone exceeds 1e-9, so an exact conversion was rejected. A unit test reproduces this at 10⁸ scale.
- **Change 1:** both checks use the same bound, `ising_roundoff_tolerance` = max(1e-9, 32·ε·(|offset| + Σ|Q| + 1)), which was already the bound used in `build`. QUBOs with ordinary coefficients keep exactly 1e-9. A wrong coefficient is still rejected (tested).
- **Cause 2, raw energies of impossible states:** candidate rotamers were screened only for intra-residue overlaps (A43). Overlaps with fixed atoms, and between states at two sites, entered the QUBO as raw Amber energies, which reach 10¹¹–10¹⁴ kcal/mol at atom-on-atom contact. Locally, on the prepared 7ZRA complex (8 sites, A45 seed-42 input), a raw Tyr H:37 candidate placed its HH 0.20 A from the frozen backbone O of Arg H:45. Amber HO hydrogens have ε = 0 (A46), so candidate relaxation pulled HH onto that O (2.7×10⁻⁶ A apart) until the energy was no longer finite. On a platform with double-precision energies, the same collapse would instead give that candidate an enormously negative energy, and the solvers would prefer a physically meaningless state.
- **Change 2a, minimizer:** a line-search step may not bring an interacting atom pair (1-2 and 1-3 excluded) that is closer than the existing 0.4-A near-coincidence floor any closer than it already is. A vetoed step is backtracked like one without sufficient decrease. A physical minimum never has such a pair, so the rule only acts on overlaps. Provenance records the number of vetoes. A unit test reproduces the collapse of a zero-ε hydrogen started 0.2 A from an acceptor without the rule and shows it does not happen with it.
- **Change 2b, geometry-forbidden discrete states:** two kinds of state are geometry-forbidden. A candidate state is forbidden if any of its atoms is closer than 1.0 A to an interacting atom outside every Active side chain. A pair of states at two different sites is forbidden if any of their interacting atoms are closer than 1.0 A. Both are judged on unrelaxed discrete coordinates, after the existing candidate relaxation, for all elements. The 1.0-A floor equals the A24/A45 heavy-atom floor and is shorter than every covalent X–H bond. Any such contact costs at least about 10⁵ kcal/mol unrelaxed, so no admissible optimum contains one. Energies of forbidden states are never evaluated. They carry a penalty F = 2B + 1 kcal/mol, where B is the sum of the absolute admissible single and pair energies. Every admissible assignment has relative energy within ±B, so any assignment containing a forbidden state lies above all of them. The one-hot λ bound is computed with F included. The decomposition anchor is the first admissible assignment in the former per-site energy order, which is the same anchor as before whenever the former anchors were admissible. The 12-sample exact-equivalence check (1e-4 kcal/mol in vacuum) draws only admissible assignments. If no admissible assignment exists, the seed fails with category `candidate_geometry`, in the denominator. The metadata record the floor, F, the counts of forbidden singles and pairs, and up to 100 examples with their closest atoms.
- **Not changed:** the Amber energy function; the energies of every admissible state and pair; the candidate pools and their retention; solver budgets; the force threshold; the geometry floors; failure accounting. XY-QAOA, SA, uniform and greedy all read the same physical terms, so all solvers see the same forbidden penalty.
- **Claim boundary:** the QUBO is exact on geometry-admissible assignments, not on forbidden ones; reports must say so. The QUBO coefficient scale changes wherever forbidden states occur, and with it QAOA's `max_coefficient` parameter scaling for those instances. That is a change to the method, so a newly frozen run is required. The 11 historical failures remain in that run's record. Local development check (prepared 7ZRA complex, 4 sites H:37/H:56/H:57/H:59, Dunbrack 2010 text library, CPU platform): 14 retained candidates, no forbidden states, roundoff bound 2.4×10⁻⁶ kcal/mol, and the quantum instance built. With all 8 sites, the same input and the double-precision Reference platform (candidate relaxation off, so raw states are kept), the QUBO built with 6 forbidden single states and 4 forbidden pairs. F was 8,970 kcal/mol, the exact-equivalence error 2.8×10⁻¹⁰ kcal/mol (check 1e-4) and the roundoff bound 9.0×10⁻⁹. The 30-qubit quantum instance built. With candidate relaxation on, the 8-site candidates relaxed without collapse on the CPU platform. The vacuum exactness check then failed there: that platform computes nonbonded energies in single precision, about 0.01–0.1 kcal/mol at these magnitudes. Formal runs use CUDA double precision, so this exactness check is not meaningful on the CPU platform. Whether 7R1Z and 9FVC now build, and how often forbidden states occur in the frozen set, is not yet verified.
