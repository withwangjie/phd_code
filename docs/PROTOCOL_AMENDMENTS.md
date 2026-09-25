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
| A1 | Resource-normalized time-to-solution (`log10_qts99`) replaces best-of-N gap (secondary since A7) | qc_benchmark, statistics, external_validation, final_report | to be completed |
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
