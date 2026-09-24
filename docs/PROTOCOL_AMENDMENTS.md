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

## A1. Primary coarse solver endpoint: resource-normalized time-to-solution

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

## A3. Earlier amendments on 2026-09-24

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
