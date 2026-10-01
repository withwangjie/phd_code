# Structure preparation and physical acceptance

This implementation strengthens the independent atomistic endpoint. The coarse
QUBO remains a solver benchmark. The training-only Amber calibration remains a
diagnostic with unused coefficients; this change does not establish a mapping
between coarse energy and structural accuracy.

## 1. Experimental input before selection

Keep the existing canonical-heavy-atom completeness rule, coherent alternate
conformer selection, graph/source coordinate agreement and exact CDR-H3 mapping.
Missing internal atoms or unobserved CDR segments are not reconstructed. Record
the reason in `eligibility.json`, including `failure_category` and the audit.
An annotation that cannot be uniquely mapped is recorded separately from an
atom-geometry failure; it is not proof that the deposited structure is wrong.

The cleaned protein topology now checks consecutive residues using the declared
C--N bond and measured distance (existing 1.9 A upper bound). Author-number gaps,
including IMGT gaps, are not used to infer missing residues. Internal breaks fail
this endpoint's continuous-protein preparation contract; independently modelled
fragments would need a separate, declared termination protocol.

The existing source-data screen at 1.0 A for distinct-residue heavy atoms (A24)
continues to apply. A supplementary topology-aware 0.4 A absolute distance floor
detects catastrophic nonbonded near-coincidences. It excludes bonded and 1--3
pairs, includes 1--4 pairs, and records the offending atoms. **0.4 A here is an
absolute distance chosen for this implementation, not a literature-established
universal cutoff or MolProbity's 0.4 A van der Waals overlap.**

## 2. Added hydrogens and generated states

Save both source and hydrogenated preparation audits in
`structure_preparation_audit.json`; the native recovery source also has
`native_preparation_audit.json`. Hydrogens participate in the new geometry audit.
Existing side-chain rotation carries bonded downstream hydrogens with the heavy
atoms. Do not infer that every hydrogen orientation is therefore physically good.

Save `candidate_geometry_audit.json` for the retained atomistic candidates. Each
candidate is assessed with the other Active sites fixed at the decomposition
anchors. This is not exhaustive validation of all multi-site assignments, nor a
new candidate filter. If decomposition fails, `allatom_build_failure.json`
preserves available preparation and candidate diagnostics.

For a newly frozen run under A45, generated recovery inputs use a fixed maximum
of 32 deterministic draws per target and seed. The first draw whose topology,
inter-residue heavy-atom distance (at least 1.0 A), and topology-excluded
all-atom nonbonded distance (at least 0.4 A) pass is used for every method and
the relax-only control. Every rejected draw is recorded. If none passes, that
seed fails and remains in the frozen denominator. Neither reference similarity,
energy nor a solver result enters selection. These are generated-input rules;
candidate or solver-output failures remain method failures, not source-data
exclusions. The draw selection changes the tested input distribution and cannot
be applied retrospectively to relabel previous runs.

## 3. Uniform relaxation and measured acceptance

All methods and the relax-only control keep the same frozen atom mask and
iteration budget. There is no reference-guided state selection, energy clipping,
new random start or silent increase of that budget. The exact-potential movable
coordinate optimizer may restart its L-BFGS-B history within that same total
iteration cap when the measured residual force remains high (A44); every restart
and the final force are recorded. This numerical continuation does not change
the physical acceptance rule or guarantee that a difficult pose will pass.

For each output, record discrete and relaxed force-group energies, closest
nonbonded pair, extreme pair count and residual forces. Force groups distinguish
bond, angle, torsion and nonbonded contributions; the nonbonded group still
combines electrostatics and Lennard-Jones energy.

The acceptance criterion uses the RMS **force components on movable atoms**,
with tolerance 10 kJ/mol/nm. Frozen-atom forces do not contribute. Record the
maximum movable-atom force too. A capped minimization is accepted as converged
only if the measured criterion passes. Zero iterations are explicitly `skipped`.
This operational rule is stricter than inferring convergence from OpenMM's return
or energy decrease, and does not assert a global minimum. Optional stage-2 loop
relaxation measures forces of its restrained objective and reports the separate
physical energy and final geometry.

Final outputs must pass topology, the extreme-overlap rule and measured
convergence. JSON and CSV retain failed outputs and their reference metrics. The
experiment evaluates all four methods plus relax-only before rejecting physical
acceptance; it does not write `completed.json` on rejection. Execution exceptions
can interrupt this evaluation and are counted as failed seeds with missing
outputs visible against the requested denominator.

`structure_quality_summary.json` counts evaluated and failed outputs.
`recovery_quality_summary.json` records requested seeds, failed seeds, requested
method outputs and retained invalid rows. A failed child's return status now
propagates to the recovery benchmark and the frozen target closure. Descriptive
means identify valid outputs explicitly; formal inference must reject a table
containing failed outputs rather than silently dropping them.

## 4. Scientific endpoint and next decision

Continue reporting side-chain RMSD, complete-chi recovery, contacts, interface
metrics and existing clash diagnostics, with relax-only as a control. Energy
decrease alone is not evidence of structural improvement or binding affinity.
The new absolute floor detects catastrophes only; passing it does not establish
normal packing, protonation correctness or experimental-quality geometry.

First verify these checks on development/training structures using the pinned
server PyRosetta and CUDA environment. Record the counts and convergence
distribution before viewing an untouched endpoint. Any changes to sampling,
repair, tolerance or relaxation budget require their own frozen amendment.
Reconsider Amber calibration only after preparing a physically valid, sufficiently
diverse training dataset and showing family-grouped out-of-sample performance.
Do not lower its current acceptance gates to make it pass.

## References and what they support

1. Gordon et al. (2023), *A comparison of the binding sites of antibodies and
   single-domain antibodies*, Frontiers in Immunology 14:1231623,
   [doi:10.3389/fimmu.2023.1231623](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2023.1231623/full).
   Their dataset excludes missing CDR backbone/anchor atoms: this supports
   explicit completeness criteria, not silently repairing this benchmark's loop.
2. Huang et al. (2020), *FASPR: an open-source tool for fast and accurate protein
   side-chain packing*, Bioinformatics 36:3758--3765,
   [doi:10.1093/bioinformatics/btaa234](https://academic.oup.com/bioinformatics/article/36/12/3758/5817305).
   Packing uses complete backbone input and considers fixed-environment and
   side-chain pair interactions. It does not validate our numerical thresholds.
3. [OpenMM LocalEnergyMinimizer documentation](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.LocalEnergyMinimizer.html).
   Documents the RMS-force tolerance and finite iteration cap. Our movable-only
   acceptance calculation and explicit skipped state are recorded study rules.
4. [OpenMM nonbonded force theory](https://docs.openmm.org/latest/userguide/theory/02_standard_forces.html).
   The Lennard-Jones repulsive term grows as inverse distance to the twelfth
   power. A near-zero separation can therefore explain enormous raw Amber
   energies without establishing that those energies are useful fitting targets.
5. Chen et al. (2010), *MolProbity: all-atom structure validation for macromolecular
   crystallography*, Acta Crystallographica D66:12--21,
   [doi:10.1107/S0907444909042073](https://doi.org/10.1107/S0907444909042073).
   Supports hydrogen-aware all-atom validation. MolProbity overlap/clashscore
   must be reported separately from the absolute-distance diagnostic here.
