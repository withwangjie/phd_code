"""Topology-aware physical diagnostics, independent of reference accuracy.

The 0.4 A floor detects catastrophic nonbonded near-coincidences only. It is
an absolute distance, NOT MolProbity's 0.4 A van der Waals overlap criterion.
No atoms, states, energies or failed method outputs are silently removed.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from nanoqc.structure.residue_tables import PEPTIDE_BOND_MAX_C_N_ANGSTROM


QUALITY_SCHEMA = "structure_physical_quality_v1"
EXTREME_NONBONDED_FLOOR_ANGSTROM = 0.4
RELAX_FORCE_TOLERANCE_KJ_MOL_NM = 10.0
GENERATED_INPUT_HEAVY_FLOOR_ANGSTROM = 1.0


class StructureQualityError(ValueError):
    """An auditable failure whose category is independent of solver score."""

    def __init__(self, message: str, *, category: str, audit: dict):
        super().__init__(message)
        self.category = category
        self.audit = audit


def generated_input_geometry_audit(topology, positions_nm: np.ndarray) -> dict:
    """Apply the existing source-heavy and all-atom hard floors to a pose.

    This is a prospective generated-input screen.  It neither evaluates the
    reference structure nor uses Amber energy or a solver outcome.
    """
    physical = topology_geometry_audit(topology, positions_nm)
    atoms = list(topology.atoms())
    xyz = np.asarray(positions_nm, dtype=float) * 10.0
    heavy = [atom for atom in atoms if atom.element is not None
             and atom.element.symbol.upper() != "H"]
    overlaps = []
    if len(heavy) > 1:
        coordinates = xyz[[atom.index for atom in heavy]]
        tree = cKDTree(coordinates)
        for left, right in sorted(tree.query_pairs(GENERATED_INPUT_HEAVY_FLOOR_ANGSTROM)):
            a, b = heavy[left], heavy[right]
            if a.residue.index == b.residue.index:
                continue
            distance = float(np.linalg.norm(coordinates[left]-coordinates[right]))
            if distance >= GENERATED_INPUT_HEAVY_FLOOR_ANGSTROM:
                continue
            overlaps.append(dict(
                distance_angstrom=distance,
                atoms=[f"{a.residue.chain.id}:{a.residue.id}:{a.residue.name}:{a.name}",
                       f"{b.residue.chain.id}:{b.residue.id}:{b.residue.name}:{b.name}"],
            ))
    return dict(
        accepted=bool(physical["topology_passed"] and physical["geometry_passed"] and not overlaps),
        source_heavy_floor_angstrom=GENERATED_INPUT_HEAVY_FLOOR_ANGSTROM,
        interresidue_heavy_overlap_count=len(overlaps),
        interresidue_heavy_overlaps=overlaps[:100],
        interresidue_heavy_pair_list_truncated=len(overlaps)>100,
        all_atom=physical,
        selection_definition="first deterministic geometry-qualified perturbation; no reference, energy or solver selection",
    )


def assert_structural_row_acceptance(rows: list[dict]) -> None:
    """Fail closed on new-contract failed rows; never remove them for inference.

    Historical CSVs without either status column keep their historical contract.
    Once either new field is present, every row must satisfy both status fields.
    """
    if not any("evaluation_status" in row or "control_evaluation_status" in row for row in rows):
        return
    invalid=[dict(target=row.get("target"),seed=row.get("seed"),method=row.get("method"),
                  status=row.get("evaluation_status"),control_status=row.get("control_evaluation_status"))
             for row in rows if row.get("evaluation_status")!="passed" or
             row.get("control_evaluation_status")!="passed"]
    if invalid:
        raise ValueError(f"Physical acceptance failed for {len(invalid)} structural rows; "
                         f"no rows excluded for inference: {invalid[:10]}")


def topology_geometry_audit(topology, positions_nm: np.ndarray) -> dict:
    """Check declared peptide bonds and all-atom, topology-excluded proximity.

    Excludes directly bonded and 1-3 atom pairs; 1-4 pairs remain diagnostic.
    Chain breaks are not inferred from author number gaps (IMGT has gaps).
    They require missing or overlong C-N bonds between consecutive residues
    within the actual topology chain. Hydrogens participate in this audit.
    """
    atoms = list(topology.atoms())
    xyz = np.asarray(positions_nm, dtype=float) * 10.0
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError("Physical audit requires finite atom-aligned positions in nm")
    if [a.index for a in atoms] != list(range(len(atoms))):
        raise ValueError("Physical audit requires consecutive topology atom indices")
    labels = [f"{a.residue.chain.id}:{a.residue.id}"
              f"{a.residue.insertionCode.strip()}:{a.residue.name}:{a.name}" for a in atoms]
    adjacency = {a.index: set() for a in atoms}
    for a, b in topology.bonds():
        adjacency[a.index].add(b.index)
        adjacency[b.index].add(a.index)
    excluded = {}
    for i in adjacency:
        neighbours = set(adjacency[i])
        for j in adjacency[i]:
            neighbours.update(adjacency[j])
        excluded[i] = neighbours | {i}

    breaks = []
    for chain in topology.chains():
        residues = list(chain.residues())
        for previous, current in zip(residues, residues[1:]):
            left = {a.name: a for a in previous.atoms()}
            right = {a.name: a for a in current.atoms()}
            if "C" not in left or "N" not in right:
                breaks.append(dict(left=f"{chain.id}:{previous.id}{previous.insertionCode.strip()}",
                                   right=f"{chain.id}:{current.id}{current.insertionCode.strip()}",
                                   reason="missing_peptide_atom", distance_angstrom=None))
                continue
            i, j = left["C"].index, right["N"].index
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            if j not in adjacency[i] or distance >= PEPTIDE_BOND_MAX_C_N_ANGSTROM:
                breaks.append(dict(left=labels[i], right=labels[j],
                                   reason="missing_peptide_bond" if j not in adjacency[i] else "overlong_peptide_bond",
                                   distance_angstrom=distance))

    closest = None
    collisions = []
    if len(atoms) > 1:
        tree = cKDTree(xyz)
        # Increase the neighbour list if bonded atoms hide the closest
        # nonbonded partner. Never use an N-by-N distance matrix.
        for i in range(len(atoms)):
            k = min(8, len(atoms))
            while True:
                distances, indices = tree.query(xyz[i], k=k)
                eligible = [(float(d), int(j)) for d, j in
                            zip(np.atleast_1d(distances), np.atleast_1d(indices))
                            if int(j) not in excluded[i]]
                if eligible:
                    distance, j = min(eligible)
                    if closest is None or distance < closest[0]:
                        closest = (distance, i, j)
                    break
                if k == len(atoms):
                    break
                k = min(2*k, len(atoms))
        for i, j in sorted(tree.query_pairs(EXTREME_NONBONDED_FLOOR_ANGSTROM)):
            if j not in excluded[i]:
                distance = float(np.linalg.norm(xyz[i] - xyz[j]))
                if distance < EXTREME_NONBONDED_FLOOR_ANGSTROM:
                    collisions.append(dict(distance_angstrom=distance, atoms=[labels[i], labels[j]]))
    return dict(schema=QUALITY_SCHEMA, atom_count=len(atoms),
                topology_passed=not breaks, peptide_breaks=breaks,
                geometry_passed=not collisions,
                extreme_nonbonded_floor_angstrom=EXTREME_NONBONDED_FLOOR_ANGSTROM,
                extreme_nonbonded_pair_count=len(collisions),
                extreme_nonbonded_pairs=collisions[:100],
                pair_list_truncated=len(collisions) > 100,
                closest_nonbonded_pair=None if closest is None else
                dict(distance_angstrom=closest[0], atoms=[labels[closest[1]], labels[closest[2]]]),
                definition="all atoms; bonded and 1-3 pairs excluded; absolute near-coincidence floor, not MolProbity")


def relaxation_force_audit(forces_kj_mol_nm: np.ndarray, movable: set[int], *,
                           iterations: int) -> dict:
    """Measure residual force on movable DOFs, excluding frozen atom forces.

    Convergence is a measured RMS-force criterion, not an assertion that the
    iteration cap was reached or a global energy minimum was obtained.
    """
    force = np.asarray(forces_kj_mol_nm, dtype=float)
    if force.ndim != 2 or force.shape[1] != 3 or not np.isfinite(force).all():
        raise ValueError("Nonfinite or malformed forces after relaxation")
    if not movable or min(movable) < 0 or max(movable) >= len(force):
        raise ValueError("Relaxation force audit requires valid movable atom indices")
    free = force[sorted(movable)]
    rms = float(np.sqrt(np.mean(free**2)))
    maximum = float(np.max(np.linalg.norm(free, axis=1)))
    return dict(relaxation_status="skipped" if iterations == 0 else
                ("converged" if rms <= RELAX_FORCE_TOLERANCE_KJ_MOL_NM else "not_converged"),
                relaxation_converged=bool(iterations > 0 and rms <= RELAX_FORCE_TOLERANCE_KJ_MOL_NM),
                movable_force_rms_kj_mol_nm=rms, movable_force_max_kj_mol_nm=maximum,
                force_tolerance_kj_mol_nm=RELAX_FORCE_TOLERANCE_KJ_MOL_NM,
                convergence_definition="RMS of force components on movable atoms; frozen forces excluded")


# ---------------------------------------------------------------------------
# Generated recovery inputs (A44)
# ---------------------------------------------------------------------------
# Perturbing every chi of every Active residue by a random signed angle can
# drive side chains into one another. Such an input is physically impossible,
# so the relax-only control started from it cannot converge, while solver
# methods (which rebuild Active side chains from the screened rotamer library)
# never inherit it. A generated input must therefore meet the same validity
# criteria the protocol already applies elsewhere: the A24 inter-residue
# heavy-atom floor that deposited inputs pass, and the all-atom near-coincidence
# floor that outputs and candidates pass. Neither looks at the reference
# structure or ranks by energy.
DEFAULT_MAX_PERTURBATION_ATTEMPTS = 1000


def interresidue_heavy_floor_audit(topology, positions_nm: np.ndarray,
                                   floor_angstrom: float) -> dict:
    """Heavy-atom pairs on distinct residues closer than ``floor_angstrom``.

    Same definition as the A24 data-audit screen
    (``audit_all_datasets.interresidue_heavy_overlap``): protein heavy atoms,
    pairs owned by different (chain, residue), no bonded exclusion, absolute
    distance.
    """
    floor = float(floor_angstrom)
    if not np.isfinite(floor) or not 0 < floor <= 1.0:
        raise ValueError("inter-residue heavy-atom floor must be in (0, 1.0] A")
    atoms = list(topology.atoms())
    xyz = np.asarray(positions_nm, dtype=float) * 10.0
    if xyz.shape != (len(atoms), 3) or not np.isfinite(xyz).all():
        raise ValueError("Generated input positions are malformed or nonfinite")
    heavy = [i for i, atom in enumerate(atoms)
             if atom.element is not None and atom.element.symbol != "H"]
    owners = {i: (atoms[i].residue.chain.index, atoms[i].residue.index) for i in heavy}
    labels = {i: f"{atoms[i].residue.chain.id}:{atoms[i].residue.id}:"
                 f"{atoms[i].residue.name}:{atoms[i].name}" for i in heavy}
    count = 0
    nearest = None
    closest = None
    if len(heavy) >= 2:
        coordinates = xyz[heavy]
        for a, b in cKDTree(coordinates).query_pairs(floor, output_type="ndarray"):
            i, j = heavy[int(a)], heavy[int(b)]
            if owners[i] == owners[j]:
                continue
            distance = float(np.linalg.norm(xyz[i] - xyz[j]))
            if distance >= floor:
                continue
            count += 1
            if nearest is None or distance < nearest:
                nearest, closest = distance, [labels[i], labels[j]]
    return dict(floor_angstrom=floor, count=count,
                minimum_distance_angstrom=nearest, closest_pair=closest,
                definition="A24 protein heavy atoms on distinct residues; absolute distance")


def generated_input_validity(topology, positions_nm: np.ndarray, *,
                             heavy_floor_angstrom: float) -> dict:
    """Whether a generated recovery input is physically possible (A44)."""
    geometry = topology_geometry_audit(topology, positions_nm)
    heavy = interresidue_heavy_floor_audit(topology, positions_nm, heavy_floor_angstrom)
    reasons = []
    if not geometry["topology_passed"]:
        reasons.append("peptide_topology")
    if not geometry["geometry_passed"]:
        reasons.append("extreme_nonbonded_overlap")
    if heavy["count"]:
        reasons.append("interresidue_heavy_atom_floor")
    return dict(valid=not reasons, reasons=reasons,
                extreme_nonbonded_pair_count=geometry["extreme_nonbonded_pair_count"],
                closest_nonbonded_pair=geometry["closest_nonbonded_pair"],
                interresidue_heavy=heavy)


def sample_valid_input(draw, validity, seed: int, *,
                       max_attempts: int = DEFAULT_MAX_PERTURBATION_ATTEMPTS):
    """Rejection-sample ``draw(rng)`` until ``validity(positions)['valid']`` (A44).

    One generator seeded with ``seed`` serves every attempt, so attempt 0
    consumes exactly the draws a single unconditioned perturbation consumed:
    a seed whose first draw is valid keeps its former input bit for bit. The
    accepted input is therefore a draw from the original perturbation
    distribution conditioned on physical validity. Every rejected attempt is
    returned for the record. Raises :class:`StructureQualityError` (category
    ``generated_input``) if no attempt is valid.
    """
    if int(max_attempts) < 1:
        raise ValueError("max_attempts must be positive")
    rng = np.random.default_rng(seed)
    rejected = []
    for attempt in range(int(max_attempts)):
        positions, records = draw(rng)
        check = validity(positions)
        if check["valid"]:
            return positions, records, dict(
                policy="original perturbation distribution conditioned on physical validity; "
                       "no reference-, energy- or outcome-based criterion",
                seed=int(seed), accepted_attempt=attempt, attempts=attempt + 1,
                max_attempts=int(max_attempts), accepted_validity=check,
                rejected_attempts=rejected)
        rejected.append(dict(attempt=attempt, reasons=check["reasons"],
                             closest_nonbonded_pair=check["closest_nonbonded_pair"],
                             interresidue_heavy_minimum_angstrom=
                             check["interresidue_heavy"]["minimum_distance_angstrom"],
                             interresidue_heavy_closest_pair=check["interresidue_heavy"]["closest_pair"]))
    raise StructureQualityError(
        f"No physically valid perturbed input in {max_attempts} draws from seed {seed}",
        category="generated_input",
        audit=dict(seed=int(seed), max_attempts=int(max_attempts), rejected_attempts=rejected[-20:],
                   rejected_count=len(rejected)))
