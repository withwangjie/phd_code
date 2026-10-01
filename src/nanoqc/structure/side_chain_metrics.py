"""Active side-chain RMSD, severe-clash detection and DockQ.

Split out of evaluate_complex_metrics.py;
see that module's docstring for the metric definitions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import numpy as np
from scipy.spatial import cKDTree
from nanoqc.structure.residue_tables import SIDECHAIN_HEAVY_ATOMS, SYMMETRIC_SWAPS
from nanoqc.structure.complex_atoms import StructureAtoms, _residue_atom_triples


# Canonical side-chain heavy-atom names from residue_tables.py (dependency-free,
# shared with subgraph_to_qubo.py's ``_SIDECHAIN_NAMES`` and
# audit_all_datasets.py's ``SIDECHAIN_HEAVY``).
# Only residues that can legally appear as an Active (redesigned) residue in
# this project's chi1 model are listed; an unrecognized residue name raises
# ValueError in :func:`_active_side_chain_error` rather than silently
# scoring zero atoms.
_SIDECHAIN_HEAVY_ATOMS: Dict[str, Tuple[str, ...]] = SIDECHAIN_HEAVY_ATOMS

# Atom-name pairs that a physically valid local symmetry can exchange without
# changing the residue's actual conformation. Every entry inside one residue
# is swapped *together* as a single alternative candidate (not each pair
# independently): for PHE/TYR this is exactly the one legal 180-degree ring
# flip about the CB-CG-CZ axis, which moves CD1<->CD2 and CE1<->CE2 at the
# same time. See the module docstring's "active_sidechain_rmsd_angstrom"
# section. Shared with subgraph_to_qubo.py's ``_SYMMETRIC_SWAPS`` via
# residue_tables.py.
_SYMMETRIC_SWAPS: Dict[str, Tuple[Tuple[str, str], ...]] = SYMMETRIC_SWAPS


# --------------------------------------------------------------------------
# Symmetry-corrected Active side-chain RMSD
# --------------------------------------------------------------------------

def _active_side_chain_error(
    ref_entry: Mapping[str, Any], pred_entry: Mapping[str, Any], rid: str,
) -> Tuple[Optional[float], int]:
    """Minimum symmetry-corrected summed squared side-chain error for one residue.

    Tries the unswapped atom assignment and, when ``ref_entry["name"]`` is
    in :data:`_SYMMETRIC_SWAPS`, exactly one additional candidate with every
    listed pair swapped together (see the module docstring); returns
    whichever gives the smaller summed squared distance.

    Returns:
        ``(summed_squared_error, atom_count)``. ``summed_squared_error`` is
        ``None`` for a residue with no side-chain heavy atoms (Gly); such
        residues contribute nothing to the aggregate RMSD.

    Raises:
        ValueError: If ``ref_entry["name"]`` is not in
            :data:`_SIDECHAIN_HEAVY_ATOMS`, or either structure is missing a
            required side-chain heavy atom at this residue.
    """
    name = ref_entry["name"]
    if name not in _SIDECHAIN_HEAVY_ATOMS:
        raise ValueError(
            f"Unsupported residue name for Active side-chain scoring at {rid}: {name!r}"
        )
    names = _SIDECHAIN_HEAVY_ATOMS[name]
    if not names:
        return None, 0
    ref_side, pred_side = ref_entry["atoms"], pred_entry["atoms"]
    missing_ref = [n for n in names if n not in ref_side]
    missing_pred = [n for n in names if n not in pred_side]
    if missing_ref or missing_pred:
        raise ValueError(
            f"Missing required side-chain heavy atoms at {rid}: "
            f"reference lacks {missing_ref}, predicted lacks {missing_pred}"
        )
    candidates: List[Mapping[str, np.ndarray]] = [pred_side]
    if name in _SYMMETRIC_SWAPS:
        swapped = dict(pred_side)
        for a, b in _SYMMETRIC_SWAPS[name]:
            swapped[a], swapped[b] = pred_side[b], pred_side[a]
        candidates.append(swapped)
    squared_errors = [
        float(sum(np.sum((candidate[n] - ref_side[n]) ** 2) for n in names))
        for candidate in candidates
    ]
    return min(squared_errors), len(names)


def compute_active_side_chain_rmsd(
    ref_atoms: StructureAtoms,
    aligned_pred_atoms: StructureAtoms,
    active_ids: Sequence[str],
) -> Tuple[Optional[float], List[Dict[str, Any]]]:
    """Symmetry-corrected side-chain heavy-atom RMSD over ``active_ids``.

    Args:
        ref_atoms: Reference structure, as returned by
            :func:`read_structure_atoms`.
        aligned_pred_atoms: Predicted structure already transformed into the
            reference (receptor) frame -- see the module docstring.
        active_ids: Residue ids to score (this project's usual convention:
            the caller-declared Active/redesigned residues).

    Returns:
        ``(rmsd_or_None, per_residue_details)``. ``rmsd_or_None`` is
        ``None`` when every Active residue is Gly (no side-chain heavy
        atoms at all to compare) or ``active_ids`` is empty.

    Raises:
        ValueError: If a residue id is missing from either structure, a
            residue-identity mismatch is found, or a residue's name is not
            in :data:`_SIDECHAIN_HEAVY_ATOMS`, or a required side-chain
            heavy atom is missing from either structure.
    """
    details: List[Dict[str, Any]] = []
    total_error = 0.0
    total_atoms = 0
    for rid in active_ids:
        if rid not in ref_atoms or rid not in aligned_pred_atoms:
            raise ValueError(f"Active residue {rid} missing from reference or predicted structure")
        ref_entry, pred_entry = ref_atoms[rid], aligned_pred_atoms[rid]
        if ref_entry["name"] != pred_entry["name"]:
            raise ValueError(
                f"Residue identity mismatch at Active residue {rid}: "
                f"reference={ref_entry['name']} predicted={pred_entry['name']}"
            )
        error, atom_count = _active_side_chain_error(ref_entry, pred_entry, rid)
        details.append(dict(
            residue_id=rid, residue_name=ref_entry["name"], sidechain_atom_count=atom_count,
            sidechain_rmsd_angstrom=math.sqrt(error / atom_count) if atom_count else None,
            symmetry_corrected=ref_entry["name"] in _SYMMETRIC_SWAPS,
        ))
        if atom_count:
            total_error += error
            total_atoms += atom_count
    rmsd = math.sqrt(total_error / total_atoms) if total_atoms else None
    return rmsd, details


# --------------------------------------------------------------------------
# Stereochemical severe-clash filter
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ClashPair:
    """One severe steric clash between two non-bonded heavy atoms."""

    residue_a: str
    atom_a: str
    residue_b: str
    atom_b: str
    distance_angstrom: float


def compute_bonded_exclusions(
    atoms: StructureAtoms,
    residue_ids: Sequence[str],
    *,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
) -> Set[frozenset]:
    """Residue-atom pairs excluded from clash scanning as genuine covalent bonds.

    Two kinds of adjacency are detected, both from ``atoms``' own geometry
    (intended to be called on the *reference* structure -- see the module
    docstring -- so the exclusion set is stable across every predicted/
    intermediate structure the same complex is scanned against):

    * Peptide bonds: within each chain, residues are ordered by
      ``(seqid, icode)``; a consecutive pair whose C(i)-N(i+1) distance is
      below ``peptide_bond_cutoff`` (a canonical peptide bond is ~1.33 A) is
      excluded. A chain break or numbering gap simply fails the distance
      test and is correctly NOT excluded, so a real steric clash across a
      break still gets flagged.
    * Disulfide bonds: every pair of CYS residues (in ``residue_ids``, not
      necessarily consecutive or same-chain) whose SG-SG distance is below
      ``disulfide_cutoff`` (a canonical S-S bond is ~2.05 A) is excluded.

    Args:
        atoms: Parsed structure (as returned by :func:`read_structure_atoms`)
            whose geometry defines bond adjacency.
        residue_ids: Residues considered for both exclusion kinds.
        peptide_bond_cutoff: Distance below which a consecutive C(i)-N(i+1)
            pair is treated as a genuine peptide bond (default 1.9 A).
        disulfide_cutoff: Distance below which a CYS SG-SG pair is treated
            as a genuine disulfide bond (default 2.3 A).

    Returns:
        A set of ``frozenset({(residue_id, atom_name), (residue_id, atom_name)})``
        pairs to exclude from clash scanning.
    """
    by_chain: Dict[str, List[str]] = {}
    for rid in residue_ids:
        by_chain.setdefault(atoms[rid]["chain"], []).append(rid)
    excluded: Set[frozenset] = set()
    for ids in by_chain.values():
        ordered = sorted(ids, key=lambda r: (atoms[r]["seqid"], atoms[r]["icode"]))
        for a, b in zip(ordered, ordered[1:]):
            atoms_a, atoms_b = atoms[a]["atoms"], atoms[b]["atoms"]
            if "C" in atoms_a and "N" in atoms_b:
                if np.linalg.norm(atoms_a["C"] - atoms_b["N"]) < peptide_bond_cutoff:
                    excluded.add(frozenset(((a, "C"), (b, "N"))))
    sulfurs = [
        rid for rid in residue_ids
        if atoms[rid]["name"] == "CYS" and "SG" in atoms[rid]["atoms"]
    ]
    for i, a in enumerate(sulfurs):
        for b in sulfurs[i + 1:]:
            if np.linalg.norm(atoms[a]["atoms"]["SG"] - atoms[b]["atoms"]["SG"]) < disulfide_cutoff:
                excluded.add(frozenset(((a, "SG"), (b, "SG"))))
    return excluded


def find_severe_clashes(
    atoms: StructureAtoms,
    residue_ids: Sequence[str],
    exclusions: Set[frozenset],
    *,
    clash_cutoff: float = 1.5,
) -> List[ClashPair]:
    """Severe steric clashes among non-covalently-bonded heavy atoms.

    Excludes atom pairs within the same residue and every pair listed in
    ``exclusions`` (see :func:`compute_bonded_exclusions`). This is a plain
    geometric distance filter, not MolProbity clashscore: there are no van
    der Waals radii and no general bonded/1-4 exclusion table beyond the two
    exclusion kinds ``exclusions`` was built from.

    Args:
        atoms: Parsed structure to scan (heavy atoms only, as returned by
            :func:`read_structure_atoms`).
        residue_ids: Residues to include in the scan.
        exclusions: Bonded atom-name pairs to exclude, from
            :func:`compute_bonded_exclusions` (normally computed once on the
            reference structure and reused for every scanned structure).
        clash_cutoff: Distance below which a non-excluded pair counts as a
            severe clash (default 1.5 A).

    Returns:
        Every severe clash found, each atom pair reported once.
    """
    triples = _residue_atom_triples(atoms, residue_ids)
    if len(triples) < 2:
        return []
    xyz = np.array([t[2] for t in triples])
    tree = cKDTree(xyz)
    severe: List[ClashPair] = []
    for i, j in sorted(tree.query_pairs(clash_cutoff)):
        rid_a, name_a, xyz_a = triples[i]
        rid_b, name_b, xyz_b = triples[j]
        if rid_a == rid_b:
            continue
        if frozenset(((rid_a, name_a), (rid_b, name_b))) in exclusions:
            continue
        distance = float(np.linalg.norm(xyz_a - xyz_b))
        if distance < clash_cutoff:
            severe.append(ClashPair(rid_a, name_a, rid_b, name_b, distance))
    return severe


# --------------------------------------------------------------------------
# DockQ combination
# --------------------------------------------------------------------------

def compute_dockq(
    fnat: Optional[float], irmsd: Optional[float], lrmsd: float
) -> Tuple[Optional[float], Optional[str]]:
    """Receptor-aligned DockQ-style combination and CAPRI-style category.

    ``dockq = (Fnat + 1/(1+(iRMSD/1.5)^2) + 1/(1+(LRMSD/8.5)^2)) / 3``

    Categories: Incorrect (< 0.23), Acceptable ([0.23, 0.49)),
    Medium ([0.49, 0.80)), High (>= 0.80). See the module docstring for why
    this is deliberately not called "the DockQ score".

    Returns:
        ``(None, None)`` when ``fnat`` or ``irmsd`` is ``None`` (undefined
        because the reference has no native contacts, or no interface
        residues were found); otherwise ``(score, category)``.

    Raises:
        ValueError: If any provided metric is non-finite.
    """
    if fnat is None or irmsd is None:
        return None, None
    if not (math.isfinite(fnat) and math.isfinite(irmsd) and math.isfinite(lrmsd)):
        raise ValueError(f"fnat={fnat}, irmsd={irmsd}, lrmsd={lrmsd} must all be finite")
    score = (fnat + 1.0 / (1.0 + (irmsd / 1.5) ** 2) + 1.0 / (1.0 + (lrmsd / 8.5) ** 2)) / 3.0
    if score < 0.23:
        category = "Incorrect"
    elif score < 0.49:
        category = "Acceptable"
    elif score < 0.80:
        category = "Medium"
    else:
        category = "High"
    return float(score), category
