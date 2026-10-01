"""Structure reading, superposition and contact geometry for complex evaluation.

Split out of evaluate_complex_metrics.py;
see that module's docstring for the metric definitions.
"""
from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import numpy as np
from scipy.spatial import cKDTree
try:
    import gemmi
except ImportError as exc:  # pragma: no cover - environment guard, not test-exercised
    raise ImportError(
        "evaluate_complex_metrics requires the 'gemmi' package "
        "(pip install gemmi==0.7.5; see requirements.txt)"
    ) from exc



PathLike = Union[str, Path]

# atom name -> read-only [3] float64 coordinate array
ResidueAtoms = Dict[str, np.ndarray]
# "CHAIN:SEQID[ICODE]" -> {"chain", "seqid", "icode", "name", "altloc", "atoms"}
StructureAtoms = Dict[str, Dict[str, Any]]


class AtomCompletenessError(ValueError):
    """A structure's heavy-atom content does not exactly match the reference's.

    Subclasses ``ValueError`` so existing ``except ValueError`` call sites
    elsewhere in this pipeline (CLI entry points, batch drivers) keep
    working unchanged; a caller that wants to distinguish this specific
    failure mode from a generic argument error can catch
    ``AtomCompletenessError`` directly. Raised by
    :func:`verify_full_chain_heavy_atom_completeness`.
    """


# --------------------------------------------------------------------------
# Gemmi parsing
# --------------------------------------------------------------------------

def _clean_code(value: str) -> str:
    """Normalize gemmi's blank altloc/insertion-code sentinels (' ' or NUL) to ''."""

    return "" if value in (" ", "\x00", "") else value


def _residue_id(chain_name: str, seqid: "gemmi.SeqId") -> str:
    """Author "CHAIN:SEQID[ICODE]" key, e.g. "H:1" or "H:100A"."""

    return f"{chain_name}:{seqid.num}{_clean_code(seqid.icode)}"


def _frozen_vector(x: float, y: float, z: float) -> np.ndarray:
    """A fixed-content, non-writeable 3-vector (atom-level write protection)."""

    array = np.array([x, y, z], dtype=np.float64)
    array.setflags(write=False)
    return array


def read_structure_atoms(path: PathLike, *, model_index: int = 0) -> StructureAtoms:
    """Parse a structure into canonical protein heavy-atom coordinates.

    Accepts mmCIF (``.cif``) or legacy PDB (``.pdb``/``.ent``) transparently
    -- gemmi sniffs the format from file content, not the extension. Only
    protein polymer residues are kept (``gemmi.find_tabulated_residue(...)
    .is_amino_acid()``); waters, ions and other heteroatom ligands are
    silently skipped, since this evaluator only scores the protein-protein
    complex. Only heavy (non-hydrogen, non-deuterium) atoms with positive
    occupancy are kept.

    When a residue has alternate conformers, the highest-mean-occupancy
    conformer is kept (ties broken lexicographically by altloc code), plus
    any atoms shared across every conformer (blank altloc). Atoms from two
    different conformers of the same residue are never mixed.

    Every returned coordinate array is read-only.

    Args:
        path: Path to an mmCIF or PDB file.
        model_index: Which model to read for a multi-model file (default 0,
            the first / only model).

    Returns:
        A mapping from residue id to ``{"chain", "seqid", "icode", "name",
        "altloc", "atoms"}``, where ``"atoms"`` maps atom name to a
        read-only ``[3]`` coordinate array.

    Raises:
        ValueError: If ``model_index`` is out of range, an author
            chain:seqid key is duplicated within the model, a residue's
            conformer selection would mix two altlocs under one atom name,
            a parsed coordinate is non-finite, or no protein heavy atoms
            were found at all.
    """
    structure = gemmi.read_structure(str(path))
    if not 0 <= model_index < len(structure):
        raise ValueError(
            f"Invalid model_index={model_index} for {path} ({len(structure)} model(s) present)"
        )
    model = structure[model_index]
    residues: StructureAtoms = {}
    for chain in model:
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info is None or not info.is_amino_acid():
                continue  # water / heteroatom ligand / nucleic acid / etc.
            rid = _residue_id(chain.name, residue.seqid)
            if rid in residues:
                raise ValueError(f"Duplicate author residue id {rid!r} in {path}")
            observed = [a for a in residue if a.element.name not in ("H", "D") and a.occ > 0]
            if not observed:
                continue  # fully unresolved / hydrogen-only residue: no usable heavy atoms
            altlocs = sorted({a.altloc for a in observed if _clean_code(a.altloc)})
            chosen = (
                min(
                    altlocs,
                    key=lambda code: (
                        -float(np.mean([a.occ for a in observed if a.altloc == code])),
                        code,
                    ),
                )
                if altlocs
                else ""
            )
            atoms: ResidueAtoms = {}
            for atom in observed:
                code = _clean_code(atom.altloc)
                if code and code != chosen:
                    continue
                name = atom.name.strip()
                if name in atoms:
                    raise ValueError(f"Duplicate atom after conformer selection: {rid}/{name} in {path}")
                xyz = _frozen_vector(atom.pos.x, atom.pos.y, atom.pos.z)
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Non-finite coordinate at {rid}/{name} in {path}")
                atoms[name] = xyz
            residues[rid] = dict(
                chain=chain.name,
                seqid=residue.seqid.num,
                icode=_clean_code(residue.seqid.icode),
                name=residue.name,
                altloc=chosen,
                atoms=atoms,
            )
    if not residues:
        raise ValueError(f"No protein heavy atoms found in {path}")
    return residues


# --------------------------------------------------------------------------
# Rigid-body alignment
# --------------------------------------------------------------------------

def kabsch_fit(moving: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Proper (no-reflection) Kabsch rotation and translation.

    Row-vector convention: ``moving @ R + t`` best superposes onto
    ``reference`` in the least-squares sense.

    Args:
        moving: ``[N, 3]`` coordinates to be transformed.
        reference: ``[N, 3]`` coordinates to superpose onto, row-matched to
            ``moving``.

    Returns:
        ``(R, t)`` with ``R`` a proper ``[3, 3]`` rotation matrix
        (``det(R) == 1``, never a reflection) and ``t`` a ``[3]``
        translation.

    Raises:
        ValueError: If the shapes disagree, fewer than 3 atoms are given,
            or the atoms are (near-)collinear, in which case the rotation
            about the collinear axis is underdetermined.
    """
    moving = np.asarray(moving, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError(f"Alignment arrays must share shape [N, 3]; got {moving.shape} vs {reference.shape}")
    if len(moving) < 3:
        raise ValueError(f"Kabsch alignment needs at least 3 atoms, got {len(moving)}")
    a = moving - moving.mean(axis=0)
    b = reference - reference.mean(axis=0)
    if min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Alignment atoms are (near-)collinear; the rotation is underdetermined")
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    translation = reference.mean(axis=0) - moving.mean(axis=0) @ rotation
    return rotation, translation


def _paired_coordinates(
    ref_atoms: StructureAtoms,
    moving_atoms: StructureAtoms,
    residue_ids: Sequence[str],
    atom_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Matched ``(moving, reference)`` heavy-atom coordinate arrays.

    When ``atom_names`` is given (e.g. the fixed backbone quartet), only
    those named atoms are required to be present in both structures --
    other atoms present in either residue are ignored. When ``atom_names``
    is ``None`` ("all observed heavy atoms"), the two residues' heavy-atom
    *sets* must match exactly, because there is no fixed reference list to
    fall back on; a partially-resolved residue must fail loudly here rather
    than silently bias the RMSD by comparing mismatched atom sets.

    Raises:
        ValueError: On a missing residue, a residue-identity (three-letter
            code) mismatch, a missing required atom, or (in "all atoms"
            mode) an atom-set mismatch.
    """
    moving_rows: List[np.ndarray] = []
    ref_rows: List[np.ndarray] = []
    for rid in residue_ids:
        if rid not in ref_atoms or rid not in moving_atoms:
            raise ValueError(f"Residue {rid} missing from reference or predicted structure")
        ref_entry, moving_entry = ref_atoms[rid], moving_atoms[rid]
        if ref_entry["name"] != moving_entry["name"]:
            raise ValueError(
                f"Residue identity mismatch at {rid}: "
                f"reference={ref_entry['name']} predicted={moving_entry['name']}"
            )
        if atom_names is None:
            ref_names = set(ref_entry["atoms"])
            moving_names = set(moving_entry["atoms"])
            if ref_names != moving_names:
                raise ValueError(
                    f"Heavy-atom set mismatch at {rid}: "
                    f"reference-only={sorted(ref_names - moving_names)}, "
                    f"predicted-only={sorted(moving_names - ref_names)}"
                )
            names: Sequence[str] = sorted(ref_names)
        else:
            names = atom_names
            missing_ref = [n for n in names if n not in ref_entry["atoms"]]
            missing_pred = [n for n in names if n not in moving_entry["atoms"]]
            if missing_ref or missing_pred:
                raise ValueError(
                    f"Missing required atoms at {rid}: "
                    f"reference lacks {missing_ref}, predicted lacks {missing_pred}"
                )
        for name in names:
            ref_rows.append(ref_entry["atoms"][name])
            moving_rows.append(moving_entry["atoms"][name])
    if not moving_rows:
        raise ValueError("No atoms selected for this RMSD/alignment computation")
    return np.array(moving_rows, dtype=np.float64), np.array(ref_rows, dtype=np.float64)


def _rmsd(moving: np.ndarray, reference: np.ndarray) -> float:
    """Plain coordinate RMSD between two matched ``[N, 3]`` arrays."""

    return float(np.sqrt(np.mean(np.sum((moving - reference) ** 2, axis=1))))


# --------------------------------------------------------------------------
# Contacts / interface / Fnat
# --------------------------------------------------------------------------

def _residue_atom_triples(
    atoms: StructureAtoms, residue_ids: Sequence[str]
) -> List[Tuple[str, str, np.ndarray]]:
    """Flatten ``(residue_id, atom_name, xyz)`` triples, atom names sorted for stability."""

    return [
        (rid, name, atoms[rid]["atoms"][name])
        for rid in residue_ids
        for name in sorted(atoms[rid]["atoms"])
    ]


def compute_contact_residue_pairs(
    atoms: StructureAtoms, group_a: Sequence[str], group_b: Sequence[str], cutoff: float
) -> Set[Tuple[str, str]]:
    """Residue pairs (one from each group) with any heavy-atom pair within ``cutoff`` A."""

    a_triples = _residue_atom_triples(atoms, group_a)
    b_triples = _residue_atom_triples(atoms, group_b)
    if not a_triples or not b_triples:
        return set()
    b_xyz = np.array([xyz for _, _, xyz in b_triples])
    tree = cKDTree(b_xyz)
    pairs: Set[Tuple[str, str]] = set()
    for rid_a, _, xyz_a in a_triples:
        for j in tree.query_ball_point(xyz_a, cutoff):
            if np.linalg.norm(xyz_a - b_xyz[j]) < cutoff:
                pairs.add((rid_a, b_triples[j][0]))
    return pairs


def compute_fnat(
    ref_atoms: StructureAtoms,
    pred_atoms: StructureAtoms,
    receptor_ids: Sequence[str],
    ligand_ids: Sequence[str],
    cutoff: float = 5.0,
) -> Tuple[Optional[float], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """Fraction of reference receptor-ligand residue contacts recovered in the prediction.

    See the module docstring's Fnat definition. Returns ``(None, native,
    predicted)`` when the reference itself has zero native contacts (Fnat is
    undefined, not zero, in that case).
    """
    native = compute_contact_residue_pairs(ref_atoms, receptor_ids, ligand_ids, cutoff)
    predicted = compute_contact_residue_pairs(pred_atoms, receptor_ids, ligand_ids, cutoff)
    fnat = len(native & predicted) / len(native) if native else None
    return fnat, native, predicted


def compute_interface_residues(
    ref_atoms: StructureAtoms,
    receptor_ids: Sequence[str],
    ligand_ids: Sequence[str],
    cutoff: float = 10.0,
) -> List[str]:
    """Reference residues (either chain) within ``cutoff`` A of the other chain."""

    pairs = compute_contact_residue_pairs(ref_atoms, receptor_ids, ligand_ids, cutoff)
    interface = {r for r, _ in pairs} | {l for _, l in pairs}
    return sorted(
        interface,
        key=lambda r: (ref_atoms[r]["chain"], ref_atoms[r]["seqid"], ref_atoms[r]["icode"]),
    )


def _chain_residue_ids(atoms: StructureAtoms, chains: Sequence[str]) -> List[str]:
    """Every residue id in ``atoms`` belonging to one of ``chains``, sorted by author order."""

    chain_set = set(chains)
    ids = [rid for rid, entry in atoms.items() if entry["chain"] in chain_set]
    return sorted(ids, key=lambda r: (atoms[r]["chain"], atoms[r]["seqid"], atoms[r]["icode"]))


# --------------------------------------------------------------------------
# Full-chain heavy-atom completeness
# --------------------------------------------------------------------------

def verify_full_chain_heavy_atom_completeness(
    ref_atoms: StructureAtoms,
    pred_atoms: StructureAtoms,
    chain_ids: Sequence[str],
) -> None:
    """Enforce full-chain, residue-by-residue heavy-atom identity, every chain, every residue.

    Closes the "non-interface heavy-atom bypass": every other check in this
    module only inspects the specific atoms a given computation actually
    reads (the fixed backbone quartet for alignment/LRMSD, all atoms of
    *interface* residues for iRMSD, side-chain atoms of *Active* residues
    for the symmetry-corrected RMSD). A residue that is none of those --
    e.g. a receptor residue far from the interface and not declared Active
    -- was never checked at all: silently deleting one of its heavy atoms
    (or adding a spurious one) from the predicted structure changed nothing
    about the reported score. This function is a single, independent,
    full-chain sweep that closes that gap. It is called, for every declared
    chain, before :func:`evaluate_complex_metrics` computes anything else --
    before either structure's atom content is trusted for alignment or any
    metric.

    Args:
        ref_atoms: Reference structure, as returned by
            :func:`read_structure_atoms` (this module's own "ref_structure").
        pred_atoms: Predicted structure, same (this module's "pred_structure").
        chain_ids: Chain identifiers (as they appear in the file) to check.
            Every residue belonging to one of these chains in EITHER
            structure is checked -- not only those present in the reference
            -- so an insertion in the prediction is caught too, not only a
            deletion.

    Raises:
        AtomCompletenessError: If a chain has a different residue-id set
            between the two structures (a missing or an inserted residue),
            a residue-identity (three-letter code) mismatch, or -- the
            specific gap this function closes -- a heavy-atom-name-set
            mismatch at ANY residue, interface or not, Active or not. The
            message names the exact residue id and the precise missing/
            extra atom names, e.g. deleting the off-interface, non-Active
            receptor atom A:20/CB from the prediction is reported as
            ``"... at A:20 (ALA): missing from prediction=['CB'], ..."``,
            never silently absorbed into a lower score.
    """
    chain_set = set(chain_ids)
    ref_ids = set(_chain_residue_ids(ref_atoms, chain_set))
    pred_ids = set(_chain_residue_ids(pred_atoms, chain_set))
    only_in_ref = sorted(ref_ids - pred_ids)
    only_in_pred = sorted(pred_ids - ref_ids)
    if only_in_ref or only_in_pred:
        raise AtomCompletenessError(
            f"Residue-set mismatch for chains {sorted(chain_set)}: "
            f"present only in reference={only_in_ref}, present only in prediction={only_in_pred}"
        )
    for rid in sorted(ref_ids, key=lambda r: (ref_atoms[r]["chain"], ref_atoms[r]["seqid"], ref_atoms[r]["icode"])):
        ref_entry, pred_entry = ref_atoms[rid], pred_atoms[rid]
        if ref_entry["name"] != pred_entry["name"]:
            raise AtomCompletenessError(
                f"Residue identity mismatch at {rid}: "
                f"reference={ref_entry['name']} predicted={pred_entry['name']}"
            )
        ref_heavy = set(ref_entry["atoms"])
        pred_heavy = set(pred_entry["atoms"])
        if ref_heavy != pred_heavy:
            raise AtomCompletenessError(
                f"Heavy-atom completeness violation at {rid} ({ref_entry['name']}): "
                f"missing from prediction={sorted(ref_heavy - pred_heavy)}, "
                f"unexpected extra atoms in prediction={sorted(pred_heavy - ref_heavy)}"
            )
