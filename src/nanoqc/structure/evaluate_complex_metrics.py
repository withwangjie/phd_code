"""Full-atom nanobody-antigen complex evaluator for the side-chain-optimization /
complex-prediction pipeline (Gemmi-based).

This module supersedes the earlier ``evaluate_complex_dockq`` evaluator (no
longer in this repository) with the three corrections
raised in the academic pre-review (R1-M5): (1) an explicit, non-official
receptor-aligned DockQ field name so it is never confused with the published
DockQ definition; (2) a symmetry-corrected side-chain heavy-atom RMSD for the
caller-declared Active (redesigned) residues, so a physically meaningless
180-degree ring/carboxylate flip does not register as a large false error;
(3) side-by-side multi-stage trajectory evaluation, so a pipeline run's
perturbed input, relax-only control, discretely-picked rotamer structure, and
both post-relaxation stages can be reported together instead of only the
final structure.

Reference frame and alignment
------------------------------
The rigid-body superposition (Kabsch algorithm) that brings a predicted
structure into the reference frame is fit **only** on the antigen
(receptor)'s backbone heavy atoms (N, CA, C, O), over receptor residues that
are *not* in the caller-declared ``active_residues`` set (in this project's
usual convention ``active_residues`` are the handful of nanobody paratope
residues being redesigned, so this exclusion is a no-op on the receptor side
and the fit uses the whole receptor backbone; the exclusion is still applied
generally in case a future active set ever includes receptor residues, e.g.
a locally flexible epitope patch). The nanobody (ligand) backbone is never
used to compute this fit -- fitting it would let a global superposition
partially explain away a real docking-pose error. It is still *transformed*
by the receptor-derived rotation/translation like every other predicted
atom, which is the entire point: LRMSD and interface metrics must reflect
the predicted pose in the receptor's own frame, not the ligand's self-best
overlay.

Metric definitions
-------------------
* **Fnat**: fraction of reference receptor-ligand residue-pair contacts
  (heavy-atom pair distance < ``fnat_cutoff``, default 5.0 A) recovered in
  the predicted structure. Computed independently in each structure's own
  coordinate frame -- internal (including inter-chain) distances are
  unaffected by a rigid-body transform of the whole structure, so Fnat needs
  no prior alignment.
* **iRMSD**: all-heavy-atom RMSD, after receptor-frame alignment, of every
  reference residue (either chain) within ``interface_cutoff`` (default
  10.0 A) of the other chain.
* **LRMSD**: nanobody backbone (N, CA, C, O) RMSD after receptor-frame
  alignment.
* **dockq_receptor_aligned_variant**:
  ``(Fnat + 1/(1+(iRMSD/1.5)**2) + 1/(1+(LRMSD/8.5)**2)) / 3``, categorized
  Incorrect (< 0.23), Acceptable ([0.23, 0.49)), Medium ([0.49, 0.80)),
  High (>= 0.80). The field is deliberately **not** named ``dockq_score``:
  this is a single receptor-only rigid-body fit with all-heavy-atom interface
  RMSD, not the original DockQ paper's separate least-squares fit of only
  the interface backbone atoms of both partners together, so it must never
  be silently compared to published-DockQ numbers. ``dockq_definition``
  in the result carries this caveat as text alongside the score.
* **active_sidechain_rmsd_angstrom**: symmetry-corrected side-chain
  heavy-atom RMSD over the caller-declared ``active_residues``, computed
  after the same receptor-frame alignment. For PHE and TYR, the aromatic
  ring's two-fold symmetry means a 180-degree flip about the CB-CG-CZ axis
  swaps CD1<->CD2 *and* CE1<->CE2 *simultaneously* (they are not
  independent degrees of freedom); this module therefore compares the
  unswapped assignment against exactly that one physically valid
  simultaneous double-swap and keeps whichever gives the lower squared
  error, per residue, exactly as :data:`_SYMMETRIC_SWAPS` in
  ``subgraph_to_qubo.py`` already does for chi1 recovery scoring elsewhere
  in this pipeline. The same table's other entries (ASP/GLU carboxylate
  O-O, ARG guanidinium N-N, VAL/LEU methyl-pair symmetry) are included for
  consistency with that established convention; they do not change PHE/TYR
  behavior and only ever apply if such a residue is itself declared Active.

Stereochemical severe-clash filter
------------------------------------
Severe clashes are counted over every non-covalently-bonded heavy-atom pair
(``clash_cutoff`` default 1.5 A) among the scored receptor+ligand residues.
Bond adjacency -- both consecutive peptide C(i)-N(i+1) pairs and CYS-CYS
disulfide SG-SG pairs -- is determined once from the **reference**
structure's own geometry (not the possibly-distorted predicted/intermediate
structure being scanned), matching this project's ``structural_quality.py``
convention; the same excluded atom-name pairs are then applied whichever
structure is scanned. This is a plain geometric distance filter, not
MolProbity clashscore: there are no van der Waals radii and no general
bonded/1-4 exclusion table beyond the two exclusions named above.

Multi-stage trajectory evaluation
------------------------------------
:func:`evaluate_trajectory` accepts an ordered mapping of stage name to
predicted-structure path (e.g. ``{"perturbed_input": ..., "relax_only":
..., "discrete_picked": ..., "stage1_relaxed": ..., "stage2_relaxed":
...}`` -- this project's canonical five pipeline checkpoints, though any
stage names and count are accepted) and evaluates every stage against the
same single reference with :func:`evaluate_complex_metrics`, returning one
full result dict per stage (tagged with ``"stage"``/``"stage_index"``) plus
stage-over-stage deltas for the headline metrics, so a full R1-M5-style
side-by-side trajectory table can be built without re-parsing the reference
once per stage.

Defensive conventions
----------------------
* Every coordinate array this module hands back or stores internally is a
  read-only numpy view (atom-level write protection): parsing and alignment
  never mutate a previously extracted array in place, they only ever build
  fresh ones.
* Residue-identity and heavy-atom-set mismatches between the reference and a
  prediction raise ``ValueError`` immediately rather than silently
  subsetting or dropping atoms, which could otherwise bias an RMSD without
  any visible symptom.
* Every returned result dict contains only JSON-serializable native Python
  types (``float``/``int``/``str``/``bool``/``list``/``dict``/``None``),
  ready for ``json.dump`` or one flattened CSV row per structure/stage (see
  :func:`to_json` / :func:`append_to_csv` / :func:`append_trajectory_to_csv`).
* This module is intentionally free of exotic dependencies: pure Python +
  numpy + scipy (``cKDTree``) + gemmi. It does not import ``subgraph_to_qubo``
  (which pulls in ``torch``/``torch_geometric`` at import time); the residue
  atom tables both modules need live in the dependency-free
  ``residue_tables`` module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
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
from nanoqc.structure.residue_tables import BACKBONE_ATOMS, SIDECHAIN_HEAVY_ATOMS, SYMMETRIC_SWAPS

# atom name -> read-only [3] float64 coordinate array
ResidueAtoms = Dict[str, np.ndarray]
# "CHAIN:SEQID[ICODE]" -> {"chain", "seqid", "icode", "name", "altloc", "atoms"}
StructureAtoms = Dict[str, Dict[str, Any]]

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


# --------------------------------------------------------------------------
# Top-level single-structure orchestrator
# --------------------------------------------------------------------------

def evaluate_complex_metrics(
    ref_path: PathLike,
    pred_path: PathLike,
    *,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    active_residues: Sequence[str] = (),
    fnat_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    clash_cutoff: float = 1.5,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
    model_index: int = 0,
) -> Dict[str, Any]:
    """Evaluate one predicted nanobody-antigen complex against a reference.

    See the module docstring for the alignment convention, metric
    definitions, and clash-exclusion convention. Nothing this function reads
    is ever mutated in place: parsing, alignment, and every downstream
    computation build fresh arrays (see :func:`read_structure_atoms`'s
    atom-level write protection).

    A thin file-reading wrapper around :func:`evaluate_complex_metrics_from_atoms`
    -- reads both structures with :func:`read_structure_atoms`, then delegates.
    Callers that already have parsed ``StructureAtoms`` in memory (e.g. a
    caller that also needs those atoms for its own, separate computation)
    should call :func:`evaluate_complex_metrics_from_atoms` directly instead,
    to avoid parsing the same files twice.

    Args:
        ref_path: Reference structure (mmCIF or PDB; format sniffed by
            gemmi from content, not the file extension).
        pred_path: Predicted / relaxed structure, same format flexibility.
        receptor_chains: Antigen chain IDs (as they appear in the file).
        ligand_chains: Nanobody chain IDs. Disjoint from
            ``receptor_chains``; never used to compute the alignment fit.
        active_residues: Residue ids (``"CHAIN:SEQID[ICODE]"``, the same
            convention :func:`read_structure_atoms` produces; this
            project's usual convention is a handful of nanobody paratope
            residues) to score with the symmetry-corrected side-chain RMSD,
            and to EXCLUDE from the alignment fit wherever they happen to
            fall on the receptor. Must be a subset of the receptor+ligand
            residues. Defaults to none.
        fnat_cutoff: Contact distance for Fnat (default 5.0 A).
        interface_cutoff: Distance defining interface residues for iRMSD
            (default 10.0 A).
        clash_cutoff: Severe-clash distance threshold (default 1.5 A).
        peptide_bond_cutoff: C(i)-N(i+1) distance below which a consecutive
            residue pair is treated as a genuine peptide bond and excluded
            from clash scanning (default 1.9 A).
        disulfide_cutoff: SG-SG distance below which a CYS pair is treated
            as a genuine disulfide bond and excluded from clash scanning
            (default 2.3 A).
        model_index: Which model to read from each file (default 0).

    Returns:
        A JSON-serializable dict with every metric described in the module
        docstring, plus alignment/clash provenance and the exact cutoffs
        and file paths used, so a saved result is fully self-describing.
        See the source for the complete key list.

    Raises:
        ValueError: On any invalid argument, chain/residue selection that
            does not exist or is not disjoint as required, a residue-
            identity or heavy-atom-set mismatch between the reference and
            prediction, or too few non-Active receptor residues to fit a
            rotation (fewer than 3).
    """
    ref_atoms = read_structure_atoms(ref_path, model_index=model_index)
    pred_atoms = read_structure_atoms(pred_path, model_index=model_index)
    return evaluate_complex_metrics_from_atoms(
        ref_atoms, pred_atoms,
        receptor_chains=receptor_chains, ligand_chains=ligand_chains,
        active_residues=active_residues, fnat_cutoff=fnat_cutoff,
        interface_cutoff=interface_cutoff, clash_cutoff=clash_cutoff,
        peptide_bond_cutoff=peptide_bond_cutoff, disulfide_cutoff=disulfide_cutoff,
        model_index=model_index, reference_path=ref_path, predicted_path=pred_path,
    )


def evaluate_complex_metrics_from_atoms(
    ref_atoms: StructureAtoms,
    pred_atoms: StructureAtoms,
    *,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    active_residues: Sequence[str] = (),
    fnat_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    clash_cutoff: float = 1.5,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
    model_index: int = 0,
    reference_path: Optional[PathLike] = None,
    predicted_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Evaluate one predicted complex against a reference, from already-parsed atoms.

    The actual computation body of :func:`evaluate_complex_metrics` (which is
    a thin ``read_structure_atoms`` + delegate wrapper around this function).
    Call this directly when the caller already has both structures parsed as
    ``StructureAtoms`` (e.g. produced by its own, separately-configured
    parser, or by a synthetic test fixture) -- this avoids a second,
    redundant file parse, and does not require ``ref_path``/``pred_path`` to
    be real, readable files.

    Args:
        ref_atoms, pred_atoms: Already-parsed reference/prediction
            structures, in the same shape :func:`read_structure_atoms`
            returns (residue id -> ``{"chain", "seqid", "icode", "name",
            "altloc", "atoms"}``).
        receptor_chains, ligand_chains, active_residues, fnat_cutoff,
            interface_cutoff, clash_cutoff, peptide_bond_cutoff,
            disulfide_cutoff: Same as :func:`evaluate_complex_metrics`.
        model_index: Recorded in the result dict for provenance only (no
            parsing happens here); pass through the value actually used to
            produce ``ref_atoms``/``pred_atoms``, if any (default 0).
        reference_path, predicted_path: Recorded in the result dict's
            ``"reference_path"``/``"predicted_path"`` fields for provenance
            only, if available (``None`` when there is no real file, e.g. a
            synthetic in-memory fixture).

    Returns, Raises: Same as :func:`evaluate_complex_metrics`.
    """
    cutoffs = (fnat_cutoff, interface_cutoff, clash_cutoff, peptide_bond_cutoff, disulfide_cutoff)
    if any((not math.isfinite(v)) or v <= 0 for v in cutoffs):
        raise ValueError(
            "fnat_cutoff, interface_cutoff, clash_cutoff, peptide_bond_cutoff and "
            "disulfide_cutoff must all be positive and finite"
        )

    receptor_chains = list(dict.fromkeys(receptor_chains))
    ligand_chains = list(dict.fromkeys(ligand_chains))
    if not receptor_chains or not ligand_chains:
        raise ValueError("receptor_chains and ligand_chains must both be nonempty")
    if set(receptor_chains) & set(ligand_chains):
        raise ValueError("receptor_chains and ligand_chains must be disjoint")

    receptor_ids = _chain_residue_ids(ref_atoms, receptor_chains)
    ligand_ids = _chain_residue_ids(ref_atoms, ligand_chains)
    if not receptor_ids:
        raise ValueError(f"No residues found in the reference for receptor_chains={receptor_chains}")
    if not ligand_ids:
        raise ValueError(f"No residues found in the reference for ligand_chains={ligand_chains}")

    active_ids = list(dict.fromkeys(active_residues))
    complex_ids = set(receptor_ids) | set(ligand_ids)
    unknown_active = [rid for rid in active_ids if rid not in complex_ids]
    if unknown_active:
        raise ValueError(
            f"active_residues must be a subset of receptor_chains/ligand_chains residues; "
            f"unknown ids: {unknown_active}"
        )

    # (P2 remediation) Full-chain, residue-by-residue heavy-atom completeness,
    # across EVERY evaluated chain -- not just the backbone/interface/Active
    # atoms each downstream computation happens to read. Must run before
    # alignment or any metric trusts either structure's atom content; see
    # verify_full_chain_heavy_atom_completeness's docstring for the exact
    # bypass this closes.
    verify_full_chain_heavy_atom_completeness(
        ref_atoms, pred_atoms, list(receptor_chains) + list(ligand_chains),
    )

    # --- (1) receptor-frame Kabsch alignment, fit only on non-Active receptor backbone ---
    active_on_receptor = set(active_ids) & set(receptor_ids)
    alignment_ids = [rid for rid in receptor_ids if rid not in active_on_receptor]
    if len(alignment_ids) < 3:
        raise ValueError(
            f"Only {len(alignment_ids)} non-Active receptor residue(s) available for alignment; "
            "need at least 3. Widen receptor_chains or shrink active_residues."
        )
    moving_alignment, fixed_alignment = _paired_coordinates(ref_atoms, pred_atoms, alignment_ids, BACKBONE_ATOMS)
    rotation, translation = kabsch_fit(moving_alignment, fixed_alignment)
    alignment_rmsd = _rmsd(moving_alignment @ rotation + translation, fixed_alignment)

    # Apply the SAME rigid transform to every scored predicted atom -- this
    # is a transform of the ligand's coordinates, never a fit of them (see
    # the module docstring). It never touches the reference, and never
    # touches pred_atoms in place -- it builds an independent copy so
    # pred_atoms stays exactly as parsed.
    aligned_pred: StructureAtoms = {}
    for rid in receptor_ids + ligand_ids:
        entry = pred_atoms[rid]
        transformed = {
            name: _frozen_vector(*(xyz @ rotation + translation))
            for name, xyz in entry["atoms"].items()
        }
        aligned_pred[rid] = {**entry, "atoms": transformed}

    # --- (2a) Fnat: frame-independent, computed on the raw (unaligned) prediction ---
    fnat, native_contacts, predicted_contacts = compute_fnat(
        ref_atoms, pred_atoms, receptor_ids, ligand_ids, fnat_cutoff,
    )

    # --- (2b) iRMSD: all heavy atoms of reference interface residues ---
    interface_ids = compute_interface_residues(ref_atoms, receptor_ids, ligand_ids, interface_cutoff)
    irmsd: Optional[float] = None
    if interface_ids:
        moving_interface, fixed_interface = _paired_coordinates(ref_atoms, aligned_pred, interface_ids)
        irmsd = _rmsd(moving_interface, fixed_interface)

    # --- (2c) LRMSD: ligand backbone only ---
    moving_ligand, fixed_ligand = _paired_coordinates(ref_atoms, aligned_pred, ligand_ids, BACKBONE_ATOMS)
    lrmsd = _rmsd(moving_ligand, fixed_ligand)

    # --- (2d) receptor-aligned DockQ-style combination ---
    dockq_score, dockq_category = compute_dockq(fnat, irmsd, lrmsd)

    # --- (3) symmetry-corrected Active side-chain RMSD ---
    active_sidechain_rmsd, active_details = compute_active_side_chain_rmsd(
        ref_atoms, aligned_pred, active_ids,
    )

    # --- (4) severe-clash filter on the predicted structure ---
    # Bond adjacency is fixed from the REFERENCE's own geometry (see
    # compute_bonded_exclusions), then applied while scanning the
    # prediction. A rigid transform of the whole structure preserves every
    # internal pairwise distance, so scanning the raw (unaligned) prediction
    # gives identical clash geometry to scanning aligned_pred, at lower cost.
    scored_ids = receptor_ids + ligand_ids
    exclusions = compute_bonded_exclusions(
        ref_atoms, scored_ids,
        peptide_bond_cutoff=peptide_bond_cutoff, disulfide_cutoff=disulfide_cutoff,
    )
    clashes = find_severe_clashes(pred_atoms, scored_ids, exclusions, clash_cutoff=clash_cutoff)

    # --- (5) standard, JSON-serializable result dict ---
    return {
        "dockq_receptor_aligned_variant": dockq_score,
        "dockq_category": dockq_category,
        "dockq_definition": (
            "dockq_receptor_aligned_variant = (Fnat + 1/(1+(iRMSD/1.5)^2) + "
            "1/(1+(LRMSD/8.5)^2)) / 3; based on a single receptor-only rigid-body "
            "fit and all-heavy-atom interface RMSD, NOT the original DockQ "
            "paper's separate interface-backbone superposition of both "
            "partners -- not directly comparable to published DockQ numbers."
        ),
        "fnat": fnat,
        "irmsd_angstrom": irmsd,
        "lrmsd_angstrom": lrmsd,
        "active_sidechain_rmsd_angstrom": active_sidechain_rmsd,
        "active_sidechain_definition": (
            "Symmetry-corrected side-chain heavy-atom RMSD over active_residues "
            "after receptor-frame alignment; PHE/TYR ring CD1/CD2+CE1/CE2 and "
            "other _SYMMETRIC_SWAPS entries are exhaustively compared as one "
            "simultaneous swap vs. the unswapped assignment, minimum kept per residue."
        ),
        "active_sidechain_per_residue": active_details,
        "native_contacts": len(native_contacts),
        "predicted_contacts": len(predicted_contacts),
        "recovered_contacts": len(native_contacts & predicted_contacts),
        "interface_residue_count": len(interface_ids),
        "interface_residues": list(interface_ids),
        "alignment_residue_count": len(alignment_ids),
        "alignment_atom_count": int(moving_alignment.shape[0]),
        "alignment_rmsd_angstrom": alignment_rmsd,
        "alignment_rotation": rotation.tolist(),
        "alignment_translation": translation.tolist(),
        "num_severe_clashes": len(clashes),
        "has_severe_clash": bool(clashes),
        "severe_clash_pairs": [
            {
                "residue_a": c.residue_a, "atom_a": c.atom_a,
                "residue_b": c.residue_b, "atom_b": c.atom_b,
                "distance_angstrom": c.distance_angstrom,
            }
            for c in clashes
        ],
        "clash_definition": (
            f"heavy-atom distance < {clash_cutoff} A; same-residue pairs, "
            "consecutive peptide-bonded C(i)-N(i+1) pairs, and CYS-CYS "
            "disulfide SG-SG pairs excluded (bond adjacency determined from "
            f"the reference structure's own geometry, C-N < {peptide_bond_cutoff} A, "
            f"SG-SG < {disulfide_cutoff} A); not MolProbity clashscore"
        ),
        "receptor_chains": list(receptor_chains),
        "ligand_chains": list(ligand_chains),
        "active_residues": sorted(
            active_ids,
            key=lambda r: (ref_atoms[r]["chain"], ref_atoms[r]["seqid"], ref_atoms[r]["icode"]),
        ),
        "receptor_residue_count": len(receptor_ids),
        "ligand_residue_count": len(ligand_ids),
        "fnat_cutoff_angstrom": fnat_cutoff,
        "interface_cutoff_angstrom": interface_cutoff,
        "clash_cutoff_angstrom": clash_cutoff,
        "peptide_bond_cutoff_angstrom": peptide_bond_cutoff,
        "disulfide_cutoff_angstrom": disulfide_cutoff,
        "model_index": model_index,
        "reference_path": str(reference_path) if reference_path is not None else None,
        "predicted_path": str(predicted_path) if predicted_path is not None else None,
    }


# --------------------------------------------------------------------------
# Multi-stage trajectory evaluation
# --------------------------------------------------------------------------

# Headline metrics tracked stage-over-stage in evaluate_trajectory's deltas.
_TRAJECTORY_HEADLINE_METRICS: Tuple[str, ...] = (
    "active_sidechain_rmsd_angstrom", "irmsd_angstrom", "fnat", "lrmsd_angstrom",
    "num_severe_clashes", "dockq_receptor_aligned_variant",
)


def evaluate_trajectory(
    ref_path: PathLike,
    stage_paths: Mapping[str, PathLike],
    *,
    receptor_chains: Sequence[str],
    ligand_chains: Sequence[str],
    active_residues: Sequence[str] = (),
    fnat_cutoff: float = 5.0,
    interface_cutoff: float = 10.0,
    clash_cutoff: float = 1.5,
    peptide_bond_cutoff: float = 1.9,
    disulfide_cutoff: float = 2.3,
    model_index: int = 0,
) -> Dict[str, Any]:
    """Evaluate every pipeline stage of one complex against the same reference.

    This project's canonical five checkpoints are ``perturbed_input``,
    ``relax_only``, ``discrete_picked``, ``stage1_relaxed`` and
    ``stage2_relaxed`` (see the module docstring), but ``stage_paths`` may
    contain any number of stages under any names; results are reported in
    the order ``stage_paths`` iterates (a plain ``dict`` preserves insertion
    order in Python).

    Args:
        ref_path: Reference structure, shared across every stage.
        stage_paths: Ordered mapping of stage name to that stage's predicted
            structure path.
        receptor_chains, ligand_chains, active_residues, fnat_cutoff,
            interface_cutoff, clash_cutoff, peptide_bond_cutoff,
            disulfide_cutoff, model_index: Forwarded unchanged to
            :func:`evaluate_complex_metrics` for every stage.

    Returns:
        A JSON-serializable dict with:

        * ``"stages"``: a list of per-stage result dicts, each the full
          :func:`evaluate_complex_metrics` output plus ``"stage"`` and
          ``"stage_index"`` (0-based, in iteration order).
        * ``"stage_order"``: the stage names in evaluated order.
        * ``"reference_path"``.

        Every stage dict also carries ``"delta_vs_previous_stage"``: a dict
        of ``{metric_name: current - previous}`` for every metric in
        :data:`_TRAJECTORY_HEADLINE_METRICS` where both this and the
        previous stage have a non-``None`` value (``num_severe_clashes``
        deltas are ``int``, the rest ``float``); a metric missing from
        either stage is simply omitted rather than raising. The first
        stage's ``"delta_vs_previous_stage"`` is always ``{}`` (no previous
        stage to compare against) -- present rather than omitted, so every
        stage dict shares one schema (this matters for
        :func:`append_trajectory_to_csv`, where all stages of one
        trajectory append to the same CSV columns).

    Raises:
        ValueError: If ``stage_paths`` is empty, or (propagated from
            :func:`evaluate_complex_metrics`) on any per-stage evaluation
            failure -- including which stage failed in the exception message.
    """
    if not stage_paths:
        raise ValueError("stage_paths must contain at least one stage")
    stages: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    for index, (stage_name, path) in enumerate(stage_paths.items()):
        try:
            result = evaluate_complex_metrics(
                ref_path, path,
                receptor_chains=receptor_chains, ligand_chains=ligand_chains,
                active_residues=active_residues, fnat_cutoff=fnat_cutoff,
                interface_cutoff=interface_cutoff, clash_cutoff=clash_cutoff,
                peptide_bond_cutoff=peptide_bond_cutoff, disulfide_cutoff=disulfide_cutoff,
                model_index=model_index,
            )
        except ValueError as exc:
            raise ValueError(f"Trajectory stage {stage_name!r} failed: {exc}") from exc
        result["stage"] = stage_name
        result["stage_index"] = index
        # Always present (empty for the first stage), so every stage dict
        # shares one schema -- e.g. for append_trajectory_to_csv, where all
        # stages of one trajectory must append to the same CSV columns.
        deltas: Dict[str, Any] = {}
        if previous is not None:
            for metric in _TRAJECTORY_HEADLINE_METRICS:
                current_value, previous_value = result.get(metric), previous.get(metric)
                if current_value is None or previous_value is None:
                    continue
                deltas[metric] = current_value - previous_value
        result["delta_vs_previous_stage"] = deltas
        stages.append(result)
        previous = result
    return {
        "stages": stages,
        "stage_order": list(stage_paths.keys()),
        "reference_path": str(ref_path),
    }


# --------------------------------------------------------------------------
# Export: standalone JSON or an appended CSV row
# --------------------------------------------------------------------------

def to_json(result: Mapping[str, Any], path: PathLike, *, indent: int = 2) -> None:
    """Write a result dict (already fully JSON-serializable) to ``path``."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=indent, sort_keys=True)
        handle.write("\n")


def flatten_for_csv(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Flatten a result dict into one CSV-ready row.

    Scalar fields (``str``/``int``/``float``/``bool``/``None``) pass through
    unchanged. List/dict-valued fields (``interface_residues``,
    ``severe_clash_pairs``, ``active_sidechain_per_residue``,
    ``alignment_rotation``, ``delta_vs_previous_stage``, ...) are
    JSON-encoded into a single string cell so the row stays one value per
    column; their counts are already available as separate scalar fields
    (``interface_residue_count``, ``num_severe_clashes``, ...) where
    applicable.
    """
    row: Dict[str, Any] = {}
    for key, value in result.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            row[key] = value
        else:
            row[key] = json.dumps(value, sort_keys=True)
    return row


def append_to_csv(result: Mapping[str, Any], path: PathLike) -> None:
    """Append one evaluation's flattened result as a row to ``path``.

    Creates the file with a header on first write. If the file already
    exists, its header's column set is compared against the new row's
    columns; a mismatch raises rather than silently producing a ragged or
    misaligned CSV.

    Raises:
        ValueError: If ``path`` already exists with a different column set.
    """
    row = flatten_for_csv(result)
    csv_path = Path(path)
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    if file_exists:
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if existing_header and set(existing_header) != set(row):
            raise ValueError(
                f"CSV schema mismatch appending to {csv_path}: "
                f"existing columns {sorted(existing_header)} != new columns {sorted(row)}"
            )
        fieldnames = existing_header or sorted(row)
    else:
        fieldnames = sorted(row)
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_trajectory_to_csv(trajectory: Mapping[str, Any], path: PathLike) -> None:
    """Append every stage of one :func:`evaluate_trajectory` result as one CSV row each.

    Each row is that stage's full result dict (including ``"stage"`` and
    ``"stage_index"``), flattened exactly like :func:`flatten_for_csv`; all
    stages of one trajectory share one schema, so a schema mismatch across
    stages -- which should not happen since every stage is produced by the
    same :func:`evaluate_complex_metrics` call shape -- still raises via
    :func:`append_to_csv`.
    """
    for stage_result in trajectory["stages"]:
        append_to_csv(stage_result, path)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _parse_stage_argument(value: str) -> Tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--stage expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--stage expects NAME=PATH, got {value!r}")
    return name, path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Reference structure (.cif or .pdb)")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred", help="Single predicted/relaxed structure (.cif or .pdb)")
    mode.add_argument(
        "--stage", action="append", type=_parse_stage_argument, dest="stages",
        metavar="NAME=PATH",
        help="One pipeline stage (repeatable); switches to trajectory mode",
    )
    parser.add_argument("--receptor-chains", required=True, nargs="+", help="Antigen (receptor) chain IDs")
    parser.add_argument("--ligand-chains", required=True, nargs="+", help="Nanobody (ligand) chain IDs")
    parser.add_argument(
        "--active-residues", nargs="*", default=(),
        help='Residue ids ("CHAIN:SEQID[ICODE]") scored for symmetry-corrected '
             "side-chain RMSD and excluded from the alignment fit where on the receptor",
    )
    parser.add_argument("--fnat-cutoff", type=float, default=5.0)
    parser.add_argument("--interface-cutoff", type=float, default=10.0)
    parser.add_argument("--clash-cutoff", type=float, default=1.5)
    parser.add_argument("--peptide-bond-cutoff", type=float, default=1.9)
    parser.add_argument("--disulfide-cutoff", type=float, default=2.3)
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--json-out", default=None, help="Write the result dict to this JSON path")
    parser.add_argument("--csv-out", default=None, help="Append the flattened result(s) to this CSV path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Evaluate one complex (or trajectory) from the command line and print/export the result."""

    args = _build_argument_parser().parse_args(argv)
    common = dict(
        receptor_chains=args.receptor_chains,
        ligand_chains=args.ligand_chains,
        active_residues=args.active_residues,
        fnat_cutoff=args.fnat_cutoff,
        interface_cutoff=args.interface_cutoff,
        clash_cutoff=args.clash_cutoff,
        peptide_bond_cutoff=args.peptide_bond_cutoff,
        disulfide_cutoff=args.disulfide_cutoff,
        model_index=args.model_index,
    )
    try:
        if args.stages:
            result: Dict[str, Any] = evaluate_trajectory(args.ref, dict(args.stages), **common)
        else:
            result = evaluate_complex_metrics(args.ref, args.pred, **common)
    except (ValueError, OSError) as exc:
        print(f"evaluate_complex_metrics: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json_out:
        to_json(result, args.json_out)
    if args.csv_out:
        if args.stages:
            append_trajectory_to_csv(result, args.csv_out)
        else:
            append_to_csv(result, args.csv_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
