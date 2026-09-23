"""Map a pruned protein-interface graph to an upper-triangular QUBO matrix.

The delivered PyG graphs contain one CA coordinate per residue, not complete
backbone/side-chain atoms. This module therefore implements a deterministic,
coarse-grained rotamer model rather than claiming atomistic Dunbrack accuracy:

* each selected VHH residue first receives a 6--12-state chi1 sub-rotamer pool;
* prior, frozen-environment and antigen-guidance energies pre-screen that pool;
* 3--6 states per residue enter the QUBO under a global <=30-variable budget;
* a local frame derived from neighbouring CA and ligand directions attaches
  residue-specific side-chain pseudo-atoms;
* softened, truncated Lennard-Jones and distance-dependent dielectric Coulomb
  terms score rotamer-environment and rotamer-rotamer interactions.

Energies are in approximate kcal/mol units. The model is suitable for QUBO
algorithm development and NISQ-scale experiments, but should be calibrated or
replaced by an all-atom force field before quantitative affinity claims.

QUBO convention
---------------
``Q`` is upper triangular, including its diagonal, and the binary objective is

``E(x) = constant_offset + sum_i Q[i,i] x_i + sum_{i<j} Q[i,j] x_i x_j``.

This is also equal to ``constant_offset + x.T @ Q @ x`` because the lower
triangle is exactly zero. One-hot penalties contribute ``-lambda`` to each
site-variable diagonal, ``+2*lambda`` between rotamers of the same site, and
``+lambda`` per site to ``constant_offset``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from nanoqc.structure.residue_tables import SIDECHAIN_HEAVY_ATOMS, SYMMETRIC_SWAPS
from nanoqc.data.safe_graph_load import load_graph


# Real-atom validation is separate from the coarse-grained QUBO force field.
# Canonical heavy-atom names follow the wwPDB amino-acid convention.
_SIDECHAIN_NAMES = {name: " ".join(atoms) for name, atoms in SIDECHAIN_HEAVY_ATOMS.items()}
_CHI_ATOMS = {
    "SER": (("N","CA","CB","OG"),),
    "THR": (("N","CA","CB","OG1"),),
    "CYS": (("N","CA","CB","SG"),),
    "VAL": (("N","CA","CB","CG1"),),
    "ILE": (("N","CA","CB","CG1"),("CA","CB","CG1","CD1")),
    "LEU": (("N","CA","CB","CG"),("CA","CB","CG","CD1")),
    "ASP": (("N","CA","CB","CG"),("CA","CB","CG","OD1")),
    "ASN": (("N","CA","CB","CG"),("CA","CB","CG","OD1")),
    "GLU": (("N","CA","CB","CG"),("CA","CB","CG","CD"),("CB","CG","CD","OE1")),
    "GLN": (("N","CA","CB","CG"),("CA","CB","CG","CD"),("CB","CG","CD","OE1")),
    "LYS": (("N","CA","CB","CG"),("CA","CB","CG","CD"),("CB","CG","CD","CE"),("CG","CD","CE","NZ")),
    "ARG": (("N","CA","CB","CG"),("CA","CB","CG","CD"),("CB","CG","CD","NE"),("CG","CD","NE","CZ")),
    "MET": (("N","CA","CB","CG"),("CA","CB","CG","SD"),("CB","CG","SD","CE")),
    "HIS": (("N","CA","CB","CG"),("CA","CB","CG","ND1")),
    "PHE": (("N","CA","CB","CG"),("CA","CB","CG","CD1")),
    "TYR": (("N","CA","CB","CG"),("CA","CB","CG","CD1")),
    "TRP": (("N","CA","CB","CG"),("CA","CB","CG","CD1")),
}
_SYMMETRIC_SWAPS = {name: list(pairs) for name, pairs in SYMMETRIC_SWAPS.items()}


def read_atomistic_structure(path: Path, model_index: int = 0) -> dict[str, dict[str, Any]]:
    """Read canonical protein heavy atoms with author chain:residue IDs.

    Choose one coherent alternate conformer per residue by mean occupancy
    (lexicographic tie break), plus shared blank-altloc atoms. Never fill a
    missing atom from another conformer. Waters/non-protein ligands are excluded;
    modified amino acids are rejected rather than silently mapped to canonical.
    """
    import gemmi

    structure = gemmi.read_structure(str(path))
    if not 0 <= model_index < len(structure):
        raise ValueError(f"Invalid model index {model_index}: {path}")
    residues: dict[str, dict[str, Any]] = {}
    for chain in structure[model_index]:
        for residue in chain:
            if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                continue
            if residue.name not in _SIDECHAIN_NAMES:
                raise ValueError(f"Unsupported modified amino acid: {chain.name}:{residue.seqid} {residue.name}")
            rid = f"{chain.name}:{residue.seqid}"
            if rid in residues:
                raise ValueError(f"Ambiguous author residue ID: {rid}")
            atoms = [a for a in residue if a.element.name not in ("H", "D") and a.occ > 0]
            labels = sorted({a.altloc for a in atoms if a.altloc not in ("\x00", " ", "")})
            label = min(labels, key=lambda c: (-np.mean([a.occ for a in atoms if a.altloc == c]), c)) if labels else ""
            coords = {}
            for atom in atoms:
                if atom.altloc not in ("\x00", " ", "", label):
                    continue
                name = atom.name.strip()
                if name in coords:
                    raise ValueError(f"Duplicate atom after conformer selection: {rid}/{name}")
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float)
                if not np.isfinite(xyz).all():
                    raise ValueError(f"Nonfinite atom: {rid}/{name}")
                coords[name] = xyz
            residues[rid] = dict(name=residue.name, atoms=coords, altloc=label)
    if not residues:
        raise ValueError(f"No canonical protein residues: {path}")
    return residues


_RESIDUE_ID_PATTERN = re.compile(r"^(?P<chain>.+):(?P<seqid>-?\d+)(?P<icode>[A-Za-z]?)$")


def _to_structure_atoms(residues: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert this module's ``read_atomistic_structure`` shape to
    ``evaluate_complex_metrics``'s ``StructureAtoms`` shape.

    ``read_atomistic_structure`` residue entries carry ``{"name", "atoms",
    "altloc"}`` keyed by author ``"CHAIN:SEQID[ICODE]"`` id;
    ``evaluate_complex_metrics.read_structure_atoms`` additionally splits
    that same id into explicit ``"chain"``/``"seqid"``/``"icode"`` fields on
    each entry. This is a pure reparsing of the SAME id string both modules
    already use (the identical ``"CHAIN:SEQID[ICODE]"`` convention
    documented in both modules' docstrings) -- it re-reads nothing from disk
    and re-derives nothing geometric, so it cannot introduce a
    parsing-vs-geometry inconsistency between the two shapes.

    Raises:
        ValueError: If a residue id does not match the expected
            ``"CHAIN:SEQID[ICODE]"`` pattern.
    """
    converted: dict[str, dict[str, Any]] = {}
    for rid, entry in residues.items():
        match = _RESIDUE_ID_PATTERN.match(rid)
        if match is None:
            raise ValueError(f"Cannot parse residue id into chain/seqid/icode: {rid!r}")
        converted[rid] = dict(
            chain=match.group("chain"), seqid=int(match.group("seqid")),
            icode=match.group("icode") or "", name=entry["name"],
            altloc=entry["altloc"], atoms=entry["atoms"],
        )
    return converted


def _rigid_fit(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return proper row-vector Kabsch rotation and translation; no reflection."""
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Alignment arrays must share shape [N,3]")
    a, b = moving - moving.mean(0), reference - reference.mean(0)
    if len(a) < 3 or min(np.linalg.matrix_rank(a), np.linalg.matrix_rank(b)) < 2:
        raise ValueError("Alignment needs at least three non-collinear atoms")
    u, _, vt = np.linalg.svd(a.T @ b)
    rotation = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return rotation, reference.mean(0) - moving.mean(0) @ rotation


def _torsion_angle_degrees(p0, p1, p2, p3) -> float:
    """Signed four-point torsion in degrees."""
    p0,p1,p2,p3=(np.asarray(p,dtype=float) for p in (p0,p1,p2,p3))
    b0=-(p1-p0); b1=p2-p1; b2=p3-p2
    norm=np.linalg.norm(b1)
    if norm<1e-10: raise ValueError("Degenerate torsion axis")
    b1=b1/norm
    v=b0-np.dot(b0,b1)*b1
    w=b2-np.dot(b2,b1)*b1
    if min(np.linalg.norm(v),np.linalg.norm(w))<1e-10:
        raise ValueError("Undefined torsion")
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))

def _backbone_phi_psi(residues: Mapping[str, Mapping[str, Any]], rid: str) -> Tuple[float,float]:
    """Backbone phi/psi for one author residue id; termini fail closed."""
    match=_RESIDUE_ID_PATTERN.match(rid)
    if match is None: raise ValueError(f"Cannot parse residue id {rid}")
    chain=match.group("chain")
    ordered=[]
    for key in residues:
        m=_RESIDUE_ID_PATTERN.match(key)
        if m and m.group("chain")==chain:
            ordered.append((int(m.group("seqid")),m.group("icode") or "",key))
    ordered.sort()
    keys=[item[2] for item in ordered]
    idx=keys.index(rid)
    if idx==0 or idx==len(keys)-1:
        raise ValueError(f"Dunbrack mode requires non-terminal Active residue: {rid}")
    prev,current,nxt=residues[keys[idx-1]],residues[rid],residues[keys[idx+1]]
    for entry,names in ((prev,("C",)),(current,("N","CA","C")),(nxt,("N",))):
        missing=[name for name in names if name not in entry["atoms"]]
        if missing: raise ValueError(f"Missing backbone atoms {missing} for Dunbrack lookup at {rid}")
    phi=_torsion_angle_degrees(prev["atoms"]["C"],current["atoms"]["N"],current["atoms"]["CA"],current["atoms"]["C"])
    psi=_torsion_angle_degrees(current["atoms"]["N"],current["atoms"]["CA"],current["atoms"]["C"],nxt["atoms"]["N"])
    return phi,psi

def _chi1_angle(atoms: Mapping[str, np.ndarray], residue_name: str) -> Optional[float]:
    """Signed N-CA-CB-X torsion in degrees; Ala/Gly have no chi1."""
    if residue_name in ("ALA", "GLY"):
        return None
    fourth = {"SER": "OG", "THR": "OG1", "CYS": "SG", "VAL": "CG1", "ILE": "CG1"}.get(residue_name, "CG")
    p0, p1, p2, p3 = [atoms[n] for n in ("N", "CA", "CB", fourth)]
    axis = p2-p1
    if np.linalg.norm(axis) < 1e-8:
        raise ValueError("Degenerate chi1 axis")
    axis = axis / np.linalg.norm(axis)
    v, w = p0-p1, p3-p2
    v, w = v-np.dot(v, axis)*axis, w-np.dot(w, axis)*axis
    if min(np.linalg.norm(v), np.linalg.norm(w)) < 1e-8:
        raise ValueError("Undefined chi1 torsion")
    return float(np.degrees(np.arctan2(np.dot(np.cross(axis, v), w), np.dot(v, w))))


def _sidechain_chi_angles(
    atoms: Mapping[str, np.ndarray], residue_name: str
) -> Tuple[float, ...]:
    """Return every defined canonical side-chain chi angle."""
    definitions=_CHI_ATOMS.get(residue_name,())
    values=[]
    for definition in definitions:
        missing=[name for name in definition if name not in atoms]
        if missing:
            raise ValueError(f"Missing chi atoms for {residue_name}: {missing}")
        values.append(_torsion_angle_degrees(*(atoms[name] for name in definition)))
    return tuple(values)



def _rotate_about_axis(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis=np.asarray(axis,dtype=float)
    norm=np.linalg.norm(axis)
    if norm<1e-10:
        raise ValueError("Degenerate rotation axis")
    axis=axis/norm
    relative=np.asarray(points,dtype=float)-origin
    theta=np.deg2rad(float(angle_degrees))
    return (origin + relative*np.cos(theta)
            + np.cross(axis,relative)*np.sin(theta)
            + np.outer(relative@axis,axis)*(1-np.cos(theta)))


def _downstream_atoms(
    bond_graph: Mapping[int,set[int]], start: int, blocked: int, allowed: set[int]
) -> set[int]:
    """Atoms on the distal side of one rotatable bond, restricted to one residue."""
    seen={blocked}; stack=[start]; result=set()
    while stack:
        atom=stack.pop()
        if atom in seen or atom not in allowed:
            continue
        seen.add(atom);result.add(atom)
        stack.extend(neighbor for neighbor in bond_graph[atom] if neighbor not in seen)
    return result


def _apply_sidechain_chis(
    positions: np.ndarray,
    atoms: Mapping[str,int],
    bond_graph: Mapping[int,set[int]],
    residue_name: str,
    targets: Sequence[float],
) -> np.ndarray:
    """Set χ1..χn sequentially for one acyclic canonical side chain."""
    definitions=_CHI_ATOMS.get(residue_name,())
    if len(targets)>len(definitions):
        raise ValueError(f"Too many chi targets for {residue_name}: {len(targets)}>{len(definitions)}")
    result=np.asarray(positions,dtype=float).copy()
    allowed=set(atoms.values())
    for definition,target in zip(definitions,targets):
        if not all(name in atoms for name in definition):
            raise ValueError(f"Missing chi atoms for {residue_name}: {definition}")
        a,b,c,d=(atoms[name] for name in definition)
        current=_torsion_angle_degrees(result[a],result[b],result[c],result[d])
        delta=((float(target)-current+180.0)%360.0)-180.0
        moving=sorted(_downstream_atoms(bond_graph,c,b,allowed))
        if d not in moving:
            raise ValueError(f"Chi downstream graph is inconsistent for {residue_name}: {definition}")
        result[moving]=_rotate_about_axis(result[moving],result[b],result[c]-result[b],delta)
        observed=_torsion_angle_degrees(result[a],result[b],result[c],result[d])
        if abs(((observed-float(target)+180.0)%360.0)-180.0)>1e-5:
            raise AssertionError(f"Failed to set torsion {definition}: target={target}, observed={observed}")
    return result


def evaluate_atomistic_prediction(
    reference_path: Path, prediction_path: Path, *, active_residues: Sequence[str],
    alignment_residues: Sequence[str], partner_residues: Sequence[str],
    contact_cutoff: float = 5.0, proximity_cutoff: float = 2.0,
    chi1_tolerance: float = 20.0, model_index: int = 0,
) -> dict[str, Any]:
    """Evaluate real predicted atoms against a held-out reference structure.

    The caller fixes all residue selections before seeing reference accuracy.
    Alignment uses N/CA/C/O of non-Active residues, never fits Active side chains.
    Report symmetry-corrected side-chain heavy-atom RMSD and chi1 recovery.
    Contacts are residue pairs (any protein heavy atoms <= cutoff). Severe
    proximity is an explicitly geometric Active-sidechain/partner atom count,
    NOT MolProbity clashscore or a bonded-exclusion-aware force-field score.
    This function does not create, minimize, or validate an all-atom force field.
    """
    if any(not math.isfinite(v) or v <= 0 for v in (contact_cutoff, proximity_cutoff, chi1_tolerance)) or chi1_tolerance > 180:
        raise ValueError("Invalid distance/torsion thresholds")
    selections = [list(active_residues), list(alignment_residues), list(partner_residues)]
    if any(not ids or len(ids) != len(set(ids)) for ids in selections):
        raise ValueError("Active, alignment and partner selections must be nonempty and unique")
    active, alignment, partners = selections
    if set(active) & (set(alignment) | set(partners)):
        raise ValueError("Active must be disjoint from alignment and partners")
    if {r.rsplit(":", 1)[0] for r in active} & {r.rsplit(":", 1)[0] for r in partners}:
        raise ValueError("Partner must be a separate chain for interface evaluation")
    ref, pred = read_atomistic_structure(reference_path, model_index), read_atomistic_structure(prediction_path, model_index)
    backbone = ("N", "CA", "C", "O")
    for rid in set(active + alignment + partners):
        if rid not in ref or rid not in pred:
            raise ValueError(f"Missing selected residue: {rid}")
        if ref[rid]["name"] != pred[rid]["name"]:
            raise ValueError(f"Residue identity mismatch: {rid}")
        required = set(backbone)
        if rid in set(active + partners):
            required.update(_SIDECHAIN_NAMES[ref[rid]["name"]].split())
        for label, structure in (("reference", ref), ("prediction", pred)):
            missing = required - structure[rid]["atoms"].keys()
            if missing:
                raise ValueError(f"{label} missing heavy atoms at {rid}: {sorted(missing)}")
    moving = np.array([pred[r]["atoms"][n] for r in alignment for n in backbone])
    fixed = np.array([ref[r]["atoms"][n] for r in alignment for n in backbone])
    rotation, translation = _rigid_fit(moving, fixed)
    aligned = {r: {n: xyz @ rotation + translation for n, xyz in entry["atoms"].items()} for r, entry in pred.items()}
    details, errors, backbone_errors, recovered = [], [], [], []
    chi_recovery_by_index: dict[int,list[bool]] = {}
    all_chi_recovered: list[bool] = []
    for rid in active:
        name, ra, pa = ref[rid]["name"], ref[rid]["atoms"], aligned[rid]
        names = _SIDECHAIN_NAMES[name].split()
        candidates = [pa]
        if name in _SYMMETRIC_SWAPS:
            swapped = dict(pa)
            for a, b in _SYMMETRIC_SWAPS[name]:
                swapped[a], swapped[b] = pa[b], pa[a]
            candidates.append(swapped)
        squared = [sum(float(np.sum((ca[n]-ra[n])**2)) for n in names) for ca in candidates]
        error = min(squared)
        errors.extend([error, len(names)])
        backbone_errors.extend(float(np.sum((pa[n]-ra[n])**2)) for n in backbone)
        ref_chis=_sidechain_chi_angles(ra,name)
        candidate_chis=[_sidechain_chi_angles(ca,name) for ca in candidates]
        chi_errors=[]
        for chi_index,ref_angle in enumerate(ref_chis):
            values=[
                abs(((chis[chi_index]-ref_angle+180.0)%360.0)-180.0)
                for chis in candidate_chis if len(chis)>chi_index
            ]
            chi_errors.append(min(values) if values else None)
        chi_error=chi_errors[0] if chi_errors else None
        if chi_error is not None:
            recovered.append(chi_error <= chi1_tolerance)
        flags=[]
        for chi_index,value in enumerate(chi_errors,1):
            if value is None:
                continue
            flag=bool(value<=chi1_tolerance)
            chi_recovery_by_index.setdefault(chi_index,[]).append(flag)
            flags.append(flag)
        if flags:
            all_chi_recovered.append(all(flags))
        details.append(dict(
            residue_id=rid,residue_name=name,sidechain_atoms=len(names),
            sidechain_rmsd_angstrom=math.sqrt(error/len(names)) if names else None,
            chi1_error_degrees=chi_error,
            chi1_recovered=chi_error <= chi1_tolerance if chi_error is not None else None,
            chi_errors_degrees=[None if value is None else float(value) for value in chi_errors],
            chi_recovered=[None if value is None else bool(value<=chi1_tolerance) for value in chi_errors],
            all_chi_recovered=(all(flags) if flags else None),
            reference_altloc=ref[rid]["altloc"],prediction_altloc=pred[rid]["altloc"]))

    def interface_metrics(structure: dict) -> tuple[set, int, int]:
        contacts, close_pairs, possible_pairs = set(), 0, 0
        for a in active:
            aa = structure[a]["atoms"]
            all_a = np.array(list(aa.values()))
            side_a = np.array([aa[n] for n in _SIDECHAIN_NAMES[structure[a]["name"]].split()]).reshape(-1, 3)
            for b in partners:
                all_b = np.array(list(structure[b]["atoms"].values()))
                if np.min(np.sum((all_a[:, None]-all_b)**2, axis=-1)) <= contact_cutoff**2:
                    contacts.add((a, b))
                distances = np.sum((side_a[:, None]-all_b)**2, axis=-1)
                close_pairs += int(np.count_nonzero(distances < proximity_cutoff**2))
                possible_pairs += int(distances.size)
        return contacts, close_pairs, possible_pairs

    ref_contacts, ref_close, _ = interface_metrics(ref)
    pred_contacts, pred_close, possible_pairs = interface_metrics(pred)
    overlap = len(ref_contacts & pred_contacts)
    precision = overlap/len(pred_contacts) if pred_contacts else None
    recall = overlap/len(ref_contacts) if ref_contacts else None
    atom_count = sum(errors[1::2])
    if not set(alignment).issubset(partners):
        raise ValueError("Alignment residues must belong to the non-Active receptor")
    # (P3 remediation) evaluate_complex_metrics.evaluate_complex_metrics_from_atoms
    # is now the single source of truth for Fnat/iRMSD/LRMSD/DockQ/severe-clash
    # metrics, replacing the legacy structural_quality.docking_quality call for
    # every metric it covers. It runs on the SAME already-parsed `ref`/`pred`
    # atoms this function read above via read_atomistic_structure (converted to
    # evaluate_complex_metrics's StructureAtoms shape by _to_structure_atoms
    # below) rather than a second, independent read_structure_atoms parse of
    # reference_path/prediction_path -- both because the two files may not
    # even exist as real files (this function's own required-atoms checks
    # above, and this project's existing offline tests, exercise it entirely
    # through synthetic in-memory fixtures), and to avoid parsing the same
    # structures twice. It additionally enforces full-chain, residue-by-residue
    # heavy-atom completeness across EVERY residue of receptor_chains/
    # ligand_chains (see verify_full_chain_heavy_atom_completeness), a
    # strictly stronger check than this function's own required-atoms loop
    # above, which only validates the active/alignment/partner selections,
    # not every receptor/ligand residue -- deleting an off-interface,
    # non-selected heavy atom (e.g. receptor A:20 CB) now halts evaluation
    # here too, not just when evaluate_complex_metrics is called directly.
    #
    # receptor_chains/ligand_chains are derived exactly as docking_quality
    # itself used to derive ligand_chains internally (chain IDs of `active`),
    # extended symmetrically to `partners` for the receptor side, so this is
    # a behavior-preserving substitution, not a change of scope. Its own
    # alignment fit spans every residue of receptor_chains (not only this
    # function's curated `alignment` subset), so it requires at least 3
    # non-Active receptor residues -- widen `partners`/`receptor_chains` if a
    # ValueError names this requirement.
    from nanoqc.structure.evaluate_complex_metrics import evaluate_complex_metrics_from_atoms as _evaluate_complex_metrics_from_atoms
    ligand_chain_ids = {r.rsplit(":", 1)[0] for r in active}
    receptor_chain_ids = {r.rsplit(":", 1)[0] for r in partners}
    complex_metrics = _evaluate_complex_metrics_from_atoms(
        _to_structure_atoms(ref), _to_structure_atoms(pred),
        receptor_chains=sorted(receptor_chain_ids), ligand_chains=sorted(ligand_chain_ids),
        active_residues=active, model_index=model_index,
        reference_path=reference_path, predicted_path=prediction_path,
    )
    # The backbone-only interface-superposition DockQ variant
    # (dockq_backbone_score/irmsd_interface_backbone_fit) has no equivalent in
    # evaluate_complex_metrics yet: evaluate_complex_metrics's own alignment
    # fits on the FULL receptor_chains backbone, not this function's curated
    # `alignment` residue subset. It remains the one field genuinely still
    # sourced from the legacy function -- using the SAME `aligned` prediction
    # this function already computed above from its own curated alignment
    # fit, not a second, independent alignment -- rather than fabricating a
    # value evaluate_complex_metrics never computed.
    from nanoqc.structure.structural_quality import docking_quality as _legacy_docking_quality
    _legacy_backbone_variant = _legacy_docking_quality(ref, pred, aligned, active, partners, _rigid_fit)
    # NOTE: evaluate_complex_metrics's own result carries alignment_atom_count/
    # alignment_rmsd_angstrom/alignment_rotation/alignment_translation keys
    # too -- describing ITS OWN full-receptor-chain alignment, a DIFFERENT
    # fit from this function's curated `alignment` residue subset used below
    # for chi1/sidechain/proximity. Splatting complex_metrics directly into
    # this function's own return dict would silently collide on those names
    # (or raise TypeError, as a literal-kwargs dict(**complex_metrics,
    # alignment_atom_count=...) does) and overwrite one alignment's numbers
    # with the other's under an identical, now-ambiguous key. To avoid that,
    # only the specific legacy-compatible fields are hoisted to the top
    # level below, and the complete, unmodified evaluate_complex_metrics
    # result is kept fully available, clearly namespaced, under
    # "complex_metrics" -- nothing from it is lost, just not silently mixed
    # with this function's own differently-scoped alignment fields.
    quality = dict(
        # Canonical new field name (single source of truth going forward).
        dockq_receptor_aligned_variant=complex_metrics["dockq_receptor_aligned_variant"],
        dockq_category=complex_metrics["dockq_category"],
        dockq_definition=complex_metrics["dockq_definition"],
        fnat=complex_metrics["fnat"],
        irmsd_angstrom=complex_metrics["irmsd_angstrom"],
        lrmsd_angstrom=complex_metrics["lrmsd_angstrom"],
        num_severe_clashes=complex_metrics["num_severe_clashes"],
        has_severe_clash=complex_metrics["has_severe_clash"],
        severe_clash_pairs=complex_metrics["severe_clash_pairs"],
        clash_definition=complex_metrics["clash_definition"],
        # Backward-compatible aliases for existing downstream consumers
        # (e.g. batch_benchmark_hard_set.py) that still index the legacy
        # structural_quality.docking_quality key names directly.
        dockq_score=complex_metrics["dockq_receptor_aligned_variant"],
        irmsd=complex_metrics["irmsd_angstrom"],
        lrmsd=complex_metrics["lrmsd_angstrom"],
        docking_native_contacts=complex_metrics["native_contacts"],
        docking_predicted_contacts=complex_metrics["predicted_contacts"],
        docking_interface_residues=complex_metrics["interface_residue_count"],
        # The backbone-only interface-superposition DockQ variant has no
        # equivalent in evaluate_complex_metrics yet (see comment above) --
        # sourced from the legacy function using this function's OWN
        # curated-alignment `aligned` prediction, not a second alignment.
        dockq_backbone_score=_legacy_backbone_variant["dockq_backbone_score"],
        irmsd_interface_backbone_fit=_legacy_backbone_variant["irmsd_interface_backbone_fit"],
        # Full evaluate_complex_metrics output (every field it returns,
        # including its own, differently-scoped alignment_*/receptor_chains/
        # active_sidechain_per_residue/etc.), for anyone who wants the
        # complete new-schema result rather than only the aliased subset above.
        complex_metrics=complex_metrics,
    )
    return dict(**quality,active_count=len(active), alignment_atom_count=len(moving),
        alignment_rmsd_angstrom=float(np.sqrt(np.mean(np.sum((moving @ rotation+translation-fixed)**2, axis=1)))),
        active_backbone_rmsd_angstrom=float(np.sqrt(np.mean(backbone_errors))),
        sidechain_heavy_atom_count=int(atom_count),
        sidechain_rmsd_angstrom=math.sqrt(sum(errors[::2])/atom_count) if atom_count else None,
        chi1_evaluable_residues=len(recovered), chi1_recovery_rate=float(np.mean(recovered)) if recovered else None,
        chi_recovery_rates={
            f"chi{index}": (float(np.mean(flags)) if flags else None)
            for index,flags in sorted(chi_recovery_by_index.items())
        },
        all_chi_recovery_rate=(float(np.mean(all_chi_recovered)) if all_chi_recovered else None),
        reference_contact_count=len(ref_contacts), prediction_contact_count=len(pred_contacts),
        recovered_contact_count=overlap, contact_precision=precision, contact_recall=recall,
        contact_f1=2*overlap/(len(ref_contacts)+len(pred_contacts)) if ref_contacts or pred_contacts else None,
        reference_severe_proximity_pairs=ref_close, prediction_severe_proximity_pairs=pred_close,
        proximity_atom_pair_denominator=possible_pairs,
        severe_proximity_pair_delta=pred_close-ref_close, per_residue=details,
        alignment_rotation=rotation.tolist(), alignment_translation=translation.tolist(),
        contact_cutoff_angstrom=contact_cutoff, proximity_cutoff_angstrom=proximity_cutoff,
        chi1_tolerance_degrees=chi1_tolerance)


AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}
COULOMB_KCAL_ANGSTROM = 332.06371


@dataclass(frozen=True)
class ForceFieldConfig:
    """Parameters for the coarse-grained non-bonded energy model."""

    cutoff_angstrom: float = 8.0
    softcore_delta_angstrom: float = 0.5
    hard_core_fraction: float = 0.72
    hard_sphere_penalty: float = 25.0
    lj_repulsion_cap: float = 50.0
    lj_attraction_cap: float = 5.0
    coulomb_cap: float = 20.0
    dielectric_base: float = 4.0
    dielectric_slope: float = 2.0
    thermal_energy_kcal: float = 0.593

    def __post_init__(self) -> None:
        """Reject non-physical or numerically unsafe parameter choices."""

        positive = {
            "cutoff_angstrom": self.cutoff_angstrom,
            "softcore_delta_angstrom": self.softcore_delta_angstrom,
            "hard_core_fraction": self.hard_core_fraction,
            "hard_sphere_penalty": self.hard_sphere_penalty,
            "lj_repulsion_cap": self.lj_repulsion_cap,
            "lj_attraction_cap": self.lj_attraction_cap,
            "coulomb_cap": self.coulomb_cap,
            "dielectric_base": self.dielectric_base,
            "thermal_energy_kcal": self.thermal_energy_kcal,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"Force-field parameters must be positive: {positive}")
        if self.dielectric_slope < 0:
            raise ValueError("dielectric_slope cannot be negative")


@dataclass(frozen=True)
class EnergyCalibration:
    """Frozen linear calibration from coarse components to all-atom energy deltas."""

    prior_weight: float = 1.0
    vhh_environment_weight: float = 1.0
    antigen_weight: float = 1.0
    pair_weight: float = 1.0
    intercept: float = 0.0
    source: str = "uncalibrated"

    @classmethod
    def from_json(cls, path: Path) -> "EnergyCalibration":
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        required=("prior_weight","vhh_environment_weight","antigen_weight","pair_weight","intercept")
        missing=[key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Calibration file missing keys: {missing}")
        values={key:float(payload[key]) for key in required}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Calibration coefficients must be finite")
        for key in ("prior_weight","vhh_environment_weight","antigen_weight","pair_weight"):
            if values[key] < 0:
                raise ValueError(f"Calibration component weight must be nonnegative: {key}={values[key]}")
        scope=str(payload.get("scope",""))
        if "training complexes only" not in scope:
            raise ValueError(
                "Calibration provenance must state that coefficients were fit on training complexes only"
            )
        if int(payload.get("n_train_complexes",0)) < 2:
            raise ValueError("Calibration must report at least two training complexes")
        return cls(**values,source=str(path))

@dataclass(frozen=True)
class RotamerTemplate:
    """One statistically defined rotamer state."""

    chi1_degrees: float
    prior_probability: float
    chi_degrees: Tuple[float, ...] = ()
    chi_sigmas: Tuple[float, ...] = ()
    source: str = "legacy"


@dataclass
class RotamerState:
    """Generated pseudo-atom representation of one rotamer microstate."""

    site_index: int
    node_index: int
    amino_acid: str
    rotamer_index: int
    chi1_degrees: float
    prior_probability: float
    positions: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    charges: np.ndarray
    chi_degrees: Tuple[float, ...] = ()
    chi_sigmas: Tuple[float, ...] = ()
    prior_energy: float = 0.0
    environment_energy: float = 0.0
    antigen_guidance_energy: float = 0.0

    @property
    def self_energy(self) -> float:
        """Return prior + VHH-fixed environment + antigen interaction once."""

        return self.prior_energy + self.environment_energy + self.antigen_guidance_energy


def chi1_well_index(angle: float) -> int:
    """Assign chi1 to the nearest gauche+, gauche- or trans well."""
    centers = (60.0, -60.0, 180.0)
    return min(range(3), key=lambda index: abs((angle - centers[index] + 180.0) % 360.0 - 180.0))


def select_chi1_well_representatives(states: Sequence[RotamerState], count: int = 3) -> list[RotamerState]:
    """Cover all chi1 wells, then fill remaining slots by calibrated energy."""
    if count < 3:
        raise ValueError("Chi1-well coverage requires at least three states")
    selected: dict[int, RotamerState] = {}
    for state in states:
        selected.setdefault(chi1_well_index(state.chi1_degrees), state)
    if len(selected) != 3:
        raise ValueError("Fixed three-state scaling requires a candidate in each chi1 well")
    chosen_ids = {id(state) for state in selected.values()}
    for state in states:
        if len(chosen_ids) >= count:
            break
        chosen_ids.add(id(state))
    if len(chosen_ids) != count:
        raise ValueError(f"Only {len(chosen_ids)} rotamers available for {count} states/site")
    return [state for state in states if id(state) in chosen_ids]


@dataclass(frozen=True)
class VariableRecord:
    """Trace one QUBO bit to its residue and rotamer state."""

    variable_index: int
    site_index: int
    node_index: int
    original_node_index: int
    residue_id: str
    amino_acid: str
    rotamer_index: int
    chi1_degrees: float
    prior_probability: float
    self_energy: float


@dataclass
class QUBOResult:
    """Complete QUBO delivery including physical components and provenance."""

    Q: np.ndarray
    variable_map: Tuple[VariableRecord, ...]
    site_to_variables: Dict[int, Tuple[int, ...]]
    lambda_value: float
    lambda_lower_bound: float
    constant_offset: float
    physical_self: np.ndarray
    physical_pair: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def energy(self, binary_state: Sequence[int]) -> float:
        """Evaluate the complete constrained QUBO energy for one bit string."""

        x = np.asarray(binary_state, dtype=np.float64)
        if x.shape != (self.Q.shape[0],):
            raise ValueError(f"Expected {self.Q.shape[0]} bits, got {x.shape}")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("binary_state must contain only 0 and 1")
        return float(self.constant_offset + x @ self.Q @ x)

    def mapping_table(self) -> list[dict[str, Any]]:
        """Return JSON/CSV-friendly variable mapping rows."""

        return [asdict(record) for record in self.variable_map]

    def export(self, output_dir: Path, stem: str = "interface") -> Dict[str, Path]:
        """Persist matrix components and a human-readable mapping manifest.

        Returns paths to an ``.npz`` bundle and a ``.json`` manifest. Existing
        files with the same names are replaced intentionally by this explicit
        method call.
        """

        if not stem or Path(stem).name != stem:
            raise ValueError("stem must be one plain filename component")
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = output_dir / f"{stem}_qubo.npz"
        manifest_path = output_dir / f"{stem}_mapping.json"
        np.savez_compressed(
            matrix_path,
            Q=self.Q,
            physical_self=self.physical_self,
            physical_pair=self.physical_pair,
            lambda_value=np.asarray(self.lambda_value),
            lambda_lower_bound=np.asarray(self.lambda_lower_bound),
            constant_offset=np.asarray(self.constant_offset),
        )
        manifest = {
            "variable_map": self.mapping_table(),
            "site_to_variables": {
                str(site): list(variables)
                for site, variables in self.site_to_variables.items()
            },
            "lambda_value": self.lambda_value,
            "lambda_lower_bound": self.lambda_lower_bound,
            "constant_offset": self.constant_offset,
            "metadata": self.metadata,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"matrix": matrix_path, "mapping": manifest_path}


_THREE_LETTER = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR",
}

def _nearest_dunbrack_bin(angle: float) -> int:
    """Nearest 10-degree backbone bin in the Dunbrack 2010 library."""
    if not math.isfinite(angle):
        raise ValueError("Dunbrack lookup requires finite backbone phi/psi")
    value=int(round(float(angle)/10.0)*10)
    while value>180: value-=360
    while value<-180: value+=360
    return value

def _load_dunbrack_bins(library_path: Path, requested_bins: set[tuple[str,int,int]]) -> Dict[tuple[str,int,int], list[RotamerTemplate]]:
    """Read requested residue/phi/psi bins from ALL.bbdep.rotamers.lib."""
    path=Path(library_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dunbrack 2010 rotamer library not found: {path}")
    found={key:[] for key in requested_bins}
    with path.open("r",encoding="utf-8",errors="replace") as handle:
        for raw in handle:
            line=raw.strip()
            if not line or line.startswith("#"): continue
            fields=line.split()
            if len(fields)<17:
                continue
            try:
                residue=fields[0].upper()
                phi=float(fields[1]); psi=float(fields[2])
                if residue not in _THREE_LETTER.values():
                    continue
                if not (-180.0 <= phi <= 180.0 and -180.0 <= psi <= 180.0):
                    raise ValueError(f"Invalid Dunbrack backbone bin: {residue} {phi} {psi}")
                if abs(phi/10.0-round(phi/10.0))>1e-6 or abs(psi/10.0-round(psi/10.0))>1e-6:
                    raise ValueError(f"Dunbrack traditional library must use 10-degree bins: {residue} {phi} {psi}")
                key=(residue,int(round(phi)),int(round(psi)))
            except ValueError:
                continue
            if key not in found: continue
            probability=float(fields[8])
            chis=tuple(float(v) for v in fields[9:13])
            sigmas=tuple(float(v) for v in fields[13:17])
            nz=[i for i,(chi,sigma) in enumerate(zip(chis,sigmas)) if abs(chi)>1e-12 or abs(sigma)>1e-12]
            n=(max(nz)+1) if nz else 1
            found[key].append(RotamerTemplate(chis[0],probability,chis[:n],sigmas[:n],"dunbrack2010"))
    missing=[key for key,rows in found.items() if not rows]
    if missing:
        raise ValueError(f"Dunbrack library lacks requested bins: {missing[:8]}")
    normalized={}
    for key,rows in found.items():
        total=sum(max(0.0,row.prior_probability) for row in rows)
        if total<=0: raise ValueError(f"Dunbrack probabilities sum to zero at {key}")
        normalized[key]=[RotamerTemplate(r.chi1_degrees,r.prior_probability/total,r.chi_degrees,r.chi_sigmas,r.source) for r in rows]
    return normalized

def _dunbrack_templates_for_site(library_bins, amino_acid: str, phi: float, psi: float, *, probability_floor: float, sigma_offsets: Sequence[float]) -> Tuple[RotamerTemplate, ...]:
    """Expand Dunbrack rotamers using their reported chi1 sigma."""
    key=(_THREE_LETTER[amino_acid],_nearest_dunbrack_bin(phi),_nearest_dunbrack_bin(psi))
    expanded=[]
    for row in library_bins[key]:
        if row.prior_probability<probability_floor: continue
        sigma1=row.chi_sigmas[0] if row.chi_sigmas else 0.0
        for z in sigma_offsets:
            angle=((row.chi1_degrees+float(z)*sigma1+180.0)%360.0)-180.0
            weight=row.prior_probability*math.exp(-0.5*float(z)**2)
            chis=list(row.chi_degrees)
            if chis:
                chis[0]=angle
            else:
                chis=[angle]
            expanded.append(RotamerTemplate(angle,weight,tuple(chis),row.chi_sigmas,"dunbrack2010"))
    if not expanded: raise ValueError(f"No Dunbrack candidates survived probability floor for {key}")
    expanded.sort(key=lambda r:(-r.prior_probability,r.chi1_degrees))
    total=sum(r.prior_probability for r in expanded)
    return tuple(RotamerTemplate(r.chi1_degrees,r.prior_probability/total,r.chi_degrees,r.chi_sigmas,r.source) for r in expanded)

# Three broad backbone-independent chi1 modes, ordered by simple residue-class
# prior. Gly/Ala entries are surrogate microstates because those residues have
# no physical chi1; this is explicitly recorded in result metadata.
_ROTAMER_PRIORS: Mapping[str, Tuple[RotamerTemplate, ...]] = {
    "A": (RotamerTemplate(60.0, 0.50), RotamerTemplate(-60.0, 0.35), RotamerTemplate(180.0, 0.15)),
    "C": (RotamerTemplate(-60.0, 0.48), RotamerTemplate(60.0, 0.32), RotamerTemplate(180.0, 0.20)),
    "D": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(60.0, 0.34), RotamerTemplate(180.0, 0.20)),
    "E": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "F": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.40), RotamerTemplate(60.0, 0.15)),
    "G": (RotamerTemplate(60.0, 0.50), RotamerTemplate(-60.0, 0.35), RotamerTemplate(180.0, 0.15)),
    "H": (RotamerTemplate(-60.0, 0.44), RotamerTemplate(180.0, 0.38), RotamerTemplate(60.0, 0.18)),
    "I": (RotamerTemplate(-60.0, 0.52), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.14)),
    "K": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.20)),
    "L": (RotamerTemplate(-60.0, 0.48), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.17)),
    "M": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "N": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(60.0, 0.34), RotamerTemplate(180.0, 0.20)),
    "P": (RotamerTemplate(30.0, 0.48), RotamerTemplate(-30.0, 0.42), RotamerTemplate(180.0, 0.10)),
    "Q": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "R": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "S": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(60.0, 0.34), RotamerTemplate(180.0, 0.20)),
    "T": (RotamerTemplate(-60.0, 0.50), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.16)),
    "V": (RotamerTemplate(-60.0, 0.52), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.14)),
    "W": (RotamerTemplate(-60.0, 0.44), RotamerTemplate(180.0, 0.41), RotamerTemplate(60.0, 0.15)),
    "Y": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.40), RotamerTemplate(60.0, 0.15)),
}

_SIDECHAIN_REACH: Mapping[str, float] = {
    "G": 1.6, "A": 1.8, "S": 2.4, "C": 2.5, "T": 2.6, "V": 2.8,
    "D": 3.0, "N": 3.1, "I": 3.2, "L": 3.3, "P": 2.6, "M": 3.7,
    "E": 3.8, "Q": 3.9, "H": 3.7, "F": 4.0, "Y": 4.2, "W": 4.5,
    "K": 4.6, "R": 4.8,
}

_NET_CHARGE: Mapping[str, float] = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
}

_FLEXIBILITY_RANK: Mapping[str, int] = {
    aa: rank for rank, aa in enumerate("GAPVITSCNDFYWHLMEQKR")
}

# Approximate side-chain torsional freedom used for adaptive state allocation.
_SIDECHAIN_CHI_COUNT: Mapping[str, int] = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 2, "G": 0, "H": 2,
    "I": 2, "K": 4, "L": 2, "M": 3, "N": 2, "P": 2, "Q": 3,
    "R": 4, "S": 1, "T": 1, "V": 1, "W": 2, "Y": 2,
}


# Residue-specific sub-rotamer expansion.  The base three broad chi1 modes are
# deliberately expanded before any energy-based filtering so that candidate
# diversity is not artificially limited by the QUBO bit budget.
_SUBROTAMER_SCHEMES: Mapping[int, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    6: ((-12.0, 12.0), (0.50, 0.50)),
    9: ((-15.0, 0.0, 15.0), (0.25, 0.50, 0.25)),
    12: ((-22.0, -7.0, 7.0, 22.0), (0.15, 0.35, 0.35, 0.15)),
}


def _raw_rotamer_pool_size(amino_acid: str) -> int:
    """Return 6/9/12 raw candidates according to side-chain torsional freedom."""

    chi = _SIDECHAIN_CHI_COUNT.get(amino_acid, 1)
    if chi <= 1:
        return 6
    if chi == 2:
        return 9
    return 12


def _expanded_rotamer_templates(amino_acid: str) -> Tuple[RotamerTemplate, ...]:
    """Expand three broad chi1 modes into a normalized 6--12-state pool."""

    pool_size = _raw_rotamer_pool_size(amino_acid)
    offsets, weights = _SUBROTAMER_SCHEMES[pool_size]
    expanded = []
    for base in _ROTAMER_PRIORS[amino_acid]:
        for offset, weight in zip(offsets, weights):
            angle = ((base.chi1_degrees + offset + 180.0) % 360.0) - 180.0
            expanded.append(
                RotamerTemplate(
                    chi1_degrees=float(angle),
                    prior_probability=float(base.prior_probability * weight),
                )
            )
    total = sum(item.prior_probability for item in expanded)
    if len(expanded) != pool_size or total <= 0:
        raise RuntimeError(f"Invalid expanded rotamer pool for {amino_acid}")
    return tuple(
        RotamerTemplate(item.chi1_degrees, item.prior_probability / total)
        for item in expanded
    )


def _normalize(vector: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return a unit vector and fail clearly for an unusable direction."""

    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def _decode_amino_acids(x: np.ndarray) -> list[str]:
    """Decode strict 20-way one-hot residue identities from graph features."""

    if x.ndim != 2 or x.shape[1] < 21:
        raise ValueError(f"Expected x with at least 21 columns, got {x.shape}")
    residue_features = x[:, :20]
    if not np.allclose(residue_features.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("The first 20 node features must be one-hot encoded")
    if not np.all((np.isclose(residue_features, 0.0)) | (np.isclose(residue_features, 1.0))):
        raise ValueError("Amino-acid features contain non-binary values")
    return [AA_ORDER[index] for index in residue_features.argmax(axis=1)]


def _atom_parameters(kind: str) -> Tuple[float, float]:
    """Return coarse LJ sigma (A) and epsilon (kcal/mol) by pseudo-element."""

    parameters = {
        "C": (3.50, 0.12),
        "N": (3.25, 0.17),
        "O": (3.00, 0.20),
        "S": (3.60, 0.25),
    }
    return parameters[kind]


def _terminal_spec(amino_acid: str) -> list[Tuple[str, float]]:
    """Return terminal pseudo-elements and partial charges for one residue."""

    if amino_acid in "DE":
        return [("O", -0.65), ("O", -0.65), ("C", 0.30)]
    if amino_acid == "K":
        return [("N", 0.90), ("C", 0.10)]
    if amino_acid == "R":
        return [("N", 0.45), ("N", 0.45), ("C", 0.10)]
    if amino_acid in "NQ":
        return [("O", -0.30), ("N", 0.30)]
    if amino_acid in "STY":
        return [("O", -0.25), ("C", 0.25)]
    if amino_acid == "H":
        return [("N", 0.10), ("C", 0.00)]
    if amino_acid in "CM":
        return [("S", 0.00)]
    return [("C", _NET_CHARGE.get(amino_acid, 0.0))]


def _nonbonded_energy(
    positions_a: np.ndarray,
    sigma_a: np.ndarray,
    epsilon_a: np.ndarray,
    charges_a: np.ndarray,
    positions_b: np.ndarray,
    sigma_b: np.ndarray,
    epsilon_b: np.ndarray,
    charges_b: np.ndarray,
    config: ForceFieldConfig,
) -> float:
    """Compute softened truncated LJ plus distance-dependent Coulomb energy."""

    if not len(positions_a) or not len(positions_b):
        return 0.0
    displacement = positions_a[:, None, :] - positions_b[None, :, :]
    distance = np.linalg.norm(displacement, axis=-1)
    active = distance < config.cutoff_angstrom
    if not np.any(active):
        return 0.0

    sigma = 0.5 * (sigma_a[:, None] + sigma_b[None, :])
    epsilon = np.sqrt(epsilon_a[:, None] * epsilon_b[None, :])
    distance_squared_soft = distance * distance + config.softcore_delta_angstrom**2
    effective_distance = np.sqrt(distance_squared_soft)

    # Exact soft-core form: 4 eps [(sigma^2/(r^2+delta^2))^6 - (...)^3].
    soft_ratio = sigma * sigma / distance_squared_soft
    lj = 4.0 * epsilon * (soft_ratio**6 - soft_ratio**3)
    cutoff_ratio = sigma * sigma / (
        config.cutoff_angstrom**2 + config.softcore_delta_angstrom**2
    )
    lj_shift = 4.0 * epsilon * (cutoff_ratio**6 - cutoff_ratio**3)
    lj = lj - lj_shift
    lj = np.clip(lj, -config.lj_attraction_cap, config.lj_repulsion_cap)

    hard_distance = config.hard_core_fraction * sigma
    overlap = np.maximum(hard_distance - distance, 0.0) / np.maximum(hard_distance, 1e-8)
    hard_penalty = config.hard_sphere_penalty * overlap**2

    charge_product = charges_a[:, None] * charges_b[None, :]
    dielectric = config.dielectric_base + config.dielectric_slope * distance
    coulomb = COULOMB_KCAL_ANGSTROM * charge_product / (
        dielectric * effective_distance
    )
    cutoff_dielectric = (
        config.dielectric_base
        + config.dielectric_slope * config.cutoff_angstrom
    )
    coulomb_shift = COULOMB_KCAL_ANGSTROM * charge_product / (
        cutoff_dielectric * config.cutoff_angstrom
    )
    coulomb = np.clip(
        coulomb - coulomb_shift,
        -config.coulomb_cap,
        config.coulomb_cap,
    )
    total = np.where(active, lj + hard_penalty + coulomb, 0.0)
    return float(total.sum())


class InterfaceQUBOBuilder:
    """Build a NISQ-sized rotamer QUBO from a pruned interface subgraph.

    Args:
        min_variables: Minimum bit count after adaptive 3--6-state allocation.
        max_variables: Hard QUBO dimension limit; must not exceed 30.
        max_sites: Maximum optimized VHH sites. Extra marked sites are ranked by
            ``interface_score`` and deterministically truncated.
        lambda_value: Optional explicit one-hot penalty. Values below the
            computed conservative lower bound are rejected.
        penalty_margin: Fractional safety margin above the physical incident
            energy bound used for automatic lambda selection.
        force_field: Coarse-grained non-bonded parameters.
    """

    def __init__(
        self,
        min_variables: int = 20,
        max_variables: int = 30,
        max_sites: int = 10,
        lambda_value: Optional[float] = None,
        penalty_margin: float = 0.10,
        force_field: Optional[ForceFieldConfig] = None,
        rotamer_mode: str = "legacy",
        rotamer_library_path: Optional[Path] = None,
        rotamer_probability_floor: float = 1e-4,
        rotamer_sigma_offsets: Sequence[float] = (-1.0, 0.0, 1.0),
        energy_calibration: Optional[EnergyCalibration] = None,
        fixed_chi1_wells: bool = False,
        fixed_states_per_site: Optional[int] = None,
    ) -> None:
        if not 2 <= min_variables <= max_variables <= 30:
            raise ValueError("Require 2 <= min_variables <= max_variables <= 30")
        if not 1 <= max_sites <= 10:
            raise ValueError("max_sites must be between 1 and 10 for >=3 states/site under <=30 variables")
        if penalty_margin <= 0:
            raise ValueError("penalty_margin must be positive")
        if lambda_value is not None and lambda_value <= 0:
            raise ValueError("lambda_value must be positive")
        self.min_variables = min_variables
        self.max_variables = max_variables
        self.max_sites = max_sites
        self.fixed_chi1_wells = bool(fixed_chi1_wells)
        self.fixed_states_per_site = (3 if fixed_chi1_wells and fixed_states_per_site is None
                                      else fixed_states_per_site)
        if self.fixed_states_per_site is not None and not self.fixed_chi1_wells:
            raise ValueError("fixed_states_per_site requires chi1-well coverage")
        if self.fixed_states_per_site is not None and not 3 <= self.fixed_states_per_site <= 6:
            raise ValueError("fixed_states_per_site must be in 3..6")
        self.lambda_value = lambda_value
        self.penalty_margin = penalty_margin
        self.force_field = force_field or ForceFieldConfig()
        self.rotamer_mode = str(rotamer_mode)
        if self.rotamer_mode not in ("legacy", "dunbrack2010"):
            raise ValueError("rotamer_mode must be legacy or dunbrack2010")
        self.rotamer_library_path = None if rotamer_library_path is None else Path(rotamer_library_path)
        self.rotamer_probability_floor = float(rotamer_probability_floor)
        self.rotamer_sigma_offsets = tuple(float(v) for v in rotamer_sigma_offsets)
        self.energy_calibration = energy_calibration or EnergyCalibration()
        if not 0.0 < self.rotamer_probability_floor < 1.0:
            raise ValueError("rotamer_probability_floor must lie in (0,1)")
        if not self.rotamer_sigma_offsets or not all(math.isfinite(v) for v in self.rotamer_sigma_offsets):
            raise ValueError("rotamer_sigma_offsets must be finite and nonempty")

    def _validate_graph(self, data: Data) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """Move required graph fields to CPU and validate their semantics."""

        if not hasattr(data, "x") or not hasattr(data, "pos"):
            raise ValueError("data must contain x and pos")
        x = data.x.detach().cpu().numpy().astype(np.float64, copy=False)
        pos = data.pos.detach().cpu().numpy().astype(np.float64, copy=False)
        if pos.shape != (x.shape[0], 3) or not np.isfinite(pos).all():
            raise ValueError(f"pos must be finite [N, 3], got {pos.shape}")
        amino_acids = _decode_amino_acids(x)
        groups = x[:, -1]
        if not np.all(np.isclose(groups, 0.0) | np.isclose(groups, 1.0)):
            raise ValueError("x[:, -1] must contain binary partner labels")
        return x, pos, amino_acids

    def _select_sites(self, data: Data, x: np.ndarray) -> np.ndarray:
        """Choose marked VHH sites, ranked by model interface score."""

        vhh = np.isclose(x[:, -1], 0.0)
        if hasattr(data, "is_active"):
            selected = data.is_active.detach().cpu().numpy().astype(bool)
            mask_name = "is_active"
        elif hasattr(data, "selected_vhh_mask"):
            selected = data.selected_vhh_mask.detach().cpu().numpy().astype(bool)
            mask_name = "selected_vhh_mask"
        else:
            selected = vhh
            mask_name = "implicit VHH mask"
        if mask_name != "implicit VHH mask":
            if selected.shape != (len(x),):
                raise ValueError(f"{mask_name} must have shape [N]")
            if np.any(selected & ~vhh):
                raise ValueError(f"{mask_name} marks a non-VHH node")
        indices = np.flatnonzero(selected)
        if self.rotamer_mode == "dunbrack2010":
            if not hasattr(data, "backbone_phi") or not hasattr(data, "backbone_psi"):
                raise ValueError("Dunbrack mode requires backbone_phi/backbone_psi")
            phi = data.backbone_phi.detach().cpu().numpy().astype(np.float64)
            psi = data.backbone_psi.detach().cpu().numpy().astype(np.float64)
            if phi.shape != (len(x),) or psi.shape != (len(x),):
                raise ValueError("backbone_phi/backbone_psi must have shape [N]")
            allowed = np.asarray([
                (amino not in {"A","G","P","C"}) and math.isfinite(phi[idx]) and math.isfinite(psi[idx])
                for idx, amino in enumerate(_decode_amino_acids(x))
            ], dtype=bool)
            indices = np.flatnonzero(selected & allowed)
        if not len(indices):
            raise ValueError("No eligible selected VHH residues were found for the requested rotamer model")

        if hasattr(data, "interface_score"):
            scores = data.interface_score.detach().cpu().numpy()
            if scores.shape != (len(x),) or not np.isfinite(scores).all():
                raise ValueError("interface_score must be finite with shape [N]")
        else:
            scores = np.zeros(len(x), dtype=np.float64)
        order = sorted(indices.tolist(), key=lambda i: (-float(scores[i]), i))
        return np.asarray(order[: self.max_sites], dtype=np.int64)

    def _allocate_rotamer_counts(
        self,
        site_indices: np.ndarray,
        amino_acids: Sequence[str],
        site_scores: Optional[Sequence[float]] = None,
    ) -> list[int]:
        """Allocate 3--6 retained states/site under the global variable budget.

        Flexibility supplies the baseline target (3/4/5/6 states for 0--1/2/3/4+
        chi torsions).  Interface importance breaks ties when the global <=30-bit
        budget requires contraction or permits expansion.
        """

        site_count = len(site_indices)
        minimum_possible = 3 * site_count
        maximum_possible = 6 * site_count
        if minimum_possible > self.max_variables:
            raise ValueError(
                f"{site_count} sites require at least {minimum_possible} variables "
                f"at 3 states/site; limit is {self.max_variables}"
            )
        if maximum_possible < self.min_variables:
            raise ValueError(
                f"{site_count} sites can provide at most {maximum_possible} variables; "
                f"need at least {self.min_variables}"
            )

        if site_scores is None:
            importance = np.zeros(site_count, dtype=np.float64)
        else:
            importance = np.asarray(site_scores, dtype=np.float64)
            if importance.shape != (site_count,) or not np.isfinite(importance).all():
                raise ValueError("site_scores must be finite with one value per selected site")

        def preferred_count(node: int) -> int:
            chi = _SIDECHAIN_CHI_COUNT.get(amino_acids[int(node)], 1)
            if chi <= 1:
                return 3
            if chi == 2:
                return 4
            if chi == 3:
                return 5
            return 6

        counts = [preferred_count(int(node)) for node in site_indices]

        # Contract least important / least flexible sites first, never below 3.
        while sum(counts) > self.max_variables:
            candidates = [idx for idx, count in enumerate(counts) if count > 3]
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda idx: (
                    float(importance[idx]),
                    _SIDECHAIN_CHI_COUNT.get(amino_acids[int(site_indices[idx])], 1),
                    _FLEXIBILITY_RANK.get(amino_acids[int(site_indices[idx])], 0),
                    int(site_indices[idx]),
                ),
            )
            counts[chosen] -= 1

        # If a caller requests a larger minimum dimension, spend remaining bits
        # on the most important/flexible sites, never above six.
        while sum(counts) < self.min_variables:
            candidates = [idx for idx, count in enumerate(counts) if count < 6]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda idx: (
                    float(importance[idx]),
                    _SIDECHAIN_CHI_COUNT.get(amino_acids[int(site_indices[idx])], 1),
                    _FLEXIBILITY_RANK.get(amino_acids[int(site_indices[idx])], 0),
                    -int(site_indices[idx]),
                ),
            )
            counts[chosen] += 1

        total = sum(counts)
        if not self.min_variables <= total <= self.max_variables:
            raise RuntimeError(f"Internal rotamer allocation error: {total} variables")
        if any(count < 3 or count > 6 for count in counts):
            raise AssertionError(f"Invalid per-site state allocation: {counts}")
        return counts


    def _local_frame(
        self,
        node_index: int,
        pos: np.ndarray,
        x: np.ndarray,
        chain_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a right-handed local frame from CA and interface directions."""

        center = pos[node_index]
        same_chain = np.flatnonzero(
            (chain_ids == chain_ids[node_index])
            & np.isclose(x[:, -1], 0.0)
            & (np.arange(len(pos)) != node_index)
        )
        ligand = np.flatnonzero(np.isclose(x[:, -1], 1.0))

        if len(same_chain):
            distances = np.linalg.norm(pos[same_chain] - center, axis=1)
            nearest = same_chain[np.argsort(distances)[:2]]
            if len(nearest) == 2:
                tangent_raw = pos[nearest[1]] - pos[nearest[0]]
            else:
                tangent_raw = pos[nearest[0]] - center
        elif len(ligand):
            nearest_ligand = ligand[np.argmin(np.linalg.norm(pos[ligand] - center, axis=1))]
            tangent_raw = pos[nearest_ligand] - center
        else:
            tangent_raw = np.array([1.0, 0.0, 0.0])
        tangent = _normalize(tangent_raw)

        candidates: list[np.ndarray] = []
        if len(ligand):
            nearest_ligand = ligand[np.argmin(np.linalg.norm(pos[ligand] - center, axis=1))]
            candidates.append(pos[nearest_ligand] - center)
        if len(same_chain):
            candidates.extend(pos[index] - center for index in same_chain[:3])
        candidates.extend(
            np.eye(3)[index] for index in np.argsort(np.abs(np.eye(3) @ tangent))
        )

        normal: Optional[np.ndarray] = None
        for candidate in candidates:
            perpendicular = candidate - np.dot(candidate, tangent) * tangent
            if np.linalg.norm(perpendicular) > 1e-8:
                normal = _normalize(perpendicular)
                break
        if normal is None:
            raise ValueError("Unable to construct a local residue frame")
        binormal = _normalize(np.cross(tangent, normal))
        return tangent, normal, binormal

    def _generate_rotamer(
        self,
        site_index: int,
        node_index: int,
        amino_acid: str,
        template: RotamerTemplate,
        pos: np.ndarray,
        x: np.ndarray,
        chain_ids: np.ndarray,
        rotamer_index: int,
    ) -> RotamerState:
        """Attach residue-specific pseudo-atoms in a local chi1 orientation."""

        tangent, normal, binormal = self._local_frame(
            node_index, pos, x, chain_ids
        )
        angle = math.radians(template.chi1_degrees)
        radial = math.cos(angle) * normal + math.sin(angle) * binormal
        direction = _normalize(0.25 * tangent + 0.9682458 * radial)
        lateral = _normalize(-math.sin(angle) * normal + math.cos(angle) * binormal)
        center = pos[node_index]
        reach = _SIDECHAIN_REACH[amino_acid]

        positions = [center + 1.53 * direction]
        kinds = ["C"]
        charges = [0.0]
        terminal = _terminal_spec(amino_acid)
        for terminal_index, (kind, charge) in enumerate(terminal):
            spread = (terminal_index - 0.5 * (len(terminal) - 1)) * 0.38
            positions.append(center + reach * direction + spread * lateral)
            kinds.append(kind)
            charges.append(charge)
        parameters = [_atom_parameters(kind) for kind in kinds]
        sigma = np.asarray([item[0] for item in parameters], dtype=np.float64)
        epsilon = np.asarray([item[1] for item in parameters], dtype=np.float64)
        return RotamerState(
            site_index=site_index,
            node_index=node_index,
            amino_acid=amino_acid,
            rotamer_index=rotamer_index,
            chi1_degrees=template.chi1_degrees,
            prior_probability=template.prior_probability,
            positions=np.asarray(positions, dtype=np.float64),
            sigma=sigma,
            epsilon=epsilon,
            charges=np.asarray(charges, dtype=np.float64),
            chi_degrees=tuple(float(v) for v in (template.chi_degrees or (template.chi1_degrees,))),
            chi_sigmas=tuple(float(v) for v in template.chi_sigmas),
        )

    def _rigid_environment(
        self,
        pos: np.ndarray,
        amino_acids: Sequence[str],
        frozen_mask: np.ndarray,
        active_mask: np.ndarray,
        vhh_mask: np.ndarray,
        excluded_node: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Represent VHH-only fixed background; antigen is scored separately."""

        if (frozen_mask.shape != (len(pos),) or active_mask.shape != (len(pos),)
                or vhh_mask.shape != (len(pos),)):
            raise ValueError("active/frozen/vhh masks must have shape [N]")
        keep = (frozen_mask | active_mask) & vhh_mask
        keep[excluded_node] = False
        environment_indices = np.flatnonzero(keep)
        env_pos = pos[environment_indices]
        env_sigma = np.full(len(env_pos), 3.50, dtype=np.float64)
        env_epsilon = np.full(len(env_pos), 0.06, dtype=np.float64)
        # Frozen residues retain coarse net charges. Other active backbones are
        # neutral here because their side-chain charge is handled by pair terms.
        charges = [
            _NET_CHARGE.get(amino_acids[index], 0.0) if frozen_mask[index] else 0.0
            for index in environment_indices
        ]
        return env_pos, env_sigma, env_epsilon, np.asarray(charges)

    def _antigen_environment(
        self,
        pos: np.ndarray,
        x: np.ndarray,
        amino_acids: Sequence[str],
        node_index: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return a coarse antigen-only environment for candidate guidance."""

        antigen = np.flatnonzero(np.isclose(x[:, -1], 1.0))
        if not len(antigen):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        center = pos[int(node_index)]
        distances = np.linalg.norm(pos[antigen] - center, axis=1)
        antigen = antigen[distances <= self.force_field.cutoff_angstrom]
        env_pos = pos[antigen]
        env_sigma = np.full(len(antigen), 3.50, dtype=np.float64)
        env_epsilon = np.full(len(antigen), 0.06, dtype=np.float64)
        env_charge = np.asarray(
            [_NET_CHARGE.get(amino_acids[int(index)], 0.0) for index in antigen],
            dtype=np.float64,
        )
        return env_pos, env_sigma, env_epsilon, env_charge

    def _lambda_lower_bound(
        self,
        self_energy: np.ndarray,
        pair_energy: np.ndarray,
        site_to_variables: Mapping[int, Tuple[int, ...]],
    ) -> float:
        """Bound any single-bit physical gain before adding one-hot penalties.

        For each variable, this uses its absolute self term plus the maximum
        absolute interaction with every *other* site. The maximum incident
        bound dominates the most attractive individual pair and is deliberately
        conservative for both missing-choice and multiple-choice violations.
        """

        variable_site = {
            variable: site
            for site, variables in site_to_variables.items()
            for variable in variables
        }
        incident_bounds = []
        for variable in range(len(self_energy)):
            bound = abs(float(self_energy[variable]))
            for other_site, other_variables in site_to_variables.items():
                if other_site == variable_site[variable]:
                    continue
                values = [
                    pair_energy[min(variable, other), max(variable, other)]
                    for other in other_variables
                ]
                bound += max(abs(float(value)) for value in values)
            incident_bounds.append(bound)
        raw_bound = max(incident_bounds, default=1.0)
        return max(1.0, raw_bound) * (1.0 + self.penalty_margin) + 1e-6

    def build(self, data: Data) -> QUBOResult:
        """Construct physical terms, inject one-hot constraints, and return Q."""

        x, pos, amino_acids = self._validate_graph(data)
        site_nodes = self._select_sites(data, x)
        if hasattr(data, "interface_score"):
            all_site_scores = data.interface_score.detach().cpu().numpy().astype(np.float64)
            site_scores = all_site_scores[site_nodes]
        else:
            site_scores = np.zeros(len(site_nodes), dtype=np.float64)
        counts = ([self.fixed_states_per_site] * len(site_nodes) if self.fixed_chi1_wells
                  else self._allocate_rotamer_counts(site_nodes, amino_acids, site_scores))
        if self.fixed_chi1_wells and not self.min_variables <= self.fixed_states_per_site * len(site_nodes) <= self.max_variables:
            raise ValueError("Fixed state policy exceeds the configured QUBO dimension")
        active_mask = np.zeros(len(x), dtype=bool)
        active_mask[site_nodes] = True
        declared_active = (
            data.is_active.detach().cpu().numpy().astype(bool)
            if hasattr(data, "is_active")
            else active_mask.copy()
        )
        if hasattr(data, "is_frozen_environment"):
            frozen_mask = data.is_frozen_environment.detach().cpu().numpy().astype(bool)
            if frozen_mask.shape != (len(x),):
                raise ValueError("is_frozen_environment must have shape [N]")
        else:
            frozen_mask = ~active_mask
        # If max_sites truncates a larger adaptive active set, keep the omitted
        # residues as a fixed background instead of silently dropping them.
        frozen_mask = frozen_mask | (declared_active & ~active_mask)
        if np.any(active_mask & frozen_mask):
            raise ValueError("Active and Frozen residue masks must be disjoint")
        if hasattr(data, "node_chain_id"):
            chain_ids = data.node_chain_id.detach().cpu().numpy()
            if chain_ids.shape != (len(x),):
                raise ValueError("node_chain_id must have shape [N]")
        else:
            chain_ids = np.where(np.isclose(x[:, -1], 0.0), 0, 1)

        vhh_mask = np.isclose(x[:, -1], 0.0)
        environments = {
            int(node_index): self._rigid_environment(
                pos, amino_acids, frozen_mask, active_mask, vhh_mask, int(node_index)
            )
            for node_index in site_nodes
        }
        antigen_environments = {
            int(node_index): self._antigen_environment(
                pos, x, amino_acids, int(node_index)
            )
            for node_index in site_nodes
        }
        if self.rotamer_mode == "dunbrack2010":
            if self.rotamer_library_path is None:
                raise ValueError("Formal Dunbrack mode requires rotamer_library_path")
            if not hasattr(data, "backbone_phi") or not hasattr(data, "backbone_psi"):
                raise ValueError("Dunbrack mode requires backbone_phi/backbone_psi graph metadata")
            phi_all=data.backbone_phi.detach().cpu().numpy().astype(np.float64)
            psi_all=data.backbone_psi.detach().cpu().numpy().astype(np.float64)
            requested=set()
            for node_index in site_nodes:
                aa=amino_acids[int(node_index)]
                if aa in "AG": continue
                requested.add((_THREE_LETTER[aa], _nearest_dunbrack_bin(phi_all[int(node_index)]), _nearest_dunbrack_bin(psi_all[int(node_index)])))
            dunbrack_bins=_load_dunbrack_bins(self.rotamer_library_path, requested) if requested else {}
        else:
            phi_all=psi_all=None
            dunbrack_bins={}

        rotamers: list[RotamerState] = []
        site_to_variables: Dict[int, Tuple[int, ...]] = {}
        raw_pool_sizes_actual: list[int] = []
        for site_index, (node_index, count) in enumerate(zip(site_nodes, counts)):
            aa = amino_acids[int(node_index)]
            if self.rotamer_mode == "dunbrack2010":
                templates = _dunbrack_templates_for_site(
                    dunbrack_bins, aa, phi_all[int(node_index)], psi_all[int(node_index)],
                    probability_floor=self.rotamer_probability_floor,
                    sigma_offsets=self.rotamer_sigma_offsets,
                )
            else:
                templates = _expanded_rotamer_templates(aa)
            raw_pool_sizes_actual.append(len(templates))
            best_probability = max(template.prior_probability for template in templates)
            candidate_states: list[RotamerState] = []
            for template_index, template in enumerate(templates):
                state = self._generate_rotamer(
                    site_index,
                    int(node_index),
                    aa,
                    template,
                    pos,
                    x,
                    chain_ids,
                    template_index,
                )
                state.prior_energy = -self.force_field.thermal_energy_kcal * math.log(
                    template.prior_probability / best_probability
                )
                state.antigen_guidance_energy = _nonbonded_energy(
                    state.positions,
                    state.sigma,
                    state.epsilon,
                    state.charges,
                    *antigen_environments[int(node_index)],
                    self.force_field,
                )
                state.environment_energy = _nonbonded_energy(
                    state.positions,
                    state.sigma,
                    state.epsilon,
                    state.charges,
                    *environments[int(node_index)],
                    self.force_field,
                )
                candidate_states.append(state)

            cal=self.energy_calibration
            candidate_states.sort(
                key=lambda state: (
                    cal.prior_weight*state.prior_energy
                    + cal.vhh_environment_weight*state.environment_energy
                    + cal.antigen_weight*state.antigen_guidance_energy,
                    state.rotamer_index,
                )
            )
            selected_states = (select_chi1_well_representatives(candidate_states, count)
                               if self.fixed_chi1_wells else candidate_states[:count])
            variable_indices = []
            for selected_index, state in enumerate(selected_states):
                state.rotamer_index = selected_index
                variable_indices.append(len(rotamers))
                rotamers.append(state)
            site_to_variables[site_index] = tuple(variable_indices)

        variable_count = len(rotamers)
        if not self.min_variables <= variable_count <= self.max_variables:
            raise RuntimeError(f"Produced invalid QUBO dimension {variable_count}")
        cal=self.energy_calibration
        raw_prior = np.asarray([state.prior_energy for state in rotamers],dtype=np.float64)
        raw_vhh_environment = np.asarray([state.environment_energy for state in rotamers],dtype=np.float64)
        raw_antigen = np.asarray([state.antigen_guidance_energy for state in rotamers],dtype=np.float64)
        physical_self = (
            cal.prior_weight*raw_prior
            + cal.vhh_environment_weight*raw_vhh_environment
            + cal.antigen_weight*raw_antigen
        )
        raw_pair = np.zeros((variable_count, variable_count), dtype=np.float64)
        for left in range(variable_count):
            for right in range(left + 1, variable_count):
                if rotamers[left].site_index == rotamers[right].site_index:
                    continue
                a, b = rotamers[left], rotamers[right]
                raw_pair[left, right] = _nonbonded_energy(
                    a.positions,
                    a.sigma,
                    a.epsilon,
                    a.charges,
                    b.positions,
                    b.sigma,
                    b.epsilon,
                    b.charges,
                    self.force_field,
                )
        physical_pair = cal.pair_weight * raw_pair

        lambda_lower_bound = self._lambda_lower_bound(
            physical_self, physical_pair, site_to_variables
        )
        if self.lambda_value is not None and self.lambda_value < lambda_lower_bound:
            raise ValueError(
                f"lambda_value={self.lambda_value:.6g} is below the conservative "
                f"lower bound {lambda_lower_bound:.6g}"
            )
        lambda_value = self.lambda_value or lambda_lower_bound

        Q = physical_pair.copy()
        diagonal = physical_self - lambda_value
        np.fill_diagonal(Q, diagonal)
        for variables in site_to_variables.values():
            for left, right in _combinations(variables):
                Q[left, right] += 2.0 * lambda_value
        Q[np.tril_indices(variable_count, k=-1)] = 0.0
        if not np.isfinite(Q).all():
            raise FloatingPointError("QUBO contains non-finite values")

        original_indices = (
            data.original_node_index.detach().cpu().numpy()
            if hasattr(data, "original_node_index")
            else np.arange(len(x))
        )
        residue_ids = (
            list(data.residue_ids)
            if hasattr(data, "residue_ids") and len(data.residue_ids) == len(x)
            else [str(index) for index in range(len(x))]
        )
        variable_map = tuple(
            VariableRecord(
                variable_index=index,
                site_index=state.site_index,
                node_index=state.node_index,
                original_node_index=int(original_indices[state.node_index]),
                residue_id=str(residue_ids[state.node_index]),
                amino_acid=state.amino_acid,
                rotamer_index=state.rotamer_index,
                chi1_degrees=state.chi1_degrees,
                prior_probability=state.prior_probability,
                self_energy=float(physical_self[index]),
            )
            for index, state in enumerate(rotamers)
        )
        max_attraction = max(
            0.0,
            -float(np.min(physical_pair)) if physical_pair.size else 0.0,
        )
        constant_offset = float(lambda_value * len(site_to_variables) + cal.intercept)
        ising_h, ising_J, ising_offset = qubo_to_ising(Q, constant_offset)
        max_equivalence_error = validate_qubo_ising_equivalence(
            Q, constant_offset, ising_h, ising_J, ising_offset
        )
        metadata: Dict[str, Any] = {
            "model": "CA-frame coarse-grained pseudo-atom force field with full Dunbrack rotamer-state provenance; pseudo-atom geometry remains chi1-oriented",
            "energy_unit": "approximate kcal/mol",
            "site_node_indices": site_nodes.tolist(),
            "rotamers_per_site": counts,
            "raw_rotamer_pool_sizes": raw_pool_sizes_actual,
            "rotamer_state_policy": (
                "Dunbrack 2010 backbone-dependent full rotamer states (chi1..chiN); chi1 sigma expansion controls pseudo-atom orientation while distal chi means are retained for exact all-atom reconstruction/calibration; retain 3--6 states/site under <=30 variables"
                if self.rotamer_mode == "dunbrack2010"
                else "legacy 6/9/12 raw chi1 sub-rotamers by flexibility; retain 3--6 states/site under <=30 variables"
            ),
            "candidate_guidance": "pre-screen by rotamer prior + VHH-only fixed-environment energy + antigen interaction energy; antigen counted once",
            "rotamer_model": self.rotamer_mode,
            "state_policy": (f"fixed_{self.fixed_states_per_site}_chi1_coverage" if self.fixed_chi1_wells and self.fixed_states_per_site != 3
                             else "fixed_three_chi1_wells" if self.fixed_chi1_wells else "adaptive_3_to_6"),
            "rotamer_library_path": (None if self.rotamer_library_path is None else str(self.rotamer_library_path)),
            "rotamer_probability_floor": self.rotamer_probability_floor,
            "rotamer_sigma_offsets": list(self.rotamer_sigma_offsets),
            "energy_calibration": asdict(cal),
            "raw_prior_energy": raw_prior.tolist(),
            "raw_vhh_environment_energy": raw_vhh_environment.tolist(),
            "raw_antigen_energy": raw_antigen.tolist(),
            "raw_pair_energy_upper": raw_pair.tolist(),
            "antigen_guidance_energy": raw_antigen.tolist(),
            "rotamer_state_records": [
                dict(
                    variable_index=int(index),
                    residue_id=str(residue_ids[state.node_index]),
                    site_index=int(state.site_index),
                    amino_acid=str(state.amino_acid),
                    rotamer_index=int(state.rotamer_index),
                    chi1_degrees=float(state.chi1_degrees),
                    chi_degrees=[float(v) for v in state.chi_degrees],
                    chi_sigmas=[float(v) for v in state.chi_sigmas],
                    prior_probability=float(state.prior_probability),
                )
                for index,state in enumerate(rotamers)
            ],
            "variable_count": variable_count,
            "active_residue_count": int(active_mask.sum()),
            "frozen_environment_count": int(frozen_mask.sum()),
            "ising_energy_equivalence_max_error": max_equivalence_error,
            "max_pair_attraction": max_attraction,
            "lambda_exceeds_max_pair_attraction": lambda_value > max_attraction,
            "surrogate_chi1_sites": [
                site for site, node in enumerate(site_nodes)
                if amino_acids[int(node)] in {"A", "G"}
            ],
            "pdb_id": getattr(data, "pdb_id", "unknown"),
            "source_id": getattr(data, "source_id", "unknown"),
            "force_field": asdict(self.force_field),
        }
        return QUBOResult(
            Q=Q,
            variable_map=variable_map,
            site_to_variables=site_to_variables,
            lambda_value=float(lambda_value),
            lambda_lower_bound=float(lambda_lower_bound),
            constant_offset=constant_offset,
            physical_self=physical_self,
            physical_pair=physical_pair,
            metadata=metadata,
        )

    def build_matrix(self, data: Data) -> np.ndarray:
        """Convenience wrapper returning only the upper-triangular Q matrix."""

        return self.build(data).Q


def _combinations(values: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """Yield sorted unique pairs from a small variable-index collection."""

    items = tuple(values)
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            yield items[left_index], items[right_index]


def qubo_to_ising(
    Q: np.ndarray,
    qubo_offset: float = 0.0,
    *,
    tolerance: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert an upper-triangular QUBO to an upper-triangular Ising model.

    Uses ``x_i = (1 - Z_i) / 2`` and returns ``(h, J, offset)`` such that

    ``H(Z) = offset + sum_i h[i] Z_i + sum_{i<j} J[i,j] Z_i Z_j``.

    Args:
        Q: Upper-triangular QUBO matrix whose diagonal stores linear terms.
        qubo_offset: Optional constant already present in the QUBO, such as the
            ``QUBOResult.constant_offset`` from one-hot expansion.
        tolerance: Maximum accepted magnitude below the diagonal.
    """

    matrix = np.asarray(Q, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Q must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all() or not math.isfinite(qubo_offset):
        raise ValueError("Q and qubo_offset must be finite")
    if np.any(np.abs(np.tril(matrix, k=-1)) > tolerance):
        raise ValueError("Q must be upper triangular under the stated convention")

    size = matrix.shape[0]
    h = -0.5 * np.diag(matrix).copy()
    J = np.zeros_like(matrix)
    offset = float(qubo_offset + 0.5 * np.trace(matrix))
    for left in range(size):
        for right in range(left + 1, size):
            coupling = matrix[left, right]
            if coupling == 0.0:
                continue
            J[left, right] = 0.25 * coupling
            h[left] -= 0.25 * coupling
            h[right] -= 0.25 * coupling
            offset += 0.25 * coupling
    return h, J, offset


def validate_qubo_ising_equivalence(
    Q: np.ndarray,
    qubo_offset: float,
    h: np.ndarray,
    J: np.ndarray,
    ising_offset: float,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Validate triangular storage and exact QUBO/Ising energy equality."""

    matrix = np.asarray(Q, dtype=np.float64)
    linear = np.asarray(h, dtype=np.float64)
    coupling = np.asarray(J, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Q must be square")
    size = matrix.shape[0]
    if coupling.shape != matrix.shape or linear.shape != (size,):
        raise ValueError("h/J dimensions are inconsistent with Q")
    if np.any(np.abs(np.tril(matrix, -1)) > tolerance):
        raise ValueError("Q contains non-zero entries below the diagonal")
    if np.any(np.abs(np.tril(coupling, 0)) > tolerance):
        raise ValueError("J must be strictly upper triangular")
    if not all(np.isfinite(value).all() for value in (matrix, linear, coupling)):
        raise ValueError("QUBO/Ising coefficients must be finite")
    if not math.isfinite(qubo_offset) or not math.isfinite(ising_offset):
        raise ValueError("QUBO/Ising offsets must be finite")

    reconstructed_q = np.zeros_like(matrix)
    reconstructed_q[np.triu_indices(size, 1)] = 4.0 * coupling[
        np.triu_indices(size, 1)
    ]
    for index in range(size):
        incident = coupling[:index, index].sum() + coupling[index, index + 1 :].sum()
        reconstructed_q[index, index] = -2.0 * linear[index] - 2.0 * incident
    if not np.allclose(reconstructed_q, matrix, atol=tolerance, rtol=0.0):
        raise AssertionError("Analytical Ising coefficients do not reconstruct Q")
    expected_offset = float(
        qubo_offset + 0.5 * np.trace(matrix) + 0.25 * np.triu(matrix, 1).sum()
    )
    if not math.isclose(ising_offset, expected_offset, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError("Ising constant offset is inconsistent with Q")

    states = [np.zeros(size), np.ones(size), *np.eye(size)]
    rng = np.random.default_rng(20260917)
    states.extend(rng.integers(0, 2, size=size).astype(float) for _ in range(16))
    maximum_error = 0.0
    for binary in states:
        spins = 1.0 - 2.0 * binary
        qubo_energy = float(qubo_offset + binary @ matrix @ binary)
        ising_energy = float(ising_offset + linear @ spins + spins @ coupling @ spins)
        maximum_error = max(maximum_error, abs(qubo_energy - ising_energy))
    if maximum_error > tolerance:
        raise AssertionError(
            f"QUBO/Ising mismatch {maximum_error:.3e} exceeds {tolerance:.3e}"
        )
    return maximum_error


def _virtual_pruned_graph(site_count: int = 6) -> Data:
    """Create a deterministic interface-like PyG graph for executable tests."""

    if site_count < 5:
        raise ValueError("At least five active sites are required by adaptive pruning")
    amino_acids = "DEKRQNSTYFIL"[:site_count]
    ligand_count = 12
    node_count = site_count + ligand_count
    x = torch.zeros((node_count, 21), dtype=torch.float32)
    for index, aa in enumerate(amino_acids):
        x[index, AA_INDEX[aa]] = 1.0
    for index in range(site_count, node_count):
        x[index, AA_INDEX["A"]] = 1.0
        x[index, -1] = 1.0

    vhh_pos = [
        [3.8 * index, 0.45 * math.sin(index), 0.20 * math.cos(index)]
        for index in range(site_count)
    ]
    ligand_pos = [
        [3.0 * index, 4.0 + 0.25 * math.cos(index), 0.35 * math.sin(index)]
        for index in range(ligand_count)
    ]
    pos = torch.tensor(vhh_pos + ligand_pos, dtype=torch.float32)
    distances = torch.cdist(pos, pos)
    edge_index = ((distances < 8.0) & (distances > 0)).nonzero().t().long()
    data = Data(x=x, pos=pos, edge_index=edge_index)
    data.is_active = torch.tensor([True] * site_count + [False] * ligand_count)
    data.is_frozen_environment = torch.tensor(
        [False] * site_count + [True] * ligand_count
    )
    data.selected_vhh_mask = data.is_active.clone()
    data.interface_score = torch.linspace(1.0, 0.1, node_count)
    data.original_node_index = torch.arange(node_count)
    data.node_chain_id = torch.tensor([0] * site_count + [1] * ligand_count)
    data.residue_ids = [
        *(f"H:{index + 1}" for index in range(site_count)),
        *(f"A:{index + 1}" for index in range(ligand_count)),
    ]
    data.pdb_id = "VIRTUAL"
    data.source_id = "subgraph_to_qubo.py::__main__"
    return data



def _openmm_context(mm: Any, system: Any, integrator: Any) -> Any:
    """Use an explicit recorded platform; never silently fall back on failure."""
    name = os.environ.get("QP_OPENMM_PLATFORM", "Reference")
    if name not in ("Reference", "CPU", "CUDA"):
        raise ValueError("QP_OPENMM_PLATFORM must be Reference, CPU or CUDA")
    properties = {}
    if name == "CPU":
        properties["Threads"] = os.environ.get("OPENMM_CPU_THREADS", "8")
    elif name == "CUDA":
        precision = os.environ.get("QP_OPENMM_PRECISION", "double")
        if precision not in ("single", "mixed", "double"):
            raise ValueError("QP_OPENMM_PRECISION must be single, mixed or double")
        properties = {"Precision": precision, "DeviceIndex": os.environ.get("QP_OPENMM_DEVICE", "0")}
    return mm.Context(system, integrator, mm.Platform.getPlatformByName(name), properties)


class AllAtomInterfaceQUBOBuilder:
    """Amber14 fixed-backbone adaptive multi-state side-chain QUBO.

    Formal all-atom validation uses complete Dunbrack 2010 side-chain rotamer
    states (chi1..chiN) at the residue's backbone phi/psi bin. Chi1 is expanded
    by the configured Dunbrack sigma offsets while distal chi values follow the
    rotamer's statistical means. Amber14 single-candidate energies pre-screen
    the pool; 3--6 states/site are retained under a global <=30-variable budget.
    An explicit chi1_angles sequence remains available only as a legacy
    controlled-ablation override.

    No native/reference structure is accepted by this builder. Missing heavy atoms and unsupported templates
    fail rather than inventing atoms. The primary protocol uses vacuum NoCutoff; optional GBN2 is a
    pre-declared sensitivity model. These energies are packing/reconstruction proxies, NOT binding free energy.
    """

    def __init__(self, structure_path: Path, active_residues: Sequence[str], *,
                 chi1_angles: Optional[Sequence[float]] = None,
                 site_scores: Optional[Sequence[float]] = None, seed: int = 42,
                 candidate_relax_iterations: int = 0,
                 rotamer_mode: str = "legacy",
                 rotamer_library_path: Optional[Path] = None,
                 rotamer_probability_floor: float = 1e-4,
                 rotamer_sigma_offsets: Sequence[float] = (-1.0,0.0,1.0),
                 solvent_model: str = "vacuum"):
        import openmm as mm
        from openmm import app, unit
        import random
        import gemmi
        self.mm, self.app, self.unit = mm, app, unit
        if candidate_relax_iterations < 0:
            raise ValueError("Candidate relaxation iterations must be nonnegative")
        self.candidate_relax_iterations = candidate_relax_iterations
        self.rotamer_mode=str(rotamer_mode)
        if self.rotamer_mode not in ("legacy","dunbrack2010"):
            raise ValueError("rotamer_mode must be legacy or dunbrack2010")
        self.rotamer_library_path=None if rotamer_library_path is None else Path(rotamer_library_path)
        self.rotamer_probability_floor=float(rotamer_probability_floor)
        self.rotamer_sigma_offsets=tuple(float(v) for v in rotamer_sigma_offsets)
        self.solvent_model=str(solvent_model).lower()
        if self.solvent_model not in ("vacuum","gbn2"):
            raise ValueError("solvent_model must be vacuum or gbn2")
        # Only the vacuum model is exactly pair-decomposable; see build().
        self.pair_decomposition_exact=self.solvent_model=="vacuum"
        if chi1_angles is not None:
            chi1_angles=tuple(float(a) for a in chi1_angles)
            if not 2 <= len(chi1_angles) <= 6 or not np.isfinite(chi1_angles).all():
                raise ValueError("Legacy chi1_angles override requires 2--6 finite angles")
            if len({round(float(a)%360,8) for a in chi1_angles})!=len(chi1_angles):
                raise ValueError("Duplicate chi1 angles modulo 360")
        self.chi1_angles_override=chi1_angles
        ids=list(active_residues)
        if not ids or len(set(ids))!=len(ids):
            raise ValueError("Active residues must be unique and nonempty")
        if site_scores is None:
            self.site_scores=np.zeros(len(ids),dtype=np.float64)
        else:
            self.site_scores=np.asarray(site_scores,dtype=np.float64)
            if self.site_scores.shape!=(len(ids),) or not np.isfinite(self.site_scores).all():
                raise ValueError("site_scores must be finite and aligned with active_residues")
        if chi1_angles is None and len(ids)*3>30:
            raise ValueError("Adaptive all-atom mode requires at most 10 Active residues under the 30-variable budget")
        if chi1_angles is not None and len(ids)*len(chi1_angles)>30:
            raise ValueError("Legacy chi1 angle override exceeds the 30-variable budget")
        structure_path=Path(structure_path)
        structure=gemmi.read_structure(str(structure_path))
        if len(structure)!=1:
            raise ValueError("Prepare a single-model structure before all-atom construction")
        if any(a.altloc not in ("\x00"," ","") for c in structure[0] for r in c for a in r):
            raise ValueError("Resolve alternate conformers before force-field preparation")
        self.source_structure=structure_path
        # Explicitly validate selected canonical heavy atoms before hydrogen addition.
        protein=read_atomistic_structure(structure_path)
        allatom_dunbrack_bins={}
        allatom_backbone_angles={}
        if self.rotamer_mode=="dunbrack2010" and chi1_angles is None:
            if self.rotamer_library_path is None:
                raise ValueError("Dunbrack all-atom mode requires rotamer_library_path")
            requested=set()
            for rid in ids:
                residue_name=protein[rid]["name"]
                aa=gemmi.find_tabulated_residue(residue_name).one_letter_code
                if aa in "AG": continue
                phi,psi=_backbone_phi_psi(protein,rid)
                allatom_backbone_angles[rid]=(phi,psi)
                requested.add((_THREE_LETTER[aa],_nearest_dunbrack_bin(phi),_nearest_dunbrack_bin(psi)))
            allatom_dunbrack_bins=_load_dunbrack_bins(self.rotamer_library_path,requested) if requested else {}
        for rid in ids:
            if rid not in protein or protein[rid]["name"] in ("ALA","GLY","PRO","CYS"):
                raise ValueError(f"Active site has no supported safe acyclic side-chain search: {rid}")
            needed=set(("N","CA","C","O"))|set(_SIDECHAIN_NAMES[protein[rid]["name"]].split())
            if needed-set(protein[rid]["atoms"]):
                raise ValueError(f"Incomplete Active heavy atoms: {rid}")
        suffix=structure_path.name.lower()
        import gzip
        opener=gzip.open if suffix.endswith(".gz") else open
        with opener(structure_path,"rt") as handle:
            parsed=(app.PDBxFile(handle) if suffix.endswith((".cif",".cif.gz")) else app.PDBFile(handle))
        if self.solvent_model=="vacuum":
            self.forcefield=app.ForceField("amber14-all.xml")
        else:
            # OpenMM's Amber implicit-solvent GBN2 parameters; no explicit
            # solvent particles are added. This is a sensitivity model, not
            # the frozen primary structural protocol.
            self.forcefield=app.ForceField("amber14-all.xml","implicit/gbn2.xml")
        modeller=app.Modeller(parsed.topology,parsed.positions)
        state=random.getstate()
        try:
            random.seed(seed)
            modeller.addHydrogens(self.forcefield, platform=mm.Platform.getPlatformByName("Reference"))
        finally:
            random.setstate(state)
        self.topology=modeller.topology
        self.base_positions=np.asarray(modeller.positions.value_in_unit(unit.nanometer),dtype=float)
        self.system=self.forcefield.createSystem(
            self.topology,nonbondedMethod=app.NoCutoff,
            constraints=None,rigidWater=False,removeCMMotion=False
        )
        self.integrator=mm.VerletIntegrator(.001)
        self.context=_openmm_context(mm, self.system, self.integrator)
        residues={}
        for residue in self.topology.residues():
            key=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            if key in residues:
                raise ValueError(f"Duplicate topology residue ID: {key}")
            residues[key]=residue
        bonds={i:set() for i in range(self.topology.getNumAtoms())}
        for a,b in self.topology.bonds():
            bonds[a.index].add(b.index); bonds[b.index].add(a.index)
        self.active_residues=ids; self.candidates=[]; self.site_to_variables={}; self.movable=set()
        raw_candidates_by_site: dict[int, list[int]] = {}
        residue_one_letter: dict[int, str] = {}
        raw_pool_sizes: dict[int, int] = {}
        for site,rid in enumerate(ids):
            residue=residues[rid]; atoms={a.name:a.index for a in residue.atoms()}
            one_letter=gemmi.find_tabulated_residue(residue.name).one_letter_code
            residue_one_letter[site]=one_letter
            ca,cb=atoms["CA"],atoms["CB"]
            if cb not in bonds[ca]:
                raise ValueError(f"Missing CA-CB bond: {rid}")
            moving={cb}; frontier=[cb]
            while frontier:
                i=frontier.pop()
                for j in bonds[i]:
                    if {i,j}=={ca,cb}: continue
                    if j not in moving: moving.add(j); frontier.append(j)
            if ca in moving or not moving.issubset(set(atoms.values())):
                raise ValueError(f"Cyclic/crosslinked Active side chain unsupported: {rid}")
            indices=np.array(sorted(moving),dtype=int)
            if self.chi1_angles_override is None:
                if self.rotamer_mode=="dunbrack2010" and one_letter not in "AG":
                    phi,psi=allatom_backbone_angles[rid]
                    templates=_dunbrack_templates_for_site(
                        allatom_dunbrack_bins,one_letter,phi,psi,
                        probability_floor=self.rotamer_probability_floor,
                        sigma_offsets=self.rotamer_sigma_offsets,
                    )
                else:
                    templates=_expanded_rotamer_templates(one_letter)
            else:
                templates=tuple(RotamerTemplate(float(angle),1.0/len(self.chi1_angles_override),(float(angle),),(),"legacy_override")
                                for angle in self.chi1_angles_override)
            raw_pool_sizes[site]=len(templates)
            variables=[]
            for template in templates:
                targets=(template.chi_degrees if (self.rotamer_mode=="dunbrack2010" and self.chi1_angles_override is None)
                         else (template.chi1_degrees,))
                full=_apply_sidechain_chis(self.base_positions,atoms,bonds,residue.name,targets)
                coordinates=full[indices].copy()
                variables.append(len(self.candidates))
                self.candidates.append(dict(
                    site=site,residue_id=rid,residue_name=residue.name,
                    angle=float(template.chi1_degrees),chi_degrees=tuple(float(v) for v in targets),
                    prior_probability=float(template.prior_probability),
                    indices=indices,positions=coordinates))
            raw_candidates_by_site[site]=variables
            self.movable.update(moving)
        if candidate_relax_iterations:
            for group in raw_candidates_by_site.values():
                moving = set(self.candidates[group[0]]["indices"])
                system = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(self.system))
                for i in range(len(self.base_positions)):
                    if i not in moving:
                        system.setParticleMass(i, 0)
                integrator = mm.VerletIntegrator(.001)
                context = _openmm_context(mm, system, integrator)
                for variable in group:
                    context.setPositions(self.positions_for_variables([variable])*unit.nanometer)
                    mm.LocalEnergyMinimizer.minimize(context, 10., candidate_relax_iterations)
                    positions = np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
                    fixed = sorted(set(range(len(positions)))-moving)
                    if not np.allclose(positions[fixed], self.base_positions[fixed], atol=1e-10, rtol=0):
                        raise AssertionError("Candidate preparation moved fixed atoms")
                    candidate = self.candidates[variable]
                    candidate["positions"] = positions[candidate["indices"]].copy()
                del context, integrator

        # Adaptive retention after optional raw-candidate relaxation.
        if self.chi1_angles_override is None:
            helper=InterfaceQUBOBuilder(
                min_variables=3*len(ids), max_variables=30, max_sites=len(ids)
            )
            pseudo_nodes=np.arange(len(ids),dtype=np.int64)
            pseudo_aas=[residue_one_letter[i] for i in range(len(ids))]
            counts=helper._allocate_rotamer_counts(
                pseudo_nodes,pseudo_aas,self.site_scores
            )
            retained_old_indices=[]; retained_groups={}
            for site,count in enumerate(counts):
                ranked=sorted(
                    raw_candidates_by_site[site],
                    key=lambda variable: (
                        self.energy(self.positions_for_variables([variable])),
                        abs(float(self.candidates[variable]["angle"])),
                        int(variable),
                    ),
                )
                chosen=ranked[:count]
                retained_groups[site]=chosen
                retained_old_indices.extend(chosen)
            old_candidates=self.candidates
            old_to_new={old:new for new,old in enumerate(retained_old_indices)}
            self.candidates=[old_candidates[old] for old in retained_old_indices]
            self.site_to_variables={
                site:tuple(old_to_new[old] for old in retained_groups[site])
                for site in range(len(ids))
            }
            self.raw_rotamer_pool_sizes=[raw_pool_sizes[i] for i in range(len(ids))]
            self.retained_rotamers_per_site=[len(self.site_to_variables[i]) for i in range(len(ids))]
        else:
            self.site_to_variables={}
            cursor=0
            for site in range(len(ids)):
                width=len(raw_candidates_by_site[site])
                self.site_to_variables[site]=tuple(range(cursor,cursor+width))
                cursor+=width
            self.raw_rotamer_pool_sizes=[len(self.chi1_angles_override)]*len(ids)
            self.retained_rotamers_per_site=[len(self.chi1_angles_override)]*len(ids)

    def positions_for_variables(self, variables: Sequence[int]) -> np.ndarray:
        """Apply zero or one candidate per site; partial assignments support decomposition."""
        positions=self.base_positions.copy(); used=set()
        for variable in variables:
            c=self.candidates[int(variable)]
            if c["site"] in used: raise ValueError("Multiple candidates for one site")
            used.add(c["site"]); positions[c["indices"]]=c["positions"]
        return positions

    def energy(self, positions: np.ndarray) -> float:
        """Full Amber14 potential including bonded and exception terms, kcal/mol."""
        self.context.setPositions(positions*self.unit.nanometer)
        value=float(self.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(self.unit.kilocalorie_per_mole))
        if not math.isfinite(value): raise FloatingPointError("Nonfinite all-atom energy")
        return value

    def positions_for_chi_assignment(
        self, assignment: Mapping[str, Sequence[float]]
    ) -> np.ndarray:
        """Apply explicit residue->chi1..chiN targets without candidate projection.

        Used by TRAIN-ONLY coarse-to-Amber calibration so the atomistic target
        is evaluated for the exact same multi-chi rotamer represented by the
        coarse candidate metadata.
        """
        positions=self.base_positions.copy()
        residue_lookup={}
        bond_graph={}
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            residue_lookup[rid]=residue
        for bond in self.topology.bonds():
            a,b=bond[0].index,bond[1].index
            bond_graph.setdefault(a,set()).add(b);bond_graph.setdefault(b,set()).add(a)
        for rid,targets in assignment.items():
            if rid not in self.active_residues:
                raise ValueError(f"Calibration assignment contains non-active residue: {rid}")
            residue=residue_lookup[rid]
            atoms={a.name:a.index for a in residue.atoms()}
            target_tuple=tuple(float(v) for v in targets)
            expected=len(_CHI_ATOMS.get(residue.name, ()))
            if expected == 0 or len(target_tuple) != expected:
                raise ValueError(
                    f"{rid} expects {expected} chi angles, received {len(target_tuple)}"
                )
            positions=_apply_sidechain_chis(
                positions, atoms, bond_graph, residue.name, target_tuple
            )
            observed=_sidechain_chi_angles(
                {name:positions[index] for name,index in atoms.items()}, residue.name
            )
            if len(observed)!=len(target_tuple):
                raise AssertionError("Explicit multi-chi assignment dimensionality mismatch")
            for got,want in zip(observed,target_tuple):
                if abs((float(got)-float(want)+180)%360-180)>1e-4:
                    raise AssertionError(
                        f"Explicit multi-chi calibration rotation mismatch for {rid}: "
                        f"observed={observed}, target={target_tuple}"
                    )
        return positions

    def energy_for_chi_assignment(
        self, assignment: Mapping[str, Sequence[float]]
    ) -> float:
        """Amber14 potential for an exact residue->chi1..chiN assignment."""
        return self.energy(self.positions_for_chi_assignment(assignment))

    def positions_for_chi1_assignment(self, assignment: Mapping[str, float]) -> np.ndarray:
        """Apply explicit chi1 angles to Active residues without candidate-set projection.

        This is used by the TRAIN-ONLY coarse-to-Amber calibration stage so the
        exact same chi1 assignment evaluated by the coarse model is evaluated by
        Amber14, rather than snapping to the all-atom builder's retained states.
        """
        positions=self.base_positions.copy()
        residue_lookup={}
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            residue_lookup[rid]=residue
        for rid,angle in assignment.items():
            if rid not in self.active_residues:
                raise ValueError(f"Calibration assignment contains non-active residue: {rid}")
            residue=residue_lookup[rid]
            atoms={a.name:a.index for a in residue.atoms()}
            ca,cb=atoms["CA"],atoms["CB"]
            site=self.active_residues.index(rid)
            indices=self.candidates[self.site_to_variables[site][0]]["indices"]
            original=_chi1_angle({n:positions[i] for n,i in atoms.items()},residue.name)
            delta=np.deg2rad((float(angle)-original+180)%360-180)
            center=positions[ca]
            axis=positions[cb]-center
            axis/=np.linalg.norm(axis)
            relative=positions[indices]-center
            rotated=(relative*np.cos(delta)+np.cross(axis,relative)*np.sin(delta)
                     +np.outer(relative@axis,axis)*(1-np.cos(delta)))
            positions[indices]=center+rotated
            checked={n:positions[i] for n,i in atoms.items()}
            observed=_chi1_angle(checked,residue.name)
            if abs((observed-float(angle)+180)%360-180)>1e-5:
                raise AssertionError("Explicit chi1 calibration rotation mismatch")
        return positions

    def energy_for_chi1_assignment(self, assignment: Mapping[str, float]) -> float:
        """Amber14 potential for an exact residue->chi1 assignment."""
        return self.energy(self.positions_for_chi1_assignment(assignment))

    def build(self) -> QUBOResult:
        """Inclusion-exclusion physical terms; validate full-assignment energy equivalence."""
        # Use a complete candidate assignment as decomposition origin. A heavily
        # clashing perturbed input would otherwise cause catastrophic cancellation.
        count=len(self.candidates)
        anchors=[min(group,key=lambda v:self.energy(self.positions_for_variables([v])))
                 for group in self.site_to_variables.values()]
        anchor_positions=self.positions_for_variables(anchors)
        def assignment(variables):
            positions=anchor_positions.copy()
            for v in variables:
                c=self.candidates[v]
                positions[c["indices"]]=c["positions"]
            return positions
        baseline=self.energy(anchor_positions)
        singles=np.array([self.energy(assignment([v]))-baseline for v in range(count)])
        pairs=np.zeros((count,count))
        for i in range(count):
            for j in range(i+1,count):
                if self.candidates[i]["site"]!=self.candidates[j]["site"]:
                    pairs[i,j]=self.energy(assignment([i,j]))-baseline-singles[i]-singles[j]
        helper=InterfaceQUBOBuilder(min_variables=2,max_variables=30)
        penalty=helper._lambda_lower_bound(singles,pairs,self.site_to_variables)
        q=pairs.copy(); np.fill_diagonal(q,singles-penalty)
        for group in self.site_to_variables.values():
            for a,b in _combinations(group): q[a,b]+=2*penalty
        records=tuple(VariableRecord(v,c["site"],c["site"],c["site"],c["residue_id"],
            __import__("gemmi").find_tabulated_residue(c["residue_name"]).one_letter_code,
            list(self.site_to_variables[c["site"]]).index(v),c["angle"],1/len(self.site_to_variables[c["site"]]),float(singles[v]))
            for v,c in enumerate(self.candidates))
        # Vacuum/NoCutoff Amber14 is exactly pair-decomposable over side-chain
        # choices, so the QUBO must reproduce the full energy (1e-4 kcal/mol).
        # Implicit-solvent GBN2 is not: Born radii depend on every atom, so the
        # same inclusion-exclusion expansion is a pairwise approximation. Its
        # error is measured on the same sampled assignments and recorded, and
        # every structure is still relaxed/scored with the full GBN2 energy.
        exact=self.pair_decomposition_exact
        rng=np.random.default_rng(918); max_error=0.; squared_errors=[]
        for _ in range(12):
            selected=[int(rng.choice(g)) for g in self.site_to_variables.values()]
            x=np.zeros(count); x[selected]=1
            actual=self.energy(self.positions_for_variables(selected))
            predicted=baseline+singles@x+x@pairs@x
            if not (math.isfinite(actual) and math.isfinite(predicted)):
                raise FloatingPointError("Non-finite all-atom energy during decomposition check")
            max_error=max(max_error,abs(actual-predicted))
            squared_errors.append((actual-predicted)**2)
            if exact and not np.isclose(actual,predicted,atol=1e-4,rtol=1e-9):
                raise ValueError("Force field is not pair-decomposable at required precision")
        rms_error=float(math.sqrt(sum(squared_errors)/len(squared_errors)))
        offset=baseline+penalty*len(self.site_to_variables)
        h,j,ising_offset=qubo_to_ising(q,offset)
        roundoff_bound=max(1e-9,32*np.finfo(float).eps*(abs(offset)+np.abs(q).sum()+1))
        if roundoff_bound>1e-3:
            raise FloatingPointError(f"All-atom coefficient dynamic range exceeds 0.001 kcal/mol precision budget: {roundoff_bound}")
        ising_error=validate_qubo_ising_equivalence(q,offset,h,j,ising_offset,tolerance=roundoff_bound)
        return QUBOResult(q,records,self.site_to_variables,penalty,penalty,offset,singles,pairs,
            dict(model="Amber14 all-atom fixed-backbone chi1 grid",energy_unit="kcal/mol",
                physical_constant_offset=baseline,all_atom_equivalence_max_error=max_error,
                all_atom_equivalence_rms_error=rms_error,all_atom_equivalence_samples=12,
                pair_decomposition=("exact" if exact else "pairwise_approximation"),
                candidate_relax_iterations=self.candidate_relax_iterations,
                decomposition_anchor_variables=anchors,ising_equivalence_max_error=ising_error,
                ising_roundoff_tolerance=roundoff_bound,
                atom_count=len(self.base_positions),
                forcefield=(["amber14-all.xml"] if self.solvent_model=="vacuum"
                            else ["amber14-all.xml","implicit/gbn2.xml"]),
                solvent=("vacuum; NoCutoff" if self.solvent_model=="vacuum" else "implicit GBN2; NoCutoff"),
                solvent_model=self.solvent_model,
                raw_rotamer_pool_sizes=self.raw_rotamer_pool_sizes,
                rotamers_per_site=self.retained_rotamers_per_site,
                site_scores=self.site_scores.tolist(),
                candidate_chi_degrees=[list(candidate.get("chi_degrees",(candidate["angle"],))) for candidate in self.candidates],
                rotamer_state_policy=(f"{self.rotamer_mode} full side-chain rotamer states (chi1..chiN) -> 3--6 retained under <=30 variables" if self.rotamer_mode=="dunbrack2010" and self.chi1_angles_override is None else "explicit legacy chi1 angle override"),
                rotamer_library_path=(None if self.rotamer_library_path is None else str(self.rotamer_library_path)),
                candidate_scope=("Dunbrack full side-chain chi state; Amber14 single-candidate prescreen, no affinity claim"
                    if self.rotamer_mode=="dunbrack2010" and self.chi1_angles_override is None
                    else "legacy chi1-only candidate; no affinity claim")))

    def write_structure(self, positions: np.ndarray, destination: Path) -> None:
        """Write author-ID CIF, with occupancy=1 for generated computational atoms."""
        import gemmi
        with Path(destination).open("w") as f:
            self.app.PDBxFile.writeFile(self.topology,positions*self.unit.nanometer,f,keepIds=True)
        structure=gemmi.read_structure(str(destination))
        for chain in structure[0]:
            for residue in chain:
                for atom in residue: atom.occ=1.
        structure.make_mmcif_document().write_file(str(destination))

    def reconstruct(self, bits: Sequence[int], destination: Path, *,
                    minimize_iterations: int = 200) -> dict[str, Any]:
        """Write a full-atom CIF before/after identically constrained local relaxation."""
        x=np.asarray(bits)
        if x.shape!=(len(self.candidates),) or not np.all((x==0)|(x==1)):
            raise ValueError("Invalid binary assignment")
        if any(x[list(g)].sum()!=1 for g in self.site_to_variables.values()):
            raise ValueError("Assignment violates site one-hot constraints")
        positions=self.positions_for_variables(np.flatnonzero(x))
        return self.relax_positions(positions,destination,minimize_iterations=minimize_iterations)

    def perturb_sidechain_chis(
        self, seed: int, min_degrees: float = 40., max_degrees: float = 120.
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Perturb every defined side-chain chi angle without reference-based rejection."""
        if not 0 < min_degrees <= max_degrees <= 180:
            raise ValueError("Require 0 < min_degrees <= max_degrees <= 180")
        rng=np.random.default_rng(seed)
        positions=self.base_positions.copy()
        residue_lookup={}
        bond_graph={i:set() for i in range(self.topology.getNumAtoms())}
        for a,b in self.topology.bonds():
            bond_graph[a.index].add(b.index);bond_graph[b.index].add(a.index)
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            residue_lookup[rid]=residue
        records=[]
        for rid in self.active_residues:
            residue=residue_lookup[rid]
            definitions=_CHI_ATOMS.get(residue.name,())
            if not definitions:
                raise ValueError(f"No supported chi definitions for Active residue {rid}")
            atoms={a.name:a.index for a in residue.atoms()}
            before=[]
            targets=[]
            deltas=[]
            # Read current torsions from the progressively unchanged input,
            # then set all target chis in one sequential internal-coordinate pass.
            for definition in definitions:
                a,b,c,d=(atoms[name] for name in definition)
                current=_torsion_angle_degrees(positions[a],positions[b],positions[c],positions[d])
                delta=float(rng.uniform(min_degrees,max_degrees)*rng.choice([-1,1]))
                before.append(current);deltas.append(delta)
                targets.append(((current+delta+180.0)%360.0)-180.0)
            positions=_apply_sidechain_chis(positions,atoms,bond_graph,residue.name,targets)
            after=[]
            for definition in definitions:
                a,b,c,d=(atoms[name] for name in definition)
                after.append(_torsion_angle_degrees(
                    positions[a],positions[b],positions[c],positions[d]))
            records.append(dict(
                residue_id=rid,seed=seed,residue_name=residue.name,
                chi_before=[float(v) for v in before],
                chi_after=[float(v) for v in after],
                delta_degrees=[float(v) for v in deltas],
            ))
        return positions,records

    def perturb_chi1(self, seed: int, min_degrees: float = 40.,
                     max_degrees: float = 120.) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Deterministic signed chi1 perturbations; no energy/reference-based rejection.

        This is a retrospective fixed-backbone recovery control. Other chi angles
        and the backbone remain input-derived, so it is not de novo prediction.
        """
        if not 0 < min_degrees <= max_degrees <= 180:
            raise ValueError("Require 0 < min_degrees <= max_degrees <= 180")
        rng=np.random.default_rng(seed);positions=self.base_positions.copy();record=[]
        residues={}
        for residue in self.topology.residues():
            residues[f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"]=residue
        for site,rid in enumerate(self.active_residues):
            atoms={a.name:a.index for a in residues[rid].atoms()}
            indices=self.candidates[self.site_to_variables[site][0]]["indices"]
            center=self.base_positions[atoms["CA"]]
            axis=self.base_positions[atoms["CB"]]-center;axis/=np.linalg.norm(axis)
            degrees=float(rng.uniform(min_degrees,max_degrees)*rng.choice([-1,1]))
            angle=np.deg2rad(degrees);relative=self.base_positions[indices]-center
            positions[indices]=center+relative*np.cos(angle)+np.cross(axis,relative)*np.sin(angle)+np.outer(relative@axis,axis)*(1-np.cos(angle))
            old=_chi1_angle({n:self.base_positions[i] for n,i in atoms.items()},residues[rid].name)
            new=_chi1_angle({n:positions[i] for n,i in atoms.items()},residues[rid].name)
            if abs((new-old-degrees+180)%360-180)>1e-6:
                raise AssertionError("Perturbed chi1 does not match requested rotation")
            record.append(dict(residue_id=rid,seed=seed,chi1_before=old,chi1_after=new,delta_degrees=degrees))
        return positions,record

    def relax_cdr_loop(self, destination: Path, cdr_residues: Sequence[str], *,
                       iterations: int = 100, restraint_k: float = 100.) -> dict[str, Any]:
        """Stage 2: loop atoms plus Active sidechains move; weak BB restraint to stage 1.

        k is kJ/mol/nm^2; potential is k/2*distance^2. Loop side chains and H
        move with their backbone to avoid stretching bonds to immobilized atoms.
        """
        if iterations <= 0 or restraint_k <= 0 or not cdr_residues:
            raise ValueError("Stage 2 requires mapped CDR residues and positive controls")
        parsed=self.app.PDBxFile(str(destination))
        positions=np.asarray(parsed.positions.value_in_unit(self.unit.nanometer))
        movable=set(self.movable); backbone=[]; found=set()
        for residue in self.topology.residues():
            rid=f"{residue.chain.id}:{residue.id}{residue.insertionCode.strip()}"
            if rid in cdr_residues:
                found.add(rid)
                for atom in residue.atoms():
                    movable.add(atom.index)
                    if atom.name in ('N','CA','C','O'): backbone.append(atom.index)
        if found!=set(cdr_residues): raise ValueError("CDR residue topology mapping failed")
        system=self.mm.XmlSerializer.deserialize(self.mm.XmlSerializer.serialize(self.system))
        frozen=sorted(set(range(len(positions)))-movable)
        for i in frozen: system.setParticleMass(i,0)
        force=self.mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
        force.addGlobalParameter('k',restraint_k)
        for name in ('x0','y0','z0'): force.addPerParticleParameter(name)
        for i in backbone: force.addParticle(i,positions[i].tolist())
        system.addForce(force)
        integrator=self.mm.VerletIntegrator(.001)
        context=_openmm_context(self.mm, system, integrator)
        context.setPositions(positions*self.unit.nanometer)
        initial=float(context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(self.unit.kilocalories_per_mole))
        self.mm.LocalEnergyMinimizer.minimize(context,10.,iterations)
        state=context.getState(getPositions=True,getEnergy=True)
        final=np.asarray(state.getPositions(asNumpy=True).value_in_unit(self.unit.nanometer))
        augmented=float(state.getPotentialEnergy().value_in_unit(self.unit.kilocalories_per_mole))
        del context,integrator
        if not np.allclose(final[frozen],positions[frozen],atol=1e-10,rtol=0):
            raise AssertionError('Stage 2 moved frozen atoms')
        if augmented>initial+1e-4: raise ValueError('Stage 2 raised restrained objective')
        self.write_structure(positions,Path(destination).with_name(Path(destination).stem+'_stage1.cif'))
        self.write_structure(final,destination)
        return dict(stage2_iterations=iterations,stage2_restraint_k_kj_mol_nm2=restraint_k,
            stage2_physical_energy_kcal=self.energy(final),stage2_restrained_energy_kcal=augmented,
            stage2_backbone_displacement_angstrom=float(10*np.sqrt(np.mean(np.sum((final[backbone]-positions[backbone])**2,axis=1)))),
            stage2_frozen_atoms=len(frozen))

    def relax_positions(self, positions: np.ndarray, destination: Path, *,
                        minimize_iterations: int = 200) -> dict[str, Any]:
        """Same constrained relaxation for sampled candidates and unsearched input."""
        if minimize_iterations<0: raise ValueError("minimize_iterations must be nonnegative")
        positions=np.asarray(positions,dtype=float).copy()
        if positions.shape!=self.base_positions.shape or not np.isfinite(positions).all():
            raise ValueError("Invalid full-atom positions")
        frozen=sorted(set(range(len(positions)))-self.movable)
        if not np.allclose(positions[frozen],self.base_positions[frozen],atol=1e-10,rtol=0):
            raise ValueError("Input changed frozen/background atoms")
        before=self.energy(positions)
        destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
        self.write_structure(positions,destination.with_name(destination.stem+"_discrete.cif"))
        if minimize_iterations:
            system=self.mm.XmlSerializer.deserialize(self.mm.XmlSerializer.serialize(self.system))
            for i in range(len(positions)):
                if i not in self.movable: system.setParticleMass(i,0)
            integrator=self.mm.VerletIntegrator(.001)
            context=_openmm_context(self.mm, system, integrator)
            context.setPositions(positions*self.unit.nanometer)
            self.mm.LocalEnergyMinimizer.minimize(context,10.,minimize_iterations)
            positions=np.asarray(context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(self.unit.nanometer))
            del context,integrator
        frozen=sorted(set(range(len(positions)))-self.movable)
        if not np.allclose(positions[frozen],self.base_positions[frozen],atol=1e-10,rtol=0):
            raise AssertionError("Frozen atoms moved during relaxation")
        after=self.energy(positions)
        if after>before+1e-4: raise ValueError("Relaxation increased potential energy")
        self.write_structure(positions,destination)
        return dict(discrete_energy_kcal=before,relaxed_energy_kcal=after,
            max_iterations=minimize_iterations,frozen_atoms=len(frozen),movable_atoms=len(self.movable),
            note="Iteration cap is not a convergence guarantee; fixed-backbone vacuum energy is not binding affinity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        help="Optional trusted local pruned PyG .pt file; defaults to a virtual graph.",
    )
    arguments = parser.parse_args()
    if arguments.graph is None:
        example = _virtual_pruned_graph(site_count=6)
    else:
        # torch.load uses pickle for PyG Data. Only load files generated locally
        # or obtained from a trusted source.
        example = load_graph(arguments.graph)

    builder = InterfaceQUBOBuilder()
    result = builder.build(example)
    Q = result.Q
    assert Q.shape[0] == Q.shape[1] <= 30
    assert Q.shape[0] >= 20
    assert np.count_nonzero(np.tril(Q, k=-1)) == 0
    assert np.isfinite(Q).all()
    assert all(3 <= len(indices) <= 6 for indices in result.site_to_variables.values())
    assert result.lambda_value >= result.lambda_lower_bound
    assert result.metadata["lambda_exceeds_max_pair_attraction"]

    h, J, offset = qubo_to_ising(Q, result.constant_offset)
    assert validate_qubo_ising_equivalence(
        Q, result.constant_offset, h, J, offset
    ) <= 1e-9
    one_hot = np.zeros(Q.shape[0], dtype=np.int8)
    for variables in result.site_to_variables.values():
        one_hot[variables[0]] = 1
    spins = 1.0 - 2.0 * one_hot
    qubo_energy = result.energy(one_hot)
    ising_energy = float(offset + h @ spins)
    for left in range(len(spins)):
        for right in range(left + 1, len(spins)):
            ising_energy += J[left, right] * spins[left] * spins[right]
    assert np.isclose(qubo_energy, ising_energy, atol=1e-8)

    print("Variable mapping")
    print("idx  site  node  residue  aa  rot  chi1  prior    E_self")
    for record in result.variable_map:
        print(
            f"{record.variable_index:>3}  {record.site_index:>4}  "
            f"{record.node_index:>4}  {record.residue_id:<7}  "
            f"{record.amino_acid:>2}  {record.rotamer_index:>3}  "
            f"{record.chi1_degrees:>5.0f}  {record.prior_probability:>5.2f}  "
            f"{record.self_energy:>9.3f}"
        )
    linear_terms = int(np.count_nonzero(np.abs(h) > 1e-12))
    coupling_terms = int(np.count_nonzero(np.abs(np.triu(J, 1)) > 1e-12))
    print(
        json.dumps(
            {
                "qubo_dimension": int(Q.shape[0]),
                "sites": len(result.site_to_variables),
                "lambda_lower_bound": result.lambda_lower_bound,
                "lambda_used": result.lambda_value,
                "ising_linear_terms": linear_terms,
                "ising_coupling_terms": coupling_terms,
                "basis_hamiltonian_terms": linear_terms + coupling_terms,
                "offset": offset,
                "qubo_ising_energy_check": qubo_energy,
            },
            indent=2,
        )
    )
