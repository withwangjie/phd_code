"""Read heavy-atom protein structures and score atomistic side-chain predictions.

Parses a PDB/mmCIF into per-residue heavy-atom records, computes backbone
phi/psi and side-chain chi angles, applies chi rotations, and evaluates a
predicted structure against the native one (symmetry-corrected RMSD, chi
recovery).
"""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import numpy as np
from nanoqc.structure.residue_tables import (
    PEPTIDE_BOND_MAX_C_N_ANGSTROM, SIDECHAIN_HEAVY_ATOMS, SYMMETRIC_SWAPS,
)



def _sha256_path(path: Optional[Path]) -> Optional[str]:
    """Return a content hash for an external scientific input file."""
    if path is None:
        return None
    resolved=Path(path)
    if not resolved.is_file():
        return None
    digest=hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    """Backbone phi/psi for one author residue id; termini and chain breaks fail closed.

    Neighbours are the residues actually peptide-bonded to ``rid`` (C-N
    distance), not the neighbours in (seqid, insertion code) order: IMGT
    numbers CDR3 insertions at position 112 in reverse (112B, 112A, 112), so
    sorting insertion codes alphabetically does not give chain order.
    """
    match=_RESIDUE_ID_PATTERN.match(rid)
    if match is None: raise ValueError(f"Cannot parse residue id {rid}")
    chain=match.group("chain")
    current=residues[rid]
    missing=[name for name in ("N","CA","C") if name not in current["atoms"]]
    if missing: raise ValueError(f"Missing backbone atoms {missing} for Dunbrack lookup at {rid}")
    same_chain=[key for key in residues
                if key!=rid and (lambda m: m is not None and m.group("chain")==chain)(_RESIDUE_ID_PATTERN.match(key))]
    if not same_chain:
        raise ValueError(f"Dunbrack mode requires non-terminal Active residue: {rid}")
    def bonded(atom_here: str, atom_there: str):
        best=None
        for key in same_chain:
            atoms=residues[key]["atoms"]
            if atom_there not in atoms: continue
            gap=float(np.linalg.norm(np.asarray(current["atoms"][atom_here],dtype=float)
                                     -np.asarray(atoms[atom_there],dtype=float)))
            if best is None or gap<best[0]: best=(gap,key)
        return best
    neighbours={}
    for side,here,there in (("preceding","N","C"),("following","C","N")):
        found=bonded(here,there)
        if found is None or found[0]>=PEPTIDE_BOND_MAX_C_N_ANGSTROM:
            gap="none" if found is None else f"{found[0]:.2f} A"
            raise ValueError(
                f"Dunbrack mode requires peptide-bonded neighbours at {rid}: "
                f"no {side} residue within the peptide C-N bond distance (closest {gap}); "
                "terminus or chain break")
        neighbours[side]=residues[found[1]]
    prev,nxt=neighbours["preceding"],neighbours["following"]
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
