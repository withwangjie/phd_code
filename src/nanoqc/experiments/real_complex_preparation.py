"""Structure preparation for real-complex recovery targets.

Protein-conformer stripping, terminal-oxygen completion and the shared
Active-site preparation (``prepare``) used by the validation-queue driver
(run_real_complex_pilot.py) and by the energy-calibration generator.
"""
from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import torch

from nanoqc.qubo.atomistic_structure import read_atomistic_structure, _SIDECHAIN_NAMES
from nanoqc.structure.physical_quality import StructureQualityError, topology_geometry_audit


def complete_terminal_oxygen(path: Path) -> list[str]:
    """Use PDBFixer only for OXT, never repair internal or side-chain atoms."""
    from pdbfixer import PDBFixer
    from openmm import app, unit
    fixer = PDBFixer(filename=str(path))
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        raise ValueError("Internal heavy-atom repair would be required")
    if any(set(names) != {"OXT"} for names in fixer.missingTerminals.values()):
        raise ValueError("Unsupported terminal atom repair")
    changes = [f"added terminal OXT {r.chain.id}:{r.id}{r.insertionCode.strip()}" for r in fixer.missingTerminals]
    if not changes:
        return changes
    before = {(a.residue.chain.id, a.residue.id, a.residue.insertionCode, a.name):np.array(x)
              for a,x in zip(fixer.topology.atoms(),fixer.positions.value_in_unit(unit.nanometer))}
    fixer.addMissingAtoms(seed=42)
    for a,x in zip(fixer.topology.atoms(),fixer.positions.value_in_unit(unit.nanometer)):
        key=(a.residue.chain.id,a.residue.id,a.residue.insertionCode,a.name)
        if key in before and not np.allclose(x,before[key],rtol=0,atol=1e-8):
            raise ValueError("Terminal completion moved an observed atom")
    with path.open("w") as handle:
        app.PDBxFile.writeFile(fixer.topology,fixer.positions,handle,keepIds=True)
    st=gemmi.read_structure(str(path))
    for c in st[0]:
        for r in c:
            for a in r:
                a.occ=1.
    st.make_mmcif_document().write_file(str(path))
    return changes


def strip_to_protein_conformer(st, residues: dict) -> list[str]:
    """Reduce a single-model gemmi structure to force-field-ready protein atoms.

    In place: removes waters, hydrogens/deuteriums and zero-occupancy atoms,
    keeps exactly one alternate conformer per residue (the one
    :func:`read_atomistic_structure` chose, passed in as ``residues``) and
    clears altloc labels. Fails closed on any non-protein component or a
    residue missing canonical heavy atoms; never repairs coordinates. Shared
    by target preparation and energy-calibration data generation.
    """
    changes = []
    for chain in st[0]:
        for k in reversed(range(len(chain))):
            r = chain[k]
            if r.is_water():
                changes.append(f"removed water {chain.name}:{r.seqid}")
                del chain[k]
                continue
            rid = f"{chain.name}:{r.seqid}"
            if rid not in residues:
                raise ValueError(f"Unsupported nonprotein component {rid}/{r.name}")
            entry = residues[rid]
            required = set(("N", "CA", "C", "O")) | set(_SIDECHAIN_NAMES[r.name].split())
            missing = required - entry["atoms"].keys()
            if missing:
                raise ValueError(f"Missing heavy atoms {rid}: {sorted(missing)}")
            label = entry["altloc"]
            for j in reversed(range(len(r))):
                atom = r[j]
                if atom.element.name in ("H", "D") or atom.occ <= 0 or atom.altloc not in ("\x00", " ", "", label):
                    del r[j]
                else:
                    atom.altloc = "\x00"
            if label:
                changes.append(f"altloc {rid}={label}")
    return changes


def prepare(graph, source: Path, destination: Path, sites: int, *, pruning: str = 'cdr',
            seed: int = 42, checkpoint: Path | None = None,
            antigen_guidance_weight: float = 0.25,
            antigen_proximity_scale: float = 6.0,
            contact_ca_cutoff: float = 8.0,
            homology_isolation: dict[str, float] | None = None,
            eligibility_only: bool = False,
            rotamer_mode: str = "dunbrack2010",
            allowed_residues: set[str] | None = None) -> dict:
    """Resolve structure and select Active sites under the shared main protocol."""
    if not 0.0 <= antigen_guidance_weight <= 1.0:
        raise ValueError("antigen_guidance_weight must be in [0,1]")
    if not np.isfinite(antigen_proximity_scale) or antigen_proximity_scale <= 0:
        raise ValueError("antigen_proximity_scale must be positive finite")
    if not np.isfinite(contact_ca_cutoff) or contact_ca_cutoff <= 0:
        raise ValueError("contact_ca_cutoff must be positive finite")
    if rotamer_mode not in ("legacy","dunbrack2010","pyrosetta_dun10"):
        raise ValueError("rotamer_mode must be legacy, dunbrack2010 or pyrosetta_dun10")
    if rotamer_mode in ("dunbrack2010","pyrosetta_dun10"):
        if not hasattr(graph,"backbone_phi") or not hasattr(graph,"backbone_psi"):
            raise ValueError("Formal Dunbrack site selection requires backbone phi/psi metadata")
        phi_values=graph.backbone_phi.detach().cpu().numpy()
        psi_values=graph.backbone_psi.detach().cpu().numpy()
        if phi_values.shape!=(len(graph.residue_ids),) or psi_values.shape!=(len(graph.residue_ids),):
            raise ValueError("backbone phi/psi metadata must align with graph residues")
    else:
        phi_values=psi_values=None
    residues = read_atomistic_structure(source)
    st = gemmi.read_structure(str(source))
    if len(st) != 1:
        raise ValueError("Multiple models")
    expected = set(graph.residue_ids)
    if set(residues) != expected:
        raise ValueError("Raw/graph protein residue identities differ")
    try:
        changes = strip_to_protein_conformer(st, residues)
    except ValueError as exc:
        raise StructureQualityError(str(exc),category="input_heavy_atoms",
            audit=dict(reason=str(exc),policy="no internal atom or loop reconstruction")) from exc
    # Audit experimental coordinates before selection or deliberate perturbation.
    # IMGT residue-number gaps are not evidence of a missing peptide bond.
    import io
    from openmm import app, unit
    parsed=app.PDBxFile(io.StringIO(st.make_mmcif_document().as_string()))
    preparation_quality=topology_geometry_audit(parsed.topology,
        np.asarray(parsed.positions.value_in_unit(unit.nanometer)))
    if not preparation_quality["topology_passed"] or not preparation_quality["geometry_passed"]:
        raise StructureQualityError("Source structure failed topology/near-coincidence audit",
            category="input_quality",audit=preparation_quality)
    for i, rid in enumerate(graph.residue_ids):
        if not np.allclose(residues[rid]["atoms"]["CA"], graph.pos[i].numpy(), atol=.002):
            raise ValueError(f"Raw/graph coordinates differ at {rid}")
    groups = dict(zip(graph.chain_ids, graph.chain_groups))
    vhh = [c for c, g in groups.items() if g == 0]
    if len(vhh) != 1:
        raise ValueError("Expected one VHH chain")
    chain_nodes = [i for i, rid in enumerate(graph.residue_ids) if rid.rsplit(":", 1)[0] == vhh[0]]
    sequence = graph.chain_sequences[graph.chain_ids.index(vhh[0])]
    cdr = graph.cdr3_seq
    if not cdr or sequence.count(cdr) != 1:
        raise StructureQualityError("CDR-H3 does not map uniquely; no loop reconstruction attempted",
            category="input_annotation",audit=dict(cdr3_sequence=cdr,
                observed_sequence=sequence,reason="cdr3_not_uniquely_observed"))
    start = sequence.index(cdr)
    cdr_ids = [graph.residue_ids[i] for i in chain_nodes[start:start+len(cdr)]]
    partners = sorted(r for r in residues if groups[r.rsplit(":", 1)[0]] == 1)
    antigen_nodes = torch.where(graph.x[:, -1] == 1)[0]
    if len(antigen_nodes) == 0:
        raise ValueError("Graph contains no antigen/group-1 residues")
    scored = []
    for i in chain_nodes:
        rid=graph.residue_ids[i]
        entry = residues[rid]
        # Formal main protocol ranks every chemically movable VHH residue.
        # Native interface distance is retained only as a baseline feature,
        # never as an oracle eligibility gate.
        if entry["name"] in ("ALA", "GLY", "PRO", "CYS"):
            continue
        if allowed_residues is not None and rid not in allowed_residues:
            continue
        if rotamer_mode in ("dunbrack2010","pyrosetta_dun10") and (
            not np.isfinite(phi_values[i]) or not np.isfinite(psi_values[i])
        ):
            continue
        ca_distances = torch.linalg.norm(graph.pos[antigen_nodes] - graph.pos[i], dim=1)
        dist = float(ca_distances.min().item())
        contact_count = int((ca_distances < float(contact_ca_cutoff)).sum().item())
        scored.append((dist, rid, contact_count, i))
    if len(scored) < sites:
        raise ValueError(f"Only {len(scored)} chemically movable VHH sites")

    # Queue-freeze eligibility must not depend on a temporary pruning/ranking
    # strategy before the EGNN checkpoint exists. In eligibility-only mode we
    # preserve the cleaned structure and expose the full chemically movable
    # residue pool; main() then verifies that at least K of these residues are
    # independently Dunbrack/Amber-compatible, without choosing the formal
    # Active set.
    if eligibility_only:
        st.make_mmcif_document().write_file(str(destination))
        return dict(
            active_residues=[], active_site_scores=[],
            eligible_residues=[item[1] for item in scored],
            alignment_residues=partners, partner_residues=partners,
            preparation_changes=changes, preparation_quality=preparation_quality, cdr3_residues=cdr_ids,
            pruning="eligibility_only", pruning_candidate_count=len(scored),
            pruning_seed=None, model_status=None,
            antigen_guidance_weight=float(antigen_guidance_weight),
            antigen_proximity_scale=float(antigen_proximity_scale),
            contact_ca_cutoff_angstrom=float(contact_ca_cutoff),
            rotamer_mode=str(rotamer_mode),
            selection_origin=(
                "eligibility-only queue freeze: no contact/distance/CDR/EGNN ranking; "
                "formal Active residues are selected only after EGNN training"
            ),
        )
    model_status=None
    if pruning=='contact':
        ordered=sorted(scored,key=lambda x:(-x[2],x[0],x[1]))
    elif pruning=='distance':
        ordered=sorted(scored,key=lambda x:(x[0],x[1]))
    elif pruning=='cdr':
        ordered=sorted(scored,key=lambda x:(x[1] not in cdr_ids,-x[2],x[1]))
    elif pruning=='random':
        ordered=[scored[i] for i in np.random.default_rng(seed).permutation(len(scored))]
    elif pruning=='egnn':
        from nanoqc.model.model_egnn_pruning import load_interface_scorer, assert_checkpoint_graph_compatible
        model,info=load_interface_scorer(checkpoint,torch_device=torch.device('cpu'),seed=seed)
        model_status=info.status
        if model_status!='checkpoint_loaded':
            raise ValueError('EGNN ablation requires a valid trained checkpoint')
        assert_checkpoint_graph_compatible(
            info, graph, homology_isolation=homology_isolation
        )
        with torch.no_grad():
            model_scores=model(graph.x,graph.pos,graph.edge_index).reshape(-1)
        candidate_indices=torch.tensor([item[3] for item in scored],dtype=torch.long)
        nearest=torch.cdist(graph.pos[candidate_indices],graph.pos[antigen_nodes]).min(1).values
        proximity=torch.exp(-nearest/float(antigen_proximity_scale))
        composite=(1.0-antigen_guidance_weight)*model_scores[candidate_indices]+antigen_guidance_weight*proximity
        composite_map={int(idx):float(score) for idx,score in zip(candidate_indices,composite)}
        ordered=sorted(scored,key=lambda x:(-composite_map[x[3]],x[1]))
    else: raise ValueError('Unknown pruning strategy')
    chosen = ordered[:sites]
    active = [item[1] for item in chosen]
    if pruning == 'egnn':
        active_site_scores = [composite_map[item[3]] for item in chosen]
    elif pruning == 'contact':
        values = np.asarray([item[2] for item in scored], dtype=float)
        denom = max(float(values.max()), 1.0)
        active_site_scores = [float(item[2]) / denom for item in chosen]
    elif pruning == 'distance':
        active_site_scores = [float(np.exp(-item[0] / antigen_proximity_scale)) for item in chosen]
    elif pruning == 'cdr':
        active_site_scores = [1.0 if item[1] in cdr_ids else 0.0 for item in chosen]
    else:
        active_site_scores = [0.0 for _ in chosen]
    st.make_mmcif_document().write_file(str(destination))
    return dict(active_residues=active, active_site_scores=active_site_scores,
                alignment_residues=partners, partner_residues=partners,
                preparation_changes=changes, preparation_quality=preparation_quality, cdr3_residues=cdr_ids,pruning=pruning,
                pruning_candidate_count=len(scored),pruning_seed=seed,model_status=model_status,
                antigen_guidance_weight=float(antigen_guidance_weight),
                antigen_proximity_scale=float(antigen_proximity_scale),
                contact_ca_cutoff_angstrom=float(contact_ca_cutoff),
                rotamer_mode=str(rotamer_mode),
                selection_origin=(f"{pruning} selection across all chemically movable VHH residues; native distance/contact used only as explicit baselines; "
                    f"EGNN path uses shared composite score=(1-w)*EGNN+w*exp(-dAg/{antigen_proximity_scale:.3g}A), w={antigen_guidance_weight:.3f}; "
                    "fixed before perturbation; retrospective control"))
