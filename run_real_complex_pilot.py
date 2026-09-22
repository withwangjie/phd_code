"""Real VHH multi-chi side-chain recovery with explicit data separation and exclusions.

This is retrospective, native-backbone-conditioned recovery, not blind docking.
The original benchmark driver remains the solver/evaluation implementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import traceback

import gemmi
import numpy as np
import parasail
import torch
from tqdm import tqdm

from generate_figure1_pymol_script import extract_source
from subgraph_to_qubo import read_atomistic_structure, _SIDECHAIN_NAMES, AllAtomInterfaceQUBOBuilder
from seed_streams import derive_streams, derive_child_seed, save_stream_map, DEFAULT_MASTER_SEED


def identity(a: str, b: str, threshold: float = .4) -> float:
    """Global identity over alignment length; length bound only rejects >=threshold."""
    if min(len(a), len(b)) / max(len(a), len(b)) < threshold:
        return 0.
    values = []
    for x, y in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(x, y, 10, 1, parasail.blosum62)
        if result.saturated:
            raise ValueError("Alignment saturation")
        values.append(result.matches / result.length)
    return max(values)


def identity_detail(a: str, b: str, threshold: float = .4) -> dict:
    """Like `identity`, but always returns the actual identity value and the
    length-ratio coverage gate -- for an auditable record, never just a
    boolean pass/fail. Uses the same alignment (parasail NW + BLOSUM62) and
    the same configurable length-ratio coverage gate as `identity`, just without
    collapsing the result to a threshold comparison."""
    coverage = min(len(a), len(b)) / max(len(a), len(b))
    if coverage < threshold:
        return dict(identity=0., coverage=coverage, length_gated=True)
    values = []
    for x, y in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(x, y, 10, 1, parasail.blosum62)
        if result.saturated:
            raise ValueError("Alignment saturation")
        values.append(result.matches / result.length)
    return dict(identity=max(values), coverage=coverage, length_gated=False)


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
    if rotamer_mode not in ("legacy","dunbrack2010"):
        raise ValueError("rotamer_mode must be legacy or dunbrack2010")
    if rotamer_mode=="dunbrack2010":
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
        raise ValueError("CDR-H3 does not map uniquely")
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
        if rotamer_mode=="dunbrack2010" and (
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
            preparation_changes=changes, cdr3_residues=cdr_ids,
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
        from batch_benchmark_hard_set import load_interface_scorer, assert_checkpoint_graph_compatible
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
                preparation_changes=changes, cdr3_residues=cdr_ids,pruning=pruning,
                pruning_candidate_count=len(scored),pruning_seed=seed,model_status=model_status,
                antigen_guidance_weight=float(antigen_guidance_weight),
                antigen_proximity_scale=float(antigen_proximity_scale),
                contact_ca_cutoff_angstrom=float(contact_ca_cutoff),
                rotamer_mode=str(rotamer_mode),
                selection_origin=(f"{pruning} selection across all chemically movable VHH residues; native distance/contact used only as explicit baselines; "
                    f"EGNN path uses shared composite score=(1-w)*EGNN+w*exp(-dAg/{antigen_proximity_scale:.3g}A), w={antigen_guidance_weight:.3f}; "
                    "fixed before perturbation; retrospective control"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset_clean_500"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("real_complex_pilot"))
    parser.add_argument("--targets", type=int, default=0,
        help="Maximum number of eligible targets to select. 0 (default) = process every qualifying "
             "target with no hidden cap; a positive value freezes an EXPLICIT subset and must be set "
             "before the run starts (never adjusted after inspecting results). The actual coverage "
             "(selected vs. examined qualifying pool) is always stated in real_complex_report.md.")
    parser.add_argument("--sites", type=int, default=6)
    parser.add_argument("--vhh-identity-threshold", type=float, default=0.80)
    parser.add_argument("--cdr-h3-identity-threshold", type=float, default=0.50)
    parser.add_argument("--antigen-identity-threshold", type=float, default=0.30)
    parser.add_argument("--antigen-min-length-coverage", type=float, default=0.70)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--perturbation-mode",choices=("multi_chi","chi1"),default="multi_chi")
    parser.add_argument("--solvent-model",choices=("vacuum","gbn2"),default="vacuum")
    parser.add_argument("--min-perturb-degrees",type=float,default=40.0)
    parser.add_argument("--max-perturb-degrees",type=float,default=120.0)
    parser.add_argument("--outputs", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=90)
    parser.add_argument("--qaoa-depth", type=int, default=2)
    parser.add_argument("--relax-iterations", type=int, default=200)
    parser.add_argument("--candidate-relax-iterations", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eligibility-only", action="store_true",
        help="With --prepare-only, freeze target membership without selecting Active residues. "
             "Requires only that at least --sites VHH residues are independently Dunbrack/Amber-compatible.")
    parser.add_argument("--eval-shots",type=int,choices=(200,500,1000))
    parser.add_argument("--loop-relax-iterations",type=int,default=100)
    parser.add_argument("--pruning",choices=('egnn','contact','distance','cdr','random'),default='egnn')
    parser.add_argument("--pdb-id",type=str)
    parser.add_argument("--pdb-allowlist-file",type=Path,default=None,
        help="Optional JSON file (a plain list of PDB IDs, or a list of dicts each with a "
             "target/pdb_id key -- e.g. an earlier run's own selected_targets.json) restricting "
             "candidates to EXACTLY this set before the eligibility loop runs. Used to make a later "
             "invocation (e.g. with --pruning egnn once a checkpoint exists) reproduce the identical "
             "frozen target set an earlier --prepare-only freeze already decided, rather than relying "
             "on re-derivation determinism alone. Identity/exclusion screening still runs against this "
             "restricted set as a safety check -- a target can still end up excluded here, but no target "
             "outside the allowlist can ever be added.")
    parser.add_argument("--checkpoint",type=Path,default=Path('quantum-protein/checkpoints_500/best_egnn_pruning.pt'))
    parser.add_argument("--cluster-map",type=Path,default=None,
        help="Frozen PDB->family/structure cluster map used for eligibility and independence checks.")
    parser.add_argument("--antigen-guidance-weight",type=float,default=0.25)
    parser.add_argument("--antigen-proximity-scale",type=float,default=6.0)
    parser.add_argument("--contact-ca-cutoff",type=float,default=8.0)
    parser.add_argument("--rotamer-mode",choices=("legacy","dunbrack2010"),default="dunbrack2010")
    parser.add_argument("--rotamer-library",type=Path)
    parser.add_argument("--rotamer-probability-floor",type=float,default=1e-4)
    parser.add_argument("--rotamer-sigma-offsets",type=float,nargs="+",default=[-1.0,0.0,1.0])
    parser.add_argument("--robust-qaoa", action="store_true")
    parser.add_argument("--qaoa-restarts",type=int,default=4)
    parser.add_argument("--qaoa-objective",choices=("mean","cvar"),default="cvar")
    parser.add_argument("--cvar-alpha",type=float,default=.1)
    parser.add_argument("--parameter-scale",choices=("max_coefficient","feasible_iqr"),default="max_coefficient")
    parser.add_argument("--exclude-pdb",nargs="+",default=[],metavar="PDB_ID",
        help="PDB IDs (case-insensitive) never eligible for selection by this invocation -- used to keep a "
             "frozen validation queue disjoint from the historical development queue (e.g. 4s10 8yvo 9gcn).")
    parser.add_argument("--exclude-pdb-file",type=Path,default=None,
        help="Optional file with one PDB ID per line, merged with --exclude-pdb.")
    parser.add_argument("--dev-exposed-pdb",nargs="+",default=[],metavar="PDB_ID",
        help="PDB IDs (case-insensitive) known to have been used/inspected during development "
             "(e.g. the historical dev queue). Recorded on every candidate's independence audit as "
             "development_exposed, DISTINCT from --exclude-pdb: exposure is tracked even for "
             "candidates that are not excluded -- exposure and the configured identity-threshold screening are different, "
             "both-necessary checks, and neither implies the other.")
    parser.add_argument("--dev-exposed-pdb-file",type=Path,default=None,
        help="Optional file with one PDB ID per line, merged with --dev-exposed-pdb.")
    parser.add_argument("--master-seed",type=int,default=DEFAULT_MASTER_SEED,
        help="Derives independent, saved optimize/sample sub-seeds (see seed_streams.py) for the "
             "structural recovery experiment of every selected target -- NEVER the same as --seeds, "
             "which are the shared chi1-perturbation repeat identities.")
    parser.add_argument("--selection-order",choices=("ascending_size","seeded_random"),default="ascending_size",
        help="'ascending_size' (default; original behaviour, smallest structure first) or 'seeded_random' "
             "(shuffled by --selection-seed) -- use seeded_random for a NEW validation queue so it does not "
             "inherit the ascending-size bias of the original development pilot.")
    parser.add_argument("--selection-seed",type=int,default=42,
        help="Shuffle seed used only when --selection-order seeded_random.")
    parser.add_argument("--queue-role",choices=("dev","validation"),default="dev",
        help="Purely descriptive tag recorded in provenance/eligibility.json/the report, so a validation-queue "
             "output directory can never be silently confused with a development pilot's.")
    args = parser.parse_args(argv)
    if not 1 <= args.sites <= 10 or args.targets < 0 or args.qaoa_depth <= 0:
        parser.error("Require 1..10 sites, positive qaoa-depth, and a nonnegative target count (0 = unlimited)")
    if args.eligibility_only and not args.prepare_only:
        parser.error("--eligibility-only is valid only together with --prepare-only")
    if not 0.0 < args.min_perturb_degrees <= args.max_perturb_degrees <= 180.0:
        parser.error("Require 0 < min-perturb-degrees <= max-perturb-degrees <= 180")
    homology = {
        "vhh_full_chain_identity": float(args.vhh_identity_threshold),
        "cdr_h3_identity": float(args.cdr_h3_identity_threshold),
        "antigen_identity": float(args.antigen_identity_threshold),
        "antigen_min_length_coverage": float(args.antigen_min_length_coverage),
    }
    if any(not 0.0 < value <= 1.0 for value in homology.values()):
        parser.error("homology thresholds/coverage must lie in (0,1]")
    from batch_benchmark_hard_set import _ablation_atomic_json, _ablation_digest, _recovery_benchmark_main
    from filelock import FileLock
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = args.dataset / "graph_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    training = [r for r in rows if r["split"] == "train"]
    candidates = sorted([r for r in rows if r["split"] == "test_snac_hard"], key=lambda r: (int(r["nodes"]),r["pdb_id"],r["path"]))
    frozen_target_metadata={}
    frozen_target_ids=set()
    frozen_identity_missing=[]
    formal_frozen_execution=bool(
        args.pdb_allowlist_file and args.queue_role=="validation" and not args.prepare_only
    )
    if args.pdb_allowlist_file:
        raw_allowlist = json.loads(args.pdb_allowlist_file.read_text(encoding="utf-8"))
        if not isinstance(raw_allowlist,list) or not raw_allowlist:
            raise ValueError("Frozen target allowlist must be a nonempty JSON list")
        for index,entry in enumerate(raw_allowlist):
            if isinstance(entry,str):
                pdb=entry.strip().lower()
                metadata={}
            elif isinstance(entry,dict):
                pdb=str(entry.get("target") or entry.get("pdb_id") or "").strip().lower()
                metadata=dict(entry)
            else:
                raise ValueError("Allowlist entries must be PDB strings or dicts")
            if not pdb:
                raise ValueError(f"Allowlist entry {index} lacks target/pdb_id")
            if pdb in frozen_target_ids:
                raise ValueError(f"Duplicate frozen target PDB ID: {pdb}")
            frozen_target_ids.add(pdb)
            frozen_target_metadata[pdb]=metadata

        if formal_frozen_execution:
            required_identity=("source_id","graph_path","graph_sha256")
            malformed=[]
            for pdb in sorted(frozen_target_ids):
                metadata=frozen_target_metadata[pdb]
                missing=[
                    key for key in required_identity
                    if not isinstance(metadata.get(key),str) or not metadata.get(key).strip()
                ]
                if not metadata.get("eligibility_compatible_residues"):
                    missing.append("eligibility_compatible_residues")
                if missing:
                    malformed.append((pdb,missing))
            if malformed:
                raise ValueError(
                    "Formal validation freeze lacks exact graph identity/compatible-residue fields: "
                    + "; ".join(f"{pdb}:{','.join(fields)}" for pdb,fields in malformed[:10])
                )

            matched=[]
            matched_ids=set()
            for row in candidates:
                pdb=row["pdb_id"].lower()
                frozen=frozen_target_metadata.get(pdb)
                if frozen is None:
                    continue
                row_path=row["path"].replace("\\","/")
                if (
                    row.get("source_id")==frozen["source_id"]
                    and row_path==str(frozen["graph_path"]).replace("\\","/")
                    and row.get("sha256","").lower()==str(frozen["graph_sha256"]).lower()
                ):
                    matched.append(row)
                    matched_ids.add(pdb)
            frozen_identity_missing=sorted(frozen_target_ids-matched_ids)
            candidates=matched
        else:
            candidates=[r for r in candidates if r["pdb_id"].lower() in frozen_target_ids]
    excluded_pdb = {x.lower() for x in args.exclude_pdb}
    if args.exclude_pdb_file:
        excluded_pdb |= {line.strip().lower() for line in args.exclude_pdb_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    dev_exposed_pdb = {x.lower() for x in args.dev_exposed_pdb}
    if args.dev_exposed_pdb_file:
        dev_exposed_pdb |= {line.strip().lower() for line in args.dev_exposed_pdb_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    if args.selection_order == "seeded_random":
        random.Random(args.selection_seed).shuffle(candidates)
    cluster_map=None
    if args.cluster_map is not None:
        raw_clusters=json.loads(args.cluster_map.read_text(encoding="utf-8"))
        if not isinstance(raw_clusters,dict) or not raw_clusters:
            raise ValueError("--cluster-map must contain a nonempty JSON object")
        cluster_map={str(k).lower():str(v) for k,v in raw_clusters.items()}
    streams = derive_streams(args.master_seed)
    # Target filtering is explicit only. A one-target run must never
    # silently become the historical 4S10 smoke case merely because eval_shots is set.
    requested_pdb=args.pdb_id
    if requested_pdb:
        candidates=[r for r in candidates if r['pdb_id'].lower()==requested_pdb.lower()]
        if not candidates: raise ValueError(f'Target {requested_pdb} absent from manifest')
        print(f'Explicit target filter: {requested_pdb}',flush=True)
    # (Version Discrepancy remediation) evaluate_complex_metrics.py is now a
    # transitive dependency of subgraph_to_qubo.py's evaluate_atomistic_prediction
    # (the single source of truth for Fnat/iRMSD/LRMSD/DockQ/severe-clash
    # metrics), so this root-directory pilot's own provenance manifest tracks
    # it exactly like every other module already fingerprinted here.
    provenance = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        graph_manifest_sha256=_ablation_digest(manifest), code_sha256={n:_ablation_digest(Path(n)) for n in
        ("run_real_complex_pilot.py","batch_benchmark_hard_set.py","subgraph_to_qubo.py","qaoa_interface_sampler.py",
         "structural_quality.py","prediction_contract.py","evaluate_complex_metrics.py")},
        checkpoint_sha256=_ablation_digest(args.checkpoint) if args.pruning=='egnn' else None,
        pdb_allowlist_sha256=(
            _ablation_digest(args.pdb_allowlist_file) if args.pdb_allowlist_file else None),
        resolved_target_filter=requested_pdb)
    with FileLock(str(out/".lock"),timeout=0):
        stamp=out/"run_manifest.json"
        if stamp.exists() and json.loads(stamp.read_text()) != provenance:
            raise ValueError("Protocol changed; use a new output directory")
        _ablation_atomic_json(stamp,provenance)
        save_stream_map(out/"seed_streams.json", args.master_seed, streams)
        cache=out/"training_sequences.json"
        if cache.exists():
            sequence_inventory=json.loads(cache.read_text())
        else:
            sequence_inventory={"vhh": {}, "antigen": {}}
            for row in tqdm(training,desc="Training sequence inventory"):
                path=args.dataset/Path(row["path"].replace("\\","/"))
                if _ablation_digest(path)!=row["sha256"]:
                    raise ValueError(f"Training graph hash mismatch {path}")
                data=torch.load(path,map_location="cpu",weights_only=False)
                for seq in getattr(data,"vhh_sequences",[]):
                    sequence_inventory["vhh"].setdefault(seq,[]).append(row["pdb_id"])
                for seq in getattr(data,"antigen_sequences",[]):
                    sequence_inventory["antigen"].setdefault(seq,[]).append(row["pdb_id"])
            _ablation_atomic_json(cache,sequence_inventory)
        train_pdb={r["pdb_id"].lower() for r in training}
        train_cdr=sorted({r["cdr3_seq"] for r in training if r["cdr3_seq"]})
        train_clusters=set()
        if cluster_map is not None:
            missing=[p for p in sorted(train_pdb) if p not in cluster_map]
            if missing:
                raise ValueError(f"Cluster map missing training PDBs, e.g. {missing[:10]}")
            train_clusters={cluster_map[p] for p in train_pdb}
        selected=[]; decisions=[]; selected_vhh=[]; selected_antigen=[]; selected_cdr=[]; selected_clusters=set(); seen=set()
        for pdb in frozen_identity_missing:
            decisions.append(dict(
                pdb_id=pdb,status="excluded",
                reason="Frozen graph identity absent or mismatched in current test_snac_hard manifest",
                development_exposed=pdb in dev_exposed_pdb,
                independence_status="frozen_graph_identity_mismatch",
            ))
        for row in tqdm(candidates,desc="Real complex eligibility"):
            if args.targets and len(selected)>=args.targets: break
            pdb=row["pdb_id"].lower()
            if pdb in seen: continue
            seen.add(pdb)
            if pdb in excluded_pdb:
                decisions.append(dict(pdb_id=pdb,status="excluded",
                    reason=f"explicitly excluded from the {args.queue_role} queue (--exclude-pdb/--exclude-pdb-file)",
                    development_exposed=pdb in dev_exposed_pdb,independence_status="excluded_pdb_overlap_explicit"))
                _ablation_atomic_json(out/"eligibility.json",decisions)
                continue
            work=out/"prepared"/pdb;work.mkdir(parents=True,exist_ok=True)
            chain_identity_audit=[];cdr3_identity=None
            independence_status=("sequence_and_family_structure_isolated" if cluster_map is not None
                                 else "sequence_isolated_family_structure_unconfirmed")
            development_exposed=pdb in dev_exposed_pdb
            try:
                if pdb in train_pdb: raise ValueError("Training PDB overlap")
                family_cluster=None
                if cluster_map is not None:
                    if pdb not in cluster_map:
                        independence_status="excluded_missing_family_cluster"
                        raise ValueError(f"Cluster map missing candidate PDB {pdb}")
                    family_cluster=cluster_map[pdb]
                    if family_cluster in train_clusters or family_cluster in selected_clusters:
                        independence_status="excluded_family_cluster_overlap"
                        raise ValueError(f"Family/structure cluster overlap: {family_cluster}")
                path=args.dataset/Path(row["path"].replace("\\","/"))
                if _ablation_digest(path)!=row["sha256"]: raise ValueError("Test graph hash mismatch")
                graph=torch.load(path,map_location="cpu",weights_only=False)
                raw=extract_source(graph.source_id,args.data_root,work/"raw.pdb")
                frozen_pool=(
                    set(frozen_target_metadata.get(pdb,{}).get("eligibility_compatible_residues",[]))
                    if args.pdb_allowlist_file and not args.prepare_only else None
                )
                config=prepare(graph,raw,work/"native.cif",args.sites,pruning=args.pruning,
                    seed=args.seeds[0],checkpoint=args.checkpoint,
                    antigen_guidance_weight=args.antigen_guidance_weight,
                    antigen_proximity_scale=args.antigen_proximity_scale,
                    contact_ca_cutoff=args.contact_ca_cutoff,
                    homology_isolation=homology,
                    eligibility_only=args.eligibility_only,
                    rotamer_mode=args.rotamer_mode,
                    allowed_residues=frozen_pool)
                chain_identity_audit=[]
                max_vhh_identity=0.0
                max_antigen_identity=0.0
                for chain_id,seq in zip(graph.chain_ids,graph.chain_sequences):
                    role="vhh" if dict(zip(graph.chain_ids,graph.chain_groups)).get(chain_id)==0 else "antigen"
                    if role=="vhh":
                        pool=list(sequence_inventory["vhh"])+selected_vhh
                        threshold=args.vhh_identity_threshold
                        coverage_gate=0.0
                    else:
                        pool=list(sequence_inventory["antigen"])+selected_antigen
                        threshold=args.antigen_identity_threshold
                        coverage_gate=args.antigen_min_length_coverage
                    best=dict(identity=0.,coverage=0.,length_gated=False)
                    for other in pool:
                        detail=identity_detail(seq,other,coverage_gate)
                        if detail["identity"]>best["identity"]:
                            best=detail
                    chain_identity_audit.append(dict(
                        chain_id=chain_id,role=role,length=len(seq),
                        best_identity=best["identity"],best_coverage=best["coverage"],
                        threshold=float(threshold),min_length_coverage=float(coverage_gate)))
                    if role=="vhh":
                        max_vhh_identity=max(max_vhh_identity,best["identity"])
                    else:
                        max_antigen_identity=max(max_antigen_identity,best["identity"])

                cdr3_identity=max(
                    (identity_detail(graph.cdr3_seq,c,0.0)["identity"] for c in (train_cdr+selected_cdr)),
                    default=0.0,
                )
                if max_vhh_identity>=args.vhh_identity_threshold:
                    independence_status="excluded_high_vhh_identity"
                    raise ValueError(
                        f"VHH full-chain identity {max_vhh_identity:.3f} >= {args.vhh_identity_threshold:.2f}"
                    )
                if cdr3_identity>=args.cdr_h3_identity_threshold:
                    independence_status="excluded_high_cdr_h3_identity"
                    raise ValueError(
                        f"CDR-H3 identity {cdr3_identity:.3f} >= {args.cdr_h3_identity_threshold:.2f}"
                    )
                if max_antigen_identity>=args.antigen_identity_threshold:
                    independence_status="excluded_high_antigen_identity"
                    raise ValueError(
                        f"Antigen identity {max_antigen_identity:.3f} >= {args.antigen_identity_threshold:.2f} "
                        f"with minimum length coverage {args.antigen_min_length_coverage:.2f}"
                    )
                config["preparation_changes"].extend(complete_terminal_oxygen(work/"native.cif"))
                if args.eligibility_only:
                    # Prove existence of K compatible sites without ranking them.
                    compatible=[]
                    compatibility_failures=[]
                    for rid in config.get("eligible_residues",[]):
                        try:
                            check=AllAtomInterfaceQUBOBuilder(
                                work/"native.cif",[rid],site_scores=[0.0],
                                rotamer_mode=args.rotamer_mode,
                                rotamer_library_path=args.rotamer_library,
                                rotamer_probability_floor=args.rotamer_probability_floor,
                                rotamer_sigma_offsets=args.rotamer_sigma_offsets,
                                solvent_model=args.solvent_model,
                            )
                            del check
                            compatible.append(rid)
                        except Exception as site_exc:
                            compatibility_failures.append(dict(
                                residue_id=rid,
                                error=f"{type(site_exc).__name__}: {site_exc}",
                            ))
                    if len(compatible) < args.sites:
                        raise ValueError(
                            f"Only {len(compatible)} independently Dunbrack/Amber-compatible VHH sites; "
                            f"need {args.sites}"
                        )
                    config["eligibility_compatible_residues"]=compatible
                    config["eligibility_site_failures"]=compatibility_failures
                    config["active_residues"]=[]
                    config["active_site_scores"]=[]
                else:
                    # Formal run: verify the actually selected Active set.
                    check=AllAtomInterfaceQUBOBuilder(
                        work/"native.cif",config["active_residues"],
                        site_scores=config.get("active_site_scores"),
                        rotamer_mode=args.rotamer_mode,
                        rotamer_library_path=args.rotamer_library,
                        rotamer_probability_floor=args.rotamer_probability_floor,
                        rotamer_sigma_offsets=args.rotamer_sigma_offsets,
                        solvent_model=args.solvent_model,
                    )
                    del check
                config.update(target=pdb,native_structure=str(work/"native.cif"),
                    candidate_relax_iterations=args.candidate_relax_iterations,
                    protocol="validation_control",graph_sha256=row["sha256"],
                    graph_path=row["path"].replace("\\","/"),source_id=graph.source_id,
                    raw_sha256=_ablation_digest(raw),native_sha256=_ablation_digest(work/"native.cif"),
                    development_exposed=development_exposed,chain_identity_audit=chain_identity_audit,
                    cdr3_identity=cdr3_identity,max_vhh_identity=max_vhh_identity,
                    max_antigen_identity=max_antigen_identity,homology_isolation=homology,
                    family_structure_cluster=family_cluster,
                    frozen_compatible_residue_pool=(
                        sorted(frozen_pool) if frozen_pool is not None else None
                    ),
                    cluster_map_sha256=(_ablation_digest(args.cluster_map) if args.cluster_map else None),
                    solvent_model=args.solvent_model,
                    rotamer_model=dict(
                        mode=args.rotamer_mode,
                        library_path=(None if args.rotamer_library is None else str(args.rotamer_library)),
                        probability_floor=float(args.rotamer_probability_floor),
                        sigma_offsets=[float(v) for v in args.rotamer_sigma_offsets],
                    ),
                    independence_status=independence_status,
                    independence=(
                        f"{independence_status}: PDB-disjoint and below layered homology thresholds "
                        f"VHH<{args.vhh_identity_threshold:.2f}, CDR-H3<{args.cdr_h3_identity_threshold:.2f}, "
                        f"antigen<{args.antigen_identity_threshold:.2f} with coverage>={args.antigen_min_length_coverage:.2f}; "
                        f"observed max VHH={max_vhh_identity:.4f}, CDR-H3={cdr3_identity:.4f}, "
                        f"antigen={max_antigen_identity:.4f}; "
                        f"family/structure cluster={family_cluster}; cluster-map checked={cluster_map is not None}; "
                        f"development_exposed={development_exposed}.") )
                _ablation_atomic_json(work/"recovery_manifest.json",config)
                selected.append(config)
                if family_cluster is not None:
                    selected_clusters.add(family_cluster)
                selected_vhh.extend(getattr(graph,"vhh_sequences",[]))
                selected_antigen.extend(getattr(graph,"antigen_sequences",[]))
                if getattr(graph,"cdr3_seq",""):
                    selected_cdr.append(graph.cdr3_seq)
                decisions.append(dict(pdb_id=pdb,status="selected",reason="",
                    development_exposed=development_exposed,independence_status=independence_status,
                    max_vhh_identity=max_vhh_identity,max_antigen_identity=max_antigen_identity,
                    cdr3_identity=cdr3_identity,homology_isolation=homology,
                    chain_identity_audit=chain_identity_audit))
            except Exception as exc:
                decisions.append(dict(pdb_id=pdb,status="excluded",reason=str(exc),
                    development_exposed=development_exposed,independence_status=independence_status,
                    chain_identity_audit=chain_identity_audit,cdr3_identity=cdr3_identity))
            _ablation_atomic_json(out/"eligibility.json",decisions)
        _ablation_atomic_json(out/"selected_targets.json",selected)
        if not selected:
            raise ValueError('No eligible targets; inspect eligibility.json')
        results=[];failed=[]
        if not args.prepare_only:
            for case in selected:
                pdb=case["target"]
                try:
                    destination=out/"results"/pdb
                    # Independent, saved, per-target optimize/sample sub-seeds
                    # (never equal to --seeds, and never each other) -- derived
                    # from the master-seed optimize/sample streams keyed by
                    # (target pdb, repeat identity), so this target's structural
                    # search and its finite-shot output sampling never share
                    # randomness, and the derivation is reproducible from
                    # (master_seed, pdb, seed) alone regardless of scheduling order.
                    optimize_seeds=[derive_child_seed(streams["optimize"],"structure",pdb,str(s)) for s in args.seeds]
                    measurement_seeds=[derive_child_seed(streams["measurement"],"structure",pdb,str(s)) for s in args.seeds]
                    sample_seeds=[derive_child_seed(streams["sample"],"structure",pdb,str(s)) for s in args.seeds]
                    status=_recovery_benchmark_main(["--manifest",str(out/"prepared"/pdb/"recovery_manifest.json"),
                        "--out-dir",str(destination),"--solvent-model",args.solvent_model,
                        "--perturbation-mode",args.perturbation_mode,
                        "--min-perturb-degrees",str(args.min_perturb_degrees),
                        "--max-perturb-degrees",str(args.max_perturb_degrees),
                        "--seeds",*[str(s) for s in args.seeds],
                        "--perturbation-mode",args.perturbation_mode,
                        "--optimize-seeds",*[str(s) for s in optimize_seeds],
                        "--measurement-seeds",*[str(s) for s in measurement_seeds],
                        "--sample-seeds",*[str(s) for s in sample_seeds],
                        "--outputs",str(args.outputs),"--max-evals",str(args.max_evals),
                        "--qaoa-depth",str(args.qaoa_depth),
                        "--relax-iterations",str(args.relax_iterations),
                        "--loop-relax-iterations",str(args.loop_relax_iterations),
                        *(["--eval-shots",str(args.eval_shots)] if args.eval_shots else []),
                        *(["--robust-qaoa","--qaoa-restarts",str(args.qaoa_restarts),
                           "--qaoa-objective",args.qaoa_objective,"--cvar-alpha",str(args.cvar_alpha),
                           "--parameter-scale",args.parameter_scale] if args.robust_qaoa else [])])
                    results.extend(csv.DictReader((destination/"recovery_metrics.csv").open(encoding="utf-8")))
                    if status: failed.append(pdb)
                except Exception:
                    failed.append(pdb)
                    with (out/"failures.log").open("a",encoding="utf-8") as f: f.write(pdb+"\n"+traceback.format_exc())
        if results:
            with (out/"real_complex_metrics.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        # Closure bookkeeping: formal validation closes against the IMMUTABLE
        # frozen denominator, not merely the subset that re-passed eligibility.
        # A frozen graph mismatch, re-preparation/eligibility failure, Active-set
        # construction failure, or solver failure therefore remains a failed
        # confirmatory target and can never silently disappear or be replaced.
        selected_ids={case["target"] for case in selected}
        solver_failed=set(failed)
        if formal_frozen_execution:
            frozen_failed=set(frozen_target_ids)-selected_ids
            frozen_failed.update(solver_failed)
            completed_ids=selected_ids-solver_failed
            failed_ids=frozen_failed
            frozen_set_accounting_ok=(
                completed_ids.isdisjoint(failed_ids)
                and completed_ids|failed_ids==set(frozen_target_ids)
            )
            closed=frozen_set_accounting_ok
        else:
            completed_ids=selected_ids-solver_failed
            failed_ids=solver_failed
            frozen_set_accounting_ok=None
            closed=(len(completed_ids)+len(failed_ids)==len(selected_ids))
        completed_targets=len(completed_ids)
        _ablation_atomic_json(out/"run_summary.json", dict(
            examined_candidates=len(decisions), qualifying_pool_size=len(candidates),
            target_cap=args.targets or None, selected_targets=len(selected),
            frozen_target_ids=(sorted(frozen_target_ids) if formal_frozen_execution else None),
            frozen_target_count=(len(frozen_target_ids) if formal_frozen_execution else None),
            structure_experiment_completed_targets=completed_targets,
            structure_experiment_completed_target_ids=sorted(completed_ids),
            structure_experiment_failed_targets=sorted(failed_ids),
            frozen_set_accounting_ok=frozen_set_accounting_ok,
            closed=closed))
        coverage_pct = (len(selected)/len(decisions)*100) if decisions else 0.0
        report=["# Real VHH retrospective side-chain recovery pilot", "",
            f"Queue role: {args.queue_role}. Selection order: {args.selection_order}{' (seed '+str(args.selection_seed)+')' if args.selection_order=='seeded_random' else ''}. "
            f"Excluded PDBs: {sorted(excluded_pdb) or 'none'}.",
            f"Target cap: {args.targets or 'unlimited (all qualifying targets)'}; qualifying candidate pool: {len(candidates)}; "
            f"examined {len(decisions)}; selected {len(selected)} (coverage of examined pool: {coverage_pct:.1f}%); "
            f"structural experiment completed {completed_targets}, failed {sorted(failed)}.",
            f"Input is native backbone/pose plus perturbed Active side chains (mode={args.perturbation_mode}). Formal EGNN Active-site selection ranks all chemically movable VHH residues with the shared antigen-guided composite score (EGNN probability plus nearest-antigen proximity); native heavy-atom <8A is no longer an oracle eligibility gate. Contact/distance/CDR/random remain explicit ablation baselines. This is a retrospective native-backbone-conditioned recovery task, not blind docking or CDR-H3 backbone prediction.",
            f"All-atom energy model: {args.solvent_model}. Formal rotamer model: {args.rotamer_mode}. In Dunbrack mode, complete backbone-dependent rotamer chi1..chiN states are constructed at each residue's phi/psi bin; chi1 is expanded by configured sigma offsets and Amber14 single-candidate energies pre-screen 3-6 retained states/site under <=30 variables. Formal multi_chi recovery perturbs all defined Active side-chain chis; chi1-only remains an explicit ablation. Backbone and background remain frozen.",
            "All methods share input/candidates, read budget and relaxation. CPU cost is not equal. Reference structure evaluates accuracy but never selects solver output.",
            f"Independence uses PDB-disjointness, layered sequence screening (VHH {args.vhh_identity_threshold*100:.0f}%, CDR-H3 {args.cdr_h3_identity_threshold*100:.0f}%, antigen {args.antigen_identity_threshold*100:.0f}% with minimum length coverage {args.antigen_min_length_coverage*100:.0f}%) and the supplied family/structure cluster map when present (details in eligibility.json). Development exposure is recorded separately. This remains a retrospective recovery benchmark.",
            "", "| Method | Targets with results | Mean RMSD gain vs input (A) | Mean gain vs relax-only (A) |", "|---|---:|---:|---:|"]
        for method in ("qaoa","sa","uniform","greedy"):
            group=[r for r in results if r["method"]==method]
            targets=sorted({r["target"] for r in group})
            if targets:
                gains=[np.mean([float(r["improvement_vs_input"]) for r in group if r["target"]==t]) for t in targets]
                controls=[np.mean([float(r["improvement_vs_relax_only"]) for r in group if r["target"]==t]) for t in targets]
                report.append(f"| {method} | {len(targets)} | {np.mean(gains):.6g} | {np.mean(controls):.6g} |")
        report += ["", "Eligibility and every exclusion reason are in eligibility.json; no replacement based on solver results. Failed/incomplete targets remain listed. Target means weight seeds within each target first. No significance or quantum advantage is inferred from a small pilot."]
        (out/"real_complex_report.md").write_text("\n".join(report),encoding="utf-8")
        print(out/"real_complex_report.md")
        if formal_frozen_execution:
            return int(bool(failed_ids) or not frozen_set_accounting_ok)
        return int(bool(failed) or len(selected)<args.targets)


if __name__=="__main__":
    raise SystemExit(main())
