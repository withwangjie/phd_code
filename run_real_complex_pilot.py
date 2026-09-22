"""Real VHH chi1 recovery pilot with explicit data separation and exclusions.

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


def _load_frozen_target_ids(path: Path) -> list[str]:
    """Load an immutable frozen-target allowlist with strict validation.

    The file may be a JSON list of PDB-id strings or the earlier
    selected_targets.json list of dictionaries carrying target/pdb_id.
    IDs are normalized to lowercase, but duplicates, empty IDs and malformed
    entries fail closed rather than silently shrinking the formal denominator.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Frozen target file must contain a nonempty JSON list")
    target_ids: list[str] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            value = entry
        elif isinstance(entry, dict):
            value = entry.get("target") or entry.get("pdb_id")
        else:
            raise ValueError(f"Frozen target entry {index} must be a string or object")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Frozen target entry {index} has no target/pdb_id")
        target_ids.append(value.strip().lower())
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Frozen target file contains duplicate PDB IDs")
    return target_ids


def _reconcile_frozen_targets(
    frozen_target_ids: list[str],
    execution_selected_ids: set[str],
    runtime_failed_ids: set[str],
) -> tuple[list[str], list[str], bool]:
    """Account for every frozen target exactly once as completed or failed.

    Any frozen target that disappears during formal re-preparation is a
    pre-execution failure, never a silent denominator reduction. Targets not
    present in the frozen set fail closed.
    """
    frozen = set(frozen_target_ids)
    selected = {str(x).lower() for x in execution_selected_ids}
    runtime_failed = {str(x).lower() for x in runtime_failed_ids}
    if not selected <= frozen:
        raise ValueError(f"Execution selected targets outside frozen set: {sorted(selected - frozen)}")
    if not runtime_failed <= selected:
        raise ValueError(f"Runtime failures outside execution-selected set: {sorted(runtime_failed - selected)}")
    pre_execution_failed = frozen - selected
    failed = pre_execution_failed | runtime_failed
    completed = selected - runtime_failed
    closed = (completed | failed) == frozen and not (completed & failed)
    return sorted(completed), sorted(failed), closed


def identity(a: str, b: str, threshold: float = .8) -> float:
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


def identity_detail(a: str, b: str) -> dict:
    """Like `identity`, but always returns the actual identity value and the
    length-ratio coverage gate -- for an auditable record, never just a
    boolean pass/fail. Uses the same alignment (parasail NW + BLOSUM62) and
    the same 80% length-ratio coverage gate as `identity`, just without
    collapsing the result to a threshold comparison."""
    coverage = min(len(a), len(b)) / max(len(a), len(b))
    if coverage < .8:
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
            seed: int = 42, checkpoint: Path | None = None) -> dict:
    """Resolve coherent altlocs and remove waters only; reject missing protein atoms."""
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
    partner_atoms = np.concatenate([list(residues[r]["atoms"].values()) for r in partners])
    scored = []
    for i in chain_nodes:
        rid=graph.residue_ids[i]
        entry = residues[rid]
        # Exclude cysteine from movable sites to avoid breaking disulfides.
        if entry["name"] in ("ALA", "GLY", "PRO", "CYS"):
            continue
        xyz = np.array(list(entry["atoms"].values()))
        dist = np.sqrt(np.min(np.sum((xyz[:, None]-partner_atoms)**2, axis=-1)))
        if dist < 8.:
            contact_count=int(np.sum(np.sum((xyz[:,None]-partner_atoms)**2,axis=-1)<25.))
            scored.append((float(dist), rid, contact_count, i))
    if len(scored) < sites:
        raise ValueError(f"Only {len(scored)} eligible VHH interface-neighborhood sites")
    model_status=None
    if pruning=='contact':
        ordered=sorted(scored,key=lambda x:(-x[2],x[0],x[1]))
    elif pruning=='cdr':
        ordered=sorted(scored,key=lambda x:(x[1] not in cdr_ids,-x[2],x[1]))
    elif pruning=='random':
        ordered=[scored[i] for i in np.random.default_rng(seed).permutation(len(scored))]
    elif pruning=='egnn':
        from batch_benchmark_hard_set import load_interface_scorer
        model,info=load_interface_scorer(checkpoint,torch_device=torch.device('cpu'),seed=seed)
        model_status=info.status
        if model_status!='checkpoint_loaded':
            raise ValueError('EGNN ablation requires a valid trained checkpoint')
        with torch.no_grad(): scores=model(graph.x,graph.pos,graph.edge_index).reshape(-1).numpy()
        ordered=sorted(scored,key=lambda x:(-float(scores[x[3]]),x[1]))
    else: raise ValueError('Unknown pruning strategy')
    active = [item[1] for item in ordered[:sites]]
    st.make_mmcif_document().write_file(str(destination))
    return dict(active_residues=active, alignment_residues=partners, partner_residues=partners,
                preparation_changes=changes, cdr3_residues=cdr_ids,pruning=pruning,
                pruning_candidate_count=len(scored),pruning_seed=seed,model_status=model_status,
                selection_origin=f"{pruning} selection from native-pose VHH heavy-atom <8A interface neighborhood, fixed before perturbation; retrospective control")


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
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--outputs", type=int, default=1000)
    parser.add_argument("--max-evals", type=int, default=90)
    parser.add_argument("--relax-iterations", type=int, default=200)
    parser.add_argument("--candidate-relax-iterations", type=int, default=100)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-shots",type=int,choices=(200,500,1000))
    parser.add_argument("--loop-relax-iterations",type=int,default=100)
    parser.add_argument("--pruning",choices=('egnn','contact','cdr','random'),default='egnn')
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
             "candidates that are not excluded -- exposure and 80%% identity screening are different, "
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
    if not 1 <= args.sites <= 10 or args.targets < 0:
        parser.error("Require 1..10 sites and a nonnegative target count (0 = unlimited)")
    from batch_benchmark_hard_set import _ablation_atomic_json, _ablation_digest, _recovery_benchmark_main
    from filelock import FileLock
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    manifest = args.dataset / "graph_manifest.csv"
    rows = list(csv.DictReader(manifest.open(encoding="utf-8-sig")))
    training = [r for r in rows if r["split"] == "train"]
    candidates = sorted([r for r in rows if r["split"] == "test_snac_hard"], key=lambda r: (int(r["nodes"]),r["pdb_id"],r["path"]))
    frozen_target_ids = None
    frozen_target_set = None
    missing_frozen_from_manifest: list[str] = []
    if args.pdb_allowlist_file:
        frozen_target_ids = _load_frozen_target_ids(args.pdb_allowlist_file)
        frozen_target_set = set(frozen_target_ids)
        manifest_candidate_ids = {r["pdb_id"].lower() for r in candidates}
        missing_frozen_from_manifest = sorted(frozen_target_set - manifest_candidate_ids)
        candidates = [r for r in candidates if r["pdb_id"].lower() in frozen_target_set]
    excluded_pdb = {x.lower() for x in args.exclude_pdb}
    if args.exclude_pdb_file:
        excluded_pdb |= {line.strip().lower() for line in args.exclude_pdb_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    dev_exposed_pdb = {x.lower() for x in args.dev_exposed_pdb}
    if args.dev_exposed_pdb_file:
        dev_exposed_pdb |= {line.strip().lower() for line in args.dev_exposed_pdb_file.read_text(encoding="utf-8").splitlines() if line.strip()}
    if args.selection_order == "seeded_random":
        random.Random(args.selection_seed).shuffle(candidates)
    streams = derive_streams(args.master_seed)
    # Explicit requested integration smoke uses 4S10, not the first small graph.
    requested_pdb=args.pdb_id or ('4s10' if args.eval_shots and args.targets==1 else None)
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
        pdb_allowlist_sha256=_ablation_digest(args.pdb_allowlist_file) if args.pdb_allowlist_file else None,
        resolved_target_filter=requested_pdb)
    with FileLock(str(out/".lock"),timeout=0):
        stamp=out/"run_manifest.json"
        if stamp.exists() and json.loads(stamp.read_text()) != provenance:
            raise ValueError("Protocol changed; use a new output directory")
        _ablation_atomic_json(stamp,provenance)
        save_stream_map(out/"seed_streams.json", args.master_seed, streams)
        cache=out/"training_sequences.json"
        if cache.exists():
            sequences=json.loads(cache.read_text())
        else:
            sequences={}
            for row in tqdm(training,desc="Training sequence inventory"):
                path=args.dataset/Path(row["path"].replace("\\","/"))
                if _ablation_digest(path)!=row["sha256"]:
                    raise ValueError(f"Training graph hash mismatch {path}")
                data=torch.load(path,map_location="cpu",weights_only=False)
                for seq in data.chain_sequences:
                    sequences.setdefault(seq,[]).append(row["pdb_id"])
            _ablation_atomic_json(cache,sequences)
        train_pdb={r["pdb_id"].lower() for r in training}
        train_cdr=sorted({r["cdr3_seq"] for r in training if r["cdr3_seq"]})
        selected=[]; decisions=[]; test_seqs=[]; seen=set()
        for pdb in missing_frozen_from_manifest:
            decisions.append(dict(
                pdb_id=pdb, status="excluded",
                reason="Frozen target absent from current test_snac_hard manifest",
                development_exposed=pdb in dev_exposed_pdb,
                independence_status="frozen_target_missing_from_manifest",
            ))
        if decisions:
            _ablation_atomic_json(out/"eligibility.json", decisions)
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
            independence_status="independence_not_confirmed"
            development_exposed=pdb in dev_exposed_pdb
            try:
                if pdb in train_pdb: raise ValueError("Training PDB overlap")
                path=args.dataset/Path(row["path"].replace("\\","/"))
                if _ablation_digest(path)!=row["sha256"]: raise ValueError("Test graph hash mismatch")
                graph=torch.load(path,map_location="cpu",weights_only=False)
                raw=extract_source(graph.source_id,args.data_root,work/"raw.pdb")
                config=prepare(graph,raw,work/"native.cif",args.sites,pruning=args.pruning,
                    seed=args.seeds[0],checkpoint=args.checkpoint)
                # Every chain is audited and RECORDED (role + actual best
                # identity/coverage against every training/already-selected
                # sequence), never collapsed to only a pass/fail boolean --
                # so a reviewer can see near-misses, not only the final
                # exclude/pass decision. Exclusion threshold is unchanged
                # (>=80% full-chain global identity).
                groups=dict(zip(graph.chain_ids,graph.chain_groups))
                pool=list(sequences)+test_seqs
                max_chain_identity=0.
                for chain_id,seq in zip(graph.chain_ids,graph.chain_sequences):
                    role="vhh" if groups.get(chain_id)==0 else "partner_or_antigen"
                    best=dict(identity=0.,coverage=0.,length_gated=True)
                    for other in pool:
                        detail=identity_detail(seq,other)
                        if detail["identity"]>best["identity"]:
                            best=detail
                    chain_identity_audit.append(dict(chain_id=chain_id,role=role,length=len(seq),
                        best_identity=best["identity"],best_coverage=best["coverage"]))
                    max_chain_identity=max(max_chain_identity,best["identity"])
                cdr3_identity=max((identity_detail(graph.cdr3_seq,c)["identity"] for c in train_cdr),default=0.)
                # Family/local-domain-level relatedness (beyond raw sequence
                # identity) is NOT checked anywhere in this project -- no
                # PDB-family/cluster map exists (see batch_benchmark_hard_set.py
                # --paired-statistics --cluster-map, an unfilled optional
                # input there). So the ceiling for a target that clears every
                # identity/exposure check below is honestly
                # "independence_not_confirmed", never "confirmed independent";
                # only exclusion statuses are ever asserted with confidence.
                if max_chain_identity>=.8:
                    independence_status="excluded_high_chain_identity"
                    raise ValueError("Full-chain >=80% global identity overlap with training or selected target")
                if cdr3_identity>=.8:
                    independence_status="excluded_high_cdr_identity"
                    raise ValueError("Training annotated CDR-H3 overlap")
                config["preparation_changes"].extend(complete_terminal_oxygen(work/"native.cif"))
                # Confirm full Amber template compatibility before freezing membership.
                check=AllAtomInterfaceQUBOBuilder(work/"native.cif",config["active_residues"])
                del check
                config.update(target=pdb,native_structure=str(work/"native.cif"),
                    candidate_relax_iterations=args.candidate_relax_iterations,
                    protocol="validation_control",graph_sha256=row["sha256"],source_id=graph.source_id,
                    raw_sha256=_ablation_digest(raw),native_sha256=_ablation_digest(work/"native.cif"),
                    development_exposed=development_exposed,chain_identity_audit=chain_identity_audit,
                    cdr3_identity=cdr3_identity,independence_status=independence_status,
                    independence=f"independence_not_confirmed: PDB-disjoint and below the 80% full-chain/"
                        f"annotated-CDR3 global identity threshold against training+selected targets "
                        f"(max_chain_identity={max_chain_identity:.4f}, cdr3_identity={cdr3_identity:.4f}); "
                        f"family/local-domain relatedness is NOT checked (no cluster map exists in this "
                        f"project) and must not be read as confirmed independence; "
                        f"development_exposed={development_exposed}.")
                _ablation_atomic_json(work/"recovery_manifest.json",config)
                selected.append(config);test_seqs.extend(graph.chain_sequences)
                decisions.append(dict(pdb_id=pdb,status="selected",reason="",
                    development_exposed=development_exposed,independence_status=independence_status,
                    max_chain_identity=max_chain_identity,cdr3_identity=cdr3_identity,
                    chain_identity_audit=chain_identity_audit))
            except Exception as exc:
                decisions.append(dict(pdb_id=pdb,status="excluded",reason=str(exc),
                    development_exposed=development_exposed,independence_status=independence_status,
                    chain_identity_audit=chain_identity_audit,cdr3_identity=cdr3_identity))
            _ablation_atomic_json(out/"eligibility.json",decisions)
        _ablation_atomic_json(out/"selected_targets.json",selected)
        execution_selected_ids = {case["target"].lower() for case in selected}
        if frozen_target_set is not None and not execution_selected_ids <= frozen_target_set:
            raise ValueError("Formal execution selected a target outside the frozen target set")
        if not selected and frozen_target_set is None:
            raise ValueError('No eligible targets; inspect eligibility.json')
        results=[]; runtime_failed=[]
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
                        "--out-dir",str(destination),"--seeds",*[str(s) for s in args.seeds],
                        "--optimize-seeds",*[str(s) for s in optimize_seeds],
                        "--measurement-seeds",*[str(s) for s in measurement_seeds],
                        "--sample-seeds",*[str(s) for s in sample_seeds],
                        "--outputs",str(args.outputs),"--max-evals",str(args.max_evals),
                        "--relax-iterations",str(args.relax_iterations),
                        "--loop-relax-iterations",str(args.loop_relax_iterations),
                        *(["--eval-shots",str(args.eval_shots)] if args.eval_shots else []),
                        *(["--robust-qaoa","--qaoa-restarts",str(args.qaoa_restarts),
                           "--qaoa-objective",args.qaoa_objective,"--cvar-alpha",str(args.cvar_alpha),
                           "--parameter-scale",args.parameter_scale] if args.robust_qaoa else [])])
                    results.extend(csv.DictReader((destination/"recovery_metrics.csv").open(encoding="utf-8")))
                    if status: runtime_failed.append(pdb)
                except Exception:
                    runtime_failed.append(pdb)
                    with (out/"failures.log").open("a",encoding="utf-8") as f: f.write(pdb+"\n"+traceback.format_exc())
        if results:
            with (out/"real_complex_metrics.csv").open("w",newline="",encoding="utf-8") as f:
                writer=csv.DictWriter(f,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        # Closure bookkeeping (requirement #5): every selected target lands in
        # EITHER completed-with-usable-metrics OR the `failed` list above by
        # construction of the try/except loop -- this identity is recorded,
        # not assumed, so a process killed mid-loop (leaving some selected
        # targets neither attempted nor recorded) is visible as a MISSING
        # run_summary.json rather than a false "completed".
        runtime_failed_ids = {p.lower() for p in runtime_failed}
        if frozen_target_ids is not None:
            completed_target_ids, failed_target_ids, closed = _reconcile_frozen_targets(
                frozen_target_ids, execution_selected_ids, runtime_failed_ids)
            accounting_target_ids = sorted(frozen_target_set)
        else:
            failed_target_ids = sorted(runtime_failed_ids)
            completed_target_ids = sorted(execution_selected_ids - runtime_failed_ids)
            accounting_target_ids = sorted(execution_selected_ids)
            closed = (
                set(completed_target_ids) | set(failed_target_ids) == set(accounting_target_ids)
                and not (set(completed_target_ids) & set(failed_target_ids))
            )
        completed_targets=len(completed_target_ids)
        _ablation_atomic_json(out/"run_summary.json", dict(
            examined_candidates=len(decisions), qualifying_pool_size=len(candidates),
            target_cap=args.targets or None, selected_targets=len(selected),
            frozen_target_ids=sorted(frozen_target_set) if frozen_target_set is not None else None,
            frozen_target_count=len(frozen_target_set) if frozen_target_set is not None else None,
            execution_selected_target_ids=sorted(execution_selected_ids),
            structure_experiment_completed_targets=completed_targets,
            structure_experiment_completed_target_ids=completed_target_ids,
            structure_experiment_failed_targets=failed_target_ids,
            frozen_set_accounting_ok=closed if frozen_target_set is not None else None,
            closed=closed))
        coverage_pct = (len(selected)/len(decisions)*100) if decisions else 0.0
        report=["# Real VHH retrospective side-chain recovery pilot", "",
            f"Queue role: {args.queue_role}. Selection order: {args.selection_order}{' (seed '+str(args.selection_seed)+')' if args.selection_order=='seeded_random' else ''}. "
            f"Excluded PDBs: {sorted(excluded_pdb) or 'none'}.",
            f"Target cap: {args.targets or 'unlimited (all qualifying targets)'}; qualifying candidate pool: {len(candidates)}; "
            f"examined {len(decisions)}; selected {len(selected)} (coverage of examined pool: {coverage_pct:.1f}%); "
            f"structural experiment completed {completed_targets}, failed {failed_target_ids}.",
            "Input is native backbone/pose plus perturbed Active chi1. Active selection uses native interface contacts. This is an oracle-conditioned retrospective prediction task, not blind docking or CDR-H3 backbone prediction.",
            "Chi1 grids seed candidates. Candidate-local and final relaxation may move all atoms downstream of CA-CB; backbone and background remain frozen. Discrete optimization selects the prepared candidate combinations.",
            "All methods share input/candidates, read budget and relaxation. CPU cost is not equal. Reference structure evaluates accuracy but never selects solver output.",
            "Independence is limited to PDB plus 80% full-chain/annotated-CDR3 global identity screening (per-chain identity/coverage recorded in eligibility.json). Family/local-domain relatedness is NOT checked (no cluster map exists in this project): every non-excluded target's status is independence_not_confirmed, never confirmed-independent. Development exposure (--dev-exposed-pdb) is recorded per target, separately from the identity screen. This is an exploratory pilot, not a fresh confirmatory test.",
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
        return int(bool(failed_target_ids) or
                   (frozen_target_set is None and bool(args.targets) and len(selected)<args.targets))


if __name__=="__main__":
    raise SystemExit(main())
