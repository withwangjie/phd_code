#!/usr/bin/env python3
"""Generate an auditable independence manifest for an external VHH graph set.

The external set is compared against the *frozen training graphs* using the
same layered sequence thresholds as the formal study plus the same frozen
PDB->family/structure cluster map. No solver outputs or validation metrics are
read. The resulting JSON is suitable for run_full_experiment.py's external
validation gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gemmi
import torch
from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.data.safe_graph_load import load_graph
from nanoqc.data.sequence_identity import nw_identity, length_coverage, partner_orientations, partner_roles_anchored

AA_ORDER="ACDEFGHIKLMNPQRSTVWY"


def verified_graph_identity(graph: object, path: Path, *, require_cdr: bool = False) -> dict:
    """Cross-check sequence metadata against the graph's encoded residues."""
    pdb=str(getattr(graph,"pdb_id","")).strip().lower()
    if len(pdb)!=4 or not pdb.isalnum():
        raise ValueError(f"{path}: graph lacks a valid four-character PDB ID")
    x=getattr(graph,"x",None)
    chain_index=getattr(graph,"node_chain_id",None)
    chains=getattr(graph,"chain_sequences",None)
    groups=getattr(graph,"chain_groups",None)
    if (not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=21
            or not isinstance(chain_index,torch.Tensor) or chain_index.ndim!=1
            or len(chain_index)!=len(x) or not isinstance(chains,list)
            or not isinstance(groups,list) or len(chains)!=len(groups) or not chains):
        raise ValueError(f"{path}: graph lacks the required residue/chain identity fields (rebuild with graph v1.9)")
    if not torch.isfinite(x).all() or not torch.all((x[:,:20]==0)|(x[:,:20]==1)):
        raise ValueError(f"{path}: invalid amino-acid one-hot node features")
    if not torch.all(x[:,:20].sum(dim=1)==1):
        raise ValueError(f"{path}: amino-acid node features are not one-hot")
    if not torch.all((x[:,20]==0)|(x[:,20]==1)):
        raise ValueError(f"{path}: invalid partner-group node features")
    if chain_index.dtype not in (torch.int32,torch.int64):
        raise ValueError(f"{path}: chain indices must be integers")
    if sorted(set(chain_index.tolist()))!=list(range(len(chains))):
        raise ValueError(f"{path}: chain indices do not match chain sequences")
    for index,(sequence,group) in enumerate(zip(chains,groups)):
        if type(sequence) is not str or group not in (0,1):
            raise ValueError(f"{path}: invalid chain sequence or partner group")
        nodes=x[chain_index==index]
        observed="".join(AA_ORDER[i] for i in nodes[:,:20].argmax(dim=1).tolist())
        if not observed or observed!=sequence or not torch.all(nodes[:,20]==group):
            raise ValueError(f"{path}: chain {index} metadata disagrees with encoded residues")
    vhh=[sequence for sequence,group in zip(chains,groups) if group==0]
    antigen=[sequence for sequence,group in zip(chains,groups) if group==1]
    if not vhh or not antigen or getattr(graph,"vhh_sequences",None)!=vhh or getattr(graph,"antigen_sequences",None)!=antigen:
        raise ValueError(f"{path}: partner sequences disagree with encoded chains")
    cdr=str(getattr(graph,"cdr3_seq","") or "")
    # External graphs (require_cdr=True) are bound to their raw structure, so
    # their CDR-H3 must be an exact substring of the encoded VHH chain.
    # Training graphs are hash-bound to this run's frozen manifest; their
    # annotated CDR-H3 may legitimately contain residues that are unmodeled in
    # the structure, so it is kept as annotated instead of aborting the audit.
    if require_cdr and (not cdr or not any(cdr in sequence for sequence in vhh)):
        raise ValueError(f"{path}: CDR-H3 sequence is absent from encoded VHH chains")
    if int(getattr(graph,"cdr3_len",-1))!=len(cdr):
        raise ValueError(f"{path}: CDR-H3 length disagrees with sequence")
    return dict(pdb_id=pdb,vhh=vhh,antigen=antigen,cdr_h3=cdr)


def source_structure_for_pdb(source_dir: Path, pdb: str) -> Path:
    suffixes=(".cif.gz",".pdb.gz",".mmcif",".cif",".pdb")
    matches=[]
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        name=path.name.lower()
        if any(name==pdb+suffix for suffix in suffixes):
            matches.append(path)
    if len(matches)!=1:
        raise ValueError(f"Expected exactly one raw PDB/mmCIF structure for {pdb} in {source_dir}; found {len(matches)}")
    return matches[0]


def verify_graph_against_source(graph: object, source: Path) -> None:
    """Bind every encoded residue and CA position to an independent raw file.

    Graphs built from the biological assembly (``structure_source`` starting
    with ``biological_assembly:``) are checked against the same assembly
    rebuilt from the raw file, so symmetry-generated chains (``A-2``) are
    verified rather than reported as absent.
    """
    structure=gemmi.read_structure(str(source))
    if not len(structure):
        raise ValueError(f"Raw structure has no model: {source}")
    while len(structure)>1:
        del structure[1]
    recorded=str(getattr(graph,"structure_source","") or "")
    if recorded.startswith("biological_assembly:"):
        from nanoqc.data.audit_all_datasets import STRUCTURE_SOURCE_KEY, biological_assembly_structure
        structure=biological_assembly_structure(structure)
        rebuilt=dict(structure.info).get(STRUCTURE_SOURCE_KEY)
        if rebuilt!=recorded:
            raise ValueError(f"Assembly choice {rebuilt} differs from graph provenance {recorded}: {source}")
    residues={}
    for chain in structure[0]:
        for residue in chain:
            info=gemmi.find_tabulated_residue(residue.name)
            if not info.is_amino_acid():
                continue
            atoms={}
            for atom in residue:
                if atom.occ>0 and not atom.element.is_hydrogen:
                    if atom.name not in atoms or atom.occ>atoms[atom.name].occ:
                        atoms[atom.name]=atom
            if "CA" not in atoms:
                continue
            aa=info.one_letter_code.upper()
            key=f"{chain.name}:{residue.seqid}"
            if key in residues:
                raise ValueError(f"Duplicate raw residue identity {key}: {source}")
            residues[key]=(aa,atoms["CA"].pos)
    ids=getattr(graph,"residue_ids",None)
    positions=getattr(graph,"pos",None)
    if (not isinstance(ids,list) or not isinstance(positions,torch.Tensor)
            or len(ids)!=len(graph.x) or tuple(positions.shape)!=(len(ids),3)
            or len(set(ids))!=len(ids)):
        raise ValueError(f"Graph lacks unique residue IDs/CA positions: {source}")
    if not torch.isfinite(positions).all():
        raise ValueError(f"Graph has nonfinite CA coordinates: {source}")
    for index,key in enumerate(ids):
        if key not in residues:
            raise ValueError(f"Graph residue {key} absent from raw structure: {source}")
        aa,ca=residues[key]
        if aa not in AA_ORDER or AA_ORDER.index(aa)!=int(graph.x[index,:20].argmax()):
            raise ValueError(f"Graph residue {key} disagrees with raw structure sequence: {source}")
        observed=positions[index].tolist()
        if max(abs(observed[i]-getattr(ca,"xyz"[i])) for i in range(3))>1e-3:
            raise ValueError(f"Graph residue {key} CA coordinate disagrees with raw structure: {source}")


def identity(a: str, b: str, min_length_coverage: float = 0.0) -> tuple[float,float]:
    a,b=str(a or ""),str(b or "")
    if not a or not b:
        return 0.0,0.0
    coverage=length_coverage(a,b)
    if coverage < min_length_coverage:
        return 0.0,coverage
    return nw_identity(a,b,saturation_message="parasail alignment saturated"),coverage


def side_max(left: list[str], right: list[str], coverage: float = 0.0) -> tuple[float,float]:
    best=(0.0,0.0)
    for a in left:
        for b in right:
            value,cov=identity(a,b,coverage)
            if value>best[0]:
                best=(value,cov)
    return best


def max_training_identities(ext: dict, train: list[dict], antigen_min_length_coverage: float) -> dict:
    """Largest layered identities of one external complex against every training graph.

    External partner roles are verified (CDR-H3 inside the encoded VHH chain);
    training complexes without anchored roles are also compared with their
    partners swapped (see sequence_identity.partner_orientations).
    """
    max_vhh=0.0;max_cdr=0.0;max_ag=0.0;ag_cov=0.0
    for tr in train:
        for ev,tv,ea,ta in partner_orientations(
                ext["vhh"],ext["antigen"],True,
                tr["vhh"],tr["antigen"],partner_roles_anchored(tr["subset_source"])):
            value,_=side_max(ev,tv)
            max_vhh=max(max_vhh,value)
            value,cov=side_max(ea,ta,antigen_min_length_coverage)
            if value>max_ag:
                max_ag=value;ag_cov=cov
        value,_=identity(ext["cdr_h3"],tr["cdr_h3"]) if ext["cdr_h3"] and tr["cdr_h3"] else (0.0,0.0)
        max_cdr=max(max_cdr,value)
    return dict(max_vhh_full_chain_identity=max_vhh,max_cdr_h3_loop_identity=max_cdr,
                max_antigen_full_chain_identity=max_ag,antigen_length_coverage=ag_cov)


def load_training_sequences(training_dataset: Path) -> list[dict]:
    """Sequence records of every hash-verified training graph in a frozen dataset."""
    manifest_path=training_dataset/"graph_manifest.json"
    graph_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows=[row for row in graph_manifest if row.get("split")=="train"]
    if not train_rows:
        raise ValueError("No training rows in graph manifest")
    train=[]
    root=training_dataset.resolve()
    for row in train_rows:
        relative=row.get("path")
        if not isinstance(relative,str) or not relative or Path(relative).is_absolute():
            raise ValueError(f"Training graph manifest path must be relative: {relative!r}")
        path=(root/relative.replace("\\","/")).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Training graph manifest path escapes dataset: {relative}")
        if not path.is_file() or sha256(path)!=row["sha256"]:
            raise ValueError(f"Training graph missing/hash mismatch: {path}")
        item=graph_sequences(path)
        if item["pdb_id"]!=str(row.get("pdb_id","")).lower():
            raise ValueError(f"Training graph PDB ID differs from frozen manifest: {path}")
        train.append(item)
    return train


def graph_sequences(path: Path, source_dir: Path | None = None) -> dict:
    digest=sha256(path)
    graph=load_graph(path)
    identity_fields=verified_graph_identity(graph,path,require_cdr=source_dir is not None)
    source=None
    if source_dir is not None:
        source=source_structure_for_pdb(source_dir,identity_fields["pdb_id"])
        verify_graph_against_source(graph,source)
    return dict(
        **identity_fields,
        graph_version=str(getattr(graph,"graph_version","")),
        subset_source=str(getattr(graph,"subset_source","") or ""),
        sha256=digest,
        source_structure=(None if source is None else str(source.resolve())),
        source_structure_sha256=(None if source is None else sha256(source)),
    )


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset",type=Path,required=True,
        help="Frozen dataset directory containing graph_manifest.json and graphs/train.")
    parser.add_argument("--external-graph-dir",type=Path,required=True)
    parser.add_argument("--external-source-dir",type=Path,required=True,
        help="Trusted raw PDB/mmCIF files named by their four-character PDB IDs.")
    parser.add_argument("--cluster-map",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--vhh-threshold",type=float,default=0.80)
    parser.add_argument("--cdr-h3-threshold",type=float,default=0.50)
    parser.add_argument("--antigen-threshold",type=float,default=0.30)
    parser.add_argument("--antigen-min-length-coverage",type=float,default=0.70)
    args=parser.parse_args()

    thresholds=(
        args.vhh_threshold,args.cdr_h3_threshold,args.antigen_threshold,
        args.antigen_min_length_coverage,
    )
    if any(not 0.0 < value <= 1.0 for value in thresholds):
        parser.error("All identity/coverage thresholds must lie in (0,1]")
    manifest_path=args.training_dataset/"graph_manifest.json"
    if not manifest_path.is_file():
        parser.error("Training graph_manifest.json not found")
    if not args.external_graph_dir.is_dir():
        parser.error("External graph directory not found")
    if not args.external_source_dir.is_dir():
        parser.error("External raw-structure directory not found")
    if not args.cluster_map.is_file():
        parser.error("Cluster map not found")

    raw_map=json.loads(args.cluster_map.read_text(encoding="utf-8"))
    cluster_map={str(k).lower():str(v) for k,v in raw_map.items()}
    if not cluster_map:
        raise ValueError("Cluster map is empty")

    train=load_training_sequences(args.training_dataset)
    train_pdb={item["pdb_id"] for item in train}
    missing_train=sorted(train_pdb-set(cluster_map))
    if missing_train:
        raise ValueError(f"Cluster map missing training PDBs: {missing_train[:20]}")
    train_clusters={cluster_map[pdb] for pdb in train_pdb}

    external_paths=sorted(args.external_graph_dir.glob("*.pt"))
    if not external_paths:
        raise ValueError("No external .pt graphs")
    audits=[]
    graph_versions=set()
    external_pdbs=set()
    for path in external_paths:
        ext=graph_sequences(path,args.external_source_dir)
        graph_versions.add(ext["graph_version"])
        pdb=ext["pdb_id"]
        if pdb in external_pdbs:
            raise ValueError(f"Duplicate external PDB ID {pdb}: {path}")
        external_pdbs.add(pdb)
        if pdb not in cluster_map:
            raise ValueError(f"Cluster map missing external PDB {pdb}")
        overlap=max_training_identities(ext,train,args.antigen_min_length_coverage)
        max_vhh=overlap["max_vhh_full_chain_identity"]
        max_cdr=overlap["max_cdr_h3_loop_identity"]
        max_ag=overlap["max_antigen_full_chain_identity"]
        ag_cov=overlap["antigen_length_coverage"]
        family_overlap=cluster_map[pdb] in train_clusters
        audits.append(dict(
            pdb_id=pdb,
            graph_sha256=ext["sha256"],
            source_structure=ext["source_structure"],
            source_structure_sha256=ext["source_structure_sha256"],
            family_cluster=cluster_map[pdb],
            family_cluster_overlap=family_overlap,
            max_vhh_full_chain_identity=max_vhh,
            max_cdr_h3_loop_identity=max_cdr,
            max_antigen_full_chain_identity=max_ag,
            antigen_length_coverage=ag_cov,
            passes=bool(
                max_vhh < args.vhh_threshold
                and max_cdr < args.cdr_h3_threshold
                and not (
                    max_ag >= args.antigen_threshold
                    and ag_cov >= args.antigen_min_length_coverage
                )
                and not family_overlap
            ),
        ))

    failed=[row for row in audits if not row["passes"]]
    payload=dict(
        graph_version=(next(iter(graph_versions)) if len(graph_versions)==1 else sorted(graph_versions)),
        training_family_overlap_zero=not any(row["family_cluster_overlap"] for row in audits),
        all_targets_pass=not failed,
        homology_isolation=dict(
            vhh_full_chain_identity=float(args.vhh_threshold),
            cdr_h3_identity=float(args.cdr_h3_threshold),
            antigen_identity=float(args.antigen_threshold),
            antigen_min_length_coverage=float(args.antigen_min_length_coverage),
        ),
        training_cluster_map_sha256=sha256(args.cluster_map),
        training_graph_manifest_sha256=sha256(manifest_path),
        external_graph_dir=str(args.external_graph_dir.resolve()),
        target_count=len(audits),
        failed_targets=[row["pdb_id"] for row in failed],
        targets=audits,
        scope="external graphs compared only against frozen training graphs and frozen family/structure clusters",
    )
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if failed:
        raise SystemExit(f"External independence failed for {len(failed)} target(s); see {args.out}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
