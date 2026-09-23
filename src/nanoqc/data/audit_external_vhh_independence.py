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

import torch
from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.data.sequence_identity import nw_identity


def identity(a: str, b: str, min_length_coverage: float = 0.0) -> tuple[float,float]:
    a,b=str(a or ""),str(b or "")
    if not a or not b:
        return 0.0,0.0
    coverage=min(len(a),len(b))/max(len(a),len(b))
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


def graph_sequences(path: Path) -> dict:
    graph=torch.load(path,map_location="cpu",weights_only=False)
    return dict(
        pdb_id=str(getattr(graph,"pdb_id",path.stem)).lower(),
        graph_version=str(getattr(graph,"graph_version","")),
        vhh=[str(x) for x in getattr(graph,"vhh_sequences",[]) if str(x)],
        antigen=[str(x) for x in getattr(graph,"antigen_sequences",[]) if str(x)],
        cdr_h3=str(getattr(graph,"cdr3_seq","") or ""),
        sha256=sha256(path),
    )


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dataset",type=Path,required=True,
        help="Frozen dataset directory containing graph_manifest.json and graphs/train.")
    parser.add_argument("--external-graph-dir",type=Path,required=True)
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
    if not args.cluster_map.is_file():
        parser.error("Cluster map not found")

    raw_map=json.loads(args.cluster_map.read_text(encoding="utf-8"))
    cluster_map={str(k).lower():str(v) for k,v in raw_map.items()}
    if not cluster_map:
        raise ValueError("Cluster map is empty")

    graph_manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    train_rows=[row for row in graph_manifest if row.get("split")=="train"]
    if not train_rows:
        raise ValueError("No training rows in graph manifest")
    train=[]
    train_pdb=set()
    for row in train_rows:
        path=args.training_dataset/Path(str(row["path"]).replace("\\","/"))
        if not path.is_file() or sha256(path)!=row["sha256"]:
            raise ValueError(f"Training graph missing/hash mismatch: {path}")
        item=graph_sequences(path)
        train.append(item);train_pdb.add(item["pdb_id"])
    missing_train=sorted(train_pdb-set(cluster_map))
    if missing_train:
        raise ValueError(f"Cluster map missing training PDBs: {missing_train[:20]}")
    train_clusters={cluster_map[pdb] for pdb in train_pdb}

    external_paths=sorted(args.external_graph_dir.glob("*.pt"))
    if not external_paths:
        raise ValueError("No external .pt graphs")
    audits=[]
    graph_versions=set()
    for path in external_paths:
        ext=graph_sequences(path)
        graph_versions.add(ext["graph_version"])
        pdb=ext["pdb_id"]
        if pdb not in cluster_map:
            raise ValueError(f"Cluster map missing external PDB {pdb}")
        max_vhh=0.0;max_cdr=0.0;max_ag=0.0;ag_cov=0.0
        for tr in train:
            value,_=side_max(ext["vhh"],tr["vhh"])
            max_vhh=max(max_vhh,value)
            value,_=identity(ext["cdr_h3"],tr["cdr_h3"]) if ext["cdr_h3"] and tr["cdr_h3"] else (0.0,0.0)
            max_cdr=max(max_cdr,value)
            value,cov=side_max(ext["antigen"],tr["antigen"],args.antigen_min_length_coverage)
            if value>max_ag:
                max_ag=value;ag_cov=cov
        family_overlap=cluster_map[pdb] in train_clusters
        audits.append(dict(
            pdb_id=pdb,
            graph_sha256=ext["sha256"],
            family_cluster=cluster_map[pdb],
            family_cluster_overlap=family_overlap,
            max_vhh_identity=max_vhh,
            max_cdr_h3_identity=max_cdr,
            max_antigen_identity=max_ag,
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
