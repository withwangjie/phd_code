#!/usr/bin/env python3
"""Pre-registered structural endpoint and energy-to-structure statistics.

Primary endpoint: post-relaxation symmetry-corrected Active side-chain
heavy-atom RMSD. Primary contrast: QAOA - SA. Repeats are averaged within PDB
before cluster-level inference. RQ5 uses Spearman association between discrete
energy and post-relaxation RMSD, with cluster bootstrap confidence intervals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def percentile_ci(values: list[float], rng: np.random.Generator, resamples: int) -> tuple[float|None,float|None]:
    if len(values)<2:
        return None,None
    array=np.asarray(values,float)
    draws=np.empty(resamples,float)
    for i in range(resamples):
        draws[i]=float(np.mean(rng.choice(array,size=len(array),replace=True)))
    return float(np.percentile(draws,2.5)),float(np.percentile(draws,97.5))


def sign_flip_p(values: list[float], seed: int) -> float|None:
    d=np.asarray(values,float)
    if len(d)<2:
        return None
    observed=abs(float(np.mean(d)))
    if len(d)<=20:
        total=1<<len(d);extreme=0
        for mask in range(total):
            signs=np.asarray([1.0 if (mask>>i)&1 else -1.0 for i in range(len(d))])
            if abs(float(np.mean(signs*d)))>=observed-1e-15:
                extreme+=1
        return extreme/total
    rng=np.random.default_rng(seed)
    trials=200000
    extreme=0
    for _ in range(trials):
        signs=rng.choice((-1.0,1.0),size=len(d))
        extreme+=abs(float(np.mean(signs*d)))>=observed-1e-15
    return (extreme+1)/(trials+1)


def grouped_primary(
    rows: list[dict], cluster_map: dict[str,str], endpoint: str, contrast: str,
    resamples: int, seed: int,
) -> dict:
    by_target=defaultdict(lambda:defaultdict(list))
    for row in rows:
        method=row.get("method","")
        target=row.get("target","").lower()
        value=row.get(endpoint)
        if method and target and value not in (None,"","None"):
            by_target[target][method].append(float(value))
    baseline_map={
        "qaoa_vs_sa":"sa",
        "qaoa_vs_greedy":"greedy",
        "qaoa_vs_uniform":"uniform",
    }
    if contrast not in baseline_map:
        raise ValueError(f"Unsupported primary contrast: {contrast}")
    baseline=baseline_map[contrast]
    by_cluster=defaultdict(list)
    excluded=[]
    for target,methods in sorted(by_target.items()):
        if "qaoa" not in methods or baseline not in methods:
            excluded.append(target);continue
        if target not in cluster_map:
            raise ValueError(f"Missing cluster mapping for structural target {target}")
        difference=float(np.mean(methods["qaoa"])-np.mean(methods[baseline]))
        by_cluster[cluster_map[target]].append(difference)
    cluster_values=[float(np.mean(values)) for _,values in sorted(by_cluster.items())]
    rng=np.random.default_rng(seed)
    low,high=percentile_ci(cluster_values,rng,resamples)
    return dict(
        endpoint=endpoint,
        contrast=f"QAOA-{baseline}; sign interpretation depends on endpoint direction",
        n_targets=sum(len(v) for v in by_cluster.values()),
        n_clusters=len(cluster_values),
        mean_difference=(None if not cluster_values else float(np.mean(cluster_values))),
        ci_low=low,ci_high=high,p_value=sign_flip_p(cluster_values,seed),
        excluded_targets=excluded,
    )


def rq5_energy_structure(rows: list[dict], cluster_map: dict[str,str], resamples: int, seed: int) -> dict:
    # Within each target/method, center both quantities so between-complex
    # absolute energy offsets cannot create a spurious pooled correlation.
    centered=[]
    grouped=defaultdict(list)
    for row in rows:
        target=row.get("target","").lower()
        method=row.get("method","")
        e=row.get("discrete_energy_kcal")
        r=row.get("sidechain_rmsd_after_relaxation") or row.get("final_rmsd")
        if target and method and e not in (None,"","None") and r not in (None,"","None"):
            grouped[(target,method)].append((float(e),float(r)))
    for (target,method),pairs in grouped.items():
        if target not in cluster_map:
            raise ValueError(f"Missing cluster mapping for structural target {target}")
        energies=np.asarray([p[0] for p in pairs],float)
        rmsds=np.asarray([p[1] for p in pairs],float)
        for e,r in zip(energies-energies.mean(),rmsds-rmsds.mean()):
            centered.append((target,cluster_map[target],method,float(e),float(r)))
    if len(centered)<3:
        return dict(n_rows=len(centered),spearman_rho=None,p_value=None,ci_low=None,ci_high=None)

    e=np.asarray([x[3] for x in centered],float)
    r=np.asarray([x[4] for x in centered],float)
    rho=float(spearmanr(e,r).statistic)
    clusters=sorted({x[1] for x in centered})
    rng=np.random.default_rng(seed)
    boots=[]
    by_cluster={cluster:[x for x in centered if x[1]==cluster] for cluster in clusters}
    if len(clusters)>=2:
        for _ in range(resamples):
            sampled=rng.choice(clusters,size=len(clusters),replace=True)
            sample=[]
            for cluster in sampled:
                sample.extend(by_cluster[str(cluster)])
            se=np.asarray([x[3] for x in sample],float)
            sr=np.asarray([x[4] for x in sample],float)
            value=spearmanr(se,sr).statistic
            if math.isfinite(float(value)):
                boots.append(float(value))

    # Cluster-aware null: preserve target/method blocks and permute only the
    # centered RMSD residuals within each block. This avoids treating repeated
    # seeds or different proteins as exchangeable independent observations.
    blocks=defaultdict(list)
    for item in centered:
        blocks[(item[0],item[2])].append(item)
    trials=max(1000,min(int(resamples),20000))
    extreme=0;valid_trials=0
    if math.isfinite(rho):
        for _ in range(trials):
            pe=[];pr=[]
            for block in blocks.values():
                be=np.asarray([x[3] for x in block],float)
                br=np.asarray([x[4] for x in block],float)
                shuffled=rng.permutation(br)
                pe.extend(be.tolist());pr.extend(shuffled.tolist())
            value=spearmanr(np.asarray(pe,float),np.asarray(pr,float)).statistic
            if math.isfinite(float(value)):
                valid_trials+=1
                if abs(float(value))>=abs(rho)-1e-15:
                    extreme+=1
    permutation_p=None if valid_trials==0 else (extreme+1)/(valid_trials+1)
    return dict(
        definition="within-target/method centered discrete energy vs post-relaxation side-chain RMSD",
        n_rows=len(centered),n_clusters=len(clusters),
        spearman_rho=rho,p_value=permutation_p,
        p_value_method="within-target/method centered-RMSD permutation",
        permutation_trials=valid_trials,
        ci_low=(None if not boots else float(np.percentile(boots,2.5))),
        ci_high=(None if not boots else float(np.percentile(boots,97.5))),
        interpretation="positive rho means higher discrete energy is associated with worse final RMSD",
    )


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics",type=Path,required=True)
    parser.add_argument("--cluster-map",type=Path,required=True)
    parser.add_argument("--out-json",type=Path,required=True)
    parser.add_argument("--out-md",type=Path,required=True)
    parser.add_argument("--primary-endpoint",default="final_rmsd",
        choices=("final_rmsd","improvement_vs_input","improvement_vs_relax_only"))
    parser.add_argument("--primary-contrast",default="qaoa_vs_sa",
        choices=("qaoa_vs_sa","qaoa_vs_greedy","qaoa_vs_uniform"))
    parser.add_argument("--resamples",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=20260917)
    args=parser.parse_args()
    if args.resamples<1000:
        parser.error("--resamples must be >=1000")
    rows=list(csv.DictReader(args.metrics.open(encoding="utf-8-sig",newline="")))
    if not rows:
        raise ValueError("Structural metrics CSV is empty")
    raw=json.loads(args.cluster_map.read_text(encoding="utf-8"))
    cluster_map={str(k).lower():str(v) for k,v in raw.items()}
    primary=grouped_primary(rows,cluster_map,args.primary_endpoint,args.primary_contrast,args.resamples,args.seed)
    rq5=rq5_energy_structure(rows,cluster_map,args.resamples,args.seed)
    payload=dict(
        primary=primary,rq5=rq5,resamples=args.resamples,seed=args.seed,
        metrics_sha256=sha256(args.metrics),cluster_map_sha256=sha256(args.cluster_map),
        multiplicity_policy="single pre-registered primary structural contrast; secondary metrics descriptive unless separately adjusted",
    )
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=[
        "# Structural primary endpoint and RQ5 analysis","",
        f"Primary endpoint: `{args.primary_endpoint}`; contrast: `{args.primary_contrast}`.",
        f"Clusters: {primary['n_clusters']}; mean difference: {primary['mean_difference']}; "
        f"95% cluster-bootstrap CI: [{primary['ci_low']}, {primary['ci_high']}]; p={primary['p_value']}.","",
        "## Energy-to-structure transfer","",
        f"Spearman rho={rq5.get('spearman_rho')} with 95% cluster-bootstrap CI "
        f"[{rq5.get('ci_low')}, {rq5.get('ci_high')}], "
        f"cluster-aware permutation p={rq5.get('p_value')}.",
        rq5.get("interpretation",""),"",
        "Repeated seeds are centered/averaged within target before family-cluster inference; "
        "PDB rows are not treated as independent proteins.",
    ]
    args.out_md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
