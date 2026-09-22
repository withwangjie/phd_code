#!/usr/bin/env python3
"""Pre-registered structural endpoint and energy-to-structure statistics.

Primary endpoint: post-relaxation symmetry-corrected Active side-chain
heavy-atom RMSD. Primary contrast: QAOA - SA. Solver values are strictly paired
within the same target and perturbation seed, then averaged within target and
family/structure cluster. RQ5 tests whether paired solver differences in
discrete energy propagate to paired differences in post-relaxation RMSD using
cluster-level Spearman inference, bootstrap confidence intervals and
permutation tests. The primary structural contrast and RQ5 form one
two-hypothesis Holm-adjusted confirmatory family.
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


def holm_adjust_named(pvalues: dict[str,float|None]) -> dict[str,float|None]:
    """Holm step-down adjustment for the pre-registered confirmatory family."""
    valid=sorted(
        ((name,float(value)) for name,value in pvalues.items()
         if value is not None and math.isfinite(float(value))),
        key=lambda item:item[1],
    )
    adjusted={name:None for name in pvalues}
    running=0.0
    m=len(valid)
    for rank,(name,value) in enumerate(valid):
        candidate=min(1.0,(m-rank)*value)
        running=max(running,candidate)
        adjusted[name]=running
    return adjusted


def grouped_primary(
    rows: list[dict], cluster_map: dict[str,str], endpoint: str, contrast: str,
    resamples: int, seed: int,
) -> dict:
    """Strict paired-seed primary contrast, then target and cluster averaging."""

    baseline_map={
        "qaoa_vs_sa":"sa",
        "qaoa_vs_greedy":"greedy",
        "qaoa_vs_uniform":"uniform",
    }
    if contrast not in baseline_map:
        raise ValueError(f"Unsupported primary contrast: {contrast}")
    baseline=baseline_map[contrast]

    # One value per target x seed x method. Duplicate rows are an error rather
    # than silently averaged because the formal recovery protocol should emit
    # exactly one row for each solver at each frozen perturbation seed.
    values={}
    duplicate_keys=[]
    for row in rows:
        method=str(row.get("method",""))
        target=str(row.get("target","")).lower()
        seed_value=str(row.get("seed",""))
        value=row.get(endpoint)
        if method not in ("qaoa",baseline) or not target or seed_value=="" or value in (None,"","None"):
            continue
        key=(target,seed_value,method)
        if key in values:
            duplicate_keys.append(key)
        else:
            values[key]=float(value)
    if duplicate_keys:
        raise ValueError(f"Duplicate structural primary rows: {duplicate_keys[:10]}")

    target_seed_sets=defaultdict(set)
    for target,seed_value,method in values:
        target_seed_sets[target].add(seed_value)

    by_cluster=defaultdict(list)
    excluded_targets=[]
    paired_seed_count=0
    incomplete_seed_count=0
    target_details=[]
    for target,seeds in sorted(target_seed_sets.items()):
        if target not in cluster_map:
            raise ValueError(f"Missing cluster mapping for structural target {target}")
        differences=[]
        missing=[]
        for seed_value in sorted(seeds):
            qkey=(target,seed_value,"qaoa")
            bkey=(target,seed_value,baseline)
            if qkey in values and bkey in values:
                differences.append(values[qkey]-values[bkey])
                paired_seed_count+=1
            else:
                missing.append(seed_value)
                incomplete_seed_count+=1
        if not differences:
            excluded_targets.append(target)
            continue
        target_difference=float(np.mean(differences))
        by_cluster[cluster_map[target]].append(target_difference)
        target_details.append(dict(
            target=target,cluster=cluster_map[target],
            paired_seeds=len(differences),missing_or_unpaired_seeds=missing,
            mean_difference=target_difference,
        ))

    cluster_values=[float(np.mean(values)) for _,values in sorted(by_cluster.items())]
    rng=np.random.default_rng(seed)
    low,high=percentile_ci(cluster_values,rng,resamples)
    return dict(
        endpoint=endpoint,
        contrast=f"QAOA-{baseline}; sign interpretation depends on endpoint direction",
        pairing_unit="same target and perturbation seed",
        n_targets=sum(len(v) for v in by_cluster.values()),
        n_clusters=len(cluster_values),
        paired_seed_count=int(paired_seed_count),
        incomplete_seed_count=int(incomplete_seed_count),
        mean_difference=(None if not cluster_values else float(np.mean(cluster_values))),
        ci_low=low,ci_high=high,p_value=sign_flip_p(cluster_values,seed),
        excluded_targets=excluded_targets,
        target_details=target_details,
    )


def rq5_energy_structure(
    rows: list[dict], cluster_map: dict[str,str], contrast: str,
    resamples: int, seed: int,
) -> dict:
    """Relate paired solver energy gains to paired structural gains.

    For the pre-registered contrast (normally QAOA-SA), compute same-target,
    same-perturbation-seed differences in discrete all-atom energy and final
    side-chain RMSD. Differences are averaged within target and then within
    family/structure cluster before Spearman inference, so repeated seeds and
    homologous PDBs are not treated as independent observations.
    """

    baseline_map={
        "qaoa_vs_sa":"sa",
        "qaoa_vs_greedy":"greedy",
        "qaoa_vs_uniform":"uniform",
    }
    if contrast not in baseline_map:
        raise ValueError(f"Unsupported RQ5 contrast: {contrast}")
    baseline=baseline_map[contrast]

    values={}
    duplicates=[]
    for row in rows:
        target=str(row.get("target","")).lower()
        seed_value=str(row.get("seed",""))
        method=str(row.get("method",""))
        energy=row.get("discrete_energy_kcal")
        rmsd=row.get("sidechain_rmsd_after_relaxation") or row.get("final_rmsd")
        if (
            target and seed_value!="" and method in ("qaoa",baseline)
            and energy not in (None,"","None") and rmsd not in (None,"","None")
        ):
            key=(target,seed_value,method)
            if key in values:
                duplicates.append(key)
            else:
                values[key]=(float(energy),float(rmsd))
    if duplicates:
        raise ValueError(f"Duplicate RQ5 rows: {duplicates[:10]}")

    seeds_by_target=defaultdict(set)
    for target,seed_value,_ in values:
        seeds_by_target[target].add(seed_value)

    target_points=[]
    incomplete_seed_count=0
    for target,seeds in sorted(seeds_by_target.items()):
        if target not in cluster_map:
            raise ValueError(f"Missing cluster mapping for structural target {target}")
        de=[];dr=[];paired=0
        for seed_value in sorted(seeds):
            qkey=(target,seed_value,"qaoa")
            bkey=(target,seed_value,baseline)
            if qkey not in values or bkey not in values:
                incomplete_seed_count+=1
                continue
            qe,qr=values[qkey];be,br=values[bkey]
            de.append(qe-be)
            dr.append(qr-br)
            paired+=1
        if paired:
            target_points.append(dict(
                target=target,cluster=cluster_map[target],paired_seeds=paired,
                delta_energy=float(np.mean(de)),
                delta_rmsd=float(np.mean(dr)),
            ))

    by_cluster=defaultdict(list)
    for point in target_points:
        by_cluster[point["cluster"]].append(point)

    cluster_points=[]
    for cluster,points in sorted(by_cluster.items()):
        cluster_points.append(dict(
            cluster=cluster,
            target_count=len(points),
            delta_energy=float(np.mean([p["delta_energy"] for p in points])),
            delta_rmsd=float(np.mean([p["delta_rmsd"] for p in points])),
        ))

    if len(cluster_points)<3:
        return dict(
            definition=f"paired QAOA-{baseline} discrete-energy difference vs final-RMSD difference",
            contrast=contrast,n_targets=len(target_points),n_clusters=len(cluster_points),
            paired_seed_count=sum(p["paired_seeds"] for p in target_points),
            incomplete_seed_count=incomplete_seed_count,
            spearman_rho=None,p_value=None,ci_low=None,ci_high=None,
            target_points=target_points,cluster_points=cluster_points,
        )

    energy=np.asarray([p["delta_energy"] for p in cluster_points],float)
    rmsd=np.asarray([p["delta_rmsd"] for p in cluster_points],float)
    rho_value=spearmanr(energy,rmsd).statistic
    rho=None if not math.isfinite(float(rho_value)) else float(rho_value)

    rng=np.random.default_rng(seed)
    boots=[]
    if rho is not None:
        n=len(cluster_points)
        for _ in range(resamples):
            idx=rng.integers(0,n,size=n)
            if len(set(idx.tolist()))<2:
                continue
            value=spearmanr(energy[idx],rmsd[idx]).statistic
            if math.isfinite(float(value)):
                boots.append(float(value))

    # Exact/Monte-Carlo permutation at the independent cluster unit.
    trials=max(1000,min(int(resamples),20000))
    extreme=0;valid=0
    if rho is not None:
        for _ in range(trials):
            permuted=rng.permutation(rmsd)
            value=spearmanr(energy,permuted).statistic
            if math.isfinite(float(value)):
                valid+=1
                if abs(float(value))>=abs(rho)-1e-15:
                    extreme+=1
    pvalue=None if valid==0 else (extreme+1)/(valid+1)

    return dict(
        definition=f"paired QAOA-{baseline} discrete-energy difference vs final-RMSD difference",
        contrast=contrast,
        difference_sign="negative delta means QAOA lower than baseline",
        n_targets=len(target_points),n_clusters=len(cluster_points),
        paired_seed_count=sum(p["paired_seeds"] for p in target_points),
        incomplete_seed_count=incomplete_seed_count,
        spearman_rho=rho,p_value=pvalue,
        p_value_method="family/structure-cluster-level RMSD-difference permutation",
        permutation_trials=valid,
        ci_low=(None if not boots else float(np.percentile(boots,2.5))),
        ci_high=(None if not boots else float(np.percentile(boots,97.5))),
        interpretation=(
            "positive rho means targets/clusters with a larger QAOA energy disadvantage "
            "also tend to have a larger QAOA RMSD disadvantage; negative energy and RMSD "
            "differences together represent solver energy gains propagating to structural gains"
        ),
        target_points=target_points,cluster_points=cluster_points,
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
    rq5=rq5_energy_structure(
        rows,cluster_map,args.primary_contrast,args.resamples,args.seed
    )
    adjusted=holm_adjust_named({
        "primary_structural_contrast":primary.get("p_value"),
        "rq5_energy_structure":rq5.get("p_value"),
    })
    primary["p_holm_confirmatory_family"]=adjusted["primary_structural_contrast"]
    rq5["p_holm_confirmatory_family"]=adjusted["rq5_energy_structure"]
    payload=dict(
        primary=primary,rq5=rq5,resamples=args.resamples,seed=args.seed,
        metrics_sha256=sha256(args.metrics),cluster_map_sha256=sha256(args.cluster_map),
        multiplicity_policy="Primary structural contrast and RQ5 form one two-hypothesis confirmatory family with Holm FWER adjustment; all other structural metrics are descriptive unless separately adjusted.",
        confirmatory_family=dict(
            hypotheses=["primary_structural_contrast","rq5_energy_structure"],
            correction="Holm FWER",adjusted_p=adjusted,
        ),
    )
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=[
        "# Structural primary endpoint and RQ5 analysis","",
        f"Primary endpoint: `{args.primary_endpoint}`; contrast: `{args.primary_contrast}`.",
        f"Clusters: {primary['n_clusters']}; mean difference: {primary['mean_difference']}; "
        f"95% cluster-bootstrap CI: [{primary['ci_low']}, {primary['ci_high']}]; "
        f"raw p={primary['p_value']}; Holm p={primary.get('p_holm_confirmatory_family')}.","",
        "## Energy-to-structure transfer","",
        f"Spearman rho={rq5.get('spearman_rho')} with 95% cluster-bootstrap CI "
        f"[{rq5.get('ci_low')}, {rq5.get('ci_high')}], "
        f"cluster-aware permutation raw p={rq5.get('p_value')}; "
        f"Holm p={rq5.get('p_holm_confirmatory_family')}.",
        rq5.get("interpretation",""),"",
        "Repeated seeds are strictly paired by target+seed, then averaged within target before family-cluster inference; "
        "PDB rows are not treated as independent proteins.",
    ]
    args.out_md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
