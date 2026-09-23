#!/usr/bin/env python3
"""Formal size-scaling analysis for the quantum-classical coarse benchmark.

Primary estimand:
    within-PDB slope of (QAOA gap - classical gap) versus log10(feasible
    configuration count), followed by equal-weight family/structure-cluster
    aggregation. Negative slope means QAOA's relative energy-gap performance
    becomes more favorable as combinatorial complexity increases.

This is a simulator-level algorithmic scaling analysis, not evidence of
hardware quantum speedup.
"""
from __future__ import annotations
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
from typing import Sequence,Optional
import numpy as np
from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.inference.paired_statistics import bootstrap_sign_flip

def _slope(xs:list[float],ys:list[float]) -> Optional[float]:
    x=np.asarray(xs,float);y=np.asarray(ys,float)
    if len(x)<3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    centered=x-x.mean()
    denom=float(np.dot(centered,centered))
    if denom<=1e-15:
        return None
    return float(np.dot(centered,y-y.mean())/denom)

def _cluster_effect(values:list[float],seed:int,resamples:int)->dict:
    arr=np.asarray(values,float)
    if not len(arr) or not np.isfinite(arr).all():
        return dict(n_clusters=0,mean_slope=None,ci_low=None,ci_high=None,p_value=None)
    result=dict(n_clusters=len(arr),mean_slope=float(arr.mean()),ci_low=None,ci_high=None,p_value=None)
    if len(arr)<2:
        return result
    ci_low,ci_high,p=bootstrap_sign_flip(arr,seed,resamples)
    result["ci_low"]=ci_low
    result["ci_high"]=ci_high
    result["p_value"]=p
    return result

def _validate_fixed_state_case(case:dict, sites:int, num_bits:int)->None:
    if case.get("config",{}).get("state_policy") != "fixed_three_chi1_wells":
        raise ValueError("scaling case lacks the fixed three-chi1-well state policy")
    groups=case.get("site_to_variables",{})
    if num_bits != 3*sites or len(groups) != sites or any(len(v)!=3 for v in groups.values()):
        raise ValueError("scaling case does not have exactly three states per site")
    records=case.get("variable_map",[])
    if len(records)!=num_bits:
        raise ValueError("scaling case variable map is incomplete")
    wells=defaultdict(set)
    for record in records:
        angle=float(record["chi1_degrees"])
        if not math.isfinite(angle):
            raise ValueError("scaling case has a nonfinite chi1 angle")
        centers=(60.0,-60.0,180.0)
        well=min(range(3),key=lambda i:abs((angle-centers[i]+180.0)%360.0-180.0))
        wells[int(record["site_index"])].add(well)
    if len(wells)!=sites or any(len(group)!=3 for group in wells.values()):
        raise ValueError("scaling case lacks three distinct chi1 wells per site")

def main(argv:Optional[Sequence[str]]=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir",type=Path,required=True)
    p.add_argument("--cluster-map",type=Path,required=True)
    p.add_argument("--out-json",type=Path,required=True)
    p.add_argument("--out-md",type=Path,required=True)
    p.add_argument("--primary-pruning",default="egnn")
    p.add_argument("--primary-outputs",type=int,default=1000)
    p.add_argument("--primary-objective",choices=("mean","cvar"),default="cvar")
    p.add_argument("--primary-restarts",type=int,default=4)
    p.add_argument("--primary-depth",type=int,default=2)
    p.add_argument("--primary-max-evals",type=int,default=90)
    p.add_argument("--primary-radius",type=float,default=6.0)
    p.add_argument("--baseline",choices=("sa","uniform","greedy"),default="sa")
    p.add_argument("--active-sites",type=int,nargs="+",required=True)
    p.add_argument("--resamples",type=int,default=10000)
    p.add_argument("--seed",type=int,default=20260917)
    args=p.parse_args(argv)
    if args.resamples<1000:
        p.error("--resamples must be >=1000")
    sizes=sorted(set(args.active_sites))
    if len(sizes)<3:
        p.error("Scaling inference requires at least three active-site levels")
    raw_map=json.loads(args.cluster_map.read_text(encoding="utf-8"))
    cluster_map={str(k).lower():str(v) for k,v in raw_map.items()}
    case_paths=sorted((args.results_dir/"cases").glob("*.json"))
    if not case_paths:
        raise ValueError("No benchmark case JSON files")

    observations=[]
    failures=[]
    for path in case_paths:
        case=json.loads(path.read_text(encoding="utf-8"))
        cfg=case.get("config",{}) or {}
        if str(cfg.get("pruning",""))!=args.primary_pruning:
            continue
        if int(cfg.get("active_sites",-1)) not in sizes:
            continue
        if int(cfg.get("depth",-1))!=args.primary_depth:
            continue
        if int(cfg.get("max_evals",-1))!=args.primary_max_evals:
            continue
        if not math.isclose(float(cfg.get("radius",float("nan"))),args.primary_radius,rel_tol=0,abs_tol=1e-12):
            continue
        rows=[
            r for r in case.get("metrics",[])
            if r.get("budget_mode")=="matched_outputs"
            and int(r.get("outputs",0) or 0)==args.primary_outputs
        ]
        qrows=[
            r for r in rows if r.get("solver")=="qaoa"
            and r.get("qaoa_objective")==args.primary_objective
            and int(r.get("qaoa_restarts",0) or 0)==args.primary_restarts
        ]
        brows=[r for r in rows if r.get("solver")==args.baseline]
        if len(qrows)!=1 or len(brows)!=1:
            failures.append(f"{path.name}: missing/ambiguous primary QAOA or {args.baseline}")
            continue
        q,b=qrows[0],brows[0]
        if q.get("termination_reason")=="all_restarts_failed":
            failures.append(f"{path.name}: QAOA all_restarts_failed")
            continue
        pdb=str(cfg.get("pdb_id") or cfg.get("target") or "").strip().lower()
        if not pdb:
            failures.append(f"{path.name}: missing PDB identity");continue
        if pdb not in cluster_map:
            failures.append(f"{path.name}: cluster map missing {pdb}");continue
        config_count=int(q.get("configuration_count",0) or 0)
        num_bits=int(q.get("num_bits",0) or 0)
        if config_count<=0 or num_bits<=0:
            failures.append(f"{path.name}: invalid complexity metadata");continue
        if config_count != 3 ** int(cfg["active_sites"]):
            failures.append(f"{path.name}: feasible configuration count is not 3^active_sites");continue
        try:
            _validate_fixed_state_case(case,int(cfg["active_sites"]),num_bits)
        except (ValueError,KeyError,TypeError) as exc:
            failures.append(f"{path.name}: {exc}");continue
        observations.append(dict(
            pdb_id=pdb,cluster=cluster_map[pdb],seed=int(cfg.get("seed",0)),
            active_sites=int(cfg["active_sites"]),num_bits=num_bits,
            configuration_count=config_count,
            log10_configuration_count=math.log10(config_count),
            qaoa_gap=float(q["gap"]),baseline_gap=float(b["gap"]),
            delta_gap=float(q["gap"])-float(b["gap"]),
            qaoa_hit=float(q["hit"]),baseline_hit=float(b["hit"]),
            delta_hit=float(q["hit"])-float(b["hit"]),
            case_file=path.name,
        ))
    if failures:
        raise ValueError("Scaling primary denominator is incomplete: "+"; ".join(failures[:20]))
    if not observations:
        raise ValueError("No scaling observations after frozen primary filtering")

    by_pdb_size=defaultdict(list)
    for row in observations:
        by_pdb_size[(row["pdb_id"],row["active_sites"])].append(row)
    pdbs=sorted({row["pdb_id"] for row in observations})
    per_pdb=[]
    for pdb in pdbs:
        observed_sizes=sorted(size for (p,size) in by_pdb_size if p==pdb)
        if observed_sizes!=sizes:
            raise ValueError(f"{pdb}: scaling sizes {observed_sizes} != expected {sizes}")
        points=[]
        for size in sizes:
            group=by_pdb_size[(pdb,size)]
            points.append(dict(
                active_sites=size,
                num_bits=float(np.mean([r["num_bits"] for r in group])),
                log10_configuration_count=float(np.mean([r["log10_configuration_count"] for r in group])),
                delta_gap=float(np.mean([r["delta_gap"] for r in group])),
                delta_hit=float(np.mean([r["delta_hit"] for r in group])),
                repeats=len(group),
            ))
        per_pdb.append(dict(
            pdb_id=pdb,cluster=cluster_map[pdb],points=points,
            slope_log10_configuration_count=_slope(
                [x["log10_configuration_count"] for x in points],
                [x["delta_gap"] for x in points]),
            slope_num_bits=_slope([x["num_bits"] for x in points],[x["delta_gap"] for x in points]),
            slope_active_sites=_slope([x["active_sites"] for x in points],[x["delta_gap"] for x in points]),
        ))
    if any(row["slope_log10_configuration_count"] is None for row in per_pdb):
        bad=[r["pdb_id"] for r in per_pdb if r["slope_log10_configuration_count"] is None]
        raise ValueError(f"Cannot estimate primary scaling slope for PDBs: {bad[:20]}")

    by_cluster=defaultdict(list)
    for row in per_pdb:
        by_cluster[row["cluster"]].append(row)
    cluster_slopes=[]
    cluster_details=[]
    for cluster,rows in sorted(by_cluster.items()):
        values=[float(r["slope_log10_configuration_count"]) for r in rows]
        slope=float(np.mean(values))
        cluster_slopes.append(slope)
        cluster_details.append(dict(cluster=cluster,pdb_count=len(rows),mean_slope=slope))
    primary=_cluster_effect(cluster_slopes,args.seed,args.resamples)

    size_summary=[]
    for size in sizes:
        rows=[r for r in observations if r["active_sites"]==size]
        size_summary.append(dict(
            active_sites=size,n_rows=len(rows),n_pdb=len({r["pdb_id"] for r in rows}),
            mean_num_bits=float(np.mean([r["num_bits"] for r in rows])),
            mean_states_per_site=float(np.mean([r["num_bits"] / size for r in rows])),
            mean_log10_configuration_count=float(np.mean([r["log10_configuration_count"] for r in rows])),
            mean_delta_gap=float(np.mean([r["delta_gap"] for r in rows])),
            mean_delta_hit=float(np.mean([r["delta_hit"] for r in rows])),
        ))
    payload=dict(
        definition="within-PDB slope of QAOA-minus-classical energy gap versus log10 feasible configuration count under a fixed three-chi1-well state policy; PDB slopes averaged within family/structure cluster",
        sign_interpretation="negative slope means QAOA relative gap improves versus the classical baseline as complexity increases",
        simulator_scope="classical exact-subspace finite-shot QAOA; not hardware quantum speedup",
        primary_predictor="log10_configuration_count",
        primary_response=f"qaoa_gap_minus_{args.baseline}_gap",
        primary_pruning=args.primary_pruning,baseline=args.baseline,
        primary_outputs=args.primary_outputs,primary_objective=args.primary_objective,
        primary_restarts=args.primary_restarts,primary_depth=args.primary_depth,
        primary_max_evals=args.primary_max_evals,primary_radius=args.primary_radius,
        active_sites=sizes,resamples=args.resamples,seed=args.seed,
        primary=primary,size_summary=size_summary,per_pdb=per_pdb,
        cluster_details=cluster_details,observation_count=len(observations),
        case_source_sha256={p.name:sha256(p) for p in case_paths},
        cluster_map_sha256=sha256(args.cluster_map),
        analysis_code_sha256=sha256(Path(__file__)),
    )
    args.out_json.parent.mkdir(parents=True,exist_ok=True)
    args.out_json.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    lines=[
        "# Quantum-classical scaling analysis","",
        f"Primary pruning: {args.primary_pruning}; baseline: {args.baseline}; p={args.primary_depth}; "
        f"outputs={args.primary_outputs}; objective={args.primary_objective}; restarts={args.primary_restarts}.",
        f"Primary predictor: log10(feasible configuration count). Active-site levels: {sizes}.",
        "Negative slope means QAOA's energy-gap difference relative to the classical baseline becomes more favorable as complexity increases.",
        "",
        f"Independent family/structure clusters: {primary['n_clusters']}; mean cluster slope: {primary['mean_slope']}; "
        f"95% bootstrap CI: [{primary['ci_low']}, {primary['ci_high']}]; sign-flip p={primary['p_value']}.",
        "",
        "| Active sites | Rows | PDBs | Mean QUBO bits | Mean states/site | Mean log10(|Omega|) | Mean QAOA-baseline gap | Mean QAOA-baseline hit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in size_summary:
        lines.append(
            f"| {row['active_sites']} | {row['n_rows']} | {row['n_pdb']} | "
            f"{row['mean_num_bits']:.6g} | {row['mean_states_per_site']:.6g} | {row['mean_log10_configuration_count']:.6g} | "
            f"{row['mean_delta_gap']:.6g} | {row['mean_delta_hit']:.6g} |"
        )
    lines += ["",
        "Every size uses exactly three rotamer states per site, with one representative from each chi1 well. The configuration count therefore changes with site count at fixed state resolution. QAOA depth and optimizer evaluations remain fixed across sizes, so this estimates fixed-resource scaling rather than equal-compute scaling.",
        "This is a fixed-resource shallow-QAOA scaling analysis; it does not establish hardware quantum advantage or computational speedup.",
    ]
    args.out_md.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
