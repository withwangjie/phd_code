"""``--paired-statistics`` mode: paired, cluster-level inference on benchmark results.
"""
from __future__ import annotations

import argparse
import math
import os
import json
from filelock import FileLock
from nanoqc.common.seed_streams import DEFAULT_MASTER_SEED
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
from collections import Counter
from nanoqc.common.repo_io import sha256_file as _ablation_digest, atomic_write_json_fsync as _ablation_atomic_json
from nanoqc.inference.paired_statistics import (
    paired_effect as _paired_effect,
    holm_adjust as _holm_adjust,
    paired_denominator_failures,
)



def _paired_statistics_main(argv: Optional[Sequence[str]] = None) -> int:
    """Analyse complete within-case pairs; average repeats within PDB/cluster."""
    parser = argparse.ArgumentParser(description="Formal pre-registered paired cluster statistics")
    parser.add_argument("--results-dir",type=Path,required=True)
    parser.add_argument("--resamples",type=int,default=10000)
    parser.add_argument("--seed",type=int,default=DEFAULT_MASTER_SEED)
    parser.add_argument("--cluster-map",type=Path,help="JSON mapping every PDB ID to antigen/sequence-family cluster")
    parser.add_argument("--budget-mode",choices=["outputs","time"],default="outputs")
    parser.add_argument("--primary-pruning", type=str, default="egnn")
    parser.add_argument("--primary-radius", type=float, default=6.0)
    parser.add_argument("--primary-depth", type=int, default=2)
    parser.add_argument("--primary-max-evals", type=int, default=90)
    parser.add_argument("--primary-outputs", type=int, default=1000)
    parser.add_argument("--primary-objective", choices=("mean", "cvar"), default="cvar")
    parser.add_argument("--primary-restarts", type=int, default=4)
    parser.add_argument("--primary-active-sites", type=int, default=6)
    parser.add_argument("--max-time-overrun-fraction",type=float,default=.10)
    args=parser.parse_args(argv)
    if args.resamples < 100 or not math.isfinite(args.max_time_overrun_fraction) or args.max_time_overrun_fraction<0:
        parser.error("Invalid resampling/overrun settings")
    paths=sorted((args.results_dir/"cases").glob("*.json"))
    if not paths:
        parser.error("No case JSON artifacts")
    cluster_map=json.loads(args.cluster_map.read_text()) if args.cluster_map else None
    grouped={}; skipped=Counter(); pair_count=Counter()
    metrics=("gap","hit","ground_probability","low_energy_mass","low_energy_coverage","entropy","log10_qts99")
    for path in paths:
        case=json.loads(path.read_text())
        # Explicit primary output-budget contrast; never silently overwrite
        # all objective/restart/budget variants in a solver-keyed dictionary.
        if str(case.get("config",{}).get("pruning","")) != args.primary_pruning:
            skipped["nonprimary_pruning"] += 1
            continue
        config=case.get("config",{}) or {}
        if not math.isclose(float(config.get("radius",float("nan"))),args.primary_radius,rel_tol=0,abs_tol=1e-12):
            skipped["nonprimary_radius"] += 1
            continue
        if int(config.get("depth",-1)) != args.primary_depth:
            skipped["nonprimary_depth"] += 1
            continue
        if int(config.get("max_evals",-1)) != args.primary_max_evals:
            skipped["nonprimary_max_evals"] += 1
            continue
        if int(case.get("config",{}).get("active_sites",args.primary_active_sites)) != args.primary_active_sites:
            skipped["nonprimary_active_sites"] += 1
            continue
        selected = [r for r in case["metrics"] if
                    (r.get("reference_outputs", r.get("outputs")) if r["solver"].endswith("_time")
                     else r.get("outputs")) == args.primary_outputs]
        rows = {r["solver"]: r for r in selected if r["solver"] != "qaoa"}
        qrows = [r for r in selected if r["solver"] == "qaoa"]
        qrows = [r for r in qrows if
                 r.get("qaoa_objective") == args.primary_objective and
                 r.get("qaoa_restarts") == args.primary_restarts]
        if len(qrows) != 1:
            skipped["missing_or_ambiguous_primary_contrast"] += 1
            continue
        rows["qaoa"] = qrows[0]
        if rows["qaoa"].get("termination_reason") == "all_restarts_failed":
            skipped["qaoa:all_restarts_failed"] += 1
            continue
        pdb=str(
            case["config"].get("pdb_id") or case["config"].get("target") or ""
        ).strip().lower()
        if not pdb:
            raise ValueError(f"Missing PDB identity: {path}")
        if cluster_map is not None and pdb not in cluster_map:
            raise ValueError(f"Missing cluster mapping: {pdb}")
        cluster=str(cluster_map[pdb]) if cluster_map is not None else pdb
        for baseline in ("sa","uniform","greedy"):
            name=baseline if args.budget_mode=="outputs" else baseline+"_time"
            a,b=rows.get("qaoa"),rows.get(name)
            if a is None or b is None:
                skipped[name+":missing_pair"]+=1; continue
            if args.budget_mode=="outputs" and a["outputs"]!=b["outputs"]:
                skipped[name+":unequal_outputs"]+=1; continue
            if args.budget_mode=="time":
                budget=b.get("budget_seconds")
                if not budget or abs(budget-a["solver_seconds"])>1e-6*max(1.,budget):
                    skipped[name+":invalid_budget"]+=1; continue
                if b["budget_overrun_seconds"]>budget*args.max_time_overrun_fraction:
                    skipped[name+":overrun"]+=1; continue
            pair_count[name]+=1
            for metric in metrics:
                # Unequal read counts bias empirical diversity/coverage; do not test them in time mode.
                if args.budget_mode=="time" and metric not in ("gap","hit","log10_qts99"):
                    continue
                av,bv=a.get(metric),b.get(metric)
                if av is None or bv is None or not np.isfinite([av,bv]).all():
                    skipped[name+":"+metric+":nonfinite"]+=1; continue
                grouped.setdefault((name,metric),{}).setdefault(cluster,{}).setdefault(pdb,[]).append(float(av)-float(bv))
    effects=[]; cluster_values=[]
    for (name,metric),clusters in sorted(grouped.items()):
        values=[]
        for cluster,pdbs in sorted(clusters.items()):
            # Equal weight to PDBs within families; repeated configs/seeds average within PDB.
            value=float(np.mean([np.mean(v) for v in pdbs.values()]))
            values.append(value)
            cluster_values.append(dict(baseline=name,metric=metric,cluster=cluster,difference=value,
                pdb_count=len(pdbs),paired_cases=sum(len(v) for v in pdbs.values())))
        effects.append(dict(baseline=name,metric=metric,**_paired_effect(values,args.seed,args.resamples)))
    tested=[e for e in effects if e["p_value"] is not None]
    for effect,pvalue in zip(tested,_holm_adjust([e["p_value"] for e in tested])):
        effect["p_holm"] = pvalue
    payload=dict(budget_mode=args.budget_mode,seed=args.seed,resamples=args.resamples,
        primary_pruning=args.primary_pruning, primary_radius=args.primary_radius,
        primary_depth=args.primary_depth, primary_max_evals=args.primary_max_evals,
        primary_outputs=args.primary_outputs, primary_objective=args.primary_objective,
        primary_restarts=args.primary_restarts, primary_active_sites=args.primary_active_sites,
        cluster_unit="provided family clusters" if cluster_map is not None else "PDB (homology dependence unresolved)",
        effects=effects,cluster_differences=cluster_values,paired_cases=dict(pair_count),exclusions=dict(skipped),
        denominator_failures=paired_denominator_failures(dict(skipped),args.budget_mode),
        source_sha256={p.name:_ablation_digest(p) for p in paths},
        cluster_map_sha256=_ablation_digest(args.cluster_map) if args.cluster_map else None,
        max_time_overrun_fraction=args.max_time_overrun_fraction,
        analysis_code_sha256=_ablation_digest(Path(__file__)))
    lines=["# Exploratory paired statistics", "", "Differences = QAOA - classical; negative gap favours QAOA, positive hit favours QAOA.",
        f"Primary contrast: pruning={args.primary_pruning}, radius={args.primary_radius}, p={args.primary_depth}, max_evals={args.primary_max_evals}, active_sites={args.primary_active_sites}, outputs={args.primary_outputs}, objective={args.primary_objective}, restarts={args.primary_restarts}. Time-mode uses the explicitly recorded budget donor at the same output budget; all other dimensions remain ablation/scaling/raw records.",
        "Repeats averaged within PDB, then PDBs within supplied families. Equal cluster weighting.",
        "95% percentile bootstrap intervals are marginal, not simultaneous. Two-sided sign-flip p values assume exchangeability/symmetry; Holm correction covers every tested contrast in this report.",
        "PDB clusters do not remove homologous-family dependence. Small cluster counts give unreliable intervals. One cluster: no CI or p value.",
        "Same-output budgets are not equal compute. Time mode compares best gap/hit only, excludes excessive soft-deadline overruns; unequal read entropy/coverage is not tested.",
        "Exact landscape preprocessing is excluded from solver time; these classical simulator runs do not establish hardware quantum advantage.",
        "", "| Baseline | Metric | Clusters | Mean difference | 95% CI | p | Holm p |", "|---|---|---:|---:|---|---|---|"]
    for e in effects:
        lines.append(f"| {e['baseline']} | {e['metric']} | {e['n_clusters']} | {e['mean_difference']:.6g} | {e['ci_low']}, {e['ci_high']} | {e['p_value']} | {e.get('p_holm')} |")
    lines += ["", "Pair/exclusion counts: "+json.dumps(dict(pairs=dict(pair_count),excluded=dict(skipped))),
        "", "Methods reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html"]
    out=args.results_dir/("statistics_"+args.budget_mode)
    with FileLock(str(out)+".lock",timeout=0):
        _ablation_atomic_json(out.with_suffix(".json"),payload)
        temp=out.with_suffix(".md.tmp")
        temp.write_text("\n".join(lines),encoding="utf-8")
        os.replace(temp,out.with_suffix(".md"))
    print(out.with_suffix(".md").resolve())
    return 0
