#!/usr/bin/env python3
"""Descriptive QAOA exploration: depth, optimizer budget and parameter transfer.

Pre-declared exploratory analyses (docs/PROTOCOL_AMENDMENTS.md A7). For each
depth p (optimizer budget proportional to the 2p parameters) it summarises, on
the held-out hard set at the primary size:

* exact log10 ground-state amplification over uniform feasible sampling for
  per-instance-trained QAOA and for QAOA run at transferred angles fitted on
  training graphs (Brandao et al. 2018; Galda et al. 2021),
* execution-only and training-inclusive log10 queries-to-solution (Ronnow et
  al. 2014; training vs execution separated as in Shaydulin et al. 2024),
* optimization shots per instance.

Values are averaged over repeats within PDB, then over PDBs within
family/structure cluster; intervals are 95% cluster-bootstrap percentile
intervals. No hypothesis tests are reported: these results are exploratory.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from nanoqc.common.repo_io import sha256_file as sha256

FIELDS = (
    ("trained_log10_amplification_exact", "qaoa", "log10_ground_amplification_exact"),
    ("transfer_log10_amplification_exact", "qaoa_transfer", "log10_ground_amplification_exact"),
    ("trained_log10_qts99_execution", "qaoa", "log10_qts99_execution"),
    ("transfer_log10_qts99_execution", "qaoa_transfer", "log10_qts99_execution"),
    ("trained_log10_qts99_with_training", "qaoa", "log10_qts99"),
    ("trained_optimization_shots", "qaoa", "total_opt_shots"),
)


def cluster_summary(values_by_cluster: dict, rng: np.random.Generator, resamples: int) -> dict:
    values = np.asarray([float(np.mean(v)) for _, v in sorted(values_by_cluster.items())], dtype=float)
    if not len(values):
        return dict(n_clusters=0, mean=None, ci_low=None, ci_high=None)
    result = dict(n_clusters=int(len(values)), mean=float(values.mean()), ci_low=None, ci_high=None)
    if len(values) >= 2:
        draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
        result.update(ci_low=float(np.percentile(draws, 2.5)), ci_high=float(np.percentile(draws, 97.5)))
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-dirs", type=Path, nargs="+", required=True,
                        help="One benchmark output directory per depth (hard set, with transfer rows)")
    parser.add_argument("--cluster-map", type=Path, required=True)
    parser.add_argument("--primary-outputs", type=int, required=True)
    parser.add_argument("--objective", choices=("mean", "cvar"), required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--active-sites", type=int, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)
    cluster_map = {str(k).lower(): str(v) for k, v in
                   json.loads(args.cluster_map.read_text(encoding="utf-8")).items()}
    rng = np.random.default_rng(args.seed)

    per_depth = []
    for directory in args.depth_dirs:
        by_pdb = defaultdict(lambda: defaultdict(list))
        depth = None
        max_evals = None
        failures = []
        for path in sorted((directory / "cases").glob("*.json")):
            case = json.loads(path.read_text(encoding="utf-8"))
            config = case["config"]
            if int(config["active_sites"]) != args.active_sites:
                continue
            depth = int(config["depth"]) if depth is None else depth
            max_evals = int(config["max_evals"]) if max_evals is None else max_evals
            if int(config["depth"]) != depth:
                raise ValueError(f"{directory}: mixed depths; pass one directory per depth")
            rows = {}
            for row in case.get("metrics", []):
                if row.get("budget_mode") != "matched_outputs" or int(row.get("outputs", 0) or 0) != args.primary_outputs:
                    continue
                if row["solver"] == "qaoa" and (row.get("qaoa_objective") != args.objective
                                                or row.get("qaoa_restarts") != args.restarts):
                    continue
                if row["solver"] in ("qaoa", "qaoa_transfer"):
                    rows[row["solver"]] = row
            if set(rows) != {"qaoa", "qaoa_transfer"}:
                failures.append(f"{path.name}: missing trained or transferred QAOA row")
                continue
            pdb = str(config.get("pdb_id") or config.get("target")).lower()
            if pdb not in cluster_map:
                raise ValueError(f"Cluster map lacks {pdb}")
            for name, solver, key in FIELDS:
                value = float(rows[solver][key])
                if not math.isfinite(value):
                    failures.append(f"{path.name}: nonfinite {key}")
                    break
                by_pdb[pdb][name].append(value)
            by_pdb[pdb]["transfer_minus_trained_log10_amplification_exact"].append(
                float(rows["qaoa_transfer"]["log10_ground_amplification_exact"])
                - float(rows["qaoa"]["log10_ground_amplification_exact"]))
        if failures:
            raise ValueError(f"{directory}: incomplete exploration denominator: {failures[:10]}")
        if depth is None:
            raise ValueError(f"{directory}: no cases at {args.active_sites} active sites")
        summary = dict(depth=depth, max_evals=max_evals, n_pdb=len(by_pdb))
        names = [name for name, _, _ in FIELDS] + ["transfer_minus_trained_log10_amplification_exact"]
        for name in names:
            by_cluster = defaultdict(list)
            for pdb, values in by_pdb.items():
                by_cluster[cluster_map[pdb]].append(float(np.mean(values[name])))
            summary[name] = cluster_summary(by_cluster, rng, args.resamples)
        per_depth.append(summary)
    per_depth.sort(key=lambda row: row["depth"])

    payload = dict(
        schema="quantum_exploration_v1", exploratory=True, hypothesis_tests=None,
        active_sites=args.active_sites, primary_outputs=args.primary_outputs,
        objective=args.objective, restarts=args.restarts, resamples=args.resamples,
        per_depth=per_depth,
        interpretation=("Exploratory, descriptive estimates with 95% cluster-bootstrap intervals; "
                        "noiseless exact-subspace simulation, not hardware results."),
        cluster_map_sha256=sha256(args.cluster_map),
        case_sources={str(d): sha256(d / "run_manifest.json") for d in args.depth_dirs},
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fmt(entry: dict) -> str:
        if entry["mean"] is None:
            return "n/a"
        interval = "" if entry["ci_low"] is None else f" [{entry['ci_low']:.3g}, {entry['ci_high']:.3g}]"
        return f"{entry['mean']:.3g}{interval}"

    lines = [
        "# QAOA exploration: depth, optimizer budget and parameter transfer", "",
        f"Primary size {args.active_sites} sites; {args.primary_outputs} output shots; objective={args.objective}, "
        f"restarts={args.restarts}. Exploratory and descriptive: cluster means with 95% cluster-bootstrap "
        "intervals, no hypothesis tests. Amplification = exact ground-state probability / uniform (log10).", "",
        "| p | max_evals | PDBs | Trained log10 amplification | Transferred log10 amplification | "
        "Transfer - trained | Trained log10 QTS99 (execution) | Transferred log10 QTS99 (execution) | "
        "Trained log10 QTS99 (with training) | Optimization shots |",
        "|---:|---:|---:|---|---|---|---|---|---|---|",
    ]
    for row in per_depth:
        lines.append(
            f"| {row['depth']} | {row['max_evals']} | {row['n_pdb']} | "
            f"{fmt(row['trained_log10_amplification_exact'])} | {fmt(row['transfer_log10_amplification_exact'])} | "
            f"{fmt(row['transfer_minus_trained_log10_amplification_exact'])} | "
            f"{fmt(row['trained_log10_qts99_execution'])} | {fmt(row['transfer_log10_qts99_execution'])} | "
            f"{fmt(row['trained_log10_qts99_with_training'])} | {fmt(row['trained_optimization_shots'])} |")
    lines += ["", payload["interpretation"]]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
