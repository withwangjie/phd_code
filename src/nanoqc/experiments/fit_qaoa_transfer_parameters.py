#!/usr/bin/env python3
"""Fit transferable QAOA angles on TRAINING graphs only.

Optimal QAOA angles concentrate across typical instances of a problem class
(Brandao et al. 2018) and can be transferred between instances without
re-optimization (Galda et al. 2021; Shaydulin et al. 2024 use fixed angles).
This module reads a benchmark run on the training split, collects each case's
optimized angles in instance-independent units (gamma multiplied by the
optimizer's parameter scale; beta unchanged), and stores their component-wise
median per (depth, active sites). Test/validation graphs are never read.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from nanoqc.common.repo_io import sha256_file as sha256


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Benchmark output directory produced on graphs/train")
    parser.add_argument("--objective", choices=("mean", "cvar"), required=True)
    parser.add_argument("--restarts", type=int, required=True)
    parser.add_argument("--min-instances", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = json.loads((args.results_dir / "run_manifest.json").read_text(encoding="utf-8"))
    input_dir = Path(str(manifest.get("arguments", {}).get("input_dir", "")))
    if input_dir.name != "train":
        raise ValueError(f"Transfer angles must be fitted on the train split, got input_dir={input_dir}")
    parameter_scale_mode = str(manifest["arguments"].get("parameter_scale", "max_coefficient"))

    collected = defaultdict(list)
    shots = defaultdict(int)
    wanted = f"qaoa_obj{args.objective}_restarts{args.restarts}_outputs"
    for path in sorted((args.results_dir / "cases").glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        config = case["config"]
        key = f"p{int(config['depth'])}_sites{int(config['active_sites'])}"
        entries = [v for k, v in (case.get("optimizations") or {}).items() if k.startswith(wanted)]
        if not entries or entries[0].get("termination_reason") == "all_restarts_failed":
            continue
        entry = entries[0]  # reused across the output curve: one optimization per case
        collected[key].append((np.asarray(entry["internal_gammas"], float), np.asarray(entry["betas"], float)))
        rows = [r for r in case.get("metrics", []) if r.get("solver") == "qaoa"
                and r.get("qaoa_objective") == args.objective and r.get("qaoa_restarts") == args.restarts]
        if rows:
            shots[key] += int(rows[0].get("total_opt_shots", 0) or 0)

    entries = {}
    for key, values in sorted(collected.items()):
        if len(values) < args.min_instances:
            raise ValueError(f"{key}: only {len(values)} training instances (< {args.min_instances})")
        gammas = np.stack([g for g, _ in values])
        betas = np.stack([b for _, b in values])
        depth = int(key.split("_")[0][1:])
        entries[key] = dict(
            depth=depth, active_sites=int(key.split("sites")[1]), n_instances=len(values),
            internal_gammas=np.median(gammas, axis=0).tolist(), betas=np.median(betas, axis=0).tolist(),
            internal_gammas_iqr=np.subtract(*np.quantile(gammas, [.75, .25], axis=0)).tolist(),
            betas_iqr=np.subtract(*np.quantile(betas, [.75, .25], axis=0)).tolist(),
            training_shots_total=shots[key],
        )
    if not entries:
        raise ValueError("No usable training optimizations found")
    payload = dict(
        schema="qaoa_transfer_parameters_v1", fit_split="train", statistic="component-wise median",
        objective=args.objective, restarts=args.restarts, parameter_scale_mode=parameter_scale_mode,
        references=["Brandao et al. 2018 (arXiv:1812.04170)", "Galda et al. 2021 (arXiv:2106.07531)"],
        source_run_manifest_sha256=sha256(args.results_dir / "run_manifest.json"),
        entries=entries,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
