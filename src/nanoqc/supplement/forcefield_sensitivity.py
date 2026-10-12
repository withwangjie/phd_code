#!/usr/bin/env python3
"""A64c: force-field sensitivity of the frozen structural experiment (exploratory).

The formal structural model is Amber ff14SB in vacuum with the A50 polar-
hydrogen Lennard-Jones shielding (a study-specific hybrid, A53). This driver
re-runs the frozen validation queue under two declared alternatives:

* ``charmm36`` -- the CHARMM36 protein force field as published (OpenMM
  charmm36.xml), whose polar hydrogens carry their own LJ terms; hydrogens are
  re-added from CHARMM templates, heavy atoms are unchanged;
* ``amber14_unshielded`` -- ff14SB without the A50 shielding, to show what the
  shielding changes.

Everything else is the formal protocol: the same recovery manifests, the same
seeds and derived optimize/measurement/sample sub-seeds, the same perturbed
input (the perturbation generator always uses the frozen model), the same
solvers and relaxation settings. Only the all-atom QUBO and relaxation use the
alternative force field. Results go to <run>/analysis/a64/forcefield/<arm>/.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

from nanoqc.common.seed_streams import derive_child_seed, derive_streams
from nanoqc.supplement.common import output_dir, write_csv

ARMS = ("charmm36", "amber14_unshielded")
# Reproduction control: the frozen model through this driver must equal the formal rows.
CONTROL_ARM = "amber14_a50"
METHODS = ("qaoa", "sa", "uniform", "greedy")


def load_run_config(run_dir: Path) -> dict:
    for candidate in ("provenance/resolved_runtime_config.yaml", "frozen_config.yaml"):
        path = Path(run_dir) / candidate
        if path.is_file():
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No resolved/frozen config in {run_dir}")


def job_argv(config: dict, manifest: Path, destination: Path, pdb: str, seeds: list[int], arm: str) -> list[str]:
    """The pilot's per-target recovery argv (run_real_complex_pilot.py), plus the force field."""
    from nanoqc.pipeline.orchestrator_common import quantum_primary
    cfg = config["structure_experiment"]
    q = quantum_primary(config)
    streams = derive_streams(config["master_seed"])
    optimize = [derive_child_seed(streams["optimize"], "structure", pdb, str(s)) for s in seeds]
    measurement = [derive_child_seed(streams["measurement"], "structure", pdb, str(s)) for s in seeds]
    sample = [derive_child_seed(streams["sample"], "structure", pdb, str(s)) for s in seeds]
    argv = ["--manifest", str(manifest), "--out-dir", str(destination),
            "--solvent-model", str(cfg.get("solvent_model", "vacuum")),
            "--structure-force-field", arm,
            "--perturbation-mode", str(cfg.get("perturbation_mode", "multi_chi")),
            "--perturbation-max-attempts", str(cfg.get("perturbation_max_attempts", 256)),
            "--min-perturb-degrees", str(cfg.get("min_perturb_degrees", 40.0)),
            "--max-perturb-degrees", str(cfg.get("max_perturb_degrees", 120.0)),
            "--min-input-heavy-distance", str((config.get("data_audit", {}) or {}).get(
                "min_interresidue_heavy_distance_angstrom", 1.0)),
            "--seeds", *[str(s) for s in seeds],
            "--optimize-seeds", *[str(s) for s in optimize],
            "--measurement-seeds", *[str(s) for s in measurement],
            "--sample-seeds", *[str(s) for s in sample],
            "--outputs", str(q.get("output_shots", 1000)), "--max-evals", str(q.get("max_evals", 90)),
            "--qaoa-depth", str(q.get("depth", 2)),
            "--relax-iterations", str(cfg.get("relax_iterations", 200)),
            "--loop-relax-iterations", str(cfg.get("loop_relax_iterations", 0))]
    if q.get("eval_shots"):
        argv += ["--eval-shots", str(q["eval_shots"])]
    if cfg.get("robust_qaoa", True):
        argv += ["--robust-qaoa", "--qaoa-restarts", str(q.get("restarts", 4)),
                 "--qaoa-objective", str(q.get("objective", "cvar")),
                 "--cvar-alpha", str(q.get("cvar_alpha", 0.1)),
                 "--parameter-scale", str(q.get("parameter_scale", "max_coefficient"))]
    return argv


def target_complete(destination: Path, seeds: list[int]) -> bool:
    """Every requested seed has a row for every method (the CSV is written incrementally)."""
    rows = read_rows(destination / "recovery_metrics.csv")
    present = {(str(r.get("seed")), r.get("method")) for r in rows}
    return all((str(seed), method) in present for seed in seeds for method in METHODS)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def compare(run_dir: Path, arms: list[str], seeds: list[int],
            targets: set[str] | None = None) -> tuple[list[dict], str]:
    """Paired rows (target, seed, method) across the formal model and each arm."""
    formal_root = Path(run_dir) / "validation_queue" / "results"
    rows = []
    for target_dir in sorted(p for p in formal_root.iterdir() if p.is_dir() and ".incomplete-" not in p.name):
        if targets is not None and target_dir.name not in targets:
            continue
        formal = {(r["seed"], r["method"]): r for r in read_rows(target_dir / "recovery_metrics.csv")}
        for arm in arms:
            arm_rows = {(r["seed"], r["method"]): r for r in read_rows(
                output_dir(run_dir, f"forcefield/{arm}") / "results" / target_dir.name / "recovery_metrics.csv")}
            for seed in seeds:
                for method in METHODS:
                    f, a = formal.get((str(seed), method)), arm_rows.get((str(seed), method))
                    rows.append(dict(
                        target=target_dir.name, seed=seed, method=method, arm=arm,
                        formal_status=None if f is None else f.get("evaluation_status"),
                        arm_status=None if a is None else a.get("evaluation_status"),
                        formal_final_rmsd=None if f is None else _float(f.get("final_rmsd")),
                        arm_final_rmsd=None if a is None else _float(a.get("final_rmsd")),
                        formal_chi1=None if f is None else _float(f.get("chi1_recovery_final")),
                        arm_chi1=None if a is None else _float(a.get("chi1_recovery_final")),
                        initial_rmsd=None if f is None else _float(f.get("initial_rmsd")),
                        arm_initial_rmsd=None if a is None else _float(a.get("initial_rmsd"))))
    return rows, _summary(rows, arms)


def _spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) < 3:
        return None
    return float(spearmanr(x, y).statistic)


def _summary(rows: list[dict], arms: list[str]) -> str:
    lines = ["# A64c Force-field sensitivity of the structural experiment (exploratory)", "",
             "Formal model: Amber ff14SB + A50 polar-hydrogen shielding (vacuum). Same manifests, seeds, "
             "perturbed inputs, solvers and relaxation; only the all-atom QUBO and relaxation change.", ""]
    for arm in arms:
        r = [x for x in rows if x["arm"] == arm]
        done = [x for x in r if x["arm_status"] is not None]
        passed = [x for x in done if x["arm_status"] == "passed"]
        lines += [f"## Arm: {arm}", "",
                  f"- Rows run: {len(done)}/{len(r)}; passed physical acceptance: {len(passed)}/{len(done)} "
                  f"(formal: {sum(x['formal_status'] == 'passed' for x in r)}/{len(r)}).",
                  f"- Identical perturbed input (initial RMSD equal to the formal row within 1e-6 A): "
                  f"{sum(1 for x in done if x['initial_rmsd'] is not None and x['arm_initial_rmsd'] is not None and abs(x['initial_rmsd'] - x['arm_initial_rmsd']) < 1e-6)}/{len(done)}.", "",
                  "| Method | Paired rows | Formal mean final RMSD (A) | Arm mean final RMSD (A) | Mean arm-formal (A) | Formal chi1 | Arm chi1 |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for method in METHODS:
            m = [x for x in passed if x["method"] == method and x["formal_status"] == "passed"
                 and x["formal_final_rmsd"] is not None and x["arm_final_rmsd"] is not None]
            if not m:
                lines.append(f"| {method} | 0 | n/a | n/a | n/a | n/a | n/a |"); continue
            lines.append(f"| {method} | {len(m)} | {np.mean([x['formal_final_rmsd'] for x in m]):.3f} | "
                         f"{np.mean([x['arm_final_rmsd'] for x in m]):.3f} | "
                         f"{np.mean([x['arm_final_rmsd'] - x['formal_final_rmsd'] for x in m]):+.3f} | "
                         f"{np.mean([x['formal_chi1'] for x in m]):.3f} | {np.mean([x['arm_chi1'] for x in m]):.3f} |")
        # QAOA - SA within each model, target-level means
        def contrast(key):
            per_target = {}
            for x in passed:
                if x["formal_status"] != "passed" or x[key] is None:
                    continue
                per_target.setdefault(x["target"], {}).setdefault(x["method"], []).append(x[key])
            diffs = [np.mean(v["qaoa"]) - np.mean(v["sa"]) for v in per_target.values() if "qaoa" in v and "sa" in v]
            return (float(np.mean(diffs)) if diffs else None), len(diffs)
        formal_diff, n_formal = contrast("formal_final_rmsd")
        arm_diff, n_arm = contrast("arm_final_rmsd")
        targets = sorted({x["target"] for x in passed})
        formal_t = [np.mean([x["formal_final_rmsd"] for x in passed if x["target"] == t and x["formal_final_rmsd"] is not None]) for t in targets]
        arm_t = [np.mean([x["arm_final_rmsd"] for x in passed if x["target"] == t and x["arm_final_rmsd"] is not None]) for t in targets]
        rho = _spearman(formal_t, arm_t)
        if arm == CONTROL_ARM:
            exact = sum(1 for x in passed if x["formal_final_rmsd"] is not None and x["arm_final_rmsd"] is not None
                        and abs(x["formal_final_rmsd"] - x["arm_final_rmsd"]) <= 1e-6)
            lines += ["", f"- Reproduction control: {exact}/{len(passed)} rows reproduce the formal final RMSD within 1e-6 A."]
        lines += ["", f"- QAOA - SA mean final RMSD (target level): formal {_fmt(formal_diff)} A over {n_formal} targets; "
                  f"{arm} {_fmt(arm_diff)} A over {n_arm} targets.",
                  f"- Spearman rank correlation of target mean final RMSD, formal vs {arm}: {_fmt(rho)} over {len(targets)} targets.", ""]
    lines += ["Exploratory (A64); descriptive, no hypothesis test; does not change any preregistered result."]
    return "\n".join(lines) + "\n"


def _fmt(value):
    return "n/a" if value is None else f"{value:+.4f}" if abs(value) < 1 else f"{value:.3f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=ARMS + (CONTROL_ARM,), default=list(ARMS))
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Restrict to these validation targets (e.g. one target for the control arm).")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Default: the run's frozen structure_experiment seeds.")
    parser.add_argument("--gpu-devices", nargs="+", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=None)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args(argv)
    config = load_run_config(args.run_dir)
    seeds = args.seeds or [int(s) for s in config["structure_experiment"].get("seeds", [42, 43, 44, 45, 46])]
    hardware = config.get("hardware", {}) or {}
    devices = args.gpu_devices or [str(d) for d in hardware.get(
        "structural_gpu_devices", [hardware.get("openmm_device", "0")])]
    per_gpu = args.workers_per_gpu or int(hardware.get("structural_workers_per_gpu", 1))
    manifests = sorted((Path(args.run_dir) / "validation_queue" / "prepared").glob("*/recovery_manifest.json"))
    if args.targets:
        wanted = {t.lower() for t in args.targets}
        manifests = [m for m in manifests if m.parent.name.lower() in wanted]
    if not manifests:
        parser.error("No frozen validation recovery manifests")
    started = time.time()
    if not args.compare_only:
        from nanoqc.pipeline.orchestrator_common import subprocess_environment
        os.environ.update(subprocess_environment(hardware, Path(config["paths"]["repo_root"])))
        from nanoqc.experiments.run_real_complex_pilot import (
            _run_recovery_on_device, _structural_worker_devices, _submit_structural_jobs)
        jobs = []
        for arm in args.arms:
            root = output_dir(args.run_dir, f"forcefield/{arm}") / "results"
            for manifest in manifests:
                pdb = manifest.parent.name
                destination = root / pdb
                if target_complete(destination, seeds):
                    continue  # resumable: a finished target is never re-run
                if destination.exists():
                    # The benchmark writes recovery_metrics.csv row by row, so an
                    # interrupted or failed attempt leaves a partial file. Keep it
                    # as evidence and re-run the target from scratch.
                    aside = destination.with_name(f"{destination.name}.incomplete-{time.strftime('%Y%m%d%H%M%S')}")
                    destination.rename(aside)
                    print(f"[forcefield] {arm}/{pdb}: incomplete earlier attempt moved to {aside.name}", flush=True)
                jobs.append((f"{arm}/{pdb}", destination, job_argv(config, manifest, destination, pdb, seeds, arm)))
        print(f"[forcefield] {len(jobs)} target jobs on GPUs {devices} x {per_gpu}", flush=True)
        if jobs:
            workers = _structural_worker_devices(min(len(jobs), len(devices) * per_gpu), devices, per_gpu)
            with contextlib.ExitStack() as stack:
                futures = (_submit_structural_jobs(jobs, workers, stack) if len(workers) > 1
                           else [(name, dest, argv, None) for name, dest, argv in jobs])
                for k, (name, destination, argv_, future) in enumerate(futures, 1):
                    try:
                        status = future.result() if future is not None else _run_recovery_on_device(argv_, workers[0])
                        print(f"[forcefield] {k}/{len(jobs)} {name} status={status} ({time.time()-started:.0f}s)", flush=True)
                    except Exception as exc:  # recorded, never hidden: the comparison shows the missing rows
                        print(f"[forcefield] {k}/{len(jobs)} {name} FAILED {type(exc).__name__}: {exc}", flush=True)
    rows, summary = compare(args.run_dir, args.arms, seeds,
                            targets={m.parent.name for m in manifests} if args.targets else None)
    out = output_dir(args.run_dir, "forcefield")
    write_csv(out / "forcefield_paired.csv", rows)
    (out / "forcefield_summary.md").write_text(summary, encoding="utf-8")
    (out / "forcefield_settings.json").write_text(json.dumps(dict(
        arms=args.arms, seeds=seeds, gpu_devices=devices, workers_per_gpu=per_gpu,
        manifests=len(manifests)), indent=2) + "\n", encoding="utf-8")
    print(f"[forcefield] done in {time.time()-started:.0f}s -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
