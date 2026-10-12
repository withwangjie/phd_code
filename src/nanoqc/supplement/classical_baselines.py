#!/usr/bin/env python3
"""A64b: budget-optimised classical baselines and exact solvers (exploratory).

The formal matched-output SA ran 100 site passes per read at every size and hit
the ground state with probability 1, so its cost reflects the chosen budget,
not what SA needs. Following the optimal time-to-solution convention (Ronnow
et al. 2014), SA is run here at several site-pass budgets and its TTS99 is the
minimum over budgets of

    reads99(passes) * queries_per_read(passes),
    reads99 = ceil(ln(0.01) / ln(1 - p_s)),  queries_per_read = 1 + passes * sites,

where p_s is the per-read probability of ending in a ground state, estimated
as (successes + 0.5) / (reads + 1) so that 0/R and R/R stay finite.

QAOA's TTS99 uses the exact noiseless ground-state probability of the frozen
fitted state: shots99 = ceil(ln(0.01)/ln(1 - p)), reported execution-only and
with the recorded optimisation shots added. Shots and classical energy queries
are different operations; the two are tabulated side by side, not equated.

Exact solvers: vectorised brute-force enumeration over the one-hot product
space and a mixed-integer program (scipy.optimize.milp / HiGHS) with the
standard product linearisation. Both must return the same optimum.
"""
from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np

from nanoqc.supplement.common import (
    GROUND_ENERGY_TOLERANCE, PRIMARY_OPTIMIZATION_KEY, output_dir, primary_cases,
    sampler_from_case, write_csv)

DEFAULT_PASSES = (1, 2, 5, 10, 20, 50, 100)
TARGET = 0.99


def reads_to_target(p: float, target: float = TARGET) -> float:
    if p >= 1.0:
        return 1.0
    if p <= 0.0:
        return math.inf
    return float(max(1, math.ceil(math.log(1.0 - target) / math.log(1.0 - p))))


def brute_force(physical_self: np.ndarray, physical_pair: np.ndarray, groups: list[list[int]]):
    """Exact minimum over the one-hot product space, vectorised by register."""
    started = time.perf_counter()
    choice = np.array(list(itertools.product(*groups)), dtype=np.int64)  # (|Omega|, sites)
    energy = physical_self[choice].sum(axis=1)
    upper = np.triu(physical_pair, 1) + np.tril(physical_pair, -1).T
    for s in range(choice.shape[1]):
        for t in range(s + 1, choice.shape[1]):
            a, b = choice[:, s], choice[:, t]
            energy = energy + upper[np.minimum(a, b), np.maximum(a, b)]
    best = int(np.argmin(energy))
    return float(energy[best]), choice[best].tolist(), time.perf_counter() - started, len(choice)


def exact_milp(physical_self: np.ndarray, physical_pair: np.ndarray, groups: list[list[int]]):
    """min h.x + sum P_ij y_ij, one-hot rows, y_ij = x_i x_j by linearisation."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix
    started = time.perf_counter()
    m = len(physical_self)
    site_of = {v: s for s, g in enumerate(groups) for v in g}
    upper = np.triu(physical_pair, 1) + np.tril(physical_pair, -1).T
    pairs = [(i, j) for i in range(m) for j in range(i + 1, m)
             if site_of[i] != site_of[j] and upper[i, j] != 0.0]
    n = m + len(pairs)
    c = np.concatenate([physical_self, [upper[i, j] for i, j in pairs]])
    rows = len(groups) + 3 * len(pairs)
    A = lil_matrix((rows, n)); lo = np.empty(rows); hi = np.empty(rows)
    r = 0
    for g in groups:
        for v in g:
            A[r, v] = 1.0
        lo[r] = hi[r] = 1.0; r += 1
    for k, (i, j) in enumerate(pairs):
        y = m + k
        A[r, y] = 1; A[r, i] = -1; lo[r], hi[r] = -np.inf, 0.0; r += 1          # y <= x_i
        A[r, y] = 1; A[r, j] = -1; lo[r], hi[r] = -np.inf, 0.0; r += 1          # y <= x_j
        A[r, y] = 1; A[r, i] = -1; A[r, j] = -1; lo[r], hi[r] = -1.0, np.inf; r += 1  # y >= x_i + x_j - 1
    integrality = np.concatenate([np.ones(m), np.zeros(len(pairs))])
    result = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi),
                  integrality=integrality, bounds=Bounds(0, 1))
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")
    x = np.round(result.x[:m]).astype(int)
    return float(result.fun), x.tolist(), time.perf_counter() - started


def sa_sweep(sampler, ground: float, passes_grid, reads: int, seed: int) -> list[dict]:
    rows = []
    sites = len(sampler.site_to_variables)
    for passes in passes_grid:
        started = time.perf_counter()
        result = sampler.simulated_annealing(num_reads=reads, site_passes=int(passes), seed=seed + int(passes))
        seconds = time.perf_counter() - started
        energies = np.asarray(result.read_energies, dtype=float)
        successes = int(np.sum(energies <= ground + GROUND_ENERGY_TOLERANCE * max(1.0, abs(ground))))
        p = (successes + 0.5) / (reads + 1.0)
        per_read = 1 + int(passes) * sites
        reads99 = reads_to_target(p)
        rows.append(dict(passes=int(passes), reads=reads, successes=successes, success_probability=p,
                         queries_per_read=per_read, reads99=reads99, tts99_queries=reads99 * per_read,
                         seconds_per_read=seconds / reads, tts99_seconds=reads99 * seconds / reads))
    return rows


def _case_task(task: tuple) -> dict:
    import json
    path, passes_grid, reads, seed = task
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    sampler = sampler_from_case(case)
    groups = [list(g) for g in sampler.site_to_variables.values()]
    h = np.asarray(sampler.physical_self, dtype=float)
    P = np.asarray(sampler.physical_pair, dtype=float)
    bf_energy, _, bf_seconds, omega = brute_force(h, P, groups)
    milp_energy, _, milp_seconds = exact_milp(h, P, groups)
    if abs(bf_energy - milp_energy) > 1e-6 * max(1.0, abs(bf_energy)):
        raise AssertionError(f"{Path(path).name}: MILP {milp_energy} != brute force {bf_energy}")
    sweep = sa_sweep(sampler, bf_energy, passes_grid, reads, seed)
    best = min(sweep, key=lambda r: r["tts99_queries"])
    metrics = next(m for m in case["metrics"] if m.get("solver") == "qaoa"
                   and m.get("qaoa_objective") == "cvar" and int(m.get("qaoa_restarts", 0)) == 4
                   and int(m.get("outputs", 0)) == 1000)
    p_qaoa = float(metrics["exact_ground_probability"])
    training = float(case["optimizations"][PRIMARY_OPTIMIZATION_KEY].get("total_opt_shots")
                     or metrics.get("total_opt_shots") or 0.0)
    shots99 = reads_to_target(p_qaoa)
    config = case["config"]
    return dict(
        case=Path(path).name, pdb=str(config.get("pdb_id", "")).lower(), active_sites=int(config["active_sites"]),
        configurations=omega, ground_energy_bruteforce=bf_energy,
        ground_energy_recorded=case.get("ground_energy"),
        recorded_minus_bruteforce=(None if case.get("ground_energy") is None
                                   else float(case["ground_energy"]) - bf_energy),
        bruteforce_seconds=bf_seconds, milp_seconds=milp_seconds,
        sa_best_passes=best["passes"], sa_best_success_probability=best["success_probability"],
        sa_tts99_queries=best["tts99_queries"], sa_tts99_seconds=best["tts99_seconds"],
        sa_formal_passes100_tts99_queries=next((r["tts99_queries"] for r in sweep if r["passes"] == 100), None),
        qaoa_exact_ground_probability=p_qaoa, qaoa_shots99_execution=shots99,
        qaoa_training_shots=training, qaoa_shots99_with_training=shots99 + training,
        sweep=sweep)


def _cluster_slope(rows: list[dict], key: str, cluster_of: dict, resamples: int, seed: int):
    """Mean within-cluster slope of log10(key) on log10(configurations) with a cluster bootstrap CI."""
    by_cluster: dict[str, dict[int, list]] = {}
    for r in rows:
        value = r[key]
        if value is None or not math.isfinite(value) or value <= 0:
            continue
        cluster = cluster_of.get(r["pdb"], r["pdb"])
        by_cluster.setdefault(cluster, {}).setdefault(r["active_sites"], []).append(
            (math.log10(r["configurations"]), math.log10(value)))
    slopes = []
    for sizes in by_cluster.values():
        points = [(np.mean([x for x, _ in v]), np.mean([y for _, y in v])) for v in sizes.values()]
        if len(points) >= 2:
            x, y = np.array(points).T
            slopes.append(float(np.polyfit(x, y, 1)[0]))
    if not slopes:
        return None, None, None, 0
    slopes = np.array(slopes); rng = np.random.default_rng(seed)
    boot = [rng.choice(slopes, len(slopes)).mean() for _ in range(resamples)]
    return float(slopes.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(slopes)


def main(argv=None) -> int:
    import json
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sites", type=int, nargs="+", default=[4, 6, 8, 10])
    parser.add_argument("--passes", type=int, nargs="+", default=list(DEFAULT_PASSES))
    parser.add_argument("--reads", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.reads < 10 or any(p <= 0 for p in args.passes):
        parser.error("reads >= 10 and positive passes required")
    out = output_dir(args.run_dir, "classical")
    tasks = [(str(path), list(args.passes), args.reads, args.seed + i)
             for i, (path, _) in enumerate(primary_cases(args.run_dir, set(args.sites)))]
    started = time.time(); results = []
    if args.workers > 1 and len(tasks) > 1:
        import concurrent.futures, multiprocessing as mp
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)),
                                                    mp_context=mp.get_context("spawn")) as pool:
            for k, row in enumerate(pool.map(_case_task, tasks), 1):
                results.append(row)
                if k % 20 == 0:
                    print(f"[classical] {k}/{len(tasks)} ({time.time()-started:.0f}s)", flush=True)
    else:
        for k, task in enumerate(tasks, 1):
            results.append(_case_task(task))
            if k % 20 == 0:
                print(f"[classical] {k}/{len(tasks)} ({time.time()-started:.0f}s)", flush=True)
    sweep_rows = [dict(case=r["case"], pdb=r["pdb"], active_sites=r["active_sites"], **s)
                  for r in results for s in r["sweep"]]
    case_rows = [{k: v for k, v in r.items() if k != "sweep"} for r in results]
    write_csv(out / "classical_cases.csv", case_rows)
    write_csv(out / "sa_budget_sweep.csv", sweep_rows)
    cluster_path = Path(args.run_dir) / "independence" / "pdb_family_clusters.json"
    cluster_of = ({str(k).lower(): str(v) for k, v in json.loads(cluster_path.read_text()).items()}
                  if cluster_path.is_file() else {})
    _write_summary(out, case_rows, cluster_of, args)
    print(f"[classical] {len(results)} cases in {time.time()-started:.0f}s -> {out}", flush=True)
    return 0


def _write_summary(out: Path, rows: list[dict], cluster_of: dict, args) -> None:
    lines = ["# A64b Budget-optimised classical baselines and exact solvers (exploratory)", "",
             "One case per PDB and size under the frozen primary filter. SA TTS99 is minimised over "
             f"site passes {args.passes} with {args.reads} reads each. QAOA TTS99 uses the exact noiseless "
             "ground probability of the frozen fitted state. Shots and energy queries are different "
             "operations and are not equated.", "",
             "| Sites | Cases | Mean configurations | Median SA best passes | Median SA TTS99 (queries) | "
             "Median SA TTS99 at formal 100 passes | Median QAOA shots99 (execution) | "
             "Median QAOA shots99 (with training) | Median brute force (s) | Median MILP (s) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for sites in sorted({r["active_sites"] for r in rows}):
        s = [r for r in rows if r["active_sites"] == sites]
        def med(k):
            values = [r[k] for r in s if r[k] is not None]
            return float(np.median(values)) if values else float("nan")
        lines.append(f"| {sites} | {len(s)} | {np.mean([r['configurations'] for r in s]):.0f} | {med('sa_best_passes'):g} | "
                     f"{med('sa_tts99_queries'):.3g} | {med('sa_formal_passes100_tts99_queries'):.3g} | "
                     f"{med('qaoa_shots99_execution'):.3g} | {med('qaoa_shots99_with_training'):.3g} | "
                     f"{med('bruteforce_seconds'):.3g} | {med('milp_seconds'):.3g} |")
    lines += ["", "## Scaling of TTS99 with log10(configurations) (within-cluster slopes, cluster bootstrap)", "",
              "| Quantity | Mean slope | 95% CI | Clusters |", "|---|---:|---|---:|"]
    for key, label in (("sa_tts99_queries", "SA optimal TTS99 (queries)"),
                       ("qaoa_shots99_execution", "QAOA shots99, execution only"),
                       ("qaoa_shots99_with_training", "QAOA shots99, with training"),
                       ("milp_seconds", "Exact MILP seconds")):
        mean, lo, hi, n = _cluster_slope(rows, key, cluster_of, args.resamples, args.seed)
        lines.append(f"| {label} | {mean if mean is None else f'{mean:.3f}'} | "
                     f"{'n/a' if lo is None else f'[{lo:.3f}, {hi:.3f}]'} | {n} |")
    mismatch = [r for r in rows if r["recorded_minus_bruteforce"] is not None
                and abs(r["recorded_minus_bruteforce"]) > 1e-6]
    lines += ["", f"Recorded ground energies differing from brute force by > 1e-6: {len(mismatch)}/{len(rows)}.",
              "MILP and brute force agree on every case (asserted).",
              "Exploratory (A64); descriptive slopes, no hypothesis test; does not change any preregistered result."]
    (out / "classical_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
