"""Re-run the relax-only control of saved seeds with the current code, in parallel.

Read-only with respect to the run: every seed is rebuilt from its own
``experiment.json`` and ``experiment/run_manifest.json`` arguments (seed,
relaxation cap, solvent, loop relaxation) exactly as the structure stage
does, and the new outputs go to ``--out``. The formal run's files, hashes
and acceptance records are never touched.

Seeds run in spawned worker processes, each pinned to one GPU of the frozen
hardware section (``structural_gpu_devices`` x ``structural_workers_per_gpu``
unless overridden). Results stream to ``results.jsonl`` as each seed
finishes, with a progress line and ETA, so an interrupted diagnostic resumes
where it stopped. A seed that raises is recorded as an error; it does not
stop the others.

    python -m nanoqc.experiments.relax_only_diagnostic RUN_DIR --out /tmp/a46 --select failed
    python -m nanoqc.experiments.relax_only_diagnostic RUN_DIR --out /tmp/a46 --seeds 7nxx/seed_43 7zra/seed_44
    python -m nanoqc.experiments.relax_only_diagnostic RUN_DIR --out /tmp/a46 --select passed --limit 24
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_DEVICE = None


def _init_worker(devices) -> None:
    global _DEVICE
    _DEVICE = devices.get()
    os.environ["QP_OPENMM_DEVICE"] = _DEVICE


def _seed_dirs(run: Path) -> list[Path]:
    root = run / "validation_queue" / "results"
    return sorted(p.parent.parent for p in root.glob("*/seed_*/experiment/relax_only_result.json"))


def _key(seed_dir: Path) -> str:
    return f"{seed_dir.parent.name}/{seed_dir.name}"


def _old_record(seed_dir: Path) -> dict:
    old = json.loads((seed_dir / "experiment" / "relax_only_result.json").read_text())
    relax = old.get("relaxation", {})
    return dict(status=old.get("evaluation_status"), reasons=old.get("evaluation_failure_reasons"),
                force_rms=old.get("movable_force_rms_kj_mol_nm"),
                stop_reason=relax.get("minimizer_stop_reason"),
                iterations=relax.get("minimizer_iterations"))


def _run_seed(seed_dir: str, out_dir: str) -> dict:
    """One seed's relax-only control, built the way structure_benchmarks builds it."""
    from nanoqc.experiments.structure_benchmarks import _structure_quality_assessment
    from nanoqc.qubo.subgraph_to_qubo import AllAtomInterfaceQUBOBuilder
    seed_dir, out = Path(seed_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    manifest = seed_dir / "experiment.json"
    case = json.loads(manifest.read_text())
    arguments = json.loads((seed_dir / "experiment" / "run_manifest.json").read_text())["arguments"]
    rotamer = case.get("rotamer_model", {}) or {}
    builder = AllAtomInterfaceQUBOBuilder(
        (manifest.parent / case["input_structure"]).resolve(), case["active_residues"],
        seed=int(arguments["seed"]), chi1_angles=case.get("chi1_angles"),
        site_scores=case.get("active_site_scores"),
        candidate_relax_iterations=int(case.get("candidate_relax_iterations", 0)),
        rotamer_mode=rotamer.get("mode", "dunbrack2010"),
        rotamer_library_path=rotamer.get("library_path"),
        rotamer_probability_floor=float(rotamer.get("probability_floor", 1e-4)),
        rotamer_sigma_offsets=rotamer.get("sigma_offsets", [-1.0, 0.0, 1.0]),
        solvent_model=arguments.get("solvent_model", "vacuum"))
    # Direct check of GPU fixed-point force saturation (A46) on the input.
    import openmm as mm
    from openmm import unit
    reference = mm.Context(builder.system, mm.VerletIntegrator(0.001),
                           mm.Platform.getPlatformByName("Reference"))
    movable = sorted(builder.movable)
    platform_forces = {}
    for name, context in (("platform", builder.context), ("reference", reference)):
        context.setPositions(builder.base_positions * unit.nanometer)
        forces = context.getState(getForces=True).getForces(asNumpy=True).value_in_unit(
            unit.kilojoules_per_mole / unit.nanometer)
        platform_forces[name] = float(abs(forces[movable]).max())
    del reference
    path = out / "relax_only.cif"
    relax = builder.relax_positions(builder.base_positions, path,
                                    minimize_iterations=int(arguments["relax_iterations"]))
    if int(arguments.get("loop_relax_iterations", 0)):
        relax.update(builder.relax_cdr_loop(path, case.get("cdr3_residues", []),
                                            iterations=int(arguments["loop_relax_iterations"])))
    quality = _structure_quality_assessment(relax)
    geometry = relax.get("stage2_physical_quality", relax["physical_quality_after"])
    record = dict(status=quality["evaluation_status"], reasons=quality["evaluation_failure_reasons"],
                  force_rms=quality["movable_force_rms_kj_mol_nm"],
                  force_max=quality["movable_force_max_kj_mol_nm"],
                  closest_pair=geometry.get("closest_nonbonded_pair"),
                  # Input after hydrogen addition: tells an overlap the input
                  # carried apart from one the relaxation created.
                  closest_pair_before=relax["physical_quality_before"].get("closest_nonbonded_pair"),
                  overlaps_before=relax["physical_quality_before"].get("extreme_nonbonded_pair_count"),
                  stop_reason=relax.get("minimizer_stop_reason"),
                  iterations=relax.get("minimizer_iterations"),
                  minimizer=relax.get("minimizer"), device=_DEVICE,
                  reference_evaluations=relax.get("minimizer_reference_platform_evaluations"),
                  input_max_force_component=platform_forces,
                  seconds=round(time.time() - started, 1))
    (out / "relax_only_result.json").write_text(json.dumps(dict(relaxation=relax, **quality),
                                                           indent=2, default=str))
    return record


def _classify(old: dict, new: dict) -> str:
    if new.get("error"):
        return "error"
    if old["status"] == "failed":
        return "fixed" if new["status"] == "passed" else "still_failing"
    return "still_passing" if new["status"] == "passed" else "regressed"


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def main(argv=None) -> int:
    import yaml
    from nanoqc.pipeline.orchestrator_common import subprocess_environment
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", help="target/seed_N; overrides --select")
    parser.add_argument("--select", choices=("failed", "passed", "all"), default="failed",
                        help="by the saved relax-only status in the run")
    parser.add_argument("--limit", type=int, default=0, help="first N selected seeds (0: all)")
    parser.add_argument("--workers-per-gpu", type=int, default=0,
                        help="default: the run's structural_workers_per_gpu")
    args = parser.parse_args(argv)
    run, out = args.run.resolve(), args.out.resolve()
    if out == run or run in out.parents:
        parser.error("--out must be outside the run directory")
    hardware = (yaml.safe_load((run / "frozen_config.yaml").read_text()) or {}).get("hardware", {}) or {}
    os.environ.update(subprocess_environment(hardware, REPO))

    available = {_key(d): d for d in _seed_dirs(run)}
    if args.seeds:
        missing = [s for s in args.seeds if s not in available]
        if missing:
            parser.error(f"not in the run: {missing}")
        chosen = [available[s] for s in args.seeds]
    else:
        chosen = [d for d in available.values()
                  if args.select == "all" or _old_record(d)["status"] == args.select]
    if args.limit > 0:
        chosen = chosen[:args.limit]

    out.mkdir(parents=True, exist_ok=True)
    ledger = out / "results.jsonl"
    done = {}
    if ledger.exists():
        for line in ledger.read_text().splitlines():
            row = json.loads(line)
            done[row["seed"]] = row
    todo = [d for d in chosen if _key(d) not in done]

    platform = os.environ["QP_OPENMM_PLATFORM"]
    if platform == "CUDA":
        gpus = [str(g) for g in hardware.get("structural_gpu_devices") or [os.environ["QP_OPENMM_DEVICE"]]]
        per_gpu = args.workers_per_gpu or int(hardware.get("structural_workers_per_gpu", 1))
        devices = [g for _ in range(per_gpu) for g in gpus]
    else:
        devices = [os.environ["QP_OPENMM_DEVICE"]] * max(1, args.workers_per_gpu or 1)
    workers = max(1, min(len(devices), len(todo)))
    print(f"OpenMM {platform}/{os.environ['QP_OPENMM_PRECISION']} | seeds: {len(chosen)} selected, "
          f"{len(chosen) - len(todo)} already done, {len(todo)} to run | "
          f"{workers} worker processes on devices {devices[:workers]}", flush=True)

    context = mp.get_context("spawn")
    queue = context.Queue()
    for device in devices[:workers]:
        queue.put(device)
    started, finished = time.time(), 0
    if todo:
        with concurrent.futures.ProcessPoolExecutor(workers, mp_context=context,
                                                    initializer=_init_worker,
                                                    initargs=(queue,)) as pool:
            futures = {pool.submit(_run_seed, str(d), str(out / _key(d))): d for d in todo}
            for future in concurrent.futures.as_completed(futures):
                seed_dir = futures[future]
                try:
                    new = future.result()
                except Exception as exc:  # record and continue with the other seeds
                    new = dict(error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc(limit=5))
                old = _old_record(seed_dir)
                row = dict(seed=_key(seed_dir), outcome=_classify(old, new), old=old, new=new)
                with ledger.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, default=str) + "\n")
                done[row["seed"]] = row
                finished += 1
                elapsed = time.time() - started
                eta = elapsed / finished * (len(todo) - finished)
                pair = (new.get("closest_pair") or {})
                before = (new.get("closest_pair_before") or {})
                detail = (new["error"] if "error" in new else
                          f"{new['status']} rms={new['force_rms']:.3g} "
                          f"closest {before.get('distance_angstrom', float('nan')):.2f}->"
                          f"{pair.get('distance_angstrom', float('nan')):.2f}A "
                          f"iters={new['iterations']} ref_evals={new['reference_evaluations']} "
                          f"input max|F| gpu={new['input_max_force_component']['platform']:.3g} "
                          f"ref={new['input_max_force_component']['reference']:.3g} "
                          f"{new['seconds']}s gpu{new['device']}")
                print(f"[{finished}/{len(todo)}] {row['seed']:<16} {row['outcome']:<14} {detail} | "
                      f"elapsed {_fmt(elapsed)} eta {_fmt(eta)}", flush=True)

    rows = [done[_key(d)] for d in chosen if _key(d) in done]
    counts = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    print("\nSummary:", json.dumps(counts))
    for row in rows:
        if row["outcome"] in ("still_failing", "regressed", "error"):
            new = row["new"]
            print(f"  {row['outcome']:<14} {row['seed']:<16} {new.get('error') or new['reasons']}\n"
                  f"      input closest={new.get('closest_pair_before')}\n"
                  f"      final closest={new.get('closest_pair')}")
    (out / "summary.json").write_text(json.dumps(dict(counts=counts, rows=rows), indent=2, default=str))
    print(f"Ledger: {ledger}")
    return 1 if any(r["outcome"] in ("still_failing", "regressed", "error") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
