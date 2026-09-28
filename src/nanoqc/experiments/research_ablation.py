"""``--research-ablation`` mode: the matched QAOA-vs-classical ablation sweep.

Pruning strategy x radius x size x depth x budget x objective cases, each
sharing one perturbed input and candidate set across solvers, with every
solver's cost recorded so equal output is never read as equal budget. Also
holds time-to-solution accounting and QAOA angle transfer.
"""
from __future__ import annotations

import argparse
import csv
import concurrent.futures
import math
import multiprocessing as mp
import os
import sys
import time
import traceback
import json
import hashlib
from filelock import FileLock
from nanoqc.common.seed_streams import derive_streams, derive_child_seed, save_stream_map, DEFAULT_MASTER_SEED
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
import numpy as np
import torch
from nanoqc.data.safe_graph_load import load_graph
from tqdm.auto import tqdm
from nanoqc.model.model_egnn_pruning import (  # re-exported: historical import location
    assert_checkpoint_graph_compatible, load_interface_scorer,
)
from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler, GROUND_ENERGY_TOLERANCE, MAX_QAOA_DEPTH
from nanoqc.qubo.subgraph_to_qubo import InterfaceQUBOBuilder, ForceFieldConfig, EnergyCalibration
from nanoqc.quantum.resource_estimation import estimate_qaoa_resources
import itertools
import platform
from collections import Counter
from nanoqc.model.model_egnn_pruning import select_ablation_active, build_ablation_subgraph
from nanoqc.common.repo_io import sha256_file as _ablation_digest, atomic_write_json_fsync as _ablation_atomic_json, repo_path
from nanoqc.experiments.benchmark_common import SHARED_HELPER_MODULES
from nanoqc.experiments.calibration_fit import fit_energy_calibration_csv


def _ablation_classical_counts(sampler: Any, reads: int, seed: int,
                     greedy: bool, max_passes: int) -> tuple[dict, int]:
    """Uniform feasible sampling or multi-start coordinate descent."""
    rng = np.random.default_rng(seed)
    groups = list(sampler.site_to_variables.values())
    counts: Counter = Counter()
    evaluations = 0
    for _ in range(reads):
        chosen = [int(rng.choice(g)) for g in groups]
        bits = np.zeros(sampler.num_variables, dtype=np.int8)
        bits[chosen] = 1
        if greedy:
            current = sampler.physical_energy(bits)
            evaluations += 1
            for _ in range(max_passes):
                changed = False
                for site in rng.permutation(len(groups)):
                    old = chosen[site]
                    best, best_e = old, current
                    for proposal in groups[site]:
                        if proposal == old:
                            continue
                        trial = bits.copy()
                        trial[old], trial[proposal] = 0, 1
                        value = sampler.physical_energy(trial)
                        evaluations += 1
                        if value < best_e - 1e-12:
                            best, best_e = proposal, value
                    if best != old:
                        bits[old], bits[best] = 0, 1
                        chosen[site], current = best, best_e
                        changed = True
                if not changed:
                    break
        counts[tuple(map(int, bits))] += 1
    return dict(counts), evaluations


QTS_TARGET_CONFIDENCE = 0.99


def queries_to_solution(counts: dict, energies: dict, ground: float, *,
                        fixed_units: float, units_per_sample: float,
                        configuration_count: Optional[int] = None) -> dict:
    """Abstract time-to-solution under a declared shot/query conversion.

    Following the time-to-solution definition of Ronnow et al. (Science 2014),
    R99 = fixed + c * ln(1-0.99)/ln(1-p), the resources needed to observe a
    ground state at least once with 99% probability. One unit is one
    measurement shot or one single-state energy query by convention, so QAOA
    optimization shots (fixed) and SA per-read energy queries (c) are charged
    alike in this descriptive metric. They are different physical operations;
    this is not hardware-normalized cost or matched total computation.
    p uses the Jeffreys estimate (k+1/2)/(n+1) (Brown, Cai & DasGupta 2001),
    which stays finite for k=0 or k=n; unlike best-of-N gap/hit it does not
    saturate when a baseline always reaches the ground state.
    """
    n = int(sum(counts.values()))
    if n <= 0 or not math.isfinite(fixed_units) or fixed_units < 0 or not math.isfinite(units_per_sample) or units_per_sample <= 0:
        raise ValueError("queries_to_solution needs samples and finite nonnegative resources")
    k = int(sum(c for s, c in counts.items() if abs(energies[s] - ground) <= GROUND_ENERGY_TOLERANCE))
    p = (k + 0.5) / (n + 1.0)
    repetitions = max(1.0, math.log(1.0 - QTS_TARGET_CONFIDENCE) / math.log(1.0 - p))
    total = float(fixed_units) + float(units_per_sample) * repetitions
    execution = float(units_per_sample) * repetitions
    result = dict(ground_hits=k, success_probability_jeffreys=p,
                  resource_fixed_units=float(fixed_units), resource_units_per_sample=float(units_per_sample),
                  queries_to_solution_99=total, log10_qts99=math.log10(total),
                  # Execution-only cost: training/optimization shots excluded
                  # (training vs execution separated as in Shaydulin et al. 2024).
                  queries_to_solution_99_execution=execution,
                  log10_qts99_execution=math.log10(execution))
    if configuration_count is not None:
        if int(configuration_count) <= 0:
            raise ValueError("configuration_count must be positive")
        # Ground-state probability relative to uniform feasible sampling
        # (n_ground/|Omega|); 0 means no concentration beyond random.
        n_ground = sum(1 for e in energies.values() if abs(e - ground) <= GROUND_ENERGY_TOLERANCE)
        result["log10_ground_amplification"] = math.log10(p * int(configuration_count) / max(1, n_ground))
    return result


def load_transfer_parameters(path: Path) -> dict:
    """Read and validate transferred QAOA angles (see fit_qaoa_transfer_parameters)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "qaoa_transfer_parameters_v1" or payload.get("fit_split") != "train":
        raise ValueError("transfer parameters must be qaoa_transfer_parameters_v1 fitted on the train split")
    for key, entry in (payload.get("entries") or {}).items():
        depth = int(entry["depth"])
        if len(entry["internal_gammas"]) != depth or len(entry["betas"]) != depth:
            raise ValueError(f"transfer entry {key} has the wrong number of angles")
        if not all(math.isfinite(float(v)) for v in [*entry["internal_gammas"], *entry["betas"]]):
            raise ValueError(f"transfer entry {key} has non-finite angles")
    return payload


def transfer_key(depth: int, active_sites: int) -> str:
    return f"p{int(depth)}_sites{int(active_sites)}"


def _exact_ground_metrics(sampler: Any, parameters: np.ndarray, truth: Any) -> dict:
    """Noiseless ground-state probability of the QAOA state at given angles.

    Exact subspace amplitudes (a simulator-level algorithm property, not a
    hardware-measurable quantity); amplification is relative to uniform
    feasible sampling, whose ground probability is n_ground/|Omega|.
    """
    amplitudes = sampler.subspace_state(np.asarray(parameters, dtype=float))
    probabilities = np.abs(amplitudes) ** 2
    probabilities = probabilities / probabilities.sum()
    energies = sampler._subspace_energies
    ground = np.abs(energies - energies.min()) <= GROUND_ENERGY_TOLERANCE
    exact = float(probabilities[ground].sum())
    uniform = float(ground.sum()) / float(truth.configuration_count)
    return dict(exact_ground_probability=exact,
                log10_ground_amplification_exact=(math.log10(exact / uniform) if exact > 0 else float("-inf")))


def _ablation_summarize(counts: dict, energies: dict, ground: float, window: float) -> dict:
    """Matched-output metrics; report low-energy mass separately from entropy."""
    n = sum(counts.values())
    low = {s for s, e in energies.items() if e <= ground + window}
    seen_low = low.intersection(counts)
    mass = sum(counts[s] for s in seen_low)
    probs = np.array(list(counts.values()), float) / n
    conditional = np.array([counts[s] / mass for s in seen_low]) if mass else np.array([])
    best = min(energies[s] for s in counts)
    return dict(outputs=n, best_energy=best, gap=max(0., best-ground),
                hit=int(abs(best-ground) <= GROUND_ENERGY_TOLERANCE),
                ground_probability=sum(c for s,c in counts.items() if abs(energies[s]-ground)<=GROUND_ENERGY_TOLERANCE)/n,
                legal_rate=1.0, entropy=float(-sum(probs*np.log(probs))),
                low_energy_mass=mass/n, low_energy_coverage=len(seen_low)/len(low),
                low_energy_conditional_entropy=float(-sum(conditional*np.log(conditional))) if mass else None,
                unique_states=len(counts))


def _ablation_run_case(data: Any, scorer: Any, config: dict, args: Any, artifact: Path) -> list[dict]:
    """One paired instance: same physical objective, candidate set and QUBO for every solver AND every
    outputs/objective/restart combination swept below -- only the output-budget curve and QAOA's own
    objective/restart ablation vary within this one case; pruning/radius/depth/max_evals/seed (the
    shared input and fixed evaluation region) are fixed by ``config`` before any solver runs."""
    begin = time.perf_counter()
    state_policy=("fixed_three_chi1_wells" if args.states_per_site==3
                  else f"fixed_{args.states_per_site}_chi1_coverage")
    config = dict(config, state_policy=state_policy, states_per_site=args.states_per_site)
    active_sites=int(config["active_sites"])
    active = select_ablation_active(
        data, config["pruning"], active_sites, config["seed"], scorer,
        antigen_guidance_weight=args.antigen_guidance_weight,
        antigen_proximity_scale=args.antigen_proximity_scale,
        contact_ca_cutoff=args.contact_ca_cutoff,
    )
    sub = build_ablation_subgraph(data, active, config["radius"])
    force_field = ForceFieldConfig(
        cutoff_angstrom=args.nonbonded_cutoff,
        softcore_delta_angstrom=args.softcore_delta,
        hard_core_fraction=args.hard_core_fraction,
        hard_sphere_penalty=args.hard_sphere_penalty,
        lj_repulsion_cap=args.lj_repulsion_cap,
        lj_attraction_cap=args.lj_attraction_cap,
        coulomb_cap=args.coulomb_cap,
        dielectric_base=args.dielectric_base,
        dielectric_slope=args.dielectric_slope,
        thermal_energy_kcal=args.thermal_energy_kcal,
    )
    calibration = (
        EnergyCalibration.from_json(args.energy_calibration_file)
        if args.energy_calibration_file is not None else EnergyCalibration()
    )
    qubo = InterfaceQUBOBuilder(
        min_variables=args.states_per_site * active_sites,
        max_variables=args.states_per_site * active_sites,
        max_sites=active_sites,
        fixed_chi1_wells=True,
        fixed_states_per_site=args.states_per_site,
        force_field=force_field,
        rotamer_mode=args.rotamer_mode,
        rotamer_library_path=args.rotamer_library,
        rotamer_probability_floor=args.rotamer_probability_floor,
        rotamer_sigma_offsets=args.rotamer_sigma_offsets,
        energy_calibration=calibration,
    ).build(sub)
    if len(qubo.site_to_variables) != active_sites:
        raise ValueError(
            f"Requested {active_sites} active sites but QUBO contains "
            f"{len(qubo.site_to_variables)} rotamer sites; refusing a mislabeled scaling case")
    quantum_instance=qubo.to_quantum_instance()
    # Independent optimize/sample seeds (never the shared perturb/input seed
    # config["seed"] above, and never each other): derived per-case, before
    # this function is ever called, from the master-seed optimize/sample
    # streams keyed by stable case labels (target/pruning/radius/depth/
    # max_evals/seed) -- see _ablation_main. Falls back to config["seed"]
    # only for a config dict built by older/external code that never set
    # these keys, so this stays runnable standalone.
    optimize_seed = config.get("optimize_seed", config["seed"])
    measurement_seed = config.get("measurement_seed", config["seed"])
    sample_seed = config.get("sample_seed", config["seed"])
    # ONE sampler instance: XYMixerQAOASampler.optimize_robust/.sample/
    # .simulated_annealing already accept explicit optimize_seed/
    # measurement_seed/sample_seed overrides directly, so a second instance
    # is not needed -- the base constructor seed below is only the fallback
    # used if a call site ever omits an explicit override.
    sampler = XYMixerQAOASampler.from_instance(
        quantum_instance,p=config["depth"],seed=optimize_seed,
        shots=args.outputs[0],initial_state="wstate",
        simulation_mode="subspace",device_name="default.qubit")
    built = time.perf_counter()
    truth = sampler.enumerate_ground_states()
    energies = sampler.feasible_energy_map()
    oracle_seconds = time.perf_counter()-built
    low_ids = {s for s,e in energies.items() if e <= truth.energy+args.energy_window}
    assert low_ids
    records, raw = [], {}
    last_optimization = {}
    optimizations = {}
    qaoa_cache = {}
    qaoa_exact = {}
    for outputs in args.outputs:
        for solver in ("qaoa", "sa", "uniform", "greedy"):
            if solver == "qaoa":
                for qaoa_objective, qaoa_restarts in itertools.product(args.qaoa_objective, args.qaoa_restarts):
                    cache_key=(qaoa_objective,qaoa_restarts)
                    if cache_key not in qaoa_cache:
                        optimize_start=time.perf_counter()
                        opt=sampler.optimize_robust(max_evals=config["max_evals"], restarts=qaoa_restarts,
                            objective=qaoa_objective, cvar_alpha=args.cvar_alpha,
                            parameter_scale=args.parameter_scale, eval_shots=args.eval_shots,
                            optimize_seed=optimize_seed, measurement_seed=measurement_seed)
                        optimization_seconds=time.perf_counter()-optimize_start
                        scale=float(opt.shot_ledger["parameter_scale"])
                        optimization=dict(success=opt.success, message=getattr(opt,"message",None),
                            evaluations=opt.evaluations, history=list(opt.history),
                            gammas=opt.gammas.tolist(), betas=opt.betas.tolist(),
                            # Instance-independent (normalised) angles for parameter transfer.
                            parameter_scale=scale, internal_gammas=(opt.gammas*scale).tolist(),
                            termination_reason=getattr(opt,"termination_reason",None))
                        qaoa_cache[cache_key]=(opt,optimization,optimization_seconds)
                        qaoa_exact[cache_key]=_exact_ground_metrics(
                            sampler, np.concatenate([opt.gammas, opt.betas]), truth)
                    opt,optimization,optimization_seconds=qaoa_cache[cache_key]
                    sample_start=time.perf_counter()
                    sampled=sampler.sample(opt,shots=outputs,ground_state=truth,sample_seed=sample_seed)
                    sampling_seconds=time.perf_counter()-sample_start
                    counts=dict(sampled.counts)
                    last_optimization=optimization
                    elapsed=optimization_seconds+sampling_seconds
                    if sum(counts.values()) != outputs or any(s not in energies for s in counts):
                        raise AssertionError("Sample count or local one-hot constraint violated.")
                    metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
                    metrics.update(queries_to_solution(counts, energies, truth.energy,
                        fixed_units=float(opt.total_opt_shots), units_per_sample=1.0,
                        configuration_count=truth.configuration_count))
                    metrics.update(qaoa_exact[cache_key])
                    # metrics owns outputs: it is the verified measured count.
                    resources=estimate_qaoa_resources(
                        quantum_instance,p=config["depth"],
                        eval_shots=args.eval_shots,max_evals=config["max_evals"],
                        output_shots=outputs,restarts=qaoa_restarts,
                    )
                    records.append(dict(config, solver=solver,
                        method_role="proposed_quantum_method",
                        qaoa_objective=qaoa_objective, qaoa_restarts=qaoa_restarts, eval_shots=args.eval_shots,
                        **metrics, num_bits=quantum_instance.num_qubits,
                        configuration_count=truth.configuration_count,
                        qaoa_parameter_count=resources.parameter_count,
                        qaoa_rz_gates=resources.rz_gates_total,
                        qaoa_zz_gates=resources.zz_gates_total,
                        qaoa_xy_gates=resources.xy_gates_total,
                        # Legacy alias retained for schema compatibility; the
                        # explicit field below is the scientifically correct name.
                        qaoa_two_qubit_gates=resources.two_qubit_gates_total,
                        qaoa_variational_two_qubit_gates=resources.variational_two_qubit_gates_total,
                        qaoa_state_preparation_two_qubit_gates=resources.state_preparation_two_qubit_gates,
                        qaoa_full_circuit_two_qubit_gates=resources.full_circuit_two_qubit_gates,
                        qaoa_gate_accounting_scope=resources.accounting_scope,
                        qaoa_max_measurement_shots=resources.max_measurement_shots,
                        qaoa_total_measurement_shots=opt.total_opt_shots+outputs,
                        solver_seconds=elapsed,
                        optimization_seconds=optimization_seconds,
                        sampling_seconds=sampling_seconds,
                        optimization_reused=(outputs != args.outputs[0]),
                        build_seconds=built-begin, oracle_seconds=oracle_seconds,
                        single_state_energy_queries=None,
                        optimizer_success=optimization.get("success"),
                        optimizer_evaluations=optimization.get("evaluations"),
                        termination_reason=optimization.get("termination_reason"),
                        optimization_energy_start=optimization["history"][0] if optimization.get("history") else None,
                        optimization_energy_end=opt.objective_value,
                        diagnostic_mean_energy_end=opt.energy,
                        total_opt_shots=opt.total_opt_shots))
                    records[-1].update(budget_mode="matched_outputs", budget_seconds=None, budget_overrun_seconds=0.0)
                    key = f"qaoa_obj{qaoa_objective}_restarts{qaoa_restarts}_outputs{outputs}"
                    optimizations[key] = dict(
                        optimization,
                        optimization_seconds=optimization_seconds,
                        reused_across_output_curve=True)
                    raw[key] = [{"bits":"".join(map(str,s)), "count":c} for s,c in sorted(counts.items())]
            else:
                start = time.perf_counter()
                queries = None
                if solver == "sa":
                    # Existing API calls each single-site proposal a sweep. Convert explicitly.
                    proposals = args.sa_passes * len(qubo.site_to_variables)
                    sampled = sampler.simulated_annealing(num_reads=outputs, site_passes=args.sa_passes, seed=sample_seed, ground_state=truth)
                    counts = dict(sampled.counts)
                    queries = outputs*(1+proposals)
                else:
                    counts, queries = _ablation_classical_counts(sampler, outputs, sample_seed+101,
                                                        solver=="greedy", args.greedy_passes)
                elapsed = time.perf_counter()-start
                if sum(counts.values()) != outputs or any(s not in energies for s in counts):
                    raise AssertionError("Sample count or local one-hot constraint violated.")
                metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
                metrics.update(queries_to_solution(counts, energies, truth.energy,
                    # >=1 unit/sample: an unevaluated uniform draw cannot be recognised
                    # as a ground state, just as each QAOA output costs one shot.
                    fixed_units=0.0, units_per_sample=max(1.0, float(queries)/float(outputs)),
                    configuration_count=truth.configuration_count))
                # Do not pass outputs twice; _ablation_summarize supplies it.
                records.append(dict(config, solver=solver,
                    method_role="classical_baseline",
                    qaoa_objective=None, qaoa_restarts=None, eval_shots=None,
                    **metrics, num_bits=quantum_instance.num_qubits,
                    configuration_count=truth.configuration_count, solver_seconds=elapsed,
                    optimization_seconds=None, sampling_seconds=None, optimization_reused=None,
                    build_seconds=built-begin, oracle_seconds=oracle_seconds,
                    single_state_energy_queries=queries,
                    optimizer_success=None, optimizer_evaluations=None, termination_reason=None,
                    optimization_energy_start=None, optimization_energy_end=None))
                records[-1].update(budget_mode="matched_outputs", budget_seconds=None, budget_overrun_seconds=0.0)
                raw[f"{solver}_outputs{outputs}"] = [{"bits":"".join(map(str,s)), "count":c} for s,c in sorted(counts.items())]
        transfer = (getattr(args, "transfer_payload", None) or {}).get("entries", {}).get(
            transfer_key(config["depth"], active_sites))
        if transfer is not None:
            # Parameter transfer (Brandao et al. 2018; Galda et al. 2021): no
            # per-instance training; angles fitted on training graphs only.
            scale = sampler.parameter_scale(args.parameter_scale)
            parameters = np.concatenate([np.asarray(transfer["internal_gammas"], float) / scale,
                                         np.asarray(transfer["betas"], float)])
            sample_start = time.perf_counter()
            sampled = sampler.sample(parameters, shots=outputs, ground_state=truth, sample_seed=sample_seed+50001)
            counts = dict(sampled.counts)
            if sum(counts.values()) != outputs or any(s not in energies for s in counts):
                raise AssertionError("Sample count or local one-hot constraint violated.")
            metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
            metrics.update(queries_to_solution(counts, energies, truth.energy, fixed_units=0.0,
                units_per_sample=1.0, configuration_count=truth.configuration_count))
            metrics.update(_exact_ground_metrics(sampler, parameters, truth))
            records.append(dict(config, solver="qaoa_transfer", method_role="proposed_quantum_method_transfer",
                qaoa_objective=args.transfer_payload.get("objective"),
                qaoa_restarts=args.transfer_payload.get("restarts"), eval_shots=None,
                **metrics, num_bits=quantum_instance.num_qubits,
                configuration_count=truth.configuration_count,
                solver_seconds=time.perf_counter()-sample_start,
                optimization_seconds=0.0, sampling_seconds=time.perf_counter()-sample_start,
                optimization_reused=None, build_seconds=built-begin, oracle_seconds=oracle_seconds,
                single_state_energy_queries=None, optimizer_success=None, optimizer_evaluations=0,
                termination_reason="transferred_parameters", total_opt_shots=0,
                transfer_fit_instances=int(transfer.get("n_instances", 0)),
                transfer_fit_training_shots=int(transfer.get("training_shots_total", 0)),
                budget_mode="matched_outputs", budget_seconds=None, budget_overrun_seconds=0.0))
            raw[f"qaoa_transfer_outputs{outputs}"] = [{"bits":"".join(map(str,b)), "count":c} for b,c in sorted(counts.items())]
        if args.time_baselines:
            qaoa_rows_this_output=[
                r for r in records
                if r["solver"]=="qaoa" and r["outputs"]==outputs
                and r.get("qaoa_objective")==args.time_donor_objective
                and r.get("qaoa_restarts")==args.time_donor_restarts
            ]
            if len(qaoa_rows_this_output)!=1:
                raise ValueError(
                    f"Expected exactly one matched-time donor for outputs={outputs}, "
                    f"objective={args.time_donor_objective}, restarts={args.time_donor_restarts}; "
                    f"found {len(qaoa_rows_this_output)}")
            donor_row=qaoa_rows_this_output[0]
            if donor_row.get("termination_reason")!="all_restarts_failed":
                budget=donor_row["solver_seconds"]
                for method in ("sa", "uniform", "greedy"):
                    counts, elapsed, queries = _time_budget_counts(sampler, method, budget,
                        sample_seed+10001, args.sa_passes, args.greedy_passes)
                    metrics = _ablation_summarize(counts, energies, truth.energy, args.energy_window)
                    metrics.update(queries_to_solution(counts, energies, truth.energy,
                        fixed_units=0.0, units_per_sample=max(1.0, float(queries)/float(sum(counts.values()))),
                        configuration_count=truth.configuration_count))
                    row = dict(donor_row)
                    row.update(metrics, solver=method+"_time", method_role="classical_baseline",
                        reference_outputs=outputs, solver_seconds=elapsed,
                        qaoa_parameter_count=None,qaoa_rz_gates=None,qaoa_zz_gates=None,
                        qaoa_xy_gates=None,qaoa_two_qubit_gates=None,
                        qaoa_max_measurement_shots=None,qaoa_total_measurement_shots=None,
                        single_state_energy_queries=queries, optimizer_success=None,
                        optimizer_evaluations=None, termination_reason=None,
                        optimization_energy_start=None, optimization_energy_end=None,
                        diagnostic_mean_energy_end=None, total_opt_shots=0,
                        budget_mode="matched_time_soft_deadline",
                        budget_seconds=budget, budget_overrun_seconds=max(0., elapsed-budget))
                    records.append(row)
                    raw[f"{method}_time_outputs{outputs}"] = [{"bits":"".join(map(str,b)), "count":c} for b,c in sorted(counts.items())]
    quantum_benchmark_contract={
        "schema":"quantum_classical_benchmark_v1",
        "proposed_method":{
            "name":"xy_qaoa",
            "role":"proposed_quantum_method",
            "simulation_scope":"exact feasible-subspace classical simulation with finite-shot objectives",
        },
        "classical_baselines":["simulated_annealing","greedy","uniform_feasible_sampling"],
        "oracle":{
            "name":"exact_feasible_enumeration",
            "role":"retrospective_ground_truth_only",
            "energy":truth.energy,
            "configuration_count":truth.configuration_count,
            "oracle_seconds":oracle_seconds,
        },
    }
    _ablation_atomic_json(artifact, dict(config=config, metrics=records, counts=raw,
        quantum_instance={
            **quantum_instance.manifest(),
            "Q":quantum_instance.Q.tolist(),
            "physical_self":quantum_instance.physical_self.tolist(),
            "physical_pair":quantum_instance.physical_pair.tolist(),
            "site_to_variables":{
                str(site):list(variables)
                for site,variables in quantum_instance.site_to_variables.items()
            },
            "ising_h":quantum_instance.ising_h.tolist(),
            "ising_J":quantum_instance.ising_J.tolist(),
        },
        quantum_benchmark=quantum_benchmark_contract,
        optimization=last_optimization, optimizations=optimizations,
        active_residue_ids=[data.residue_ids[i] for i in active.tolist()],
        frozen_residue_ids=[sub.residue_ids[i] for i in range(sub.num_nodes) if sub.is_frozen_environment[i]],
        physical_self=qubo.physical_self.tolist(), physical_pair=qubo.physical_pair.tolist(),
        site_to_variables=qubo.site_to_variables, variable_map=[vars(r) for r in qubo.variable_map],
        rotamer_state_records=qubo.metadata.get("rotamer_state_records",[]),
        ground_energy=truth.energy, total_seconds=time.perf_counter()-begin,
        scope=("coarse-grained fixed-backbone; classical exact subspace simulation; "
               "matched-output records plus optional matched-time controls; budget_mode disambiguates")))
    return records


def _ablation_export_results(out: Path) -> None:
    """Rebuild CSV from atomic per-case records after interruptions."""
    rows = []
    for path in sorted((out/"cases").glob("*.json")):
        rows.extend(json.loads(path.read_text())["metrics"])
    if not rows:
        return
    temp = out/"metrics.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, out/"metrics.csv")
    report = ["# Exploratory matched-output benchmark", "",
        "Classical simulation of a coarse-grained fixed-backbone model. No structural-accuracy or quantum-speedup claim.",
        "Rows are correlated target × seed × configuration runs, NOT independent proteins.",
        "Compare solvers within an instance. Do not compare raw energies across different prunings/radii.",
        "Exact oracle preprocessing is separately timed. Equal outputs do not mean equal compute budgets.",
        "", f"Completed cases: {len(list((out/'cases').glob('*.json')))}", "",
        "| Solver | Runs | Hit fraction | Mean gap | Low-energy coverage |",
        "|---|---:|---:|---:|---:|"]
    for solver in sorted({r["solver"] for r in rows}):
        group = [r for r in rows if r["solver"]==solver]
        report.append(f"| {solver} | {len(group)} | {np.mean([r['hit'] for r in group]):.4f} | "
                      f"{np.mean([r['gap'] for r in group]):.6g} | {np.mean([r['low_energy_coverage'] for r in group]):.4f} |")
    (out/"summary.md").write_text("\n".join(report), encoding="utf-8")


_ABLATION_SCORER = None
_ABLATION_MODEL_INFO = None


def _ablation_append_results(out: Path, key: str) -> None:
    """Parent-only durable append; atomic case files rebuild CSV after a crash."""
    rows = json.loads((out/"cases"/(key+".json")).read_text(encoding="utf-8"))["metrics"]
    path = out/"metrics.csv"
    new_file = not path.exists()
    fields = list(dict.fromkeys(k for row in rows for k in row))
    if not new_file:
        with path.open(newline="", encoding="utf-8") as handle:
            fields = next(csv.reader(handle))
    # A case may legitimately add a metric column as the pipeline evolves.
    # Rebuild the aggregate with the union schema instead of aborting the run.
    extra = [k for row in rows for k in row if k not in fields]
    if extra:
        existing = list(csv.DictReader(path.open(newline="", encoding="utf-8"))) if path.exists() else []
        fields = list(dict.fromkeys(fields + extra))
        merged = existing + rows
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(merged)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temp, path)
        return
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _ablation_worker(task: tuple) -> tuple:
    """Own one atomic case file; leave aggregate CSV and failure logs to parent."""
    global _ABLATION_SCORER, _ABLATION_MODEL_INFO
    path, config, args, artifact, key = task
    try:
        torch.set_num_threads(args.omp_threads)
        data = load_graph(path)
        if config["pruning"] == "egnn" and _ABLATION_SCORER is None:
            _ABLATION_SCORER, _ABLATION_MODEL_INFO = load_interface_scorer(
                args.checkpoint, torch_device=torch.device("cpu"), seed=42)
            if _ABLATION_MODEL_INFO.status != "checkpoint_loaded":
                raise ValueError("A trained matching checkpoint is mandatory")
            _ABLATION_SCORER.eval()
        if config["pruning"] == "egnn":
            assert_checkpoint_graph_compatible(
                _ABLATION_MODEL_INFO, data,
                homology_isolation={
                    "vhh_full_chain_identity": args.vhh_identity_threshold,
                    "cdr_h3_identity": args.cdr_h3_identity_threshold,
                    "antigen_identity": args.antigen_identity_threshold,
                    "antigen_min_length_coverage": args.antigen_min_length_coverage,
                },
            )
        config["pdb_id"] = getattr(data, "pdb_id", "")
        _ablation_run_case(data, _ABLATION_SCORER, config, args, artifact)
        return key, config, None
    except Exception:
        return key, config, traceback.format_exc()


def _ablation_dispatch(tasks: Iterable[tuple], workers: int) -> Iterable[tuple]:
    """Bound queued work; a crashed process propagates as a systemic failure."""
    if workers == 1:
        for task in tasks:
            yield _ablation_worker(task)
        return
    iterator = iter(tasks)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        pending = set()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * workers:
                task = next(iterator, None)
                if task is None:
                    exhausted = True
                else:
                    pending.add(pool.submit(_ablation_worker, task))
            if pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    yield future.result()

def _ablation_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Matched-output research ablations on the existing coarse-grained model.")
    parser.add_argument("--input-dir", type=Path, default=Path("dataset_clean_500/graphs/test_snac_hard"))
    parser.add_argument("--checkpoint", type=Path, default=Path("quantum-protein/checkpoints_500/best_egnn_pruning.pt"))
    parser.add_argument("--out-dir", type=Path, default=Path("benchmark_results_ablation"))
    parser.add_argument("--fit-energy-calibration-csv", type=Path)
    parser.add_argument("--fit-energy-calibration-out", type=Path)
    parser.add_argument("--calibration-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,43,44])
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED,
        help="Derives independent, saved, per-case optimize/sample sub-seeds (see seed_streams.py) -- "
             "NEVER the same as --seeds, which are the shared perturb/site-selection repeat identities.")
    parser.add_argument("--pruning", nargs="+", choices=["egnn","contact","distance","cdr","random"], default=["egnn","contact","random"])
    parser.add_argument("--radii", type=float, nargs="+", default=[6.,10.])
    parser.add_argument("--depths", type=int, nargs="+", default=[1,2,3])
    parser.add_argument("--max-evals", type=int, nargs="+", default=[90,300])
    parser.add_argument("--active-sites", type=int, nargs="+", default=[6],
        help="Active-site scaling axis. Each value defines a separate paired QUBO case family.")
    parser.add_argument("--states-per-site", type=int, default=3,
        help="Fixed retained rotamers per site (3..6), covering all three chi1 wells.")
    parser.add_argument("--vhh-identity-threshold", type=float, default=0.80)
    parser.add_argument("--cdr-h3-identity-threshold", type=float, default=0.50)
    parser.add_argument("--antigen-identity-threshold", type=float, default=0.30)
    parser.add_argument("--antigen-min-length-coverage", type=float, default=0.70)
    parser.add_argument("--antigen-guidance-weight", type=float, default=0.25,
        help="Blend weight for label-free nearest-antigen proximity in EGNN site ranking.")
    parser.add_argument("--antigen-proximity-scale", type=float, default=6.0,
        help="Exponential decay length in Angstrom for nearest-antigen proximity.")
    parser.add_argument("--contact-ca-cutoff", type=float, default=8.0,
        help="C-alpha distance cutoff in Angstrom for the contact-count baseline.")
    parser.add_argument("--nonbonded-cutoff", type=float, default=8.0)
    parser.add_argument("--softcore-delta", type=float, default=0.5)
    parser.add_argument("--hard-core-fraction", type=float, default=0.72)
    parser.add_argument("--hard-sphere-penalty", type=float, default=25.0)
    parser.add_argument("--lj-repulsion-cap", type=float, default=50.0)
    parser.add_argument("--lj-attraction-cap", type=float, default=5.0)
    parser.add_argument("--coulomb-cap", type=float, default=20.0)
    parser.add_argument("--dielectric-base", type=float, default=4.0)
    parser.add_argument("--dielectric-slope", type=float, default=2.0)
    parser.add_argument("--thermal-energy-kcal", type=float, default=0.593)
    parser.add_argument("--rotamer-mode", choices=("legacy","dunbrack2010","pyrosetta_dun10"), default="dunbrack2010")
    parser.add_argument("--rotamer-library", type=Path)
    parser.add_argument("--rotamer-probability-floor", type=float, default=1e-4)
    parser.add_argument("--rotamer-sigma-offsets", type=float, nargs="+", default=[-1.0,0.0,1.0])
    parser.add_argument("--energy-calibration-file", type=Path)
    parser.add_argument("--require-calibrated-energy", action="store_true")
    parser.add_argument("--outputs", type=int, nargs="+", default=[1000],
        help="Output-budget curve (e.g. 10 30 100 300 1000): each value is a fully matched-output "
             "comparison across all four solvers, so equal-output at one budget is never conflated "
             "with equal-output at a different budget, and neither is conflated with equal total "
             "compute -- see solver_seconds/oracle_seconds/single_state_energy_queries per row.")
    parser.add_argument("--qaoa-objective", nargs="+", choices=("mean","cvar"), default=["cvar"],
        help="QAOA optimizer objective ablation dimension (mean vs CVaR low-energy tail).")
    parser.add_argument("--qaoa-restarts", type=int, nargs="+", default=[4],
        help="QAOA multistart ablation dimension (1 = single-start, N = N-start COBYLA sharing one "
             "--max-evals budget per case, per optimize_robust's own documented budget sharing).")
    parser.add_argument("--cvar-alpha", type=float, default=.1)
    parser.add_argument("--eval-shots", type=int, default=500,
        help="Finite measurement shots per objective evaluation inside optimize_robust (never the "
             "exact analytic expectation) -- the NISQ-realistic finite-shot CVaR/mean estimate.")
    parser.add_argument("--parameter-scale", choices=("max_coefficient","feasible_iqr"), default="max_coefficient")
    parser.add_argument("--transfer-parameters", type=Path, default=None,
                        help="qaoa_transfer_parameters_v1 JSON fitted on TRAINING graphs; adds untrained "
                             "'qaoa_transfer' rows sampled at the transferred angles.")
    parser.add_argument("--time-baselines", action="store_true", help="Add classical restart baselines using QAOA solver wall time; record soft-deadline overrun.")
    parser.add_argument("--time-donor-objective", choices=("mean","cvar"), default="cvar")
    parser.add_argument("--time-donor-restarts", type=int, default=4)
    parser.add_argument("--sa-passes", type=int, default=100)
    parser.add_argument("--greedy-passes", type=int, default=50)
    parser.add_argument("--energy-window", type=float, default=2.)
    parser.add_argument("--max-targets", type=int, default=0)
    parser.add_argument("--target-selection-seed", type=int, default=None,
        help="When --max-targets is positive, deterministically shuffle the sorted input graphs before truncation.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--omp-threads", type=int, default=2)
    args = parser.parse_args(argv)
    if args.fit_energy_calibration_csv is not None:
        if args.fit_energy_calibration_out is None:
            parser.error("--fit-energy-calibration-out is required with --fit-energy-calibration-csv")
        payload=fit_energy_calibration_csv(
            args.fit_energy_calibration_csv,args.fit_energy_calibration_out,args.calibration_ridge_alpha
        )
        print(json.dumps(payload,indent=2,sort_keys=True))
        return 0
    if (not args.active_sites or any(site < 4 or site > 10 for site in args.active_sites)
            or len(args.active_sites) != len(set(args.active_sites))
            or args.states_per_site not in (3,4,5,6)
            or any(site * args.states_per_site > 30 for site in args.active_sites)
            or any(not 0.0 < value <= 1.0 for value in (
                args.vhh_identity_threshold, args.cdr_h3_identity_threshold,
                args.antigen_identity_threshold, args.antigen_min_length_coverage))
            or not 0.0 <= args.antigen_guidance_weight <= 1.0
            or min(*args.outputs, args.sa_passes, args.greedy_passes, *args.max_evals) <= 0
            or min(args.qaoa_restarts) <= 0 or args.eval_shots <= 0):
        parser.error("Require unique active-site values in 4..10, antigen-guidance-weight in [0,1], and positive budgets.")
    positive_scientific = (
        args.antigen_proximity_scale, args.contact_ca_cutoff, args.nonbonded_cutoff,
        args.softcore_delta, args.hard_core_fraction, args.hard_sphere_penalty,
        args.lj_repulsion_cap, args.lj_attraction_cap, args.coulomb_cap,
        args.dielectric_base, args.thermal_energy_kcal,
    )
    if any((not math.isfinite(v)) or v <= 0 for v in positive_scientific):
        parser.error("Scientific distance/energy parameters must be positive and finite.")
    if not math.isfinite(args.dielectric_slope) or args.dielectric_slope < 0:
        parser.error("dielectric-slope must be finite and nonnegative.")
    if args.time_baselines:
        if args.time_donor_objective not in args.qaoa_objective:
            parser.error("--time-donor-objective must be included in --qaoa-objective")
        if args.time_donor_restarts not in args.qaoa_restarts:
            parser.error("--time-donor-restarts must be included in --qaoa-restarts")
    if args.rotamer_mode=="dunbrack2010" and (args.rotamer_library is None or not args.rotamer_library.is_file()):
        parser.error("--rotamer-library must point to ALL.bbdep.rotamers.lib in dunbrack2010 mode")
    if not 0.0 < args.rotamer_probability_floor < 1.0 or not args.rotamer_sigma_offsets:
        parser.error("Invalid rotamer probability floor or sigma offsets")
    if args.require_calibrated_energy and (args.energy_calibration_file is None or not args.energy_calibration_file.is_file()):
        parser.error("--require-calibrated-energy requires an existing --energy-calibration-file")
    if any(not math.isfinite(r) or r<=0 for r in args.radii) or any(not 1<=p<=MAX_QAOA_DEPTH for p in args.depths):
        parser.error(f"Require positive finite radii and depths in 1..{MAX_QAOA_DEPTH}.")
    if args.max_targets < 0 or args.energy_window < 0 or not math.isfinite(args.energy_window):
        parser.error("Invalid target limit or energy window.")
    if args.workers < 1 or args.omp_threads < 1:
        parser.error("workers and omp-threads must be positive")
    args.transfer_payload = None
    if args.transfer_parameters is not None:
        args.transfer_payload = load_transfer_parameters(args.transfer_parameters)
        if args.transfer_payload["parameter_scale_mode"] != args.parameter_scale:
            parser.error("--transfer-parameters were fitted with a different --parameter-scale")
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(args.omp_threads)
    torch.set_num_threads(args.omp_threads)
    files = sorted(args.input_dir.glob("*.pt"))
    if args.max_targets:
        if args.target_selection_seed is not None:
            rng=np.random.default_rng(args.target_selection_seed)
            order=rng.permutation(len(files))
            files=[files[int(i)] for i in order[:args.max_targets]]
        else:
            files = files[:args.max_targets]
    if not files:
        parser.error("No input graphs.")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out/".lock"), timeout=0):
        provenance = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            input_sha256={str(p.resolve()):_ablation_digest(p) for p in files},
            code_sha256={n:_ablation_digest(repo_path(n)) for n in
                ("model_egnn_pruning.py","subgraph_to_qubo.py","qaoa_interface_sampler.py","batch_benchmark_hard_set.py",
                 *SHARED_HELPER_MODULES)},
            checkpoint_sha256=_ablation_digest(args.checkpoint) if "egnn" in args.pruning and args.checkpoint.exists() else None,
            transfer_parameters_sha256=(_ablation_digest(args.transfer_parameters) if args.transfer_parameters else None),
            python=sys.version, numpy=np.__version__, torch=torch.__version__, platform=platform.platform(),
            pennylane=__import__("pennylane").__version__, scipy=__import__("scipy").__version__)
        manifest = out/"run_manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != provenance:
            raise ValueError("Output provenance differs; use a new --out-dir.")
        _ablation_atomic_json(manifest, provenance)
        streams = derive_streams(args.master_seed)
        save_stream_map(out/"seed_streams.json", args.master_seed, streams)
        (out/"cases").mkdir(exist_ok=True)
        scorer = None
        if "egnn" in args.pruning:
            scorer, status = load_interface_scorer(args.checkpoint, torch_device=torch.device("cpu"), seed=42)
            if status.status != "checkpoint_loaded":
                raise ValueError("A trained matching checkpoint is mandatory; no random fallback.")
            assert_checkpoint_graph_compatible(
                status, load_graph(files[0]),
                homology_isolation={
                    "vhh_full_chain_identity": args.vhh_identity_threshold,
                    "cdr_h3_identity": args.cdr_h3_identity_threshold,
                    "antigen_identity": args.antigen_identity_threshold,
                    "antigen_min_length_coverage": args.antigen_min_length_coverage,
                },
            )
            scorer.eval()
        _ablation_export_results(out)
        # Treat repeated CLI values as one grid point.  Artifact keys are
        # intentionally content-addressed, so counting duplicate tuples would
        # otherwise make completion impossible even though every artifact exists.
        settings = list(dict.fromkeys(itertools.product(
            args.pruning,args.radii,args.depths,args.max_evals,args.active_sites,args.seeds)))
        total_planned = len(files)*len(settings)
        failed_keys_path = out/"failed_case_keys.json"
        failed_keys = set(json.loads(failed_keys_path.read_text())) if failed_keys_path.exists() else set()
        failed = 0
        tasks = []
        for path, setting in itertools.product(files, settings):
            config = dict(target=path.stem, pdb_id=path.stem,
                          pruning=setting[0], radius=setting[1],
                          depth=setting[2], max_evals=setting[3],
                          active_sites=setting[4], seed=setting[5])
            labels = ("ablation", config["target"], config["pruning"], str(config["radius"]),
                      str(config["depth"]), str(config["max_evals"]),
                      str(config["active_sites"]), str(config["seed"]))
            for stream in ("optimize", "measurement", "sample"):
                config[stream+"_seed"] = derive_child_seed(streams[stream], *labels)
            key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:24]
            artifact = out/"cases"/(key+".json")
            if artifact.exists():
                failed_keys.discard(key)
                continue
            tasks.append((path, config, args, artifact, key))
        for key, config, error in tqdm(_ablation_dispatch(tasks, args.workers),
                                      total=len(tasks), desc="Pending ablation cases"):
            if error is None:
                failed_keys.discard(key)
                _ablation_append_results(out, key)
            else:
                failed += 1
                failed_keys.add(key)
                with (out/"failed_cases.log").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(config)+"\n"+error+"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            _ablation_atomic_json(failed_keys_path, sorted(failed_keys))
        _ablation_atomic_json(failed_keys_path, sorted(failed_keys))
        _ablation_export_results(out)
        cases_completed_total = len(list((out/"cases").glob("*.json")))
        failures_total = len(failed_keys)
        gap = total_planned - cases_completed_total - failures_total
        # "Closed" per requirement #5: every planned case is accounted for
        # as EITHER a completed artifact OR an entry in the failure list --
        # never inferred from returncode or from any single file's mere
        # existence. A per-instance failure is expected and does not by
        # itself mean the stage failed; an unclosed gap (a case neither
        # completed nor recorded as failed -- e.g. the process was killed
        # mid-case) does.
        summary = dict(total_cases_planned=total_planned, cases_completed_total=cases_completed_total,
            failures_total=failures_total, gap=gap, closed=(gap==0),
            failures_this_invocation=failed)
        _ablation_atomic_json(out/"run_summary.json", summary)
        print(f"Results: {out}; failures this invocation: {failed}; "
              f"planned={total_planned} completed={cases_completed_total} failed_total={failures_total} closed={summary['closed']}")
        return 1 if (failed or not summary["closed"]) else 0



def _time_budget_counts(sampler: Any, method: str, seconds: float, seed: int,
                        sa_passes: int, greedy_passes: int) -> tuple[dict, float, int]:
    """Independent restarts to a soft wall deadline, checked between complete reads.

    Includes solver call/setup overhead. One final read may exceed the budget;
    this is recorded, never silently presented as strict equal-time computation.
    """
    if seconds <= 0 or not math.isfinite(seconds):
        raise ValueError("Time budget must be positive and finite")
    if method not in ("sa", "uniform", "greedy"):
        raise ValueError("Unknown baseline")
    start = time.perf_counter()
    counts: Counter = Counter()
    queries, restart = 0, 0
    while not counts or time.perf_counter()-start < seconds:
        if method == "sa":
            result = sampler.simulated_annealing(num_reads=1, site_passes=sa_passes,
                                                 seed=seed+restart)
            current = result.counts
            used = 1 + sa_passes*len(sampler.site_to_variables)
        else:
            current, used = _ablation_classical_counts(sampler, 1, seed+restart,
                                                       method == "greedy", greedy_passes)
            # Uniform draws need one objective query each for best-energy search.
            if method == "uniform":
                for bits in current:
                    sampler.physical_energy(bits)
                used = 1
        counts.update(current)
        queries += used
        restart += 1
    return dict(counts), time.perf_counter()-start, queries
