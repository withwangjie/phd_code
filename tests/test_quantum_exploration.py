"""Quantum-intrinsic primary endpoint, depth range and parameter transfer (A7)."""
from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import nanoqc.pipeline.run_full_experiment as full
from nanoqc.experiments import batch_benchmark_hard_set as bbh
from nanoqc.experiments import fit_qaoa_transfer_parameters as fit
from nanoqc.inference import analyze_quantum_exploration as exploration
from nanoqc.qubo.subgraph_to_qubo import _virtual_pruned_graph
from nanoqc.solvers import qaoa_interface_sampler as sampler_module

REPO = Path(__file__).resolve().parents[1]


def test_depth_cap_is_shared_and_enforced():
    assert full.MAX_QAOA_DEPTH == sampler_module.MAX_QAOA_DEPTH == 12
    groups = {0: [0, 1, 2], 1: [3, 4, 5]}
    h, J = np.zeros(6), np.zeros((6, 6))
    sampler_module.XYMixerQAOASampler(h, J, groups, p=6, simulation_mode="subspace")
    with pytest.raises(ValueError):
        sampler_module.XYMixerQAOASampler(h, J, groups, p=13, simulation_mode="subspace")


def _args(**extra):
    base = dict(states_per_site=3, antigen_guidance_weight=0.25, antigen_proximity_scale=6.0, contact_ca_cutoff=8.0,
                nonbonded_cutoff=8.0, softcore_delta=0.5, hard_core_fraction=0.72, hard_sphere_penalty=25.0,
                lj_repulsion_cap=50.0, lj_attraction_cap=5.0, coulomb_cap=20.0, dielectric_base=4.0,
                dielectric_slope=2.0, thermal_energy_kcal=0.593, energy_calibration_file=None, rotamer_mode="legacy",
                rotamer_library=None, rotamer_probability_floor=1e-4, rotamer_sigma_offsets=[-1.0, 0.0, 1.0],
                outputs=[200], qaoa_objective=["cvar"], qaoa_restarts=[4], cvar_alpha=0.1,
                parameter_scale="max_coefficient", eval_shots=100, sa_passes=20, greedy_passes=10,
                energy_window=2.0, time_baselines=False, time_donor_objective="cvar", time_donor_restarts=4,
                transfer_payload=None)
    base.update(extra)
    return argparse.Namespace(**base)


def _case(tmp_path: Path, transfer_payload=None):
    data = _virtual_pruned_graph(5)
    for name in ("is_active", "is_frozen_environment", "selected_vhh_mask"):
        delattr(data, name)
    config = dict(pruning="contact", active_sites=5, radius=6.0, seed=7, depth=2, max_evals=40,
                  pdb_id="virtual", optimize_seed=1, measurement_seed=2, sample_seed=3)
    out = tmp_path / "case.json"
    rows = bbh._ablation_run_case(data, None, config, _args(transfer_payload=transfer_payload), out)
    return rows, json.loads(out.read_text())


def test_benchmark_rows_carry_quantum_intrinsic_metrics_and_transfer_rows(tmp_path: Path):
    rows, case = _case(tmp_path)
    qaoa = next(r for r in rows if r["solver"] == "qaoa")
    assert 0.0 < qaoa["exact_ground_probability"] <= 1.0
    assert math.isfinite(qaoa["log10_ground_amplification_exact"])
    assert qaoa["log10_qts99_execution"] < qaoa["log10_qts99"]            # training shots excluded
    optimization = next(iter(case["optimizations"].values()))
    assert len(optimization["internal_gammas"]) == 2 and optimization["parameter_scale"] > 0
    payload = dict(schema="qaoa_transfer_parameters_v1", fit_split="train", objective="cvar", restarts=4,
                   parameter_scale_mode="max_coefficient",
                   entries={"p2_sites5": dict(depth=2, active_sites=5, n_instances=12, training_shots_total=1,
                                              internal_gammas=optimization["internal_gammas"],
                                              betas=optimization["betas"])})
    (tmp_path / "t").mkdir()
    rows_t, _ = _case(tmp_path / "t", transfer_payload=payload)
    transfer = next(r for r in rows_t if r["solver"] == "qaoa_transfer")
    assert transfer["total_opt_shots"] == 0 and transfer["optimizer_evaluations"] == 0
    # Same angles and same instance: the exact amplification matches the trained row.
    assert transfer["log10_ground_amplification_exact"] == pytest.approx(qaoa["log10_ground_amplification_exact"])


def _fit_case(depth: int, sites: int, gammas, betas) -> dict:
    return dict(config=dict(depth=depth, active_sites=sites, pdb_id="x"),
                optimizations={f"qaoa_objcvar_restarts4_outputs1000": dict(
                    internal_gammas=list(gammas), betas=list(betas), termination_reason="max_evaluations_reached")},
                metrics=[dict(solver="qaoa", qaoa_objective="cvar", qaoa_restarts=4, total_opt_shots=100)])


def test_transfer_fit_uses_training_split_and_median(tmp_path: Path):
    results = tmp_path / "fit"
    (results / "cases").mkdir(parents=True)
    (results / "run_manifest.json").write_text(json.dumps({"arguments": {
        "input_dir": str(tmp_path / "dataset" / "graphs" / "train"), "parameter_scale": "max_coefficient"}}))
    for i, g in enumerate((0.1, 0.2, 0.9)):
        (results / "cases" / f"c{i}.json").write_text(json.dumps(_fit_case(1, 6, [g], [0.3 + i])))
    out = tmp_path / "transfer.json"
    assert fit.main(["--results-dir", str(results), "--objective", "cvar", "--restarts", "4",
                     "--min-instances", "3", "--out", str(out)]) == 0
    entry = json.loads(out.read_text())["entries"]["p1_sites6"]
    assert entry["internal_gammas"] == pytest.approx([0.2]) and entry["betas"] == pytest.approx([1.3])
    assert entry["n_instances"] == 3 and entry["training_shots_total"] == 300
    bad = json.loads((results / "run_manifest.json").read_text())
    bad["arguments"]["input_dir"] = str(tmp_path / "graphs" / "test_snac_hard")
    (results / "run_manifest.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="train split"):
        fit.main(["--results-dir", str(results), "--objective", "cvar", "--restarts", "4", "--out", str(out)])


def test_exploration_summary_per_depth(tmp_path: Path):
    cluster_map = {}
    dirs = []
    for depth in (1, 2):
        directory = tmp_path / f"p{depth}"
        (directory / "cases").mkdir(parents=True)
        (directory / "run_manifest.json").write_text("{}")
        for i in range(4):
            pdb = f"p{i}"
            cluster_map[pdb] = f"c{i}"
            base = dict(budget_mode="matched_outputs", outputs=1000, log10_qts99_execution=2.0,
                        log10_qts99=4.0, total_opt_shots=1000)
            rows = [dict(base, solver="qaoa", qaoa_objective="cvar", qaoa_restarts=4,
                         log10_ground_amplification_exact=1.0 * depth),
                    dict(base, solver="qaoa_transfer", log10_ground_amplification_exact=0.8 * depth)]
            (directory / "cases" / f"{pdb}.json").write_text(json.dumps(dict(
                config=dict(depth=depth, max_evals=40 * depth, active_sites=6, pdb_id=pdb), metrics=rows)))
        dirs.append(directory)
    (tmp_path / "clusters.json").write_text(json.dumps(cluster_map))
    out_json, out_md = tmp_path / "s.json", tmp_path / "s.md"
    assert exploration.main(["--depth-dirs", *map(str, dirs), "--cluster-map", str(tmp_path / "clusters.json"),
                             "--primary-outputs", "1000", "--objective", "cvar", "--restarts", "4",
                             "--active-sites", "6", "--resamples", "200",
                             "--out-json", str(out_json), "--out-md", str(out_md)]) == 0
    payload = json.loads(out_json.read_text())
    assert payload["hypothesis_tests"] is None and [r["depth"] for r in payload["per_depth"]] == [1, 2]
    second = payload["per_depth"][1]
    assert second["trained_log10_amplification_exact"]["mean"] == pytest.approx(2.0)
    assert second["transfer_minus_trained_log10_amplification_exact"]["mean"] == pytest.approx(-0.4)


def test_exploration_config_is_validated():
    config = yaml.safe_load((REPO / "configs" / "full_experiment_config.yaml").read_text(encoding="utf-8"))
    full._validate_scientific_config(copy.deepcopy(config))
    config["quantum_exploration"]["evals_per_parameter"] = 1
    with pytest.raises(ValueError, match="evals_per_parameter"):
        full._validate_scientific_config(config)


def test_confirmatory_family_is_quantum_intrinsic():
    source = inspect.getsource(full.Orchestrator.stage_statistics)
    assert '"amplification:primary"' in source and '"scaling:primary"' in source
    assert 'effect["gatekeeping_family"]="secondary"' in source


def test_transfer_parameter_file_is_validated(tmp_path: Path):
    good = dict(schema="qaoa_transfer_parameters_v1", fit_split="train", parameter_scale_mode="max_coefficient",
                entries={"p2_sites6": dict(depth=2, active_sites=6, internal_gammas=[0.1, 0.2], betas=[0.3, 0.4])})
    path = tmp_path / "t.json"
    path.write_text(json.dumps(good))
    assert bbh.load_transfer_parameters(path)["entries"]["p2_sites6"]["depth"] == 2
    for broken in (dict(good, fit_split="test_snac_hard"),
                   dict(good, entries={"p2_sites6": dict(depth=2, active_sites=6, internal_gammas=[0.1], betas=[0.3, 0.4])})):
        path.write_text(json.dumps(broken))
        with pytest.raises(ValueError):
            bbh.load_transfer_parameters(path)
