"""Resource-normalized primary endpoint (log10 queries-to-solution, 99%)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nanoqc.experiments.batch_benchmark_hard_set import (
    _ablation_classical_counts, _ablation_summarize, queries_to_solution,
)
from nanoqc.inference import analyze_quantum_scaling as scaling
from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler


def test_queries_to_solution_formula_and_jeffreys_estimate():
    energies = {(1, 0): -1.0, (0, 1): 0.0}
    counts = {(1, 0): 3, (0, 1): 7}
    result = queries_to_solution(counts, energies, -1.0, fixed_units=100.0, units_per_sample=2.0)
    p = (3 + 0.5) / (10 + 1)
    expected = 100.0 + 2.0 * math.log(0.01) / math.log(1 - p)
    assert result["ground_hits"] == 3
    assert result["success_probability_jeffreys"] == pytest.approx(p)
    assert result["queries_to_solution_99"] == pytest.approx(expected)
    assert result["log10_qts99"] == pytest.approx(math.log10(expected))
    # Zero hits stays finite instead of infinite/undefined.
    missed = queries_to_solution({(0, 1): 10}, energies, -1.0, fixed_units=0.0, units_per_sample=1.0)
    assert missed["ground_hits"] == 0 and math.isfinite(missed["log10_qts99"])
    with pytest.raises(ValueError):
        queries_to_solution(counts, energies, -1.0, fixed_units=0.0, units_per_sample=0.0)


def test_endpoint_separates_methods_that_saturate_best_of_n():
    rng = np.random.default_rng(3)
    sites, k = 6, 3
    groups = {s: list(range(s * k, (s + 1) * k)) for s in range(sites)}
    h = rng.normal(0, 2, sites * k)
    J = np.triu(rng.normal(0, 1, (sites * k, sites * k)), 1)
    for group in groups.values():
        for a in group:
            for b in group:
                if a < b:
                    J[a, b] = 0.0
    sampler = XYMixerQAOASampler(h, J, groups, p=2, seed=1, simulation_mode="subspace")
    truth = sampler.enumerate_ground_states()
    energies = sampler.feasible_energy_map()
    sa = sampler.simulated_annealing(num_reads=300, site_passes=100, seed=4, ground_state=truth)
    sa_counts = dict(sa.counts)
    uniform_counts, _ = _ablation_classical_counts(sampler, 300, 5, False, 50)
    sa_summary = _ablation_summarize(sa_counts, energies, truth.energy, 2.0)
    assert sa_summary["hit"] == 1 and sa_summary["gap"] == 0.0   # best-of-N saturated
    sa_qts = queries_to_solution(sa_counts, energies, truth.energy,
                                 fixed_units=0.0, units_per_sample=1 + 100 * sites)
    uniform_qts = queries_to_solution(uniform_counts, energies, truth.energy,
                                      fixed_units=0.0, units_per_sample=1.0)
    assert math.isfinite(sa_qts["log10_qts99"]) and math.isfinite(uniform_qts["log10_qts99"])
    assert abs(sa_qts["log10_qts99"] - uniform_qts["log10_qts99"]) > 0.1


def _scaling_case(pdb: str, sites: int, delta: float) -> dict:
    wells = (60.0, -60.0, 180.0)
    variable_map = [dict(site_index=s, chi1_degrees=wells[i]) for s in range(sites) for i in range(3)]
    site_to_variables = {str(s): [3 * s, 3 * s + 1, 3 * s + 2] for s in range(sites)}
    shared = dict(budget_mode="matched_outputs", outputs=1000, configuration_count=3 ** sites,
                  num_bits=3 * sites, qaoa_parameter_count=4, qaoa_xy_gates=6 * sites,
                  qaoa_zz_gates=sites, qaoa_two_qubit_gates=7 * sites, gap=0.0, hit=1)
    rows = [dict(shared, solver="qaoa", qaoa_objective="cvar", qaoa_restarts=4,
                 termination_reason="max_evaluations_reached", log10_qts99=4.0 + delta,
                 log10_qts99_execution=2.0 + delta, log10_ground_amplification_exact=1.0 + delta),
            dict(shared, solver="sa", log10_qts99=4.0, log10_qts99_execution=2.0)]
    return dict(config=dict(pruning="egnn", active_sites=sites, depth=2, max_evals=90, radius=6.0,
                            seed=1, pdb_id=pdb, state_policy="fixed_three_chi1_wells"),
                metrics=rows, site_to_variables=site_to_variables, variable_map=variable_map)


def test_scaling_uses_quantum_amplification_as_primary_response(tmp_path: Path):
    cases = tmp_path / "cases"
    cases.mkdir()
    slope = 0.5
    cluster_map = {}
    for index in range(4):
        pdb = f"p{index}"
        cluster_map[pdb] = f"c{index}"
        for sites in (4, 6, 8):
            delta = slope * math.log10(3 ** sites)
            (cases / f"{pdb}_{sites}.json").write_text(json.dumps(_scaling_case(pdb, sites, delta)))
    (tmp_path / "clusters.json").write_text(json.dumps(cluster_map))
    out_json, out_md = tmp_path / "scaling.json", tmp_path / "scaling.md"
    assert scaling.main([
        "--results-dir", str(tmp_path), "--cluster-map", str(tmp_path / "clusters.json"),
        "--out-json", str(out_json), "--out-md", str(out_md), "--active-sites", "4", "6", "8",
        "--resamples", "1000",
    ]) == 0
    payload = json.loads(out_json.read_text())
    assert payload["primary_response"] == "qaoa_log10_amplification_exact"
    assert payload["primary"]["mean_slope"] == pytest.approx(slope)
    amplification = payload["primary_amplification"]
    assert amplification["active_sites"] == 6
    assert amplification["mean_log10_amplification"] == pytest.approx(1.0 + slope * math.log10(3 ** 6))
    assert payload["per_pdb"][0]["descriptive_slopes"]["delta_log10_qts99"] == pytest.approx(slope)
    assert "amplification" in out_md.read_text()
