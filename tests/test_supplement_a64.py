"""A64 supplementary experiments: simulator correctness and end-to-end runs."""
import json
import math

import numpy as np
import pytest

from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler
from nanoqc.supplement import noise_simulation as noise
from nanoqc.supplement.common import PRIMARY_OPTIMIZATION_KEY, feasible_ground


def _instance(sites=3, states=3, seed=0):
    rng = np.random.default_rng(seed)
    m = sites * states
    pair = np.triu(rng.normal(size=(m, m)), 1)
    groups = {s: list(range(s * states, (s + 1) * states)) for s in range(sites)}
    for g in groups.values():
        for a in g:
            for b in g:
                pair[min(a, b), max(a, b)] = 0.0 if a != b else pair[min(a, b), max(a, b)]
    return rng.normal(size=m), pair, groups


def _sampler(seed=0, p=2):
    h, J, groups = _instance(seed=seed)
    return XYMixerQAOASampler(h, J, groups, p=p, simulation_mode="subspace")


def test_full_register_matches_exact_subspace():
    sampler = _sampler()
    params = np.array([0.31, -0.17, 0.42, 0.23])
    circuit = noise.FullRegisterCircuit(sampler, params)
    full = circuit.run()
    assert full.sum() == pytest.approx(1.0)
    assert full[~circuit.feasible].sum() < 1e-12          # noiseless XY-QAOA stays one-hot
    sub = np.abs(sampler.subspace_state(params)) ** 2
    for index, bits in enumerate(sampler.feasible_energy_map()):
        assert full[circuit.index_of(bits)] == pytest.approx(sub[index], abs=1e-12)


def test_pauli_errors_leave_the_feasible_space_and_keep_norm():
    sampler = _sampler(seed=1)
    circuit = noise.FullRegisterCircuit(sampler, np.array([0.2, 0.1, 0.3, 0.4]))
    first_xy = next(i for i, g in enumerate(circuit.gates) if g[0] == "xy")
    probabilities = circuit.run({first_xy: 4})              # X on the first wire
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[~circuit.feasible].sum() > 0.5


def test_white_noise_limits_and_critical_epsilon():
    full = noise.white_noise(0.4, 1, 27, 9, 1.0)
    assert full["postselected"] == pytest.approx(0.4)
    dead = noise.white_noise(0.4, 1, 27, 9, 0.0)
    assert dead["log10_amplification"] == pytest.approx(0.0)  # random = uniform feasible
    e1 = noise.critical_epsilon(0.4, 1, 27, 9, 100, 0.0)
    e10 = noise.critical_epsilon(0.4, 1, 27, 9, 100, 0.5)
    assert 0 < e10 < e1 <= 1.0


def test_trajectories_agree_with_white_noise_in_expectation():
    sampler = _sampler(seed=2)
    params = np.array([0.3, -0.2, 0.5, 0.25])
    circuit = noise.FullRegisterCircuit(sampler, params)
    ground, _, omega = feasible_ground(sampler)
    idx = [circuit.index_of(b) for b in ground]
    ideal = circuit.run()
    row = noise.trajectory_metrics(circuit, idx, omega, 0.02, 400, np.random.default_rng(3), ideal)
    # The white-noise model is an approximation; it should be in the same range.
    assert abs(row["feasible_probability"] - row["white_noise_feasible_probability"]) < 0.15
    assert row["raw_ground_probability"] < float(ideal[idx].sum())


def _write_case(run, name, sampler, sites, pdb, depth=2):
    params = np.array([0.3, -0.2, 0.5, 0.25])
    ground, energy, omega = feasible_ground(sampler)
    probs = np.abs(sampler.subspace_state(params)) ** 2
    keys = list(sampler.feasible_energy_map())
    exact = float(sum(probs[i] for i, b in enumerate(keys) if tuple(b) in set(ground)))
    circuit = noise.FullRegisterCircuit(sampler, params)
    metrics = [dict(solver="qaoa", qaoa_objective="cvar", qaoa_restarts=4, outputs=1000,
                    exact_ground_probability=exact,
                    log10_ground_amplification_exact=math.log10(exact / (len(ground) / omega)),
                    qaoa_two_qubit_gates=circuit.two_qubit_gates)]
    case = dict(
        config=dict(pdb_id=pdb, pruning="egnn", radius=6.0, depth=depth, max_evals=90, active_sites=sites),
        metrics=metrics, ground_energy=energy,
        optimizations={PRIMARY_OPTIMIZATION_KEY: dict(gammas=params[:2].tolist(), betas=params[2:].tolist(),
                                                       total_opt_shots=45000)},
        quantum_instance=dict(physical_self=sampler.physical_self.tolist(),
                              physical_pair=sampler.physical_pair.tolist(),
                              site_to_variables={str(k): v for k, v in sampler.site_to_variables.items()}))
    folder = run / "qc_benchmark" / "cases"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(case))


def test_noise_main_end_to_end(tmp_path):
    _write_case(tmp_path, "a.json", _sampler(seed=4), 3, "1abc")
    _write_case(tmp_path, "b.json", _sampler(seed=5), 3, "2xyz")
    assert noise.main(["--run-dir", str(tmp_path), "--trajectory-sites", "3", "--trajectories", "5",
                       "--epsilons", "0", "0.001", "--compiled-two-qubit-gates", "3:60"]) == 0
    out = tmp_path / "analysis" / "a64" / "noise"
    assert (out / "noise_summary.md").is_file()
    rows = (out / "noise_white_model.csv").read_text().splitlines()
    assert len(rows) == 1 + 2 * 2                            # 2 cases x (logical, compiled)


def test_errors_inside_diagonal_segments_match_naive_reference():
    sampler = _sampler(seed=6)
    circuit = noise.FullRegisterCircuit(sampler, np.array([0.3, -0.2, 0.5, 0.25]))
    zz = [i for i, g in enumerate(circuit.gates) if g[0] == "zz"]
    xy = [i for i, g in enumerate(circuit.gates) if g[0] == "xy"]
    errors = {zz[1]: 7, zz[-1]: 13, xy[2]: 2}
    labels = ("I", "X", "Y", "Z")
    state = circuit.initial_state()
    for position, gate in enumerate(circuit.gates):
        state = circuit.apply(state, gate)
        if position in errors:
            state = circuit.pauli(state, gate[1][0], labels[errors[position] // 4])
            state = circuit.pauli(state, gate[1][1], labels[errors[position] % 4])
    assert np.allclose(circuit.run(errors), np.abs(state) ** 2, atol=1e-12)


def test_exact_solvers_agree_with_sampler_energies():
    from nanoqc.supplement import classical_baselines as cb
    for seed in range(3):
        sampler = _sampler(seed=10 + seed)
        groups = [list(g) for g in sampler.site_to_variables.values()]
        h, P = np.asarray(sampler.physical_self), np.asarray(sampler.physical_pair)
        bf, bf_choice, _, omega = cb.brute_force(h, P, groups)
        mi, mi_x, _ = cb.exact_milp(h, P, groups)
        exact = min(sampler.feasible_energy_map().values())
        assert bf == pytest.approx(exact, abs=1e-9) and mi == pytest.approx(exact, abs=1e-7)
        assert omega == 27
        bits = np.zeros(len(h), dtype=int); bits[bf_choice] = 1
        assert sampler.physical_energy(bits) == pytest.approx(exact, abs=1e-9)


def test_reads_to_target():
    from nanoqc.supplement.classical_baselines import reads_to_target
    assert reads_to_target(1.0) == 1 and reads_to_target(0.0) == math.inf
    assert reads_to_target(0.5) == 7                       # 1 - 0.5**7 >= 0.99 > 1 - 0.5**6


def test_classical_main_end_to_end(tmp_path):
    from nanoqc.supplement import classical_baselines as cb
    _write_case(tmp_path, "a.json", _sampler(seed=4), 3, "1abc")
    _write_case(tmp_path, "b.json", _sampler(seed=5), 3, "2xyz")
    assert cb.main(["--run-dir", str(tmp_path), "--sites", "3", "--passes", "1", "5",
                    "--reads", "20", "--resamples", "200"]) == 0
    out = tmp_path / "analysis" / "a64" / "classical"
    text = (out / "classical_summary.md").read_text()
    assert "MILP and brute force agree" in text
    assert len((out / "sa_budget_sweep.csv").read_text().splitlines()) == 1 + 2 * 2


def _fake_run(tmp_path):
    import yaml
    config = dict(master_seed=4050350448, paths=dict(repo_root=str(tmp_path)),
                  structure_experiment=dict(seeds=[42, 43], solvent_model="vacuum"),
                  quantum_protocol=dict(primary=dict(depth=2, max_evals=90, output_shots=1000, restarts=4,
                                                     objective="cvar", cvar_alpha=0.1, eval_shots=500,
                                                     parameter_scale="max_coefficient")))
    (tmp_path / "provenance").mkdir()
    (tmp_path / "provenance" / "resolved_runtime_config.yaml").write_text(yaml.safe_dump(config))
    return config


def test_forcefield_argv_is_accepted_by_the_recovery_parser(tmp_path, monkeypatch):
    from nanoqc.supplement import forcefield_sensitivity as ff
    from nanoqc.experiments import structure_benchmarks as sb
    from nanoqc.common.seed_streams import derive_child_seed, derive_streams
    config = _fake_run(tmp_path)
    argv = ff.job_argv(config, tmp_path / "missing.json", tmp_path / "out", "7abc", [42, 43], "charmm36")
    assert argv[argv.index("--structure-force-field") + 1] == "charmm36"
    expected = derive_child_seed(derive_streams(config["master_seed"])["optimize"], "structure", "7abc", "42")
    assert int(argv[argv.index("--optimize-seeds") + 1]) == expected   # identical to the pilot's derivation
    try:
        sb._recovery_benchmark_main(argv)
    except SystemExit as exc:                                          # argparse rejection would be exit 2
        assert exc.code != 2
    except Exception:
        pass                                                           # missing manifest, after parsing


def test_forcefield_compare_pairs_rows(tmp_path):
    from nanoqc.supplement import forcefield_sensitivity as ff
    import csv as _csv
    fields = ["target", "seed", "method", "evaluation_status", "final_rmsd", "chi1_recovery_final", "initial_rmsd"]
    def write(path, offset):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as h:
            w = _csv.DictWriter(h, fieldnames=fields); w.writeheader()
            for t_rmsd, method in ((1.0, "qaoa"), (1.1, "sa"), (1.05, "uniform"), (1.0, "greedy")):
                w.writerow(dict(target="7abc", seed=42, method=method, evaluation_status="passed",
                                final_rmsd=t_rmsd + offset, chi1_recovery_final=0.8, initial_rmsd=2.0))
    write(tmp_path / "validation_queue/results/7abc/recovery_metrics.csv", 0.0)
    write(tmp_path / "analysis/a64/forcefield/charmm36/results/7abc/recovery_metrics.csv", 0.2)
    write(tmp_path / "analysis/a64/forcefield/amber14_a50/results/7abc/recovery_metrics.csv", 0.0)
    rows, summary = ff.compare(tmp_path, ["charmm36", "amber14_a50"], [42])
    assert len(rows) == 8
    assert "+0.200" in summary
    assert "Reproduction control: 4/4" in summary


def _tabulated_system(nbfix=False):
    import openmm as mm
    sig = [0.30, 0.35, 0.25]; eps = [0.5, 0.2, 0.1]
    A = np.zeros((3, 3)); B = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.5 * (sig[i] + sig[j]); e = math.sqrt(eps[i] * eps[j])
            A[i, j] = 4 * e * s ** 12; B[i, j] = 4 * e * s ** 6
    if nbfix:
        A[0, 1] = A[1, 0] = A[0, 1] * 1.3
    system = mm.System(); nb = mm.NonbondedForce(); nb.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    custom = mm.CustomNonbondedForce("acoef(type1, type2)/r^12 - bcoef(type1, type2)/r^6;")
    custom.addTabulatedFunction("acoef", mm.Discrete2DFunction(3, 3, A.T.flatten().tolist()))
    custom.addTabulatedFunction("bcoef", mm.Discrete2DFunction(3, 3, B.T.flatten().tolist()))
    custom.addPerParticleParameter("type")
    for i, charge in enumerate((0.3, -0.2, 0.1, -0.2)):
        system.addParticle(12.0); nb.addParticle(charge, 0.1, 0.0); custom.addParticle([i % 3])
    nb.addException(0, 1, 0.0, 0.1, 0.0); custom.addExclusion(0, 1)
    system.addForce(nb); system.addForce(custom)
    positions = np.array([[0, 0, 0], [0.4, 0, 0], [0, 0.45, 0], [0.5, 0.5, 0.1]], dtype=float)
    return system, positions


def test_tabulated_lj_transfer_is_exact_for_lorentz_berthelot():
    import openmm as mm
    from nanoqc.qubo.allatom_qubo import _tabulated_lj_to_nonbonded
    system, positions = _tabulated_system()
    audit = _tabulated_lj_to_nonbonded(system, positions)
    assert audit["converted"] and audit["lj_types"] == 3
    assert abs(audit["energy_after_kj_mol"] - audit["energy_before_kj_mol"]) < 1e-9
    assert not any(isinstance(f, mm.CustomNonbondedForce) for f in system.getForces())


def test_tabulated_lj_transfer_refuses_nbfix():
    from nanoqc.qubo.allatom_qubo import _tabulated_lj_to_nonbonded
    system, positions = _tabulated_system(nbfix=True)
    with pytest.raises(ValueError, match="NBFIX"):
        _tabulated_lj_to_nonbonded(system, positions)
