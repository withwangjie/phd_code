"""A54: geometry-forbidden penalties stay in the cost but not in the angle scale."""
import numpy as np

from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler


def _sampler(**kw):
    # Two sites x two states. Variable 1 is a forbidden state priced at F = 1e6.
    physical_self = np.array([1.0, 1e6, 0.5, 2.0])
    physical_pair = np.zeros((4, 4))
    physical_pair[0, 2] = -0.75
    physical_pair[0, 3] = 1e6                     # a forbidden pair
    return XYMixerQAOASampler(physical_self, physical_pair, {0: (0, 1), 1: (2, 3)},
                              p=1, shots=64, simulation_mode="subspace", **kw)


def test_scale_ignores_forbidden_terms_but_cost_keeps_them() -> None:
    plain = _sampler()
    masked = _sampler(scale_excluded_variables=[1], scale_excluded_pairs=[[0, 3]])
    assert plain.gamma_scale > 1e5                     # the penalty set the old scale
    assert masked.gamma_scale < 10                     # physical terms only
    assert np.array_equal(plain.ising_h, masked.ising_h)
    assert np.array_equal(plain.ising_J, masked.ising_J)
    assert plain.feasible_energy_map() == masked.feasible_energy_map()
    assert plain.enumerate_ground_states().energy == masked.enumerate_ground_states().energy


def test_iqr_scale_uses_admissible_states_only() -> None:
    masked = _sampler(scale_excluded_variables=[1], scale_excluded_pairs=[[0, 3]])
    plain = _sampler()
    assert masked.parameter_scale("feasible_iqr") < plain.parameter_scale("feasible_iqr")


def test_without_forbidden_states_nothing_changes() -> None:
    a = _sampler()
    b = _sampler(scale_excluded_variables=(), scale_excluded_pairs=())
    assert a.gamma_scale == b.gamma_scale
    assert a.parameter_scale("feasible_iqr") == b.parameter_scale("feasible_iqr")


def test_forbidden_penalty_separates_every_assignment_exhaustively() -> None:
    import itertools
    from nanoqc.qubo.allatom_qubo import _forbidden_state_penalty
    rng = np.random.default_rng(5)
    sites = {0: (0, 1, 2), 1: (3, 4), 2: (5, 6, 7)}
    n = 8
    for trial in range(200):
        singles = rng.normal(scale=50, size=n)
        pairs = np.triu(rng.normal(scale=30, size=(n, n)), 1)
        for group in sites.values():
            for a in group:
                for b in group:
                    if a < b:
                        pairs[a, b] = 0.0
        single_ok = rng.random(n) > 0.2
        pair_ok = rng.random((n, n)) > 0.2
        pair_ok = pair_ok & pair_ok.T
        F = _forbidden_state_penalty(singles, pairs, sites, single_ok, pair_ok)
        model_s = np.where(single_ok, singles, F)
        model_p = pairs.copy()
        for i in range(n):
            for j in range(i + 1, n):
                if not (single_ok[i] and single_ok[j]):
                    model_p[i, j] = 0.0
                elif not pair_ok[i, j]:
                    model_p[i, j] = F
        admissible, forbidden = [], []
        for chosen in itertools.product(*sites.values()):
            e = sum(model_s[v] for v in chosen) + sum(model_p[min(a, b), max(a, b)]
                                                      for a, b in itertools.combinations(chosen, 2))
            ok = all(single_ok[v] for v in chosen) and all(pair_ok[a, b] for a, b in itertools.combinations(chosen, 2))
            (admissible if ok else forbidden).append(e)
        if admissible and forbidden:
            assert min(forbidden) > max(admissible)


def test_sa_temperature_scale_ignores_forbidden_terms() -> None:
    masked = _sampler(scale_excluded_variables=[1], scale_excluded_pairs=[[0, 3]])
    result = masked.simulated_annealing(num_reads=8, site_passes=5, seed=1)
    best = min(result.counts, key=lambda bits: masked.feasible_energy_map()[bits])
    assert masked.feasible_energy_map()[best] == masked.enumerate_ground_states().energy
