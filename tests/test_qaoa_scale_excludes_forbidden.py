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
