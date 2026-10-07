"""A48: calibration rows use only geometry-admissible rotamer assignments."""
from types import SimpleNamespace

import numpy as np

from nanoqc.experiments.generate_energy_calibration_dataset import calibration_assignments


def _qubo():
    # Two sites, three states each; state energies make (0, 3) the ground state.
    physical_self = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    return SimpleNamespace(site_to_variables={0: (0, 1, 2), 1: (3, 4, 5)},
                           physical_self=physical_self, physical_pair=np.zeros((6, 6)))


def test_impossible_states_are_never_sampled_and_the_anchor_is_admissible() -> None:
    impossible = {(0, 3), (1, 4)}
    chosen = calibration_assignments(_qubo(), 6, np.random.default_rng(1),
                                     admissible=lambda s: tuple(s) not in impossible)
    assert not {tuple(s) for s in chosen} & impossible
    # The lowest-energy admissible state is the anchor.
    assert chosen[0] in ([0, 4], [1, 3])
    assert len(chosen) == 6


def test_without_a_filter_the_former_sampling_is_unchanged() -> None:
    chosen = calibration_assignments(_qubo(), 4, np.random.default_rng(1))
    assert chosen[0] == [0, 3]
