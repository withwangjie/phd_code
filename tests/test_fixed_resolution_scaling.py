"""Fixed-resolution scaling retains one state from each chi1 well."""

from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from nanoqc.inference.analyze_quantum_scaling import _validate_fixed_state_case
from nanoqc.qubo.subgraph_to_qubo import AA_INDEX, InterfaceQUBOBuilder, select_chi1_well_representatives


def test_chi1_selection_rejects_three_states_from_one_well():
    states = [SimpleNamespace(chi1_degrees=angle) for angle in (-65, -55, -45, 60, 180)]
    chosen = select_chi1_well_representatives(states)
    assert [state.chi1_degrees for state in chosen] == [-65, 60, 180]
    chosen_four = select_chi1_well_representatives(states, 4)
    assert [state.chi1_degrees for state in chosen_four] == [-65, -55, 60, 180]

    try:
        select_chi1_well_representatives(states[:3])
    except ValueError as exc:
        assert "each chi1 well" in str(exc)
    else:
        raise AssertionError("One-well candidate pool was accepted")


def test_scaling_analysis_rejects_legacy_adaptive_case():
    case = {
        "config": {"state_policy": "fixed_three_chi1_wells"},
        "site_to_variables": {"0": [0, 1, 2]},
        "variable_map": [
            {"site_index": 0, "chi1_degrees": angle}
            for angle in (-60, 60, 180)
        ],
    }
    _validate_fixed_state_case(case, 1, 3)
    case["config"]["state_policy"] = "adaptive_3_to_6"
    try:
        _validate_fixed_state_case(case, 1, 3)
    except ValueError as exc:
        assert "fixed three-chi1-well" in str(exc)
    else:
        raise AssertionError("Legacy adaptive scaling case was accepted")


def test_four_site_six_state_qubo_keeps_fixed_resolution():
    features = torch.zeros((6, 21))
    features[:4, AA_INDEX["S"]] = 1
    features[4:, AA_INDEX["Y"]] = 1
    features[4:, 20] = 1
    positions = torch.tensor([
        [0., 0., 0.], [3.8, 0., 0.], [7.6, 0., 0.], [11.4, 0., 0.],
        [0., 5., 0.], [5., 5., 0.],
    ])
    graph = Data(x=features, pos=positions,
                 is_active=torch.tensor([1, 1, 1, 1, 0, 0], dtype=torch.bool))
    qubo = InterfaceQUBOBuilder(
        min_variables=24, max_variables=24, max_sites=4,
        rotamer_mode="legacy", fixed_chi1_wells=True, fixed_states_per_site=6,
    ).build(graph)

    assert len(qubo.variable_map) == 24
    assert all(len(variables) == 6 for variables in qubo.site_to_variables.values())
    assert qubo.metadata["state_policy"] == "fixed_6_chi1_coverage"
