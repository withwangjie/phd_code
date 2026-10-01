"""Geometry-only admission for prospective structural recovery inputs."""

import numpy as np
import pytest
from openmm.app import Topology, element

from nanoqc.experiments import structure_benchmarks
from nanoqc.structure.physical_quality import (
    StructureQualityError, generated_input_geometry_audit,
)


def _two_atom_topology(*, same_residue=False, second_element=element.carbon):
    topology = Topology()
    chain = topology.addChain("H")
    first = topology.addResidue("ALA", chain, id="1")
    topology.addAtom("CA", element.carbon, first)
    second = first if same_residue else topology.addResidue(
        "ALA", topology.addChain("A"), id="2")
    topology.addAtom("CB", second_element, second)
    return topology


def test_generated_input_rejects_cross_residue_heavy_clash_above_all_atom_floor():
    topology = _two_atom_topology()
    audit = generated_input_geometry_audit(topology, np.array([[0, 0, 0], [.08, 0, 0]]))
    assert audit["all_atom"]["geometry_passed"]
    assert audit["interresidue_heavy_overlap_count"] == 1
    assert not audit["accepted"]


def test_generated_input_distinguishes_same_residue_and_hydrogen_pairs():
    same = generated_input_geometry_audit(
        _two_atom_topology(same_residue=True), np.array([[0, 0, 0], [.08, 0, 0]]))
    assert same["accepted"]
    hydrogen = generated_input_geometry_audit(
        _two_atom_topology(second_element=element.hydrogen),
        np.array([[0, 0, 0], [.03, 0, 0]]))
    assert hydrogen["interresidue_heavy_overlap_count"] == 0
    assert not hydrogen["all_atom"]["geometry_passed"]
    assert not hydrogen["accepted"]


def test_generated_input_first_valid_draw_and_exhaustion_are_audited(monkeypatch):
    calls = []

    class Generator:
        topology = object()

        def perturb_sidechain_chis(self, seed, minimum, maximum):
            calls.append(seed)
            return np.array([len(calls)]), {"chi": len(calls)}

    monkeypatch.setattr(structure_benchmarks, "generated_input_geometry_audit",
                        lambda _topology, positions, **_floor: {"accepted": positions[0] == 3})
    positions, _, attempts = structure_benchmarks._qualified_perturbation(
        Generator(), 42, "multi_chi", 40, 120, 4, heavy_floor_angstrom=1.0)
    assert positions.tolist() == [3]
    assert len(attempts) == 3
    assert attempts[0]["draw_seed"] == 42
    assert len(set(calls)) == 3
    calls.clear()
    with pytest.raises(StructureQualityError) as exc:
        structure_benchmarks._qualified_perturbation(
            Generator(), 42, "multi_chi", 40, 120, 2, heavy_floor_angstrom=1.0)
    assert exc.value.category == "generated_input_geometry"
    assert len(exc.value.audit["attempts"]) == 2


def test_configured_heavy_floor_is_the_floor_applied_and_recorded():
    """The ledger's floor must be the floor the check used (merge of A44 into A45)."""
    topology = _two_atom_topology()

    class Generator:
        def __init__(self):
            self.topology = topology

        def perturb_sidechain_chis(self, seed, minimum, maximum):
            # A 0.8 A cross-residue heavy pair: invalid at 1.0 A, valid at 0.5 A.
            return np.array([[0, 0, 0], [.08, 0, 0]]), {"seed": seed}

    with pytest.raises(StructureQualityError) as rejected:
        structure_benchmarks._qualified_perturbation(
            Generator(), 42, "multi_chi", 40, 120, 3, heavy_floor_angstrom=1.0)
    first = rejected.value.audit["attempts"][0]["quality"]
    assert first["source_heavy_floor_angstrom"] == 1.0
    assert first["interresidue_heavy_overlap_count"] == 1

    _, _, attempts = structure_benchmarks._qualified_perturbation(
        Generator(), 42, "multi_chi", 40, 120, 3, heavy_floor_angstrom=0.5)
    assert len(attempts) == 1
    assert attempts[0]["quality"]["source_heavy_floor_angstrom"] == 0.5
    assert attempts[0]["quality"]["accepted"]
    with pytest.raises(ValueError, match="floor"):
        generated_input_geometry_audit(topology, np.zeros((2, 3)), heavy_floor_angstrom=1.5)
