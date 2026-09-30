"""The fixed-backbone optimizer must actually reduce movable-atom forces."""

import numpy as np
import openmm as mm
from openmm import unit

from nanoqc.qubo.allatom_qubo import (
    _internal_candidate_overlaps, _minimize_movable_positions,
)


def test_exact_movable_minimization_leaves_frozen_atom_unchanged() -> None:
    system = mm.System()
    system.addParticle(0.0)
    system.addParticle(12.0)
    force = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", 1000.0)
    for name in ("x0", "y0", "z0"):
        force.addPerParticleParameter(name)
    force.addParticle(1, [0.0, 0.0, 0.0])
    system.addForce(force)
    integrator = mm.VerletIntegrator(0.001)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    start = np.array([[2.0, 3.0, 4.0], [1.0, -1.0, 0.5]])
    final, provenance = _minimize_movable_positions(context, start, {1}, 200, unit)
    assert np.array_equal(final[0], start[0])
    assert np.linalg.norm(final[1]) < 1e-6
    assert provenance["minimizer"] == "scipy_lbfgsb_exact_movable"
    assert provenance["minimizer_iterations"] <= 200


def test_candidate_screen_rejects_only_nonbonded_internal_coincidences() -> None:
    coordinates = np.array([[0., 0., 0.], [0.1, 0., 0.],
                            [0.2, 0., 0.], [0.001, 0., 0.]])
    atoms = dict(A=0, B=1, C=2, D=3)
    bonds = {0: {1}, 1: {0, 2}, 2: {1}, 3: set()}
    overlap = _internal_candidate_overlaps(coordinates, atoms, bonds)
    assert len(overlap) == 1
    assert overlap[0]["atoms"] == ["A", "D"]
    coordinates[3] = [1., 0., 0.]
    assert _internal_candidate_overlaps(coordinates, atoms, bonds) == []
