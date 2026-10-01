"""The fixed-backbone optimizer must actually reduce movable-atom forces."""

import numpy as np
import openmm as mm
from openmm import unit

from nanoqc.qubo.allatom_qubo import (
    _capped_lbfgs, _internal_candidate_overlaps, _minimize_movable_positions,
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
    # Stops on the exact force criterion: RMS of k*x over 3 coordinates <= 10.
    assert 1000.0 * np.linalg.norm(final[1]) / np.sqrt(3) <= 10.0
    assert provenance["minimizer"] == "lbfgs_exact_movable_capped_atom_step"
    assert provenance["minimizer_force_verified_success"]
    assert provenance["minimizer_stop_reason"] == "force_tolerance"
    assert provenance["minimizer_iterations"] <= 200


def test_huge_energy_with_large_force_does_not_stop_on_relative_change() -> None:
    """A severely clashing pose has an enormous energy whose relative change is
    tiny while forces are still large; only the force criterion may stop."""
    def objective(x):
        return 1e15 + 0.5 * 1000.0 * float(x @ x), 1000.0 * x
    result = _capped_lbfgs(objective, np.array([0.5, 0.0, 0.0]), max_iterations=500,
                           max_atom_step=0.03, force_tolerance=10.0)
    assert result["stop"] == "force_tolerance"
    assert np.sqrt(np.mean(result["gradient"] ** 2)) <= 10.0


def test_every_iteration_moves_each_atom_at_most_the_step_cap() -> None:
    steps = []
    target = np.array([1.0, -2.0, 0.5, 0.0, 0.0, 3.0])

    def objective(x):
        d = x - target
        return 0.5 * 100.0 * float(d @ d), 100.0 * d
    result = _capped_lbfgs(objective, np.zeros(6), max_iterations=1000, max_atom_step=0.03,
                           force_tolerance=1e-3, callback=lambda x, s: steps.append(s.copy()))
    assert result["stop"] == "force_tolerance"
    assert np.allclose(result["x"], target, atol=1e-4)
    per_atom = np.linalg.norm(np.array(steps).reshape(len(steps), -1, 3), axis=2)
    assert per_atom.max() <= 0.03 + 1e-12
    # Atom 2 travels 3 nm, so the cap must not end the run before it arrives.
    assert result["iterations"] >= 3.0 / 0.03


def test_iteration_cap_is_reported_without_success() -> None:
    def objective(x):
        return 0.5 * float(x @ x), x
    result = _capped_lbfgs(objective, np.array([10.0, 0.0, 0.0]), max_iterations=5,
                           max_atom_step=0.03, force_tolerance=1e-6)
    assert result["stop"] == "iteration_cap" and result["iterations"] == 5


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


def _tethered_atom_near_unscreened_charge(well_x: float):
    """One movable atom tethered at the origin and attracted to a point with no
    short-range repulsion, like an Amber HO hydrogen near an oppositely charged
    carbon. The physical minimum sits next to the tether; the well is singular."""
    system = mm.System()
    system.addParticle(0.0)
    system.addParticle(1.0)
    force = mm.CustomExternalForce("0.5*k*(x^2+y^2+z^2) - c/sqrt((x-ax)^2+y^2+z^2)")
    force.addGlobalParameter("k", 1000.0)
    force.addGlobalParameter("c", 1.0)
    force.addGlobalParameter("ax", well_x)
    force.addParticle(1, [])
    system.addForce(force)
    return mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))


def test_minimizer_does_not_jump_into_an_unscreened_coulomb_singularity() -> None:
    """A46: an uncapped first L-BFGS-B step (1 nm) lands in the singular well."""
    for well_x in (-0.5, -0.502, -0.51):
        context = _tethered_atom_near_unscreened_charge(well_x)
        start = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        final, record = _minimize_movable_positions(context, start, {1}, 1000, unit)
        assert abs(final[1, 0] - well_x) > 0.4, "collapsed into the singular well"
        assert abs(final[1, 0]) < 0.01, "should stop at the tethered physical minimum"
        assert record["minimizer_force_verified_success"]
