"""The fixed-backbone optimizer must actually reduce movable-atom forces."""

import numpy as np
import openmm as mm
from types import SimpleNamespace
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


class _SaturatingState:
    def __init__(self, state, limit):
        self.state, self.limit = state, limit

    def getPotentialEnergy(self):
        return self.state.getPotentialEnergy()

    def getForces(self, asNumpy=True):
        forces = self.state.getForces(asNumpy=True)
        # Fixed-point accumulation with 32 fractional bits: sums past 2**31
        # wrap around in two's complement.
        exact = forces.value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        values = np.mod(exact + self.limit, 2 * self.limit) - self.limit
        return values * (unit.kilojoules_per_mole / unit.nanometer)


class _SaturatingPlatformContext:
    """Stands in for a GPU context whose fixed-point force accumulator saturates."""

    def __init__(self, context, limit):
        self.context, self.limit = context, limit

    def setPositions(self, positions):
        self.context.setPositions(positions)

    def getState(self, **kwargs):
        return _SaturatingState(self.context.getState(**kwargs), self.limit)

    def getSystem(self):
        return self.context.getSystem()

    def getPlatform(self):
        return SimpleNamespace(getName=lambda: "CUDA")


def _lennard_jones_pair(separation_nm: float):
    """A frozen and a movable atom with only a Lennard-Jones interaction."""
    system = mm.System()
    system.addParticle(0.0)
    system.addParticle(12.0)
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for _ in range(2):
        force.addParticle(0.0, 0.3, 0.5)
    system.addForce(force)
    real = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    start = np.array([[0.0, 0.0, 0.0], [separation_nm, 0.0, 0.0]])
    return real, start


def test_overlap_forces_are_recomputed_on_the_reference_platform() -> None:
    """A46: a GPU fixed-point accumulator wraps above 2**31 kJ/mol/nm."""
    from nanoqc.qubo import allatom_qubo
    limit = allatom_qubo.FIXED_POINT_FORCE_LIMIT_KJ_MOL_NM
    real, start = _lennard_jones_pair(0.05)        # force about 1e12 kJ/mol/nm
    final, record = _minimize_movable_positions(
        _SaturatingPlatformContext(real, limit), start, {1}, 1000, unit)
    assert record["minimizer_reference_platform_evaluations"] > 0
    assert record["minimizer_force_verified_success"]
    assert 0.3 < final[1, 0] < 0.4      # near the Lennard-Jones minimum at 0.337 nm
    assert np.array_equal(final[0], start[0])


def test_wrapped_forces_break_the_minimizer_without_the_reference_check(monkeypatch) -> None:
    from nanoqc.qubo import allatom_qubo
    monkeypatch.setattr(allatom_qubo._ExactForces, "needs_reference", lambda self, positions: False)
    real, start = _lennard_jones_pair(0.05)
    _, record = _minimize_movable_positions(
        _SaturatingPlatformContext(real, allatom_qubo.FIXED_POINT_FORCE_LIMIT_KJ_MOL_NM),
        start, {1}, 1000, unit)
    assert not record["minimizer_force_verified_success"]


def test_well_separated_atoms_keep_the_gpu_result() -> None:
    from nanoqc.qubo import allatom_qubo
    real, start = _lennard_jones_pair(0.6)
    exact = allatom_qubo._ExactForces(
        _SaturatingPlatformContext(real, allatom_qubo.FIXED_POINT_FORCE_LIMIT_KJ_MOL_NM), {1})
    # sigma 0.3 nm, epsilon 0.5 kJ/mol: 48*0.5*0.3**12/(2**31/64) to the 1/13.
    assert abs(exact.safe_distance_nm - (48 * 0.5 * 0.3 ** 12 / (2 ** 31 / 64)) ** (1 / 13)) < 1e-12
    assert not exact.needs_reference(start)
    assert exact.needs_reference(np.array([[0.0, 0.0, 0.0], [0.9 * exact.safe_distance_nm, 0, 0]]))


def test_an_unscreened_hydrogen_overlap_is_never_deepened() -> None:
    """A47: an Amber HO hydrogen (epsilon 0) that starts 0.2 A from an acceptor
    would collapse onto it; no step may bring such a pair closer."""
    system = mm.System()
    system.addParticle(0.0)                        # frozen acceptor
    system.addParticle(1.0)                        # movable polar hydrogen
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    nonbonded.addParticle(-0.5, 0.3, 0.5)
    nonbonded.addParticle(0.4, 1.0, 0.0)           # OpenMM's zero-epsilon convention
    system.addForce(nonbonded)
    tether = mm.CustomExternalForce("0.5*k*((x-0.1)^2+y^2+z^2)")
    tether.addGlobalParameter("k", 1e5)
    tether.addParticle(1, [])
    system.addForce(tether)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    start = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0]])
    final, record = _minimize_movable_positions(context, start, {1}, 1000, unit)
    assert np.linalg.norm(final[1] - final[0]) >= 0.02 - 1e-12
    assert np.isfinite(final).all()
    assert record["minimizer_overlap_floor_vetoes"] > 0


def test_admissible_assignment_avoids_forbidden_states_and_pairs() -> None:
    from nanoqc.qubo.allatom_qubo import _admissible_assignment
    sites = {0: (0, 1), 1: (2, 3), 2: (4, 5)}
    single_ok = np.array([False, True, True, True, True, True])
    pair_ok = np.ones((6, 6), dtype=bool)
    for a, b in ((1, 2), (3, 4)):
        pair_ok[a, b] = pair_ok[b, a] = False
    assert _admissible_assignment(sites, single_ok, pair_ok, order=list) == [1, 3, 5]
    pair_ok[3, 5] = pair_ok[5, 3] = False
    assert _admissible_assignment(sites, single_ok, pair_ok, order=list) is None


def test_impossible_contact_uses_the_a45_floors() -> None:
    from nanoqc.qubo.allatom_qubo import _impossible_contact
    assert _impossible_contact(0.39, heavy_pair=False)      # any atoms under 0.4 A
    assert not _impossible_contact(0.7, heavy_pair=False)    # a hydrogen contact relaxation removes
    assert _impossible_contact(0.9, heavy_pair=True)         # heavy atoms under 1.0 A
    assert not _impossible_contact(1.05, heavy_pair=True)


def test_selection_admissibility_reads_the_forbidden_lists() -> None:
    from nanoqc.qubo.allatom_qubo import selection_is_geometry_admissible
    metadata = dict(forbidden_variables=[2], forbidden_variable_pairs=[[0, 4]])
    assert selection_is_geometry_admissible(metadata, [1, 0, 0, 1, 0, 0])
    assert not selection_is_geometry_admissible(metadata, [0, 0, 1, 1, 0, 0])   # forbidden state
    assert not selection_is_geometry_admissible(metadata, [1, 0, 0, 0, 1, 0])   # forbidden pair
    assert selection_is_geometry_admissible({}, [1, 0, 1])                      # no forbidden lists


def test_assignment_search_limit_is_a_typed_signal(monkeypatch) -> None:
    from nanoqc.qubo import allatom_qubo
    monkeypatch.setattr(allatom_qubo, "_ASSIGNMENT_SEARCH_NODE_LIMIT", 3)
    sites = {0: (0, 1), 1: (2, 3), 2: (4, 5)}
    pair_ok = np.ones((6, 6), dtype=bool)
    for v in (4, 5):
        for u in range(4):
            pair_ok[u, v] = pair_ok[v, u] = False         # site 2 always conflicts
    import pytest
    with pytest.raises(allatom_qubo._SearchLimitExceeded):
        allatom_qubo._admissible_assignment(sites, np.ones(6, dtype=bool), pair_ok, order=list)
