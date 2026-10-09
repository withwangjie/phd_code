"""A50: zero-LJ hydroxyl hydrogens get a small repulsive core and 1-4 terms."""
import math

import openmm as mm
from openmm import app, unit

from nanoqc.qubo.allatom_qubo import (
    POLAR_HYDROGEN_EPSILON_KJ_MOL, POLAR_HYDROGEN_SIGMA_NM, _shield_zero_lj_polar_hydrogens,
)


def _chain():
    """C3-C2-C1-O-H: (C3, O) is an ordinary 1-4 pair, (C2, H) the hydroxyl one."""
    topology = app.Topology()
    residue = topology.addResidue("TYR", topology.addChain("H"))
    elements = [app.element.carbon] * 3 + [app.element.oxygen, app.element.hydrogen]
    atoms = [topology.addAtom(n, e, residue) for n, e in zip(("C3", "C2", "C1", "O", "H"), elements)]
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4)]
    for a, b in bonds:
        topology.addBond(atoms[a], atoms[b])
    system = mm.System()
    force = mm.NonbondedForce()
    for q, s, e in ((-0.1, 0.34, 0.36), (-0.2, 0.34, 0.36), (0.2, 0.34, 0.36),
                    (-0.5, 0.30, 0.88), (0.4, 1.0, 0.0)):
        system.addParticle(12.0)
        force.addParticle(q, s, e)
    force.createExceptionsFromBonds(bonds, 1 / 1.2, 0.5)
    system.addForce(force)
    return system, topology, force


def test_hydroxyl_hydrogen_and_its_one_four_pairs_are_shielded() -> None:
    system, topology, force = _chain()
    record = _shield_zero_lj_polar_hydrogens(system, topology)
    assert record["shielded_hydrogens"] == 1 and record["one_four_lj_scale"] == 0.5
    _, sigma, epsilon = force.getParticleParameters(4)
    assert math.isclose(sigma.value_in_unit(unit.nanometer), POLAR_HYDROGEN_SIGMA_NM)
    assert math.isclose(epsilon.value_in_unit(unit.kilojoule_per_mole), POLAR_HYDROGEN_EPSILON_KJ_MOL)
    for k in range(force.getNumExceptions()):
        i, j, q, s, e = force.getExceptionParameters(k)
        if {i, j} == {1, 4}:
            assert math.isclose(e.value_in_unit(unit.kilojoule_per_mole),
                                0.5 * math.sqrt(0.36 * POLAR_HYDROGEN_EPSILON_KJ_MOL))
            assert math.isclose(s.value_in_unit(unit.nanometer), 0.5 * (0.34 + POLAR_HYDROGEN_SIGMA_NM))
        if {i, j} == {0, 3}:          # untouched ordinary 1-4 pair
            assert math.isclose(e.value_in_unit(unit.kilojoule_per_mole), 0.5 * math.sqrt(0.36 * 0.88))
        if {i, j} in ({2, 4}, {3, 4}):   # 1-3 and 1-2 stay excluded
            assert e.value_in_unit(unit.kilojoule_per_mole) == 0.0


def test_the_one_four_hydroxyl_pair_is_repulsive_at_short_range() -> None:
    def pair_energy(system, distance_nm):
        force = [f for f in system.getForces() if isinstance(f, mm.NonbondedForce)][0]
        for k in range(force.getNumExceptions()):
            i, j, q, s, e = force.getExceptionParameters(k)
            if {i, j} == {1, 4}:
                q, s, e = (x.value_in_unit(u) for x, u in ((q, unit.elementary_charge ** 2),
                                                              (s, unit.nanometer), (e, unit.kilojoule_per_mole)))
                return 138.935456 * q / distance_nm + 4 * e * ((s / distance_nm) ** 12 - (s / distance_nm) ** 6)
    plain, _, _ = _chain()
    shielded, topology, _ = _chain()
    _shield_zero_lj_polar_hydrogens(shielded, topology)
    assert pair_energy(plain, 0.01) < pair_energy(plain, 0.1) < 0          # singular attraction
    assert pair_energy(shielded, 0.01) > pair_energy(shielded, 0.1) > 0     # repulsive wall


def test_nonbonded_tables_match_unit_converted_parameters() -> None:
    from nanoqc.qubo.allatom_qubo import _NonbondedTables
    system, topology, force = _chain()
    _shield_zero_lj_polar_hydrogens(system, topology)
    tables = _NonbondedTables.of(system)
    for i in range(force.getNumParticles()):
        q, s, e = force.getParticleParameters(i)
        assert tables.charge[i] == q.value_in_unit(unit.elementary_charge)
        assert tables.sigma[i] == s.value_in_unit(unit.nanometer)
        assert tables.epsilon[i] == e.value_in_unit(unit.kilojoule_per_mole)
    for k in range(force.getNumExceptions()):
        i, j, qq, s, e = force.getExceptionParameters(k)
        pair = (min(i, j), max(i, j))
        if qq.value_in_unit(unit.elementary_charge ** 2) == 0 and e.value_in_unit(unit.kilojoule_per_mole) == 0:
            assert pair in tables.excluded
        else:
            assert tables.exceptions[pair] == (qq.value_in_unit(unit.elementary_charge ** 2),
                                               s.value_in_unit(unit.nanometer),
                                               e.value_in_unit(unit.kilojoule_per_mole))
