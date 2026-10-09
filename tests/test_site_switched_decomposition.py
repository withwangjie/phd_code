"""A56: the site-switched decomposition cannot leak a clash into another term."""
import itertools

import numpy as np
import openmm as mm
from openmm import app, unit

from nanoqc.qubo.allatom_qubo import AllAtomInterfaceQUBOBuilder


def _builder():
    """One fixed atom and two single-atom Active sites with two states each.

    State v0 (site 0) and state w0 (site 1) sit 0.2 A apart: a forbidden pair.
    Under the A47 anchor decomposition, single(v0) would contain that clash
    whenever the site-1 anchor is w0.
    """
    topology = app.Topology()
    chain = topology.addChain("H")
    atoms = []
    for name in ("ENV", "LEU", "LEU"):
        residue = topology.addResidue(name, chain)
        atoms.append(topology.addAtom("C", app.element.carbon, residue))
    system = mm.System()
    nonbonded = mm.NonbondedForce()
    nonbonded.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
    for charge in (-0.3, 0.2, 0.1):
        system.addParticle(12.0)
        nonbonded.addParticle(charge, 0.34, 0.36)
    system.addForce(nonbonded)
    b = object.__new__(AllAtomInterfaceQUBOBuilder)
    b.mm, b.app, b.unit = mm, app, unit
    b.system, b.topology = system, topology
    b.context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    b.base_positions = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [1.2, 0.0, 0.0]])
    states = {0: [(0.6, 0.5, 0.0), (0.6, -0.5, 0.0)], 1: [(0.6, 0.52, 0.0), (1.4, 0.4, 0.0)]}
    b.candidates = [dict(site=s, residue_id=f"H:{s + 2}", residue_name="LEU", angle=0.0,
                         chi_degrees=(0.0,), prior_probability=0.5,
                         indices=np.array([s + 1]), positions=np.array([xyz]))
                    for s in (0, 1) for xyz in states[s]]
    b.site_to_variables = {0: (0, 1), 1: (2, 3)}
    b.pair_decomposition_exact = True
    b.candidate_quality_exclusions = []
    b.preparation_quality = dict(schema="test")
    b.candidate_relax_iterations = 0
    b.polar_hydrogen_shielding = {}
    b.raw_rotamer_pool_sizes = [2, 2]
    b.retained_rotamers_per_site = [2, 2]
    b.site_scores = np.zeros(2)
    b.solvent_model = "vacuum"
    b.rotamer_mode, b.rotamer_library_path, b.chi1_angles_override = "legacy", None, None
    return b


def test_forbidden_pair_is_isolated_and_admissible_assignments_are_exact() -> None:
    b = _builder()
    q = b.build()
    md = q.metadata
    assert md["forbidden_variable_pairs"] == [[0, 2]]
    F = md["forbidden_state_penalty_kcal"]
    assert q.physical_pair[0, 2] == F
    # No admissible term carries the 0.2-A clash (which is ~1e13 kcal/mol).
    admissible = [abs(q.physical_self[v]) for v in range(4)] + [abs(q.physical_pair[0, 3]),
                                                                abs(q.physical_pair[1, 2]),
                                                                abs(q.physical_pair[1, 3])]
    assert max(admissible) < 1e3 < F * 1e9
    for chosen in itertools.product((0, 1), (2, 3)):
        x = np.zeros(4); x[list(chosen)] = 1
        predicted = md["physical_constant_offset"] + q.physical_self @ x + x @ q.physical_pair @ x
        if list(chosen) == [0, 2]:
            assert predicted > max(md["physical_constant_offset"] + q.physical_self @ y + y @ q.physical_pair @ y
                                   for y in [np.eye(4)[[a, c]].sum(0) for a, c in ((0, 3), (1, 2), (1, 3))])
        else:
            actual = b.energy(b.positions_for_variables(list(chosen)))
            assert abs(actual - predicted) < 1e-6
