"""Physical acceptance must not be inferred from energy decrease or file existence."""
import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
from openmm import app, unit
from nanoqc.structure.physical_quality import (StructureQualityError, topology_geometry_audit,
                                              relaxation_force_audit)
from nanoqc.experiments.structure_benchmarks import _structure_quality_assessment
from nanoqc.qubo.allatom_qubo import AllAtomInterfaceQUBOBuilder
from nanoqc.inference.analyze_structure_recovery import grouped_primary, rq5_energy_structure


def test_hydrogen_near_coincidence_is_not_hidden_by_heavy_atom_screen():
    topology=app.Topology()
    left=topology.addResidue("TYR",topology.addChain("H"),"108")
    right=topology.addResidue("ASP",topology.addChain("A"),"110")
    topology.addAtom("HE2",app.element.hydrogen,left)
    topology.addAtom("OD1",app.element.oxygen,right)
    audit=topology_geometry_audit(topology,np.array([[0.,0.,0.],[.0045566,0.,0.]]))
    assert not audit["geometry_passed"]
    assert audit["extreme_nonbonded_pair_count"]==1
    assert audit["closest_nonbonded_pair"]["distance_angstrom"]==pytest.approx(.045566)
    assert audit["closest_nonbonded_pair"]["atoms"]==["H:108:TYR:HE2","A:110:ASP:OD1"]


def test_bonds_and_one_three_pairs_are_excluded_but_one_four_is_not():
    topology=app.Topology();res=topology.addResidue("ALA",topology.addChain("H"),"1")
    atoms=[topology.addAtom(name,app.element.carbon,res) for name in ("N","CA","C","CB")]
    for a,b in zip(atoms,atoms[1:]):topology.addBond(a,b)
    audit=topology_geometry_audit(topology,np.array([[0.,0.,0.],[.01,0.,0.],[.02,0.,0.],[.03,0.,0.]]))
    assert audit["extreme_nonbonded_pair_count"]==1
    assert audit["closest_nonbonded_pair"]["atoms"]==["H:1:ALA:N","H:1:ALA:CB"]


@pytest.mark.parametrize("second_id",["2","100"])
def test_peptide_connectivity_uses_coordinates_and_bonds_not_number_gaps(second_id):
    topology=app.Topology();chain=topology.addChain("H")
    a=topology.addAtom("C",app.element.carbon,topology.addResidue("ALA",chain,"1"))
    b=topology.addAtom("N",app.element.nitrogen,topology.addResidue("SER",chain,second_id))
    topology.addBond(a,b)
    assert topology_geometry_audit(topology,[[0,0,0],[.133,0,0]])["topology_passed"]
    bad=topology_geometry_audit(topology,[[0,0,0],[.4,0,0]])
    assert not bad["topology_passed"]
    assert bad["peptide_breaks"][0]["reason"]=="overlong_peptide_bond"


def test_frozen_forces_do_not_determine_convergence_and_zero_iterations_are_explicit():
    forces=np.array([[1e12,1e12,1e12],[1.,1.,1.]])
    assert relaxation_force_audit(forces,{1},iterations=200)["relaxation_converged"]
    skipped=relaxation_force_audit(forces,{1},iterations=0)
    assert skipped["relaxation_status"]=="skipped" and not skipped["relaxation_converged"]
    forces[1]=100.
    audit=relaxation_force_audit(forces,{1},iterations=200)
    audit["physical_quality_after"]=dict(topology_passed=True,geometry_passed=True,extreme_nonbonded_pair_count=0)
    audit.update(discrete_energy_kcal=1e20,relaxed_energy_kcal=-3000.)
    result=_structure_quality_assessment(audit)
    assert result["evaluation_status"]=="failed"
    assert "relaxation_not_converged" in result["evaluation_failure_reasons"]


@pytest.mark.parametrize("analysis",[grouped_primary,rq5_energy_structure])
def test_failed_method_or_control_rows_cannot_enter_structural_inference(analysis):
    rows=[dict(target="9gcn",seed=1,method="qaoa",final_rmsd=.1,
               evaluation_status="passed",control_evaluation_status="failed")]
    with pytest.raises(ValueError,match="no rows excluded for inference"):
        if analysis is grouped_primary:
            analysis(rows,{},"final_rmsd","qaoa_vs_sa",1000,42)
        else:
            analysis(rows,{},"qaoa_vs_sa",1000,42)


def test_real_openmm_relaxation_preserves_frozen_coordinates_and_reports_forces(tmp_path,monkeypatch):
    monkeypatch.setenv("QP_OPENMM_PLATFORM","Reference")
    builder=AllAtomInterfaceQUBOBuilder.__new__(AllAtomInterfaceQUBOBuilder)
    builder.mm,builder.app,builder.unit=openmm,app,unit
    builder.topology=app.Topology()
    for label in ("H","A"):
        residue=builder.topology.addResidue("ALA",builder.topology.addChain(label),"1")
        builder.topology.addAtom("CA",app.element.carbon,residue)
    builder.base_positions=np.array([[0.,0.,0.],[.2,0.,0.]])
    builder.movable={1};builder.system=openmm.System()
    for _ in range(2):builder.system.addParticle(12.)
    force=openmm.CustomBondForce("0.5*k*(r-r0)^2")
    force.addGlobalParameter("k",1000.);force.addGlobalParameter("r0",.15)
    force.addBond(0,1,[]);builder.system.addForce(force)
    builder.energy_force_groups={"0:CustomBondForce":0}
    builder.integrator=openmm.VerletIntegrator(.001)
    builder.context=openmm.Context(builder.system,builder.integrator,openmm.Platform.getPlatformByName("Reference"))
    result=builder.relax_positions(builder.base_positions,tmp_path/"relaxed.cif",minimize_iterations=200)
    parsed=app.PDBxFile(str(tmp_path/"relaxed.cif"))
    final=np.array(parsed.positions.value_in_unit(unit.nanometer))
    assert np.array_equal(final[0],builder.base_positions[0])
    assert result["relaxed_energy_kcal"]<result["discrete_energy_kcal"]
    assert result["relaxation_converged"]
    assert sum(result["relaxed_energy_components_kcal"].values())==pytest.approx(result["relaxed_energy_kcal"])
    assert (tmp_path/"relaxed_discrete.cif").exists()
    assert _structure_quality_assessment(result)["evaluation_status"]=="passed"
    del builder.context,builder.integrator


def test_device_resource_errors_never_become_scientific_exclusions():
    """A GPU memory limit is a scheduling artifact, not a property of a complex (A35)."""
    from nanoqc.common.device_errors import (DeviceResourceError, as_resource_error,
                                             is_resource_error)
    from nanoqc.experiments import run_real_complex_pilot as pilot

    resource = [RuntimeError("CUDA error: out of memory"),
                Exception("Error launching kernel: cudaErrorMemoryAllocation"),
                MemoryError(),
                Exception("All CUDA-capable devices are busy or unavailable")]
    scientific = [ValueError("Raw/graph protein residue identities differ"),
                  ValueError("Only 4 independently Dunbrack/Amber-compatible VHH sites; need 6"),
                  Exception("missing backbone on reread: H:31")]
    assert all(is_resource_error(exc) for exc in resource)
    assert not any(is_resource_error(exc) for exc in scientific)
    # A wrapped cause is still recognized through the exception chain.
    try:
        try:
            raise RuntimeError("out of memory")
        except RuntimeError as cause:
            raise ValueError("preparation failed") from cause
    except ValueError as chained:
        assert is_resource_error(chained)

    wrapped = as_resource_error(resource[0], device="1")
    record = pilot._preparation_error(wrapped)
    assert record["category"] == DeviceResourceError.category
    with pytest.raises(DeviceResourceError):
        pilot._raise_preparation_error(record)
    # A genuine scientific exclusion still raises the structure-quality error.
    quality = pilot._preparation_error(
        StructureQualityError("bad geometry", category="input_heavy_atoms", audit={}))
    with pytest.raises(StructureQualityError):
        pilot._raise_preparation_error(quality)


def _two_residue_topology(separation_angstrom, *, same_residue=False, hydrogen=False):
    """Two backbone-free fragments whose probe atoms sit ``separation`` apart."""
    from openmm.app import Topology, element
    top = Topology(); chain = top.addChain("A")
    first = top.addResidue("ALA", chain, id="1")
    second = first if same_residue else top.addResidue("ALA", chain, id="2")
    probe = element.hydrogen if hydrogen else element.carbon
    # OpenMM requires each residue's atoms to be added contiguously.
    top.addAtom("CB", element.carbon, first)
    top.addAtom("CA", element.carbon, first)
    top.addAtom("HB1" if hydrogen else "CG", probe, second)
    xyz = np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [separation_angstrom, 0.0, 0.0]]) / 10.0
    return top, xyz


def test_generated_input_heavy_floor_matches_the_a24_definition():
    """Distinct-residue heavy pairs below the floor; same residue and hydrogens ignored (A44)."""
    from nanoqc.structure.physical_quality import interresidue_heavy_floor_audit
    top, xyz = _two_residue_topology(0.8)
    audit = interresidue_heavy_floor_audit(top, xyz, 1.0)
    assert audit["count"] == 1 and abs(audit["minimum_distance_angstrom"] - 0.8) < 1e-9
    assert audit["closest_pair"] == ["A:1:ALA:CB", "A:2:ALA:CG"]
    assert interresidue_heavy_floor_audit(*_two_residue_topology(1.2), 1.0)["count"] == 0
    assert interresidue_heavy_floor_audit(*_two_residue_topology(0.8, same_residue=True), 1.0)["count"] == 0
    assert interresidue_heavy_floor_audit(*_two_residue_topology(0.8, hydrogen=True), 1.0)["count"] == 0
    with pytest.raises(ValueError, match="floor"):
        interresidue_heavy_floor_audit(top, xyz, 1.5)


def test_rejection_sampling_keeps_valid_first_draws_and_records_every_rejection():
    """Attempt 0 is the former unconditioned draw; rejections come from the same stream (A44)."""
    from nanoqc.structure.physical_quality import sample_valid_input

    def draw(rng):
        value = float(rng.uniform())
        return np.array([value]), [dict(value=value)]

    def validity_above(threshold):
        def check(positions):
            ok = float(positions[0]) > threshold
            return dict(valid=ok, reasons=[] if ok else ["interresidue_heavy_atom_floor"],
                        closest_nonbonded_pair=None,
                        interresidue_heavy=dict(minimum_distance_angstrom=None, closest_pair=None))
        return check

    # Always valid: identical to one draw from a fresh generator with that seed.
    positions, records, audit = sample_valid_input(draw, validity_above(-1.0), 7)
    assert positions[0] == np.random.default_rng(7).uniform()
    assert audit["accepted_attempt"] == 0 and audit["rejected_attempts"] == []

    # Conditioned: the accepted value is the first draw of the same stream above 0.9.
    stream = np.random.default_rng(7)
    expected_index = next(i for i in range(10_000) if stream.uniform() > 0.9)
    positions, records, audit = sample_valid_input(draw, validity_above(0.9), 7)
    assert audit["accepted_attempt"] == expected_index
    assert len(audit["rejected_attempts"]) == expected_index
    assert positions[0] > 0.9 and records == [dict(value=float(positions[0]))]
    # Deterministic for a seed.
    again = sample_valid_input(draw, validity_above(0.9), 7)
    assert again[0][0] == positions[0] and again[2]["accepted_attempt"] == expected_index


def test_no_valid_draw_is_a_recorded_failure_not_a_silent_skip():
    from nanoqc.structure.physical_quality import sample_valid_input
    never = lambda positions: dict(valid=False, reasons=["extreme_nonbonded_overlap"],
                                   closest_nonbonded_pair=None,
                                   interresidue_heavy=dict(minimum_distance_angstrom=None,
                                                           closest_pair=None))
    with pytest.raises(StructureQualityError) as caught:
        sample_valid_input(lambda rng: (np.zeros(1), []), never, 3, max_attempts=5)
    assert caught.value.category == "generated_input"
    assert caught.value.audit["rejected_count"] == 5


def test_recovery_benchmark_uses_the_validated_perturbation():
    """The shared recovery input is drawn through perturb_valid_input with the A24 floor (A44)."""
    import inspect
    from nanoqc.experiments import structure_benchmarks
    from nanoqc.pipeline import stages_structure
    source = inspect.getsource(structure_benchmarks._recovery_benchmark_main)
    assert "generator.perturb_valid_input(" in source
    assert "generator.perturb_sidechain_chis(" not in source
    assert "input_validity=input_validity" in source
    orchestrator = inspect.getsource(stages_structure)
    assert orchestrator.count('"min_interresidue_heavy_distance_angstrom"') == 2
