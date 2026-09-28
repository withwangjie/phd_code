"""Full-manifest diagnosis retains failed states and cannot become a fit input."""
import csv
import json

import numpy as np
import pytest

from nanoqc.experiments import training_energy_diagnostic as diagnostic
from nanoqc.experiments.calibration_fit import fit_energy_calibration_csv
from nanoqc.common.repo_io import sha256_file


def test_raw_measurement_survives_failed_relaxation(tmp_path,monkeypatch):
    class Builder:
        topology=object()
        def positions_for_chi_assignment(self,angles):return np.ones((2,3))
        def write_structure(self,positions,path):path.write_text("raw state retained")
        def energy(self,positions):return 1e20
        def energy_components(self):return {"NonbondedForce":1e20}
        def relax_positions(self,*args,**kwargs):raise RuntimeError("minimizer failed")
    monkeypatch.setattr(diagnostic,"topology_geometry_audit",lambda *args:dict(
        geometry_passed=False,closest_nonbonded_pair=dict(distance_angstrom=.045566,atoms=["H:108:HE2","H:110:OD1"])))
    result=diagnostic.diagnostic_assignment(Builder(),{"H:108":[60.,90.]},tmp_path/"state.cif",200)
    assert result["assignment_status"]=="failed"
    assert result["raw_amber_kcal"]==1e20 and result["relaxed_amber_kcal"] is None
    saved=json.loads((tmp_path/"state.json").read_text())
    assert saved["audit"]["raw_geometry"]["closest_nonbonded_pair"]["distance_angstrom"]==.045566
    assert saved["audit"]["raw_energy_components_kcal"]["NonbondedForce"]==1e20
    assert (tmp_path/"state_discrete.cif").is_file()


def _tables(tmp_path):
    rows=[dict(training_row_index=0,assignment_index=i,pdb_id="9gcn",family_cluster="family_1",
        diagnostic_mode="raw_relaxed_training_only_v1",coarse_delta=i,
        raw_amber_kcal=i if i!=2 else "",relaxed_amber_kcal=8-i if i!=2 else "",
        assignment_status="failed" if i==2 else "evaluated",relaxation_status="not_converged" if i==2 else "converged",
        raw_geometry_passed=i!=1,relaxed_geometry_passed=i!=1,relaxation_converged=i not in (1,2)) for i in range(8)]
    csv_path=tmp_path/"states.csv"
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    provenance=dict(complexes_discovered=3,active_sites=6,assignments_per_complex=8,
        input_quality_exclusions=[dict(training_row_index=1,error="missing heavy atoms")],
        failures=[dict(training_row_index=2,error="candidate construction failed")],csv_sha256=sha256_file(csv_path))
    prov_path=tmp_path/"provenance.json";prov_path.write_text(json.dumps(provenance))
    return rows,csv_path,prov_path


def test_full_denominator_keeps_preparation_failures_and_state_failures(tmp_path):
    _,csv_path,prov_path=_tables(tmp_path)
    summary=diagnostic.summarize(csv_path,prov_path,tmp_path)
    assert summary["discovered_complexes"]==3 and summary["evaluated_complexes"]==1
    assert summary["unavailable_complexes"]==2 and summary["states_recorded"]==8
    assert summary["assignment_status_counts"]["failed"]==1
    assert summary["raw_extreme_geometry_failures"]==1
    assert summary["within_complex_ranks"][0]["relaxed_all"]["spearman_rho"]==-1.
    assert summary["within_complex_ranks"][0]["relaxed_screened"]["n_states"]==6
    assert not summary["coefficients_fitted"]
    assert (tmp_path/"dual_energy_report.md").is_file()


def test_missing_complex_cannot_be_reported_as_full_training(tmp_path):
    _,csv_path,prov_path=_tables(tmp_path)
    provenance=json.loads(prov_path.read_text());provenance["complexes_discovered"]=4
    prov_path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError,match="manifest denominator"):
        diagnostic.summarize(csv_path,prov_path,tmp_path)


def test_all_failed_preparations_still_produce_a_full_diagnostic_report(tmp_path):
    rows,csv_path,prov_path=_tables(tmp_path)
    with csv_path.open("w",newline="",encoding="utf-8") as handle:
        csv.DictWriter(handle,fieldnames=list(rows[0])).writeheader()
    provenance=json.loads(prov_path.read_text())
    provenance["failures"].append(dict(training_row_index=0,error="no candidates"))
    provenance["csv_sha256"]=sha256_file(csv_path)
    prov_path.write_text(json.dumps(provenance))
    summary=diagnostic.summarize(csv_path,prov_path,tmp_path)
    assert summary["complex_denominator_closed"] and summary["states_recorded"]==0
    assert summary["unavailable_complexes"]==3


def test_diagnostic_csv_cannot_be_used_by_calibration_fitter(tmp_path):
    _,csv_path,_=_tables(tmp_path)
    with pytest.raises(ValueError,match="not a calibration fitting input"):
        fit_energy_calibration_csv(csv_path,tmp_path/"fit.json",1.)


def test_full_command_preserves_frozen_settings_and_overrides_target_cap(tmp_path):
    config={
        "master_seed":4050350448,"paths":{"repo_root":str(tmp_path/"repo"),"data_root":"data"},
        "queue_freeze":{"homology_isolation":{}},"structure_experiment":{"solvent_model":"gbn2"},
        "qc_benchmark":{"checkpoint":"best.pt","rotamer_model":{"mode":"pyrosetta_dun10","sigma_offsets":[-1.,0.,1.]},
            "energy_calibration":{"max_complexes":7,"assignments_per_complex":64,"active_sites":6},
            "coarse_force_field":{"coulomb_cap":17.}},
    }
    command=diagnostic.diagnostic_command(tmp_path/"run",config,tmp_path/"diagnostic",workers=2,devices=["0","1"],iterations=200)
    value=lambda flag:command[command.index(flag)+1]
    assert value("--max-complexes")=="0"
    assert value("--solvent-model")=="gbn2" and value("--coulomb-cap")=="17.0"
    assert value("--assignments-per-complex")=="64" and value("--workers")=="2"
    assert value("--checkpoint")==str(tmp_path/"run"/"checkpoints"/"best.pt")
    assert command[-3:]==["--gpu-devices","0","1"]


def test_constant_rank_is_explicit_without_scipy_warning():
    rows=[dict(coarse_delta=0,raw_amber_kcal=i,raw_geometry_passed=True) for i in range(8)]
    rank=diagnostic.rank_diagnostic(rows,"raw_amber_kcal",valid_only=False)
    assert rank["spearman_rho"] is None and rank["status"]=="constant_input"
