"""Full training-manifest structure and raw/relaxed Amber diagnostics; no fit."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from filelock import FileLock
import numpy as np
from scipy.stats import spearmanr

from nanoqc.common.repo_io import atomic_write_json_fsync, sha256_file, repo_path, MODULE_LAYOUT
from nanoqc.common.seed_streams import derive_streams, derive_child_seed
from nanoqc.structure.physical_quality import topology_geometry_audit


DIAGNOSTIC_FIELDS = (
    "diagnostic_mode", "coarse_delta", "raw_amber_kcal", "relaxed_amber_kcal",
    "relaxed_amber_delta_kcal", "raw_geometry_passed", "relaxed_geometry_passed",
    "raw_min_nonbonded_angstrom", "relaxed_min_nonbonded_angstrom",
    "relaxation_status", "relaxation_converged", "movable_force_rms_kj_mol_nm",
    "movable_force_max_kj_mol_nm", "assignment_status", "assignment_error",
    "raw_positions_sha256", "diagnostic_artifact",
)


def diagnostic_assignment(builder, angles: dict, destination: Path, iterations: int) -> dict:
    """Always retain the state assessment; a failed anchor never selects a replacement."""
    destination.parent.mkdir(parents=True,exist_ok=True)
    result={key:None for key in DIAGNOSTIC_FIELDS}
    result.update(diagnostic_mode="raw_relaxed_training_only_v1",assignment_status="failed",
                  assignment_error=None,diagnostic_artifact=str(destination.with_suffix(".json")))
    audit={}
    try:
        positions=builder.positions_for_chi_assignment(angles)
        result["raw_positions_sha256"]=hashlib.sha256(
            np.asarray(positions,dtype="<f8").tobytes()).hexdigest()
        raw_path=destination.with_name(destination.stem+"_discrete.cif")
        builder.write_structure(positions,raw_path)
        audit["raw_structure"]=str(raw_path)
        audit["raw_geometry"]=topology_geometry_audit(builder.topology,positions)
        result["raw_geometry_passed"]=audit["raw_geometry"]["geometry_passed"]
        pair=audit["raw_geometry"]["closest_nonbonded_pair"]
        result["raw_min_nonbonded_angstrom"]=None if pair is None else pair["distance_angstrom"]
        # Raw energy and its force components are preserved even if relaxation fails.
        result["raw_amber_kcal"]=builder.energy(positions)
        audit["raw_energy_components_kcal"]=builder.energy_components()
        audit["relaxation"]=builder.relax_positions(positions,destination,minimize_iterations=iterations)
        relaxation=audit["relaxation"]
        result["relaxed_amber_kcal"]=relaxation["relaxed_energy_kcal"]
        final=relaxation["physical_quality_after"]
        result["relaxed_geometry_passed"]=final["geometry_passed"]
        pair=final["closest_nonbonded_pair"]
        result["relaxed_min_nonbonded_angstrom"]=None if pair is None else pair["distance_angstrom"]
        for name in ("relaxation_status","relaxation_converged","movable_force_rms_kj_mol_nm",
                     "movable_force_max_kj_mol_nm"):
            result[name]=relaxation[name]
        result["assignment_status"]="evaluated"
    except Exception as exc:
        result["assignment_error"]=f"{type(exc).__name__}: {exc}"
    atomic_write_json_fsync(destination.with_suffix(".json"),dict(
        result=result,chi_assignment=angles,audit=audit,
        policy="no energy/clash rejection, no alternate anchor, no fitted coefficients"))
    return result


def _number(value):
    if value in (None,"","None"):return None
    number=float(value)
    return number if math.isfinite(number) else None


def rank_diagnostic(rows: list[dict], target: str, *, valid_only: bool) -> dict:
    """Within-complex descriptive rank only; constant groups remain visible."""
    pairs=[]
    for row in rows:
        if valid_only:
            flag="raw_geometry_passed" if target=="raw_amber_kcal" else "relaxed_geometry_passed"
            if str(row.get(flag)).lower()!="true":continue
            if target=="relaxed_amber_kcal" and str(row.get("relaxation_converged")).lower()!="true":continue
        x,y=_number(row.get("coarse_delta")),_number(row.get(target))
        if x is not None and y is not None:pairs.append((x,y))
    rho=None;status="insufficient_states"
    if len(pairs)>=3:
        x,y=np.asarray(pairs).T
        if np.ptp(x)==0 or np.ptp(y)==0:status="constant_input"
        else:rho=float(spearmanr(x,y).statistic);status="estimable"
    return dict(n_states=len(pairs),spearman_rho=rho,status=status,
                subset="passes_extreme_geometry_and_relaxed_force_if_applicable" if valid_only else "all_finite_states",
                claim="descriptive only; subset is not a calibrated full-space surrogate")


def summarize(csv_path: Path, provenance_path: Path, output: Path) -> dict:
    with csv_path.open(newline="",encoding="utf-8-sig") as handle:rows=list(csv.DictReader(handle))
    provenance=json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("csv_sha256")!=sha256_file(csv_path):
        raise ValueError("Diagnostic CSV differs from its generated provenance hash")
    by_complex={}
    for row in rows:by_complex.setdefault(int(row["training_row_index"]),[]).append(row)
    ranks=[]
    planned_states=min(provenance["assignments_per_complex"],3**provenance["active_sites"])
    for index,group in sorted(by_complex.items()):
        indices=[int(row["assignment_index"]) for row in group]
        if len(indices)!=len(set(indices)) or set(indices)!=set(range(planned_states)):
            raise ValueError(f"Incomplete or duplicate diagnostic state denominator for training row {index}")
        ranks.append(dict(training_row_index=index,pdb_id=group[0]["pdb_id"],family_cluster=group[0]["family_cluster"],
            sampled_states=len(group),
            raw_all=rank_diagnostic(group,"raw_amber_kcal",valid_only=False),
            raw_screened=rank_diagnostic(group,"raw_amber_kcal",valid_only=True),
            relaxed_all=rank_diagnostic(group,"relaxed_amber_kcal",valid_only=False),
            relaxed_screened=rank_diagnostic(group,"relaxed_amber_kcal",valid_only=True)))
    total=provenance["complexes_discovered"]
    excluded=provenance["input_quality_exclusions"];failed=provenance["failures"]
    ids=[int(x["training_row_index"]) for x in excluded+failed]+list(by_complex)
    if len(ids)!=len(set(ids)) or set(ids)!=set(range(total)):
        raise ValueError("Full training diagnostic does not close its manifest denominator")
    result=dict(scope="entire frozen training manifest; no target cap; no validation/test graphs",
        discovered_complexes=total,evaluated_complexes=len(by_complex),
        input_quality_exclusions=excluded,complex_execution_failures=failed,
        complex_denominator_closed=True,states_recorded=len(rows),
        requested_states_per_eligible_complex=provenance["assignments_per_complex"],
        state_denominator_closed=True,planned_states_per_evaluated_complex=planned_states,
        unavailable_complexes=len(excluded)+len(failed),
        assignment_status_counts=dict(Counter(row["assignment_status"] for row in rows)),
        relaxation_status_counts=dict(Counter(str(row.get("relaxation_status")) for row in rows)),
        raw_extreme_geometry_failures=sum(str(row["raw_geometry_passed"]).lower()=="false" for row in rows),
        relaxed_extreme_geometry_failures=sum(str(row["relaxed_geometry_passed"]).lower()=="false" for row in rows),
        within_complex_ranks=ranks,csv_sha256=sha256_file(csv_path),
        provenance_sha256=sha256_file(provenance_path),coefficients_fitted=False,
        limitations="extreme geometry screen is not full stereochemical validation; relaxed energies are not a pairwise QUBO or affinity")
    atomic_write_json_fsync(output/"dual_energy_summary.json",result)
    lines=["# Full training structure and dual-energy diagnostic","",
        f"Training complexes: {total}; evaluated: {len(by_complex)}; input exclusions: {len(excluded)}; execution failures: {len(failed)}.",
        f"Recorded states: {len(rows)}; statuses: {result['assignment_status_counts']}.",
        f"Relaxation statuses: {result['relaxation_status_counts']}.",
        "All finite-state ranks and screened-subset ranks are shown separately. Missing and constant correlations remain explicit.",
        "No regression fit, coefficient application, validation/test evaluation or inference is performed.","",
        "| Training row | PDB | States | Raw rho (all) | Relaxed rho (all) | Raw rho (screened) | Relaxed rho (screened) |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for rank in ranks:
        values=[rank[name]["spearman_rho"] for name in ("raw_all","relaxed_all","raw_screened","relaxed_screened")]
        lines.append(f"| {rank['training_row_index']} | {rank['pdb_id']} | {rank['sampled_states']} | "+
                     " | ".join("n/a" if value is None else f"{value:.4g}" for value in values)+" |")
    (output/"dual_energy_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result


def diagnostic_command(run: Path, config: dict, output: Path, *, workers: int,
                       devices: list[str], iterations: int) -> list[str]:
    """Use frozen scientific settings but enforce the complete training manifest."""
    paths=config["paths"];repo=Path(paths["repo_root"])
    def source(value):
        path=Path(value);return str(path if path.is_absolute() else repo/path)
    def artifact(value):
        path=Path(value);return str(path if path.is_absolute() else run/path)
    qc=config["qc_benchmark"];cal=qc["energy_calibration"];rot=qc["rotamer_model"]
    ff=qc.get("coarse_force_field",{});hom=config["queue_freeze"]["homology_isolation"]
    options={
        "dataset":artifact(paths.get("dataset_dir","dataset")),"data-root":source(paths["data_root"]),
        "checkpoint":artifact(str(Path(paths.get("checkpoint_dir","checkpoints"))/qc.get("checkpoint","best_egnn_pruning.pt"))),
        "cluster-map":str(run/"independence"/"pdb_family_clusters.json"),
        "selection-mode":cal.get("selection_mode","egnn"),"rotamer-mode":rot["mode"],
        "rotamer-library":source(rot.get("library_path","data/rotamer/ALL.bbdep.rotamers.lib")),
        "rotamer-probability-floor":rot.get("probability_floor",1e-4),
        "out-csv":str(output/"dual_energy_train.csv"),"out-provenance":str(output/"dual_energy_train.provenance.json"),
        "diagnostic-artifacts":str(output/"structures"),"diagnostic-iterations":iterations,
        "max-complexes":0,"assignments-per-complex":cal.get("assignments_per_complex",64),
        "active-sites":cal.get("active_sites",6),"radius":cal.get("radius_angstrom",6.),
        "seed":derive_child_seed(derive_streams(config["master_seed"])["partition"],"energy_calibration"),
        "solvent-model":config.get("structure_experiment",{}).get("solvent_model","vacuum"),"workers":workers,
        "vhh-identity-threshold":hom.get("vhh_full_chain_identity",.8),"cdr-h3-identity-threshold":hom.get("cdr_h3_identity",.5),
        "antigen-identity-threshold":hom.get("antigen_identity",.3),"antigen-min-length-coverage":hom.get("antigen_min_length_coverage",.7),
        "antigen-guidance-weight":qc.get("antigen_guidance_weight",.25),
        "antigen-proximity-scale":qc.get("antigen_proximity_scale_angstrom",6.),"contact-ca-cutoff":qc.get("contact_ca_cutoff_angstrom",8.),
    }
    for cli,key,default in (("nonbonded-cutoff","cutoff_angstrom",8.),("softcore-delta","softcore_delta_angstrom",.5),
        ("hard-core-fraction","hard_core_fraction",.72),("hard-sphere-penalty","hard_sphere_penalty",25.),
        ("lj-repulsion-cap","lj_repulsion_cap",50.),("lj-attraction-cap","lj_attraction_cap",5.),
        ("coulomb-cap","coulomb_cap",20.),("dielectric-base","dielectric_base",4.),
        ("dielectric-slope","dielectric_slope",2.),("thermal-energy-kcal","thermal_energy_kcal",.593)):
        options[cli]=ff.get(key,default)
    command=[sys.executable,"-m","nanoqc.experiments.generate_energy_calibration_dataset","--dual-energy-diagnostic"]
    for key,value in options.items():command.extend(["--"+key,str(value)])
    command.extend(["--rotamer-sigma-offsets",*[str(x) for x in rot.get("sigma_offsets",[-1.,0.,1.])],
                    "--gpu-devices",*devices])
    return command


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True,help="New diagnostic directory outside the frozen run")
    parser.add_argument("--workers",type=int,default=2)
    parser.add_argument("--gpu-devices",nargs="+",default=["0","1"])
    parser.add_argument("--iterations",type=int,default=200)
    args=parser.parse_args()
    run=args.run_dir.resolve(strict=True);output=args.out_dir.resolve()
    if output==run or output.is_relative_to(run):parser.error("Diagnostic output must be outside the frozen run")
    if args.iterations<1 or args.workers<1 or args.workers>len(args.gpu_devices):parser.error("Invalid iteration/worker settings")
    if len(set(args.gpu_devices))!=len(args.gpu_devices) or any(not d.isdecimal() for d in args.gpu_devices):parser.error("Distinct numeric GPU indices required")
    if output.exists() and any(output.iterdir()):parser.error("Use a new empty output directory; previous diagnostics are preserved")
    config=json.loads((run/"run_manifest.json").read_text(encoding="utf-8"))["config"]
    from nanoqc.qubo.rotamer_library import rotamer_source_metadata
    rot=config["qc_benchmark"]["rotamer_model"]
    if rot["mode"]=="pyrosetta_dun10":
        version=rotamer_source_metadata(rot["mode"],None)["pyrosetta_version"]
        if rot.get("version_contains") and rot["version_contains"] not in version:
            raise ValueError("PyRosetta version differs from the frozen diagnostic source protocol")
    hardware=config.get("hardware",{})
    for variable,key,default in (("QP_OPENMM_PLATFORM","openmm_platform","Reference"),
                                 ("QP_OPENMM_PRECISION","openmm_precision","double")):
        os.environ.setdefault(variable,str(hardware.get(key,default)))
    command=diagnostic_command(run,config,output,workers=args.workers,devices=args.gpu_devices,iterations=args.iterations)
    output.mkdir(parents=True,exist_ok=True)
    with FileLock(str(output/".lock"),timeout=0):
        atomic_write_json_fsync(output/"diagnostic_manifest.json",dict(
            command=command,source_run_manifest_sha256=sha256_file(run/"run_manifest.json"),
            code_sha256={name:sha256_file(repo_path(name)) for name in MODULE_LAYOUT},
            openmm_platform=os.environ.get("QP_OPENMM_PLATFORM","Reference"),
            openmm_precision=os.environ.get("QP_OPENMM_PRECISION","double"),
            scope="full training manifest; previous target cap overridden to zero; no fit"))
        with (output/"generation.log").open("w",encoding="utf-8") as log:
            status=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
        csv_path=output/"dual_energy_train.csv";prov=output/"dual_energy_train.provenance.json"
        if not csv_path.is_file() or not prov.is_file():
            print(f"Diagnostic generation failed (exit={status}); inspect {output/'generation.log'}",file=sys.stderr);return 1
        summary=summarize(csv_path,prov,output)
        print(f"Full training diagnostic: {summary['discovered_complexes']} complexes; {summary['states_recorded']} states; {output/'dual_energy_report.md'}")
        return int(bool(status) or not summary["complex_denominator_closed"])


if __name__=="__main__":raise SystemExit(main())
