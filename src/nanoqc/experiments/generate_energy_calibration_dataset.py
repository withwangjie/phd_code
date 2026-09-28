#!/usr/bin/env python3
"""Generate TRAIN-ONLY paired coarse/Amber calibration rows.

The exact same full Dunbrack residue->chi1..chiN assignment represented by a
coarse QUBO bit is reconstructed atomistically and evaluated by Amber14. The
coarse pseudo-atom geometry remains chi1-oriented, so this calibration measures
how well its component scores rank the corresponding full side-chain states.
Rows are within-complex deltas relative to a fixed deterministic anchor, which
removes arbitrary per-complex absolute energy offsets before cross-complex fit.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from nanoqc.reporting.generate_figure1_pymol_script import extract_source
from nanoqc.model.model_egnn_pruning import select_ablation_active, build_ablation_subgraph
from nanoqc.experiments.run_real_complex_pilot import complete_terminal_oxygen, strip_to_protein_conformer
from nanoqc.qubo.subgraph_to_qubo import (
    AllAtomInterfaceQUBOBuilder,
    read_atomistic_structure,
    EnergyCalibration,
    ForceFieldConfig,
    InterfaceQUBOBuilder,
)
from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.common.seed_streams import DEFAULT_MASTER_SEED
from nanoqc.data.safe_graph_load import load_graph


def _manifest_graph_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Graph manifest path must be relative")
    base = root.resolve()
    path = (base / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError(f"Graph manifest path escapes dataset or is missing: {relative}")
    return path


def assignment_components(qubo, selected: list[int]) -> tuple[float,float,float,float]:
    meta=qubo.metadata
    prior=np.asarray(meta["raw_prior_energy"],dtype=float)
    vhh=np.asarray(meta["raw_vhh_environment_energy"],dtype=float)
    antigen=np.asarray(meta["raw_antigen_energy"],dtype=float)
    pair=np.asarray(meta["raw_pair_energy_upper"],dtype=float)
    x=np.zeros(len(prior),dtype=float)
    x[selected]=1.0
    return (
        float(prior@x),
        float(vhh@x),
        float(antigen@x),
        float(x@pair@x),
    )


def chi_assignment(qubo, selected: list[int]) -> dict[str,tuple[float,...]]:
    """Recover the exact multi-chi rotamer represented by each selected QUBO bit."""
    by_variable={
        int(row["variable_index"]):row
        for row in qubo.metadata.get("rotamer_state_records",[])
    }
    result={}
    for index in selected:
        record=by_variable.get(int(index))
        if record is None:
            raise ValueError(
                f"Formal Dunbrack calibration requires complete rotamer_state_records; "
                f"missing QUBO variable {index}"
            )
        chis=tuple(float(v) for v in record.get("chi_degrees",[]))
        if not chis:
            raise ValueError(f"Missing multi-chi metadata for QUBO variable {index}")
        result[str(record["residue_id"])]=chis
    return result


def legal_assignment(groups: dict[int,tuple[int,...]], rng: np.random.Generator) -> list[int]:
    return [int(rng.choice(group)) for _,group in sorted(groups.items())]


def coarse_physical_energy(qubo, selected: list[int]) -> float:
    x=np.zeros(len(qubo.physical_self),dtype=float)
    x[selected]=1.0
    return float(qubo.physical_self@x + x@qubo.physical_pair@x)


def calibration_assignments(qubo, count: int, rng: np.random.Generator) -> list[list[int]]:
    """Mix exact low-energy feasible states with uniform legal states.

    The feasible space is at most 6^8 for the supported generic builder, but
    formal calibration uses six sites. To keep this stage bounded, exact
    enumeration is required only when <=100,000 configurations; otherwise the
    stage fails rather than silently changing its sampling protocol.
    """
    groups=[tuple(group) for _,group in sorted(qubo.site_to_variables.items())]
    total=1
    for group in groups:
        total*=len(group)
    if total>100000:
        raise ValueError(
            f"Calibration feasible space {total} exceeds frozen exact-enumeration cap 100000"
        )
    scored=[]
    for state in itertools.product(*groups):
        selected=[int(v) for v in state]
        scored.append((coarse_physical_energy(qubo,selected),tuple(selected)))
    scored.sort(key=lambda item:(item[0],item[1]))
    if not scored:
        raise ValueError("No feasible calibration assignments")

    wanted=min(int(count),len(scored))
    low_count=max(1,wanted//2)
    chosen=[list(state) for _,state in scored[:low_count]]
    seen={tuple(x) for x in chosen}
    attempts=0
    while len(chosen)<wanted and attempts<max(1000,wanted*100):
        proposal=legal_assignment(qubo.site_to_variables,rng)
        key=tuple(proposal);attempts+=1
        if key not in seen:
            seen.add(key);chosen.append(proposal)
    if len(chosen)<wanted:
        for _,state in scored[low_count:]:
            if state not in seen:
                seen.add(state);chosen.append(list(state))
                if len(chosen)>=wanted:
                    break
    return chosen


def is_input_quality_exclusion(exc: Exception) -> bool:
    """Recognize missing observed atoms before any calibration energy is fit."""
    return isinstance(exc, ValueError) and (
        str(exc).startswith("Missing heavy atoms ")
        or str(exc) == "Internal heavy-atom repair would be required"
    )


def _calibration_shard(argv: list[str], device: str, log_path: Path) -> int:
    env=os.environ.copy()
    env["QP_OPENMM_DEVICE"]=device
    with log_path.open("w",encoding="utf-8") as log:
        return subprocess.run([sys.executable,"-m",
            "nanoqc.experiments.generate_energy_calibration_dataset",*argv],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=False).returncode


def _run_parallel_shards(args: argparse.Namespace) -> int:
    """Evaluate disjoint training complexes on separate GPUs; merge in input order."""
    args.out_csv.parent.mkdir(parents=True,exist_ok=True)
    provenance_path=args.out_provenance or args.out_csv.with_suffix(".provenance.json")
    with tempfile.TemporaryDirectory(prefix="calibration_shards_",dir=args.out_csv.parent) as scratch:
        root=Path(scratch)
        jobs=[]
        for index in range(args.workers):
            csv_path=root/f"shard_{index}.csv"
            prov_path=root/f"shard_{index}.json"
            log_path=root/f"shard_{index}.log"
            child=[*sys.argv[1:],"--workers","1","--shard-count",str(args.workers),
                "--shard-index",str(index),"--out-csv",str(csv_path),
                "--out-provenance",str(prov_path)]
            jobs.append((child,args.gpu_devices[index],csv_path,prov_path,log_path))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(_calibration_shard,child,device,log)
                for child,device,_,_,log in jobs]
            codes=[future.result() for future in futures]
        if any(codes):
            detail="; ".join(f"shard {i} exit={code}: {jobs[i][4].read_text(encoding='utf-8')[-2000:]}"
                for i,code in enumerate(codes) if code)
            raise RuntimeError("Calibration shard failed: "+detail)
        chunks=[];provenances=[];fieldnames=None
        for _,_,csv_path,prov_path,_ in jobs:
            with csv_path.open(newline="",encoding="utf-8") as handle:
                reader=csv.DictReader(handle)
                if fieldnames is None: fieldnames=reader.fieldnames
                elif fieldnames!=reader.fieldnames: raise ValueError("Calibration shard schema mismatch")
                chunks.extend(reader)
            provenances.append(json.loads(prov_path.read_text(encoding="utf-8")))
        for key in ("source_manifest_sha256","cluster_map_sha256","checkpoint_sha256",
                    "rotamer_mode","seed_policy"):
            if any(part.get(key)!=provenances[0].get(key) for part in provenances[1:]):
                raise ValueError(f"Calibration shards disagree on {key}")
        chunks.sort(key=lambda row:(int(row["training_row_index"]),int(row["assignment_index"])))
        succeeded={int(row["training_row_index"]) for row in chunks}
        excluded={int(record["training_row_index"]) for part in provenances
                  for record in part["input_quality_exclusions"]}
        failed={int(record["training_row_index"]) for part in provenances
                for record in part["failures"]}
        expected=set(range(sum(int(part["complexes_discovered"]) for part in provenances)))
        if (succeeded & excluded or succeeded & failed or excluded & failed
                or succeeded | excluded | failed != expected):
            raise ValueError("Calibration shards do not close the training manifest denominator")
        temp_csv=args.out_csv.with_suffix(args.out_csv.suffix+".tmp")
        with temp_csv.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fieldnames)
            writer.writeheader();writer.writerows(chunks)
        temp_csv.replace(args.out_csv)
        provenance=provenances[0]
        for key in ("complexes_discovered","complexes_attempted","complexes_succeeded",
                    "rows_written"):
            provenance[key]=sum(int(part[key]) for part in provenances)
        for key in ("failures","input_quality_exclusions"):
            provenance[key]=sorted((record for part in provenances for record in part[key]),
                key=lambda record:int(record["training_row_index"]))
        discovered=provenance["complexes_discovered"]
        attempted=provenance["complexes_attempted"]
        provenance["input_quality_exclusion_fraction"]=(
            len(provenance["input_quality_exclusions"])/discovered if discovered else 0.0)
        provenance["generation_failure_fraction"]=(
            len(provenance["failures"])/attempted if attempted else 1.0)
        provenance.update(parallel_workers=args.workers,gpu_devices=args.gpu_devices,
            shard_count=args.workers,seed_policy="per_training_manifest_row_seedsequence",
            csv_sha256=sha256(args.out_csv))
        provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        if not chunks: raise RuntimeError("No calibration rows generated")
    print(args.out_csv)
    return 0


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True,
        help="Graph dataset directory containing graph_manifest.json and graphs/train.")
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--rotamer-mode",choices=("dunbrack2010","pyrosetta_dun10"),default="dunbrack2010")
    parser.add_argument("--rotamer-library",type=Path)
    parser.add_argument("--selection-mode",choices=("egnn","distance"),default="egnn")
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--vhh-identity-threshold",type=float,default=0.80)
    parser.add_argument("--cdr-h3-identity-threshold",type=float,default=0.50)
    parser.add_argument("--antigen-identity-threshold",type=float,default=0.30)
    parser.add_argument("--antigen-min-length-coverage",type=float,default=0.70)
    parser.add_argument("--antigen-guidance-weight",type=float,default=0.25)
    parser.add_argument("--out-csv",type=Path,required=True)
    parser.add_argument("--out-provenance",type=Path)
    parser.add_argument("--cluster-map",type=Path,
        help="Frozen PDB->family/structure cluster map used for grouped calibration CV.")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--gpu-devices",nargs="+",default=["0"])
    parser.add_argument("--shard-index",type=int,default=0,help=argparse.SUPPRESS)
    parser.add_argument("--shard-count",type=int,default=1,help=argparse.SUPPRESS)
    parser.add_argument("--max-complexes",type=int,default=0)
    parser.add_argument("--assignments-per-complex",type=int,default=64)
    parser.add_argument("--active-sites",type=int,default=6)
    parser.add_argument("--radius",type=float,default=6.0)
    parser.add_argument("--seed",type=int,default=DEFAULT_MASTER_SEED)
    parser.add_argument("--antigen-proximity-scale",type=float,default=6.0)
    parser.add_argument("--contact-ca-cutoff",type=float,default=8.0)
    parser.add_argument("--nonbonded-cutoff",type=float,default=8.0)
    parser.add_argument("--softcore-delta",type=float,default=0.5)
    parser.add_argument("--hard-core-fraction",type=float,default=0.72)
    parser.add_argument("--hard-sphere-penalty",type=float,default=25.0)
    parser.add_argument("--lj-repulsion-cap",type=float,default=50.0)
    parser.add_argument("--lj-attraction-cap",type=float,default=5.0)
    parser.add_argument("--coulomb-cap",type=float,default=20.0)
    parser.add_argument("--dielectric-base",type=float,default=4.0)
    parser.add_argument("--dielectric-slope",type=float,default=2.0)
    parser.add_argument("--thermal-energy-kcal",type=float,default=0.593)
    parser.add_argument("--rotamer-probability-floor",type=float,default=1e-4)
    parser.add_argument("--rotamer-sigma-offsets",type=float,nargs="+",default=[-1.0,0.0,1.0])
    parser.add_argument("--solvent-model",choices=("vacuum","gbn2"),default="vacuum",
        help="Must match the frozen primary structural energy model.")
    args=parser.parse_args()

    if not 5 <= args.active_sites <= 8:
        parser.error("--active-sites must be in 5..8")
    if args.assignments_per_complex < 8:
        parser.error("--assignments-per-complex must be >=8")
    if args.max_complexes < 0:
        parser.error("--max-complexes must be >=0")
    if args.workers < 1 or args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Invalid calibration worker or shard count")
    if args.workers > len(args.gpu_devices) or len(set(args.gpu_devices))!=len(args.gpu_devices):
        parser.error("Each calibration worker requires a distinct gpu-device")
    if any(not device.isdecimal() for device in args.gpu_devices):
        parser.error("gpu-devices must be nonnegative CUDA device indices")
    if args.rotamer_mode=="dunbrack2010" and (args.rotamer_library is None or not args.rotamer_library.is_file()):
        parser.error("Dunbrack rotamer library not found")
    if args.workers>1:
        if args.shard_count!=1:
            parser.error("Only the parent calibration process can create shards")
        return _run_parallel_shards(args)
    homology_isolation=dict(
        vhh_full_chain_identity=float(args.vhh_identity_threshold),
        cdr_h3_identity=float(args.cdr_h3_identity_threshold),
        antigen_identity=float(args.antigen_identity_threshold),
        antigen_min_length_coverage=float(args.antigen_min_length_coverage),
    )
    if any(not 0.0 < value <= 1.0 for value in homology_isolation.values()):
        parser.error("homology thresholds/coverage must lie in (0,1]")
    if not 0.0 <= args.antigen_guidance_weight <= 1.0:
        parser.error("--antigen-guidance-weight must be in [0,1]")
    scorer=None; model_info=None
    if args.selection_mode=="egnn":
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("--selection-mode egnn requires --checkpoint")
        from nanoqc.model.model_egnn_pruning import load_interface_scorer
        scorer,model_info=load_interface_scorer(
            args.checkpoint,torch_device=torch.device("cpu"),seed=args.seed
        )
        if model_info.status!="checkpoint_loaded":
            raise ValueError("Calibration requires a valid trained EGNN checkpoint")
        scorer.eval()

    cluster_map=None
    cluster_map_sha256=None
    if args.cluster_map is not None:
        if not args.cluster_map.is_file():
            parser.error("--cluster-map not found")
        raw=json.loads(args.cluster_map.read_text(encoding="utf-8"))
        cluster_map={str(k).lower():str(v) for k,v in raw.items()}
        if not cluster_map or any(not k or not v for k,v in cluster_map.items()):
            raise ValueError("Invalid/empty calibration cluster map")
        cluster_map_sha256=sha256(args.cluster_map)

    manifest_path=args.dataset/"graph_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    train=[row for row in manifest if row.get("split")=="train"]
    train=sorted(train,key=lambda row:(str(row.get("pdb_id","")).lower(),row["path"]))
    if args.max_complexes:
        train=train[:args.max_complexes]
    train=[(index,row) for index,row in enumerate(train)
           if index % args.shard_count == args.shard_index]
    if not train:
        raise ValueError("No training graphs found")

    force_field=ForceFieldConfig(
        cutoff_angstrom=args.nonbonded_cutoff,
        softcore_delta_angstrom=args.softcore_delta,
        hard_core_fraction=args.hard_core_fraction,
        hard_sphere_penalty=args.hard_sphere_penalty,
        lj_repulsion_cap=args.lj_repulsion_cap,
        lj_attraction_cap=args.lj_attraction_cap,
        coulomb_cap=args.coulomb_cap,
        dielectric_base=args.dielectric_base,
        dielectric_slope=args.dielectric_slope,
        thermal_energy_kcal=args.thermal_energy_kcal,
    )

    args.out_csv.parent.mkdir(parents=True,exist_ok=True)
    fieldnames=[
        "pdb_id","family_cluster","split","training_row_index","assignment_index",
        "prior_energy","vhh_environment_energy","antigen_energy","pair_energy",
        "amber_delta_kcal","anchor_amber_kcal","active_residues","chi_assignment",
        "graph_sha256","source_id",
    ]
    written=0
    failures=[]
    input_quality_exclusions=[]
    with args.out_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames)
        writer.writeheader()
        for training_row_index,row in train:
            # Buffer one complex at a time. A failed complex must contribute
            # zero rows; otherwise low-energy assignments written before the
            # exception would bias the calibration fit.
            complex_rows=[]
            pdb=str(row.get("pdb_id","")).lower()
            rng=np.random.default_rng(np.random.SeedSequence([args.seed,training_row_index]))
            graph_path=_manifest_graph_path(args.dataset, row.get("path"))
            try:
                if cluster_map is not None and pdb not in cluster_map:
                    raise ValueError(f"Calibration cluster map missing training PDB {pdb}")
                family_cluster=(cluster_map[pdb] if cluster_map is not None else pdb)
                if sha256(graph_path)!=row["sha256"]:
                    raise ValueError("Training graph SHA256 mismatch")
                data=load_graph(graph_path)
                if args.selection_mode=="egnn":
                    from nanoqc.model.model_egnn_pruning import assert_checkpoint_graph_compatible
                    assert_checkpoint_graph_compatible(
                        model_info,data,homology_isolation=homology_isolation
                    )
                active=select_ablation_active(
                    data,args.selection_mode,args.active_sites,args.seed,scorer,
                    antigen_guidance_weight=args.antigen_guidance_weight,
                    antigen_proximity_scale=args.antigen_proximity_scale,
                    contact_ca_cutoff=args.contact_ca_cutoff,
                )
                sub=build_ablation_subgraph(data,active,args.radius)
                coarse=InterfaceQUBOBuilder(
                    min_variables=3*args.active_sites,max_variables=3*args.active_sites,max_sites=args.active_sites,
                    force_field=force_field,rotamer_mode=args.rotamer_mode,
                    rotamer_library_path=args.rotamer_library,
                    rotamer_probability_floor=args.rotamer_probability_floor,
                    rotamer_sigma_offsets=args.rotamer_sigma_offsets,
                    fixed_chi1_wells=True,fixed_states_per_site=3,
                    energy_calibration=EnergyCalibration(),
                ).build(sub)
                active_residues=[]
                for site in sorted(coarse.site_to_variables):
                    first=coarse.variable_map[coarse.site_to_variables[site][0]]
                    active_residues.append(first.residue_id)

                with tempfile.TemporaryDirectory(prefix=f"cal_{pdb}_") as temp:
                    temp=Path(temp)
                    extracted=extract_source(data.source_id,args.data_root,temp/"raw_structure")
                    from nanoqc.data.audit_all_datasets import materialize_graph_complex
                    extracted=materialize_graph_complex(extracted,data,temp/"graph_complex.cif")
                    import gemmi
                    structure=gemmi.read_structure(str(extracted))
                    if not len(structure):
                        raise ValueError("Source structure has no coordinate model")
                    while len(structure)>1:
                        del structure[1]
                    model_one=temp/"model1.cif"
                    structure.make_mmcif_document().write_file(str(model_one))
                    # Same force-field preparation as validation targets
                    # (run_real_complex_pilot.prepare): drop waters/H/zero-occupancy
                    # atoms, keep one alternate conformer, and bind the raw
                    # protein residues and CA coordinates to the training graph.
                    residues=read_atomistic_structure(model_one)
                    if set(residues)!=set(data.residue_ids):
                        raise ValueError("Raw/graph protein residue identities differ")
                    for node,rid in enumerate(data.residue_ids):
                        if not np.allclose(residues[rid]["atoms"]["CA"],data.pos[node].numpy(),atol=.002):
                            raise ValueError(f"Raw/graph coordinates differ at {rid}")
                    strip_to_protein_conformer(structure,residues)
                    local=temp/"native.cif"
                    structure.make_mmcif_document().write_file(str(local))
                    complete_terminal_oxygen(local)
                    atomistic=AllAtomInterfaceQUBOBuilder(
                        local,active_residues,site_scores=[1.0]*len(active_residues),
                        rotamer_mode=args.rotamer_mode,rotamer_library_path=args.rotamer_library,
                        rotamer_probability_floor=args.rotamer_probability_floor,
                        rotamer_sigma_offsets=args.rotamer_sigma_offsets,
                        solvent_model=args.solvent_model,
                    )

                    assignments=calibration_assignments(
                        coarse,args.assignments_per_complex,rng
                    )
                    anchor=assignments[0]
                    anchor_components=np.asarray(assignment_components(coarse,anchor),dtype=float)
                    anchor_angles=chi_assignment(coarse,anchor)
                    anchor_amber=atomistic.energy_for_chi_assignment(anchor_angles)

                    for index,selected in enumerate(assignments):
                        components=np.asarray(assignment_components(coarse,selected),dtype=float)-anchor_components
                        angles=chi_assignment(coarse,selected)
                        amber=atomistic.energy_for_chi_assignment(angles)-anchor_amber
                        complex_rows.append(dict(
                            pdb_id=pdb,family_cluster=family_cluster,split="train",
                            training_row_index=training_row_index,assignment_index=index,
                            prior_energy=components[0],
                            vhh_environment_energy=components[1],
                            antigen_energy=components[2],
                            pair_energy=components[3],
                            amber_delta_kcal=amber,
                            anchor_amber_kcal=anchor_amber,
                            active_residues=json.dumps(active_residues,separators=(",",":")),
                            chi_assignment=json.dumps(angles,separators=(",",":"),sort_keys=True),
                            graph_sha256=row["sha256"],source_id=data.source_id,
                        ))
                    for output_row in complex_rows:
                        writer.writerow(output_row)
                        written+=1
                    del atomistic
            except Exception as exc:
                record=dict(pdb_id=pdb,source_id=row.get("source_id"),
                            training_row_index=training_row_index,
                            error=f"{type(exc).__name__}: {exc}")
                if is_input_quality_exclusion(exc):
                    input_quality_exclusions.append(record)
                else:
                    failures.append(record)
            handle.flush()

    attempted_complexes=len(train)-len(input_quality_exclusions)
    succeeded_complexes=attempted_complexes-len(failures)
    from nanoqc.qubo.subgraph_to_qubo import rotamer_source_metadata
    provenance=dict(
        scope="training complexes only",
        source_manifest=str(manifest_path),
        source_manifest_sha256=sha256(manifest_path),
        rotamer_mode=args.rotamer_mode,
        rotamer_library=(str(args.rotamer_library) if args.rotamer_mode=="dunbrack2010" else None),
        rotamer_library_sha256=(sha256(args.rotamer_library) if args.rotamer_mode=="dunbrack2010" else None),
        rotamer_source_metadata=rotamer_source_metadata(args.rotamer_mode,args.rotamer_library),
        active_site_selection=(
            "formal EGNN + antigen-proximity selector on training complexes only"
            if args.selection_mode=="egnn"
            else "distance baseline on training complexes only"
        ),
        selection_mode=args.selection_mode,
        checkpoint=(None if args.checkpoint is None else str(args.checkpoint)),
        checkpoint_sha256=(None if args.checkpoint is None else sha256(args.checkpoint)),
        homology_isolation=homology_isolation,
        antigen_guidance_weight=float(args.antigen_guidance_weight),
        assignment_sampling=(
            "exact feasible-space ranking; lowest-energy half plus unique uniform legal samples; "
            "anchor is exact coarse physical ground state"
        ),
        rotamer_state_policy="fixed_three_chi1_wells; exactly three retained states per site, one real Dunbrack sample per chi1 well including top positive-probability sample below the global floor when required",
        active_sites=args.active_sites,radius=args.radius,
        solvent_model=args.solvent_model,
        assignments_per_complex=args.assignments_per_complex,
        rows_written=written,complexes_discovered=len(train),
        complexes_attempted=attempted_complexes,
        complexes_succeeded=succeeded_complexes,
        input_quality_exclusions=input_quality_exclusions,
        input_quality_exclusion_fraction=(0.0 if not train else len(input_quality_exclusions)/len(train)),
        generation_failure_fraction=(1.0 if not attempted_complexes else len(failures)/attempted_complexes),
        cluster_map=(None if args.cluster_map is None else str(args.cluster_map)),
        cluster_map_sha256=cluster_map_sha256,
        failures=failures,seed=args.seed,
        parallel_workers=args.workers,gpu_devices=args.gpu_devices,
        shard_count=args.shard_count,seed_policy="per_training_manifest_row_seedsequence",
        force_field=force_field.__dict__,
        csv_sha256=sha256(args.out_csv),
    )
    provenance_path=args.out_provenance or args.out_csv.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if written==0 and args.shard_count==1:
        raise RuntimeError("No calibration rows generated")
    print(args.out_csv)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
