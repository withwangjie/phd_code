#!/usr/bin/env python3
"""Generate TRAIN-ONLY paired coarse/Amber calibration rows.

The exact same residue->chi1 assignment is evaluated by the coarse QUBO
component model and by Amber14. Rows are within-complex deltas relative to a
fixed deterministic anchor assignment, which removes arbitrary per-complex
absolute energy offsets before the cross-complex ridge fit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from generate_figure1_pymol_script import extract_source
from model_egnn_pruning import select_ablation_active, build_ablation_subgraph
from run_real_complex_pilot import complete_terminal_oxygen
from subgraph_to_qubo import (
    AllAtomInterfaceQUBOBuilder,
    EnergyCalibration,
    ForceFieldConfig,
    InterfaceQUBOBuilder,
)


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


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


def chi1_assignment(qubo, selected: list[int]) -> dict[str,float]:
    records={row.variable_index:row for row in qubo.variable_map}
    return {records[index].residue_id:float(records[index].chi1_degrees) for index in selected}


def legal_assignment(groups: dict[int,tuple[int,...]], rng: np.random.Generator) -> list[int]:
    return [int(rng.choice(group)) for _,group in sorted(groups.items())]


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,required=True,
        help="Graph dataset directory containing graph_manifest.json and graphs/train.")
    parser.add_argument("--data-root",type=Path,required=True)
    parser.add_argument("--rotamer-library",type=Path,required=True)
    parser.add_argument("--out-csv",type=Path,required=True)
    parser.add_argument("--out-provenance",type=Path)
    parser.add_argument("--max-complexes",type=int,default=0)
    parser.add_argument("--assignments-per-complex",type=int,default=64)
    parser.add_argument("--active-sites",type=int,default=6)
    parser.add_argument("--radius",type=float,default=6.0)
    parser.add_argument("--seed",type=int,default=20260917)
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
    args=parser.parse_args()

    if not 5 <= args.active_sites <= 8:
        parser.error("--active-sites must be in 5..8")
    if args.assignments_per_complex < 8:
        parser.error("--assignments-per-complex must be >=8")
    if args.max_complexes < 0:
        parser.error("--max-complexes must be >=0")
    if not args.rotamer_library.is_file():
        parser.error("Dunbrack rotamer library not found")

    manifest_path=args.dataset/"graph_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    train=[row for row in manifest if row.get("split")=="train"]
    train=sorted(train,key=lambda row:(str(row.get("pdb_id","")).lower(),row["path"]))
    if args.max_complexes:
        train=train[:args.max_complexes]
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
        "pdb_id","split","assignment_index",
        "prior_energy","vhh_environment_energy","antigen_energy","pair_energy",
        "amber_delta_kcal","anchor_amber_kcal","active_residues","chi1_assignment",
        "graph_sha256","source_id",
    ]
    written=0
    failures=[]
    rng=np.random.default_rng(args.seed)

    with args.out_csv.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fieldnames)
        writer.writeheader()
        for row in train:
            pdb=str(row.get("pdb_id","")).lower()
            graph_path=args.dataset/Path(row["path"].replace("\\","/"))
            try:
                if sha256(graph_path)!=row["sha256"]:
                    raise ValueError("Training graph SHA256 mismatch")
                data=torch.load(graph_path,map_location="cpu",weights_only=False)
                active=select_ablation_active(
                    data,"distance",args.active_sites,args.seed,None,
                    antigen_proximity_scale=args.antigen_proximity_scale,
                    contact_ca_cutoff=args.contact_ca_cutoff,
                )
                sub=build_ablation_subgraph(data,active,args.radius)
                coarse=InterfaceQUBOBuilder(
                    min_variables=3*args.active_sites,max_variables=30,max_sites=args.active_sites,
                    force_field=force_field,rotamer_mode="dunbrack2010",
                    rotamer_library_path=args.rotamer_library,
                    rotamer_probability_floor=args.rotamer_probability_floor,
                    rotamer_sigma_offsets=args.rotamer_sigma_offsets,
                    energy_calibration=EnergyCalibration(),
                ).build(sub)
                active_residues=[]
                for site in sorted(coarse.site_to_variables):
                    first=coarse.variable_map[coarse.site_to_variables[site][0]]
                    active_residues.append(first.residue_id)

                with tempfile.TemporaryDirectory(prefix=f"cal_{pdb}_") as temp:
                    temp=Path(temp)
                    extracted=extract_source(data.source_id,args.data_root,temp/"raw_structure")
                    local=temp/("native"+extracted.suffix)
                    if extracted.resolve()!=local.resolve():
                        shutil.copy2(extracted,local)
                    complete_terminal_oxygen(local)
                    atomistic=AllAtomInterfaceQUBOBuilder(
                        local,active_residues,site_scores=[1.0]*len(active_residues),
                        rotamer_mode="dunbrack2010",rotamer_library_path=args.rotamer_library,
                        rotamer_probability_floor=args.rotamer_probability_floor,
                        rotamer_sigma_offsets=args.rotamer_sigma_offsets,
                    )

                    anchor=[
                        min(group,key=lambda index:coarse.physical_self[index])
                        for _,group in sorted(coarse.site_to_variables.items())
                    ]
                    anchor_components=np.asarray(assignment_components(coarse,anchor),dtype=float)
                    anchor_angles=chi1_assignment(coarse,anchor)
                    anchor_amber=atomistic.energy_for_chi1_assignment(anchor_angles)

                    seen={tuple(anchor)}
                    assignments=[anchor]
                    max_attempts=max(100,args.assignments_per_complex*20)
                    for _ in range(max_attempts):
                        if len(assignments)>=args.assignments_per_complex:
                            break
                        proposal=legal_assignment(coarse.site_to_variables,rng)
                        key=tuple(proposal)
                        if key not in seen:
                            seen.add(key);assignments.append(proposal)

                    for index,selected in enumerate(assignments):
                        components=np.asarray(assignment_components(coarse,selected),dtype=float)-anchor_components
                        angles=chi1_assignment(coarse,selected)
                        amber=atomistic.energy_for_chi1_assignment(angles)-anchor_amber
                        writer.writerow(dict(
                            pdb_id=pdb,split="train",assignment_index=index,
                            prior_energy=components[0],
                            vhh_environment_energy=components[1],
                            antigen_energy=components[2],
                            pair_energy=components[3],
                            amber_delta_kcal=amber,
                            anchor_amber_kcal=anchor_amber,
                            active_residues=json.dumps(active_residues,separators=(",",":")),
                            chi1_assignment=json.dumps(angles,separators=(",",":"),sort_keys=True),
                            graph_sha256=row["sha256"],source_id=data.source_id,
                        ))
                        written+=1
                    del atomistic
            except Exception as exc:
                failures.append(dict(pdb_id=pdb,error=f"{type(exc).__name__}: {exc}"))
            handle.flush()

    provenance=dict(
        scope="training complexes only",
        source_manifest=str(manifest_path),
        source_manifest_sha256=sha256(manifest_path),
        rotamer_library=str(args.rotamer_library),
        rotamer_library_sha256=sha256(args.rotamer_library),
        active_site_selection="distance baseline; no EGNN/test outcome dependence",
        active_sites=args.active_sites,radius=args.radius,
        assignments_per_complex=args.assignments_per_complex,
        rows_written=written,complexes_attempted=len(train),
        failures=failures,seed=args.seed,
        force_field=force_field.__dict__,
        csv_sha256=sha256(args.out_csv),
    )
    provenance_path=args.out_provenance or args.out_csv.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    if written==0:
        raise RuntimeError("No calibration rows generated")
    print(args.out_csv)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
