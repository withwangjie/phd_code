#!/usr/bin/env python3
"""External biological baselines for the frozen structural validation queue.

FASPR is treated as a mature side-chain packing baseline, not as a
matched-compute solver baseline. Optional Phenix clashscore supplies a standard
steric-quality metric in addition to the project's internal geometric clash
diagnostic. Missing configured executables fail closed.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import tempfile
from pathlib import Path

import gemmi

from nanoqc.qubo.subgraph_to_qubo import evaluate_atomistic_prediction
from nanoqc.common.repo_io import sha256_file as sha256


def cif_to_pdb(source: Path, destination: Path) -> None:
    structure=gemmi.read_structure(str(source))
    if len(structure)!=1:
        raise ValueError(f"Expected one model: {source}")
    structure.write_pdb(str(destination))


def restrict_sidechain_packing(
    input_pdb: Path, packed_pdb: Path, destination: Path, active_residues: list[str]
) -> None:
    """Keep FASPR changes only on the declared Active side chains.

    FASPR repacks every eligible residue by default. The formal comparison in
    this project optimizes only Active side chains, so backbone atoms and all
    non-Active side-chain atoms are restored from the identical perturbed
    input before scoring.
    """
    source = gemmi.read_structure(str(input_pdb))
    packed = gemmi.read_structure(str(packed_pdb))
    if len(source) != 1 or len(packed) != 1:
        raise ValueError("Active-only FASPR restriction requires one-model structures")
    active = {str(rid) for rid in active_residues}
    backbone = {"N", "CA", "C", "O"}
    source_atoms = {}
    for chain in source[0]:
        for residue in chain:
            rid = f"{chain.name}:{residue.seqid}"
            source_atoms[rid] = {str(atom.name).strip(): atom for atom in residue}
    for chain in packed[0]:
        for residue in chain:
            rid = f"{chain.name}:{residue.seqid}"
            reference = source_atoms.get(rid)
            if reference is None:
                raise ValueError(f"FASPR output residue is absent from input: {rid}")
            keep_sidechain = rid in active
            for atom in residue:
                name = str(atom.name).strip()
                if name in backbone or not keep_sidechain:
                    original = reference.get(name)
                    if original is None:
                        raise ValueError(f"Input is missing atom {rid}:{name}")
                    atom.pos = original.pos
                    atom.occ = original.occ
                    atom.b_iso = original.b_iso
                    atom.altloc = original.altloc
    packed.make_mmcif_document().write_file(str(destination))


def run_checked(argv: list[str], *, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    result=subprocess.run(argv,capture_output=True,text=True,timeout=timeout)
    if result.returncode!=0:
        raise RuntimeError(
            f"External command failed ({result.returncode}): {' '.join(argv)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def phenix_clashscore(executable: Path, structure: Path) -> float:
    result=run_checked([str(executable),str(structure)])
    text=result.stdout+"\n"+result.stderr
    patterns=[
        r"clashscore\s*=\s*([0-9]+(?:\.[0-9]+)?)",
        r"clashscore\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pattern in patterns:
        match=re.search(pattern,text,re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError("Could not parse Phenix clashscore output")


def evaluated_row(
    *, target: str, seed: str, method: str, reference: Path, prediction: Path,
    active: list[str], alignment: list[str], partners: list[str],
    phenix_executable: Path | None,
) -> dict:
    metrics=evaluate_atomistic_prediction(
        reference,prediction,
        active_residues=active,
        alignment_residues=alignment,
        partner_residues=partners,
    )
    row=dict(
        target=target,seed=seed,method=method,
        final_rmsd=metrics["sidechain_rmsd_angstrom"],
        chi1_recovery=metrics["chi1_recovery_rate"],
        all_chi_recovery=metrics.get("all_chi_recovery_rate"),
        chi_recovery_rates=json.dumps(metrics.get("chi_recovery_rates",{}),sort_keys=True),
        contact_f1=metrics.get("contact_f1"),
        fnat=metrics["fnat"],irmsd=metrics.get("irmsd"),lrmsd=metrics.get("lrmsd"),
        num_severe_clashes=metrics["num_severe_clashes"],
        prediction_sha256=sha256(prediction),
    )
    if phenix_executable is not None:
        row["molprobity_clashscore"]=phenix_clashscore(phenix_executable,prediction)
        row["phenix_executable_sha256"]=sha256(phenix_executable)
    return row


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir",type=Path,required=True)
    parser.add_argument("--faspr",type=Path,required=True)
    parser.add_argument("--phenix-clashscore",type=Path)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--timeout-seconds",type=int,default=1800)
    parser.add_argument("--expected-seeds",type=int,nargs="+",required=True,
        help="Frozen structural-recovery seeds; every target must have every seed.")
    args=parser.parse_args()
    if not args.faspr.is_file():
        parser.error(f"FASPR executable not found: {args.faspr}")
    if args.phenix_clashscore is not None and not args.phenix_clashscore.is_file():
        parser.error(f"Phenix clashscore executable not found: {args.phenix_clashscore}")
    if args.timeout_seconds<=0:
        parser.error("--timeout-seconds must be positive")
    if len(args.expected_seeds)!=len(set(args.expected_seeds)) or any(seed<0 for seed in args.expected_seeds):
        parser.error("--expected-seeds must be unique nonnegative integers")

    prepared=args.validation_dir/"prepared"
    results_root=args.validation_dir/"results"
    manifests=sorted(prepared.glob("*/recovery_manifest.json"))
    if not manifests:
        raise ValueError("No frozen recovery manifests found")
    args.out_dir.mkdir(parents=True,exist_ok=True)
    rows=[];failures=[]
    for manifest_path in manifests:
        case=json.loads(manifest_path.read_text(encoding="utf-8"))
        target=str(case.get("target") or manifest_path.parent.name).lower()
        native=Path(case["native_structure"])
        if not native.is_absolute():
            native=(manifest_path.parent/native).resolve()
        active=case["active_residues"]
        alignment=case["alignment_residues"]
        partners=case["partner_residues"]
        expected_seed_names=[str(seed) for seed in args.expected_seeds]
        available={
            path.name.removeprefix("seed_"):path
            for path in (results_root/target).glob("seed_*") if path.is_dir()
        }
        missing_seed_dirs=[seed for seed in expected_seed_names if seed not in available]
        if missing_seed_dirs:
            failures.append(dict(
                target=target,seed=",".join(missing_seed_dirs),
                error=f"Missing structural recovery seed directories: {missing_seed_dirs}",
            ))
        for seed in expected_seed_names:
            if seed not in available:
                continue
            seed_dir=available[seed]
            perturbed=seed_dir/"perturbed_input.cif"
            if not perturbed.is_file():
                failures.append(dict(
                    target=target,seed=seed,error=f"Missing perturbed input: {perturbed}"
                ))
                continue
            out_case=args.out_dir/target/f"seed_{seed}"
            out_case.mkdir(parents=True,exist_ok=True)
            try:
                input_pdb=out_case/"perturbed_input.pdb"
                prediction_pdb=out_case/"faspr_prediction.pdb"
                cif_to_pdb(perturbed,input_pdb)
                run_checked(
                    [str(args.faspr),"-i",str(input_pdb),"-o",str(prediction_pdb)],
                    timeout=args.timeout_seconds,
                )
                if not prediction_pdb.is_file() or prediction_pdb.stat().st_size==0:
                    raise RuntimeError("FASPR did not produce a nonempty prediction")
                active_only_prediction=out_case/"faspr_active_only.cif"
                restrict_sidechain_packing(
                    input_pdb,prediction_pdb,active_only_prediction,active
                )
                row=evaluated_row(
                    target=target,seed=seed,method="faspr_active_only",
                    reference=native,prediction=active_only_prediction,
                    active=active,alignment=alignment,partners=partners,
                    phenix_executable=args.phenix_clashscore,
                )
                row["faspr_executable_sha256"]=sha256(args.faspr)
                row["packing_scope"]="active_sidechains_only"
                rows.append(row)

                # Apply the same evaluator and standard clashscore to the
                # internal methods' final relaxed structures. These rows are
                # not a matched-compute baseline; they provide a common
                # structural-quality scale across all reconstruction methods.
                experiment_dir=seed_dir/"experiment"
                for internal_method in ("qaoa","sa","uniform","greedy"):
                    internal=experiment_dir/f"{internal_method}_relaxed.cif"
                    if not internal.is_file():
                        raise RuntimeError(
                            f"Missing final relaxed structure for {internal_method}: {internal}"
                        )
                    internal_pdb=out_case/f"{internal_method}_relaxed.pdb"
                    cif_to_pdb(internal,internal_pdb)
                    rows.append(evaluated_row(
                        target=target,seed=seed,method=internal_method,
                        reference=native,prediction=internal_pdb,
                        active=active,alignment=alignment,partners=partners,
                        phenix_executable=args.phenix_clashscore,
                    ))
            except Exception as exc:
                failures.append(dict(target=target,seed=seed,error=f"{type(exc).__name__}: {exc}"))

    if not rows:
        raise RuntimeError(f"No external baseline completed; failures={failures[:5]}")
    fields=sorted({key for row in rows for key in row})
    with (args.out_dir/"external_baseline_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    provenance=dict(
        scope="frozen validation structural-quality benchmark: Active-only FASPR packing plus common evaluation/clashscore for internal methods",
        faspr=str(args.faspr),faspr_sha256=sha256(args.faspr),
        phenix_clashscore=(None if args.phenix_clashscore is None else str(args.phenix_clashscore)),
        failures=failures,rows=len(rows),targets=len({r["target"] for r in rows}),
        expected_seeds=[int(v) for v in args.expected_seeds],
        packing_scope="active_sidechains_only; backbone and non-Active sidechains restored from perturbed input before scoring",
    )
    (args.out_dir/"run_summary.json").write_text(
        json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 1 if failures else 0


if __name__=="__main__":
    raise SystemExit(main())
