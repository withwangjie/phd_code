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
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

import gemmi

from subgraph_to_qubo import evaluate_atomistic_prediction


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def cif_to_pdb(source: Path, destination: Path) -> None:
    structure=gemmi.read_structure(str(source))
    if len(structure)!=1:
        raise ValueError(f"Expected one model: {source}")
    structure.write_pdb(str(destination))


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


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir",type=Path,required=True)
    parser.add_argument("--faspr",type=Path,required=True)
    parser.add_argument("--phenix-clashscore",type=Path)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--timeout-seconds",type=int,default=1800)
    args=parser.parse_args()
    if not args.faspr.is_file():
        parser.error(f"FASPR executable not found: {args.faspr}")
    if args.phenix_clashscore is not None and not args.phenix_clashscore.is_file():
        parser.error(f"Phenix clashscore executable not found: {args.phenix_clashscore}")
    if args.timeout_seconds<=0:
        parser.error("--timeout-seconds must be positive")

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
        seed_dirs=sorted((results_root/target).glob("seed_*"))
        for seed_dir in seed_dirs:
            perturbed=seed_dir/"perturbed_input.cif"
            if not perturbed.is_file():
                continue
            seed=seed_dir.name.removeprefix("seed_")
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
                metrics=evaluate_atomistic_prediction(
                    native,prediction_pdb,
                    active_residues=active,
                    alignment_residues=alignment,
                    partner_residues=partners,
                )
                row=dict(
                    target=target,seed=seed,method="faspr",
                    final_rmsd=metrics["sidechain_rmsd_angstrom"],
                    chi1_recovery=metrics["chi1_recovery_rate"],
                    fnat=metrics["fnat"],irmsd=metrics.get("irmsd"),
                    lrmsd=metrics.get("lrmsd"),
                    num_severe_clashes=metrics["num_severe_clashes"],
                    prediction_sha256=sha256(prediction_pdb),
                    faspr_executable_sha256=sha256(args.faspr),
                )
                if args.phenix_clashscore is not None:
                    row["molprobity_clashscore"]=phenix_clashscore(
                        args.phenix_clashscore,prediction_pdb)
                    row["phenix_executable_sha256"]=sha256(args.phenix_clashscore)
                rows.append(row)
            except Exception as exc:
                failures.append(dict(target=target,seed=seed,error=f"{type(exc).__name__}: {exc}"))

    if not rows:
        raise RuntimeError(f"No external baseline completed; failures={failures[:5]}")
    fields=sorted({key for row in rows for key in row})
    with (args.out_dir/"external_baseline_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    provenance=dict(
        scope="frozen validation structural baseline; not matched-compute QAOA comparison",
        faspr=str(args.faspr),faspr_sha256=sha256(args.faspr),
        phenix_clashscore=(None if args.phenix_clashscore is None else str(args.phenix_clashscore)),
        failures=failures,rows=len(rows),targets=len({r["target"] for r in rows}),
    )
    (args.out_dir/"run_summary.json").write_text(
        json.dumps(provenance,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
