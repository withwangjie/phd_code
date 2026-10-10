#!/usr/bin/env python3
"""External biological baselines for the frozen structural validation queue.

FASPR is treated as a mature side-chain packing baseline, not as a
matched-compute solver baseline. Optional Phenix clashscore supplies a standard
steric-quality metric in addition to the project's internal geometric clash
diagnostic. Missing configured executables fail closed.
"""
from __future__ import annotations
from nanoqc.common.device_errors import raise_if_resource_error

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

import gemmi

from nanoqc.qubo.subgraph_to_qubo import evaluate_atomistic_prediction
from nanoqc.common.repo_io import sha256_file as sha256


FASPR_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})


def _single_model(path: Path) -> gemmi.Structure:
    structure = gemmi.read_structure(str(path))
    if len(structure) != 1:
        raise ValueError(f"Expected one model: {path}")
    structure.remove_hydrogens()
    return structure


def _residues(structure: gemmi.Structure) -> list[tuple[gemmi.Chain, gemmi.Residue]]:
    return [(chain, residue) for chain in structure[0] for residue in chain]


def write_faspr_input(source: Path, destination: Path) -> list[str]:
    """Write a heavy-atom PDB that FASPR's fixed-column reader cannot misread.

    FASPR reads only ATOM records, a one-character chain ID (column 22), and
    starts a new residue whenever columns 23-27 (number plus insertion code)
    change, without looking at the chain. Author chain names such as ``A-2``
    cannot be written to PDB at all, and IMGT insertion codes or a repeated
    number across a chain boundary could merge residues. The FASPR input
    therefore uses one-character chain IDs in chain order and one global
    consecutive residue number, with every residue an ATOM record. The
    returned list gives the residue order, which is how the output is mapped
    back to the original author identities.
    """
    structure = _single_model(source)
    rows = _residues(structure)
    chains = list(structure[0])
    if len(chains) > len(FASPR_CHAIN_IDS):
        raise ValueError(f"FASPR input supports at most {len(FASPR_CHAIN_IDS)} chains: {source}")
    if len(rows) > 9999:
        raise ValueError(f"FASPR input supports at most 9999 residues: {source}")
    order = []
    number = 0
    for index, chain in enumerate(chains):
        order_chain = chain.name
        chain.name = FASPR_CHAIN_IDS[index]
        for residue in chain:
            if not gemmi.find_tabulated_residue(residue.name).is_amino_acid():
                raise ValueError(f"FASPR input contains a non-amino-acid residue "
                                 f"{order_chain}:{residue.seqid} {residue.name}: {source}")
            order.append(f"{order_chain}:{residue.seqid}:{residue.name}")
            number += 1
            residue.seqid = gemmi.SeqId(number, " ")
            residue.het_flag = "A"
    structure.setup_entities()
    text = structure.make_pdb_string(gemmi.PdbWriteOptions(minimal=True))
    destination.write_text("\n".join(line for line in text.splitlines() if line.strip()) + "\n")
    return order


def restrict_sidechain_packing(
    source: Path, packed_pdb: Path, order: list[str], destination: Path,
    active_residues: list[str],
) -> None:
    """Keep FASPR changes only on the declared Active side chains.

    FASPR repacks every eligible residue. The formal comparison optimizes only
    Active side chains, so the prediction is the identical perturbed input
    (heavy atoms, original author chain names and numbering) with only the
    Active side-chain atoms replaced by FASPR's. Backbone atoms, the
    C-terminal OXT that FASPR does not write, and non-Active side chains are
    therefore exactly the input's. A side-chain atom FASPR fails to build for
    an Active residue stays missing and fails the completeness check.
    """
    structure = _single_model(source)
    packed = gemmi.read_structure(str(packed_pdb))
    if len(packed) != 1:
        raise ValueError("FASPR output must have one model")
    rows = _residues(structure)
    packed_rows = [residue for chain in packed[0] for residue in chain]
    if len(rows) != len(order) or len(packed_rows) != len(order):
        raise ValueError(f"FASPR output has {len(packed_rows)} residues; input has {len(order)}")
    active = {str(rid) for rid in active_residues}
    seen = set()
    for (chain, residue), packed_residue, expected in zip(rows, packed_rows, order):
        rid = f"{chain.name}:{residue.seqid}"
        if f"{rid}:{residue.name}" != expected or packed_residue.name != residue.name:
            raise ValueError(f"FASPR output residue order differs from input at {expected}")
        if rid not in active:
            continue
        seen.add(rid)
        for index in reversed(range(len(residue))):
            if residue[index].name.strip() not in BACKBONE_ATOMS:
                del residue[index]
        for atom in packed_residue:
            if atom.name.strip() not in BACKBONE_ATOMS and not atom.element.is_hydrogen:
                residue.add_atom(atom)
    if seen != active:
        raise ValueError(f"Active residues absent from the FASPR input: {sorted(active - seen)}")
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(destination))


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
    if not (args.faspr.parent/"dun2010bbdep.bin").is_file():
        parser.error(f"FASPR rotamer library not found beside executable: {args.faspr.parent/'dun2010bbdep.bin'}")
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
                input_pdb=out_case/"faspr_input.pdb"
                prediction_pdb=out_case/"faspr_prediction.pdb"
                order=write_faspr_input(perturbed,input_pdb)
                run_checked(
                    [str(args.faspr),"-i",str(input_pdb),"-o",str(prediction_pdb)],
                    timeout=args.timeout_seconds,
                )
                if not prediction_pdb.is_file() or prediction_pdb.stat().st_size==0:
                    raise RuntimeError("FASPR did not produce a nonempty prediction")
                active_only_prediction=out_case/"faspr_active_only.cif"
                restrict_sidechain_packing(
                    perturbed,prediction_pdb,order,active_only_prediction,active
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
                # internal methods' final relaxed structures, read as the
                # mmCIF files they are (PDB cannot hold every author chain
                # name). These rows are not a matched-compute baseline; they
                # provide a common structural-quality scale across methods.
                experiment_dir=seed_dir/"experiment"
                for internal_method in ("qaoa","sa","uniform","greedy"):
                    internal=experiment_dir/f"{internal_method}_relaxed.cif"
                    if not internal.is_file():
                        raise RuntimeError(
                            f"Missing final relaxed structure for {internal_method}: {internal}"
                        )
                    rows.append(evaluated_row(
                        target=target,seed=seed,method=internal_method,
                        reference=native,prediction=internal,
                        active=active,alignment=alignment,partners=partners,
                        phenix_executable=args.phenix_clashscore,
                    ))
            except Exception as exc:
                raise_if_resource_error(exc, stage_hint="external structure baseline",
                                        record_path=args.out_dir/"device_resource_failure.json")
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
