"""All-atom modes: ``--structure-evaluation``, ``--allatom-experiment`` and
``--recovery-benchmark`` (retrospective side-chain recovery).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
import traceback
import json
from filelock import FileLock
from pathlib import Path
from typing import Optional, Sequence
import numpy as np
from tqdm.auto import tqdm
from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler
from nanoqc.common.repo_io import sha256_file as _ablation_digest, atomic_write_json_fsync as _ablation_atomic_json, repo_path
from nanoqc.experiments.benchmark_common import SHARED_HELPER_MODULES
from nanoqc.common.device_errors import raise_if_resource_error
from nanoqc.common.seed_streams import derive_child_seed
from nanoqc.experiments.research_ablation import _ablation_classical_counts, _ablation_summarize
from nanoqc.common.device_errors import raise_if_resource_error
from nanoqc.common.seed_streams import derive_child_seed
from nanoqc.experiments.research_ablation import _ablation_classical_counts, _ablation_summarize
from nanoqc.structure.physical_quality import (
    DEFAULT_MAX_PERTURBATION_ATTEMPTS,
    StructureQualityError,
    generated_input_geometry_audit,
)


def _qualified_perturbation(generator, seed: int, mode: str, minimum: float,
                            maximum: float, max_attempts: int):
    """Take the first geometry-valid draw without inspecting energy or reference."""
    if max_attempts < 1 or mode not in ("multi_chi", "chi1"):
        raise ValueError("Invalid generated-input perturbation protocol")
    attempts = []
    for index in range(max_attempts):
        draw_seed = seed if index == 0 else derive_child_seed(
            seed, "generated_input_geometry", str(index))
        if mode == "multi_chi":
            positions, angles = generator.perturb_sidechain_chis(draw_seed, minimum, maximum)
        else:
            positions, angles = generator.perturb_chi1(draw_seed, minimum, maximum)
        quality = generated_input_geometry_audit(generator.topology, positions)
        attempts.append(dict(attempt=index, draw_seed=draw_seed, angles=angles, quality=quality))
        if quality["accepted"]:
            return positions, angles, attempts
    raise StructureQualityError(
        f"No geometry-qualified generated input in {max_attempts} fixed attempts",
        category="generated_input_geometry", audit=dict(seed=seed, attempts=attempts))



def _structure_quality_assessment(relaxation: dict) -> dict:
    """Assess a saved output, without changing its energy or reference metrics."""
    failures=[]
    geometry=relaxation.get("stage2_physical_quality",relaxation["physical_quality_after"])
    forces=relaxation.get("stage2_force_quality",relaxation)
    if not geometry["topology_passed"]: failures.append("peptide_topology")
    if not geometry["geometry_passed"]: failures.append("extreme_nonbonded_overlap")
    # Skipped minimization is explicit and cannot be described as converged.
    if not forces["relaxation_converged"]: failures.append("relaxation_"+forces["relaxation_status"])
    return dict(evaluation_status="passed" if not failures else "failed",
                evaluation_failure_reasons=failures,
                movable_force_rms_kj_mol_nm=forces["movable_force_rms_kj_mol_nm"],
                movable_force_max_kj_mol_nm=forces["movable_force_max_kj_mol_nm"],
                extreme_nonbonded_pair_count=geometry["extreme_nonbonded_pair_count"])




def _structure_evaluation_main(argv: Optional[Sequence[str]] = None) -> int:
    """Evaluate explicit real-atom predictions; never reconstruct pseudo-atoms as native atoms.

    Manifest is a JSON list. Paths are relative to the manifest, not the CWD.
    Each case declares target, method, reference, prediction, active_residues,
    alignment_residues, partner_residues, selection_origin and protocol.
    """
    from nanoqc.qubo.subgraph_to_qubo import evaluate_atomistic_prediction
    parser = argparse.ArgumentParser(description="Real-atom structural evaluation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("structure_results"))
    args = parser.parse_args(argv)
    manifest = args.manifest.resolve()
    cases = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        parser.error("Manifest must be a nonempty list")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # (Version Discrepancy remediation) evaluate_complex_metrics.py is now a
    # transitive dependency via evaluate_atomistic_prediction's call below,
    # so it is tracked here exactly like every other module this manifest
    # already fingerprints.
    source_hashes = {n:_ablation_digest(repo_path(n)) for n in
                     ("batch_benchmark_hard_set.py", "subgraph_to_qubo.py", "evaluate_complex_metrics.py",
                      "model_egnn_pruning.py", *SHARED_HELPER_MODULES)}
    import gemmi
    provenance = dict(manifest_sha256=_ablation_digest(manifest), code_sha256=source_hashes,
        numpy=np.__version__, gemmi=gemmi.__version__, python=sys.version)
    rows, failures = [], 0
    with FileLock(str(out/".lock"), timeout=0):
        record = out/"run_manifest.json"
        if record.exists() and json.loads(record.read_text()) != provenance:
            raise ValueError("Manifest/code/version changed; use a new output directory")
        _ablation_atomic_json(record, provenance)
        (out/"cases").mkdir(exist_ok=True)
        temp_csv=out/"structure_metrics.csv.tmp"
        with temp_csv.open("w",newline="",encoding="utf-8") as handle:
            writer=None
            for index, case in enumerate(tqdm(cases, desc="Real-atom evaluation")):
                try:
                    required=("target","method","reference","prediction","active_residues",
                              "alignment_residues","partner_residues","selection_origin","protocol")
                    if any(key not in case for key in required):
                        raise ValueError("Missing manifest fields: "+str(set(required)-case.keys()))
                    if case["protocol"] not in ("prediction","validation_control") or not case["selection_origin"]:
                        raise ValueError("Declare prediction/validation_control and selection_origin")
                    for key in ("active_residues","alignment_residues","partner_residues"):
                        if not isinstance(case[key],list) or any(not isinstance(r,str) for r in case[key]):
                            raise ValueError(f"{key} must be a list of author chain:residue IDs")
                    reference=(manifest.parent/str(case["reference"])).resolve()
                    prediction=(manifest.parent/str(case["prediction"])).resolve()
                    hashes=dict(reference=_ablation_digest(reference),prediction=_ablation_digest(prediction))
                    if hashes["reference"]==hashes["prediction"] and case["protocol"]!="validation_control":
                        raise ValueError("Identical reference/prediction must be labelled validation_control")
                    artifact=out/"cases"/f"{index:06d}.json"
                    signature=dict(case=case,structure_sha256=hashes)
                    if artifact.exists():
                        saved=json.loads(artifact.read_text())
                        if saved["signature"]!=signature:
                            raise ValueError("Input structure changed; use a new output directory")
                        result=saved["metrics"]
                    else:
                        result=evaluate_atomistic_prediction(reference,prediction,
                            active_residues=case["active_residues"], alignment_residues=case["alignment_residues"],
                            partner_residues=case["partner_residues"], model_index=int(case.get("model_index",0)),
                            contact_cutoff=float(case.get("contact_cutoff",5.)),
                            proximity_cutoff=float(case.get("proximity_cutoff",2.)),
                            chi1_tolerance=float(case.get("chi1_tolerance",20.)))
                        _ablation_atomic_json(artifact,dict(signature=signature,metrics=result))
                    row=dict(case_index=index,target=case["target"],method=case["method"],
                             protocol=case["protocol"],selection_origin=case["selection_origin"],**hashes)
                    row.update({k:v for k,v in result.items() if not isinstance(v,(list,dict))})
                    if writer is None:
                        writer=csv.DictWriter(handle,fieldnames=list(row),extrasaction="ignore"); writer.writeheader()
                    writer.writerow(row); handle.flush(); os.fsync(handle.fileno()); rows.append(row)
                except Exception as exc:
                    raise_if_resource_error(exc, stage_hint="structure evaluation",
                                            record_path=out/"device_resource_failure.json")
                    failures+=1
                    with (out/"failed_cases.log").open("a",encoding="utf-8") as log:
                        log.write(json.dumps(dict(case_index=index,case=case))+"\n"+traceback.format_exc()+"\n")
                        log.flush(); os.fsync(log.fileno())
        os.replace(temp_csv,out/"structure_metrics.csv")
        report=["# Real-atom structural validation", "",f"Requested: {len(cases)}; successful: {len(rows)}; failed: {failures}.",
            "Validation controls are implementation checks, not structure prediction results.",
            "Alignment uses declared non-Active N/CA/C/O atoms. Active side chains are not independently fitted.",
            "Symmetry correction covers ASP/GLU/ARG/PHE/TYR; aromatic paired swaps are coupled. Prochiral VAL/LEU methyls are not swapped.",
            "Chi1 is a partial torsion metric, not full rotamer recovery. Gly has no side-chain heavy atoms; Ala/Gly lack chi1.",
            "Contact precision/recall concern declared Active-partner residue pairs; partner selection defines the denominator.",
            "Severe proximity counts are geometric <2 A (unless configured) sidechain/partner pairs, not MolProbity clashscore or force-field energy.",
            "No all-atom candidate generation, minimization, docking or structural-accuracy claim is supplied by this evaluator.",
            "", "| Target | Method | Protocol | Side-chain RMSD (A) | Chi1 recovery | Contact F1 |", "|---|---|---|---:|---:|---:|"]
        for row in rows:
            report.append(f"| {row['target']} | {row['method']} | {row['protocol']} | {row['sidechain_rmsd_angstrom']} | {row['chi1_recovery_rate']} | {row['contact_f1']} |")
        report += ["", "Atom parsing follows Gemmi author IDs: https://gemmi.readthedocs.io/en/stable/mol.html"]
        temporary=out/"structure_report.md.tmp"
        temporary.write_text("\n".join(report),encoding="utf-8")
        os.replace(temporary,out/"structure_report.md")
    print(f"Structure results: {out}; failures: {failures}")
    return 1 if failures else 0



def _allatom_experiment_main(argv: Optional[Sequence[str]] = None) -> int:
    """One explicit all-atom case: QUBO -> matched-output solvers -> CIF -> evaluation."""
    from nanoqc.qubo.subgraph_to_qubo import AllAtomInterfaceQUBOBuilder, evaluate_atomistic_prediction
    import openmm
    parser=argparse.ArgumentParser(description="All-atom fixed-backbone multi-chi side-chain experiment")
    parser.add_argument("--eval-shots",type=int,choices=(200,500,1000))
    parser.add_argument("--loop-relax-iterations",type=int,default=0)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--outputs",type=int,default=1000)
    parser.add_argument("--max-evals",type=int,default=90)
    parser.add_argument("--qaoa-depth",type=int,default=2)
    parser.add_argument("--sa-passes",type=int,default=100)
    parser.add_argument("--robust-qaoa",action="store_true",help="Exact-subspace multistart optimization; total max-evals budget")
    parser.add_argument("--qaoa-restarts",type=int,default=4)
    parser.add_argument("--qaoa-objective",choices=("mean","cvar"),default="cvar")
    parser.add_argument("--cvar-alpha",type=float,default=.1)
    parser.add_argument("--parameter-scale",choices=("max_coefficient","feasible_iqr"),default="max_coefficient")
    parser.add_argument("--relax-iterations",type=int,default=200)
    parser.add_argument("--solvent-model",choices=("vacuum","gbn2"),default="vacuum")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--optimize-seed",type=int,default=None,
        help="Independent optimizer sub-seed (defaults to --seed when omitted, for standalone-"
             "invocation compatibility). Drives XYMixerQAOASampler search RNG only.")
    parser.add_argument("--sample-seed",type=int,default=None,
        help="Independent final-output-sampling sub-seed (defaults to --seed when omitted). Drives "
             "the finite-shot draw for every solver reported output, never the QAOA search itself.")
    parser.add_argument("--measurement-seed",type=int,default=None,
        help="Independent in-search finite-shot measurement sub-seed (defaults to --seed when omitted). "
             "Drives optimize_robust's CVaR/mean objective-evaluation draws WHILE searching -- distinct "
             "from both --optimize-seed (restart initialization) and --sample-seed (final output draw).")
    args=parser.parse_args(argv)
    if min(args.outputs,args.max_evals,args.sa_passes,args.qaoa_depth)<=0 or args.relax_iterations<0:
        parser.error("Invalid budgets")
    optimize_seed = args.optimize_seed if args.optimize_seed is not None else args.seed
    measurement_seed = args.measurement_seed if args.measurement_seed is not None else args.seed
    sample_seed = args.sample_seed if args.sample_seed is not None else args.seed
    manifest=args.manifest.resolve(); case=json.loads(manifest.read_text())
    from nanoqc.common.prediction_contract import validate_prediction_contract
    validate_prediction_contract(case, manifest.parent)
    if case.get("protocol") not in ("validation_control","prediction") or not case.get("selection_origin"):
        parser.error("Manifest requires protocol and selection_origin")
    source=(manifest.parent/case["input_structure"]).resolve()
    reference=(manifest.parent/case["reference_structure"]).resolve() if case.get("reference_structure") else None
    if reference and _ablation_digest(source)==_ablation_digest(reference) and case["protocol"]!="validation_control":
        parser.error("Native-input controls must be declared validation_control")
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        resolved_optimize_seed=optimize_seed,resolved_measurement_seed=measurement_seed,resolved_sample_seed=sample_seed,
        manifest_sha256=_ablation_digest(manifest),input_sha256=_ablation_digest(source),
        reference_sha256=_ablation_digest(reference) if reference else None,
        code_sha256={n:_ablation_digest(repo_path(n)) for n in
            # (Version Discrepancy remediation) evaluate_complex_metrics.py
            # added: evaluate_atomistic_prediction (called via evaluate()
            # below) now depends on it as the single source of truth for
            # Fnat/iRMSD/LRMSD/DockQ/severe-clash metrics; structural_quality.py
            # is kept because evaluate_atomistic_prediction still sources the
            # legacy backbone-only DockQ variant from it (see subgraph_to_qubo.py).
            ("batch_benchmark_hard_set.py","subgraph_to_qubo.py","qaoa_interface_sampler.py",
             "prediction_contract.py","structural_quality.py","evaluate_complex_metrics.py",
             "model_egnn_pruning.py",*SHARED_HELPER_MODULES)},openmm=openmm.__version__)
    with FileLock(str(out/".lock"),timeout=0):
        marker=out/"run_manifest.json"
        if marker.exists() and json.loads(marker.read_text())!=provenance:
            raise ValueError("All-atom provenance changed; use a new output directory")
        _ablation_atomic_json(marker,provenance)
        if (out/"completed.json").exists():
            saved=json.loads((out/"completed.json").read_text())
            if all((out/name).exists() and _ablation_digest(out/name)==digest for name,digest in saved["artifacts"].items()):
                print(f"Verified completed experiment: {out}");return 0
            raise ValueError("Completed artifacts changed or missing")
        print("Preparing complete atoms and Amber14 parameters",flush=True)
        rotamer_cfg=case.get("rotamer_model",{}) or {}
        builder=AllAtomInterfaceQUBOBuilder(
            source,case["active_residues"],seed=args.seed,
            chi1_angles=case.get("chi1_angles"),
            site_scores=case.get("active_site_scores"),
            candidate_relax_iterations=int(case.get("candidate_relax_iterations",0)),
            rotamer_mode=rotamer_cfg.get("mode","dunbrack2010"),
            rotamer_library_path=rotamer_cfg.get("library_path"),
            rotamer_probability_floor=float(rotamer_cfg.get("probability_floor",1e-4)),
            rotamer_sigma_offsets=rotamer_cfg.get("sigma_offsets",[-1.0,0.0,1.0]),
            solvent_model=args.solvent_model)
        builder.write_structure(builder.base_positions,out/"prepared_input.cif")
        _ablation_atomic_json(out/"structure_preparation_audit.json",dict(
            source=builder.input_quality,after_hydrogen_addition=builder.preparation_quality,
            input_role=case["protocol"],
            generated_input_policy="perturbed recovery inputs are retained, not relabelled as source exclusions"))
        # Candidate coordinates make reconstruction independently auditable.
        np.savez_compressed(out/"candidate_coordinates.npz",base_positions_nm=builder.base_positions,
            **{f"indices_{i}":c["indices"] for i,c in enumerate(builder.candidates)},
            **{f"positions_nm_{i}":c["positions"] for i,c in enumerate(builder.candidates)})
        print("Decomposing full force-field energies and checking equivalence",flush=True)
        try:
            qubo=builder.build()
        except Exception as exc:
            raise_if_resource_error(exc, stage_hint="allatom QUBO build",
                                    record_path=out/"device_resource_failure.json")
            _ablation_atomic_json(out/"allatom_build_failure.json",dict(
                error=str(exc),category="candidate_or_model_construction",
                preparation=builder.preparation_quality,
                decomposition=getattr(builder,"decomposition_quality",None)))
            raise
        _ablation_atomic_json(out/"candidate_geometry_audit.json",builder.decomposition_quality)
        qubo.export(out,"allatom")
        # ONE sampler instance: XYMixerQAOASampler.optimize_robust/.sample/
        # .simulated_annealing already accept explicit optimize_seed/
        # measurement_seed/sample_seed overrides directly.
        quantum_instance=qubo.to_quantum_instance()
        sampler=XYMixerQAOASampler.from_instance(
            quantum_instance,simulation_mode="subspace",p=args.qaoa_depth,
            seed=optimize_seed,shots=args.outputs,initial_state="wstate")
        truth=sampler.enumerate_ground_states();energies=sampler.feasible_energy_map()
        records=[]
        def evaluate(path):
            return evaluate_atomistic_prediction(reference,path,active_residues=case["active_residues"],
                alignment_residues=case["alignment_residues"],partner_residues=case["partner_residues"])
        if reference:
            _ablation_atomic_json(out/"initial_structure_metrics.json",evaluate(out/"prepared_input.cif"))
        relax_only_path=out/"relax_only.cif"
        relax_only=builder.relax_positions(builder.base_positions,relax_only_path,
                                           minimize_iterations=args.relax_iterations)
        if args.loop_relax_iterations:
            relax_only.update(builder.relax_cdr_loop(relax_only_path,case.get('cdr3_residues',[]),iterations=args.loop_relax_iterations))
        _ablation_atomic_json(out/"relax_only_result.json",dict(relaxation=relax_only,
            **_structure_quality_assessment(relax_only),
            structure_after_relaxation=evaluate(relax_only_path) if reference else None))
        quality_outcomes={"relax_only":_structure_quality_assessment(relax_only)}
        with (out/"allatom_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=None
            for method in ("qaoa","sa","uniform","greedy"):
                print(f"Sampling and reconstructing {method}",flush=True)
                start=time.perf_counter()
                if method=="qaoa":
                    # (Version Discrepancy remediation) optimize_robust's
                    # CVaR-quantile keyword is cvar_alpha, not the bare alpha
                    # used in earlier revisions of this call site -- passing
                    # alpha= against the current qaoa_interface_sampler.py
                    # raises TypeError immediately (an unexpected-keyword
                    # error, not a silent misconfiguration), but the fix
                    # keeps this entry point in sync with the single source
                    # of truth rather than leaving it broken.
                    #
                    # --eval-shots has no CLI default (None), and
                    # --robust-qaoa can be passed without it -- this used to
                    # fall back to the now-removed analytic-expectation
                    # mode, which optimize_robust no longer accepts
                    # (eval_shots must be a positive integer). Resolve to
                    # this project's default of 500 whenever --robust-qaoa
                    # is set but --eval-shots wasn't, so this entry point
                    # never passes eval_shots=None into optimize_robust.
                    resolved_eval_shots=args.eval_shots or 500
                    opt=(sampler.optimize_robust(max_evals=args.max_evals,restarts=args.qaoa_restarts,
                        objective=args.qaoa_objective,cvar_alpha=args.cvar_alpha,parameter_scale=args.parameter_scale,eval_shots=resolved_eval_shots,
                        optimize_seed=optimize_seed,measurement_seed=measurement_seed)
                        if args.robust_qaoa or args.eval_shots else sampler.optimize(method="cobyla",max_iterations=args.max_evals))
                    sampled=sampler.sample(opt,shots=args.outputs,ground_state=truth,sample_seed=sample_seed);counts=dict(sampled.counts)
                    # Honest convergence/measurement-accounting fields
                    # (optimizer_success, termination_reason, gamma_best,
                    # beta_best, best_mean_energy) mirrored into the saved
                    # artifact alongside the pre-existing fields, so a saved
                    # optimization.json is fully self-describing without
                    # requiring the reader to re-derive them from history.
                    _ablation_atomic_json(out/"optimization.json",dict(energy=opt.energy,history=list(opt.history),
                        success=opt.success,evaluations=opt.evaluations,gammas=opt.gammas.tolist(),betas=opt.betas.tolist(),
                        objective_name=opt.objective_name,objective_value=opt.objective_value,
                        restart_records=opt.restart_records,history_kind=opt.objective_name,
                        total_opt_shots=opt.total_opt_shots,
                        eval_shots=resolved_eval_shots if (args.robust_qaoa or args.eval_shots) else None,
                        optimizer_success=opt.optimizer_success,termination_reason=opt.termination_reason,
                        gamma_best=opt.gamma_best.tolist() if opt.gamma_best is not None else None,
                        beta_best=opt.beta_best.tolist() if opt.beta_best is not None else None,
                        best_mean_energy=opt.best_mean_energy))
                elif method=="sa":
                    counts=dict(sampler.simulated_annealing(num_reads=args.outputs,site_passes=args.sa_passes,seed=sample_seed,ground_state=truth).counts)
                else:
                    counts,_=_ablation_classical_counts(sampler,args.outputs,sample_seed+101,method=="greedy",50)
                elapsed=time.perf_counter()-start
                selected=min(counts,key=lambda bits:(energies[bits],bits))
                prediction=out/(method+"_relaxed.cif")
                relaxation=builder.reconstruct(selected,prediction,minimize_iterations=args.relax_iterations)
                if args.loop_relax_iterations:
                    relaxation.update(builder.relax_cdr_loop(prediction,case.get('cdr3_residues',[]),iterations=args.loop_relax_iterations))
                if sum(counts.values()) != args.outputs:
                    raise AssertionError('Solver output budget mismatch')
                frequencies=np.array(list(counts.values()),dtype=float)/args.outputs
                budget_metrics=dict(total_opt_shots=opt.total_opt_shots if method=='qaoa' else 0,
                    bitstring_entropy=float(-np.sum(frequencies*np.log(frequencies))),
                    bitstring_entropy_upper_bound=float(np.log(args.outputs)),
                    low_energy_fraction=sum(c for b,c in counts.items() if energies[b]<=truth.energy+2.)/args.outputs)
                expected=energies[selected]+qubo.metadata["physical_constant_offset"]
                qubo_energy_discrepancy=float(relaxation["discrete_energy_kcal"]-expected)
                if qubo.metadata.get("pair_decomposition","exact")=="exact":
                    if not np.isclose(expected,relaxation["discrete_energy_kcal"],atol=1e-4,rtol=1e-9):
                        raise AssertionError("Selected structure and QUBO energy disagree")
                elif not math.isfinite(qubo_energy_discrepancy):
                    raise FloatingPointError("Non-finite full-energy/QUBO discrepancy")
                # Pairwise-approximate models (GBN2): the full-energy minus QUBO
                # energy of the selected structure is recorded, never hidden.
                relaxation=dict(relaxation,qubo_energy_kcal=float(expected),
                    qubo_energy_discrepancy_kcal=qubo_energy_discrepancy,
                    pair_decomposition=qubo.metadata.get("pair_decomposition","exact"))
                before=evaluate(prediction.with_name(prediction.stem+"_discrete.cif")) if reference else None
                after=evaluate(prediction) if reference else None
                result=dict(**budget_metrics,method=method,protocol=case["protocol"],selected_bits=list(selected),
                    **_structure_quality_assessment(relaxation),
                    counts=[dict(bits=list(b),count=c) for b,c in sorted(counts.items())],
                    sampling=_ablation_summarize(counts,energies,truth.energy,2.),relaxation=relaxation,
                    structure_before_relaxation=before,structure_after_relaxation=after,solver_seconds=elapsed)
                _ablation_atomic_json(out/(method+"_result.json"),result)
                quality_keys=('dockq_score','fnat','irmsd','lrmsd','dockq_category','num_severe_clashes','has_severe_clash','dockq_definition','dockq_backbone_score')
                result.update({k:after[k] for k in quality_keys} if after else {})
                _ablation_atomic_json(out/(method+"_result.json"),result)
                row=dict(**budget_metrics,**({k:after[k] for k in quality_keys} if after else {}),method=method,protocol=case["protocol"],outputs=sum(counts.values()),
                    bits=len(selected),ground_gap=energies[selected]-truth.energy,solver_seconds=elapsed,**relaxation,
                    sidechain_rmsd_before=before["sidechain_rmsd_angstrom"] if before else None,
                    sidechain_rmsd_after=after["sidechain_rmsd_angstrom"] if after else None)
                quality_outcomes[method]=_structure_quality_assessment(relaxation)
                row.update(quality_outcomes[method])
                if writer is None:
                    writer=csv.DictWriter(handle,fieldnames=list(row),extrasaction="ignore");writer.writeheader()
                writer.writerow(row);handle.flush();os.fsync(handle.fileno());records.append(row)
        lines=["# All-atom fixed-backbone experiment", "", "Protocol: "+case["protocol"],
            f"Amber14 potential ({qubo.metadata.get('solvent')}); not binding free energy. "
            f"Candidate states: {qubo.metadata.get('model')}; pair decomposition: {qubo.metadata.get('pair_decomposition')}.",
            "Same output count and same relaxation conditions; not equal total computational cost. The native reference never enters candidate selection or energy ranking.",
            "Only the lowest discrete-energy sampled state per method is relaxed; no reference-based selection. Iteration cap does not guarantee convergence.",
            "Sampling uses classical exact-subspace simulation. These results do not establish quantum advantage.",
            "", "| Method | Discrete energy (kcal/mol) | Relaxed energy | SC RMSD before | SC RMSD after | Physical acceptance | RMS force (kJ/mol/nm) |", "|---|---:|---:|---:|---:|---|---:|"]
        for r in records:
            lines.append(f"| {r['method']} | {r['discrete_energy_kcal']:.6g} | {r['relaxed_energy_kcal']:.6g} | {r['sidechain_rmsd_before']} | {r['sidechain_rmsd_after']} | {r['evaluation_status']} | {r['movable_force_rms_kj_mol_nm']:.6g} |")
        (out/"allatom_report.md").write_text("\n".join(lines),encoding="utf-8")
        _ablation_atomic_json(out/"structure_quality_summary.json",dict(
            outputs_requested=5,outputs_evaluated=len(quality_outcomes),
            outputs_failed=sum(x["evaluation_status"]=="failed" for x in quality_outcomes.values()),
            outcomes=quality_outcomes,
            failure_category="method_output; not an input-data exclusion"))
        if any(x["evaluation_status"]=="failed" for x in quality_outcomes.values()):
            print("Structure quality acceptance failed; all evaluated outputs and metrics retained",flush=True)
            return 1
        files=[p for p in out.iterdir() if p.is_file() and p.name not in (".lock","completed.json")]
        _ablation_atomic_json(out/"completed.json",dict(artifacts={p.name:_ablation_digest(p) for p in files}))
        print(out/"allatom_report.md")
    return 0



def _recovery_comparison(initial: float, relax_only: float, final: float) -> dict:
    """Positive gains mean structural improvement; negative values are retained."""
    if not np.isfinite([initial,relax_only,final]).all():
        raise ValueError("Recovery RMSDs must be finite")
    return dict(initial_rmsd=initial,relax_only_rmsd=relax_only,final_rmsd=final,
        improvement_vs_input=initial-final,improvement_vs_relax_only=relax_only-final,
        better_than_input=bool(final<initial-1e-6),better_than_relax_only=bool(final<relax_only-1e-6))


def _recovery_benchmark_main(argv: Optional[Sequence[str]] = None) -> int:
    """Retrospective multi-seed side-chain perturbation/recovery, with relax-only control."""
    from nanoqc.qubo.subgraph_to_qubo import AllAtomInterfaceQUBOBuilder
    import openmm
    parser=argparse.ArgumentParser(description="Fixed-backbone perturbation-recovery control")
    parser.add_argument("--eval-shots",type=int,choices=(200,500,1000))
    parser.add_argument("--loop-relax-iterations",type=int,default=0)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out-dir",type=Path,required=True)
    parser.add_argument("--seeds",nargs="+",type=int,default=[42,43,44])
    parser.add_argument("--optimize-seeds",nargs="+",type=int,default=None,
        help="Per-seed independent optimizer sub-seeds, positionally matched to --seeds "
             "(defaults elementwise to --seeds when omitted, for standalone-invocation compatibility). "
             "Never the same values as --seeds or --sample-seeds.")
    parser.add_argument("--sample-seeds",nargs="+",type=int,default=None,
        help="Per-seed independent final-sampling sub-seeds, positionally matched to --seeds "
             "(defaults elementwise to --seeds when omitted).")
    parser.add_argument("--measurement-seeds",nargs="+",type=int,default=None,
        help="Per-seed independent in-search measurement sub-seeds, positionally matched to --seeds "
             "(defaults elementwise to --seeds when omitted). Drives optimize_robust's finite-shot "
             "CVaR/mean draws WHILE searching -- distinct from --optimize-seeds and --sample-seeds.")
    parser.add_argument("--perturbation-mode",choices=("multi_chi","chi1"),default="multi_chi")
    parser.add_argument("--solvent-model",choices=("vacuum","gbn2"),default="vacuum")
    parser.add_argument("--min-perturb-degrees",type=float,default=40.)
    parser.add_argument("--max-perturb-degrees",type=float,default=120.)
    parser.add_argument("--perturbation-max-attempts", "--max-perturbation-attempts",
        dest="perturbation_max_attempts", type=int,
        default=DEFAULT_MAX_PERTURBATION_ATTEMPTS,
        help="Fixed upper bound for geometry-only generated-input draws per seed")
    # A44: generated inputs meet the A24 inter-residue heavy-atom floor (the
    # orchestrator passes data_audit.min_interresidue_heavy_distance_angstrom)
    # and the all-atom near-coincidence floor; invalid draws are redrawn.
    parser.add_argument("--min-input-heavy-distance", type=float, default=1.0)

    parser.add_argument("--outputs",type=int,default=1000)
    parser.add_argument("--max-evals",type=int,default=90)
    parser.add_argument("--qaoa-depth",type=int,default=2)
    parser.add_argument("--sa-passes",type=int,default=100)
    parser.add_argument("--relax-iterations",type=int,default=200)
    parser.add_argument("--robust-qaoa",action="store_true")
    parser.add_argument("--qaoa-restarts",type=int,default=4)
    parser.add_argument("--qaoa-objective",choices=("mean","cvar"),default="cvar")
    parser.add_argument("--cvar-alpha",type=float,default=.1)
    parser.add_argument("--parameter-scale",choices=("max_coefficient","feasible_iqr"),default="max_coefficient")
    args=parser.parse_args(argv)
    if len(args.seeds)!=len(set(args.seeds)) or min(args.seeds)<0:
        parser.error("Use unique nonnegative seeds")
    if args.optimize_seeds is not None and len(args.optimize_seeds)!=len(args.seeds):
        parser.error("--optimize-seeds must match --seeds length")
    if args.sample_seeds is not None and len(args.sample_seeds)!=len(args.seeds):
        parser.error("--sample-seeds must match --seeds length")
    if args.measurement_seeds is not None and len(args.measurement_seeds)!=len(args.seeds):
        parser.error("--measurement-seeds must match --seeds length")
    if not 0<args.min_perturb_degrees<=args.max_perturb_degrees<=180:
        parser.error("Invalid perturbation angle range")
    if not 0 < args.min_perturb_degrees <= args.max_perturb_degrees <= 180:
        parser.error("Invalid perturbation angle range")
    if args.perturbation_max_attempts < 1:
        parser.error("--perturbation-max-attempts must be positive")
    if not 0 < args.min_input_heavy_distance <= 1.0:
        parser.error("Require 0 < --min-input-heavy-distance <= 1.0")

    if min(args.outputs,args.max_evals,args.sa_passes,args.qaoa_depth)<=0 or args.relax_iterations<0:
        parser.error("Invalid solver budgets")
    manifest=args.manifest.resolve();case=json.loads(manifest.read_text())
    native=(manifest.parent/case["native_structure"]).resolve()
    for key in ("active_residues","alignment_residues","partner_residues","selection_origin"):
        if not case.get(key): parser.error("Missing "+key)
    out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    provenance=dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        input_sha256=_ablation_digest(native),manifest_sha256=_ablation_digest(manifest),
        code_sha256={n:_ablation_digest(repo_path(n)) for n in
            ("batch_benchmark_hard_set.py","subgraph_to_qubo.py","qaoa_interface_sampler.py",
             "model_egnn_pruning.py",*SHARED_HELPER_MODULES)},openmm=openmm.__version__)
    rows=[];failures=0;failure_records=[]
    with FileLock(str(out/".lock"),timeout=0):
        record=out/"run_manifest.json"
        if record.exists() and json.loads(record.read_text())!=provenance:
            raise ValueError("Recovery provenance changed; use a new output directory")
        _ablation_atomic_json(record,provenance)
        rotamer_cfg=case.get("rotamer_model",{}) or {}
        generator=AllAtomInterfaceQUBOBuilder(
            native,case["active_residues"],
            site_scores=case.get("active_site_scores"),seed=42,
            rotamer_mode=rotamer_cfg.get("mode","dunbrack2010"),
            rotamer_library_path=rotamer_cfg.get("library_path"),
            rotamer_probability_floor=float(rotamer_cfg.get("probability_floor",1e-4)),
            rotamer_sigma_offsets=rotamer_cfg.get("sigma_offsets",[-1.0,0.0,1.0]),
            solvent_model=args.solvent_model)
        _ablation_atomic_json(out/"native_preparation_audit.json",dict(
            source=generator.input_quality,after_hydrogen_addition=generator.preparation_quality))
        if not generator.input_quality["geometry_passed"]:
            raise StructureQualityError("Native source has an extreme nonbonded overlap",
                category="input_geometry",audit=generator.input_quality)
        with (out/"recovery_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=None
            for idx, seed in enumerate(args.seeds):
                optimize_seed = args.optimize_seeds[idx] if args.optimize_seeds is not None else seed
                measurement_seed = args.measurement_seeds[idx] if args.measurement_seeds is not None else seed
                sample_seed = args.sample_seeds[idx] if args.sample_seeds is not None else seed
                try:
                    directory=out/f"seed_{seed}";directory.mkdir(exist_ok=True)
                    perturbed=directory/"perturbed_input.cif";metadata=directory/"perturbation.json"
                    attempts_path=directory/"perturbation_attempts.json"
                    if metadata.exists():
                        saved=json.loads(metadata.read_text())
                        if not perturbed.exists() or _ablation_digest(perturbed)!=saved["structure_sha256"]:
                            raise ValueError("Perturbed input changed or missing")
                        if not attempts_path.is_file() or _ablation_digest(attempts_path)!=saved.get("attempts_sha256"):
                            raise ValueError("Generated-input geometry ledger changed or missing")
                    else:
                        try:
                            positions, angles, attempts = _qualified_perturbation(
                                generator, seed, args.perturbation_mode,
                                args.min_perturb_degrees, args.max_perturb_degrees,
                                args.perturbation_max_attempts)
                        except StructureQualityError as exc:
                            _ablation_atomic_json(attempts_path, exc.audit)
                            raise
                        _ablation_atomic_json(attempts_path, dict(
                            seed=seed,
                            selected_attempt=attempts[-1]["attempt"],
                            max_attempts=args.perturbation_max_attempts,
                            attempts=attempts,
                            min_input_heavy_distance_angstrom=args.min_input_heavy_distance))
                        perturb_protocol = (
                            "retrospective "
                            + ("multi-chi" if args.perturbation_mode == "multi_chi" else "chi1-only")
                            + " recovery; first deterministic perturbation satisfying the existing "
                            "A24/A28 absolute-distance floors and the all-atom near-coincidence floor; "
                            "no energy/reference/solver-based selection")
                        generator.write_structure(positions, perturbed)
                        _ablation_atomic_json(metadata, dict(
                            seed=seed, angles=angles, structure_sha256=_ablation_digest(perturbed),
                            attempts_sha256=_ablation_digest(attempts_path),
                            selected_attempt=attempts[-1]["attempt"],
                            perturbation_mode=args.perturbation_mode,
                            protocol=perturb_protocol,
                            input_validity=dict(
                                min_interresidue_heavy_distance_angstrom=args.min_input_heavy_distance,
                                attempts=attempts,
                            ),
                        ))

                    child_case=dict(input_structure=str(perturbed),reference_structure=str(native),
                        cdr3_residues=case.get('cdr3_residues',[]),pruning=case.get('pruning'),
                        candidate_relax_iterations=int(case.get("candidate_relax_iterations",0)),
                        active_residues=case["active_residues"],
                        active_site_scores=case.get("active_site_scores"),
                        rotamer_model=case.get("rotamer_model"),
                        alignment_residues=case["alignment_residues"],
                        partner_residues=case["partner_residues"],protocol="validation_control",
                        selection_origin=case["selection_origin"]+"; retrospective fixed-backbone perturbation-recovery")
                    child_manifest=directory/"experiment.json"
                    _ablation_atomic_json(child_manifest,child_case)
                    experiment=directory/"experiment"
                    child_status=_allatom_experiment_main(["--manifest",str(child_manifest),"--out-dir",str(experiment),
                        "--outputs",str(args.outputs),"--max-evals",str(args.max_evals),
                        "--qaoa-depth",str(args.qaoa_depth),"--sa-passes",str(args.sa_passes),
                        "--relax-iterations",str(args.relax_iterations),"--seed",str(seed),
                        "--optimize-seed",str(optimize_seed),"--measurement-seed",str(measurement_seed),"--sample-seed",str(sample_seed),
                        "--loop-relax-iterations",str(args.loop_relax_iterations),
                        "--solvent-model",args.solvent_model,
                        *(["--eval-shots",str(args.eval_shots)] if args.eval_shots else []),
                        *(["--robust-qaoa","--qaoa-restarts",str(args.qaoa_restarts),
                           "--qaoa-objective",args.qaoa_objective,"--cvar-alpha",str(args.cvar_alpha),
                           "--parameter-scale",args.parameter_scale] if args.robust_qaoa else [])])
                    initial=json.loads((experiment/"initial_structure_metrics.json").read_text())
                    control=json.loads((experiment/"relax_only_result.json").read_text())
                    for method in ("qaoa","sa","uniform","greedy"):
                        result=json.loads((experiment/(method+"_result.json")).read_text())
                        final=result["structure_after_relaxation"]
                        recovery=_recovery_comparison(initial["sidechain_rmsd_angstrom"],
                            control["structure_after_relaxation"]["sidechain_rmsd_angstrom"],final["sidechain_rmsd_angstrom"])
                        final_energy=result["relaxation"].get("stage2_physical_energy_kcal",result["relaxation"]["relaxed_energy_kcal"])
                        energy_drop=result["relaxation"]["discrete_energy_kcal"]-final_energy
                        rmsd_before=result["structure_before_relaxation"]["sidechain_rmsd_angstrom"]
                        row=dict(target=case.get("target",native.stem),seed=seed,optimize_seed=optimize_seed,
                            evaluation_status=result["evaluation_status"],
                            evaluation_failure_reasons=json.dumps(result["evaluation_failure_reasons"]),
                            control_evaluation_status=control["evaluation_status"],
                            measurement_seed=measurement_seed,sample_seed=sample_seed,method=method,**recovery,
                            **{k:result[k] for k in ('dockq_score','fnat','irmsd','lrmsd','dockq_category','num_severe_clashes','has_severe_clash','total_opt_shots','bitstring_entropy','low_energy_fraction','dockq_definition','dockq_backbone_score')},
                            chi1_recovery_initial=initial["chi1_recovery_rate"],chi1_recovery_final=final["chi1_recovery_rate"],
                            all_chi_recovery_initial=initial.get("all_chi_recovery_rate"),
                            all_chi_recovery_final=final.get("all_chi_recovery_rate"),
                            chi_recovery_rates_initial=json.dumps(initial.get("chi_recovery_rates",{}),sort_keys=True),
                            chi_recovery_rates_final=json.dumps(final.get("chi_recovery_rates",{}),sort_keys=True),
                            contact_f1_initial=initial["contact_f1"],contact_f1_final=final["contact_f1"],
                            discrete_energy_kcal=result["relaxation"]["discrete_energy_kcal"],
                            final_physical_energy_kcal=final_energy,
                            sidechain_rmsd_before_relaxation=rmsd_before,
                            sidechain_rmsd_after_relaxation=final["sidechain_rmsd_angstrom"],
                            relaxation_energy_drop=energy_drop,
                            relaxation_energy_down_rmsd_up=bool(energy_drop>1e-6 and final["sidechain_rmsd_angstrom"]>rmsd_before+1e-6))
                        if writer is None:
                            writer=csv.DictWriter(handle,fieldnames=list(row),extrasaction="ignore");writer.writeheader()
                        writer.writerow(row);handle.flush();os.fsync(handle.fileno());rows.append(row)
                    if child_status:
                        raise StructureQualityError("Recovery outputs failed physical acceptance",
                            category="method_output",audit=json.loads((experiment/"structure_quality_summary.json").read_text()))
                except Exception as exc:
                    raise_if_resource_error(exc, stage_hint="recovery seed",
                                            record_path=out/"device_resource_failure.json")
                    failures+=1
                    failure_records.append(dict(seed=seed,error=str(exc),
                        category=getattr(exc,"category","execution"),audit=getattr(exc,"audit",None)))
                    with (out/"failed_cases.log").open("a",encoding="utf-8") as f:
                        f.write(f"seed={seed}\n"+traceback.format_exc()+"\n");f.flush()
        lines=["# Retrospective side-chain perturbation-recovery", "",
            f"Perturbation mode: {args.perturbation_mode}.",
            f"One target; requested seeds={len(args.seeds)}, failed seeds={failures}. Repeated seeds are not independent proteins; no inferential p values.",
            "Positive RMSD gains mean improvement. All methods share the perturbed input and relaxation protocol. Relax-only isolates local minimization without discrete search.",
            ("Native backbone remains fixed; formal multi_chi mode perturbs all defined Active side-chain chis before recovery. "
             "The chi1 mode is a controlled ablation. This is neither de novo prediction nor blind docking."),
            "Generated inputs use the first geometry-qualified perturbation within the fixed attempt cap. Rejected draws and exhausted seeds remain in the ledger; selection never uses energy, reference similarity or solver outcomes. Energy decrease alone is not accuracy.",
            "", "| Method | Successful seeds | Mean gain vs input (A) | Mean gain vs relax-only (A) | Energy-down/RMSD-up cases |", "|---|---:|---:|---:|---:|"]
        for method in ("qaoa","sa","uniform","greedy"):
            group=[r for r in rows if r["method"]==method and
                   r["evaluation_status"]=="passed" and r["control_evaluation_status"]=="passed"]
            if group:
                lines.append(f"| {method} | {len(group)} | {np.mean([r['improvement_vs_input'] for r in group]):.6g} | {np.mean([r['improvement_vs_relax_only'] for r in group]):.6g} | {sum(r['relaxation_energy_down_rmsd_up'] for r in group)} |")
        (out/"recovery_report.md").write_text("\n".join(lines),encoding="utf-8")
        _ablation_atomic_json(out/"recovery_quality_summary.json",dict(
            requested_seeds=len(args.seeds),failed_seeds=failures,failures=failure_records,
            requested_method_outputs=4*len(args.seeds),recorded_method_outputs=len(rows),
            invalid_recorded_outputs=sum(r["evaluation_status"]!="passed" for r in rows),
            rows_retained_including_invalid=True))
        print(out/"recovery_report.md")
    return 1 if failures else 0
