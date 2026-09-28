"""Final-report sections on pruning, the quantum problem, the QAOA protocol and search performance.

Split out of generate_final_research_report.py, which re-exports every name here.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from scipy.stats import spearmanr
from nanoqc.reporting.report_common import ReportContext, _filter_qc_rows, _fmt, _formal_statistics_payload, _paired_inference_multiplicity_note, _primary_active_sites, _primary_pruning, _quantum_primary, _read_csv_rows, _read_json, stage_ok



def training_coarse_atomistic_rank_diagnostic(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Describe within-complex raw-energy ranks without claiming validation.

    The calibration CSV maps each coarse bit assignment to the same full-chi
    atomistic assignment. Its energies are unrelaxed, training-only values.
    Constant or nonfinite groups are counted rather than silently omitted.
    """
    by_pdb: Dict[str, List[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Coarse/atomistic rank diagnostic requires training rows only")
        try:
            coarse = sum(float(row[key]) for key in (
                "prior_energy", "vhh_environment_energy", "antigen_energy", "pair_energy"))
            atomistic = float(row["amber_delta_kcal"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Incomplete coarse/atomistic calibration row") from exc
        if not math.isfinite(coarse) or not math.isfinite(atomistic):
            raise ValueError("Nonfinite coarse/atomistic calibration energy")
        by_pdb[str(row["pdb_id"]).lower()].append((coarse, atomistic))
    per_pdb = []
    for pdb, values in sorted(by_pdb.items()):
        coarse = np.asarray([v[0] for v in values], dtype=float)
        atomistic = np.asarray([v[1] for v in values], dtype=float)
        estimable = len(values) >= 3 and len(set(coarse)) > 1 and len(set(atomistic)) > 1
        rho = float(spearmanr(coarse, atomistic).statistic) if estimable else None
        if rho is not None and not math.isfinite(rho):
            rho = None
        per_pdb.append(dict(pdb_id=pdb, assignments=len(values), spearman_rho=rho,
                            max_abs_amber_delta_kcal=float(np.max(np.abs(atomistic)))))
    estimable_rhos = [r["spearman_rho"] for r in per_pdb if r["spearman_rho"] is not None]
    return dict(scope="training_only_unrelaxed_same_assignment",
                n_pdb=len(per_pdb), n_estimable=len(estimable_rhos),
                n_nonestimable=len(per_pdb)-len(estimable_rhos),
                median_within_pdb_spearman=(float(np.median(estimable_rhos)) if estimable_rhos else None),
                per_pdb=per_pdb)


# ---------------------------------------------------------------------------
# Section 2: pruning contribution
# ---------------------------------------------------------------------------

def section_pruning_contribution(ctx: ReportContext) -> List[str]:
    lines = ["## 6. Quantum problem reduction: EGNN pruning contribution", ""]
    checkpoint_dir = ctx.run_dir / str((ctx.frozen_config.get("paths", {}) or {}).get("checkpoint_dir", "checkpoints"))
    geometry_path = checkpoint_dir / "geometry_baseline.json"
    training_summary = _read_json(checkpoint_dir / "training_summary.json") or {}
    geometry = _read_json(geometry_path) or {}
    if geometry:
        lines.append("### 6.1 Leakage-controlled node-classification validation")
        lines.append("")
        lines.append("| Model | Validation ROC-AUC | Validation PR-AUC |")
        lines.append("|---|---:|---:|")
        egnn_metrics = training_summary.get("best_validation", {}) or {}
        if egnn_metrics:
            lines.append(
                f"| EGNN | {_fmt(egnn_metrics.get('roc_auc'))} | {_fmt(egnn_metrics.get('pr_auc'))} |"
            )
        lines.append(
            f"| Train-only geometry logistic | {_fmt(geometry.get('validation_roc_auc'))} | "
            f"{_fmt(geometry.get('validation_pr_auc'))} |"
        )
        lines.append("")
        lines.append(
            "The EGNN backbone follows the E(n)-equivariant formulation [R1]. "
        "The geometry logistic baseline is fitted only on training graphs from simple residue/geometry "
            "features and evaluated on the same homology-isolated validation fold. Its purpose is to test "
            "whether EGNN performance exceeds a low-capacity geometry shortcut rather than merely distance."
        )
        lines.append("")
    lines.append("### 6.2 Downstream pruning ablation")
    lines.append("")
    if not stage_ok(ctx, "qc_benchmark"):
        lines += ["qc_benchmark stage did not complete; no pruning-ablation numbers are reported.", ""]
        return lines
    all_rows = _read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv")
    rows = _filter_qc_rows(all_rows, "matched_outputs")
    primary_sites=_primary_active_sites(ctx)
    rows=[r for r in rows if int(float(r.get("active_sites",primary_sites)))==primary_sites]
    if not rows:
        lines += ["No matched-output rows were found in `qc_benchmark/metrics.csv`.", ""]
        return lines
    prunings = sorted({r.get("pruning", "") for r in rows if r.get("pruning")})
    lines.append(f"Five-way pruning-strategy comparison at the frozen primary size ({primary_sites} active sites), "
                  "sharing the same perturbed input, site count, and "
                  "evaluation region (radius/depth) within each case; only the pruning STRATEGY differs "
                  "between compared rows. Downstream search performance (hit fraction, energy gap), not just "
                  "input-graph AUC, is what is compared here.")
    lines.append("")
    lines.append("| Pruning | Rows | Mean hit fraction | Mean gap | Mean low-energy coverage | Mean recorded solver cost |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for pruning in prunings:
        group = [r for r in rows if r.get("pruning") == pruning]

        def mean(field: str) -> Optional[float]:
            values = [float(r[field]) for r in group if r.get(field) not in (None, "", "None")]
            return sum(values) / len(values) if values else None

        lines.append(f"| {pruning} | {len(group)} | {_fmt(mean('hit'))} | {_fmt(mean('gap'))} | "
                      f"{_fmt(mean('low_energy_coverage'))} | {_fmt(mean('solver_seconds'))} |")
    lines.append("")
    lines.append("Candidate sets differ across pruning methods, so gaps compare solver quality within each "
                  "instance's own candidate space, not a biological superiority claim about one pruning method "
                  "over another purely from this table. QAOA solver_seconds is standalone-equivalent "
                  "(measured optimization + sampling) because optimization is cached across output budgets.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Sections 1-3: quantum encoding, frozen protocol, and search/scaling performance
# ---------------------------------------------------------------------------

def section_quantum_problem_encoding(ctx: ReportContext) -> List[str]:
    lines=["## 1. Quantum problem encoding",""]
    if not stage_ok(ctx,"qc_benchmark"):
        lines += ["qc_benchmark did not complete; no formal quantum-instance summary is reported.",""]
        return lines
    case_paths=sorted((ctx.run_dir/"qc_benchmark"/"cases").glob("*.json"))
    payloads=[]
    for path in case_paths:
        data=_read_json(path) or {}
        instance=data.get("quantum_instance")
        if isinstance(instance,dict) and instance.get("schema")=="quantum_optimization_instance_v1":
            payloads.append(instance)
    if not payloads:
        lines += ["No completed self-contained quantum-instance artifacts were found.",""]
        return lines
    qubits=[int(p.get("num_qubits",0) or 0) for p in payloads]
    configs=[int(p.get("feasible_configuration_count",0) or 0) for p in payloads]
    lines += [
        f"- Self-contained quantum instances: {len(payloads)} case artifacts.",
        f"- Encoding: one-hot rotamer registers; logical qubits range {min(qubits)}--{max(qubits)}.",
        f"- Feasible-state counts range {min(configs)}--{max(configs)}.",
        "- Each case stores the full penalized QUBO/Ising representation and the penalty-free physical "
        "Hamiltonian terms used by the feasibility-preserving XY-QAOA solver.",
        "- Exact feasible enumeration is retained only as a retrospective oracle, not as part of the proposed solver.",
        "",
    ]
    return lines


def section_quantum_protocol(ctx: ReportContext) -> List[str]:
    lines=["## 2. Frozen QAOA protocol and logical resources",""]
    qp=ctx.frozen_config.get("quantum_protocol",{}) or {}
    primary=qp.get("primary",{}) or {}
    ablation=qp.get("benchmark_ablation",{}) or {}
    sensitivity=qp.get("development_sensitivity",{}) or {}
    lines += [
        f"- Algorithm={qp.get('algorithm')}; encoding={qp.get('encoding')}; mixer={qp.get('mixer')}; initial_state={qp.get('initial_state')}.",
        f"- Primary protocol: p={primary.get('depth')}, max_evals={primary.get('max_evals')}, "
        f"objective={primary.get('objective')}, CVaR alpha={primary.get('cvar_alpha')}, "
        f"restarts={primary.get('restarts')}, eval_shots={primary.get('eval_shots')}, "
        f"output_shots={primary.get('output_shots')}.",
        f"- Predeclared benchmark ablation: objectives={ablation.get('objectives')}, restarts={ablation.get('restarts')}.",
        f"- Development-only sensitivity: p={sensitivity.get('depths')}, max_evals={sensitivity.get('max_evals')}, "
        f"eval_shots={sensitivity.get('eval_shots')}, CVaR alpha={sensitivity.get('cvar_alpha')}.",
        "- QAOA follows Farhi et al. [R7]; feasibility-preserving alternating-operator mixers follow "
        "Hadfield et al. [R28]; finite-shot CVaR follows Barkoutsos et al. [R8].",
        "- Logical qubit and gate counts are pre-transpilation algorithmic resources, not hardware-native "
        "counts; reporting solution quality together with resource use follows quantum-optimization benchmarking guidance [R20,R29].",
        "",
    ]
    return lines


def section_search_performance(ctx: ReportContext) -> List[str]:
    lines = ["## 3. Quantum-classical search performance and scaling", ""]
    if not stage_ok(ctx, "qc_benchmark"):
        lines += ["qc_benchmark stage did not complete; no search-performance numbers are reported.", ""]
        return lines
    all_rows = _read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv")
    if not all_rows:
        lines += ["`qc_benchmark/metrics.csv` is empty or missing.", ""]
        return lines
    primary_sites=_primary_active_sites(ctx)
    primary_pruning=_primary_pruning(ctx)
    stats_cfg=ctx.frozen_config.get("statistics",{}) or {}
    qprimary=_quantum_primary(ctx)
    primary_radius=float(stats_cfg.get("primary_radius",6.0))
    primary_depth=int(qprimary.get("depth",2))
    primary_max_evals=int(qprimary.get("max_evals",90))
    rows=[
        r for r in _filter_qc_rows(all_rows,"matched_outputs")
        if int(float(r.get("active_sites",primary_sites)))==primary_sites
        and str(r.get("pruning",""))==primary_pruning
        and abs(float(r.get("radius",primary_radius))-primary_radius)<=1e-12
        and int(float(r.get("depth",primary_depth)))==primary_depth
        and int(float(r.get("max_evals",primary_max_evals)))==primary_max_evals
    ]

    calibration_cfg=((ctx.frozen_config.get("qc_benchmark",{}) or {}).get("energy_calibration",{}) or {})
    calibration_raw=calibration_cfg.get("calibration_file","calibration/coarse_to_amber.json")
    calibration_path=Path(calibration_raw)
    if not calibration_path.is_absolute():
        calibration_path=ctx.run_dir/calibration_path
    calibration=_read_json(calibration_path) or {}
    lines.append("### 3.0 Training-only coarse-to-Amber calibration")
    lines.append("")
    diagnostic=calibration_cfg.get("mode","frozen")=="diagnostic"
    assessment=_read_json(ctx.run_dir/"calibration"/"diagnostic_assessment.json") or {}
    if diagnostic and assessment.get("fit_status")!="completed":
        calibration={}
    if diagnostic:
        lines.append(
            "- Protocol amendment: Amber calibration is a reported diagnostic and its fitted "
            "coefficients are **not** used by any solver benchmark. Solver energies refer to "
            "the frozen coarse-grained surrogate; atomistic structural outcomes are evaluated separately."
        )
        lines.append(
            f"- Amber fit status: {assessment.get('fit_status', 'unavailable')}; "
            f"historical acceptance: {'passed' if assessment.get('accepted') is True else 'failed'}; "
            f"reasons={assessment.get('failure_reasons', [])}."
        )
    if calibration:
        lines.append(
            f"- Training complexes={calibration.get('n_train_complexes')}; samples={calibration.get('n_train_samples')}; "
            f"CV RMSE={_fmt(calibration.get('cv_rmse_kcal'))} kcal/mol; "
            f"CV MAE={_fmt(calibration.get('cv_mae_kcal'))}; "
            f"CV R²={_fmt(calibration.get('cv_r2'))}; "
            f"CV Spearman={_fmt(calibration.get('cv_spearman'))}."
        )
        lines.append(
            f"- Uncalibrated RMSE={_fmt(calibration.get('uncalibrated_rmse_kcal'))}; "
            f"uncalibrated Spearman={_fmt(calibration.get('uncalibrated_spearman'))}; "
            f"RMSE improvement={_fmt(calibration.get('calibration_rmse_improvement_kcal'))} kcal/mol."
        )
        if not diagnostic:
            lines.append(
                "- Side-chain state construction follows the backbone-dependent Dunbrack rotamer framework [R2]. "
                "Calibration uses training complexes only, PDB-grouped cross-validation, nonnegative component "
                "weights, and must satisfy the frozen acceptance thresholds before the formal benchmark can run."
            )
    else:
        lines.append("No successful Amber diagnostic fit was produced; see the assessment and stage log."
                     if diagnostic else "Frozen calibration artifact is missing or unreadable.")
    lines.append("")
    if diagnostic:
        training_csv=Path(calibration_cfg.get("training_csv","calibration/coarse_to_amber_train.csv"))
        if not training_csv.is_absolute():
            training_csv=ctx.run_dir/training_csv
        training_rows=_read_csv_rows(training_csv)
        lines.append("#### Training-only coarse versus atomistic rank diagnostic")
        lines.append("")
        if training_rows and all("amber_delta_kcal" in row for row in training_rows):
            ranking=training_coarse_atomistic_rank_diagnostic(training_rows)
            lines.append(
                f"The same full-chi assignment was scored by the uncalibrated coarse surrogate and "
                f"unrelaxed Amber14 on {ranking['n_pdb']} training complexes. Within-PDB Spearman rank "
                f"correlation is estimable for {ranking['n_estimable']}; "
                f"median rho={_fmt(ranking['median_within_pdb_spearman'])}. "
                f"It is undefined for {ranking['n_nonestimable']} complexes."
            )
            lines.append("| Training PDB | Assignments | Within-PDB Spearman rho | Largest absolute raw Amber delta (kcal/mol) |")
            lines.append("|---|---:|---:|---:|")
            for item in ranking["per_pdb"]:
                lines.append(f"| {item['pdb_id']} | {item['assignments']} | "
                             f"{_fmt(item['spearman_rho'])} | {_fmt(item['max_abs_amber_delta_kcal'])} |")
            lines.append("")
        else:
            lines.append("No complete training assignment pairs were available for a rank diagnostic.")
        lines.append("This diagnostic is training-only and uses raw, unrelaxed energies. Extreme overlaps can "
                     "dominate Amber values. It does not validate coarse energy on held-out structures, "
                     "establish cross-stage rotamer equality, or show that a solver gain improves RMSD.")
        lines.append("")

    lines.append("### 3.1 Output-budget curve (equal OUTPUT count across solvers; not equal total compute)")
    lines.append("")
    outputs_values = sorted({r.get("outputs", "") for r in rows if r.get("outputs")}, key=lambda v: float(v))
    lines.append("| Outputs | Solver | Rows | Mean hit fraction | Mean gap | Mean single-state energy queries |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for outputs in outputs_values:
        for solver in sorted({r.get("solver", "") for r in rows if r.get("outputs") == outputs}):
            group = [r for r in rows if r.get("outputs") == outputs and r.get("solver") == solver]

            def mean(field: str) -> Optional[float]:
                values = [float(r[field]) for r in group if r.get(field) not in (None, "", "None")]
                return sum(values) / len(values) if values else None

            lines.append(f"| {outputs} | {solver} | {len(group)} | {_fmt(mean('hit'))} | {_fmt(mean('gap'))} | "
                          f"{_fmt(mean('single_state_energy_queries'))} |")
    lines.append("")

    lines.append("### 3.2 Active-site scaling analysis")
    lines.append("")
    primary_outputs=int(qprimary.get("output_shots",1000))
    primary_objective=str(qprimary.get("objective","cvar"))
    primary_restarts=int(qprimary.get("restarts",4))
    scaling_rows=[
        r for r in _filter_qc_rows(all_rows,"matched_outputs")
        if int(float(r.get("outputs",0) or 0))==primary_outputs
        and str(r.get("pruning",""))==primary_pruning
        and abs(float(r.get("radius",primary_radius))-primary_radius)<=1e-12
        and int(float(r.get("depth",primary_depth)))==primary_depth
        and int(float(r.get("max_evals",primary_max_evals)))==primary_max_evals
    ]
    site_values=sorted({
        int(float(r.get("active_sites")))
        for r in scaling_rows if r.get("active_sites") not in (None,"","None")
    })
    lines.append(
        f"Descriptive scaling on pruning={primary_pruning}, radius={primary_radius}, p={primary_depth}, "
        f"max_evals={primary_max_evals}, outputs={primary_outputs}. QAOA uses the frozen primary "
        f"objective={primary_objective}, restarts={primary_restarts}; classical solvers use their matched-output rows."
    )
    lines.append("")
    lines.append("| Active sites | Solver | Rows | Mean logical qubits | Mean 2q gates | Mean feasible configurations | Mean hit | Mean gap |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for sites in site_values:
        for solver in ("qaoa","sa","uniform","greedy"):
            group=[
                r for r in scaling_rows
                if int(float(r.get("active_sites",0) or 0))==sites and r.get("solver")==solver
                and (solver!="qaoa" or (
                    r.get("qaoa_objective")==primary_objective
                    and int(float(r.get("qaoa_restarts",0) or 0))==primary_restarts
                ))
            ]
            if not group:
                continue
            def scale_mean(field: str) -> Optional[float]:
                values=[float(r[field]) for r in group if r.get(field) not in (None,"","None")]
                return sum(values)/len(values) if values else None
            lines.append(
                f"| {sites} | {solver} | {len(group)} | {_fmt(scale_mean('num_bits'))} | "
                f"{_fmt(scale_mean('qaoa_two_qubit_gates') if solver=='qaoa' else None)} | "
                f"{_fmt(scale_mean('configuration_count'))} | {_fmt(scale_mean('hit'))} | "
                f"{_fmt(scale_mean('gap'))} |"
            )
    lines.append("")
    lines.append(
        f"The {primary_sites}-site condition is the pre-registered confirmatory size. "
        "The use of scaling analysis is motivated by the combinatorial nature of fixed-backbone rotamer search [R10-R12]. "
        "The other site counts are pre-declared scaling conditions used to assess how relative "
        "quantum-classical performance changes with QUBO/problem size; they are not pooled into the primary test."
    )
    exploration_md=ctx.run_dir/"quantum_exploration"/"summary.md"
    if exploration_md.is_file():
        lines.append("### Exploratory QAOA analyses (depth, optimizer budget, parameter transfer)")
        lines.append("")
        lines.extend(line for line in exploration_md.read_text(encoding="utf-8").splitlines()[2:])
        lines.append("")
    lines.append("")
    scaling_stats=_read_json(ctx.run_dir/"statistics"/"quantum_scaling_statistics.json") or {}
    if scaling_stats:
        amplification=scaling_stats.get("primary_amplification",{}) or {}
        if amplification:
            lines.append(
                f"Primary quantum-intrinsic endpoint: at {amplification.get('active_sites')} sites the mean "
                f"log10 exact ground-state amplification of QAOA over uniform feasible sampling is "
                f"{_fmt(amplification.get('mean_log10_amplification'))} "
                f"(95% cluster-bootstrap CI [{_fmt(amplification.get('ci_low'))}, {_fmt(amplification.get('ci_high'))}]; "
                f"clusters={amplification.get('n_clusters')}; raw sign-flip p={_fmt(amplification.get('p_value'))}; "
                f"gatekeeping-adjusted p={_fmt(amplification.get('p_gatekeeping_adjusted'))}). "
                "0 means no concentration beyond random sampling; values are noiseless-simulator properties.")
            lines.append("")
        primary_scaling=scaling_stats.get("primary",{}) or {}
        lines.append(
            f"Formal scaling inference uses `{scaling_stats.get('primary_predictor')}` as the primary "
            f"complexity axis and `{scaling_stats.get('primary_response')}` as the response. "
            f"Independent clusters={primary_scaling.get('n_clusters')}; "
            f"mean within-PDB/cluster slope={_fmt(primary_scaling.get('mean_slope'))}; "
            f"95% cluster-bootstrap CI=[{_fmt(primary_scaling.get('ci_low'))}, "
            f"{_fmt(primary_scaling.get('ci_high'))}]; "
            f"raw sign-flip p={_fmt(primary_scaling.get('p_value'))}; "
            f"gatekeeping-adjusted p (primary family)={_fmt(primary_scaling.get('p_gatekeeping_adjusted'))}."
        )
        lines.append(
            "Positive slope means QAOA concentrates relatively more probability on the ground state "
            "(relative to uniform sampling) as the feasible configuration space grows. Every scaling size uses three states "
            "per site, one from each chi1 well; the analysis rejects cases lacking this policy. "
            "QAOA depth and optimizer evaluations stay fixed. Added sites also change which residues and "
            "energy landscape are studied, so this is a finite-range, fixed-budget trend, not an "
            "asymptotic complexity exponent or equal-compute comparison. Exact enumeration remains feasible "
            "through 10 sites (59,049 assignments). This simulator-level analysis does not establish "
            "hardware quantum speedup."
        )
        lines.append("")
    elif stage_ok(ctx,"statistics"):
        lines.append("Formal scaling statistics are missing or unreadable despite a completed statistics stage.")
        lines.append("")

    lines.append("### 3.3 QAOA objective/restart ablation at the primary size (mean vs. CVaR, single- vs. multi-start)")
    lines.append("")
    qaoa_rows = [r for r in rows if r.get("solver") == "qaoa"]
    combos = sorted({(r.get("qaoa_objective", ""), r.get("qaoa_restarts", "")) for r in qaoa_rows})
    lines.append("| Objective | Restarts | Rows | Mean hit fraction | Mean gap | Termination reasons |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for objective, restarts in combos:
        group = [r for r in qaoa_rows if r.get("qaoa_objective") == objective and r.get("qaoa_restarts") == restarts]

        def mean(field: str) -> Optional[float]:
            values = [float(r[field]) for r in group if r.get(field) not in (None, "", "None")]
            return sum(values) / len(values) if values else None

        reasons: Dict[str, int] = {}
        for r in group:
            reason = r.get("termination_reason") or r.get("optimizer_success")
            reasons[str(reason)] = reasons.get(str(reason), 0) + 1
        lines.append(f"| {objective or 'n/a'} | {restarts or 'n/a'} | {len(group)} | {_fmt(mean('hit'))} | "
                      f"{_fmt(mean('gap'))} | {reasons} |")
    lines.append("")
    lines.append("A run whose termination reason is `max_evaluations_reached` (budget exhausted) is never "
                  "reported above as `converged`; the raw `termination_reason` distribution is shown as-is.")
    lines.append("")

    lines.append("### 3.4 Development-only QAOA hyperparameter sensitivity")
    lines.append("")
    sensitivity_rows=_read_csv_rows(ctx.run_dir/"method_sensitivity"/"sensitivity_summary.csv")
    if sensitivity_rows:
        lines.append(
            "This table is development-only and preserves the depth, evaluation-budget, measurement-shot, "
            "and CVaR-alpha axes separately; validation/test results are not used to choose these settings."
        )
        lines.append("")
        lines.append("| Sites | p | Max evals | Eval shots | CVaR alpha | Rows | Mean hit | Mean gap |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in sensitivity_rows:
            lines.append(
                f"| {row.get('active_sites')} | {row.get('depth')} | {row.get('max_evals')} | "
                f"{row.get('eval_shots')} | {row.get('cvar_alpha')} | {row.get('rows')} | "
                f"{_fmt(row.get('mean_hit'))} | {_fmt(row.get('mean_gap'))} |"
            )
    else:
        lines.append("No completed development-only sensitivity aggregate was found.")
    lines.append("")

    lines.append("### 3.5 Frozen primary paired inference")
    lines.append("")
    if not stage_ok(ctx, "statistics"):
        lines.append("Statistics stage did not complete; no formal paired inference is reported.")
        lines.append("")
        return lines
    for mode in ("outputs","time"):
        payload=_formal_statistics_payload(ctx,mode)
        label="Matched outputs" if mode=="outputs" else "Matched time"
        lines.append(f"#### {label}")
        lines.append("")
        if not payload:
            lines.append(f"`statistics_{mode}.json` is missing or unreadable.")
            lines.append("")
            continue
        lines.append(
            f"Frozen primary contrast: pruning={payload.get('primary_pruning')}, radius={payload.get('primary_radius')}, "
            f"p={payload.get('primary_depth')}, max_evals={payload.get('primary_max_evals')}, "
            f"active_sites={payload.get('primary_active_sites')}, "
            f"outputs={payload.get('primary_outputs')}, objective={payload.get('primary_objective')}, "
            f"restarts={payload.get('primary_restarts')}; "
            f"cluster unit: {payload.get('cluster_unit','n/a')}.")
        lines.append("")
        lines.append("| Baseline | Metric | Family | Clusters | Mean QAOA-classical difference | 95% CI | p | Holm p (all effects, descriptive) | Gatekeeping-adjusted p |")
        lines.append("|---|---|---|---:|---:|---|---:|---:|---:|")
        effects=payload.get("effects") or []
        if effects:
            for effect in effects:
                lines.append(
                    f"| {effect.get('baseline')} | {effect.get('metric')} | {effect.get('gatekeeping_family','n/a')} | {effect.get('n_clusters',0)} | "
                    f"{_fmt(effect.get('mean_difference'))} | "
                    f"{_fmt(effect.get('ci_low'))}, {_fmt(effect.get('ci_high'))} | "
                    f"{_fmt(effect.get('p_value'))} | {_fmt(effect.get('p_holm'))} | "
                    f"{_fmt(effect.get('p_gatekeeping_adjusted'))} |")
        else:
            lines.append("| — | — | — | 0 | n/a | n/a | n/a | n/a | n/a |")
        lines.append("")
        lines.append(_paired_inference_multiplicity_note(mode))
        lines.append("")
        lines.append(f"Paired-case counts: `{json.dumps(payload.get('paired_cases',{}),sort_keys=True)}`.")
        lines.append(f"Exclusions: `{json.dumps(payload.get('exclusions',{}),sort_keys=True)}`.")
        if mode=="time":
            lines.append(
                "Matched-time results use only explicitly recorded time-budget controls passing "
                "the frozen overrun rule; they are reported separately from matched-output inference.")
        lines.append("")
    return lines
