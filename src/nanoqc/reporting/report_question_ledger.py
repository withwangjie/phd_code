"""Research-question ledger: each central question, its evidence, and its verdict.

Every verdict follows one pre-declared rule (RESULTS_CONTRACT.md, "Research-question
ledger"), applied mechanically to this run's own artifacts:

* the stage that produces the evidence did not complete -> ``not established``;
* the multiplicity-adjusted p value is missing (gatekeeping closed, too few
  clusters, not estimable) -> ``not testable``;
* adjusted p < alpha -> ``null rejected`` with the sign of the estimate;
* otherwise -> ``null not rejected``.

Descriptive questions are labelled ``descriptive`` and never receive a verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from nanoqc.reporting.report_common import (
    ReportContext, _filter_qc_rows, _fmt, _formal_statistics_payload, _read_csv_rows, _read_json, stage_ok,
)

DEFAULT_ALPHA = 0.05


def _alpha(ctx: ReportContext) -> float:
    return float((ctx.frozen_config.get("statistics", {}) or {}).get("alpha", DEFAULT_ALPHA))


def verdict(adjusted_p: Optional[float], estimate: Optional[float], alpha: float, *,
            positive: str, negative: str, evidence_complete: bool = True) -> str:
    """The pre-declared verdict rule; never reads anything but its arguments."""
    if not evidence_complete:
        return "not established (evidence stage incomplete)"
    if adjusted_p is None:
        return "not testable (no adjusted p value: gatekeeping closed, too few clusters or not estimable)"
    if float(adjusted_p) < alpha:
        if estimate is None or float(estimate) == 0.0:
            return f"null rejected (adjusted p={_fmt(adjusted_p)} < {alpha}); direction undetermined"
        return (f"null rejected (adjusted p={_fmt(adjusted_p)} < {alpha}): "
                + (positive if float(estimate) > 0 else negative))
    return f"null not rejected (adjusted p={_fmt(adjusted_p)} >= {alpha})"


def _encoding_evidence(ctx: ReportContext) -> Dict[str, Any]:
    coarse_errors: List[float] = []
    for path in sorted((ctx.run_dir / "qc_benchmark" / "cases").glob("*.json")):
        instance = (_read_json(path) or {}).get("quantum_instance") or {}
        value = (instance.get("metadata") or {}).get("ising_energy_equivalence_max_error")
        if value is not None:
            coarse_errors.append(float(value))
    structural = dict(instances=0, max_equivalence_error=None, with_forbidden_states=0,
                      raw_fallback=0, forbidden_selected_outputs=0, selected_outputs=0)
    root = ctx.run_dir / "validation_queue" / "results"
    errors: List[float] = []
    for mapping in sorted(root.glob("*/seed_*/experiment/allatom_mapping.json")):
        metadata = (_read_json(mapping) or {}).get("metadata") or {}
        structural["instances"] += 1
        if metadata.get("all_atom_equivalence_max_error") is not None:
            errors.append(float(metadata["all_atom_equivalence_max_error"]))
        if metadata.get("forbidden_variables") or metadata.get("forbidden_variable_pairs"):
            structural["with_forbidden_states"] += 1
        if metadata.get("no_admissible_assignment_raw_fallback"):
            structural["raw_fallback"] += 1
        for method in ("qaoa", "sa", "uniform", "greedy"):
            result = _read_json(mapping.parent / f"{method}_result.json") or {}
            admissible = (result.get("relaxation") or {}).get("selected_state_geometry_admissible")
            if admissible is not None:
                structural["selected_outputs"] += 1
                structural["forbidden_selected_outputs"] += int(admissible is False)
    if errors:
        structural["max_equivalence_error"] = max(errors)
    return dict(coarse_instances=len(coarse_errors),
                coarse_max_error=max(coarse_errors) if coarse_errors else None,
                structural=structural)


def _cvar_description(ctx: ReportContext) -> str:
    rows = [r for r in _filter_qc_rows(_read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv"),
                                       "matched_outputs")
            if r.get("solver") == "qaoa"]
    parts = []
    for objective in ("mean", "cvar"):
        hits = [float(r["hit"]) for r in rows
                if r.get("qaoa_objective") == objective and r.get("hit") not in (None, "", "None")]
        if hits:
            parts.append(f"{objective}: mean hit fraction {_fmt(sum(hits) / len(hits))} over {len(hits)} rows")
    return "; ".join(parts) if parts else "no QAOA objective-ablation rows recorded"


def section_research_question_ledger(ctx: ReportContext) -> List[str]:
    alpha = _alpha(ctx)
    qc_done = stage_ok(ctx, "qc_benchmark") and stage_ok(ctx, "statistics")
    structure_done = stage_ok(ctx, "structure_experiment") and stage_ok(ctx, "statistics")
    scaling = _read_json(ctx.run_dir / "statistics" / "quantum_scaling_statistics.json") or {}
    amplification = scaling.get("primary_amplification", {}) or {}
    slope = scaling.get("primary", {}) or {}
    structure = _read_json(ctx.run_dir / "statistics" / "structure_statistics.json") or {}
    primary_structure = structure.get("primary", {}) or {}
    rq5 = structure.get("rq5", {}) or {}
    outputs_payload = _formal_statistics_payload(ctx, "outputs") or {}
    secondary = [e for e in outputs_payload.get("effects", []) or []
                 if e.get("gatekeeping_family") == "secondary"]
    encoding = _encoding_evidence(ctx)
    s = encoding["structural"]

    q2 = verdict(amplification.get("p_gatekeeping_adjusted"), amplification.get("mean_log10_amplification"),
                 alpha, evidence_complete=qc_done,
                 positive="QAOA concentrates probability on the ground state beyond uniform feasible sampling",
                 negative="QAOA places less probability on the ground state than uniform feasible sampling")
    q3 = verdict(slope.get("p_gatekeeping_adjusted"), slope.get("mean_slope"), alpha,
                 evidence_complete=qc_done,
                 positive="QAOA's amplification grows with the feasible configuration count over the studied range",
                 negative="QAOA's amplification shrinks as the feasible configuration count grows over the studied range")
    q5_primary = verdict(primary_structure.get("p_holm_confirmatory_family"),
                         primary_structure.get("mean_difference"), alpha, evidence_complete=structure_done,
                         positive="QAOA has the larger value of the primary structural endpoint than SA",
                         negative="QAOA has the smaller value of the primary structural endpoint than SA")
    rq5_not_estimable = str(rq5.get("estimability", "")).startswith("not_estimable_")
    q5_transfer = ("not testable (not estimable under the pre-specified rule)" if rq5_not_estimable else
                   verdict(rq5.get("p_holm_confirmatory_family"), rq5.get("spearman_rho"), alpha,
                           evidence_complete=structure_done,
                           positive="discrete-energy differences and structural differences are positively rank-correlated",
                           negative="discrete-energy differences and structural differences are negatively rank-correlated"))
    primary_family_rejected = all(
        p is not None and float(p) < alpha
        for p in (amplification.get("p_gatekeeping_adjusted"), slope.get("p_gatekeeping_adjusted")))

    lines = ["## R. Research-question ledger (logical closure)", "",
             f"Each central question (README, \"Central research questions\") is answered only from this run's "
             f"artifacts, with the pre-declared verdict rule (alpha={alpha}; RESULTS_CONTRACT.md). A question whose "
             "evidence stage did not complete is `not established`; nothing is carried over from earlier runs.", "",
             "| Question | Evidence | Estimate | Verdict |", "|---|---|---|---|"]
    lines.append(
        "| Q1 Encoding fidelity | qc_benchmark cases; structural `allatom_mapping.json` | "
        f"coarse max QUBO/Ising error {_fmt(encoding['coarse_max_error'])} over {encoding['coarse_instances']} instances; "
        f"structural max Amber/QUBO error {_fmt(s['max_equivalence_error'])} kcal/mol over {s['instances']} instances "
        f"(exact on admissible assignments; {s['with_forbidden_states']} with geometry-forbidden states, "
        f"{s['raw_fallback']} raw fallbacks; A47) | descriptive (each instance is gated by its own fail-closed check) |")
    lines.append(
        f"| Q2 Ground-state amplification (primary family) | statistics/quantum_scaling_statistics.json | "
        f"mean log10 amplification {_fmt(amplification.get('mean_log10_amplification'))} "
        f"[{_fmt(amplification.get('ci_low'))}, {_fmt(amplification.get('ci_high'))}], "
        f"clusters={amplification.get('n_clusters')} | {q2} |")
    lines.append(
        f"| Q3 Scaling slope (primary family) | statistics/quantum_scaling_statistics.json | "
        f"mean slope {_fmt(slope.get('mean_slope'))} [{_fmt(slope.get('ci_low'))}, {_fmt(slope.get('ci_high'))}], "
        f"clusters={slope.get('n_clusters')} | {q3} |")
    if secondary:
        for effect in secondary:
            lines.append(
                f"| Q2 QAOA vs {effect.get('baseline')} ({effect.get('metric')}; secondary family) | "
                f"qc_benchmark/statistics_outputs.json | mean difference {_fmt(effect.get('mean_difference'))}, "
                f"clusters={effect.get('n_clusters')} | "
                + verdict(effect.get("p_gatekeeping_adjusted"), effect.get("mean_difference"), alpha,
                          evidence_complete=qc_done, positive="QAOA higher", negative="QAOA lower")
                + " |")
    else:
        lines.append("| Q2 QAOA vs classical (secondary family) | qc_benchmark/statistics_outputs.json | "
                     "no secondary effects recorded | not established |")
    lines.append(f"| Q4 CVaR vs mean objective | qc_benchmark/metrics.csv | {_cvar_description(ctx)} | "
                 "descriptive (preregistered ablation; no hypothesis test) |")
    lines.append(
        f"| Q5 Structural endpoint, QAOA vs SA | statistics/structure_statistics.json | "
        f"{primary_structure.get('endpoint')}: mean difference {_fmt(primary_structure.get('mean_difference'))} "
        f"[{_fmt(primary_structure.get('ci_low'))}, {_fmt(primary_structure.get('ci_high'))}], "
        f"clusters={primary_structure.get('n_clusters')} | {q5_primary} |")
    lines.append(
        f"| Q5 Energy-to-structure transfer (RQ5) | statistics/structure_statistics.json | "
        f"Spearman rho {_fmt(rq5.get('spearman_rho'))} [{_fmt(rq5.get('ci_low'))}, {_fmt(rq5.get('ci_high'))}] | "
        f"{q5_transfer} |")
    lines.append("| Q6 Problem reduction (EGNN selection, rotamer coverage) | checkpoints/training_summary.json; "
                 "section 6 | see section 6 | descriptive (enabling component, not a confirmatory endpoint) |")
    lines.append("")

    lines.append("### R.1 How the answers connect")
    lines.append("")
    lines.append(
        "- Q1 is a precondition: every coarse and structural instance passed its own fail-closed QUBO/Ising and "
        "energy-equivalence check, or it would have failed its case. Structural instances are exact on "
        "geometry-admissible assignments; "
        f"{s['forbidden_selected_outputs']}/{s['selected_outputs']} solver selections contained a "
        "geometry-forbidden state and were judged by physical acceptance only (A47, A51).")
    if not qc_done:
        lines.append("- Q2/Q3 are not established in this run, so no statement about QAOA's algorithmic behaviour "
                     "and no QAOA-vs-classical comparison can be made.")
    elif primary_family_rejected:
        lines.append("- Both primary quantum-intrinsic hypotheses (Q2 amplification, Q3 slope) were rejected, so the "
                     "secondary QAOA-vs-classical comparisons were tested under serial gatekeeping; their verdicts "
                     "above are confirmatory at the family-wise level.")
    else:
        lines.append("- At least one primary quantum-intrinsic hypothesis was not rejected, so serial gatekeeping "
                     "keeps every QAOA-vs-classical comparison closed: those differences are descriptive only and "
                     "support no claim in either direction.")
    if not structure_done:
        lines.append("- Q5 is not established: the structural stage or statistics did not complete, so nothing "
                     "is concluded about whether discrete-energy gains reach all-atom structure.")
    else:
        lines.append(
            "- Q5 asks whether solver-level discrete energy gains reach all-atom structure after reconstruction and "
            "identical relaxation. Its verdicts above stand on their own (structural Holm family); a coarse "
            "search result never substitutes for them, because the coarse surrogate is not calibrated to Amber "
            "(training-only diagnostic, A23/A48) and the structural QUBO is a separate all-atom construction.")
    lines.append(
        "- Scope of every verdict: fixed backbone and known pose; noiseless exact-subspace simulation, so no "
        "hardware or speed claim; finite-range scaling; the structural energy is Amber ff14SB in vacuum with "
        "the A50 polar-hydrogen modification; QAOA parameter scaling is instance-dependent where forbidden "
        "states exist (A51). Failed cases remain in every denominator.")
    lines.append("")
    return lines
