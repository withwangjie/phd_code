"""Final-report sections on structural recovery, external baselines and robustness.

Split out of generate_final_research_report.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from nanoqc.reporting.report_common import ReportContext, _fmt, _read_csv_rows, _read_json, _read_text, stage_ok



# ---------------------------------------------------------------------------
# Section 4: structural benefit (energy-down vs RMSD-up; paired target-level)
# ---------------------------------------------------------------------------

def _load_recovery_rows(directory: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not directory.is_dir():
        return rows
    combined = directory / "real_complex_metrics.csv"
    if combined.is_file():
        return _read_csv_rows(combined)
    for sub in directory.iterdir():
        candidate = sub / "recovery_metrics.csv"
        if candidate.is_file():
            rows.extend(_read_csv_rows(candidate))
        nested = sub / "results" / sub.name / "recovery_metrics.csv"
        if nested.is_file():
            rows.extend(_read_csv_rows(nested))
    return rows


def _target_level_descriptive_difference(
    rows: List[Dict[str, str]], metric: str, baseline: str
) -> Dict[str, Any]:
    by_target: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        target=row.get("target","")
        method=row.get("method","")
        value=row.get(metric)
        if value in (None,"","None"):
            continue
        by_target.setdefault(target,{}).setdefault(method,[]).append(float(value))
    differences=[]
    for target,methods in sorted(by_target.items()):
        if "qaoa" not in methods or baseline not in methods:
            continue
        differences.append(
            sum(methods["qaoa"])/len(methods["qaoa"])
            - sum(methods[baseline])/len(methods[baseline])
        )
    return dict(
        n_targets=len(differences),
        mean_difference=(None if not differences else sum(differences)/len(differences)),
    )


def section_structural_benefit(ctx: ReportContext) -> List[str]:
    lines = ["## 4. Structure reconstruction, relaxation, and the energy-structure relationship "
             "(does a lower-energy conformer mean a more accurate structure?)", ""]
    if not stage_ok(ctx, "structure_experiment"):
        lines += ["structure_experiment stage did not complete; no structural-benefit numbers are reported.", ""]
        return lines

    structure_stats = _read_json(ctx.run_dir / "statistics" / "structure_statistics.json") or {}
    if structure_stats:
        primary=structure_stats.get("primary",{}) or {}
        rq5=structure_stats.get("rq5",{}) or {}
        lines.append("### 4.0 Pre-registered confirmatory structural analysis")
        lines.append("")
        lines.append(
            f"- Primary endpoint: {primary.get('endpoint')}; contrast: {primary.get('contrast')}; "
            f"clusters={primary.get('n_clusters')}; mean difference={_fmt(primary.get('mean_difference'))}; "
            f"95% cluster-bootstrap CI=[{_fmt(primary.get('ci_low'))}, {_fmt(primary.get('ci_high'))}]; "
            f"raw sign-flip p={_fmt(primary.get('p_value'))}; "
            f"Holm-adjusted p={_fmt(primary.get('p_holm_confirmatory_family'))}."
        )
        lines.append(
            f"- RQ5 energy-to-structure transfer: Spearman rho={_fmt(rq5.get('spearman_rho'))}; "
            f"95% cluster-bootstrap CI=[{_fmt(rq5.get('ci_low'))}, {_fmt(rq5.get('ci_high'))}]; "
            f"raw cluster-aware permutation p={_fmt(rq5.get('p_value'))}; "
            f"Holm-adjusted p={_fmt(rq5.get('p_holm_confirmatory_family'))}; "
            f"estimability={rq5.get('estimability', 'estimable')}."
        )
        if str(rq5.get("estimability", "")).startswith("not_estimable_"):
            lines.append(
                "- RQ5 is not estimable under the pre-specified rule (a cluster-level difference is constant; "
                f"mean delta energy={_fmt(rq5.get('mean_delta_energy'))}, "
                f"mean delta RMSD={_fmt(rq5.get('mean_delta_rmsd'))}). It is reported descriptively and is "
                "excluded from the Holm family; it is neither a positive nor a negative finding."
            )
        lines.append(
            "- These are the only inferential structural results. Additional method/metric tables below "
            "are descriptive and are not separate hypothesis tests."
        )
        lines.append("")

    dev_enabled=bool(((ctx.frozen_config.get("queue_freeze",{}) or {}).get("dev_queue",{}) or {}).get("enabled",True))
    validation_rows = _load_recovery_rows(ctx.run_dir / "validation_queue")
    queues=[("4.2 Frozen, blind validation queue (requirement #6)",validation_rows)]
    if dev_enabled:
        queues.insert(0,("4.1 Historical development queue (NOT a confirmatory test)",
                         _load_recovery_rows(ctx.run_dir/"dev_queue")))
    for label, rows in queues:
        lines.append(f"### {label}")
        lines.append("")
        if not rows:
            lines.append("No recovery_metrics.csv rows found for this queue.")
            lines.append("")
            continue
        targets = sorted({r.get("target", "") for r in rows})
        energy_down_rmsd_up = sum(1 for r in rows if str(r.get("relaxation_energy_down_rmsd_up")).lower() == "true")
        lines.append(f"- Targets with at least one recorded row: {len(targets)}. Total method x seed rows: {len(rows)}.")
        lines.append(f"- Rows where relaxation lowered energy but worsened side-chain RMSD "
                      f"(`relaxation_energy_down_rmsd_up`): {energy_down_rmsd_up}/{len(rows)}.")
        lines.append("")
        lines.append("Descriptive target-level QAOA-vs-classical differences "
                     "(repeats averaged within target first; no additional p values):")
        lines.append("")
        lines.append("| Baseline | Metric | Targets paired | Mean difference |")
        lines.append("|---|---|---:|---:|")
        for baseline in ("sa", "uniform", "greedy"):
            for metric in ("improvement_vs_input", "final_rmsd"):
                effect=_target_level_descriptive_difference(rows,metric,baseline)
                lines.append(
                    f"| {baseline} | {metric} | {effect.get('n_targets',0)} | "
                    f"{_fmt(effect.get('mean_difference'))} |"
                )
        lines.append("")

    lines += _structural_protocol_execution(ctx.run_dir / "validation_queue", ctx)

    # Failure denominators, preserved explicitly rather than dropped.
    for label, directory in (("dev queue", ctx.run_dir / "dev_queue"), ("validation queue", ctx.run_dir / "validation_queue")):
        if label=="dev queue" and not dev_enabled:continue
        failures_log = directory / "failures.log"
        if failures_log.is_file():
            failed = [line for line in _read_text(failures_log).splitlines() if line and not line.startswith(" ")]
            lines.append(f"- {label}: `failures.log` present, {len(failed)} logged failure line(s) preserved verbatim "
                          f"(see `{failures_log}`).")
    lines.append("")
    return lines


def _structural_protocol_execution(queue: Path, ctx: ReportContext) -> List[str]:
    """How the A45-A51 structural protocol actually executed in this run."""
    import collections
    root = queue / "results"
    expected_seeds = len((ctx.frozen_config.get("structure_experiment", {}) or {}).get("seeds", []) or [])
    summaries = sorted(root.glob("*/recovery_quality_summary.json"))
    failures = collections.Counter()
    for path in summaries:
        for failure in (_read_json(path) or {}).get("failures", []) or []:
            failures[str(failure.get("category"))] += 1
    attempts = []
    for path in sorted(root.glob("*/seed_*/perturbation.json")):
        value = (_read_json(path) or {}).get("selected_attempt")
        if value is not None:
            attempts.append(int(value))
    stops = collections.Counter()
    reference_outputs = vetoed_outputs = outputs = forbidden_selected = 0
    for result_path in sorted(root.glob("*/seed_*/experiment/*_result.json")):
        relaxation = (_read_json(result_path) or {}).get("relaxation") or {}
        outputs += 1
        stops[str(relaxation.get("minimizer_stop_reason"))] += 1
        reference_outputs += int(bool(relaxation.get("minimizer_reference_platform_evaluations")))
        vetoed_outputs += int(bool(relaxation.get("minimizer_overlap_floor_vetoes")))
        forbidden_selected += int(relaxation.get("selected_state_geometry_admissible") is False)
    instances = with_forbidden = fallback = shielded = 0
    for mapping in sorted(root.glob("*/seed_*/experiment/allatom_mapping.json")):
        metadata = (_read_json(mapping) or {}).get("metadata") or {}
        instances += 1
        with_forbidden += int(bool(metadata.get("forbidden_variables") or metadata.get("forbidden_variable_pairs")))
        fallback += int(bool(metadata.get("no_admissible_assignment_raw_fallback")))
        shielded += int(bool((metadata.get("polar_hydrogen_shielding") or {}).get("shielded_hydrogens")))
    lines = ["### 4.3 Structural protocol execution (A45-A51)", ""]
    if not summaries:
        lines += ["No per-target recovery summaries were found.", ""]
        return lines
    sorted_attempts = sorted(attempts)
    median = sorted_attempts[len(sorted_attempts) // 2] if sorted_attempts else None
    lines += [
        f"- Targets with a recovery summary: {len(summaries)}; seeds per target: {expected_seeds}. "
        f"Recorded seed failures by category: {dict(failures) or 'none'} (all retained in denominators).",
        f"- Generated inputs (A45/A51): {len(attempts)} seeds received a geometry-qualified input; selected attempt "
        f"index median {median}, maximum {max(attempts) if attempts else None} (0 = the seed's own first draw).",
        f"- Relaxations: {outputs} outputs; minimizer stop reasons {dict(stops)}; {reference_outputs} used "
        f"double-precision Reference evaluations for close contacts (A46); {vetoed_outputs} had an overlap-floor "
        "veto (A47).",
        f"- Structural QUBOs: {instances} instances; {with_forbidden} contained geometry-forbidden states, "
        f"{fallback} used the raw-energy fallback (A47); {forbidden_selected} solver selections were forbidden "
        f"states (judged by physical acceptance only, A51); polar-hydrogen shielding applied in {shielded} (A50).",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Section 5: external validation and development-only robustness
# ---------------------------------------------------------------------------

def section_external_and_robustness(ctx: ReportContext) -> List[str]:
    lines=["## 7. External validation and robustness", ""]

    # External VHH coarse benchmark.
    ext_metrics=ctx.run_dir/"external_validation"/"vhh_coarse"/"metrics.csv"
    if ext_metrics.is_file():
        rows=_read_csv_rows(ext_metrics)
        lines.append("### 7.1 External VHH benchmark")
        lines.append("")
        lines.append(
            f"- External graph rows recorded: {len(rows)}; target set is required to pass the frozen "
            "sequence + family/structure independence manifest before this stage runs."
        )
        by_solver={}
        for row in rows:
            solver=row.get("solver","")
            if solver:
                by_solver.setdefault(solver,[]).append(row)
        lines.append("| Solver | Rows | Mean gap | Mean hit fraction |")
        lines.append("|---|---:|---:|---:|")
        for solver,group in sorted(by_solver.items()):
            def mean(field: str):
                vals=[float(r[field]) for r in group if r.get(field) not in (None,"","None")]
                return None if not vals else sum(vals)/len(vals)
            lines.append(
                f"| {solver} | {len(group)} | {_fmt(mean('gap'))} | {_fmt(mean('hit'))} |"
            )
        lines.append("")
    else:
        lines.append("### 7.1 External VHH benchmark")
        lines.append("")
        lines.append("No completed external VHH metrics were found for this run.")
        lines.append("")

    # Mature biological side-chain packing baseline.
    baseline_path=ctx.run_dir/"external_validation"/"structural_baselines"/"external_baseline_metrics.csv"
    if baseline_path.is_file():
        rows=_read_csv_rows(baseline_path)
        lines.append("### 7.2 FASPR / standard clashscore baseline")
        lines.append("")
        lines.append("| Method | Rows | Targets | Mean RMSD | Mean χ1 recovery | Mean all-χ recovery | Mean contact F1 | Mean Phenix clashscore |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for method in sorted({r.get("method","") for r in rows if r.get("method")}):
            group=[r for r in rows if r.get("method")==method]
            def _mean(field):
                values=[float(r[field]) for r in group if r.get(field) not in (None,"","None")]
                return None if not values else sum(values)/len(values)
            lines.append(
                f"| {method} | {len(group)} | {len({r.get('target','') for r in group})} | "
                f"{_fmt(_mean('final_rmsd'))} | {_fmt(_mean('chi1_recovery'))} | "
                f"{_fmt(_mean('all_chi_recovery'))} | {_fmt(_mean('contact_f1'))} | "
                f"{_fmt(_mean('molprobity_clashscore'))} |"
            )
        lines.append("")
        lines.append(
            "- FASPR is run through the same Active-only packing scope as the internal methods: its "
            "backbone and non-Active side chains are restored from the perturbed input before scoring. "
            "Phenix clashscore and the common structural evaluator are applied to FASPR and to the final "
            "QAOA/SA/uniform/greedy structures on the same target/seed inputs; solver inference remains separate."
        )
        lines.append("")
    else:
        lines.append("### 7.2 FASPR / standard clashscore baseline")
        lines.append("")
        lines.append("No completed external structural-baseline metrics were found for this run.")
        lines.append("")

    # Development-only solvent sensitivity.
    dev_enabled=bool(((ctx.frozen_config.get("queue_freeze",{}) or {}).get("dev_queue",{}) or {}).get("enabled",True))
    sensitivity_dirs=sorted(ctx.run_dir.glob("dev_queue_solvent_*")) if dev_enabled else []
    lines.append("### 7.3 Development-only solvent-model sensitivity")
    lines.append("")
    if not dev_enabled:
        lines.append("Development target experiments and their solvent-model sensitivity are disabled by protocol.")
    elif sensitivity_dirs:
        lines.append("| Solvent model | Rows | Mean final RMSD | Mean improvement vs input |")
        lines.append("|---|---:|---:|---:|")
        for directory in sensitivity_dirs:
            rows=_load_recovery_rows(directory)
            finals=[float(r["final_rmsd"]) for r in rows if r.get("final_rmsd") not in (None,"","None")]
            gains=[float(r["improvement_vs_input"]) for r in rows
                   if r.get("improvement_vs_input") not in (None,"","None")]
            model=directory.name.removeprefix("dev_queue_solvent_")
            lines.append(
                f"| {model} | {len(rows)} | "
                f"{_fmt(sum(finals)/len(finals) if finals else None)} | "
                f"{_fmt(sum(gains)/len(gains) if gains else None)} |"
            )
        lines.append("")
        lines.append(
            "This sensitivity analysis is development-only. Validation targets remain on the frozen "
            "primary solvent model and are not used to select the solvent treatment."
        )
    else:
        lines.append("No development solvent-sensitivity results were found.")
    lines.append("")
    return lines
