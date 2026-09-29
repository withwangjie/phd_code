#!/usr/bin/env python3
"""generate_final_research_report.py -- recompute FINAL_RESEARCH_REPORT.md
from one run_full_experiment.py run directory's raw per-instance artifacts.

Never reads a cached/previous summary of itself: every number in the
generated report is recomputed here, directly, from the run's own
per-case/per-target CSV and JSON records, each run's own stage_status/*.json
markers, and (for data-audit counts) the repository-root audit artifacts
that stage produced. This is deliberate -- a report that trusted an
upstream script's own prose summary could silently drift from what actually
happened; recomputation from raw records is the only way "budget exhaustion
was reported as convergence" or "process startup was reported as
experiment completion" cannot slip through.

Organized around the quantum-computing research object: quantum-instance
encoding -> frozen QAOA protocol/resources -> quantum-classical benchmark and
scaling -> structural/physical validation -> data reliability and problem
reduction -> robustness/cost/applicability. A stage-by-stage
completed/failed/incomplete count table remains up front so every claim is
traceable to completed run artifacts.

Usage:
    python -m nanoqc.reporting.generate_final_research_report --run-dir experiments_full_run_20260920_010203
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))


# Reused, not reimplemented: the same bootstrap-CI / sign-flip-test / Holm
# correction machinery batch_benchmark_hard_set.py's own --paired-statistics
# uses, applied here to the real-atom structural-recovery CSVs (which that
# entrypoint cannot read directly -- see run_full_experiment.py's
# stage_statistics comment for why).

# Definitions now live in focused modules; re-exported so every existing
# `from nanoqc.reporting.generate_final_research_report import ...` keeps working.
from nanoqc.reporting.report_common import (  # noqa: E402,F401
    _read_json,
    _read_csv_rows,
    _read_text,
    _filter_qc_rows,
    _formal_statistics_payload,
    _paired_inference_multiplicity_note,
    _primary_active_sites,
    _primary_pruning,
    _quantum_primary,
    _fmt,
    ReportContext,
    STAGE_ORDER,
    stage_ok,
)
from nanoqc.reporting.report_sections_quantum import (  # noqa: E402,F401
    training_coarse_atomistic_rank_diagnostic,
    section_pruning_contribution,
    section_quantum_problem_encoding,
    section_quantum_protocol,
    section_search_performance,
)
from nanoqc.reporting.report_sections_structure import (  # noqa: E402,F401
    _load_recovery_rows,
    _target_level_paired_differences,
    _target_level_descriptive_difference,
    section_structural_benefit,
    section_external_and_robustness,
)


def section_stage_table(ctx: ReportContext) -> List[str]:
    lines = ["## 0. Stage completion", "",
             "Every number below is scoped to only the stages that actually completed; "
             "a stage marked `failed` or `not_started` means the sections depending on it "
             "report an explicit gap rather than a silently stale or fabricated number.",
             "", "| Stage | Status | Detail |", "|---|---|---|"]
    for stage in STAGE_ORDER:
        record = ctx.stage_status.get(stage)
        status = record["status"] if record else "not_started"
        detail = (record or {}).get("detail", "")
        if stage=="final_report" and status=="running":
            status="in_progress (this report)"
            detail="The report is being generated; terminal completion is recorded immediately after successful write."
        detail = detail.replace("\n", " ")[:200]
        lines.append(f"| {stage} | {status} | {detail} |")
    return lines


# ---------------------------------------------------------------------------
# Section 1: data reliability
# ---------------------------------------------------------------------------

def section_data_reliability(ctx: ReportContext) -> List[str]:
    lines = ["## 5. Data reliability and problem preparation", ""]
    if not stage_ok(ctx, "data_audit"):
        lines += ["Data audit stage did not complete; no data-reliability numbers are reported here.", ""]
        return lines

    inventory = _read_json(ctx.run_dir / "audit" / "data_audit_inventory.json") or {}
    lines.append("### 5.1 Raw structure audit (audit_all_datasets.py)")
    lines.append("")
    if inventory:
        lines.append(f"Audit inventory recorded (see `data_audit_report.md` for full per-reason breakdown, "
                      f"and `data_audit_details.csv`/`.jsonl` for the per-structure ledger this run consumed).")
        quality_protocol=inventory.get("structure_quality_protocol", {}) or {}
        if quality_protocol:
            lines.append(f"- High-confidence structure-quality protocol: {json.dumps(quality_protocol, sort_keys=True)}")
    else:
        lines.append("`data_audit_inventory.json` was not found or not parseable in this run's audit directory.")
    lines.append("")

    if not stage_ok(ctx, "queue_freeze"):
        lines += ["Queue-freeze stage did not complete; dedup/isolation and validation-queue "
                   "eligibility numbers are not reported.", ""]
        return lines

    dataset_dir = ctx.resolve_run((ctx.frozen_config.get("paths", {}) or {}).get("dataset_dir", "dataset"))
    run_summary = _read_json(dataset_dir / "run_summary.json") or {}
    summary_homology = run_summary.get("homology_isolation", {}) or {}
    homology = summary_homology or ((ctx.frozen_config.get("queue_freeze", {}) or {}).get("homology_isolation", {}) or {})
    cdr_h3_threshold = float(homology.get("cdr_h3_identity", 0.50))
    vhh_threshold = float(homology.get("vhh_full_chain_identity", 0.80))
    antigen_threshold = float(homology.get("antigen_identity", 0.30))
    antigen_coverage = float(homology.get("antigen_min_length_coverage", 0.70))
    lines.append("### 5.2 Deduplication, isolation, and split (build_final_pyg_dataset.py --no-cap)")
    lines.append("")
    lines.append(f"- Cap removed (requirement #1): `no_cap` semantics used; "
                  f"`target_hard` field in run_summary.json is informational only when uncapped.")
    lines.append(f"- Admission by source: {json.dumps(run_summary.get('admission', {}))}")
    lines.append(f"- CDR-H3 >=16aa eligible pool: {run_summary.get('long_eligible', 'n/a')}; "
                  f"unique CDR sequences: {run_summary.get('unique_long_cdr', 'n/a')}; "
                  f"{cdr_h3_threshold*100:.0f}%-CDR-H3-loop-identity initial clusters: {run_summary.get('clusters', 'n/a')}.")
    lines.append(
        f"- Final layered graph-level isolation: VHH full-chain<{vhh_threshold:.2f}, "
        f"CDR-H3 loop<{cdr_h3_threshold:.2f}, antigen full-chain<{antigen_threshold:.2f} "
        f"with antigen length coverage>={antigen_coverage:.2f}. "
        f"Observed train-hard maxima: "
        f"{json.dumps((run_summary.get('validation', {}) or {}).get('train_hard_layered_cross_max', {}))}"
    )
    graphs = run_summary.get("graphs", "n/a")
    complete = run_summary.get("complete", None)
    lines.append(f"- Total graphs written this run: {graphs}. Pipeline-level `complete` flag: {complete}.")
    exclusions_path = dataset_dir / "excluded_samples.csv"
    exclusion_rows = _read_csv_rows(exclusions_path)
    lines.append(f"- Excluded samples (quality/dedup/isolation reasons preserved verbatim): "
                  f"{len(exclusion_rows)} rows in `{exclusions_path.name}`.")
    failures_rows = _read_csv_rows(dataset_dir / "processing_failures.csv")
    lines.append(f"- Processing failures during graph construction: {len(failures_rows)}.")
    lines.append("")

    validation_dir = ctx.run_dir / "validation_queue"
    freeze_dir = validation_dir / "freeze"
    eligibility = _read_json(freeze_dir / "eligibility.json") or []
    selected = _read_json(freeze_dir / "selected_targets.json") or []
    excluded = [d for d in eligibility if d.get("status") == "excluded"]
    dev_pdb = ((ctx.frozen_config.get("queue_freeze", {}) or {}).get("dev_queue", {}) or {}).get("excluded_pdb", [])
    lines.append("### 5.3 Frozen, blind validation-target queue (requirement #2)")
    lines.append("")
    lines.append(f"- Historical development targets permanently excluded from this queue: {dev_pdb}.")
    lines.append(f"- Candidates examined: {len(eligibility)}; selected (frozen): {len(selected)}; "
                  f"excluded: {len(excluded)}.")
    if excluded:
        reason_counts: Dict[str, int] = {}
        for entry in excluded:
            reason = str(entry.get("reason", "unknown"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        top_reasons = sorted(reason_counts.items(), key=lambda kv: -kv[1])[:10]
        lines.append(f"- Top exclusion reasons: {top_reasons}")
    lines.append("- Selection order: seeded-random (never ascending-structure-size), so this queue does not "
                  "inherit the historical dev pilot's smallest-first bias. Every examined PDB and its exact "
                  "exclusion reason is preserved in `validation_queue/freeze/eligibility.json`; nothing was replaced "
                  "for solver convenience after being selected.")
    # (requirement #1) Independence is never reported as a single pass/fail
    # gate: every examined candidate (selected or excluded) carries a
    # recorded chain-level identity/coverage audit, a CDR-H3 identity value,
    # a development-exposure flag, and an explicit independence_status --
    # surfaced here directly from eligibility.json, never re-derived or
    # summarized away.
    status_counts: Dict[str, int] = {}
    exposed_count = 0
    for entry in eligibility:
        status_counts[str(entry.get("independence_status", "unrecorded"))] = \
            status_counts.get(str(entry.get("independence_status", "unrecorded")), 0) + 1
        if entry.get("development_exposed"):
            exposed_count += 1
    lines.append(f"- Independence status distribution over all {len(eligibility)} examined candidates: "
                  f"{dict(sorted(status_counts.items()))}. Development-exposed (per --dev-exposed-pdb): "
                  f"{exposed_count}/{len(eligibility)}.")
    cluster_cfg=((ctx.frozen_config.get("queue_freeze", {}) or {}).get("independence_clustering", {}) or {})
    validation_meta=run_summary.get("validation", {}) or {}
    lines.append(
        f"- Family/structure clustering required={cluster_cfg.get('required', False)}; "
        f"map={cluster_cfg.get('cluster_map')}; dataset builder recorded map use="
        f"{validation_meta.get('family_cluster_map_used')} and train-hard cluster overlap="
        f"{validation_meta.get('family_cluster_train_hard_overlap')}."
    )
    if selected:
        vhh_ids = [float(d.get("max_vhh_full_chain_identity", 0.0)) for d in selected]
        antigen_ids = [float(d.get("max_antigen_full_chain_identity", 0.0)) for d in selected]
        cdr_ids = [float(d.get("max_cdr_h3_loop_identity", 0.0)) for d in selected]
        if vhh_ids:
            lines.append(
                f"- Layered homology isolation: VHH full-chain < {vhh_threshold:.2f}, "
                f"CDR-H3 loop < {cdr_h3_threshold:.2f}, antigen full-chain < {antigen_threshold:.2f} "
                f"with minimum length coverage {antigen_coverage:.2f}. "
                f"Selected-target maxima: VHH full-chain {max(vhh_ids):.3f}, "
                f"CDR-H3 loop {max(cdr_ids):.3f}, antigen full-chain {max(antigen_ids):.3f}."
            )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 6: cost
# ---------------------------------------------------------------------------

def section_cost(ctx: ReportContext) -> List[str]:
    lines = ["## 8. Computational cost and resource accounting", ""]
    if stage_ok(ctx, "qc_benchmark"):
        all_rows = _read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv")
        rows = _filter_qc_rows(all_rows, "matched_outputs")
        if rows:
            by_size_solver: Dict[tuple[int,str], List[float]] = {}
            for r in rows:
                solver=r.get("solver","")
                if r.get("solver_seconds") not in (None,"","None"):
                    sites=int(float(r.get("active_sites",_primary_active_sites(ctx))))
                    by_size_solver.setdefault((sites,solver),[]).append(float(r["solver_seconds"]))
            lines.append("### 8.1 Coarse-grained matched-output solver cost (seconds, mean per row)")
            lines.append("")
            lines.append("| Active sites | Solver | Cases | Mean solver_seconds |")
            lines.append("|---:|---|---:|---:|")
            for (sites,solver),values in sorted(by_size_solver.items()):
                lines.append(f"| {sites} | {solver} | {len(values)} | {_fmt(sum(values)/len(values))} |")
            lines.append("")
            lines.append("Only matched-output rows are summarized here; matched-time controls are kept separate. "
                          "For QAOA, `solver_seconds` is a standalone-equivalent cost: measured optimization time "
                          "plus the measured sampling time for that output budget. The optimization is physically "
                          "run once per objective/restart variant and reused across the output curve, so this is not "
                          "the incremental wall-clock time of each cached row. Oracle/exact-landscape preprocessing "
                          "(`oracle_seconds`/`build_seconds`) is excluded.")
            lines.append("")
            qprimary=_quantum_primary(ctx)
            stats_cfg=ctx.frozen_config.get("statistics",{}) or {}
            primary_pruning=str(stats_cfg.get("primary_pruning","egnn"))
            primary_radius=float(stats_cfg.get("primary_radius",6.0))
            qrows=[
                r for r in rows
                if r.get("solver")=="qaoa"
                and str(r.get("pruning",""))==primary_pruning
                and abs(float(r.get("radius",primary_radius))-primary_radius)<=1e-12
                and r.get("qaoa_objective")==str(qprimary.get("objective","cvar"))
                and int(float(r.get("qaoa_restarts",0) or 0))==int(qprimary.get("restarts",4))
                and int(float(r.get("outputs",0) or 0))==int(qprimary.get("output_shots",1000))
            ]
            if qrows:
                lines.append("### 8.2 Logical QAOA resources (pre-transpilation)")
                lines.append("")
                lines.append("| Active sites | Cases | Mean qubits | Mean 2q gates | Mean XY | Mean ZZ | Mean total measurement shots |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|")
                for sites in sorted({int(float(r["active_sites"])) for r in qrows}):
                    group=[r for r in qrows if int(float(r["active_sites"]))==sites]
                    def qmean(field: str) -> Optional[float]:
                        values=[float(r[field]) for r in group if r.get(field) not in (None,"","None")]
                        return sum(values)/len(values) if values else None
                    lines.append(
                        f"| {sites} | {len(group)} | {_fmt(qmean('num_bits'))} | "
                        f"{_fmt(qmean('qaoa_two_qubit_gates'))} | {_fmt(qmean('qaoa_xy_gates'))} | "
                        f"{_fmt(qmean('qaoa_zz_gates'))} | {_fmt(qmean('qaoa_total_measurement_shots'))} |"
                    )
                lines.append("")
                lines.append(
                    "These are logical, pre-transpilation counts for the implemented cost and XY-mixer layers; "
                    "they are not hardware-native gate counts and exclude decomposition of local W-state StatePrep. "
                    "This separation follows resource-transparent quantum-optimization benchmarking guidance [R20,R29]."
                )
                lines.append("")
            lines.append("### 8.3 Measurement shots and classical energy queries")
            lines.append("")
            lines.append("| Active sites | Solver | Rows | Mean QAOA measurement shots | Mean classical single-state energy queries | Mean solver seconds |")
            lines.append("|---:|---|---:|---:|---:|---:|")
            for sites in sorted({int(float(r["active_sites"])) for r in rows if r.get("active_sites") not in (None,"","None")}):
                for solver in ("qaoa","sa","greedy","uniform"):
                    group=[r for r in rows if r.get("solver")==solver
                           and int(float(r.get("active_sites",0) or 0))==sites
                           and int(float(r.get("outputs",0) or 0))==int(qprimary.get("output_shots",1000))
                           and str(r.get("pruning",""))==primary_pruning
                           and r.get("budget_mode") in (None,"", "matched_outputs")
                           and (solver!="qaoa" or (r.get("qaoa_objective")==str(qprimary.get("objective","cvar"))
                               and int(float(r.get("qaoa_restarts",0) or 0))==int(qprimary.get("restarts",4))))]
                    if not group:
                        continue
                    def mean_field(field: str) -> Optional[float]:
                        values=[float(r[field]) for r in group if r.get(field) not in (None,"","None")]
                        return sum(values)/len(values) if values else None
                    lines.append(f"| {sites} | {solver} | {len(group)} | "
                                 f"{_fmt(mean_field('qaoa_total_measurement_shots'))} | "
                                 f"{_fmt(mean_field('single_state_energy_queries'))} | "
                                 f"{_fmt(mean_field('solver_seconds'))} |")
            lines.append("")
            lines.append("Shots and single-state energy queries are different operations. The legacy "
                         "cross-method `log10_qts99` adds them as abstract accounting units under an explicit "
                         "one-shot-equals-one-query convention; it is not a hardware-normalized cost or runtime "
                         "measure. Circuit state preparation and transpilation are not included.")
            lines.append("")
    for label, directory in (("dev queue", ctx.run_dir / "dev_queue"), ("validation queue", ctx.run_dir / "validation_queue")):
        if label=="dev queue" and not bool(((ctx.frozen_config.get("queue_freeze",{}) or {}).get("dev_queue",{}) or {}).get("enabled",True)):continue
        rows = _load_recovery_rows(directory)
        if not rows:
            continue
        shots: List[float] = [float(r["total_opt_shots"]) for r in rows if r.get("total_opt_shots") not in (None, "", "None")]
        if shots:
            lines.append(f"- {label}: mean `total_opt_shots` per method x seed row: {_fmt(sum(shots)/len(shots))} "
                          f"(n={len(shots)}).")
    lines.append("")
    lines.append("Equal output/shot counts across solvers are matched-output comparisons, not matched-time or "
                  "matched-compute comparisons; see the raw per-case JSON (`qc_benchmark/cases/*.json`) for the "
                  "full cost breakdown behind every summary number above.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 7: failures and incomplete work (planned/completed/failed closure,
# consolidated from every stage's own run_summary.json rather than scattered
# across the sections that happen to touch each artifact)
# ---------------------------------------------------------------------------

def section_failures_and_incomplete(ctx: ReportContext) -> List[str]:
    lines = ["## 9. Failures and incomplete work", "",
        "Every number below comes from a stage's own planned/completed/failed reconciliation "
        "(`run_summary.json`), never inferred from a subprocess return code or from a single "
        "artifact's mere existence -- `closed` requires `completed + failed == planned`, and a "
        "stage that is not closed is reported as such here even if it also produced partial results.",
        ""]
    any_incomplete = False
    for stage in STAGE_ORDER:
        record = ctx.stage_status.get(stage)
        status = record.get("status") if record else "not_started"
        if status not in ("completed", "completed_with_failures"):
            any_incomplete = True
            lines.append(f"- Stage `{stage}`: **{status}**. "
                          f"{(record or {}).get('detail', 'No detail recorded.')}")
    qc_summary = _read_json(ctx.run_dir / "qc_benchmark" / "run_summary.json")
    if qc_summary:
        lines.append(f"- `qc_benchmark`: planned={qc_summary.get('total_cases_planned')} "
                      f"completed={qc_summary.get('cases_completed_total')} "
                      f"failed={qc_summary.get('failures_total')} gap={qc_summary.get('gap')} "
                      f"closed={qc_summary.get('closed')}.")
        if not qc_summary.get("closed"):
            any_incomplete = True
    for label, directory in (("dev_queue", ctx.run_dir / "dev_queue"), ("validation_queue", ctx.run_dir / "validation_queue")):
        if label=="dev_queue" and not bool(((ctx.frozen_config.get("queue_freeze",{}) or {}).get("dev_queue",{}) or {}).get("enabled",True)):continue
        summary = _read_json(directory / "run_summary.json")
        if summary:
            lines.append(f"- `{label}`: qualifying_pool={summary.get('qualifying_pool_size')} "
                          f"target_cap={summary.get('target_cap')} "
                          f"examined={summary.get('examined_candidates')} "
                          f"selected={summary.get('selected_targets')} "
                          f"completed={summary.get('structure_experiment_completed_targets')} "
                          f"failed_targets={summary.get('structure_experiment_failed_targets')} "
                          f"closed={summary.get('closed')}.")
            if not summary.get("closed") or summary.get("structure_experiment_failed_targets"):
                any_incomplete = True
    if not any_incomplete:
        lines.append("- No stage-level or target-level incompleteness recorded: every stage that ran is "
                      "closed with `completed + failed == planned`, and no per-target/per-case failure "
                      "was left outside its own failure list.")
    lines.append("")
    lines.append("Per-instance (single case/single target) failure is expected in an exploratory pipeline "
                  "and does not by itself invalidate a stage's other results -- what matters is that every "
                  "failure is accounted for in a failure list (`failed_cases.log`, "
                  "`structure_experiment_failed_targets`, `failures.log`) rather than silently dropped from "
                  "the planned count, which is exactly what `closed` above verifies.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 8: applicability boundary
# ---------------------------------------------------------------------------

def section_applicability_boundary(ctx: ReportContext) -> List[str]:
    lines = ["## 10. Applicability boundary", "",
        "- Fixed backbone, known binding pose, local side-chain optimization only. Not blind docking, not "
        "CDR-H3 backbone prediction, not de novo complex structure prediction.",
        "- 4S10/8YVO/9GCN remain development-only regression targets (selection order: ascending structure "
        "size; already inspected during development). They are never reported above as a confirmatory result.",
        "- The validation queue in section 1.3 is the only queue in this run intended to support a "
        "confirmatory claim, and only to the extent its own target count and per-target variance support one "
        "- A small queue (see section 1.3 for its actual selected count) supports stability/sanity checking, "
        "not a general statistical-power guarantee.",
        f"- Independence checking is PDB-disjoint plus layered sequence isolation using published anti-leakage precedents: "
        f"VHH full-chain < {100.0 * float((((ctx.frozen_config.get('queue_freeze', {}) or {}).get('homology_isolation', {}) or {}).get('vhh_full_chain_identity', 0.80))):.0f}%, "
        f"CDR-H3 loop < {100.0 * float((((ctx.frozen_config.get('queue_freeze', {}) or {}).get('homology_isolation', {}) or {}).get('cdr_h3_identity', 0.50))):.0f}%, "
        f"antigen full-chain < {100.0 * float((((ctx.frozen_config.get('queue_freeze', {}) or {}).get('homology_isolation', {}) or {}).get('antigen_identity', 0.30))):.0f}% "
        f"with minimum length coverage {100.0 * float((((ctx.frozen_config.get('queue_freeze', {}) or {}).get('homology_isolation', {}) or {}).get('antigen_min_length_coverage', 0.70))):.0f}%. "
        "Formal runs additionally require the frozen family/structure cluster map recorded in section 1; "
        "independence claims are limited to those explicitly encoded sequence and cluster criteria, not arbitrary remote homology.",
        "- QAOA follows the variational optimization framework of Farhi et al. [R7]. CVaR optimization follows [R8]; "
        "alpha=0.1 is literature-supported as an empirical CVaR setting, while primary p=2 remains a preregistered "
        "depth choice checked by development-only p=1/2/3 sensitivity.",
        "- Classical simulation of the discrete QAOA circuit in the feasible subspace is exact-subspace "
        "classical simulation, not a quantum-hardware run; nothing in this report should be read as a "
        "hardware or NISQ-noise result.",
        "- \"Budget exhausted\" (`termination_reason == max_evaluations_reached`) is reported as exactly "
        "that, never as algorithmic convergence -- see section 3.2's raw termination-reason distribution.",
        "- A stage that only started successfully (subprocess launched, provenance file written) but produced "
        "no usable per-case artifacts is marked `failed` in section 0, never silently reported as complete.",
        "- This report, and the code it summarizes, exist in exactly three distinct, never-conflated states: "
        "(a) *code implemented* -- a feature exists in source but has not been exercised at all; (b) "
        "*offline check passed* -- syntax/unit-level verification only (`py_compile`, a standalone function "
        "test), never run against real data or the full pipeline; (c) *real experiment run* -- this run "
        "directory's own stage_status/run_summary.json/CSV artifacts, produced by an actual execution. "
        "Section 0 and this section state, per stage, which of these three applies. "
        "A claim never advances from (a)/(b) to (c) "
        "without an actual run producing the artifact that backs it.",
        "- DockQ-like fields in this project's CSVs come from this repository's own evaluator "
        "(`evaluate_complex_metrics.py`/`structural_quality.py`), not the official DockQ reference implementation; "
        "`dockq_definition` on every row records the exact variant. Treat them as project-specific structural-quality "
        "metrics rather than official DockQ values when comparing with literature.",
        "- No *hardware* quantum advantage or quantum speedup is claimed: every QAOA result here is an "
        "exact-subspace classical simulation of a finite-shot circuit. The report may describe a simulator-level "
        "QAOA relative performance advantage or size-dependent quantum-classical trend only when directly supported "
        "by the recorded matched-output/matched-time data; such evidence is algorithmic and does not establish "
        "hardware quantum advantage.",
        "",
    ]
    return lines


def section_literature_basis() -> List[str]:
    evidence_path=REPO_ROOT/"docs"/"METHODS_EVIDENCE.md"
    lines=["## 11. Methodological literature basis",""]
    if not evidence_path.is_file():
        lines += ["`METHODS_EVIDENCE.md` is missing; formal literature traceability is unavailable.",""]
        return lines
    text=evidence_path.read_text(encoding="utf-8")
    marker="## References"
    if marker not in text:
        lines += ["Reference section missing from `METHODS_EVIDENCE.md`.",""]
        return lines
    refs=text.split(marker,1)[1].strip()
    lines += [
        "The formal protocol follows the evidence classes recorded in `METHODS_EVIDENCE.md`: direct literature basis, literature-informed preregistration, and study-specific preregistration.",
        "Exact numerical values are described as literature-based only when the cited paper directly supports that definition/value in a comparable setting.",
        "",
        refs,
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compile_report(run_dir: Path) -> str:
    ctx = ReportContext(run_dir)
    lines: List[str] = [
        "# Final Quantum Optimization Research Report", "",
        f"Run directory: `{run_dir}`",
        f"Master seed: {ctx.frozen_config.get('master_seed', 'n/a')} "
        f"(derived streams: {ctx.seed_streams.get('streams', {})})",
        f"Git commit: {ctx.run_manifest.get('git_commit', 'n/a')}",
        "",
        "Recomputed entirely from this run's own raw per-case/per-target records; see section 0 for exactly "
        "which stages completed and therefore which sections below rest on complete data.",
        "",
    ]
    lines += section_stage_table(ctx)
    lines += [""]
    lines += section_quantum_problem_encoding(ctx)
    lines += section_quantum_protocol(ctx)
    lines += section_search_performance(ctx)
    lines += section_structural_benefit(ctx)
    lines += section_data_reliability(ctx)
    lines += section_pruning_contribution(ctx)
    lines += section_external_and_robustness(ctx)
    lines += section_cost(ctx)
    lines += section_failures_and_incomplete(ctx)
    lines += section_applicability_boundary(ctx)
    lines += section_literature_basis()
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                         help="Defaults to <run-dir>/FINAL_RESEARCH_REPORT.md")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        parser.error(f"Run directory not found: {run_dir}")
    out_path = args.out or (run_dir / "FINAL_RESEARCH_REPORT.md")
    report = compile_report(run_dir)
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    temp.write_text(report, encoding="utf-8")
    temp.replace(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
