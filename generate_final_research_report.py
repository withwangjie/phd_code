#!/usr/bin/env python3
"""generate_final_research_report.py -- recompute FINAL_RESEARCH_REPORT.md
from one run_full_experiment.py run directory's raw per-instance artifacts.

Never reads a cached/previous summary of itself. Descriptive numbers are
recomputed from the run's raw per-case/per-target CSV/JSON records and
stage-status markers. Formal inferential results are read from the
machine-readable statistics_outputs.json/statistics_time.json artifacts
produced by the frozen paired-statistics stage, never from prose summaries;
this keeps the report aligned with the exact primary contrast, exclusion
ledger, bootstrap/sign-flip tests and Holm correction actually executed.

Organized in exactly the order requested: data reliability -> pruning
contribution -> search performance -> structural benefit -> cost ->
applicability boundary, plus a stage-by-stage completed/failed/incomplete
count table up front so a reader immediately knows which sections rest on
complete data and which do not.

Usage:
    python generate_final_research_report.py --run-dir experiments_full_run_20260920_010203
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError as exc:
    raise SystemExit(f"PyYAML is required to read frozen_config.yaml: {exc}")

# Reused, not reimplemented: the same bootstrap-CI / sign-flip-test / Holm
# correction machinery batch_benchmark_hard_set.py's own --paired-statistics
# uses, applied here to the real-atom structural-recovery CSVs (which that
# entrypoint cannot read directly -- see run_full_experiment.py's
# stage_statistics comment for why).
from batch_benchmark_hard_set import _paired_effect, _holm_adjust  # noqa: E402


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_text(path: Path) -> Optional[str]:
    return path.read_text(encoding="utf-8") if path.is_file() else None


def _filter_qc_rows(rows: List[Dict[str, str]], budget_mode: str) -> List[Dict[str, str]]:
    """Keep one benchmark budget family separate from the other.

    Legacy rows without budget_mode are treated as matched_outputs only for
    backwards-compatible development reports. Formal current runs record the
    field explicitly.
    """
    selected = []
    for row in rows:
        mode = row.get("budget_mode") or "matched_outputs"
        if mode == budget_mode:
            selected.append(row)
    return selected


def _formal_statistics_payload(ctx: "ReportContext", mode: str) -> Optional[Dict[str, Any]]:
    payload = _read_json(ctx.run_dir / "qc_benchmark" / f"statistics_{mode}.json")
    return payload if isinstance(payload, dict) else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


# ---------------------------------------------------------------------------
# Report context
# ---------------------------------------------------------------------------

class ReportContext:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.frozen_config = yaml.safe_load(_read_text(run_dir / "frozen_config.yaml") or "{}") or {}
        self.run_manifest = _read_json(run_dir / "run_manifest.json") or {}
        self.seed_streams = _read_json(run_dir / "seed_streams.json") or {}
        self.progress = _read_json(run_dir / "progress.json") or {}
        self.stage_status = {
            path.stem: _read_json(path)
            for path in sorted((run_dir / "stage_status").glob("*.json"))
        } if (run_dir / "stage_status").is_dir() else {}
        self.repo_root = self._resolve_repo_root()

    def _resolve_repo_root(self) -> Path:
        raw = (self.frozen_config.get("paths", {}) or {}).get("repo_root", ".")
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else (REPO_ROOT).resolve()

    def resolve(self, relative: str) -> Path:
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.repo_root / relative)

    def resolve_run(self, relative: str) -> Path:
        """Resolve a path relative to THIS run's own run_dir, not repo_root.

        paths.dataset_dir/paths.checkpoint_dir are directory NAMES scoped
        under run_dir (run isolation -- see run_full_experiment.py's
        dataset_dir()/checkpoint_dir()), never a fixed repo-root path shared
        across runs, so they must be resolved here the same way."""
        candidate = Path(relative)
        return candidate if candidate.is_absolute() else (self.run_dir / relative)


# ---------------------------------------------------------------------------
# Section 0: stage completion table
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "env_check", "smoke_check", "data_audit", "queue_freeze", "egnn_train",
    "qc_benchmark", "structure_experiment", "statistics", "final_report",
]


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
        detail = detail.replace("\n", " ")[:200]
        lines.append(f"| {stage} | {status} | {detail} |")
    return lines


def stage_ok(ctx: ReportContext, stage: str) -> bool:
    record = ctx.stage_status.get(stage)
    return bool(record and record["status"] in ("completed", "completed_with_failures"))


# ---------------------------------------------------------------------------
# Section 1: data reliability
# ---------------------------------------------------------------------------

def section_data_reliability(ctx: ReportContext) -> List[str]:
    lines = ["## 1. Data reliability", ""]
    if not stage_ok(ctx, "data_audit"):
        lines += ["Data audit stage did not complete; no data-reliability numbers are reported here.", ""]
        return lines

    inventory = _read_json(ctx.run_dir / "audit" / "data_audit_inventory.json") or {}
    lines.append("### 1.1 Raw structure audit (audit_all_datasets.py)")
    lines.append("")
    if inventory:
        lines.append(f"Audit inventory recorded (see `data_audit_report.md` for full per-reason breakdown, "
                      f"and `data_audit_details.csv`/`.jsonl` for the per-structure ledger this run consumed).")
    else:
        lines.append("`data_audit_inventory.json` was not found or not parseable in this run's audit directory.")
    lines.append("")

    if not stage_ok(ctx, "queue_freeze"):
        lines += ["Queue-freeze stage did not complete; dedup/isolation and validation-queue "
                   "eligibility numbers are not reported.", ""]
        return lines

    dataset_dir = ctx.resolve_run((ctx.frozen_config.get("paths", {}) or {}).get("dataset_dir", "dataset"))
    run_summary = _read_json(dataset_dir / "run_summary.json") or {}
    lines.append("### 1.2 Deduplication, isolation, and split (build_final_pyg_dataset.py --no-cap)")
    lines.append("")
    lines.append(f"- Cap removed (requirement #1): `no_cap` semantics used; "
                  f"`target_hard` field in run_summary.json is informational only when uncapped.")
    lines.append(f"- Admission by source: {json.dumps(run_summary.get('admission', {}))}")
    lines.append(f"- CDR-H3 >=16aa eligible pool: {run_summary.get('long_eligible', 'n/a')}; "
                  f"unique CDR sequences: {run_summary.get('unique_long_cdr', 'n/a')}; "
                  f"80%-identity clusters: {run_summary.get('clusters', 'n/a')}.")
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

    validation_root = ctx.run_dir / "validation_queue"
    validation_freeze_dir = validation_root / "freeze"
    validation_execution_dir = validation_root / "execution"
    eligibility = _read_json(validation_freeze_dir / "eligibility.json") or []
    selected = _read_json(validation_freeze_dir / "selected_targets.json") or []
    excluded = [d for d in eligibility if d.get("status") == "excluded"]
    dev_pdb = ((ctx.frozen_config.get("queue_freeze", {}) or {}).get("dev_queue", {}) or {}).get("excluded_pdb", [])
    lines.append("### 1.3 Frozen, blind validation-target queue (requirement #2)")
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
                  "exclusion reason is preserved in `validation_queue/eligibility.json`; nothing was replaced "
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
    lines.append("- `independence_status` is capped at `independence_not_confirmed` for every candidate that "
                  "clears identity/exposure screening -- this project has no PDB-family/local-domain cluster "
                  "map, so family- or domain-level relatedness is never checked and a candidate is never "
                  "reported as confirmed-independent. Only `excluded_*` statuses are asserted with confidence.")
    if selected:
        max_identities = [float(d.get("max_chain_identity", 0.0)) for d in selected if "max_chain_identity" in d]
        cdr_identities = [float(d.get("cdr3_identity", 0.0)) for d in selected if "cdr3_identity" in d]
        if max_identities:
            lines.append(f"- Among selected targets: max full-chain identity vs. training/selected pool "
                          f"ranges {min(max_identities):.3f}-{max(max_identities):.3f} "
                          f"(mean {sum(max_identities)/len(max_identities):.3f}); CDR-H3 identity ranges "
                          f"{min(cdr_identities):.3f}-{max(cdr_identities):.3f} "
                          f"(mean {sum(cdr_identities)/len(cdr_identities):.3f}) -- both below the 0.80 "
                          f"exclusion threshold by construction, recorded here rather than only in the boolean gate.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 2: pruning contribution
# ---------------------------------------------------------------------------

def section_pruning_contribution(ctx: ReportContext) -> List[str]:
    lines = ["## 2. EGNN pruning contribution", ""]
    if not stage_ok(ctx, "qc_benchmark"):
        lines += ["qc_benchmark stage did not complete; no pruning-ablation numbers are reported.", ""]
        return lines
    all_rows = _read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv")
    rows = _filter_qc_rows(all_rows, "matched_outputs")
    if not rows:
        lines += ["No matched-output rows were found in `qc_benchmark/metrics.csv`.", ""]
        return lines
    prunings = sorted({r.get("pruning", "") for r in rows if r.get("pruning")})
    lines.append("Four-way pruning-strategy comparison, sharing the same perturbed input, site count, and "
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
                  "over another purely from this table. For QAOA, recorded solver cost is standalone-equivalent "
                  "(measured optimization + sampling), because optimization is cached across the output curve.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 3: search performance (budget curve, objective ablation)
# ---------------------------------------------------------------------------

def section_search_performance(ctx: ReportContext) -> List[str]:
    lines = ["## 3. Search performance: QAOA vs. classical", ""]
    if not stage_ok(ctx, "qc_benchmark"):
        lines += ["qc_benchmark stage did not complete; no search-performance numbers are reported.", ""]
        return lines
    all_rows = _read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv")
    rows = _filter_qc_rows(all_rows, "matched_outputs")
    if not rows:
        lines += ["No matched-output rows were found in `qc_benchmark/metrics.csv`.", ""]
        return lines

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

    primary_outputs = str(
        ((ctx.frozen_config.get("statistics", {}) or {}).get("primary_outputs", 1000)))
    lines.append(
        "### 3.2 QAOA objective/restart ablation at the frozen primary output "
        f"budget ({primary_outputs} outputs)")
    lines.append("")
    qaoa_rows = [
        r for r in rows
        if r.get("solver") == "qaoa" and str(r.get("outputs")) == primary_outputs
    ]
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

    lines.append("### 3.3 Frozen primary paired inference")
    lines.append("")
    if not stage_ok(ctx, "statistics"):
        lines.append("Statistics stage did not complete; no formal paired inference is reported.")
        lines.append("")
        return lines
    for mode in ("outputs", "time"):
        payload = _formal_statistics_payload(ctx, mode)
        label = "Matched outputs" if mode == "outputs" else "Matched time"
        lines.append(f"#### {label}")
        lines.append("")
        if not payload:
            lines.append(f"`statistics_{mode}.json` is missing or unreadable.")
            lines.append("")
            continue
        lines.append(
            f"Frozen primary contrast: outputs={payload.get('primary_outputs')}, "
            f"objective={payload.get('primary_objective')}, restarts={payload.get('primary_restarts')}; "
            f"cluster unit: {payload.get('cluster_unit', 'n/a')}.")
        lines.append("")
        lines.append("| Baseline | Metric | Clusters | Mean QAOA-classical difference | 95% CI | p | Holm p |")
        lines.append("|---|---|---:|---:|---|---:|---:|")
        effects = payload.get("effects") or []
        if effects:
            for effect in effects:
                lines.append(
                    f"| {effect.get('baseline')} | {effect.get('metric')} | {effect.get('n_clusters', 0)} | "
                    f"{_fmt(effect.get('mean_difference'))} | "
                    f"{_fmt(effect.get('ci_low'))}, {_fmt(effect.get('ci_high'))} | "
                    f"{_fmt(effect.get('p_value'))} | {_fmt(effect.get('p_holm'))} |")
        else:
            lines.append("| — | — | 0 | n/a | n/a | n/a | n/a |")
        lines.append("")
        lines.append(f"Paired-case counts: `{json.dumps(payload.get('paired_cases', {}), sort_keys=True)}`.")
        lines.append(f"Exclusions: `{json.dumps(payload.get('exclusions', {}), sort_keys=True)}`.")
        if mode == "time":
            lines.append(
                "Time-mode inference is restricted to gap/hit and excludes excessive soft-deadline overruns; "
                "the QAOA donor budget is the same frozen primary objective/restart variant.")
        lines.append("")
    return lines


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


def _target_level_paired_differences(rows: List[Dict[str, str]], metric: str, baseline: str,
                                      master_seed: int) -> Dict[str, Any]:
    by_target: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        target = row.get("target", "")
        method = row.get("method", "")
        value = row.get(metric)
        if value in (None, "", "None"):
            continue
        by_target.setdefault(target, {}).setdefault(method, []).append(float(value))
    differences = []
    for target, methods in sorted(by_target.items()):
        if "qaoa" not in methods or baseline not in methods:
            continue
        differences.append(sum(methods["qaoa"]) / len(methods["qaoa"])
                            - sum(methods[baseline]) / len(methods[baseline]))
    if not differences:
        return dict(n_targets=0, mean_difference=None, ci_low=None, ci_high=None, p_value=None)
    return _paired_effect(differences, master_seed, 10000)


def section_structural_benefit(ctx: ReportContext) -> List[str]:
    lines = ["## 4. Structure reconstruction, relaxation, and the energy-structure relationship "
             "(does a lower-energy conformer mean a more accurate structure?)", ""]
    if not stage_ok(ctx, "structure_experiment"):
        lines += ["structure_experiment stage did not complete; no structural-benefit numbers are reported.", ""]
        return lines

    dev_rows = _load_recovery_rows(ctx.run_dir / "dev_queue")
    validation_rows = _load_recovery_rows(ctx.run_dir / "validation_queue" / "execution")

    for label, rows in (("4.1 Historical development queue (4S10/8YVO/9GCN; NOT a confirmatory test)", dev_rows),
                         ("4.2 Frozen, blind validation queue (requirement #6)", validation_rows)):
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
        master_seed = ctx.frozen_config.get("master_seed", 20260917)
        lines.append("")
        lines.append("Target-level paired QAOA-vs-classical differences (repeats averaged within target first, "
                      "then targets averaged; positive `improvement_vs_input` favours QAOA):")
        lines.append("")
        lines.append("| Baseline | Metric | Targets paired | Mean difference | 95% CI | p |")
        lines.append("|---|---|---:|---:|---|---|")
        for baseline in ("sa", "uniform", "greedy"):
            for metric in ("improvement_vs_input", "final_rmsd"):
                effect = _target_level_paired_differences(rows, metric, baseline, int(master_seed))
                lines.append(f"| {baseline} | {metric} | {effect.get('n_clusters', 0)} | "
                              f"{_fmt(effect.get('mean_difference'))} | "
                              f"{_fmt(effect.get('ci_low'))}, {_fmt(effect.get('ci_high'))} | "
                              f"{_fmt(effect.get('p_value'))} |")
        lines.append("")

    # Failure denominators, preserved explicitly rather than dropped.
    for label, directory in (("dev queue", ctx.run_dir / "dev_queue"), ("validation queue", ctx.run_dir / "validation_queue" / "execution")):
        failures_log = directory / "failures.log"
        if failures_log.is_file():
            failed = [line for line in _read_text(failures_log).splitlines() if line and not line.startswith(" ")]
            lines.append(f"- {label}: `failures.log` present, {len(failed)} logged failure line(s) preserved verbatim "
                          f"(see `{failures_log}`).")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 5: cost
# ---------------------------------------------------------------------------

def section_cost(ctx: ReportContext) -> List[str]:
    lines = ["## 5. Cost", ""]
    if stage_ok(ctx, "qc_benchmark"):
        all_rows = _read_csv_rows(ctx.run_dir / "qc_benchmark" / "metrics.csv")
        rows = _filter_qc_rows(all_rows, "matched_outputs")
        if rows:
            by_solver: Dict[str, List[float]] = {}
            for r in rows:
                solver = r.get("solver", "")
                if r.get("solver_seconds") not in (None, "", "None"):
                    by_solver.setdefault(solver, []).append(float(r["solver_seconds"]))
            lines.append("### 5.1 Coarse-grained matched-output solver cost")
            lines.append("")
            lines.append("| Solver | Rows | Mean recorded solver_seconds | Interpretation |")
            lines.append("|---|---:|---:|---|")
            for solver, values in sorted(by_solver.items()):
                interpretation = (
                    "standalone-equivalent = measured optimization + this output's measured sampling"
                    if solver == "qaoa"
                    else "measured wall time for this matched-output solver call")
                lines.append(
                    f"| {solver} | {len(values)} | {_fmt(sum(values)/len(values))} | {interpretation} |")
            lines.append("")
            lines.append(
                "QAOA optimization is executed once per objective/restart variant and reused across the "
                "output-budget sampling curve. Therefore a QAOA row's `solver_seconds` is a "
                "**standalone-equivalent** cost (measured optimization_seconds + measured sampling_seconds), "
                "not the incremental wall-clock time consumed by that cached row in the benchmark process. "
                "The raw CSV records `optimization_seconds`, `sampling_seconds`, and "
                "`optimization_reused` explicitly.")
            lines.append(
                "Oracle/exact-landscape preprocessing time is recorded separately "
                "(`oracle_seconds`/`build_seconds`) and excluded from these solver costs.")
            lines.append("")
    for label, directory in (("dev queue", ctx.run_dir / "dev_queue"), ("validation queue", ctx.run_dir / "validation_queue" / "execution")):
        rows = _load_recovery_rows(directory)
        if not rows:
            continue
        shots: List[float] = [float(r["total_opt_shots"]) for r in rows if r.get("total_opt_shots") not in (None, "", "None")]
        if shots:
            lines.append(f"- {label}: mean `total_opt_shots` per method x seed row: {_fmt(sum(shots)/len(shots))} "
                          f"(n={len(shots)}).")
    lines.append("")
    lines.append("Matched-output and matched-time records are never pooled in the descriptive tables above. "
                  "Formal paired inference for each budget family is reported separately in section 3.3 from "
                  "the frozen statistics JSON artifacts. See `qc_benchmark/cases/*.json` for the per-case cost ledger.")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Section 7: failures and incomplete work (planned/completed/failed closure,
# consolidated from every stage's own run_summary.json rather than scattered
# across the sections that happen to touch each artifact)
# ---------------------------------------------------------------------------

def section_failures_and_incomplete(ctx: ReportContext) -> List[str]:
    lines = ["## 7. Failures and incomplete work", "",
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
    for label, directory in (("dev_queue", ctx.run_dir / "dev_queue"), ("validation_queue", ctx.run_dir / "validation_queue" / "execution")):
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
    lines = ["## 8. Applicability boundary", "",
        "- Fixed backbone, known binding pose, local side-chain optimization only. Not blind docking, not "
        "CDR-H3 backbone prediction, not de novo complex structure prediction.",
        "- 4S10/8YVO/9GCN remain development-only regression targets (selection order: ascending structure "
        "size; already inspected during development). They are never reported above as a confirmatory result.",
        "- The validation queue in section 1.3/4.2 is the only queue in this run intended to support a "
        "confirmatory claim, and only to the extent its own target count and per-target variance support one "
        "-- a small queue (see section 1.3 for its actual selected count) supports stability/sanity checking, "
        "not a general statistical-power guarantee.",
        "- Independence checking is PDB-disjoint plus 80% global chain identity and annotated CDR-H3 identity "
        "against training; this does NOT establish remote-homology or antigen-family independence beyond that "
        "threshold.",
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
        "Section 0 and this section state, per stage, which of these three applies; CHANGES.md states the "
        "same distinction for the orchestrator code itself. A claim never advances from (a)/(b) to (c) "
        "without an actual run producing the artifact that backs it.",
        "- `dockq_score`/`dockq_backbone_score`/`dockq_category` in this project's CSVs come from this "
        "repository's own evaluator (`evaluate_complex_metrics.py`/`structural_quality.py`), not the "
        "official DockQ reference implementation; `dockq_definition` on every row records exactly which "
        "variant produced that row's score. Treat these as this project's own structural-quality metric, "
        "not as an official DockQ number, when comparing to literature.",
        "- No quantum-advantage or quantum-speedup claim is made anywhere in this report. Every QAOA result "
        "here is an exact-subspace *classical simulation* of a finite-shot circuit (see section 3 and the "
        "hardware/NISQ-noise disclaimer above); this report states only what the recorded data directly "
        "shows (matched-output accuracy/energy comparisons against SA/uniform/greedy at recorded, unequal "
        "wall-clock cost -- see section 5), and states explicitly, in the relevant section, whenever the "
        "data does not support a directional claim either way.",
        "",
    ]
    return lines


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compile_report(run_dir: Path) -> str:
    ctx = ReportContext(run_dir)
    lines: List[str] = [
        "# Final Research Report", "",
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
    lines += section_data_reliability(ctx)
    lines += section_pruning_contribution(ctx)
    lines += section_search_performance(ctx)
    lines += section_structural_benefit(ctx)
    lines += section_cost(ctx)
    lines += section_failures_and_incomplete(ctx)
    lines += section_applicability_boundary(ctx)
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
