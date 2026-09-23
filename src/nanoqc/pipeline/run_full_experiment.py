#!/usr/bin/env python3
"""run_full_experiment.py -- single unattended entry point for the full
nanobody-interface quantum/classical benchmark research pipeline.

This is an ORCHESTRATOR, not a reimplementation: every stage below shells
out to an existing, already-implemented module in this repository
(audit_all_datasets.py, build_final_pyg_dataset.py,
train_egnn_pruning.py, batch_benchmark_hard_set.py, run_real_complex_pilot.py,
generate_final_research_report.py) via subprocess. This script's only job
is to chain them safely, unattended, with provenance, resumability, and
honest failure reporting -- it contains no scientific logic of its own.

Stage order (each stage checks its own prerequisite stage's completion
marker before running; toggled individually in full_experiment_config.yaml
under `stages:`):

    env_check -> smoke_check -> data_audit -> queue_freeze (dataset split,
    no cap, + validation-queue selection+freeze) -> egnn_train ->
    energy_calibration -> method_sensitivity ->
    qc_benchmark (pruning x budget x objective ablation) ->
    structure_experiment (dev queue + validation queue recovery-benchmark) ->
    external_validation -> statistics (paired + structural analysis) ->
    final_report

Every run gets its own fresh, uniquely timestamped directory under
`paths.run_root` (never reused, never overwritten); raw data, historical
result directories (real_complex_pilot_v1/v2/v3, dataset_clean_500,
checkpoints_500, benchmark_results_*, ...) and existing trained weights are
never written to or deleted by this script. Use --resume <run_dir> to
continue a specific interrupted run (re-derives seeds/hashes and verifies
they match the original launch); a bare re-invocation without --resume
always starts a brand-new run directory, and a concurrent second
invocation anywhere under the same run_root fails fast via a shared lock
file rather than racing.

    python -m nanoqc.pipeline.run_full_experiment --config configs/full_experiment_config.yaml
    python -m nanoqc.pipeline.run_full_experiment --config configs/full_experiment_config.yaml --resume experiments_full_run_20260920_010203
    python -m nanoqc.pipeline.run_full_experiment --config configs/full_experiment_config.yaml --resume <run_dir> --only qc_benchmark
    python -m nanoqc.pipeline.run_full_experiment --config configs/full_experiment_config.yaml --smoke-only

Requires PyYAML and ``filelock`` (both listed in requirements.txt).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

try:
    import yaml  # PyYAML
except ImportError as exc:  # pragma: no cover - reported, not silently swallowed
    raise SystemExit(
        "PyYAML is required (`pip install pyyaml`) to parse full_experiment_config.yaml. "
        f"Import failed: {exc}"
    )

try:
    from filelock import FileLock, Timeout as FileLockTimeout
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "The `filelock` package is required (listed in requirements.txt). "
        f"Import failed: {exc}"
    )

REPO_ROOT = Path(__file__).resolve().parents[3]
# Allow `python src/nanoqc/pipeline/run_full_experiment.py` as well as
# `python -m nanoqc.pipeline.run_full_experiment` with src/ on PYTHONPATH.
sys.path.insert(0, str(REPO_ROOT / "src"))

from nanoqc.common.seed_streams import derive_streams, derive_child_seed, save_stream_map, verify_stream_map, DEFAULT_MASTER_SEED  # noqa: E402
from nanoqc.common.repo_io import sha256_file as sha256_of, repo_path, module_name, DOCS_DIR, CONFIGS_DIR  # noqa: E402
from nanoqc.inference.paired_statistics import paired_denominator_failures, holm_step_down  # noqa: E402


# Detail prefix of the failure recorded when a stage is refused because a
# prerequisite is incomplete. Such a stage never started, so a later resume
# retries it automatically once its prerequisites are complete.
PREREQUISITE_BLOCK_PREFIX = "Prerequisite stage '"


def quantum_protocol(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the frozen top-level quantum protocol."""
    value=config.get("quantum_protocol",{}) or {}
    if not isinstance(value,dict):
        raise ValueError("quantum_protocol must be a mapping")
    return value


def quantum_primary(config: Mapping[str, Any]) -> Dict[str, Any]:
    value=quantum_protocol(config).get("primary",{}) or {}
    if not isinstance(value,dict):
        raise ValueError("quantum_protocol.primary must be a mapping")
    return value


def quantum_benchmark_ablation(config: Mapping[str, Any]) -> Dict[str, Any]:
    value=quantum_protocol(config).get("benchmark_ablation",{}) or {}
    if not isinstance(value,dict):
        raise ValueError("quantum_protocol.benchmark_ablation must be a mapping")
    return value


def quantum_development_sensitivity(config: Mapping[str, Any]) -> Dict[str, Any]:
    value=quantum_protocol(config).get("development_sensitivity",{}) or {}
    if not isinstance(value,dict):
        raise ValueError("quantum_protocol.development_sensitivity must be a mapping")
    return value


def primary_qc_effect_name(baseline: str, budget_mode: str) -> str:
    """Effect name written by batch_benchmark_hard_set --paired-statistics.

    Matched-output contrasts are named after the baseline (``sa``); matched-time
    contrasts carry a ``_time`` suffix (``sa_time``). Keep in sync with
    ``_paired_statistics_main``.
    """
    return str(baseline) if budget_mode == "outputs" else f"{baseline}_time"


def rq5_inference_failures(rq5: dict, min_clusters: int) -> list[str]:
    """Require an estimable, fully reported confirmatory RQ5 result."""
    if not isinstance(rq5,dict):
        return ["RQ5 energy-structure inference payload is not an object"]
    failures=[]
    try:
        clusters=int(rq5.get("n_clusters",0) or 0)
    except (TypeError,ValueError):
        clusters=0
    if clusters<min_clusters:
        failures.append(
            f"RQ5 energy-structure inference has {clusters} clusters; requires >= {min_clusters}"
        )
    missing=[]
    for field in ("spearman_rho","p_value","ci_low","ci_high","p_holm_confirmatory_family"):
        try:
            valid=math.isfinite(float(rq5.get(field)))
        except (TypeError,ValueError):
            valid=False
        if not valid:
            missing.append(field)
    if missing:
        failures.append(
            "RQ5 energy-structure inference is undefined or incomplete "
            f"({', '.join(missing)}); check for constant cluster-level energy/RMSD differences"
        )
    return failures

# ---------------------------------------------------------------------------
# Scripts this orchestrator shells out to. Every one of these is fingerprinted
# (SHA-256) into run_manifest.json for provenance, exactly like every other
# entrypoint in this repository already fingerprints its own code
# dependencies (_ablation_main / _recovery_benchmark_main / build_manifest).
# ---------------------------------------------------------------------------
ORCHESTRATED_SCRIPTS: List[str] = [
    "run_full_experiment.py",
    "resolve_server_config.py",
    "seed_streams.py",
    "audit_all_datasets.py",
    "build_final_pyg_dataset.py",
    "build_independence_cluster_map.py",
    "train_egnn_pruning.py",
    "generate_energy_calibration_dataset.py",
    "batch_benchmark_hard_set.py",
    "run_real_complex_pilot.py",
    "run_external_structure_baselines.py",
    "audit_external_vhh_independence.py",
    "generate_final_research_report.py",
    "analyze_structure_recovery.py",
    "analyze_quantum_scaling.py",
    "model_egnn_pruning.py",
    "subgraph_to_qubo.py",
    "qaoa_interface_sampler.py",
    "evaluate_complex_metrics.py",
    "prediction_contract.py",
    "structural_quality.py",
    # imported by run_real_complex_pilot / generate_energy_calibration_dataset
    # for extract_source, so it is part of the executed code.
    "generate_figure1_pymol_script.py",
    "repo_io.py",
    "sequence_identity.py",
    "safe_graph_load.py",
    "residue_tables.py",
    "paired_statistics.py",
]

STAGE_ORDER: List[str] = [
    "env_check",
    "smoke_check",
    "data_audit",
    "queue_freeze",
    "egnn_train",
    "energy_calibration",
    "method_sensitivity",
    "qc_benchmark",
    "structure_experiment",
    "external_validation",
    "statistics",
    "final_report",
]

# Single source of truth for the stage dependency DAG. Previously this list
# was duplicated inline at every run_stage(...) call site in run_all() below,
# which made it easy for a resume/staleness check to walk only the direct
# prerequisite (one hop) and silently miss a staleness two hops away (e.g.
# data_audit changing without queue_freeze being re-run would not be caught
# when validating egnn_train, since egnn_train's only recorded prerequisite
# is queue_freeze). _validate_upstream_chain_fresh below walks this dict
# recursively instead.
STAGE_PREREQUISITES: Dict[str, List[str]] = {
    "env_check": [],
    "smoke_check": ["env_check"],
    "data_audit": ["env_check", "smoke_check"],
    "queue_freeze": ["data_audit"],
    "egnn_train": ["queue_freeze"],
    "energy_calibration": ["queue_freeze", "egnn_train"],
    "method_sensitivity": ["egnn_train", "energy_calibration"],
    "qc_benchmark": ["egnn_train", "energy_calibration", "method_sensitivity"],
    "structure_experiment": ["queue_freeze", "egnn_train"],
    "external_validation": ["qc_benchmark", "structure_experiment"],
    "statistics": ["qc_benchmark", "structure_experiment", "external_validation"],
    "final_report": ["statistics"],
}
assert list(STAGE_PREREQUISITES) == STAGE_ORDER


# ---------------------------------------------------------------------------
# Small, dependency-free helpers (mirroring the atomic-write / hashing
# patterns already used throughout batch_benchmark_hard_set.py /
# build_final_pyg_dataset.py, kept local here rather than importing private
# helpers across modules).
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=False, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_run_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root),
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def package_versions(names: Sequence[str]) -> Dict[str, Optional[str]]:
    """Best-effort installed-version report; missing packages record None,
    never raise -- this is a diagnostic record, not a hard gate, since
    run_full_experiment.py must not install anything itself."""
    import importlib
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception:
            versions[name] = None
            continue
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _validate_scientific_config(config: Dict[str, Any]) -> None:
    """Fail before any stage when scientific protocol settings are inconsistent."""

    qf = config.get("queue_freeze", {}) or {}
    graph = qf.get("graph_build", {}) or {}
    train = config.get("egnn_train", {}) or {}
    qc = config.get("qc_benchmark", {}) or {}
    structure = config.get("structure_experiment", {}) or {}
    qproto = quantum_protocol(config)
    qprimary = quantum_primary(config)
    qablation = quantum_benchmark_ablation(config)
    qsensitivity = quantum_development_sensitivity(config)
    validation = qf.get("validation_queue", {}) or {}
    clustering = qf.get("independence_clustering", {}) or {}
    if str(qproto.get("algorithm","")) != "xy_qaoa":
        raise ValueError("quantum_protocol.algorithm must be xy_qaoa")
    if str(qproto.get("encoding","")) != "one_hot_rotamer_registers":
        raise ValueError("quantum_protocol.encoding must be one_hot_rotamer_registers")
    if str(qproto.get("mixer","")) != "local_xy":
        raise ValueError("quantum_protocol.mixer must be local_xy")
    if str(qproto.get("initial_state","")) != "wstate":
        raise ValueError(
            "quantum_protocol.initial_state must be wstate (product of local one-hot W states)"
        )
    if str(qproto.get("simulation_scope","")) != "exact_feasible_subspace_classical_simulation":
        raise ValueError(
            "quantum_protocol.simulation_scope must explicitly identify exact feasible-subspace classical simulation"
        )

    legacy_qc_keys={
        "depths","max_evals","qaoa_objective","qaoa_restarts",
        "cvar_alpha","eval_shots","parameter_scale",
    }
    stale_qc=sorted(k for k in legacy_qc_keys if k in qc)
    if stale_qc:
        raise ValueError(
            f"QAOA settings must live only under quantum_protocol; remove qc_benchmark keys {stale_qc}"
        )
    legacy_structure_keys={
        "outputs","max_evals","qaoa_depth","eval_shots","qaoa_restarts",
        "qaoa_objective","cvar_alpha","parameter_scale",
    }
    stale_structure=sorted(k for k in legacy_structure_keys if k in structure)
    if stale_structure:
        raise ValueError(
            f"QAOA settings must live only under quantum_protocol; remove structure_experiment keys {stale_structure}"
        )
    stats_probe=config.get("statistics",{}) or {}
    legacy_stats_keys={
        "primary_depth","primary_max_evals","primary_outputs",
        "primary_objective","primary_restarts",
    }
    stale_stats=sorted(k for k in legacy_stats_keys if k in stats_probe)
    if stale_stats:
        raise ValueError(
            f"QAOA primary settings must live only under quantum_protocol.primary; remove statistics keys {stale_stats}"
        )

    depth=int(qprimary.get("depth",0) or 0)
    if depth not in (1,2,3):
        raise ValueError("quantum_protocol.primary.depth must be one of 1, 2, 3")
    for key in ("max_evals","restarts","eval_shots","output_shots"):
        value=qprimary.get(key)
        if isinstance(value,bool) or value is None or int(value)!=value or int(value)<=0:
            raise ValueError(f"quantum_protocol.primary.{key} must be a positive integer")
    if str(qprimary.get("objective","")) not in ("mean","cvar"):
        raise ValueError("quantum_protocol.primary.objective must be mean or cvar")
    alpha=float(qprimary.get("cvar_alpha",0.0))
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ValueError("quantum_protocol.primary.cvar_alpha must lie in (0,1]")
    if str(qprimary.get("parameter_scale","")) not in ("max_coefficient","feasible_iqr"):
        raise ValueError(
            "quantum_protocol.primary.parameter_scale must be max_coefficient or feasible_iqr"
        )

    objectives=[str(v) for v in qablation.get("objectives",[])]
    restarts=[int(v) for v in qablation.get("restarts",[])]
    if not objectives or any(v not in ("mean","cvar") for v in objectives) or len(objectives)!=len(set(objectives)):
        raise ValueError("quantum_protocol.benchmark_ablation.objectives must be unique mean/cvar values")
    if not restarts or any(v<=0 for v in restarts) or len(restarts)!=len(set(restarts)):
        raise ValueError("quantum_protocol.benchmark_ablation.restarts must be unique positive integers")
    if str(qprimary["objective"]) not in objectives or int(qprimary["restarts"]) not in restarts:
        raise ValueError("quantum_protocol primary objective/restarts must be present in benchmark_ablation")

    qc_sensitivity=qc.get("sensitivity",{}) or {}
    stale_nested=sorted(
        k for k in ("depths","max_evals","eval_shots","cvar_alpha","qaoa_objective","qaoa_restarts")
        if k in qc_sensitivity
    )
    if stale_nested:
        raise ValueError(
            f"Quantum sensitivity axes must live only under quantum_protocol.development_sensitivity; "
            f"remove qc_benchmark.sensitivity keys {stale_nested}"
        )
    if structure.get("robust_qaoa",True) is not True:
        raise ValueError("Formal structure_experiment.robust_qaoa must remain true for the frozen quantum protocol")

    sensitivity_depths=[int(v) for v in qsensitivity.get("depths",[])]
    if not sensitivity_depths or any(v not in (1,2,3) for v in sensitivity_depths):
        raise ValueError("quantum_protocol.development_sensitivity.depths must use supported p in {1,2,3}")
    for key in ("max_evals","eval_shots"):
        values=[int(v) for v in qsensitivity.get(key,[])]
        if not values or any(v<=0 for v in values) or len(values)!=len(set(values)):
            raise ValueError(f"quantum_protocol.development_sensitivity.{key} must be unique positive integers")
    sensitivity_alpha=[float(v) for v in qsensitivity.get("cvar_alpha",[])]
    if (not sensitivity_alpha or any((not math.isfinite(v)) or not 0.0<v<=1.0 for v in sensitivity_alpha)
            or len(sensitivity_alpha)!=len(set(sensitivity_alpha))):
        raise ValueError("quantum_protocol.development_sensitivity.cvar_alpha must be unique values in (0,1]")
    if int(qprimary["depth"]) not in sensitivity_depths:
        raise ValueError("quantum primary depth must be included in development_sensitivity.depths")
    if int(qprimary["max_evals"]) not in [int(v) for v in qsensitivity.get("max_evals",[])]:
        raise ValueError("quantum primary max_evals must be included in development_sensitivity.max_evals")
    if int(qprimary["eval_shots"]) not in [int(v) for v in qsensitivity.get("eval_shots",[])]:
        raise ValueError("quantum primary eval_shots must be included in development_sensitivity.eval_shots")
    if float(qprimary["cvar_alpha"]) not in sensitivity_alpha:
        raise ValueError("quantum primary cvar_alpha must be included in development_sensitivity.cvar_alpha")

    if clustering.get("required", False) and not clustering.get("cluster_map"):
        raise ValueError("queue_freeze.independence_clustering.cluster_map is required")

    homology = qf.get("homology_isolation", {}) or {}
    required_homology = {
        "vhh_full_chain_identity": float(homology.get("vhh_full_chain_identity", 0.80)),
        "cdr_h3_identity": float(homology.get("cdr_h3_identity", 0.50)),
        "antigen_identity": float(homology.get("antigen_identity", 0.30)),
        "antigen_min_length_coverage": float(homology.get("antigen_min_length_coverage", 0.70)),
    }
    if any(not 0.0 < value <= 1.0 for value in required_homology.values()):
        raise ValueError(f"homology_isolation values must lie in (0,1]: {required_homology}")

    positive_graph = {
        "interface_label_cutoff_angstrom": float(graph.get("interface_label_cutoff_angstrom", 5.0)),
        "intra_chain_ca_cutoff_angstrom": float(graph.get("intra_chain_ca_cutoff_angstrom", 8.0)),
    }
    if any((not math.isfinite(v)) or v <= 0 for v in positive_graph.values()):
        raise ValueError(f"Graph distance parameters must be positive finite: {positive_graph}")
    if int(graph.get("cross_partner_knn_k", 3)) < 1:
        raise ValueError("graph_build.cross_partner_knn_k must be >=1")
    if int(graph.get("min_interface_residues", 15)) < 1:
        raise ValueError("graph_build.min_interface_residues must be >=1")

    if not 0.0 <= float(qc.get("antigen_guidance_weight", 0.25)) <= 1.0:
        raise ValueError("qc_benchmark.antigen_guidance_weight must be in [0,1]")
    if not 0.0 <= float(structure.get("antigen_guidance_weight", 0.25)) <= 1.0:
        raise ValueError("structure_experiment.antigen_guidance_weight must be in [0,1]")

    shared_pairs = (
        ("antigen_guidance_weight", 0.25),
        ("antigen_proximity_scale_angstrom", 6.0),
        ("contact_ca_cutoff_angstrom", 8.0),
    )
    qc_rot = qc.get("rotamer_model", {}) or {}
    st_rot = structure.get("rotamer_model", {}) or {}
    rotamer_keys = ("mode", "library_path", "probability_floor", "sigma_offsets")
    for key in rotamer_keys:
        if qc_rot.get(key) != st_rot.get(key):
            raise ValueError(
                f"Rotamer protocol mismatch for {key}: "
                f"qc_benchmark={qc_rot.get(key)!r}, structure_experiment={st_rot.get(key)!r}"
            )
    if qc_rot.get("mode", "dunbrack2010") == "dunbrack2010":
        floor = float(qc_rot.get("probability_floor", 1e-4))
        offsets = qc_rot.get("sigma_offsets", [-1.0, 0.0, 1.0])
        if not 0.0 < floor < 1.0 or not offsets:
            raise ValueError("Invalid Dunbrack probability floor or sigma offsets")
        if any(not math.isfinite(float(v)) for v in offsets):
            raise ValueError("Dunbrack sigma offsets must be finite")

    if clustering.get("required", False):
        pair_tsv=clustering.get("pair_tsv")
        if not clustering.get("cluster_map"):
            raise ValueError("independence_clustering.cluster_map is required")
        if pair_tsv is not None:
            min_score=float(clustering.get("min_score",0.50))
            if not math.isfinite(min_score):
                raise ValueError("independence_clustering.min_score must be finite")
            if str(clustering.get("score_semantics","")).lower() not in ("qtmscore","ttmscore"):
                raise ValueError("independence_clustering.score_semantics must be qtmscore or ttmscore")
            for key in ("query_column","target_column","score_column"):
                if int(clustering.get(key,0)) < 0:
                    raise ValueError(f"independence_clustering.{key} must be nonnegative")

    solvent_model=str(structure.get("solvent_model","vacuum")).lower()
    if solvent_model not in ("vacuum","gbn2"):
        raise ValueError("structure_experiment.solvent_model must be vacuum or gbn2")
    sensitivity=[str(v).lower() for v in structure.get("solvent_sensitivity",[])]
    if any(v not in ("vacuum","gbn2") for v in sensitivity):
        raise ValueError("structure_experiment.solvent_sensitivity supports only vacuum/gbn2")
    if len(sensitivity)!=len(set(sensitivity)):
        raise ValueError("structure_experiment.solvent_sensitivity must not contain duplicates")
    if str(structure.get("perturbation_mode","multi_chi")) not in ("multi_chi","chi1"):
        raise ValueError("structure_experiment.perturbation_mode must be multi_chi or chi1")
    seeds=[int(v) for v in structure.get("seeds",[42,43,44,45,46])]
    if len(seeds)<3 or len(seeds)!=len(set(seeds)) or min(seeds)<0:
        raise ValueError("structure_experiment.seeds must contain >=3 unique nonnegative values")
    if int(qc.get("repeats",10)) < 3:
        raise ValueError("qc_benchmark.repeats must be >=3")
    max_failure_fraction=float(qc.get("max_failure_fraction",0.0))
    if not math.isfinite(max_failure_fraction) or not 0.0 <= max_failure_fraction < 1.0:
        raise ValueError("qc_benchmark.max_failure_fraction must be in [0,1)")

    stats=config.get("statistics",{}) or {}
    if stats.get("cluster_map") not in (None, ""):
        raise ValueError(
            "statistics.cluster_map is obsolete: formal statistics must reuse the "
            "run-local frozen cluster map produced by queue_freeze; remove the key"
        )
    if stats.get("primary_structural_endpoint","final_rmsd") not in (
        "final_rmsd","improvement_vs_input","improvement_vs_relax_only"
    ):
        raise ValueError("Invalid statistics.primary_structural_endpoint")
    if stats.get("primary_structural_contrast","qaoa_vs_sa") not in (
        "qaoa_vs_sa","qaoa_vs_greedy","qaoa_vs_uniform"
    ):
        raise ValueError("Invalid statistics.primary_structural_contrast")
    if int(stats.get("resamples",10000)) < 1000:
        raise ValueError("statistics.resamples must be >=1000")
    if int(stats.get("min_qc_clusters",10)) < 2:
        raise ValueError("statistics.min_qc_clusters must be >=2")
    if int(stats.get("min_scaling_clusters",10)) < 2:
        raise ValueError("statistics.min_scaling_clusters must be >=2")
    if str(stats.get("primary_qc_baseline","sa")) not in ("sa","uniform","greedy"):
        raise ValueError("statistics.primary_qc_baseline must be sa, uniform, or greedy")
    if str(stats.get("primary_qc_metric","gap")) not in (
        "gap","hit","ground_probability","low_energy_mass","low_energy_coverage","entropy"
    ):
        raise ValueError("Invalid statistics.primary_qc_metric")
    if int(stats.get("min_primary_clusters",10)) < 2:
        raise ValueError("statistics.min_primary_clusters must be >=2")
    if int(stats.get("min_rq5_clusters",10)) < 2:
        raise ValueError("statistics.min_rq5_clusters must be >=2")

    external=config.get("external_validation",{}) or {}
    if external.get("required",False):
        ext=external.get("external_vhh",{}) or {}
        structural=external.get("structural_baselines",{}) or {}
        if ext.get("required",False) and not ext.get("graph_dir"):
            raise ValueError("external_validation.external_vhh.graph_dir is required")
        if ext.get("required",False) and not ext.get("source_structure_dir"):
            raise ValueError("external_validation.external_vhh.source_structure_dir is required")
        if ext.get("independence_manifest"):
            # The independence audit is bound to this run's frozen dataset and
            # cluster map, so it is always regenerated inside the run directory.
            raise ValueError(
                "external_validation.external_vhh.independence_manifest is obsolete: the audit "
                "is regenerated as <run_dir>/external_validation/external_vhh_independence_manifest.json; "
                "remove the key")
        if structural.get("required",False):
            if not structural.get("faspr_executable") or not structural.get("phenix_clashscore_executable"):
                raise ValueError("Required structural baseline executables must be configured")

    calibration_cfg = qc.get("energy_calibration", {}) or {}
    mode = str(calibration_cfg.get("mode", "frozen"))
    if mode not in ("frozen", "off"):
        raise ValueError("energy_calibration.mode must be frozen or off")
    ridge_alpha = float(calibration_cfg.get("ridge_alpha", 1.0))
    if not math.isfinite(ridge_alpha) or ridge_alpha < 0:
        raise ValueError("energy_calibration.ridge_alpha must be finite and nonnegative")

    for key, default in shared_pairs:
        left = float(qc.get(key, default))
        right = float(structure.get(key, default))
        if key != "antigen_guidance_weight" and (
            not math.isfinite(left) or left <= 0 or not math.isfinite(right) or right <= 0
        ):
            raise ValueError(f"{key} must be positive finite in coarse and structure protocols")
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"Shared site-selection parameter mismatch for {key}: "
                f"qc_benchmark={left}, structure_experiment={right}"
            )

    qc_depths=[int(qprimary["depth"])]
    qc_sites=[int(v) for v in qc.get("active_sites",[6])]
    if not qc_sites or len(qc_sites)!=len(set(qc_sites)) or any(v<4 or v>10 for v in qc_sites):
        raise ValueError("qc_benchmark.active_sites must contain unique integers in 4..10")
    resolution=(qc.get("sensitivity",{}) or {}).get("rotamer_resolution",{}) or {}
    if resolution:
        site_levels=[int(v) for v in resolution.get("active_sites",[])]
        state_levels=[int(v) for v in resolution.get("states_per_site",[])]
        if (not site_levels or not state_levels or len(site_levels)!=len(set(site_levels))
                or len(state_levels)!=len(set(state_levels))
                or any(site not in (4,5) for site in site_levels)
                or any(state not in (3,4,5,6) for state in state_levels)
                or any(site*state>30 for site in site_levels for state in state_levels)):
            raise ValueError("rotamer_resolution requires unique 4-5 sites and 3-6 states/site within 30 bits")
    primary_pruning=str(stats.get("primary_pruning","egnn"))
    if primary_pruning not in [str(v) for v in qc.get("pruning",["egnn"])]:
        raise ValueError(
            f"statistics.primary_pruning={primary_pruning!r} must be present in "
            f"qc_benchmark.pruning={qc.get('pruning')}")
    primary_radius=float(stats.get("primary_radius",6.0))
    if primary_radius not in [float(v) for v in qc.get("radii",[6.0])]:
        raise ValueError("statistics.primary_radius must be present in qc_benchmark.radii")
    primary_depth=int(qprimary["depth"])
    primary_max_evals=int(qprimary["max_evals"])
    if int(qprimary["output_shots"]) not in [int(v) for v in qc.get("outputs",[1000])]:
        raise ValueError(
            "quantum_protocol.primary.output_shots must be present in qc_benchmark.outputs"
        )
    stats_primary_sites=int(stats.get("primary_active_sites",6))
    if stats_primary_sites not in qc_sites:
        raise ValueError(
            f"statistics.primary_active_sites={stats_primary_sites} must be present in "
            f"qc_benchmark.active_sites={qc_sites}")
    validation_sites = int(validation.get("sites", 6))
    if validation_sites != stats_primary_sites:
        raise ValueError(
            f"Formal structural active-site count must match the primary confirmatory size: "
            f"statistics.primary_active_sites={stats_primary_sites}, "
            f"validation_queue={validation_sites}"
        )

    ff = qc.get("coarse_force_field", {}) or {}
    positive_ff = {
        "cutoff_angstrom": float(ff.get("cutoff_angstrom", 8.0)),
        "softcore_delta_angstrom": float(ff.get("softcore_delta_angstrom", 0.5)),
        "hard_core_fraction": float(ff.get("hard_core_fraction", 0.72)),
        "hard_sphere_penalty": float(ff.get("hard_sphere_penalty", 25.0)),
        "lj_repulsion_cap": float(ff.get("lj_repulsion_cap", 50.0)),
        "lj_attraction_cap": float(ff.get("lj_attraction_cap", 5.0)),
        "coulomb_cap": float(ff.get("coulomb_cap", 20.0)),
        "dielectric_base": float(ff.get("dielectric_base", 4.0)),
        "thermal_energy_kcal": float(ff.get("thermal_energy_kcal", 0.593)),
    }
    if any((not math.isfinite(v)) or v <= 0 for v in positive_ff.values()):
        raise ValueError(f"coarse_force_field positive parameters invalid: {positive_ff}")
    dielectric_slope = float(ff.get("dielectric_slope", 2.0))
    if not math.isfinite(dielectric_slope) or dielectric_slope < 0:
        raise ValueError("coarse_force_field.dielectric_slope must be finite and nonnegative")


def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a top-level YAML mapping")
    _validate_scientific_config(config)
    return config


def resolve_path(config: Dict[str, Any], relative: str) -> Path:
    root = Path(config["paths"]["repo_root"]).resolve()
    candidate = Path(relative)
    return candidate if candidate.is_absolute() else (root / candidate)


def apply_runtime_mode_overrides(config: Dict[str, Any], *, smoke_only: bool) -> Dict[str, Any]:
    import copy
    resolved=copy.deepcopy(config)
    if smoke_only:
        stages=resolved.setdefault("stages",{})
        stages["env_check"]=True
        stages["smoke_check"]=True
    return resolved


# ---------------------------------------------------------------------------
# Stage bookkeeping
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    stage: str
    status: str  # "completed" | "completed_with_failures" | "failed" | "skipped"
    started_utc: str
    finished_utc: str
    returncode: Optional[int]
    detail: str
    argv: List[str] = field(default_factory=list)
    log_path: Optional[str] = None
    artifacts_ok: bool = True

    def to_json(self) -> Dict[str, Any]:
        return dict(
            stage=self.stage, status=self.status, started_utc=self.started_utc,
            finished_utc=self.finished_utc, returncode=self.returncode, detail=self.detail,
            argv=self.argv, log_path=self.log_path, artifacts_ok=self.artifacts_ok,
        )


class Orchestrator:
    def __init__(self, config: Dict[str, Any], run_dir: Path, *, only: Optional[str] = None,
                 smoke_only: bool = False, force_restage: Optional[List[str]] = None):
        self.config = config
        self.run_dir = run_dir
        self.only = only
        self.smoke_only = smoke_only
        self.force_restage = set(force_restage or [])
        self.repo_root = Path(config["paths"]["repo_root"]).resolve()
        self.status_dir = run_dir / "stage_status"
        self.log_dir = run_dir / "logs"
        self.results_manifest_dir = run_dir / "results_manifests"
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_manifest_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = run_dir / "progress.json"
        self.venv_python = self._venv_python()

    # -- environment -----------------------------------------------------
    def _venv_python(self) -> str:
        """Prefer this project's own .venv interpreter (POSIX or Windows
        layout) over whatever `python` happens to resolve to on PATH, since
        every existing script in this repo assumes it is invoked from that
        venv; fall back to sys.executable if no .venv is present (e.g. this
        script itself was already launched from inside an active venv)."""
        posix = self.repo_root / ".venv" / "bin" / "python"
        windows = self.repo_root / ".venv" / "Scripts" / "python.exe"
        if posix.is_file():
            return str(posix)
        if windows.is_file():
            return str(windows)
        return sys.executable

    # -- run-scoped dataset/checkpoint directories (requirement #4) ---------
    def dataset_dir(self) -> Path:
        """This run's own dataset directory (run_dir/<paths.dataset_dir>).

        NEVER a fixed, repo-root-level path shared across runs: two runs
        launched from the same repo/config never overwrite or silently read
        each other's dataset -- each run gets its own directory under its
        own uniquely timestamped run_dir."""
        return self.run_dir / self.config["paths"].get("dataset_dir", "dataset")

    def checkpoint_dir(self) -> Path:
        """This run's own checkpoint directory (run_dir/<paths.checkpoint_dir>);
        see dataset_dir() -- same run-isolation guarantee."""
        return self.run_dir / self.config["paths"].get("checkpoint_dir", "checkpoints")

    def frozen_cluster_map_path(self) -> Path:
        """Run-local family/structure cluster map used by every downstream stage."""
        return self.run_dir/"independence"/"pdb_family_clusters.json"

    # -- status persistence ------------------------------------------------
    def _stage_status_path(self, stage: str) -> Path:
        return self.status_dir / f"{stage}.json"

    def _load_stage_status(self, stage: str) -> Optional[Dict[str, Any]]:
        path = self._stage_status_path(stage)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _save_stage_status(self, result: StageResult) -> None:
        atomic_write_json(self._stage_status_path(result.stage), result.to_json())
        self._update_progress()

    def _update_progress(self) -> None:
        statuses = {}
        for stage in STAGE_ORDER:
            record = self._load_stage_status(stage)
            statuses[stage] = record["status"] if record else "not_started"
        atomic_write_json(self.progress_path, dict(
            updated_utc=utc_timestamp(), run_dir=str(self.run_dir), stages=statuses,
        ))

    # -- subprocess execution ----------------------------------------------
    def _run_subprocess(self, stage: str, argv: Sequence[str], *, cwd: Optional[Path] = None,
                         env: Optional[Dict[str, str]] = None) -> tuple[int, Path]:
        log_path = self.log_dir / f"{stage}.log"
        started = utc_timestamp()
        full_env = dict(os.environ)
        hardware = self.config.get("hardware", {})
        threads = str(hardware.get("cpu_threads_per_process", 2))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            full_env[name] = threads
        full_env["PYTHONUNBUFFERED"] = "1"
        # Stage entry points run as `python -m nanoqc...` from the repository root.
        src_root = str(Path(self.repo_root) / "src")
        full_env["PYTHONPATH"] = os.pathsep.join(
            [src_root, *[p for p in full_env.get("PYTHONPATH", "").split(os.pathsep) if p and p != src_root]])
        full_env["OPENMM_CPU_THREADS"] = str(hardware.get("openmm_cpu_threads", 8))
        full_env["QP_OPENMM_PLATFORM"] = str(hardware.get("openmm_platform", "Reference"))
        full_env["QP_OPENMM_DEVICE"] = str(hardware.get("openmm_device", "0"))
        full_env["QP_OPENMM_PRECISION"] = str(hardware.get("openmm_precision", "double"))
        if env:
            full_env.update(env)
        timeout_seconds=float(self.config.get("control",{}).get("stage_timeout_seconds",0) or 0)
        timeout=timeout_seconds if timeout_seconds>0 else None
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n=== {started} :: {' '.join(argv)} ===\n")
            log_handle.flush()
            use_process_group=os.name=="posix"
            process=subprocess.Popen(
                argv,cwd=str(cwd or self.repo_root),env=full_env,
                stdout=log_handle,stderr=subprocess.STDOUT,
                start_new_session=use_process_group,
            )
            try:
                return process.wait(timeout=timeout),log_path
            except subprocess.TimeoutExpired:
                if use_process_group:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL); process.wait()
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                log_handle.write(
                    f"\n[orchestrator] stage timeout after {timeout_seconds:.3f} seconds; "
                    "subprocess process group terminated.\n")
                log_handle.flush()
                return 124,log_path

    def _stage_result_roots(self, stage: str) -> list[Path]:
        mapping={
            "env_check":[self.run_dir/"env_check.json"],
            "smoke_check":[self.run_dir/"smoke_check"],
            "data_audit":[self.run_dir/"audit"],
            "queue_freeze":[
                self.dataset_dir(),self.run_dir/"validation_queue"/"freeze",
                self.run_dir/"independence",
            ],
            "egnn_train":[self.checkpoint_dir()],
            "energy_calibration":[self.run_dir/"calibration"],
            "method_sensitivity":[self.run_dir/"method_sensitivity"],
            "qc_benchmark":[self.run_dir/"qc_benchmark"],
            "structure_experiment":[
                self.run_dir/"dev_queue",self.run_dir/"validation_queue",
                *sorted(self.run_dir.glob("dev_queue_solvent_*")),
            ],
            "external_validation":[self.run_dir/"external_validation"],
            "statistics":[self.run_dir/"statistics",self.run_dir/"qc_benchmark"],
            "final_report":[self.run_dir/(self.config.get("final_report",{}) or {}).get(
                "filename","FINAL_RESEARCH_REPORT.md")],
        }
        return mapping.get(stage,[])

    def _upstream_fingerprints(self, prerequisites: Sequence[str]) -> Dict[str, Optional[str]]:
        """Content fingerprint of each prerequisite's recorded results (None when absent).

        Hashes the status and the (path, size, sha256) artifact list of the
        prerequisite's results manifest, not its timestamp, so a bit-identical
        re-run does not invalidate downstream results but any changed artifact does.
        """
        fingerprints: Dict[str, Optional[str]] = {}
        for prereq in prerequisites:
            path=self.results_manifest_dir/f"{prereq}.json"
            try:
                manifest=json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            except (OSError,json.JSONDecodeError):
                manifest={"unreadable":True}
            if not isinstance(manifest,dict):
                fingerprints[prereq]=None
                continue
            content=json.dumps({"status":manifest.get("status"),
                                "artifacts":manifest.get("artifacts")},sort_keys=True)
            fingerprints[prereq]=hashlib.sha256(content.encode("utf-8")).hexdigest()
        return fingerprints

    def _validate_upstream_chain_fresh(
        self, stage: str, _seen: Optional[set] = None
    ) -> tuple[bool, str]:
        """Recursively confirm every ancestor of ``stage`` is still fresh.

        ``run_stage``'s existing resume check only compares ``stage``'s own
        direct prerequisites' fingerprints against what was recorded when
        ``stage`` completed -- a re-run of a stage TWO OR MORE hops upstream
        (e.g. data_audit re-run, invalidating queue_freeze's inputs, without
        queue_freeze itself being re-run) would leave queue_freeze's own
        recorded fingerprint of data_audit unchanged only if queue_freeze
        also re-ran; if it did not, queue_freeze's manifest is now stale but
        the direct one-hop check on egnn_train (whose only recorded
        prerequisite is queue_freeze) would not catch it. This walks the
        full ancestor chain via STAGE_PREREQUISITES instead.

        Returns ``(True, "")`` for a stage name not present in
        STAGE_PREREQUISITES (nothing to check) or with no completed manifest
        yet (nothing to compare against -- run_stage's own prerequisite loop
        already enforces the prerequisite has completed before this runs).
        """
        seen = _seen if _seen is not None else set()
        if stage in seen:
            return True, ""
        seen.add(stage)
        if stage not in STAGE_PREREQUISITES:
            return True, ""
        manifest_path = self.results_manifest_dir / f"{stage}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            manifest = None
        if not isinstance(manifest, dict):
            return True, ""
        recorded_upstream = manifest.get("upstream_results_manifest_sha256") or {}
        prereqs = STAGE_PREREQUISITES[stage]
        current_upstream = self._upstream_fingerprints(prereqs)
        if recorded_upstream != current_upstream:
            changed = sorted(
                name for name in set(current_upstream) | set(recorded_upstream)
                if recorded_upstream.get(name) != current_upstream.get(name))
            return False, f"stage '{stage}' upstream changed: {', '.join(changed) or 'unrecorded upstream binding'}"
        for prereq in prereqs:
            ok, detail = self._validate_upstream_chain_fresh(prereq, seen)
            if not ok:
                return False, detail
        return True, ""

    def _write_stage_results_manifest(self, stage: str, result: StageResult,
                                      validation_detail: str,
                                      upstream: Optional[Dict[str, Optional[str]]] = None) -> Path:
        files=[]
        seen=set()
        for root in self._stage_result_roots(stage):
            candidates=[root] if root.is_file() else (list(root.rglob("*")) if root.is_dir() else [])
            for file in candidates:
                if not file.is_file():
                    continue
                try:
                    rel=str(file.relative_to(self.run_dir))
                except ValueError:
                    rel=str(file.resolve())
                if rel in seen:
                    continue
                seen.add(rel)
                size=file.stat().st_size
                row={"path":rel,"size_bytes":size}
                if size<=64*1024*1024:
                    try: row["sha256"]=sha256_of(file)
                    except OSError: row["sha256"]=None
                files.append(row)
        path=self.results_manifest_dir/f"{stage}.json"
        atomic_write_json(path,{
            "stage":stage,
            "status":result.status,
            "generated_utc":utc_timestamp(),
            "validation_detail":validation_detail,
            # Binds this stage's results to the exact upstream results it
            # consumed; a resumed run rejects them if an upstream stage has
            # been re-run since (see run_stage).
            "upstream_results_manifest_sha256":dict(upstream or {}),
            "artifact_count":len(files),
            "artifacts":sorted(files,key=lambda x:x["path"]),
        })
        return path

    # -- generic stage runner ----------------------------------------------
    def run_stage(self, stage: str, prerequisites: Sequence[str],
                  fn: Callable[[], StageResult]) -> StageResult:
        if self.only is not None and stage != self.only:
            existing = self._load_stage_status(stage)
            if existing and existing["status"] in ("completed", "completed_with_failures"):
                return StageResult(stage, "skipped", utc_timestamp(), utc_timestamp(),
                                    None, "Skipped (--only targets a different stage; already completed).")
            return StageResult(stage, "skipped", utc_timestamp(), utc_timestamp(),
                                None, "Skipped (--only targets a different stage; not yet run).")
        if not self.config.get("stages", {}).get(stage, True):
            # Persist an auditable skipped marker. A disabled stage is NOT
            # generally equivalent to completion: only smoke_check is an
            # explicitly optional prerequisite. Scientific dependencies
            # remain fail-closed and cannot be bypassed with a stage toggle.
            result = StageResult(stage, "skipped", utc_timestamp(), utc_timestamp(),
                                  None, "Skipped (disabled in full_experiment_config.yaml).")
            self._save_stage_status(result)
            return result
        for prereq in prerequisites:
            record = self._load_stage_status(prereq)
            optional_skip = bool(
                prereq=="smoke_check" and record and record["status"]=="skipped"
            )
            if not record or (record["status"] not in ("completed", "completed_with_failures") and not optional_skip):
                result = StageResult(stage, "failed", utc_timestamp(), utc_timestamp(), None,
                                      f"Prerequisite stage '{prereq}' has not completed; refusing to start.")
                self._save_stage_status(result)
                return result
            if not optional_skip:
                prereq_ok,prereq_detail=self._validate_completed_stage_artifacts(prereq)
                if not prereq_ok:
                    result=StageResult(
                        stage,"failed",utc_timestamp(),utc_timestamp(),1,
                        f"Prerequisite stage '{prereq}' has a completed marker but failed artifact "
                        f"verification: {prereq_detail}")
                    self._save_stage_status(result)
                    return result
                chain_ok,chain_detail=self._validate_upstream_chain_fresh(prereq)
                if not chain_ok:
                    result=StageResult(
                        stage,"failed",utc_timestamp(),utc_timestamp(),1,
                        f"Prerequisite stage '{prereq}' has a completed marker and intact artifacts, "
                        f"but its upstream chain is stale: {chain_detail}")
                    self._save_stage_status(result)
                    return result
        existing = self._load_stage_status(stage)
        if existing and stage not in self.force_restage:
            if existing["status"] in ("completed", "completed_with_failures"):
                resume_ok,resume_detail=self._validate_completed_stage_artifacts(stage)
                if not resume_ok:
                    result=StageResult(
                        stage,"failed",existing["started_utc"],utc_timestamp(),1,
                        "Resume integrity check failed: "+resume_detail+
                        f" Use --force-restage {stage} after investigating.",
                        existing.get("argv",[]),existing.get("log_path"),False)
                    self._save_stage_status(result)
                    return result
                try:
                    recorded=json.loads((self.results_manifest_dir/f"{stage}.json").read_text(encoding="utf-8"))
                except (OSError,json.JSONDecodeError):
                    recorded=None
                recorded_upstream=(recorded or {}).get("upstream_results_manifest_sha256") if isinstance(recorded,dict) else None
                current_upstream=self._upstream_fingerprints(prerequisites)
                if recorded_upstream!=current_upstream:
                    changed=sorted(
                        name for name in set(current_upstream)|set(recorded_upstream or {})
                        if (recorded_upstream or {}).get(name)!=current_upstream.get(name))
                    result=StageResult(
                        stage,"failed",existing["started_utc"],utc_timestamp(),1,
                        "Resume integrity check failed: upstream stage results changed since this "
                        f"stage completed ({', '.join(changed) or 'unrecorded upstream binding'}). "
                        f"Its results were built on superseded inputs; use --force-restage {stage} "
                        "(with a fresh output) or start a new run.",
                        existing.get("argv",[]),existing.get("log_path"),False)
                    self._save_stage_status(result)
                    return result
                print(f"[{stage}] Already {existing['status']}; verified artifacts and skipping "
                      f"(use --force-restage {stage} to redo).")
                return StageResult(stage, existing["status"], existing["started_utc"],
                                    existing["finished_utc"], existing["returncode"],
                                    "Resumed with artifact verification: "+resume_detail,
                                    existing.get("argv", []), existing.get("log_path"), True)
            if existing["status"] == "failed" and str(existing.get("detail","")).startswith(
                    PREREQUISITE_BLOCK_PREFIX):
                # Refused earlier only because a prerequisite was incomplete; the
                # stage never ran. Prerequisites are verified above, so run it now.
                print(f"[{stage}] Previously blocked by a prerequisite; prerequisites now verified, running.")
            elif existing["status"] == "failed":
                print(f"[{stage}] Previously FAILED systemically: {existing['detail']}")
                print(f"[{stage}] Not auto-retrying. Pass --force-restage {stage} to retry after investigating.")
                return StageResult(stage, "failed", existing["started_utc"], utc_timestamp(),
                                    existing["returncode"], "Not retried: " + existing["detail"])
        print(f"[{stage}] Starting.")
        running=StageResult(
            stage,"running",utc_timestamp(),utc_timestamp(),None,
            "Stage started; terminal status not yet recorded.")
        self._save_stage_status(running)
        try:
            result = fn()
        except Exception:
            result = StageResult(stage, "failed", utc_timestamp(), utc_timestamp(), None,
                                  "Unhandled exception in orchestrator stage function:\n" + traceback.format_exc())
        validation_detail="stage did not claim completion"
        if result.status in ("completed","completed_with_failures"):
            artifacts_ok,validation_detail=self._validate_completed_stage_artifacts(
                stage,require_results_manifest=False)
            if not artifacts_ok:
                result=StageResult(
                    stage,"failed",result.started_utc,utc_timestamp(),1,
                    "Stage returned completion but required experimental results failed validation: "
                    +validation_detail,
                    result.argv,result.log_path,False,
                )
        self._write_stage_results_manifest(
            stage,result,validation_detail,upstream=self._upstream_fingerprints(prerequisites))
        self._save_stage_status(result)
        print(f"[{stage}] {result.status}: {result.detail}")
        return result

    # -- artifact acceptance -------------------------------------------------
    @staticmethod
    def _artifacts_present(paths: Sequence[Path]) -> tuple[bool, str]:
        missing = [str(p) for p in paths if not p.is_file() or p.stat().st_size == 0]
        if missing:
            return False, "Missing or empty expected artifact(s): " + ", ".join(missing)
        return True, "All expected artifacts present and non-empty."

    def _validate_completed_stage_artifacts(
        self, stage: str, *, require_results_manifest: bool = True
    ) -> tuple[bool, str]:
        """Revalidate critical experimental results before trusting completion."""
        def require(paths: Sequence[Path]) -> tuple[bool, str]:
            return self._artifacts_present(paths)

        def read_json(path: Path) -> tuple[Optional[dict], Optional[str]]:
            try:
                value=json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                return None,f"Unreadable JSON {path}: {type(exc).__name__}: {exc}"
            if not isinstance(value,dict):
                return None,f"Expected JSON object at {path}"
            return value,None

        if require_results_manifest and stage!="smoke_check":
            manifest_path=self.results_manifest_dir/f"{stage}.json"
            ok,detail=require([manifest_path])
            if not ok:
                return False,"Missing stage results manifest: "+detail
            manifest,error=read_json(manifest_path)
            if error:return False,error
            if manifest.get("stage")!=stage or manifest.get("status") not in ("completed","completed_with_failures"):
                return False,f"Invalid results manifest status for {stage}: {manifest.get('status')}"
            artifacts=manifest.get("artifacts")
            if (not isinstance(artifacts,list) or not artifacts
                    or manifest.get("artifact_count")!=len(artifacts)):
                return False,f"Invalid or empty results manifest artifact list for {stage}"
            seen_paths=set()
            run_root=self.run_dir.resolve()
            for entry in artifacts:
                if not isinstance(entry,dict) or not isinstance(entry.get("path"),str):
                    return False,f"Malformed results manifest artifact for {stage}"
                rel=Path(entry["path"])
                if rel.is_absolute() or entry["path"] in seen_paths:
                    return False,f"Unsafe or duplicate results manifest path for {stage}: {rel}"
                seen_paths.add(entry["path"])
                path=(run_root/rel).resolve()
                if not path.is_relative_to(run_root) or not path.is_file():
                    return False,f"Results manifest artifact missing/outside run: {rel}"
                expected_size=entry.get("size_bytes")
                if type(expected_size) is not int or path.stat().st_size!=expected_size:
                    return False,f"Results manifest artifact size mismatch: {rel}"
                expected_sha=entry.get("sha256")
                if expected_sha is not None:
                    if not isinstance(expected_sha,str) or sha256_of(path)!=expected_sha:
                        return False,f"Results manifest artifact sha256 mismatch: {rel}"
                elif expected_size<=64*1024*1024:
                    return False,f"Results manifest lacks sha256 for small artifact: {rel}"

        if stage=="smoke_check":
            summary=self.run_dir/"smoke_check"/"smoke_summary.json"
            return require([summary]) if summary.exists() else (True,"optional smoke skipped before scientific stages")
        if stage=="env_check":
            return require([self.run_dir/"env_check.json"])
        if stage=="data_audit":
            audit_root=self.run_dir/"audit"
            required=[audit_root/name for name in (
                "data_audit_report.md","data_audit_details.csv","data_audit_details.jsonl",
                "data_audit_inventory.json","data_audit_db55_pairs.json")]
            ok,detail=require(required)
            if not ok:return ok,detail
            inventory,error=read_json(audit_root/"data_audit_inventory.json")
            if error:return False,error
            if inventory.get("partial_run"):
                return False,"Formal data audit is marked partial_run=true"
            expected=int(inventory.get("tasks",-1))
            observed=sum(
                1 for line in (audit_root/"data_audit_details.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()
            )
            if expected<0 or observed!=expected:
                return False,f"Data-audit denominator mismatch: discovered={expected}, audited_rows={observed}"
            return True,f"data audit closed exactly over {observed} discovered structures"
        if stage=="queue_freeze":
            dataset=self.dataset_dir()
            freeze=self.run_dir/"validation_queue"/"freeze"
            cluster_path=self.frozen_cluster_map_path()
            cluster_provenance=cluster_path.with_suffix(".provenance.json")
            universe=self.run_dir/"audit"/"cluster_universe.txt"
            ok,detail=require([
                dataset/"graph_manifest.csv",dataset/"graph_manifest.json",
                dataset/"run_summary.json",dataset/"graph_dataset_delivery_report.md",
                dataset/"cdr3_clusters.json",dataset/"excluded_samples.csv",
                dataset/"processing_failures.csv",
                cluster_path,cluster_provenance,universe,
                freeze/"selected_targets.json",freeze/"eligibility.json",freeze/"run_manifest.json",
                freeze/"freeze_manifest.json",
            ])
            if not ok:return ok,detail
            summary,error=read_json(dataset/"run_summary.json")
            if error:return False,error
            if not summary.get("complete"):
                return False,"dataset run_summary.json is not complete"
            # Verify frozen-input hashes before trusting the contents of
            # selected_targets.json (a tampered file must report provenance
            # mismatch, not whatever its forged targets happen to lack).
            freeze_manifest,error=read_json(freeze/"freeze_manifest.json")
            if error:return False,error
            expected_pairs={
                "selected_targets_sha256": freeze/"selected_targets.json",
                "eligibility_sha256": freeze/"eligibility.json",
                "graph_manifest_sha256": dataset/"graph_manifest.csv",
            }
            for key,path in expected_pairs.items():
                expected=freeze_manifest.get(key)
                if not expected or sha256_of(path)!=expected:
                    return False,f"Frozen validation provenance mismatch for {key}: {path}"
            try:
                frozen=json.loads((freeze/"selected_targets.json").read_text(encoding="utf-8"))
            except Exception as exc:
                return False,f"Unreadable frozen selected_targets.json: {exc}"
            if not isinstance(frozen,list) or not frozen:
                return False,"Frozen validation selected_targets.json is empty or invalid"
            for row in frozen:
                pdb=str((row or {}).get("target","")).strip().lower()
                if not pdb:
                    return False,"Frozen validation selected target lacks target identity"
                recovery_manifest=freeze/"prepared"/pdb/"recovery_manifest.json"
                if not recovery_manifest.is_file():
                    # prepare-only layout may store prepared targets one level above freeze.
                    recovery_manifest=self.run_dir/"validation_queue"/"freeze"/"prepared"/pdb/"recovery_manifest.json"
                ok,detail=require([recovery_manifest])
                if not ok:
                    return False,f"Frozen target {pdb} lacks recovery_manifest.json: {detail}"
            cluster_setting=((self.config.get("queue_freeze",{}) or {}).get("independence_clustering",{}) or {}).get("cluster_map")
            expected_cluster=freeze_manifest.get("cluster_map_sha256")
            if cluster_setting:
                cluster_path=self.frozen_cluster_map_path()
                if not cluster_path.is_file():
                    return False,f"Run-local frozen cluster map missing: {cluster_path}"
                if expected_cluster!=sha256_of(cluster_path):
                    return False,"Run-local frozen cluster map sha256 mismatch"
                cluster_provenance=cluster_path.with_suffix(".provenance.json")
                expected_prov=freeze_manifest.get("cluster_map_provenance_sha256")
                if not cluster_provenance.is_file() or expected_prov!=sha256_of(cluster_provenance):
                    return False,"Frozen cluster-map provenance sha256 mismatch"
                if ((self.config.get("queue_freeze",{}) or {}).get("independence_clustering",{}) or {}).get("pair_tsv"):
                    provenance,error=read_json(cluster_provenance)
                    if error:return False,error
                    if int(provenance.get("skipped_rows",-1))!=0:
                        return False,"Frozen cluster-map provenance reports malformed pair rows"
                universe=self.run_dir/"audit"/"cluster_universe.txt"
                expected_universe=freeze_manifest.get("cluster_universe_sha256")
                if not universe.is_file() or expected_universe!=sha256_of(universe):
                    return False,"Frozen clustering universe sha256 mismatch"
            return True,"queue-freeze artifacts and frozen-input hashes verified"
        if stage=="egnn_train":
            checkpoint=self.checkpoint_dir()/"best_egnn_pruning.pt"
            summary_path=self.checkpoint_dir()/"training_summary.json"
            ok,detail=require([
                checkpoint,summary_path,
                self.checkpoint_dir()/"geometry_baseline.json",
                self.checkpoint_dir()/"egnn_training_history.csv",
                self.checkpoint_dir()/"egnn_training_summary.md",
            ])
            if not ok:return ok,detail
            summary,error=read_json(summary_path)
            if error:return False,error
            if summary.get("status")!="complete":
                return False,"training_summary.json status is not complete"
            checkpoint_info=summary.get("checkpoint") or {}
            if checkpoint_info.get("strict_reload_verified") is not True:
                return False,"training_summary.json does not record strict_reload_verified=true"
            expected_sha=checkpoint_info.get("sha256")
            if not expected_sha:
                return False,"training_summary.json has no checkpoint sha256"
            if sha256_of(checkpoint)!=expected_sha:
                return False,"EGNN checkpoint sha256 mismatch"
            return True,"EGNN checkpoint and training summary verified"
        if stage=="energy_calibration":
            qc=self.config.get("qc_benchmark",{}) or {}
            cal=qc.get("energy_calibration",{}) or {}
            training=self.run_dir/cal.get("training_csv","calibration/coarse_to_amber_train.csv")
            calibration=self.run_dir/cal.get("calibration_file","calibration/coarse_to_amber.json")
            provenance=training.with_suffix(".provenance.json")
            ok,detail=require([
                training,provenance,calibration,
                self.run_dir/"calibration"/"calibration_report.md",
            ])
            if not ok:return ok,detail
            payload,error=read_json(calibration)
            if error:return False,error
            limits=cal.get("acceptance",{}) or {}
            checks=[
                ("cv_rmse_kcal","max_cv_rmse_kcal",lambda value,limit:value<=limit),
                ("cv_mae_kcal","max_cv_mae_kcal",lambda value,limit:value<=limit),
                ("cv_r2","min_cv_r2",lambda value,limit:value>=limit),
                ("cv_spearman","min_cv_spearman",lambda value,limit:value>=limit),
                ("calibration_rmse_improvement_kcal","min_rmse_improvement_kcal",
                    lambda value,limit:value>=limit),
            ]
            for metric,key,predicate in checks:
                if key not in limits:
                    continue
                value=payload.get(metric)
                limit=float(limits[key])
                if value is None or not math.isfinite(float(value)) or not predicate(float(value),limit):
                    return False,f"Calibration resume acceptance failed: {metric}={value}, {key}={limit}"
            if limits.get("require_family_grouped_cv",False) and payload.get("cv_grouping")!="family_cluster":
                return False,"Calibration resume requires family_cluster grouped CV"
            if int(payload.get("n_train_complexes",0) or 0)<int(limits.get("min_train_complexes",0)):
                return False,"Calibration resume has insufficient training complexes"
            if int(payload.get("n_train_groups",0) or 0)<int(limits.get("min_train_groups",0)):
                return False,"Calibration resume has insufficient family groups"
            observed_folds=int(payload.get("cv_fold_count",len(payload.get("cv_folds",[]) or [])) or 0)
            if observed_folds<int(limits.get("min_cv_folds",0)):
                return False,"Calibration resume has insufficient CV folds"
            return True,"energy calibration artifacts and acceptance thresholds verified"
        if stage=="method_sensitivity":
            qc=self.config.get("qc_benchmark",{}) or {}
            cfg=qc.get("sensitivity",{}) or {}
            qsensitivity=quantum_development_sensitivity(self.config)
            missing=[]
            for shots in qsensitivity.get("eval_shots",[200,500,1000]):
                for alpha in qsensitivity.get("cvar_alpha",[0.05,0.1,0.25,0.5,1.0]):
                    sub=self.run_dir/"method_sensitivity"/f"shots_{shots}_alpha_{str(alpha).replace('.','p')}"
                    required=[
                        sub/"run_manifest.json",sub/"run_summary.json",sub/"metrics.csv",
                        sub/"summary.md",sub/"failed_case_keys.json",sub/"seed_streams.json",
                    ]
                    ok,detail=require(required)
                    if not ok:
                        missing.append(detail);continue
                    summary_path=sub/"run_summary.json"
                    summary,error=read_json(summary_path)
                    if (error or not summary.get("closed")
                            or int(summary.get("failures_total",0) or 0)!=0):
                        missing.append(str(summary_path))
                        continue
                    case_count=len(list((sub/"cases").glob("*.json")))
                    if case_count!=int(summary.get("cases_completed_total",0) or 0):
                        missing.append(
                            f"{sub}: case count mismatch files={case_count}, "
                            f"summary={summary.get('cases_completed_total')}")
            aggregate=[
                self.run_dir/"method_sensitivity"/"sensitivity_summary.csv",
                self.run_dir/"method_sensitivity"/"sensitivity_summary.json",
                self.run_dir/"method_sensitivity"/"sensitivity_summary.md",
            ]
            ok,detail=require(aggregate)
            if not ok: missing.append(detail)
            resolution=(cfg.get("rotamer_resolution",{}) or {})
            if resolution:
                root=self.run_dir/"method_sensitivity"/"rotamer_resolution"
                ok,detail=require([root/"summary.json",root/"summary.csv",root/"summary.md"])
                if not ok: missing.append(detail)
                for sites in resolution.get("active_sites",[4,5]):
                    for states in resolution.get("states_per_site",[3,4,5,6]):
                        sub=root/f"sites_{sites}_states_{states}"
                        ok,detail=require([sub/"run_manifest.json",sub/"run_summary.json",sub/"metrics.csv"])
                        if not ok:
                            missing.append(detail);continue
                        summary,error=read_json(sub/"run_summary.json")
                        if error or not summary.get("closed") or int(summary.get("failures_total",0) or 0)!=0:
                            missing.append(str(sub/"run_summary.json"))
            if missing:
                return False,"Sensitivity sub-runs/results missing/not closed: "+", ".join(missing[:10])
            return True,"all sensitivity sub-runs and aggregate results are closed"
        if stage=="qc_benchmark":
            root=self.run_dir/"qc_benchmark"
            ok,detail=require([
                root/"run_manifest.json",root/"run_summary.json",root/"metrics.csv",
                root/"summary.md",root/"failed_case_keys.json",root/"seed_streams.json",
            ])
            if not ok:return ok,detail
            summary,error=read_json(root/"run_summary.json")
            if error:return False,error
            if not summary.get("closed") or int(summary.get("cases_completed_total",0) or 0)<=0:
                return False,"qc_benchmark run_summary.json is not closed with completed cases"
            failed=int(summary.get("failures_total",0) or 0)
            planned=int(summary.get("total_cases_planned",0) or 0)
            allowed=float((self.config.get("qc_benchmark",{}) or {}).get("max_failure_fraction",0.0))
            if planned<=0 or failed<0 or failed/planned>allowed:
                return False,(
                    f"qc_benchmark failure fraction {failed}/{planned} exceeds "
                    f"max_failure_fraction={allowed}")
            case_count=len(list((root/"cases").glob("*.json")))
            if case_count!=int(summary.get("cases_completed_total",0) or 0):
                return False,f"qc_benchmark case artifact count mismatch: files={case_count}, summary={summary.get('cases_completed_total')}"
            return True,"qc_benchmark metrics/report/cases and closure verified"
        if stage=="structure_experiment":
            dev_root=self.run_dir/"dev_queue"
            val_root=self.run_dir/"validation_queue"
            dev=dev_root/"run_summary.json"
            validation=val_root/"run_summary.json"
            ok,detail=require([
                dev_root/"run_manifest.json",dev_root/"seed_streams.json",
                dev_root/"eligibility.json",dev_root/"selected_targets.json",
                dev,dev_root/"real_complex_metrics.csv",dev_root/"real_complex_report.md",
                val_root/"run_manifest.json",val_root/"seed_streams.json",
                val_root/"eligibility.json",val_root/"selected_targets.json",
                val_root/"run_summary.json",val_root/"real_complex_metrics.csv",
                val_root/"real_complex_report.md",
            ])
            if not ok:return ok,detail
            dev_summary,error=read_json(dev)
            if error:return False,error
            val_summary,error=read_json(validation)
            if error:return False,error
            if not dev_summary.get("closed"):
                return False,"dev_queue run_summary.json is not closed"
            if not val_summary.get("closed") or val_summary.get("frozen_set_accounting_ok") is not True:
                return False,"validation execution is not closed against frozen denominator"
            if val_summary.get("structure_experiment_failed_targets"):
                return False,"validation execution has failed targets in the confirmatory queue"
            expected_seeds={str(int(v)) for v in self.config.get("structure_experiment",{}).get(
                "seeds",[42,43,44,45,46])}
            expected_methods={"qaoa","sa","uniform","greedy"}
            for root,summary,label in (
                (dev_root,dev_summary,"dev_queue"),
                (val_root,val_summary,"validation_queue"),
            ):
                for pdb in summary.get("structure_experiment_completed_target_ids",[]) or []:
                    target=(root/str(pdb)/"results"/str(pdb)) if label=="dev_queue" else (root/"results"/str(pdb))
                    metrics_path=target/"recovery_metrics.csv"
                    ok,detail=require([
                        target/"run_manifest.json",metrics_path,target/"recovery_report.md",
                    ])
                    if not ok:
                        return False,f"{label}/{pdb} missing completed-target results: {detail}"
                    with metrics_path.open(newline="",encoding="utf-8") as handle:
                        rows=list(csv.DictReader(handle))
                    observed=[(str(r.get("seed","")),str(r.get("method",""))) for r in rows]
                    expected={(seed,method) for seed in expected_seeds for method in expected_methods}
                    if len(observed)!=len(set(observed)):
                        return False,f"{label}/{pdb} contains duplicate seed/method structural rows"
                    if set(observed)!=expected:
                        missing=sorted(expected-set(observed))
                        extra=sorted(set(observed)-expected)
                        return False,(
                            f"{label}/{pdb} structural denominator mismatch: "
                            f"missing={missing[:10]}, extra={extra[:10]}")
            for solvent in [
                str(v) for v in self.config.get("structure_experiment",{}).get("solvent_sensitivity",[])
                if str(v)!=str(self.config.get("structure_experiment",{}).get("solvent_model","vacuum"))
            ]:
                solvent_root=self.run_dir/f"dev_queue_solvent_{solvent}"
                ok,detail=require([
                    solvent_root/"run_summary.json",
                    solvent_root/"recovery_metrics.csv",
                    solvent_root/"recovery_report.md",
                ])
                if not ok:
                    return False,f"Missing development solvent-sensitivity results: {detail}"
                solvent_summary,error=read_json(solvent_root/"run_summary.json")
                if error:return False,error
                if not solvent_summary.get("closed") or solvent_summary.get("failed_target_ids"):
                    return False,f"Development solvent sensitivity {solvent} is not closed/clean"
            return True,"structural aggregate, per-target denominators, and sensitivity results verified"
        if stage=="external_validation":
            cfg=self.config.get("external_validation",{}) or {}
            paths=[]
            ext=cfg.get("external_vhh",{}) or {}
            if ext.get("required",False):
                root=self.run_dir/"external_validation"/"vhh_coarse"
                paths += [
                    root/"run_manifest.json",root/"run_summary.json",root/"metrics.csv",
                    root/"summary.md",root/"seed_streams.json",
                    root/"statistics_outputs.json",root/"statistics_outputs.md",
                    self.run_dir/"external_validation"/"external_vhh_independence_manifest.json",
                ]
            structural=cfg.get("structural_baselines",{}) or {}
            if structural.get("required",False):
                root=self.run_dir/"external_validation"/"structural_baselines"
                paths += [
                    root/"run_summary.json",root/"external_baseline_metrics.csv",
                    root/"external_baseline_report.md",
                ]
            ok,detail=require(paths)
            if not ok:return ok,detail
            for path in [p for p in paths if p.name=="run_summary.json"]:
                summary,error=read_json(path)
                if error:return False,error
                if summary.get("closed") is False or summary.get("failures"):
                    return False,f"External validation summary not closed/clean: {path}"
                if "failures_total" in summary and int(summary.get("failures_total",0) or 0)!=0:
                    return False,f"External validation summary reports failed cases: {path}"
                if path.parent.name=="vhh_coarse" and "cases_completed_total" in summary:
                    case_count=len(list((path.parent/"cases").glob("*.json")))
                    if case_count!=int(summary.get("cases_completed_total",0) or 0):
                        return False,(
                            f"External VHH case count mismatch: files={case_count}, "
                            f"summary={summary.get('cases_completed_total')}")
            return True,"external validation raw/aggregate/statistical artifacts verified"
        if stage=="statistics":
            root=self.run_dir/"qc_benchmark"
            cfg=self.config.get("statistics",{}) or {}
            modes=cfg.get("budget_modes",["outputs","time"])
            paths=[
                root/f"statistics_{mode}.{suffix}"
                for mode in modes for suffix in ("json","md")
            ]
            paths += [
                self.run_dir/"statistics"/"quantum_scaling_statistics.json",
                self.run_dir/"statistics"/"quantum_scaling_statistics.md",
                self.run_dir/"statistics"/"structure_statistics.json",
                self.run_dir/"statistics"/"structure_statistics.md",
            ]
            ok,detail=require(paths)
            if not ok:return ok,detail
            for mode in modes:
                paired,error=read_json(root/f"statistics_{mode}.json")
                if error:return False,error
                exclusions=paired.get("exclusions") or {}
                denominator_failures=paired_denominator_failures(exclusions,mode)
                if denominator_failures:
                    return False,(f"{mode} primary paired-statistics denominator is incomplete: "
                                  f"{denominator_failures}")
                effect=next((entry for entry in paired.get("effects",[])
                             if entry.get("baseline")==primary_qc_effect_name(
                                 cfg.get("primary_qc_baseline","sa"),mode)
                             and entry.get("metric")==str(cfg.get("primary_qc_metric","gap"))),None)
                clusters=0 if effect is None else int(effect.get("n_clusters",0) or 0)
                if clusters<int(cfg.get("min_qc_clusters",10)):
                    return False,f"{mode} primary coarse contrast has insufficient clusters: {clusters}"
            scaling,error=read_json(self.run_dir/"statistics"/"quantum_scaling_statistics.json")
            if error:return False,error
            scaling_clusters=int((scaling.get("primary") or {}).get("n_clusters",0) or 0)
            if scaling_clusters<int(cfg.get("min_scaling_clusters",10)):
                return False,f"Scaling inference has insufficient clusters: {scaling_clusters}"
            structure,error=read_json(self.run_dir/"statistics"/"structure_statistics.json")
            if error:return False,error
            primary_clusters=int((structure.get("primary") or {}).get("n_clusters",0) or 0)
            if primary_clusters<int(cfg.get("min_primary_clusters",10)):
                return False,f"Primary structural inference has insufficient clusters: {primary_clusters}"
            min_rq5=int(cfg.get("min_rq5_clusters",10))
            failures=rq5_inference_failures(structure.get("rq5") or {},min_rq5)
            if failures:return False,"; ".join(failures)
            return True,"statistics artifacts and all formal inference gates verified"
        if stage=="final_report":
            filename=(self.config.get("final_report",{}) or {}).get(
                "filename","FINAL_RESEARCH_REPORT.md")
            return require([self.run_dir/filename])
        return False,f"No resume artifact policy defined for stage {stage!r}"

    # ================================================================
    # Stage 0: environment check
    # ================================================================
    def stage_env_check(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config.get("env_check", {})
        required = cfg.get("required_python_packages", [])
        versions = package_versions(required)
        missing_packages = [name for name, version in versions.items() if version is None]
        missing_resources: list[str] = []
        checks: dict[str, Any] = {}

        # Every orchestrated source file is part of the frozen executable protocol.
        for name in ORCHESTRATED_SCRIPTS:
            path=repo_path(name, self.repo_root)
            checks[f"script:{name}"]=path.is_file()
            if not path.is_file():
                missing_resources.append(str(path))

        qc=self.config.get("qc_benchmark", {}) or {}
        rot=qc.get("rotamer_model", {}) or {}
        if rot.get("mode","dunbrack2010")=="dunbrack2010":
            library=resolve_path(self.config,rot.get("library_path","data/rotamer/ALL.bbdep.rotamers.lib"))
            checks["dunbrack_library"]=library.is_file()
            if not library.is_file():
                missing_resources.append(str(library))

        clustering=((self.config.get("queue_freeze", {}) or {}).get("independence_clustering", {}) or {})
        if clustering.get("required",False):
            cluster_map=resolve_path(self.config,clustering.get("cluster_map","")) if clustering.get("cluster_map") else None
            pairs=resolve_path(self.config,clustering.get("pair_tsv","")) if clustering.get("pair_tsv") else None
            available=bool((cluster_map and cluster_map.is_file()) or (pairs and pairs.is_file()))
            checks["independence_cluster_input"]=available
            if not available:
                missing_resources.append(f"cluster_map_or_pair_tsv:{cluster_map}|{pairs}")

        external=self.config.get("external_validation", {}) or {}
        if external.get("required",False):
            ext=external.get("external_vhh", {}) or {}
            if ext.get("required",False):
                graph_dir=resolve_path(self.config,ext.get("graph_dir",""))
                source_dir=resolve_path(self.config,ext.get("source_structure_dir",""))
                graph_ok=graph_dir.is_dir() and any(graph_dir.glob("*.pt"))
                checks["external_vhh_graphs"]=graph_ok
                checks["external_vhh_raw_structures"]=source_dir.is_dir()
                # The independence manifest is generated run-locally during
                # external_validation, after queue_freeze has frozen this run's
                # training dataset and cluster map; it is never read from the repo.
                if not graph_ok:
                    missing_resources.append(str(graph_dir))
                if not source_dir.is_dir():
                    missing_resources.append(str(source_dir))
            structural=external.get("structural_baselines", {}) or {}
            if structural.get("required",False):
                faspr=Path(structural.get("faspr_executable",""))
                phenix=Path(structural.get("phenix_clashscore_executable",""))
                checks["faspr_executable"]=faspr.is_file()
                checks["phenix_clashscore_executable"]=phenix.is_file()
                if not faspr.is_file():
                    missing_resources.append(str(faspr))
                if not phenix.is_file():
                    missing_resources.append(str(phenix))

        # GBN2 is a declared development-only sensitivity dependency.
        gbn2_error=None
        solvent_models=(self.config.get("structure_experiment", {}) or {}).get("solvent_sensitivity",[])
        if "gbn2" in [str(v).lower() for v in solvent_models] and "openmm" not in missing_packages:
            try:
                from openmm import app as _openmm_app
                _openmm_app.ForceField("amber14-all.xml","implicit/gbn2.xml")
                checks["openmm_gbn2_parameters"]=True
            except Exception as exc:
                checks["openmm_gbn2_parameters"]=False
                gbn2_error=f"{type(exc).__name__}: {exc}"
                missing_resources.append("OpenMM implicit/gbn2.xml")

        missing_resources=sorted(set(missing_resources))
        record = dict(
            python=sys.version, platform=platform.platform(),
            git_commit=git_commit_hash(self.repo_root),
            package_versions=versions, missing_packages=missing_packages,
            resource_checks=checks, missing_resources=missing_resources,
            gbn2_error=gbn2_error,
        )
        atomic_write_json(self.run_dir / "env_check.json", record)
        failed=bool(missing_packages or missing_resources)
        status = "failed" if failed else "completed"
        detail = (
            f"Missing packages={missing_packages}; missing resources={missing_resources}"
            if failed else "Python dependencies and all declared formal external resources verified."
        )
        return StageResult(
            "env_check", status, started, utc_timestamp(), 1 if failed else 0,
            detail, artifacts_ok=not failed
        )

    # ================================================================
    # Stage 1: smoke check -- tiny fast pass through each real entrypoint,
    # in its OWN throwaway directory, never mixed into the real run's
    # results ("检查结果与正式实验分开保存").
    # ================================================================
    def stage_smoke_check(self) -> StageResult:
        started = utc_timestamp()
        smoke_cfg = self.config.get("smoke_check", {})
        smoke_dir = self.run_dir / "smoke_check"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        checks: List[tuple[str, List[str]]] = []
        smoke_rotamer_library = resolve_path(
            self.config,
            (self.config.get("qc_benchmark", {}).get("rotamer_model", {}) or {}).get(
                "library_path", "data/rotamer/ALL.bbdep.rotamers.lib"
            ),
        )
        if not smoke_rotamer_library.is_file():
            return StageResult(
                "smoke_check", "failed", started, utc_timestamp(), None,
                f"Required Dunbrack library missing: {smoke_rotamer_library}",
            )

        smoke_input_dir = self.dataset_dir() / self.config["qc_benchmark"]["input_dir"]
        if smoke_input_dir.is_dir() and any(smoke_input_dir.glob("*.pt")):
            checks.append(("qc_benchmark_smoke", [
                self.venv_python, "-m", module_name("batch_benchmark_hard_set.py"), "--research-ablation",
                "--input-dir", str(smoke_input_dir),
                "--checkpoint", str(self.checkpoint_dir() / self.config["qc_benchmark"]["checkpoint"]),
                "--out-dir", str(smoke_dir / "qc_benchmark"),
                "--max-targets", str(smoke_cfg.get("ablation_max_targets", 1)),
                "--pruning", "egnn", "contact", "cdr", "random",
                "--seeds", "42",
                "--radii", "6",
                "--depths", "2",
                "--max-evals", "12",
                "--active-sites", "5",
                "--outputs", "20",
                "--sa-passes", "5",
                "--greedy-passes", "5",
                "--rotamer-mode", "dunbrack2010",
                "--rotamer-library", str(smoke_rotamer_library),
            ]))
        else:
            print("[smoke_check] qc_benchmark input_dir not yet built; skipping that sub-check "
                  "(expected before queue_freeze has run).")

        smoke_target = smoke_cfg.get("recovery_pilot_pdb_id")
        smoke_argv = [
            self.venv_python, "-m", module_name("run_real_complex_pilot.py"),
            "--out-dir", str(smoke_dir / "real_complex"),
            "--targets", str(smoke_cfg.get("recovery_pilot_targets", 1)),
            "--sites", str(smoke_cfg.get("recovery_pilot_sites", 6)),
            "--seeds", *[str(s) for s in smoke_cfg.get("recovery_pilot_seeds", [42])],
            "--max-evals", str(smoke_cfg.get("recovery_pilot_max_evals", 12)),
            "--outputs", str(smoke_cfg.get("recovery_pilot_outputs", 20)),
            "--pruning", "contact",  # avoid requiring a trained checkpoint for the smoke check
            "--rotamer-mode", "dunbrack2010",
            "--rotamer-library", str(smoke_rotamer_library),
        ]
        if smoke_target:
            smoke_argv += ["--pdb-id", str(smoke_target)]
        checks.append(("real_complex_smoke", smoke_argv))

        failures = []
        for name, argv in checks:
            returncode, log_path = self._run_subprocess(f"smoke_{name}", argv)
            if returncode != 0:
                failures.append(f"{name} exited {returncode} (see {log_path})")
        status = "completed" if not failures else "failed"
        detail = "All smoke checks passed." if not failures else "; ".join(failures)
        atomic_write_json(smoke_dir/"smoke_summary.json",{
            "status":status,
            "checks":[{"name":name,"argv":argv} for name,argv in checks],
            "failures":failures,
            "closed":True,
        })
        return StageResult("smoke_check", status, started, utc_timestamp(), 0 if not failures else 1, detail)

    # ================================================================
    # Stage 2: data audit
    # ================================================================
    def stage_data_audit(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config.get("data_audit", {})
        argv = [
            self.venv_python, "-m", module_name("audit_all_datasets.py"),
            "--data", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--workers", str(cfg.get("workers", 4)),
            "--limit", str(cfg.get("limit", 0)),
            "--out", str(self.run_dir / "audit"),
            "--max-resolution", str(cfg.get("max_resolution_angstrom", 3.0)),
            "--min-interface-occupancy", str(cfg.get("min_interface_occupancy", 0.90)),
        ]
        if cfg.get("allow_interface_altloc", False):
            argv.append("--allow-interface-altloc")
        if cfg.get("allow_unknown_resolution", False):
            argv.append("--allow-unknown-resolution")
        if cfg.get("allow_incomplete_interface_sidechains", False):
            argv.append("--allow-incomplete-interface-sidechains")
        returncode, log_path = self._run_subprocess("data_audit", argv)
        expected = [self.run_dir / "audit" / name for name in
                    ("data_audit_report.md", "data_audit_details.csv",
                     "data_audit_details.jsonl", "data_audit_inventory.json", "data_audit_db55_pairs.json")]
        ok, artifact_detail = self._artifacts_present(expected)
        if returncode != 0 or not ok:
            return StageResult("data_audit", "failed", started, utc_timestamp(), returncode,
                                f"audit_all_datasets.py exited {returncode}; {artifact_detail} (see {log_path})",
                                argv, str(log_path), ok)
        return StageResult("data_audit", "completed", started, utc_timestamp(), returncode,
                            artifact_detail, argv, str(log_path), ok)

    # ================================================================
    # Stage 3: queue freeze + isolation (no cap) + graph construction,
    # THEN the new frozen, blind validation-target queue.
    # ================================================================
    def stage_queue_freeze(self) -> StageResult:
        started = utc_timestamp()
        streams = derive_streams(self.config["master_seed"])
        qf_cfg = self.config["queue_freeze"]
        dataset_dir = self.dataset_dir()

        # 3a. Uncapped, deduplicated, isolated graph construction / split.
        graph_argv = [
            self.venv_python, "-m", module_name("build_final_pyg_dataset.py"),
            "--out", str(dataset_dir),
            "--workers", str(qf_cfg["graph_build"].get("workers", 2)),
            "--audit-dir", str(self.run_dir / "audit"),
            "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--vhh-identity-threshold", str(qf_cfg["homology_isolation"].get("vhh_full_chain_identity", 0.80)),
            "--cdr-h3-identity-threshold", str(qf_cfg["homology_isolation"].get("cdr_h3_identity", 0.50)),
            "--antigen-identity-threshold", str(qf_cfg["homology_isolation"].get("antigen_identity", 0.30)),
            "--antigen-min-length-coverage", str(qf_cfg["homology_isolation"].get("antigen_min_length_coverage", 0.70)),
            "--interface-label-cutoff", str(qf_cfg["graph_build"].get("interface_label_cutoff_angstrom", 5.0)),
            "--intra-chain-ca-cutoff", str(qf_cfg["graph_build"].get("intra_chain_ca_cutoff_angstrom", 8.0)),
            "--cross-partner-knn-k", str(qf_cfg["graph_build"].get("cross_partner_knn_k", 3)),
            "--min-interface-residues", str(qf_cfg["graph_build"].get("min_interface_residues", 15)),
        ]
        clustering_cfg = qf_cfg.get("independence_clustering", {}) or {}
        source_cluster_map_path = (
            resolve_path(self.config, clustering_cfg.get("cluster_map", ""))
            if clustering_cfg.get("cluster_map") else None
        )
        cluster_map_path=self.frozen_cluster_map_path()
        cluster_map_path.parent.mkdir(parents=True,exist_ok=True)
        pair_setting=clustering_cfg.get("pair_tsv")
        pair_path=resolve_path(self.config,pair_setting) if pair_setting else None
        # Preserve singleton structures even when the external pair table omits
        # self hits: derive a frozen PDB universe from this run's audit ledger.
        universe_path=self.run_dir/"audit"/"cluster_universe.txt"
        audit_jsonl=self.run_dir/"audit"/"data_audit_details.jsonl"
        if audit_jsonl.is_file() and not universe_path.is_file():
            ids=set()
            for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                pdb=str(row.get("pdb_id","")).strip().lower()
                # Only real four-character PDB IDs can be structurally clustered;
                # audited non-PDB files (e.g. CAPRI models named "T37_...") fall
                # back to name[:4] in the audit and are never study graphs.
                if re.fullmatch(r"[a-z0-9]{4}",pdb):
                    ids.add(pdb)
            # External VHH structures are part of the SAME frozen structural
            # similarity universe. They must not be appended as untracked
            # singleton clusters only at external-validation time.
            external_cfg=(self.config.get("external_validation",{}) or {}).get("external_vhh",{}) or {}
            if external_cfg.get("required",False):
                external_graph_dir=resolve_path(self.config,external_cfg.get("graph_dir",""))
                external_source_dir=resolve_path(self.config,external_cfg.get("source_structure_dir",""))
                if not external_graph_dir.is_dir():
                    return StageResult(
                        "queue_freeze","failed",started,utc_timestamp(),None,
                        f"External VHH graph directory required for frozen clustering universe: {external_graph_dir}")
                if not external_source_dir.is_dir():
                    return StageResult(
                        "queue_freeze","failed",started,utc_timestamp(),None,
                        f"External VHH raw-structure directory required: {external_source_dir}")
                from nanoqc.data.audit_external_vhh_independence import graph_sequences
                external_ids=set()
                for graph_path in sorted(external_graph_dir.glob("*.pt")):
                    pdb=graph_sequences(graph_path,external_source_dir)["pdb_id"]
                    if not pdb:
                        return StageResult(
                            "queue_freeze","failed",started,utc_timestamp(),None,
                            f"External graph lacks PDB identity: {graph_path}")
                    external_ids.add(pdb)
                if not external_ids:
                    return StageResult(
                        "queue_freeze","failed",started,utc_timestamp(),None,
                        f"No external VHH graphs found for frozen clustering universe: {external_graph_dir}")
                ids.update(external_ids)
            universe_path.write_text("\n".join(sorted(ids))+"\n",encoding="utf-8")

        if clustering_cfg.get("required",False) and pair_path is not None and pair_path.is_file():
            universe_ids={
                line.strip().lower() for line in universe_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            query_col=int(clustering_cfg.get("query_column",0))
            target_col=int(clustering_cfg.get("target_column",1))
            covered_ids=set()
            for raw_line in pair_path.read_text(encoding="utf-8-sig").splitlines():
                line=raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                fields=line.split("\t")
                if len(fields)<=max(query_col,target_col):
                    continue
                for idx in (query_col,target_col):
                    token=Path(fields[idx].strip()).name
                    lower=token.lower()
                    for suffix in (".cif.gz",".pdb.gz",".cif",".pdb",".mmcif"):
                        if lower.endswith(suffix):
                            token=token[:-len(suffix)]
                            break
                    if token:
                        covered_ids.add(token[:4].lower() if len(token)>=4 else token.lower())
            missing_pair_coverage=sorted(universe_ids-covered_ids)
            if missing_pair_coverage:
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Frozen structure-similarity pair table does not demonstrate query/target coverage "
                    f"for the complete internal+external universe: {missing_pair_coverage[:20]}"
                )

        if not cluster_map_path.is_file():
            if pair_path is not None and pair_path.is_file():
                cluster_argv=[
                    self.venv_python,"-m", module_name("build_independence_cluster_map.py"),
                    "--pairs",str(pair_path),"--out-json",str(cluster_map_path),
                    "--min-score",str(clustering_cfg.get("min_score",0.50)),
                    "--query-column",str(clustering_cfg.get("query_column",0)),
                    "--target-column",str(clustering_cfg.get("target_column",1)),
                    "--score-column",str(clustering_cfg.get("score_column",2)),
                    "--score-semantics",str(clustering_cfg.get("score_semantics","unspecified")),
                ]
                if universe_path.is_file():
                    cluster_argv += ["--universe",str(universe_path)]
                cluster_rc,cluster_log=self._run_subprocess("build_independence_cluster_map",cluster_argv)
                if cluster_rc!=0 or not cluster_map_path.is_file():
                    return StageResult(
                        "queue_freeze","failed",started,utc_timestamp(),cluster_rc,
                        f"Family/structure cluster-map generation failed; see {cluster_log}",
                        cluster_argv,str(cluster_log),False,
                    )
            elif source_cluster_map_path is not None and source_cluster_map_path.is_file():
                shutil.copy2(source_cluster_map_path,cluster_map_path)
                source_prov=source_cluster_map_path.with_suffix(".provenance.json")
                if source_prov.is_file():
                    shutil.copy2(source_prov,cluster_map_path.with_suffix(".provenance.json"))
            elif clustering_cfg.get("required",False):
                return StageResult(
                    "queue_freeze", "failed", started, utc_timestamp(), None,
                    f"Required cluster map missing and no usable frozen pair TSV/source map is available: "
                    f"source_map={source_cluster_map_path}, pairs={pair_path}",
                )

        # A pre-existing map is accepted only if its provenance binds it to
        # the configured frozen pair table and score threshold.
        if cluster_map_path is not None and cluster_map_path.is_file() and pair_path is not None:
            provenance_path=cluster_map_path.with_suffix(".provenance.json")
            if not provenance_path.is_file():
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    f"Cluster map lacks provenance: {provenance_path}"
                )
            cluster_prov=json.loads(provenance_path.read_text(encoding="utf-8"))
            expected_pair_sha=sha256_of(pair_path) if pair_path.is_file() else None
            if cluster_prov.get("source_pairs_sha256") != expected_pair_sha:
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Cluster-map provenance does not match configured pair TSV"
                )
            if not math.isclose(
                float(cluster_prov.get("min_score",float("nan"))),
                float(clustering_cfg.get("min_score",0.50)),
                rel_tol=0.0,abs_tol=1e-12
            ):
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Cluster-map provenance min_score does not match frozen config"
                )
            for key in ("query_column","target_column","score_column"):
                if int(cluster_prov.get(key,-1)) != int(clustering_cfg.get(key,{"query_column":0,"target_column":1,"score_column":2}[key])):
                    return StageResult(
                        "queue_freeze","failed",started,utc_timestamp(),None,
                        f"Cluster-map provenance {key} does not match frozen config"
                    )
            if str(cluster_prov.get("score_semantics","")) != str(clustering_cfg.get("score_semantics","unspecified")):
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Cluster-map score semantics do not match frozen config"
                )
            if (cluster_prov.get("score_header_validated") is not True
                    or cluster_prov.get("score_field") != str(clustering_cfg.get("score_semantics",""))):
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Cluster-map provenance lacks a validated TM-score column header"
                )
            if int(cluster_prov.get("skipped_rows",-1)) != 0:
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Cluster-map provenance reports malformed pair rows"
                )
            if universe_path.is_file():
                universe_sha=sha256_of(universe_path)
                if cluster_prov.get("universe_sha256") != universe_sha:
                    return StageResult(
                        "queue_freeze","failed",started,utc_timestamp(),None,
                        "Cluster-map provenance is not bound to this run's audited PDB universe"
                    )
        if cluster_map_path is not None and cluster_map_path.is_file() and universe_path.is_file():
            cluster_map_payload=json.loads(cluster_map_path.read_text(encoding="utf-8"))
            required_universe={
                line.strip().lower() for line in universe_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            missing_universe=sorted(required_universe-set(str(k).lower() for k in cluster_map_payload))
            if missing_universe:
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    f"Frozen cluster map does not cover the complete internal+external universe: "
                    f"{missing_universe[:20]}"
                )
        if clustering_cfg.get("required", False) and (
            cluster_map_path is None or not cluster_map_path.is_file()
        ):
            return StageResult(
                "queue_freeze","failed",started,utc_timestamp(),None,
                f"Required family/domain/structure cluster map missing: {cluster_map_path}",
            )
        if cluster_map_path is not None and cluster_map_path.is_file():
            graph_argv += ["--cluster-map", str(cluster_map_path)]
        if qf_cfg["graph_build"].get("no_cap", True):
            graph_argv += ["--no-cap", "--partition-seed", str(streams["partition"])]
        else:
            graph_argv += ["--target-hard", str(qf_cfg["graph_build"].get("target_hard_if_capped", 500)),
                            "--partition-seed", str(streams["partition"])]
        if (dataset_dir / "run_summary.json").is_file():
            graph_argv.append("--resume")
        returncode, graph_log = self._run_subprocess("queue_freeze_graph_build", graph_argv)
        graph_expected = [dataset_dir / name for name in
                           ("graph_manifest.csv", "graph_manifest.json",
                            "run_summary.json", "graph_dataset_delivery_report.md")]
        graph_ok, graph_detail = self._artifacts_present(graph_expected)
        if returncode != 0 or not graph_ok:
            return StageResult("queue_freeze", "failed", started, utc_timestamp(), returncode,
                                f"build_final_pyg_dataset.py exited {returncode}; {graph_detail} (see {graph_log})",
                                graph_argv, str(graph_log), graph_ok)

        # 3b. Frozen, blind validation-target queue (explicitly excludes the
        # historical dev queue; seeded-random, never "smallest first").
        #
        # (requirement #1/#3, corrected) `vq_cfg["pruning"]` ("egnn") is the
        # REAL, final validation protocol -- it is never silently downgraded
        # here. But queue_freeze runs BEFORE egnn_train in STAGE_ORDER, so
        # this bootstrap call cannot use a trained checkpoint yet. Since
        # eligibility (PDB overlap / chain identity / CDR-H3 identity /
        # structural viability such as "enough chemically movable
        # VHH candidate sites") does NOT depend on pruning strategy -- only the
        # final residue ranking within an already-eligible target does --
        # this bootstrap call uses `eligibility_bootstrap_pruning` (a cheap,
        # checkpoint-free strategy, e.g. "contact") ONLY to decide TRUE
        # target membership; it is refused if ever misconfigured to "egnn"
        # (that would reintroduce the not-yet-trained-weights dependency).
        # The exact same frozen set is then reproduced explicitly in
        # stage_structure_experiment via --pdb-allowlist-file (never by
        # re-derivation alone), where REAL site selection happens with the
        # actual `pruning` ("egnn") and the by-then-trained checkpoint.
        vq_cfg = qf_cfg["validation_queue"]
        dev_cfg = qf_cfg["dev_queue"]
        # Target membership is frozen without any residue-ranking strategy.
        # --eligibility-only verifies only that enough chemically movable,
        # Dunbrack/Amber-compatible VHH sites exist. Formal EGNN ranking is
        # deferred until stage_structure_experiment, after training.
        bootstrap_pruning = vq_cfg.get("eligibility_bootstrap_pruning", "contact")
        validation_seed = save_derived_child(streams, "perturb", "validation_queue_selection_order")
        validation_root = self.run_dir / "validation_queue"
        validation_dir = validation_root / "freeze"
        vq_argv = [
            self.venv_python, "-m", module_name("run_real_complex_pilot.py"),
            "--dataset", str(dataset_dir),
            "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--out-dir", str(validation_dir),
            "--targets", str(vq_cfg.get("target_count", 0)),
            "--sites", str(vq_cfg.get("sites", 6)),
            "--vhh-identity-threshold", str(qf_cfg["homology_isolation"].get("vhh_full_chain_identity", 0.80)),
            "--cdr-h3-identity-threshold", str(qf_cfg["homology_isolation"].get("cdr_h3_identity", 0.50)),
            "--antigen-identity-threshold", str(qf_cfg["homology_isolation"].get("antigen_identity", 0.30)),
            "--antigen-min-length-coverage", str(qf_cfg["homology_isolation"].get("antigen_min_length_coverage", 0.70)),
            "--antigen-proximity-scale", str(self.config.get("structure_experiment", {}).get("antigen_proximity_scale_angstrom", 6.0)),
            "--contact-ca-cutoff", str(self.config.get("structure_experiment", {}).get("contact_ca_cutoff_angstrom", 8.0)),
            "--rotamer-mode", str(self.config.get("structure_experiment", {}).get("rotamer_model", {}).get("mode", "dunbrack2010")),
            "--rotamer-library", str(resolve_path(self.config, self.config.get("structure_experiment", {}).get("rotamer_model", {}).get("library_path", "data/rotamer/ALL.bbdep.rotamers.lib"))),
            "--rotamer-probability-floor", str(self.config.get("structure_experiment", {}).get("rotamer_model", {}).get("probability_floor", 1e-4)),
            "--rotamer-sigma-offsets", *[str(v) for v in self.config.get("structure_experiment", {}).get("rotamer_model", {}).get("sigma_offsets", [-1.0,0.0,1.0])],
            "--seeds", str(streams["perturb"]),
            "--master-seed", str(self.config["master_seed"]),
            "--pruning", bootstrap_pruning,
            "--eligibility-only",
            "--exclude-pdb", *dev_cfg.get("excluded_pdb", []),
            "--dev-exposed-pdb", *dev_cfg.get("excluded_pdb", []),
            "--selection-order", vq_cfg.get("selection_order", "seeded_random"),
            "--selection-seed", str(validation_seed),
            "--queue-role", "validation",
            "--prepare-only",
        ]
        if cluster_map_path is not None and cluster_map_path.is_file():
            vq_argv += ["--cluster-map", str(cluster_map_path)]
        returncode, vq_log = self._run_subprocess("queue_freeze_validation_queue", vq_argv)
        vq_expected = [validation_dir / name for name in ("eligibility.json", "selected_targets.json")]
        vq_ok, vq_detail = self._artifacts_present(vq_expected)
        if returncode != 0 or not vq_ok:
            return StageResult("queue_freeze", "failed", started, utc_timestamp(), returncode,
                                f"run_real_complex_pilot.py (validation queue) exited {returncode}; {vq_detail} (see {vq_log})",
                                vq_argv, str(vq_log), vq_ok)
        selected_path=validation_dir/"selected_targets.json"
        eligibility_path=validation_dir/"eligibility.json"
        freeze_manifest_path=validation_dir/"freeze_manifest.json"
        cluster_provenance_path=cluster_map_path.with_suffix(".provenance.json")
        freeze_manifest=dict(
            schema_version=2,
            selected_targets_sha256=sha256_of(selected_path),
            eligibility_sha256=sha256_of(eligibility_path),
            graph_manifest_sha256=sha256_of(dataset_dir/"graph_manifest.csv"),
            cluster_map_sha256=(sha256_of(cluster_map_path) if cluster_map_path.is_file() else None),
            cluster_map_provenance_sha256=(
                sha256_of(cluster_provenance_path) if cluster_provenance_path.is_file() else None),
            cluster_universe_sha256=(
                sha256_of(universe_path) if universe_path.is_file() else None),
        )
        atomic_write_json(freeze_manifest_path,freeze_manifest)
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        cap_label = vq_cfg.get("target_count", 0) or "unlimited (all qualifying targets)"
        detail = (f"Graph build: {graph_detail} Validation queue: {len(selected)} targets frozen "
                  f"(cap: {cap_label}; dev queue excluded+exposure-flagged: {dev_cfg.get('excluded_pdb', [])}; "
                  f"selected_targets_sha256={freeze_manifest['selected_targets_sha256']}). {vq_detail}")
        return StageResult("queue_freeze", "completed", started, utc_timestamp(), 0, detail,
                            graph_argv + ["&&"] + vq_argv, f"{graph_log};{vq_log}", True)

    # ================================================================
    # Stage 4: EGNN training
    # ================================================================
    def stage_egnn_train(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config["egnn_train"]
        homology = self.config["queue_freeze"]["homology_isolation"]
        dataset_dir = self.dataset_dir()
        checkpoint_dir = self.checkpoint_dir()
        streams = derive_streams(self.config["master_seed"])
        train_data_dir = dataset_dir / "graphs" / "train"
        graphs = list(train_data_dir.glob("*.pt")) if train_data_dir.is_dir() else []
        argv = [
            self.venv_python, "-m", module_name("train_egnn_pruning.py"),
            "--data-dir", str(train_data_dir),
            "--expected-graphs", str(len(graphs)),
            "--checkpoint-dir", str(checkpoint_dir),
            "--max-epochs", str(cfg.get("max_epochs", 50)),
            "--patience", str(cfg.get("patience", 5)),
            "--batch-size", str(cfg.get("batch_size", 2)),
            "--hidden-dim", str(cfg.get("hidden_dim", 32)),
            "--num-layers", str(cfg.get("num_layers", 4)),
            "--dropout", str(cfg.get("dropout", 0.1)),
            "--coord-scale", str(cfg.get("coord_scale", 0.1)),
            "--geometry-baseline-contact-cutoff", str(cfg.get("geometry_baseline_contact_cutoff_angstrom", 8.0)),
            "--geometry-baseline-proximity-scale", str(cfg.get("geometry_baseline_proximity_scale_angstrom", 6.0)),
            "--learning-rate", str(cfg.get("learning_rate", 1e-3)),
            "--weight-decay", str(cfg.get("weight_decay", 1e-5)),
            "--gradient-clip", str(cfg.get("gradient_clip", 5.0)),
            "--vhh-identity-threshold", str(homology.get("vhh_full_chain_identity", 0.80)),
            "--cdr-h3-identity-threshold", str(homology.get("cdr_h3_identity", 0.50)),
            "--antigen-identity-threshold", str(homology.get("antigen_identity", 0.30)),
            "--antigen-min-length-coverage", str(homology.get("antigen_min_length_coverage", 0.70)),
            "--device", cfg.get("device", "auto"),
            "--threads", str(cfg.get("threads", 16)),
            "--num-workers", str(cfg.get("num_workers", 8)),
            "--seed", str(streams["train"]),
            "--amp" if cfg.get("amp", True) else "--no-amp",
            "--pin-memory" if cfg.get("pin_memory", True) else "--no-pin-memory",
        ]
        ranks = int(cfg.get("nproc_per_node", 1))
        if ranks < 1:
            raise ValueError("egnn_train.nproc_per_node must be positive")
        if ranks > 1:
            argv[1:1] = ["-m", "torch.distributed.run", "--standalone",
                         "--nnodes=1", f"--nproc_per_node={ranks}"]
        if (checkpoint_dir / "last_egnn_pruning.pt").is_file():
            argv += ["--resume"]
        returncode, log_path = self._run_subprocess("egnn_train", argv)
        expected = [checkpoint_dir / name for name in ("best_egnn_pruning.pt", "training_summary.json")]
        ok, detail = self._artifacts_present(expected)
        status = "completed" if (returncode == 0 and ok) else "failed"
        return StageResult("egnn_train", status, started, utc_timestamp(), returncode,
                            f"{detail} (train stream seed {streams['train']}, {len(graphs)} training graphs)",
                            argv, str(log_path), ok)

    # ================================================================
    # Stage 5: TRAIN-only coarse-to-Amber energy calibration
    # ================================================================
    def stage_energy_calibration(self) -> StageResult:
        started = utc_timestamp()
        qc_cfg = self.config["qc_benchmark"]
        cal_cfg = qc_cfg.get("energy_calibration", {}) or {}
        rot_cfg = qc_cfg.get("rotamer_model", {}) or {}
        ff = qc_cfg.get("coarse_force_field", {}) or {}
        training_csv = self.run_dir / cal_cfg.get("training_csv", "calibration/coarse_to_amber_train.csv")
        calibration_file = self.run_dir / cal_cfg.get("calibration_file", "calibration/coarse_to_amber.json")
        provenance = training_csv.with_suffix(".provenance.json")
        training_csv.parent.mkdir(parents=True, exist_ok=True)
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        rotamer_library = resolve_path(
            self.config, rot_cfg.get("library_path", "data/rotamer/ALL.bbdep.rotamers.lib")
        )
        if not rotamer_library.is_file():
            return StageResult(
                "energy_calibration", "failed", started, utc_timestamp(), None,
                f"Required Dunbrack library missing: {rotamer_library}",
            )
        if cal_cfg.get("selection_mode","egnn")=="egnn":
            checkpoint=self.checkpoint_dir()/qc_cfg.get("checkpoint","best_egnn_pruning.pt")
            if not checkpoint.is_file():
                return StageResult(
                    "energy_calibration","failed",started,utc_timestamp(),None,
                    f"EGNN-selected calibration requires trained checkpoint: {checkpoint}",
                )
        argv = [
            self.venv_python, "-m", module_name("generate_energy_calibration_dataset.py"),
            "--dataset", str(self.dataset_dir()),
            "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--rotamer-library", str(rotamer_library),
            "--selection-mode", str(cal_cfg.get("selection_mode","egnn")),
            "--checkpoint", str(self.checkpoint_dir()/qc_cfg.get("checkpoint","best_egnn_pruning.pt")),
            "--vhh-identity-threshold", str(self.config["queue_freeze"]["homology_isolation"].get("vhh_full_chain_identity",0.80)),
            "--cdr-h3-identity-threshold", str(self.config["queue_freeze"]["homology_isolation"].get("cdr_h3_identity",0.50)),
            "--antigen-identity-threshold", str(self.config["queue_freeze"]["homology_isolation"].get("antigen_identity",0.30)),
            "--antigen-min-length-coverage", str(self.config["queue_freeze"]["homology_isolation"].get("antigen_min_length_coverage",0.70)),
            "--antigen-guidance-weight", str(qc_cfg.get("antigen_guidance_weight",0.25)),
            "--out-csv", str(training_csv),
            "--out-provenance", str(provenance),
            "--assignments-per-complex", str(cal_cfg.get("assignments_per_complex", 64)),
            "--active-sites", str(cal_cfg.get("active_sites", qc_cfg.get("active_sites", 6))),
            "--radius", str(cal_cfg.get("radius_angstrom", qc_cfg.get("radii", [6.0])[0])),
            "--seed", str(derive_streams(self.config["master_seed"])["partition"]),
            "--antigen-proximity-scale", str(qc_cfg.get("antigen_proximity_scale_angstrom", 6.0)),
            "--contact-ca-cutoff", str(qc_cfg.get("contact_ca_cutoff_angstrom", 8.0)),
            "--nonbonded-cutoff", str(ff.get("cutoff_angstrom", 8.0)),
            "--softcore-delta", str(ff.get("softcore_delta_angstrom", 0.5)),
            "--hard-core-fraction", str(ff.get("hard_core_fraction", 0.72)),
            "--hard-sphere-penalty", str(ff.get("hard_sphere_penalty", 25.0)),
            "--lj-repulsion-cap", str(ff.get("lj_repulsion_cap", 50.0)),
            "--lj-attraction-cap", str(ff.get("lj_attraction_cap", 5.0)),
            "--coulomb-cap", str(ff.get("coulomb_cap", 20.0)),
            "--dielectric-base", str(ff.get("dielectric_base", 4.0)),
            "--dielectric-slope", str(ff.get("dielectric_slope", 2.0)),
            "--thermal-energy-kcal", str(ff.get("thermal_energy_kcal", 0.593)),
            "--rotamer-probability-floor", str(rot_cfg.get("probability_floor", 1e-4)),
            "--rotamer-sigma-offsets", *[str(v) for v in rot_cfg.get("sigma_offsets", [-1.0,0.0,1.0])],
            "--solvent-model", str((self.config.get("structure_experiment",{}) or {}).get("solvent_model","vacuum")),
        ]
        cluster_path=self.frozen_cluster_map_path()
        if cluster_path is not None:
            if not cluster_path.is_file():
                return StageResult(
                    "energy_calibration","failed",started,utc_timestamp(),None,
                    f"Calibration requires frozen family/structure cluster map: {cluster_path}",
                )
            argv += ["--cluster-map", str(cluster_path)]
        max_complexes = int(cal_cfg.get("max_complexes", 0))
        if max_complexes:
            argv += ["--max-complexes", str(max_complexes)]
        returncode, log_path = self._run_subprocess("energy_calibration_dataset", argv)
        if returncode != 0 or not training_csv.is_file() or not provenance.is_file():
            return StageResult(
                "energy_calibration", "failed", started, utc_timestamp(), returncode,
                f"Calibration dataset generation failed; see {log_path}", argv, str(log_path), False,
            )

        generation=json.loads(provenance.read_text(encoding="utf-8"))
        limits=cal_cfg.get("acceptance", {}) or {}
        generation_failure_fraction=float(generation.get("generation_failure_fraction",1.0))
        max_failure_fraction=float(limits.get("max_generation_failure_fraction",1.0))
        if not math.isfinite(generation_failure_fraction) or generation_failure_fraction > max_failure_fraction:
            return StageResult(
                "energy_calibration","failed",started,utc_timestamp(),None,
                f"Calibration row generation failure fraction {generation_failure_fraction:.4f} "
                f"exceeds {max_failure_fraction:.4f}",
                argv,str(log_path),False,
            )
        if cluster_path is not None:
            expected_cluster_sha=sha256_of(cluster_path)
            if generation.get("cluster_map_sha256") != expected_cluster_sha:
                return StageResult(
                    "energy_calibration","failed",started,utc_timestamp(),None,
                    "Calibration dataset provenance is not bound to the frozen cluster map",
                    argv,str(log_path),False,
                )

        fit_argv = [
            self.venv_python, "-m", module_name("batch_benchmark_hard_set.py"), "--research-ablation",
            "--fit-energy-calibration-csv", str(training_csv),
            "--fit-energy-calibration-out", str(calibration_file),
            "--calibration-ridge-alpha", str(cal_cfg.get("ridge_alpha", 1.0)),
        ]
        fit_returncode, fit_log = self._run_subprocess("energy_calibration_fit", fit_argv)
        ok = fit_returncode == 0 and calibration_file.is_file() and calibration_file.stat().st_size > 0
        acceptance_detail = ""
        if ok:
            payload=json.loads(calibration_file.read_text(encoding="utf-8"))
            limits=cal_cfg.get("acceptance", {}) or {}
            checks=[
                ("cv_rmse_kcal","max_cv_rmse_kcal",lambda value,limit:value<=limit),
                ("cv_mae_kcal","max_cv_mae_kcal",lambda value,limit:value<=limit),
                ("cv_r2","min_cv_r2",lambda value,limit:value>=limit),
                ("cv_spearman","min_cv_spearman",lambda value,limit:value>=limit),
                ("calibration_rmse_improvement_kcal","min_rmse_improvement_kcal",
                    lambda value,limit:value>=limit),
            ]
            failed_checks=[]
            min_train_complexes=int(limits.get("min_train_complexes",0))
            min_train_groups=int(limits.get("min_train_groups",0))
            min_cv_folds=int(limits.get("min_cv_folds",0))
            if limits.get("require_family_grouped_cv",False) and payload.get("cv_grouping")!="family_cluster":
                failed_checks.append(
                    f"cv_grouping={payload.get('cv_grouping')} but family_cluster grouping is required"
                )
            observed_complexes=int(payload.get("n_train_complexes",0) or 0)
            observed_groups=int(payload.get("n_train_groups",0) or 0)
            observed_folds=int(payload.get("cv_fold_count",len(payload.get("cv_folds",[]) or [])) or 0)
            if observed_complexes < min_train_complexes:
                failed_checks.append(
                    f"n_train_complexes={observed_complexes} violates min_train_complexes={min_train_complexes}"
                )
            if observed_groups < min_train_groups:
                failed_checks.append(
                    f"n_train_groups={observed_groups} violates min_train_groups={min_train_groups}"
                )
            if observed_folds < min_cv_folds:
                failed_checks.append(
                    f"cv_fold_count={observed_folds} violates min_cv_folds={min_cv_folds}"
                )
            for metric,key,predicate in checks:
                if key not in limits:
                    continue
                value=payload.get(metric)
                limit=float(limits[key])
                if value is None or not math.isfinite(float(value)) or not predicate(float(value),limit):
                    failed_checks.append(f"{metric}={value} violates {key}={limit}")
            if failed_checks:
                ok=False
                acceptance_detail="; ".join(failed_checks)
        calibration_report=self.run_dir/"calibration"/"calibration_report.md"
        calibration_report.parent.mkdir(parents=True,exist_ok=True)
        payload_for_report={}
        if calibration_file.is_file():
            try: payload_for_report=json.loads(calibration_file.read_text(encoding="utf-8"))
            except Exception: payload_for_report={}
        calibration_report.write_text("\n".join([
            "# Energy calibration result",
            "",
            f"Status: {'accepted' if ok else 'failed'}",
            f"Training CSV: {training_csv}",
            f"Calibration JSON: {calibration_file}",
            f"Training complexes: {payload_for_report.get('n_train_complexes','n/a')}",
            f"Training family groups: {payload_for_report.get('n_train_groups','n/a')}",
            f"CV folds: {payload_for_report.get('cv_fold_count','n/a')}",
            f"CV RMSE (kcal/mol): {payload_for_report.get('cv_rmse_kcal','n/a')}",
            f"CV MAE (kcal/mol): {payload_for_report.get('cv_mae_kcal','n/a')}",
            f"CV R2: {payload_for_report.get('cv_r2','n/a')}",
            f"CV Spearman: {payload_for_report.get('cv_spearman','n/a')}",
            f"RMSE improvement (kcal/mol): {payload_for_report.get('calibration_rmse_improvement_kcal','n/a')}",
            "",
            f"Acceptance detail: {acceptance_detail or 'all configured acceptance gates passed'}",
        ])+"\n",encoding="utf-8")
        return StageResult(
            "energy_calibration", "completed" if ok else "failed", started, utc_timestamp(),
            fit_returncode,
            (f"Training-only calibration frozen and accepted at {calibration_file}" if ok
             else f"Calibration failed acceptance: {acceptance_detail or 'fit/artifact failure'}; see {fit_log}"),
            fit_argv, str(fit_log), ok,
        )

    # ================================================================
    # Stage 6: development-only method sensitivity (never hard/validation data)
    # ================================================================
    def stage_method_sensitivity(self) -> StageResult:
        started=utc_timestamp()
        qc=self.config["qc_benchmark"]
        cfg=qc.get("sensitivity", {}) or {}
        qprimary=quantum_primary(self.config)
        qsensitivity=quantum_development_sensitivity(self.config)
        homology=self.config["queue_freeze"]["homology_isolation"]
        rot=qc.get("rotamer_model", {}) or {}
        ff=qc.get("coarse_force_field", {}) or {}
        calibration=self.run_dir/(qc.get("energy_calibration", {}) or {}).get(
            "calibration_file","calibration/coarse_to_amber.json")
        out=self.run_dir/"method_sensitivity"
        repeats=[derive_child_seed(
            derive_streams(self.config["master_seed"])["perturb"],"sensitivity_repeat",str(i))
            for i in range(int(cfg.get("repeats",3)))]
        target_selection_seed=derive_child_seed(
            derive_streams(self.config["master_seed"])["partition"],
            "method_sensitivity_target_subset"
        )
        # eval_shots and CVaR alpha are separate axes; batch driver accepts one
        # of each per invocation, so run a frozen Cartesian set of sub-runs.
        failures=[];logs=[];argvs=[]
        for shots in qsensitivity.get("eval_shots",[200,500,1000]):
            for alpha in qsensitivity.get("cvar_alpha",[0.05,0.1,0.25,0.5,1.0]):
                sub=out/f"shots_{shots}_alpha_{str(alpha).replace('.','p')}"
                argv=[
                    self.venv_python,"-m", module_name("batch_benchmark_hard_set.py"),"--research-ablation",
                    "--input-dir",str(self.dataset_dir()/cfg.get("input_dir","graphs/train")),
                    "--checkpoint",str(self.checkpoint_dir()/qc.get("checkpoint","best_egnn_pruning.pt")),
                    "--out-dir",str(sub),"--pruning","egnn",
                    "--radii",str(qc.get("radii",[6.0])[0]),
                    "--depths",*[str(v) for v in qsensitivity.get("depths",[1,2,3])],
                    "--max-evals",*[str(v) for v in qsensitivity.get("max_evals",[90,180,300])],
                    "--active-sites",str(cfg.get("active_sites",
                        self.config.get("statistics",{}).get("primary_active_sites",6))),
                    "--vhh-identity-threshold",str(homology.get("vhh_full_chain_identity",0.80)),
                    "--cdr-h3-identity-threshold",str(homology.get("cdr_h3_identity",0.50)),
                    "--antigen-identity-threshold",str(homology.get("antigen_identity",0.30)),
                    "--antigen-min-length-coverage",str(homology.get("antigen_min_length_coverage",0.70)),
                    "--antigen-guidance-weight",str(qc.get("antigen_guidance_weight",0.25)),
                    "--antigen-proximity-scale",str(qc.get("antigen_proximity_scale_angstrom",6.0)),
                    "--contact-ca-cutoff",str(qc.get("contact_ca_cutoff_angstrom",8.0)),
                    "--nonbonded-cutoff",str(ff.get("cutoff_angstrom",8.0)),
                    "--softcore-delta",str(ff.get("softcore_delta_angstrom",0.5)),
                    "--hard-core-fraction",str(ff.get("hard_core_fraction",0.72)),
                    "--hard-sphere-penalty",str(ff.get("hard_sphere_penalty",25.0)),
                    "--lj-repulsion-cap",str(ff.get("lj_repulsion_cap",50.0)),
                    "--lj-attraction-cap",str(ff.get("lj_attraction_cap",5.0)),
                    "--coulomb-cap",str(ff.get("coulomb_cap",20.0)),
                    "--dielectric-base",str(ff.get("dielectric_base",4.0)),
                    "--dielectric-slope",str(ff.get("dielectric_slope",2.0)),
                    "--thermal-energy-kcal",str(ff.get("thermal_energy_kcal",0.593)),
                    "--rotamer-mode",str(rot.get("mode","dunbrack2010")),
                    "--rotamer-library",str(resolve_path(self.config,rot.get("library_path","data/rotamer/ALL.bbdep.rotamers.lib"))),
                    "--rotamer-probability-floor",str(rot.get("probability_floor",1e-4)),
                    "--rotamer-sigma-offsets",*[str(v) for v in rot.get("sigma_offsets",[-1,0,1])],
                    "--energy-calibration-file",str(calibration),"--require-calibrated-energy",
                    "--outputs",str(cfg.get("outputs",300)),"--qaoa-objective","cvar",
                    "--qaoa-restarts",str(qprimary.get("restarts",4)),
                    "--cvar-alpha",str(alpha),"--eval-shots",str(shots),
                    "--parameter-scale",str(qprimary.get("parameter_scale","max_coefficient")),
                    "--sa-passes",str(qc.get("sa_passes",100)),
                    "--greedy-passes",str(qc.get("greedy_passes",50)),
                    "--energy-window",str(qc.get("energy_window",2.0)),
                    "--max-targets",str(cfg.get("max_targets",20)),
                    "--target-selection-seed",str(target_selection_seed),
                    "--workers",str(qc.get("workers",1)),
                    "--omp-threads",str(self.config.get("hardware",{}).get("cpu_threads_per_process",2)),
                    "--seeds",*[str(v) for v in repeats],"--master-seed",str(self.config["master_seed"]),
                ]
                rc,log=self._run_subprocess(
                    f"sensitivity_shots_{shots}_alpha_{str(alpha).replace('.','p')}",argv)
                logs.append(str(log));argvs.append(argv)
                summary_path=sub/"run_summary.json"
                if rc!=0 or not summary_path.is_file():
                    failures.append(f"shots={shots}, alpha={alpha}, exit={rc}")
                    continue
                try:
                    summary=json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    failures.append(
                        f"shots={shots}, alpha={alpha}, unreadable run_summary.json: {exc}")
                    continue
                if not summary.get("closed"):
                    failures.append(
                        f"shots={shots}, alpha={alpha}, sensitivity sub-run not closed")
                    continue
                if int(summary.get("failures_total",0) or 0)!=0:
                    failures.append(
                        f"shots={shots}, alpha={alpha}, failures_total="
                        f"{summary.get('failures_total')}")
        aggregate_rows=[]
        for shots in qsensitivity.get("eval_shots",[200,500,1000]):
            for alpha in qsensitivity.get("cvar_alpha",[0.05,0.1,0.25,0.5,1.0]):
                sub=out/f"shots_{shots}_alpha_{str(alpha).replace('.','p')}"
                metrics=sub/"metrics.csv"
                if not metrics.is_file():
                    continue
                with metrics.open(newline="",encoding="utf-8") as handle:
                    rows=list(csv.DictReader(handle))
                qrows=[r for r in rows if r.get("solver")=="qaoa"]
                groups={}
                for row in qrows:
                    key=(
                        int(float(row.get("active_sites",cfg.get("active_sites",6)))),
                        int(float(row.get("depth",0) or 0)),
                        int(float(row.get("max_evals",0) or 0)),
                    )
                    groups.setdefault(key,[]).append(row)
                for (active_sites,depth,max_evals),group in sorted(groups.items()):
                    def mean_field(field):
                        vals=[float(r[field]) for r in group if r.get(field) not in (None,"","None")]
                        return (sum(vals)/len(vals)) if vals else None
                    aggregate_rows.append({
                        "active_sites":active_sites,"depth":depth,"max_evals":max_evals,
                        "eval_shots":shots,"cvar_alpha":alpha,"rows":len(group),
                        "mean_hit":mean_field("hit"),"mean_gap":mean_field("gap"),
                        "mean_ground_probability":mean_field("ground_probability"),
                        "mean_low_energy_mass":mean_field("low_energy_mass"),
                        "mean_solver_seconds":mean_field("solver_seconds"),
                    })
        resolution_cfg=cfg.get("rotamer_resolution",{}) or {}
        resolution_rows=[]
        resolution_case_keys={}
        if resolution_cfg:
            if not argvs:
                failures.append("rotamer resolution sensitivity has no base arguments")
            else:
                def replace_option(arguments, name, values):
                    arguments=list(arguments)
                    if name not in arguments:
                        return arguments+[name,*values]
                    start=arguments.index(name)+1
                    end=start
                    while end<len(arguments) and not arguments[end].startswith("--"):
                        end+=1
                    return arguments[:start]+values+arguments[end:]
                for sites in resolution_cfg.get("active_sites",[4,5]):
                    for states in resolution_cfg.get("states_per_site",[3,4,5,6]):
                        sub=out/"rotamer_resolution"/f"sites_{sites}_states_{states}"
                        command=list(argvs[0])
                        for name,values in (
                            ("--out-dir",[str(sub)]),
                            ("--active-sites",[str(sites)]),
                            ("--states-per-site",[str(states)]),
                            ("--depths",[str(qprimary.get("depth",2))]),
                            ("--max-evals",[str(qprimary.get("max_evals",90))]),
                            ("--eval-shots",[str(qprimary.get("eval_shots",500))]),
                            ("--cvar-alpha",[str(qprimary.get("cvar_alpha",0.1))]),
                        ):
                            command=replace_option(command,name,values)
                        rc,log=self._run_subprocess(f"rotamer_resolution_{sites}_{states}",command)
                        logs.append(str(log));argvs.append(command)
                        summary_path=sub/"run_summary.json"
                        if rc!=0 or not summary_path.is_file():
                            failures.append(f"rotamer resolution {sites} sites/{states} states failed (exit={rc})")
                            continue
                        summary=json.loads(summary_path.read_text(encoding="utf-8"))
                        if not summary.get("closed") or int(summary.get("failures_total",0) or 0)!=0:
                            failures.append(f"rotamer resolution {sites} sites/{states} states incomplete")
                            continue
                        case_keys={path.name for path in (sub/"cases").glob("*.json")}
                        if int(summary.get("cases_completed_total",0) or 0)!=len(case_keys):
                            failures.append(f"rotamer resolution {sites}/{states} case count mismatch")
                            continue
                        previous=resolution_case_keys.setdefault(int(sites),case_keys)
                        if case_keys!=previous:
                            failures.append(f"rotamer resolution {sites}/{states} is not paired on the same cases")
                            continue
                        metrics_path=sub/"metrics.csv"
                        if not metrics_path.is_file():
                            failures.append(f"rotamer resolution {sites} sites/{states} states lacks metrics")
                            continue
                        with metrics_path.open(newline="",encoding="utf-8") as handle:
                            metrics=list(csv.DictReader(handle))
                        for solver in ("qaoa","sa","greedy","uniform"):
                            selected=[row for row in metrics if row.get("solver")==solver
                                      and row.get("budget_mode")=="matched_outputs"
                                      and (solver!="qaoa" or (row.get("qaoa_objective")=="cvar"
                                          and int(float(row.get("qaoa_restarts",0) or 0))==4))]
                            if not selected:
                                failures.append(f"rotamer resolution {sites}/{states} lacks {solver} rows")
                                continue
                            resolution_rows.append(dict(active_sites=int(sites),states_per_site=int(states),
                                solver=solver,rows=len(selected),
                                mean_hit=sum(float(row["hit"]) for row in selected)/len(selected),
                                mean_gap=sum(float(row["gap"]) for row in selected)/len(selected),
                                mean_solver_seconds=sum(float(row["solver_seconds"]) for row in selected)/len(selected)))
                        oracle_times=[]
                        for case_path in sorted((sub/"cases").glob("*.json")):
                            case=json.loads(case_path.read_text(encoding="utf-8"))
                            # The exact-oracle time is measured once per case and
                            # repeated on every metrics row of that case.
                            case_oracle={float(row["oracle_seconds"])
                                         for row in (case.get("metrics") or [])
                                         if row.get("oracle_seconds") not in (None,"")}
                            if len(case_oracle)!=1:
                                failures.append(
                                    f"rotamer resolution {sites}/{states} case {case_path.name} "
                                    "lacks one consistent exact-oracle timing")
                                oracle_times=[]
                                break
                            oracle_times.append(case_oracle.pop())
                        if not oracle_times:
                            failures.append(f"rotamer resolution {sites}/{states} lacks exact-oracle timing")
                        else:
                            resolution_rows.append(dict(active_sites=int(sites),states_per_site=int(states),
                                solver="exact_enumeration",rows=len(oracle_times),mean_hit=1.0,mean_gap=0.0,
                                mean_solver_seconds=sum(oracle_times)/len(oracle_times)))
        if resolution_cfg:
            expected_rows=(len(resolution_cfg.get("active_sites",[4,5]))
                           *len(resolution_cfg.get("states_per_site",[3,4,5,6]))*5)
            if len(resolution_rows)!=expected_rows:
                failures.append(f"rotamer resolution has {len(resolution_rows)}/{expected_rows} solver summaries")
        if resolution_cfg:
            resolution_root=out/"rotamer_resolution"
            resolution_root.mkdir(parents=True,exist_ok=True)
            atomic_write_json(resolution_root/"summary.json",dict(
                scope="training-only representation and solver sensitivity",
                protocol="fixed sites, 3-6 states/site with chi1-well coverage; same target subset and repeat seeds; exact enumeration timed after QUBO build",
                rows=resolution_rows,failures=failures))
            with (resolution_root/"summary.csv").open("w",newline="",encoding="utf-8") as handle:
                writer=csv.DictWriter(handle,fieldnames=["active_sites","states_per_site","solver","rows","mean_hit","mean_gap","mean_solver_seconds"])
                writer.writeheader();writer.writerows(resolution_rows)
            resolution_md=["# Rotamer-resolution sensitivity","",
                "Training-only paired cases. Each state count covers all three chi1 wells. Exact enumeration is timed after QUBO construction.",
                "These results compare solver behavior and do not measure native chi1/chi2 or all-atom recovery.","",
                "| Sites | States/site | Solver | Rows | Mean hit | Mean gap | Mean solver seconds |",
                "|---:|---:|---|---:|---:|---:|---:|"]
            for row in resolution_rows:
                resolution_md.append(f"| {row['active_sites']} | {row['states_per_site']} | {row['solver']} | {row['rows']} | {row['mean_hit']:.6g} | {row['mean_gap']:.6g} | {row['mean_solver_seconds']:.6g} |")
            (resolution_root/"summary.md").write_text("\n".join(resolution_md)+"\n",encoding="utf-8")
        out.mkdir(parents=True,exist_ok=True)
        summary_csv=out/"sensitivity_summary.csv"
        fields=["active_sites","depth","max_evals","eval_shots","cvar_alpha","rows",
                "mean_hit","mean_gap","mean_ground_probability","mean_low_energy_mass",
                "mean_solver_seconds"]
        with summary_csv.open("w",newline="",encoding="utf-8") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(aggregate_rows)
        atomic_write_json(out/"sensitivity_summary.json",{
            "scope":"development-only",
            "primary_protocol_unchanged":True,
            "subruns_planned":len(qsensitivity.get("eval_shots",[200,500,1000]))*len(qsensitivity.get("cvar_alpha",[0.05,0.1,0.25,0.5,1.0])),
            "subruns_summarized":len(aggregate_rows),
            "failures":failures,
            "rows":aggregate_rows,
        })
        md=["# Development-only QAOA sensitivity","","Validation/test data were not used to select hyperparameters.","",
            "| sites | p | max evals | eval shots | CVaR alpha | QAOA rows | mean hit | mean gap | mean ground probability | mean low-energy mass | mean solver seconds |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in aggregate_rows:
            md.append("| {active_sites} | {depth} | {max_evals} | {eval_shots} | {cvar_alpha} | {rows} | {mean_hit} | {mean_gap} | {mean_ground_probability} | {mean_low_energy_mass} | {mean_solver_seconds} |".format(**row))
        (out/"sensitivity_summary.md").write_text("\n".join(md)+"\n",encoding="utf-8")
        return StageResult(
            "method_sensitivity","completed" if not failures else "failed",
            started,utc_timestamp(),0 if not failures else 1,
            "Development sensitivity completed; primary validation settings unchanged."
            if not failures else "; ".join(failures),
            argvs[-1] if argvs else [],";".join(logs),not failures,
        )

    # ================================================================
    # Stage 7: quantum-vs-classical ablation benchmark
    # ================================================================
    def stage_qc_benchmark(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config["qc_benchmark"]
        qprimary=quantum_primary(self.config)
        qablation=quantum_benchmark_ablation(self.config)
        out_dir = self.run_dir / "qc_benchmark"
        # (requirement #2) --seeds are the shared repeat/perturb identities
        # (site-selection input, shared by all four solvers within a case) --
        # NEVER the optimize/sample streams themselves. _ablation_main now
        # derives each case's own independent optimize_seed/sample_seed from
        # --master-seed internally (keyed by stable per-case labels), so
        # this orchestrator only needs to pass --master-seed and a list of
        # repeat identities, not pre-derive optimize/sample values by hand.
        repeat_seeds = [derive_child_seed(derive_streams(self.config["master_seed"])["perturb"],
                                           "ablation_repeat", str(i)) for i in range(cfg.get("repeats", 3))]
        argv = [
            self.venv_python, "-m", module_name("batch_benchmark_hard_set.py"), "--research-ablation",
            "--input-dir", str(self.dataset_dir() / cfg["input_dir"]),
            "--checkpoint", str(self.checkpoint_dir() / cfg["checkpoint"]),
            "--out-dir", str(out_dir),
            "--seeds", *[str(s) for s in repeat_seeds],
            "--master-seed", str(self.config["master_seed"]),
            "--pruning", *cfg.get("pruning", ["egnn", "contact", "distance", "cdr", "random"]),
            "--radii", *[str(r) for r in cfg.get("radii", [6.0, 10.0])],
            "--depths", str(qprimary.get("depth",2)),
            "--max-evals", str(qprimary.get("max_evals",90)),
            "--active-sites", *[str(v) for v in cfg.get("active_sites", [6])],
            "--states-per-site", "3",
            "--vhh-identity-threshold", str(self.config["queue_freeze"]["homology_isolation"].get("vhh_full_chain_identity", 0.80)),
            "--cdr-h3-identity-threshold", str(self.config["queue_freeze"]["homology_isolation"].get("cdr_h3_identity", 0.50)),
            "--antigen-identity-threshold", str(self.config["queue_freeze"]["homology_isolation"].get("antigen_identity", 0.30)),
            "--antigen-min-length-coverage", str(self.config["queue_freeze"]["homology_isolation"].get("antigen_min_length_coverage", 0.70)),
            "--antigen-guidance-weight", str(cfg.get("antigen_guidance_weight", 0.25)),
            "--antigen-proximity-scale", str(cfg.get("antigen_proximity_scale_angstrom", 6.0)),
            "--contact-ca-cutoff", str(cfg.get("contact_ca_cutoff_angstrom", 8.0)),
            "--nonbonded-cutoff", str(cfg.get("coarse_force_field", {}).get("cutoff_angstrom", 8.0)),
            "--softcore-delta", str(cfg.get("coarse_force_field", {}).get("softcore_delta_angstrom", 0.5)),
            "--hard-core-fraction", str(cfg.get("coarse_force_field", {}).get("hard_core_fraction", 0.72)),
            "--hard-sphere-penalty", str(cfg.get("coarse_force_field", {}).get("hard_sphere_penalty", 25.0)),
            "--lj-repulsion-cap", str(cfg.get("coarse_force_field", {}).get("lj_repulsion_cap", 50.0)),
            "--lj-attraction-cap", str(cfg.get("coarse_force_field", {}).get("lj_attraction_cap", 5.0)),
            "--coulomb-cap", str(cfg.get("coarse_force_field", {}).get("coulomb_cap", 20.0)),
            "--dielectric-base", str(cfg.get("coarse_force_field", {}).get("dielectric_base", 4.0)),
            "--dielectric-slope", str(cfg.get("coarse_force_field", {}).get("dielectric_slope", 2.0)),
            "--thermal-energy-kcal", str(cfg.get("coarse_force_field", {}).get("thermal_energy_kcal", 0.593)),
            "--rotamer-mode", str(cfg.get("rotamer_model", {}).get("mode", "dunbrack2010")),
            "--rotamer-library", str(resolve_path(self.config, cfg.get("rotamer_model", {}).get("library_path", "data/rotamer/ALL.bbdep.rotamers.lib"))),
            "--rotamer-probability-floor", str(cfg.get("rotamer_model", {}).get("probability_floor", 1e-4)),
            "--rotamer-sigma-offsets", *[str(v) for v in cfg.get("rotamer_model", {}).get("sigma_offsets", [-1.0,0.0,1.0])],
            "--outputs", *[str(o) for o in cfg.get("outputs", [10, 30, 100, 300, 1000])],
            "--qaoa-objective", *[str(v) for v in qablation.get("objectives",["mean","cvar"])],
            "--qaoa-restarts", *[str(r) for r in qablation.get("restarts",[1,4])],
            "--cvar-alpha", str(qprimary.get("cvar_alpha",0.1)),
            "--eval-shots", str(qprimary.get("eval_shots",500)),
            "--parameter-scale", str(qprimary.get("parameter_scale","max_coefficient")),
            "--sa-passes", str(cfg.get("sa_passes", 100)),
            "--greedy-passes", str(cfg.get("greedy_passes", 50)),
            "--energy-window", str(cfg.get("energy_window", 2.0)),
            "--time-donor-objective", str(qprimary.get("objective","cvar")),
            "--time-donor-restarts", str(qprimary.get("restarts",4)),
            "--max-targets", str(cfg.get("max_targets", 0)),
            "--workers", str(cfg.get("workers", 1)),
            "--omp-threads", str(self.config.get("hardware", {}).get("cpu_threads_per_process", 2)),
        ]
        calibration_cfg = cfg.get("energy_calibration", {}) or {}
        calibration_file = self.run_dir / calibration_cfg.get(
            "calibration_file", "calibration/coarse_to_amber.json"
        )
        if calibration_cfg.get("require_calibrated", False):
            argv.append("--require-calibrated-energy")
        if calibration_file.is_file():
            argv += ["--energy-calibration-file", str(calibration_file)]
        elif calibration_cfg.get("require_calibrated", False):
            return StageResult(
                "qc_benchmark", "failed", started, utc_timestamp(), None,
                f"Required run-specific frozen calibration missing: {calibration_file}",
            )
        if cfg.get("time_baselines", True):
            argv.append("--time-baselines")
        returncode, log_path = self._run_subprocess("qc_benchmark", argv)
        expected = [out_dir / "run_manifest.json"]
        ok, artifact_detail = self._artifacts_present(expected)
        summary_path = out_dir / "run_summary.json"
        # (requirement #5) Completion is decided from _ablation_main's own
        # planned/completed/failed reconciliation (run_summary.json), never
        # from returncode + "some artifact exists" alone: a per-instance
        # failure is expected and does not by itself mean the stage failed,
        # but an UNCLOSED run (a case neither completed nor recorded as
        # failed -- e.g. the process was killed mid-case) must not be
        # reported as complete just because metrics.csv happens to exist.
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
        if returncode not in (0, 1) or not ok:
            status = "failed"
            detail = f"Benchmark process failed (exit={returncode}): {artifact_detail}"
        elif summary is None:
            status = "failed"
            detail = (f"No run_summary.json (planned/completed/failed reconciliation) was produced; "
                      f"cannot confirm completion. {artifact_detail}")
        elif not summary.get("closed"):
            status = "failed"
            detail = (f"Not closed: planned={summary['total_cases_planned']} "
                      f"completed={summary['cases_completed_total']} failed={summary['failures_total']} "
                      f"gap={summary['gap']} -- re-run (resumable) to close the gap. {artifact_detail}")
        elif summary.get("cases_completed_total", 0) == 0:
            status = "failed"
            detail = "No case completed successfully; refusing to accept an all-failed benchmark."
        elif returncode == 1 and not summary.get("failures_this_invocation", 0):
            status = "failed"
            detail = "Nonzero exit is inconsistent with case summary; inspect the stage log."
        elif (
            summary.get("failures_total",0) / max(1,summary.get("total_cases_planned",1))
            > float(cfg.get("max_failure_fraction",0.0))
        ):
            status = "failed"
            failure_fraction=summary.get("failures_total",0)/max(1,summary.get("total_cases_planned",1))
            detail = (
                f"Closed but failure fraction {failure_fraction:.6f} exceeds frozen "
                f"max_failure_fraction={float(cfg.get('max_failure_fraction',0.0)):.6f}; "
                f"planned={summary['total_cases_planned']} completed={summary['cases_completed_total']} "
                f"failed={summary['failures_total']} (see {out_dir}/failed_cases.log)."
            )
        elif summary.get("failures_total", 0) > 0:
            status = "completed_with_failures"
            detail = (f"Closed within allowed failure fraction: planned={summary['total_cases_planned']} "
                      f"completed={summary['cases_completed_total']} failed={summary['failures_total']} "
                      f"(see {out_dir}/failed_cases.log). {artifact_detail}")
        else:
            status = "completed"
            detail = (f"Closed, no failures: planned={summary['total_cases_planned']} "
                      f"completed={summary['cases_completed_total']}. {artifact_detail}")
        return StageResult("qc_benchmark", status, started, utc_timestamp(), returncode, detail,
                            argv, str(log_path), ok)

    # ================================================================
    # Stage 6: real-atom structural experiment (dev queue + validation queue)
    # ================================================================
    def stage_structure_experiment(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config["structure_experiment"]
        qprimary=quantum_primary(self.config)
        dataset_dir = self.dataset_dir()
        checkpoint_dir = self.checkpoint_dir()
        validation_dir = self.run_dir / "validation_queue"
        dev_dir = self.run_dir / "dev_queue"
        qf_cfg = self.config["queue_freeze"]

        def shared_flags() -> List[str]:
            flags = [
                "--outputs", str(qprimary.get("output_shots",1000)),
                "--max-evals", str(qprimary.get("max_evals",90)),
                "--qaoa-depth", str(qprimary.get("depth",2)),
                "--perturbation-mode", str(cfg.get("perturbation_mode", "multi_chi")),
                "--solvent-model", str(cfg.get("solvent_model", "vacuum")),
                "--min-perturb-degrees", str(cfg.get("min_perturb_degrees", 40.0)),
                "--max-perturb-degrees", str(cfg.get("max_perturb_degrees", 120.0)),
                "--relax-iterations", str(cfg.get("relax_iterations", 200)),
                "--candidate-relax-iterations", str(cfg.get("candidate_relax_iterations", 100)),
                "--antigen-guidance-weight", str(cfg.get("antigen_guidance_weight", 0.25)),
                "--antigen-proximity-scale", str(cfg.get("antigen_proximity_scale_angstrom", 6.0)),
                "--contact-ca-cutoff", str(cfg.get("contact_ca_cutoff_angstrom", 8.0)),
                "--rotamer-mode", str(cfg.get("rotamer_model", {}).get("mode", "dunbrack2010")),
                "--rotamer-library", str(resolve_path(self.config, cfg.get("rotamer_model", {}).get("library_path", "data/rotamer/ALL.bbdep.rotamers.lib"))),
                "--rotamer-probability-floor", str(cfg.get("rotamer_model", {}).get("probability_floor", 1e-4)),
                "--rotamer-sigma-offsets", *[str(v) for v in cfg.get("rotamer_model", {}).get("sigma_offsets", [-1.0,0.0,1.0])],
                "--vhh-identity-threshold", str(qf_cfg["homology_isolation"].get("vhh_full_chain_identity", 0.80)),
                "--cdr-h3-identity-threshold", str(qf_cfg["homology_isolation"].get("cdr_h3_identity", 0.50)),
                "--antigen-identity-threshold", str(qf_cfg["homology_isolation"].get("antigen_identity", 0.30)),
                "--antigen-min-length-coverage", str(qf_cfg["homology_isolation"].get("antigen_min_length_coverage", 0.70)),
                "--loop-relax-iterations", str(cfg.get("loop_relax_iterations", 100)),
                "--eval-shots", str(qprimary.get("eval_shots",500)),
                "--seeds", *[str(s) for s in cfg.get("seeds", [42, 43, 44])],
                # (requirement #2) --master-seed lets run_real_complex_pilot.py
                # derive its own independent, saved --optimize-seeds/
                # --sample-seeds per selected target from the master-seed
                # optimize/sample streams -- never passed as flat --seeds values.
                "--master-seed", str(self.config["master_seed"]),
            ]
            cluster_path=self.frozen_cluster_map_path()
            if not cluster_path.is_file():
                raise FileNotFoundError(f"Required run-local family/structure cluster map missing: {cluster_path}")
            flags += ["--cluster-map", str(cluster_path)]
            if cfg.get("robust_qaoa", True):
                flags += ["--robust-qaoa", "--qaoa-restarts", str(qprimary.get("restarts",4)),
                          "--qaoa-objective", str(qprimary.get("objective","cvar")),
                          "--cvar-alpha", str(qprimary.get("cvar_alpha",0.1)),
                          "--parameter-scale", str(qprimary.get("parameter_scale","max_coefficient"))]
            return flags

        runs = [
            ("dev_queue", dev_dir, qf_cfg["dev_queue"], list(qf_cfg["dev_queue"].get("excluded_pdb", []))),
            ("validation_queue", validation_dir, qf_cfg["validation_queue"], None),
        ]
        failures = []
        queue_partial = []
        logs = []
        argvs = []
        for label, out_dir, queue_cfg, explicit_targets in runs:
            # The real site selection: by this stage egnn_train has already
            # completed (a prerequisite of this stage -- see run_all()), so
            # this run's own checkpoint_dir()/best_egnn_pruning.pt is
            # guaranteed to exist, unlike at queue_freeze time (requirement
            # #1). The strategy itself is READ FROM CONFIG (queue_cfg
            # "pruning", default "egnn" -- the formal main protocol), never
            # silently hardcoded: a user who explicitly wants the formal
            # structural experiment run under a different single strategy
            # (or, for a full five-way ALL-ATOM ablation, one
            # structure_experiment invocation per strategy, each with its
            # own --out-dir) sets it here, explicitly, per requirement #3 --
            # this orchestrator never substitutes contact/distance/cdr/random for the
            # main strategy on its own.
            queue_pruning = queue_cfg.get("pruning", "egnn")
            argv = [
                self.venv_python, "-m", module_name("run_real_complex_pilot.py"),
                "--dataset", str(dataset_dir),
                "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
                "--out-dir", str(out_dir),
                "--sites", str(queue_cfg.get("sites", 6)),
                "--pruning", queue_pruning,
                "--checkpoint", str(checkpoint_dir / "best_egnn_pruning.pt"),
                "--dev-exposed-pdb", *qf_cfg["dev_queue"].get("excluded_pdb", []),
                "--queue-role", "dev" if label == "dev_queue" else "validation",
            ] + shared_flags()
            if label == "dev_queue":
                # Historical dev targets are run individually. Each subprocess
                # therefore has an exact denominator of one target.
                # Historical dev targets are each run individually against
                # their own already-frozen manifest from real_complex_pilot_v3
                # if present, otherwise freshly (re-)selected+frozen here
                # under out_dir/<pdb>/, keeping the dev queue fully separate
                # from the validation queue's own directory.
                dev_completed=[]; dev_failed=[]; dev_summaries={}
                for pdb in explicit_targets:
                    sub_argv = argv + [
                        "--targets", "1",
                        "--pdb-id", pdb,
                        "--out-dir", str(out_dir / pdb),
                    ]
                    returncode, log_path = self._run_subprocess(f"structure_experiment_dev_{pdb}", sub_argv)
                    logs.append(str(log_path)); argvs.append(sub_argv)
                    summary_path = out_dir / pdb / "run_summary.json"
                    if summary_path.is_file():
                        summary=json.loads(summary_path.read_text(encoding="utf-8"))
                        dev_summaries[pdb]=summary
                        if summary.get("closed") and not summary.get("structure_experiment_failed_targets"):
                            dev_completed.append(pdb.lower())
                        else:
                            dev_failed.append(pdb.lower())
                    else:
                        dev_failed.append(pdb.lower())
                    if returncode not in (0,) and not (out_dir / pdb / "real_complex_metrics.csv").is_file():
                        failures.append(f"dev target {pdb} exited {returncode} with no usable metrics (see {log_path})")
                planned=[p.lower() for p in explicit_targets]
                dev_closed=(
                    set(dev_completed)|set(dev_failed)==set(planned)
                    and not (set(dev_completed)&set(dev_failed))
                )

                # Aggregate every child dev run into one queue-level result set.
                dev_eligibility=[];dev_selected=[];dev_metrics=[]
                child_manifest_sha256={}
                for pdb in explicit_targets:
                    child=out_dir/pdb
                    child_manifest=child/"run_manifest.json"
                    if child_manifest.is_file():
                        child_manifest_sha256[pdb.lower()]=sha256_of(child_manifest)
                    for name,destination in (
                        ("eligibility.json",dev_eligibility),
                        ("selected_targets.json",dev_selected),
                    ):
                        p=child/name
                        if p.is_file():
                            payload=json.loads(p.read_text(encoding="utf-8"))
                            if isinstance(payload,list):
                                destination.extend(payload)
                    metrics_path=child/"real_complex_metrics.csv"
                    if metrics_path.is_file():
                        with metrics_path.open(newline="",encoding="utf-8") as handle:
                            dev_metrics.extend(csv.DictReader(handle))
                atomic_write_json(out_dir/"run_manifest.json",{
                    "queue_role":"dev",
                    "master_seed":self.config["master_seed"],
                    "planned_target_ids":sorted(planned),
                    "child_run_manifest_sha256":child_manifest_sha256,
                    "checkpoint_sha256":sha256_of(checkpoint_dir/"best_egnn_pruning.pt"),
                    "protocol":"historical development/regression queue; never confirmatory validation",
                })
                save_stream_map(
                    out_dir/"seed_streams.json",self.config["master_seed"],
                    derive_streams(self.config["master_seed"]))
                atomic_write_json(out_dir/"eligibility.json",dev_eligibility)
                atomic_write_json(out_dir/"selected_targets.json",dev_selected)
                if dev_metrics:
                    fields=list(dict.fromkeys(k for row in dev_metrics for k in row))
                    with (out_dir/"real_complex_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
                        writer=csv.DictWriter(handle,fieldnames=fields)
                        writer.writeheader();writer.writerows(dev_metrics)
                report=[
                    "# Development structural recovery queue","",
                    "Historical development/regression targets only; never used as confirmatory validation.",
                    f"Planned targets: {len(planned)}; completed: {len(dev_completed)}; failed: {len(dev_failed)}.",
                    f"Aggregated metric rows: {len(dev_metrics)}.",
                    "",
                    f"Completed target IDs: {sorted(dev_completed)}",
                    f"Failed target IDs: {sorted(dev_failed)}",
                ]
                (out_dir/"real_complex_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
                atomic_write_json(out_dir/"run_summary.json",dict(
                    queue_role="dev",
                    planned_target_ids=sorted(planned),
                    selected_targets=len(dev_selected),
                    structure_experiment_completed_targets=len(dev_completed),
                    structure_experiment_completed_target_ids=sorted(dev_completed),
                    structure_experiment_failed_targets=sorted(dev_failed),
                    closed=dev_closed,
                    child_run_summaries=dev_summaries,
                ))
                if not dev_closed:
                    failures.append(
                        f"dev_queue root accounting not closed: planned={sorted(planned)} "
                        f"completed={sorted(dev_completed)} failed={sorted(dev_failed)}")
                elif dev_failed:
                    queue_partial.append(
                        f"dev_queue: completed with target failures {sorted(dev_failed)}")
                continue
            # (requirement #1/#3) Reproduce EXACTLY the target set queue_freeze
            # already froze -- via an explicit allowlist file, never by
            # trusting that re-running the same eligibility logic with a
            # different --pruning value happens to select the same targets.
            # A target can still fail re-verification here (recorded, not
            # silently dropped), but no target outside the frozen set can
            # ever be added.
            frozen_root=validation_dir/"freeze"
            frozen_targets=frozen_root/"selected_targets.json"
            freeze_manifest_path=frozen_root/"freeze_manifest.json"
            if not frozen_targets.is_file() or not freeze_manifest_path.is_file():
                failures.append(
                    f"{label}: frozen target file/manifest missing; refusing confirmatory execution")
                continue
            try:
                freeze_manifest=json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"{label}: unreadable freeze_manifest.json: {exc}")
                continue
            expected_allowlist_sha=freeze_manifest.get("selected_targets_sha256")
            actual_allowlist_sha=sha256_of(frozen_targets)
            if not expected_allowlist_sha or actual_allowlist_sha!=expected_allowlist_sha:
                failures.append(
                    f"{label}: frozen selected_targets sha256 mismatch; "
                    f"expected={expected_allowlist_sha}, actual={actual_allowlist_sha}")
                continue
            argv += ["--targets", str(queue_cfg.get("target_count", 0)),
                     "--pdb-allowlist-file", str(frozen_targets)]
            returncode, log_path = self._run_subprocess(f"structure_experiment_{label}", argv)
            logs.append(str(log_path)); argvs.append(argv)
            # (requirement #5) Reconciled against run_real_complex_pilot.py's
            # own run_summary.json (selected vs. completed vs. failed target
            # counts), never inferred from returncode or CSV existence alone.
            summary_path = out_dir / "run_summary.json"
            if not summary_path.is_file():
                failures.append(f"{label}: no run_summary.json produced (cannot confirm completion; see {log_path})")
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if not summary.get("closed"):
                failures.append(f"{label}: not closed -- selected={summary.get('selected_targets')} "
                                 f"completed={summary.get('structure_experiment_completed_targets')} "
                                 f"failed={summary.get('structure_experiment_failed_targets')} (see {log_path})")
            elif label=="validation_queue" and summary.get("frozen_set_accounting_ok") is not True:
                failures.append(
                    f"{label}: frozen denominator accounting failed; "
                    f"frozen={summary.get('frozen_target_ids')} "
                    f"completed={summary.get('structure_experiment_completed_target_ids')} "
                    f"failed={summary.get('structure_experiment_failed_targets')} (see {log_path})")
            elif summary.get("structure_experiment_failed_targets"):
                message=(f"{label}: completed with target failures "
                         f"{summary['structure_experiment_failed_targets']}")
                if label=="validation_queue":
                    failures.append(message+"; confirmatory queue must be complete")
                else:
                    queue_partial.append(message)

        # Pre-declared solvent sensitivity uses development targets only.
        # Validation remains on the frozen primary solvent protocol.
        sensitivity_models=[str(v) for v in cfg.get("solvent_sensitivity", [])]
        primary_solvent=str(cfg.get("solvent_model","vacuum"))
        for solvent in sensitivity_models:
            if solvent==primary_solvent:
                continue
            if solvent not in ("vacuum","gbn2"):
                failures.append(f"Unsupported solvent sensitivity model: {solvent}")
                continue
            manifests=sorted(dev_dir.glob("*/prepared/*/recovery_manifest.json"))
            if not manifests:
                failures.append(f"solvent sensitivity {solvent}: no frozen dev recovery manifests")
                continue
            solvent_root=self.run_dir/f"dev_queue_solvent_{solvent}"
            solvent_completed=[];solvent_failed=[];solvent_rows=[]
            for manifest in manifests:
                target=manifest.parent.name
                out=solvent_root/"results"/target
                optimize_seeds=[
                    derive_child_seed(derive_streams(self.config["master_seed"])["optimize"],
                                      "solvent_sensitivity",solvent,target,str(seed))
                    for seed in cfg.get("seeds",[42,43,44,45,46])
                ]
                measurement_seeds=[
                    derive_child_seed(derive_streams(self.config["master_seed"])["measurement"],
                                      "solvent_sensitivity",solvent,target,str(seed))
                    for seed in cfg.get("seeds",[42,43,44,45,46])
                ]
                sample_seeds=[
                    derive_child_seed(derive_streams(self.config["master_seed"])["sample"],
                                      "solvent_sensitivity",solvent,target,str(seed))
                    for seed in cfg.get("seeds",[42,43,44,45,46])
                ]
                sensitivity_argv=[
                    self.venv_python,"-m", module_name("batch_benchmark_hard_set.py"),"--recovery-benchmark",
                    "--manifest",str(manifest),"--out-dir",str(out),
                    "--solvent-model",solvent,
                    "--perturbation-mode",str(cfg.get("perturbation_mode","multi_chi")),
                    "--min-perturb-degrees",str(cfg.get("min_perturb_degrees",40.0)),
                    "--max-perturb-degrees",str(cfg.get("max_perturb_degrees",120.0)),
                    "--outputs",str(qprimary.get("output_shots",1000)),
                    "--max-evals",str(qprimary.get("max_evals",90)),
                    "--sa-passes",str(cfg.get("sa_passes",100)),
                    "--relax-iterations",str(cfg.get("relax_iterations",200)),
                    "--loop-relax-iterations",str(cfg.get("loop_relax_iterations",100)),
                    "--seeds",*[str(v) for v in cfg.get("seeds",[42,43,44,45,46])],
                    "--optimize-seeds",*[str(v) for v in optimize_seeds],
                    "--measurement-seeds",*[str(v) for v in measurement_seeds],
                    "--sample-seeds",*[str(v) for v in sample_seeds],
                    "--robust-qaoa","--qaoa-restarts",str(qprimary.get("restarts",4)),
                    "--qaoa-depth",str(qprimary.get("depth",2)),
                    "--qaoa-objective",str(qprimary.get("objective","cvar")),
                    "--cvar-alpha",str(qprimary.get("cvar_alpha",0.1)),
                    "--parameter-scale",str(qprimary.get("parameter_scale","max_coefficient")),
                    "--eval-shots",str(qprimary.get("eval_shots",500)),
                ]
                rc,log=self._run_subprocess(f"dev_solvent_{solvent}_{target}",sensitivity_argv)
                logs.append(str(log));argvs.append(sensitivity_argv)
                metrics_path=out/"recovery_metrics.csv"
                if rc!=0 or not metrics_path.is_file():
                    solvent_failed.append(target.lower())
                    failures.append(f"dev solvent sensitivity {solvent}/{target} failed (exit={rc}; see {log})")
                else:
                    solvent_completed.append(target.lower())
                    with metrics_path.open(newline="",encoding="utf-8") as handle:
                        solvent_rows.extend(csv.DictReader(handle))
            solvent_root.mkdir(parents=True,exist_ok=True)
            if solvent_rows:
                fields=list(dict.fromkeys(k for row in solvent_rows for k in row))
                with (solvent_root/"recovery_metrics.csv").open("w",newline="",encoding="utf-8") as handle:
                    writer=csv.DictWriter(handle,fieldnames=fields)
                    writer.writeheader();writer.writerows(solvent_rows)
            solvent_planned=sorted({m.parent.name.lower() for m in manifests})
            solvent_closed=(
                set(solvent_completed)|set(solvent_failed)==set(solvent_planned)
                and not (set(solvent_completed)&set(solvent_failed))
            )
            atomic_write_json(solvent_root/"run_summary.json",{
                "scope":"development-only solvent sensitivity",
                "solvent_model":solvent,
                "planned_target_ids":solvent_planned,
                "completed_target_ids":sorted(solvent_completed),
                "failed_target_ids":sorted(solvent_failed),
                "rows":len(solvent_rows),
                "closed":solvent_closed,
            })
            (solvent_root/"recovery_report.md").write_text("\n".join([
                f"# Development solvent sensitivity: {solvent}","",
                "Development-only robustness analysis; confirmatory validation solvent is unchanged.",
                f"Planned targets: {len(solvent_planned)}; completed: {len(solvent_completed)}; failed: {len(solvent_failed)}.",
                f"Aggregated metric rows: {len(solvent_rows)}.",
            ])+"\n",encoding="utf-8")
            if not solvent_closed:
                failures.append(f"dev solvent sensitivity {solvent}: target accounting not closed")

        status = "failed" if failures else ("completed_with_failures" if queue_partial else "completed")
        detail = "; ".join(failures + queue_partial) if (failures or queue_partial) else \
            "Dev queue and validation queue structural experiments closed with no target failures."
        return StageResult("structure_experiment", status, started, utc_timestamp(), 0 if not failures else 1,
                            detail, argvs, ";".join(logs), not failures)

    # ================================================================
    # Stage 7: external VHH validation and mature structural baselines
    # ================================================================
    def stage_external_validation(self) -> StageResult:
        started=utc_timestamp()
        cfg=self.config.get("external_validation", {}) or {}
        failures=[];logs=[];argvs=[]
        qc=self.config["qc_benchmark"]
        qprimary=quantum_primary(self.config)
        homology=self.config["queue_freeze"]["homology_isolation"]
        rot=qc.get("rotamer_model", {}) or {}
        ff=qc.get("coarse_force_field", {}) or {}
        calibration=self.run_dir/(qc.get("energy_calibration", {}) or {}).get(
            "calibration_file","calibration/coarse_to_amber.json")
        checkpoint=self.checkpoint_dir()/qc.get("checkpoint","best_egnn_pruning.pt")
        rotamer_library=resolve_path(
            self.config,rot.get("library_path","data/rotamer/ALL.bbdep.rotamers.lib"))

        ext=cfg.get("external_vhh", {}) or {}
        if ext.get("required", False):
            graph_dir=resolve_path(self.config,ext.get("graph_dir",""))
            source_dir=resolve_path(self.config,ext.get("source_structure_dir",""))
            run_external_root=self.run_dir/"external_validation"
            run_external_root.mkdir(parents=True,exist_ok=True)
            # Always regenerated for this run (never reused from another run or
            # from an earlier failed attempt), because it certifies this run's
            # frozen training graphs and cluster map.
            independence=run_external_root/"external_vhh_independence_manifest.json"
            if independence.exists():
                independence.unlink()
            ext_failures=[]
            if not graph_dir.is_dir() or not any(graph_dir.glob("*.pt")):
                ext_failures.append(f"Required external VHH graph set missing/empty: {graph_dir}")
            if not source_dir.is_dir():
                ext_failures.append(f"Required external VHH raw structures missing: {source_dir}")
            if graph_dir.is_dir() and source_dir.is_dir() and any(graph_dir.glob("*.pt")):
                cluster_path=self.frozen_cluster_map_path()
                if cluster_path is None or not cluster_path.is_file():
                    ext_failures.append(
                        "Cannot generate external independence manifest without frozen cluster map"
                    )
                else:
                    audit_argv=[
                        self.venv_python,"-m", module_name("audit_external_vhh_independence.py"),
                        "--training-dataset",str(self.dataset_dir()),
                        "--external-graph-dir",str(graph_dir),
                        "--external-source-dir",str(source_dir),
                        "--cluster-map",str(cluster_path),
                        "--out",str(independence),
                        "--vhh-threshold",str(homology.get("vhh_full_chain_identity",0.80)),
                        "--cdr-h3-threshold",str(homology.get("cdr_h3_identity",0.50)),
                        "--antigen-threshold",str(homology.get("antigen_identity",0.30)),
                        "--antigen-min-length-coverage",
                            str(homology.get("antigen_min_length_coverage",0.70)),
                    ]
                    audit_rc,audit_log=self._run_subprocess("audit_external_vhh_independence",audit_argv)
                    logs.append(str(audit_log));argvs.append(audit_argv)
                    if audit_rc!=0 or not independence.is_file():
                        ext_failures.append(
                            f"External VHH independence audit failed (exit={audit_rc}; see {audit_log})"
                        )
            if not independence.is_file():
                ext_failures.append(f"Required external independence manifest missing: {independence}")
            else:
                manifest=json.loads(independence.read_text(encoding="utf-8"))
                current_cluster_path=self.frozen_cluster_map_path()
                current_cluster_sha=(
                    sha256_of(current_cluster_path)
                    if current_cluster_path is not None and current_cluster_path.is_file()
                    else None
                )
                if not manifest.get("training_family_overlap_zero",False):
                    ext_failures.append(
                        "External independence manifest does not certify zero training-family overlap"
                    )
                if manifest.get("graph_version")!="1.6":
                    ext_failures.append(
                        f"External VHH graph version must be 1.6, got {manifest.get('graph_version')}"
                    )
                if manifest.get("training_cluster_map_sha256") != current_cluster_sha:
                    ext_failures.append(
                        "External independence manifest is not bound to the current training cluster map"
                    )
                training_manifest=self.dataset_dir()/"graph_manifest.json"
                current_training_manifest_sha=(
                    sha256_of(training_manifest) if training_manifest.is_file() else None
                )
                if manifest.get("training_graph_manifest_sha256") != current_training_manifest_sha:
                    ext_failures.append(
                        "External independence manifest is not bound to the current training graph manifest"
                    )
                from nanoqc.data.audit_external_vhh_independence import graph_sequences
                current_external_records=[
                    graph_sequences(path,source_dir) for path in sorted(graph_dir.glob("*.pt"))
                ]
                audit_hashes=sorted(
                    str(row.get("graph_sha256",""))
                    for row in (manifest.get("targets") or [])
                    if row.get("graph_sha256")
                )
                current_external_hashes=sorted(row["sha256"] for row in current_external_records)
                audit_bindings=sorted(
                    (str(row.get("pdb_id","")).lower(),str(row.get("graph_sha256","")),
                     str(row.get("source_structure_sha256","")))
                    for row in (manifest.get("targets") or [])
                )
                current_bindings=sorted(
                    (row["pdb_id"],row["sha256"],row["source_structure_sha256"])
                    for row in current_external_records
                )
                if (audit_hashes != current_external_hashes
                        or audit_bindings != current_bindings
                        or manifest.get("target_count") != len(current_external_hashes)):
                    ext_failures.append(
                        "External graph files do not match the graph hashes certified by independence manifest"
                    )
                expected_homology={
                    "vhh_full_chain_identity":float(homology.get("vhh_full_chain_identity",0.80)),
                    "cdr_h3_identity":float(homology.get("cdr_h3_identity",0.50)),
                    "antigen_identity":float(homology.get("antigen_identity",0.30)),
                    "antigen_min_length_coverage":float(homology.get("antigen_min_length_coverage",0.70)),
                }
                observed_homology=manifest.get("homology_isolation")
                if observed_homology != expected_homology:
                    ext_failures.append(
                        f"External homology protocol mismatch: expected={expected_homology}, "
                        f"observed={observed_homology}"
                    )
                audits=manifest.get("targets")
                if not isinstance(audits,list) or not audits:
                    ext_failures.append("External independence manifest requires nonempty per-target audits")
                else:
                    from nanoqc.data.audit_external_vhh_independence import source_structure_for_pdb
                    for audit in audits:
                        try:
                            pdb=str(audit["pdb_id"]).lower()
                            source=source_structure_for_pdb(source_dir,pdb)
                            if (audit.get("source_structure")!=str(source.resolve())
                                    or audit.get("source_structure_sha256")!=sha256_of(source)):
                                ext_failures.append(f"{pdb}: raw structure provenance mismatch")
                            if float(audit["max_vhh_identity"]) >= expected_homology["vhh_full_chain_identity"]:
                                ext_failures.append(f"{pdb}: VHH identity overlap")
                            if float(audit["max_cdr_h3_identity"]) >= expected_homology["cdr_h3_identity"]:
                                ext_failures.append(f"{pdb}: CDR-H3 identity overlap")
                            if (
                                float(audit["max_antigen_identity"]) >= expected_homology["antigen_identity"]
                                and float(audit.get("antigen_length_coverage",1.0))
                                    >= expected_homology["antigen_min_length_coverage"]
                            ):
                                ext_failures.append(f"{pdb}: antigen identity overlap")
                            if bool(audit.get("family_cluster_overlap",True)):
                                ext_failures.append(f"{pdb}: family/structure cluster overlap or unverified")
                        except (KeyError,TypeError,ValueError) as exc:
                            ext_failures.append(f"Malformed external target audit: {audit!r} ({exc})")
            failures.extend(ext_failures)
            if not ext_failures:
                out=self.run_dir/"external_validation"/"vhh_coarse"
                external_repeats=[
                    derive_child_seed(
                        derive_streams(self.config["master_seed"])["perturb"],
                        "external_vhh_repeat",str(i)
                    )
                    for i in range(int(ext.get("repeats",10)))
                ]
                argv=[
                    self.venv_python,"-m", module_name("batch_benchmark_hard_set.py"),"--research-ablation",
                    "--input-dir",str(graph_dir),"--checkpoint",str(checkpoint),"--out-dir",str(out),
                    "--pruning",str(self.config.get("statistics",{}).get("primary_pruning","egnn")),
                    "--radii",str(self.config.get("statistics",{}).get("primary_radius",6.0)),
                    "--depths",str(qprimary.get("depth",2)),
                    "--max-evals",str(qprimary.get("max_evals",90)),
                    "--active-sites",str(
                        self.config.get("statistics",{}).get("primary_active_sites",6)),
                    "--vhh-identity-threshold",str(homology.get("vhh_full_chain_identity",0.80)),
                    "--cdr-h3-identity-threshold",str(homology.get("cdr_h3_identity",0.50)),
                    "--antigen-identity-threshold",str(homology.get("antigen_identity",0.30)),
                    "--antigen-min-length-coverage",str(homology.get("antigen_min_length_coverage",0.70)),
                    "--antigen-guidance-weight",str(qc.get("antigen_guidance_weight",0.25)),
                    "--antigen-proximity-scale",str(qc.get("antigen_proximity_scale_angstrom",6.0)),
                    "--contact-ca-cutoff",str(qc.get("contact_ca_cutoff_angstrom",8.0)),
                    "--nonbonded-cutoff",str(ff.get("cutoff_angstrom",8.0)),
                    "--softcore-delta",str(ff.get("softcore_delta_angstrom",0.5)),
                    "--hard-core-fraction",str(ff.get("hard_core_fraction",0.72)),
                    "--hard-sphere-penalty",str(ff.get("hard_sphere_penalty",25.0)),
                    "--lj-repulsion-cap",str(ff.get("lj_repulsion_cap",50.0)),
                    "--lj-attraction-cap",str(ff.get("lj_attraction_cap",5.0)),
                    "--coulomb-cap",str(ff.get("coulomb_cap",20.0)),
                    "--dielectric-base",str(ff.get("dielectric_base",4.0)),
                    "--dielectric-slope",str(ff.get("dielectric_slope",2.0)),
                    "--thermal-energy-kcal",str(ff.get("thermal_energy_kcal",0.593)),
                    "--rotamer-mode",str(rot.get("mode","dunbrack2010")),
                    "--rotamer-library",str(rotamer_library),
                    "--rotamer-probability-floor",str(rot.get("probability_floor",1e-4)),
                    "--rotamer-sigma-offsets",*[str(v) for v in rot.get("sigma_offsets",[-1,0,1])],
                    "--energy-calibration-file",str(calibration),"--require-calibrated-energy",
                    "--outputs",str(qprimary.get("output_shots",1000)),
                    "--qaoa-objective",str(qprimary.get("objective","cvar")),
                    "--qaoa-restarts",str(qprimary.get("restarts",4)),
                    "--cvar-alpha",str(qprimary.get("cvar_alpha",0.1)),
                    "--eval-shots",str(qprimary.get("eval_shots",500)),
                    "--parameter-scale",str(qprimary.get("parameter_scale","max_coefficient")),
                    "--sa-passes",str(qc.get("sa_passes",100)),
                    "--greedy-passes",str(qc.get("greedy_passes",50)),
                    "--energy-window",str(qc.get("energy_window",2.0)),
                    "--max-targets",str(ext.get("max_targets",0)),
                    "--workers",str(qc.get("workers",1)),
                    "--omp-threads",str(self.config.get("hardware",{}).get("cpu_threads_per_process",2)),
                    "--seeds",*[str(v) for v in external_repeats],
                    "--master-seed",str(self.config["master_seed"]),
                ]
                rc,log=self._run_subprocess("external_vhh_benchmark",argv)
                logs.append(str(log));argvs.append(argv)
                summary_path=out/"run_summary.json"
                if rc!=0 or not summary_path.is_file():
                    failures.append(f"External VHH benchmark failed (exit={rc}; see {log})")
                else:
                    external_summary=json.loads(summary_path.read_text(encoding="utf-8"))
                    if not external_summary.get("closed",False):
                        failures.append("External VHH benchmark is not closed")
                    if int(external_summary.get("failures_total",0) or 0)!=0:
                        failures.append(
                            f"External VHH benchmark has {external_summary.get('failures_total')} failed cases"
                        )
                    if not failures:
                        cluster_path=self.frozen_cluster_map_path()
                        if cluster_path is None or not cluster_path.is_file():
                            failures.append("External paired statistics require frozen family/structure cluster map")
                        else:
                            stats_argv=[
                                self.venv_python,"-m", module_name("batch_benchmark_hard_set.py"),"--paired-statistics",
                                "--results-dir",str(out),
                                "--resamples",str(self.config.get("statistics",{}).get("resamples",10000)),
                                "--seed",str(self.config["master_seed"]),
                                "--cluster-map",str(cluster_path),
                                "--budget-mode","outputs",
                                "--primary-pruning",str(
                                    self.config.get("statistics",{}).get("primary_pruning","egnn")),
                                "--primary-radius",str(
                                    self.config.get("statistics",{}).get("primary_radius",6.0)),
                                "--primary-depth",str(qprimary.get("depth",2)),
                                "--primary-max-evals",str(qprimary.get("max_evals",90)),
                                "--primary-outputs",str(qprimary.get("output_shots",1000)),
                                "--primary-objective",str(qprimary.get("objective","cvar")),
                                "--primary-restarts",str(qprimary.get("restarts",4)),
                                "--primary-active-sites",str(
                                    self.config.get("statistics",{}).get("primary_active_sites",6)),
                            ]
                            stats_rc,stats_log=self._run_subprocess("external_vhh_statistics",stats_argv)
                            logs.append(str(stats_log));argvs.append(stats_argv)
                            stats_json=out/"statistics_outputs.json"
                            if stats_rc!=0 or not stats_json.is_file():
                                failures.append(
                                    f"External VHH paired statistics failed (exit={stats_rc}; see {stats_log})"
                                )
                            else:
                                stats_payload=json.loads(stats_json.read_text(encoding="utf-8"))
                                ext_exclusions=stats_payload.get("exclusions",{}) or {}
                                ext_denominator_failures=paired_denominator_failures(
                                    ext_exclusions,"outputs")
                                if ext_denominator_failures:
                                    failures.append(
                                        "External VHH paired-statistics denominator is incomplete: "
                                        f"{ext_denominator_failures}")
                                sa_gap=next(
                                    (e for e in stats_payload.get("effects",[])
                                     if e.get("baseline")=="sa" and e.get("metric")=="gap"),
                                    None,
                                )
                                observed_clusters=0 if sa_gap is None else int(sa_gap.get("n_clusters",0) or 0)
                                required_clusters=int(ext.get("min_clusters",10))
                                if observed_clusters < required_clusters:
                                    failures.append(
                                        f"External VHH QAOA-vs-SA gap contrast has {observed_clusters} "
                                        f"independent clusters; requires >= {required_clusters}"
                                    )

        structural=cfg.get("structural_baselines", {}) or {}
        if structural.get("required", False):
            faspr=Path(structural.get("faspr_executable",""))
            phenix=Path(structural.get("phenix_clashscore_executable",""))
            if not faspr.is_file():
                failures.append(f"Required FASPR executable missing: {faspr}")
            if not phenix.is_file():
                failures.append(f"Required Phenix clashscore executable missing: {phenix}")
            if faspr.is_file() and phenix.is_file():
                out=self.run_dir/"external_validation"/"structural_baselines"
                argv=[
                    self.venv_python,"-m", module_name("run_external_structure_baselines.py"),
                    "--validation-dir",str(self.run_dir/"validation_queue"),
                    "--faspr",str(faspr),"--phenix-clashscore",str(phenix),
                    "--out-dir",str(out),
                    "--timeout-seconds",str(structural.get("timeout_seconds",1800)),
                    "--expected-seeds",*[str(v) for v in self.config["structure_experiment"].get("seeds",[42,43,44,45,46])],
                ]
                rc,log=self._run_subprocess("external_structure_baselines",argv)
                logs.append(str(log));argvs.append(argv)
                baseline_summary=out/"run_summary.json"
                if rc!=0 or not (out/"external_baseline_metrics.csv").is_file() or not baseline_summary.is_file():
                    failures.append(f"External structural baselines failed (exit={rc}; see {log})")
                else:
                    summary=json.loads(baseline_summary.read_text(encoding="utf-8"))
                    report_path=out/"external_baseline_report.md"
                    metrics_path=out/"external_baseline_metrics.csv"
                    row_count=0; methods=set(); targets=set()
                    if metrics_path.is_file():
                        with metrics_path.open(newline="",encoding="utf-8") as handle:
                            baseline_rows=list(csv.DictReader(handle))
                        row_count=len(baseline_rows)
                        methods={r.get("method","") for r in baseline_rows if r.get("method")}
                        targets={r.get("target","") for r in baseline_rows if r.get("target")}
                    report_path.write_text("\n".join([
                        "# External structural baseline result",
                        "",
                        f"Rows: {row_count}",
                        f"Targets: {len(targets)}",
                        f"Methods: {', '.join(sorted(methods)) if methods else 'n/a'}",
                        f"Failures: {len(summary.get('failures',[]) or [])}",
                        "",
                        "FASPR is a mature biological packing baseline; Phenix clashscore/common structural evaluation is applied consistently to external and internal structures.",
                    ])+"\n",encoding="utf-8")
                    if summary.get("failures"):
                        failures.append(
                            f"External structural baselines contain {len(summary['failures'])} failures"
                        )

        status="completed" if not failures else "failed"
        return StageResult(
            "external_validation",status,started,utc_timestamp(),0 if not failures else 1,
            "External validation complete." if not failures else "; ".join(failures),
            argvs[-1] if argvs else [],";".join(logs),not failures,
        )

    # ================================================================
    # Stage 8: paired statistics
    # ================================================================
    def stage_statistics(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config["statistics"]
        # --paired-statistics is specifically shaped for _ablation_main's
        # "<out-dir>/cases/*.json" layout (the coarse-grained qc_benchmark
        # ablation), not run_real_complex_pilot.py's per-target/per-seed CSV
        # layout (dev_queue/validation_queue) -- it errors ("No case JSON
        # artifacts") against the latter. The real-atom structural paired
        # analysis is handled separately by analyze_structure_recovery.py
        # using the pre-registered primary endpoint/contrast and cluster-aware
        # energy-to-structure inference.
        #
        # Each subprocess below gets its own child seed derived from the
        # "inference" stream (see seed_streams.py) instead of the bare
        # master seed, so paired-statistics per budget mode, quantum
        # scaling, and structural recovery -- three nominally-independent
        # confirmatory analyses -- never silently share one RNG stream.
        streams = derive_streams(self.config["master_seed"])
        # Statistical independence units must be identical to the cluster map
        # frozen by queue_freeze for this exact run.  Never resolve a second
        # repository-level statistics.cluster_map: that could make split
        # isolation and inferential clustering use different partitions.
        statistics_cluster_path = self.frozen_cluster_map_path()
        results_dirs = [self.run_dir / "qc_benchmark"]
        logs, argvs, failures = [], [], []
        qc_cfg = self.config["qc_benchmark"]
        qprimary=quantum_primary(self.config)
        primary_pruning = str(cfg.get("primary_pruning","egnn"))
        primary_radius = float(cfg.get("primary_radius",qc_cfg.get("radii",[6.0])[0]))
        primary_depth = int(qprimary.get("depth",2))
        primary_max_evals = int(qprimary.get("max_evals",90))
        primary_outputs = int(qprimary.get("output_shots",1000))
        primary_objective = str(qprimary.get("objective","cvar"))
        primary_restarts = int(qprimary.get("restarts",4))
        primary_active_sites = int(cfg.get("primary_active_sites", 6))
        if primary_active_sites not in [int(v) for v in qc_cfg.get("active_sites",[6])]:
            failures.append(
                f"statistics primary_active_sites={primary_active_sites} is not present in "
                f"qc_benchmark.active_sites")
        if primary_outputs not in qc_cfg.get("outputs", []):
            failures.append(
                f"statistics primary_outputs={primary_outputs} is not present in qc_benchmark.outputs")
        qablation=quantum_benchmark_ablation(self.config)
        if primary_objective not in [str(v) for v in qablation.get("objectives",[])]:
            failures.append(
                f"quantum primary objective={primary_objective} is not present in benchmark_ablation.objectives")
        if primary_restarts not in [int(v) for v in qablation.get("restarts",[])]:
            failures.append(
                f"quantum primary restarts={primary_restarts} is not present in benchmark_ablation.restarts")
        for results_dir in results_dirs:
            if not results_dir.is_dir():
                failures.append(f"statistics input directory is missing: {results_dir}")
                continue
            if failures:
                continue
            for budget_mode in cfg.get("budget_modes", ["outputs", "time"]):
                argv = [
                    self.venv_python, "-m", module_name("batch_benchmark_hard_set.py"), "--paired-statistics",
                    "--results-dir", str(results_dir),
                    "--resamples", str(cfg.get("resamples", 10000)),
                    "--seed", str(derive_child_seed(streams["inference"], "paired_statistics", budget_mode)),
                    "--budget-mode", budget_mode,
                    "--primary-pruning", primary_pruning,
                    "--primary-radius", str(primary_radius),
                    "--primary-depth", str(primary_depth),
                    "--primary-max-evals", str(primary_max_evals),
                    "--primary-outputs", str(primary_outputs),
                    "--primary-objective", primary_objective,
                    "--primary-restarts", str(primary_restarts),
                    "--primary-active-sites", str(primary_active_sites),
                    "--max-time-overrun-fraction", str(cfg.get("max_time_overrun_fraction", 0.10)),
                ]
                cluster_path=statistics_cluster_path
                if not cluster_path.is_file():
                    failures.append(f"Missing required run-local cluster map for statistics: {cluster_path}")
                    continue
                argv += ["--cluster-map", str(cluster_path)]
                returncode, log_path = self._run_subprocess(
                    f"statistics_{results_dir.name}_{budget_mode}", argv)
                logs.append(str(log_path)); argvs.append(argv)
                expected = [
                    results_dir / f"statistics_{budget_mode}.json",
                    results_dir / f"statistics_{budget_mode}.md",
                ]
                artifacts_ok, artifact_detail = self._artifacts_present(expected)
                if returncode != 0:
                    failures.append(f"{results_dir.name}/{budget_mode} exited {returncode} (see {log_path})")
                elif not artifacts_ok:
                    failures.append(
                        f"{results_dir.name}/{budget_mode} did not produce required statistics artifacts: "
                        f"{artifact_detail} (see {log_path})")
                else:
                    stats_payload=json.loads(expected[0].read_text(encoding="utf-8"))
                    exclusions=stats_payload.get("exclusions",{}) or {}
                    denominator_failures=paired_denominator_failures(exclusions,budget_mode)
                    if denominator_failures:
                        failures.append(
                            f"{results_dir.name}/{budget_mode}: paired-statistics denominator "
                            f"is incomplete: {denominator_failures}")
                    primary_effect=next(
                        (e for e in stats_payload.get("effects",[])
                         if e.get("baseline")==primary_qc_effect_name(
                             cfg.get("primary_qc_baseline","sa"),budget_mode)
                         and e.get("metric")==str(cfg.get("primary_qc_metric","gap"))),
                        None,
                    )
                    observed_clusters=0 if primary_effect is None else int(
                        primary_effect.get("n_clusters",0) or 0)
                    min_qc_clusters=int(cfg.get("min_qc_clusters",10))
                    if observed_clusters < min_qc_clusters:
                        failures.append(
                            f"{results_dir.name}/{budget_mode}: primary coarse contrast "
                            f"{cfg.get('primary_qc_baseline','sa')}/{cfg.get('primary_qc_metric','gap')} "
                            f"has {observed_clusters} independent clusters; requires >= {min_qc_clusters}")
        scaling_json=self.run_dir/"statistics"/"quantum_scaling_statistics.json"
        scaling_md=self.run_dir/"statistics"/"quantum_scaling_statistics.md"
        scaling_json.parent.mkdir(parents=True,exist_ok=True)
        cluster_path=statistics_cluster_path
        if cluster_path.is_file():
            scaling_argv=[
                self.venv_python,"-m", module_name("analyze_quantum_scaling.py"),
                "--results-dir",str(self.run_dir/"qc_benchmark"),
                "--cluster-map",str(cluster_path),
                "--out-json",str(scaling_json),
                "--out-md",str(scaling_md),
                "--primary-pruning",primary_pruning,
                "--primary-outputs",str(primary_outputs),
                "--primary-objective",primary_objective,
                "--primary-restarts",str(primary_restarts),
                "--primary-depth",str(primary_depth),
                "--primary-max-evals",str(primary_max_evals),
                "--primary-radius",str(primary_radius),
                "--baseline",str(cfg.get("primary_qc_baseline","sa")),
                "--active-sites",*[str(v) for v in qc_cfg.get("active_sites",[4,6,8,10])],
                "--resamples",str(cfg.get("resamples",10000)),
                "--seed",str(derive_child_seed(streams["inference"], "quantum_scaling")),
            ]
            scaling_rc,scaling_log=self._run_subprocess("quantum_scaling_statistics",scaling_argv)
            logs.append(str(scaling_log));argvs.append(scaling_argv)
            if scaling_rc!=0 or not scaling_json.is_file() or not scaling_md.is_file():
                failures.append(f"quantum scaling statistics exited {scaling_rc} (see {scaling_log})")
            else:
                scaling_payload=json.loads(scaling_json.read_text(encoding="utf-8"))
                scaling_clusters=int((scaling_payload.get("primary",{}) or {}).get("n_clusters",0) or 0)
                min_scaling=int(cfg.get("min_scaling_clusters",10))
                if scaling_clusters < min_scaling:
                    failures.append(
                        f"Scaling inference has {scaling_clusters} independent clusters; "
                        f"requires >= {min_scaling}")
                # Put the scaling slope in the same matched-output inferential
                # family as the QC effects. This prevents the scaling result
                # from being presented as an unadjusted confirmatory test.
                output_stats_path=self.run_dir / "qc_benchmark" / "statistics_outputs.json"
                if output_stats_path.is_file():
                    output_payload=json.loads(output_stats_path.read_text(encoding="utf-8"))
                    p_entries=[]
                    for effect in output_payload.get("effects",[]):
                        value=effect.get("p_value")
                        if value is not None and math.isfinite(float(value)):
                            p_entries.append(("qc:"+str(effect.get("baseline"))+":"+str(effect.get("metric")),effect))
                    scaling_primary=scaling_payload.get("primary",{}) or {}
                    scaling_p=scaling_primary.get("p_value")
                    if scaling_p is not None and math.isfinite(float(scaling_p)):
                        p_entries.append(("scaling:primary",scaling_primary))
                    if p_entries:
                        adjusted=holm_step_down([float(item[1].get("p_value")) for item in p_entries])
                        for (_, target), value in zip(p_entries, adjusted):
                            target["p_holm_global_qc_scaling"]=value
                        output_payload["multiplicity_family"]="matched-output QC effects plus primary scaling slope"
                        output_payload["multiplicity_n_tests"]=len(p_entries)
                        output_stats_path.write_text(json.dumps(output_payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
                        scaling_payload["multiplicity_family"]="matched-output QC effects plus primary scaling slope"
                        scaling_payload["multiplicity_n_tests"]=len(p_entries)
                        scaling_payload["primary"]=scaling_primary
                        scaling_json.write_text(json.dumps(scaling_payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
                        scaling_md.write_text(scaling_md.read_text(encoding="utf-8")+
                            f"\nHolm family: matched-output QC effects plus primary scaling slope (n={len(p_entries)}); "
                            f"adjusted scaling p={scaling_primary.get('p_holm_global_qc_scaling')}.\n",encoding="utf-8")
        else:
            failures.append(f"Quantum scaling statistics require run-local cluster map: {cluster_path}")

        validation_metrics = self.run_dir / "validation_queue" / "real_complex_metrics.csv"
        cluster_path=statistics_cluster_path
        if validation_metrics.is_file() and cluster_path.is_file():
            structure_json = self.run_dir / "statistics" / "structure_statistics.json"
            structure_md = self.run_dir / "statistics" / "structure_statistics.md"
            structure_json.parent.mkdir(parents=True, exist_ok=True)
            structure_argv = [
                self.venv_python, "-m", module_name("analyze_structure_recovery.py"),
                "--metrics", str(validation_metrics),
                "--cluster-map", str(cluster_path),
                "--out-json", str(structure_json),
                "--out-md", str(structure_md),
                "--primary-endpoint", str(cfg.get("primary_structural_endpoint", "final_rmsd")),
                "--primary-contrast", str(cfg.get("primary_structural_contrast", "qaoa_vs_sa")),
                "--resamples", str(cfg.get("resamples", 10000)),
                "--seed", str(derive_child_seed(streams["inference"], "structure_recovery")),
                "--expected-targets-file", str(
                    self.run_dir/"validation_queue"/"freeze"/"selected_targets.json"),
                "--expected-seeds", *[
                    str(v) for v in self.config.get("structure_experiment",{}).get(
                        "seeds",[42,43,44,45,46])
                ],
            ]
            returncode, log_path = self._run_subprocess("structure_statistics", structure_argv)
            if returncode != 0 or not structure_json.is_file():
                failures.append(f"structure statistics exited {returncode} (see {log_path})")
            else:
                structure_payload=json.loads(structure_json.read_text(encoding="utf-8"))
                primary_clusters=int((structure_payload.get("primary",{}) or {}).get("n_clusters",0) or 0)
                min_primary=int(cfg.get("min_primary_clusters",10))
                min_rq5=int(cfg.get("min_rq5_clusters",10))
                if primary_clusters < min_primary:
                    failures.append(
                        f"Primary structural inference has {primary_clusters} clusters; "
                        f"requires >= {min_primary}"
                    )
                failures.extend(rq5_inference_failures(
                    structure_payload.get("rq5") or {},min_rq5))
        elif not validation_metrics.is_file():
            failures.append(f"Missing validation structural metrics: {validation_metrics}")
        else:
            failures.append(f"Primary structural statistics require run-local frozen cluster map: {cluster_path}")

        status = "failed" if failures else "completed"
        detail = "; ".join(failures) if failures else (
            "Paired solver statistics plus pre-registered primary structural/RQ5 analysis completed."
        )
        return StageResult(
            "statistics", status, started, utc_timestamp(), 1 if failures else 0,
            detail, argvs, ";".join(logs), not failures
        )

    # ================================================================
    # Stage 8: final report
    # ================================================================
    def stage_final_report(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config.get("final_report", {})
        out_path = self.run_dir / cfg.get("filename", "FINAL_RESEARCH_REPORT.md")
        argv = [
            self.venv_python, "-m", module_name("generate_final_research_report.py"),
            "--run-dir", str(self.run_dir),
            "--out", str(out_path),
        ]
        returncode, log_path = self._run_subprocess("final_report", argv)
        ok, detail = self._artifacts_present([out_path])
        status = "completed" if (returncode == 0 and ok) else "failed"
        return StageResult("final_report", status, started, utc_timestamp(), returncode, detail,
                            argv, str(log_path), ok)

    # ================================================================
    def run_all(self) -> Dict[str, StageResult]:
        results: Dict[str, StageResult] = {}
        for stage in STAGE_ORDER:
            results[stage] = self.run_stage(
                stage, STAGE_PREREQUISITES[stage], getattr(self, f"stage_{stage}"))
            if stage == "smoke_check" and self.smoke_only:
                return results
        return results


def save_derived_child(streams: Dict[str, int], stream_name: str, *labels: str) -> int:
    from nanoqc.common.seed_streams import derive_child_seed
    return derive_child_seed(streams[stream_name], *labels)


# ---------------------------------------------------------------------------
# Run-directory / manifest / lock management
# ---------------------------------------------------------------------------

def build_run_manifest(config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    evidence_path=repo_root/DOCS_DIR/"METHODS_EVIDENCE.md"
    results_contract_path=repo_root/DOCS_DIR/"RESULTS_CONTRACT.md"
    # Fail closed: a missing orchestrated file must never silently drop out of
    # the code fingerprint that resume compares against.
    missing=[name for name in ORCHESTRATED_SCRIPTS if not repo_path(name, repo_root).is_file()]
    if missing:
        raise FileNotFoundError(f"Orchestrated source files missing under {repo_root}: {missing}")
    return dict(
        generated_utc=utc_timestamp(),
        git_commit=git_commit_hash(repo_root),
        master_seed=config["master_seed"],
        code_sha256={name: sha256_of(repo_path(name, repo_root)) for name in ORCHESTRATED_SCRIPTS},
        methods_evidence_sha256=sha256_of(evidence_path) if evidence_path.is_file() else None,
        results_contract_sha256=(
            sha256_of(results_contract_path) if results_contract_path.is_file() else None
        ),
        config=config,
    )


def new_run_dir(config: Dict[str, Any]) -> Path:
    root = resolve_path(config, config["paths"]["run_root"])
    root.mkdir(parents=True, exist_ok=True)
    prefix = config["paths"].get("run_prefix", "experiments_full_run_")
    candidate = root / f"{prefix}{utc_run_stamp()}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{prefix}{utc_run_stamp()}_{suffix}"
    candidate.mkdir(parents=True)
    return candidate


def resolve_resume_dir(config: Dict[str, Any], resume: str) -> Path:
    root = resolve_path(config, config["paths"]["run_root"])
    candidate = Path(resume)
    if not candidate.is_absolute():
        candidate = root / resume
    if not candidate.is_dir():
        raise SystemExit(f"--resume target does not exist or is not a directory: {candidate}")
    return candidate


def write_run_inventory(run_dir: Path, results: Dict[str, StageResult]) -> None:
    """Write one run-scoped artifact index and terminal summary.

    The inventory itself is excluded while scanning to avoid self-reference.
    Large binary result files are indexed by path/size; files <=64 MiB also
    receive SHA256 for convenient integrity checks without re-hashing very
    large datasets/checkpoints at shutdown.
    """
    inventory_path=run_dir/"artifact_inventory.json"
    summary_path=run_dir/"RUN_SUMMARY.json"
    entries=[]
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
        if path in (inventory_path,summary_path):
            continue
        rel=path.relative_to(run_dir).as_posix()
        size=path.stat().st_size
        item={"path":rel,"size_bytes":size}
        if size <= 64*1024*1024:
            try:
                item["sha256"]=sha256_of(path)
            except OSError:
                item["sha256"]=None
        entries.append(item)
    atomic_write_json(inventory_path,{
        "generated_utc":utc_timestamp(),
        "run_dir":str(run_dir),
        "file_count":len(entries),
        "total_size_bytes":sum(int(x["size_bytes"]) for x in entries),
        "artifacts":entries,
    })
    stage_status={name: result.status for name,result in results.items()}
    failed=sorted(name for name,status in stage_status.items() if status=="failed")
    audit_path=run_dir/"EXPERIMENT_RESULTS_AUDIT.json"
    audit_ok=(
        bool(json.loads(audit_path.read_text(encoding="utf-8")).get("all_required_results_present"))
        if audit_path.is_file() else False
    )
    atomic_write_json(summary_path,{
        "generated_utc":utc_timestamp(),
        "run_dir":str(run_dir),
        "status":"completed" if (not failed and audit_ok) else "failed",
        "failed_stages":failed,
        "stages":stage_status,
        "artifact_inventory":"artifact_inventory.json",
        "results_audit":"EXPERIMENT_RESULTS_AUDIT.json" if audit_path.is_file() else None,
        "results_audit_ok":audit_ok,
        "final_report":"FINAL_RESEARCH_REPORT.md" if (run_dir/"FINAL_RESEARCH_REPORT.md").is_file() else None,
    })


def audit_experiment_results(
    orchestrator: Orchestrator, results: Dict[str, StageResult]
) -> tuple[bool, dict]:
    """Re-validate every completed stage and write a run-level results audit."""
    records=[]
    all_ok=True
    for stage in STAGE_ORDER:
        result=results.get(stage)
        if result is None:
            continue
        if result.status in ("completed","completed_with_failures"):
            ok,detail=orchestrator._validate_completed_stage_artifacts(stage)
            if ok:
                chain_ok,chain_detail=orchestrator._validate_upstream_chain_fresh(stage)
                if not chain_ok:
                    ok,detail=False,"artifacts intact but upstream chain is stale: "+chain_detail
        elif result.status=="skipped":
            existing=orchestrator._load_stage_status(stage)
            protocol_disabled=not orchestrator.config.get("stages",{}).get(stage,True)
            optional_smoke=(stage=="smoke_check")
            if existing and existing.get("status") in ("completed","completed_with_failures"):
                ok,detail=orchestrator._validate_completed_stage_artifacts(stage)
                detail="skipped this invocation; existing completed artifacts revalidated: "+detail
                if ok:
                    chain_ok,chain_detail=orchestrator._validate_upstream_chain_fresh(stage)
                    if not chain_ok:
                        ok,detail=False,"artifacts intact but upstream chain is stale: "+chain_detail
            elif protocol_disabled or optional_smoke:
                ok,detail=True,"stage skipped by frozen protocol as optional/disabled"
            else:
                ok,detail=False,"required stage skipped without a previously completed auditable result"
        else:
            ok,detail=False,"stage failed; no complete experimental result set"
        records.append({
            "stage":stage,
            "status":result.status,
            "results_contract_ok":ok,
            "detail":detail,
        })
        if result.status in ("completed","completed_with_failures") and not ok:
            all_ok=False
        if result.status=="skipped" and not ok:
            all_ok=False
        if result.status=="failed":
            all_ok=False
    payload={
        "generated_utc":utc_timestamp(),
        "run_dir":str(orchestrator.run_dir),
        "all_required_results_present":all_ok,
        "stages":records,
    }
    atomic_write_json(orchestrator.run_dir/"EXPERIMENT_RESULTS_AUDIT.json",payload)
    lines=[
        "# Experiment results audit","",
        f"All required results present: **{all_ok}**","",
        "| Stage | Status | Result contract | Detail |",
        "|---|---|---|---|",
    ]
    for row in records:
        safe=str(row["detail"]).replace("|","\\|").replace("\n"," ")
        lines.append(
            f"| {row['stage']} | {row['status']} | "
            f"{'PASS' if row['results_contract_ok'] else 'FAIL'} | {safe} |"
        )
    (orchestrator.run_dir/"EXPERIMENT_RESULTS_AUDIT.md").write_text(
        "\n".join(lines)+"\n",encoding="utf-8")
    return all_ok,payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / CONFIGS_DIR / "full_experiment_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                         help="Continue a specific previous run directory (name or path) instead of "
                              "starting a fresh one.")
    parser.add_argument("--run-dir", type=Path, default=None,
                         help="Use this pre-created directory for a NEW run. Intended for the one-click "
                              "launcher so preflight/orchestrator logs and every artifact share one run directory.")
    parser.add_argument("--only", type=str, default=None, choices=STAGE_ORDER,
                         help="Run only this stage (its prerequisites must already be completed).")
    parser.add_argument("--force-restage", type=str, nargs="+", default=[],
                         choices=STAGE_ORDER,
                         help="Re-run these stage(s) even if already marked completed/failed.")
    parser.add_argument("--smoke-only", action="store_true",
                         help="Run only env_check + smoke_check, then stop (for a fast preflight pass).")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.smoke_only and args.resume:
        parser.error("--smoke-only creates a separate disposable preflight run and cannot be combined with --resume")
    if args.resume and args.run_dir is not None:
        parser.error("--resume and --run-dir are mutually exclusive")

    config = apply_runtime_mode_overrides(load_config(args.config), smoke_only=args.smoke_only)
    repo_root = Path(config["paths"]["repo_root"]).resolve()
    lock_root = resolve_path(config, config["paths"]["run_root"])
    lock_root.mkdir(parents=True, exist_ok=True)
    lock_path = lock_root / config.get("control", {}).get("lock_file_name", ".run_full_experiment.lock")

    try:
        lock = FileLock(str(lock_path), timeout=0)
        lock.acquire()
    except FileLockTimeout:
        print(f"ERROR: another run_full_experiment.py is already active (lock held: {lock_path}). "
              "Refusing to start a second, concurrent run.", file=sys.stderr)
        return 1

    try:
        if args.resume:
            run_dir = resolve_resume_dir(config, args.resume)
            manifest_path = run_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise SystemExit(f"--resume target has no run_manifest.json (not a run_full_experiment.py "
                                  f"run directory?): {run_dir}")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            current = build_run_manifest(config, repo_root)
            if (previous.get("code_sha256") != current["code_sha256"]
                    or previous.get("methods_evidence_sha256") != current["methods_evidence_sha256"]
                    or previous.get("results_contract_sha256") != current["results_contract_sha256"]
                    or previous.get("config") != current["config"]):
                raise SystemExit(
                    "Refusing to resume: orchestrated code, methods evidence, results contract, or config differs from the original launch. "
                    "Start a fresh run directory for changed code/evidence/config (this repository's established "
                    "rule: a changed source hash, literature-evidence hash, or configuration always gets a new output directory)."
                )
            seed_map_path = run_dir / "seed_streams.json"
            if seed_map_path.is_file() and not verify_stream_map(seed_map_path):
                raise SystemExit("Refusing to resume: seed_streams.json no longer matches seed_streams.py's "
                                  "derivation (seed_streams.py itself changed?).")
            print(f"Resuming run: {run_dir}")
        else:
            if args.run_dir is not None:
                run_dir=args.run_dir.expanduser().resolve()
                configured_root=resolve_path(config,config["paths"]["run_root"]).resolve()
                try:
                    run_dir.relative_to(configured_root)
                except ValueError:
                    raise SystemExit(
                        f"--run-dir must be inside configured run_root {configured_root}: {run_dir}")
                run_dir.mkdir(parents=True,exist_ok=True)
                if (run_dir/"run_manifest.json").exists():
                    raise SystemExit(
                        f"Refusing fresh launch into an existing formal run with run_manifest.json: {run_dir}")
            else:
                run_dir = new_run_dir(config)
            manifest = build_run_manifest(config, repo_root)
            atomic_write_json(run_dir / "run_manifest.json", manifest)
            frozen_config_path = run_dir / "frozen_config.yaml"
            frozen_config_path.write_text(
                yaml.safe_dump(config,sort_keys=False,allow_unicode=True),
                encoding="utf-8",
            )
            streams = derive_streams(config["master_seed"])
            save_stream_map(run_dir / "seed_streams.json", config["master_seed"], streams)
            print(f"New run: {run_dir}")
            print(f"Derived seed streams: {streams}")

        orchestrator = Orchestrator(config, run_dir, only=args.only, smoke_only=args.smoke_only,
                                     force_restage=args.force_restage)
        results = orchestrator.run_all()

        print("\n=== Stage summary ===")
        overall_ok = True
        for stage in STAGE_ORDER:
            result = results.get(stage)
            if result is None:
                continue
            print(f"  {stage:24s} {result.status}")
            if result.status == "failed":
                overall_ok = False
        audit_ok,_=audit_experiment_results(orchestrator,results)
        if not audit_ok:
            overall_ok=False
        write_run_inventory(run_dir,results)
        print(f"\nRun directory: {run_dir}")
        print(f"Results audit: {run_dir/'EXPERIMENT_RESULTS_AUDIT.json'}")
        print(f"Artifact inventory: {run_dir/'artifact_inventory.json'}")
        print(f"Run summary: {run_dir/'RUN_SUMMARY.json'}")
        return 0 if overall_ok else 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
