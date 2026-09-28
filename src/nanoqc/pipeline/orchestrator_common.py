"""Constants, stage bookkeeping and small helpers shared by the orchestrator.

Stage order and prerequisites, the code-fingerprint file list, quantum
protocol accessors, cluster-adequacy arithmetic, ``StageResult`` and
filesystem/time helpers. Nothing here launches a stage.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]


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


def cluster_adequacy(config: Dict[str, Any], dataset_dir: Path, selected_targets_path: Path,
                     cluster_map_path: Path) -> Dict[str, Any]:
    """Outcome-free count of independent clusters available to each inference.

    Cluster-level inference is unreliable with few clusters (Cameron & Miller
    2015), and the statistics stage enforces preregistered minima. Counting
    the clusters once the queue and split are frozen -- before any EGNN
    training, benchmark or structure run -- turns a late, costly failure into
    an early one without looking at any result (PROTOCOL_AMENDMENTS.md A5).
    """
    cluster_map={str(k).lower():str(v) for k,v in
                 json.loads(Path(cluster_map_path).read_text(encoding="utf-8")).items()}
    manifest=json.loads((Path(dataset_dir)/"graph_manifest.json").read_text(encoding="utf-8"))
    hard_pdbs=sorted({str(r.get("pdb_id","")).lower() for r in manifest if r.get("split")=="test_snac_hard"})
    selected=json.loads(Path(selected_targets_path).read_text(encoding="utf-8"))
    validation_pdbs=sorted({str((e.get("target") or e.get("pdb_id")) if isinstance(e,dict) else e).lower()
                            for e in selected})
    def clusters(pdbs):
        missing=[p for p in pdbs if p not in cluster_map]
        return len({cluster_map[p] for p in pdbs if p in cluster_map}),missing
    holdout_pdbs=sorted({str(r.get("pdb_id","")).lower() for r in manifest if r.get("split")=="holdout"})
    hard_clusters,hard_missing=clusters(hard_pdbs)
    validation_clusters,validation_missing=clusters(validation_pdbs)
    holdout_clusters,holdout_missing=clusters(holdout_pdbs)
    stats=config.get("statistics",{}) or {}
    external=((config.get("external_validation",{}) or {}).get("external_vhh",{}) or {})
    requirements=[
        ("coarse primary QC contrast (test_snac_hard)",hard_clusters,int(stats.get("min_qc_clusters",10))),
        ("scaling slope (test_snac_hard)",hard_clusters,int(stats.get("min_scaling_clusters",10))),
        ("structural primary endpoint (validation queue)",validation_clusters,int(stats.get("min_primary_clusters",10))),
        ("RQ5 (validation queue)",validation_clusters,int(stats.get("min_rq5_clusters",10))),
    ]
    # The antigen-fold holdout is scored by the external stage, which enforces
    # the same minimum on family/structure clusters (A5 counts it here too, so
    # a shortfall surfaces before any training rather than at the last stage).
    if holdout_pdbs and external.get("required",False):
        requirements.append(("antigen-fold holdout (external_validation)",holdout_clusters,
                             int(external.get("min_clusters",10))))
    shortfalls=[f"{name}: {have} independent clusters < required {need}"
                for name,have,need in requirements if have<need]
    if hard_missing or validation_missing or holdout_missing:
        shortfalls.append(
            f"cluster map lacks PDBs: {(hard_missing+validation_missing+holdout_missing)[:20]}")
    return dict(
        schema="cluster_adequacy_v1",outcome_free=True,
        test_snac_hard_pdbs=len(hard_pdbs),test_snac_hard_clusters=hard_clusters,
        validation_queue_pdbs=len(validation_pdbs),validation_queue_clusters=validation_clusters,
        holdout_pdbs=len(holdout_pdbs),holdout_clusters=holdout_clusters,
        requirements=[dict(inference=n,available=h,required=r) for n,h,r in requirements],
        adequate=not shortfalls,shortfalls=shortfalls,
    )


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
    # Pre-specified non-estimable outcomes (constant cluster-level difference)
    # are reported results, not failures; see docs/PROTOCOL_AMENDMENTS.md A2.
    if str(rq5.get("estimability","")).startswith("not_estimable_"):
        return failures
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
# Must equal build_final_pyg_dataset.VERSION (kept literal so the orchestrator
# does not import torch/PyG at start-up; a test pins the two together).
REQUIRED_GRAPH_VERSION = "1.11"
# Must equal qaoa_interface_sampler.MAX_QAOA_DEPTH (literal to avoid importing
# PennyLane at start-up; a test pins the two together).
MAX_QAOA_DEPTH = 12

ORCHESTRATED_SCRIPTS: List[str] = [
    "run_full_experiment.py",
    # The orchestrator itself is split across these modules; each is fingerprinted
    # so a change to any stage's orchestration invalidates a resume.
    "orchestrator_common.py",
    "config_validation.py",
    "run_records.py",
    "stages_data.py",
    "stages_training.py",
    "stages_quantum.py",
    "stages_structure.py",
    "stages_reporting.py",
    "stage_contracts.py",
    "resolve_server_config.py",
    "seed_streams.py",
    "audit_all_datasets.py",
    "build_final_pyg_dataset.py",
    "build_external_vhh_graphs.py",
    "build_foldseek_pairs.py",
    "build_independence_cluster_map.py",
    "train_egnn_pruning.py",
    "egnn_seed_sensitivity.py",
    "generate_energy_calibration_dataset.py",
    "batch_benchmark_hard_set.py",
    "run_real_complex_pilot.py",
    "run_external_structure_baselines.py",
    "audit_external_vhh_independence.py",
    "carve_holdout_clusters.py",
    "generate_final_research_report.py",
    "analyze_structure_recovery.py",
    "analyze_quantum_scaling.py",
    "analyze_quantum_exploration.py",
    "fit_qaoa_transfer_parameters.py",
    "model_egnn_pruning.py",
    "subgraph_to_qubo.py",
    "qaoa_interface_sampler.py",
    # Formal QUBO/Ising instance and QAOA logical-resource accounting,
    # imported by subgraph_to_qubo / qaoa_interface_sampler / the benchmark.
    "instance.py",
    "resource_estimation.py",
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
    "quantum_exploration",
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
    "quantum_exploration": ["egnn_train", "energy_calibration"],
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
    """Best-effort installed-version report; missing packages record None.

    Imported modules without ``__version__`` fall back to installed
    distribution metadata; this remains diagnostic and installs nothing.
    """
    import importlib
    from importlib import metadata
    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            module = importlib.import_module(name)
        except Exception:
            versions[name] = None
            continue
        module_version = getattr(module, "__version__", None)
        if module_version:
            versions[name] = str(module_version)
            continue
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unknown"
    return versions


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


def calibration_solver_args(calibration_cfg: Dict[str, Any], calibration_file: Path,
                            *, force_required: bool = False) -> List[str]:
    """Never feed diagnostic fit coefficients into a solver invocation."""
    if str(calibration_cfg.get("mode", "frozen")) != "frozen":
        return []
    required = bool(calibration_cfg.get("require_calibrated", False) or force_required)
    argv = ["--require-calibrated-energy"] if required else []
    if calibration_file.is_file() or force_required:
        argv += ["--energy-calibration-file", str(calibration_file)]
    return argv
