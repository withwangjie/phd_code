#!/usr/bin/env python3
"""run_full_experiment.py -- single unattended entry point for the full
nanobody-interface quantum/classical benchmark research pipeline.

This is an ORCHESTRATOR, not a reimplementation: every stage below shells
out to an existing, already-implemented, already-documented module in this
repository (audit_all_datasets.py, build_final_pyg_dataset.py,
train_egnn_pruning.py, batch_benchmark_hard_set.py, run_real_complex_pilot.py,
generate_final_research_report.py) via subprocess, exactly as
RESEARCH_EXPERIMENTS_README.md / REAL_COMPLEX_EXPERIMENT_PROTOCOL.md already
document running them by hand, one command at a time. This script's only job
is to chain them safely, unattended, with provenance, resumability, and
honest failure reporting -- it contains no scientific logic of its own.

Stage order (each stage checks its own prerequisite stage's completion
marker before running; toggled individually in full_experiment_config.yaml
under `stages:`):

    env_check -> smoke_check -> data_audit -> queue_freeze (dataset split,
    no cap, + validation-queue selection+freeze) -> egnn_train ->
    qc_benchmark (pruning x budget x objective ablation) ->
    structure_experiment (dev queue + validation queue recovery-benchmark)
    -> statistics (paired analysis) -> final_report

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

    python run_full_experiment.py --config full_experiment_config.yaml
    python run_full_experiment.py --config full_experiment_config.yaml --resume experiments_full_run_20260920_010203
    python run_full_experiment.py --config full_experiment_config.yaml --only qc_benchmark
    python run_full_experiment.py --config full_experiment_config.yaml --smoke-only

Requires PyYAML (``pip install pyyaml``; add it to requirements-quantum.txt
if it is not already installed) and ``filelock`` (already a project
dependency via requirements-quantum.txt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

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
        "The `filelock` package is required (already listed in requirements-quantum.txt). "
        f"Import failed: {exc}"
    )

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from seed_streams import derive_streams, derive_child_seed, save_stream_map, verify_stream_map, DEFAULT_MASTER_SEED  # noqa: E402

# ---------------------------------------------------------------------------
# Scripts this orchestrator shells out to. Every one of these is fingerprinted
# (SHA-256) into run_manifest.json for provenance, exactly like every other
# entrypoint in this repository already fingerprints its own code
# dependencies (_ablation_main / _recovery_benchmark_main / build_manifest).
# ---------------------------------------------------------------------------
ORCHESTRATED_SCRIPTS: List[str] = [
    "run_full_experiment.py",
    "seed_streams.py",
    "audit_all_datasets.py",
    "build_final_pyg_dataset.py",
    "train_egnn_pruning.py",
    "batch_benchmark_hard_set.py",
    "run_real_complex_pilot.py",
    "generate_final_research_report.py",
    "model_egnn_pruning.py",
    "subgraph_to_qubo.py",
    "qaoa_interface_sampler.py",
    "evaluate_complex_metrics.py",
    "prediction_contract.py",
    "structural_quality.py",
]

STAGE_ORDER: List[str] = [
    "env_check",
    "smoke_check",
    "data_audit",
    "queue_freeze",
    "egnn_train",
    "qc_benchmark",
    "structure_experiment",
    "statistics",
    "final_report",
]


# ---------------------------------------------------------------------------
# Small, dependency-free helpers (mirroring the atomic-write / hashing
# patterns already used throughout batch_benchmark_hard_set.py /
# build_final_pyg_dataset.py, kept local here rather than importing private
# helpers across modules).
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    validation = qf.get("validation_queue", {}) or {}

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

    qc_sites = int(qc.get("active_sites", 6))
    validation_sites = int(validation.get("sites", 6))
    if not 5 <= qc_sites <= 8:
        raise ValueError("qc_benchmark.active_sites must be in 5..8")
    if validation_sites != qc_sites:
        raise ValueError(
            f"Formal active-site count must match: qc_benchmark={qc_sites}, validation_queue={validation_sites}"
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
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
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
        full_env["OPENMM_CPU_THREADS"] = str(hardware.get("openmm_cpu_threads", 8))
        full_env["QP_OPENMM_PLATFORM"] = str(hardware.get("openmm_platform", "Reference"))
        full_env["QP_OPENMM_DEVICE"] = str(hardware.get("openmm_device", "0"))
        full_env["QP_OPENMM_PRECISION"] = str(hardware.get("openmm_precision", "double"))
        if env:
            full_env.update(env)
        with log_path.open("a", encoding="utf-8") as log_handle:
            log_handle.write(f"\n=== {started} :: {' '.join(argv)} ===\n")
            log_handle.flush()
            process = subprocess.run(
                argv, cwd=str(cwd or self.repo_root), env=full_env,
                stdout=log_handle, stderr=subprocess.STDOUT,
            )
        return process.returncode, log_path

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
            # (packaging-round fix) A stage disabled via
            # `stages.<name>: false` in config -- e.g. smoke_check disabled
            # for a formal deployment run that must never execute a
            # trial/smoke sub-run -- must still PERSIST a "skipped" status
            # marker. Previously this branch returned without calling
            # _save_stage_status(), so on a fresh run_dir no
            # stage_status/<stage>.json ever existed for it; any downstream
            # stage listing it as a prerequisite would then find `record is
            # None` below and refuse to start ("Prerequisite stage has not
            # completed"), silently deadlocking the whole pipeline the
            # first time anyone disabled a stage on a brand-new run rather
            # than an already-completed one. Persisting "skipped" here (and
            # accepting "skipped" as a satisfied prerequisite just below)
            # makes disabling a stage in config behave the way the config
            # file's own comment already promises ("allows re-running a
            # subset without editing the script") for a fresh run too, not
            # only for a --resume of a run where it had previously run.
            result = StageResult(stage, "skipped", utc_timestamp(), utc_timestamp(),
                                  None, "Skipped (disabled in full_experiment_config.yaml).")
            self._save_stage_status(result)
            return result
        for prereq in prerequisites:
            record = self._load_stage_status(prereq)
            optional_skip = (prereq == "smoke_check" and record
                             and record["status"] == "skipped")
            if not record or (record["status"] not in ("completed", "completed_with_failures") and not optional_skip):
                result = StageResult(stage, "failed", utc_timestamp(), utc_timestamp(), None,
                                      f"Prerequisite stage '{prereq}' has not completed; refusing to start.")
                self._save_stage_status(result)
                return result
        existing = self._load_stage_status(stage)
        if existing and stage not in self.force_restage:
            if existing["status"] in ("completed", "completed_with_failures"):
                print(f"[{stage}] Already {existing['status']}; skipping (use --force-restage {stage} to redo).")
                return StageResult(stage, existing["status"], existing["started_utc"],
                                    existing["finished_utc"], existing["returncode"],
                                    "Resumed: previously " + existing["status"] + ".",
                                    existing.get("argv", []), existing.get("log_path"),
                                    existing.get("artifacts_ok", True))
            if existing["status"] == "failed":
                print(f"[{stage}] Previously FAILED systemically: {existing['detail']}")
                print(f"[{stage}] Not auto-retrying. Pass --force-restage {stage} to retry after investigating.")
                return StageResult(stage, "failed", existing["started_utc"], utc_timestamp(),
                                    existing["returncode"], "Not retried: " + existing["detail"])
        print(f"[{stage}] Starting.")
        try:
            result = fn()
        except Exception:
            result = StageResult(stage, "failed", utc_timestamp(), utc_timestamp(), None,
                                  "Unhandled exception in orchestrator stage function:\n" + traceback.format_exc())
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

    # ================================================================
    # Stage 0: environment check
    # ================================================================
    def stage_env_check(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config.get("env_check", {})
        required = cfg.get("required_python_packages", [])
        versions = package_versions(required)
        missing = [name for name, version in versions.items() if version is None]
        record = dict(
            python=sys.version, platform=platform.platform(),
            git_commit=git_commit_hash(self.repo_root),
            package_versions=versions, missing_packages=missing,
        )
        atomic_write_json(self.run_dir / "env_check.json", record)
        status = "completed" if not missing else "failed"
        detail = "Environment recorded." if not missing else f"Missing required packages: {missing}"
        return StageResult("env_check", status, started, utc_timestamp(), 0, detail)

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

        smoke_input_dir = self.dataset_dir() / self.config["qc_benchmark"]["input_dir"]
        if smoke_input_dir.is_dir() and any(smoke_input_dir.glob("*.pt")):
            checks.append(("qc_benchmark_smoke", [
                self.venv_python, "batch_benchmark_hard_set.py", "--research-ablation",
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
            ]))
        else:
            print("[smoke_check] qc_benchmark input_dir not yet built; skipping that sub-check "
                  "(expected before queue_freeze has run).")

        checks.append(("real_complex_smoke", [
            self.venv_python, "run_real_complex_pilot.py",
            "--out-dir", str(smoke_dir / "real_complex"),
            "--targets", str(smoke_cfg.get("recovery_pilot_targets", 1)),
            "--sites", str(smoke_cfg.get("recovery_pilot_sites", 6)),
            "--seeds", *[str(s) for s in smoke_cfg.get("recovery_pilot_seeds", [42])],
            "--max-evals", str(smoke_cfg.get("recovery_pilot_max_evals", 12)),
            "--outputs", str(smoke_cfg.get("recovery_pilot_outputs", 20)),
            "--pruning", "contact",  # avoid requiring a trained checkpoint for the smoke check
        ]))

        failures = []
        for name, argv in checks:
            returncode, log_path = self._run_subprocess(f"smoke_{name}", argv)
            if returncode != 0:
                failures.append(f"{name} exited {returncode} (see {log_path})")
        status = "completed" if not failures else "failed"
        detail = "All smoke checks passed." if not failures else "; ".join(failures)
        return StageResult("smoke_check", status, started, utc_timestamp(), 0 if not failures else 1, detail)

    # ================================================================
    # Stage 2: data audit
    # ================================================================
    def stage_data_audit(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config.get("data_audit", {})
        argv = [
            self.venv_python, "audit_all_datasets.py",
            "--data", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--workers", str(cfg.get("workers", 4)),
            "--limit", str(cfg.get("limit", 0)),
            "--out", str(self.run_dir / "audit"),
        ]
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
            self.venv_python, "build_final_pyg_dataset.py",
            "--out", str(dataset_dir),
            "--workers", str(qf_cfg["graph_build"].get("workers", 2)),
            "--audit-dir", str(self.run_dir / "audit"),
            "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--cdr-h3-identity-threshold", str(qf_cfg["homology_isolation"].get("cdr_h3_identity", 0.50)),
            "--interface-label-cutoff", str(qf_cfg["graph_build"].get("interface_label_cutoff_angstrom", 5.0)),
            "--intra-chain-ca-cutoff", str(qf_cfg["graph_build"].get("intra_chain_ca_cutoff_angstrom", 8.0)),
            "--cross-partner-knn-k", str(qf_cfg["graph_build"].get("cross_partner_knn_k", 3)),
            "--min-interface-residues", str(qf_cfg["graph_build"].get("min_interface_residues", 15)),
        ]
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
        bootstrap_pruning = vq_cfg.get("eligibility_bootstrap_pruning", "contact")
        if bootstrap_pruning == "egnn":
            return StageResult(
                "queue_freeze", "failed", started, utc_timestamp(), None,
                "full_experiment_config.yaml queue_freeze.validation_queue.eligibility_bootstrap_pruning "
                "is 'egnn', but queue_freeze runs before egnn_train (STAGE_ORDER) so no trained checkpoint "
                "exists yet. The bootstrap eligibility pass must use a checkpoint-free strategy (e.g. "
                "'contact'); the real 'egnn' protocol in validation_queue.pruning is applied later, "
                "per-target, in stage_structure_experiment once egnn_train has completed.")
        validation_seed = save_derived_child(streams, "perturb", "validation_queue_selection_order")
        validation_dir = self.run_dir / "validation_queue"
        vq_argv = [
            self.venv_python, "run_real_complex_pilot.py",
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
            "--exclude-pdb", *dev_cfg.get("excluded_pdb", []),
            "--dev-exposed-pdb", *dev_cfg.get("excluded_pdb", []),
            "--selection-order", vq_cfg.get("selection_order", "seeded_random"),
            "--selection-seed", str(validation_seed),
            "--queue-role", "validation",
            "--prepare-only",
        ]
        returncode, vq_log = self._run_subprocess("queue_freeze_validation_queue", vq_argv)
        vq_expected = [validation_dir / name for name in ("eligibility.json", "selected_targets.json")]
        vq_ok, vq_detail = self._artifacts_present(vq_expected)
        if returncode != 0 or not vq_ok:
            return StageResult("queue_freeze", "failed", started, utc_timestamp(), returncode,
                                f"run_real_complex_pilot.py (validation queue) exited {returncode}; {vq_detail} (see {vq_log})",
                                vq_argv, str(vq_log), vq_ok)
        selected = json.loads((validation_dir / "selected_targets.json").read_text(encoding="utf-8"))
        cap_label = vq_cfg.get("target_count", 0) or "unlimited (all qualifying targets)"
        detail = (f"Graph build: {graph_detail} Validation queue: {len(selected)} targets frozen "
                  f"(cap: {cap_label}; dev queue excluded+exposure-flagged: {dev_cfg.get('excluded_pdb', [])}). {vq_detail}")
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
            self.venv_python, "train_egnn_pruning.py",
            "--data-dir", str(train_data_dir),
            "--expected-graphs", str(len(graphs)),
            "--checkpoint-dir", str(checkpoint_dir),
            "--max-epochs", str(cfg.get("max_epochs", 50)),
            "--patience", str(cfg.get("patience", 5)),
            "--batch-size", str(cfg.get("batch_size", 2)),
            "--hidden-dim", str(cfg.get("hidden_dim", 32)),
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
    # Stage 5: quantum-vs-classical ablation benchmark
    # ================================================================
    def stage_qc_benchmark(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config["qc_benchmark"]
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
            self.venv_python, "batch_benchmark_hard_set.py", "--research-ablation",
            "--input-dir", str(self.dataset_dir() / cfg["input_dir"]),
            "--checkpoint", str(self.checkpoint_dir() / cfg["checkpoint"]),
            "--out-dir", str(out_dir),
            "--seeds", *[str(s) for s in repeat_seeds],
            "--master-seed", str(self.config["master_seed"]),
            "--pruning", *cfg.get("pruning", ["egnn", "contact", "distance", "cdr", "random"]),
            "--radii", *[str(r) for r in cfg.get("radii", [6.0, 10.0])],
            "--depths", *[str(d) for d in cfg.get("depths", [1, 2, 3])],
            "--max-evals", *[str(m) for m in cfg.get("max_evals", [90, 300])],
            "--active-sites", str(cfg.get("active_sites", 6)),
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
            "--qaoa-objective", *cfg.get("qaoa_objective", ["mean", "cvar"]),
            "--qaoa-restarts", *[str(r) for r in cfg.get("qaoa_restarts", [1, 4])],
            "--cvar-alpha", str(cfg.get("cvar_alpha", 0.1)),
            "--eval-shots", str(cfg.get("eval_shots", 500)),
            "--parameter-scale", cfg.get("parameter_scale", "max_coefficient"),
            "--sa-passes", str(cfg.get("sa_passes", 100)),
            "--greedy-passes", str(cfg.get("greedy_passes", 50)),
            "--energy-window", str(cfg.get("energy_window", 2.0)),
            "--max-targets", str(cfg.get("max_targets", 0)),
            "--workers", str(cfg.get("workers", 1)),
            "--omp-threads", str(self.config.get("hardware", {}).get("cpu_threads_per_process", 2)),
        ]
        calibration_cfg = cfg.get("energy_calibration", {}) or {}
        calibration_file = resolve_path(self.config, calibration_cfg.get("calibration_file", "calibration/coarse_to_amber.json"))
        if calibration_cfg.get("require_calibrated", False):
            argv.append("--require-calibrated-energy")
        if calibration_file.is_file():
            argv += ["--energy-calibration-file", str(calibration_file)]
        elif calibration_cfg.get("require_calibrated", False):
            return StageResult("qc_benchmark", "failed", started, utc_timestamp(), None,
                f"Required frozen energy calibration is missing: {calibration_file}")
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
        elif summary.get("failures_total", 0) > 0:
            status = "completed_with_failures"
            detail = (f"Closed with per-instance failures: planned={summary['total_cases_planned']} "
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
        dataset_dir = self.dataset_dir()
        checkpoint_dir = self.checkpoint_dir()
        validation_dir = self.run_dir / "validation_queue"
        dev_dir = self.run_dir / "dev_queue"
        qf_cfg = self.config["queue_freeze"]

        def shared_flags() -> List[str]:
            flags = [
                "--outputs", str(cfg.get("outputs", 1000)),
                "--max-evals", str(cfg.get("max_evals", 90)),
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
                "--eval-shots", str(cfg.get("eval_shots", 500)),
                "--seeds", *[str(s) for s in cfg.get("seeds", [42, 43, 44])],
                # (requirement #2) --master-seed lets run_real_complex_pilot.py
                # derive its own independent, saved --optimize-seeds/
                # --sample-seeds per selected target from the master-seed
                # optimize/sample streams -- never passed as flat --seeds values.
                "--master-seed", str(self.config["master_seed"]),
            ]
            if cfg.get("robust_qaoa", True):
                flags += ["--robust-qaoa", "--qaoa-restarts", str(cfg.get("qaoa_restarts", 4)),
                          "--qaoa-objective", cfg.get("qaoa_objective", "cvar"),
                          "--cvar-alpha", str(cfg.get("cvar_alpha", 0.1)),
                          "--parameter-scale", cfg.get("parameter_scale", "max_coefficient")]
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
                self.venv_python, "run_real_complex_pilot.py",
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
                argv += ["--targets", str(len(explicit_targets) or 3)]
                for pdb in explicit_targets:
                    # dev queue is one explicit historical target per invocation;
                    # run_real_complex_pilot.py accepts a single --pdb-id filter,
                    # so each historical target gets its own sub-run/out-dir.
                    pass
                # Historical dev targets are each run individually against
                # their own already-frozen manifest from real_complex_pilot_v3
                # if present, otherwise freshly (re-)selected+frozen here
                # under out_dir/<pdb>/, keeping the dev queue fully separate
                # from the validation queue's own directory.
                for pdb in explicit_targets:
                    sub_argv = argv + ["--pdb-id", pdb, "--out-dir", str(out_dir / pdb)]
                    returncode, log_path = self._run_subprocess(f"structure_experiment_dev_{pdb}", sub_argv)
                    logs.append(str(log_path)); argvs.append(sub_argv)
                    if returncode not in (0,) and not (out_dir / pdb / "real_complex_metrics.csv").is_file():
                        failures.append(f"dev target {pdb} exited {returncode} with no usable metrics (see {log_path})")
                continue
            # (requirement #1/#3) Reproduce EXACTLY the target set queue_freeze
            # already froze -- via an explicit allowlist file, never by
            # trusting that re-running the same eligibility logic with a
            # different --pruning value happens to select the same targets.
            # A target can still fail re-verification here (recorded, not
            # silently dropped), but no target outside the frozen set can
            # ever be added.
            frozen_targets = validation_dir / "selected_targets.json"
            argv += ["--targets", str(queue_cfg.get("target_count", 0))]
            if frozen_targets.is_file():
                argv += ["--pdb-allowlist-file", str(frozen_targets)]
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
            elif summary.get("structure_experiment_failed_targets"):
                queue_partial.append(f"{label}: completed with target failures "
                                      f"{summary['structure_experiment_failed_targets']}")

        status = "failed" if failures else ("completed_with_failures" if queue_partial else "completed")
        detail = "; ".join(failures + queue_partial) if (failures or queue_partial) else \
            "Dev queue and validation queue structural experiments closed with no target failures."
        return StageResult("structure_experiment", status, started, utc_timestamp(), 0 if not failures else 1,
                            detail, argvs, ";".join(logs), not failures)

    # ================================================================
    # Stage 7: paired statistics
    # ================================================================
    def stage_statistics(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config["statistics"]
        # --paired-statistics is specifically shaped for _ablation_main's
        # "<out-dir>/cases/*.json" layout (the coarse-grained qc_benchmark
        # ablation), not run_real_complex_pilot.py's per-target/per-seed CSV
        # layout (dev_queue/validation_queue) -- it errors ("No case JSON
        # artifacts") against the latter. The real-atom structural paired
        # analysis (energy-down vs RMSD-up, within-target-then-across-target
        # QAOA-vs-classical differences) is computed directly by
        # generate_final_research_report.py from real_complex_metrics.csv /
        # recovery_metrics.csv instead, in the "structural benefit" section.
        results_dirs = [self.run_dir / "qc_benchmark"]
        logs, argvs, failures = [], [], []
        for results_dir in results_dirs:
            if not results_dir.is_dir():
                continue
            for budget_mode in cfg.get("budget_modes", ["outputs", "time"]):
                argv = [
                    self.venv_python, "batch_benchmark_hard_set.py", "--paired-statistics",
                    "--results-dir", str(results_dir),
                    "--resamples", str(cfg.get("resamples", 10000)),
                    "--seed", str(self.config["master_seed"]),
                    "--budget-mode", budget_mode,
                    "--max-time-overrun-fraction", str(cfg.get("max_time_overrun_fraction", 0.10)),
                ]
                if cfg.get("cluster_map"):
                    argv += ["--cluster-map", str(resolve_path(self.config, cfg["cluster_map"]))]
                returncode, log_path = self._run_subprocess(
                    f"statistics_{results_dir.name}_{budget_mode}", argv)
                logs.append(str(log_path)); argvs.append(argv)
                if returncode != 0:
                    failures.append(f"{results_dir.name}/{budget_mode} exited {returncode} (see {log_path})")
        status = "completed_with_failures" if failures else "completed"
        detail = "; ".join(failures) if failures else "Paired statistics computed for every available results directory/budget mode."
        return StageResult("statistics", status, started, utc_timestamp(), 0, detail, argvs, ";".join(logs), True)

    # ================================================================
    # Stage 8: final report
    # ================================================================
    def stage_final_report(self) -> StageResult:
        started = utc_timestamp()
        cfg = self.config.get("final_report", {})
        out_path = self.run_dir / cfg.get("filename", "FINAL_RESEARCH_REPORT.md")
        argv = [
            self.venv_python, "generate_final_research_report.py",
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
        results["env_check"] = self.run_stage("env_check", [], self.stage_env_check)
        results["smoke_check"] = self.run_stage("smoke_check", ["env_check"], self.stage_smoke_check)
        if self.smoke_only:
            return results
        results["data_audit"] = self.run_stage("data_audit", ["env_check", "smoke_check"], self.stage_data_audit)
        results["queue_freeze"] = self.run_stage("queue_freeze", ["data_audit"], self.stage_queue_freeze)
        results["egnn_train"] = self.run_stage("egnn_train", ["queue_freeze"], self.stage_egnn_train)
        results["qc_benchmark"] = self.run_stage("qc_benchmark", ["egnn_train"], self.stage_qc_benchmark)
        results["structure_experiment"] = self.run_stage(
            "structure_experiment", ["queue_freeze", "egnn_train"], self.stage_structure_experiment)
        results["statistics"] = self.run_stage(
            "statistics", ["qc_benchmark", "structure_experiment"], self.stage_statistics)
        results["final_report"] = self.run_stage(
            "final_report", ["statistics"], self.stage_final_report)
        return results


def save_derived_child(streams: Dict[str, int], stream_name: str, *labels: str) -> int:
    from seed_streams import derive_child_seed
    return derive_child_seed(streams[stream_name], *labels)


# ---------------------------------------------------------------------------
# Run-directory / manifest / lock management
# ---------------------------------------------------------------------------

def build_run_manifest(config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    return dict(
        generated_utc=utc_timestamp(),
        git_commit=git_commit_hash(repo_root),
        master_seed=config["master_seed"],
        code_sha256={name: sha256_of(repo_root / name) for name in ORCHESTRATED_SCRIPTS
                     if (repo_root / name).is_file()},
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("full_experiment_config.yaml"))
    parser.add_argument("--resume", type=str, default=None,
                         help="Continue a specific previous run directory (name or path) instead of "
                              "starting a fresh one.")
    parser.add_argument("--only", type=str, default=None, choices=STAGE_ORDER,
                         help="Run only this stage (its prerequisites must already be completed).")
    parser.add_argument("--force-restage", type=str, nargs="+", default=[],
                         choices=STAGE_ORDER,
                         help="Re-run these stage(s) even if already marked completed/failed.")
    parser.add_argument("--smoke-only", action="store_true",
                         help="Run only env_check + smoke_check, then stop (for a fast preflight pass).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    config = load_config(args.config)
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
            if previous.get("code_sha256") != current["code_sha256"] or previous.get("config") != current["config"]:
                raise SystemExit(
                    "Refusing to resume: orchestrated code or config differs from the original launch. "
                    "Start a fresh run directory for changed code/config (this repository's established "
                    "rule: a changed source hash or configuration always gets a new output directory)."
                )
            seed_map_path = run_dir / "seed_streams.json"
            if seed_map_path.is_file() and not verify_stream_map(seed_map_path):
                raise SystemExit("Refusing to resume: seed_streams.json no longer matches seed_streams.py's "
                                  "derivation (seed_streams.py itself changed?).")
            print(f"Resuming run: {run_dir}")
        else:
            run_dir = new_run_dir(config)
            manifest = build_run_manifest(config, repo_root)
            atomic_write_json(run_dir / "run_manifest.json", manifest)
            frozen_config_path = run_dir / "frozen_config.yaml"
            shutil.copy2(args.config, frozen_config_path)
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
        print(f"\nRun directory: {run_dir}")
        return 0 if overall_ok else 1
    finally:
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
