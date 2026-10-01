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

Where the orchestrator's code lives (all in ``nanoqc.pipeline``):

    run_full_experiment   this entry point: ``Orchestrator`` core (run_stage,
                          resume/upstream checks, stage status) and ``main``
    orchestrator_common   stage order, prerequisites, code-fingerprint list,
                          ``StageResult``, shared helpers
    config_validation     ``load_config`` and the frozen-protocol validator
    run_records           run manifest, run directories, inventory, results audit
    stages_data           env_check, smoke_check, data_audit, queue_freeze
    stages_training       egnn_train, energy_calibration, method_sensitivity
    stages_quantum        qc_benchmark, quantum_exploration
    stages_structure      structure_experiment, external_validation
    stages_reporting      statistics, final_report
    stage_contracts       per-stage result contracts (docs/RESULTS_CONTRACT.md)

Every name the rest of the repository imports from this module is still
importable from here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import traceback
from nanoqc.common.gpu_runtime import GPUStageMonitor
from nanoqc.common.cpu_runtime import CPUStageMonitor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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

from nanoqc.common.seed_streams import derive_streams, save_stream_map, verify_stream_map  # noqa: E402
# repo_path and sha256_of are re-exported for the scripts/amend_failed_*.py recovery helpers.
from nanoqc.common.repo_io import sha256_file as sha256_of, repo_path, CONFIGS_DIR  # noqa: E402,F401
from nanoqc.pipeline.orchestrator_common import (  # noqa: E402,F401
    MAX_QAOA_DEPTH,
    subprocess_environment,
    ORCHESTRATED_SCRIPTS,
    PREREQUISITE_BLOCK_PREFIX,
    REPO_ROOT,
    REQUIRED_GRAPH_VERSION,
    STAGE_ORDER,
    STAGE_PREREQUISITES,
    StageResult,
    apply_runtime_mode_overrides,
    atomic_write_json,
    calibration_solver_args,
    cluster_adequacy,
    git_commit_hash,
    package_versions,
    primary_qc_effect_name,
    quantum_benchmark_ablation,
    quantum_development_sensitivity,
    quantum_primary,
    quantum_protocol,
    resolve_path,
    rq5_inference_failures,
    utc_run_stamp,
    utc_timestamp,
)
from nanoqc.pipeline.config_validation import (  # noqa: E402,F401
    _validate_scientific_config,
    load_config,
)
from nanoqc.pipeline.run_records import (  # noqa: E402,F401
    audit_experiment_results,
    build_run_manifest,
    new_run_dir,
    resolve_resume_dir,
    save_derived_child,
    write_run_inventory,
)
from nanoqc.pipeline.stages_data import DataStagesMixin  # noqa: E402
from nanoqc.pipeline.stages_training import TrainingStagesMixin  # noqa: E402
from nanoqc.pipeline.stages_quantum import QuantumStagesMixin  # noqa: E402
from nanoqc.pipeline.stages_structure import StructureStagesMixin  # noqa: E402
from nanoqc.pipeline.stages_reporting import ReportingStagesMixin  # noqa: E402
from nanoqc.pipeline.stage_contracts import StageContractsMixin  # noqa: E402


class Orchestrator(
    DataStagesMixin,
    TrainingStagesMixin,
    QuantumStagesMixin,
    StructureStagesMixin,
    ReportingStagesMixin,
    StageContractsMixin,
):
    """Runs the formal stages in order with resume, provenance and result audit.

    Core bookkeeping lives here; each stage group is a mixin in its own module
    (``stages_data``, ``stages_training``, ``stages_quantum``,
    ``stages_structure``, ``stages_reporting``, ``stage_contracts``).
    """
    def __init__(self, config: Dict[str, Any], run_dir: Path, *, only: Optional[str] = None,
                 smoke_only: bool = False, force_restage: Optional[List[str]] = None,
                 stop_after: Optional[str] = None):
        self.config = config
        self.stop_after = stop_after
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

    def external_vhh_dirs(self) -> Tuple[Path, Path, bool]:
        """Graph and raw-structure directories of the external VHH set.

        Empty configuration means this run's own antigen-fold holdout (A10),
        which lives inside the run directory and exists only after
        queue_freeze; the third value says which of the two it is.
        """
        ext=(self.config.get("external_validation",{}) or {}).get("external_vhh",{}) or {}
        if ext.get("graph_dir"):
            return (resolve_path(self.config,ext["graph_dir"]),
                    resolve_path(self.config,ext.get("source_structure_dir","")), False)
        dataset=self.dataset_dir()
        return dataset/"graphs"/"holdout", dataset/"holdout_source_structures", True

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
        hardware = self.config.get("hardware", {})
        full_env = subprocess_environment(hardware, Path(self.repo_root))
        if env:
            full_env.update(env)
        timeout_seconds=float(self.config.get("control",{}).get("stage_timeout_seconds",0) or 0)
        timeout=timeout_seconds if timeout_seconds>0 else None
        with log_path.open("a", encoding="utf-8") as log_handle, GPUStageMonitor(
                log_path.with_suffix(".gpu.csv"),enabled=hardware.get("gpu_monitor_enabled",False),
                interval=hardware.get("gpu_monitor_interval_seconds",10)), CPUStageMonitor(
                log_path.with_suffix(".cpu.csv"),enabled=hardware.get("cpu_monitor_enabled",False),
                interval=hardware.get("cpu_monitor_interval_seconds",10)) as cpu_monitor:
            log_handle.write(f"\n=== {started} :: {' '.join(argv)} ===\n")
            log_handle.flush()
            use_process_group=os.name=="posix"
            process=subprocess.Popen(
                argv,cwd=str(cwd or self.repo_root),env=full_env,
                stdout=log_handle,stderr=subprocess.STDOUT,
                start_new_session=use_process_group,
            )
            cpu_monitor.watch(process.pid)
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
            "quantum_exploration":[self.run_dir/"quantum_exploration"],
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

        # Called through the class, not ``self``: the results contract is a fixed
        # acceptance table that a subclass must not be able to override or weaken.
        return StageContractsMixin._validate_stage_contract(self, stage, require, read_json)

    # ================================================================
    def run_all(self) -> Dict[str, StageResult]:
        results: Dict[str, StageResult] = {}
        for stage in STAGE_ORDER:
            results[stage] = self.run_stage(
                stage, STAGE_PREREQUISITES[stage], getattr(self, f"stage_{stage}"))
            if stage == "smoke_check" and self.smoke_only:
                return results
            if self.stop_after is not None and stage == self.stop_after:
                return results
        return results


def write_incomplete_research_report(
    run_dir: Path, results: Dict[str, StageResult]
) -> Path:
    """Leave a readable, explicitly non-confirmatory report after a failed run.

    The formal final_report stage and its prerequisite gates remain failed.
    Scientific failures are never converted to completed stage statuses.
    """
    path = run_dir / "INCOMPLETE_RESEARCH_REPORT.md"
    header = [
        "# Incomplete research run", "",
        "This run did not satisfy every formal acceptance gate. Its recorded "
        "outputs are diagnostic; no failed structural or statistical endpoint "
        "is presented as confirmatory evidence.", "",
    ]
    try:
        from nanoqc.reporting.generate_final_research_report import compile_report
        body = compile_report(run_dir)
    except Exception as exc:
        # Reporting itself must not hide an earlier scientific failure.  Even
        # if the rich renderer cannot read a partial artifact, preserve every
        # stage status and its recorded failure reason.
        header.extend([
            f"Detailed report rendering failed: {type(exc).__name__}: {exc}", "",
            "## Stage status", "",
        ])
        for stage in STAGE_ORDER:
            result = results.get(stage)
            if result is None:
                header.append(f"- **{stage}: not started**")
            else:
                header.append(f"- **{stage}: {result.status}** — {result.detail}")
        body = ""
    content = "\n".join(header) + "\n" + body + "\n"
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(content, encoding="utf-8")
    temp.replace(path)
    return path


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
    parser.add_argument("--stop-after", type=str, default=None, choices=STAGE_ORDER,
                         help="Stop after this stage (e.g. queue_freeze to check independent-cluster "
                              "adequacy before training). The run is resumable with --resume and is "
                              "not a completed formal run.")
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
                    or previous.get("protocol_amendments_sha256") != current["protocol_amendments_sha256"]
                    or previous.get("config") != current["config"]):
                raise SystemExit(
                    "Refusing to resume: orchestrated code, methods evidence, results contract, protocol amendments, or config differs from the original launch. "
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
                                     force_restage=args.force_restage, stop_after=args.stop_after)
        results = orchestrator.run_all()
        if args.stop_after is not None:
            stopped_ok = all(r.status != "failed" for r in results.values())
            print(f"\nStopped after {args.stop_after} (not a completed formal run; resume with --resume {run_dir.name}).")
            for stage, result in results.items():
                print(f"  {stage:24s} {result.status}")
            adequacy = run_dir/"independence"/"cluster_adequacy.json"
            if adequacy.is_file():
                print(f"Cluster adequacy: {adequacy}")
            return 0 if stopped_ok else 1

        # A failed scientific gate must still leave a readable account of the
        # full run.  This is deliberately outside the final_report stage: it
        # cannot turn failed structural inference into formal completion.
        if not args.only and not args.smoke_only and any(
            result.status == "failed" for result in results.values()
        ):
            report_path = write_incomplete_research_report(run_dir, results)
            print(f"Incomplete-run report: {report_path}")

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
