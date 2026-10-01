"""Orchestrator stages ``structure_experiment`` and ``external_validation``.

All-atom side-chain recovery on the frozen queues, and the frozen pipeline
scored on the antigen-fold holdout (or a certified external VHH set).
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import csv
import json
from pathlib import Path
from typing import List

from nanoqc.common.seed_streams import derive_streams, derive_child_seed, save_stream_map
from nanoqc.common.repo_io import sha256_file as sha256_of, module_name
from nanoqc.inference.paired_statistics import paired_denominator_failures
from nanoqc.pipeline.orchestrator_common import (
    REQUIRED_GRAPH_VERSION,
    StageResult,
    atomic_write_json,
    calibration_solver_args,
    quantum_primary,
    resolve_path,
    utc_timestamp,
)


class StructureStagesMixin:
    """All-atom recovery and external validation, mixed into ``Orchestrator``.

    Relies on the attributes and core helpers ``Orchestrator`` defines
    (``config``, ``run_dir``, ``streams``, ``run_stage`` ...).
    """

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
                "--perturbation-max-attempts", str(cfg.get("perturbation_max_attempts", 32)),
                "--solvent-model", str(cfg.get("solvent_model", "vacuum")),
                "--min-perturb-degrees", str(cfg.get("min_perturb_degrees", 40.0)),
                "--max-perturb-degrees", str(cfg.get("max_perturb_degrees", 120.0)),
                # A44: the same floor deposited inputs pass in data_audit (A24).
                "--min-input-heavy-distance", str((self.config.get("data_audit", {}) or {}).get(
                    "min_interresidue_heavy_distance_angstrom", 1.0)),
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
                "--loop-relax-iterations", str(cfg.get("loop_relax_iterations", 0)),
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

        dev_cfg=qf_cfg["dev_queue"]
        dev_enabled=bool(dev_cfg.get("enabled",True))
        runs=[("validation_queue",validation_dir,qf_cfg["validation_queue"],None)]
        if dev_enabled:
            requested_dev=[str(pdb).lower() for pdb in dev_cfg.get("target_pdb_ids",dev_cfg.get("excluded_pdb",[]))]
            if len(requested_dev)!=len(set(requested_dev)):
                raise ValueError("Development structural target IDs must be unique")
            dev_split=dev_cfg.get("candidate_split","train")
            with (dataset_dir/"graph_manifest.csv").open(encoding="utf-8-sig",newline="") as handle:
                available_dev={row["pdb_id"].lower() for row in csv.DictReader(handle) if row["split"]==dev_split}
            planned_dev=[pdb for pdb in requested_dev if pdb in available_dev]
            unavailable_dev=[pdb for pdb in requested_dev if pdb not in available_dev]
            dev_dir.mkdir(parents=True,exist_ok=True)
            atomic_write_json(dev_dir/"target_availability.json",dict(
                requested_target_ids=requested_dev,available_target_ids=planned_dev,
                unavailable_target_ids=unavailable_dev,candidate_split=dev_split,
                graph_manifest_sha256=sha256_of(dataset_dir/"graph_manifest.csv"),
                historical_validation_exclusions=dev_cfg.get("excluded_pdb",[])))
            if not planned_dev:
                return StageResult("structure_experiment","failed",started,utc_timestamp(),None,
                    f"No development targets present in split {dev_split}; inspect {dev_dir/'target_availability.json'}")
            runs.insert(0,("dev_queue",dev_dir,dev_cfg,planned_dev))
        atomic_write_json(self.run_dir/"structure_execution_plan.json",dict(
            development_targets_enabled=dev_enabled,execution_queues=[run[0] for run in runs],
            development_solvent_sensitivity_enabled=dev_enabled and bool(cfg.get("solvent_sensitivity",[]))))
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
            if label == "validation_queue":
                gpu_devices=[str(device) for device in self.config.get("hardware",{}).get(
                    "structural_gpu_devices",[self.config.get("hardware",{}).get("openmm_device","0")])]
                hardware=self.config.get("hardware",{})
                argv += ["--target-workers",str(hardware.get("structural_target_workers",len(gpu_devices))),
                         "--preparation-workers",str(hardware.get("structural_prepare_workers",1)),
                         "--workers-per-gpu",str(max(hardware.get("structural_workers_per_gpu",1),
                                                     hardware.get("structural_prepare_workers_per_gpu",1))),
                         "--gpu-devices",*gpu_devices]
            if label == "dev_queue":
                argv += ["--candidate-split",str(dev_split)]
                # Historical dev targets are run individually. Each subprocess
                # therefore has an exact denominator of one target.
                # Historical dev targets are each run individually against
                # their own already-frozen manifest from real_complex_pilot_v3
                # if present, otherwise freshly (re-)selected+frozen here
                # under out_dir/<pdb>/, keeping the dev queue fully separate
                # from the validation queue's own directory.
                dev_completed=[]; dev_failed=[]; dev_summaries={}
                if len({str(pdb).lower() for pdb in explicit_targets})!=len(explicit_targets):
                    raise ValueError("Development structural target IDs must be unique")
                gpu_devices=[str(device) for device in self.config.get("hardware",{}).get(
                    "structural_gpu_devices",[self.config.get("hardware",{}).get("openmm_device","0")])]
                def run_dev(task):
                    index,pdb=task
                    sub_argv = argv + [
                        "--targets", "1",
                        "--pdb-id", pdb,
                        "--out-dir", str(out_dir / pdb),
                        "--gpu-devices",gpu_devices[index % len(gpu_devices)],
                    ]
                    returncode,log_path=self._run_subprocess(
                        f"structure_experiment_dev_{pdb}",sub_argv,
                        env={"QP_OPENMM_DEVICE":gpu_devices[index % len(gpu_devices)],
                             "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
                    return pdb,sub_argv,returncode,log_path
                with contextlib.ExitStack() as stack:
                    dev_workers=max(1,min(len(explicit_targets),int(self.config.get("hardware",{}).get(
                        "structural_target_workers",len(gpu_devices)))))
                    pools=[stack.enter_context(concurrent.futures.ThreadPoolExecutor(max_workers=1))
                           for _ in range(dev_workers)]
                    futures=[pools[index % len(pools)].submit(run_dev,(index,pdb))
                             for index,pdb in enumerate(explicit_targets)]
                    dev_runs=[future.result() for future in futures]
                for pdb,sub_argv,returncode,log_path in dev_runs:
                    logs.append(str(log_path)); argvs.append(sub_argv)
                    summary_path = out_dir / pdb / "run_summary.json"
                    if returncode != 0:
                        # Never accept stale artifacts from a previous protocol/run.
                        dev_failed.append(pdb.lower())
                        failures.append(
                            f"dev target {pdb} exited {returncode}; refusing any pre-existing "
                            f"summary/metrics in {out_dir / pdb} (see {log_path})"
                        )
                        continue
                    if summary_path.is_file():
                        summary=json.loads(summary_path.read_text(encoding="utf-8"))
                        dev_summaries[pdb]=summary
                        if summary.get("closed") and not summary.get("structure_experiment_failed_targets"):
                            dev_completed.append(pdb.lower())
                        else:
                            dev_failed.append(pdb.lower())
                    else:
                        dev_failed.append(pdb.lower())
                        failures.append(f"dev target {pdb} returned 0 but produced no run_summary.json")
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
                    "requested_target_ids":requested_dev,
                    "unavailable_target_ids":unavailable_dev,
                    "candidate_split":dev_split,
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
                    f"Requested targets unavailable in input split: {unavailable_dev}",
                    f"Candidate split: {dev_split}; training recovery is developmental, not independent validation.",
                ]
                (out_dir/"real_complex_report.md").write_text("\n".join(report)+"\n",encoding="utf-8")
                atomic_write_json(out_dir/"run_summary.json",dict(
                    queue_role="dev",
                    planned_target_ids=sorted(planned),
                    requested_target_ids=requested_dev,
                    unavailable_target_ids=unavailable_dev,
                    candidate_split=dev_split,
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
            # Non-zero subprocess status is authoritative. A previous run may
            # have left closed summaries/metrics in this directory; those must
            # never mask a protocol mismatch or current execution failure.
            if returncode != 0:
                failures.append(
                    f"{label}: subprocess exited {returncode}; refusing any pre-existing "
                    f"run_summary.json/metrics in {out_dir} (see {log_path})"
                )
                continue
            # (requirement #5) Reconciled against run_real_complex_pilot.py's
            # own run_summary.json (selected vs. completed vs. failed target
            # counts) after successful subprocess exit.
            summary_path = out_dir / "run_summary.json"
            if not summary_path.is_file():
                failures.append(f"{label}: exit 0 but no run_summary.json produced (see {log_path})")
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
        sensitivity_models=[str(v) for v in cfg.get("solvent_sensitivity", [])] if dev_enabled else []
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
                    "--perturbation-max-attempts",str(cfg.get("perturbation_max_attempts",32)),
                    "--min-perturb-degrees",str(cfg.get("min_perturb_degrees",40.0)),
                    "--max-perturb-degrees",str(cfg.get("max_perturb_degrees",120.0)),
                    "--min-input-heavy-distance",str((self.config.get("data_audit",{}) or {}).get(
                        "min_interresidue_heavy_distance_angstrom",1.0)),
                    "--outputs",str(qprimary.get("output_shots",1000)),
                    "--max-evals",str(qprimary.get("max_evals",90)),
                    "--sa-passes",str(cfg.get("sa_passes",100)),
                    "--relax-iterations",str(cfg.get("relax_iterations",200)),
                    "--loop-relax-iterations",str(cfg.get("loop_relax_iterations",0)),
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
            ("Dev queue and validation queue structural experiments closed with no target failures." if dev_enabled
             else "Frozen validation structural experiment closed; development target experiments disabled.")
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
            graph_dir,source_dir,from_run=self.external_vhh_dirs()
            run_external_root=self.run_dir/"external_validation"
            run_external_root.mkdir(parents=True,exist_ok=True)
            # Always regenerated for this run (never reused from another run or
            # from an earlier failed attempt), because it certifies this run's
            # frozen training graphs and cluster map.
            independence=run_external_root/"external_vhh_independence_manifest.json"
            if independence.exists():
                independence.unlink()
            ext_failures=[]
            label="antigen-fold holdout" if from_run else "external VHH"
            if not graph_dir.is_dir() or not any(graph_dir.glob("*.pt")):
                ext_failures.append(f"Required {label} graph set missing/empty: {graph_dir}")
            if not source_dir.is_dir():
                ext_failures.append(f"Required {label} raw structures missing: {source_dir}")
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
                        "--workers",str(self.config.get("hardware",{}).get("external_audit_workers",1)),
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
                if manifest.get("graph_version")!=REQUIRED_GRAPH_VERSION:
                    ext_failures.append(
                        f"External VHH graph version must be {REQUIRED_GRAPH_VERSION}, got {manifest.get('graph_version')}"
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
                            if float(audit["max_vhh_full_chain_identity"]) >= expected_homology["vhh_full_chain_identity"]:
                                ext_failures.append(f"{pdb}: VHH full-chain identity overlap")
                            if float(audit["max_cdr_h3_loop_identity"]) >= expected_homology["cdr_h3_identity"]:
                                ext_failures.append(f"{pdb}: CDR-H3 loop identity overlap")
                            if (
                                float(audit["max_antigen_full_chain_identity"]) >= expected_homology["antigen_identity"]
                                and float(audit.get("antigen_length_coverage",1.0))
                                    >= expected_homology["antigen_min_length_coverage"]
                            ):
                                ext_failures.append(f"{pdb}: antigen full-chain identity overlap")
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
                argv += calibration_solver_args(qc.get("energy_calibration", {}) or {},
                                                calibration, force_required=True)
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
                                "--seed",str(derive_child_seed(
                                    derive_streams(self.config["master_seed"])["inference"],
                                    "external_vhh_statistics")),
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
                                primary_metric=str(self.config.get("statistics",{}).get("primary_qc_metric","log10_qts99"))
                                sa_primary=next(
                                    (e for e in stats_payload.get("effects",[])
                                     if e.get("baseline")=="sa" and e.get("metric")==primary_metric),
                                    None,
                                )
                                observed_clusters=0 if sa_primary is None else int(sa_primary.get("n_clusters",0) or 0)
                                required_clusters=int(ext.get("min_clusters",10))
                                if observed_clusters < required_clusters:
                                    failures.append(
                                        f"External VHH QAOA-vs-SA {primary_metric} contrast has {observed_clusters} "
                                        f"independent clusters; requires >= {required_clusters}"
                                    )

        structural=cfg.get("structural_baselines", {}) or {}
        if structural.get("required", False):
            faspr=Path(structural.get("faspr_executable",""))
            phenix=Path(structural.get("phenix_clashscore_executable",""))
            if not faspr.is_file():
                failures.append(f"Required FASPR executable missing: {faspr}")
            if not (faspr.parent/"dun2010bbdep.bin").is_file():
                failures.append(f"Required FASPR rotamer library missing: {faspr.parent/'dun2010bbdep.bin'}")
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
