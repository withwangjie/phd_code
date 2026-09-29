"""Orchestrator stages ``egnn_train``, ``energy_calibration`` and ``method_sensitivity``.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import math
from pathlib import Path
from typing import List, Optional

from nanoqc.common.seed_streams import derive_streams, derive_child_seed
from nanoqc.common.repo_io import sha256_file as sha256_of, module_name
from nanoqc.pipeline.orchestrator_common import (
    StageResult,
    atomic_write_json,
    calibration_solver_args,
    quantum_development_sensitivity,
    quantum_primary,
    resolve_path,
    utc_timestamp,
)


class TrainingStagesMixin:
    """EGNN training, Amber calibration and development sensitivity, mixed into ``Orchestrator``.

    Relies on the attributes and core helpers ``Orchestrator`` defines
    (``config``, ``run_dir``, ``streams``, ``run_stage`` ...).
    """

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
        argvs = [argv]; logs = [str(log_path)]
        replicates = int(cfg.get("seed_replicates", 0) or 0)
        if status == "completed" and replicates > 0:
            # Development-only seed sensitivity (Bouthillier et al. 2021;
            # PROTOCOL_AMENDMENTS.md A6): same split, different training
            # seeds. The primary checkpoint above stays the only formal model.
            replicate_checkpoints = []
            for index in range(1, replicates + 1):
                replicate_dir = checkpoint_dir / "seed_replicates" / f"r{index}"
                seed_position = argv.index("--seed") + 1
                replicate_argv = list(argv)
                replicate_argv[seed_position] = str(
                    derive_child_seed(streams["train"], "egnn_seed_replicate", str(index)))
                replicate_argv[replicate_argv.index("--checkpoint-dir") + 1] = str(replicate_dir)
                if "--resume" in replicate_argv:
                    replicate_argv.remove("--resume")
                if (replicate_dir / "last_egnn_pruning.pt").is_file():
                    replicate_argv += ["--resume"]
                rc, replicate_log = self._run_subprocess(f"egnn_train_seed_replicate_{index}", replicate_argv)
                argvs.append(replicate_argv); logs.append(str(replicate_log))
                if rc != 0 or not (replicate_dir / "best_egnn_pruning.pt").is_file():
                    status = "failed"
                    detail += f"; seed replicate {index} failed (see {replicate_log})"
                    break
                replicate_checkpoints.append(replicate_dir / "best_egnn_pruning.pt")
            if status == "completed":
                sensitivity_dir = checkpoint_dir / "seed_sensitivity"
                qc = self.config.get("qc_benchmark", {}) or {}
                analysis_argv = [
                    self.venv_python, "-m", module_name("egnn_seed_sensitivity.py"),
                    "--data-dir", str(train_data_dir),
                    "--primary-checkpoint", str(checkpoint_dir / "best_egnn_pruning.pt"),
                    "--replicate-checkpoints", *[str(p) for p in replicate_checkpoints],
                    "--active-sites", str(self.config.get("statistics", {}).get("primary_active_sites", 6)),
                    "--antigen-guidance-weight", str(qc.get("antigen_guidance_weight", 0.25)),
                    "--antigen-proximity-scale", str(qc.get("antigen_proximity_scale_angstrom", 6.0)),
                    "--contact-ca-cutoff", str(qc.get("contact_ca_cutoff_angstrom", 8.0)),
                    "--vhh-identity-threshold", str(homology.get("vhh_full_chain_identity", 0.80)),
                    "--cdr-h3-identity-threshold", str(homology.get("cdr_h3_identity", 0.50)),
                    "--antigen-identity-threshold", str(homology.get("antigen_identity", 0.30)),
                    "--antigen-min-length-coverage", str(homology.get("antigen_min_length_coverage", 0.70)),
                    "--seed", str(derive_child_seed(streams["train"], "egnn_seed_sensitivity")),
                    "--out-json", str(sensitivity_dir / "summary.json"),
                    "--out-md", str(sensitivity_dir / "summary.md"),
                ]
                rc, analysis_log = self._run_subprocess("egnn_seed_sensitivity", analysis_argv)
                argvs.append(analysis_argv); logs.append(str(analysis_log))
                sens_ok, sens_detail = self._artifacts_present(
                    [sensitivity_dir / "summary.json", sensitivity_dir / "summary.md"])
                if rc != 0 or not sens_ok:
                    status = "failed"
                    detail += f"; seed-sensitivity analysis failed: {sens_detail} (see {analysis_log})"
                else:
                    detail += f"; seed sensitivity: {replicates} replicate(s), {sensitivity_dir / 'summary.md'}"
        return StageResult("egnn_train", status, started, utc_timestamp(), returncode,
                            f"{detail} (train stream seed {streams['train']}, {len(graphs)} training graphs)",
                            argvs if len(argvs) > 1 else argv, ";".join(logs), status == "completed")

    # ================================================================
    # Stage 5: TRAIN-only coarse-to-Amber energy calibration
    # ================================================================
    def stage_energy_calibration(self) -> StageResult:
        started = utc_timestamp()
        qc_cfg = self.config["qc_benchmark"]
        cal_cfg = qc_cfg.get("energy_calibration", {}) or {}
        diagnostic = cal_cfg.get("mode", "frozen") == "diagnostic"
        rot_cfg = qc_cfg.get("rotamer_model", {}) or {}
        ff = qc_cfg.get("coarse_force_field", {}) or {}
        training_csv = self.run_dir / cal_cfg.get("training_csv", "calibration/coarse_to_amber_train.csv")
        calibration_file = self.run_dir / cal_cfg.get("calibration_file", "calibration/coarse_to_amber.json")
        provenance = training_csv.with_suffix(".provenance.json")
        training_csv.parent.mkdir(parents=True, exist_ok=True)
        calibration_file.parent.mkdir(parents=True, exist_ok=True)
        def diagnostic_failure(detail: str, argv: Optional[List[str]] = None,
                               log_path: Optional[Path] = None,
                               returncode: Optional[int] = None) -> StageResult:
            if not diagnostic:
                return StageResult("energy_calibration", "failed", started, utc_timestamp(),
                                   returncode, detail, argv or [],
                                   str(log_path) if log_path else None, False)
            report=self.run_dir/"calibration"/"calibration_report.md"
            report.write_text("# Amber calibration diagnostic\n\nFit status: failed\n"
                              f"Failure: {detail}\nCoefficients applied to solvers: no\n",
                              encoding="utf-8")
            atomic_write_json(self.run_dir/"calibration"/"diagnostic_assessment.json",{
                "schema":"amber_calibration_diagnostic_v1",
                "fit_status":"failed",
                "accepted":False,
                "applied_to_solver":False,
                "failure_reasons":[detail],
                "training_csv_sha256":sha256_of(training_csv) if training_csv.is_file() else None,
                "provenance_sha256":sha256_of(provenance) if provenance.is_file() else None,
                "calibration_sha256":sha256_of(calibration_file) if calibration_file.is_file() else None,
            })
            return StageResult("energy_calibration", "completed_with_failures", started,
                               utc_timestamp(), returncode, detail, argv or [],
                               str(log_path) if log_path else None, True)
        rotamer_library = resolve_path(
            self.config, rot_cfg.get("library_path", "data/rotamer/ALL.bbdep.rotamers.lib")
        )
        if rot_cfg.get("mode","dunbrack2010")=="dunbrack2010" and not rotamer_library.is_file():
            return diagnostic_failure(f"Required Dunbrack library missing: {rotamer_library}")
        if cal_cfg.get("selection_mode","egnn")=="egnn":
            checkpoint=self.checkpoint_dir()/qc_cfg.get("checkpoint","best_egnn_pruning.pt")
            if not checkpoint.is_file():
                return diagnostic_failure(
                    f"EGNN-selected calibration requires trained checkpoint: {checkpoint}")
        argv = [
            self.venv_python, "-m", module_name("generate_energy_calibration_dataset.py"),
            "--dataset", str(self.dataset_dir()),
            "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--rotamer-mode", str(rot_cfg.get("mode","dunbrack2010")),
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
            "--seed", str(derive_child_seed(
                derive_streams(self.config["master_seed"])["partition"], "energy_calibration")),
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
        calibration_devices=[str(device) for device in self.config.get("hardware",{}).get(
            "structural_gpu_devices",[self.config.get("hardware",{}).get("openmm_device","0")])]
        hardware=self.config.get("hardware",{})
        argv += ["--workers",str(hardware.get("calibration_workers",len(calibration_devices))),
                 "--workers-per-gpu",str(hardware.get("calibration_workers_per_gpu",1)),
                 "--gpu-devices",*calibration_devices]
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
            return diagnostic_failure(f"Calibration dataset generation failed; see {log_path}",
                                      argv, log_path, returncode)

        generation=json.loads(provenance.read_text(encoding="utf-8"))
        limits=cal_cfg.get("acceptance", {}) or {}
        generation_failure_fraction=float(generation.get("generation_failure_fraction",1.0))
        max_failure_fraction=float(limits.get("max_generation_failure_fraction",1.0))
        attempted_complexes=int(generation.get("complexes_attempted",0))
        quality_exclusion_fraction=float(generation.get("input_quality_exclusion_fraction",1.0))
        quality_limit=limits.get("max_input_quality_exclusion_fraction")
        failed_checks=[]
        if not math.isfinite(quality_exclusion_fraction):
            return StageResult("energy_calibration","failed",started,utc_timestamp(),None,
                               "Calibration input-quality exclusion fraction is nonfinite",
                               argv,str(log_path),False)
        if quality_limit is not None and quality_exclusion_fraction > float(quality_limit):
            detail=(f"Calibration input-quality exclusion fraction {quality_exclusion_fraction:.4f} "
                    f"exceeds configured limit {float(quality_limit):.4f}")
            if not diagnostic:
                return StageResult("energy_calibration","failed",started,utc_timestamp(),None,
                                   detail,argv,str(log_path),False)
            failed_checks.append(detail)
        min_eligible=int(limits.get("min_eligible_complexes",0))
        if attempted_complexes < min_eligible:
            detail=f"Calibration has only {attempted_complexes} eligible complexes; minimum is {min_eligible}"
            if not diagnostic:
                return StageResult("energy_calibration","failed",started,utc_timestamp(),None,
                                   detail,argv,str(log_path),False)
            failed_checks.append(detail)
        if not math.isfinite(generation_failure_fraction):
            return StageResult("energy_calibration","failed",started,utc_timestamp(),None,
                               "Calibration row generation failure fraction is nonfinite",
                               argv,str(log_path),False)
        if generation_failure_fraction > max_failure_fraction:
            detail=(f"Calibration row generation failure fraction {generation_failure_fraction:.4f} "
                    f"exceeds {max_failure_fraction:.4f}")
            if not diagnostic:
                return StageResult("energy_calibration","failed",started,utc_timestamp(),None,
                                   detail,argv,str(log_path),False)
            failed_checks.append(detail)
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
        fit_artifacts_ok = fit_returncode == 0 and calibration_file.is_file() and calibration_file.stat().st_size > 0
        if diagnostic and not fit_artifacts_ok:
            detail=f"Amber fit failed or its output is missing; see {fit_log}"
            if failed_checks:
                detail="; ".join([*failed_checks,detail])
            return diagnostic_failure(detail,
                                      fit_argv, fit_log, fit_returncode)
        ok = fit_artifacts_ok
        acceptance_detail = ""
        if ok:
            try:
                payload=json.loads(calibration_file.read_text(encoding="utf-8"))
                if not isinstance(payload,dict):
                    raise ValueError("Amber fit JSON must contain an object")
            except (OSError,ValueError) as exc:
                if diagnostic:
                    return diagnostic_failure(
                        f"Amber fit output is unreadable: {type(exc).__name__}: {exc}",
                        fit_argv,fit_log,fit_returncode)
                raise
            limits=cal_cfg.get("acceptance", {}) or {}
            checks=[
                ("cv_rmse_kcal","max_cv_rmse_kcal",lambda value,limit:value<=limit),
                ("cv_mae_kcal","max_cv_mae_kcal",lambda value,limit:value<=limit),
                ("cv_r2","min_cv_r2",lambda value,limit:value>=limit),
                ("cv_spearman","min_cv_spearman",lambda value,limit:value>=limit),
                ("calibration_rmse_improvement_kcal","min_rmse_improvement_kcal",
                    lambda value,limit:value>=limit),
            ]
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
            f"Protocol mode: {'diagnostic (not applied to solvers)' if diagnostic else 'frozen calibration'}",
            f"Status: {'accepted' if ok else 'failed'}",
            f"Training CSV: {training_csv}",
            f"Calibration JSON: {calibration_file}",
            f"Discovered training complexes: {generation.get('complexes_discovered','n/a')}",
            f"Input-quality exclusions: {len(generation.get('input_quality_exclusions',[]))} "
            f"({generation.get('input_quality_exclusion_fraction','n/a')})",
            f"Eligible calibration attempts: {generation.get('complexes_attempted','n/a')}",
            f"Generation failures among eligible attempts: {len(generation.get('failures',[]))} "
            f"({generation.get('generation_failure_fraction','n/a')})",
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
        if diagnostic and fit_artifacts_ok:
            atomic_write_json(self.run_dir/"calibration"/"diagnostic_assessment.json", {
                "schema": "amber_calibration_diagnostic_v1",
                "fit_status": "completed",
                "accepted": bool(ok),
                "applied_to_solver": False,
                "failure_reasons": failed_checks,
                "training_csv_sha256": sha256_of(training_csv),
                "provenance_sha256": sha256_of(provenance),
                "calibration_sha256": sha256_of(calibration_file),
            })
        if diagnostic and fit_artifacts_ok:
            return StageResult(
                "energy_calibration", "completed" if ok else "completed_with_failures",
                started, utc_timestamp(), fit_returncode,
                ("Amber diagnostic passed its historical acceptance checks; coefficients remain unused"
                 if ok else "Amber diagnostic failed acceptance; coefficients remain unused: "
                         + acceptance_detail),
                fit_argv, str(fit_log), True,
            )
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
        sensitivity_workers=max(1,min(int(qc.get("workers",1)),int(
            self.config.get("hardware",{}).get("sensitivity_case_workers",1))))
        per_case_workers=max(1,int(qc.get("workers",1))//sensitivity_workers)
        sensitivity_jobs=[]
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
                    "--outputs",str(cfg.get("outputs",300)),"--qaoa-objective","cvar",
                    "--qaoa-restarts",str(qprimary.get("restarts",4)),
                    "--cvar-alpha",str(alpha),"--eval-shots",str(shots),
                    "--parameter-scale",str(qprimary.get("parameter_scale","max_coefficient")),
                    "--sa-passes",str(qc.get("sa_passes",100)),
                    "--greedy-passes",str(qc.get("greedy_passes",50)),
                    "--energy-window",str(qc.get("energy_window",2.0)),
                    "--max-targets",str(cfg.get("max_targets",20)),
                    "--target-selection-seed",str(target_selection_seed),
                    "--workers",str(per_case_workers),
                    "--omp-threads",str(self.config.get("hardware",{}).get("cpu_threads_per_process",2)),
                    "--seeds",*[str(v) for v in repeats],"--master-seed",str(self.config["master_seed"]),
                ]
                argv += calibration_solver_args(qc.get("energy_calibration", {}) or {},
                                                calibration, force_required=True)
                sensitivity_jobs.append((shots,alpha,sub,argv))
        if len({job[2] for job in sensitivity_jobs})!=len(sensitivity_jobs):
            raise ValueError("Method sensitivity cases must have unique output directories")
        def run_sensitivity_case(job):
            shots,alpha,sub,argv=job
            rc,log=self._run_subprocess(
                f"sensitivity_shots_{shots}_alpha_{str(alpha).replace('.','p')}",argv)
            return shots,alpha,sub,argv,rc,log
        with concurrent.futures.ThreadPoolExecutor(max_workers=sensitivity_workers) as pool:
            sensitivity_results=pool.map(run_sensitivity_case,sensitivity_jobs)
            for shots,alpha,sub,argv,rc,log in sensitivity_results:
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
                resolution_jobs=[]
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
                        resolution_jobs.append((sites,states,sub,command))
                if len({job[2] for job in resolution_jobs})!=len(resolution_jobs):
                    raise ValueError("Rotamer resolution cases must have unique output directories")
                def run_resolution_case(job):
                    sites,states,sub,command=job
                    rc,log=self._run_subprocess(f"rotamer_resolution_{sites}_{states}",command)
                    return sites,states,sub,command,rc,log
                with concurrent.futures.ThreadPoolExecutor(max_workers=sensitivity_workers) as pool:
                    resolution_results=pool.map(run_resolution_case,resolution_jobs)
                    for sites,states,sub,command,rc,log in resolution_results:
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
