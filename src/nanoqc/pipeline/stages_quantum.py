"""Orchestrator stages ``qc_benchmark`` and ``quantum_exploration``.

Both run the coarse QUBO solver benchmark: the matched QAOA-vs-classical
ablation, and the exploratory depth/budget and parameter-transfer study.
"""
from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path
from typing import List, Optional, Sequence

from nanoqc.common.seed_streams import derive_streams, derive_child_seed
from nanoqc.common.repo_io import module_name
from nanoqc.pipeline.orchestrator_common import (
    StageResult,
    calibration_solver_args,
    quantum_benchmark_ablation,
    quantum_primary,
    resolve_path,
    utc_timestamp,
)


class QuantumStagesMixin:
    """Coarse QUBO solver benchmark stages, mixed into ``Orchestrator``.

    Relies on the attributes and core helpers ``Orchestrator`` defines
    (``config``, ``run_dir``, ``streams``, ``run_stage`` ...).
    """

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
        if (calibration_cfg.get("mode", "frozen") == "frozen"
                and calibration_cfg.get("require_calibrated", False)
                and not calibration_file.is_file()):
            return StageResult(
                "qc_benchmark", "failed", started, utc_timestamp(), None,
                f"Required run-specific frozen calibration missing: {calibration_file}",
            )
        argv += calibration_solver_args(calibration_cfg, calibration_file)
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
    # Stage 5b: pre-declared exploratory QAOA analyses (A7)
    # ================================================================
    def _coarse_benchmark_argv(self, *, input_dir: Path, out_dir: Path, seeds: Sequence[int],
                               depth: int, max_evals: int, active_sites: Sequence[int],
                               outputs: Sequence[int], max_targets: int,
                               target_selection_seed: Optional[int] = None,
                               worker_override: Optional[int] = None) -> List[str]:
        """Benchmark argv with the frozen coarse model and primary QAOA settings."""
        cfg = self.config["qc_benchmark"]
        qprimary = quantum_primary(self.config)
        homology = self.config["queue_freeze"]["homology_isolation"]
        ff = cfg.get("coarse_force_field", {}) or {}
        rot = cfg.get("rotamer_model", {}) or {}
        argv = [
            self.venv_python, "-m", module_name("batch_benchmark_hard_set.py"), "--research-ablation",
            "--input-dir", str(input_dir), "--checkpoint", str(self.checkpoint_dir() / cfg["checkpoint"]),
            "--out-dir", str(out_dir), "--seeds", *[str(v) for v in seeds],
            "--master-seed", str(self.config["master_seed"]), "--pruning", "egnn",
            "--radii", str(self.config.get("statistics", {}).get("primary_radius", cfg.get("radii", [6.0])[0])),
            "--depths", str(depth), "--max-evals", str(max_evals),
            "--active-sites", *[str(v) for v in active_sites], "--states-per-site", "3",
            "--vhh-identity-threshold", str(homology.get("vhh_full_chain_identity", 0.80)),
            "--cdr-h3-identity-threshold", str(homology.get("cdr_h3_identity", 0.50)),
            "--antigen-identity-threshold", str(homology.get("antigen_identity", 0.30)),
            "--antigen-min-length-coverage", str(homology.get("antigen_min_length_coverage", 0.70)),
            "--antigen-guidance-weight", str(cfg.get("antigen_guidance_weight", 0.25)),
            "--antigen-proximity-scale", str(cfg.get("antigen_proximity_scale_angstrom", 6.0)),
            "--contact-ca-cutoff", str(cfg.get("contact_ca_cutoff_angstrom", 8.0)),
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
            "--rotamer-mode", str(rot.get("mode", "dunbrack2010")),
            "--rotamer-library", str(resolve_path(self.config, rot.get("library_path", "data/rotamer/ALL.bbdep.rotamers.lib"))),
            "--rotamer-probability-floor", str(rot.get("probability_floor", 1e-4)),
            "--rotamer-sigma-offsets", *[str(v) for v in rot.get("sigma_offsets", [-1.0, 0.0, 1.0])],
            "--outputs", *[str(v) for v in outputs],
            "--qaoa-objective", str(qprimary.get("objective", "cvar")),
            "--qaoa-restarts", str(qprimary.get("restarts", 4)),
            "--cvar-alpha", str(qprimary.get("cvar_alpha", 0.1)),
            "--eval-shots", str(qprimary.get("eval_shots", 500)),
            "--parameter-scale", str(qprimary.get("parameter_scale", "max_coefficient")),
            "--sa-passes", str(cfg.get("sa_passes", 100)),
            "--greedy-passes", str(cfg.get("greedy_passes", 50)),
            "--energy-window", str(cfg.get("energy_window", 2.0)),
            "--max-targets", str(max_targets),
            "--workers", str(worker_override if worker_override is not None else cfg.get("workers", 1)),
            "--omp-threads", str(self.config.get("hardware", {}).get("cpu_threads_per_process", 2)),
        ]
        if target_selection_seed is not None:
            argv += ["--target-selection-seed", str(target_selection_seed)]
        calibration_cfg = cfg.get("energy_calibration", {}) or {}
        calibration_file = self.run_dir / calibration_cfg.get("calibration_file", "calibration/coarse_to_amber.json")
        argv += calibration_solver_args(calibration_cfg, calibration_file)
        return argv

    def _closed_benchmark(self, out_dir: Path) -> tuple[bool, str]:
        path = out_dir / "run_summary.json"
        if not path.is_file():
            return False, f"no run_summary.json in {out_dir}"
        summary = json.loads(path.read_text(encoding="utf-8"))
        limit = float(self.config["qc_benchmark"].get("max_failure_fraction", 0.0))
        fraction = summary.get("failures_total", 0) / max(1, summary.get("total_cases_planned", 1))
        if not summary.get("closed") or summary.get("cases_completed_total", 0) == 0 or fraction > limit:
            return False, (f"{out_dir}: closed={summary.get('closed')} completed={summary.get('cases_completed_total')} "
                           f"failure fraction={fraction:.4f} (limit {limit})")
        return True, f"{out_dir}: closed"

    def stage_quantum_exploration(self) -> StageResult:
        """Depth x optimizer budget, and parameter transfer (exploratory; A7).

        For each pre-declared depth p the optimizer budget is
        evals_per_parameter x 2p. Transfer angles are fitted on TRAINING
        graphs only and then applied, untrained, to the hard set alongside
        per-instance-trained QAOA. Descriptive output only; no hypothesis
        tests and no influence on the confirmatory analyses.
        """
        started = utc_timestamp()
        cfg = self.config.get("quantum_exploration", {}) or {}
        qprimary = quantum_primary(self.config)
        streams = derive_streams(self.config["master_seed"])
        root = self.run_dir / "quantum_exploration"
        depths = [int(v) for v in cfg.get("depths", [1, 2, 3, 4, 6])]
        per_parameter = int(cfg.get("evals_per_parameter", 20))
        sites = [int(cfg.get("active_sites", self.config.get("statistics", {}).get("primary_active_sites", 6)))]
        output_shots = int(qprimary.get("output_shots", 1000))
        transfer_cfg = cfg.get("transfer", {}) or {}
        repeats = [derive_child_seed(streams["perturb"], "exploration_repeat", str(i))
                   for i in range(int(cfg.get("repeats", 3)))]
        fit_seed = [derive_child_seed(streams["perturb"], "transfer_fit_repeat", "0")]
        argvs, logs, failures, hard_dirs = [], [], [], []
        if len(set(depths))!=len(depths) or not depths:
            raise ValueError("Exploration depths must be nonempty and unique")
        depth_workers=min(len(depths),int(self.config.get("hardware",{}).get(
            "exploration_depth_workers",1)))
        if depth_workers<1:
            raise ValueError("exploration_depth_workers must be positive")
        per_depth_workers=max(1,int(self.config["qc_benchmark"].get("workers",1))//depth_workers)
        def run_depth(depth: int) -> tuple[list,list,list,Optional[Path]]:
            depth_argvs=[];depth_logs=[];depth_failures=[]
            max_evals = per_parameter * 2 * depth
            fit_dir = root / "transfer_fit" / f"p{depth}"
            fit_argv = self._coarse_benchmark_argv(
                input_dir=self.dataset_dir() / "graphs" / "train", out_dir=fit_dir, seeds=fit_seed,
                depth=depth, max_evals=max_evals, active_sites=sites, outputs=[output_shots],
                max_targets=int(transfer_cfg.get("train_max_targets", 40)),
                target_selection_seed=derive_child_seed(streams["partition"], "transfer_fit_targets"),
                worker_override=per_depth_workers)
            rc, log = self._run_subprocess(f"quantum_exploration_fit_p{depth}", fit_argv)
            depth_argvs.append(fit_argv); depth_logs.append(str(log))
            ok, detail = self._closed_benchmark(fit_dir)
            if rc not in (0, 1) or not ok:
                depth_failures.append(f"transfer fit p={depth}: exit={rc}; {detail}")
                return depth_argvs,depth_logs,depth_failures,None
            transfer_file = root / "transfer_parameters" / f"p{depth}.json"
            fit_params_argv = [
                self.venv_python, "-m", module_name("fit_qaoa_transfer_parameters.py"),
                "--results-dir", str(fit_dir), "--objective", str(qprimary.get("objective", "cvar")),
                "--restarts", str(qprimary.get("restarts", 4)),
                "--min-instances", str(int(transfer_cfg.get("min_instances", 10))),
                "--out", str(transfer_file)]
            rc, log = self._run_subprocess(f"quantum_exploration_transfer_p{depth}", fit_params_argv)
            depth_argvs.append(fit_params_argv); depth_logs.append(str(log))
            if rc != 0 or not transfer_file.is_file():
                depth_failures.append(f"transfer parameter fit p={depth} failed (see {log})")
                return depth_argvs,depth_logs,depth_failures,None
            hard_dir = root / "hard_set" / f"p{depth}"
            hard_argv = self._coarse_benchmark_argv(
                input_dir=self.dataset_dir() / self.config["qc_benchmark"]["input_dir"], out_dir=hard_dir,
                seeds=repeats, depth=depth, max_evals=max_evals, active_sites=sites, outputs=[output_shots],
                max_targets=int(self.config["qc_benchmark"].get("max_targets", 0)),
                worker_override=per_depth_workers) + [
                "--transfer-parameters", str(transfer_file)]
            rc, log = self._run_subprocess(f"quantum_exploration_hard_p{depth}", hard_argv)
            depth_argvs.append(hard_argv); depth_logs.append(str(log))
            ok, detail = self._closed_benchmark(hard_dir)
            if rc not in (0, 1) or not ok:
                depth_failures.append(f"hard-set exploration p={depth}: exit={rc}; {detail}")
                return depth_argvs,depth_logs,depth_failures,None
            return depth_argvs,depth_logs,depth_failures,hard_dir
        with concurrent.futures.ThreadPoolExecutor(max_workers=depth_workers) as pool:
            # map preserves the frozen depth order while subprocesses run concurrently.
            for depth_argvs,depth_logs,depth_failures,hard_dir in pool.map(run_depth,depths):
                argvs.extend(depth_argvs);logs.extend(depth_logs);failures.extend(depth_failures)
                if hard_dir is not None:
                    hard_dirs.append(hard_dir)
        if not failures:
            analysis_argv = [
                self.venv_python, "-m", module_name("analyze_quantum_exploration.py"),
                "--depth-dirs", *[str(d) for d in hard_dirs],
                "--cluster-map", str(self.frozen_cluster_map_path()),
                "--primary-outputs", str(output_shots),
                "--objective", str(qprimary.get("objective", "cvar")),
                "--restarts", str(qprimary.get("restarts", 4)),
                "--active-sites", str(sites[0]),
                "--resamples", str(self.config.get("statistics", {}).get("resamples", 10000)),
                "--seed", str(derive_child_seed(streams["inference"], "quantum_exploration")),
                "--out-json", str(root / "summary.json"), "--out-md", str(root / "summary.md")]
            rc, log = self._run_subprocess("quantum_exploration_analysis", analysis_argv)
            argvs.append(analysis_argv); logs.append(str(log))
            ok, detail = self._artifacts_present([root / "summary.json", root / "summary.md"])
            if rc != 0 or not ok:
                failures.append(f"exploration analysis failed: {detail} (see {log})")
        status = "failed" if failures else "completed"
        detail = "; ".join(failures) if failures else (
            f"Exploratory depth/budget and parameter-transfer analyses for p={depths} "
            f"({per_parameter} evaluations per parameter): {root / 'summary.md'}")
        return StageResult("quantum_exploration", status, started, utc_timestamp(),
                           1 if failures else 0, detail, argvs, ";".join(logs), not failures)
