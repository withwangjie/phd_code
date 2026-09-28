"""Orchestrator stages ``statistics`` and ``final_report``.
"""
from __future__ import annotations

import concurrent.futures
import json

from nanoqc.common.seed_streams import derive_streams, derive_child_seed
from nanoqc.common.repo_io import module_name
from nanoqc.inference.paired_statistics import serial_gatekeeping, paired_denominator_failures
from nanoqc.pipeline.orchestrator_common import (
    StageResult,
    primary_qc_effect_name,
    quantum_benchmark_ablation,
    quantum_primary,
    rq5_inference_failures,
    utc_timestamp,
)


class ReportingStagesMixin:
    """Paired statistics and the final report, mixed into ``Orchestrator``.

    Relies on the attributes and core helpers ``Orchestrator`` defines
    (``config``, ``run_dir``, ``streams``, ``run_stage`` ...).
    """

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
        paired_jobs=[]
        budget_modes=list(cfg.get("budget_modes", ["outputs", "time"]))
        if len(set(budget_modes))!=len(budget_modes):
            raise ValueError("Paired-statistics budget modes must be unique")
        for results_dir in results_dirs:
            if not results_dir.is_dir():
                failures.append(f"statistics input directory is missing: {results_dir}")
                continue
            if failures:
                continue
            for budget_mode in budget_modes:
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
                paired_jobs.append((results_dir,budget_mode,argv))
        def run_paired_mode(job):
            results_dir,budget_mode,argv=job
            returncode,log_path=self._run_subprocess(
                f"statistics_{results_dir.name}_{budget_mode}",argv)
            return results_dir,budget_mode,argv,returncode,log_path
        paired_workers=max(1,min(len(paired_jobs),int(self.config.get("hardware",{}).get(
            "statistics_mode_workers",1))))
        with concurrent.futures.ThreadPoolExecutor(max_workers=paired_workers) as pool:
            for results_dir,budget_mode,argv,returncode,log_path in pool.map(run_paired_mode,paired_jobs):
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
                         and e.get("metric")==str(cfg.get("primary_qc_metric","log10_qts99"))),
                        None,
                    )
                    observed_clusters=0 if primary_effect is None else int(
                        primary_effect.get("n_clusters",0) or 0)
                    min_qc_clusters=int(cfg.get("min_qc_clusters",10))
                    if observed_clusters < min_qc_clusters:
                        failures.append(
                            f"{results_dir.name}/{budget_mode}: primary coarse contrast "
                            f"{cfg.get('primary_qc_baseline','sa')}/{cfg.get('primary_qc_metric','log10_qts99')} "
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
                "--primary-active-sites",str(primary_active_sites),
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
                # Serial gatekeeping (Dmitrienko & Tamhane 2007; FDA 2022
                # multiple-endpoints guidance; PROTOCOL_AMENDMENTS.md A4, A7):
                # the two quantum-intrinsic confirmatory hypotheses (primary-
                # size QAOA ground-state amplification and its scaling slope)
                # are Holm-adjusted on their own; every QAOA-vs-classical
                # effect is secondary and gated behind them.
                output_stats_path=self.run_dir / "qc_benchmark" / "statistics_outputs.json"
                if output_stats_path.is_file():
                    output_payload=json.loads(output_stats_path.read_text(encoding="utf-8"))
                    scaling_primary=scaling_payload.get("primary",{}) or {}
                    amplification_primary=scaling_payload.get("primary_amplification",{}) or {}
                    if not amplification_primary:
                        failures.append("Scaling statistics lack the primary-size amplification test")
                    primary_family={"amplification:primary":amplification_primary.get("p_value"),
                                    "scaling:primary":scaling_primary.get("p_value")}
                    secondary_family={}
                    targets={"amplification:primary":amplification_primary,"scaling:primary":scaling_primary}
                    for effect in output_payload.get("effects",[]):
                        key="qc:"+str(effect.get("baseline"))+":"+str(effect.get("metric"))
                        targets[key]=effect
                        if str(effect.get("metric","")).startswith("log10_qts99"):
                            effect["gatekeeping_family"]="descriptive_abstract_resource"
                            effect["p_gatekeeping_adjusted"]=None
                        else:
                            secondary_family[key]=effect.get("p_value")
                            effect["gatekeeping_family"]="secondary"
                    scaling_primary["gatekeeping_family"]="primary"
                    amplification_primary["gatekeeping_family"]="primary"
                    adjusted=serial_gatekeeping(primary_family,secondary_family)
                    for key,value in adjusted.items():
                        targets[key]["p_gatekeeping_adjusted"]=value
                    family_note=(
                        "serial gatekeeping: primary family {primary-size QAOA ground-state amplification, "
                        "amplification scaling slope} Holm-adjusted alone; QAOA-vs-classical matched-output "
                        "effects other than abstract shot/query QTS99 are secondary, tested only after both "
                        "primary hypotheses are rejected; QTS99 is descriptive because shots and classical "
                        "energy queries are not equivalent computational costs")
                    output_payload["multiplicity_procedure"]=family_note
                    output_payload["multiplicity_primary_family"]=sorted(primary_family)
                    output_stats_path.write_text(json.dumps(output_payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
                    scaling_payload["multiplicity_procedure"]=family_note
                    scaling_payload["multiplicity_primary_family"]=sorted(primary_family)
                    scaling_payload["primary"]=scaling_primary
                    scaling_payload["primary_amplification"]=amplification_primary
                    scaling_json.write_text(json.dumps(scaling_payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
                    scaling_md.write_text(scaling_md.read_text(encoding="utf-8")+
                        f"\nMultiplicity: {family_note}; gatekeeping-adjusted scaling p="
                        f"{scaling_primary.get('p_gatekeeping_adjusted')}.\n",encoding="utf-8")
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
            logs.append(str(log_path)); argvs.append(structure_argv)
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
