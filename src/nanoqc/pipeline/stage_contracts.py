"""Per-stage result contracts checked before a stage may count as complete.

See docs/RESULTS_CONTRACT.md. The generic results-manifest checks stay in
``Orchestrator._validate_completed_stage_artifacts``, which calls
``StageContractsMixin._validate_stage_contract`` through the class so a
subclass cannot override the acceptance table.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

from nanoqc.common.repo_io import sha256_file as sha256_of
from nanoqc.inference.paired_statistics import paired_denominator_failures
from nanoqc.pipeline.orchestrator_common import (
    primary_qc_effect_name,
    quantum_development_sensitivity,
    rq5_inference_failures,
)


class StageContractsMixin:
    """The per-stage result-contract table, mixed into ``Orchestrator``.

    Relies on the attributes and core helpers ``Orchestrator`` defines
    (``config``, ``run_dir``, ``streams``, ``run_stage`` ...).
    """

    def _validate_stage_contract(
        self, stage: str, require: Callable[[Sequence[Path]], tuple[bool, str]],
        read_json: Callable[[Path], tuple[Optional[dict], Optional[str]]],
    ) -> tuple[bool, str]:
        """Per-stage mandatory result contract (docs/RESULTS_CONTRACT.md).

        Called by ``Orchestrator._validate_completed_stage_artifacts`` after the
        generic results-manifest checks; ``require`` and ``read_json`` are the
        helpers defined there.
        """
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
            observed=0; evaluated=0; excluded=0; atom_pairs=0
            floor=float((self.config.get("data_audit",{}) or {}).get(
                "min_interresidue_heavy_distance_angstrom",1.0))
            with (audit_root/"data_audit_details.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    observed+=1
                    if row.get("valid") and row.get("pairs"):
                        evaluated+=1
                    count=row.get("interresidue_heavy_overlap_count")
                    if type(count) is not int or count<0:
                        return False,"Data audit lacks a valid per-row heavy-atom overlap count"
                    atom_pairs+=count
                    if count:
                        excluded+=1
                        distance=row.get("interresidue_heavy_min_distance_angstrom")
                        pair=row.get("interresidue_heavy_closest_pair")
                        if (not row.get("valid") or not row.get("pairs")
                                or distance is None or not 0<=float(distance)<floor
                                or not isinstance(pair,list) or len(pair)!=2):
                            return False,"Data audit has an unaccounted nonphysical overlap"
                        if (row.get("subset") in ("snac_db","sabdab_vhh")
                                and row.get("structure_quality_status")!="fail"):
                            return False,"Formal nonphysical structure was not quality-excluded"
            if expected<0 or observed!=expected:
                return False,f"Data-audit denominator mismatch: discovered={expected}, audited_rows={observed}"
            protocol=inventory.get("structure_quality_protocol") or {}
            if protocol.get("min_interresidue_heavy_distance_angstrom")!=floor:
                return False,"Data audit did not apply the frozen heavy-atom distance floor"
            overlap_audit=inventory.get("interresidue_heavy_overlap_audit") or {}
            if (overlap_audit.get("scope")!="strongest_allowed_protein_chain_pair_in_first_model"
                    or type(overlap_audit.get("evaluated_rows")) is not int
                    or type(overlap_audit.get("excluded_rows")) is not int
                    or type(overlap_audit.get("overlap_atom_pairs")) is not int):
                return False,"Data audit lacks closed nonphysical-overlap counts"
            if not (0<=overlap_audit["excluded_rows"]<=overlap_audit["evaluated_rows"]<=observed):
                return False,"Data-audit nonphysical-overlap denominators are inconsistent"
            if (overlap_audit["evaluated_rows"]!=evaluated
                    or overlap_audit["excluded_rows"]!=excluded
                    or overlap_audit["overlap_atom_pairs"]!=atom_pairs):
                return False,"Data-audit nonphysical-overlap counts do not match per-row evidence"
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
                self.run_dir/"independence"/"cluster_adequacy.json",
            ])
            if not ok:return ok,detail
            clustering_cfg=((self.config.get("queue_freeze",{}) or {}).get("independence_clustering",{}) or {})
            if clustering_cfg.get("build_per_run",False):
                min_interface=int(((self.config.get("queue_freeze",{}) or {}).get("graph_build",{}) or {})
                                  .get("min_interface_residues",15))
                ok,detail=self._verify_run_local_foldseek_pairs(
                    universe,self.run_dir/"audit"/"data_audit_details.jsonl",min_interface)
                if not ok:return False,detail
                provenance,error=read_json(cluster_provenance)
                if error:return False,error
                if provenance.get("source_pairs_sha256")!=sha256_of(self.run_local_foldseek_pairs()):
                    return False,"Frozen cluster map does not match run-local Foldseek pairs"
            adequacy,error=read_json(self.run_dir/"independence"/"cluster_adequacy.json")
            if error:return False,error
            if not adequacy.get("adequate"):
                return False,"cluster_adequacy.json reports insufficient independent clusters: "+"; ".join(adequacy.get("shortfalls",[]))
            summary,error=read_json(dataset/"run_summary.json")
            if error:return False,error
            if not summary.get("complete"):
                return False,"dataset run_summary.json is not complete"
            holdout_cfg=((self.config.get("queue_freeze",{}) or {}).get("antigen_fold_holdout",{}) or {})
            if holdout_cfg.get("enabled",False):
                holdout_ok,holdout_detail=self._validate_frozen_antigen_holdout()
                if not holdout_ok:
                    return False,"Frozen antigen-fold holdout failed verification: "+holdout_detail
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
            replicates=int((self.config.get("egnn_train",{}) or {}).get("seed_replicates",0) or 0)
            if replicates>0:
                sensitivity=self.checkpoint_dir()/"seed_sensitivity"
                ok,detail=require([sensitivity/"summary.json",sensitivity/"summary.md"])
                if not ok:return ok,detail
                payload,error=read_json(sensitivity/"summary.json")
                if error:return False,error
                if int(payload.get("models",0) or 0)!=replicates+1:
                    return False,f"seed sensitivity covers {payload.get('models')} models; expected {replicates+1}"
                if (payload.get("checkpoints",{}).get("primary",{}) or {}).get("sha256")!=expected_sha:
                    return False,"seed sensitivity was computed against a different primary checkpoint"
            return True,"EGNN checkpoint and training summary verified"
        if stage=="quantum_exploration":
            root=self.run_dir/"quantum_exploration"
            ok,detail=require([root/"summary.json",root/"summary.md"])
            if not ok:return ok,detail
            payload,error=read_json(root/"summary.json")
            if error:return False,error
            depths=[int(v) for v in (self.config.get("quantum_exploration",{}) or {}).get("depths",[1,2,3,4,6])]
            observed=sorted(int(row.get("depth",-1)) for row in payload.get("per_depth",[]))
            if observed!=sorted(depths):
                return False,f"exploration summary covers depths {observed}; expected {sorted(depths)}"
            return True,"Quantum exploration summary verified"
        if stage=="energy_calibration":
            qc=self.config.get("qc_benchmark",{}) or {}
            cal=qc.get("energy_calibration",{}) or {}
            diagnostic=cal.get("mode","frozen")=="diagnostic"
            training=self.run_dir/cal.get("training_csv","calibration/coarse_to_amber_train.csv")
            calibration=self.run_dir/cal.get("calibration_file","calibration/coarse_to_amber.json")
            provenance=training.with_suffix(".provenance.json")
            if diagnostic:
                assessment_path=self.run_dir/"calibration"/"diagnostic_assessment.json"
                assessment,error=read_json(assessment_path)
                if error:return False,error
                if (assessment.get("schema")!="amber_calibration_diagnostic_v1"
                        or assessment.get("applied_to_solver") is not False
                        or not isinstance(assessment.get("accepted"),bool)
                        or not isinstance(assessment.get("failure_reasons"),list)
                        or assessment.get("fit_status") not in ("completed","failed")):
                    return False,"Calibration diagnostic assessment is incomplete"
                if assessment["accepted"] != (assessment["fit_status"]=="completed"
                                               and not assessment["failure_reasons"]):
                    return False,"Calibration diagnostic acceptance does not match recorded failures"
                for key,path in (("training_csv_sha256",training),
                                 ("provenance_sha256",provenance),
                                 ("calibration_sha256",calibration)):
                    observed=sha256_of(path) if path.is_file() else None
                    if assessment.get(key)!=observed:
                        return False,f"Calibration diagnostic artifact is stale: {key}"
                if assessment["fit_status"]=="failed":
                    ok,detail=require([self.run_dir/"calibration"/"calibration_report.md"])
                    if not ok:return ok,detail
                    if assessment["accepted"] or not assessment["failure_reasons"]:
                        return False,"Failed Amber diagnostic lacks explicit failure reasons"
                    return True,"Amber diagnostic failure recorded; no coefficients applied to solvers"
            ok,detail=require([
                training,provenance,calibration,
                self.run_dir/"calibration"/"calibration_report.md",
            ])
            if not ok:return ok,detail
            generation,error=read_json(provenance)
            if error:return False,error
            discovered=int(generation.get("complexes_discovered",-1))
            attempted=int(generation.get("complexes_attempted",-1))
            exclusions=generation.get("input_quality_exclusions")
            failures=generation.get("failures")
            if (not isinstance(exclusions,list) or not isinstance(failures,list)
                    or discovered<=0 or attempted<=0
                    or discovered!=attempted+len(exclusions)
                    or attempted!=int(generation.get("complexes_succeeded",-1))+len(failures)):
                return False,"Calibration input-quality and generation denominators do not close"
            exclusion_fraction=float(generation.get("input_quality_exclusion_fraction",float("nan")))
            failure_fraction=float(generation.get("generation_failure_fraction",float("nan")))
            if (not math.isfinite(exclusion_fraction)
                    or not math.isclose(exclusion_fraction,len(exclusions)/discovered,abs_tol=1e-12)
                    or not math.isfinite(failure_fraction)
                    or not math.isclose(failure_fraction,len(failures)/attempted,abs_tol=1e-12)):
                return False,"Calibration exclusion or failure fraction does not match counts"
            payload,error=read_json(calibration)
            if error:return False,error
            limits=cal.get("acceptance",{}) or {}
            quality_limit=limits.get("max_input_quality_exclusion_fraction")
            if not diagnostic and quality_limit is not None and exclusion_fraction>float(quality_limit):
                return False,"Calibration input-quality exclusion fraction exceeds frozen limit"
            min_eligible=int(limits.get("min_eligible_complexes",0))
            if not diagnostic and attempted < min_eligible:
                return False,f"Calibration has only {attempted} eligible complexes; minimum is {min_eligible}"
            if not diagnostic and failure_fraction>float(limits.get("max_generation_failure_fraction",0.0)):
                return False,"Calibration generation failure fraction exceeds frozen limit"
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
                if not diagnostic and (value is None or not math.isfinite(float(value))):
                    return False,f"Calibration metric missing or nonfinite: {metric}={value}"
                if not diagnostic and not predicate(float(value),limit):
                    return False,f"Calibration resume acceptance failed: {metric}={value}, {key}={limit}"
            if (not diagnostic and limits.get("require_family_grouped_cv",False)
                    and payload.get("cv_grouping")!="family_cluster"):
                return False,"Calibration resume requires family_cluster grouped CV"
            if not diagnostic and int(payload.get("n_train_complexes",0) or 0)<int(limits.get("min_train_complexes",0)):
                return False,"Calibration resume has insufficient training complexes"
            if not diagnostic and int(payload.get("n_train_groups",0) or 0)<int(limits.get("min_train_groups",0)):
                return False,"Calibration resume has insufficient family groups"
            observed_folds=int(payload.get("cv_fold_count",len(payload.get("cv_folds",[]) or [])) or 0)
            if not diagnostic and observed_folds<int(limits.get("min_cv_folds",0)):
                return False,"Calibration resume has insufficient CV folds"
            return True,("Amber diagnostic artifacts verified; coefficients excluded from solver calls"
                         if diagnostic else "energy calibration artifacts and acceptance thresholds verified")
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
            dev_enabled=bool(((self.config.get("queue_freeze",{}) or {}).get("dev_queue",{}) or {}).get("enabled",True))
            dev_root=self.run_dir/"dev_queue"
            val_root=self.run_dir/"validation_queue"
            dev=dev_root/"run_summary.json"
            validation=val_root/"run_summary.json"
            required=[
                val_root/"run_manifest.json",val_root/"seed_streams.json",
                val_root/"eligibility.json",val_root/"selected_targets.json",
                val_root/"run_summary.json",val_root/"real_complex_metrics.csv",
                val_root/"real_complex_report.md",
            ]
            if dev_enabled:
                required += [dev_root/"run_manifest.json",dev_root/"seed_streams.json",
                    dev_root/"eligibility.json",dev_root/"selected_targets.json",
                    dev,dev_root/"real_complex_metrics.csv",dev_root/"real_complex_report.md"]
            ok,detail=require(required)
            if not ok:return ok,detail
            dev_summary={}
            if dev_enabled:
                dev_summary,error=read_json(dev)
                if error:return False,error
            val_summary,error=read_json(validation)
            if error:return False,error
            if dev_enabled and not dev_summary.get("closed"):
                return False,"dev_queue run_summary.json is not closed"
            if not val_summary.get("closed") or val_summary.get("frozen_set_accounting_ok") is not True:
                return False,"validation execution is not closed against frozen denominator"
            if val_summary.get("structure_experiment_failed_targets"):
                return False,"validation execution has failed targets in the confirmatory queue"
            expected_seeds={str(int(v)) for v in self.config.get("structure_experiment",{}).get(
                "seeds",[42,43,44,45,46])}
            expected_methods={"qaoa","sa","uniform","greedy"}
            queues=[(val_root,val_summary,"validation_queue")]
            if dev_enabled:queues.insert(0,(dev_root,dev_summary,"dev_queue"))
            for root,summary,label in queues:
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
                if dev_enabled and str(v)!=str(self.config.get("structure_experiment",{}).get("solvent_model","vacuum"))
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
                             and entry.get("metric")==str(cfg.get("primary_qc_metric","log10_qts99"))),None)
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
