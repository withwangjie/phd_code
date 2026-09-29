"""Orchestrator stages ``env_check``, ``smoke_check``, ``data_audit`` and ``queue_freeze``.

``queue_freeze`` is the preregistration point: it freezes the clustering
universe, the structure cluster map, the graph dataset, the antigen-fold
holdout and the validation queue before any training. This module also owns
the run-local Foldseek pair table and the frozen-holdout verification it uses.
"""
from __future__ import annotations

import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any, List

from nanoqc.common.seed_streams import derive_streams
from nanoqc.common.repo_io import sha256_file as sha256_of, repo_path, module_name
from nanoqc.pipeline.orchestrator_common import (
    ORCHESTRATED_SCRIPTS,
    StageResult,
    atomic_write_json,
    cluster_adequacy,
    git_commit_hash,
    package_versions,
    resolve_path,
    utc_timestamp,
)
from nanoqc.pipeline.run_records import save_derived_child


class DataStagesMixin:
    """Data audit and queue freeze, mixed into ``Orchestrator``.

    Relies on the attributes and core helpers ``Orchestrator`` defines
    (``config``, ``run_dir``, ``streams``, ``run_stage`` ...).
    """

    def run_local_foldseek_pairs(self) -> Path:
        return self.run_dir / "independence" / "foldseek_pairs.tsv"

    def foldseek_executable(self) -> str|None:
        configured=(self.config.get("runtime_resolution",{}) or {}).get("foldseek_executable")
        return shutil.which(str(configured or os.environ.get("QP_FOLDSEEK") or "foldseek"))

    def _verify_run_local_foldseek_pairs(self, universe: Path, audit_jsonl: Path,
                                         min_interface_residues: int) -> tuple[bool, str]:
        pairs=self.run_local_foldseek_pairs()
        manifest_path=pairs.with_suffix(".manifest.json")
        if not pairs.is_file() or not manifest_path.is_file():
            return False,"Run-local Foldseek table or manifest is missing"
        try:
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            return False,f"Run-local Foldseek manifest is unreadable: {exc}"
        expected={
            "schema":"foldseek_pairs_v1",
            "score":"mintmscore",
            "pair_table_sha256":sha256_of(pairs),
            "universe_sha256":sha256_of(universe),
            "audit_ledger_sha256":sha256_of(audit_jsonl),
            "min_interface_residues":min_interface_residues,
        }
        for key,value in expected.items():
            if manifest.get(key)!=value:
                return False,f"Run-local Foldseek {key} does not match this run's audited input"
        return True,"run-local Foldseek table and manifest verified"

    def _ensure_run_local_foldseek_pairs(self, universe: Path, audit_jsonl: Path,
                                         min_interface_residues: int) -> tuple[bool, str]:
        """Build once per run; verify the frozen table on every stage retry."""
        pairs=self.run_local_foldseek_pairs()
        manifest_path=pairs.with_suffix(".manifest.json")

        if pairs.is_file() and manifest_path.is_file():
            return self._verify_run_local_foldseek_pairs(universe,audit_jsonl,min_interface_residues)

        foldseek=self.foldseek_executable()
        if not foldseek:
            return False,"Foldseek executable missing; set QP_FOLDSEEK or add foldseek to PATH"
        pairs.parent.mkdir(parents=True,exist_ok=True)
        argv=[
            self.venv_python,"-m",module_name("build_foldseek_pairs.py"),
            "--universe",str(universe),"--audit-dir",str(audit_jsonl.parent),
            "--data-root",str(resolve_path(self.config,self.config["paths"]["data_root"])),
            "--out",str(pairs),"--work-dir",str(pairs.parent/"foldseek_work"),
            "--foldseek",foldseek,
            "--threads",str(max(1,int((self.config.get("data_audit",{}) or {}).get("workers",8)))),
            "--prepare-workers",str(max(1,int((self.config.get("hardware",{}) or {}).get("foldseek_prepare_workers",1)))),
            "--min-interface-residues",str(min_interface_residues),
        ]
        if pairs.exists():
            # An interrupted build can leave the TSV before writing its manifest.
            # Both files are run-local and no completed stage has consumed them.
            argv.append("--force")
        rc,log=self._run_subprocess("queue_freeze_foldseek",argv)
        if rc!=0:
            return False,f"Run-local Foldseek build exited {rc}; see {log}"
        return self._verify_run_local_foldseek_pairs(universe,audit_jsonl,min_interface_residues)

    def _validate_frozen_antigen_holdout(self) -> tuple[bool, str]:
        """Verify an already-carved primary-SNAC holdout before queue recovery."""
        holdout_json=self.run_dir/"independence"/"antigen_fold_holdout.json"
        if not holdout_json.is_file():
            return False,"holdout manifest is absent"
        try:
            payload=json.loads(holdout_json.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError) as exc:
            return False,f"unreadable holdout manifest: {type(exc).__name__}: {exc}"
        if not isinstance(payload,dict) or payload.get("schema")!="antigen_fold_holdout_v2":
            return False,"invalid antigen-fold holdout schema"
        if payload.get("primary_source")!="snac_db" or payload.get("auxiliary_source")!="sabdab_vhh":
            return False,"holdout source-role contract is invalid"

        dataset=self.dataset_dir().resolve()
        expected_graph_dir=(dataset/"graphs"/"holdout").resolve()
        expected_quarantine_dir=(dataset/"graphs"/"holdout_quarantine").resolve()
        expected_source_dir=(dataset/"holdout_source_structures").resolve()
        try:
            graph_dir=Path(str(payload["graph_dir"])).resolve()
            quarantine_dir=Path(str(payload["quarantine_dir"])).resolve()
            source_dir=Path(str(payload["source_structure_dir"])).resolve()
        except (KeyError,TypeError,ValueError) as exc:
            return False,f"holdout manifest lacks resolved directories: {exc}"
        if (graph_dir!=expected_graph_dir or quarantine_dir!=expected_quarantine_dir
                or source_dir!=expected_source_dir):
            return False,"holdout manifest directories do not match this run's dataset"
        if not graph_dir.is_dir() or not source_dir.is_dir():
            return False,"holdout graph/source directory is missing"

        manifest_path=dataset/"graph_manifest.json"
        audit_jsonl=self.run_dir/"audit"/"data_audit_details.jsonl"
        if not manifest_path.is_file() or not audit_jsonl.is_file():
            return False,"dataset graph manifest or audit ledger is missing"
        try:
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            audit_rows=[json.loads(line) for line in audit_jsonl.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
        except (OSError,json.JSONDecodeError) as exc:
            return False,f"unreadable holdout provenance input: {type(exc).__name__}: {exc}"
        if not isinstance(manifest,list):
            return False,"graph_manifest.json must be a list"
        by_path={str(row.get("path","")).replace("\\","/"):row
                 for row in manifest if isinstance(row,dict)}
        audit_by_id={}
        for row in audit_rows:
            source_id=str((row or {}).get("id",""))
            if not source_id:
                continue
            if source_id in audit_by_id:
                return False,f"duplicate audit source_id in frozen ledger: {source_id}"
            audit_by_id[source_id]=row

        targets=payload.get("targets")
        quarantined=payload.get("quarantined",[])
        if not isinstance(targets,list) or not targets:
            return False,"holdout manifest has no targets"
        if not isinstance(quarantined,list):
            return False,"holdout quarantine manifest is malformed"

        target_paths=set()
        target_pdbs=set()
        target_by_pdb={}
        for row in targets:
            if not isinstance(row,dict):
                return False,"malformed holdout target record"
            rel=str(row.get("path","")).replace("\\","/")
            expected_sha=str(row.get("sha256",""))
            pdb=str(row.get("pdb_id","")).lower()
            source_id=str(row.get("source_id",""))
            raw_sha=str(row.get("source_structure_sha256",""))
            if (not rel.startswith("graphs/holdout/") or not expected_sha
                    or row.get("subset_source")!="snac_db" or not source_id
                    or len(raw_sha)!=64 or len(pdb)!=4):
                return False,f"invalid primary holdout target binding: {row!r}"
            if pdb in target_pdbs:
                return False,f"primary holdout has duplicate PDB target: {pdb}"
            path=(dataset/rel).resolve()
            if not path.is_relative_to(dataset) or not path.is_file():
                return False,f"holdout graph missing/outside dataset: {rel}"
            manifest_row=by_path.get(rel)
            if (not manifest_row or manifest_row.get("split")!="holdout"
                    or manifest_row.get("subset_source")!="snac_db"
                    or str(manifest_row.get("source_id",""))!=source_id
                    or str(manifest_row.get("source_structure_sha256",""))!=raw_sha):
                return False,f"graph manifest does not bind primary holdout graph: {rel}"
            actual_sha=sha256_of(path)
            if actual_sha!=expected_sha or str(manifest_row.get("sha256",""))!=actual_sha:
                return False,f"holdout graph hash mismatch: {rel}"
            audit_row=audit_by_id.get(source_id)
            if (not audit_row
                    or str(audit_row.get("pdb_id","")).lower()!=pdb
                    or audit_row.get("subset")!="snac_db"
                    or str(audit_row.get("source_structure_sha256",""))!=raw_sha):
                return False,f"holdout target is not bound to its exact audited SNAC source: {source_id}"
            target_paths.add(rel);target_pdbs.add(pdb);target_by_pdb[pdb]=row

        quarantine_paths=set()
        for row in quarantined:
            if not isinstance(row,dict):
                return False,"malformed holdout quarantine record"
            rel=str(row.get("path","")).replace("\\","/")
            expected_sha=str(row.get("sha256",""))
            if not rel.startswith("graphs/holdout_quarantine/") or not expected_sha:
                return False,f"invalid holdout quarantine binding: {row!r}"
            path=(dataset/rel).resolve()
            if not path.is_relative_to(dataset) or not path.is_file():
                return False,f"quarantined graph missing/outside dataset: {rel}"
            manifest_row=by_path.get(rel)
            if not manifest_row or manifest_row.get("split")!="holdout_quarantine":
                return False,f"graph manifest does not bind quarantined graph: {rel}"
            actual_sha=sha256_of(path)
            if actual_sha!=expected_sha or str(manifest_row.get("sha256",""))!=actual_sha:
                return False,f"quarantined graph hash mismatch: {rel}"
            quarantine_paths.add(rel)
        if quarantine_paths and not quarantine_dir.is_dir():
            return False,"holdout quarantine directory is missing"

        manifest_holdout={path for path,row in by_path.items() if row.get("split")=="holdout"}
        manifest_quarantine={path for path,row in by_path.items() if row.get("split")=="holdout_quarantine"}
        if manifest_holdout!=target_paths:
            return False,"graph manifest holdout rows differ from frozen primary target set"
        if manifest_quarantine!=quarantine_paths:
            return False,"graph manifest quarantine rows differ from frozen quarantine set"
        if target_paths & quarantine_paths:
            return False,"holdout target/quarantine path overlap"

        sources=payload.get("source_structures")
        if not isinstance(sources,dict) or set(sources)!=target_pdbs:
            return False,"holdout source-structure bindings differ from primary PDB targets"
        for pdb,binding in sources.items():
            if not isinstance(binding,dict):
                return False,f"malformed source binding for {pdb}"
            path=Path(str(binding.get("path",""))).resolve()
            expected_sha=str(binding.get("sha256",""))
            source_id=str(binding.get("source_id",""))
            audited_sha=str(binding.get("audited_source_sha256",""))
            target=target_by_pdb[pdb]
            if (not path.is_relative_to(source_dir) or not path.is_file() or not expected_sha
                    or source_id!=target["source_id"]
                    or audited_sha!=target["source_structure_sha256"]):
                return False,f"invalid exact holdout source structure for {pdb}"
            if sha256_of(path)!=expected_sha:
                return False,f"holdout source-structure hash mismatch for {pdb}"
        return True,(f"verified {len(targets)} primary SNAC holdout graph(s) and "
                     f"{len(quarantined)} quarantined component member(s)")

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

        if "anarci" in required:
            hmmscan = shutil.which("hmmscan")
            checks["anarci_hmmscan"] = hmmscan
            if not hmmscan:
                missing_resources.append("hmmscan (HMMER; required by ANARCI)")

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
        elif rot.get("mode")=="pyrosetta_dun10":
            try:
                from nanoqc.qubo.subgraph_to_qubo import PyRosettaRotamerProvider
                provider=PyRosettaRotamerProvider()
                checks["pyrosetta_dun10_version"]=provider.version
                expected=str(rot.get("version_contains", ""))
                if not expected or expected not in provider.version:
                    raise RuntimeError(f"PyRosetta version does not match frozen protocol: {provider.version!r}")
                provider.load_bins({("LYS",-60,-40)})
                checks["pyrosetta_dun10_sample"]=True
            except Exception as exc:
                checks["pyrosetta_dun10_sample"]=False
                missing_resources.append(f"PyRosetta dun10 unavailable: {type(exc).__name__}: {exc}")

        clustering=((self.config.get("queue_freeze", {}) or {}).get("independence_clustering", {}) or {})
        if clustering.get("required",False):
            build_per_run=bool(clustering.get("build_per_run",False))
            foldseek=self.foldseek_executable() if build_per_run else None
            checks["foldseek_executable"]=foldseek if build_per_run else "prebuilt_table"
            cluster_map=resolve_path(self.config,clustering.get("cluster_map","")) if clustering.get("cluster_map") else None
            pairs=resolve_path(self.config,clustering.get("pair_tsv","")) if clustering.get("pair_tsv") else None
            available=bool(foldseek) if build_per_run else bool((cluster_map and cluster_map.is_file()) or (pairs and pairs.is_file()))
            checks["independence_cluster_input"]=available
            if not available:
                missing_resources.append("foldseek (set QP_FOLDSEEK or add to PATH)" if build_per_run
                                         else f"cluster_map_or_pair_tsv:{cluster_map}|{pairs}")

        external=self.config.get("external_validation", {}) or {}
        if external.get("required",False):
            ext=external.get("external_vhh", {}) or {}
            if ext.get("required",False):
                graph_dir,source_dir,from_run=self.external_vhh_dirs()
                # The antigen-fold holdout is produced by this run's own
                # queue_freeze, so it cannot exist yet at env_check time.
                graph_ok=from_run or (graph_dir.is_dir() and any(graph_dir.glob("*.pt")))
                checks["external_vhh_graphs"]=graph_ok
                checks["external_vhh_source"]=("run_antigen_fold_holdout" if from_run else str(graph_dir))
                checks["external_vhh_raw_structures"]=from_run or source_dir.is_dir()
                # The independence manifest is generated run-locally during
                # external_validation, after queue_freeze has frozen this run's
                # training dataset and cluster map; it is never read from the repo.
                if not graph_ok:
                    missing_resources.append(str(graph_dir))
                if not from_run and not source_dir.is_dir():
                    missing_resources.append(str(source_dir))
            structural=external.get("structural_baselines", {}) or {}
            if structural.get("required",False):
                faspr=Path(structural.get("faspr_executable",""))
                phenix=Path(structural.get("phenix_clashscore_executable",""))
                checks["faspr_executable"]=faspr.is_file()
                rotamer_binary=faspr.parent/"dun2010bbdep.bin"
                checks["faspr_rotamer_binary"]=rotamer_binary.is_file()
                checks["phenix_clashscore_executable"]=phenix.is_file()
                if not faspr.is_file():
                    missing_resources.append(str(faspr))
                if not rotamer_binary.is_file():
                    missing_resources.append(str(rotamer_binary))
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
        smoke_rotamer_mode=(self.config.get("qc_benchmark",{}).get("rotamer_model",{}) or {}).get("mode","dunbrack2010")
        if smoke_rotamer_mode=="dunbrack2010" and not smoke_rotamer_library.is_file():
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
                "--rotamer-mode", str(smoke_rotamer_mode),
                "--rotamer-library", str(smoke_rotamer_library),
            ]))
        else:
            print("[smoke_check] qc_benchmark input_dir not yet built; skipping that sub-check "
                  "(expected before queue_freeze has run).")

        # Like the benchmark sub-check above, this one reads THIS run's dataset;
        # without --dataset the pilot falls back to its standalone default path.
        # smoke_check runs before queue_freeze, so on a fresh run there is
        # nothing to read yet and the sub-check is skipped rather than failed.
        smoke_target = smoke_cfg.get("recovery_pilot_pdb_id")
        smoke_manifest = self.dataset_dir() / "graph_manifest.csv"
        smoke_argv = [
            self.venv_python, "-m", module_name("run_real_complex_pilot.py"),
            "--dataset", str(self.dataset_dir()),
            "--data-root", str(resolve_path(self.config, self.config["paths"]["data_root"])),
            "--out-dir", str(smoke_dir / "real_complex"),
            "--targets", str(smoke_cfg.get("recovery_pilot_targets", 1)),
            "--sites", str(smoke_cfg.get("recovery_pilot_sites", 6)),
            "--seeds", *[str(s) for s in smoke_cfg.get("recovery_pilot_seeds", [42])],
            "--max-evals", str(smoke_cfg.get("recovery_pilot_max_evals", 12)),
            "--outputs", str(smoke_cfg.get("recovery_pilot_outputs", 20)),
            "--pruning", "contact",  # avoid requiring a trained checkpoint for the smoke check
            "--rotamer-mode", str(smoke_rotamer_mode),
            "--rotamer-library", str(smoke_rotamer_library),
        ]
        if smoke_target:
            smoke_argv += ["--pdb-id", str(smoke_target)]
        if smoke_manifest.is_file():
            checks.append(("real_complex_smoke", smoke_argv))
        else:
            print("[smoke_check] dataset graph manifest not yet built; skipping the recovery sub-check "
                  "(expected before queue_freeze has run).")

        failures = []
        for name, argv in checks:
            returncode, log_path = self._run_subprocess(f"smoke_{name}", argv)
            if returncode != 0:
                failures.append(f"{name} exited {returncode} (see {log_path})")
        status = "completed" if not failures else "failed"
        detail = ("; ".join(failures) if failures else
                  f"All {len(checks)} smoke check(s) passed." if checks else
                  "No smoke check could run yet: this run has no dataset, which is expected before "
                  "queue_freeze. Re-run with --only smoke_check --resume <run> to exercise them.")
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
            "--interface-contact-cutoff", str(cfg.get("interface_contact_cutoff_angstrom", 4.5)),
            "--max-resolution", str(cfg.get("max_resolution_angstrom", 3.0)),
            "--min-interresidue-heavy-distance", str(cfg.get(
                "min_interresidue_heavy_distance_angstrom",1.0)),
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
            "--interface-label-cutoff", str(qf_cfg["graph_build"].get("interface_label_cutoff_angstrom", 4.5)),
            "--interface-sensitivity-cutoffs", *[
                str(v) for v in qf_cfg["graph_build"].get(
                    "interface_sensitivity_cutoffs_angstrom", [3.5, 5.0])
            ],
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
            from nanoqc.data.audit_all_datasets import formal_clustering_pdb_ids
            audit_rows=[]
            for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    audit_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            # Match the graph-builder's source precedence and row-level QC
            # before clustering: excluded structures must not bridge otherwise
            # independent antigen-fold components.
            ids=set(formal_clustering_pdb_ids(
                audit_rows,
                qf_cfg["graph_build"].get("min_interface_residues", 15),
            ))
            # External VHH structures are part of the SAME frozen structural
            # similarity universe. They must not be appended as untracked
            # singleton clusters only at external-validation time.
            external_cfg=(self.config.get("external_validation",{}) or {}).get("external_vhh",{}) or {}
            external_graph_dir,external_source_dir,external_from_run=self.external_vhh_dirs()
            # A holdout carved from this run's own audited data is already in
            # the universe; only a genuinely external set adds PDBs to it.
            if external_cfg.get("required",False) and not external_from_run:
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

        if clustering_cfg.get("required",False) and clustering_cfg.get("build_per_run",False):
            external_cfg=(self.config.get("external_validation",{}) or {}).get("external_vhh",{}) or {}
            if external_cfg.get("required",False) and external_cfg.get("graph_dir"):
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),None,
                    "Run-local Foldseek auto-build requires the internal antigen-fold holdout; "
                    "external VHH candidates need their own declared Foldseek preparation")
            pair_path=self.run_local_foldseek_pairs()
            ok,detail=self._ensure_run_local_foldseek_pairs(
                universe_path,audit_jsonl,int(qf_cfg["graph_build"].get("min_interface_residues",15)))
            if not ok:
                return StageResult("queue_freeze","failed",started,utc_timestamp(),None,detail)

        if clustering_cfg.get("required",False) and pair_path is not None and pair_path.is_file():
            universe_ids={
                line.strip().lower() for line in universe_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            query_col=int(clustering_cfg.get("query_column",0))
            target_col=int(clustering_cfg.get("target_column",1))
            covered_ids=set()
            # Streamed: this table has millions of rows, and reading it whole
            # held the entire file plus a list of its lines in memory (A36).
            with pair_path.open(encoding="utf-8-sig") as pair_handle:
                for raw_line in pair_handle:
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
                    f"Frozen structure-similarity pair table {pair_path} covers "
                    f"{len(universe_ids) - len(missing_pair_coverage)} of this run's {len(universe_ids)} "
                    f"universe PDBs; {len(missing_pair_coverage)} are not demonstrated as searched, e.g. "
                    f"{missing_pair_coverage[:20]}. Rebuild it against THIS run's universe: "
                    f"prepare_external_vhh.sh foldseek --run-dir {self.run_dir} --force"
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
        if qf_cfg["graph_build"].get("include_db55_auxiliary", False):
            graph_argv.append("--include-db55-auxiliary")
        if qf_cfg["graph_build"].get("no_cap", True):
            graph_argv += ["--no-cap", "--partition-seed", str(streams["partition"])]
        else:
            graph_argv += ["--target-hard", str(qf_cfg["graph_build"].get("target_hard_if_capped", 500)),
                            "--partition-seed", str(streams["partition"])]
        holdout_cfg = qf_cfg.get("antigen_fold_holdout", {}) or {}
        holdout_json = self.run_dir / "independence" / "antigen_fold_holdout.json"
        reuse_holdout=False
        if holdout_cfg.get("enabled", False) and holdout_json.is_file():
            reuse_holdout,holdout_detail=self._validate_frozen_antigen_holdout()
            if not reuse_holdout:
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),1,
                    "Existing antigen-fold holdout is inconsistent and cannot be safely rebuilt in-place: "
                    +holdout_detail+". Start a fresh run directory.",
                )

        graph_expected = [dataset_dir / name for name in
                           ("graph_manifest.csv", "graph_manifest.json",
                            "run_summary.json", "graph_dataset_delivery_report.md")]
        if reuse_holdout:
            graph_ok,graph_detail=self._artifacts_present(graph_expected)
            graph_log=self.log_dir/"queue_freeze_graph_build.log"
            returncode=0
            if not graph_ok:
                return StageResult(
                    "queue_freeze","failed",started,utc_timestamp(),1,
                    "Frozen holdout is valid but its parent graph-build artifacts are incomplete: "
                    +graph_detail+". Start a fresh run directory.",
                )
            graph_detail="Reused verified frozen antigen-fold holdout; graph rebuild skipped."
        else:
            if (dataset_dir / "run_summary.json").is_file():
                graph_argv.append("--resume")
            returncode, graph_log = self._run_subprocess("queue_freeze_graph_build", graph_argv)
            graph_ok, graph_detail = self._artifacts_present(graph_expected)
            if returncode != 0 or not graph_ok:
                return StageResult("queue_freeze", "failed", started, utc_timestamp(), returncode,
                                    f"build_final_pyg_dataset.py exited {returncode}; {graph_detail} (see {graph_log})",
                                    graph_argv, str(graph_log), graph_ok)

        # 3a2. Antigen-fold holdout (PROTOCOL_AMENDMENTS.md A10). Carved BEFORE
        # the validation queue is frozen and before any training, so the queue
        # is selected from the data the pipeline will actually train on, and
        # held-out complexes can never reach training or calibration.
        if holdout_cfg.get("enabled", False) and not reuse_holdout:
            holdout_argv = [
                self.venv_python, "-m", module_name("carve_holdout_clusters.py"),
                "--dataset-dir", str(dataset_dir),
                "--audit-dir", str(self.run_dir / "audit"),
                "--fold", str(holdout_cfg.get("fold", 1)),  # "1" or "1,2" (A13)
                "--min-clusters", str(holdout_cfg.get("min_components", 10)),
                "--min-train-components", str(holdout_cfg.get("min_train_components", 20)),
                "--out-json", str(holdout_json),
            ]
            holdout_rc, holdout_log = self._run_subprocess("carve_holdout_clusters", holdout_argv)
            if holdout_rc != 0 or not holdout_json.is_file():
                return StageResult(
                    "queue_freeze", "failed", started, utc_timestamp(), holdout_rc,
                    f"Antigen-fold holdout could not be carved (exit={holdout_rc}; see {holdout_log})",
                    holdout_argv, str(holdout_log), False)

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
        hardware=self.config.get("hardware",{})
        vq_argv += ["--preparation-workers",str(hardware.get("structural_prepare_workers",1)),
                    "--workers-per-gpu",str(hardware.get("structural_prepare_workers_per_gpu",1)),
                    "--gpu-devices",*[str(d) for d in hardware.get("structural_gpu_devices",
                                                                      [hardware.get("openmm_device","0")])]]
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
        if not cluster_map_path.is_file():
            return StageResult("queue_freeze","failed",started,utc_timestamp(),1,
                               f"Cluster adequacy check needs the run-local cluster map: {cluster_map_path}",
                               graph_argv + ["&&"] + vq_argv, f"{graph_log};{vq_log}", False)
        adequacy=cluster_adequacy(self.config,dataset_dir,selected_path,cluster_map_path)
        atomic_write_json(self.run_dir/"independence"/"cluster_adequacy.json",adequacy)
        if not adequacy["adequate"]:
            return StageResult("queue_freeze","failed",started,utc_timestamp(),1,
                               "Insufficient independent clusters for preregistered inference, detected "
                               "before any training or outcome: "+"; ".join(adequacy["shortfalls"]),
                               graph_argv + ["&&"] + vq_argv, f"{graph_log};{vq_log}", False)
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        cap_label = vq_cfg.get("target_count", 0) or "unlimited (all qualifying targets)"
        detail = (f"Graph build: {graph_detail} Validation queue: {len(selected)} targets frozen "
                  f"(cap: {cap_label}; dev queue excluded+exposure-flagged: {dev_cfg.get('excluded_pdb', [])}; "
                  f"selected_targets_sha256={freeze_manifest['selected_targets_sha256']}). {vq_detail}")
        return StageResult("queue_freeze", "completed", started, utc_timestamp(), 0, detail,
                            graph_argv + ["&&"] + vq_argv, f"{graph_log};{vq_log}", True)
