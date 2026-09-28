"""Run-level records: manifests, run directories, inventory and results audit.

``build_run_manifest`` fingerprints every orchestrated module;
``audit_experiment_results`` decides whether a run may be called complete.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from nanoqc.common.repo_io import sha256_file as sha256_of, repo_path, DOCS_DIR
from typing import TYPE_CHECKING

from nanoqc.pipeline.orchestrator_common import (
    ORCHESTRATED_SCRIPTS,
    STAGE_ORDER,
    StageResult,
    atomic_write_json,
    git_commit_hash,
    resolve_path,
    utc_run_stamp,
    utc_timestamp,
)

if TYPE_CHECKING:  # annotation only; importing it at runtime would be circular
    from nanoqc.pipeline.run_full_experiment import Orchestrator


def save_derived_child(streams: Dict[str, int], stream_name: str, *labels: str) -> int:
    from nanoqc.common.seed_streams import derive_child_seed
    return derive_child_seed(streams[stream_name], *labels)


# ---------------------------------------------------------------------------
# Run-directory / manifest / lock management
# ---------------------------------------------------------------------------

def build_run_manifest(config: Dict[str, Any], repo_root: Path) -> Dict[str, Any]:
    evidence_path=repo_root/DOCS_DIR/"METHODS_EVIDENCE.md"
    results_contract_path=repo_root/DOCS_DIR/"RESULTS_CONTRACT.md"
    amendments_path=repo_root/DOCS_DIR/"PROTOCOL_AMENDMENTS.md"
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
        # Binds each run to the protocol amendments in force when it launched.
        protocol_amendments_sha256=(
            sha256_of(amendments_path) if amendments_path.is_file() else None
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
