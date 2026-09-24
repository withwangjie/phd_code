#!/usr/bin/env python3
"""Two-pass preparation of the external VHH set (docs/PROTOCOL_AMENDMENTS.md A8).

The formal run needs the external graphs and the Foldseek pair table before
it starts: env_check requires both, and queue_freeze adds the external PDB IDs
to the clustering universe. Training-overlap checks, however, need the
training set that queue_freeze builds. The loop is broken in two passes, and
neither pass reads any experimental outcome:

  audit   Standalone data audit with the frozen config's gates, giving the
          study PDB IDs (excluded from the external set and part of the
          Foldseek universe).
  pass1   Candidate selection without training data, then graph building.
          Writes foldseek_universe.txt (study + external IDs). The Foldseek
          pair table must cover this universe.
  (run)   ``deploy_launch.sh --stop-after queue_freeze`` builds the training
          set and cluster map; no model is trained and no outcome exists.
  pass2   Re-selects against that run's training set and cluster map,
          then rebuilds the graphs from the survivors. Keep the pass-1 pair
          table unchanged: it still links the same internal PDBs, so the
          fresh formal run rebuilds the identical training/test split.

Every formal run then re-certifies independence with
audit_external_vhh_independence.py.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import yaml

from nanoqc.common.repo_io import REPO_ROOT, sha256_file as sha256

PDB_ID = re.compile(r"[a-z0-9]{4}")


def load_config(path: Optional[Path]) -> dict:
    if path is None:
        resolved = REPO_ROOT / ".runtime" / "resolved_runtime_config.yaml"
        path = resolved if resolved.is_file() else REPO_ROOT / "configs" / "full_experiment_config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_source"] = str(path)
    return config


def repo_path(value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def settings(config: dict, prep_dir: Optional[Path]) -> dict:
    external = config["external_validation"]["external_vhh"]
    homology = config["queue_freeze"]["homology_isolation"]
    return dict(
        prep=(prep_dir or REPO_ROOT / "data" / "external_vhh" / "prep"),
        graph_dir=repo_path(external["graph_dir"]),
        source_dir=repo_path(os.environ.get("QP_EXTERNAL_VHH_SOURCE_DIR") or external["source_structure_dir"]),
        min_clusters=int(external.get("min_clusters", 10)),
        pair_tsv=repo_path(config["queue_freeze"]["independence_clustering"]["pair_tsv"]),
        thresholds=["--vhh-threshold", str(homology.get("vhh_full_chain_identity", 0.80)),
                    "--cdr-h3-threshold", str(homology.get("cdr_h3_identity", 0.50)),
                    "--antigen-threshold", str(homology.get("antigen_identity", 0.30)),
                    "--antigen-min-length-coverage", str(homology.get("antigen_min_length_coverage", 0.70))],
        max_resolution=str(config.get("data_audit", {}).get("max_resolution_angstrom", 3.0)),
    )


def study_pdb_ids(audit_jsonl: Path) -> list[str]:
    """Same rule as queue_freeze's clustering universe: every audited four-character PDB ID."""
    ids = set()
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            pdb = str(json.loads(line).get("pdb_id", "")).strip().lower()
        except json.JSONDecodeError:
            continue
        if PDB_ID.fullmatch(pdb):
            ids.add(pdb)
    return sorted(ids)


def run_module(module: str, argv: list[str]) -> None:
    print(f"[prepare_external_vhh] python -m {module} {' '.join(argv)}", flush=True)
    subprocess.run([sys.executable, "-m", module, *argv], check=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cmd_audit(args, config, s) -> int:
    audit_cfg = config.get("data_audit", {}) or {}
    data_root = args.data_root or repo_path(os.environ.get("QP_DATA_ROOT") or config["paths"]["data_root"])
    argv = ["--data", str(data_root), "--workers", str(audit_cfg.get("workers", 4)),
            "--limit", str(audit_cfg.get("limit", 0)), "--out", str(s["prep"] / "audit"),
            "--max-resolution", s["max_resolution"],
            "--min-interface-occupancy", str(audit_cfg.get("min_interface_occupancy", 0.90))]
    for key, flag in (("allow_interface_altloc", "--allow-interface-altloc"),
                      ("allow_unknown_resolution", "--allow-unknown-resolution"),
                      ("allow_incomplete_interface_sidechains", "--allow-incomplete-interface-sidechains")):
        if audit_cfg.get(key, False):
            argv.append(flag)
    run_module("nanoqc.data.audit_all_datasets", argv)
    ids = study_pdb_ids(s["prep"] / "audit" / "data_audit_details.jsonl")
    (s["prep"] / "study_pdb_ids.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    print(f"[prepare_external_vhh] {len(ids)} study PDB IDs -> {s['prep'] / 'study_pdb_ids.txt'}")
    return 0


def _select(args, s, out: Path, extra: list[str]) -> dict:
    ids = s["prep"] / "study_pdb_ids.txt"
    if not ids.is_file():
        raise SystemExit("Run the 'audit' step first (study_pdb_ids.txt missing)")
    run_module("nanoqc.data.select_external_vhh_candidates", [
        "--sabdab-summary", *map(str, args.sabdab_summary), "--released-after", args.released_after,
        "--structures-dir", str(s["source_dir"]), *(["--download"] if args.download else []),
        "--exclude-pdbs", str(ids), "--min-clusters", str(s["min_clusters"]),
        "--max-resolution", s["max_resolution"], *s["thresholds"], *extra, "--out-dir", str(out)])
    return json.loads((out / "candidates.json").read_text(encoding="utf-8"))


def _build(s, candidates: Path) -> dict:
    graph_dir = s["graph_dir"]
    if graph_dir.exists() and any(graph_dir.iterdir()):
        backup = s["prep"] / f"graphs_backup_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
        shutil.move(str(graph_dir), str(backup))
        print(f"[prepare_external_vhh] previous graphs moved to {backup}")
    run_module("nanoqc.data.build_external_vhh_graphs", [
        "--candidates", str(candidates), "--structures-dir", str(s["source_dir"]), "--out-dir", str(graph_dir)])
    return json.loads((graph_dir / "external_graph_manifest.json").read_text(encoding="utf-8"))


def cmd_pass1(args, config, s) -> int:
    out = s["prep"] / "selection_pass1"
    extra = ["--training-dataset", str(args.training_dataset)] if args.training_dataset else []
    selection = _select(args, s, out, extra)
    manifest = _build(s, out / "candidates.json")
    external = sorted(g["pdb_id"] for g in manifest["graphs"])
    study = (s["prep"] / "study_pdb_ids.txt").read_text(encoding="utf-8").split()
    universe = sorted(set(study) | set(external))
    (s["prep"] / "foldseek_universe.txt").write_text("".join(f"{i}\n" for i in universe), encoding="utf-8")
    write_json(s["prep"] / "pass1_record.json", dict(
        released_after=args.released_after, selected=selection["selected"],
        independent_groups=selection["independent_groups"], graphs=len(external),
        build_failures=manifest["failures"], external_pdb_ids=external,
        candidates_sha256=sha256(out / "candidates.json"), universe_size=len(universe)))
    print(f"[prepare_external_vhh] pass 1: {len(external)} external graphs, "
          f"{selection['independent_groups']} independent groups (required {s['min_clusters']}).")
    print(f"Next: Foldseek over the {len(universe)} PDB IDs in {s['prep'] / 'foldseek_universe.txt'} "
          f"-> {s['pair_tsv']}, then ./scripts/deploy_launch.sh --stop-after queue_freeze, "
          "then this script's pass2 with that run directory.")
    return 0


def cmd_pass2(args, config, s) -> int:
    run_dir = args.run_dir.resolve()
    dataset = run_dir / config["paths"].get("dataset_dir", "dataset")
    cluster_map = run_dir / "independence" / "pdb_family_clusters.json"
    provenance = cluster_map.with_suffix(".provenance.json")
    for required in (dataset / "graph_manifest.json", cluster_map, provenance):
        if not required.is_file():
            raise SystemExit(f"{required} missing: run --stop-after queue_freeze to completion first")
    pass1 = json.loads((s["prep"] / "pass1_record.json").read_text(encoding="utf-8"))
    frozen_pairs = json.loads(provenance.read_text(encoding="utf-8")).get("source_pairs_sha256")
    if not s["pair_tsv"].is_file() or sha256(s["pair_tsv"]) != frozen_pairs:
        raise SystemExit("The Foldseek pair table differs from the one this run clustered with; "
                         "restore the pass-1 table before pass 2")
    if args.released_after != pass1["released_after"]:
        raise SystemExit(f"--released-after must stay {pass1['released_after']} (pass 1)")
    out = s["prep"] / "selection_pass2"
    selection = _select(args, s, out, ["--training-dataset", str(dataset), "--cluster-map", str(cluster_map)])
    manifest = _build(s, out / "candidates.json")
    kept = sorted(g["pdb_id"] for g in manifest["graphs"])
    dropped = sorted(set(pass1["external_pdb_ids"]) - set(kept))
    added = sorted(set(kept) - set(pass1["external_pdb_ids"]))
    if added:
        raise SystemExit(f"pass 2 added PDBs absent from the pass-1 Foldseek universe: {added}")
    write_json(s["prep"] / "pass2_record.json", dict(
        queue_freeze_run=str(run_dir), training_graph_manifest_sha256=sha256(dataset / "graph_manifest.json"),
        cluster_map_sha256=sha256(cluster_map), pair_tsv_sha256=frozen_pairs,
        selected=selection["selected"], independent_groups=selection["independent_groups"],
        adequate=selection["adequate"], cdr3_agreement_with_training=selection.get("cdr3_agreement_with_training"),
        external_pdb_ids=kept, dropped_after_training_check=dropped, build_failures=manifest["failures"],
        candidates_sha256=sha256(out / "candidates.json")))
    agreement = selection.get("cdr3_agreement_with_training") or {}
    print(f"[prepare_external_vhh] pass 2: {len(kept)} external graphs ({len(dropped)} dropped for training overlap); "
          f"{selection['independent_groups']} independent groups (required {s['min_clusters']}): "
          f"{'adequate' if selection['adequate'] else 'NOT adequate'}.")
    if agreement.get("compared"):
        print(f"CDR-H3 agreement with training SNAC annotation: {agreement['agreed']}/{agreement['compared']}.")
    print("Next: start a FRESH formal run (./scripts/deploy_launch.sh) with the unchanged pair table.")
    return 0 if selection["adequate"] else 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None,
                        help="Default: .runtime/resolved_runtime_config.yaml, else configs/full_experiment_config.yaml")
    parser.add_argument("--prep-dir", type=Path, default=None, help="Default: data/external_vhh/prep")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Standalone data audit -> study_pdb_ids.txt")
    audit.add_argument("--data-root", type=Path, default=None)
    for name in ("pass1", "pass2"):
        p = sub.add_parser(name)
        p.add_argument("--sabdab-summary", type=Path, nargs="+", required=True)
        p.add_argument("--released-after", required=True, help="YYYY-MM-DD, after the training data snapshot")
        p.add_argument("--download", action="store_true")
        if name == "pass1":
            p.add_argument("--training-dataset", type=Path, default=None,
                           help="Optional earlier dataset for an early, non-binding overlap check")
        else:
            p.add_argument("--run-dir", type=Path, required=True, help="The --stop-after queue_freeze run")
    args = parser.parse_args(argv)
    if args.command != "audit":
        dt.date.fromisoformat(args.released_after)
    config = load_config(args.config)
    s = settings(config, args.prep_dir)
    s["prep"].mkdir(parents=True, exist_ok=True)
    return dict(audit=cmd_audit, pass1=cmd_pass1, pass2=cmd_pass2)[args.command](args, config, s)


if __name__ == "__main__":
    raise SystemExit(main())
