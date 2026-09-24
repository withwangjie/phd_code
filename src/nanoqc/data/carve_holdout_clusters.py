#!/usr/bin/env python3
"""Carve the antigen-fold holdout out of the training split (PROTOCOL_AMENDMENTS.md A10).

The holdout answers one question: does the frozen pipeline still hold on
antigen folds it never saw in training? It is a cluster-level holdout of the
kind current antibody benchmarks use, taken from the same audited snapshot,
not a separate database.

Complexes are grouped by the study's own layered isolation relation
(``train_egnn_pruning.layered_components``): VHH full-chain identity, CDR-H3
loop identity, antigen full-chain identity with coverage, or a shared frozen
family/structure cluster. Whole components go to the holdout, so no holdout
complex is homologous to any training complex under any of those criteria.

Which components go is decided by the same deterministic fold hash that
already assigns the internal validation fold, on a different fold index: the
component's member file names, never an outcome. Removing whole components
leaves every other component and its fold unchanged, so the internal
validation split is exactly what it would have been.

Graph files move from ``graphs/train`` to ``graphs/holdout`` and the graph
manifest is rewritten, so training and calibration read the training
directory and cannot reach the holdout. One audited raw structure per holdout
PDB is copied beside them, which lets the unchanged independence audit bind
every holdout graph to a raw file exactly as it does for an external set.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.model import train_egnn_pruning as training

HOLDOUT_SPLIT = "holdout"
STRUCTURE_SUFFIXES = (".cif.gz", ".pdb.gz", ".mmcif", ".cif", ".pdb")


def source_files_by_pdb(audit_jsonl: Path) -> dict[str, dict]:
    """One audited, readable source file per four-character PDB ID."""
    chosen: dict[str, dict] = {}
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pdb = str(row.get("pdb_id", "")).strip().lower()
        if len(pdb) != 4 or not pdb.isalnum() or not row.get("valid", True):
            continue
        chosen.setdefault(pdb, dict(path=row["path"], member=row.get("member", "")))
    return chosen


def copy_source_structure(source: dict, pdb: str, destination: Path) -> Path:
    """Write the audited raw file as ``<pdb><suffix>``, extracting a zip member."""
    name = str(source.get("member") or source["path"])
    suffix = next((s for s in STRUCTURE_SUFFIXES if name.lower().endswith(s)), None)
    if suffix is None:
        raise ValueError(f"{pdb}: unsupported raw structure {name}")
    if source.get("member"):
        with zipfile.ZipFile(source["path"]) as archive:
            payload = archive.read(source["member"])
    else:
        payload = Path(source["path"]).read_bytes()
    if suffix.endswith(".gz"):
        payload = gzip.decompress(payload)
        suffix = suffix[: -len(".gz")]
    out = destination / f"{pdb}{suffix}"
    out.write_bytes(payload)
    return out


def select_components(components: Sequence[Sequence[Path]], fold: int,
                      pool_size: Optional[int] = None) -> list[list[Path]]:
    """Components hashed to ``fold``; a component pinned to training (A11) never is."""
    pool = sum(len(c) for c in components) if pool_size is None else pool_size
    return [list(component) for component in components if training.assigned_fold(component, pool) == fold]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="This run's dataset directory")
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory with data_audit_details.jsonl")
    parser.add_argument("--fold", type=int, default=1,
                        help=f"Fold index held out, of {training.SPLIT_FOLDS} "
                             f"(the internal validation fold is {training.VALIDATION_FOLD})")
    parser.add_argument("--min-clusters", type=int, required=True,
                        help="Preregistered minimum of independent holdout components")
    parser.add_argument("--min-train-components", type=int, default=2,
                        help="Components that must remain for training plus internal validation")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.fold == training.VALIDATION_FOLD or not 0 <= args.fold < training.SPLIT_FOLDS:
        parser.error(f"--fold must be in [0,{training.SPLIT_FOLDS}) and differ from the internal "
                     f"validation fold {training.VALIDATION_FOLD}")
    train_dir = args.dataset_dir / "graphs" / "train"
    holdout_dir = args.dataset_dir / "graphs" / HOLDOUT_SPLIT
    manifest_path = args.dataset_dir / "graph_manifest.json"
    for required in (train_dir, manifest_path, args.audit_dir / "data_audit_details.jsonl"):
        if not required.exists():
            raise SystemExit(f"Required input missing: {required}")
    if holdout_dir.exists() and any(holdout_dir.glob("*.pt")):
        raise SystemExit(f"{holdout_dir} already holds graphs; the holdout is carved once per run")

    paths = sorted(train_dir.glob("*.pt"), key=lambda path: path.name.lower())
    if not paths:
        raise SystemExit(f"No training graphs in {train_dir}")
    components = training.layered_components(paths)
    holdout_components = select_components(components, args.fold, len(paths))
    remaining = len(components) - len(holdout_components)
    pinned = [c for c in components if training.pinned_to_training(c, len(paths))]
    # Training later splits the smaller remaining pool; the pinned set must not
    # change with it, or the internal validation split would differ from the
    # one this carve assumes.
    remaining_pool = len(paths) - sum(len(c) for c in holdout_components)
    repinned = [c for c in components if c not in holdout_components
                and training.pinned_to_training(c, remaining_pool)]
    # Compare the components themselves, not their sizes: two distinct
    # components of equal size must not cancel out and hide a change.
    def signature(cs):
        return sorted(tuple(sorted(path.name for path in c)) for c in cs)
    if signature(repinned) != signature(pinned):
        raise SystemExit(f"Components pinned to training differ before ({sorted(map(len, pinned))}) and after "
                         f"({sorted(map(len, repinned))}) the carve; a component sits at the 1/"
                         f"{training.SPLIT_FOLDS} pin boundary. Record this and amend the pin rule.")
    if len(holdout_components) < args.min_clusters:
        raise SystemExit(
            f"Antigen-fold holdout has {len(holdout_components)} independent component(s) at fold "
            f"{args.fold}, below the preregistered minimum {args.min_clusters}. The training pool is too "
            f"small or too homologous for this holdout; it is an outcome-free data-composition failure.")
    if remaining < args.min_train_components:
        raise SystemExit(f"Only {remaining} component(s) would remain for training; need "
                         f"{args.min_train_components}")

    holdout_paths = sorted((path for component in holdout_components for path in component),
                           key=lambda path: path.name.lower())
    sources = source_files_by_pdb(args.audit_dir / "data_audit_details.jsonl")
    holdout_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = args.dataset_dir / "holdout_source_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_relative = {str(row.get("path", "")).replace("\\", "/"): row for row in manifest}
    moved, structures = [], {}
    for path in holdout_paths:
        relative = f"graphs/train/{path.name}"
        row = by_relative.get(relative)
        if row is None:
            raise SystemExit(f"Training graph absent from the manifest: {relative}")
        pdb = str(row.get("pdb_id", "")).lower()
        if pdb not in sources:
            raise SystemExit(f"No audited raw structure for holdout PDB {pdb}")
        if pdb not in structures:
            structures[pdb] = copy_source_structure(sources[pdb], pdb, structures_dir)
        destination = holdout_dir / path.name
        shutil.move(str(path), str(destination))
        row["split"] = HOLDOUT_SPLIT
        row["path"] = f"graphs/{HOLDOUT_SPLIT}/{path.name}"
        row["sha256"] = sha256(destination)
        moved.append(dict(pdb_id=pdb, path=row["path"], sha256=row["sha256"],
                          family_structure_cluster=row.get("family_structure_cluster", "")))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_csv = args.dataset_dir / "graph_manifest.csv"
    if manifest_csv.is_file() and manifest:
        with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)

    by_cluster: dict[str, list[str]] = defaultdict(list)
    for row in moved:
        by_cluster[row["family_structure_cluster"]].append(row["pdb_id"])
    payload = dict(
        schema="antigen_fold_holdout_v1", split=HOLDOUT_SPLIT, fold=args.fold,
        folds=training.SPLIT_FOLDS, internal_validation_fold=training.VALIDATION_FOLD,
        selection="layered isolation components whose deterministic name-hash fold equals --fold, "
                  "except components pinned to training",
        criteria=dict(vhh_full_chain_identity=training.VHH_IDENTITY_THRESHOLD,
                      cdr_h3_identity=training.CDR_H3_IDENTITY_THRESHOLD,
                      antigen_identity=training.ANTIGEN_IDENTITY_THRESHOLD,
                      antigen_min_length_coverage=training.ANTIGEN_MIN_LENGTH_COVERAGE),
        components_total=len(components), holdout_components=len(holdout_components),
        pinned_to_training=dict(rule=f"component size >= {training.PIN_MIN_COMPONENT} and > 1/"
                                     f"{training.SPLIT_FOLDS} of the pool (PROTOCOL_AMENDMENTS.md A11)",
                                pool=len(paths), component_sizes=sorted((len(c) for c in pinned), reverse=True),
                                graphs=sum(len(c) for c in pinned)),
        training_components_remaining=remaining, min_clusters=args.min_clusters,
        graphs=len(moved), structure_clusters=sorted(by_cluster), graph_dir=str(holdout_dir.resolve()),
        source_structure_dir=str(structures_dir.resolve()),
        source_structures={pdb: dict(path=str(path.resolve()), sha256=sha256(path))
                           for pdb, path in sorted(structures.items())},
        targets=moved,
        scope=("outcome-free: components are chosen by a hash of their member file names, "
               "before any training, benchmark or structural result exists"),
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[carve_holdout_clusters] held out {len(moved)} graph(s) in {len(holdout_components)} "
          f"independent component(s) at fold {args.fold}; {remaining} component(s) remain for training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
