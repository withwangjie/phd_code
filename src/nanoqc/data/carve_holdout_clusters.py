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

Selected components are removed from ``graphs/train`` as a whole. Exactly one
SNAC-DB primary graph per PDB is moved to ``graphs/holdout`` and is scored;
SAbDab auxiliary graphs and duplicate same-PDB SNAC graphs from those selected
components move to ``graphs/holdout_quarantine``, so they cannot train and
cannot become primary evaluation units. Each scored graph is bound to the exact
audit source by ``source_id`` and the audit-time raw-structure SHA-256.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.model import train_egnn_pruning as training

HOLDOUT_SPLIT = "holdout"
STRUCTURE_SUFFIXES = (".cif.gz", ".pdb.gz", ".mmcif", ".cif", ".pdb")


def source_files_by_id(audit_jsonl: Path) -> dict[str, dict]:
    """Audited valid sources keyed by the exact graph source_id."""
    chosen: dict[str, dict] = {}
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("valid", True):
            continue
        source_id = str(row.get("id", "")).strip()
        if not source_id:
            continue
        if source_id in chosen:
            raise ValueError(f"Duplicate audit source_id {source_id!r}")
        chosen[source_id] = dict(
            id=source_id,
            pdb_id=str(row.get("pdb_id", "")).strip().lower(),
            subset=str(row.get("subset", "")),
            path=row["path"],
            member=row.get("member", ""),
            source_structure_sha256=str(row.get("source_structure_sha256", "")),
        )
    return chosen


def copy_source_structure(source: dict, pdb: str, destination: Path,
                          expected_sha256: str) -> Path:
    """Copy one exact audited source after re-verifying its raw bytes."""
    name = str(source.get("member") or source["path"])
    suffix = next((s for s in STRUCTURE_SUFFIXES if name.lower().endswith(s)), None)
    if suffix is None:
        raise ValueError(f"{pdb}: unsupported raw structure {name}")
    if source.get("member"):
        with zipfile.ZipFile(source["path"]) as archive:
            payload = archive.read(source["member"])
    else:
        payload = Path(source["path"]).read_bytes()
    recorded = str(source.get("source_structure_sha256") or "")
    actual = hashlib.sha256(payload).hexdigest()
    if not expected_sha256 or recorded != expected_sha256:
        raise ValueError(
            f"{pdb}: graph/audit raw-structure SHA-256 mismatch "
            f"(graph={expected_sha256!r}, audit={recorded!r})"
        )
    if actual != expected_sha256:
        raise ValueError(
            f"{pdb}: raw structure changed since data audit "
            f"(expected {expected_sha256}, observed {actual})"
        )
    if suffix.endswith(".gz"):
        payload = gzip.decompress(payload)
        suffix = suffix[: -len(".gz")]
    out = destination / f"{pdb}{suffix}"
    out.write_bytes(payload)
    return out


def write_outputs(manifest_path: Path, manifest: list[dict], out_json: Path, payload: dict) -> None:
    """Restore earlier metadata if any output write fails."""
    manifest_csv = manifest_path.with_suffix(".csv")
    paths = [manifest_path, out_json]
    if manifest_csv.is_file() and manifest:
        paths.append(manifest_csv)
    originals = {path: path.read_bytes() if path.is_file() else None for path in paths}
    try:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if manifest_csv in paths:
            with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
                writer.writeheader()
                writer.writerows(manifest)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except Exception:
        for path, original in originals.items():
            if original is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(original)
        raise


def parse_folds(value: str) -> list[int]:
    """``"1"`` or ``"1,2"`` -> a sorted list of distinct fold indices."""
    folds = sorted({int(part) for part in str(value).replace(" ", "").split(",") if part})
    if not folds:
        raise ValueError("no fold given")
    return folds


def holdout_folds(folds: int, count: int, validation_fold: int) -> list[int]:
    """The ``count`` lowest fold indices that are not the internal validation fold.

    A rule, never a search: with too few independent components in one fold,
    the holdout takes the next fold up, not the fold that happens to hold most
    of them (PROTOCOL_AMENDMENTS.md A13).
    """
    available = [f for f in range(folds) if f != validation_fold]
    if not 1 <= count <= len(available):
        raise ValueError(f"count must be in [1,{len(available)}]")
    return available[:count]


def select_components(components: Sequence[Sequence[Path]], folds: Sequence[int],
                      pool_size: Optional[int] = None) -> list[list[Path]]:
    """Components hashed to any of ``folds``; a component pinned to training (A11) never is."""
    wanted = set(folds)
    pool = sum(len(c) for c in components) if pool_size is None else pool_size
    return [list(component) for component in components
            if training.assigned_fold(component, pool) in wanted]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="This run's dataset directory")
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory with data_audit_details.jsonl")
    parser.add_argument("--fold", default="1",
                        help=f"Fold index, or comma-separated indices, held out of {training.SPLIT_FOLDS} "
                             f"(the internal validation fold is {training.VALIDATION_FOLD}). Several folds "
                             f"are held out together when one does not reach --min-clusters (A13).")
    parser.add_argument("--min-clusters", type=int, required=True,
                        help="Preregistered minimum of independent holdout components")
    parser.add_argument("--min-train-components", type=int, default=2,
                        help="Components that must remain for training plus internal validation")
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        folds = parse_folds(args.fold)
    except ValueError as exc:
        parser.error(f"--fold: {exc}")
    if any(f == training.VALIDATION_FOLD or not 0 <= f < training.SPLIT_FOLDS for f in folds):
        parser.error(f"--fold must be in [0,{training.SPLIT_FOLDS}) and differ from the internal "
                     f"validation fold {training.VALIDATION_FOLD}")
    if len(folds) >= training.SPLIT_FOLDS - 1:
        parser.error(f"--fold cannot hold out every fold but the internal validation one "
                     f"({training.SPLIT_FOLDS - 1} available); training would keep only pinned components")
    train_dir = args.dataset_dir / "graphs" / "train"
    holdout_dir = args.dataset_dir / "graphs" / HOLDOUT_SPLIT
    quarantine_split = "holdout_quarantine"
    quarantine_dir = args.dataset_dir / "graphs" / quarantine_split
    manifest_path = args.dataset_dir / "graph_manifest.json"
    audit_jsonl = args.audit_dir / "data_audit_details.jsonl"
    for required in (train_dir, manifest_path, audit_jsonl):
        if not required.exists():
            raise SystemExit(f"Required input missing: {required}")
    for directory in (holdout_dir, quarantine_dir):
        if directory.exists() and any(directory.glob("*.pt")):
            raise SystemExit(f"{directory} already holds graphs; the holdout is carved once per run")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise SystemExit("graph_manifest.json must be a list")
    by_relative = {str(row.get("path", "")).replace("\\", "/"): row
                   for row in manifest if isinstance(row, dict)}

    paths = sorted(train_dir.glob("*.pt"), key=lambda path: path.name.lower())
    if not paths:
        raise SystemExit(f"No training graphs in {train_dir}")

    def manifest_row(path: Path) -> dict:
        relative = f"graphs/train/{path.name}"
        row = by_relative.get(relative)
        if row is None:
            raise SystemExit(f"Training graph absent from the manifest: {relative}")
        if sha256(path) != row.get("sha256"):
            raise SystemExit(f"Training graph differs from the frozen manifest: {relative}")
        return row

    components = training.layered_components(paths)
    candidate_components = select_components(components, folds, len(paths))
    # SAbDab is an auxiliary training source. It participates in the layered
    # components so it can quarantine homologues of a primary SNAC target, but
    # an auxiliary-only component is not removed merely because its hash lands
    # on a holdout fold.
    holdout_components = [
        component for component in candidate_components
        if any(manifest_row(path).get("subset_source") == "snac_db" for path in component)
    ]
    remaining = len(components) - len(holdout_components)
    pinned = [c for c in components if training.pinned_to_training(c, len(paths))]
    remaining_pool = len(paths) - sum(len(c) for c in holdout_components)
    repinned = [c for c in components if c not in holdout_components
                and training.pinned_to_training(c, remaining_pool)]

    def signature(cs):
        return sorted(tuple(sorted(path.name for path in c)) for c in cs)

    if signature(repinned) != signature(pinned):
        raise SystemExit(f"Components pinned to training differ before ({sorted(map(len, pinned))}) and after "
                         f"({sorted(map(len, repinned))}) the carve; a component sits at the 1/"
                         f"{training.SPLIT_FOLDS} pin boundary. Record this and amend the pin rule.")
    if len(holdout_components) < args.min_clusters:
        raise SystemExit(
            f"Primary SNAC antigen-fold holdout has {len(holdout_components)} independent component(s) "
            f"at fold(s) {','.join(map(str, folds))}, below the preregistered minimum {args.min_clusters}. "
            f"The primary pool is too small or too homologous for this holdout; holding out one more fold "
            f"(A13) is the preregistered response.")
    if remaining < args.min_train_components:
        raise SystemExit(f"Only {remaining} component(s) would remain for training; need "
                         f"{args.min_train_components}")

    holdout_paths = sorted((path for component in holdout_components for path in component),
                           key=lambda path: path.name.lower())
    plan = []
    for path in holdout_paths:
        row = manifest_row(path)
        pdb = str(row.get("pdb_id", "")).strip().lower()
        source_id = str(row.get("source_id", "")).strip()
        source_sha = str(row.get("source_structure_sha256", "")).strip()
        if len(pdb) != 4 or not pdb.isalnum():
            raise SystemExit(f"Holdout graph lacks valid PDB identity: {path}")
        if not source_id or not source_sha:
            raise SystemExit(f"Holdout graph lacks exact audited-source provenance: {path}")
        plan.append(dict(path=path, row=row, pdb=pdb, source_id=source_id,
                         source_sha=source_sha,
                         subset_source=str(row.get("subset_source", ""))))

    # Primary evaluation is SNAC-only. Keep one deterministic graph per PDB to
    # match the downstream one-raw-structure-per-PDB contract; every other
    # member of the selected component remains isolated in quarantine.
    primary_by_pdb: dict[str, list[dict]] = defaultdict(list)
    for item in plan:
        if item["subset_source"] == "snac_db":
            primary_by_pdb[item["pdb"]].append(item)
    target_paths = {
        min(items, key=lambda item: item["path"].name.lower())["path"]
        for items in primary_by_pdb.values()
    }
    target_plan, quarantine_plan = [], []
    for item in plan:
        if item["path"] in target_paths:
            target_plan.append(item)
        else:
            item["quarantine_reason"] = (
                "auxiliary_sabdab_component_member"
                if item["subset_source"] == "sabdab_vhh"
                else "duplicate_primary_pdb"
            )
            quarantine_plan.append(item)
    if not target_plan:
        raise SystemExit("Selected holdout components contain no primary SNAC targets")

    sources = source_files_by_id(audit_jsonl)
    for item in target_plan:
        source = sources.get(item["source_id"])
        if source is None:
            raise SystemExit(f"No exact audited source_id for holdout graph: {item['source_id']}")
        if source["pdb_id"] != item["pdb"]:
            raise SystemExit(
                f"Holdout graph/source PDB mismatch: {item['pdb']} vs {source['pdb_id']} "
                f"for {item['source_id']}")
        if source["subset"] != item["subset_source"]:
            raise SystemExit(
                f"Holdout graph/source subset mismatch: {item['subset_source']} vs {source['subset']} "
                f"for {item['source_id']}")
        if source["source_structure_sha256"] != item["source_sha"]:
            raise SystemExit(f"Holdout graph/audit source hash mismatch for {item['source_id']}")
        item["source"] = source

    structures_dir = args.dataset_dir / "holdout_source_structures"
    with tempfile.TemporaryDirectory(prefix=".holdout_sources_", dir=args.dataset_dir) as staging:
        staged = {}
        target_by_pdb = {item["pdb"]: item for item in target_plan}
        if len(target_by_pdb) != len(target_plan):
            raise SystemExit("Primary holdout unexpectedly contains duplicate PDB targets")
        for pdb, item in sorted(target_by_pdb.items()):
            staged[pdb] = copy_source_structure(
                item["source"], pdb, Path(staging), item["source_sha"])

        if structures_dir.exists() and any(structures_dir.iterdir()):
            raise SystemExit(f"{structures_dir} already holds raw structures; inspect the previous carve")
        holdout_dir.mkdir(parents=True, exist_ok=True)
        if quarantine_plan:
            quarantine_dir.mkdir(parents=True, exist_ok=True)
        structures_dir.mkdir(parents=True, exist_ok=True)

        moved, quarantined, structures = [], [], {}
        moved_graphs: list[tuple[Path, Path]] = []
        moved_sources: list[Path] = []
        try:
            for pdb, path in staged.items():
                destination = structures_dir / path.name
                shutil.move(str(path), str(destination))
                item = target_by_pdb[pdb]
                structures[pdb] = dict(
                    path=destination,
                    source_id=item["source_id"],
                    audited_source_sha256=item["source_sha"],
                )
                moved_sources.append(destination)

            for item in target_plan:
                path, row, pdb = item["path"], item["row"], item["pdb"]
                destination = holdout_dir / path.name
                shutil.move(str(path), str(destination))
                moved_graphs.append((destination, path))
                row["split"] = HOLDOUT_SPLIT
                row["path"] = f"graphs/{HOLDOUT_SPLIT}/{path.name}"
                row["sha256"] = sha256(destination)
                moved.append(dict(
                    pdb_id=pdb, path=row["path"], sha256=row["sha256"],
                    subset_source=row.get("subset_source", ""),
                    source_id=item["source_id"],
                    source_structure_sha256=item["source_sha"],
                    family_structure_cluster=row.get("family_structure_cluster", ""),
                ))

            for item in quarantine_plan:
                path, row, pdb = item["path"], item["row"], item["pdb"]
                destination = quarantine_dir / path.name
                shutil.move(str(path), str(destination))
                moved_graphs.append((destination, path))
                row["split"] = quarantine_split
                row["path"] = f"graphs/{quarantine_split}/{path.name}"
                row["sha256"] = sha256(destination)
                quarantined.append(dict(
                    pdb_id=pdb, path=row["path"], sha256=row["sha256"],
                    subset_source=row.get("subset_source", ""),
                    source_id=item["source_id"],
                    source_structure_sha256=item["source_sha"],
                    family_structure_cluster=row.get("family_structure_cluster", ""),
                    reason=item["quarantine_reason"],
                ))

            by_cluster: dict[str, list[str]] = defaultdict(list)
            for row in moved:
                by_cluster[row["family_structure_cluster"]].append(row["pdb_id"])
            payload = dict(
                schema="antigen_fold_holdout_v2",
                split=HOLDOUT_SPLIT,
                quarantine_split=quarantine_split,
                primary_source="snac_db",
                auxiliary_source="sabdab_vhh",
                fold=(folds[0] if len(folds) == 1 else folds),
                folds_held_out=folds,
                folds=training.SPLIT_FOLDS,
                internal_validation_fold=training.VALIDATION_FOLD,
                selection=(
                    "layered SNAC+SAbDab isolation components containing at least one SNAC primary graph "
                    "whose deterministic name-hash fold equals --fold; whole selected components leave "
                    "training, but only one deterministic SNAC graph per PDB is scored"
                ),
                criteria=dict(
                    vhh_full_chain_identity=training.VHH_IDENTITY_THRESHOLD,
                    cdr_h3_identity=training.CDR_H3_IDENTITY_THRESHOLD,
                    antigen_identity=training.ANTIGEN_IDENTITY_THRESHOLD,
                    antigen_min_length_coverage=training.ANTIGEN_MIN_LENGTH_COVERAGE,
                ),
                components_total=len(components),
                holdout_components=len(holdout_components),
                pinned_to_training=dict(
                    rule=f"component size >= {training.PIN_MIN_COMPONENT} and > 1/"
                         f"{training.SPLIT_FOLDS} of the pool (PROTOCOL_AMENDMENTS.md A11)",
                    pool=len(paths),
                    component_sizes=sorted((len(c) for c in pinned), reverse=True),
                    graphs=sum(len(c) for c in pinned),
                ),
                training_components_remaining=remaining,
                min_clusters=args.min_clusters,
                graphs=len(moved),
                removed_from_training_graphs=len(plan),
                quarantine_graphs=len(quarantined),
                structure_clusters=sorted(by_cluster),
                graph_dir=str(holdout_dir.resolve()),
                quarantine_dir=str(quarantine_dir.resolve()),
                source_structure_dir=str(structures_dir.resolve()),
                source_structures={
                    pdb: dict(
                        path=str(binding["path"].resolve()),
                        sha256=sha256(binding["path"]),
                        source_id=binding["source_id"],
                        audited_source_sha256=binding["audited_source_sha256"],
                    )
                    for pdb, binding in sorted(structures.items())
                },
                targets=moved,
                quarantined=quarantined,
                scope=(
                    "outcome-free: component membership, fold assignment, source role and same-PDB "
                    "tie-breaking are fixed before any training, benchmark or structural result exists"
                ),
            )
            write_outputs(manifest_path, manifest, args.out_json, payload)
        except Exception:
            for destination, original in reversed(moved_graphs):
                if destination.exists():
                    shutil.move(str(destination), str(original))
            for destination in moved_sources:
                destination.unlink(missing_ok=True)
            raise

    print(
        f"[carve_holdout_clusters] primary SNAC holdout {len(moved)} graph(s) in "
        f"{len(holdout_components)} independent component(s) at fold {args.fold}; "
        f"{len(quarantined)} auxiliary/duplicate graph(s) quarantined; "
        f"{remaining} component(s) remain for training"
    )
    return 0


if __name__ == "__main__":
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
