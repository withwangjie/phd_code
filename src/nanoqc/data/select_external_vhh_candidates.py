#!/usr/bin/env python3
"""Select external VHH-antigen complexes and count their independent clusters.

Input is the SAbDab summary table (e.g. the nanobody summary
``sabdab_nano_summary_all.tsv``; several files may be given). Each PDB entry
passes these gates in order:

A single-domain SAbDab export lists only single-domain chains, so a Fab in
the same entry leaves no trace in the metadata. Such chains are found in the
structure instead (step 2), and the entry is rejected as in the training
audit.

1. Metadata (same rules as the training ``sabdab_vhh`` subset):
   - released after ``--released-after``, and absent from ``--exclude-pdbs``
     (the study's audited PDB universe);
   - at least one protein or peptide antigen chain, no scFv;
   - at least one VHH chain and no VH/VL antibody chain in the entry;
   - numeric resolution <= the audit limit.
2. Structure: builds the biological assembly, annotates the CDRs, keeps the
   antigen chains within 7.5 A of the paratope, and applies the audit's
   interface quality gates and minimum interface size
   (``build_external_vhh_graphs.prepare_external_complex``). An Ig variable
   domain that SAbDab annotates neither as this entry's VHH nor as its
   antigen rejects the entry.
   Each graph is one VHH-antigen complex, as in the SNAC-DB per-VHH complexes
   behind the training and hard test sets. Entries with several nanobodies
   give one complex per PDB: the passing VHH chain with the largest interface
   (ties broken alphabetically). The other antibody chains are never antigen.
   This choice depends only on structure, never on any outcome.
3. Training overlap (with ``--training-dataset``): the same layered identities
   as the formal external audit (VHH full chain, CDR-H3 loop, antigen full
   chain with coverage) against every training graph. Optionally the
   family/structure cluster map as well.

The complexes that pass are grouped by the same layered homology rule among
themselves (and by shared structure cluster when a map is given). The number
of groups is compared with ``external_validation.external_vhh.min_clusters``.

Formal independence is not decided here: every formal run re-certifies it
with ``audit_external_vhh_independence.py`` against its own frozen training
set and cluster map. Run this selection before that, to learn whether an
external set can reach the required cluster count at all.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import nanoqc.data.audit_all_datasets as audit
import nanoqc.data.build_final_pyg_dataset as builder
from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.data.build_external_vhh_graphs import prepare_external_complex

RCSB_DOWNLOAD = "https://files.rcsb.org/download/{pdb}.cif"
_MISSING = {"", "na", "nan", "none"}
_DATE_FORMATS = ("%m/%d/%y", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%d/%m/%Y")


def _present(value: object) -> bool:
    return str(value or "").strip().lower() not in _MISSING


def parse_date(value: str) -> dt.date:
    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised SAbDab date {value!r}")


def read_sabdab(paths: Iterable[Path]) -> dict[str, list[dict]]:
    """SAbDab summary rows grouped by lower-case PDB ID (duplicates kept once)."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                pdb = str(row.get("pdb", "")).strip().lower()
                if len(pdb) != 4:
                    continue
                if row not in grouped[pdb]:
                    grouped[pdb].append(row)
    return grouped


def metadata_gate(pdb: str, rows: list[dict], *, released_after: dt.date, excluded: set[str],
                  max_resolution: float) -> dict:
    """Entry-level decision from SAbDab metadata alone."""
    reasons = []
    vhh = sorted({r["Hchain"].strip() for r in rows if _present(r.get("Hchain")) and not _present(r.get("Lchain"))})
    if any(_present(r.get("Lchain")) for r in rows):
        reasons.append("contains_vh_vl_antibody")
    if not vhh:
        reasons.append("no_vhh_chain")
    if any(str(r.get("scfv", "")).strip().lower() == "true" for r in rows):
        reasons.append("scfv")
    antigen_rows = [r for r in rows if _present(r.get("antigen_chain"))]
    if not antigen_rows:
        reasons.append("no_antigen")
    # SAbDab's antigen_type is the DEDUPLICATED SET of types over the antigen
    # chains, not one token per chain (e.g. five chains -> "ION|HAPTEN|PROTEIN").
    # The rule is therefore "at least one polypeptide antigen chain", as in
    # recent SAbDab-derived benchmarks, not "nothing but polypeptide": the
    # graph encodes only amino-acid residues, so ions, glycans and ligands are
    # ignored anyway, and SNAC-DB [R34] likewise curates the protein complex.
    # Requiring their absence would make the external set stricter than the
    # training set it validates.
    types = {t.strip().lower() for r in antigen_rows for t in str(r.get("antigen_type", "")).split("|")}
    if antigen_rows and not types & {"protein", "peptide"}:
        reasons.append("no_polypeptide_antigen")
    try:
        released = min(parse_date(r["date"]) for r in rows)
        if released <= released_after:
            reasons.append("released_before_cutoff")
    except (KeyError, ValueError):
        released = None
        reasons.append("unknown_release_date")
    if pdb in excluded:
        reasons.append("in_study_universe")
    try:
        resolution = float(rows[0].get("resolution", ""))
    except ValueError:
        resolution = None
    if resolution is None:
        reasons.append("unknown_resolution")
    elif resolution > max_resolution:
        reasons.append("resolution_above_limit")
    return dict(pdb_id=pdb, vhh_chains=vhh, vhh_chain="",
                sabdab_antigen_chains=sorted({c.strip() for r in antigen_rows
                                              for c in str(r["antigen_chain"]).split("|") if c.strip()}),
                antigen_name=str((antigen_rows or rows)[0].get("antigen_name", "")),
                method=str(rows[0].get("method", "")), resolution=resolution,
                release_date=(released.isoformat() if released else None), reasons=reasons)


def fetch_structure(pdb: str, directory: Path, download: bool) -> Optional[Path]:
    """The one raw file for ``pdb`` (same lookup as the formal audit), downloading if asked."""
    from nanoqc.data.audit_external_vhh_independence import source_structure_for_pdb
    if directory.is_dir():
        try:
            return source_structure_for_pdb(directory, pdb)
        except ValueError as exc:
            if "found 0" not in str(exc):
                raise
    if not download:
        return None
    import requests
    directory.mkdir(parents=True, exist_ok=True)
    response = requests.get(RCSB_DOWNLOAD.format(pdb=pdb.upper()), timeout=120)
    response.raise_for_status()
    path = directory / f"{pdb}.cif"
    path.write_bytes(response.content)
    return path


def choose_vhh(source: Path, vhh_chains: Sequence[str],
               annotated_antigen_chains: Sequence[str] = ()) -> tuple[Optional[dict], list[dict]]:
    """Prepare every annotated VHH chain; keep the passing one with the largest interface."""
    attempts, passing = [], []
    for chain in vhh_chains:
        others = [c for c in vhh_chains if c != chain]
        try:
            prepared = prepare_external_complex(source, chain, others, annotated_antigen_chains)
        except Exception as exc:
            attempts.append(dict(vhh_chain=chain, reasons=[f"structure_{type(exc).__name__}: {exc}"]))
            continue
        quality = prepared["quality"]
        reasons = list(quality["structure_quality_reasons"])
        if quality["interface_residues"] < builder.MIN_INTERFACE_RESIDUES:
            reasons.append("weak_interface")
        # Entry-level rule, as in the training audit: a complex that also holds
        # a VH/VL antibody is not a single-VHH entry, whatever the metadata says.
        if prepared["unannotated_antibody_chains"]:
            reasons.append("contains_unannotated_antibody_chain")
        attempts.append(dict(vhh_chain=chain, interface_residues=quality["interface_residues"], reasons=reasons))
        if not reasons:
            passing.append((-quality["interface_residues"], chain, dict(prepared, vhh_chain=chain)))
    if not passing:
        return None, attempts
    return min(passing, key=lambda item: item[:2])[2], attempts


def cdr3_method_agreement(train: list[dict]) -> dict:
    """How often this module's CDR-H3 equals the training (SNAC IMGT) annotation."""
    from nanoqc.data.build_external_vhh_graphs import annotate_cdrs
    compared = agreed = 0
    methods = set()
    for record in train:
        if record["subset_source"] != "snac_db" or not record["cdr_h3"] or len(record["vhh"]) != 1:
            continue
        if record["cdr_h3"] not in record["vhh"][0]:
            continue  # annotation includes residues unmodelled in the structure
        compared += 1
        try:
            cdrs = annotate_cdrs(record["vhh"][0])
        except ValueError:
            continue
        methods.add(cdrs["method"])
        agreed += cdrs["cdr3"] == record["cdr_h3"]
    return dict(compared=compared, agreed=agreed, fraction=(agreed / compared if compared else None),
                methods=sorted(methods))


def layered_homologous(a: dict, b: dict, *, vhh: float, cdr: float, antigen: float, coverage: float) -> bool:
    """The training split's layered rule between two external complexes."""
    if builder.global_identity(a["vhh_sequence"], b["vhh_sequence"]) >= vhh:
        return True
    if builder.cdr_h3_loop_identity(a["cdr3"], b["cdr3"]) >= cdr:
        return True
    return builder.side_identity(a["antigen_sequences"], b["antigen_sequences"],
                                 min_length_coverage=coverage) >= antigen


def independent_groups(rows: list[dict], cluster_map: dict[str, str], **thresholds) -> dict[str, int]:
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            same_cluster = (rows[i]["pdb_id"] in cluster_map and rows[j]["pdb_id"] in cluster_map
                            and cluster_map[rows[i]["pdb_id"]] == cluster_map[rows[j]["pdb_id"]])
            if same_cluster or layered_homologous(rows[i], rows[j], **thresholds):
                parent[find(j)] = find(i)
    roots = sorted({find(i) for i in range(len(rows))})
    label = {root: f"external_group_{k + 1:03d}" for k, root in enumerate(roots)}
    return {rows[i]["pdb_id"]: label[find(i)] for i in range(len(rows))}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sabdab-summary", type=Path, nargs="+", required=True)
    parser.add_argument("--released-after", type=dt.date.fromisoformat, required=True,
                        help="YYYY-MM-DD; must postdate the training data snapshot")
    parser.add_argument("--structures-dir", type=Path, required=True)
    parser.add_argument("--download", action="store_true", help="Fetch missing mmCIF files from RCSB")
    parser.add_argument("--exclude-pdbs", type=Path, nargs="*", default=[],
                        help="Files with one PDB ID per line (e.g. the audited study universe)")
    parser.add_argument("--training-dataset", type=Path, default=None,
                        help="Dataset directory with graph_manifest.json and graphs/train")
    parser.add_argument("--cluster-map", type=Path, default=None)
    parser.add_argument("--min-clusters", type=int, default=10)
    parser.add_argument("--max-resolution", type=float, default=audit.MAX_RESOLUTION_ANGSTROM)
    parser.add_argument("--vhh-threshold", type=float, default=builder.VHH_IDENTITY_THRESHOLD)
    parser.add_argument("--cdr-h3-threshold", type=float, default=builder.CDR_H3_IDENTITY_THRESHOLD)
    parser.add_argument("--antigen-threshold", type=float, default=builder.ANTIGEN_IDENTITY_THRESHOLD)
    parser.add_argument("--antigen-min-length-coverage", type=float, default=builder.ANTIGEN_MIN_LENGTH_COVERAGE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    excluded = {line.strip().lower()[:4] for path in args.exclude_pdbs
                for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    cluster_map = ({str(k).lower(): str(v) for k, v in json.loads(args.cluster_map.read_text(encoding="utf-8")).items()}
                   if args.cluster_map else {})
    train = []
    if args.training_dataset is not None:
        from nanoqc.data.audit_external_vhh_independence import load_training_sequences
        train = load_training_sequences(args.training_dataset)
    train_clusters = {cluster_map[t["pdb_id"]] for t in train if t["pdb_id"] in cluster_map}
    cdr_agreement = cdr3_method_agreement(train) if train else None

    candidates = []
    for pdb, rows in sorted(read_sabdab(args.sabdab_summary).items()):
        entry = metadata_gate(pdb, rows, released_after=args.released_after, excluded=excluded,
                              max_resolution=args.max_resolution)
        candidates.append(entry)
        if entry["reasons"]:
            continue
        try:
            source = fetch_structure(pdb, args.structures_dir, args.download)
            if source is None:
                entry["reasons"].append("raw_structure_missing")
                continue
            entry.update(source_structure=str(source.resolve()), source_structure_sha256=sha256(source))
            prepared, attempts = choose_vhh(source, entry["vhh_chains"], entry["sabdab_antigen_chains"])
        except Exception as exc:
            entry["reasons"].append(f"structure_{type(exc).__name__}: {exc}")
            continue
        entry["vhh_attempts"] = attempts
        if prepared is None:
            entry["reasons"].append("no_vhh_chain_passes_structure_gates")
            continue
        quality = prepared["quality"]
        entry.update(vhh_chain=prepared["vhh_chain"],
                     other_antibody_chains=[c for c in entry["vhh_chains"] if c != prepared["vhh_chain"]],
                     vhh_sequence=prepared["vhh_sequence"], antigen_sequences=prepared["antigen_sequences"],
                     antigen_chains=prepared["antigen_chains"], cdr3=prepared["cdrs"]["cdr3"],
                     cdr_annotation_method=prepared["cdrs"]["method"],
                     antigen_contact_basis=prepared["meta"]["antigen_contact_basis"],
                     unannotated_antibody_chains=prepared["unannotated_antibody_chains"],
                     interface_residues=quality["interface_residues"], quality=quality)
        if train:
            from nanoqc.data.audit_external_vhh_independence import max_training_identities
            overlap = max_training_identities(
                dict(vhh=[entry["vhh_sequence"]], antigen=entry["antigen_sequences"], cdr_h3=entry["cdr3"]),
                train, args.antigen_min_length_coverage)
            entry["training_overlap"] = overlap
            if overlap["max_vhh_full_chain_identity"] >= args.vhh_threshold:
                entry["reasons"].append("training_vhh_identity")
            if overlap["max_cdr_h3_loop_identity"] >= args.cdr_h3_threshold:
                entry["reasons"].append("training_cdr_h3_identity")
            if (overlap["max_antigen_full_chain_identity"] >= args.antigen_threshold
                    and overlap["antigen_length_coverage"] >= args.antigen_min_length_coverage):
                entry["reasons"].append("training_antigen_identity")
        if cluster_map:
            if pdb not in cluster_map:
                entry["reasons"].append("absent_from_cluster_map")
            elif cluster_map[pdb] in train_clusters:
                entry["reasons"].append("training_structure_cluster")

    for entry in candidates:
        entry["selected"] = not entry["reasons"]
    selected = [c for c in candidates if c["selected"]]
    groups = independent_groups(selected, cluster_map, vhh=args.vhh_threshold, cdr=args.cdr_h3_threshold,
                                antigen=args.antigen_threshold, coverage=args.antigen_min_length_coverage)
    for entry in selected:
        entry["external_group"] = groups[entry["pdb_id"]]
    # One representative per group, as for the hard test set: layered-homologous
    # complexes (e.g. one VHH on two antigens) are not independent even when
    # Foldseek puts their antigens in different clusters. Structure-only choice.
    representatives = {}
    for entry in sorted(selected, key=lambda e: (-e["interface_residues"], e["pdb_id"])):
        representatives.setdefault(entry["external_group"], entry["pdb_id"])
    for entry in candidates:
        entry["representative"] = entry["pdb_id"] in set(representatives.values())
    n_groups = len(set(groups.values()))
    reason_counts: dict[str, int] = defaultdict(int)
    for entry in candidates:
        for reason in entry["reasons"]:
            reason_counts[reason.split(":")[0]] += 1

    payload = dict(
        schema="external_vhh_candidates_v1", released_after=args.released_after.isoformat(),
        sabdab_summaries={str(p): sha256(p) for p in args.sabdab_summary},
        excluded_pdb_count=len(excluded), training_checked=bool(train), training_graphs=len(train),
        cluster_map_sha256=(sha256(args.cluster_map) if args.cluster_map else None),
        homology_isolation=dict(vhh_full_chain_identity=args.vhh_threshold, cdr_h3_identity=args.cdr_h3_threshold,
                                antigen_identity=args.antigen_threshold,
                                antigen_min_length_coverage=args.antigen_min_length_coverage),
        entries=len(candidates), selected=len(selected), independent_groups=n_groups,
        min_clusters=args.min_clusters, adequate=n_groups >= args.min_clusters,
        cdr3_agreement_with_training=cdr_agreement,
        rejection_counts=dict(sorted(reason_counts.items())), candidates=candidates,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "candidates.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                                   encoding="utf-8")
    fields = ["pdb_id", "selected", "external_group", "representative", "vhh_chain", "antigen_chains", "antigen_name",
              "release_date", "method", "resolution", "cdr3", "cdr_annotation_method", "interface_residues", "reasons"]
    with (args.out_dir / "candidates.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for entry in candidates:
            writer.writerow({**entry, "antigen_chains": ",".join(entry.get("antigen_chains", [])),
                             "reasons": ";".join(entry["reasons"])})
    (args.out_dir / "selected_pdbs.txt").write_text("".join(f"{c['pdb_id']}\n" for c in selected), encoding="utf-8")
    lines = [
        "# External VHH candidate selection", "",
        f"SAbDab entries: {len(candidates)}; released after {args.released_after}; selected: {len(selected)}; "
        f"independent groups: {n_groups} (required {args.min_clusters}: "
        f"{'adequate' if payload['adequate'] else 'NOT adequate'}).", "",
        f"Training overlap checked: {'yes, ' + str(len(train)) + ' training graphs' if train else 'no (pass --training-dataset)'}; "
        f"structure clusters: {'yes' if cluster_map else 'no (pass --cluster-map after Foldseek)'}.", "",
        ("CDR-H3 agreement with the training SNAC IMGT annotation: "
         + (f"{cdr_agreement['agreed']}/{cdr_agreement['compared']} "
            f"({', '.join(cdr_agreement['methods']) or 'n/a'})" if cdr_agreement else "not checked")), "",
        "## Rejections", "", "| Reason | Entries |", "|---|---:|",
        *[f"| {k} | {v} |" for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1])], "",
        "## Selected", "", "| PDB | Group | Representative | VHH | Antigen chains | Antigen | CDR-H3 | Interface residues |",
        "|---|---|---|---|---|---|---|---:|",
        *[f"| {c['pdb_id']} | {c['external_group']} | {'yes' if c['representative'] else ''} | {c['vhh_chain']} | "
          f"{','.join(c['antigen_chains'])} | "
          f"{c['antigen_name']} | {c['cdr3']} | {c['interface_residues']} |" for c in selected], "",
        "Formal independence is re-certified in every run by audit_external_vhh_independence.py.",
    ]
    (args.out_dir / "selection_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"selected {len(selected)} of {len(candidates)}; independent groups {n_groups} "
          f"(required {args.min_clusters})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
