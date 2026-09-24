#!/usr/bin/env python3
"""Build the frozen Foldseek pair table from antigen chains only.

Structure clusters describe antigen families. Every VHH, VH/VL and TCR
variable domain has the same immunoglobulin fold, so antibody chains would
link almost every complex into one single-linkage component. This module
therefore keeps, for each PDB of the clustering universe, only its
non-antibody protein chains, runs an all-versus-all Foldseek search and writes
``query<TAB>target<TAB>mintmscore`` with the header ``build_independence_cluster_map.py``
requires.

Scores are symmetric (PROTOCOL_AMENDMENTS.md A11). Foldseek's ``qtmscore`` is
normalized by the query chain alone, so a short chain (a ~57-residue helix
such as a G-protein gamma subunit) scores high against any protein holding a
similar helix; under single linkage such chains joined 80% of the study into
one component. A chain pair therefore counts only through
``min(qtmscore(q->t), qtmscore(t->q))``, the TM-score normalized by the longer
chain, which requires the alignment to cover both chains: the structural
counterpart of the antigen sequence rule's length coverage. A PDB pair takes
the maximum of that symmetric score over its chain pairs.

A chain annotated as antigen (SNAC ``Chain_Ag``, SAbDab antigen chains of
external entries) is always kept. Otherwise antibody chains are recognised
by, in order:
  1. SNAC/SAbDab annotation: the chain sequence matches an annotated VH, VL,
     VHH or TCR sequence of that PDB (the audit's rule: observed >= 70
     residues, >= 70% of the annotated length, subsequence);
  2. SAbDab chain IDs for external candidates;
  3. an immunoglobulin variable-domain detector: ANARCI when installed,
     otherwise the conserved V-domain anchors (Cys ~IMGT 23, Trp ~41,
     Cys ~104 and the J-region [WF]-G-x-G of FR4).
The method is recorded per chain.

Chains shorter than ``--min-chain-length`` residues are not searched, because
peptides have no structure family. A PDB with no remaining antigen chain gets
one explicit self row (TM-score 1.0). Its structure has no antigen-family
relation, so it can only be a singleton, and it is listed under ``self_only``
in the manifest. A PDB whose antigen chains Foldseek returns no hit for makes
the build fail unless ``--allow-missing-hits`` is given. Sequence-level
isolation still applies to all PDBs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import gemmi

import nanoqc.data.audit_all_datasets as audit
from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.data.build_independence_cluster_map import norm_id

MIN_CHAIN_LENGTH = 20
# Exhaustive all-versus-all; E <= 10 bounds the output (-e inf would write
# every pair, N^2 rows). Pairs with TM-score >= 0.5 are expected to have
# E <= 10; override with --foldseek-arg when needed.
FOLDSEEK_ARGS = ("--exhaustive-search", "1", "-e", "10")
# Symmetric chain-pair TM-score, normalized by the longer chain (A11).
SCORE_FIELD = "mintmscore"
_FR4 = re.compile(r"[WF]G.G")


def chain_sequences(structure) -> list[tuple[str, str]]:
    """(chain name, one-letter sequence) of every protein chain in model 1."""
    out = []
    for chain in structure[0]:
        sequence = "".join(gemmi.find_tabulated_residue(r.name).one_letter_code.upper()
                           for r in chain if gemmi.find_tabulated_residue(r.name).is_amino_acid())
        if sequence:
            out.append((chain.name, sequence))
    return out


def ig_variable_domain_by_motif(sequence: str) -> bool:
    """Conserved V-domain anchors: C(23) .. W(41) .. C(104) .. [WF]G.G (FR4)."""
    for first in (i for i, aa in enumerate(sequence) if aa == "C"):
        if "W" not in sequence[first + 10:first + 21]:
            continue
        for second in (j for j in range(first + 60, min(first + 86, len(sequence))) if sequence[j] == "C"):
            if _FR4.search(sequence[second + 4:second + 44]):
                return True
    return False


def ig_variable_domain(sequence: str) -> tuple[bool, str]:
    try:
        from anarci import anarci  # optional dependency
    except ImportError:
        return ig_variable_domain_by_motif(sequence), "v_domain_motif"
    numbering, _, _ = anarci([("chain", sequence)], scheme="imgt", output=False)
    return bool(numbering and numbering[0]), "anarci"


def annotated_antibody(pdb: str, sequence: str) -> bool:
    """The audit's annotation match (nano_features): observed chain within an annotated Ig sequence."""
    return any(len(sequence) >= 70 and len(sequence) >= .7 * len(record["sequence"])
               and audit.is_subsequence(sequence, record["sequence"])
               for record in audit.PDB_ANNOTATIONS.get(pdb.upper(), []))


def annotated_antigen_chains(source: dict) -> set[str]:
    """SNAC ``Chain_Ag`` of a curated complex file, found the way the audit finds it."""
    name = Path(source.get("member") or source["path"]).stem
    parent = Path(source["path"]).parent
    row = audit.ANNOTATIONS.get((str(parent), name))
    if row is None and not source.get("member"):
        row = audit.ANNOTATIONS.get((str(parent.parent), name))
    return set(audit.literal(row.get("Chain_Ag"), [])) if row else set()


def audit_sources(audit_jsonl: Path) -> dict[str, list[dict]]:
    """Readable audited structure files per four-character PDB ID (DB5.5: bound files only).

    IDs whose every file failed the audit's parse map to an empty list.
    """
    sources: dict[str, list[dict]] = defaultdict(list)
    for line in audit_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pdb = str(row.get("pdb_id", "")).strip().lower()
        if not re.fullmatch(r"[a-z0-9]{4}", pdb):
            continue
        name = Path(row.get("member") or row["path"]).name.lower()
        if row.get("subset") == "test_db55" and "_b." not in name:
            continue
        sources.setdefault(pdb, [])
        if not row.get("valid", True):
            continue
        sources[pdb].append(dict(path=row["path"], member=row.get("member", ""), subset=row.get("subset", ""),
                                 id=row.get("id", "")))
    return sources


def antigen_structure(pdb: str, sources: Iterable[dict], sabdab_antibody_chains: Iterable[str],
                      min_length: int, sabdab_antigen_chains: Iterable[str] = ()) -> tuple[Optional[gemmi.Structure], dict]:
    """One structure holding the PDB's non-antibody chains, and the per-chain record."""
    sabdab = set(sabdab_antibody_chains)
    model = gemmi.Model("1")
    kept_sequences: dict[str, str] = {}
    chains = []
    unreadable = []
    for source in sources:
        try:
            structure, _ = audit._read_raw_structure(dict(source))
        except Exception as exc:
            unreadable.append(dict(source=source.get("id") or source["path"], error=f"{type(exc).__name__}: {exc}"))
            continue
        antigen_annotated = annotated_antigen_chains(source) | set(sabdab_antigen_chains)
        while len(structure) > 1:
            del structure[1]
        structure.setup_entities()
        sequences = dict(chain_sequences(structure))
        for chain in structure[0]:
            sequence = sequences.get(chain.name, "")
            if not sequence:
                continue
            record = dict(source=source.get("id") or source["path"], chain=chain.name, length=len(sequence))
            if chain.name in antigen_annotated:
                record.update(role="antigen", method="antigen_annotation")
            elif annotated_antibody(pdb, sequence):
                record.update(role="antibody", method="annotation")
            elif chain.name in sabdab:
                record.update(role="antibody", method="sabdab_chain_id")
            else:
                is_ig, method = ig_variable_domain(sequence)
                record.update(role="antibody" if is_ig else "antigen", method=method)
            if record["role"] == "antigen" and len(sequence) < min_length:
                record.update(role="skipped", method=f"shorter_than_{min_length}")
            if record["role"] == "antigen":
                if kept_sequences.get(chain.name) == sequence:
                    record.update(role="duplicate", method="same_chain_other_file")
                else:
                    # Foldseek appends the chain name to the entry; keep it parseable (<=4 alnum).
                    base = chain.name if re.fullmatch(r"[A-Za-z0-9]{1,3}", chain.name) else "X"
                    name, suffix = base, 2
                    while name in kept_sequences or not re.fullmatch(r"[A-Za-z0-9]{1,4}", name):
                        name = f"{base}{suffix}"
                        suffix += 1
                    copy = chain.clone()
                    copy.name = name
                    model.add_chain(copy)
                    kept_sequences[name] = sequence
                    record["written_as"] = name
            chains.append(record)
    if not len(model):
        return None, dict(chains=chains, unreadable=unreadable)
    out = gemmi.Structure()
    out.add_model(model)
    out.setup_entities()
    return out, dict(chains=chains, unreadable=unreadable)


def symmetric_pdb_scores(raw: Path) -> tuple[dict[tuple[str, str], float], set[str]]:
    """PDB-pair scores from chain-level Foldseek rows, and every PDB with any hit.

    A chain pair scores ``min`` of its two directed qtmscores; a pair seen in
    one direction only scores 0 (its reverse alignment fell below the search
    E-value). A PDB pair keeps the maximum over its chain pairs, in both orders.
    """
    directed: dict[tuple[str, str], float] = {}
    for line in Path(raw).read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) < 3 or not line.strip():
            continue
        key = (fields[0].strip(), fields[1].strip())
        directed[key] = max(float(fields[2]), directed.get(key, 0.0))
    best: dict[tuple[str, str], float] = {}
    seen: set[str] = set()
    for (query, target), score in directed.items():
        pair = (norm_id(query), norm_id(target))
        seen.update(pair)
        reverse = directed.get((target, query))
        if reverse is None:
            continue
        best[pair] = max(min(score, reverse), best.get(pair, 0.0))
    return best, seen


def resolve_foldseek(value: str) -> str:
    """The Foldseek executable, or a message naming what was found instead."""
    located = shutil.which(value)
    if located:
        return located
    path = Path(value)
    if path.is_dir():
        for candidate in (path / "bin" / "foldseek", path / "foldseek"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate.resolve())
        raise SystemExit(f"--foldseek {value} is a directory; pass the executable, e.g. {path / 'bin' / 'foldseek'}")
    if not path.exists():
        raise SystemExit(f"--foldseek {value} not found; pass the executable's path or put it on PATH")
    if not os.access(path, os.X_OK):
        raise SystemExit(f"--foldseek {value} is not executable; chmod +x it")
    return str(path.resolve())


def run_foldseek(foldseek: str, input_dir: Path, work: Path, entries: int, threads: int,
                 extra: Sequence[str]) -> Path:
    raw = work / "foldseek_raw.m8"
    tmp = work / "foldseek_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    command = [foldseek, "easy-search", str(input_dir), str(input_dir), str(raw), str(tmp),
               *extra, "--max-seqs", str(max(entries, 1)), "--threads", str(threads),
               "--format-output", "query,target,qtmscore"]
    print("[build_foldseek_pairs] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return raw


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--universe", type=Path, required=True, help="One PDB ID per line (foldseek_universe.txt)")
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory with data_audit_details.jsonl")
    parser.add_argument("--data-root", type=Path, required=True, help="Raw data root (SNAC/SAbDab annotations)")
    parser.add_argument("--external-candidates", type=Path, default=None,
                        help="candidates.json from select_external_vhh_candidates.py")
    parser.add_argument("--external-structures", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True, help="Pair table to write (pair_tsv)")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--foldseek", default="foldseek")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--foldseek-arg", action="append", default=None,
                        help=f"Replaces the default search options {' '.join(FOLDSEEK_ARGS)} (repeatable)")
    parser.add_argument("--min-chain-length", type=int, default=MIN_CHAIN_LENGTH)
    parser.add_argument("--allow-missing-hits", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing pair table")
    parser.add_argument("--reuse-raw", type=Path, default=None,
                        help="Score an existing foldseek_raw.m8 of the same antigen chains instead of searching")
    args = parser.parse_args(argv)
    # Fail before building thousands of antigen structures, not after.
    args.foldseek = resolve_foldseek(args.foldseek)
    if args.reuse_raw is not None and not args.reuse_raw.is_file():
        raise SystemExit(f"--reuse-raw {args.reuse_raw} not found")

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; the pair table is frozen once used (pass --force to rebuild)")
    universe = sorted({line.strip().lower() for line in args.universe.read_text(encoding="utf-8").splitlines()
                       if line.strip()})
    audit.load_annotations(args.data_root.resolve())
    sources = audit_sources(args.audit_dir / "data_audit_details.jsonl")
    external: dict[str, dict] = {}
    if args.external_candidates is not None:
        payload = json.loads(args.external_candidates.read_text(encoding="utf-8"))
        external = {c["pdb_id"]: c for c in payload["candidates"] if c.get("selected")}
    if external and args.external_structures is None:
        raise SystemExit("--external-structures is required with --external-candidates")

    from nanoqc.data.audit_external_vhh_independence import source_structure_for_pdb
    input_dir = args.work_dir / "antigen_structures"
    if input_dir.exists():
        shutil.rmtree(input_dir)
    input_dir.mkdir(parents=True)
    records, self_only, entries = {}, [], 0
    for pdb in universe:
        pdb_sources = list(sources.get(pdb, []))
        antibody_chains: list[str] = []
        antigen_chains: list[str] = []
        if pdb in external:
            path = source_structure_for_pdb(args.external_structures, pdb)
            pdb_sources = [dict(path=str(path), member="", subset="external_vhh", id=str(path))]
            antibody_chains = list(external[pdb].get("vhh_chains") or [external[pdb]["vhh_chain"]])
            antigen_chains = list(external[pdb].get("sabdab_antigen_chains") or [])
        if pdb not in sources and pdb not in external:
            raise SystemExit(f"No audited or external structure for universe PDB {pdb}")
        structure, record = antigen_structure(pdb, pdb_sources, antibody_chains, args.min_chain_length,
                                              antigen_chains)
        if structure is None:
            self_only.append(pdb)
            # Unparseable files never become graphs; they are in the universe only for coverage.
            record["self_only_reason"] = ("no_readable_structure" if not record["chains"]
                                          else "no_antigen_chain_for_structure_search")
        else:
            structure.make_mmcif_document().write_file(str(input_dir / f"{pdb}.cif"))
            entries += len(structure[0])
        records[pdb] = record
    if entries == 0:
        raise SystemExit("No antigen chains to search")

    if args.reuse_raw is not None:
        # The chains written above are the searched input; only the scoring rule changed.
        raw = args.reuse_raw
        print(f"[build_foldseek_pairs] reusing the Foldseek search {raw} (no new search)", flush=True)
    else:
        raw = run_foldseek(args.foldseek, input_dir, args.work_dir, entries, args.threads,
                           args.foldseek_arg or FOLDSEEK_ARGS)
    best, seen = symmetric_pdb_scores(raw)
    unknown = sorted(seen - set(universe))
    if unknown:
        raise SystemExit(f"Foldseek returned entries outside the universe: {unknown[:10]}")
    searched = [pdb for pdb in universe if pdb not in self_only]
    stray = sorted(seen - set(searched))
    if stray:
        raise SystemExit(f"Foldseek output has hits for PDBs with no searched antigen chain: {stray[:10]} "
                         "(a --reuse-raw file from different inputs?)")
    missing = sorted(set(searched) - seen)
    if missing and not args.allow_missing_hits:
        raise SystemExit(f"Foldseek returned no hit (not even self) for {len(missing)} PDB(s): {missing[:20]}; "
                         "inspect them or pass --allow-missing-hits")
    for pdb in missing:
        records[pdb]["self_only_reason"] = "no_foldseek_hit"
    for pdb in self_only + missing:
        best[(pdb, pdb)] = 1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write(f"query\ttarget\t{SCORE_FIELD}\n")
        for (query, target), score in sorted(best.items()):
            handle.write(f"{query}\t{target}\t{score:.4f}\n")
    version = subprocess.run([args.foldseek, "version"], capture_output=True, text=True, check=False).stdout.strip()
    manifest = dict(
        schema="foldseek_pairs_v1", pair_table=str(args.out.resolve()), pair_table_sha256=sha256(args.out),
        raw_output_sha256=sha256(raw), foldseek_version=version or None,
        foldseek_options=list(args.foldseek_arg or FOLDSEEK_ARGS), score=SCORE_FIELD,
        score_rule="PDB pair: max over chain pairs of min(qtmscore(q->t), qtmscore(t->q))",
        raw_output=str(Path(raw).resolve()), reused_raw_output=args.reuse_raw is not None,
        min_chain_length=args.min_chain_length, universe_size=len(universe),
        universe_sha256=sha256(args.universe), searched_pdbs=len(searched), chains_searched=entries,
        self_only=sorted(self_only + missing), allow_missing_hits=bool(args.allow_missing_hits),
        external_candidates_sha256=(sha256(args.external_candidates) if args.external_candidates else None),
        per_pdb=records,
    )
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[build_foldseek_pairs] {len(best)} pair rows over {len(universe)} PDBs "
          f"({len(self_only)} without antigen chain, {len(missing)} without hit) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
