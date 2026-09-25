#!/usr/bin/env python3
"""Fetch each PDB entry's resolution from RCSB into a declared metadata table.

Some structure files carry no resolution record. RCSB's downloaded biological
assembly files are one case: they hold ``_exptl.method`` but no ``_refine``
block, and their ``_exptl.entry_id`` is a placeholder, so the audit's
structure-quality gate would reject the whole subset for a missing field
rather than for quality (PROTOCOL_AMENDMENTS.md A15).

This writes ``pdb<TAB>resolution<TAB>method`` for the requested entries, which
``audit_all_datasets.load_annotations`` reads by PDB ID exactly as it reads the
SNAC and SAbDab tables, recording the file and its SHA-256 in the audit report.
The value is the deposition's own ``rcsb_entry_info.resolution_combined``; an
entry with none (NMR, some cryo-EM) is written with an empty resolution and
stays excluded by the gate.

Run it once and keep the table under the data root: the audit then never needs
the network, and the values are fixed by the recorded hash rather than by
whatever RCSB serves later. ``--resume`` continues an interrupted fetch.

With no arguments it does the whole job for this repository: it finds the most
recent run's audit, takes every valid structure that audit could not date
(whatever the subset), and writes the data root's ``entry_resolution.tsv``::

    python -m nanoqc.data.fetch_entry_resolution

``--missing-from-audit`` names a different audit, and ``--ids`` / ``--from-dir``
fetch a named set instead (an ID list, or the PDB IDs in structure file names).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

try:
    from nanoqc.common.repo_io import REPO_ROOT
except ModuleNotFoundError:  # run as a plain file: python src/nanoqc/data/fetch_entry_resolution.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from nanoqc.common.repo_io import REPO_ROOT

ENDPOINT = "https://data.rcsb.org/graphql"
# Verified response shape:
# {"data":{"entries":[{"rcsb_id":"10BT","rcsb_entry_info":{"resolution_combined":[1.99]},
#                      "exptl":[{"method":"X-RAY DIFFRACTION"}]}]}}
QUERY = "{entries(entry_ids:[%s]){rcsb_id rcsb_entry_info{resolution_combined} exptl{method}}}"
CHUNK = 150
USER_AGENT = "nanoqc-fetch-entry-resolution/1 (+https://data.rcsb.org)"
OUT_SUFFIX = "entry_resolution.tsv"  # any *entry_resolution.tsv under the data root is read
FIELDS = ("pdb", "resolution", "method")
_PDB_ID = re.compile(r"^(?:pdb_0000|pdb)?([0-9][A-Za-z0-9]{3})(?=$|[_.-])", re.IGNORECASE)


def data_root() -> Path:
    """The configured raw data root (``QP_DATA_ROOT``), else ``<repo>/data``."""
    return Path(os.environ.get("QP_DATA_ROOT") or (REPO_ROOT / "data"))


def latest_audit_dir(runs_dir: Optional[Path] = None) -> Optional[Path]:
    """The audit whose details file was most recently updated."""
    runs = runs_dir if runs_dir is not None else REPO_ROOT / "runs"
    if not runs.is_dir():
        return None
    candidates = [d / "audit" for d in runs.iterdir() if d.is_dir()]
    candidates = [a for a in candidates if (a / "data_audit_details.jsonl").is_file()]
    return max(candidates, key=lambda a: (
        (a / "data_audit_details.jsonl").stat().st_mtime_ns, str(a)
    )) if candidates else None


def entry_ids(values: Iterable[str]) -> list[str]:
    """Distinct four-character PDB IDs, upper case, from lines or file names."""
    seen: dict[str, None] = {}
    for value in values:
        # Match the filename/ID boundary. An unanchored search reads the
        # zero-padding of SAbDab2 names (pdb_000010zo.cif) as PDB ID 0000.
        match = _PDB_ID.match(Path(str(value).strip()).name)
        if match:
            seen.setdefault(match.group(1).upper(), None)
    return list(seen)


def resolution_of(entry: dict) -> Optional[float]:
    """The entry's worst reported resolution, or None when it reports none.

    ``resolution_combined`` may list one value per method; the gate is an
    upper bound, so the largest is the conservative reading (A12).
    """
    info = entry.get("rcsb_entry_info") or {}
    values = info.get("resolution_combined") or []
    numbers = [float(v) for v in values
               if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0]
    return max(numbers) if numbers else None


def methods_of(entry: dict) -> str:
    return ";".join(sorted({str(m.get("method", "")).strip()
                            for m in (entry.get("exptl") or []) if m.get("method")}))


def _post(body: bytes, timeout: float) -> dict:
    request = urllib.request.Request(ENDPOINT, data=body, headers={
        "Content-Type": "application/json", "Accept": "application/json", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as handle:
        return json.loads(handle.read().decode("utf-8"))


def fetch_chunk(ids: Sequence[str], timeout: float, retries: int) -> list[dict]:
    """The ``entries`` list for ``ids``; retries transient failures, fails loudly otherwise."""
    body = json.dumps({"query": QUERY % ",".join(f'"{i}"' for i in ids)}).encode("utf-8")
    last: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            payload = _post(body, timeout)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
            continue
        if payload.get("errors"):
            raise SystemExit(f"RCSB rejected the query: {json.dumps(payload['errors'][:2])[:400]}")
        entries = (payload.get("data") or {}).get("entries")
        if entries is None:
            raise SystemExit(f"Unexpected RCSB response (no data.entries): {json.dumps(payload)[:400]}")
        return [e for e in entries if e]
    raise SystemExit(f"RCSB request failed after {retries + 1} attempt(s): {type(last).__name__}: {last}")


def read_table(path: Path) -> dict[str, tuple[str, str]]:
    """Rows already fetched, for --resume; an unreadable or foreign file is ignored."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or not {"pdb", "resolution"} <= set(reader.fieldnames):
            return {}
        return {str(row["pdb"]).strip().upper(): (str(row.get("resolution") or "").strip(),
                                                  str(row.get("method") or "").strip())
                for row in reader if str(row.get("pdb") or "").strip()}


def write_table(path: Path, rows: dict[str, tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(FIELDS) + "\n")
        for pdb in sorted(rows):
            resolution, method = rows[pdb]
            handle.write(f"{pdb}\t{resolution}\t{method}\n")
    tmp.replace(path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--missing-from-audit", type=Path, metavar="AUDIT_DIR",
                        help="Take the PDB IDs of every valid audited structure with no resolution from "
                             "AUDIT_DIR/data_audit_details.jsonl (one table for every subset that needs one)")
    parser.add_argument("--ids", type=Path, help="One PDB ID per line (e.g. non_redundant_pdb_ids.txt)")
    parser.add_argument("--from-dir", type=Path, help="Take IDs from the structure file names in this directory")
    parser.add_argument("--subset", action="append", default=None, metavar="NAME",
                        help="With --missing-from-audit, restrict to these audited subsets (repeatable)")
    parser.add_argument("--out", type=Path, default=None,
                        help=f"Table to write; the name must end with '{OUT_SUFFIX}', which is what the audit "
                             f"reads. Default: <data root>/{OUT_SUFFIX}")
    parser.add_argument("--resume", action="store_true", help="Keep rows already in --out and fetch only the rest")
    parser.add_argument("--chunk", type=int, default=CHUNK, help="Entries per request")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    if not (args.missing_from_audit or args.ids or args.from_dir):
        # No source named: do the whole job for this repository.
        args.missing_from_audit = latest_audit_dir()
        if args.missing_from_audit is None:
            parser.error("no finished audit under runs/; run the audit first, or pass --ids/--from-dir")
        print(f"[fetch_entry_resolution] using the most recent audit {args.missing_from_audit}", flush=True)
    if args.subset and not args.missing_from_audit:
        parser.error("--subset needs --missing-from-audit")
    if args.out is None:
        args.out = data_root() / OUT_SUFFIX
        print(f"[fetch_entry_resolution] writing {args.out}", flush=True)
    if not args.out.name.endswith(OUT_SUFFIX):
        parser.error(f"--out must end with '{OUT_SUFFIX}'; that suffix is what the audit reads")
    if args.chunk < 1:
        parser.error("--chunk must be positive")

    sources: list[str] = []
    if args.missing_from_audit:
        details = args.missing_from_audit / "data_audit_details.jsonl"
        if not details.is_file():
            raise SystemExit(f"{details} not found; run the audit first")
        wanted = set(args.subset or ())
        seen_subsets: dict[str, int] = {}
        for line in details.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("valid") or row.get("resolution_angstrom"):
                continue
            subset = str(row.get("subset", ""))
            if wanted and subset not in wanted:
                continue
            pdb = str(row.get("pdb_id") or "").strip()
            if pdb:
                sources.append(pdb)
                seen_subsets[subset] = seen_subsets.get(subset, 0) + 1
        summary = ", ".join(f"{k} {v}" for k, v in sorted(seen_subsets.items()))
        print(f"[fetch_entry_resolution] audited structures without a resolution: {summary or 'none'}",
              flush=True)
    if args.ids:
        if not args.ids.is_file():
            raise SystemExit(f"--ids {args.ids} not found")
        sources += args.ids.read_text(encoding="utf-8").splitlines()
    if args.from_dir:
        if not args.from_dir.is_dir():
            raise SystemExit(f"--from-dir {args.from_dir} is not a directory")
        sources += [p.name for p in sorted(args.from_dir.iterdir()) if p.is_file()]
    if not sources:
        parser.error("give --missing-from-audit, --ids and/or --from-dir "
                     "(the audit reported nothing to fetch)")
    ids = entry_ids(sources)
    if not ids:
        raise SystemExit("No four-character PDB ID found in the input")

    rows = read_table(args.out) if args.resume else {}
    if args.resume and rows:
        print(f"[fetch_entry_resolution] resuming: {len(rows)} row(s) already in {args.out}", flush=True)
    todo = [i for i in ids if i not in rows]
    if not todo:
        print(f"[fetch_entry_resolution] nothing to fetch; {len(rows)} row(s) already in {args.out}")
        return 0

    done = 0
    try:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start:start + args.chunk]
            for entry in fetch_chunk(chunk, args.timeout, args.retries):
                pdb = str(entry.get("rcsb_id", "")).strip().upper()
                if not pdb:
                    continue
                resolution = resolution_of(entry)
                rows[pdb] = ("" if resolution is None else f"{resolution:g}", methods_of(entry))
            done += len(chunk)
            print(f"[fetch_entry_resolution] {done}/{len(todo)}", flush=True)
    finally:
        # Whatever was fetched is written, so --resume can continue after an interruption.
        if rows:
            write_table(args.out, rows)

    absent = [i for i in ids if i not in rows]
    without = sum(1 for pdb in ids if pdb in rows and not rows[pdb][0])
    print(f"[fetch_entry_resolution] {len(rows)} row(s) -> {args.out} "
          f"({without} without a resolution, {len(absent)} not returned by RCSB)")
    if absent:
        print(f"[fetch_entry_resolution] not returned: {absent[:10]}"
              f"{'...' if len(absent) > 10 else ''}; rerun with --resume to retry them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
