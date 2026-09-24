#!/usr/bin/env python3
"""Fetch each PDB entry's resolution from RCSB into a declared metadata table.

Some structure files carry no resolution record. RCSB's downloaded biological
assembly files are one case: they hold ``_exptl.method`` but no ``_refine``
block, and their ``_exptl.entry_id`` is a placeholder, so the audit's
structure-quality gate would reject the whole subset for a missing field
rather than for quality (PROTOCOL_AMENDMENTS.md A15).

This writes ``pdb<TAB>resolution<TAB>method`` for the requested entries, which
``audit_all_datasets.load_annotations`` reads by PDB ID exactly as it reads the
SNAC and SAbDab tables, recording the file and its hash in the audit report.
The value is the deposition's own ``rcsb_entry_info.resolution_combined``; an
entry with none (NMR, and some cryo-EM) is written as an empty field and stays
excluded by the gate.

Run it once, keep the table beside the structures, and commit or archive it:
after that the audit needs no network.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence

ENDPOINT = "https://data.rcsb.org/graphql"
QUERY = ("{entries(entry_ids:[%s]){rcsb_id rcsb_entry_info{resolution_combined} exptl{method}}}")
CHUNK = 150


def entry_ids(values: Iterable[str]) -> list[str]:
    """Distinct four-character PDB IDs, upper case, from lines or file names."""
    out = []
    for value in values:
        match = re.search(r"[0-9][A-Za-z0-9]{3}", str(value).strip())
        if match and match.group(0).upper() not in out:
            out.append(match.group(0).upper())
    return out


def request(ids: Sequence[str], timeout: float, retries: int) -> list[dict]:
    body = json.dumps({"query": QUERY % ",".join(f'"{i}"' for i in ids)}).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(ENDPOINT, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
            if payload.get("errors"):
                raise SystemExit(f"RCSB reported: {payload['errors'][:1]}")
            entries = (payload.get("data") or {}).get("entries")
            if entries is None:
                raise SystemExit(f"Unexpected RCSB response: {str(payload)[:300]}")
            return [e for e in entries if e]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise SystemExit(f"RCSB request failed after {retries + 1} attempts: {last}")


def resolution_of(entry: dict) -> Optional[float]:
    values = ((entry.get("rcsb_entry_info") or {}).get("resolution_combined") or [])
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and v > 0]
    return max(numbers) if numbers else None  # worst value, as the gate is an upper bound


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=Path, help="One PDB ID per line (e.g. non_redundant_pdb_ids.txt)")
    parser.add_argument("--from-dir", type=Path, help="Take IDs from structure file names in this directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="Table to write; name it *entry_resolution.tsv so the audit reads it")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    if not args.out.name.endswith("entry_resolution.tsv"):
        raise SystemExit("--out must end with 'entry_resolution.tsv'; that is what the audit reads")
    sources: list[str] = []
    if args.ids:
        sources += args.ids.read_text(encoding="utf-8").splitlines()
    if args.from_dir:
        sources += [p.name for p in sorted(args.from_dir.iterdir()) if p.is_file()]
    if not sources:
        raise SystemExit("Give --ids and/or --from-dir")
    ids = entry_ids(sources)
    if not ids:
        raise SystemExit("No four-character PDB ID found in the input")

    rows, missing = [], 0
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start:start + CHUNK]
        for entry in request(chunk, args.timeout, args.retries):
            resolution = resolution_of(entry)
            missing += resolution is None
            methods = [m.get("method", "") for m in (entry.get("exptl") or [])]
            rows.append((str(entry.get("rcsb_id", "")).upper(),
                         "" if resolution is None else f"{resolution:g}",
                         ";".join(sorted({m for m in methods if m}))))
        print(f"[fetch_entry_resolution] {min(start + CHUNK, len(ids))}/{len(ids)}", flush=True)

    returned = {r[0] for r in rows}
    absent = [i for i in ids if i not in returned]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("pdb\tresolution\tmethod\n")
        for pdb, resolution, method in sorted(rows):
            handle.write(f"{pdb}\t{resolution}\t{method}\n")
    print(f"[fetch_entry_resolution] {len(rows)} entries -> {args.out} "
          f"({missing} without a resolution, {len(absent)} not returned by RCSB)")
    if absent:
        print(f"[fetch_entry_resolution] not returned: {absent[:10]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
