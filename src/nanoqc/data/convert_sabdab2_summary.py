#!/usr/bin/env python3
"""Convert a SAbDab2 single-domain CSV into the classic SAbDab summary TSV.

``select_external_vhh_candidates.py`` reads the classic SAbDab summary
columns (``pdb``, ``Hchain``, ``Lchain``, ``antigen_chain``, ``antigen_type``,
``date``, ``resolution``, ``method``, ``scfv``). The SAbDab2 single-domain
export uses different column names, identifiers and vocabularies. This module
renames them without changing any selection rule:

* ``PDB`` (``pdb_0000<id>``) -> ``pdb`` (the four-character ID);
* ``date`` ``YYYY/MM/DD`` -> ISO ``YYYY-MM-DD``;
* ``antigen_type`` ``PROTEIN|SUGAR`` -> ``protein|sugar`` (tokens lower-cased,
  never dropped, so the selector's antigen rule decides unchanged);
* ``method`` ``XRAY`` -> ``X-RAY DIFFRACTION``, ``ELECTRON_MICROSCOPY`` ->
  ``ELECTRON MICROSCOPY``, and so on;
* empty ``Lchain``/``resolution`` -> ``NA``, as in the classic file.

``type`` selects which single-domain rows are kept (default ``SD-H``: the
camelid VHH class of the training and hard test sets). ``SD-L`` single-domain
light chains and shark ``VNAR`` domains are a different molecule class and are
dropped by default, with their counts recorded.

SAbDab's own ``VH`` and ``CDR-H1..H3`` sequences are carried through as extra
columns. They are the same IMGT annotation the training set uses, so the
graph builder can prefer them over its own CDR detection.

**Caveat recorded in the manifest:** a single-domain export lists only
single-domain chains. A PDB that also contains a Fab shows no ``Lchain`` here,
so the selector's VH/VL exclusion cannot fire from this file alone. Pass the
full SAbDab summary as a second ``--sabdab-summary`` file, or rely on the
structure-level antibody detection, to exclude those entries.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

from nanoqc.common.repo_io import sha256_file as sha256

# Classic SAbDab summary columns the selector reads, plus carried annotations.
OUTPUT_FIELDS = ("pdb", "Hchain", "Lchain", "antigen_chain", "antigen_type", "antigen_name",
                 "date", "resolution", "method", "scfv", "VH", "CDR-H1", "CDR-H2", "CDR-H3")
METHODS = {
    "XRAY": "X-RAY DIFFRACTION",
    "ELECTRON_MICROSCOPY": "ELECTRON MICROSCOPY",
    "SOLUTION_NMR": "SOLUTION NMR",
    "ELECTRON_CRYSTALLOGRAPHY": "ELECTRON CRYSTALLOGRAPHY",
    "NEUTRON_DIFFRACTION": "NEUTRON DIFFRACTION",
}
_MISSING = {"", "na", "none", "nan"}


def pdb_id(value: str) -> str:
    """Four-character PDB ID of a SAbDab2 ``pdb_0000<id>`` identifier."""
    token = str(value or "").strip().lower()
    if token.startswith("pdb_") and len(token) == 12:
        token = token[-4:]
    if len(token) != 4 or not token.isalnum():
        raise ValueError(f"not a four-character PDB identifier: {value!r}")
    return token


def iso_date(value: str) -> str:
    return dt.datetime.strptime(str(value).strip(), "%Y/%m/%d").date().isoformat()


def _clean(value: str) -> str:
    text = str(value or "").strip()
    return "NA" if text.lower() in _MISSING else text


def convert_row(row: dict) -> dict:
    """One classic-format row; raises when a required field cannot be mapped."""
    return {
        "pdb": pdb_id(row["PDB"]),
        "Hchain": _clean(row.get("Hchain")),
        "Lchain": _clean(row.get("Lchain")),
        "antigen_chain": " | ".join(part.strip() for part in str(row.get("antigen_chain") or "").split("|")
                                    if part.strip()) or "NA",
        "antigen_type": " | ".join(part.strip().lower() for part in str(row.get("antigen_type") or "").split("|")
                                   if part.strip()) or "NA",
        "antigen_name": _clean(row.get("antigen_name")),
        "date": iso_date(row["date"]),
        "resolution": _clean(row.get("resolution")),
        "method": METHODS.get(str(row.get("method") or "").strip().upper(),
                              str(row.get("method") or "").strip().replace("_", " ").upper() or "NA"),
        "scfv": "False",  # single-domain export: no scFv rows
        "VH": _clean(row.get("VH")),
        "CDR-H1": _clean(row.get("CDR-H1")),
        "CDR-H2": _clean(row.get("CDR-H2")),
        "CDR-H3": _clean(row.get("CDR-H3")),
    }


def convert(source: Path, keep_types: Sequence[str]) -> tuple[list[dict], dict]:
    wanted = {t.strip().upper() for t in keep_types}
    rows, failures = [], []
    kinds: Counter = Counter()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"PDB", "Hchain", "date", "type"} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{source} is not a SAbDab2 single-domain export (missing {sorted(missing)})")
        for number, row in enumerate(reader, 2):
            kind = str(row.get("type") or "").strip().upper()
            kinds[kind] += 1
            if kind not in wanted:
                continue
            try:
                rows.append(convert_row(row))
            except (KeyError, ValueError) as exc:
                failures.append(dict(line=number, pdb=str(row.get("PDB", "")), reason=str(exc)))
    unique = list({tuple(sorted(row.items())): row for row in rows}.values())
    unique.sort(key=lambda row: (row["pdb"], row["Hchain"], row["antigen_chain"]))
    stats = dict(rows_by_type=dict(sorted(kinds.items())), kept_types=sorted(wanted),
                 converted_rows=len(rows), unique_rows=len(unique),
                 entries=len({row["pdb"] for row in unique}), failures=failures)
    return unique, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="SAbDab2 single-domain CSV")
    parser.add_argument("--out", type=Path, required=True, help="Classic-format TSV to write")
    parser.add_argument("--keep-types", nargs="+", default=["SD-H"],
                        help="SAbDab2 'type' values to keep (default: SD-H, the camelid VHH class)")
    args = parser.parse_args(argv)

    rows, stats = convert(args.source, args.keep_types)
    if not rows:
        raise SystemExit(f"No rows of type {args.keep_types} in {args.source}")
    if stats["failures"]:
        raise SystemExit(f"{len(stats['failures'])} row(s) could not be converted: {stats['failures'][:5]}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    dates = sorted(row["date"] for row in rows)
    manifest = dict(
        schema="sabdab2_conversion_v1", source=str(args.source.resolve()), source_sha256=sha256(args.source),
        output=str(args.out.resolve()), output_sha256=sha256(args.out), date_range=[dates[0], dates[-1]],
        caveat=("A single-domain export lists only single-domain chains, so an entry that also contains a Fab "
                "shows no Lchain here and the selector's VH/VL exclusion cannot fire from this file alone."),
        **stats)
    manifest_path = args.out.with_suffix(".conversion.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[convert_sabdab2_summary] {stats['unique_rows']} rows / {stats['entries']} PDB entries "
          f"({dates[0]} .. {dates[-1]}) -> {args.out}")
    for kind, count in manifest["rows_by_type"].items():
        if kind not in manifest["kept_types"]:
            print(f"  dropped type {kind}: {count} row(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
