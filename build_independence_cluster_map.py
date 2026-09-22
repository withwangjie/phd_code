#!/usr/bin/env python3
"""Build a frozen PDB->family/structure cluster map from a pairwise similarity TSV.

The input is expected to contain at least query and target identifiers in the
first two columns. A third numeric score column is optional; when present,
--min-score filters edges before connected-component clustering.

This script intentionally does not run Foldseek/MMseqs itself. It consumes an
externally generated, frozen pairwise-similarity table and records provenance
so queue freezing can be reproduced exactly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def norm_id(value: str) -> str:
    value = Path(str(value).strip()).name
    for suffix in (".cif.gz", ".pdb.gz", ".cif", ".pdb", ".mmcif"):
        if value.lower().endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value[:4].lower() if len(value) >= 4 else value.lower()


class DSU:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        self.add(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-provenance", type=Path)
    parser.add_argument("--min-score", type=float, default=None)
    parser.add_argument("--delimiter", default="\t")
    parser.add_argument("--query-column", type=int, default=0)
    parser.add_argument("--target-column", type=int, default=1)
    parser.add_argument("--score-column", type=int, default=2)
    parser.add_argument("--universe", type=Path,
        help="Optional newline-delimited PDB IDs that must all appear in the output map.")
    args = parser.parse_args()

    if not args.pairs.is_file():
        parser.error("--pairs file not found")
    if args.min_score is not None and not math.isfinite(args.min_score):
        parser.error("--min-score must be finite")

    dsu = DSU()
    total_rows = kept_edges = skipped_rows = 0
    with args.pairs.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            total_rows += 1
            fields = line.split(args.delimiter)
            need = max(args.query_column, args.target_column,
                       args.score_column if args.min_score is not None else 0)
            if len(fields) <= need:
                skipped_rows += 1
                continue
            q = norm_id(fields[args.query_column])
            t = norm_id(fields[args.target_column])
            if not q or not t:
                skipped_rows += 1
                continue
            if args.min_score is not None:
                try:
                    score = float(fields[args.score_column])
                except ValueError:
                    skipped_rows += 1
                    continue
                if score < args.min_score:
                    continue
            dsu.add(q); dsu.add(t); dsu.union(q, t); kept_edges += 1

    universe: list[str] = []
    if args.universe is not None:
        if not args.universe.is_file():
            parser.error("--universe file not found")
        universe = [norm_id(line) for line in args.universe.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        for pdb in universe:
            dsu.add(pdb)

    roots: dict[str, list[str]] = {}
    for item in sorted(dsu.parent):
        roots.setdefault(dsu.find(item), []).append(item)

    components = sorted(roots.values(), key=lambda xs: (xs[0], len(xs)))
    mapping: dict[str, str] = {}
    for index, members in enumerate(components, 1):
        cluster = f"cluster_{index:06d}"
        for member in members:
            mapping[member] = cluster

    if universe:
        missing = sorted(set(universe) - set(mapping))
        if missing:
            raise ValueError(f"Universe IDs missing from cluster map: {missing[:20]}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sizes = sorted((len(x) for x in components), reverse=True)
    provenance = dict(
        source_pairs=str(args.pairs.resolve()),
        source_pairs_sha256=sha256(args.pairs),
        universe=(None if args.universe is None else str(args.universe.resolve())),
        universe_sha256=(None if args.universe is None else sha256(args.universe)),
        min_score=args.min_score,
        delimiter=args.delimiter,
        query_column=args.query_column,
        target_column=args.target_column,
        score_column=args.score_column,
        total_rows=total_rows,
        kept_edges=kept_edges,
        skipped_rows=skipped_rows,
        ids=len(mapping),
        components=len(components),
        largest_component=(max(sizes) if sizes else 0),
        cluster_map_sha256=sha256(args.out_json),
        semantics="connected components over frozen external pairwise structural/family similarity edges",
    )
    out_prov = args.out_provenance or args.out_json.with_suffix(".provenance.json")
    out_prov.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
