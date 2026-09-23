"""List recent PDB entries mentioning a nanobody or VHH for manual curation.

Discovery is deliberately separate from admission: a text hit does not prove
that an entry contains a usable VHH-antigen complex.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import urllib.parse
import urllib.request


def search(term: str, since: str, *, entity_only: bool = False) -> list[str]:
    query = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text" if entity_only else "full_text", "parameters": (
                {"attribute": "rcsb_polymer_entity.pdbx_description", "operator": "contains_phrase", "value": term}
                if entity_only else {"value": term}
            )},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater_or_equal", "value": since,
            }},
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 10000}},
    }
    encoded = urllib.parse.urlencode({"json": json.dumps(query, separators=(",", ":"))})
    with urllib.request.urlopen("https://search.rcsb.org/rcsbsearch/v2/query?" + encoded, timeout=60) as response:
        payload = json.load(response)
    return [hit["identifier"] for hit in payload.get("result_set", [])]


def entry_metadata_many(ids: list[str]) -> list[dict[str, str]]:
    query = """query ($ids: [String!]!) {
      entries(entry_ids: $ids) {
        rcsb_id
        rcsb_accession_info { initial_release_date }
        struct { title }
        exptl { method }
        rcsb_entry_info { resolution_combined }
        rcsb_entry_container_identifiers { polymer_entity_ids }
        polymer_entities {
          rcsb_polymer_entity { pdbx_description }
          rcsb_polymer_entity_container_identifiers { entity_id auth_asym_ids }
        }
      }
    }"""
    request = urllib.request.Request(
        "https://data.rcsb.org/graphql",
        data=json.dumps({"query": query, "variables": {"ids": ids}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            break
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    if payload.get("errors"):
        raise ValueError(payload["errors"])
    records = []
    for entry in (payload.get("data") or {}).get("entries") or []:
        if entry is None:
            continue
        records.append({
            "pdb_id": entry["rcsb_id"],
            "release_date": str((entry.get("rcsb_accession_info") or {}).get("initial_release_date", "")),
            "title": str((entry.get("struct") or {}).get("title", "")),
            "method": ";".join(item.get("method", "") for item in entry.get("exptl") or []),
            "resolution": ";".join(str(x) for x in (entry.get("rcsb_entry_info") or {}).get("resolution_combined") or []),
            "entity_ids": ";".join(str(x) for x in (entry.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []),
            "entities": " | ".join(
                str((entity.get("rcsb_polymer_entity_container_identifiers") or {}).get("entity_id", ""))
                + ":" + ",".join((entity.get("rcsb_polymer_entity_container_identifiers") or {}).get("auth_asym_ids") or [])
                + ":" + str((entity.get("rcsb_polymer_entity") or {}).get("pdbx_description", ""))
                for entity in entry.get("polymer_entities") or []
            ),
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2026-05-26")
    parser.add_argument("--audit-csv", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--entity-only", action="store_true")
    args = parser.parse_args()
    matches: dict[str, set[str]] = {}
    for term in ("nanobody", "VHH", "single-domain antibody"):
        ids = search(term, args.since, entity_only=args.entity_only)
        print(term, len(ids), flush=True)
        for pdb_id in ids:
            matches.setdefault(pdb_id, set()).add(term)
    existing: set[str] = set()
    if args.audit_csv:
        with args.audit_csv.open(encoding="utf-8-sig", newline="") as handle:
            existing = {str(row["pdb_id"]).upper() for row in csv.DictReader(handle)}
    ids = sorted(set(matches) - existing)
    records = []
    for start in range(0, len(ids), 50):
        records.extend(entry_metadata_many(ids[start:start + 50]))
    for row in records:
        row["matched_terms"] = ";".join(sorted(matches[row["pdb_id"]]))
    fields = ["pdb_id", "release_date", "title", "method", "resolution", "entity_ids", "entities", "matched_terms"]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(records)
    print(f"{len(matches)} search hits; {len(existing)} audited PDB IDs; {len(records)} non-overlap candidates")
    for row in records:
        print(row["pdb_id"], row["release_date"], row["resolution"], row["title"][:100], sep="\t")


if __name__ == "__main__":
    main()
