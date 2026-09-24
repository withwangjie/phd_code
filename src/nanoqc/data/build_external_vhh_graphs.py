#!/usr/bin/env python3
"""Build graph-v1.8 external VHH graphs from raw PDB/mmCIF files.

External complexes get the training-graph definition of one VHH-antigen
complex: the first author-determined biological assembly, one annotated VHH
chain as partner 0, antigen = assembly chains with a CA/CB within 7.5 A of
the VHH paratope (SAbDab rule [METHODS_EVIDENCE R33]). Copies of that VHH and
every other annotated antibody chain of the entry are never antigen, as in the
SNAC-DB per-VHH complexes that supply the training and hard test sets. The
graph also keeps 5 A heavy-atom interface labels, 8 A intra-chain CA edges and
fixed cross-partner KNN. Labelling, edges and validation are the unchanged
``build_final_pyg_dataset.make_graph`` code; the audit's structure-quality
gates (resolution, interface side-chain completeness, altloc, occupancy) are
applied to the VHH-antigen interface.

CDRs use IMGT numbering from ANARCI when it is installed (CDR1 27-38, CDR2
56-65, CDR3 105-117). Without ANARCI only CDR-H3 is located, between the
conserved FR3 cysteine motif and the FR4 W-G-x-G motif (the IMGT 104/118
anchors), and the paratope falls back to the whole VHH chain, as for
training complexes whose CDRs cannot be mapped. The method is recorded per
graph.

Candidates come from ``select_external_vhh_candidates.py``; only those it
marks ``selected`` are built. Independence from the training set is not
decided here: ``audit_external_vhh_independence.py`` certifies it inside
every formal run.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Sequence

import gemmi
import numpy as np
import torch
from scipy.spatial import cKDTree

import nanoqc.data.audit_all_datasets as audit
import nanoqc.data.build_final_pyg_dataset as builder
from nanoqc.common.repo_io import sha256_file as sha256

SUBSET = "external_vhh"
IMGT_CDRS = {"cdr1": (27, 38), "cdr2": (56, 65), "cdr3": (105, 117)}
# FR3 ends in Y-[YFHW]-C (IMGT 102-104); FR4 starts with W-G-x-G (IMGT 118-121).
_FR3_MOTIF = re.compile(r"Y[YFHW]C")
_FR4_MOTIF = re.compile(r"WG.G")
CDR3_MAX_LENGTH = 40


def cdr3_by_motif(sequence: str) -> str:
    """CDR-H3 (IMGT 105-117) between the FR3 cysteine and the FR4 tryptophan.

    Uses the last W-G-x-G motif and the earliest FR3 motif whose cysteine lies
    3..CDR3_MAX_LENGTH residues before it; raises when no such pair exists.
    """
    ends = [m.start() for m in _FR4_MOTIF.finditer(sequence)]
    if not ends:
        raise ValueError("no FR4 W-G-x-G motif; CDR-H3 cannot be located")
    end = ends[-1]
    for match in _FR3_MOTIF.finditer(sequence):
        start = match.end()  # first residue after the conserved cysteine
        if 3 <= end - start <= CDR3_MAX_LENGTH:
            return sequence[start:end]
    raise ValueError("no FR3 cysteine motif within CDR-H3 range of FR4; CDR-H3 cannot be located")


def _anarci_cdrs(sequence: str) -> Optional[dict]:
    try:
        from anarci import anarci  # optional dependency
    except ImportError:
        return None
    numbering, _, _ = anarci([("vhh", sequence)], scheme="imgt", output=False)
    if not numbering or not numbering[0]:
        raise ValueError("ANARCI found no variable domain in the VHH chain")
    domain = numbering[0][0][0]
    loops = {name: "".join(aa for (number, _), aa in domain if lo <= number <= hi and aa != "-")
             for name, (lo, hi) in IMGT_CDRS.items()}
    if not all(loops.values()):
        raise ValueError(f"ANARCI IMGT numbering left an empty CDR: {loops}")
    return dict(loops, method="anarci_imgt")


def annotate_cdrs(sequence: str) -> dict:
    """IMGT CDRs of a VHH sequence; ``cdr1``/``cdr2`` are empty without ANARCI."""
    annotated = _anarci_cdrs(sequence)
    if annotated is None:
        annotated = dict(cdr1="", cdr2="", cdr3=cdr3_by_motif(sequence), method="imgt_anchor_motif")
    if sequence.count(annotated["cdr3"]) != 1:
        raise ValueError("CDR-H3 does not map uniquely onto the VHH chain")
    return annotated


def _paratope(anchor: dict, cdrs: dict) -> tuple[list, str]:
    nodes = anchor["nodes"]
    loops = [cdrs.get(key, "") for key in ("cdr1", "cdr2", "cdr3")]
    sequence = "".join(node["aa"] for node in nodes)
    if all(loops) and all(sequence.count(loop) == 1 for loop in loops):
        indices = sorted({i for loop in loops
                          for i in range(sequence.index(loop), sequence.index(loop) + len(loop))})
        return [nodes[i] for i in indices], "vhh_cdr1_cdr2_cdr3"
    return nodes, "whole_vhh_chain_cdr_unmapped"


def _author_chain(name: str) -> str:
    """Author chain of an assembly chain (symmetry copies are named ``<chain>-<operator>``)."""
    return name.split("-", 1)[0]


def contacting_antigen_chains(chains: list, cdrs: dict, meta: dict,
                              other_antibody_chains: Sequence[str] = ()) -> tuple[list, dict]:
    """SAbDab antigen rule inside the assembly; antibody chains are never antigen."""
    anchor = next(c for c in chains if c["group"] == 0)
    paratope, basis = _paratope(anchor, cdrs)
    tree = cKDTree(builder._ca_cb_coordinates(paratope))
    anchor_sequence = "".join(node["aa"] for node in anchor["nodes"])
    cutoff = builder.ANTIGEN_CHAIN_CONTACT_ANGSTROM
    kept, dropped = [], []
    for chain in chains:
        if chain["group"] == 0:
            kept.append(chain)
            continue
        if "".join(node["aa"] for node in chain["nodes"]) == anchor_sequence:
            dropped.append(dict(chain=chain["name"], reason="copy_of_vhh"))
            continue
        if _author_chain(chain["name"]) in set(other_antibody_chains):
            dropped.append(dict(chain=chain["name"], reason="other_antibody_chain"))
            continue
        distance = float(tree.query(builder._ca_cb_coordinates(chain["nodes"]), k=1)[0].min())
        if distance <= cutoff:
            kept.append(chain)
        else:
            dropped.append(dict(chain=chain["name"], reason=f"no_paratope_contact_within_{cutoff:g}A",
                                min_ca_cb_distance=round(distance, 3)))
    if not any(c["group"] == 1 for c in kept):
        raise ValueError(f"no partner chain within {cutoff:g} A of the paratope in the biological assembly")
    meta = dict(meta,
                antigen_chain_rule=(f"biological-assembly chains with any CA/CB within {cutoff:g} A "
                                    "of a paratope CA/CB (SAbDab)"),
                antigen_contact_basis=basis, dropped_chains=dropped)
    return kept, meta


def interface_quality(model, vhh_name: str, antigen_names: Sequence[str], resolution: Optional[float]) -> dict:
    """The audit's structure-quality gates, on every VHH-antigen interface residue."""
    chains = {chain["name"]: chain for chain in audit.chain_data(model)[0]}
    vhh = chains[vhh_name]
    records = []
    for name in antigen_names:
        antigen = chains[name]
        left, right = audit.contact_residue_ids(vhh, antigen)
        for chain, ids in ((vhh, left), (antigen, right)):
            for rid in ids:
                records.append(dict(chain=chain["name"], rid=int(rid),
                                    missing_sidechain=chain["residue_missing_sidechain"][rid],
                                    altloc=bool(chain["residue_altloc"][rid]),
                                    min_occupancy=float(chain["residue_min_occ"][rid])))
    unique = list({(r["chain"], r["rid"]): r for r in records}.values())
    missing = sum(bool(r["missing_sidechain"]) for r in unique)
    altloc = sum(r["altloc"] for r in unique)
    min_occ = min((r["min_occupancy"] for r in unique), default=None)
    reasons = []
    if resolution is None:
        reasons.append("unknown_resolution")
    elif resolution > audit.MAX_RESOLUTION_ANGSTROM:
        reasons.append("resolution_above_limit")
    if missing:
        reasons.append("incomplete_interface_sidechain")
    if altloc:
        reasons.append("interface_altloc")
    if min_occ is not None and min_occ < audit.MIN_INTERFACE_OCCUPANCY:
        reasons.append("low_interface_occupancy")
    return dict(resolution_angstrom=resolution, interface_residues=len(unique),
                interface_missing_sidechain_residues=missing,
                interface_altloc_residues=altloc, interface_min_occupancy=min_occ,
                structure_quality_status="pass" if not reasons else "fail",
                structure_quality_reasons=reasons)


def prepare_external_complex(structure_path: Path, vhh_chain: str,
                             other_antibody_chains: Sequence[str] = ()) -> dict:
    """Assembly, partner chains, CDRs and quality for one external complex."""
    st = gemmi.read_structure(str(structure_path))
    if not len(st):
        raise ValueError("no coordinate model")
    while len(st) > 1:
        del st[1]
    resolution = float(getattr(st, "resolution", 0.0) or 0.0)
    resolution = resolution if math.isfinite(resolution) and resolution > 0 else None
    st = audit.biological_assembly_structure(st)
    chains = builder.build_atoms(st)
    anchors = [c for c in chains if c["name"] == vhh_chain]
    if len(anchors) != 1:
        raise ValueError(f"VHH chain {vhh_chain} absent or ambiguous in the biological assembly")
    for chain in chains:
        chain["group"] = 0 if chain["name"] == vhh_chain else 1
    vhh_sequence = "".join(node["aa"] for node in anchors[0]["nodes"])
    cdrs = annotate_cdrs(vhh_sequence)
    meta = dict(structure_source=str(dict(st.info)[audit.STRUCTURE_SOURCE_KEY]))
    kept, meta = contacting_antigen_chains(chains, cdrs, meta, other_antibody_chains)
    for chain in kept:
        chain["complex_meta"] = meta
    antigen_names = [c["name"] for c in kept if c["group"] == 1]
    quality = interface_quality(st[0], vhh_chain, antigen_names, resolution)
    return dict(chains=kept, meta=meta, cdrs=cdrs, quality=quality, vhh_sequence=vhh_sequence,
                antigen_sequences=["".join(n["aa"] for n in c["nodes"]) for c in kept if c["group"] == 1],
                antigen_chains=antigen_names)


def build_graph(prepared: dict, pdb_id: str, structure_path: Path):
    """Graph-v1.8 Data object for a prepared complex that passed quality gates."""
    if prepared["quality"]["structure_quality_status"] != "pass":
        raise ValueError(f"structure quality failed: {prepared['quality']['structure_quality_reasons']}")
    row = dict(pdb_id=pdb_id.upper(), subset=SUBSET, id=f"{SUBSET}/{pdb_id.lower()}",
               cdr3_sequences=[prepared["cdrs"]["cdr3"]], vhh_status="pass")
    graph = builder.make_graph(row, SUBSET, chains=prepared["chains"])
    graph.cdr_annotation_method = prepared["cdrs"]["method"]
    graph.cdr1_seq = prepared["cdrs"]["cdr1"]
    graph.cdr2_seq = prepared["cdrs"]["cdr2"]
    graph.source_structure_sha256 = sha256(structure_path)
    graph.structure_quality = json.dumps(prepared["quality"], sort_keys=True)
    return graph


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="candidates.json written by select_external_vhh_candidates.py")
    parser.add_argument("--structures-dir", type=Path, required=True,
                        help="Raw <pdb>.cif/.pdb files (the formal external source_structure_dir)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Graph directory (formal graph_dir)")
    args = parser.parse_args(argv)

    from nanoqc.data.audit_external_vhh_independence import source_structure_for_pdb
    payload = json.loads(args.candidates.read_text(encoding="utf-8"))
    selected = [row for row in payload["candidates"] if row.get("selected")]
    if not selected:
        raise ValueError("No selected candidates in the candidate file")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stale = sorted(args.out_dir.glob("*.pt"))
    if stale:
        raise ValueError(f"Output directory already holds graphs; use an empty directory: {stale[:5]}")
    built, failures = [], []
    for row in selected:
        pdb = str(row["pdb_id"]).lower()
        try:
            source = source_structure_for_pdb(args.structures_dir, pdb)
            if row.get("source_structure_sha256") and sha256(source) != row["source_structure_sha256"]:
                raise ValueError("raw structure changed since candidate selection")
            prepared = prepare_external_complex(source, row["vhh_chain"], row.get("other_antibody_chains", []))
            graph = build_graph(prepared, pdb, source)
            path = args.out_dir / f"{SUBSET}__{pdb.upper()}.pt"
            torch.save(graph, path)
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            builder.validate_graph(loaded)
            built.append(dict(pdb_id=pdb, path=path.name, sha256=sha256(path), nodes=int(graph.num_nodes),
                              vhh_chain=row["vhh_chain"], antigen_chains=prepared["antigen_chains"],
                              cdr3=prepared["cdrs"]["cdr3"], cdr_annotation_method=prepared["cdrs"]["method"],
                              antigen_contact_basis=prepared["meta"]["antigen_contact_basis"],
                              num_interface_residues=int(graph.num_interface_residues),
                              source_structure=str(source.resolve()), source_structure_sha256=sha256(source)))
        except Exception as exc:  # recorded, never silently dropped
            failures.append(dict(pdb_id=pdb, reason=type(exc).__name__, detail=str(exc)))
    manifest = dict(schema="external_vhh_graphs_v1", graph_version=builder.VERSION, subset=SUBSET,
                    candidates_sha256=sha256(args.candidates), graphs=built, failures=failures,
                    note="Independence from training is certified per run by audit_external_vhh_independence.py.")
    (args.out_dir / "external_graph_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"built {len(built)} graph(s); {len(failures)} failure(s)")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
