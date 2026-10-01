"""Per-graph sequence records, partner-side identities and assignment digests for the EGNN split.

Split out of train_egnn_pruning.py. The
component/fold assignment itself (``split_paths``) stays there because it
reads the isolation thresholds that ``main`` rebinds from the command line.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple
import torch

from nanoqc.data.sequence_identity import nw_identity, length_coverage, partner_roles_anchored


def _sequence_identity(a: str, b: str, *, min_length_coverage: float = 0.0) -> float:
    """Symmetric global identity over alignment length with an explicit length gate."""

    a, b = str(a or ""), str(b or "")
    if not a or not b:
        return 0.0
    coverage = length_coverage(a, b)
    if coverage < min_length_coverage:
        return 0.0
    if a == b:
        return 1.0
    return nw_identity(a, b, saturation_message="alignment score saturation while building the split")


def _side_identity(
    left: Sequence[str],
    right: Sequence[str],
    *,
    min_length_coverage: float = 0.0,
) -> float:
    """Maximum global identity across two partner-side sequence sets."""

    return max(
        (
            _sequence_identity(a, b, min_length_coverage=min_length_coverage)
            for a in left for b in right if a and b
        ),
        default=0.0,
    )


def _split_records(paths: Sequence[Path]) -> List[Tuple]:
    """Per-graph isolation fields, loaded once and shared by every consumer."""
    records = []
    for path in paths:
        data = torch.load(path, map_location="cpu", weights_only=False)
        vhh = tuple(sorted(set(str(s) for s in getattr(data, "vhh_sequences", []) if str(s))))
        antigen = tuple(sorted(set(str(s) for s in getattr(data, "antigen_sequences", []) if str(s))))
        if not vhh or not antigen:
            raise ValueError(
                f"{path.name} lacks full-chain vhh_sequences/antigen_sequences; "
                "rebuild graphs with build_final_pyg_dataset.py >= 1.2"
            )
        cdr3 = str(getattr(data, "cdr3_seq", "") or "")
        family_cluster = str(getattr(data, "family_structure_cluster", "") or "")
        if not family_cluster:
            raise ValueError(
                f"{path.name} lacks family_structure_cluster; rebuild formal graphs with "
                "build_final_pyg_dataset.py graph version >=1.8"
            )
        anchored = partner_roles_anchored(getattr(data, "subset_source", ""))
        records.append((path, vhh, antigen, cdr3, family_cluster, anchored))
    return records

def _family_structure_assignment_digest(paths: Sequence[Path]) -> str:
    """Hash graph-name/PDB/family assignments that govern component splitting."""
    digest=hashlib.sha256()
    for path in sorted(paths,key=lambda p:p.name.lower()):
        data=torch.load(path,map_location="cpu",weights_only=False)
        family=str(getattr(data,"family_structure_cluster","") or "")
        pdb=str(getattr(data,"pdb_id","") or "").lower()
        if not family or not pdb:
            raise ValueError(f"{path.name} lacks PDB/family cluster metadata")
        digest.update(f"{path.name}\t{pdb}\t{family}\n".encode("utf-8"))
    return digest.hexdigest()


def _paths_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
