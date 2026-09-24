"""Canonical amino-acid atom tables (wwPDB heavy-atom names), dependency-free.

Formerly duplicated in evaluate_complex_metrics.py (tuples),
subgraph_to_qubo.py (space-joined strings / lists) and audit_all_datasets.py
(sets). Each module keeps its historical container type by deriving it from
the tables below, so lookups behave exactly as before.
"""
from __future__ import annotations

from typing import Dict, Tuple

BACKBONE_ATOMS: Tuple[str, ...] = ("N", "CA", "C", "O")

SIDECHAIN_HEAVY_ATOMS: Dict[str, Tuple[str, ...]] = {
    "GLY": (), "ALA": ("CB",), "SER": ("CB", "OG"), "CYS": ("CB", "SG"),
    "THR": ("CB", "OG1", "CG2"), "VAL": ("CB", "CG1", "CG2"),
    "ILE": ("CB", "CG1", "CG2", "CD1"), "LEU": ("CB", "CG", "CD1", "CD2"),
    "ASP": ("CB", "CG", "OD1", "OD2"), "ASN": ("CB", "CG", "OD1", "ND2"),
    "GLU": ("CB", "CG", "CD", "OE1", "OE2"), "GLN": ("CB", "CG", "CD", "OE1", "NE2"),
    "LYS": ("CB", "CG", "CD", "CE", "NZ"),
    "ARG": ("CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    "MET": ("CB", "CG", "SD", "CE"), "PRO": ("CB", "CG", "CD"),
    "HIS": ("CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    "PHE": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"),
    "TRP": ("CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

# Atom-name pairs exchanged together by a physically valid local symmetry
# (for PHE/TYR: the single 180-degree ring flip moving CD1<->CD2 and CE1<->CE2).
# VAL CG1/CG2 and LEU CD1/CD2 are prochiral, stereochemically distinct methyls
# (IUPAC naming fixes which is which), so they are deliberately NOT swapped:
# doing so would both lower side-chain RMSD and let a ~120-degree chi error
# count as recovered.
SYMMETRIC_SWAPS: Dict[str, Tuple[Tuple[str, str], ...]] = {
    "ASP": (("OD1", "OD2"),), "GLU": (("OE1", "OE2"),),
    "ARG": (("NH1", "NH2"),),
    "PHE": (("CD1", "CD2"), ("CE1", "CE2")),
    "TYR": (("CD1", "CD2"), ("CE1", "CE2")),
}
