"""Backbone-dependent rotamer sources and per-site candidate pools.

Loads Dunbrack 2010 samples from the installed PyRosetta dun10 database (or
the legacy text library), records their provenance, and expands them into the
fixed-resolution rotamer templates each Active site offers.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple
from nanoqc.qubo.atomistic_structure import _sha256_path
from nanoqc.qubo.qubo_types import RotamerTemplate, chi1_well_index



_THREE_LETTER = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS",
    "I":"ILE","K":"LYS","L":"LEU","M":"MET","N":"ASN","P":"PRO","Q":"GLN",
    "R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR",
}

def _nearest_dunbrack_bin(angle: float) -> int:
    """Nearest 10-degree backbone bin in the Dunbrack 2010 library."""
    if not math.isfinite(angle):
        raise ValueError("Dunbrack lookup requires finite backbone phi/psi")
    value=int(round(float(angle)/10.0)*10)
    while value>180: value-=360
    while value<-180: value+=360
    return value

def _load_dunbrack_bins(library_path: Path, requested_bins: set[tuple[str,int,int]]) -> Dict[tuple[str,int,int], list[RotamerTemplate]]:
    """Read requested residue/phi/psi bins from ALL.bbdep.rotamers.lib."""
    path=Path(library_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dunbrack 2010 rotamer library not found: {path}")
    found={key:[] for key in requested_bins}
    with path.open("r",encoding="utf-8",errors="replace") as handle:
        for raw in handle:
            line=raw.strip()
            if not line or line.startswith("#"): continue
            fields=line.split()
            if len(fields)<17:
                continue
            try:
                residue=fields[0].upper()
                phi=float(fields[1]); psi=float(fields[2])
                if residue not in _THREE_LETTER.values():
                    continue
                if not (-180.0 <= phi <= 180.0 and -180.0 <= psi <= 180.0):
                    raise ValueError(f"Invalid Dunbrack backbone bin: {residue} {phi} {psi}")
                if abs(phi/10.0-round(phi/10.0))>1e-6 or abs(psi/10.0-round(psi/10.0))>1e-6:
                    raise ValueError(f"Dunbrack traditional library must use 10-degree bins: {residue} {phi} {psi}")
                key=(residue,int(round(phi)),int(round(psi)))
            except ValueError:
                continue
            if key not in found: continue
            probability=float(fields[8])
            chis=tuple(float(v) for v in fields[9:13])
            sigmas=tuple(float(v) for v in fields[13:17])
            nz=[i for i,(chi,sigma) in enumerate(zip(chis,sigmas)) if abs(chi)>1e-12 or abs(sigma)>1e-12]
            n=(max(nz)+1) if nz else 1
            found[key].append(RotamerTemplate(chis[0],probability,chis[:n],sigmas[:n],"dunbrack2010"))
    missing=[key for key,rows in found.items() if not rows]
    if missing:
        raise ValueError(f"Dunbrack library lacks requested bins: {missing[:8]}")
    normalized={}
    for key,rows in found.items():
        total=sum(max(0.0,row.prior_probability) for row in rows)
        if total<=0: raise ValueError(f"Dunbrack probabilities sum to zero at {key}")
        normalized[key]=[RotamerTemplate(r.chi1_degrees,r.prior_probability/total,r.chi_degrees,r.chi_sigmas,r.source) for r in rows]
    return normalized


class PyRosettaRotamerProvider:
    """Read backbone-dependent dun10 samples from Rosetta's installed database.

    PyRosetta is imported lazily so legacy analyses remain usable without its
    separately licensed distribution.  The formal mode never falls back to a
    text library when the requested Rosetta API or database is unavailable.
    """

    source = "pyrosetta_dun10"
    init_options = "-mute all -dun10"

    def __init__(self) -> None:
        try:
            import pyrosetta
        except ImportError as exc:
            raise RuntimeError("PyRosetta dun10 mode requires an installed PyRosetta distribution") from exc
        if not pyrosetta.rosetta.basic.was_init_called():
            pyrosetta.init(self.init_options)
        # Do not query ``get_boolean_option('dun10')`` here.  Some PyRosetta
        # builds do not register that legacy OptionKey in the Python binding;
        # the getter then aborts the entire process instead of raising Python
        # an exception.  ``-dun10`` is supplied on initialization above, and
        # load_bins() validates the actual Dunbrack sample API and data.
        self.pyrosetta = pyrosetta
        self.rosetta = pyrosetta.rosetta
        self.version = pyrosetta.version()

    def load_bins(self, requested_bins: set[tuple[str, int, int]]) -> Dict[tuple[str, int, int], list[RotamerTemplate]]:
        rosetta = self.rosetta
        factory = rosetta.core.pack.dunbrack.RotamerLibrary.get_instance()
        found = {}
        for residue, phi, psi in sorted(requested_bins):
            one_letter = next((aa for aa, three in _THREE_LETTER.items() if three == residue), None)
            if one_letter is None:
                raise ValueError(f"Unsupported Rosetta rotamer residue: {residue}")
            library = factory.get_library_by_aa(rosetta.core.chemical.aa_from_oneletter_code(one_letter))
            if library is None or not hasattr(library, "get_all_rotamer_samples"):
                raise RuntimeError(f"PyRosetta dun10 samples unavailable for {residue}")
            backbone = rosetta.utility.fixedsizearray1_double_5_t()
            backbone[1], backbone[2] = float(phi), float(psi)
            samples = library.get_all_rotamer_samples(backbone)
            rows = []
            for sample in samples:
                n_chi = int(sample.nchi())
                probability = float(sample.probability())
                if n_chi < 1 or n_chi > 4 or not math.isfinite(probability) or probability < 0:
                    raise ValueError(f"Invalid PyRosetta dun10 sample for {residue} {phi} {psi}")
                means, sigmas = sample.chi_mean(), sample.chi_sd()
                chis = tuple(float(means[i]) for i in range(1, n_chi + 1))
                deviations = tuple(float(sigmas[i]) for i in range(1, n_chi + 1))
                if not all(math.isfinite(v) for v in (*chis, *deviations)) or any(v < 0 for v in deviations):
                    raise ValueError(f"Nonfinite PyRosetta dun10 chi for {residue} {phi} {psi}")
                rows.append(RotamerTemplate(chis[0], probability, chis, deviations, self.source))
            total = sum(row.prior_probability for row in rows)
            if total <= 0:
                raise ValueError(f"PyRosetta dun10 has no nonzero samples for {residue} {phi} {psi}")
            found[(residue, phi, psi)] = [
                RotamerTemplate(row.chi1_degrees, row.prior_probability / total,
                                row.chi_degrees, row.chi_sigmas, row.source)
                for row in rows
            ]
        return found


def _load_rotamer_bins(mode: str, library_path: Optional[Path],
                       requested: set[tuple[str, int, int]]) -> Dict[tuple[str, int, int], list[RotamerTemplate]]:
    if mode == "pyrosetta_dun10":
        return PyRosettaRotamerProvider().load_bins(requested)
    if mode == "dunbrack2010":
        if library_path is None:
            raise ValueError("Dunbrack text mode requires rotamer_library_path")
        return _load_dunbrack_bins(library_path, requested)
    raise ValueError(f"No backbone-dependent rotamer provider for {mode}")


def rotamer_source_metadata(mode: str, library_path: Optional[Path]) -> dict:
    if mode == "pyrosetta_dun10":
        provider = PyRosettaRotamerProvider()
        return {"rotamer_source": provider.source,
                "pyrosetta_version": provider.version,
                "pyrosetta_init_options": provider.init_options,
                "rotamer_library_path": None, "rotamer_library_sha256": None}
    return {"rotamer_source": mode,
            "rotamer_library_path": None if library_path is None else str(library_path),
            "rotamer_library_sha256": _sha256_path(library_path)}

def _dunbrack_templates_for_site(library_bins, amino_acid: str, phi: float, psi: float, *, probability_floor: float, sigma_offsets: Sequence[float], ensure_chi1_wells: bool = False) -> Tuple[RotamerTemplate, ...]:
    """Expand Dunbrack samples, retaining rare real wells when required.

    The global probability floor can erase an entire chi1 well even when the
    source library contains it. A fixed three-well experiment keeps the most
    probable *observed library sample* in each such well. Its original tiny
    probability is preserved; no rotamer or probability is fabricated.
    """
    key=(_THREE_LETTER[amino_acid],_nearest_dunbrack_bin(phi),_nearest_dunbrack_bin(psi))
    source_rows=library_bins[key]
    retained=[row for row in source_rows if row.prior_probability>=probability_floor]
    if ensure_chi1_wells:
        present={chi1_well_index(row.chi1_degrees) for row in retained}
        for well in range(3):
            if well in present:
                continue
            candidates=[row for row in source_rows
                        if row.prior_probability>0 and chi1_well_index(row.chi1_degrees)==well]
            if candidates:
                retained.append(max(candidates,key=lambda row:row.prior_probability))
    expanded=[]
    for row in retained:
        sigma1=row.chi_sigmas[0] if row.chi_sigmas else 0.0
        for z in sigma_offsets:
            angle=((row.chi1_degrees+float(z)*sigma1+180.0)%360.0)-180.0
            weight=row.prior_probability*math.exp(-0.5*float(z)**2)
            chis=list(row.chi_degrees)
            if chis:
                chis[0]=angle
            else:
                chis=[angle]
            expanded.append(RotamerTemplate(angle,weight,tuple(chis),row.chi_sigmas,row.source))
    if not expanded: raise ValueError(f"No Dunbrack candidates survived probability floor for {key}")
    expanded.sort(key=lambda r:(-r.prior_probability,r.chi1_degrees))
    total=sum(r.prior_probability for r in expanded)
    return tuple(RotamerTemplate(r.chi1_degrees,r.prior_probability/total,r.chi_degrees,r.chi_sigmas,r.source) for r in expanded)

# Three broad backbone-independent chi1 modes, ordered by simple residue-class
# prior. Gly/Ala entries are surrogate microstates because those residues have
# no physical chi1; this is explicitly recorded in result metadata.
_ROTAMER_PRIORS: Mapping[str, Tuple[RotamerTemplate, ...]] = {
    "A": (RotamerTemplate(60.0, 0.50), RotamerTemplate(-60.0, 0.35), RotamerTemplate(180.0, 0.15)),
    "C": (RotamerTemplate(-60.0, 0.48), RotamerTemplate(60.0, 0.32), RotamerTemplate(180.0, 0.20)),
    "D": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(60.0, 0.34), RotamerTemplate(180.0, 0.20)),
    "E": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "F": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.40), RotamerTemplate(60.0, 0.15)),
    "G": (RotamerTemplate(60.0, 0.50), RotamerTemplate(-60.0, 0.35), RotamerTemplate(180.0, 0.15)),
    "H": (RotamerTemplate(-60.0, 0.44), RotamerTemplate(180.0, 0.38), RotamerTemplate(60.0, 0.18)),
    "I": (RotamerTemplate(-60.0, 0.52), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.14)),
    "K": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.20)),
    "L": (RotamerTemplate(-60.0, 0.48), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.17)),
    "M": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "N": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(60.0, 0.34), RotamerTemplate(180.0, 0.20)),
    "P": (RotamerTemplate(30.0, 0.48), RotamerTemplate(-30.0, 0.42), RotamerTemplate(180.0, 0.10)),
    "Q": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "R": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.35), RotamerTemplate(60.0, 0.20)),
    "S": (RotamerTemplate(-60.0, 0.46), RotamerTemplate(60.0, 0.34), RotamerTemplate(180.0, 0.20)),
    "T": (RotamerTemplate(-60.0, 0.50), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.16)),
    "V": (RotamerTemplate(-60.0, 0.52), RotamerTemplate(180.0, 0.34), RotamerTemplate(60.0, 0.14)),
    "W": (RotamerTemplate(-60.0, 0.44), RotamerTemplate(180.0, 0.41), RotamerTemplate(60.0, 0.15)),
    "Y": (RotamerTemplate(-60.0, 0.45), RotamerTemplate(180.0, 0.40), RotamerTemplate(60.0, 0.15)),
}

_SIDECHAIN_REACH: Mapping[str, float] = {
    "G": 1.6, "A": 1.8, "S": 2.4, "C": 2.5, "T": 2.6, "V": 2.8,
    "D": 3.0, "N": 3.1, "I": 3.2, "L": 3.3, "P": 2.6, "M": 3.7,
    "E": 3.8, "Q": 3.9, "H": 3.7, "F": 4.0, "Y": 4.2, "W": 4.5,
    "K": 4.6, "R": 4.8,
}

_NET_CHARGE: Mapping[str, float] = {
    "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1,
}

_FLEXIBILITY_RANK: Mapping[str, int] = {
    aa: rank for rank, aa in enumerate("GAPVITSCNDFYWHLMEQKR")
}

# Approximate side-chain torsional freedom used for adaptive state allocation.
_SIDECHAIN_CHI_COUNT: Mapping[str, int] = {
    "A": 0, "C": 1, "D": 2, "E": 3, "F": 2, "G": 0, "H": 2,
    "I": 2, "K": 4, "L": 2, "M": 3, "N": 2, "P": 2, "Q": 3,
    "R": 4, "S": 1, "T": 1, "V": 1, "W": 2, "Y": 2,
}


# Residue-specific sub-rotamer expansion.  The base three broad chi1 modes are
# deliberately expanded before any energy-based filtering so that candidate
# diversity is not artificially limited by the QUBO bit budget.
_SUBROTAMER_SCHEMES: Mapping[int, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    6: ((-12.0, 12.0), (0.50, 0.50)),
    9: ((-15.0, 0.0, 15.0), (0.25, 0.50, 0.25)),
    12: ((-22.0, -7.0, 7.0, 22.0), (0.15, 0.35, 0.35, 0.15)),
}


def _raw_rotamer_pool_size(amino_acid: str) -> int:
    """Return 6/9/12 raw candidates according to side-chain torsional freedom."""

    chi = _SIDECHAIN_CHI_COUNT.get(amino_acid, 1)
    if chi <= 1:
        return 6
    if chi == 2:
        return 9
    return 12


def _expanded_rotamer_templates(amino_acid: str) -> Tuple[RotamerTemplate, ...]:
    """Expand three broad chi1 modes into a normalized 6--12-state pool."""

    pool_size = _raw_rotamer_pool_size(amino_acid)
    offsets, weights = _SUBROTAMER_SCHEMES[pool_size]
    expanded = []
    for base in _ROTAMER_PRIORS[amino_acid]:
        for offset, weight in zip(offsets, weights):
            angle = ((base.chi1_degrees + offset + 180.0) % 360.0) - 180.0
            expanded.append(
                RotamerTemplate(
                    chi1_degrees=float(angle),
                    prior_probability=float(base.prior_probability * weight),
                )
            )
    total = sum(item.prior_probability for item in expanded)
    if len(expanded) != pool_size or total <= 0:
        raise RuntimeError(f"Invalid expanded rotamer pool for {amino_acid}")
    return tuple(
        RotamerTemplate(item.chi1_degrees, item.prior_probability / total)
        for item in expanded
    )
