"""Shared helpers for the A64 supplementary experiments.

Everything here reads a frozen run directory and never writes inside it except
under ``<run>/analysis/a64/``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from nanoqc.solvers.qaoa_interface_sampler import XYMixerQAOASampler

# Frozen primary matched-output QAOA variant whose fitted angles are evaluated.
PRIMARY_OPTIMIZATION_KEY = "qaoa_objcvar_restarts4_outputs1000"
PRIMARY_FILTER = dict(pruning="egnn", radius=6.0, depth=2, max_evals=90)
# Same tolerance the formal benchmark uses to call a state a ground state.
GROUND_ENERGY_TOLERANCE = 1e-9


def output_dir(run_dir: Path, name: str) -> Path:
    path = Path(run_dir) / "analysis" / "a64" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def primary_cases(run_dir: Path, sites: set[int] | None = None,
                  one_per_pdb: bool = True) -> Iterator[tuple[Path, dict]]:
    """Frozen primary-filter coarse cases, deterministic order.

    With ``one_per_pdb`` the first case (sorted file name) of each PDB and size
    is used, so a PDB is never weighted by how many repeats it has.
    """
    seen = set()
    for path in sorted((Path(run_dir) / "qc_benchmark" / "cases").glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        config = case.get("config") or {}
        if any(_differs(config.get(k), v) for k, v in PRIMARY_FILTER.items()):
            continue
        n_sites = int(config.get("active_sites", 0))
        if sites is not None and n_sites not in sites:
            continue
        if PRIMARY_OPTIMIZATION_KEY not in (case.get("optimizations") or {}):
            continue
        pdb = str(config.get("pdb_id") or config.get("target") or path.stem).lower()
        if one_per_pdb:
            if (pdb, n_sites) in seen:
                continue
            seen.add((pdb, n_sites))
        yield path, case


def _differs(observed, expected) -> bool:
    if observed is None:
        return True
    if isinstance(expected, float):
        return not math.isclose(float(observed), expected)
    if isinstance(expected, int):
        return int(observed) != expected
    return str(observed).lower() != str(expected).lower()


def sampler_from_case(case: dict) -> XYMixerQAOASampler:
    """Rebuild the exact solver-facing sampler from the stored quantum instance."""
    instance = case["quantum_instance"]
    sites = {int(k) if str(k).lstrip("-").isdigit() else k: list(v)
             for k, v in instance["site_to_variables"].items()}
    depth = int((case.get("config") or {}).get("depth", 2))
    return XYMixerQAOASampler(
        np.asarray(instance["physical_self"], dtype=float),
        np.asarray(instance["physical_pair"], dtype=float),
        sites, p=depth, simulation_mode="subspace")


def primary_angles(case: dict) -> np.ndarray:
    optimization = case["optimizations"][PRIMARY_OPTIMIZATION_KEY]
    return np.concatenate([np.asarray(optimization["gammas"], dtype=float),
                           np.asarray(optimization["betas"], dtype=float)])


def feasible_ground(sampler: XYMixerQAOASampler) -> tuple[list[tuple[int, ...]], float, int]:
    """Ground bitstrings, ground energy and feasible-space size."""
    energies = sampler.feasible_energy_map()
    minimum = min(energies.values())
    ground = [tuple(int(b) for b in bits) for bits, value in energies.items()
              if abs(value - minimum) <= GROUND_ENERGY_TOLERANCE]
    return ground, float(minimum), len(energies)


def write_csv(path: Path, rows: list[dict]) -> None:
    import csv
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
