"""Spawned-process helpers for build_final_pyg_dataset.

Each helper returns results in submission order, so the parent consumes them
exactly as the former sequential loops did. Workers first copy the parent's
command-line settings (``main`` rebinds them with ``global``) and load the same
audit annotations. Homology workers receive only the four attributes that
``layered_graph_homology`` reads.
"""
from __future__ import annotations

import concurrent.futures
import multiprocessing as mp
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import torch

# Every name in build_final_pyg_dataset.main's `global` statements except PEAK_RSS.
SETTINGS = (
    "RESUME", "VHH_IDENTITY_THRESHOLD", "CDR_H3_IDENTITY_THRESHOLD",
    "ANTIGEN_IDENTITY_THRESHOLD", "ANTIGEN_MIN_LENGTH_COVERAGE",
    "INTERFACE_LABEL_CUTOFF_ANGSTROM", "INTERFACE_SENSITIVITY_CUTOFFS_ANGSTROM",
    "INTRA_CHAIN_CA_CUTOFF_ANGSTROM", "CROSS_PARTNER_KNN_K", "MIN_INTERFACE_RESIDUES",
)
HOMOLOGY_FIELDS = ("vhh_sequences", "antigen_sequences", "cdr3_seq", "subset_source")

_REFERENCE: List[SimpleNamespace] = []


def _setup(settings: Dict[str, Any], annotations: Optional[tuple], reference: tuple) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    from nanoqc.data import audit_all_datasets as audit
    from nanoqc.data import build_final_pyg_dataset as build
    for name, value in settings.items():
        setattr(build, name, value)
    if annotations is not None:
        audit.install_annotations(annotations)
    _REFERENCE[:] = list(reference)


def pool(workers: int, annotations: bool = False,
         reference: Sequence[SimpleNamespace] = ()) -> concurrent.futures.ProcessPoolExecutor:
    """A spawned pool; ``annotations`` hands workers the parent's audit tables (A36).

    The parent has already loaded them, so the snapshot is passed rather than
    re-read: ``load_annotations`` walks the entire data root per process.
    """
    from nanoqc.data import audit_all_datasets as audit
    from nanoqc.data import build_final_pyg_dataset as build
    settings = {name: getattr(build, name) for name in SETTINGS}
    snapshot = audit.annotation_snapshot() if annotations else None
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, int(workers)), mp_context=mp.get_context("spawn"),
        initializer=_setup, initargs=(settings, snapshot, tuple(reference)))


def sequences(graph: Any) -> SimpleNamespace:
    """Copy only the attributes layered_graph_homology reads (absent stays absent)."""
    return SimpleNamespace(**{k: getattr(graph, k) for k in HOMOLOGY_FIELDS if hasattr(graph, k)})


def save_graph_task(args: tuple, kwargs: dict) -> tuple:
    """(record, error, this worker's sampled peak RSS)."""
    from nanoqc.data import build_final_pyg_dataset as build
    record, error = build.save_graph(*args, **kwargs)
    return record, error, int(build.PEAK_RSS)


def homology_row(item: SimpleNamespace, stop: Optional[int] = None) -> List[dict]:
    """layered_graph_homology(item, reference[j]) for j in reference order (up to ``stop``)."""
    from nanoqc.data import build_final_pyg_dataset as build
    refs = _REFERENCE if stop is None else _REFERENCE[:stop]
    return [build.layered_graph_homology(item, ref) for ref in refs]


def reference_row(index: int) -> List[dict]:
    """Row ``index`` of the lower-triangular reference-vs-reference homology table."""
    return homology_row(_REFERENCE[index], index)
