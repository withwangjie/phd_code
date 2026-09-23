#!/usr/bin/env python3
"""seed_streams.py -- master-seed -> independent, saved, reproducible random streams.

Single source of truth for turning ONE master seed (20260917 by default, the
same literal value ``build_final_pyg_dataset.py`` and ``train_egnn_pruning.py``
already hardcode) into independent, named sub-seeds for every stage of
``run_full_experiment.py`` that consumes randomness on its own: dataset
PARTITION (train/test_snac_hard cluster split and shuffle order), EGNN TRAIN
(weight init, loader shuffling, DDP-rank offsets), structural PERTURB (chi1
perturbation degrees, validation-queue seeded-random target ordering),
optimizer/OPTIMIZE (QAOA parameter initialization / COBYLA restart starting
points), in-search MEASUREMENT (the finite-shot CVaR/mean draws
``optimize_robust`` takes WHILE searching, to score each candidate parameter
point -- independent of both the search's own initialization/restart RNG
and the final output draw below), and final-output SAMPLE (the finite-shot
bitstring draw actually reported as each method's output, and every
classical baseline's own RNG).

Before this module, ``build_final_pyg_dataset.py`` and ``train_egnn_pruning.py``
both consumed the bare literal ``20260917`` directly -- i.e. partition and
train were NOT independent streams, they were the same number reused. This
module derives six independently derived child streams from one
``numpy.random.SeedSequence(master_seed)`` (via ``.spawn()``, the standard
NumPy-recommended way to get non-overlapping streams from one root seed) and
persists the full derivation as JSON, so any run's exact seed provenance is
reproducible and auditable without re-deriving anything.

Usage as a library (from ``run_full_experiment.py``)::

    from seed_streams import derive_streams, derive_child_seed, save_stream_map
    streams = derive_streams(20260917)
    save_stream_map(run_dir / "seed_streams.json", 20260917, streams)
    partition_seed = streams["partition"]
    # A per-target, per-purpose sub-seed (e.g. one perturbation seed per
    # validation-queue target), independent across targets/purposes too:
    target_seed = derive_child_seed(streams["perturb"], "perturb", "8YVO")

Usage from the command line (inspect/regenerate a mapping standalone)::

    python -m nanoqc.common.seed_streams --master-seed 20260917 --out seed_streams.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# Fixed, ordered stream names. Order matters: streams are spawned from the
# master SeedSequence in this exact order, so changing the order (or adding
# a name in the middle rather than at the end) would silently change every
# existing stream's derived value -- append new names at the end only, once
# a run has actually executed against a given list. "measurement" was added
# between "optimize" and "sample" (its documented pipeline position) before
# any pipeline execution occurred this session, so no already-produced
# results were invalidated by the reordering; any future addition must be
# appended at the end instead.
STREAM_NAMES: List[str] = ["partition", "train", "perturb", "optimize", "measurement", "sample"]

# The literal value already hardcoded (as SEED = 20260917) in
# build_final_pyg_dataset.py and train_egnn_pruning.py before this module
# existed. Kept here as the documented default, never silently changed.
DEFAULT_MASTER_SEED = 20260917

_UINT32_MAX = 2**32 - 1


def _seedsequence_to_int(seq: "np.random.SeedSequence") -> int:
    """One reproducible, plain, non-negative 32-bit int from a SeedSequence.

    Suitable directly as a CLI ``--seed``/``--seeds`` integer for every
    existing entrypoint in this project (argparse ``type=int``), which is
    why this collapses a SeedSequence (NumPy's own richer object) down to a
    single plain int rather than exposing the object itself.
    """
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def derive_streams(master_seed: int = DEFAULT_MASTER_SEED) -> Dict[str, int]:
    """Derive the six named top-level streams from one master seed.

    Returns a dict, in ``STREAM_NAMES`` order, mapping each stream name to
    one plain int seed. Deterministic: calling this twice with the same
    ``master_seed`` always returns the same mapping.
    """
    if not isinstance(master_seed, int) or master_seed < 0:
        raise ValueError("master_seed must be a non-negative int")
    root = np.random.SeedSequence(master_seed)
    children = root.spawn(len(STREAM_NAMES))
    return {name: _seedsequence_to_int(child) for name, child in zip(STREAM_NAMES, children)}


def derive_child_seed(stream_seed: int, *labels: str) -> int:
    """A further-independent sub-seed for one (stream, *labels) combination.

    Used to give e.g. each validation-queue target its own perturbation seed
    that is independent both of every other target AND of the top-level
    stream seed itself, while remaining fully deterministic and reproducible
    from (stream_seed, labels) alone -- no separate bookkeeping is required
    to reconstruct it later, though ``save_stream_map`` still records
    explicitly-requested children for auditability.

    Labels are folded into the SeedSequence's entropy pool as stable 32-bit
    integers (via a truncated SHA-256 digest of each label, not Python's
    salted ``hash()``, which is not reproducible across processes/runs).
    """
    if not labels:
        raise ValueError("At least one label is required to derive a child seed")
    entropy: List[int] = [int(stream_seed)]
    for label in labels:
        digest = hashlib.sha256(str(label).encode("utf-8")).digest()[:4]
        entropy.append(int.from_bytes(digest, "big"))
    return _seedsequence_to_int(np.random.SeedSequence(entropy))


def save_stream_map(
    path: Path,
    master_seed: int,
    streams: Dict[str, int],
    *,
    children: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Persist the full derivation as indented, human-readable JSON.

    ``children`` (optional) records any ``derive_child_seed`` calls already
    made for this run, keyed first by stream name then by a
    ``"|".join(labels)`` string, purely for audit/debugging -- children are
    always independently reproducible from ``streams`` + labels alone, so
    this is a convenience record, not load-bearing state.
    """
    payload = {
        "master_seed": int(master_seed),
        "derivation_method": "numpy.random.SeedSequence(master_seed).spawn(len(STREAM_NAMES))",
        "stream_names_in_spawn_order": STREAM_NAMES,
        "streams": {name: int(seed) for name, seed in streams.items()},
        "children": {
            stream: {label: int(seed) for label, seed in mapping.items()}
            for stream, mapping in (children or {}).items()
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_stream_map(path: Path) -> Dict[str, object]:
    """Read back a mapping written by ``save_stream_map``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_stream_map(path: Path) -> bool:
    """Recompute every stream in a saved mapping and confirm it still matches.

    Used by ``run_full_experiment.py`` on ``--resume`` to guarantee a
    continued run derives exactly the same seeds as the original launch
    (rather than silently drifting if this module's derivation logic ever
    changes between the original launch and the resume attempt).
    """
    payload = load_stream_map(path)
    recomputed = derive_streams(int(payload["master_seed"]))
    return recomputed == {k: int(v) for k, v in payload["streams"].items()}


def _main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-seed", type=int, default=DEFAULT_MASTER_SEED)
    parser.add_argument("--out", type=Path, default=Path("seed_streams.json"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    streams = derive_streams(args.master_seed)
    save_stream_map(args.out, args.master_seed, streams)
    print(f"Master seed {args.master_seed} -> {streams}")
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
