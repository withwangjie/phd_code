#!/usr/bin/env python3
"""EGNN training-seed sensitivity (development data only).

Random seeds (initialization, data order, DDP partitions) are a major and
usually unreported source of variance in learned benchmarks (Bouthillier et
al., MLSys 2021). The formal pipeline keeps ONE preregistered EGNN checkpoint;
this analysis trains nothing itself and never selects a model. It compares the
primary checkpoint with replicate checkpoints trained on the same split with
different seeds, and reports:

* the spread of the best internal-validation ROC-AUC, and
* the stability of the formal Active-site selection (EGNN + antigen guidance,
  top-k chemically movable VHH residues) on the internal validation fold,
  as per-graph Jaccard overlap with the primary checkpoint's selection.

Only training-split graphs are read; the internal validation fold is rebuilt
with the same deterministic ``split_paths`` used for training. No test or
validation-queue data are touched.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from nanoqc.common.repo_io import sha256_file as sha256
from nanoqc.data.safe_graph_load import load_graph
from nanoqc.model import train_egnn_pruning as training
from nanoqc.model.model_egnn_pruning import load_interface_scorer, select_ablation_active


def jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    a, b = set(int(v) for v in left), set(int(v) for v in right)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def best_validation_auc(checkpoint: Path) -> float:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value = float((payload.get("early_stopping") or {}).get("best_auc", float("nan")))
    if not math.isfinite(value):
        raise ValueError(f"{checkpoint}: no finite best validation ROC-AUC")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Training-split graph directory")
    parser.add_argument("--primary-checkpoint", type=Path, required=True)
    parser.add_argument("--replicate-checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--active-sites", type=int, default=6)
    parser.add_argument("--antigen-guidance-weight", type=float, default=0.25)
    parser.add_argument("--antigen-proximity-scale", type=float, default=6.0)
    parser.add_argument("--contact-ca-cutoff", type=float, default=8.0)
    parser.add_argument("--vhh-identity-threshold", type=float, default=training.VHH_IDENTITY_THRESHOLD)
    parser.add_argument("--cdr-h3-identity-threshold", type=float, default=training.CDR_H3_IDENTITY_THRESHOLD)
    parser.add_argument("--antigen-identity-threshold", type=float, default=training.ANTIGEN_IDENTITY_THRESHOLD)
    parser.add_argument("--antigen-min-length-coverage", type=float, default=training.ANTIGEN_MIN_LENGTH_COVERAGE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args(argv)

    # Rebuild the exact internal validation fold used in training.
    training.VHH_IDENTITY_THRESHOLD = float(args.vhh_identity_threshold)
    training.CDR_H3_IDENTITY_THRESHOLD = float(args.cdr_h3_identity_threshold)
    training.ANTIGEN_IDENTITY_THRESHOLD = float(args.antigen_identity_threshold)
    training.ANTIGEN_MIN_LENGTH_COVERAGE = float(args.antigen_min_length_coverage)
    paths = sorted(args.data_dir.glob("*.pt"), key=lambda path: path.name.lower())
    if not paths:
        raise ValueError(f"No training graphs in {args.data_dir}")
    _, validation_paths = training.split_paths(paths, args.seed)

    checkpoints = [args.primary_checkpoint, *args.replicate_checkpoints]
    labels = ["primary", *[f"replicate_{i + 1}" for i in range(len(args.replicate_checkpoints))]]
    aucs = {label: best_validation_auc(path) for label, path in zip(labels, checkpoints)}
    scorers = {}
    for label, path in zip(labels, checkpoints):
        scorer, info = load_interface_scorer(path, torch_device=torch.device("cpu"), seed=args.seed)
        if info.status != "checkpoint_loaded":
            raise ValueError(f"{path}: {info.status} ({info.warning})")
        scorer.eval()
        scorers[label] = scorer

    per_graph = []
    skipped = []
    for path in validation_paths:
        data = load_graph(path)
        try:
            selections = {
                label: select_ablation_active(
                    data, "egnn", args.active_sites, args.seed, scorer,
                    antigen_guidance_weight=args.antigen_guidance_weight,
                    antigen_proximity_scale=args.antigen_proximity_scale,
                    contact_ca_cutoff=args.contact_ca_cutoff,
                ).tolist()
                for label, scorer in scorers.items()
            }
        except ValueError as exc:  # too few movable residues: not selectable at this size
            skipped.append(dict(graph=path.name, reason=str(exc)))
            continue
        per_graph.append(dict(
            graph=path.name,
            jaccard_vs_primary={label: jaccard(selections["primary"], selections[label])
                                for label in labels[1:]},
            pairwise_jaccard_mean=float(np.mean([
                jaccard(selections[a], selections[b]) for a, b in itertools.combinations(labels, 2)])),
            identical_to_primary={label: sorted(selections["primary"]) == sorted(selections[label])
                                  for label in labels[1:]},
        ))
    if not per_graph:
        raise ValueError("No internal-validation graph supports the requested Active-site count")

    vs_primary = [v for row in per_graph for v in row["jaccard_vs_primary"].values()]
    auc_values = np.asarray(list(aucs.values()), dtype=float)
    payload = dict(
        schema="egnn_seed_sensitivity_v1",
        scope="development only: training split, internal validation fold; never used for model selection",
        reference="Bouthillier et al., Accounting for Variance in Machine Learning Benchmarks, MLSys 2021",
        models=len(labels),
        checkpoints={label: dict(path=str(path), sha256=sha256(path)) for label, path in zip(labels, checkpoints)},
        best_validation_roc_auc=aucs,
        roc_auc_mean=float(auc_values.mean()),
        roc_auc_sd=float(auc_values.std(ddof=1)) if len(auc_values) > 1 else None,
        roc_auc_range=[float(auc_values.min()), float(auc_values.max())],
        active_sites=args.active_sites,
        validation_graphs=len(validation_paths),
        graphs_compared=len(per_graph),
        graphs_skipped=skipped,
        site_jaccard_vs_primary_mean=float(np.mean(vs_primary)),
        site_jaccard_vs_primary_min=float(np.min(vs_primary)),
        site_selection_identical_fraction=float(np.mean([
            v for row in per_graph for v in row["identical_to_primary"].values()])),
        site_jaccard_pairwise_mean=float(np.mean([row["pairwise_jaccard_mean"] for row in per_graph])),
        per_graph=per_graph,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# EGNN training-seed sensitivity (development only)", "",
        f"Models: {len(labels)} (primary + {len(labels) - 1} seed replicates, same training split).",
        f"Best internal-validation ROC-AUC: mean {payload['roc_auc_mean']:.4f}, "
        f"SD {payload['roc_auc_sd'] if payload['roc_auc_sd'] is None else round(payload['roc_auc_sd'], 4)}, "
        f"range {payload['roc_auc_range'][0]:.4f}-{payload['roc_auc_range'][1]:.4f}.",
        f"Active-site selection ({args.active_sites} sites) on {len(per_graph)} internal-validation graphs "
        f"({len(skipped)} skipped): Jaccard vs primary mean {payload['site_jaccard_vs_primary_mean']:.3f}, "
        f"min {payload['site_jaccard_vs_primary_min']:.3f}; identical selection in "
        f"{100 * payload['site_selection_identical_fraction']:.1f}% of comparisons.",
        "",
        "The formal pipeline uses only the preregistered primary checkpoint. These replicates "
        "quantify seed variance (Bouthillier et al. 2021) and are never used to choose a model.",
    ]
    args.out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
