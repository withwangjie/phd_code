"""Supervised training for the EGNN residue-interface pruning scorer.

Labels are stored during graph construction from a versioned inter-partner
heavy-atom cutoff recorded in each graph. Cross-partner graph connectivity is
fixed-KNN rather than contact-threshold connectivity, so edge existence does
not reconstruct the target rule. Training and validation are separated by
joint connected components using the configured bilateral full-chain VHH /
antigen identity threshold.

Linux dual-T4 launch example::

    torchrun --standalone --nproc_per_node=2 train_egnn_pruning.py \
        --device cuda --num-workers 8 --threads 8

Each rank preloads all graphs once into host RAM. Rank-local
``DistributedSampler`` instances partition training batches, while rank 0 owns
validation, early stopping, history, and checkpoint writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch_geometric.data import Data
from torch.utils.data.distributed import DistributedSampler

from nanoqc.model.model_egnn_pruning import EGNNInterfaceScorer
from nanoqc.common.repo_io import sha256_file as _file_sha256
from nanoqc.data.sequence_identity import partner_orientations

# Names from the split-out modules that this module or its callers use.
from nanoqc.model.egnn_metrics import (  # noqa: E402,F401
    BinaryMetrics,
    fit_geometry_logistic_baseline,
    count_training_labels,
)
from nanoqc.model.egnn_graph_data import (  # noqa: E402,F401
    InterfaceGraphDataset,
    graph_protocol,
    preload_graphs,
    seed_everything,
    _make_loader,
)
from nanoqc.model.egnn_training_loop import (  # noqa: E402,F401
    train_one_epoch,
    validate,
    _initialize_distributed,
    _broadcast_metrics,
    _atomic_torch_save,
    _write_history,
    verify_checkpoint,
)
from nanoqc.model.egnn_split_records import (  # noqa: E402,F401
    _sequence_identity,
    _side_identity,
    _split_records,
    _family_structure_assignment_digest,
    _paths_digest,
)


SEED = 4050350448


VHH_IDENTITY_THRESHOLD = 0.80
CDR_H3_IDENTITY_THRESHOLD = 0.50
ANTIGEN_IDENTITY_THRESHOLD = 0.30
ANTIGEN_MIN_LENGTH_COVERAGE = 0.70


SPLIT_FOLDS = 5
VALIDATION_FOLD = 0
# A component holding more than one fold's share of the pool cannot be held
# out without becoming most of that fold; it always trains (A11). The floor
# keeps small pools on the plain hash.
PIN_MIN_COMPONENT = 50


def _components_from_records(records: Sequence[Tuple]) -> List[List[Path]]:
    """Union-find over the layered isolation edges between loaded records."""
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for i in range(len(records)):
        _, vhh_i, antigen_i, cdr_i, family_i, anchored_i = records[i]
        for j in range(i):
            _, vhh_j, antigen_j, cdr_j, family_j, anchored_j = records[j]
            # Unanchored partner roles (e.g. train_rcsb) are also compared with
            # partners swapped, so cross-role homology is not missed.
            same_vhh = same_antigen = False
            for lv, rv, la, ra in partner_orientations(
                    vhh_i, antigen_i, anchored_i, vhh_j, antigen_j, anchored_j):
                same_vhh = same_vhh or _side_identity(lv, rv) >= VHH_IDENTITY_THRESHOLD
                same_antigen = same_antigen or (
                    _side_identity(la, ra, min_length_coverage=ANTIGEN_MIN_LENGTH_COVERAGE)
                    >= ANTIGEN_IDENTITY_THRESHOLD
                )
            same_cdr = (
                bool(cdr_i and cdr_j)
                and _sequence_identity(cdr_i, cdr_j) >= CDR_H3_IDENTITY_THRESHOLD
            )
            same_family_structure = family_i == family_j
            if same_vhh or same_cdr or same_antigen or same_family_structure:
                union(i, j)

    components: Dict[int, List[int]] = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    if len(components) < 2:
        raise RuntimeError("Layered VHH/CDR-H3/antigen/family clustering produced fewer than two components")
    return [[records[i][0] for i in indices]
            for _, indices in sorted(components.items(),
                                     key=lambda item: min(records[i][0].name for i in item[1]))]


def layered_components(paths: Sequence[Path]) -> List[List[Path]]:
    """Connected components of the layered isolation relation, ordered by first name.

    Two complexes are joined when any criterion fires: VHH full-chain identity,
    CDR-H3 loop identity, antigen full-chain identity with coverage, or a shared
    frozen family/structure cluster. A component may therefore never be split
    across train, internal validation or the antigen-fold holdout.
    """
    return _components_from_records(_split_records(paths))


def component_fold(component: Sequence[Path]) -> int:
    """Deterministic fold of a layered component, from its member file names."""
    signature = "\n".join(sorted(path.name for path in component)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(signature).digest()[:8], "big") % SPLIT_FOLDS


def pinned_to_training(component: Sequence[Path], pool_size: int) -> bool:
    """True for a component larger than 1/SPLIT_FOLDS of the pool (and >= PIN_MIN_COMPONENT)."""
    return len(component) >= PIN_MIN_COMPONENT and len(component) * SPLIT_FOLDS > pool_size


def assigned_fold(component: Sequence[Path], pool_size: int) -> Optional[int]:
    """The component's held-out fold, or None when it is pinned to training."""
    return None if pinned_to_training(component, pool_size) else component_fold(component)


def split_paths(paths: Sequence[Path], seed: int) -> Tuple[List[Path], List[Path]]:
    """Layered sequence + family/structure component split; no random 90/10 partition."""
    del seed
    records = _split_records(paths)
    ordered_components = _components_from_records(records)
    train_paths: List[Path] = []
    validation_paths: List[Path] = []
    for component in ordered_components:
        target = validation_paths if assigned_fold(component, len(paths)) == VALIDATION_FOLD else train_paths
        target.extend(component)

    train_paths = sorted(train_paths, key=lambda path: path.name.lower())
    validation_paths = sorted(validation_paths, key=lambda path: path.name.lower())
    if not validation_paths:
        chosen = set(ordered_components[0])
        validation_paths = sorted(chosen, key=lambda p: p.name.lower())
        train_paths = sorted([p for p in paths if p not in chosen], key=lambda p: p.name.lower())
    if not train_paths:
        chosen = set(ordered_components[-1])
        train_paths = sorted(chosen, key=lambda p: p.name.lower())
        validation_paths = sorted([p for p in paths if p not in chosen], key=lambda p: p.name.lower())
    if not train_paths or not validation_paths:
        raise RuntimeError("Bilateral component split produced an empty partition")

    train_set, validation_set = set(train_paths), set(validation_paths)
    max_vhh_cross = 0.0
    max_cdr_cross = 0.0
    max_antigen_cross = 0.0
    train_families=set()
    validation_families=set()
    for path, _vhh, _antigen, _cdr, family, _anchored in records:
        if path in train_set:
            train_families.add(family)
        if path in validation_set:
            validation_families.add(family)
    family_overlap=sorted(train_families & validation_families)
    for train_path, train_vhh, train_antigen, train_cdr, _train_family, train_anchored in records:
        if train_path not in train_set:
            continue
        for val_path, val_vhh, val_antigen, val_cdr, _val_family, val_anchored in records:
            if val_path not in validation_set:
                continue
            # Same partner orientations as the component union above, so the
            # post-hoc audit checks exactly the criterion used to build the split.
            for lv, rv, la, ra in partner_orientations(
                    train_vhh, train_antigen, train_anchored,
                    val_vhh, val_antigen, val_anchored):
                max_vhh_cross = max(max_vhh_cross, _side_identity(lv, rv))
                max_antigen_cross = max(
                    max_antigen_cross,
                    _side_identity(la, ra, min_length_coverage=ANTIGEN_MIN_LENGTH_COVERAGE),
                )
            if train_cdr and val_cdr:
                max_cdr_cross = max(max_cdr_cross, _sequence_identity(train_cdr, val_cdr))
    if (max_vhh_cross >= VHH_IDENTITY_THRESHOLD
            or max_cdr_cross >= CDR_H3_IDENTITY_THRESHOLD
            or max_antigen_cross >= ANTIGEN_IDENTITY_THRESHOLD
            or family_overlap):
        raise AssertionError(
            "Layered sequence/family leakage across train/validation: "
            f"max VHH full-chain={max_vhh_cross:.3f}, max CDR-H3 loop={max_cdr_cross:.3f}, "
            f"max antigen full-chain={max_antigen_cross:.3f}, family_overlap={family_overlap[:10]}"
        )
    return train_paths, validation_paths


def run_split_identity(
    train_paths: Sequence[Path], validation_paths: Sequence[Path]
) -> Dict[str, Any]:
    """Split digests and graph protocol, which are fixed for a whole run (A37).

    Recomputing these inside every epoch's checkpoint payload re-read every
    training and validation graph from disk each epoch, although the file set and
    its frozen family/structure assignment cannot change during a run (a resume
    compares these same digests). Computing them once also surfaces missing
    PDB/family metadata before the first epoch instead of after it.
    """
    return {
        "train_names_sha256": _paths_digest(train_paths),
        "validation_names_sha256": _paths_digest(validation_paths),
        "family_structure_assignment_sha256": _family_structure_assignment_digest(
            [*train_paths, *validation_paths]
        ),
        "graph_protocol": graph_protocol(
            torch.load(train_paths[0], map_location="cpu", weights_only=False)),
    }


def _checkpoint_payload(
    *,
    model: EGNNInterfaceScorer,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    epoch: int,
    metrics: BinaryMetrics,
    train_loss: float,
    model_config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
    history: Sequence[Mapping[str, Any]],
    best_auc: float,
    best_epoch: int,
    epochs_without_improvement: int,
    train_positives: int,
    train_negatives: int,
    split_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    identity = dict(split_identity) if split_identity is not None else run_split_identity(
        train_paths, validation_paths)
    return {
        "format_version": 2,
        "model_class": "EGNNInterfaceScorer",
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_metrics": asdict(metrics),
        "training_config": dict(training_config),
        "split": {
            "partition_method": "layered_vhh_cdrh3_antigen_family_components_sha256_5fold",
            "homology_isolation": {
                "vhh_full_chain_identity": float(VHH_IDENTITY_THRESHOLD),
                "cdr_h3_identity": float(CDR_H3_IDENTITY_THRESHOLD),
                "antigen_identity": float(ANTIGEN_IDENTITY_THRESHOLD),
                "antigen_min_length_coverage": float(ANTIGEN_MIN_LENGTH_COVERAGE),
            },
            "partition_seed": None,
            "training_seed": SEED,
            "train_count": len(train_paths),
            "validation_count": len(validation_paths),
            "train_names_sha256": identity["train_names_sha256"],
            "validation_names_sha256": identity["validation_names_sha256"],
            "family_structure_assignment_sha256": identity["family_structure_assignment_sha256"],
        },
        "graph_protocol": identity["graph_protocol"],
        "label_definition": "inter-partner heavy-atom cutoff label stored independently during graph construction",
        "history": list(history),
        "early_stopping": {
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
        "label_counts": {
            "train_positives": train_positives,
            "train_negatives": train_negatives,
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }


def restore_training_state(
    checkpoint_path: Path,
    *,
    model: EGNNInterfaceScorer,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    model_config: Mapping[str, Any],
    train_paths: Sequence[Path],
    validation_paths: Sequence[Path],
) -> Tuple[int, List[Dict[str, Any]], float, int, int]:
    """Restore a completed epoch and verify model/split compatibility."""

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if dict(payload.get("model_config", {})) != dict(model_config):
        raise RuntimeError(
            "Resume checkpoint model_config does not match current arguments: "
            f"saved={payload.get('model_config')}, current={dict(model_config)}"
        )
    split = payload.get("split", {})
    expected_train_hash = _paths_digest(train_paths)
    expected_validation_hash = _paths_digest(validation_paths)
    if split.get("train_names_sha256") != expected_train_hash or split.get(
        "validation_names_sha256"
    ) != expected_validation_hash:
        raise RuntimeError("Resume checkpoint uses a different graph split")
    current_family_digest=_family_structure_assignment_digest([*train_paths,*validation_paths])
    if split.get("family_structure_assignment_sha256") != current_family_digest:
        raise RuntimeError(
            "Resume checkpoint family/structure assignments differ from the current graphs"
        )
    saved_homology = split.get("homology_isolation")
    current_homology = {
        "vhh_full_chain_identity": float(VHH_IDENTITY_THRESHOLD),
        "cdr_h3_identity": float(CDR_H3_IDENTITY_THRESHOLD),
        "antigen_identity": float(ANTIGEN_IDENTITY_THRESHOLD),
        "antigen_min_length_coverage": float(ANTIGEN_MIN_LENGTH_COVERAGE),
    }
    if saved_homology != current_homology:
        raise RuntimeError(
            f"Resume checkpoint homology protocol mismatch: saved={saved_homology}, "
            f"current={current_homology}"
        )
    current_protocol = graph_protocol(
        torch.load(train_paths[0], map_location="cpu", weights_only=False)
    )
    if payload.get("graph_protocol") != current_protocol:
        raise RuntimeError(
            f"Resume checkpoint graph protocol mismatch: saved={payload.get('graph_protocol')}, "
            f"current={current_protocol}"
        )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if payload.get("grad_scaler_state_dict"):
        scaler.load_state_dict(payload["grad_scaler_state_dict"])
    history = [dict(row) for row in payload.get("history", [])]
    early = payload.get("early_stopping", {})
    if history:
        fallback_auc = max(float(row["roc_auc"]) for row in history)
        fallback_best_epoch = int(
            max(history, key=lambda row: float(row["roc_auc"]))["epoch"]
        )
    else:
        fallback_auc = float(payload["validation_metrics"]["roc_auc"])
        fallback_best_epoch = int(payload["epoch"])
    best_auc = float(early.get("best_auc", fallback_auc))
    best_epoch = int(early.get("best_epoch", fallback_best_epoch))
    epochs_without = int(early.get("epochs_without_improvement", 0))
    rng = payload.get("rng_state", {})
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and rng.get("cuda"):
            current_device = torch.cuda.current_device()
            saved_cuda_states = rng["cuda"]
            state_index = min(current_device, len(saved_cuda_states) - 1)
            torch.cuda.set_rng_state(saved_cuda_states[state_index], current_device)
    start_epoch = int(payload["epoch"]) + 1
    return start_epoch, history, best_auc, best_epoch, epochs_without


def write_summary_json(
    path: Path,
    *,
    total_epochs: int,
    best_payload: Mapping[str, Any],
    best_checkpoint: Path,
    total_seconds: float,
    train_count: int,
    validation_count: int,
    train_positives: int,
    train_negatives: int,
    resumed_from: Optional[Path],
) -> None:
    """Write the required machine-readable final convergence summary."""

    metrics = dict(best_payload["validation_metrics"])
    payload = {
        "status": "complete",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_epochs": total_epochs,
        "best_epoch": int(best_payload["epoch"]),
        "best_validation": {
            "bce_with_logits_loss": float(metrics["loss"]),
            "roc_auc": float(metrics["roc_auc"]),
            "pr_auc": float(metrics["pr_auc"]),
            "best_threshold": float(metrics["threshold"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1_score": float(metrics["f1"]),
            "threshold_0_5": {
                "precision": float(metrics["default_precision"]),
                "recall": float(metrics["default_recall"]),
                "f1_score": float(metrics["default_f1"]),
            },
        },
        "split": {
            "partition_method": "layered_vhh_cdrh3_antigen_family_components_sha256_5fold",
            "partition_seed": None,
            "training_seed": SEED,
            "train_graphs": train_count,
            "validation_graphs": validation_count,
        },
        "labels": {
            "definition": best_payload["label_definition"],
            "task_scope": "known-pose heavy-atom interface classification from residue-level geometry",
            "recoverable_from_input_edges": False,
            "unknown_interface_prediction_validated": False,
            "validation_scope": "known-pose, layered VHH/CDR-H3/antigen plus family/structure-isolated components",
            "deterministic_baseline": (
                "cross-edge existence cannot reconstruct labels; fixed-KNN cross edges and "
                "heavy-atom cutoff labels are stored as separate graph-protocol fields"
            ),
            "train_positives": train_positives,
            "train_negatives": train_negatives,
        },
        "runtime_seconds_this_invocation": total_seconds,
        "resumed_from": str(resumed_from.resolve()) if resumed_from else None,
        "model_config": dict(best_payload["model_config"]),
        "training_config": dict(best_payload["training_config"]),
        "checkpoint": {
            "path": str(best_checkpoint.resolve()),
            "bytes": best_checkpoint.stat().st_size,
            "sha256": _file_sha256(best_checkpoint),
            "strict_reload_verified": True,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def write_summary(
    path: Path,
    *,
    total_epochs: int,
    best_payload: Mapping[str, Any],
    best_checkpoint: Path,
    total_seconds: float,
    train_count: int,
    validation_count: int,
    train_positives: int,
    train_negatives: int,
) -> None:
    metrics = best_payload["validation_metrics"]
    lines = [
        "# EGNN 界面打分网络训练摘要",
        "",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 收敛结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 实际训练 Epoch | {total_epochs} |",
        f"| 最佳 Epoch | {best_payload['epoch']} |",
        f"| 最佳验证 BCEWithLogitsLoss | {metrics['loss']:.6f} |",
        f"| 最佳验证 ROC-AUC | {metrics['roc_auc']:.6f} |",
        f"| 最佳验证 PR-AUC | {metrics['pr_auc']:.6f} |",
        f"| 最佳 F1 阈值 | {metrics['threshold']:.6f} |",
        f"| Precision（最佳 F1 阈值） | {metrics['precision']:.6f} |",
        f"| Recall（最佳 F1 阈值） | {metrics['recall']:.6f} |",
        f"| F1-score（最佳 F1 阈值） | {metrics['f1']:.6f} |",
        f"| Precision（阈值 0.5） | {metrics['default_precision']:.6f} |",
        f"| Recall（阈值 0.5） | {metrics['default_recall']:.6f} |",
        f"| F1-score（阈值 0.5） | {metrics['default_f1']:.6f} |",
        f"| 训练耗时 | {total_seconds:.1f} s |",
        "",
        "## 数据与监督定义",
        "",
        f"- 分层同源隔离：训练 {train_count}，验证 {validation_count}；VHH≥{VHH_IDENTITY_THRESHOLD:.2f}、CDR-H3≥{CDR_H3_IDENTITY_THRESHOLD:.2f}、或抗原≥{ANTIGEN_IDENTITY_THRESHOLD:.2f}且长度覆盖≥{ANTIGEN_MIN_LENGTH_COVERAGE:.2f}时并入同一连通分量。",
        "- 分量按SHA-256确定性映射到5个fold，fold 0用于验证；不使用随机90/10切分。",
        f"- 训练节点标签：正例 {train_positives:,}，负例 {train_negatives:,}。",
        f"- 图协议：{json.dumps(best_payload.get('graph_protocol', {}), ensure_ascii=False, sort_keys=True)}",
        "- 界面标签与图边独立构建；固定KNN跨伙伴边不复用heavy-atom标签阈值。",
        "- 因此跨伙伴edge existence不再能确定性重建界面标签；仍需通过独立验证AUC/PR-AUC评估泛化。",
        "",
        "## 权重完整性",
        "",
        f"- 最佳权重：`{best_checkpoint.resolve()}`",
        f"- 文件大小：{best_checkpoint.stat().st_size:,} bytes",
        "- 已重新加载并以 `strict=True` 验证全部参数键和张量形状。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("./dataset_clean/graphs/train"))
    parser.add_argument(
        "--expected-graphs",
        type=int,
        default=6904,
        help="Fail fast unless this many graph files are present.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints"))
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--coord-scale", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--geometry-baseline-contact-cutoff", type=float, default=8.0)
    parser.add_argument("--geometry-baseline-proximity-scale", type=float, default=6.0)
    parser.add_argument("--vhh-identity-threshold", type=float, default=VHH_IDENTITY_THRESHOLD)
    parser.add_argument("--cdr-h3-identity-threshold", type=float, default=CDR_H3_IDENTITY_THRESHOLD)
    parser.add_argument("--antigen-identity-threshold", type=float, default=ANTIGEN_IDENTITY_THRESHOLD)
    parser.add_argument("--antigen-min-length-coverage", type=float, default=ANTIGEN_MIN_LENGTH_COVERAGE)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable CUDA float16 autocast and GradScaler (default: disabled; the formal "
             "protocol trains in FP32, PROTOCOL_AMENDMENTS.md A20).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="Resume from CHECKPOINT, or from checkpoints/last_egnn_pruning.pt when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Overrides the module-level SEED constant (default: the project's master seed, "
             "4050350448) for weight init, loader shuffling and DDP-rank seed offsets; pass an "
             "independently-derived train-stream seed here rather than reusing the bare master seed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    global SEED, VHH_IDENTITY_THRESHOLD, CDR_H3_IDENTITY_THRESHOLD
    global ANTIGEN_IDENTITY_THRESHOLD, ANTIGEN_MIN_LENGTH_COVERAGE
    SEED = args.seed
    thresholds = (
        args.vhh_identity_threshold, args.cdr_h3_identity_threshold,
        args.antigen_identity_threshold, args.antigen_min_length_coverage,
    )
    if any(not 0.0 < value <= 1.0 for value in thresholds):
        raise ValueError("homology thresholds/coverage must lie in (0,1]")
    VHH_IDENTITY_THRESHOLD = float(args.vhh_identity_threshold)
    CDR_H3_IDENTITY_THRESHOLD = float(args.cdr_h3_identity_threshold)
    ANTIGEN_IDENTITY_THRESHOLD = float(args.antigen_identity_threshold)
    ANTIGEN_MIN_LENGTH_COVERAGE = float(args.antigen_min_length_coverage)  # Every existing seed_everything(SEED + rank) / training_summary.json
                       # "seed": SEED usage below transparently picks up the CLI override.
    if args.max_epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("max-epochs, patience, and batch-size must be positive")
    if args.hidden_dim <= 0 or args.num_layers <= 0 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("hidden-dim, num-layers and learning-rate must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if not math.isfinite(args.coord_scale) or args.coord_scale <= 0:
        raise ValueError("coord-scale must be positive finite")
    if (not math.isfinite(args.geometry_baseline_contact_cutoff)
            or args.geometry_baseline_contact_cutoff <= 0
            or not math.isfinite(args.geometry_baseline_proximity_scale)
            or args.geometry_baseline_proximity_scale <= 0):
        raise ValueError("Geometry baseline distance parameters must be positive finite")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight-decay must be finite and nonnegative")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        raise ValueError("gradient-clip must be positive and finite")
    if args.num_workers < 0 or args.threads <= 0:
        raise ValueError("num-workers must be nonnegative and threads must be positive")
    rank, local_rank, world_size, device = _initialize_distributed(args.device)
    is_main_process = rank == 0
    paths = sorted(args.data_dir.glob("*.pt"), key=lambda path: path.name.lower())
    if args.expected_graphs <= 0:
        raise ValueError("--expected-graphs must be positive")
    if len(paths) != args.expected_graphs:
        raise RuntimeError(
            f"Expected {args.expected_graphs} training graphs, found {len(paths)}"
        )
    torch.set_num_threads(args.threads)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    seed_everything(SEED + rank)
    train_paths, validation_paths = split_paths(paths, SEED)
    if is_main_process:
        print(
            f"Graph split: train={len(train_paths)}, validation={len(validation_paths)}, "
            f"seed={SEED}, world_size={world_size}, device={device}, "
            f"threads/rank={torch.get_num_threads()}, workers/rank={args.num_workers}"
        )

    # Every rank owns a complete RAM-resident list. Linux fork/CoW keeps the
    # immutable graph tensors largely shared; all epochs after this point are
    # free of filesystem reads. A barrier prevents rank 0 from entering
    # validation while a peer is still loading.
    data_list: List[Data] = preload_graphs(paths, show_progress=is_main_process)
    if dist.is_initialized():
        dist.barrier()
    path_to_index = {path: index for index, path in enumerate(paths)}
    train_data = [data_list[path_to_index[path]] for path in train_paths]
    validation_data = [data_list[path_to_index[path]] for path in validation_paths]
    geometry_baseline = None
    if is_main_process:
        geometry_baseline = fit_geometry_logistic_baseline(
            train_data, validation_data,
            contact_cutoff=args.geometry_baseline_contact_cutoff,
            proximity_scale=args.geometry_baseline_proximity_scale,
        )
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (args.checkpoint_dir/"geometry_baseline.json").write_text(
            json.dumps(geometry_baseline,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"
        )
    if is_main_process:
        print(f"RAM preload complete: {len(data_list):,} graphs on rank {rank}")

    if is_main_process:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    best_path = args.checkpoint_dir / "best_egnn_pruning.pt"
    last_path = args.checkpoint_dir / "last_egnn_pruning.pt"
    history_path = args.checkpoint_dir / "egnn_training_history.csv"
    summary_path = args.checkpoint_dir / "egnn_training_summary.md"
    summary_json_path = args.checkpoint_dir / "training_summary.json"

    label_counts = torch.zeros(2, dtype=torch.long, device=device)
    if is_main_process:
        positive, negative = count_training_labels(train_data)
        label_counts[:] = torch.tensor([positive, negative], device=device)
    if dist.is_initialized():
        dist.broadcast(label_counts, src=0)
    train_positives, train_negatives = (int(value) for value in label_counts.tolist())
    pos_weight_value = train_negatives / train_positives
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    if is_main_process:
        print(
            f"Training labels: positive={train_positives:,}, negative={train_negatives:,}, "
            f"BCE pos_weight={pos_weight_value:.4f}"
        )

    model_config = {
        "input_dim": 21,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "edge_attr_dim": 0,
        "dropout": args.dropout,
        "coord_scale": args.coord_scale,
    }
    training_config = {
        "seed": SEED,
        "split": (
            f"layered VHH/CDR-H3/antigen/family connected components "
            f"(VHH {VHH_IDENTITY_THRESHOLD:.2f}, CDR-H3 {CDR_H3_IDENTITY_THRESHOLD:.2f}, "
            f"antigen {ANTIGEN_IDENTITY_THRESHOLD:.2f} with coverage {ANTIGEN_MIN_LENGTH_COVERAGE:.2f}); "
            "validation fold 0; no random 90/10"
        ),
        "homology_isolation": {
            "vhh_full_chain_identity": float(VHH_IDENTITY_THRESHOLD),
            "cdr_h3_identity": float(CDR_H3_IDENTITY_THRESHOLD),
            "antigen_identity": float(ANTIGEN_IDENTITY_THRESHOLD),
            "antigen_min_length_coverage": float(ANTIGEN_MIN_LENGTH_COVERAGE),
        },
        "graph_protocol": graph_protocol(train_data[0]),
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "gradient_clip": args.gradient_clip,
        "device": str(device),
        "distributed": world_size > 1,
        "world_size": world_size,
        "local_rank": local_rank,
        "in_memory": True,
        "threads": args.threads,
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
        "amp_requested": args.amp,
        "amp_enabled": bool(args.amp and device.type == "cuda"),
        "pos_weight": pos_weight_value,
        "geometry_baseline_contact_cutoff": args.geometry_baseline_contact_cutoff,
        "geometry_baseline_proximity_scale": args.geometry_baseline_proximity_scale,
    }
    base_model = EGNNInterfaceScorer(**model_config).to(device)
    optimizer = AdamW(
        base_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    effective_pin_memory = bool(args.pin_memory and device.type == "cuda")
    if is_main_process:
        print(
            f"AMP enabled={amp_enabled}; pin_memory={effective_pin_memory}; "
            f"persistent_workers={args.num_workers > 0}"
        )

    history: List[Dict[str, Any]] = []
    best_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1
    resume_path: Optional[Path] = None
    if args.resume is not None:
        resume_path = last_path if args.resume == "auto" else Path(args.resume)
        (
            start_epoch,
            history,
            best_auc,
            best_epoch,
            epochs_without_improvement,
        ) = restore_training_state(
            resume_path,
            model=base_model,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            train_paths=train_paths,
            validation_paths=validation_paths,
        )
        if is_main_process:
            print(
                f"Resumed from {resume_path.resolve()}: next_epoch={start_epoch}, "
                f"best_epoch={best_epoch}, best_ROC_AUC={best_auc:.6f}, "
                f"patience={epochs_without_improvement}/{args.patience}"
            )
    model: torch.nn.Module = base_model
    if dist.is_initialized():
        model = DDP(
            base_model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    train_dataset = InterfaceGraphDataset(train_data)
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=SEED,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    train_loader = _make_loader(
        train_data,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        seed=SEED,
        num_workers=args.num_workers,
        pin_memory=effective_pin_memory,
        sampler=train_sampler,
    )
    validation_loader = (
        _make_loader(
            validation_data,
            batch_size=args.batch_size,
            shuffle=False,
            seed=SEED,
            num_workers=args.num_workers,
            pin_memory=effective_pin_memory,
        )
        if is_main_process
        else None
    )
    training_start = time.perf_counter()
    # Fixed for the whole run: computed once instead of re-reading every graph
    # from disk in each epoch's checkpoint payload (A37).
    split_identity = run_split_identity(train_paths, validation_paths) if is_main_process else None

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start = time.perf_counter()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device=device,
            pos_weight=pos_weight,
            gradient_clip=args.gradient_clip,
            epoch=epoch,
            amp_enabled=amp_enabled,
            pin_memory=effective_pin_memory,
            is_main_process=is_main_process,
        )
        metrics: Optional[BinaryMetrics] = None
        if is_main_process:
            if validation_loader is None:
                raise RuntimeError("Rank 0 validation loader is missing")
            metrics = validate(
                base_model,
                validation_loader,
                device=device,
                pos_weight=pos_weight,
                epoch=epoch,
                amp_enabled=amp_enabled,
                pin_memory=effective_pin_memory,
            )
        metrics = _broadcast_metrics(metrics, rank=rank, device=device)
        improved = math.isfinite(metrics.roc_auc) and metrics.roc_auc > best_auc + 1e-6
        if improved:
            best_auc = metrics.roc_auc
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics.loss,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "best_f1_threshold": metrics.threshold,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "default_precision": metrics.default_precision,
            "default_recall": metrics.default_recall,
            "default_f1": metrics.default_f1,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "is_best": int(improved),
        }
        if is_main_process:
            history.append(row)
            payload = _checkpoint_payload(
                model=base_model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                metrics=metrics,
                train_loss=train_loss,
                model_config=model_config,
                training_config=training_config,
                train_paths=train_paths,
                validation_paths=validation_paths,
                history=history,
                best_auc=best_auc,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
                train_positives=train_positives,
                train_negatives=train_negatives,
                split_identity=split_identity,
            )
            _atomic_torch_save(payload, last_path)
            if improved:
                _atomic_torch_save(payload, best_path)
            _write_history(history, history_path)
            print(
                f"Epoch {epoch:03d}: train_loss={train_loss:.6f} "
                f"val_loss={metrics.loss:.6f} ROC-AUC={metrics.roc_auc:.6f} "
                f"PR-AUC={metrics.pr_auc:.6f} F1={metrics.f1:.6f} "
                f"threshold={metrics.threshold:.6f} best_epoch={best_epoch} "
                f"patience={epochs_without_improvement}/{args.patience}"
            )
        if epochs_without_improvement >= args.patience:
            if is_main_process:
                print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}.")
            break

    total_seconds = time.perf_counter() - training_start
    if dist.is_initialized():
        dist.barrier()
    if not is_main_process:
        dist.destroy_process_group()
        return 0
    best_payload = verify_checkpoint(best_path, device)
    write_summary(
        summary_path,
        total_epochs=len(history),
        best_payload=best_payload,
        best_checkpoint=best_path,
        total_seconds=total_seconds,
        train_count=len(train_paths),
        validation_count=len(validation_paths),
        train_positives=train_positives,
        train_negatives=train_negatives,
    )
    write_summary_json(
        summary_json_path,
        total_epochs=len(history),
        best_payload=best_payload,
        best_checkpoint=best_path,
        total_seconds=total_seconds,
        train_count=len(train_paths),
        validation_count=len(validation_paths),
        train_positives=train_positives,
        train_negatives=train_negatives,
        resumed_from=resume_path,
    )
    best_metrics = best_payload["validation_metrics"]
    print("Training complete")
    print(f"Total epochs: {len(history)}")
    print(f"Best epoch: {best_payload['epoch']}")
    print(f"Best validation ROC-AUC: {best_metrics['roc_auc']:.6f}")
    print(f"Best validation PR-AUC: {best_metrics['pr_auc']:.6f}")
    print(
        f"Best-F1 threshold metrics: threshold={best_metrics['threshold']:.6f}, "
        f"precision={best_metrics['precision']:.6f}, "
        f"recall={best_metrics['recall']:.6f}, F1={best_metrics['f1']:.6f}"
    )
    print(
        f"Checkpoint verified: {best_path.resolve()} "
        f"({best_path.stat().st_size:,} bytes)"
    )
    print(f"Summary: {summary_path.resolve()}")
    print(f"JSON summary: {summary_json_path.resolve()}")
    baseline_path=args.checkpoint_dir/"geometry_baseline.json"
    if baseline_path.is_file():
        baseline=json.loads(baseline_path.read_text(encoding="utf-8"))
        print(f"Geometry baseline validation ROC-AUC: {baseline['validation_roc_auc']:.6f}")
        print(f"Geometry baseline validation PR-AUC: {baseline['validation_pr_auc']:.6f}")
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
