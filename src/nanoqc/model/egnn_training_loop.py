"""One training epoch, validation, DDP helpers and checkpoint/history file I/O.

Split out of train_egnn_pruning.py, which re-exports every name here. The
split, checkpoint-payload and summary code stays there because it reads the
isolation thresholds that ``main`` rebinds from the command line.
"""
from __future__ import annotations

import csv
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm
from nanoqc.model.model_egnn_pruning import EGNNInterfaceScorer
from nanoqc.model.egnn_metrics import BinaryMetrics, best_f1_threshold, classification_metrics, pr_auc_score_binary, roc_auc_score_binary



def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    *,
    device: torch.device,
    pos_weight: Tensor,
    gradient_clip: float,
    epoch: int,
    amp_enabled: bool,
    pin_memory: bool,
    is_main_process: bool,
) -> float:
    """Train over every graph once and return node-weighted BCE loss."""

    model.train()
    weighted_loss_sum = 0.0
    node_count = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} train",
        unit="batch",
        dynamic_ncols=True,
        disable=not is_main_process,
    )
    for batch in progress:
        batch = batch.to(device, non_blocking=pin_memory)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            logits = model(
                batch.x, batch.pos, batch.edge_index, return_logits=True
            )
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.y,
                pos_weight=pos_weight,
                reduction="mean",
            )
        # One device synchronization per step: reading the scalar once serves both
        # the A20 non-finite guard and the loss accounting, which previously
        # synchronized twice (torch.isfinite, then float(loss)). Converting a
        # float32 nan/inf to a Python float preserves it, so the guard is
        # unchanged, and the accumulated value is bit-for-bit the same (A37).
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        nodes = int(batch.num_nodes)
        weighted_loss_sum += loss_value * nodes
        node_count += nodes
        if is_main_process:
            progress.set_postfix(loss=f"{weighted_loss_sum / node_count:.4f}")
    totals = torch.tensor(
        [weighted_loss_sum, float(node_count)], dtype=torch.float64, device=device
    )
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return float(totals[0].item() / max(totals[1].item(), 1.0))


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    pos_weight: Tensor,
    epoch: int,
    amp_enabled: bool,
    pin_memory: bool,
) -> BinaryMetrics:
    """Evaluate BCE, ROC-AUC, PR-AUC, and thresholded classification metrics."""

    model.eval()
    loss_sum = 0.0
    node_count = 0
    labels_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    progress = tqdm(loader, desc=f"Epoch {epoch:03d} valid", unit="batch", dynamic_ncols=True)
    for batch in progress:
        batch = batch.to(device, non_blocking=pin_memory)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            logits = model(
                batch.x, batch.pos, batch.edge_index, return_logits=True
            )
            batch_loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.y,
                pos_weight=pos_weight,
                reduction="sum",
            )
        loss_sum += float(batch_loss)
        node_count += int(batch.num_nodes)
        labels_parts.append(batch.y.detach().cpu().numpy().astype(np.int8, copy=False))
        score_parts.append(torch.sigmoid(logits).detach().cpu().numpy())
    labels = np.concatenate(labels_parts)
    scores = np.concatenate(score_parts)
    threshold, precision, recall, f1 = best_f1_threshold(labels, scores)
    default_precision, default_recall, default_f1 = classification_metrics(
        labels, scores, 0.5
    )
    return BinaryMetrics(
        loss=loss_sum / max(node_count, 1),
        roc_auc=roc_auc_score_binary(labels, scores),
        pr_auc=pr_auc_score_binary(labels, scores),
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        default_precision=default_precision,
        default_recall=default_recall,
        default_f1=default_f1,
        positives=int(labels.sum()),
        negatives=int(len(labels) - labels.sum()),
    )


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _write_history(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "epoch",
        "train_loss",
        "val_loss",
        "roc_auc",
        "pr_auc",
        "best_f1_threshold",
        "precision",
        "recall",
        "f1",
        "default_precision",
        "default_recall",
        "default_f1",
        "epoch_seconds",
        "is_best",
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def verify_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    """Reload the best file and strictly validate its model state."""

    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Checkpoint was not created correctly: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"model_config", "model_state_dict", "epoch", "validation_metrics"}
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Checkpoint missing fields: {sorted(missing)}")
    candidate = EGNNInterfaceScorer(**payload["model_config"])
    candidate.load_state_dict(payload["model_state_dict"], strict=True)
    candidate.to(device).eval()
    return payload


def _initialize_distributed(device_argument: str) -> Tuple[int, int, int, torch.device]:
    """Initialize torchrun DDP and bind each rank to one local CUDA device."""

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    cuda_requested = device_argument == "auto" or device_argument.startswith("cuda")
    use_cuda = cuda_requested and torch.cuda.is_available()
    if device_argument.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if use_cuda:
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds {torch.cuda.device_count()} CUDA devices"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu" if device_argument == "auto" else device_argument)
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return rank, local_rank, world_size, device


def _broadcast_metrics(
    metrics: Optional[BinaryMetrics], *, rank: int, device: torch.device
) -> BinaryMetrics:
    """Broadcast rank-0 validation metrics to every training rank."""

    if not dist.is_initialized():
        if metrics is None:
            raise RuntimeError("Validation metrics are missing")
        return metrics
    payload: List[Optional[Dict[str, Any]]] = [
        asdict(metrics) if rank == 0 and metrics is not None else None
    ]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        raise RuntimeError("Rank 0 did not broadcast validation metrics")
    return BinaryMetrics(**payload[0])
