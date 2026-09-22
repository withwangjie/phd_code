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
import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import parasail
import torch
from scipy.optimize import minimize
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from model_egnn_pruning import EGNNInterfaceScorer


SEED = 20260917


@dataclass(frozen=True)
class BinaryMetrics:
    """Validation metrics at one epoch and one decision threshold."""

    loss: float
    roc_auc: float
    pr_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    default_precision: float
    default_recall: float
    default_f1: float
    positives: int
    negatives: int


class InterfaceGraphDataset(Dataset):
    """RAM-resident graph dataset with labels prepared during preload."""

    def __init__(self, data_list: Sequence[Data]) -> None:
        super().__init__(root=None)
        self.data_list = list(data_list)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, index: int) -> Data:
        return self.data_list[index]


def graph_protocol(data: Data) -> Dict[str, Any]:
    """Return the versioned scientific graph protocol encoded in one PyG graph."""

    required = (
        "graph_version", "edge_policy", "label_policy",
        "intra_chain_ca_cutoff_angstrom", "cross_partner_knn_k",
        "interface_label_cutoff_angstrom", "min_interface_residues",
    )
    missing = [name for name in required if not hasattr(data, name)]
    if missing:
        raise ValueError(f"Graph lacks protocol metadata {missing}; rebuild with dataset version >=1.3")
    protocol = {
        "graph_version": str(data.graph_version),
        "edge_policy": str(data.edge_policy),
        "label_policy": str(data.label_policy),
        "intra_chain_ca_cutoff_angstrom": float(data.intra_chain_ca_cutoff_angstrom),
        "cross_partner_knn_k": int(data.cross_partner_knn_k),
        "interface_label_cutoff_angstrom": float(data.interface_label_cutoff_angstrom),
        "min_interface_residues": int(data.min_interface_residues),
    }
    if protocol["edge_policy"] != "intra_chain_ca_radius_plus_cross_partner_knn":
        raise ValueError("Unexpected graph edge policy; rebuild with current dataset builder")
    if protocol["label_policy"] != "cross_partner_heavy_atom_cutoff":
        raise ValueError("Unexpected graph label policy; rebuild with current dataset builder")
    if (protocol["intra_chain_ca_cutoff_angstrom"] <= 0
            or protocol["cross_partner_knn_k"] <= 0
            or protocol["interface_label_cutoff_angstrom"] <= 0
            or protocol["min_interface_residues"] <= 0):
        raise ValueError(f"Invalid graph protocol values: {protocol}")
    return protocol


def preload_graphs(paths: Sequence[Path], *, show_progress: bool) -> List[Data]:
    """Load every graph once, require one protocol, and attach labels."""

    iterator: Iterable[Path] = paths
    if show_progress:
        iterator = tqdm(paths, desc="Preloading graphs into RAM", unit="graph", dynamic_ncols=True)
    data_list = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in iterator
    ]
    expected_protocol: Optional[Dict[str, Any]] = None
    for data in data_list:
        protocol = graph_protocol(data)
        if expected_protocol is None:
            expected_protocol = protocol
        elif protocol != expected_protocol:
            raise ValueError(
                f"Mixed graph protocols in one training run: {expected_protocol} vs {protocol}"
            )
        data.y = interface_labels(data)
    return data_list


def interface_labels(data: Data) -> Tensor:
    """Return independently constructed heavy-atom interface labels.

    The exact heavy-atom cutoff is read from graph protocol metadata. Labels
    are never reconstructed from CA graph edges.
    """

    if not hasattr(data, "x") or not hasattr(data, "edge_index"):
        raise ValueError("Graph must contain x and edge_index")
    if data.x.ndim != 2 or data.x.size(1) != 21:
        raise ValueError(f"Expected x=[N,21], got {tuple(data.x.shape)}")
    if not hasattr(data, "interface_label"):
        raise ValueError(
            "Graph is missing interface_label; rebuild graphs with "
            "build_final_pyg_dataset.py version >= 1.4"
        )
    labels = data.interface_label.detach().cpu().to(torch.float32)
    if labels.shape != (data.num_nodes,):
        raise ValueError(
            f"interface_label must have shape [N], got {tuple(labels.shape)}"
        )
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("interface_label must contain only binary 0/1 values")
    graph_protocol(data)
    return labels

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CPU training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_loader_worker(worker_id: int) -> None:
    """Give every DataLoader process a deterministic independent RNG stream."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


VHH_IDENTITY_THRESHOLD = 0.80
CDR_H3_IDENTITY_THRESHOLD = 0.50
ANTIGEN_IDENTITY_THRESHOLD = 0.30
ANTIGEN_MIN_LENGTH_COVERAGE = 0.70


def _sequence_identity(a: str, b: str, *, min_length_coverage: float = 0.0) -> float:
    """Symmetric global identity over alignment length with an explicit length gate."""

    a, b = str(a or ""), str(b or "")
    if not a or not b:
        return 0.0
    coverage = min(len(a), len(b)) / max(len(a), len(b))
    if coverage < min_length_coverage:
        return 0.0
    if a == b:
        return 1.0
    values = []
    for left, right in ((a, b), (b, a)):
        result = parasail.nw_stats_striped_32(left, right, 10, 1, parasail.blosum62)
        if result.saturated:
            raise ValueError("alignment score saturation while building the split")
        values.append(result.matches / result.length)
    return max(values)


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


SPLIT_FOLDS = 5
VALIDATION_FOLD = 0


def split_paths(paths: Sequence[Path], seed: int) -> Tuple[List[Path], List[Path]]:
    """Layered VHH/CDR-H3/antigen component split; no random 90/10 partition."""
    del seed
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
        records.append((path, vhh, antigen, cdr3))

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
        _, vhh_i, antigen_i, cdr_i = records[i]
        for j in range(i):
            _, vhh_j, antigen_j, cdr_j = records[j]
            same_vhh = _side_identity(vhh_i, vhh_j) >= VHH_IDENTITY_THRESHOLD
            same_cdr = (
                bool(cdr_i and cdr_j)
                and _sequence_identity(cdr_i, cdr_j) >= CDR_H3_IDENTITY_THRESHOLD
            )
            same_antigen = (
                _side_identity(
                    antigen_i, antigen_j,
                    min_length_coverage=ANTIGEN_MIN_LENGTH_COVERAGE,
                ) >= ANTIGEN_IDENTITY_THRESHOLD
            )
            if same_vhh or same_cdr or same_antigen:
                union(i, j)

    components: Dict[int, List[int]] = {}
    for index in range(len(records)):
        components.setdefault(find(index), []).append(index)
    if len(components) < 2:
        raise RuntimeError("Layered VHH/CDR-H3/antigen clustering produced fewer than two components")

    train_paths: List[Path] = []
    validation_paths: List[Path] = []
    ordered_components = sorted(
        components.items(),
        key=lambda item: min(records[i][0].name for i in item[1]),
    )
    for _, indices in ordered_components:
        signature = "\n".join(sorted(records[i][0].name for i in indices)).encode("utf-8")
        fold = int.from_bytes(hashlib.sha256(signature).digest()[:8], "big") % SPLIT_FOLDS
        target = validation_paths if fold == VALIDATION_FOLD else train_paths
        target.extend(records[i][0] for i in indices)

    train_paths = sorted(train_paths, key=lambda path: path.name.lower())
    validation_paths = sorted(validation_paths, key=lambda path: path.name.lower())
    if not validation_paths:
        _, indices = ordered_components[0]
        chosen = {records[i][0] for i in indices}
        validation_paths = sorted(chosen, key=lambda p: p.name.lower())
        train_paths = sorted([p for p in paths if p not in chosen], key=lambda p: p.name.lower())
    if not train_paths:
        _, indices = ordered_components[-1]
        chosen = {records[i][0] for i in indices}
        train_paths = sorted(chosen, key=lambda p: p.name.lower())
        validation_paths = sorted([p for p in paths if p not in chosen], key=lambda p: p.name.lower())
    if not train_paths or not validation_paths:
        raise RuntimeError("Bilateral component split produced an empty partition")

    train_set, validation_set = set(train_paths), set(validation_paths)
    max_vhh_cross = 0.0
    max_cdr_cross = 0.0
    max_antigen_cross = 0.0
    for train_path, train_vhh, train_antigen, train_cdr in records:
        if train_path not in train_set:
            continue
        for val_path, val_vhh, val_antigen, val_cdr in records:
            if val_path not in validation_set:
                continue
            max_vhh_cross = max(max_vhh_cross, _side_identity(train_vhh, val_vhh))
            if train_cdr and val_cdr:
                max_cdr_cross = max(max_cdr_cross, _sequence_identity(train_cdr, val_cdr))
            max_antigen_cross = max(
                max_antigen_cross,
                _side_identity(
                    train_antigen, val_antigen,
                    min_length_coverage=ANTIGEN_MIN_LENGTH_COVERAGE,
                ),
            )
    if (max_vhh_cross >= VHH_IDENTITY_THRESHOLD
            or max_cdr_cross >= CDR_H3_IDENTITY_THRESHOLD
            or max_antigen_cross >= ANTIGEN_IDENTITY_THRESHOLD):
        raise AssertionError(
            "Layered sequence leakage across train/validation: "
            f"max VHH={max_vhh_cross:.3f}, max CDR-H3={max_cdr_cross:.3f}, "
            f"max antigen={max_antigen_cross:.3f}"
        )
    return train_paths, validation_paths

def _paths_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()



def _geometry_baseline_arrays(
    data_list: Sequence[Data], *, contact_cutoff: float, proximity_scale: float
) -> tuple[np.ndarray,np.ndarray,list[str]]:
    """Residue-type + partner geometry features; never reads heavy-atom labels as inputs."""
    rows=[]; labels=[]
    names=[*(f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"),
           "partner_group","nearest_partner_ca","contact_count","antigen_proximity"]
    for data in data_list:
        pos=data.pos.detach().cpu()
        group=data.x[:,-1].detach().cpu()
        onehot=data.x[:,:20].detach().cpu().numpy().astype(np.float64)
        nearest=np.empty(data.num_nodes,dtype=np.float64)
        contacts=np.empty(data.num_nodes,dtype=np.float64)
        for value in (0.0,1.0):
            source=torch.where(group==value)[0]
            partner=torch.where(group!=value)[0]
            if not len(source) or not len(partner):
                raise ValueError("Geometry baseline requires both partner groups")
            distances=torch.cdist(pos[source],pos[partner])
            nearest[source.numpy()]=distances.min(1).values.numpy()
            contacts[source.numpy()]=(distances<float(contact_cutoff)).sum(1).numpy()
        proximity=np.exp(-nearest/float(proximity_scale))
        rows.append(np.column_stack([onehot,group.numpy(),nearest,contacts,proximity]))
        labels.append(data.y.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(rows),np.concatenate(labels),names


def fit_geometry_logistic_baseline(
    train_data: Sequence[Data], validation_data: Sequence[Data], *,
    contact_cutoff: float = 8.0, proximity_scale: float = 6.0, l2: float = 1e-3
) -> dict[str,Any]:
    """Train-only weighted logistic baseline for geometry-shortcut auditing."""
    x_train,y_train,names=_geometry_baseline_arrays(
        train_data,contact_cutoff=contact_cutoff,proximity_scale=proximity_scale)
    x_val,y_val,_=_geometry_baseline_arrays(
        validation_data,contact_cutoff=contact_cutoff,proximity_scale=proximity_scale)
    mean=x_train.mean(0);std=x_train.std(0)
    std[std<1e-8]=1.0
    xt=(x_train-mean)/std;xv=(x_val-mean)/std
    positives=max(float(y_train.sum()),1.0);negatives=max(float(len(y_train)-y_train.sum()),1.0)
    sample_weight=np.where(y_train>0.5,negatives/positives,1.0)
    def objective(theta):
        bias=float(theta[0]);weights=theta[1:]
        z=bias+xt@weights
        loss=np.sum(sample_weight*(np.logaddexp(0.0,z)-y_train*z))/sample_weight.sum()
        loss+=0.5*l2*float(weights@weights)
        p=1.0/(1.0+np.exp(-np.clip(z,-50,50)))
        residual=sample_weight*(p-y_train)/sample_weight.sum()
        grad=np.concatenate([[residual.sum()],xt.T@residual+l2*weights])
        return float(loss),grad
    fit=minimize(objective,np.zeros(xt.shape[1]+1),jac=True,method="L-BFGS-B",
                 options={"maxiter":500,"ftol":1e-12})
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f"Geometry logistic baseline fit failed: {fit.message}")
    scores=1.0/(1.0+np.exp(-np.clip(fit.x[0]+xv@fit.x[1:],-50,50)))
    threshold,precision,recall,f1=best_f1_threshold(y_val.astype(np.int8),scores)
    return dict(
        model="weighted logistic regression",
        features=names,contact_ca_cutoff_angstrom=float(contact_cutoff),
        antigen_proximity_scale_angstrom=float(proximity_scale),l2=float(l2),
        train_nodes=int(len(y_train)),validation_nodes=int(len(y_val)),
        validation_roc_auc=roc_auc_score_binary(y_val.astype(np.int8),scores),
        validation_pr_auc=pr_auc_score_binary(y_val.astype(np.int8),scores),
        validation_best_f1=float(f1),validation_best_threshold=float(threshold),
        validation_precision=float(precision),validation_recall=float(recall),
        coefficients={name:float(value) for name,value in zip(["intercept",*names],fit.x)},
        standardization_mean=mean.tolist(),standardization_std=std.tolist(),
        optimizer_success=bool(fit.success),optimizer_message=str(fit.message),
        scope="fit on EGNN training split only; evaluated on the same homology-isolated validation split",
    )


def count_training_labels(data_list: Sequence[Data]) -> Tuple[int, int]:
    """Count RAM-resident positive and negative nodes for BCE pos_weight."""

    positives = 0
    total = 0
    for data in data_list:
        labels = data.y
        positives += int(labels.sum().item())
        total += int(labels.numel())
    negatives = total - positives
    if positives <= 0 or negatives <= 0:
        raise RuntimeError(
            f"Training labels require both classes, got positive={positives}, negative={negatives}"
        )
    return positives, negatives


def roc_auc_score_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC with average ranks for tied predictions."""

    labels = labels.astype(np.int8, copy=False)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def pr_auc_score_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute step-integrated PR-AUC (average precision)."""

    labels = labels.astype(np.int8, copy=False)
    positives = int(labels.sum())
    if positives == 0:
        return math.nan
    order = np.argsort(-scores, kind="mergesort")
    ranked = labels[order]
    true_positives = np.cumsum(ranked)
    precision = true_positives / np.arange(1, len(ranked) + 1)
    return float(precision[ranked == 1].sum() / positives)


def classification_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """Return precision, recall, and F1 at a probability threshold."""

    predicted = scores >= threshold
    positive = labels == 1
    tp = int(np.sum(predicted & positive))
    fp = int(np.sum(predicted & ~positive))
    fn = int(np.sum(~predicted & positive))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    """Find the validation threshold maximizing F1 without quadratic scans."""

    labels = labels.astype(np.int8, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    ranked_labels = labels[order]
    ranked_scores = scores[order]
    tp = np.cumsum(ranked_labels)
    fp = np.cumsum(1 - ranked_labels)
    positives = int(labels.sum())
    fn = positives - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(positives, 1)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-15)
    # Only evaluate boundaries that change the predicted set.
    boundary = np.r_[ranked_scores[:-1] != ranked_scores[1:], True]
    candidate_indices = np.flatnonzero(boundary)
    best_index = int(candidate_indices[np.argmax(f1[candidate_indices])])
    return (
        float(ranked_scores[best_index]),
        float(precision[best_index]),
        float(recall[best_index]),
        float(f1[best_index]),
    )


def _make_loader(
    data_list: Sequence[Data],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
    sampler: Optional[DistributedSampler] = None,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    options: Dict[str, Any] = {
        "dataset": InterfaceGraphDataset(data_list),
        "batch_size": batch_size,
        "shuffle": shuffle if sampler is None else False,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
        "generator": generator,
        "worker_init_fn": seed_loader_worker,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(
        **options,
    )


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
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
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            logits = model(
                batch.x, batch.pos, batch.edge_index, return_logits=True
            )
            loss = F.binary_cross_entropy_with_logits(
                logits,
                batch.y,
                pos_weight=pos_weight,
                reduction="mean",
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        nodes = int(batch.num_nodes)
        weighted_loss_sum += float(loss.detach()) * nodes
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
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
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


def _checkpoint_payload(
    *,
    model: EGNNInterfaceScorer,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
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
) -> Dict[str, Any]:
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
            "partition_method": "layered_vhh_cdrh3_antigen_components_sha256_5fold",
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
            "train_names_sha256": _paths_digest(train_paths),
            "validation_names_sha256": _paths_digest(validation_paths),
        },
        "graph_protocol": graph_protocol(torch.load(train_paths[0], map_location="cpu", weights_only=False)),
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


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def restore_training_state(
    checkpoint_path: Path,
    *,
    model: EGNNInterfaceScorer,
    optimizer: AdamW,
    scaler: torch.cuda.amp.GradScaler,
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
            "partition_method": "layered_vhh_cdrh3_antigen_components_sha256_5fold",
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
            "validation_scope": "known-pose, layered VHH/CDR-H3/antigen homology-isolated components",
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
        default=True,
        help="Enable CUDA float16 autocast and GradScaler (default: enabled).",
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
             "20260917) for weight init, loader shuffling and DDP-rank seed offsets; pass an "
             "independently-derived train-stream seed here rather than reusing the bare master seed.",
    )
    return parser


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
            f"layered VHH/CDR-H3/antigen connected components "
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
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
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
