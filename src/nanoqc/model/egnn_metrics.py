"""Binary interface-classification metrics and the geometry logistic baseline.

Split out of train_egnn_pruning.py, which re-exports every name here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple
import numpy as np
import torch
from scipy.optimize import minimize
from torch_geometric.data import Data



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
