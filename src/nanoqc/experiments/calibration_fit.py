"""Fit the training-only coarse-to-Amber ridge calibration from its CSV.

Run through ``--research-ablation --fit-energy-calibration-csv``. Rows are
grouped by PDB for SHA-256 five-fold cross-validation; component weights are
nonnegative and the intercept is free.
"""
from __future__ import annotations

import csv
import math
import hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr
from nanoqc.common.repo_io import sha256_file as _ablation_digest, atomic_write_json_fsync as _ablation_atomic_json



def fit_energy_calibration_csv(input_csv: Path, output_json: Path, ridge_alpha: float) -> dict:
    """Fit nonnegative coarse-component weights to Amber delta-E using TRAIN complexes only.

    Rows from the same family/structure cluster are kept together when the
    CSV supplies a nonempty family_cluster column; otherwise grouping falls
    back to PDB. This prevents related complexes from crossing calibration
    fit/evaluation folds.
    """
    if ridge_alpha < 0 or not math.isfinite(ridge_alpha):
        raise ValueError("ridge_alpha must be finite and nonnegative")
    with Path(input_csv).open(newline="",encoding="utf-8-sig") as handle:
        rows=list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Calibration CSV is empty")
    if "diagnostic_mode" in rows[0]:
        raise ValueError("Dual-energy diagnostic CSV is not a calibration fitting input")
    required=(
        "pdb_id","split","prior_energy","vhh_environment_energy",
        "antigen_energy","pair_energy","amber_delta_kcal",
    )
    missing=[key for key in required if key not in rows[0]]
    if missing:
        raise ValueError(f"Calibration CSV missing columns: {missing}")
    if any(str(row["split"]).lower()!="train" for row in rows):
        raise ValueError("Energy calibration may use training rows only")
    pdb_ids=[str(row["pdb_id"]).strip().lower() for row in rows]
    if any(not pdb for pdb in pdb_ids):
        raise ValueError("Calibration rows require nonempty pdb_id values")

    feature_names=("prior_energy","vhh_environment_energy","antigen_energy","pair_energy")
    X=np.asarray([[float(row[k]) for k in feature_names] for row in rows],dtype=float)
    y=np.asarray([float(row["amber_delta_kcal"]) for row in rows],dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Calibration data contain non-finite values")
    family_present = "family_cluster" in rows[0]
    family_ids=[str(row.get("family_cluster","")).strip() for row in rows]
    if family_present and any(not value for value in family_ids):
        raise ValueError("family_cluster column must be complete when present")
    group_ids=family_ids if family_present else pdb_ids
    grouping_name="family_cluster" if family_present else "pdb_id"
    if len(set(group_ids)) < 5:
        raise ValueError(
            f"Calibration requires at least five distinct {grouping_name} groups for grouped 5-fold CV"
        )

    def fit_coefficients(x: np.ndarray, target: np.ndarray) -> np.ndarray:
        design=np.column_stack([np.ones(len(x)),x])
        if ridge_alpha:
            ridge=np.zeros((x.shape[1],1+x.shape[1]),dtype=float)
            ridge[:,1:]=math.sqrt(float(ridge_alpha))*np.eye(x.shape[1])
            design_aug=np.vstack([design,ridge])
            target_aug=np.concatenate([target,np.zeros(x.shape[1],dtype=float)])
        else:
            design_aug,target_aug=design,target
        lower=np.asarray([-np.inf,0.0,0.0,0.0,0.0],dtype=float)
        upper=np.full(5,np.inf,dtype=float)
        result=lsq_linear(design_aug,target_aug,bounds=(lower,upper),method="trf")
        if not result.success or not np.isfinite(result.x).all():
            raise RuntimeError(f"Nonnegative ridge calibration failed: {result.message}")
        return result.x

    unique_pdbs=sorted(set(pdb_ids))
    unique_groups=sorted(set(group_ids))
    # Deterministic hashed ordering followed by round-robin allocation keeps
    # entire family/structure groups together and guarantees five nonempty
    # folds whenever at least five groups are available.
    ordered_groups=sorted(
        unique_groups,
        key=lambda group:(hashlib.sha256(group.encode("utf-8")).hexdigest(),group)
    )
    fold_of={group:index%5 for index,group in enumerate(ordered_groups)}
    cv_rows=[]
    cv_predictions=np.full(len(y),np.nan,dtype=float)
    uncalibrated_predictions=X.sum(axis=1)
    for fold in sorted(set(fold_of.values())):
        test_mask=np.asarray([fold_of[group]==fold for group in group_ids],dtype=bool)
        train_mask=~test_mask
        if not test_mask.any() or not train_mask.any():
            continue
        beta=fit_coefficients(X[train_mask],y[train_mask])
        pred=np.column_stack([np.ones(test_mask.sum()),X[test_mask]])@beta
        cv_predictions[test_mask]=pred
        residual=y[test_mask]-pred
        cv_rows.append(dict(
            fold=int(fold),
            train_complexes=int(len({p for p,m in zip(pdb_ids,train_mask) if m})),
            test_complexes=int(len({p for p,m in zip(pdb_ids,test_mask) if m})),
            train_groups=int(len({g for g,m in zip(group_ids,train_mask) if m})),
            test_groups=int(len({g for g,m in zip(group_ids,test_mask) if m})),
            test_rows=int(test_mask.sum()),
            rmse_kcal=float(np.sqrt(np.mean(residual**2))),
            mae_kcal=float(np.mean(np.abs(residual))),
        ))

    beta=fit_coefficients(X,y)
    design=np.column_stack([np.ones(len(X)),X])
    pred=design@beta
    residual=y-pred
    ss_tot=float(np.sum((y-y.mean())**2))
    cv_mask=np.isfinite(cv_predictions)
    cv_r2=None
    cv_spearman=None
    if cv_mask.any():
        cv_target=y[cv_mask];cv_pred=cv_predictions[cv_mask]
        cv_ss_tot=float(np.sum((cv_target-cv_target.mean())**2))
        if cv_ss_tot>0:
            cv_r2=float(1.0-np.sum((cv_target-cv_pred)**2)/cv_ss_tot)
        if len(cv_target)>=3:
            rho=spearmanr(cv_target,cv_pred).statistic
            if math.isfinite(float(rho)):
                cv_spearman=float(rho)

    baseline_residual=y-uncalibrated_predictions
    baseline_ss_tot=float(np.sum((y-y.mean())**2))
    baseline_r2=(
        None if baseline_ss_tot<=0
        else float(1.0-np.sum(baseline_residual**2)/baseline_ss_tot)
    )
    baseline_spearman=None
    if len(y)>=3:
        baseline_rho=spearmanr(y,uncalibrated_predictions).statistic
        if math.isfinite(float(baseline_rho)):
            baseline_spearman=float(baseline_rho)

    payload={
        "intercept":float(beta[0]),
        "prior_weight":float(beta[1]),
        "vhh_environment_weight":float(beta[2]),
        "antigen_weight":float(beta[3]),
        "pair_weight":float(beta[4]),
        "ridge_alpha":float(ridge_alpha),
        "coefficient_constraint":"nonnegative component weights; unconstrained intercept",
        "n_train_samples":int(len(y)),
        "n_train_complexes":int(len(unique_pdbs)),
        "n_train_groups":int(len(unique_groups)),
        "cv_grouping":grouping_name,
        "train_rmse_kcal":float(np.sqrt(np.mean(residual**2))),
        "train_mae_kcal":float(np.mean(np.abs(residual))),
        "train_r2":(None if ss_tot<=0 else float(1.0-np.sum(residual**2)/ss_tot)),
        "cv_scheme":f"deterministic {grouping_name}-grouped SHA256-order round-robin 5-fold",
        "cv_folds":cv_rows,
        "cv_fold_count":int(len(cv_rows)),
        "cv_rmse_kcal":(
            None if not cv_mask.any()
            else float(np.sqrt(np.mean((y[cv_mask]-cv_predictions[cv_mask])**2)))
        ),
        "cv_mae_kcal":(
            None if not cv_mask.any()
            else float(np.mean(np.abs(y[cv_mask]-cv_predictions[cv_mask])))
        ),
        "cv_r2":cv_r2,
        "cv_spearman":cv_spearman,
        "uncalibrated_rmse_kcal":float(np.sqrt(np.mean(baseline_residual**2))),
        "uncalibrated_mae_kcal":float(np.mean(np.abs(baseline_residual))),
        "uncalibrated_r2":baseline_r2,
        "uncalibrated_spearman":baseline_spearman,
        "calibration_rmse_improvement_kcal":(
            None if not cv_mask.any()
            else float(np.sqrt(np.mean(baseline_residual[cv_mask]**2))
                       - np.sqrt(np.mean((y[cv_mask]-cv_predictions[cv_mask])**2)))
        ),
        "input_sha256":_ablation_digest(Path(input_csv)),
        "scope":"fit on training complexes only; freeze coefficients before validation/test",
    }
    _ablation_atomic_json(Path(output_json),payload)
    return payload
