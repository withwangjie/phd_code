"""Cluster-level paired inference kernels shared by formal statistics stages.

The QC/scaling and structural analyses use the same two-sided sign-flip test:
exact enumeration for at most 16 independent clusters, otherwise the caller's
configured Monte-Carlo resample count, with the same numerical tolerance and
finite-sample correction. QC/scaling keeps its historical RNG draw order so
existing benchmark results remain reproducible.

Structural percentile bootstrap confidence intervals remain a separate
resampling operation; only the sign-flip hypothesis-test kernel is shared.
``holm_step_down`` is the Holm adjustment shared by QC and structural
confirmatory analyses.
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import numpy as np
from nanoqc.common.seed_streams import DEFAULT_MASTER_SEED


SIGN_FLIP_EXACT_MAX_CLUSTERS = 16

PAIRED_OUTPUT_METRICS = (
    "gap", "hit", "ground_probability", "low_energy_mass",
    "low_energy_coverage", "entropy", "log10_qts99",
)
PAIRED_TIME_METRICS = ("gap", "hit", "log10_qts99")


def paired_denominator_failures(exclusions: dict, budget_mode: str) -> dict[str, int]:
    """Return paired-case exclusions that invalidate formal inference.

    All reported QAOA-vs-classical contrasts are part of the paired-statistics
    family. In time mode only gap/hit/log10_qts99 are analyzed; same-output diversity
    metrics are intentionally omitted and therefore are not denominator losses.
    """
    if budget_mode not in ("outputs", "time"):
        raise ValueError("budget_mode must be 'outputs' or 'time'")
    exclusions = exclusions if isinstance(exclusions, dict) else {}
    invalid: dict[str, int] = {}

    def add(key: str) -> None:
        try:
            count = int(exclusions.get(key, 0) or 0)
        except (TypeError, ValueError):
            count = 1
        if count > 0:
            invalid[key] = count

    add("qaoa:all_restarts_failed")
    add("missing_or_ambiguous_primary_contrast")
    for baseline in ("sa", "uniform", "greedy"):
        name = baseline + ("_time" if budget_mode == "time" else "")
        add(name + ":missing_pair")
        if budget_mode == "outputs":
            add(name + ":unequal_outputs")
            metrics = PAIRED_OUTPUT_METRICS
        else:
            add(name + ":invalid_budget")
            add(name + ":overrun")
            metrics = PAIRED_TIME_METRICS
        for metric in metrics:
            add(name + ":" + metric + ":nonfinite")
    return invalid


def _sign_flip_pvalue(d: np.ndarray, rng: np.random.Generator, resamples: int) -> float:
    """Shared two-sided sign-flip kernel for independent cluster differences."""
    observed = abs(float(d.mean()))
    tolerance = 1e-12 * max(1.0, observed)
    if len(d) <= SIGN_FLIP_EXACT_MAX_CLUSTERS:
        extreme = sum(
            abs(float(np.dot(signs, d) / len(d))) >= observed - tolerance
            for signs in itertools.product((-1, 1), repeat=len(d))
        )
        return float(extreme / (2 ** len(d)))

    extreme = 0
    for start in range(0, resamples, 256):
        size = min(256, resamples - start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(d)))
        extreme += int(
            np.count_nonzero(np.abs(signs @ d / len(d)) >= observed - tolerance)
        )
    return float((extreme + 1) / (resamples + 1))


def sign_flip_pvalue(values: Sequence[float], seed: int, resamples: int) -> Optional[float]:
    """Two-sided cluster sign-flip p-value under the shared formal protocol."""
    d = np.asarray(values, dtype=float)
    if d.ndim != 1 or not np.isfinite(d).all():
        raise ValueError("sign-flip values must be a finite 1-D sequence")
    if len(d) < 2:
        return None
    if resamples < 1:
        raise ValueError("resamples must be positive")
    return _sign_flip_pvalue(d, np.random.default_rng(seed), resamples)


def bootstrap_sign_flip(d: np.ndarray, seed: int, resamples: int
                        ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Percentile bootstrap CI of the mean and two-sided sign-flip p-value.

    ``d`` must be a finite 1-D array. Returns ``(ci_low, ci_high, p_value)``;
    all three are ``None`` for fewer than two clusters. Exact enumeration of
    all 2**n sign vectors for n <= 16, otherwise ``resamples`` Monte-Carlo
    sign vectors with the (extreme+1)/(resamples+1) correction.
    """
    if len(d) < 2:
        return None, None, None
    rng = np.random.default_rng(seed)
    boot = []
    for start in range(0, resamples, 256):
        size = min(256, resamples-start)
        boot.extend(d[rng.integers(0, len(d), (size, len(d)))].mean(axis=1))
    ci_low, ci_high = float(np.quantile(boot, .025)), float(np.quantile(boot, .975))
    pvalue = _sign_flip_pvalue(d, rng, resamples)
    return ci_low, ci_high, pvalue


def holm_step_down(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down family-wise adjustment; output keeps the input order.

    Ties are harmless: tied p-values receive the same adjusted value whichever
    order the sort visits them in, because the running maximum is monotone.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(p)-rank)*p[index])
        adjusted[index] = min(1., running)
    return adjusted.tolist()


def serial_gatekeeping(primary: dict, secondary: dict) -> dict:
    """Serial gatekeeping with Holm inside each family (Dmitrienko & Tamhane 2007).

    The primary (confirmatory) family is Holm-adjusted on its own at the full
    alpha. Secondary hypotheses are tested only after every primary
    hypothesis is rejected; their adjusted p value is
    max(largest primary adjusted p, Holm-adjusted p within the secondary
    family), which controls the family-wise error rate strongly across both
    families. Missing/non-finite p values stay None, and a missing primary p
    leaves the whole secondary family untestable (None).
    """
    def holm(pvalues: dict) -> dict:
        valid = {k: float(v) for k, v in pvalues.items()
                 if v is not None and np.isfinite(float(v))}
        names = sorted(valid)
        adjusted = dict(zip(names, holm_step_down([valid[n] for n in names]))) if names else {}
        return {k: adjusted.get(k) for k in pvalues}

    primary_adjusted = holm(primary)
    if not primary or any(value is None for value in primary_adjusted.values()):
        return {**primary_adjusted, **{k: None for k in secondary}}
    gate = max(primary_adjusted.values())
    secondary_adjusted = {
        k: (None if v is None else max(gate, v)) for k, v in holm(secondary).items()
    }
    return {**primary_adjusted, **secondary_adjusted}


# Call-site wrappers (formerly batch_benchmark_hard_set._paired_effect /
# _holm_adjust, which remain importable there under their old names).

def paired_effect(values: Sequence[float], seed: int = DEFAULT_MASTER_SEED,
                   resamples: int = 10000) -> dict:
    """Mean paired cluster difference, percentile CI, two-sided sign-flip test.

    Sign exchangeability/symmetry under the null and independent clusters are
    assumptions, not guaranteed by observational benchmark data.
    Kernel: :func:`paired_statistics.bootstrap_sign_flip`.
    """
    d = np.asarray(values, dtype=float)
    if d.ndim != 1 or not len(d) or not np.isfinite(d).all() or resamples < 100:
        raise ValueError("Finite nonempty differences and >=100 resamples required")
    result = dict(n_clusters=len(d), mean_difference=float(d.mean()),
                  ci_low=None, ci_high=None, p_value=None)
    if len(d) < 2:
        return result
    ci_low, ci_high, pvalue = bootstrap_sign_flip(d, seed, resamples)
    result.update(ci_low=ci_low, ci_high=ci_high, p_value=pvalue)
    return result


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error adjustment, original ordering retained."""
    p = np.asarray(pvalues,dtype=float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Invalid p values")
    return holm_step_down(p)
