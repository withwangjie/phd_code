"""Cluster-level paired inference kernels shared by the QC statistics stages.

``bootstrap_sign_flip`` is the exact algorithm formerly copied into
``batch_benchmark_hard_set._paired_effect`` (now :func:`paired_effect`) and
``analyze_quantum_scaling._cluster_effect`` (same RNG, same draw order, same
batching of 256, same tolerance), so both wrappers produce bit-identical
results. ``holm_step_down`` is the Holm adjustment shared by
:func:`holm_adjust` (formerly ``batch_benchmark_hard_set._holm_adjust``) and
``analyze_structure_recovery.holm_adjust_named``.

``analyze_structure_recovery.percentile_ci`` / ``sign_flip_p`` are NOT routed
through here: they implement a different resampling scheme (per-draw
``rng.choice``, exact enumeration up to 20 clusters, 200 000 Monte-Carlo
trials, tolerance 1e-15), and merging them would change reported numbers.
"""
from __future__ import annotations

import itertools
from typing import Optional, Sequence, Tuple

import numpy as np


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
    observed = abs(float(d.mean()))
    tolerance = 1e-12*max(1., observed)
    if len(d) <= 16:
        extreme = sum(abs(float(np.dot(signs, d)/len(d))) >= observed-tolerance
                      for signs in itertools.product((-1, 1), repeat=len(d)))
        pvalue = extreme/(2**len(d))
    else:
        extreme = 0
        for start in range(0, resamples, 256):
            size = min(256, resamples-start)
            signs = rng.choice([-1., 1.], size=(size, len(d)))
            extreme += int(np.count_nonzero(np.abs(signs@d/len(d)) >= observed-tolerance))
        pvalue = (extreme+1)/(resamples+1)
    return ci_low, ci_high, float(pvalue)


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


# Call-site wrappers (formerly batch_benchmark_hard_set._paired_effect /
# _holm_adjust, which remain importable there under their old names).

def paired_effect(values: Sequence[float], seed: int = 20260917,
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
