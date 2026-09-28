"""Result records, the CVaR objective helpers and QAOA constants.

Split out of qaoa_interface_sampler.py, which re-exports every name here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import numpy as np



BitString = Tuple[int, ...]


@dataclass(frozen=True)
class OptimizationResult:
    """Optimized QAOA angles and objective-evaluation history."""

    gammas: np.ndarray
    betas: np.ndarray
    energy: float
    history: Tuple[float, ...]
    success: bool
    message: str
    evaluations: int
    objective_name: str = "mean"
    objective_value: Optional[float] = None
    restart_records: Tuple[dict, ...] = ()
    total_opt_shots: int = 0
    # --- Finite-shot measurement bookkeeping (see optimize_robust) ---
    # ``nfev`` duplicates ``evaluations`` under the name conventional in the
    # optimization literature; both always agree by construction.
    nfev: int = 0
    eval_shots: Optional[int] = None
    shot_ledger: Mapping[str, Any] = field(default_factory=dict)
    # --- Honest convergence reporting (see optimize_robust) ---
    # ``optimizer_success`` is a stricter synonym of ``success`` under the
    # literal name academic reviewers asked for; both always agree by
    # construction. ``termination_reason`` says *why* the run stopped:
    # ``"converged"`` only when every restart's own COBYLA call satisfied
    # its internal convergence test, ``"max_evaluations_reached"`` when the
    # evaluation budget -- not algorithmic convergence -- is what stopped
    # the search, or ``"mixed_or_incomplete_convergence"``/
    # ``"restart_raised_exception"``/``"not_applicable"`` otherwise. Never
    # ``"converged"`` merely because the budget happened not to run out.
    optimizer_success: bool = False
    termination_reason: str = "not_applicable"
    # ``gamma_best``/``beta_best``/``best_mean_energy`` are explicit aliases
    # for ``gammas``/``betas``/``energy`` under the names requested for
    # direct citation in write-ups; all three pairs always hold identical
    # values by construction.
    gamma_best: Optional[np.ndarray] = None
    beta_best: Optional[np.ndarray] = None
    best_mean_energy: Optional[float] = None
    # (P2 remediation) Populated only for the ``termination_reason ==
    # "all_restarts_failed"`` case -- see optimize_robust's docstring --
    # with a small diagnostic dict (``{"error": ..., "restart_exception_messages": [...]}``).
    # ``None`` in every other case.
    raw_result: Optional[Dict[str, Any]] = None


class OptimizationCollapseError(RuntimeError):
    """No point was ever successfully evaluated in ``optimize_robust``.

    Raised only for the genuinely unrecoverable case: not even the single,
    cost-free, always-attempted uniform-state baseline evaluation
    succeeded (so ``best`` is empty and there is no point of any kind, not
    even a degenerate one, to return). This is distinct from -- and much
    rarer than -- every *restart* raising: when the baseline succeeds but
    every restart's COBYLA call subsequently raises, ``optimize_robust``
    does not raise; it returns a real (if degenerate, baseline-only)
    ``OptimizationResult`` with ``termination_reason ==
    "all_restarts_failed"`` and ``optimizer_success = False``, so a caller
    can distinguish total algorithmic collapse from ordinary budget
    exhaustion without a crash. See ``optimize_robust``'s docstring.
    """


# Absolute energy tolerance defining "at the exact ground energy" everywhere
# (ground-state enumeration, hit, ground-state probability), shared with the
# coarse benchmark summaries so both report the same hit semantics.
GROUND_ENERGY_TOLERANCE = 1e-9
# Largest supported QAOA depth. p up to 12 has been studied for noiseless
# QAOA scaling (Shaydulin et al., Sci. Adv. 2024); deeper circuits need
# proportionally larger optimizer budgets (see quantum_exploration).
MAX_QAOA_DEPTH = 12


def lower_tail_cvar(energies: np.ndarray, probabilities: np.ndarray, alpha: float) -> float:
    """Exact lower-tail CVaR over a *known* probability distribution.

    This is the analytic (infinite-shot) CVaR: it includes fractional
    probability mass at the quantile boundary and requires the caller to
    already know each outcome's exact probability. Use this only when the
    full distribution is available (e.g. state-vector simulation). For a
    finite measurement sample where every shot carries equal weight
    ``1/eval_shots``, use :func:`finite_shot_cvar` instead -- it matches the
    literature's usual finite-shot CVaR-VQE/QAOA aggregation (Barkoutsos
    et al. 2020), which averages the literal lowest-energy shots rather than
    interpolating a fractional boundary weight.
    """
    e, p = np.asarray(energies, dtype=float), np.asarray(probabilities, dtype=float)
    if not 0 < alpha <= 1 or e.ndim != 1 or e.shape != p.shape or not len(e):
        raise ValueError("Invalid CVaR dimensions or alpha")
    if not np.isfinite(e).all() or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1.):
        raise ValueError("CVaR requires finite energies and normalized nonnegative probabilities")
    order = np.argsort(e, kind="stable")
    mass = p[order] / p.sum()
    taken = np.minimum(mass, np.maximum(0., alpha - (np.cumsum(mass) - mass)))
    return float(taken @ e[order] / alpha)


def finite_shot_cvar(sampled_energies: Sequence[float], alpha: float) -> float:
    """Finite-shot CVaR estimator: plain mean of the lowest-energy shots.

    Sorts ``sampled_energies`` (one physical/Ising energy per measured
    bitstring) ascending and returns the arithmetic mean of the lowest
    ``ceil(alpha * len(sampled_energies))`` of them. Every shot is an
    equally-weighted point mass drawn from the true, unknown Born
    distribution -- unlike :func:`lower_tail_cvar`, there is no fractional
    boundary weight to interpolate, because a single shot cannot be split.

    Args:
        sampled_energies: 1-D array-like of finite-shot measured energies.
        alpha: Lower-tail quantile fraction in ``(0, 1]``.

    Raises:
        ValueError: If the sample is empty/non-1D, ``alpha`` is out of
            range, or any sampled energy is non-finite.
    """
    try:
        e = np.asarray(sampled_energies, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("sampled_energies must be convertible to a float array") from exc
    if e.ndim != 1 or e.size == 0:
        raise ValueError("finite_shot_cvar requires a nonempty 1-D sample of energies")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must lie in (0, 1]")
    if not np.isfinite(e).all():
        raise ValueError("finite_shot_cvar requires finite sampled energies")
    tail_count = int(math.ceil(alpha * e.size))
    tail_count = max(1, min(tail_count, e.size))
    order = np.argsort(e, kind="stable")
    tail = e[order[:tail_count]]
    return float(tail.mean())


@dataclass(frozen=True)
class GroundStateResult:
    """Exact optimum over the feasible product of local one-hot states."""

    energy: float
    states: Tuple[BitString, ...]
    configuration_count: int


@dataclass(frozen=True)
class QuantumSampleResult:
    """Summary of finite-shot samples from the optimized QAOA circuit.

    ``bitstring_entropy``, ``low_energy_fraction`` and ``ground_state_hit``
    are computed purely from this finite output sample (never from the
    exact analytic distribution), matching the empirical accounting this
    project's benchmark scripts already compute by hand from raw
    ``counts`` -- see :meth:`XYMixerQAOASampler.sample`.
    """

    counts: Mapping[BitString, int]
    shots: int
    legal_rate: float
    best_energy: float
    best_states: Tuple[BitString, ...]
    ground_state_success_probability: float
    mean_energy: float
    bitstring_entropy: float
    low_energy_fraction: float
    ground_state_hit: bool


@dataclass(frozen=True)
class AnnealingResult:
    """Results from repeated simulated-annealing reads."""

    counts: Mapping[BitString, int]
    best_energy: float
    best_state: BitString
    ground_state_success_probability: float
    read_energies: np.ndarray
    best_energy_trace: Tuple[float, ...]
