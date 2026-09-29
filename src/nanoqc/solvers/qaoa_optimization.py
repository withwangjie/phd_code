"""Variational optimization of XYMixerQAOASampler: single-start, robust multi-start and measurement-ledger solves.

The methods below are XYMixerQAOASampler's own (it inherits them from this
mixin); they were split out of qaoa_interface_sampler.py unchanged.
"""
from __future__ import annotations
from nanoqc.common.device_errors import raise_if_resource_error

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from nanoqc.solvers.qaoa_results import GroundStateResult, OptimizationCollapseError, OptimizationResult, finite_shot_cvar


class QAOAOptimizationMixin:
    """Optimization methods of :class:`XYMixerQAOASampler` (see qaoa_interface_sampler.py)."""

    def optimize(
        self,
        *,
        method: str = "cobyla",
        max_iterations: int = 30,
        learning_rate: float = 0.08,
        initial_parameters: Optional[Sequence[float]] = None,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> OptimizationResult:
        """Optimize QAOA angles with COBYLA or PennyLane's Adam optimizer."""

        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        rng = np.random.default_rng(self.seed)
        if initial_parameters is None:
            # COBYLA performs substantially better when its coordinates have
            # comparable scales.  Optimize dimensionless gamma coordinates and
            # divide by the largest Ising coefficient before circuit execution.
            internal_initial = np.concatenate(
                [rng.uniform(0.0, 0.6, self.p), rng.uniform(0.1, 0.8, self.p)]
            )
        else:
            initial = np.asarray(initial_parameters, dtype=np.float64)
            if initial.shape != (2 * self.p,):
                raise ValueError(f"Expected {2 * self.p} initial parameters")
            internal_initial = initial.copy()
            internal_initial[: self.p] *= self.gamma_scale

        def physical_parameters(internal: Sequence[float]):
            values = pnp.array(internal)
            return pnp.concatenate(
                [values[: self.p] / self.gamma_scale, values[self.p :]]
            )

        method_name = method.lower()
        history: List[float] = []
        if method_name == "cobyla":

            def objective(parameters: np.ndarray) -> float:
                energy = self.expected_energy(physical_parameters(parameters))
                history.append(energy)
                if progress_callback is not None:
                    progress_callback(len(history), energy)
                return energy

            result = minimize(
                objective,
                internal_initial,
                method="COBYLA",
                options={
                    "maxiter": int(max_iterations),
                    "rhobeg": 0.35,
                    "catol": 1e-7,
                },
            )
            optimum = np.asarray(physical_parameters(result.x), dtype=np.float64)
            energy = self.expected_energy(optimum)
            if not history or abs(history[-1] - energy) > 1e-12:
                history.append(energy)
            converged = bool(result.success)
            gamma_best, beta_best = optimum[: self.p].copy(), optimum[self.p :].copy()
            return OptimizationResult(
                gammas=gamma_best,
                betas=beta_best,
                energy=energy,
                history=tuple(history),
                success=converged,
                message=str(result.message),
                evaluations=int(result.nfev),
                nfev=int(result.nfev),
                optimizer_success=converged,
                termination_reason="converged" if converged else "max_evaluations_reached",
                gamma_best=gamma_best,
                beta_best=beta_best,
                best_mean_energy=energy,
            )

        if method_name == "adam":
            if self.simulation_mode == "subspace":
                raise ValueError("subspace simulation currently supports COBYLA only")
            parameters = pnp.array(internal_initial, requires_grad=True)
            optimizer = qml.AdamOptimizer(stepsize=float(learning_rate))

            def differentiable_objective(values):
                return self._energy_qnode(physical_parameters(values)) + self.ising_offset

            for _ in range(max_iterations):
                parameters, energy_before = optimizer.step_and_cost(
                    differentiable_objective, parameters
                )
                history.append(float(energy_before))
            final_energy = float(differentiable_objective(parameters))
            history.append(final_energy)
            optimum = np.asarray(physical_parameters(parameters), dtype=np.float64)
            gamma_best, beta_best = optimum[: self.p].copy(), optimum[self.p :].copy()
            return OptimizationResult(
                gammas=gamma_best,
                betas=beta_best,
                energy=final_energy,
                history=tuple(history),
                success=True,
                message="Adam completed the requested number of iterations",
                evaluations=max_iterations + 1,
                nfev=max_iterations + 1,
                # Adam has no internal convergence test (unlike COBYLA's catol
                # trust-region criterion): it simply runs the fixed iteration
                # budget, so "converged" would be an unsupported claim.
                optimizer_success=True,
                termination_reason="fixed_iteration_budget_completed",
                gamma_best=gamma_best,
                beta_best=beta_best,
                best_mean_energy=final_energy,
            )

        raise ValueError("method must be 'cobyla' or 'adam'")

    def parameter_scale(self, mode: str = "max_coefficient") -> float:
        """Gamma normalisation: physical gamma = internal gamma / scale.

        ``max_coefficient`` uses the largest Ising coefficient; ``feasible_iqr``
        the interquartile range of feasible energies (classical preprocessing).
        Shared by ``optimize_robust`` and by parameter transfer, so transferred
        internal angles are rescaled exactly as the optimizer scaled them.
        """
        if mode not in ("max_coefficient", "feasible_iqr"):
            raise ValueError("parameter_scale must be 'max_coefficient' or 'feasible_iqr'")
        self.subspace_state(np.zeros(2 * self.p))
        scale = self.gamma_scale if mode == "max_coefficient" else max(
            float(np.subtract(*np.quantile(self._subspace_energies, [.75, .25]))), 1.)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Computed parameter scale is not a positive finite number")
        return float(scale)

    def optimize_robust(
        self, *, max_evals: int = 90, restarts: int = 4,
        objective: str = "cvar", cvar_alpha: float = 0.1,
        parameter_scale: str = "max_coefficient",
        eval_shots: int = 500,
        optimize_seed: Optional[int] = None,
        measurement_seed: Optional[int] = None,
    ) -> OptimizationResult:
        """Budgeted, finite-shot multistart COBYLA over the exact-subspace simulation.

        Every objective evaluation -- including the initial uniform-state
        baseline -- draws exactly ``eval_shots`` finite measurement shots
        from the current parameterized wavefunction's Born distribution and
        scores them with :func:`finite_shot_cvar` (or a plain empirical
        mean). The exact analytic state-vector expectation is **never**
        read as the optimized objective (only kept internally, per
        evaluation, as ``best_mean_energy`` bookkeeping once the search is
        done) -- via ``simulation_mode='subspace'`` this is a restricted-
        Hilbert-space *classical simulation* of what a real device would
        report, not a hardware execution, but the optimizer sees exactly
        the same finite-shot noise a real QPU run would produce.

        All ``restarts`` COBYLA restarts share one measurement budget
        ``max_evals`` (including the uniform-state baseline); the budget is
        a hard cutoff -- ``len(history) <= max_evals`` always holds, even
        when scipy's COBYLA is interrupted mid-restart by its own
        ``maxiter``. The best point *ever evaluated* is tracked
        independently of each restart's ``OptimizeResult.success`` flag, so
        a restart that COBYLA reports as unconverged (``success=False``,
        e.g. because its ``maxiter`` budget ran out) still contributes its
        evaluations to ``best`` if any of them improved on the incumbent --
        ``gamma_best``/``beta_best`` (aliases of ``gammas``/``betas``) and
        ``best_mean_energy`` (alias of ``energy``) are exported regardless
        of whether any restart actually converged. Selection uses only the
        declared objective, never reference RMSD or ground-state
        probability. Physical energies and phase evolution are not
        clipped. Feasible-IQR scaling requires enumerating the legal space
        and must be reported as classical preprocessing, not a hardware
        speedup. ``history`` records the running finite-shot objective
        (CVaR or mean); ``energy``/``best_mean_energy`` always record the
        exact analytic mean physical energy AT the best point found (a
        diagnostic quantity, never what COBYLA actually optimized).

        Convergence reporting is deliberately strict, per academic-review
        feedback that a budget-exhausted run must never be described as
        converged: ``optimizer_success`` (and its legacy alias ``success``)
        is ``True`` only when *every* restart's own COBYLA call satisfied
        its internal convergence test (``catol``) before exhausting its
        share of the evaluation budget. ``termination_reason`` explains why
        the run actually stopped:

        * ``"converged"`` -- every restart converged on its own terms.
        * ``"max_evaluations_reached"`` -- the run stopped because the
          evaluation budget ran out (the common case: COBYLA's convergence
          test rarely fires cleanly against a stochastic, finite-shot
          objective), never because of algorithmic convergence.
        * ``"mixed_or_incomplete_convergence"`` -- some restarts converged
          and some did not, but the global ``max_evals`` budget was not
          the limiting factor.
        * ``"all_restarts_failed"`` -- **total algorithmic collapse**:
          every one of the ``restarts`` COBYLA calls raised an exception
          (see ``restart_records`` for each one's message), so the
          returned parameters are the best successfully scored point,
          which may come from evaluations completed before an exception.
          The baseline is returned only if no better point was scored. Distinct from
          (and takes priority in classification over) both
          ``"max_evaluations_reached"`` and
          ``"mixed_or_incomplete_convergence"``, which both describe a run
          where at least some genuine optimization occurred; conflating
          this case with those would camouflage total collapse as an
          unremarkable partial result. ``raw_result`` is populated with a
          diagnostic dict (``{"error": ..., "restart_exception_messages":
          [...]}``) only in this case, ``None`` otherwise. This does NOT
          raise (unlike the even-rarer case where the baseline itself
          fails -- see Raises below) because a well-formed, if degenerate,
          result is still returned; ``optimizer_success`` is ``False`` and
          MUST be checked before trusting the result as a real
          optimization outcome.
        * ``"restart_raised_exception"`` -- set per-restart (see
          ``restart_records``) when a restart's COBYLA call itself raised;
          the shot budget it already consumed is still counted and other
          restarts still run. When this is true of *every* restart, the
          aggregate ``termination_reason`` is ``"all_restarts_failed"``
          (above), not a per-restart detail the caller must reconstruct.

        Each entry of ``restart_records`` additionally carries its own
        ``termination_reason`` with the same vocabulary, computed from that
        restart's own ``evaluations``/``budget``/``result.success``, so a
        caller can audit exactly which restart(s) actually converged.

        Args:
            max_evals: Hard cap on total objective evaluations across every
                restart plus the one uniform-state baseline evaluation.
            restarts: Number of independent COBYLA restarts sharing the
                ``max_evals`` budget (as evenly as possible).
            optimize_seed: Optional restart-initialization seed, default self.seed.
            measurement_seed: Optional objective-measurement seed, default
                self.seed + 190917. Independent of final sample_seed.
            objective: ``"cvar"`` (default) or ``"mean"``.
            cvar_alpha: Lower-tail CVaR quantile in ``(0, 1]`` -- the
                fraction of the ``eval_shots`` lowest-energy shots averaged
                by :func:`finite_shot_cvar`. Ignored when
                ``objective == "mean"``. Named explicitly (not the bare
                ``alpha`` used in earlier revisions) to avoid ambiguity
                with unrelated significance/learning-rate parameters
                elsewhere in this codebase.
            parameter_scale: ``"max_coefficient"`` (default) rescales gamma
                angles by the largest Ising coefficient; ``"feasible_iqr"``
                rescales by the interquartile range of feasible energies
                (requires enumerating the feasible space up front).
            eval_shots: Finite measurement shots drawn per objective
                evaluation during optimization (default 500). Must be a
                positive integer; there is no analytic-expectation escape
                hatch -- every evaluation is shot-noise-limited.

        Returns:
            An :class:`OptimizationResult` whose ``shot_ledger`` field
            records ``nfev``, ``eval_shots``, ``total_opt_shots`` (the
            total measurement shots this optimization run consumed --
            ``nfev * eval_shots``), ``cvar_alpha``, ``objective``,
            ``restarts`` and ``max_evals``; ``total_opt_shots`` and
            ``nfev`` are also mirrored as top-level fields for convenience.
            This method performs no *final output* sampling on its own --
            pair it with :meth:`sample` (or use
            :meth:`solve_with_measurement_ledger`, which composes both and
            returns the full measurement ledger, including the output shot
            count, as a plain ``dict``).

        Raises:
            ValueError: On any invalid argument, including an ``eval_shots``
                that is not a positive integer, or a ``max_evals`` too
                small to cover the uniform-state baseline plus ``2*p+2``
                evaluations per restart.
            OptimizationCollapseError: Only in the genuinely unrecoverable
                case where even the uniform-state baseline itself never
                completed, so no point of any kind was evaluated. This is
                NOT raised when every *restart* fails but the baseline
                succeeded -- that case returns normally with
                ``termination_reason == "all_restarts_failed"`` and
                ``optimizer_success = False`` (see above), so a caller can
                inspect the diagnostic ``raw_result`` rather than having to
                catch an exception for an outcome that still carries a
                well-formed (if degenerate) result.
        """
        if self.simulation_mode != "subspace":
            raise ValueError("Robust budgeted CVaR currently requires simulation_mode='subspace'")
        if (
            eval_shots is None
            or isinstance(eval_shots, bool)
            or not isinstance(eval_shots, (int, float))
            or int(eval_shots) != eval_shots
            or eval_shots <= 0
        ):
            raise ValueError(
                "eval_shots must be a positive integer -- there is no analytic-expectation "
                "fallback; every objective evaluation is finite-shot by design"
            )
        eval_shots = int(eval_shots)
        if objective not in ("mean", "cvar"):
            raise ValueError("objective must be 'mean' or 'cvar'")
        if not 0 < cvar_alpha <= 1:
            raise ValueError("cvar_alpha must lie in (0, 1]")
        if isinstance(restarts, bool) or int(restarts) != restarts or restarts < 1:
            raise ValueError("restarts must be a positive integer")
        restarts = int(restarts)
        if isinstance(max_evals, bool) or int(max_evals) != max_evals:
            raise ValueError("max_evals must be an integer")
        max_evals = int(max_evals)
        min_budget = 1 + restarts * (2 * self.p + 2)
        if max_evals < min_budget:
            raise ValueError(
                f"max_evals={max_evals} is too small for {restarts} restart(s) at "
                f"p={self.p}; need at least {min_budget} (1 uniform-state baseline + "
                f"2*p+2 evaluations per restart)"
            )
        if parameter_scale not in ("max_coefficient", "feasible_iqr"):
            raise ValueError("parameter_scale must be 'max_coefficient' or 'feasible_iqr'")

        scale = self.parameter_scale(parameter_scale)
        energies = self._subspace_energies
        # Explicit stream overrides separate initialization from noisy objectives.
        # None preserves historical replay; callers should persist supplied seeds.
        resolved_optimize_seed = self.seed if optimize_seed is None else optimize_seed
        resolved_measurement_seed = self.seed + 190917 if measurement_seed is None else measurement_seed
        rng = np.random.default_rng(resolved_optimize_seed)
        measurement_rng = np.random.default_rng(resolved_measurement_seed)
        history: List[float] = []
        best: dict = {}
        records: List[dict] = []
        nfev_counter = 0  # Explicit objective-evaluation counter (== len(history) by construction).
        evaluation_limit = 1  # Baseline first; each restart receives its own hard ceiling.

        class EvaluationBudgetReached(RuntimeError):
            """Stop before another measurement exceeds the allocated quota."""

        def evaluate(internal: np.ndarray) -> float:
            """Score one parameter point from ``eval_shots`` finite measurement shots.

            Samples ``eval_shots`` bitstrings (represented here by their
            Ising energies) from the exact Born distribution of the
            restricted-subspace wavefunction at ``internal`` (rescaled to
            physical gamma units) -- i.e. classically simulates finite-shot
            measurement noise. The declared ``objective`` (CVaR via
            :func:`finite_shot_cvar`, or a plain empirical mean) is
            computed *only* from those sampled energies; the exact
            analytic mean is separately retained purely as a diagnostic
            (``best_mean_energy`` bookkeeping), never as the value COBYLA
            actually receives or optimizes.
            """
            nonlocal nfev_counter
            if nfev_counter >= evaluation_limit:
                raise EvaluationBudgetReached("Objective measurement quota exhausted")
            parameters = np.asarray(internal, dtype=float).copy()
            if not np.isfinite(parameters).all():
                raise ValueError("COBYLA proposed non-finite parameters")
            parameters[: self.p] /= scale
            amplitudes = self.subspace_state(parameters)
            probabilities = np.abs(amplitudes) ** 2
            probability_mass = probabilities.sum()
            if not np.isfinite(probability_mass) or probability_mass <= 0:
                raise ValueError("Subspace wavefunction did not yield a normalizable distribution")
            probabilities = probabilities / probability_mass
            # Diagnostic only: the exact analytic mean, never handed to COBYLA as `value`.
            diagnostic_mean = float(probabilities @ energies)
            samples = measurement_rng.choice(energies, size=eval_shots, p=probabilities)
            if objective == "mean":
                value = float(samples.mean())
            else:
                value = finite_shot_cvar(samples, cvar_alpha)
            nfev_counter += 1
            history.append(value)
            if not best or value < best['objective']:
                best.update(objective=value, mean=diagnostic_mean, parameters=parameters.copy(),
                            evaluation=nfev_counter)
            return value

        try:
            evaluate(np.zeros(2 * self.p))  # Preparation is free; measurement is counted.
        except Exception as exc:
            raise_if_resource_error(exc, stage_hint="QAOA baseline")
            raise OptimizationCollapseError(
                "Uniform-state baseline evaluation failed; no usable optimization result"
            ) from exc
        quotas = [int(v) for v in np.full(restarts, (max_evals - 1) // restarts)]
        for i in range((max_evals - 1) % restarts):
            quotas[i] += 1
        for i, quota in enumerate(quotas):
            initial = np.concatenate([rng.uniform(0., .6, self.p), rng.uniform(.1, .8, self.p)])
            before = len(history)
            evaluation_limit = before + quota
            try:
                result = minimize(evaluate, initial, method="COBYLA", options={
                    "maxiter": quota, "rhobeg": .35, "catol": 1e-7})
            except EvaluationBudgetReached:
                records.append(dict(restart=i, budget=quota, evaluations=len(history)-before,
                    success=False, message="Hard objective quota reached",
                    termination_reason="max_evaluations_reached",
                    initial_objective=history[before] if len(history)>before else None,
                    final_objective=history[-1] if len(history)>before else None,
                    parameter_scale=scale))
                continue
            except Exception as exc:  # noqa: BLE001 - defensive: one bad restart must not lose the shot budget
                raise_if_resource_error(exc, stage_hint="QAOA restart")
                used = len(history) - before
                records.append(dict(
                    restart=i, budget=quota, evaluations=used,
                    success=False, message=f"restart raised {type(exc).__name__}: {exc}",
                    termination_reason="restart_raised_exception",
                    initial_objective=history[before] if used else None,
                    final_objective=history[-1] if used else None,
                    parameter_scale=scale,
                ))
                continue
            used = len(history) - before
            converged = bool(result.success) and used > 0 and used < quota
            restart_reason = (
                "max_evaluations_reached" if used >= quota
                else ("converged" if converged else "stopped_early_without_convergence")
            )
            records.append(dict(restart=i, budget=quota, evaluations=used,
                success=converged, message=str(result.message),
                termination_reason=restart_reason,
                initial_objective=history[before] if used else None,
                final_objective=float(result.fun) if used else None,
                parameter_scale=scale))
        if len(history) > max_evals:
            raise AssertionError("Measurement budget hard cutoff was violated")
        if nfev_counter != len(history):
            raise AssertionError("Internal nfev counter drifted from the evaluation history")
        if not best:
            # Genuinely unrecoverable: even the cost-free uniform-state
            # baseline (evaluated unconditionally, outside any try/except,
            # before the restart loop) never completed. There is no point
            # of any kind -- not even a degenerate one -- to return.
            raise OptimizationCollapseError(
                "optimize_robust evaluated no points whatsoever; the uniform-state "
                "baseline itself never completed (see the original exception above, "
                "if any, via exception chaining)"
            )

        nfev = nfev_counter
        total_opt_shots = nfev * eval_shots
        # All optimizer calls can fail after performing valid evaluations.
        # Preserve those evaluations and distinguish status from point provenance.
        all_restarts_failed = bool(records) and all(
            r["termination_reason"] == "restart_raised_exception" for r in records
        )
        raw_result: Optional[Dict[str, Any]] = None
        # Honest, strict aggregate convergence classification -- see the docstring.
        if all_restarts_failed:
            termination_reason = "all_restarts_failed"
            optimizer_success = False
            raw_result = {
                "error": "All restarts raised exceptions; completed evaluations are retained.",
                "restart_count": len(records),
                "search_evaluations": nfev - 1,
                "best_point_source": "baseline" if best['evaluation'] == 1 else "search",
                "restart_exception_messages": [r["message"] for r in records],
            }
        elif records and all(r["termination_reason"] == "converged" for r in records):
            termination_reason = "converged"
            optimizer_success = True
        elif nfev >= max_evals:
            termination_reason = "max_evaluations_reached"
            optimizer_success = False
        else:
            termination_reason = "mixed_or_incomplete_convergence"
            optimizer_success = False
        if all_restarts_failed:
            message = (
                "termination_reason=all_restarts_failed; every restart's COBYLA call "
                "raised an exception (see restart_records for the per-restart messages) "
                f"; {nfev - 1} search evaluations completed; returned point source="
                f"{'baseline' if best['evaluation'] == 1 else 'search'}; "
                "optimizer_success is False and must be checked by any "
                "caller before trusting this result as a real optimization outcome"
            )
        else:
            message = (
                f"termination_reason={termination_reason}; best point tracked across all "
                "evaluations regardless of any single restart's own convergence flag "
                "(see restart_records for per-restart detail); a False optimizer_success "
                "does not mean the returned parameters are unusable, only that COBYLA's "
                "own convergence test was not satisfied before the evaluation budget ran out"
            )
        shot_ledger: Dict[str, Any] = {
            "nfev": nfev,
            "eval_shots": eval_shots,
            "total_opt_shots": total_opt_shots,
            "restarts": restarts,
            "max_evals": max_evals,
            "objective": objective,
            "cvar_alpha": cvar_alpha,
            "best_evaluation": best['evaluation'],
            "optimize_seed": int(resolved_optimize_seed),
            "measurement_seed": int(resolved_measurement_seed),
            "best_point_source": "baseline" if best['evaluation'] == 1 else "search",
            # Gamma normalisation used by the optimizer: physical gamma =
            # internal gamma / parameter_scale. Needed to transfer parameters
            # between instances with different energy scales.
            "parameter_scale": float(scale),
            "parameter_scale_mode": parameter_scale,
        }
        gamma_best = best['parameters'][: self.p].copy()
        beta_best = best['parameters'][self.p:].copy()
        # Even when optimizer_success is False for the whole run (e.g. every restart's
        # maxiter budget ran out before its internal convergence test was met, or every
        # restart raised -- termination_reason=="all_restarts_failed"), `best` still
        # holds the lowest-objective point evaluated across the entire run, so the caller gets a
        # well-formed answer -- see gamma_best/beta_best below -- it just may not
        # reflect any real search; check optimizer_success/termination_reason first.
        return OptimizationResult(
            gammas=gamma_best, betas=beta_best,
            energy=best['mean'], history=tuple(history), success=optimizer_success,
            message=message,
            evaluations=nfev, objective_name=objective, objective_value=best['objective'],
            restart_records=tuple(records), total_opt_shots=total_opt_shots,
            nfev=nfev, eval_shots=eval_shots, shot_ledger=shot_ledger,
            optimizer_success=optimizer_success, termination_reason=termination_reason,
            gamma_best=gamma_best, beta_best=beta_best, best_mean_energy=best['mean'],
            raw_result=raw_result,
        )

    def solve_with_measurement_ledger(
        self, *,
        max_evals: int = 90, restarts: int = 4,
        objective: str = "cvar", cvar_alpha: float = 0.1,
        eval_shots: int = 500,
        parameter_scale: str = "max_coefficient",
        output_shots: int = 1000,
        energy_window: float = 2.0,
        ground_state: Optional[GroundStateResult] = None,
    ) -> Dict[str, Any]:
        """Run the budgeted finite-shot CVaR optimizer, then draw the final output sample.

        Composes :meth:`optimize_robust` (parameter search under a finite
        measurement budget) with :meth:`sample` (drawing ``output_shots``
        output measurements at the best parameters found -- ``gamma_best``/
        ``beta_best`` -- regardless of whether ``optimize_robust`` reports
        ``optimizer_success``), and returns everything as a single plain
        ``dict`` so it can be logged, serialized, or inspected without
        importing the dataclasses defined in this module.

        Args:
            max_evals, restarts, objective, cvar_alpha, eval_shots,
                parameter_scale: Forwarded to :meth:`optimize_robust`.
            output_shots: Number of output measurement shots drawn from the
                optimized circuit after optimization completes (default
                1000, matching this project's ``--outputs`` CLI
                convention). Must be a positive integer.
            energy_window: Forwarded to :meth:`sample` as its
                ``energy_window`` -- the energy window (default 2.0, same
                units as the physical objective) above the exact ground
                energy used for ``low_energy_fraction``.
            ground_state: Optional precomputed :class:`GroundStateResult`
                to avoid re-enumerating the feasible space when the caller
                already has one.

        Returns:
            A dict with keys ``"gammas"``/``"gamma_best"``,
            ``"betas"``/``"beta_best"``, ``"energy"``/``"best_mean_energy"``,
            ``"objective_name"``, ``"objective_value"``, ``"history"``,
            ``"optimizer_success"``, ``"termination_reason"``,
            ``"message"``, ``"restart_records"``, ``"quantum_sample"`` (the
            full :class:`QuantumSampleResult` from the final measurement,
            including ``bitstring_entropy``/``low_energy_fraction``/
            ``ground_state_hit``), the same three output-sampling
            diagnostics mirrored as top-level keys for convenience, and
            ``"measurement_ledger"`` -- a nested dict with ``nfev``,
            ``eval_shots``, ``total_opt_shots``, ``output_shots``, and
            ``total_measurement_shots`` (``total_opt_shots +
            output_shots``: every measurement -- optimization and output
            combined -- consumed by this call).

        Raises:
            ValueError: If ``output_shots`` is not a positive integer, or
                propagated from :meth:`optimize_robust` / :meth:`sample`.
        """
        if isinstance(output_shots, bool) or int(output_shots) != output_shots or output_shots <= 0:
            raise ValueError("output_shots must be a positive integer")
        output_shots = int(output_shots)

        optimized = self.optimize_robust(
            max_evals=max_evals, restarts=restarts, objective=objective, cvar_alpha=cvar_alpha,
            eval_shots=eval_shots, parameter_scale=parameter_scale,
        )
        # Always sample at the best parameters found, independent of optimizer_success.
        quantum_sample = self.sample(
            optimized, shots=output_shots, ground_state=ground_state, energy_window=energy_window,
        )

        total_opt_shots = int(optimized.shot_ledger.get("total_opt_shots", optimized.total_opt_shots))
        measurement_ledger: Dict[str, Any] = {
            "nfev": optimized.nfev,
            "eval_shots": optimized.eval_shots,
            "total_opt_shots": total_opt_shots,
            "output_shots": output_shots,
            "total_measurement_shots": total_opt_shots + output_shots,
            "restarts": restarts,
            "max_evals": max_evals,
            "objective": objective,
            "cvar_alpha": cvar_alpha,
        }
        return {
            "gammas": optimized.gammas,
            "betas": optimized.betas,
            "gamma_best": optimized.gamma_best,
            "beta_best": optimized.beta_best,
            "energy": optimized.energy,
            "best_mean_energy": optimized.best_mean_energy,
            "objective_name": optimized.objective_name,
            "objective_value": optimized.objective_value,
            "history": optimized.history,
            "optimizer_success": optimized.optimizer_success,
            "termination_reason": optimized.termination_reason,
            "success": optimized.success,
            "message": optimized.message,
            "restart_records": optimized.restart_records,
            "quantum_sample": quantum_sample,
            "bitstring_entropy": quantum_sample.bitstring_entropy,
            "low_energy_fraction": quantum_sample.low_energy_fraction,
            "ground_state_hit": quantum_sample.ground_state_hit,
            "measurement_ledger": measurement_ledger,
        }
