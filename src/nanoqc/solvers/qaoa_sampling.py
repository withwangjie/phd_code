"""Exact enumeration, measurement sampling and the simulated-annealing baseline of XYMixerQAOASampler.

The methods below are XYMixerQAOASampler's own (it inherits them from this
mixin); they were split out of qaoa_interface_sampler.py unchanged.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import pennylane as qml
from nanoqc.solvers.qaoa_results import AnnealingResult, BitString, GROUND_ENERGY_TOLERANCE, GroundStateResult, OptimizationResult, QuantumSampleResult


class QAOASamplingMixin:
    """Enumeration, sampling and annealing methods of :class:`XYMixerQAOASampler` (see qaoa_interface_sampler.py)."""

    def enumerate_ground_states(self, tolerance: float = GROUND_ENERGY_TOLERANCE) -> GroundStateResult:
        """Exactly enumerate the feasible rotamer assignments.

        The current builder emits 3--6 candidates per residue under a 30-bit budget; legacy 2-state registers remain supported for regression tests. The
        legal product space small enough for an exact benchmark oracle.
        """

        energy_map = self.feasible_energy_map()
        minimum = min(energy_map.values())
        ground_states = tuple(
            state
            for state, energy in energy_map.items()
            if math.isclose(energy, minimum, abs_tol=tolerance, rel_tol=0.0)
        )
        return GroundStateResult(
            energy=minimum,
            states=ground_states,
            configuration_count=len(energy_map),
        )

    def feasible_energy_map(self) -> Mapping[BitString, float]:
        """Return exact physical energies for every legal product assignment."""

        if self._feasible_energy_cache is None:
            energies: Dict[BitString, float] = {}
            for assignment in itertools.product(*self.site_to_variables.values()):
                state = np.zeros(self.num_variables, dtype=np.int8)
                state[np.asarray(assignment, dtype=np.int64)] = 1
                bitstring = tuple(int(value) for value in state)
                energies[bitstring] = self.physical_energy(state)
            self._feasible_energy_cache = energies
        return self._feasible_energy_cache

    def low_energy_states(
        self, energy_ceiling: float, *, tolerance: float = GROUND_ENERGY_TOLERANCE
    ) -> Tuple[BitString, ...]:
        """Enumerate legal states whose energy is at most ``energy_ceiling``."""

        return tuple(
            state
            for state, energy in self.feasible_energy_map().items()
            if energy <= energy_ceiling + tolerance
        )

    def sample(
        self,
        optimization: OptimizationResult | Sequence[float],
        *,
        shots: Optional[int] = None,
        ground_state: Optional[GroundStateResult] = None,
        energy_window: float = 2.0,
        sample_seed: Optional[int] = None,
    ) -> QuantumSampleResult:
        """Draw finite-shot bitstrings and report feasibility, success rate and diagnostics.

        Always executes at the parameters actually supplied (typically the
        best-ever point from :meth:`optimize_robust`/:meth:`optimize`,
        ``gamma_best``/``beta_best``), independent of that optimization's
        own convergence flag -- callers should draw this final output
        sample even when ``optimizer_success`` was ``False``.

        Args:
            optimization: An :class:`OptimizationResult` (its
                ``gammas``/``betas`` are used) or a flat ``[2*p]``
                parameter array directly.
            shots: Number of output measurement shots (default
                ``self.shots``). Must be positive.
            sample_seed: Explicit final-readout seed. None preserves the legacy
                self.seed + 1 behavior. Supply distinct seeds for independent repeats.
            ground_state: Optional precomputed :class:`GroundStateResult`
                to avoid re-enumerating the feasible space.
            energy_window: Energy window (same units as the physical
                objective, default 2.0) above the exact ground energy used
                for ``low_energy_fraction``; matches this project's
                existing hand-computed ``low_energy_fraction`` convention
                (``energies <= E_ground + energy_window``). Must be
                finite and non-negative.

        Returns:
            A :class:`QuantumSampleResult`. Its ``bitstring_entropy``,
            ``low_energy_fraction`` and ``ground_state_hit`` are computed
            purely from this finite output sample:

            * ``bitstring_entropy``: empirical Shannon entropy, in nats,
              of the observed bitstring frequency distribution
              (``-sum(p * ln(p))`` over the distinct sampled bitstrings).
            * ``low_energy_fraction``: fraction of the ``shots`` samples
              whose physical energy falls within ``energy_window`` of the
              exact ground energy (``energy <= E_ground + energy_window``).
            * ``ground_state_hit``: whether at least one of the ``shots``
              samples exactly matched a known global-optimal (ground)
              state -- equivalently ``ground_state_success_probability > 0``.

        Raises:
            ValueError: If ``shots``/``energy_window`` are invalid, or the
                parameter array has the wrong shape.
            AssertionError: If particle-number preservation failed (an
                illegal one-hot state was sampled).
        """

        requested_shots = self.shots if shots is None else shots
        if isinstance(requested_shots, bool) or int(requested_shots) != requested_shots:
            raise ValueError("shots must be a positive integer")
        sample_count = int(requested_shots)
        if sample_count <= 0:
            raise ValueError("shots must be positive")
        if not math.isfinite(energy_window) or energy_window < 0:
            raise ValueError("energy_window must be a finite, non-negative number")
        if isinstance(optimization, OptimizationResult):
            parameters = np.concatenate([optimization.gammas, optimization.betas])
        else:
            parameters = np.asarray(optimization, dtype=np.float64)
        if parameters.shape != (2 * self.p,) or not np.isfinite(parameters).all():
            raise ValueError(f"Expected {2 * self.p} optimized parameters")
        readout_seed = self.seed + 1 if sample_seed is None else sample_seed

        if self.simulation_mode == "subspace":
            probabilities = np.abs(self.subspace_state(parameters)) ** 2
            probabilities /= probabilities.sum()
            indices = np.random.default_rng(readout_seed).choice(len(probabilities), size=sample_count, p=probabilities)
            raw_samples = self._subspace_bits[indices]
        else:
            device = qml.device(self.device_name, wires=self.num_variables,
                                shots=sample_count, seed=readout_seed)

            @qml.qnode(device, interface=None, diff_method=None)
            def sampling_qnode(flat_parameters: Sequence[float]):
                self._ansatz(flat_parameters)
                return qml.sample(wires=range(self.num_variables))

            raw_samples = np.asarray(sampling_qnode(parameters), dtype=np.int8)
        if raw_samples.ndim == 1:
            raw_samples = raw_samples[None, :]
        bitstrings = [tuple(int(value) for value in row) for row in raw_samples]
        counts: Counter[BitString] = Counter(bitstrings)
        legal = np.fromiter(
            (self.is_legal(bitstring) for bitstring in bitstrings),
            dtype=np.bool_,
            count=len(bitstrings),
        )
        legal_rate = float(np.mean(legal))
        if legal_rate < 1.0 - 1e-12:
            raise AssertionError(
                "Particle-number preservation failed: sampled an illegal one-hot state"
            )

        energies = np.asarray(
            [self.physical_energy(bitstring) for bitstring in bitstrings],
            dtype=np.float64,
        )
        best_energy = float(np.min(energies))
        best_states = tuple(
            sorted(
                {
                    bitstrings[index]
                    for index in np.flatnonzero(
                        np.isclose(energies, best_energy, atol=GROUND_ENERGY_TOLERANCE, rtol=0.0)
                    )
                }
            )
        )
        exact = ground_state or self.enumerate_ground_states()
        success_probability = float(
            np.mean(np.isclose(energies, exact.energy, atol=GROUND_ENERGY_TOLERANCE, rtol=0.0))
        )
        # Empirical Shannon entropy (nats) of the observed bitstring distribution.
        # `counts` holds only observed (nonzero-count) bitstrings, so no 0*ln(0) term arises.
        frequencies = np.array(list(counts.values()), dtype=np.float64) / sample_count
        bitstring_entropy = float(-np.sum(frequencies * np.log(frequencies)))
        low_energy_fraction = float(np.mean(energies <= exact.energy + energy_window))
        ground_state_hit = bool(success_probability > 0.0)
        return QuantumSampleResult(
            counts=dict(counts),
            shots=sample_count,
            legal_rate=legal_rate,
            best_energy=best_energy,
            best_states=best_states,
            ground_state_success_probability=success_probability,
            mean_energy=float(np.mean(energies)),
            bitstring_entropy=bitstring_entropy,
            low_energy_fraction=low_energy_fraction,
            ground_state_hit=ground_state_hit,
        )

    def simulated_annealing(
        self,
        *,
        num_reads: int = 256,
        sweeps: int = 800,
        site_passes: Optional[int] = None,
        seed: Optional[int] = None,
        initial_temperature: Optional[float] = None,
        final_temperature: Optional[float] = None,
        ground_state: Optional[GroundStateResult] = None,
    ) -> AnnealingResult:
        """Run feasible SA. Legacy sweeps counts single-site proposals.

        site_passes, when supplied, overrides sweeps with passes * site_count.
        Sites are sampled randomly; one pass is expected coverage, not a scan.
        """
        if site_passes is not None:
            if isinstance(site_passes, bool) or int(site_passes) != site_passes or site_passes <= 0:
                raise ValueError("site_passes must be a positive integer")
            sweeps = int(site_passes) * len(self.site_to_variables)

        if num_reads <= 0 or sweeps <= 0:
            raise ValueError("num_reads and sweeps must be positive")
        coefficient_scale = max(
            float(np.max(np.abs(self.physical_self), initial=0.0)),
            float(np.max(np.abs(self.physical_pair), initial=0.0)),
            1e-3,
        )
        start_temp = (
            2.5 * coefficient_scale
            if initial_temperature is None
            else float(initial_temperature)
        )
        end_temp = (
            max(1e-4, 0.0025 * coefficient_scale)
            if final_temperature is None
            else float(final_temperature)
        )
        if start_temp <= 0.0 or end_temp <= 0.0 or start_temp <= end_temp:
            raise ValueError("Require initial_temperature > final_temperature > 0")

        rng = np.random.default_rng(self.seed + 2 if seed is None else seed)
        groups = tuple(self.site_to_variables.values())
        read_energies = np.empty(num_reads, dtype=np.float64)
        read_states: List[BitString] = []
        best_energy_trace: List[float] = []
        global_best = math.inf
        global_state: Optional[BitString] = None
        cooling = (end_temp / start_temp) ** (1.0 / max(1, sweeps - 1))

        for read in range(num_reads):
            state = np.zeros(self.num_variables, dtype=np.int8)
            chosen = []
            for group in groups:
                variable = int(rng.choice(group))
                chosen.append(variable)
                state[variable] = 1
            energy = self.physical_energy(state)
            temperature = start_temp

            for _ in range(sweeps):
                site = int(rng.integers(0, len(groups)))
                alternatives = [
                    variable for variable in groups[site] if variable != chosen[site]
                ]
                proposal = int(rng.choice(alternatives))
                previous = chosen[site]
                # The one-hot move changes only one variable.  Evaluate its
                # local QUBO delta instead of rescanning the complete energy
                # table for every proposal (the latter was a measurable cost
                # in matched-time SA runs).
                delta = float(self.physical_self[proposal] - self.physical_self[previous])
                for selected in chosen:
                    if selected == previous:
                        continue
                    left, right = sorted((proposal, int(selected)))
                    new_pair = float(self.physical_pair[left, right])
                    left, right = sorted((previous, int(selected)))
                    old_pair = float(self.physical_pair[left, right])
                    delta += new_pair - old_pair
                state[previous] = 0
                state[proposal] = 1
                proposed_energy = energy + delta
                accept = delta <= 0.0 or rng.random() < math.exp(
                    -delta / max(temperature, 1e-15)
                )
                if accept:
                    chosen[site] = proposal
                    energy = proposed_energy
                else:
                    state[proposal] = 0
                    state[previous] = 1
                temperature *= cooling

            final_state = tuple(int(value) for value in state)
            read_states.append(final_state)
            read_energies[read] = energy
            if energy < global_best:
                global_best = float(energy)
                global_state = final_state
            best_energy_trace.append(global_best)

        if global_state is None:
            raise RuntimeError("Simulated annealing produced no states")
        exact = ground_state or self.enumerate_ground_states()
        success_probability = float(
            np.mean(np.isclose(read_energies, exact.energy, atol=GROUND_ENERGY_TOLERANCE, rtol=0.0))
        )
        return AnnealingResult(
            counts=dict(Counter(read_states)),
            best_energy=global_best,
            best_state=global_state,
            ground_state_success_probability=success_probability,
            read_energies=read_energies,
            best_energy_trace=tuple(best_energy_trace),
        )
