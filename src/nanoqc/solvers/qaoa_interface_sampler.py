"""Particle-number-preserving XY-mixer QAOA for interface rotamer sampling.

The module consumes the penalty-free physical terms exported by
``subgraph_to_qubo.py``.  Every residue site is represented by a local one-hot
register.  Both the diagonal cost unitary and the XY mixer preserve each
register's Hamming weight, so a weight-one initial state remains feasible
throughout the circuit without a one-hot penalty.

The exported physical objective follows the upper-triangular binary convention

    E(x) = sum_i a_i x_i + sum_{i<j} b_ij x_i x_j.

Consequently, the gate coefficients are obtained from ``x=(1-Z)/2`` before
applying ``RZ(2*gamma*h_i)`` and ``IsingZZ(2*gamma*J_ij)``.  Applying the raw
binary coefficients directly as Pauli coefficients would optimize a different
energy landscape.
"""

from __future__ import annotations

import argparse
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize

from nanoqc.qubo.subgraph_to_qubo import InterfaceQUBOBuilder, _virtual_pruned_graph, qubo_to_ising
from nanoqc.quantum.instance import QuantumOptimizationInstance


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


class XYMixerQAOASampler:
    """Penalty-free QAOA sampler on a product of one-hot residue registers.

    Args:
        physical_self: Binary linear coefficients with shape ``[M]``.
        physical_pair: Strictly upper-triangular binary pair coefficients with
            shape ``[M, M]``.
        site_to_variables: Mapping from residue-site index to its 2--6 global
            binary-variable indices.
        p: QAOA depth. Supported values are 1, 2, and 3.
        shots: Default number of measurement shots.
        initial_state: Must be ``"wstate"``. Each residue register is prepared
            as an equal local Hamming-weight-one Dicke state.
        device_name: PennyLane device used for exact optimization and sampling.
        seed: Reproducibility seed.
        coefficient_tolerance: Coefficients below this magnitude are omitted
            from the circuit.

    Notes:
        PennyLane defines ``IsingXY(phi)`` as
        ``exp(+i phi (XX+YY)/4)``.  Therefore ``phi=-2*beta`` implements the
        requested ``exp(-i beta (XX+YY)/2)``.  For multi-state sites, applying
        all local pairs sequentially is a first-order product formula for the
        sum mixer; every factor separately preserves local Hamming weight.
    """

    def __init__(
        self,
        physical_self: Sequence[float],
        physical_pair: Sequence[Sequence[float]],
        site_to_variables: Mapping[int, Sequence[int]],
        *,
        p: int = 2,
        shots: int = 1024,
        initial_state: str = "wstate",
        device_name: str = "default.qubit",
        seed: int = 7,
        coefficient_tolerance: float = 1e-10,
        simulation_mode: str = "pennylane",
    ) -> None:
        self.physical_self = np.asarray(physical_self, dtype=np.float64)
        self.physical_pair = np.asarray(physical_pair, dtype=np.float64)
        self.site_to_variables = {
            int(site): tuple(int(variable) for variable in variables)
            for site, variables in sorted(site_to_variables.items())
        }
        self.p = int(p)
        self.shots = int(shots)
        self.initial_state = str(initial_state).lower()
        self.device_name = str(device_name)
        self.seed = int(seed)
        self.coefficient_tolerance = float(coefficient_tolerance)
        self.num_variables = int(self.physical_self.size)
        if simulation_mode not in ("pennylane", "subspace"):
            raise ValueError("simulation_mode must be pennylane or subspace")
        self.simulation_mode = simulation_mode

        self._validate_inputs()
        self._variable_to_site = {
            variable: site
            for site, variables in self.site_to_variables.items()
            for variable in variables
        }
        self._mixer_pairs = tuple(
            (site, left, right)
            for site, variables in self.site_to_variables.items()
            for left, right in itertools.combinations(variables, 2)
        )
        if any(
            self._variable_to_site[left] != site
            or self._variable_to_site[right] != site
            for site, left, right in self._mixer_pairs
        ):
            raise AssertionError("XY mixer pair crosses residue-register boundaries")
        self._feasible_energy_cache: Optional[Dict[BitString, float]] = None

        physical_qubo = self.physical_pair.copy()
        np.fill_diagonal(physical_qubo, self.physical_self)
        self.ising_h, self.ising_J, self.ising_offset = qubo_to_ising(physical_qubo)
        self.gamma_scale = max(
            float(np.max(np.abs(self.ising_h), initial=0.0)),
            float(np.max(np.abs(self.ising_J), initial=0.0)),
            1.0,
        )
        self._hamiltonian = self._build_cost_hamiltonian()
        self._assert_xy_number_conservation()
        self._energy_qnode = self._make_energy_qnode()

    @classmethod
    def from_instance(
        cls,
        instance: QuantumOptimizationInstance,
        *,
        p: int = 2,
        shots: int = 1024,
        initial_state: str = "wstate",
        device_name: str = "default.qubit",
        seed: int = 7,
        coefficient_tolerance: float = 1e-10,
        simulation_mode: str = "pennylane",
    ) -> "XYMixerQAOASampler":
        """Construct the solver strictly from the frozen quantum-instance contract."""
        return cls(
            instance.physical_self,
            instance.physical_pair,
            instance.site_to_variables,
            p=p,
            shots=shots,
            initial_state=initial_state,
            device_name=device_name,
            seed=seed,
            coefficient_tolerance=coefficient_tolerance,
            simulation_mode=simulation_mode,
        )

    def _validate_inputs(self) -> None:
        """Validate shape, one-hot partition, and NISQ-size assumptions."""

        m = self.num_variables
        if isinstance(self.p, bool) or not 1 <= self.p <= MAX_QAOA_DEPTH:
            raise ValueError(f"p must be an integer in 1..{MAX_QAOA_DEPTH}")
        if self.shots <= 0:
            raise ValueError("shots must be positive")
        if self.initial_state != "wstate":
            raise ValueError("initial_state must be 'wstate' for legal Dicke preparation")
        if m == 0 or m > 30:
            raise ValueError(f"Expected 1--30 variables, received {m}")
        if self.physical_pair.shape != (m, m):
            raise ValueError(
                f"physical_pair must have shape {(m, m)}, got {self.physical_pair.shape}"
            )
        if not np.isfinite(self.physical_self).all() or not np.isfinite(
            self.physical_pair
        ).all():
            raise ValueError("Physical coefficients must all be finite")
        lower = np.tril(self.physical_pair, k=-1)
        if not np.allclose(lower, 0.0, atol=self.coefficient_tolerance):
            raise ValueError("physical_pair must use the upper-triangular convention")
        if not np.allclose(
            np.diag(self.physical_pair), 0.0, atol=self.coefficient_tolerance
        ):
            raise ValueError("physical_pair diagonal must be zero; use physical_self")

        variables: List[int] = []
        for site, group in self.site_to_variables.items():
            if not 2 <= len(group) <= 6:
                raise ValueError(
                    f"Site {site} must contain 2 to 6 candidate variables, got {len(group)}"
                )
            if len(set(group)) != len(group):
                raise ValueError(f"Site {site} contains duplicate variable indices")
            variables.extend(group)
        if sorted(variables) != list(range(m)):
            raise ValueError(
                "site_to_variables must partition every variable index exactly once"
            )

    @staticmethod
    def _assert_xy_number_conservation(tolerance: float = 1e-12) -> None:
        """Prove the two-qubit XY generator commutes with excitation number.

        In the computational basis, ``H_XY=(XX+YY)/2`` only connects ``|01>``
        and ``|10>``.  The explicit commutator check below guards the gate-sign
        and basis convention used by this implementation.
        """

        pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        pauli_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
        number = np.diag([0.0, 1.0, 1.0, 2.0]).astype(np.complex128)
        h_xy = 0.5 * (
            np.kron(pauli_x, pauli_x) + np.kron(pauli_y, pauli_y)
        )
        commutator = h_xy @ number - number @ h_xy
        if np.linalg.norm(commutator) > tolerance:
            raise AssertionError("XY generator does not conserve particle number")

    @property
    def feasible_configuration_count(self) -> int:
        """Return the size of the local one-hot search space."""

        return math.prod(len(group) for group in self.site_to_variables.values())

    def physical_energy(self, bitstring: Sequence[int]) -> float:
        """Evaluate the penalty-free binary physical objective."""

        bits = np.asarray(bitstring, dtype=np.float64)
        if bits.shape != (self.num_variables,):
            raise ValueError(
                f"Expected bitstring shape {(self.num_variables,)}, got {bits.shape}"
            )
        return float(
            self.physical_self @ bits + np.einsum(
                "i,ij,j->", bits, self.physical_pair, bits, optimize=True
            )
        )

    def is_legal(self, bitstring: Sequence[int]) -> bool:
        """Return whether every residue register has Hamming weight one."""

        bits = np.asarray(bitstring, dtype=np.int8)
        if bits.shape != (self.num_variables,):
            return False
        if np.any((bits != 0) & (bits != 1)):
            return False
        return all(int(bits[list(group)].sum()) == 1 for group in self.site_to_variables.values())

    def _build_cost_hamiltonian(self) -> qml.Hamiltonian:
        """Construct the Pauli Hamiltonian equivalent to the binary objective."""

        coefficients: List[float] = []
        operators: List[qml.operation.Operator] = []
        tol = self.coefficient_tolerance
        for wire, coefficient in enumerate(self.ising_h):
            if abs(float(coefficient)) > tol:
                coefficients.append(float(coefficient))
                operators.append(qml.PauliZ(wire))
        for left in range(self.num_variables):
            for right in range(left + 1, self.num_variables):
                coefficient = float(self.ising_J[left, right])
                if abs(coefficient) > tol:
                    coefficients.append(coefficient)
                    operators.append(qml.PauliZ(left) @ qml.PauliZ(right))
        if not coefficients:
            coefficients = [0.0]
            operators = [qml.Identity(0)]
        return qml.Hamiltonian(coefficients, operators)

    def _prepare_initial_state(self) -> None:
        """Prepare an exact product of local equal-amplitude W/Dicke states."""

        for group in self.site_to_variables.values():
            size = len(group)
            local_state = np.zeros(2**size, dtype=np.complex128)
            for local_wire in range(size):
                basis_index = 1 << (size - 1 - local_wire)
                local_state[basis_index] = 1.0 / math.sqrt(size)
            if not np.isclose(np.vdot(local_state, local_state).real, 1.0):
                raise AssertionError("Local W-state amplitudes are not normalized")
            qml.StatePrep(local_state, wires=group, normalize=False)

    def _apply_cost(self, gamma: float) -> None:
        """Apply the exact diagonal physical-cost evolution, up to global phase."""

        tol = self.coefficient_tolerance
        for wire, coefficient in enumerate(self.ising_h):
            if abs(float(coefficient)) > tol:
                qml.RZ(2.0 * gamma * float(coefficient), wires=wire)
        for left in range(self.num_variables):
            for right in range(left + 1, self.num_variables):
                coefficient = float(self.ising_J[left, right])
                if abs(coefficient) > tol:
                    qml.IsingZZ(
                        2.0 * gamma * coefficient, wires=(left, right)
                    )

    def _apply_xy_mixer(self, beta: float, layer: int) -> None:
        """Apply local number-preserving XY rotations to every candidate pair."""

        for site in self.site_to_variables:
            pairs = [
                (left, right)
                for pair_site, left, right in self._mixer_pairs
                if pair_site == site
            ]
            if layer % 2:
                pairs.reverse()
            for left, right in pairs:
                qml.IsingXY(-2.0 * beta, wires=(left, right))

    def _ansatz(self, flat_parameters: Sequence[float]) -> None:
        """Build the depth-p alternating cost/mixer circuit."""

        self._prepare_initial_state()
        gammas = flat_parameters[: self.p]
        betas = flat_parameters[self.p :]
        for layer in range(self.p):
            self._apply_cost(gammas[layer])
            self._apply_xy_mixer(betas[layer], layer)

    def _make_energy_qnode(self):
        """Create the exact expectation-value QNode used by the optimizer."""

        device = qml.device(
            self.device_name, wires=self.num_variables, shots=None, seed=self.seed
        )

        @qml.qnode(device, interface="autograd", diff_method="best")
        def energy_qnode(flat_parameters: Sequence[float]):
            self._ansatz(flat_parameters)
            return qml.expval(self._hamiltonian)

        return energy_qnode

    def expected_energy(self, flat_parameters: Sequence[float]) -> float:
        """Return the exact expected physical energy, including Ising offset."""

        if self.simulation_mode == "subspace":
            state = self.subspace_state(flat_parameters)
            return float(np.abs(state) ** 2 @ self._subspace_energies)
        # COBYLA needs values only; autograd/adjoint would compute unused Jacobians.
        if not hasattr(self, "_forward_energy_qnode"):
            self._forward_energy_qnode = qml.QNode(
                self._energy_qnode.func, self._energy_qnode.device,
                interface=None, diff_method=None,
            )
        parameters = np.asarray(flat_parameters, dtype=np.float64)
        return float(self._forward_energy_qnode(parameters) + self.ising_offset)

    def subspace_state(self, parameters: Sequence[float]) -> np.ndarray:
        """Exactly simulate the same ordered XY gates in the legal product basis.

        This is classical restricted-state simulation, not a hardware execution.
        The physical cost includes the global Ising offset, irrelevant to probabilities.
        """
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (2 * self.p,):
            raise ValueError("Expected 2*p parameters")
        if not hasattr(self, "_subspace_bits"):
            self._subspace_bits = np.asarray(list(self.feasible_energy_map()), dtype=np.int8)
            bits = self._subspace_bits
            # Match coefficient truncation in the PennyLane circuit exactly.
            z = 1.0 - 2.0 * bits
            h = np.where(np.abs(self.ising_h) > self.coefficient_tolerance, self.ising_h, 0)
            j = np.where(np.abs(self.ising_J) > self.coefficient_tolerance, self.ising_J, 0)
            self._subspace_energies = z @ h + np.einsum('bi,ij,bj->b', z, j, z) + self.ising_offset
        dims = tuple(len(g) for g in self.site_to_variables.values())
        state = np.ones(dims, dtype=np.complex128) / math.sqrt(math.prod(dims))
        for layer in range(self.p):
            state *= np.exp(-1j * parameters[layer] * self._subspace_energies.reshape(dims))
            beta = parameters[self.p + layer]
            for axis, group in enumerate(self.site_to_variables.values()):
                pairs = list(itertools.combinations(range(len(group)), 2))
                if layer % 2:
                    pairs.reverse()
                view = np.moveaxis(state, axis, 0)
                for left, right in pairs:
                    a, b = view[left].copy(), view[right].copy()
                    view[left] = math.cos(beta) * a - 1j * math.sin(beta) * b
                    view[right] = math.cos(beta) * b - 1j * math.sin(beta) * a
        return state.reshape(-1)

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


def _format_trace(history: Sequence[float]) -> str:
    """Format a compact convergence trace without hiding the final value."""

    selected = list(enumerate(history, start=1))
    return ", ".join(f"{step}:{energy:.6f}" for step, energy in selected)


def _positive_int(raw: str) -> int:
    """argparse type= callback: a strictly positive integer."""

    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{raw!r} is not an integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{raw!r} must be positive")
    return value


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("robust", "exact"), default="robust",
        help="'robust' (default): budgeted multistart COBYLA over a finite-shot "
             "CVaR objective (optimize_robust / solve_with_measurement_ledger). "
             "'exact': legacy single-start optimizer over the noiseless analytic "
             "expectation value (optimize).",
    )
    parser.add_argument("--iterations", type=_positive_int, default=18,
                         help="max_iterations for --mode exact")
    parser.add_argument("--shots", type=_positive_int, default=1024,
                         help="Shots for the non-robust-mode final sample() call")
    parser.add_argument("--eval-shots", type=_positive_int, default=500,
                         help="Finite measurement shots per COBYLA objective "
                              "evaluation in --mode robust")
    parser.add_argument("--cvar-alpha", type=float, default=0.1,
                         help="CVaR lower-tail quantile in (0, 1] for --mode robust "
                              "(matches this project's --cvar-alpha convention, e.g. "
                              "in batch_benchmark_hard_set.py)")
    parser.add_argument("--restarts", type=_positive_int, default=4,
                         help="COBYLA restarts sharing --max-evals in --mode robust")
    parser.add_argument("--max-evals", type=_positive_int, default=90,
                         help="Total objective-evaluation budget across all restarts "
                              "(including the uniform-state baseline) in --mode robust")
    parser.add_argument("--outputs", type=_positive_int, default=1000,
                         help="Output measurement shots drawn after optimization "
                              "in --mode robust (matches this project's --outputs "
                              "convention)")
    parser.add_argument("--sa-reads", type=_positive_int, default=192)
    parser.add_argument("--sa-sweeps", type=_positive_int, default=600)
    parser.add_argument("--optimizer", choices=("cobyla", "adam"), default="cobyla",
                         help="Optimizer for --mode exact")
    parser.add_argument("--initial-state", choices=("wstate",), default="wstate")
    parser.add_argument("--device", default="lightning.qubit")
    parser.add_argument("--seed", type=int, default=19)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the required two-layer virtual-interface integration test."""

    args = _build_argument_parser().parse_args(argv)
    if not 0 < args.cvar_alpha <= 1:
        raise SystemExit("--cvar-alpha must lie in (0, 1]")
    graph = _virtual_pruned_graph()
    qubo = InterfaceQUBOBuilder().build(graph)
    sampler = XYMixerQAOASampler(
        qubo.physical_self,
        qubo.physical_pair,
        qubo.site_to_variables,
        p=2,
        shots=args.shots,
        initial_state=args.initial_state,
        device_name=args.device,
        seed=args.seed,
        simulation_mode="subspace" if args.mode == "robust" else "pennylane",
    )

    # Verify binary and Pauli objectives agree for a deterministic legal state.
    check_state = np.zeros(sampler.num_variables, dtype=np.int8)
    for group in sampler.site_to_variables.values():
        check_state[group[0]] = 1
    spins = 1.0 - 2.0 * check_state
    pauli_energy = float(
        sampler.ising_offset
        + sampler.ising_h @ spins
        + np.einsum("i,ij,j->", spins, sampler.ising_J, spins, optimize=True)
    )
    assert np.isclose(
        sampler.physical_energy(check_state), pauli_energy, atol=1e-8
    ), "Binary-to-Ising conversion changed the physical objective"

    ground = sampler.enumerate_ground_states()

    if args.mode == "robust":
        solved = sampler.solve_with_measurement_ledger(
            max_evals=args.max_evals, restarts=args.restarts,
            cvar_alpha=args.cvar_alpha, eval_shots=args.eval_shots,
            output_shots=args.outputs, ground_state=ground,
        )
        history = solved["history"]
        energy = solved["energy"]
        quantum = solved["quantum_sample"]
        ledger = solved["measurement_ledger"]
        optimizer_success = solved["optimizer_success"]
        termination_reason = solved["termination_reason"]
        optimizer_label = f"COBYLA x{args.restarts} restarts (finite-shot CVaR)"
    else:
        optimized = sampler.optimize(method=args.optimizer, max_iterations=args.iterations)
        history = optimized.history
        energy = optimized.energy
        quantum = sampler.sample(optimized, shots=args.shots, ground_state=ground)
        ledger = None
        optimizer_success = optimized.optimizer_success
        termination_reason = optimized.termination_reason
        optimizer_label = args.optimizer.upper()

    annealing = sampler.simulated_annealing(
        num_reads=args.sa_reads,
        sweeps=args.sa_sweeps,
        ground_state=ground,
    )

    assert quantum.legal_rate == 1.0
    assert sampler.is_legal(annealing.best_state)
    print("XY-Mixer QAOA virtual rotamer benchmark")
    print(f"PennyLane version: {qml.__version__}")
    print(
        f"Sites / variables / feasible states: "
        f"{len(sampler.site_to_variables)} / {sampler.num_variables} / "
        f"{ground.configuration_count}"
    )
    print(f"Depth / optimizer: p={sampler.p} / {optimizer_label}")
    print(f"Convergence trace (evaluation:energy): {_format_trace(history)}")
    print(f"Optimized expected physical energy: {energy:.8f}")
    print(f"Exact feasible ground energy:       {ground.energy:.8f}")
    print(f"optimizer_success / termination_reason: {optimizer_success} / {termination_reason}")
    print(f"Measurement legality:              {100.0 * quantum.legal_rate:.2f}%")
    print(
        f"QAOA best / ground success:         {quantum.best_energy:.8f} / "
        f"{100.0 * quantum.ground_state_success_probability:.2f}% "
        f"({quantum.shots} shots)"
    )
    print(
        f"bitstring_entropy (nats) / low_energy_fraction / ground_state_hit: "
        f"{quantum.bitstring_entropy:.4f} / {quantum.low_energy_fraction:.4f} / "
        f"{quantum.ground_state_hit}"
    )
    print(
        f"SA best / ground success:           {annealing.best_energy:.8f} / "
        f"{100.0 * annealing.ground_state_success_probability:.2f}% "
        f"({args.sa_reads} reads)"
    )
    print(f"QAOA best state: {''.join(map(str, quantum.best_states[0]))}")
    print(f"SA best state:   {''.join(map(str, annealing.best_state))}")
    if ledger is not None:
        print(f"Measurement ledger (dict):          {ledger}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
