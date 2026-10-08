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
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pennylane as qml

from nanoqc.qubo.subgraph_to_qubo import InterfaceQUBOBuilder, _virtual_pruned_graph, qubo_to_ising
from nanoqc.quantum.instance import QuantumOptimizationInstance

# Names from the split-out modules that this module or its callers use.
from nanoqc.solvers.qaoa_results import (  # noqa: E402,F401
    BitString,
    GROUND_ENERGY_TOLERANCE,
    MAX_QAOA_DEPTH,
    finite_shot_cvar,
)
from nanoqc.solvers.qaoa_optimization import QAOAOptimizationMixin
from nanoqc.solvers.qaoa_sampling import QAOASamplingMixin


class XYMixerQAOASampler(QAOAOptimizationMixin, QAOASamplingMixin):
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
        scale_excluded_variables: Sequence[int] = (),
        scale_excluded_pairs: Sequence[Sequence[int]] = (),
    ) -> None:
        self.physical_self = np.asarray(physical_self, dtype=np.float64)
        # A54: geometry-forbidden states (A47) stay in the cost Hamiltonian but
        # are a constraint encoding, like the one-hot penalties XY-QAOA omits,
        # so they do not set the angle scale.
        self.scale_excluded_variables = tuple(sorted({int(v) for v in scale_excluded_variables}))
        self.scale_excluded_pairs = tuple(sorted({(min(int(a), int(b)), max(int(a), int(b)))
                                                  for a, b in scale_excluded_pairs}))
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
        scale_qubo = physical_qubo.copy()
        for v in self.scale_excluded_variables:
            if not 0 <= v < self.num_variables:
                raise ValueError("Scale-excluded variable outside the instance")
            scale_qubo[v, :] = 0.0
            scale_qubo[:, v] = 0.0
        for a, b in self.scale_excluded_pairs:
            if not (0 <= a < b < self.num_variables):
                raise ValueError("Scale-excluded pair outside the instance")
            scale_qubo[a, b] = 0.0
        scale_h, scale_j, _ = qubo_to_ising(scale_qubo)
        self.gamma_scale = max(
            float(np.max(np.abs(scale_h), initial=0.0)),
            float(np.max(np.abs(scale_j), initial=0.0)),
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
            scale_excluded_variables=(instance.metadata or {}).get("forbidden_variables", ()) or (),
            scale_excluded_pairs=(instance.metadata or {}).get("forbidden_variable_pairs", ()) or (),
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
