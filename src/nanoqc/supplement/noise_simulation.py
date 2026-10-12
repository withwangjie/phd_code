#!/usr/bin/env python3
"""A64a: depolarizing-noise sensitivity of the frozen XY-QAOA (exploratory).

The formal benchmark simulates QAOA exactly in the feasible (one-hot) subspace.
Noise leaves that subspace, so here the identical circuit is simulated on the
full 2^n register, gate by gate in the solver's own order:

* product of local W states (prepared noiselessly; its gates are not modelled);
* per layer, RZ(2*gamma*h_i) on every wire, then IsingZZ(2*gamma*J_ij) in
  (i, j) order, then IsingXY(-2*beta) over each register's pairs, reversed on
  odd layers -- exactly XYMixerQAOASampler._ansatz;
* after every two-qubit gate (ZZ and XY), a two-qubit depolarizing channel of
  probability eps: one of the 15 non-identity Paulis, uniformly.

The channel is sampled by quantum trajectories; averaging |psi|^2 over
trajectories is an unbiased estimate of the noisy output distribution.
Single-qubit gate errors, readout error, idling and crosstalk are not modelled,
so the results are optimistic. The angles are the frozen noiselessly fitted
primary angles; they are not re-optimised under noise.

Outputs are reported raw and after discarding non-one-hot outcomes
(post-selection, a classical check of every register's Hamming weight).

The global white-noise model rho = F rho_ideal + (1-F) I/2^n with
F = (1-eps)^N2q is compared against the trajectories and then applied to every
primary case at each size, with logical and heavy-hex compiled gate counts.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from nanoqc.supplement.common import (
    feasible_ground, output_dir, primary_angles, primary_cases, sampler_from_case, write_csv)

DEFAULT_EPSILONS = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
# A63 heavy-hex compilation (mean two-qubit gates, Qiskit 2.5.2, opt. level 3).
DEFAULT_COMPILED_TWO_QUBIT_GATES = {4: 259.0, 6: 802.3, 8: 1235.3, 10: 1641.0}
MAX_TRAJECTORY_QUBITS = 20


class FullRegisterCircuit:
    """The solver's XY-QAOA circuit as an explicit gate list on n qubits."""

    def __init__(self, sampler, parameters: np.ndarray):
        self.n = int(sampler.num_variables)
        if self.n > MAX_TRAJECTORY_QUBITS:
            raise ValueError(f"Trajectory simulation is limited to {MAX_TRAJECTORY_QUBITS} qubits")
        self.p = int(sampler.p)
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape != (2 * self.p,):
            raise ValueError("Expected 2*p parameters")
        tol = sampler.coefficient_tolerance
        h = np.asarray(sampler.ising_h, dtype=float)
        J = np.asarray(sampler.ising_J, dtype=float)
        self.groups = [list(g) for g in sampler.site_to_variables.values()]
        self.gates: list[tuple] = []
        for layer in range(self.p):
            gamma, beta = parameters[layer], parameters[self.p + layer]
            for wire in range(self.n):
                if abs(h[wire]) > tol:
                    self.gates.append(("rz", (wire,), 2.0 * gamma * h[wire]))
            for left in range(self.n):
                for right in range(left + 1, self.n):
                    if abs(J[left, right]) > tol:
                        self.gates.append(("zz", (left, right), 2.0 * gamma * J[left, right]))
            for site in sampler.site_to_variables:
                pairs = [(a, b) for pair_site, a, b in sampler._mixer_pairs if pair_site == site]
                if layer % 2:
                    pairs.reverse()
                for a, b in pairs:
                    self.gates.append(("xy", (a, b), -2.0 * beta))
        self.two_qubit_gates = sum(1 for g in self.gates if g[0] != "rz")
        # z_i = +1 for bit 0, -1 for bit 1; wire 0 is the most significant bit.
        index = np.arange(2 ** self.n, dtype=np.int64)
        self.bits = ((index[:, None] >> (self.n - 1 - np.arange(self.n))) & 1).astype(np.int8)
        self.z = (1 - 2 * self.bits).astype(np.float64)
        self.feasible = np.all(
            [self.bits[:, g].sum(axis=1) == 1 for g in self.groups], axis=0)

    def initial_state(self) -> np.ndarray:
        state = np.zeros(2 ** self.n, dtype=np.complex128)
        state[self.feasible] = 1.0
        return state / math.sqrt(float(self.feasible.sum()))

    def index_of(self, bits) -> int:
        return int("".join(str(int(b)) for b in bits), 2)

    # --- gate kernels on a flat statevector -----------------------------
    def _tensor(self, state):
        return state.reshape((2,) * self.n)

    def apply(self, state: np.ndarray, gate: tuple) -> np.ndarray:
        kind, wires, angle = gate
        if kind == "rz":
            return state * np.exp(-0.5j * angle * self.z[:, wires[0]])
        if kind == "zz":
            a, b = wires
            return state * np.exp(-0.5j * angle * self.z[:, a] * self.z[:, b])
        a, b = wires
        t = self._tensor(state).copy()
        index01 = [slice(None)] * self.n; index01[a] = 0; index01[b] = 1
        index10 = [slice(None)] * self.n; index10[a] = 1; index10[b] = 0
        s01, s10 = t[tuple(index01)].copy(), t[tuple(index10)].copy()
        c, s = math.cos(angle / 2.0), math.sin(angle / 2.0)
        t[tuple(index01)] = c * s01 + 1j * s * s10
        t[tuple(index10)] = c * s10 + 1j * s * s01
        return t.reshape(-1)

    def pauli(self, state: np.ndarray, wire: int, label: str) -> np.ndarray:
        if label == "I":
            return state
        out = state
        if label in ("Z", "Y"):
            out = out * self.z[:, wire]
        if label in ("X", "Y"):
            out = np.flip(self._tensor(out), axis=wire).reshape(-1)
        return out  # Y up to a global phase, irrelevant to probabilities

    def draw_errors(self, rng: np.random.Generator, eps: float) -> dict[int, int]:
        """Gate index -> one of the 15 non-identity two-qubit Paulis (1..15)."""
        two_qubit = [i for i, g in enumerate(self.gates) if g[0] != "rz"]
        hit = rng.random(len(two_qubit)) < eps
        return {two_qubit[i]: int(rng.integers(1, 16)) for i in np.flatnonzero(hit)}

    def _diagonal_term(self, gate: tuple) -> np.ndarray:
        kind, wires, angle = gate
        if kind == "rz":
            return angle * self.z[:, wires[0]]
        return angle * self.z[:, wires[0]] * self.z[:, wires[1]]

    def _segments(self) -> list[tuple[int, int, np.ndarray]]:
        """Maximal runs of diagonal gates with their cached total phase."""
        if not hasattr(self, "_segment_cache"):
            segments, position = [], 0
            while position < len(self.gates):
                if self.gates[position][0] == "xy":
                    position += 1
                    continue
                first = position
                total = np.zeros(2 ** self.n, dtype=np.float64)
                while position < len(self.gates) and self.gates[position][0] != "xy":
                    total += self._diagonal_term(self.gates[position]); position += 1
                segments.append((first, position, total))
            self._segment_cache = segments
        return self._segment_cache

    def run(self, errors: dict[int, int] | None = None) -> np.ndarray:
        """Gate by gate in the solver's order; Pauli errors after two-qubit gates.

        Diagonal gates commute, so each run of them is applied as one phase,
        cached when the run has no error and split at each error otherwise.
        """
        state = self.initial_state()
        labels = ("I", "X", "Y", "Z")
        errors = errors or {}
        starts = {first: (last, total) for first, last, total in self._segments()}
        position = 0
        while position < len(self.gates):
            if position in starts:
                last, total = starts[position]
                inside = sorted(k for k in errors if position <= k < last)
                if not inside:
                    state = state * np.exp(-0.5j * total)
                else:
                    cursor = position
                    for stop in inside + [last]:
                        # Gates cursor..stop (inclusive for an error, exclusive for the end).
                        upper = stop + 1 if stop < last else last
                        if upper > cursor:
                            phase = np.zeros(2 ** self.n, dtype=np.float64)
                            for k in range(cursor, upper):
                                phase += self._diagonal_term(self.gates[k])
                            state = state * np.exp(-0.5j * phase)
                        if stop < last:
                            wires = self.gates[stop][1]
                            code = errors[stop]
                            state = self.pauli(state, wires[0], labels[code // 4])
                            state = self.pauli(state, wires[1], labels[code % 4])
                        cursor = upper
                position = last
                continue
            gate = self.gates[position]
            state = self.apply(state, gate)
            code = errors.get(position)
            if code:
                state = self.pauli(state, gate[1][0], labels[code // 4])
                state = self.pauli(state, gate[1][1], labels[code % 4])
            position += 1
        return np.abs(state) ** 2


def trajectory_metrics(circuit: FullRegisterCircuit, ground_indices: list[int], omega: int,
                       eps: float, trajectories: int, rng: np.random.Generator,
                       ideal: np.ndarray) -> dict:
    """Noisy distribution by trajectories; error-free trajectories reuse ``ideal``."""
    feasible = circuit.feasible
    ground = np.asarray(ground_indices, dtype=np.int64)
    raw_ground, raw_feasible = [], []
    for _ in range(trajectories):
        errors = circuit.draw_errors(rng, eps) if eps > 0 else {}
        # An error-free trajectory is exactly the ideal circuit.
        probabilities = circuit.run(errors) if errors else ideal
        raw_ground.append(float(probabilities[ground].sum()))
        raw_feasible.append(float(probabilities[feasible].sum()))
    raw_ground = np.asarray(raw_ground); raw_feasible = np.asarray(raw_feasible)
    mean_ground, mean_feasible = float(raw_ground.mean()), float(raw_feasible.mean())
    post = mean_ground / mean_feasible if mean_feasible > 0 else float("nan")
    uniform = len(ground_indices) / omega
    fidelity = (1.0 - eps) ** circuit.two_qubit_gates
    model = white_noise(float(ideal[ground].sum()), len(ground_indices), omega, circuit.n, fidelity)
    return dict(
        epsilon=eps, trajectories=trajectories, two_qubit_gates=circuit.two_qubit_gates,
        raw_ground_probability=mean_ground,
        raw_ground_probability_se=float(raw_ground.std(ddof=1) / math.sqrt(trajectories)) if trajectories > 1 else 0.0,
        feasible_probability=mean_feasible,
        postselected_ground_probability=post,
        log10_postselected_amplification=(math.log10(post / uniform) if post > 0 else float("-inf")),
        white_noise_fidelity=fidelity,
        white_noise_raw_ground_probability=model["raw_ground"],
        white_noise_feasible_probability=model["feasible"],
        white_noise_log10_postselected_amplification=model["log10_amplification"],
    )


def _trajectory_case(task: tuple) -> list[dict]:
    path, sites, epsilons, trajectories, seed = task
    path = Path(path)
    case = json.loads(path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(seed)
    sampler = sampler_from_case(case)
    circuit = FullRegisterCircuit(sampler, primary_angles(case))
    ground, _, omega = feasible_ground(sampler)
    ground_indices = [circuit.index_of(bits) for bits in ground]
    ideal = circuit.run()
    subspace = np.abs(sampler.subspace_state(primary_angles(case))) ** 2
    ground_set = set(ground)
    noiseless_check = abs(float(ideal[ground_indices].sum()) - float(sum(
        subspace[i] for i, bits in enumerate(sampler.feasible_energy_map())
        if tuple(int(b) for b in bits) in ground_set)))
    if noiseless_check > 1e-9:
        raise AssertionError(f"{path.name}: full-register and subspace simulations disagree ({noiseless_check})")
    rows = []
    for eps in epsilons:
        row = trajectory_metrics(circuit, ground_indices, omega, eps,
                                 1 if eps == 0 else trajectories, rng, ideal)
        row.update(case=path.name, pdb=str(case["config"].get("pdb_id", "")).lower(),
                   active_sites=sites, qubits=circuit.n, noiseless_subspace_agreement=noiseless_check,
                   trajectory_seed=seed)
        rows.append(row)
    return rows


def white_noise(ideal_ground: float, n_ground: int, omega: int, n_qubits: int, fidelity: float) -> dict:
    """Global depolarizing model; post-selection discards non-one-hot outcomes."""
    dimension = 2.0 ** n_qubits
    raw = fidelity * ideal_ground + (1.0 - fidelity) * n_ground / dimension
    feasible = fidelity + (1.0 - fidelity) * omega / dimension
    post = raw / feasible
    uniform = n_ground / omega
    return dict(raw_ground=raw, feasible=feasible, postselected=post,
                log10_amplification=(math.log10(post / uniform) if post > 0 else float("-inf")))


def critical_epsilon(ideal_ground: float, n_ground: int, omega: int, n_qubits: int,
                     two_qubit_gates: float, target_log10: float = 0.0) -> float:
    """Largest eps whose post-selected log10 amplification stays >= target (bisection)."""
    def amp(eps):
        return white_noise(ideal_ground, n_ground, omega, n_qubits,
                           (1.0 - eps) ** two_qubit_gates)["log10_amplification"]
    if amp(0.0) < target_log10:
        return 0.0
    lo, hi = 0.0, 1.0
    if amp(hi) >= target_log10:
        return 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if amp(mid) >= target_log10 else (lo, mid)
    return lo


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPSILONS))
    parser.add_argument("--trajectory-sites", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--trajectory-cases", type=int, default=10,
                        help="Distinct PDBs per size for trajectory validation (sorted, outcome-blind).")
    parser.add_argument("--trajectories", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--workers", type=int, default=1, help="Spawned processes over trajectory cases.")
    parser.add_argument("--compiled-two-qubit-gates", type=str,
                        default=",".join(f"{k}:{v}" for k, v in DEFAULT_COMPILED_TWO_QUBIT_GATES.items()),
                        help="sites:gates pairs for the heavy-hex white-noise column (A63).")
    args = parser.parse_args(argv)
    if any(not 0.0 <= e < 1.0 for e in args.epsilons) or args.trajectories < 2:
        parser.error("epsilons must lie in [0,1) and trajectories >= 2")
    compiled = {int(k): float(v) for k, v in (item.split(":") for item in args.compiled_two_qubit_gates.split(","))}
    out = output_dir(args.run_dir, "noise")
    rng = np.random.default_rng(args.seed)
    started = time.time()

    # 1) Trajectory validation on the smallest sizes; one independent stream per case.
    tasks = []
    for sites in args.trajectory_sites:
        for path, _ in list(primary_cases(args.run_dir, {sites}))[: args.trajectory_cases]:
            tasks.append((str(path), sites, list(args.epsilons), args.trajectories,
                          int(rng.integers(0, 2**63 - 1))))
    trajectory_rows = []
    if args.workers > 1 and len(tasks) > 1:
        import concurrent.futures, multiprocessing as mp
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=min(args.workers, len(tasks)), mp_context=mp.get_context("spawn")) as pool:
            for rows in pool.map(_trajectory_case, tasks):
                trajectory_rows.extend(rows)
                print(f"[trajectories] {rows[0]['case']} done ({time.time()-started:.0f}s)", flush=True)
    else:
        for task in tasks:
            rows = _trajectory_case(task)
            trajectory_rows.extend(rows)
            print(f"[trajectories] {rows[0]['case']} done ({time.time()-started:.0f}s)", flush=True)
    write_csv(out / "noise_trajectories.csv", trajectory_rows)

    # 2) White-noise model on every primary case and size.
    model_rows = []
    for path, case in primary_cases(args.run_dir, one_per_pdb=False):
        metrics = next(m for m in case["metrics"] if m.get("solver") == "qaoa"
                       and m.get("qaoa_objective") == "cvar" and int(m.get("qaoa_restarts", 0)) == 4
                       and int(m.get("outputs", 0)) == 1000)
        sites = int(case["config"]["active_sites"])
        ideal = float(metrics["exact_ground_probability"])
        registers = [len(v) for v in case["quantum_instance"]["site_to_variables"].values()]
        n_qubits = int(sum(registers))
        omega = int(math.prod(registers))
        uniform_ground = ideal / 10 ** float(metrics["log10_ground_amplification_exact"])
        n_ground = max(1, round(uniform_ground * omega))
        logical = float(metrics.get("qaoa_two_qubit_gates"))
        for label, gates in (("logical", logical), ("heavy_hex_compiled", compiled.get(sites))):
            if gates is None:
                continue
            row = dict(case=path.name, pdb=str(case["config"].get("pdb_id", "")).lower(),
                       active_sites=sites, qubits=n_qubits, gate_count_model=label, two_qubit_gates=gates,
                       ideal_ground_probability=ideal,
                       critical_epsilon_amplification_1x=critical_epsilon(ideal, n_ground, omega, n_qubits, gates, 0.0),
                       critical_epsilon_amplification_10x=critical_epsilon(ideal, n_ground, omega, n_qubits, gates, 1.0))
            for eps in args.epsilons:
                model = white_noise(ideal, n_ground, omega, n_qubits, (1.0 - eps) ** gates)
                row[f"log10_amplification_eps_{eps:g}"] = model["log10_amplification"]
                row[f"feasible_probability_eps_{eps:g}"] = model["feasible"]
            model_rows.append(row)
    write_csv(out / "noise_white_model.csv", model_rows)
    _write_summary(out, trajectory_rows, model_rows, args)
    print(f"[noise] done in {time.time()-started:.0f}s -> {out}", flush=True)
    return 0


def _write_summary(out: Path, trajectory_rows: list[dict], model_rows: list[dict], args) -> None:
    lines = ["# A64a Noise sensitivity (exploratory)", "",
             "Frozen noiselessly fitted primary angles (CVaR, 4 restarts, p=2) evaluated under two-qubit "
             "depolarizing noise. W-state preparation, single-qubit gates, readout and idling are noiseless; "
             "results are therefore optimistic. Post-selection discards outcomes that are not one-hot.", "",
             "## Trajectory validation of the white-noise model", "",
             "| Sites | eps | Cases | Mean log10 amplification (trajectories) | Mean log10 amplification (white-noise model) | Mean feasible probability |",
             "|---:|---:|---:|---:|---:|---:|"]
    for sites in sorted({r["active_sites"] for r in trajectory_rows}):
        for eps in args.epsilons:
            rows = [r for r in trajectory_rows if r["active_sites"] == sites and r["epsilon"] == eps]
            if rows:
                lines.append(f"| {sites} | {eps:g} | {len(rows)} | "
                             f"{np.mean([r['log10_postselected_amplification'] for r in rows]):.3f} | "
                             f"{np.mean([r['white_noise_log10_postselected_amplification'] for r in rows]):.3f} | "
                             f"{np.mean([r['feasible_probability'] for r in rows]):.3f} |")
    lines += ["", "## White-noise model over all primary cases", "",
              "| Sites | Gate count | Cases | Mean 2q gates | " +
              " | ".join(f"log10 amp. eps={e:g}" for e in args.epsilons) +
              " | Median eps where amplification falls to 1x | Median eps where it falls to 10x |",
              "|---:|---|---:|---:|" + "---:|" * (len(args.epsilons) + 2)]
    for sites in sorted({r["active_sites"] for r in model_rows}):
        for label in ("logical", "heavy_hex_compiled"):
            rows = [r for r in model_rows if r["active_sites"] == sites and r["gate_count_model"] == label]
            if not rows:
                continue
            cells = [f"{np.mean([r[f'log10_amplification_eps_{e:g}'] for r in rows]):.3f}" for e in args.epsilons]
            lines.append(f"| {sites} | {label} | {len(rows)} | {np.mean([r['two_qubit_gates'] for r in rows]):.0f} | "
                         + " | ".join(cells) +
                         f" | {np.median([r['critical_epsilon_amplification_1x'] for r in rows]):.2e}"
                         f" | {np.median([r['critical_epsilon_amplification_10x'] for r in rows]):.2e} |")
    lines += ["", "Exploratory (A64); no hypothesis test; does not change any preregistered result. "
              "Noiseless-subspace agreement of the full-register simulator is asserted per case (<= 1e-9)."]
    (out / "noise_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
