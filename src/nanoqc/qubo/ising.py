"""QUBO-to-Ising conversion and its exact equivalence check.
"""
from __future__ import annotations

import math
from typing import Tuple
import numpy as np



def qubo_to_ising(
    Q: np.ndarray,
    qubo_offset: float = 0.0,
    *,
    tolerance: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Convert an upper-triangular QUBO to an upper-triangular Ising model.

    Uses ``x_i = (1 - Z_i) / 2`` and returns ``(h, J, offset)`` such that

    ``H(Z) = offset + sum_i h[i] Z_i + sum_{i<j} J[i,j] Z_i Z_j``.

    Args:
        Q: Upper-triangular QUBO matrix whose diagonal stores linear terms.
        qubo_offset: Optional constant already present in the QUBO, such as the
            ``QUBOResult.constant_offset`` from one-hot expansion.
        tolerance: Maximum accepted magnitude below the diagonal.
    """

    matrix = np.asarray(Q, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Q must be square, got {matrix.shape}")
    if not np.isfinite(matrix).all() or not math.isfinite(qubo_offset):
        raise ValueError("Q and qubo_offset must be finite")
    if np.any(np.abs(np.tril(matrix, k=-1)) > tolerance):
        raise ValueError("Q must be upper triangular under the stated convention")

    size = matrix.shape[0]
    h = -0.5 * np.diag(matrix).copy()
    J = np.zeros_like(matrix)
    offset = float(qubo_offset + 0.5 * np.trace(matrix))
    for left in range(size):
        for right in range(left + 1, size):
            coupling = matrix[left, right]
            if coupling == 0.0:
                continue
            J[left, right] = 0.25 * coupling
            h[left] -= 0.25 * coupling
            h[right] -= 0.25 * coupling
            offset += 0.25 * coupling
    return h, J, offset


def validate_qubo_ising_equivalence(
    Q: np.ndarray,
    qubo_offset: float,
    h: np.ndarray,
    J: np.ndarray,
    ising_offset: float,
    *,
    tolerance: float = 1e-9,
) -> float:
    """Validate triangular storage and exact QUBO/Ising energy equality."""

    matrix = np.asarray(Q, dtype=np.float64)
    linear = np.asarray(h, dtype=np.float64)
    coupling = np.asarray(J, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Q must be square")
    size = matrix.shape[0]
    if coupling.shape != matrix.shape or linear.shape != (size,):
        raise ValueError("h/J dimensions are inconsistent with Q")
    if np.any(np.abs(np.tril(matrix, -1)) > tolerance):
        raise ValueError("Q contains non-zero entries below the diagonal")
    if np.any(np.abs(np.tril(coupling, 0)) > tolerance):
        raise ValueError("J must be strictly upper triangular")
    if not all(np.isfinite(value).all() for value in (matrix, linear, coupling)):
        raise ValueError("QUBO/Ising coefficients must be finite")
    if not math.isfinite(qubo_offset) or not math.isfinite(ising_offset):
        raise ValueError("QUBO/Ising offsets must be finite")

    reconstructed_q = np.zeros_like(matrix)
    reconstructed_q[np.triu_indices(size, 1)] = 4.0 * coupling[
        np.triu_indices(size, 1)
    ]
    for index in range(size):
        incident = coupling[:index, index].sum() + coupling[index, index + 1 :].sum()
        reconstructed_q[index, index] = -2.0 * linear[index] - 2.0 * incident
    if not np.allclose(reconstructed_q, matrix, atol=tolerance, rtol=0.0):
        raise AssertionError("Analytical Ising coefficients do not reconstruct Q")
    expected_offset = float(
        qubo_offset + 0.5 * np.trace(matrix) + 0.25 * np.triu(matrix, 1).sum()
    )
    if not math.isclose(ising_offset, expected_offset, abs_tol=tolerance, rel_tol=0.0):
        raise AssertionError("Ising constant offset is inconsistent with Q")

    states = [np.zeros(size), np.ones(size), *np.eye(size)]
    rng = np.random.default_rng(20260917)
    states.extend(rng.integers(0, 2, size=size).astype(float) for _ in range(16))
    maximum_error = 0.0
    for binary in states:
        spins = 1.0 - 2.0 * binary
        qubo_energy = float(qubo_offset + binary @ matrix @ binary)
        ising_energy = float(ising_offset + linear @ spins + spins @ coupling @ spins)
        maximum_error = max(maximum_error, abs(qubo_energy - ising_energy))
    if maximum_error > tolerance:
        raise AssertionError(
            f"QUBO/Ising mismatch {maximum_error:.3e} exceeds {tolerance:.3e}"
        )
    return maximum_error
