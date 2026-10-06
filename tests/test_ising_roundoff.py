"""QUBO/Ising equivalence is checked against float64 rounding, not a fixed 1e-9."""
import numpy as np
import pytest

from nanoqc.qubo.ising import (
    ising_roundoff_tolerance, qubo_to_ising, validate_qubo_ising_equivalence,
)


def _large_qubo(seed: int = 3, size: int = 24, scale: float = 1e8) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    q = np.triu(rng.normal(size=(size, size)) * scale)
    return q, float(rng.normal() * scale)


def test_exact_conversion_of_large_coefficients_is_accepted() -> None:
    q, offset = _large_qubo()
    h, j, ising_offset = qubo_to_ising(q, offset)
    # The default tolerance follows the coefficient magnitude.
    error = validate_qubo_ising_equivalence(q, offset, h, j, ising_offset)
    assert error <= ising_roundoff_tolerance(q, offset)


def test_the_former_fixed_tolerance_rejected_an_exact_conversion() -> None:
    q, offset = _large_qubo()
    h, j, ising_offset = qubo_to_ising(q, offset)
    with pytest.raises(AssertionError):
        validate_qubo_ising_equivalence(q, offset, h, j, ising_offset, tolerance=1e-9)


def test_a_wrong_coefficient_is_still_rejected() -> None:
    q, offset = _large_qubo()
    h, j, ising_offset = qubo_to_ising(q, offset)
    j = j.copy()
    j[0, 1] += 1e-3 * abs(j[0, 1]) + 1.0
    with pytest.raises(AssertionError):
        validate_qubo_ising_equivalence(q, offset, h, j, ising_offset)


def test_ordinary_qubos_keep_the_historical_tolerance() -> None:
    q = np.triu(np.ones((4, 4)))
    assert ising_roundoff_tolerance(q, 0.0) == 1e-9
