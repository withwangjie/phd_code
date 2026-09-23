"""Logical QAOA resource accounting.

Counts describe the circuit implemented by XYMixerQAOASampler before hardware
transpilation. They are not device-specific gate counts and do not include a
decomposition of PennyLane StatePrep for local W states.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from nanoqc.quantum.instance import QuantumOptimizationInstance


@dataclass(frozen=True)
class QAOAResourceEstimate:
    num_qubits: int
    qaoa_depth: int
    parameter_count: int
    nonzero_linear_cost_terms: int
    nonzero_pair_cost_terms: int
    mixer_pairs_per_layer: int
    rz_gates_total: int
    zz_gates_total: int
    xy_gates_total: int
    two_qubit_gates_total: int
    feasible_configuration_count: int
    eval_shots: Optional[int] = None
    max_evals: Optional[int] = None
    output_shots: Optional[int] = None
    restarts: Optional[int] = None
    max_optimization_shots: Optional[int] = None
    max_measurement_shots: Optional[int] = None
    accounting_scope: str = (
        "logical pre-transpilation QAOA gates; local W-state StatePrep decomposition excluded"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def estimate_qaoa_resources(
    instance: QuantumOptimizationInstance,
    *,
    p: int,
    coefficient_tolerance: float = 1e-10,
    eval_shots: Optional[int] = None,
    max_evals: Optional[int] = None,
    output_shots: Optional[int] = None,
    restarts: Optional[int] = None,
) -> QAOAResourceEstimate:
    """Estimate logical pre-transpilation resources for the implemented XY-QAOA.

    The count is exact for gates emitted by the implemented cost and XY-mixer
    layers at the supplied coefficient tolerance. It deliberately excludes
    decomposition of local W-state StatePrep, routing/SWAPs, error-correction
    overhead, and backend-native transpilation. These fields therefore support
    algorithmic scaling and benchmark transparency rather than a hardware-
    resource or quantum-advantage claim.
    """
    if p < 1:
        raise ValueError("p must be positive")
    if coefficient_tolerance < 0 or not math.isfinite(float(coefficient_tolerance)):
        raise ValueError("coefficient_tolerance must be finite and nonnegative")

    for name,value in (
        ("eval_shots",eval_shots),("max_evals",max_evals),
        ("output_shots",output_shots),("restarts",restarts),
    ):
        if value is not None and (isinstance(value,bool) or int(value) != value or int(value) <= 0):
            raise ValueError(f"{name} must be a positive integer when supplied")

    physical_h,physical_j,_=instance.physical_ising()
    linear=int(np.count_nonzero(np.abs(physical_h) > coefficient_tolerance))
    upper=np.triu(physical_j,1)
    pair=int(np.count_nonzero(np.abs(upper) > coefficient_tolerance))
    mixer=int(sum(len(group)*(len(group)-1)//2 for group in instance.site_to_variables.values()))

    eval_shots_i=None if eval_shots is None else int(eval_shots)
    max_evals_i=None if max_evals is None else int(max_evals)
    output_shots_i=None if output_shots is None else int(output_shots)
    restarts_i=None if restarts is None else int(restarts)
    max_opt=None
    if eval_shots_i is not None and max_evals_i is not None:
        max_opt=eval_shots_i*max_evals_i
    max_measure=None
    if max_opt is not None and output_shots_i is not None:
        max_measure=max_opt+output_shots_i

    return QAOAResourceEstimate(
        num_qubits=instance.num_qubits,
        qaoa_depth=int(p),
        parameter_count=2*int(p),
        nonzero_linear_cost_terms=linear,
        nonzero_pair_cost_terms=pair,
        mixer_pairs_per_layer=mixer,
        rz_gates_total=int(p)*linear,
        zz_gates_total=int(p)*pair,
        xy_gates_total=int(p)*mixer,
        two_qubit_gates_total=int(p)*(pair+mixer),
        feasible_configuration_count=instance.feasible_configuration_count,
        eval_shots=eval_shots_i,
        max_evals=max_evals_i,
        output_shots=output_shots_i,
        restarts=restarts_i,
        max_optimization_shots=max_opt,
        max_measurement_shots=max_measure,
    )
