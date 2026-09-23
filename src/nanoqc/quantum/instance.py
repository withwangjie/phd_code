"""Formal quantum-optimization instance independent of protein-specific builders.

The object is the boundary between problem preparation and solver research:
upstream code may use EGNN, rotamer libraries and molecular energies to build
the instance, while downstream quantum/classical solvers consume only this
frozen QUBO/Ising representation and its one-hot register partition.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class QuantumOptimizationInstance:
    """Frozen constrained QUBO/Ising instance used by quantum experiments.

    Q and ising_* describe the full one-hot-penalized formulation.
    physical_self and physical_pair describe the penalty-free physical
    objective used inside the feasible subspace preserved by the XY mixer.
    """

    Q: np.ndarray
    constant_offset: float
    physical_self: np.ndarray
    physical_pair: np.ndarray
    site_to_variables: Mapping[int, Sequence[int]]
    ising_h: np.ndarray
    ising_J: np.ndarray
    ising_offset: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        q=np.asarray(self.Q,dtype=float)
        physical_self=np.asarray(self.physical_self,dtype=float)
        physical_pair=np.asarray(self.physical_pair,dtype=float)
        ising_h=np.asarray(self.ising_h,dtype=float)
        ising_J=np.asarray(self.ising_J,dtype=float)
        n=int(physical_self.size)
        if n < 1:
            raise ValueError("QuantumOptimizationInstance requires at least one binary variable")
        for name,matrix,shape in (
            ("Q",q,(n,n)),
            ("physical_pair",physical_pair,(n,n)),
            ("ising_J",ising_J,(n,n)),
        ):
            if matrix.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {matrix.shape}")
        if ising_h.shape != (n,):
            raise ValueError(f"ising_h must have shape {(n,)}, got {ising_h.shape}")
        if not all(np.isfinite(x).all() for x in (q,physical_self,physical_pair,ising_h,ising_J)):
            raise ValueError("QuantumOptimizationInstance coefficients must be finite")
        if not math.isfinite(float(self.constant_offset)) or not math.isfinite(float(self.ising_offset)):
            raise ValueError("QuantumOptimizationInstance offsets must be finite")

        normalized={
            int(site):tuple(int(v) for v in variables)
            for site,variables in sorted(self.site_to_variables.items())
        }
        variables=[v for group in normalized.values() for v in group]
        if sorted(variables) != list(range(n)):
            raise ValueError("site_to_variables must partition every variable exactly once")
        if any(len(group) < 2 for group in normalized.values()):
            raise ValueError("Every one-hot register must contain at least two states")

        object.__setattr__(self,"Q",q.copy())
        object.__setattr__(self,"physical_self",physical_self.copy())
        object.__setattr__(self,"physical_pair",physical_pair.copy())
        object.__setattr__(self,"ising_h",ising_h.copy())
        object.__setattr__(self,"ising_J",ising_J.copy())
        object.__setattr__(self,"site_to_variables",normalized)
        object.__setattr__(self,"metadata",dict(self.metadata))

    @property
    def num_qubits(self) -> int:
        """Logical binary-variable/qubit count for the QAOA encoding."""
        return int(self.physical_self.size)

    @property
    def register_sizes(self) -> Tuple[int, ...]:
        return tuple(len(group) for group in self.site_to_variables.values())

    @property
    def feasible_configuration_count(self) -> int:
        """Product-space size under exactly-one-state-per-register feasibility."""
        return int(math.prod(self.register_sizes))

    def manifest(self) -> Dict[str, Any]:
        """JSON-friendly quantum-instance contract."""
        return {
            "schema": "quantum_optimization_instance_v1",
            "encoding": "one_hot_rotamer_registers",
            "num_qubits": self.num_qubits,
            "num_registers": len(self.site_to_variables),
            "register_sizes": list(self.register_sizes),
            "feasible_configuration_count": self.feasible_configuration_count,
            "full_qubo_includes_one_hot_penalties": True,
            "xy_cost_uses_penalty_free_feasible_subspace_objective": True,
            "constant_offset": float(self.constant_offset),
            "ising_offset": float(self.ising_offset),
            "metadata": dict(self.metadata),
        }
