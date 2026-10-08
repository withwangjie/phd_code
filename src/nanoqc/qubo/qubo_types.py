"""Data types shared by the coarse and all-atom QUBO builders.

Force-field and calibration parameters, rotamer templates/states, the
variable map and the ``QUBOResult`` container with its quantum-instance export.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
import numpy as np
from nanoqc.quantum.instance import QuantumOptimizationInstance
from nanoqc.qubo.ising import qubo_to_ising, validate_qubo_ising_equivalence



AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: index for index, aa in enumerate(AA_ORDER)}
COULOMB_KCAL_ANGSTROM = 332.06371


@dataclass(frozen=True)
class ForceFieldConfig:
    """Parameters for the coarse-grained non-bonded energy model."""

    cutoff_angstrom: float = 8.0
    softcore_delta_angstrom: float = 0.5
    hard_core_fraction: float = 0.72
    hard_sphere_penalty: float = 25.0
    lj_repulsion_cap: float = 50.0
    lj_attraction_cap: float = 5.0
    coulomb_cap: float = 20.0
    dielectric_base: float = 4.0
    dielectric_slope: float = 2.0
    thermal_energy_kcal: float = 0.593

    def __post_init__(self) -> None:
        """Reject non-physical or numerically unsafe parameter choices."""

        positive = {
            "cutoff_angstrom": self.cutoff_angstrom,
            "softcore_delta_angstrom": self.softcore_delta_angstrom,
            "hard_core_fraction": self.hard_core_fraction,
            "hard_sphere_penalty": self.hard_sphere_penalty,
            "lj_repulsion_cap": self.lj_repulsion_cap,
            "lj_attraction_cap": self.lj_attraction_cap,
            "coulomb_cap": self.coulomb_cap,
            "dielectric_base": self.dielectric_base,
            "thermal_energy_kcal": self.thermal_energy_kcal,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError(f"Force-field parameters must be positive: {positive}")
        if self.dielectric_slope < 0:
            raise ValueError("dielectric_slope cannot be negative")


@dataclass(frozen=True)
class EnergyCalibration:
    """Frozen linear calibration from coarse components to all-atom energy deltas."""

    prior_weight: float = 1.0
    vhh_environment_weight: float = 1.0
    antigen_weight: float = 1.0
    pair_weight: float = 1.0
    intercept: float = 0.0
    source: str = "uncalibrated"

    @classmethod
    def from_json(cls, path: Path) -> "EnergyCalibration":
        payload=json.loads(Path(path).read_text(encoding="utf-8"))
        required=("prior_weight","vhh_environment_weight","antigen_weight","pair_weight","intercept")
        missing=[key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Calibration file missing keys: {missing}")
        values={key:float(payload[key]) for key in required}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Calibration coefficients must be finite")
        for key in ("prior_weight","vhh_environment_weight","antigen_weight","pair_weight"):
            if values[key] < 0:
                raise ValueError(f"Calibration component weight must be nonnegative: {key}={values[key]}")
        scope=str(payload.get("scope",""))
        if "training complexes only" not in scope:
            raise ValueError(
                "Calibration provenance must state that coefficients were fit on training complexes only"
            )
        if int(payload.get("n_train_complexes",0)) < 2:
            raise ValueError("Calibration must report at least two training complexes")
        return cls(**values,source=str(path))

@dataclass(frozen=True)
class RotamerTemplate:
    """One statistically defined rotamer state."""

    chi1_degrees: float
    prior_probability: float
    chi_degrees: Tuple[float, ...] = ()
    chi_sigmas: Tuple[float, ...] = ()
    source: str = "legacy"


@dataclass
class RotamerState:
    """Generated pseudo-atom representation of one rotamer microstate."""

    site_index: int
    node_index: int
    amino_acid: str
    rotamer_index: int
    chi1_degrees: float
    prior_probability: float
    positions: np.ndarray
    sigma: np.ndarray
    epsilon: np.ndarray
    charges: np.ndarray
    chi_degrees: Tuple[float, ...] = ()
    chi_sigmas: Tuple[float, ...] = ()
    prior_energy: float = 0.0
    environment_energy: float = 0.0
    antigen_guidance_energy: float = 0.0

    @property
    def self_energy(self) -> float:
        """Return prior + VHH-fixed environment + antigen interaction once."""

        return self.prior_energy + self.environment_energy + self.antigen_guidance_energy


def chi1_well_index(angle: float) -> int:
    """Assign chi1 to the nearest gauche+, gauche- or trans well."""
    centers = (60.0, -60.0, 180.0)
    return min(range(3), key=lambda index: abs((angle - centers[index] + 180.0) % 360.0 - 180.0))


def select_chi1_well_representatives(states: Sequence[RotamerState], count: int = 3) -> list[RotamerState]:
    """Cover all chi1 wells, then fill remaining slots by calibrated energy."""
    if count < 3:
        raise ValueError("Chi1-well coverage requires at least three states")
    selected: dict[int, RotamerState] = {}
    for state in states:
        selected.setdefault(chi1_well_index(state.chi1_degrees), state)
    if len(selected) != 3:
        raise ValueError("Fixed three-state scaling requires a candidate in each chi1 well")
    chosen_ids = {id(state) for state in selected.values()}
    for state in states:
        if len(chosen_ids) >= count:
            break
        chosen_ids.add(id(state))
    if len(chosen_ids) != count:
        raise ValueError(f"Only {len(chosen_ids)} rotamers available for {count} states/site")
    return [state for state in states if id(state) in chosen_ids]


@dataclass(frozen=True)
class VariableRecord:
    """Trace one QUBO bit to its residue and rotamer state.

    ``node_index``/``original_node_index`` are PyG graph node indices for the
    coarse builder and -1 for all-atom (mmCIF-derived) candidates.
    """

    variable_index: int
    site_index: int
    node_index: int
    original_node_index: int
    residue_id: str
    amino_acid: str
    rotamer_index: int
    chi1_degrees: float
    prior_probability: float
    self_energy: float


@dataclass
class QUBOResult:
    """Complete QUBO delivery including physical components and provenance."""

    Q: np.ndarray
    variable_map: Tuple[VariableRecord, ...]
    site_to_variables: Dict[int, Tuple[int, ...]]
    lambda_value: float
    lambda_lower_bound: float
    constant_offset: float
    physical_self: np.ndarray
    physical_pair: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def energy(self, binary_state: Sequence[int]) -> float:
        """Evaluate the complete constrained QUBO energy for one bit string."""

        x = np.asarray(binary_state, dtype=np.float64)
        if x.shape != (self.Q.shape[0],):
            raise ValueError(f"Expected {self.Q.shape[0]} bits, got {x.shape}")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("binary_state must contain only 0 and 1")
        return float(self.constant_offset + x @ self.Q @ x)

    def mapping_table(self) -> list[dict[str, Any]]:
        """Return JSON/CSV-friendly variable mapping rows."""

        return [asdict(record) for record in self.variable_map]

    def to_quantum_instance(self) -> QuantumOptimizationInstance:
        """Freeze this protein-derived QUBO behind the solver-facing quantum contract."""
        ising_h,ising_J,ising_offset=qubo_to_ising(self.Q,self.constant_offset)
        validate_qubo_ising_equivalence(
            self.Q,self.constant_offset,ising_h,ising_J,ising_offset
        )
        return QuantumOptimizationInstance(
            Q=self.Q,
            constant_offset=self.constant_offset,
            physical_self=self.physical_self,
            physical_pair=self.physical_pair,
            site_to_variables=self.site_to_variables,
            ising_h=ising_h,
            ising_J=ising_J,
            ising_offset=ising_offset,
            metadata={
                "source_model": self.metadata.get("model"),
                "pdb_id": self.metadata.get("pdb_id"),
                "state_policy": self.metadata.get("state_policy"),
                "rotamer_library_path": self.metadata.get("rotamer_library_path"),
                "rotamer_library_sha256": self.metadata.get("rotamer_library_sha256"),
                "rotamer_source": self.metadata.get("rotamer_source"),
                "pyrosetta_version": self.metadata.get("pyrosetta_version"),
                "pyrosetta_init_options": self.metadata.get("pyrosetta_init_options"),
                "lambda_value": self.lambda_value,
                "lambda_lower_bound": self.lambda_lower_bound,
                "ising_energy_equivalence_max_error": self.metadata.get(
                    "ising_energy_equivalence_max_error"
                ),
                # A54: geometry-forbidden states (A47) carry a constraint
                # penalty; the solver excludes them from its angle scale only.
                "forbidden_variables": list(self.metadata.get("forbidden_variables", []) or []),
                "forbidden_variable_pairs": [list(pair) for pair in
                                             (self.metadata.get("forbidden_variable_pairs", []) or [])],
            },
        )

    def export(self, output_dir: Path, stem: str = "interface") -> Dict[str, Path]:
        """Persist matrix components and a human-readable mapping manifest.

        Returns paths to the matrix bundle, variable-mapping manifest, and
        solver-facing quantum-instance manifest. Existing files with the same
        names are replaced intentionally by this explicit method call.
        """

        if not stem or Path(stem).name != stem:
            raise ValueError("stem must be one plain filename component")
        output_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = output_dir / f"{stem}_qubo.npz"
        manifest_path = output_dir / f"{stem}_mapping.json"
        np.savez_compressed(
            matrix_path,
            Q=self.Q,
            physical_self=self.physical_self,
            physical_pair=self.physical_pair,
            lambda_value=np.asarray(self.lambda_value),
            lambda_lower_bound=np.asarray(self.lambda_lower_bound),
            constant_offset=np.asarray(self.constant_offset),
        )
        manifest = {
            "variable_map": self.mapping_table(),
            "site_to_variables": {
                str(site): list(variables)
                for site, variables in self.site_to_variables.items()
            },
            "lambda_value": self.lambda_value,
            "lambda_lower_bound": self.lambda_lower_bound,
            "constant_offset": self.constant_offset,
            "metadata": self.metadata,
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        quantum_path=output_dir / f"{stem}_quantum_instance.json"
        quantum_path.write_text(
            json.dumps(self.to_quantum_instance().manifest(),ensure_ascii=False,indent=2)+"\n",
            encoding="utf-8",
        )
        return {"matrix": matrix_path, "mapping": manifest_path, "quantum_instance": quantum_path}
