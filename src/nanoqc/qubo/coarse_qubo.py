"""Coarse-grained interface QUBO builder (``InterfaceQUBOBuilder``).

chi1-oriented pseudo-atom energies (prior, VHH environment, antigen, pair)
with the capped non-bonded proxy, encoded as a one-hot constrained QUBO.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple
import numpy as np
from torch_geometric.data import Data
from nanoqc.qubo.ising import qubo_to_ising, validate_qubo_ising_equivalence
from nanoqc.qubo.qubo_types import AA_ORDER, COULOMB_KCAL_ANGSTROM, EnergyCalibration, ForceFieldConfig, QUBOResult, RotamerState, RotamerTemplate, VariableRecord, select_chi1_well_representatives
from nanoqc.qubo.rotamer_library import _FLEXIBILITY_RANK, _NET_CHARGE, _SIDECHAIN_CHI_COUNT, _SIDECHAIN_REACH, _THREE_LETTER, _dunbrack_templates_for_site, _expanded_rotamer_templates, _load_rotamer_bins, _nearest_dunbrack_bin, rotamer_source_metadata



def _normalize(vector: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return a unit vector and fail clearly for an unusable direction."""

    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def _decode_amino_acids(x: np.ndarray) -> list[str]:
    """Decode strict 20-way one-hot residue identities from graph features."""

    if x.ndim != 2 or x.shape[1] < 21:
        raise ValueError(f"Expected x with at least 21 columns, got {x.shape}")
    residue_features = x[:, :20]
    if not np.allclose(residue_features.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("The first 20 node features must be one-hot encoded")
    if not np.all((np.isclose(residue_features, 0.0)) | (np.isclose(residue_features, 1.0))):
        raise ValueError("Amino-acid features contain non-binary values")
    return [AA_ORDER[index] for index in residue_features.argmax(axis=1)]


def _atom_parameters(kind: str) -> Tuple[float, float]:
    """Return coarse LJ sigma (A) and epsilon (kcal/mol) by pseudo-element."""

    parameters = {
        "C": (3.50, 0.12),
        "N": (3.25, 0.17),
        "O": (3.00, 0.20),
        "S": (3.60, 0.25),
    }
    return parameters[kind]


def _terminal_spec(amino_acid: str) -> list[Tuple[str, float]]:
    """Return terminal pseudo-elements and partial charges for one residue."""

    if amino_acid in "DE":
        return [("O", -0.65), ("O", -0.65), ("C", 0.30)]
    if amino_acid == "K":
        return [("N", 0.90), ("C", 0.10)]
    if amino_acid == "R":
        return [("N", 0.45), ("N", 0.45), ("C", 0.10)]
    if amino_acid in "NQ":
        return [("O", -0.30), ("N", 0.30)]
    if amino_acid in "STY":
        return [("O", -0.25), ("C", 0.25)]
    if amino_acid == "H":
        return [("N", 0.10), ("C", 0.00)]
    if amino_acid in "CM":
        return [("S", 0.00)]
    return [("C", _NET_CHARGE.get(amino_acid, 0.0))]


# Lateral spacing between terminal pseudo-atoms (see _generate_rotamer).
_TERMINAL_SPREAD_ANGSTROM = 0.38
# Largest distance of any candidate pseudo-atom from its own CA: the 1.53 A
# CB-like atom, or a terminal atom at ``reach`` along ``direction`` plus an
# orthogonal lateral spread. Used for an exact antigen neighbour pre-filter.
_MAX_PSEUDO_ATOM_OFFSET = max(
    [1.53] + [
        math.hypot(reach, 0.5 * (len(_terminal_spec(aa)) - 1) * _TERMINAL_SPREAD_ANGSTROM)
        for aa, reach in _SIDECHAIN_REACH.items()
    ]
) + 1e-6


def _nonbonded_energy(
    positions_a: np.ndarray,
    sigma_a: np.ndarray,
    epsilon_a: np.ndarray,
    charges_a: np.ndarray,
    positions_b: np.ndarray,
    sigma_b: np.ndarray,
    epsilon_b: np.ndarray,
    charges_b: np.ndarray,
    config: ForceFieldConfig,
) -> float:
    """Compute softened truncated LJ plus distance-dependent Coulomb energy."""

    if not len(positions_a) or not len(positions_b):
        return 0.0
    displacement = positions_a[:, None, :] - positions_b[None, :, :]
    distance = np.linalg.norm(displacement, axis=-1)
    active = distance < config.cutoff_angstrom
    if not np.any(active):
        return 0.0

    sigma = 0.5 * (sigma_a[:, None] + sigma_b[None, :])
    epsilon = np.sqrt(epsilon_a[:, None] * epsilon_b[None, :])
    distance_squared_soft = distance * distance + config.softcore_delta_angstrom**2
    effective_distance = np.sqrt(distance_squared_soft)

    # Exact soft-core form: 4 eps [(sigma^2/(r^2+delta^2))^6 - (...)^3].
    soft_ratio = sigma * sigma / distance_squared_soft
    lj = 4.0 * epsilon * (soft_ratio**6 - soft_ratio**3)
    cutoff_ratio = sigma * sigma / (
        config.cutoff_angstrom**2 + config.softcore_delta_angstrom**2
    )
    lj_shift = 4.0 * epsilon * (cutoff_ratio**6 - cutoff_ratio**3)
    lj = lj - lj_shift
    lj = np.clip(lj, -config.lj_attraction_cap, config.lj_repulsion_cap)

    hard_distance = config.hard_core_fraction * sigma
    overlap = np.maximum(hard_distance - distance, 0.0) / np.maximum(hard_distance, 1e-8)
    hard_penalty = config.hard_sphere_penalty * overlap**2

    charge_product = charges_a[:, None] * charges_b[None, :]
    dielectric = config.dielectric_base + config.dielectric_slope * distance
    coulomb = COULOMB_KCAL_ANGSTROM * charge_product / (
        dielectric * effective_distance
    )
    cutoff_dielectric = (
        config.dielectric_base
        + config.dielectric_slope * config.cutoff_angstrom
    )
    coulomb_shift = COULOMB_KCAL_ANGSTROM * charge_product / (
        cutoff_dielectric * config.cutoff_angstrom
    )
    coulomb = np.clip(
        coulomb - coulomb_shift,
        -config.coulomb_cap,
        config.coulomb_cap,
    )
    total = np.where(active, lj + hard_penalty + coulomb, 0.0)
    return float(total.sum())


class InterfaceQUBOBuilder:
    """Build a NISQ-sized rotamer QUBO from a pruned interface subgraph.

    COARSE geometry model: rotamer states here are generated from a local
    frame built by ``_local_frame`` (nearest same-chain CA neighbors plus the
    nearest ligand atom), NOT a real N-CA-CB-CG dihedral. This builder's
    rotamer angles are therefore NOT directly comparable to, or
    interchangeable with, ``AllAtomInterfaceQUBOBuilder``'s all-atom chi1
    angles (real N-CA-CB-CG dihedral) -- do not feed one model's solved
    angles into the other's ``chi1_angles`` override.

    Args:
        min_variables: Minimum bit count after adaptive 3--6-state allocation.
        max_variables: Hard QUBO dimension limit; must not exceed 30.
        max_sites: Maximum optimized VHH sites. Extra marked sites are ranked by
            ``interface_score`` and deterministically truncated.
        lambda_value: Optional explicit one-hot penalty. Values below the
            computed conservative lower bound are rejected.
        penalty_margin: Fractional safety margin above the physical incident
            energy bound used for automatic lambda selection.
        force_field: Coarse-grained non-bonded parameters.
    """

    def __init__(
        self,
        min_variables: int = 20,
        max_variables: int = 30,
        max_sites: int = 10,
        lambda_value: Optional[float] = None,
        penalty_margin: float = 0.10,
        force_field: Optional[ForceFieldConfig] = None,
        rotamer_mode: str = "legacy",
        rotamer_library_path: Optional[Path] = None,
        rotamer_probability_floor: float = 1e-4,
        rotamer_sigma_offsets: Sequence[float] = (-1.0, 0.0, 1.0),
        energy_calibration: Optional[EnergyCalibration] = None,
        fixed_chi1_wells: bool = False,
        fixed_states_per_site: Optional[int] = None,
    ) -> None:
        if not 2 <= min_variables <= max_variables <= 30:
            raise ValueError("Require 2 <= min_variables <= max_variables <= 30")
        if not 1 <= max_sites <= 10:
            raise ValueError("max_sites must be between 1 and 10 for >=3 states/site under <=30 variables")
        if penalty_margin <= 0:
            raise ValueError("penalty_margin must be positive")
        if lambda_value is not None and lambda_value <= 0:
            raise ValueError("lambda_value must be positive")
        self.min_variables = min_variables
        self.max_variables = max_variables
        self.max_sites = max_sites
        self.fixed_chi1_wells = bool(fixed_chi1_wells)
        self.fixed_states_per_site = (3 if fixed_chi1_wells and fixed_states_per_site is None
                                      else fixed_states_per_site)
        if self.fixed_states_per_site is not None and not self.fixed_chi1_wells:
            raise ValueError("fixed_states_per_site requires chi1-well coverage")
        if self.fixed_states_per_site is not None and not 3 <= self.fixed_states_per_site <= 6:
            raise ValueError("fixed_states_per_site must be in 3..6")
        self.lambda_value = lambda_value
        self.penalty_margin = penalty_margin
        self.force_field = force_field or ForceFieldConfig()
        self.rotamer_mode = str(rotamer_mode)
        if self.rotamer_mode not in ("legacy", "dunbrack2010", "pyrosetta_dun10"):
            raise ValueError("rotamer_mode must be legacy, dunbrack2010 or pyrosetta_dun10")
        self.rotamer_library_path = None if rotamer_library_path is None else Path(rotamer_library_path)
        self.rotamer_probability_floor = float(rotamer_probability_floor)
        self.rotamer_sigma_offsets = tuple(float(v) for v in rotamer_sigma_offsets)
        self.energy_calibration = energy_calibration or EnergyCalibration()
        if not 0.0 < self.rotamer_probability_floor < 1.0:
            raise ValueError("rotamer_probability_floor must lie in (0,1)")
        if not self.rotamer_sigma_offsets or not all(math.isfinite(v) for v in self.rotamer_sigma_offsets):
            raise ValueError("rotamer_sigma_offsets must be finite and nonempty")

    def _validate_graph(self, data: Data) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """Move required graph fields to CPU and validate their semantics."""

        if not hasattr(data, "x") or not hasattr(data, "pos"):
            raise ValueError("data must contain x and pos")
        x = data.x.detach().cpu().numpy().astype(np.float64, copy=False)
        pos = data.pos.detach().cpu().numpy().astype(np.float64, copy=False)
        if pos.shape != (x.shape[0], 3) or not np.isfinite(pos).all():
            raise ValueError(f"pos must be finite [N, 3], got {pos.shape}")
        amino_acids = _decode_amino_acids(x)
        groups = x[:, -1]
        if not np.all(np.isclose(groups, 0.0) | np.isclose(groups, 1.0)):
            raise ValueError("x[:, -1] must contain binary partner labels")
        return x, pos, amino_acids

    def _select_sites(self, data: Data, x: np.ndarray) -> np.ndarray:
        """Choose marked VHH sites, ranked by model interface score."""

        vhh = np.isclose(x[:, -1], 0.0)
        if hasattr(data, "is_active"):
            selected = data.is_active.detach().cpu().numpy().astype(bool)
            mask_name = "is_active"
        elif hasattr(data, "selected_vhh_mask"):
            selected = data.selected_vhh_mask.detach().cpu().numpy().astype(bool)
            mask_name = "selected_vhh_mask"
        else:
            selected = vhh
            mask_name = "implicit VHH mask"
        if mask_name != "implicit VHH mask":
            if selected.shape != (len(x),):
                raise ValueError(f"{mask_name} must have shape [N]")
            if np.any(selected & ~vhh):
                raise ValueError(f"{mask_name} marks a non-VHH node")
        indices = np.flatnonzero(selected)
        if self.rotamer_mode in ("dunbrack2010", "pyrosetta_dun10"):
            if not hasattr(data, "backbone_phi") or not hasattr(data, "backbone_psi"):
                raise ValueError("Dunbrack mode requires backbone_phi/backbone_psi")
            phi = data.backbone_phi.detach().cpu().numpy().astype(np.float64)
            psi = data.backbone_psi.detach().cpu().numpy().astype(np.float64)
            if phi.shape != (len(x),) or psi.shape != (len(x),):
                raise ValueError("backbone_phi/backbone_psi must have shape [N]")
            allowed = np.asarray([
                (amino not in {"A","G","P","C"}) and math.isfinite(phi[idx]) and math.isfinite(psi[idx])
                for idx, amino in enumerate(_decode_amino_acids(x))
            ], dtype=bool)
            indices = np.flatnonzero(selected & allowed)
        if not len(indices):
            raise ValueError("No eligible selected VHH residues were found for the requested rotamer model")

        if hasattr(data, "interface_score"):
            scores = data.interface_score.detach().cpu().numpy()
            if scores.shape != (len(x),) or not np.isfinite(scores).all():
                raise ValueError("interface_score must be finite with shape [N]")
        else:
            scores = np.zeros(len(x), dtype=np.float64)
        order = sorted(indices.tolist(), key=lambda i: (-float(scores[i]), i))
        return np.asarray(order[: self.max_sites], dtype=np.int64)

    def _allocate_rotamer_counts(
        self,
        site_indices: np.ndarray,
        amino_acids: Sequence[str],
        site_scores: Optional[Sequence[float]] = None,
    ) -> list[int]:
        """Allocate 3--6 retained states/site under the global variable budget.

        Flexibility supplies the baseline target (3/4/5/6 states for 0--1/2/3/4+
        chi torsions).  Interface importance breaks ties when the global <=30-bit
        budget requires contraction or permits expansion.
        """

        site_count = len(site_indices)
        minimum_possible = 3 * site_count
        maximum_possible = 6 * site_count
        if minimum_possible > self.max_variables:
            raise ValueError(
                f"{site_count} sites require at least {minimum_possible} variables "
                f"at 3 states/site; limit is {self.max_variables}"
            )
        if maximum_possible < self.min_variables:
            raise ValueError(
                f"{site_count} sites can provide at most {maximum_possible} variables; "
                f"need at least {self.min_variables}"
            )

        if site_scores is None:
            importance = np.zeros(site_count, dtype=np.float64)
        else:
            importance = np.asarray(site_scores, dtype=np.float64)
            if importance.shape != (site_count,) or not np.isfinite(importance).all():
                raise ValueError("site_scores must be finite with one value per selected site")

        def preferred_count(node: int) -> int:
            chi = _SIDECHAIN_CHI_COUNT.get(amino_acids[int(node)], 1)
            if chi <= 1:
                return 3
            if chi == 2:
                return 4
            if chi == 3:
                return 5
            return 6

        counts = [preferred_count(int(node)) for node in site_indices]

        # Contract least important / least flexible sites first, never below 3.
        while sum(counts) > self.max_variables:
            candidates = [idx for idx, count in enumerate(counts) if count > 3]
            if not candidates:
                break
            chosen = min(
                candidates,
                key=lambda idx: (
                    float(importance[idx]),
                    _SIDECHAIN_CHI_COUNT.get(amino_acids[int(site_indices[idx])], 1),
                    _FLEXIBILITY_RANK.get(amino_acids[int(site_indices[idx])], 0),
                    int(site_indices[idx]),
                ),
            )
            counts[chosen] -= 1

        # If a caller requests a larger minimum dimension, spend remaining bits
        # on the most important/flexible sites, never above six.
        while sum(counts) < self.min_variables:
            candidates = [idx for idx, count in enumerate(counts) if count < 6]
            if not candidates:
                break
            chosen = max(
                candidates,
                key=lambda idx: (
                    float(importance[idx]),
                    _SIDECHAIN_CHI_COUNT.get(amino_acids[int(site_indices[idx])], 1),
                    _FLEXIBILITY_RANK.get(amino_acids[int(site_indices[idx])], 0),
                    -int(site_indices[idx]),
                ),
            )
            counts[chosen] += 1

        total = sum(counts)
        if not self.min_variables <= total <= self.max_variables:
            raise RuntimeError(f"Internal rotamer allocation error: {total} variables")
        if any(count < 3 or count > 6 for count in counts):
            raise AssertionError(f"Invalid per-site state allocation: {counts}")
        return counts


    def _local_frame(
        self,
        node_index: int,
        pos: np.ndarray,
        x: np.ndarray,
        chain_ids: np.ndarray,
        antigen_pos: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a right-handed local frame from CA and interface directions.

        This is a COARSE geometric approximation (nearest same-chain CA
        neighbors plus nearest ligand atom), not a real N-CA-CB-CG sidechain
        dihedral frame. See the AllAtomInterfaceQUBOBuilder docstring for the
        physically-real chi1 frame used in all-atom validation; the two are
        not interchangeable.
        """

        center = pos[node_index]
        same_chain = np.flatnonzero(
            (chain_ids == chain_ids[node_index])
            & np.isclose(x[:, -1], 0.0)
            & (np.arange(len(pos)) != node_index)
        )
        ligand_pos = (pos[np.isclose(x[:, -1], 1.0)] if antigen_pos is None else antigen_pos)

        if len(same_chain):
            distances = np.linalg.norm(pos[same_chain] - center, axis=1)
            nearest = same_chain[np.argsort(distances)[:2]]
            if len(nearest) == 2:
                tangent_raw = pos[nearest[1]] - pos[nearest[0]]
            else:
                tangent_raw = pos[nearest[0]] - center
        elif len(ligand_pos):
            nearest_ligand = ligand_pos[np.argmin(np.linalg.norm(ligand_pos - center, axis=1))]
            tangent_raw = nearest_ligand - center
        else:
            tangent_raw = np.array([1.0, 0.0, 0.0])
        tangent = _normalize(tangent_raw)

        candidates: list[np.ndarray] = []
        if len(ligand_pos):
            nearest_ligand = ligand_pos[np.argmin(np.linalg.norm(ligand_pos - center, axis=1))]
            candidates.append(nearest_ligand - center)
        if len(same_chain):
            candidates.extend(pos[index] - center for index in same_chain[:3])
        candidates.extend(
            np.eye(3)[index] for index in np.argsort(np.abs(np.eye(3) @ tangent))
        )

        normal: Optional[np.ndarray] = None
        for candidate in candidates:
            perpendicular = candidate - np.dot(candidate, tangent) * tangent
            if np.linalg.norm(perpendicular) > 1e-8:
                normal = _normalize(perpendicular)
                break
        if normal is None:
            raise ValueError("Unable to construct a local residue frame")
        binormal = _normalize(np.cross(tangent, normal))
        return tangent, normal, binormal

    def _generate_rotamer(
        self,
        site_index: int,
        node_index: int,
        amino_acid: str,
        template: RotamerTemplate,
        pos: np.ndarray,
        x: np.ndarray,
        chain_ids: np.ndarray,
        rotamer_index: int,
        antigen_pos: Optional[np.ndarray] = None,
    ) -> RotamerState:
        """Attach residue-specific pseudo-atoms in a local chi1 orientation."""

        tangent, normal, binormal = self._local_frame(
            node_index, pos, x, chain_ids, antigen_pos
        )
        angle = math.radians(template.chi1_degrees)
        radial = math.cos(angle) * normal + math.sin(angle) * binormal
        direction = _normalize(0.25 * tangent + 0.9682458 * radial)
        lateral = _normalize(-math.sin(angle) * normal + math.cos(angle) * binormal)
        center = pos[node_index]
        reach = _SIDECHAIN_REACH[amino_acid]

        positions = [center + 1.53 * direction]
        kinds = ["C"]
        charges = [0.0]
        terminal = _terminal_spec(amino_acid)
        for terminal_index, (kind, charge) in enumerate(terminal):
            spread = (terminal_index - 0.5 * (len(terminal) - 1)) * _TERMINAL_SPREAD_ANGSTROM
            positions.append(center + reach * direction + spread * lateral)
            kinds.append(kind)
            charges.append(charge)
        parameters = [_atom_parameters(kind) for kind in kinds]
        sigma = np.asarray([item[0] for item in parameters], dtype=np.float64)
        epsilon = np.asarray([item[1] for item in parameters], dtype=np.float64)
        return RotamerState(
            site_index=site_index,
            node_index=node_index,
            amino_acid=amino_acid,
            rotamer_index=rotamer_index,
            chi1_degrees=template.chi1_degrees,
            prior_probability=template.prior_probability,
            positions=np.asarray(positions, dtype=np.float64),
            sigma=sigma,
            epsilon=epsilon,
            charges=np.asarray(charges, dtype=np.float64),
            chi_degrees=tuple(float(v) for v in (template.chi_degrees or (template.chi1_degrees,))),
            chi_sigmas=tuple(float(v) for v in template.chi_sigmas),
        )

    def _rigid_environment(
        self,
        pos: np.ndarray,
        amino_acids: Sequence[str],
        frozen_mask: np.ndarray,
        active_mask: np.ndarray,
        vhh_mask: np.ndarray,
        excluded_node: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Represent VHH-only fixed background; antigen is scored separately."""

        if (frozen_mask.shape != (len(pos),) or active_mask.shape != (len(pos),)
                or vhh_mask.shape != (len(pos),)):
            raise ValueError("active/frozen/vhh masks must have shape [N]")
        keep = (frozen_mask | active_mask) & vhh_mask
        keep[excluded_node] = False
        environment_indices = np.flatnonzero(keep)
        env_pos = pos[environment_indices]
        env_sigma = np.full(len(env_pos), 3.50, dtype=np.float64)
        env_epsilon = np.full(len(env_pos), 0.06, dtype=np.float64)
        # Frozen residues retain coarse net charges. Other active backbones are
        # neutral here because their side-chain charge is handled by pair terms.
        charges = [
            _NET_CHARGE.get(amino_acids[index], 0.0) if frozen_mask[index] else 0.0
            for index in environment_indices
        ]
        return env_pos, env_sigma, env_epsilon, np.asarray(charges)

    def _full_vhh_environment(
        self,
        vhh_pos: np.ndarray,
        vhh_amino_acids: Sequence[str],
        vhh_index: np.ndarray,
        optimized_indices: set,
        own_index: int,
        center: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fixed VHH background from the full complex, cutoff-limited only.

        Every VHH residue except the site itself contributes a CA bead.
        Residues whose side chains are being optimized are neutral here (their
        side-chain charges enter through pair terms); all other VHH residues,
        including declared-but-unselected ones, keep their coarse net charge,
        matching ``_rigid_environment``. The CA pre-filter is exact for the
        same reason as in ``_antigen_environment``.
        """
        reach = self.force_field.cutoff_angstrom + _MAX_PSEUDO_ATOM_OFFSET
        keep = np.flatnonzero(
            (vhh_index != own_index)
            & (np.linalg.norm(vhh_pos - center, axis=1) < reach)
        )
        env_pos = vhh_pos[keep]
        env_sigma = np.full(len(keep), 3.50, dtype=np.float64)
        env_epsilon = np.full(len(keep), 0.06, dtype=np.float64)
        charges = np.asarray([
            0.0 if int(vhh_index[i]) in optimized_indices
            else _NET_CHARGE.get(vhh_amino_acids[int(i)], 0.0)
            for i in keep
        ], dtype=np.float64)
        return env_pos, env_sigma, env_epsilon, charges

    def _antigen_environment(
        self,
        antigen_pos: np.ndarray,
        antigen_amino_acids: Sequence[str],
        center: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Coarse antigen environment limited only by the atom-pair cutoff.

        A candidate pseudo-atom lies at most ``_MAX_PSEUDO_ATOM_OFFSET`` from
        its CA, so an antigen bead can reach any pseudo-atom within the
        non-bonded cutoff only if its CA-CA distance is below
        cutoff + that offset. This neighbour pre-filter is therefore exact:
        it never drops a pair that ``_nonbonded_energy`` would score.
        """

        if not len(antigen_pos):
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.float64),
            )
        reach = self.force_field.cutoff_angstrom + _MAX_PSEUDO_ATOM_OFFSET
        keep = np.flatnonzero(np.linalg.norm(antigen_pos - center, axis=1) < reach)
        env_pos = antigen_pos[keep]
        env_sigma = np.full(len(keep), 3.50, dtype=np.float64)
        env_epsilon = np.full(len(keep), 0.06, dtype=np.float64)
        env_charge = np.asarray(
            [_NET_CHARGE.get(antigen_amino_acids[int(index)], 0.0) for index in keep],
            dtype=np.float64,
        )
        return env_pos, env_sigma, env_epsilon, env_charge

    def _lambda_lower_bound(
        self,
        self_energy: np.ndarray,
        pair_energy: np.ndarray,
        site_to_variables: Mapping[int, Tuple[int, ...]],
    ) -> float:
        """Bound any single-bit physical gain before adding one-hot penalties.

        For each variable, this uses its absolute self term plus the maximum
        absolute interaction with every *other* site. The maximum incident
        bound dominates the most attractive individual pair and is deliberately
        conservative for both missing-choice and multiple-choice violations.
        """

        variable_site = {
            variable: site
            for site, variables in site_to_variables.items()
            for variable in variables
        }
        incident_bounds = []
        for variable in range(len(self_energy)):
            bound = abs(float(self_energy[variable]))
            for other_site, other_variables in site_to_variables.items():
                if other_site == variable_site[variable]:
                    continue
                values = [
                    pair_energy[min(variable, other), max(variable, other)]
                    for other in other_variables
                ]
                bound += max(abs(float(value)) for value in values)
            incident_bounds.append(bound)
        raw_bound = max(incident_bounds, default=1.0)
        return max(1.0, raw_bound) * (1.0 + self.penalty_margin) + 1e-6

    def build(self, data: Data) -> QUBOResult:
        """Construct physical terms, inject one-hot constraints, and return Q."""

        x, pos, amino_acids = self._validate_graph(data)
        site_nodes = self._select_sites(data, x)
        if hasattr(data, "interface_score"):
            all_site_scores = data.interface_score.detach().cpu().numpy().astype(np.float64)
            site_scores = all_site_scores[site_nodes]
        else:
            site_scores = np.zeros(len(site_nodes), dtype=np.float64)
        counts = ([self.fixed_states_per_site] * len(site_nodes) if self.fixed_chi1_wells
                  else self._allocate_rotamer_counts(site_nodes, amino_acids, site_scores))
        if self.fixed_chi1_wells and not self.min_variables <= self.fixed_states_per_site * len(site_nodes) <= self.max_variables:
            raise ValueError("Fixed state policy exceeds the configured QUBO dimension")
        active_mask = np.zeros(len(x), dtype=bool)
        active_mask[site_nodes] = True
        declared_active = (
            data.is_active.detach().cpu().numpy().astype(bool)
            if hasattr(data, "is_active")
            else active_mask.copy()
        )
        if hasattr(data, "is_frozen_environment"):
            frozen_mask = data.is_frozen_environment.detach().cpu().numpy().astype(bool)
            if frozen_mask.shape != (len(x),):
                raise ValueError("is_frozen_environment must have shape [N]")
        else:
            frozen_mask = ~active_mask
        # If max_sites truncates a larger adaptive active set, keep the omitted
        # residues as a fixed background instead of silently dropping them.
        frozen_mask = frozen_mask | (declared_active & ~active_mask)
        if np.any(active_mask & frozen_mask):
            raise ValueError("Active and Frozen residue masks must be disjoint")
        if hasattr(data, "node_chain_id"):
            chain_ids = data.node_chain_id.detach().cpu().numpy()
            if chain_ids.shape != (len(x),):
                raise ValueError("node_chain_id must have shape [N]")
        else:
            chain_ids = np.where(np.isclose(x[:, -1], 0.0), 0, 1)

        vhh_mask = np.isclose(x[:, -1], 0.0)
        if (hasattr(data, "vhh_context_pos") and hasattr(data, "vhh_context_x")
                and hasattr(data, "vhh_context_index") and hasattr(data, "original_node_index")):
            vhh_context_pos = data.vhh_context_pos.detach().cpu().numpy().astype(np.float64)
            vhh_context_aa = _decode_amino_acids(
                data.vhh_context_x.detach().cpu().numpy().astype(np.float64))
            vhh_context_index = data.vhh_context_index.detach().cpu().numpy().astype(np.int64)
            if (vhh_context_pos.shape != (len(vhh_context_aa), 3)
                    or vhh_context_index.shape != (len(vhh_context_aa),)
                    or not np.isfinite(vhh_context_pos).all()):
                raise ValueError("vhh_context_* must be finite and aligned")
            subgraph_to_complex = data.original_node_index.detach().cpu().numpy().astype(np.int64)
            optimized = {int(subgraph_to_complex[int(node)]) for node in site_nodes}
            environments = {
                int(node_index): self._full_vhh_environment(
                    vhh_context_pos, vhh_context_aa, vhh_context_index, optimized,
                    int(subgraph_to_complex[int(node_index)]), pos[int(node_index)],
                )
                for node_index in site_nodes
            }
            vhh_scope = "full_complex_vhh"
        else:
            environments = {
                int(node_index): self._rigid_environment(
                    pos, amino_acids, frozen_mask, active_mask, vhh_mask, int(node_index)
                )
                for node_index in site_nodes
            }
            vhh_scope = "graph_vhh_nodes"
        if hasattr(data, "antigen_context_pos") and hasattr(data, "antigen_context_x"):
            antigen_pos = data.antigen_context_pos.detach().cpu().numpy().astype(np.float64)
            antigen_amino_acids = _decode_amino_acids(
                data.antigen_context_x.detach().cpu().numpy().astype(np.float64))
            if antigen_pos.shape != (len(antigen_amino_acids), 3) or not np.isfinite(antigen_pos).all():
                raise ValueError("antigen_context_pos must be finite [M, 3] aligned with antigen_context_x")
            antigen_scope = "full_complex_antigen"
        else:
            antigen_mask = np.isclose(x[:, -1], 1.0)
            antigen_pos = pos[antigen_mask]
            antigen_amino_acids = [aa for aa, keep in zip(amino_acids, antigen_mask) if keep]
            antigen_scope = "graph_antigen_nodes"
        antigen_environments = {
            int(node_index): self._antigen_environment(
                antigen_pos, antigen_amino_acids, pos[int(node_index)]
            )
            for node_index in site_nodes
        }
        if self.rotamer_mode in ("dunbrack2010", "pyrosetta_dun10"):
            if not hasattr(data, "backbone_phi") or not hasattr(data, "backbone_psi"):
                raise ValueError("Dunbrack mode requires backbone_phi/backbone_psi graph metadata")
            phi_all=data.backbone_phi.detach().cpu().numpy().astype(np.float64)
            psi_all=data.backbone_psi.detach().cpu().numpy().astype(np.float64)
            requested=set()
            for node_index in site_nodes:
                aa=amino_acids[int(node_index)]
                if aa in "AG": continue
                requested.add((_THREE_LETTER[aa], _nearest_dunbrack_bin(phi_all[int(node_index)]), _nearest_dunbrack_bin(psi_all[int(node_index)])))
            dunbrack_bins=_load_rotamer_bins(self.rotamer_mode, self.rotamer_library_path, requested) if requested else {}
        else:
            phi_all=psi_all=None
            dunbrack_bins={}

        rotamers: list[RotamerState] = []
        site_to_variables: Dict[int, Tuple[int, ...]] = {}
        raw_pool_sizes_actual: list[int] = []
        for site_index, (node_index, count) in enumerate(zip(site_nodes, counts)):
            aa = amino_acids[int(node_index)]
            if self.rotamer_mode in ("dunbrack2010", "pyrosetta_dun10"):
                templates = _dunbrack_templates_for_site(
                    dunbrack_bins, aa, phi_all[int(node_index)], psi_all[int(node_index)],
                    probability_floor=self.rotamer_probability_floor,
                    sigma_offsets=self.rotamer_sigma_offsets,
                    ensure_chi1_wells=self.fixed_chi1_wells,
                )
            else:
                templates = _expanded_rotamer_templates(aa)
            raw_pool_sizes_actual.append(len(templates))
            best_probability = max(template.prior_probability for template in templates)
            candidate_states: list[RotamerState] = []
            for template_index, template in enumerate(templates):
                state = self._generate_rotamer(
                    site_index,
                    int(node_index),
                    aa,
                    template,
                    pos,
                    x,
                    chain_ids,
                    template_index,
                    antigen_pos,
                )
                state.prior_energy = -self.force_field.thermal_energy_kcal * math.log(
                    template.prior_probability / best_probability
                )
                state.antigen_guidance_energy = _nonbonded_energy(
                    state.positions,
                    state.sigma,
                    state.epsilon,
                    state.charges,
                    *antigen_environments[int(node_index)],
                    self.force_field,
                )
                state.environment_energy = _nonbonded_energy(
                    state.positions,
                    state.sigma,
                    state.epsilon,
                    state.charges,
                    *environments[int(node_index)],
                    self.force_field,
                )
                candidate_states.append(state)

            cal=self.energy_calibration
            candidate_states.sort(
                key=lambda state: (
                    cal.prior_weight*state.prior_energy
                    + cal.vhh_environment_weight*state.environment_energy
                    + cal.antigen_weight*state.antigen_guidance_energy,
                    state.rotamer_index,
                )
            )
            selected_states = (select_chi1_well_representatives(candidate_states, count)
                               if self.fixed_chi1_wells else candidate_states[:count])
            variable_indices = []
            for selected_index, state in enumerate(selected_states):
                state.rotamer_index = selected_index
                variable_indices.append(len(rotamers))
                rotamers.append(state)
            site_to_variables[site_index] = tuple(variable_indices)

        variable_count = len(rotamers)
        if not self.min_variables <= variable_count <= self.max_variables:
            raise RuntimeError(f"Produced invalid QUBO dimension {variable_count}")
        cal=self.energy_calibration
        raw_prior = np.asarray([state.prior_energy for state in rotamers],dtype=np.float64)
        raw_vhh_environment = np.asarray([state.environment_energy for state in rotamers],dtype=np.float64)
        raw_antigen = np.asarray([state.antigen_guidance_energy for state in rotamers],dtype=np.float64)
        physical_self = (
            cal.prior_weight*raw_prior
            + cal.vhh_environment_weight*raw_vhh_environment
            + cal.antigen_weight*raw_antigen
        )
        raw_pair = np.zeros((variable_count, variable_count), dtype=np.float64)
        for left in range(variable_count):
            for right in range(left + 1, variable_count):
                if rotamers[left].site_index == rotamers[right].site_index:
                    continue
                a, b = rotamers[left], rotamers[right]
                raw_pair[left, right] = _nonbonded_energy(
                    a.positions,
                    a.sigma,
                    a.epsilon,
                    a.charges,
                    b.positions,
                    b.sigma,
                    b.epsilon,
                    b.charges,
                    self.force_field,
                )
        physical_pair = cal.pair_weight * raw_pair

        lambda_lower_bound = self._lambda_lower_bound(
            physical_self, physical_pair, site_to_variables
        )
        if self.lambda_value is not None and self.lambda_value < lambda_lower_bound:
            raise ValueError(
                f"lambda_value={self.lambda_value:.6g} is below the conservative "
                f"lower bound {lambda_lower_bound:.6g}"
            )
        lambda_value = self.lambda_value or lambda_lower_bound

        Q = physical_pair.copy()
        diagonal = physical_self - lambda_value
        np.fill_diagonal(Q, diagonal)
        for variables in site_to_variables.values():
            for left, right in _combinations(variables):
                Q[left, right] += 2.0 * lambda_value
        Q[np.tril_indices(variable_count, k=-1)] = 0.0
        if not np.isfinite(Q).all():
            raise FloatingPointError("QUBO contains non-finite values")

        original_indices = (
            data.original_node_index.detach().cpu().numpy()
            if hasattr(data, "original_node_index")
            else np.arange(len(x))
        )
        residue_ids = (
            list(data.residue_ids)
            if hasattr(data, "residue_ids") and len(data.residue_ids) == len(x)
            else [str(index) for index in range(len(x))]
        )
        variable_map = tuple(
            VariableRecord(
                variable_index=index,
                site_index=state.site_index,
                node_index=state.node_index,
                original_node_index=int(original_indices[state.node_index]),
                residue_id=str(residue_ids[state.node_index]),
                amino_acid=state.amino_acid,
                rotamer_index=state.rotamer_index,
                chi1_degrees=state.chi1_degrees,
                prior_probability=state.prior_probability,
                self_energy=float(physical_self[index]),
            )
            for index, state in enumerate(rotamers)
        )
        max_attraction = max(
            0.0,
            -float(np.min(physical_pair)) if physical_pair.size else 0.0,
        )
        constant_offset = float(lambda_value * len(site_to_variables) + cal.intercept)
        ising_h, ising_J, ising_offset = qubo_to_ising(Q, constant_offset)
        max_equivalence_error = validate_qubo_ising_equivalence(
            Q, constant_offset, ising_h, ising_J, ising_offset
        )
        metadata: Dict[str, Any] = {
            "model": "CA-frame coarse-grained pseudo-atom force field with full Dunbrack rotamer-state provenance; pseudo-atom geometry remains chi1-oriented",
            "energy_unit": "approximate kcal/mol",
            "site_node_indices": site_nodes.tolist(),
            "rotamers_per_site": counts,
            "raw_rotamer_pool_sizes": raw_pool_sizes_actual,
            "rotamer_state_policy": (
                "Dunbrack 2010 backbone-dependent full rotamer states (chi1..chiN); chi1 sigma expansion controls pseudo-atom orientation while distal chi means are retained for exact all-atom reconstruction/calibration; retain 3--6 states/site under <=30 variables"
                if self.rotamer_mode in ("dunbrack2010", "pyrosetta_dun10")
                else "legacy 6/9/12 raw chi1 sub-rotamers by flexibility; retain 3--6 states/site under <=30 variables"
            ),
            "candidate_guidance": "pre-screen by rotamer prior + VHH-only fixed-environment energy + antigen interaction energy; antigen counted once",
            "antigen_environment_scope": antigen_scope,
            "vhh_environment_scope": vhh_scope,
            "antigen_environment_rule": (
                "every antigen residue of the source complex, limited only by the "
                "coarse atom-pair non-bonded cutoff"
                if antigen_scope == "full_complex_antigen"
                else "antigen nodes present in the supplied graph"),
            "rotamer_model": self.rotamer_mode,
            "probability_floor_policy": (
                "retain_positive-probability_top_library_sample_per_missing_chi1_well"
                if self.fixed_chi1_wells and self.rotamer_mode in ("dunbrack2010","pyrosetta_dun10")
                else "global_probability_floor"
            ),
            "state_policy": (f"fixed_{self.fixed_states_per_site}_chi1_coverage" if self.fixed_chi1_wells and self.fixed_states_per_site != 3
                             else "fixed_three_chi1_wells" if self.fixed_chi1_wells else "adaptive_3_to_6"),
            **rotamer_source_metadata(self.rotamer_mode, self.rotamer_library_path),
            "rotamer_probability_floor": self.rotamer_probability_floor,
            "rotamer_sigma_offsets": list(self.rotamer_sigma_offsets),
            "energy_calibration": asdict(cal),
            "raw_prior_energy": raw_prior.tolist(),
            "raw_vhh_environment_energy": raw_vhh_environment.tolist(),
            "raw_antigen_energy": raw_antigen.tolist(),
            "raw_pair_energy_upper": raw_pair.tolist(),
            "antigen_guidance_energy": raw_antigen.tolist(),
            "rotamer_state_records": [
                dict(
                    variable_index=int(index),
                    residue_id=str(residue_ids[state.node_index]),
                    site_index=int(state.site_index),
                    amino_acid=str(state.amino_acid),
                    rotamer_index=int(state.rotamer_index),
                    chi1_degrees=float(state.chi1_degrees),
                    chi_degrees=[float(v) for v in state.chi_degrees],
                    chi_sigmas=[float(v) for v in state.chi_sigmas],
                    prior_probability=float(state.prior_probability),
                )
                for index,state in enumerate(rotamers)
            ],
            "variable_count": variable_count,
            "active_residue_count": int(active_mask.sum()),
            "frozen_environment_count": int(frozen_mask.sum()),
            "ising_energy_equivalence_max_error": max_equivalence_error,
            "max_pair_attraction": max_attraction,
            "lambda_exceeds_max_pair_attraction": lambda_value > max_attraction,
            "surrogate_chi1_sites": [
                site for site, node in enumerate(site_nodes)
                if amino_acids[int(node)] in {"A", "G"}
            ],
            "pdb_id": getattr(data, "pdb_id", "unknown"),
            "source_id": getattr(data, "source_id", "unknown"),
            "force_field": asdict(self.force_field),
        }
        return QUBOResult(
            Q=Q,
            variable_map=variable_map,
            site_to_variables=site_to_variables,
            lambda_value=float(lambda_value),
            lambda_lower_bound=float(lambda_lower_bound),
            constant_offset=constant_offset,
            physical_self=physical_self,
            physical_pair=physical_pair,
            metadata=metadata,
        )

def _combinations(values: Iterable[int]) -> Iterable[Tuple[int, int]]:
    """Yield sorted unique pairs from a small variable-index collection."""

    items = tuple(values)
    for left_index in range(len(items)):
        for right_index in range(left_index + 1, len(items)):
            yield items[left_index], items[right_index]
